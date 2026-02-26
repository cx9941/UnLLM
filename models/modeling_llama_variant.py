from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from dataclasses import dataclass
from transformers import LlamaPreTrainedModel, LlamaModel, AutoTokenizer
from .utils import sample_tail_normal_distribution, orthogonality_loss, contrastive_loss
logger = logging.get_logger(__name__)
import torch.nn.functional as F

# tok_llama32 = AutoTokenizer.from_pretrained('llms/Llama-3.2-8B-Instruct')

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

from transformers.utils import ModelOutput
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    scores: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_type_ids: Optional[Tuple[torch.FloatTensor, ...]] = None
    lm_head_weight: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class CausalLMOutputWithCheck(ModelOutput):
    logits: torch.FloatTensor = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.propotype = {}

        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        
        

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        logits = logits.float()

        loss = None

        if 'token_type_ids' not in kwargs:
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values
            )
        
        loss_fct = CrossEntropyLoss()

        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)

        gen_labels = input_ids.clone()
        gen_labels[kwargs['token_type_ids']<=0] = -100
        shift_gen_labels = gen_labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
        loss_gen = loss_fct(shift_logits, shift_gen_labels)

        class_labels = input_ids.clone()
        class_labels[kwargs['token_type_ids']!=3] = -100
        shift_class_labels = class_labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
        loss_class = loss_fct(shift_logits, shift_class_labels)

        labels[kwargs['token_type_ids']<=1] = -100

        loss = loss_gen + loss_class

        shift_hidden_states = hidden_states[:,:-1,:].contiguous().to(shift_logits.device)
        shift_token_type_ids = kwargs['token_type_ids'][...,1:].contiguous().to(shift_logits.device)
        class_hidden_states = shift_hidden_states[shift_token_type_ids==3].reshape(logits.shape[0], -1, shift_hidden_states.shape[-1])

        
        ood_hidden_states = class_hidden_states[:,-1]

        loss_con = contrastive_loss(ood_hidden_states, kwargs['class_golds'], temperature=kwargs['temperature'].mean().item())
        loss += loss_con * kwargs['con_weight'].mean()

        epsilon = int(kwargs['epsilon'].mean())

        self.propotype_update(class_hidden_states, kwargs['class_golds'], num_samples=kwargs['vos_neg_sample'].mean().int().item(), epsilon=epsilon)

        if self.training:
            if kwargs['neg_weight'].mean() != 0:
                neg_sample_list = [self.propotype[i]['neg_samples'] for i in self.propotype if 'neg_samples' in self.propotype]
                if len(neg_sample_list) > 0:
                    neg_sample = torch.concat(neg_sample_list)
                    pos_sample = class_hidden_states[kwargs['class_golds']!=0]
                    neg_sample = neg_sample.to(pos_sample.device, dtype=pos_sample.dtype)
                    e_dim = int(kwargs['e_dim'].mean())
                    loss_neg = orthogonality_loss(neg_sample=neg_sample, pos_sample=pos_sample, k=e_dim)
                    loss += loss_neg * kwargs['neg_weight'].mean()
        # tok_llama32.decode(input_ids[0][kwargs['token_type_ids'][0]==3])
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=class_hidden_states,
            token_type_ids=kwargs['token_type_ids'],
            lm_head_weight=self.lm_head.weight[kwargs['class_token'][0]].unsqueeze(0)
        )

    
    def check_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(outputs[0], lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(outputs[0].to(self.lm_head.weight.dtype))
        logits = logits.float()

        return CausalLMOutputWithCheck(
            logits=logits,
            attentions=outputs.attentions,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        
        
        
        if past_key_values is not None:
            if inputs_embeds is not None:  
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def propotype_update(self, class_hidden_states, class_golds, num_samples=20, epsilon=2):
        for i,v in enumerate(class_golds.tolist()):
            if v not in self.propotype:
                self.propotype[v] = {'hidden_states': class_hidden_states[i].detach().cpu(), 'num': 1, 'queue_hidden':[class_hidden_states[i].detach().cpu()]}
            else:
                self.propotype[v]['hidden_states'] = (self.propotype[v]['hidden_states'] * self.propotype[v]['num'] + class_hidden_states[i].detach().cpu()) / (self.propotype[v]['num'] + 1)
                self.propotype[v]['num'] = self.propotype[v]['num'] + 1
                self.propotype[v]['queue_hidden'].append(class_hidden_states[i].detach().cpu())
                if len(self.propotype[v]['queue_hidden']) > 10:
                    self.propotype[v]['queue_hidden'].pop(0)
                if len(self.propotype[v]['queue_hidden']) >= 10:
                    sample_hidden = torch.concat([_.unsqueeze(0) for _ in self.propotype[v]['queue_hidden']], dim=0)
                    neg_samples = sample_tail_normal_distribution(sample_hidden, num_samples=num_samples, epsilon=epsilon)
                    if neg_samples is not None:
                        self.propotype[v]['neg_samples'] = neg_samples
                    elif 'neg_samples' in self.propotype[v]:
                        del self.propotype[v]['neg_samples']