from tqdm import tqdm
import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union
import datasets
import torch
import torch.nn as nn
import math
from accelerate.state import PartialState
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from huggingface_hub.utils._deprecation import _deprecate_arguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
import copy
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from transformers.integrations.deepspeed import deepspeed_init
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, has_length, denumpify_detensorize, speed_metrics
from transformers.trainer_pt_utils import EvalLoopContainer, IterableDatasetShard, find_batch_size, nested_detach
from transformers.utils import is_torch_xla_available
from trl.trainer.sft_config import SFTConfig
from trl import SFTTrainer
from peft import PeftConfig
from utils.utils import get_latest_checkpoint, get_best_checkpoint
import os
from transformers.integrations.tpu import tpu_spmd_dataloader
import re
import json
import pandas as pd
from transformers.utils import logging
from utils.bert_embedding import obtain_bert_embedding
from utils.utils import compute_hidden_score, compute_attention_score, compute_perplexity, compute_logit_entropy, evaluate_predictions
from qwen_vl_utils import process_vision_info
logger = logging.get_logger(__name__)

class OHTC_Trainer(SFTTrainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        num_class: Optional[int] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,  
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        compute_anology_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = None,
        chars_per_token: Optional[float] = None,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: Optional[int] = None,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        eval_packing: Optional[bool] = None,
        dataset_name: Optional[str] = None,
        rate: Optional[float] = None,
        process_gather_logits=None,
        train_ablation=None,
        mode=None,
        model_name_or_path=None,
        segment_length=None,
        neg_sample=None,
        propotype_path=None,
        processor=None,
    ):
        self.train_ablation = train_ablation
        self.model_name_or_path = model_name_or_path
        self.segment_length = segment_length
        self.neg_sample = neg_sample
        if mode == 'train':
            self.eval_ablation_list = ['binary']
        else:
            if mode in ['eval-ood', 'test-ood']:
                self.eval_ablation_list = ['cosine_distance-scores', 'maha_distance-scores', 'msp_logits-scores', 'energy_logits-scores']
            elif mode in ['train-ood']:
                self.eval_ablation_list = ['binary']
            else:
                self.eval_ablation_list = ['binary', 'class', 'interfere-pca-X_correct', 'interfere-pca-lm_weight', 'interfere-mean-X_correct','interfere-mean-lm_weight']
                if segment_length == 1 and neg_sample == 0:
                    self.eval_ablation_list = ['binary', 'class']

        self.dataset_name = dataset_name
        self.rate = rate
        self.mode = mode
        self.num_class = num_class
        self.process_gather_logits = process_gather_logits
        self.propotype_path = propotype_path
        self.processor = processor

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,  
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            packing=packing,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            infinite=infinite,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            dataset_num_proc=dataset_num_proc,
            dataset_batch_size=dataset_batch_size,
            neftune_noise_alpha=neftune_noise_alpha,
            model_init_kwargs=model_init_kwargs,
            dataset_kwargs=dataset_kwargs,
            eval_packing=eval_packing,
        )

        if test_dataset is not None:
            _multiple = isinstance(test_dataset, dict)
            _test_datasets = test_dataset if _multiple else {"singleton": test_dataset}

            eval_packing = packing if eval_packing is None else eval_packing

            for _eval_dataset_name, _eval_dataset in _test_datasets.items():
                _test_datasets[_eval_dataset_name] = self._prepare_dataset(
                    _eval_dataset,
                    tokenizer,
                    eval_packing,
                    dataset_text_field,
                    max_seq_length,
                    formatting_func,
                    num_of_sequences,
                    chars_per_token,
                    **dataset_kwargs,
                )
            if not _multiple:
                test_dataset = _test_datasets["singleton"]

        self.test_dataset = test_dataset
        self.compute_anology_metrics = compute_anology_metrics

    def self_check(self, test_dataset, metrics_dir, checkpoint_dir=None):

        if checkpoint_dir is not None:
            checkpoint_path = get_latest_checkpoint(checkpoint_dir)
            self._load_from_checkpoint(checkpoint_path)

        model = self.model
        model.eval()
        model = self.accelerator.prepare(model)

        label_idx = pd.read_csv(f'data/{self.dataset_name}/{self.dataset_name}_{self.rate}/idx.txt', sep='\t')
        label_idx = label_idx.sort_values(by='index')[['node', 'index']]
        label_idx.columns = ['label', 'idx']

        label_list = ['OOD'] + label_idx['label'].tolist()
        label_list_expand = ['000.It is OOD'] + [f"{i:03d}.{label_list[i]}" for i in range(1, len(label_list))]

        train_data = pd.read_csv(f'data/{self.dataset_name}/origin_data/train.tsv', sep='\t')
        eval_data = pd.read_csv(f'data/{self.dataset_name}/origin_data/eval.tsv', sep='\t')

        df = pd.concat([train_data, eval_data])
        df['label'] = df['label'].apply(lambda x: x.replace('_', ' '))
        df = pd.merge(df, label_idx, on='label', how='left').fillna(0)
        df['idx'] = df['idx'].apply(int)

        # class_golds = torch.load(f"{self.args.output_dir}/ood_eval_scores/test_golds.pt", weights_only=False)

        eval_ablation = 'check'

        if os.path.exists(f"{self.args.output_dir}/ood_eval_scores/test_binary_preds.pt"):
            preds = torch.load(f"{self.args.output_dir}/ood_eval_scores/test_binary_preds.pt", weights_only=False)
        else:
            preds = torch.ones(len(test_dataset)).long()

        def update_content(example, index):
            if preds[index] != 0:
                predicted_label_expand = label_list_expand[preds[index]]
                content = re.sub(r"Answer:\s*.+?\n", f"Answer: {predicted_label_expand}.\n", example["content"])
                example["content"] = content
            return example
    
        modify_test_dataset = test_dataset.map(
            lambda example, idx: update_content(example, idx),
            with_indices=True
        )

        valid_indices = [i for i, pred in enumerate(preds) if pred != 0]
        modify_test_dataset = modify_test_dataset.select(valid_indices)

        tokenizer = self.tokenizer
        args = self.args

        _multiple = isinstance(modify_test_dataset, dict)
        _test_datasets = modify_test_dataset if _multiple else {"singleton": modify_test_dataset}

        test_packing = False
        tokenizer.padding_side = "left" 
        for _test_dataset_name, _test_dataset in _test_datasets.items():
            _test_datasets[_test_dataset_name] = self._prepare_dataset(
                _test_dataset,
                tokenizer,
                test_packing,
                args.dataset_text_field,
                args.max_seq_length,
                None,
                args.num_of_sequences,
                args.chars_per_token,
                remove_unused_columns=args.remove_unused_columns if args is not None else True,
                **args.dataset_kwargs,
            )
        if not _multiple:
            modify_test_dataset = _test_datasets["singleton"]
        
        if len(modify_test_dataset) != 0:

            modify_test_dataset = modify_test_dataset.add_column("idx", [i for i in range(len(modify_test_dataset['input_ids']))])
            
            test_dataloader = self.get_eval_dataloader(modify_test_dataset)
            
            test_dataloader = self.accelerator.prepare(test_dataloader)

            if not os.path.exists(f"{self.args.output_dir}/ood_eval_scores/test_{eval_ablation}.pt1"):
            
                ans = {"attentions": {}, "hidden_states": {},"logit_entropy": {}}
                self.accelerator.wait_for_everyone()
                with torch.cuda.amp.autocast():
                    for index, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

                        unwrapped_model = self.accelerator.unwrap_model(model)
                        with torch.inference_mode():
                            generated_tokens = unwrapped_model.check_forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True, output_attentions=True)
                            
                            attentions = self.accelerator.gather_for_metrics(generated_tokens.attentions)
                            hidden_states = self.accelerator.gather_for_metrics(generated_tokens.hidden_states)
                            logits = self.accelerator.gather_for_metrics(generated_tokens.logits)
                            
                        self.accelerator.wait_for_everyone()

                        attn_cpu = attentions[-1].detach().cpu()
                        hidden_cpu = hidden_states[-1].detach().cpu()
                        logits_cpu = logits.detach().cpu()

                        ans["attentions"][index] = compute_attention_score(attn_cpu)
                        ans["hidden_states"][index] = compute_hidden_score(hidden_cpu)
                        ans["logit_entropy"][index] = compute_logit_entropy(logits_cpu, k=10)

                self.accelerator.wait_for_everyone()

                scores = {}
                for key in ans:
                    scores[key] = torch.concat([ans[key][i] for i in sorted(list(ans[key].keys()))], dim=0)

                if self.accelerator.is_main_process:
                    torch.save(scores, f"{self.args.output_dir}/ood_eval_scores/test_{eval_ablation}.pt")
            
            else:
                scores = torch.load(f"{self.args.output_dir}/ood_eval_scores/test_{eval_ablation}.pt", weights_only=False)
            for key in scores:
                if key != 'attentions':
                    scores[key] = 1 - (scores[key] - scores[key].min()) / (scores[key].max() - scores[key].min())
                else:
                    scores[key] = (scores[key] - scores[key].min()) / (scores[key].max() - scores[key].min())
            
            test_data = pd.read_csv(f'data/{self.dataset_name}/origin_data/test.tsv', sep='\t')
            test_data['label'] = test_data['label'].apply(lambda x: x.replace('_', ' '))
            test_data['label'] = test_data['label'].apply(lambda x: x if x in label_list[1:] else label_list[0])
            test_data['preds'] = [label_list[i] for i in preds]
            sub_df = test_data[:len(scores[key])]
            for key in scores:
                sub_df[f'{key}_scores'] = scores[key]
                sub_df[f'{key}_preds'] = sub_df.apply(lambda x: x['preds'] if x[f'{key}_scores'] < 0.5 else label_list[0], axis=1)
            
            sub_df.to_csv(f"{metrics_dir}/test-self_check_preds.csv")
            
            results = evaluate_predictions(sub_df.drop('text', axis=1), label_col='label', ood_class='OOD', test_columns=["attentions_preds", "hidden_states_preds", "logit_entropy_preds"])
            metric_df = pd.DataFrame(results).T
            metric_df.to_csv(f"{metrics_dir}/test-self_check.csv")


    def analogy(self, test_dataset, metrics_dir, checkpoint_dir=None):

        if checkpoint_dir is not None:
            checkpoint_path = get_latest_checkpoint(checkpoint_dir)
            self._load_from_checkpoint(checkpoint_path)

        model = self.model
        model.eval()
        model = self.accelerator.prepare(model)

        label_idx = pd.read_csv(f'data/{self.dataset_name}/{self.dataset_name}_{self.rate}.txt', header=None)[0].tolist()
        label_list = ['OOD'] + label_idx
        label_list_expand = ['000.It is OOD'] + [f"{i:03d}.{label_list[i]}" for i in range(1, len(label_list))]

        train_data = pd.read_csv(f'data/{self.dataset_name}/origin_data/train.tsv', sep='\t')
        eval_data = pd.read_csv(f'data/{self.dataset_name}/origin_data/eval.tsv', sep='\t')
        test_data = pd.read_csv(f'data/{self.dataset_name}/origin_data/test.tsv', sep='\t')

        df = pd.concat([train_data, eval_data])
        # df['label'] = df['label'].apply(lambda x: x.replace('_', ' '))
        df['idx'] = df['label'].apply(lambda x: label_list.index(x) if x in label_list else 0)
        

        if not os.path.exists(f"data/bert_embeddings/{self.dataset_name}/cosine_sim.pt"):
            os.makedirs(f'data/bert_embeddings/{self.dataset_name}', exist_ok=True)
            model_name = 'llms/bert-base-chinese' if self.dataset_name =='thucnews' else 'llms/bert-base-uncased'
            if os.path.exists(f"data/bert_embeddings/{self.dataset_name}/traineval_bert.pt"):
                bert_embeddings_traineval = torch.load(f"data/bert_embeddings/{self.dataset_name}/traineval_bert.pt")
                bert_embeddings_test = torch.load(f"data/bert_embeddings/{self.dataset_name}/test_bert.pt")
            else:
                bert_embeddings_traineval, bert_embeddings_test= obtain_bert_embedding(train_data, eval_data, test_data, model_name)
                torch.save(bert_embeddings_traineval, f"data/bert_embeddings/{self.dataset_name}/traineval_bert.pt")
                torch.save(bert_embeddings_test, f"data/bert_embeddings/{self.dataset_name}/test_bert.pt")

            eps=1e-8
            x1_norm = bert_embeddings_traineval / (bert_embeddings_traineval.norm(dim=1, keepdim=True) + eps)
            x2_norm = bert_embeddings_test / (bert_embeddings_test.norm(dim=1, keepdim=True) + eps)
            cosine_sim = torch.mm(x2_norm, x1_norm.T)
            torch.save(cosine_sim, f"data/bert_embeddings/{self.dataset_name}/cosine_sim.pt")
        else:
            cosine_sim = torch.load(f"data/bert_embeddings/{self.dataset_name}/cosine_sim.pt")

        class_golds = torch.load(f"{self.args.output_dir}/ood_eval_scores/test_golds.pt", weights_only=False)

        anology_metrics = {}
        for eval_ablation in self.eval_ablation_list:

            preds = torch.load(f"{self.args.output_dir}/ood_eval_scores/test_{eval_ablation}_preds.pt", weights_only=False)

            def update_content(example, index):
                if preds[index] != 0:
                    predicted_label_expand = label_list_expand[preds[index]]
                    content = re.sub(r"Answer:\s*.+?\n", f"Answer: {predicted_label_expand}.\n", example["content"])

                    df['cosine_sim'] = cosine_sim[index // self.segment_length].tolist()
                    sub_df = df[df['idx'] == preds[index].item()]
                    sub_df = sub_df.sort_values('cosine_sim', ascending=False)
                    examples = sub_df['text'].tolist()[:20]

                    new_content = content + f'<|im_start|>user\nRecall relevant exemplars of {predicted_label_expand}: {examples}.\nDoes the text strictly align with the specified scope of {predicted_label_expand}? Please start by answering Yes or No.\n<|im_end|>\n'
                    new_content += '<|im_start|>assistant\nAnswer: '
                    example["content"] = new_content
                return example
        
            modify_test_dataset = test_dataset.map(
                lambda example, idx: update_content(example, idx),
                with_indices=True
            )

            valid_indices = [i for i, pred in enumerate(preds) if pred != 0]
            modify_test_dataset = modify_test_dataset.select(valid_indices)

            tokenizer = self.tokenizer
            args = self.args

            _multiple = isinstance(modify_test_dataset, dict)
            _test_datasets = modify_test_dataset if _multiple else {"singleton": modify_test_dataset}

            test_packing = False
            tokenizer.padding_side = "left" 
            for _test_dataset_name, _test_dataset in _test_datasets.items():
                _test_datasets[_test_dataset_name] = self._prepare_dataset(
                    _test_dataset,
                    tokenizer,
                    test_packing,
                    args.dataset_text_field,
                    args.max_seq_length,
                    None,
                    args.num_of_sequences,
                    args.chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **args.dataset_kwargs,
                )
            if not _multiple:
                modify_test_dataset = _test_datasets["singleton"]
            
            if len(modify_test_dataset) != 0:

                modify_test_dataset = modify_test_dataset.add_column("idx", [i for i in range(len(modify_test_dataset['input_ids']))])
                
                test_dataloader = self.get_eval_dataloader(modify_test_dataset)
                
                test_dataloader = self.accelerator.prepare(test_dataloader)

                

                if not os.path.exists(f"{self.args.output_dir}/ood_eval_scores/test_{eval_ablation}_analogy.pt"):
                
                    ans = {}
                    self.accelerator.wait_for_everyone()
                    with torch.cuda.amp.autocast():
                        for index, batch in tqdm(enumerate(# The code `test_dataloader` is likely a
                        # variable or object being defined in
                        # Python. Without seeing the specific
                        # implementation of `test_dataloader`, it
                        # is not possible to determine exactly what
                        # it is doing. Typically, a variable with a
                        # name like `test_dataloader` might be used
                        # to store or represent a data loader
                        # object used for testing purposes in
                        # machine learning or data processing
                        # tasks.
                        test_dataloader), total=len(test_dataloader)):
                            
                            
                            unwrapped_model = self.accelerator.unwrap_model(model)
                            with torch.inference_mode():
                                generated_tokens = unwrapped_model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=2, return_dict_in_generate=True, output_scores=True)
                                
                                scores = self.accelerator.gather_for_metrics(generated_tokens.scores)

                                no_pro = scores[0][:, tokenizer.encode(' No')[-1]]
                                yes_pro = scores[0][:, tokenizer.encode(' Yes')[-1]]
                                logits = torch.concat([no_pro.unsqueeze(-1), yes_pro.unsqueeze(-1)], dim=-1)
                                
                            self.accelerator.wait_for_everyone()
                            ans[index] = logits

                    self.accelerator.wait_for_everyone()
                    
                    gathered_logits = torch.concat([ans[i].detach().cpu() for i in sorted(list(ans.keys()))], dim=0)

                    if self.accelerator.is_main_process:
                        torch.save(gathered_logits, f"{self.args.output_dir}/ood_eval_scores/test_{eval_ablation}_analogy.pt")
                
                else:
                    gathered_logits = torch.load(f"{self.args.output_dir}/ood_eval_scores/test_{eval_ablation}_analogy.pt")
                
                selected_logits = torch.load(f"{self.args.output_dir}/ood_eval_scores/test_selected_logits.pt", weights_only=False)
                sequence_scores = selected_logits.prod(axis=-2)
                test_softmax = torch.softmax(gathered_logits, dim=-1)

            for mask_rate in [0.5, 0.6, 0.7, 0.8, 0.9]:
                analogy_preds = copy.deepcopy(preds)
                if len(modify_test_dataset) != 0:
                    non_zero_indices = np.where(analogy_preds != 0)
                    mask = (test_softmax[:, 0] > mask_rate)
                    analogy_preds[non_zero_indices[0][mask]] = 0 
                anology_metrics[f"{eval_ablation}-{mask_rate}"] = self.compute_anology_metrics(analogy_preds, class_golds, sequence_scores)

        df_metrics = (pd.DataFrame(anology_metrics).T[['accuracy','f1-score','K-f1','N-f1']]*100).round(2)
        df_metrics = df_metrics.reset_index()
        df_metrics.columns = ['eval_ablation','ACC','F1','K-F1','N-F1']
        df_metrics['eval_ablation'] = df_metrics['eval_ablation'].apply(lambda x:x.replace('interfere-', '').replace('-lm', '').replace('-X', ''))
        df_metrics.to_csv(f'{metrics_dir}/{self.mode}.csv', sep='\t', index=None)

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        if tokenizer.name_or_path.split('/')[-1] in ['Meta-Llama-3.1-8B-Instruct', 'DeepSeek-R1-Distill-Llama-8B']:
            start_token = tokenizer.encode('\n\nQuestions: Given the')[2:]
            id_token = tokenizer.encode('\nAnswer:')[2:]
            length = len(tokenizer.encode('001')) - 1 if len(tokenizer.encode('001'))  > 2 else len(tokenizer.encode('001'))
        elif tokenizer.name_or_path.split('/')[-1] in ['Llama-3.2-8B-Instruct']:
            start_token = tokenizer.encode('\n\nQuestions: Given the')[2:]
            id_token = tokenizer.encode('\nAnswer: ')[2:]
            length = len(tokenizer.encode('001')) - 1
        elif tokenizer.name_or_path.split('/')[-1] in ['Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct']:
            start_token = tokenizer.encode('\n\nQuestions: Given the')[1:]
            id_token = tokenizer.encode('\nAnswer: ')[1:]
            length = len(tokenizer.encode('001'))

        def equal_list(x, y):
            if len(x) != len(y):
                return False
            for i,v in zip(x, y):
                if i != v:
                    return False
            return True

        def tokenize(element):
            if self.processor is not None:
                outputs = {"input_ids": [], 'attention_mask': [], 'pixel_values': [], 'image_grid_thw': []}
                for text, image in zip(element[dataset_text_field], element['image_inputs']):
                    records = self.processor(
                        text=text,
                        images=[image],
                        padding=False,
                        max_length=max_seq_length,
                    ).data
                    outputs['input_ids'].append(records['input_ids'][0])
                    outputs['attention_mask'].append(records['attention_mask'][0])
                    outputs['pixel_values'].append(records['pixel_values'])
                    outputs['image_grid_thw'].append(records['image_grid_thw'][0])
            else:
                outputs = tokenizer(
                    element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    padding=False,
                    max_length=max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                )

            token_type_ids = [[0 for j in i] for i in outputs["input_ids"]]
            
            for i in range(len(token_type_ids)):
                tag = False
                for j in range(len(token_type_ids[i])):
                    if j+len(start_token) < len(token_type_ids[i]) and equal_list(outputs['input_ids'][i][j:j+len(start_token)], start_token):
                        tag = True

                    if tag and equal_list(outputs['input_ids'][i][j-len(id_token):j], id_token):
                        for k in range(j, len(token_type_ids[i])):
                            token_type_ids[i][k] = 1
                        for k in range(j, j + length):
                            token_type_ids[i][k] = 3
                        tag = False

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True
            
            if self.processor is None:
                return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "token_type_ids": token_type_ids, "labels":  element['labels']}
            else:
                return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "token_type_ids": token_type_ids, "labels":  element['labels'], "pixel_values": outputs['pixel_values'], "image_grid_thw": outputs['image_grid_thw']}

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names,
            "batch_size": self.dataset_batch_size,
        }
        if isinstance(dataset, datasets.Dataset):
            map_kwargs["num_proc"] = self.dataset_num_proc  
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)
        return tokenized_dataset
    
    def test(self, output_dir=None, metrics_dir=None):
        if output_dir is not None:
            checkpoint_path = get_best_checkpoint(output_dir)
            # if checkpoint_path is not None:
            self._load_from_checkpoint(checkpoint_path)
        with torch.no_grad():
            metrics = self.evaluate(eval_dataset=self.test_dataset if 'test' in self.mode else self.eval_dataset)
        
        json.dump(metrics, open(f"{metrics_dir}/{self.mode}.json", "w"))
        
        ans = {}
        for k,v in metrics.items():
            if len(k.split('--')) > 1:
                metric, suffix = k.split('--')
                metric = '_'.join(metric.split('_')[1:])
                if suffix not in ans:
                    ans[suffix] = {}
                ans[suffix][metric] = v
        df = pd.DataFrame(ans).T.reset_index().round(4)
        for k,v in metrics.items():
            if len(k.split('--')) == 1:
                k = '_'.join(k.split('_')[1:])
                df[k] = v
        df['index'] = df['index'].apply(lambda x:x.replace('interfere-', '').replace('-lm', '').replace('-X', ''))
        sub_df = df[['index', 'accuracy', 'f1-score', 'K-f1', 'N-f1']]
        sub_df.columns = ['eval_ablation', 'ACC', 'F1', 'K-F1', 'N-F1']
        for col in sub_df.columns[-4:]:
            sub_df[col] = sub_df[col]*100
        sub_df = sub_df.round(2)
        sub_df.to_csv(f"{metrics_dir}/{self.mode}.csv", sep='\t', index=None)

    def save_train_embedding(self, output_dir=None, metrics_dir=None):
        if output_dir is not None:
            checkpoint_path = get_best_checkpoint(output_dir)
            self._load_from_checkpoint(checkpoint_path)
        with torch.no_grad():
            output = self.evaluate_train(eval_dataset=self.train_dataset)
        json.dump(output.metrics, open(f"{metrics_dir}/{self.mode}.json", "w"))

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_ablation_list: Optional[List[str]] = None,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        eval_ablation_list = eval_ablation_list if eval_ablation_list is not None else self.eval_ablation_list

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            
            if model is not self.model:
                self.model_wrapped = model

            
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        
        
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None

        
        observed_num_examples = 0

        
        for step, inputs in enumerate(dataloader):
            
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                
                if batch_size is None:
                    batch_size = observed_batch_size

            
            
            if self.mode == 'eval' and 'interfere' in eval_ablation_list: 
                inputs['output_hidden_states'] = True
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_xla_available():
                xm.mark_step()

            
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))

            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)
            
            if logits is not None:
                if self.process_gather_logits is not None:
                    logits = self.process_gather_logits(logits, labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            
            delattr(self, "_past")

        
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        
        
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs), eval_ablation_list=eval_ablation_list)
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), eval_ablation_list=eval_ablation_list)
        elif metrics is None:
            metrics = {}

        
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def save_propotype(self):
        
        
        torch.save(self.model.propotype, self.propotype_path)

    def evaluate_train(
        self,
        eval_dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        train_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            train_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            eval_ablation_list=['binary']
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        return output

    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval, metric_key_prefix='eval')
        
        
        
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not skip_scheduler:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics[metric_to_check])
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
        return metrics