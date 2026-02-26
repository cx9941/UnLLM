from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

@dataclass
class SelfDataCollator():
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    con_weight: Optional[float] = 0
    temperature: Optional[float] = 1
    e_dim: Optional[int] = 1
    epsilon: Optional[int] = 2
    neg_weight: Optional[float] = 0
    vos_neg_sample: Optional[float] = 0
    num_labels: Optional[int] = 0
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        unpadable_keys = {"pixel_values"}
        filtered_examples = [
            {k: v for k, v in example.items() if k not in unpadable_keys} for example in examples
        ]
        if isinstance(examples[0], Mapping):
            # batch = pad_without_fast_tokenizer_warning(
                # self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            # )
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, filtered_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if "pixel_values" in examples[0]:
                batch['pixel_values'] = torch.concat([torch.tensor(i['pixel_values']) for i in examples], dim=0)

        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }


        labels = batch.pop('labels')

        if self.tokenizer.name_or_path.split('/')[-1] in ['Meta-Llama-3.1-8B-Instruct']:
            class_token = torch.tensor([[self.tokenizer.encode(f' {i:03d}')[1:] for i in range(self.num_labels + 1)] for _ in range(len(labels))], device=labels.device)
        elif self.tokenizer.name_or_path.split('/')[-1] in ['Llama-3.2-8B-Instruct']:
            class_token = torch.tensor([[self.tokenizer.encode(f'{i:03d}')[1:] for i in range(self.num_labels + 1)] for _ in range(len(labels))], device=labels.device)
        elif self.tokenizer.name_or_path.split('/')[-1] in ['Qwen2.5-7B-Instruct']:
            class_token = torch.tensor([[self.tokenizer.encode(f' {i:03d}')[1:] for i in range(self.num_labels + 1)] for _ in range(len(labels))], device=labels.device)
        elif self.tokenizer.name_or_path.split('/')[-1] in ['Qwen2.5-VL-7B-Instruct']:
            class_token = torch.tensor([[self.tokenizer.encode(f' {i:03d}')[1:] for i in range(self.num_labels + 1)] for _ in range(len(labels))], device=labels.device)

        batch['labels'] = batch['input_ids'].clone()
        batch['class_token'] = class_token
        batch['con_weight'] = torch.ones(labels.shape) * self.con_weight
        batch['e_dim'] = torch.ones(labels.shape) * self.e_dim
        batch['epsilon'] = torch.ones(labels.shape) * self.epsilon
        batch['neg_weight'] = torch.ones(labels.shape) * self.neg_weight
        batch['vos_neg_sample'] = torch.ones(labels.shape) * self.vos_neg_sample
        batch['temperature'] = torch.ones(labels.shape) * self.temperature
        A = batch['labels'][batch['token_type_ids']==3].reshape(batch['labels'].shape[0], -1)
        matches = torch.all(A[:, np.newaxis, :] == class_token, dim=2).int() 
        batch['class_golds'] = torch.argmax(matches, dim=1).to(labels.device)
        assert (matches.sum(dim=-1)!=1).sum().item() == 0
        return batch