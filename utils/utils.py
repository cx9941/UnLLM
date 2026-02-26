import json
import os
from enum import Enum
from datasets import Dataset
import pandas as pd
import torch
from tqdm import tqdm
tqdm.pandas()
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import numpy as np
from peft import LoraConfig
import re
import random
from sklearn.metrics import classification_report
import torch.nn.functional as F
import copy
from qwen_vl_utils import process_vision_info
DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
from modelscope import AutoProcessor

class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

def format_nested_dict(d, indent=0):
    formatted_string = ""
    for key, value in d.items():
        formatted_string += "  " * indent + f"{key}:\n"
        if isinstance(value, dict):
            formatted_string += format_nested_dict(value, indent + 1)
        elif isinstance(value, list):
            for item in value:
                formatted_string += "  " * (indent + 1) + f"- {item}\n"
    return formatted_string

def generate_prompt(text, label_candidates, label, candidate_explanation=None, candidate_class=None, level=['root'], segment_length = 2, neg_sample=0, apply_chat_template=False, split='train'):
    prompt_start = f"You are an expert in classification with a focus on Out-of-Distribution (OOD) detection. Your task is to accurately classify a given piece of text/image into one of the provided categories only if it strictly matches the meaning and scope of the category definition. If the text/image does not match any category definition, it should be identified as “Out-of-Distribution (OOD)\n"
    prompt_start += "text/image to Classify:\n"
    label_candidates = ['It is OOD'] + label_candidates

    def gen(candidates, label):
        candidates_id = ["000.It is OOD"] + [f"{j:03d}.{v}" for j, v in enumerate(label_candidates) if v in candidates]
        candidates_id = random.sample(candidates_id, len(candidates_id))
        content = f"\n\nQuestions: Given the candidate categories: {candidates_id}, which category does the text/image belong to?\n"
        if label not in candidates:
            label = "It is OOD"
        idx = label_candidates.index(label)
        content += f"Answer: " + f"{idx:03d}.{label}."
        return content, idx
    
    ood_content = []
    pos_content_levels = [prompt_start + text]

    pos_candidates = copy.deepcopy(label_candidates)
    
    new_content_gen = []
    content_labels = []
    if candidate_class == None:
        random.shuffle(pos_candidates)
    else:
        candidate_class = [_.replace('_', ' ') for _ in candidate_class]
        new_pos_candidates = [_ for _ in candidate_class if _ in pos_candidates]
        assert set(pos_candidates) == set(new_pos_candidates)
        pos_candidates = new_pos_candidates
    
    page_size = (len(pos_candidates) - 1) // segment_length + 1

    candidates_list = [pos_candidates[s*page_size: (s+1)*page_size] if s != segment_length - 1 else pos_candidates[-page_size:] for s in range(segment_length)]
        
    
    for candidates in candidates_list:
        content_gen, idx = gen(candidates, label)
        new_content_gen.append(content_gen)
        content_labels.append(idx)
    pos_content_levels = [key + value for key in pos_content_levels for value in new_content_gen]

    ood_content += pos_content_levels

    if 'train' in split or 'eval' in split:
        if neg_sample > 0:
            neg_content_levels = [prompt_start + text]
            neg_candidates = [i for i in pos_candidates if i != level[-1]]
            new_content_gen = []
            for _ in range(neg_sample):
                random.shuffle(neg_candidates)
                
                candidates_list = [neg_candidates[s*page_size: (s+1)*page_size] if s != segment_length - 1 else neg_candidates[-page_size:] for s in range(segment_length)]

                
                for candidates in candidates_list:
                    content_gen, idx = gen(candidates, label)
                    new_content_gen.append(content_gen)
                    content_labels.append(idx)
            neg_content_levels = [key + value for key in neg_content_levels for value in new_content_gen]
            ood_content += neg_content_levels

    return ood_content, content_labels

def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    # if hasattr(tokenizer, "name_or_path"):
    #     processor = tokenizer
    #     tokenizer = processor.tokenizer
    is_multimodal = not hasattr(tokenizer, "name_or_path")
    raw_datasets = DatasetDict()
    label_candidates = pd.read_csv(f'data/{data_args.dataset_name}/{data_args.dataset_name}_{data_args.rate}.txt', header=None)[0].tolist()
    
    
    candidate_explanation = None
    assert data_args.train_ablation in ['retrival', 'binary', 'nosft']
    if data_args.train_ablation in ['retrival', 'nosft']:
        candidate_class = pd.read_csv(f'data/{data_args.dataset_name}/origin_data/candidate_class.tsv', sep='\t')
        candidate_class['candidates'] = candidate_class['candidates'].apply(eval)
        candidate_class = {i:v for i,v in zip(candidate_class['text'], candidate_class['candidates'])}
    else:
        candidate_class = None
    split_set = data_args.splits.split(",")
    for split in split_set:
        df = pd.read_csv(f'data/{data_args.dataset_name}/origin_data/{split}.tsv', sep='\t')
        if split in ['train', 'eval']:
            df = df[df['label'].isin(label_candidates)]
        
        print("max the text length")
        df['level'] = df.progress_apply(lambda x: [x[col] for col in df.columns if 'l' in col], axis=1)
        df['candidate_class'] = df['text'].apply(lambda x: candidate_class[x] if candidate_class != None else None)
        df['text'] = df['text'].progress_apply(lambda x: tokenizer.decode(tokenizer(x)['input_ids'][:512], skip_special_tokens=True) if not is_multimodal else x)
        df['mode'] = split

        random_num = data_args.random_sample if split == 'train' else 1
        sum_df = []
        for i in range(random_num):
            df[[f'content', f'labels']] = df.progress_apply(lambda x: generate_prompt(x['text'], label_candidates, x['label'], candidate_explanation, segment_length=data_args.segment_length, neg_sample=data_args.neg_sample, apply_chat_template=apply_chat_template, split=split), axis=1, result_type='expand')
            sum_df.append(df)

        df = pd.concat(sum_df)
        df = df.explode(['content', 'labels'])

        final_df = {i:df[i].tolist() for i in df.columns if i not in ['l0', 'candidate_class']}
        dataset = Dataset.from_dict(final_df)

        assert [int(re.findall(r'which category does the text/image belong to\?\nAnswer: (.*?)\.', i)[0]) for i in dataset['content']] == dataset['labels']
        
        raw_datasets[split.split('/')[-1]] = dataset
    
    def map_process(samples):
        batch = []
        image_inputs = []

        for idx, conversation in enumerate(samples["content"]):
            system_prompt = "You are an expert in classification with a focus on Out-of-Distribution (OOD) detection. Your task is to accurately classify a given piece of text/image into one of the provided categories only if it strictly matches the meaning and scope of the category definition. If the text/image does not match any category definition, it should be identified as “Out-of-Distribution (OOD)\n"
            tag_seq = 'Answer:'
            question = conversation[conversation.find(system_prompt)+len(system_prompt):conversation.find(tag_seq)]
            answer = conversation[conversation.find(tag_seq):]
            if not is_multimodal:
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            else:
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": samples['text'][idx],
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                    {"role": "assistant", "content": answer},
                ]
                images, _ = process_vision_info(conversation)
                image_inputs.append(images)

            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        if not is_multimodal:
            return {"content": batch}
        else:
            return {"content": batch, "image_inputs": image_inputs}
    
    if apply_chat_template:
        raw_datasets = raw_datasets.map(
                map_process,
                batched=True,
            )
    
    for split in ['train', 'eval', 'test']:
        raw_datasets[split] = raw_datasets[split] if split in raw_datasets else None

    print(f"Size of the train set: {len(raw_datasets['train'])}. Size of the validation set: {len( raw_datasets['eval'])}")
    print(f"A sample of train dataset: {raw_datasets['train'][0]}")
    return raw_datasets

def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )

        if 'llama' in args.model_name_or_path.lower():

            from models.modeling_llama import LlamaForCausalLM
            
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )
        elif 'qwen2.5-vl' in args.model_name_or_path.lower():
            from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )

        elif 'qwen2' in args.model_name_or_path.lower():
            from models.modeling_qwen2 import Qwen2ForCausalLM
            
            model = Qwen2ForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )

        
    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE



    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token


    if 'qwen2.5-vl' in args.model_name_or_path.lower():
        
        tokenizer = AutoProcessor.from_pretrained("llms/Qwen2.5-VL-7B-Instruct")
        # tokenizer = processor.tokenizer

        # 只对 tokenizer 设置 special tokens
        if special_tokens is not None:
            tokenizer.tokenizer.pad_token = special_tokens.pad_token.value
            tokenizer.tokenizer.bos_token = special_tokens.bos_token.value
            tokenizer.tokenizer.eos_token = special_tokens.eos_token.value
            tokenizer.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens.list()})
            tokenizer.tokenizer.chat_template = chat_template

        # resize
        model.resize_token_embeddings(len(tokenizer.tokenizer), pad_to_multiple_of=8)

    if args.use_unsloth:
        
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer


def load_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.output_dir,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer


def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if re.match(r'checkpoint-\d+', d)]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
    latest_checkpoint = os.path.join(output_dir, checkpoints[0])
    return latest_checkpoint

def get_best_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if re.match(r'checkpoint-\d+', d)]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=False)
    latest_checkpoint = os.path.join(output_dir, checkpoints[0])
    return latest_checkpoint

import torch

def compute_hidden_score(hidden_states):
    """
    高效计算每个样本的 Hidden Score（使用对称特征值分解）
    :param hidden_states: Tensor, shape (B, L, d)
    :return: Tensor, shape (B,)
    """
    B, L, d = hidden_states.shape
    hs_T = hidden_states.transpose(1, 2)  # (B, d, L)
    cov_matrices = hs_T @ hidden_states  # (B, d, d)

    # 用 eigvalsh 替代 SVD
    eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, d)
    safe_eigenvalues = torch.clamp(eigenvalues, min=1e-6)
    log_det = torch.sum(torch.log(safe_eigenvalues), dim=1)

    # 计算 logdet
    # log_det = torch.sum(torch.log(eigenvalues + 1e-6), dim=1)
    hidden_scores = log_det / L

    return hidden_scores

def compute_attention_score(attention_map):
    """
    为每个样本计算 Attention Score（无 for-loop）
    :param attention_map: Tensor, shape (B, H, L, L)
    :return: Tensor, shape (B,)
    """
    B, H, L, _ = attention_map.shape

    # 取每个注意力矩阵的对角线: shape -> (B, H, L)
    diagonal_entries = torch.diagonal(attention_map, dim1=-2, dim2=-1)  # (B, H, L)

    # 避免 log(0)，加 epsilon
    log_diag = torch.log(diagonal_entries + 1e-6)  # (B, H, L)

    # 对每个 head 的对角线求和，再对所有 head 求平均
    # (B, H, L) -> (B, H) -> (B,)
    log_det = log_diag.sum(dim=-1).mean(dim=-1)  # 按 L 再按 H 平均

    # 归一化（除以 L）
    attention_scores = log_det / L

    return attention_scores  # shape: (B,)

def compute_perplexity(logits, targets):
    """
    计算序列困惑度 (PPL)
    :param logits: 模型输出 logits, shape (batch_size, seq_len, vocab_size)
    :param targets: 实际目标序列, shape (batch_size, seq_len)
    :return: PPL Score
    """
    log_probs = F.log_softmax(logits, dim=-1)
    target_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    avg_log_prob = target_probs.mean(dim=-1)  # 平均 log 概率
    perplexity = torch.exp(-avg_log_prob).mean().item()
    return perplexity

def compute_logit_entropy(logits, k):
    """
    为每个样本计算 Logit 熵（只考虑前 k 个高概率 token）
    :param logits: Tensor, shape (B, L, V)
    :param k: int, top-k token
    :return: Tensor, shape (B,), 每个样本的 logit entropy
    """
    # logits: (B, L, V)
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=-1)  # (B, L, V)

    # 取前 k 的概率值: (B, L, k)
    topk_probs, _ = torch.topk(probs, k=k, dim=-1)

    # 避免 log(0)
    log_topk_probs = torch.log(topk_probs + 1e-10)

    # 计算熵: (B, L)
    entropy_per_token = - (topk_probs * log_topk_probs).sum(dim=-1)

    # 平均所有 token 熵: (B,)
    entropy_per_sample = entropy_per_token.mean(dim=-1)

    return entropy_per_sample  # shape: (B,)

from sklearn.metrics import f1_score, accuracy_score

def evaluate_predictions(df, label_col='label', ood_class='OOD', test_columns=[]):
    results = {}
    label_true = df[label_col]
    
    for col in test_columns:
        if col == label_col:
            continue

        pred = df[col]

        # Overall metrics
        acc = accuracy_score(label_true, pred)
        f1 = f1_score(label_true, pred, average='macro', zero_division=0)

        # Mask for OOD
        is_ood = label_true == ood_class
        is_known = ~is_ood

        # K-F1 (known classes only)
        k_f1 = f1_score(label_true[is_known], pred[is_known], average='macro', zero_division=0)

        # N-F1 (OOD class only, treated as binary classification)
        ood_true_binary = label_true == ood_class
        ood_pred_binary = pred == ood_class
        n_f1 = f1_score(ood_true_binary, ood_pred_binary, average='binary', zero_division=0)

        results[col] = {
            'ACC': acc,
            'F1': f1,
            'K-F1': k_f1,
            'N-F1': n_f1
        }
    
    return results