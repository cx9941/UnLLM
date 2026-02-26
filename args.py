from dataclasses import dataclass, field
from typing import Optional
import argparse
import os
import time

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="Meta-Llama-3-8B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    mode: str = field(
        default='train',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    gpu_id: str = field(
        default='3',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="demo",
        metadata={"help": "The preference dataset to use."},
    )
    rate: Optional[float] = field(
        default=0.25,
        metadata={"help": "The known proportion."},
    )
    segment_length: Optional[int] = field(
        default=2,
        metadata={"help": "The known proportion."},
    )
    vos_neg_sample: Optional[int] = field(
        default=2,
        metadata={"help": "The known proportion."},
    )
    epsilon: Optional[int] = field(
        default=2,
        metadata={"help": "The known proportion."},
    )
    e_dim: Optional[int] = field(
        default=2,
        metadata={"help": "The known proportion."},
    )
    neg_sample: Optional[int] = field(
        default=2,
        metadata={"help": "The known proportion."},
    )
    random_sample: Optional[int] = field(
        default=1,
        metadata={"help": "The known proportion."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,valid,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    train_ablation: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    con_weight: Optional[float] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    neg_weight: Optional[float] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    temperature: Optional[float] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    metric_dir: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    gen_dir: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="clinc")
parser.add_argument("--rate", default="0.25")
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--con_weight", default=0.1, type=float)
parser.add_argument("--neg_weight", default=0.1, type=float)
parser.add_argument("--model_name_or_path", default="Meta-Llama-3.1-8B-Instruct", choices=['Meta-Llama-3.1-8B', 'Meta-Llama-3-8B', 'Meta-Llama-2-7b-hf', 'Meta-Llama-3.1-8B-Instruct', 't5-large', 'DeepSeek-R1-Distill-Llama-8B', 'Llama-3.2-8B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct'])
parser.add_argument("--default_config", default="configs/args.json")
parser.add_argument("--metric_for_best_model", default="OOD-N_recall--binary")
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--per_device_train_batch_size", default=2, type=int)
parser.add_argument("--per_device_eval_batch_size", default=4, type=int)
parser.add_argument("--seed", default=5, type=int)
parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
parser.add_argument("--eval_step_num", default=2, type=int)
parser.add_argument("--mode", default="test-analogy", choices=['train', 'test', 'eval', 'eval-ood', 'test-ood', 'test-analogy', 'train-ood', 'test-self_check'], type=str)
parser.add_argument("--report_to", default="wandb", type=str)
parser.add_argument("--run_name", default="ACL25-LLM4Open", type=str)
parser.add_argument("--train_ablation", default="binary", type=str, choices=['binary', 'retrival', 'nosft'])
parser.add_argument("--shot_num", default=-1, type=int)
parser.add_argument("--epsilon", default=2, type=int)
parser.add_argument("--e_dim", default=20, type=int)
parser.add_argument("--vos_neg_sample", default=20, type=int)
parser.add_argument("--segment_length", default=1, type=int)
parser.add_argument("--neg_sample", default=0, type=int)
parser.add_argument("--random_sample", default=1, type=int)
parser.add_argument("--gpu_id", default="3", type=str)
parser.add_argument("--use_flash_attn", default="yes", type=str)
parser.add_argument("--dataloader_drop_last", default=False, type=bool)
parser.add_argument("--remove_unused_columns", default=False, type=bool)

custom_args = parser.parse_args()
import json

metric_for_best_model_map = {
    "0.25": "OOD-N_recall--binary",
    "0.5": "precision--binary",
    "0.75": "precision--binary",
}

with open(custom_args.default_config, 'r') as r:
    config_content = json.load(r)

custom_args.config_file = f"configs/json/{custom_args.dataset_name}/{custom_args.dataset_name}-{custom_args.rate}-{custom_args.shot_num}/{custom_args.model_name_or_path}-{custom_args.random_sample}-{custom_args.segment_length}-{custom_args.neg_sample}-{custom_args.vos_neg_sample}-{custom_args.train_ablation}_{custom_args.neg_weight}_{custom_args.con_weight}_{custom_args.temperature}-sft-qlora-dsz3-seed_{custom_args.seed}-{custom_args.epsilon}-{custom_args.e_dim}.json"
sample_num = json.load(open('data/statics/sample_num.json', 'r'))

os.makedirs(f"configs/json/{custom_args.dataset_name}/{custom_args.dataset_name}-{custom_args.rate}-{custom_args.shot_num}", exist_ok=True)

with open(custom_args.config_file, 'w') as w:
    config_content['dataset_name'] = custom_args.dataset_name
    config_content['rate'] = custom_args.rate
    config_content['output_dir'] = f"outputs/{custom_args.dataset_name}/{custom_args.dataset_name}-{custom_args.rate}-{custom_args.shot_num}/{custom_args.model_name_or_path}-{custom_args.random_sample}-{custom_args.segment_length}-{custom_args.neg_sample}-{custom_args.vos_neg_sample}-{custom_args.train_ablation}_{custom_args.neg_weight}_{custom_args.con_weight}_{custom_args.temperature}-sft-qlora-dsz3-seed_{custom_args.seed}-{custom_args.epsilon}-{custom_args.e_dim}"
    config_content['metric_dir'] = f"metrics_results/{custom_args.dataset_name}/{custom_args.dataset_name}-{custom_args.rate}-{custom_args.shot_num}/{custom_args.model_name_or_path}-{custom_args.random_sample}-{custom_args.segment_length}-{custom_args.neg_sample}-{custom_args.vos_neg_sample}-{custom_args.train_ablation}_{custom_args.neg_weight}_{custom_args.con_weight}_{custom_args.temperature}-sft-qlora-dsz3-seed_{custom_args.seed}-{custom_args.epsilon}-{custom_args.e_dim}"
    config_content['gen_dir'] = f"gen_results/{custom_args.dataset_name}/{custom_args.dataset_name}-{custom_args.rate}-{custom_args.shot_num}/{custom_args.model_name_or_path}-{custom_args.random_sample}-{custom_args.segment_length}-{custom_args.neg_sample}-{custom_args.vos_neg_sample}-{custom_args.train_ablation}_{custom_args.neg_weight}_{custom_args.con_weight}_{custom_args.temperature}-sft-qlora-dsz3-seed_{custom_args.seed}-{custom_args.epsilon}-{custom_args.e_dim}"
    config_content['model_name_or_path'] = f"llms/{custom_args.model_name_or_path}"    
    config_content['per_device_eval_batch_size'] = custom_args.per_device_eval_batch_size   
    config_content['per_device_train_batch_size'] = custom_args.per_device_train_batch_size    
    config_content['seed'] = custom_args.seed    
    config_content['mode'] = custom_args.mode
    config_content['gradient_accumulation_steps'] = custom_args.gradient_accumulation_steps
    config_content['metric_for_best_model'] = 'f1-score--binary' if custom_args.neg_sample == 0 else metric_for_best_model_map[config_content['rate']]
    config_content['load_best_model_at_end'] = custom_args.load_best_model_at_end
    config_content['eval_strategy'] = 'steps'
    config_content['save_strategy'] = 'steps'
    config_content['train_ablation'] = custom_args.train_ablation
    config_content['con_weight'] = custom_args.con_weight
    config_content['neg_weight'] = custom_args.neg_weight
    config_content['temperature'] = custom_args.temperature
    config_content['segment_length'] = custom_args.segment_length
    config_content['neg_sample'] = custom_args.neg_sample
    config_content['random_sample'] = custom_args.random_sample
    config_content['vos_neg_sample'] = custom_args.vos_neg_sample
    config_content['epsilon'] = custom_args.epsilon
    config_content['e_dim'] = custom_args.e_dim
    config_content['use_flash_attn'] = custom_args.use_flash_attn == 'yes' if 'self_check' not in custom_args.mode else False
    
    config_content['eval_steps'] = max(sample_num[custom_args.dataset_name][custom_args.rate]['train'] // (custom_args.per_device_train_batch_size * custom_args.gradient_accumulation_steps * len(custom_args.gpu_id.split(',')) * custom_args.eval_step_num), 5)
    # config_content['eval_steps'] = 2
    config_content['save_steps'] = max(sample_num[custom_args.dataset_name][custom_args.rate]['train'] // (custom_args.per_device_train_batch_size * custom_args.gradient_accumulation_steps * len(custom_args.gpu_id.split(',')) * custom_args.eval_step_num), 5)
    if custom_args.shot_num != -1:
        config_content['splits'] =  f"shot_{custom_args.shot_num}/train,shot_{custom_args.shot_num}/eval,test"
        config_content['eval_strategy'] = 'epoch'
        config_content['num_train_epochs'] = 10
    if custom_args.rate == "0.25":
        config_content['num_train_epochs'] = 1
    
    config_content['remove_unused_columns'] = custom_args.remove_unused_columns

    if custom_args.mode == 'train':
        
        os.environ["WANDB_DISABLED"]="true"
    else:
        os.environ["WANDB_DISABLED"]="true"

    json.dump(config_content, w)
    w.flush()
    time.sleep(10)

if custom_args.mode in ['train']:
    os.makedirs(f"{config_content['output_dir']}/ood_eval_scores", exist_ok=True)

if custom_args.mode in ['eval', 'test', 'test-analogy', 'train-ood', 'eval-ood', 'test-ood']:
    os.makedirs(f"{config_content['metric_dir']}", exist_ok=True)

if custom_args.mode == 'gen':
    os.makedirs(f"{config_content['gen_dir']}", exist_ok=True)