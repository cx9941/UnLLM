from args import custom_args
import os
os.environ["WANDB_PROJECT"] = "NeurIPS25"


from transformers import HfArgumentParser, TrainingArguments, set_seed
from trainer import OHTC_Trainer
from utils.utils import create_and_prepare_model, create_datasets, get_latest_checkpoint
from utils.metric import Metrics
from args import ModelArguments, DataTrainingArguments
from utils.dataset_utils import SelfDataCollator
import pandas as pd

def main(model_args, data_args, training_args):
    if data_args.dataset_name != 'demo' and 'self_check' not in model_args.mode:
        # if os.path.exists(f"{data_args.metric_dir}/{model_args.mode}.csv") or os.path.exists(f"{data_args.metric_dir}/{model_args.mode}.json"):
        #     print(f"This model has been trained and tested on {model_args.mode}")
        #     exit()

        if model_args.mode == 'train' and os.path.exists(f"{data_args.metric_dir}/eval.csv"):
            print("This model has been trained")
            exit()

        # if model_args.mode == 'train' and get_latest_checkpoint(training_args.output_dir):
        #     print("This model has been trained")
        #     exit()
    
    set_seed(training_args.seed)
    print(training_args)
    print(data_args)
    print(model_args)

    data_args.num_labels = len(pd.read_csv(f'data/{data_args.dataset_name}/{data_args.dataset_name}_{data_args.rate}.txt', header=None))

    
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}
    
    datasets = create_datasets(tokenizer,data_args,training_args,apply_chat_template= 'Instruct' in model_args.model_name_or_path)
    
    print('Data Load Over')
    metric = Metrics(data_args, model_args, training_args)
    print('Metric Load Over')
    trainer = OHTC_Trainer(
        model=model,
        num_class=len(metric.id_list),
        model_name_or_path=model_args.model_name_or_path,
        mode=model_args.mode,
        processor=None if hasattr(tokenizer, 'name_or_path') else tokenizer,
        tokenizer=tokenizer if hasattr(tokenizer, 'name_or_path') else tokenizer.tokenizer,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'] if model_args.mode == 'train' else datasets['eval'],
        # eval_dataset= datasets['eval'],
        test_dataset=datasets['test'],
        peft_config=peft_config,
        packing=data_args.packing,
        compute_metrics=metric.compute_metrics,
        compute_anology_metrics=metric.compute_anology_metrics,
        process_gather_logits=metric.process_gather_logits,
        preprocess_logits_for_metrics=metric.preprocess_logits_for_metrics,
        data_collator=SelfDataCollator(tokenizer if hasattr(tokenizer, 'name_or_path') else tokenizer.tokenizer, con_weight=data_args.con_weight, e_dim=data_args.e_dim, epsilon=data_args.epsilon,temperature=data_args.temperature, neg_weight=data_args.neg_weight, vos_neg_sample=data_args.vos_neg_sample, num_labels=data_args.num_labels).torch_call,
        dataset_kwargs={
            "append_concat_token": data_args.append_concat_token,
            "add_special_tokens": data_args.add_special_tokens,
        },
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        dataset_name=data_args.dataset_name,
        rate=data_args.rate,
        train_ablation=data_args.train_ablation,
        segment_length=data_args.segment_length,
        neg_sample=data_args.neg_sample,
        propotype_path=f"{training_args.output_dir}/ood_eval_scores/prototypes.pt"
    )

    print('Trainer Ready')
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    if model_args.mode == 'train':
        trainer.train()
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model()
        print('Train Over')
    elif model_args.mode in ['eval', 'test', 'eval-ood', 'test-ood']:
        
        if data_args.train_ablation == 'nosft':
            trainer.test(None, data_args.metric_dir)
        else:
            trainer.test(training_args.output_dir, data_args.metric_dir)

    elif model_args.mode == 'train-ood':
        trainer.save_train_embedding(training_args.output_dir, data_args.metric_dir)

    elif model_args.mode == 'test-analogy':
        for eval_ablation in trainer.eval_ablation_list:
            os.makedirs(f"{data_args.gen_dir}/{eval_ablation}", exist_ok=True)
        trainer.analogy(datasets['test'], data_args.metric_dir)
    elif model_args.mode == 'test-self_check':
        trainer.self_check(datasets['test'], data_args.metric_dir, training_args.output_dir)
    else:
        assert False

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=custom_args.config_file)
    main(model_args, data_args, training_args)
