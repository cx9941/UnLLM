set -o errexit
gpu_id='0'
export CUDA_VISIBLE_DEVICES=$gpu_id
config_file='configs/yaml/1-deepspeed_config_z3_qlora.yaml'

per_device_train_batch_size=4
per_device_eval_batch_size=8

vos_neg_sample=20
neg_weight=0.0
con_weight=0.1
seed=5
segment_length=2
random_sample=1
neg_sample=0
model_name_or_path='Meta-Llama-3.1-8B-Instruct'
train_ablation='binary'
temperature=1.0
e_dim=20
epsilon=2

for dataset_name in clinc
do
for rate in 0.25 0.5 0.75
do
for segment_length in $segment_length
do
python  main.py --dataset_name $dataset_name --rate $rate --model_name_or_path $model_name_or_path --per_device_train_batch_size $per_device_train_batch_size --mode 'train' --seed $seed --gpu_id  $gpu_id --train_ablation ${train_ablation} --con_weight ${con_weight} --temperature $temperature --neg_sample $neg_sample --segment_length $segment_length --neg_weight $neg_weight --vos_neg_sample $vos_neg_sample --random_sample $random_sample --epsilon $epsilon --e_dim $e_dim
for mode in 'eval' 'test'
do
python main.py --dataset_name $dataset_name  --rate $rate  --model_name_or_path $model_name_or_path  --per_device_eval_batch_size $per_device_eval_batch_size  --mode $mode --seed $seed --gpu_id  $gpu_id --train_ablation ${train_ablation} --con_weight ${con_weight} --temperature $temperature --neg_sample $neg_sample --segment_length $segment_length  --neg_weight $neg_weight --vos_neg_sample $vos_neg_sample --random_sample $random_sample --epsilon $epsilon --e_dim $e_dim
done
done
done
done