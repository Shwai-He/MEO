do_train=False
model_name_or_path=gpt2
dataset_name=wikitext
dataset_config_name=wikitext-2-raw-v1

per_device_train_batch_size=8
per_device_eval_batch_size=8

num_train_epochs=10
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=epoch
k=2
n_experts=8
seed=42

use_moe=none
# use_moe=Conv1D_MoE
# use_moe=Conv1D_MEO
moe_level=task
log_out=log.out
learning_rate=1e-5

output_dir=./checkpoints/${model_name_or_path##*/}/${use_moe}/${n_experts}_${k}_${moe_level}_${learning_rate}

echo "${output_dir}"
mkdir -p ${output_dir}

echo  --use_moe ${use_moe} \
      --moe_level ${moe_level} \
      --k ${k} \
      --n_experts ${n_experts} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train ${do_train} \
      --do_eval \
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      --learning_rate ${learning_rate} \
      > ${output_dir}/config.txt


if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

python run_clm.py \
      --use_moe ${use_moe} \
      --moe_level ${moe_level} \
      --k ${k} \
      --n_experts ${n_experts} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train ${do_train} \
      --do_eval \
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      --learning_rate ${learning_rate} \


      