#!/usr/bin/bash

#SBATCH --job-name=eval-moe
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --quotatype=auto

#SBATCH --nodes=1
#SBATCH --gres=gpu:1

source ~/anaconda3/bin/activate smoe

TASK_NAME=sst2
use_moe=MEO
# use_moe=MoE
# use_moe=none

moe_level=task

k=12
n_experts=16

model_name_or_path=bert-base-uncased
# model_name_or_path=prajjwal1/bert-small
do_train=False
per_device_train_batch_size=16
per_device_eval_batch_size=32
cache_dir=../../cache_dir
num_train_epochs=10
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=steps
save_steps=999999999
weight_decay=0.1
learning_rate=1e-5
seed=42
output_dir=./checkpoints/${model_name_or_path##*/}/${TASK_NAME}/${use_moe}/${n_experts}_${k}_${moe_level}_${learning_rate}

echo "${output_dir}"
mkdir -p ${output_dir}

echo "${cache_dir}"
mkdir -p ${cache_dir}

echo  --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --use_moe ${use_moe} \
      --moe_level ${moe_level} \
      --k ${k} \
      --n_experts ${n_experts} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --overwrite_output_dir \
      --do_train ${do_train}\
      --do_eval \
      --weight_decay ${weight_decay} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      > ${output_dir}/config.txt


if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/log.out
fi

python ./run_glue.py \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --use_moe ${use_moe} \
      --moe_level ${moe_level} \
      --k ${k} \
      --n_experts ${n_experts} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --overwrite_output_dir \
      --do_train ${do_train}\
      --do_eval \
      --weight_decay ${weight_decay} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
