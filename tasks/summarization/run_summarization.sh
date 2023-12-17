do_train=False
model_name_or_path=facebook/bart-large
per_device_train_batch_size=8
per_device_eval_batch_size=16
dataset_name=xsum
num_train_epochs=4
weight_decay=0.1
learning_rate=1e-4

dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=steps
eval_accumulation_steps=20

use_moe=none
# use_moe=MoE
# use_moe=MEO
moe_level=sequence
n_experts=8
k=2

metric_for_best_model=rouge2
SAVE=./checkpoints/${model_name_or_path##*/}/${use_moe}/${n_experts}_${k}_${moe_level}_${learning_rate}

echo "${SAVE}"
mkdir -p ${SAVE}

echo  --model_name_or_path ${model_name_or_path} \
      --dataset_name ${dataset_name} \
      --do_train ${do_train} --do_eval --overwrite_output_dir \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --output_dir ${SAVE} \
      --num_train_epochs ${num_train_epochs} --learning_rate ${learning_rate} \
      --weight_decay ${weight_decay} --metric_for_best_model ${metric_for_best_model} \
      --val_max_target_length 60 --max_eval_samples 1600 \
      --num_beams 6 --max_length 60 --min_length 10 --no_repeat_ngram_size 3 \
      --use_moe ${use_moe} --moe_level ${moe_level} --n_experts ${n_experts} --k ${k} \
      --evaluation_strategy ${evaluation_strategy} --save_strategy ${save_strategy} --save_steps ${save_steps} \
      --eval_accumulation_steps ${eval_accumulation_steps} 

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

python run_summarization.py \
      --model_name_or_path ${model_name_or_path} \
      --dataset_name ${dataset_name} \
      --do_train ${do_train} --do_eval --overwrite_output_dir \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --output_dir ${SAVE} \
      --num_train_epochs ${num_train_epochs} --learning_rate ${learning_rate} \
      --weight_decay ${weight_decay} --metric_for_best_model ${metric_for_best_model} \
      --val_max_target_length 60 --max_eval_samples 1600 \
      --num_beams 6 --max_length 60 --min_length 10 --no_repeat_ngram_size 3 \
      --use_moe ${use_moe} --moe_level ${moe_level} --n_experts ${n_experts} --k ${k} \
      --evaluation_strategy ${evaluation_strategy} --save_strategy ${save_strategy} --save_steps ${save_steps} \
      --eval_accumulation_steps ${eval_accumulation_steps} 
