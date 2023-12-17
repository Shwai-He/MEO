model_name_or_path=bert-base-cased
use_moe=none
# use_moe=MoE
# use_moe=MEO
dataset_name=wikitext
dataset_config_name=wikitext-2-raw-v1

per_device_train_batch_size=16
per_device_eval_batch_size=16

num_train_epochs=10
dataloader_num_workers=16

save_strategy=steps
evaluation_strategy=steps
save_steps=500
eval_steps=500
seed=42

log_out=log.out
output_dir=./checkpoints/${model_name_or_path##*/}/${use_moe}

echo "${output_dir}"
mkdir -p ${output_dir}

echo  --use_moe ${use_moe} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train \
      --do_eval \
      --eval_steps ${eval_steps} \
      --save_steps ${save_steps} \
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      > ${output_dir}/config.txt

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

python run_mlm.py \
      --use_moe ${use_moe} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train \
      --do_eval \
      --eval_steps ${eval_steps} \
      --save_steps ${save_steps} \
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      > ${output_dir}/${log_out} & echo $! > ${output_dir}/log.txt &