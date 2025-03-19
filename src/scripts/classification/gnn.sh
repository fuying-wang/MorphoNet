#!/bin/bash

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("$@")

feat='uni_pt_patch_features'
input_dim=1024
# input_dim=65536
mag='20x'
patch_size=256

model_tuple='GNN,default'
bag_size=-1
batch_size=1
lr_scheduler='cosine'
opt='adamW'
max_epoch=20
lr=0.0001
wd=0.00001
save_dir_root='results'

es_min_epochs=15
num_layers=3
pooling='attention'
es_metric='loss'

IFS=',' read -r model config_suffix <<< "${model_tuple}"
model_config=${model}_${config_suffix}
feat_name=$(echo $feat | sed 's/^extracted-//')
exp_code=${task}::${model_config}::${feat_name}
save_dir=${save_dir_root}/${exp_code}

if [[ $bag_size == "-1" ]]; then
	curr_bag_size=$bag_size
	curr_batch_size=1
	grad_accum=32
else
	if [[ $patch_size == 512 ]]; then
		curr_bag_size='1024'
	else
		curr_bag_size=$bag_size
	fi
	curr_batch_size=$batch_size
	grad_accum=0
fi


th=0.00005
if awk "BEGIN {exit !($lr <= $th)}"; then
  warmup=0
  curr_lr_scheduler='constant'
else
  curr_lr_scheduler=$lr_scheduler
  warmup=1
fi

# Identify feature paths
all_feat_dirs=""
for dataroot_path in "${dataroots[@]}"; do
  # feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}/${feat}/h5_files
  # feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}/${feat}/WSI_graph
  # feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}/${feat}/superpatch_graph
  feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}/${feat}/superpatch_graph
  # feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}/${feat}/superpatch_leiden_mean_graph
  # feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}/${feat}/superpatch_connected_graph
  if ! test -d $feat_dir
  then
    continue
  fi

  if [[ -z ${all_feat_dirs} ]]; then
    all_feat_dirs=${feat_dir}
  else
    all_feat_dirs=${all_feat_dirs},${feat_dir}
  fi
done

RUN_CMD="python3 -m debugpy --listen 3000 --wait-for-client"
# Actual command
# cmd="CUDA_VISIBLE_DEVICES=$gpuid ${RUN_CMD} training/main_classification.py \\
cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_classification \\
--data_source ${all_feat_dirs} \\
--results_dir ${save_dir} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--task ${task} \\
--target_col ${target_col} \\
--model_type ${model} \\
--model_config ${model}_default \\
--in_dim ${input_dim} \\
--opt ${opt} \\
--lr ${lr} \\
--lr_scheduler ${curr_lr_scheduler} \\
--accum_steps ${grad_accum} \\
--wd ${wd} \\
--warmup_epochs ${warmup} \\
--max_epochs ${max_epoch} \\
--train_bag_size ${bag_size} \\
--batch_size ${batch_size} \\
--in_dropout 0 \\
--seed 1 \\
--num_workers 8 \\
--early_stopping \\
--num_layers ${num_layers} \\
--pooling ${pooling} \\
--es_min_epochs ${es_min_epochs} \\
--es_metric ${es_metric} \\
"

# Specifiy prototype path if load_proto is True
# if [[ $load_proto -eq 1 ]]; then
#   cmd="$cmd --load_proto \\
#   --proto_path "splits/${split_dir}/prototypes/prototypes_c${out_size}_${feat_name}_faiss_num_${proto_num_samples}.pkl" \\
#   "
# fi

if [[ $load_proto -eq 1 ]]; then
  cmd="$cmd --load_proto \\
  --proto_path "splits/${split_dir}/prototypes/prototypes_${feat_name}_nproto_${out_size}.pkl" \\
  "
fi

eval "$cmd"