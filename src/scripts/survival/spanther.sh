#!/bin/bash
gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("$@")

feat='uni_pt_patch_features'
input_dim=1024
mag='20x'
patch_size=256

bag_size='-1'
batch_size=64
# out_size=32
out_size=16
out_type='allcat'
model_tuple='SPANTHER,default'
lin_emb_model='LinearEmb' # 'LinearEmb'; 'IndivMLPEmb_Shared', 'IndivMLPEmb_Indiv', 'IndivMLPEmb_SharedPost', 'IndivMLPEmb_IndivPost', 'IndivMLPEmb_SharedIndiv' or 'IndivMLPEmb_SharedIndivPost'
max_epoch=50
lr=0.0001
wd=0.00001
lr_scheduler='cosine'
opt='adamW'
grad_accum=1
loss_fn='cox'
n_label_bin=4
alpha=0.5
em_step=1
load_proto=0
es_flag=0
tau=1.0
eps=1
n_fc_layer=0
proto_num_samples='1.0e+05'
save_dir_root=results
num_layers=1
num_mixtures=16
num_heads=1
sigma=1.
pooling="ot"

IFS=',' read -r model config_suffix <<< "${model_tuple}"
model_config=${model}_${config_suffix}
feat_name=$(echo $feat | sed 's/^extracted-//')
exp_code=${task}::${model_config}::${feat_name}
save_dir=${save_dir_root}/${exp_code}

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
  # feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}/${feat}/WSI_graph_v2
  # feat_dir=/home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_KIRC/superpatch
  feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}/${feat}/superpatch_connected_graph
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
# cmd="CUDA_VISIBLE_DEVICES=$gpuid ${RUN_CMD} training/main_survival.py \\
cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_survival \\
--data_source ${all_feat_dirs} \\
--results_dir ${save_dir} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--task ${task} \\
--target_col ${target_col} \\
--model_type ${model} \\
--model_config ${model}_default \\
--n_fc_layers ${n_fc_layer} \\
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
--num_workers 4 \\
--em_iter ${em_step} \\
--tau ${tau} \\
--n_proto ${out_size} \\
--out_type ${out_type} \\
--loss_fn ${loss_fn} \\
--nll_alpha ${alpha} \\
--n_label_bins ${n_label_bin} \\
--early_stopping ${es_flag} \\
--emb_model_type ${lin_emb_model} \\
--ot_eps ${eps} \\
--fix_proto \\
--num_layers ${num_layers} \\
--num_mixtures ${num_mixtures} \\
--num_heads ${num_heads} \\
--sigma ${sigma} \\
--pooling ${pooling} \\
"
# --skip_exist \\
# --concat \\

# Specifiy prototype path if load_proto is True
# prototypes_uni_pt_patch_features_nproto_32
# if [[ $load_proto -eq 1 ]]; then
#   cmd="$cmd --load_proto \\
#   --proto_path "splits/${split_dir}/prototypes/prototypes_c${out_size}_${feat_name}_faiss_num_${proto_num_samples}.pkl" \\
#   "
# fi

# if [[ $load_proto -eq 1 ]]; then
#   cmd="$cmd --load_proto \\
#   --proto_path "splits/${split_dir}/prototypes/prototypes_${feat_name}_nproto_${out_size}_norm.pkl" \\
#   "
# fi

eval "$cmd"