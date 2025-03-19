#!/bin/bash
# ./scripts/sae/brca.sh 6 sae

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_BRCA'
)

task='BRCA_SAE'
target_col='oncotree_code'
split_names='train,val,test'

split_dir="classification/BRCA"
bash "./scripts/sae/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"