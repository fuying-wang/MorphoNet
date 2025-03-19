#!/bin/bash
#  ./scripts/embedding/ucec.sh 0 ot

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_UCEC'
)

split_names='train,test'

split_dir='survival/TCGA_UCEC_overall_survival_k=0'
bash "./scripts/embedding/${config}.sh" $gpuid $split_dir $split_names "${dataroots[@]}"