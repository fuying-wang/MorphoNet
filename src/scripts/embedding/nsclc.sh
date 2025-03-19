#!/bin/bash
# ./scripts/embedding/nsclc.sh 0 ot

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_NSCLC'
)

split_names='train,val,test'

split_dir="classification/NSCLC"
bash "./scripts/embedding/${config}.sh" $gpuid $split_dir $split_names "${dataroots[@]}"