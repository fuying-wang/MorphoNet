#!/bin/bash
# ./scripts/embedding/panda.sh 4 ot

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	"/data1/r20user2/wsi_data/PANDA"
)

split_names='train,val,test_K,test_R'
split_dir="classification/panda_wholesight"
bash "./scripts/embedding/${config}.sh" $gpuid $split_dir $split_names "${dataroots[@]}"