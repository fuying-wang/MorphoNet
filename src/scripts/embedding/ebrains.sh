#!/bin/bash
# ./scripts/embedding/ebrains.sh 2 ot

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	"/data1/r20user2/wsi_data/EBRAINS"
)

split_names='train,val,test'

split_dir='classification/ebrains'
bash "./scripts/embedding/${config}.sh" $gpuid $split_dir $split_names "${dataroots[@]}"