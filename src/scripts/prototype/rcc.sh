#!/bin/bash
# ./scripts/prototype/rcc.sh 0

gpuid=$1

declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_RCC'
)
split_dir="classification/RCC"
split_names="train"
bash "./scripts/prototype/clustering.sh" $gpuid $split_dir $split_names "${dataroots[@]}"