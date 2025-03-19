#!/bin/bash
# ./scripts/prototype/ebrains.sh 7

gpuid=$1

declare -a dataroots=(
	"/data1/r20user2/wsi_data/EBRAINS"
)
split_dir="classification/ebrains"
split_names="train"
bash "./scripts/prototype/clustering.sh" $gpuid $split_dir $split_names "${dataroots[@]}"

# Loop through different folds
# for k in 0 1 2 3 4; do
# 	split_dir="survival/TCGA_NSCLC_overall_survival_k=${k}"
# 	split_names="train"
# 	bash "./scripts/prototype/clustering.sh" $gpuid $split_dir $split_names "${dataroots[@]}"
# done