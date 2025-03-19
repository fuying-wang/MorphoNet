#!/bin/bash

gpuid=$1

declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_NSCLC'
)

# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_LUSC_overall_survival_k=${k}"
	split_names="train"
	bash "./scripts/prototype/clustering.sh" $gpuid $split_dir $split_names "${dataroots[@]}"
done