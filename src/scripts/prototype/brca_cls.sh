#!/bin/bash
# ./scripts/prototype/brca_cls.sh 1

gpuid=$1

declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_BRCA'
)
split_dir="classification/BRCA"
split_names="train"
bash "./scripts/prototype/clustering.sh" $gpuid $split_dir $split_names "${dataroots[@]}"
