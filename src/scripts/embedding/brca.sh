#!/bin/bash

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_BRCA'
)

split_names='train,test'

split_dir='survival/TCGA_BRCA_overall_survival_k=1'
bash "./scripts/embedding/${config}.sh" $gpuid $split_dir $split_names "${dataroots[@]}"