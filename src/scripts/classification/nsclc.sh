#!/bin/bash
# ./scripts/classification/nsclc.sh 2 deepattnmisl
# ./scripts/classification/nsclc.sh 5 spanther
# ./scripts/classification/nsclc.sh 1 panther
# ./scripts/classification/nsclc.sh 1 h2t
# ./scripts/classification/nsclc.sh 4 ot
# ./scripts/classification/nsclc.sh 1 protocounts
# ./scripts/classification/nsclc.sh 1 abmil
# ./scripts/classification/nsclc.sh 1 deepsets
# ./scripts/classification/nsclc.sh 1 transmil
# ./scripts/classification/nsclc.sh 1 dsmil
# ./scripts/classification/nsclc.sh 1 ilra
# ./scripts/classification/nsclc.sh 0 gnn
# ./scripts/classification/nsclc.sh 0 graphtransformer
gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_NSCLC'
)

task='NSCLC_classification'
target_col='OncoTreeCode'
split_names='train,val,test'

split_dir="classification/NSCLC"
bash "./scripts/classification/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"