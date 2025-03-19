#!/bin/bash
# ./scripts/classification/ebrains_coarse.sh 2 deepattnmisl
# ./scripts/classification/ebrains_coarse.sh 5 spanther
# ./scripts/classification/ebrains_coarse.sh 5 panther
# ./scripts/classification/ebrains_coarse.sh 3 h2t
# ./scripts/classification/ebrains_coarse.sh 4 ot
# ./scripts/classification/ebrains_coarse.sh 3 protocounts
# ./scripts/classification/ebrains_coarse.sh 6 abmil
# ./scripts/classification/ebrains_coarse.sh 7 deepsets
# ./scripts/classification/ebrains_coarse.sh 7 transmil
# ./scripts/classification/ebrains_coarse.sh 7 dsmil
# ./scripts/classification/ebrains_coarse.sh 7 ilra
# ./scripts/classification/ebrains_coarse.sh 3 gnn
# ./scripts/classification/ebrains_coarse.sh 3 graphtransformer

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	"/data1/r20user2/wsi_data/EBRAINS"
)

task='ebrains_subtyping_coarse'
target_col='diagnosis_group'
split_dir='classification/ebrains'
split_names='train,val,test'

bash "./scripts/classification/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
