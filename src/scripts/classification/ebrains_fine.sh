#!/bin/bash
# ./scripts/classification/ebrains_fine.sh 4 spanther
# ./scripts/classification/ebrains_fine.sh 2 panther
# ./scripts/classification/ebrains_fine.sh 3 h2t
# ./scripts/classification/ebrains_fine.sh 3 ot
# ./scripts/classification/ebrains_fine.sh 3 protocounts
# ./scripts/classification/ebrains_fine.sh 6 abmil
# ./scripts/classification/ebrains_fine.sh 7 deepsets
# ./scripts/classification/ebrains_fine.sh 1 transmil
# ./scripts/classification/ebrains_fine.sh 7 dsmil
# ./scripts/classification/ebrains_fine.sh 7 ilra
# ./scripts/classification/ebrains_fine.sh 6 gnn
# ./scripts/classification/ebrains_fine.sh 1 graphtransformer
# ./scripts/classification/ebrains_fine.sh 6 longmil

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	"/data1/r20user2/wsi_data/EBRAINS"
)

task='ebrains_subtyping_fine'
target_col='diagnosis'
split_dir='classification/ebrains'
split_names='train,val,test'

bash "./scripts/classification/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
