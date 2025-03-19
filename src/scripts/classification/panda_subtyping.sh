#!/bin/bash
# ./scripts/classification/panda_subtyping.sh 6 panther
# ./scripts/classification/panda_subtyping.sh 3 h2t
# ./scripts/classification/panda_subtyping.sh 2 ot
# ./scripts/classification/panda_subtyping.sh 5 gnn
# ./scripts/classification/panda_subtyping.sh 3 protocounts
# ./scripts/classification/panda_subtyping.sh 4 abmil
# ./scripts/classification/panda_subtyping.sh 6 deepattnmisl
# ./scripts/classification/panda_subtyping.sh 7 deepsets
# ./scripts/classification/panda_subtyping.sh 6 transmil
# ./scripts/classification/panda_subtyping.sh 7 dsmil
# ./scripts/classification/panda_subtyping.sh 7 ilra
# ./scripts/classification/panda_subtyping.sh 2 graphtransformer
# ./scripts/classification/panda_subtyping.sh 7 longmil

gpuid=$1
config=$2
# check=$2
# wandb_project=$4

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/PANDA'
)

task='panda'
target_col='isup_grade'
split_dir="classification/panda_wholesight"
split_names='train,val,test_K,test_R'

bash "./scripts/classification/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
# bash "./scripts/classification/${config}.sh" $gpuid $check $wandb_project $task $target_col $split_dir $split_names "${dataroots[@]}"
