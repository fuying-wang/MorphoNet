#!/bin/bash
# ./scripts/survival/kirc_surv.sh 5 spanther
# ./scripts/survival/kirc_surv.sh 4 panther
# ./scripts/survival/kirc_surv.sh 3 h2t
# ./scripts/survival/kirc_surv.sh 4 ot
# ./scripts/survival/kirc_surv.sh 3 protocounts
# ./scripts/survival/kirc_surv.sh 5 abmil
# ./scripts/survival/kirc_surv.sh 7 deepsets
# ./scripts/survival/kirc_surv.sh 4 deepattnmisl
# ./scripts/survival/kirc_surv.sh 3 transmil
# ./scripts/survival/kirc_surv.sh 7 dsmil
# ./scripts/survival/kirc_surv.sh 7 ilra
# ./scripts/survival/kirc_surv.sh 0 gnn

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_RCC'
)


task='KIRC_survival'
target_col='dss_survival_days'
# split_names='train,test,filtered_test_cptac'
split_names='train,test'

# Loop through different folds
for k in 0 1 2 3 4; do
# for k in 4; do
	split_dir="survival/TCGA_KIRC_overall_survival_k=${k}"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
done

# for k in 0; do
# 	split_dir="survival/TCGA_KIRC_overall_survival_k=${k}"
# 	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
# done