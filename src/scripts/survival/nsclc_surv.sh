#!/bin/bash
# ./scripts/survival/nsclc_surv.sh 6 panther
# ./scripts/survival/nsclc_surv.sh 3 h2t
# ./scripts/survival/nsclc_surv.sh 3 ot
# ./scripts/survival/nsclc_surv.sh 3 protocounts
# ./scripts/survival/nsclc_surv.sh 6 abmil

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_NSCLC'
)


task='NSCLC_survival'
target_col='dss_survival_days'
split_names='train,test'

# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_NSCLC_overall_survival_k=${k}"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
done