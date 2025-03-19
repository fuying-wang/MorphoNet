#!/bin/bash
# ./scripts/survival/lusc_surv.sh 4 panther
# ./scripts/survival/lusc_surv.sh 3 h2t
# ./scripts/survival/lusc_surv.sh 3 ot
# ./scripts/survival/lusc_surv.sh 3 protocounts
# ./scripts/survival/lusc_surv.sh 6 abmil
# ./scripts/survival/lusc_surv.sh 7 deepsets
# ./scripts/survival/lusc_surv.sh 7 transmil
# ./scripts/survival/lusc_surv.sh 7 dsmil
# ./scripts/survival/lusc_surv.sh 7 ilra

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_NSCLC'
)


task='LUSC_survival'
target_col='dss_survival_days'
split_names='train,test_tcgalusc'

# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_LUSC_overall_survival_k=${k}"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
done