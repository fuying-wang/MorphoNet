#!/bin/bash
# ./scripts/survival/ucec_surv.sh 5 deepattnmisl
# ./scripts/survival/ucec_surv.sh 4 spanther
# ./scripts/survival/ucec_surv.sh 2 panther
# ./scripts/survival/ucec_surv.sh 3 h2t
# ./scripts/survival/ucec_surv.sh 3 ot
# ./scripts/survival/ucec_surv.sh 3 protocounts
# ./scripts/survival/ucec_surv.sh 1 abmil
# ./scripts/survival/ucec_surv.sh 7 deepsets
# ./scripts/survival/ucec_surv.sh 7 transmil
# ./scripts/survival/ucec_surv.sh 7 dsmil
# ./scripts/survival/ucec_surv.sh 7 ilra
# ./scripts/survival/ucec_surv.sh 7 gnn
# ./scripts/survival/ucec_surv.sh 7 graphtransformer

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_UCEC'
)


task='UCEC_survival'
target_col='dss_survival_days'
split_names='train,test'

# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_UCEC_overall_survival_k=${k}"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
done