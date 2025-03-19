#!/bin/bash
# ./scripts/survival/blca_surv.sh 4 deepattnmisl
# ./scripts/survival/blca_surv.sh 4 spanther
# ./scripts/survival/blca_surv.sh 3 panther
# ./scripts/survival/blca_surv.sh 3 h2t
# ./scripts/survival/blca_surv.sh 3 ot
# ./scripts/survival/blca_surv.sh 3 protocounts
# ./scripts/survival/blca_surv.sh 4 abmil
# ./scripts/survival/blca_surv.sh 7 deepsets
# ./scripts/survival/blca_surv.sh 7 transmil
# ./scripts/survival/blca_surv.sh 7 dsmil
# ./scripts/survival/blca_surv.sh 5 ilra
# ./scripts/survival/blca_surv.sh 5 gnn
# ./scripts/survival/blca_surv.sh 5 graphtransformer


gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_BLCA'
)


task='BLCA_survival'
target_col='dss_survival_days'
split_names='train,test'

# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_BLCA_overall_survival_k=${k}"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
done