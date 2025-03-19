#!/bin/bash
# ./scripts/survival/brca_surv.sh 4 spanther
# ./scripts/survival/brca_surv.sh 2 panther
# ./scripts/survival/brca_surv.sh 3 h2t
# ./scripts/survival/brca_surv.sh 3 ot
# ./scripts/survival/brca_surv.sh 3 protocounts
# ./scripts/survival/brca_surv.sh 1 abmil
# ./scripts/survival/brca_surv.sh 7 deepsets
# ./scripts/survival/brca_surv.sh 6 transmil
# ./scripts/survival/brca_surv.sh 7 dsmil
# ./scripts/survival/brca_surv.sh 7 ilra
# ./scripts/survival/brca_surv.sh 3 deepattnmisl
# ./scripts/survival/brca_surv.sh 3 gnn
# ./scripts/survival/brca_surv.sh 3 graphtransformer

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_BRCA'
)


task='BRCA_survival'
target_col='dss_survival_days'
split_names='train,test'


# Loop through different folds
for k in 0 1 2 3 4; do
# for k in 0; do
	split_dir="survival/TCGA_BRCA_overall_survival_k=${k}"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
done