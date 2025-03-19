#!/bin/bash
# ./scripts/survival/luad_surv.sh 3 spanther
# ./scripts/survival/luad_surv.sh 1 panther
# ./scripts/survival/luad_surv.sh 3 h2t
# ./scripts/survival/luad_surv.sh 2 ot
# ./scripts/survival/luad_surv.sh 3 protocounts
# ./scripts/survival/luad_surv.sh 1 abmil
# ./scripts/survival/luad_surv.sh 7 deepsets
# ./scripts/survival/luad_surv.sh 5 transmil
# ./scripts/survival/luad_surv.sh 7 dsmil
# ./scripts/survival/luad_surv.sh 7 ilra
# ./scripts/survival/luad_surv.sh 6 gnn
# ./scripts/survival/luad_surv.sh 7 graphtransformer

# export PATH="/home/fwu/Documents/MyProjects/SPANTHER/src:$PATH"

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_NSCLC'
)


task='LUAD_survival'
target_col='dss_survival_days'
split_names='train,test_tcgaluad'

# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_LUAD_overall_survival_k=${k}"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
done