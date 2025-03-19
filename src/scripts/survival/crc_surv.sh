#!/bin/bash
# ./scripts/survival/crc_surv.sh 3 deepattnmisl
# ./scripts/survival/crc_surv.sh 2 panther
# ./scripts/survival/crc_surv.sh 3 h2t
# ./scripts/survival/crc_surv.sh 3 ot
# ./scripts/survival/crc_surv.sh 3 protocounts
# ./scripts/survival/crc_surv.sh 1 abmil
# ./scripts/survival/crc_surv.sh 7 deepsets
# ./scripts/survival/crc_surv.sh 7 transmil
# ./scripts/survival/crc_surv.sh 7 dsmil
# ./scripts/survival/crc_surv.sh 7 ilra
# ./scripts/survival/crc_surv.sh 4 gnn
# ./scripts/survival/crc_surv.sh 5 graphtransformer

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'/data1/r20user2/wsi_data/TCGA_COADREAD'
	# '/data1/r20user2/wsi_data/TCGA_COAD'
	# '/data1/r20user2/wsi_data/TCGA_READ'
)


task='COADREAD_survival'
target_col='dss_survival_days'
split_names='train,test'

for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_COADREAD_overall_survival_k=${k}"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
done