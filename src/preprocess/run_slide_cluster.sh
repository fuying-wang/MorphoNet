# CUDA_VISIBLE_DEVICES=4 python create_slide_level_prototypes.py \
# --data_source /data1/r20user2/wsi_data/TCGA_BLCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
# --dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_BLCA/survival.csv \
# --cluster_method faiss

CUDA_VISIBLE_DEVICES=4 python create_slide_level_prototypes.py \
--data_source /data1/r20user2/wsi_data/TCGA_RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
--dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_KIRC/survival.csv \
--cluster_method leiden

CUDA_VISIBLE_DEVICES=4 python create_slide_level_prototypes.py \
--data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
--dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_LUAD/survival.csv \
--cluster_method faiss

CUDA_VISIBLE_DEVICES=4 python create_slide_level_prototypes.py \
--data_source /data1/r20user2/wsi_data/TCGA_BRCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
--dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_BRCA/survival.csv \
--cluster_method faiss

CUDA_VISIBLE_DEVICES=4 python create_slide_level_prototypes.py \
--data_source /data1/r20user2/wsi_data/EBRAINS/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
--dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/ebrains/classification.csv \
--cluster_method faiss

CUDA_VISIBLE_DEVICES=4 python create_slide_level_prototypes.py \
--data_source /data1/r20user2/wsi_data/PANDA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
--dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/panda_wholesight/classification.csv \
--cluster_method faiss
