# CUDA_VISIBLE_DEVICES=6 python create_slide_level_prototypes.py \
# --data_source /data1/r20user2/wsi_data/TCGA_BLCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
# --dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_BLCA/survival.csv \
# --cluster_method faiss

# CUDA_VISIBLE_DEVICES=6 python create_slide_level_prototypes.py \
# --data_source /data1/r20user2/wsi_data/TCGA_RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
# --dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_KIRC/survival.csv \
# --cluster_method faiss

# CUDA_VISIBLE_DEVICES=6 python create_slide_level_prototypes.py \
# --data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
# --dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_LUAD/survival.csv \
# --cluster_method faiss

# CUDA_VISIBLE_DEVICES=6 python create_slide_level_prototypes.py \
# --data_source /data1/r20user2/wsi_data/TCGA_COADREAD/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
# --dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_COADREAD/survival.csv \
# --cluster_method faiss

# CUDA_VISIBLE_DEVICES=6 python create_slide_level_prototypes.py \
# --data_source /data1/r20user2/wsi_data/TCGA_UCEC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
# --dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_UCEC/survival.csv \
# --cluster_method faiss

# CUDA_VISIBLE_DEVICES=6 python create_slide_level_prototypes.py \
# --data_source /data1/r20user2/wsi_data/TCGA_BRCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
# --dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_BRCA/survival.csv \
# --cluster_method faiss

# CUDA_VISIBLE_DEVICES=6 python create_slide_level_prototypes.py \
# --data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
# --dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_NSCLC/classification.csv \
# --cluster_method faiss

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_KIRC \
    --task survival --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv  \
    --data_source /data1/r20user2/wsi_data/TCGA_RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset NSCLC \
    --task classification --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset panda_wholesight \
    --task classification --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/PANDA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset ebrains \
    --task classification --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/EBRAINS/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_BRCA \
    --task survival --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_BRCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_COADREAD \
    --task survival --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_COADREAD/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_LUAD \
    --task survival --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_LUAD/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_BLCA \
    --task survival --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_BLCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_UCEC \
    --task survival --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_UCEC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset CPTAC_KIRC \
    --task survival --agg_type ot \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/CPTAC_RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 