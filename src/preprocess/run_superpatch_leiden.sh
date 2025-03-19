CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset ebrains \
    --task classification --agg_type mean \
    --cluster_method leiden \
    --skip_existed \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/EBRAINS/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset panda_wholesight \
    --task classification --agg_type mean \
    --cluster_method leiden \
    --skip_existed \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/PANDA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 


CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_KIRC \
    --task survival --agg_type mean \
    --cluster_method leiden \
    --skip_existed \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_LUAD \
    --task survival --agg_type mean \
    --cluster_method leiden \
    --skip_existed \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=6 python create_superpatch.py --dataset TCGA_BRCA \
    --task survival --agg_type mean \
    --cluster_method leiden \
    --skip_existed \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_BRCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 