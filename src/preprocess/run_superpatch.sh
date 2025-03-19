CUDA_VISIBLE_DEVICES=1 python create_superpatch_graph.py --dataset TCGA_NSCLC \
    --source_dir superpatch_leiden_mean \
    --save_dir superpatch_leiden_mean_graph

CUDA_VISIBLE_DEVICES=1 python create_superpatch_graph.py --dataset EBRAINS \
    --source_dir superpatch_leiden_mean \
    --save_dir superpatch_leiden_mean_graph

CUDA_VISIBLE_DEVICES=1 python create_superpatch_graph.py --dataset PANDA \
    --source_dir superpatch_leiden_mean \
    --save_dir superpatch_leiden_mean_graph

CUDA_VISIBLE_DEVICES=1 python create_superpatch_graph.py --dataset TCGA_BRCA \
    --source_dir superpatch_leiden_mean \
    --save_dir superpatch_leiden_mean_graph

CUDA_VISIBLE_DEVICES=1 python create_superpatch_graph.py --dataset TCGA_RCC \
    --source_dir superpatch_leiden_mean \
    --save_dir superpatch_leiden_mean_graph