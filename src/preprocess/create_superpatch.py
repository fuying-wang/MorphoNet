import sys
sys.path.append('../../src')

import faiss
import os
import h5py
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import scanpy as sc
import ipdb
from utils.file_utils import save_hdf5
from mil_models.OT.otk.layers import OTKernel
from utils.proto_utils import cluster
from utils.file_utils import save_pkl, load_pkl, j_


'''
CUDA_VISIBLE_DEVICES=3 python create_superpatch.py --dataset TCGA_KIRC \
    --task survival --agg_type mean \
    --cluster_method kmeans \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv  \
    --data_source /data1/r20user2/wsi_data/TCGA_RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=3 python create_superpatch.py --dataset panda_wholesight \
    --task classification --agg_type mean \
    --cluster_method kmeans \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv  \
    --data_source /data1/r20user2/wsi_data/PANDA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

# CUDA_VISIBLE_DEVICES=4 python create_superpatch.py --dataset NSCLC \
#     --task classification --agg_type mean \
#     --cluster_method kmeans \
#     --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
#     --data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=5 python create_superpatch.py --dataset panda_wholesight \
    --task classification --agg_type mean \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/PANDA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=5 python create_superpatch.py --dataset ebrains \
    --task classification --agg_type mean \
    --cluster_method leiden \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/EBRAINS/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=5 python create_superpatch.py --dataset TCGA_BRCA \
    --task survival --agg_type mean \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_BRCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=5 python create_superpatch.py --dataset TCGA_COADREAD \
    --task survival --agg_type mean \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_COADREAD/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=5 python create_superpatch.py --dataset TCGA_LUAD \
    --task survival --agg_type mean \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_LUAD/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=5 python create_superpatch.py --dataset TCGA_BLCA \
    --task survival --agg_type mean \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_BLCA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=5 python create_superpatch.py --dataset TCGA_UCEC \
    --task survival --agg_type mean \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/TCGA_UCEC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 

CUDA_VISIBLE_DEVICES=5 python create_superpatch.py --dataset CPTAC_KIRC \
    --task survival --agg_type mean \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --data_source /data1/r20user2/wsi_data/CPTAC_RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files 
'''

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument("--task", type=str, choices=["survival", "classification"],
                    default="survival")
parser.add_argument("--dataset", type=str, required=True,
                    default="TCGA_KIRC")
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument("--dataset_csv_dir", type=str, default=None)
parser.add_argument('--split_names', type=str, default='train',
                    help='delimited list for specifying names within each split')
parser.add_argument("--agg_type", type=str, default="ot",
                    choices=["ot", "mean"])
parser.add_argument("--n_proto", type=int, default=16)
parser.add_argument("--cluster_method", type=str, default="kmeans")
parser.add_argument("--skip_existed", action="store_true", default=False)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = os.path.join(args.dataset_csv_dir, args.dataset, f"{args.task}.csv")
if args.cluster_method == "spatialleiden":
    slide_level_prototype = os.path.join(args.data_source, f"../cluster_feats")
elif args.cluster_method == "kmeans":
    slide_level_prototype = os.path.join(args.data_source, f"../faiss_cluster_feats")
elif args.cluster_method == "leiden":
    slide_level_prototype = os.path.join(args.data_source, f"../leiden_cluster_feats")
else:
    raise NotImplementedError(f"Cluster method {args.cluster_method} not implemented")
args.data_source = [src for src in args.data_source.split(',')][0]
save_dir = os.path.join(args.data_source, f"../superpatch_{args.cluster_method}_{args.agg_type}")

# split_dir = "/home/fywang/Documents/SPANTHER/src/splits"
# check this for survival tasks
# proto_path = j_(split_dir, args.task, args.dataset, f"prototypes/prototypes_c{args.n_proto}_uni_pt_patch_features_faiss_num_1.0e+05.pkl")
# assert os.path.exists(proto_path), f"proto_path: {proto_path} does not exist"
# prototypes = load_pkl(proto_path)['prototypes'].squeeze()
os.makedirs(save_dir, exist_ok=True)


# def create_region_representations(x):
#     attention = OTKernel(in_dim=1024, 
#                          out_size=args.n_proto, 
#                          distance="euclidean",
#                          heads=1, 
#                          max_iter=100, 
#                          eps=0.1).to(device)
#     attention.weight.data.copy_(torch.from_numpy(prototypes).to(device))

#     x = torch.tensor(x).to(device)
#     out = attention(x.unsqueeze(0))

#     # weighted_mean
#     out = out.mean(dim=1).detach().cpu().numpy()
#     # allcat
#     # out = out.reshape(1, -1).detach().cpu().numpy()

#     return out


def main():
    df = pd.read_csv(csv_path)
    for row in tqdm(df.itertuples(), total=len(df)):
        slide_id = row.slide_id
        save_path = os.path.join(save_dir, f"{slide_id}.h5")

        # make sure both are true 
        if args.skip_existed and os.path.exists(save_path):
            print(f"Skipping {slide_id} as it already exists")
            continue
        
        if args.cluster_method == "spatialLeiden":
            slide_prototype_path = os.path.join(slide_level_prototype, f"{slide_id}.h5ad")
            assert os.path.exists(slide_prototype_path), f"slide_prototype_path: {slide_prototype_path} does not exist"
            adata = sc.read(slide_prototype_path)
            spatial_leiden_clusters = adata.obs['spatialleiden'].values.astype(np.int32)
            num_clusters = len(np.unique(spatial_leiden_clusters))
        elif args.cluster_method == "kmeans":
            slide_prototype_path = os.path.join(slide_level_prototype, f"{slide_id}_faiss_num16.pt")
            h5_file = os.path.join(args.data_source, f"{slide_id}.h5")
            with h5py.File(h5_file, 'r') as f:
                features = f['features'][:]
                coords = f['coords'][:]
            proto = torch.load(slide_prototype_path)
            num_clusters = len(np.unique(proto))
        elif args.cluster_method == "leiden":
            slide_prototype_path = os.path.join(slide_level_prototype, f"{slide_id}.h5ad")
            assert os.path.exists(slide_prototype_path), f"slide_prototype_path: {slide_prototype_path} does not exist"
            adata = sc.read(slide_prototype_path)
            leiden_clusters = adata.obs['leiden'].values.astype(np.int32)
            num_clusters = len(np.unique(leiden_clusters))
        else:
            raise NotImplementedError(f"Cluster method {args.cluster_method} not implemented")

        center_embeddings = []
        center_coords = []
        for i in range(num_clusters):   
            if args.cluster_method == "spatialLeiden":
                cluster_embeddings = adata[adata.obs['spatialleiden'] == i].X
                cluster_coords = adata[adata.obs['spatialleiden'] == i].obsm['spatial']
            elif args.cluster_method == "kmeans":
                cluster_embeddings = features[proto == i]
                cluster_coords = coords[proto == i]
            elif args.cluster_method == "leiden":
                cluster_embeddings = adata.X[leiden_clusters == i]
                cluster_coords = adata.obsm['spatial'][leiden_clusters == i]
            else:
                raise NotImplementedError(f"Cluster method {args.cluster_method} not implemented")
            # TODO: maybe need hierarchical clustering to find a better representation ...
            if args.agg_type == "ot":
                raise NotImplementedError("OT not implemented")
                # agg_center_embeddings = create_region_representations(cluster_embeddings)
                # center_embeddings.append(agg_center_embeddings[0])
            elif args.agg_type == "mean":
                center_embeddings.append(np.mean(cluster_embeddings, axis=0))
            center_coords.append(np.mean(cluster_coords, axis=0))
        
        center_embeddings = np.array(center_embeddings)
        center_coords = np.array(center_coords)

        asset_dict = {'features': center_embeddings, 'coords': center_coords}
        save_hdf5(save_path, asset_dict, attr_dict=None, mode='w')


if __name__ == "__main__":
    main()