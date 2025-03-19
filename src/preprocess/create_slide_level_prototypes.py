"""
This will perform spatial clustering on patches of the training data

spatial-aware leiden clustering: 
https://spatialleiden.readthedocs.io/stable/usage.html
"""
import sys
sys.path.append('../../src')
# from __future__ import print_function

import os
from os.path import join as j_
import ipdb

import faiss
import random
import numpy as np
import pandas as pd
import argparse
import torch
import time
from torch.utils.data import DataLoader
from wsi_datasets import WSIProtoDataset
from utils.utils import seed_torch, read_splits
from utils.file_utils import save_pkl
from tqdm import tqdm

# from utils.proto_utils import cluster
import scanpy as sc
import spatialleiden as sl
import squidpy as sq

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

'''
CUDA_VISIBLE_DEVICES=1 python create_slide_level_prototypes.py \
--data_source /data1/r20user2/wsi_data/TCGA_RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \
--dataset_csv /home/fywang/Documents/SPANTHER/src/dataset_csv/TCGA_KIRC/survival.csv \
--cluster_method faiss

'''

def build_datasets(csv_splits, batch_size=1, num_workers=2, train_kwargs={}):
    dataset_splits = {}
    for k in csv_splits.keys(): # ['train']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy()
        dataset = WSIProtoDataset(df, **dataset_kwargs)

        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')

    return dataset_splits


def slide_level_cluster(data_loader, dataset_csv, 
                        patch_size=512, cluster_method="spatialleiden"):
    """
    """
    save_dir = j_(args.data_source[0], f'../{cluster_method}_cluster_feats')
    os.makedirs(save_dir, exist_ok=True)
    # save_dir = j_(os.path.dirname(dataset_csv), f'prototypes_{cluster_method}')
    for batch in tqdm(data_loader):       
        slide_id = batch['slide_id'][0]
        coords = batch['coords'][0] / patch_size  # (n_instances, instance_dim)
        data = batch['img'][0]                    # (n_instances, instance_dim)

        with torch.no_grad():
            coord_out = coords.detach().cpu().numpy()  # Remove batch dim
            feat_out = data.detach().cpu().numpy()     # Remove batch dim

        if cluster_method == "spatialleiden":
            if os.path.exists(j_(save_dir, f'{slide_id}.h5ad')):
                continue
            # create anndata object
            adata = sc.AnnData(feat_out)
            adata.obsm["spatial"] = coord_out
            # PCA for dimensionality reduction
            n_components = min(65, adata.shape[0], adata.shape[1]) - 1
            sc.pp.pca(adata, n_comps=n_components, random_state=seed)

            try:
                sc.pp.neighbors(adata, random_state=seed)
                sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=8)
                adata.obsp["spatial_connectivities"] = sl.distance2connectivity(
                    adata.obsp["spatial_distances"]
                )
                sl.spatialleiden(adata, layer_ratio=1.8, directed=(False, True), seed=seed)

                os.makedirs(save_dir, exist_ok=True)
                adata.write_h5ad(j_(save_dir, f'{slide_id}.h5ad'))
            except Exception as e:
                print(e)
                continue
        elif cluster_method == "faiss":
            cluster_num = 16
            if os.path.exists(j_(save_dir, f'{slide_id}_faiss_num{cluster_num}.pt')):
                continue
            if feat_out.shape[0] < cluster_num:
                cluster_num = feat_out.shape[0]
                # print(f'Not enough patches for clustering: {feat_out.shape[0]}')
                # print(f"slide id: {slide_id}")
                # continue
            kmeans = faiss.Kmeans(feat_out.shape[1], 
                                  cluster_num, 
                                  niter=50, 
                                  nredo=5,
                                  verbose=True,
                                  max_points_per_centroid=100000,
                                  gpu=1)
            kmeans.train(feat_out)
            labels = kmeans.assign(feat_out)[1]
            # cluster_feats = []
            # for i in range(cluster_num):
            #     class_i_index = np.where(labels == i)[0]
            #     cluster_feats.append(torch.from_numpy(feat_out[class_i_index]).unsqueeze(0))
            
            torch.save(labels, j_(save_dir, f'{slide_id}_faiss_num16.pt'))
        elif cluster_method == "leiden":
            if os.path.exists(j_(save_dir, f'{slide_id}.h5ad')):
                continue
            # create anndata object
            adata = sc.AnnData(feat_out)
            adata.obsm["spatial"] = coord_out
            # PCA for dimensionality reduction
            n_components = min(65, adata.shape[0], adata.shape[1]) - 1
            sc.pp.pca(adata, n_comps=n_components, random_state=seed)

            try:
                sc.pp.neighbors(adata, random_state=seed)
                sc.tl.leiden(adata, resolution=1.0, key_added="leiden")
                os.makedirs(save_dir, exist_ok=True)
                adata.write_h5ad(j_(save_dir, f'{slide_id}.h5ad'))
            except Exception as e:
                print(e)
                continue
            

def main(args):
    train_kwargs = dict(data_source=args.data_source)
    
    csv_splits = {}
    csv_splits["train"] = pd.read_csv(args.dataset_csv)

    dataset_splits = build_datasets(csv_splits,
                                    batch_size=1,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs)

    loader_train = dataset_splits['train']

    weights = slide_level_cluster(loader_train, 
                                  dataset_csv=args.dataset_csv,
                                  patch_size=args.patch_size,
                                  cluster_method=args.cluster_method)


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--patch_size', type=int, default=512, help="patch size under 40x magnification")
# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--dataset_csv', type=str, default=None)
parser.add_argument('--cluster_method', type=str, default='spatialleiden',
                    choices=["spatialleiden", "faiss", "leiden"])
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()


if __name__ == "__main__":
    print(args.dataset_csv)

    args.data_source = [src for src in args.data_source.split(',')]
    results = main(args)