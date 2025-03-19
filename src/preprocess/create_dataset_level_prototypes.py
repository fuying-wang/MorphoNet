import sys
sys.path.append('../../src')

import faiss
import os
import h5py
import random
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import scanpy as sc
import ipdb
from torch.utils.data import DataLoader
from wsi_datasets import WSIProtoDataset
from utils.utils import seed_torch, read_splits
from utils.file_utils import save_pkl
from utils.proto_utils import cluster


'''
CUDA_VISIBLE_DEVICES=0 python create_dataset_level_prototypes.py --dataset ebrains \
    --task classification \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --split_dir /home/fywang/Documents/SPANTHER/src/splits \
    --data_source /data1/r20user2/wsi_data/EBRAINS/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_mean \
    --n_proto 16

CUDA_VISIBLE_DEVICES=0 python create_dataset_level_prototypes.py --dataset panda_wholesight \
    --task classification \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --split_dir /home/fywang/Documents/SPANTHER/src/splits \
    --data_source /data1/r20user2/wsi_data/PANDA/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_mean \
    --n_proto 16

CUDA_VISIBLE_DEVICES=0 python create_dataset_level_prototypes.py --dataset NSCLC \
    --task classification \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --split_dir /home/fywang/Documents/SPANTHER/src/splits \
    --data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_mean \
    --n_proto 32

--data_source /disk1/fywang/WSI/RCC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files \

CUDA_VISIBLE_DEVICES=2 python create_dataset_level_prototypes.py --dataset TCGA_KIRC \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --split_dir /home/fywang/Documents/SPANTHER/src/splits \
    --data_source /data1/r20user2/wsi_data/TCGA_RCC/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_mean \
    --n_proto 16

CUDA_VISIBLE_DEVICES=2 python create_dataset_level_prototypes.py --dataset TCGA_BRCA \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --split_dir /home/fywang/Documents/SPANTHER/src/splits \
    --data_source /data1/r20user2/wsi_data/TCGA_BRCA/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_mean \
    --n_proto 16 --norm

CUDA_VISIBLE_DEVICES=2 python create_dataset_level_prototypes.py --dataset TCGA_LUAD \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --split_dir /home/fywang/Documents/SPANTHER/src/splits \
    --data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_mean \
    --n_proto 16 
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
parser.add_argument("--split_dir", type=str, default=None)
parser.add_argument("--num_folds", type=int, default=5)

# model / loss fn args ###
parser.add_argument('--n_proto', type=int, help='Number of prototypes', default=32)
parser.add_argument('--n_proto_patches', type=int, default=100000,
                    help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
parser.add_argument('--n_init', type=int, default=5,
                    help='Number of different KMeans initialization (for FAISS)')
parser.add_argument('--n_iter', type=int, default=50,
                    help='Number of iterations for Kmeans clustering')
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--norm', action="store_true", default=False)
parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='kmeans')

parser.add_argument("--dataset_csv_dir", type=str, default=None)
parser.add_argument('--split_names', type=str, default='train',
                    help='delimited list for specifying names within each split')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

csv_path = os.path.join(args.dataset_csv_dir, args.dataset, f"{args.task}.csv")
# slide_level_prototype = os.path.join(args.dataset_csv_dir, args.dataset, "prototypes")
split_dir = os.path.join(args.split_dir, args.task)
args.data_source = [src for src in args.data_source.split(',')]

if args.task == "survival":
    args.num_folds = 5
elif args.task == "classification":
    args.num_folds = 1


def main():
    for fold in range(args.num_folds):
        if args.task == "survival":
            args.split_dir = os.path.join(split_dir, f"{args.dataset}_overall_{args.task}_k={fold}")
        elif args.task == "classification":
            args.split_dir = os.path.join(split_dir, f"{args.dataset}")
        print(f"current split_dir: {args.split_dir}")
        # train_kwargs = dict(data_source=args.data_source)

        seed_torch(args.seed)
        csv_splits = read_splits(args)
        print('\nsuccessfully read splits for: ', list(csv_splits.keys()))

        # dataset_splits = dict()
        for k in csv_splits.keys():
            df = csv_splits[k]
            all_mean_embeddings = []
            for row in tqdm(df.itertuples(), total=len(df)):
                slide_id = row.slide_id
                h5_path = os.path.join(args.data_source[0], f"{slide_id}.h5")
                assert os.path.exists(h5_path), f"h5_path: {h5_path} does not exist"
                with h5py.File(h5_path, 'r') as f:
                    embeddings = f['features'][:]
                # slide_prototype_path = os.path.join(slide_level_prototype, f"{slide_id}.h5ad")
                # assert os.path.exists(slide_prototype_path), f"slide_prototype_path: {slide_prototype_path} does not exist"
                # adata = sc.read(slide_prototype_path)
                # spatial_leiden_clusters = adata.obs['spatialleiden'].values.astype(np.int32)
                # num_clusters = len(np.unique(spatial_leiden_clusters))

                # center_embeddings = []
                # for i in range(num_clusters):
                #     # using the mean embeddings of the cluster as the prototype
                #     cluster_embeddings = adata[adata.obs['spatialleiden'] == i].X
                #     center_embeddings.append(np.mean(cluster_embeddings, axis=0))
                all_mean_embeddings.extend(embeddings)
            
            all_mean_embeddings = np.array(all_mean_embeddings)

            if args.norm:
                all_mean_embeddings = all_mean_embeddings / np.linalg.norm(all_mean_embeddings, axis=1, keepdims=True)

            # all_mean_embeddings = all_mean_embeddings / np.linalg.norm(all_mean_embeddings, axis=1, keepdims=True)
            # print(f"all_mean_embeddings: {all_mean_embeddings.shape}")
            # sim_matrix = all_mean_embeddings @ all_mean_embeddings.T

            # from sklearn.decomposition import PCA
            # import matplotlib.pyplot as plt
            # pca = PCA(n_components=2)
            # pca.fit(all_mean_embeddings)
            # all_mean_embeddings = pca.transform(all_mean_embeddings)
            # plt.scatter(all_mean_embeddings[:, 0], all_mean_embeddings[:, 1], s=1)
            # plt.savefig('pca.png')
            
            # TODO: we have tried DBSCAN, GaussianMixture, BayesianGaussianMixture, but they are too slow
            # # from sklearn.cluster import DBSCAN
            # # from sklearn.mixture import BayesianGaussianMixture
            # from sklearn.mixture import GaussianMixture
            # # clustering = DBSCAN(eps=0.25, min_samples=5).fit(all_mean_embeddings)
            # clustering = GaussianMixture(n_components=args.n_proto,
            #                              random_state=args.seed).fit(all_mean_embeddings)
            # # clustering = BayesianGaussianMixture(n_components=args.n_proto, 
            # #                                      random_state=args.seed).fit(all_mean_embeddings)
            # labels = clustering.predict(all_mean_embeddings)

            # unique_cluster_labels = np.unique(clustering.labels_)
            # num_clusters = len(np.delete(unique_cluster_labels, np.where(unique_cluster_labels == -1)))
            # print(f"There are total {num_clusters} clusters")
            # print(f"number of noise: {len(np.where(clustering.labels_ == -1)[0])}")
            # cluster_center_embs = []
            # for idx in unique_cluster_labels:
            #     if idx == -1:
            #         continue
            #     cluster_idx = np.where(clustering.labels_ == idx)
            #     cluster_embeddings = all_mean_embeddings[cluster_idx]
            #     cluster_center = np.mean(cluster_embeddings, axis=0)
            #     cluster_center_embs.append(cluster_center)
            
            # cluster_center_embs = np.array(cluster_center_embs)
            
            # since the above clustering algorithms are too slow for large high-dimension data, we use Faiss Kmeans
            numOfGPUs = torch.cuda.device_count()
            print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...")
            print(f"\tNum of clusters {args.n_proto}, num of iter {args.n_iter}")

            kmeans = faiss.Kmeans(all_mean_embeddings.shape[1], 
                                args.n_proto, 
                                niter=args.n_iter, 
                                nredo=args.n_init,
                                verbose=True, 
                                max_points_per_centroid=args.n_proto_patches,
                                gpu=numOfGPUs)
            kmeans.train(all_mean_embeddings)
            weights = kmeans.centroids[np.newaxis, ...]
            if args.norm:
                weights = weights / np.linalg.norm(weights, axis=2, keepdims=True)

            os.makedirs(os.path.join(args.split_dir, 'prototypes'), exist_ok=True)
            if args.norm:
                save_fpath = os.path.join(args.split_dir, 'prototypes',
                                          f"prototypes_{args.data_source[0].split('/')[-2]}_nproto_{args.n_proto}_embed_dim{all_mean_embeddings.shape[1]}_norm.pkl")
            else:
                save_fpath = os.path.join(args.split_dir, 'prototypes',
                                          f"prototypes_{args.data_source[0].split('/')[-2]}_nproto_{args.n_proto}_embed_dim{all_mean_embeddings.shape[1]}.pkl")
            save_pkl(save_fpath, {'prototypes': weights})


if __name__ == "__main__":
    main()