import argparse
import os
import ipdb
import random
import torch
import h5py
import numpy as np
import pandas as pd
import numpy as np
import squidpy as sq
from torch_geometric.data import Data as geomData
from tqdm import tqdm
import nmslib

'''
CUDA_VISIBLE_DEVICES=0 python create_superpatch_graph.py --dataset TCGA_NSCLC \
    --source_dir superpatch_kmeans_mean \
    --save_dir superpatch_kmeans_mean_graph

CUDA_VISIBLE_DEVICES=0 python create_superpatch_graph.py --dataset EBRAINS \
    --source_dir superpatch_kmeans_mean \
    --save_dir superpatch_kmeans_mean_graph
'''

parser = argparse.ArgumentParser(description="Construct superpatch graph")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data_source", type=str, default="/data1/r20user2/wsi_data")
parser.add_argument("--dataset_name", type=str, default="TCGA_RCC")
parser.add_argument("--magnitude", type=int, default=20)
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--skip_existed", action="store_true", default=False)
parser.add_argument("--source_dir", type=str, default="superpatch_mean")
parser.add_argument("--save_dir", type=str, default="superpatch_graph")
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# args.h5_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/uni_pt_patch_features/h5_files"
args.h5_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/uni_pt_patch_features/{args.source_dir}"
args.save_path = f"extracted_mag{args.magnitude}x_patch{args.patch_size}/uni_pt_patch_features"

args.h5_path = os.path.join(args.data_source, args.dataset_name, args.h5_path)
args.save_path = os.path.join(args.data_source, args.dataset_name, args.save_path, args.save_dir)
os.makedirs(args.save_path, exist_ok=True)
patch_size = args.patch_size * (2 ** (40 / args.magnitude - 1))

# csv_path = os.path.join(args.dataset_csv_dir, args.dataset, f"{args.task}.csv")
# args.data_source = [src for src in args.data_source.split(',')]
# superpatch_dir = os.path.join(args.data_source, "superpatch")


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 180}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def feat_query(self, vector, topn):
        indices, dist = self.index_.knnQuery(vector, k=topn)
        indices = indices[dist <= 0.2]

        return indices

    def spatial_query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        indices = indices[dist <= 1500]

        return indices


def pt2graph(wsi_h5, radius=5, patch_size=1024):
    coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]
    cur_coords = coords

    model = Hnsw(space='l2')
    model.fit(cur_coords)

    edge_spatial_list = []
    for v_idx in range(num_patches):
        neighbors = model.spatial_query(cur_coords[v_idx], topn=radius)[1:]
        a = np.array([v_idx] * len(neighbors))
        edge_spatial_list.append(np.stack([a, neighbors]))

    edge_spatial = np.hstack(edge_spatial_list).astype(np.int64)
    edge_spatial = torch.tensor(edge_spatial).long()

    model = Hnsw(space='cosinesimil')
    model.fit(features)

    # for each patch, find 4 most similar patches
    edge_latent_list = []
    for v_idx in range(num_patches):
        neighbors = model.feat_query(features[v_idx], topn=radius)[1:]
        a = np.array([v_idx] * len(neighbors))
        edge_latent_list.append(np.stack([a, neighbors]))

    edge_latent = np.hstack(edge_latent_list).astype(np.int64)
    edge_latent = torch.tensor(edge_latent).long()

    G = geomData(x=torch.Tensor(features),
                 edge_index=edge_spatial,
                 edge_latent=edge_latent,
                 centroid=torch.Tensor(coords))
    return G


def createDir_h5toPyG(h5_path, save_path, skip_existed=True, patch_size=1024):
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        graph_file = os.path.join(save_path, h5_fname[:-3]+'.pt')

        if skip_existed:
            if os.path.exists(graph_file):
                continue

        pbar.set_description('%s - Creating Graph' % (h5_fname[:12]))

        try:
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            G = pt2graph(wsi_h5, patch_size=patch_size)
            torch.save(G, graph_file)
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')


def main():
    createDir_h5toPyG(args.h5_path, args.save_path,
                      args.skip_existed, patch_size)


if __name__ == "__main__":
    main()
