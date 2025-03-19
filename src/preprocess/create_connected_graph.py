import sys
sys.path.append('../../src')
import os
import ipdb
import argparse
import pandas as pd
import scanpy as sc 
import random
import numpy as np
from tqdm import tqdm
from pprint import pprint
import nmslib
import torch
from sklearn.neighbors import KDTree
from skimage.measure import regionprops
from scipy.ndimage import label as sc_label
from scipy.ndimage import generate_binary_structure
from torch_geometric.data import Data as geomData
from mil_models.OT.otk.layers import OTKernel
from mil_models.PANTHER.layers import PANTHERBase
from utils.file_utils import save_hdf5
from utils.proto_utils import cluster
from utils.file_utils import save_pkl, load_pkl, j_
'''
CUDA_VISIBLE_DEVICES=4 python create_connected_graph.py --dataset panda_wholesight \
    --task classification \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --agg_type panther \
    --n_proto 64 \
    --data_source /data1/r20user2/wsi_data/PANDA/extracted_mag20x_patch256/uni_pt_patch_features/h5_files

CUDA_VISIBLE_DEVICES=3 python create_connected_graph.py --dataset ebrains \
    --task classification \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --agg_type ot \
    --n_proto 64 \
    --data_source /data1/r20user2/wsi_data/EBRAINS/extracted_mag20x_patch256/uni_pt_patch_features/h5_files

CUDA_VISIBLE_DEVICES=3 python create_connected_graph.py --dataset NSCLC \
    --task classification \
    --dataset_csv_dir /home/fywang/Documents/SPANTHER/src/dataset_csv \
    --agg_type ot \
    --n_proto 16 \
    --data_source /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files
'''

parser = argparse.ArgumentParser(description='Create connected graph from leiden clustering results')
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
parser.add_argument('--skip_existed', action='store_true')
parser.add_argument("--agg_type", type=str, default="ot",
                    choices=["ot", "mean", "panther"])
parser.add_argument("--n_proto", type=int, default=64)
parser.add_argument("--split_dir", type=str, default="/home/fywang/Documents/SPANTHER/src/splits")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = os.path.join(args.dataset_csv_dir, args.dataset, f"{args.task}.csv")
slide_level_prototype = os.path.join(args.data_source, "../cluster_feats")
# slide_level_prototype = os.path.join(args.dataset_csv_dir, args.dataset, "prototypes")
if args.task == "classification":
    save_dir = os.path.join(args.data_source, f"../superpatch_connected_{args.agg_type}_C{args.n_proto}_graph")
    save_h5_dir = os.path.join(args.data_source, f"../superpatch_connected_{args.agg_type}_C{args.n_proto}")
else:
    raise NotImplementedError
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_h5_dir, exist_ok=True)
# check this for survival tasks
# using superpatch prototypes
proto_path = j_(args.split_dir, args.task, args.dataset, 
                f"prototypes/prototypes_c{args.n_proto}_uni_pt_patch_features_superpatch_mean_faiss_num_1.0e+05.pkl")
assert os.path.exists(proto_path), f"proto_path: {proto_path} does not exist"
prototypes = load_pkl(proto_path)['prototypes'].squeeze()
pprint(vars(args))

# define model for aggregation
if args.agg_type == "ot":
    attention = OTKernel(in_dim=1024, 
                         out_size=args.n_proto, 
                         distance="euclidean",
                         heads=1, 
                         max_iter=100, 
                         eps=0.1).to(device)
    attention.weight.data.copy_(torch.from_numpy(prototypes).to(device))
elif args.agg_type == "panther":
    attention = PANTHERBase(
        1024, 
        p=args.n_proto, 
        L=1,
        tau=1., 
        out="allcat", 
        ot_eps=1,
        load_proto=True, 
        proto_path=proto_path,
        fix_proto=True).to(device)
elif args.agg_type == "mean":
    pass
else:
    raise NotImplementedError


def create_region_representations(x, mode="ot"):
    x = torch.tensor(x).to(device).unsqueeze(0)
    if mode == "ot":
        out = attention(x)
        # weighted_mean
        # out = out.mean(dim=1).detach().cpu().numpy()
        # allcat
        out = out.reshape(1, -1).detach().cpu().numpy()
    else:
        out, qqs = attention(x)
        out = out.reshape(1, -1).detach().cpu().numpy()
    return out

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
        filter_indices = indices[dist <= 800]
        if len(filter_indices) == 1:
            # can't find neighbor point ...
            return indices[:2]
        else:
            return filter_indices

def rowwise_isin(array, values):
    """
    Written by ChatGPT :)
    Checks if each row in 'array' is in 'values' row-wise.
    
    Parameters:
    - array: np.ndarray, the main array where rows are to be checked
    - values: np.ndarray, the array containing rows to check against
    
    Returns:
    - np.ndarray of bools, where each element corresponds to whether
      a row in 'array' is found in 'values'
    """
    # Ensure both arrays are 2D
    array = np.atleast_2d(array)
    values = np.atleast_2d(values)

    # Use broadcasting to compare all rows of 'array' to all rows of 'values'
    # Find the rows in 'values' that match each row in 'array'
    matches = (array[:, None] == values).all(axis=2)

    # If a row in 'array' has any matching row in 'values', mark it as True
    return matches.any(axis=1)


def main():
    df = pd.read_csv(csv_path)
    for row in tqdm(df.itertuples(), total=len(df)):
        slide_id = row.slide_id
        save_path = os.path.join(save_dir, f"{slide_id}.pt")
        save_h5_path = os.path.join(save_h5_dir, f"{slide_id}.h5")

        if args.skip_existed and os.path.exists(save_path):
            continue

        if args.skip_existed and os.path.exists(save_h5_path):
            continue

        slide_prototype_path = os.path.join(slide_level_prototype, f"{slide_id}.h5ad")
        assert os.path.exists(slide_prototype_path), f"slide_prototype_path: {slide_prototype_path} does not exist"
        slide_prototype = sc.read(slide_prototype_path)
        # spatial_leiden_clusters = adata.obs['spatialleiden'].values.astype(np.int32)
        # num_clusters = len(np.unique(spatial_leiden_clusters))
        slide_embeddings = slide_prototype.X
        slide_coords = slide_prototype.obsm["spatial"]
        tree = KDTree(slide_coords)
        max_coords = np.max(slide_prototype.obsm["spatial"], axis=0) + 1

        image = np.zeros((int(max_coords[0]), int(max_coords[1]))) 
        unique_labels = np.unique(slide_prototype.obs["spatialleiden"])
        for label in unique_labels:
            coords = slide_prototype.obsm["spatial"][slide_prototype.obs["spatialleiden"] == label]
            for coord in coords:
                image[int(coord[0]), int(coord[1])] = label + 1

        print("Create nodes from connected components:")
        node_embs = []
        centroids = []
        labels = []
        unique_labels = np.unique(image)
        for label_value in unique_labels:
            if label_value == 0:  # Skip background if applicable
                continue
            binary_mask = (image == label_value).astype(np.uint8)
            
            # Get connected components
            s = generate_binary_structure(2, 2)
            labeled_array, num_features = sc_label(binary_mask, structure=s)
            regions = regionprops(labeled_array)
            
            # Add each connected component as a node in the graph and store centroid info
            for region in regions:
                
                # if region.area < 15:
                #     continue
                centroid = region.centroid
                centroids.append(centroid)
                labels.append(label_value)
                
                region_embeddings = []
                for coord in region.coords:
                    _, indices = tree.query(coord.reshape(1, -1), k=1)
                    region_embeddings.append(slide_embeddings[indices[0]].reshape(-1))
                region_embeddings = np.array(region_embeddings)

                # TODO: maybe need hierarchical clustering to find a better representation ...
                if args.agg_type in ["ot", "panther"]:
                    agg_center_embeddings = create_region_representations(region_embeddings, mode=args.agg_type)
                    node_embedding = agg_center_embeddings[0]
                elif args.agg_type == "mean":
                    node_embedding = np.mean(region_embeddings, axis=0)
                else:
                    raise NotImplementedError
                node_embs.append(node_embedding)

        # Convert centroids to array format for k-NN computation
        centroids = np.array(centroids)
        node_embs = np.array(node_embs)

        if len(node_embs) == 0:
            print(f"Slide {slide_id} has no connected components")
            continue
        
        asset_dict = {'features': node_embs, 'coords': centroids}
        save_hdf5(save_h5_path, asset_dict, attr_dict=None, mode='w')

        num_patches = len(node_embs)

        model = Hnsw(space='l2')
        model.fit(centroids)

        print("Create edges:")
        radius = 5
        edge_spatial_list = []
        for v_idx in range(num_patches):
            neighbors = model.spatial_query(centroids[v_idx], topn=radius)[1:]
            a = np.array([v_idx] * len(neighbors))
            edge_spatial_list.append(np.stack([a, neighbors]))

        edge_spatial = np.hstack(edge_spatial_list).astype(np.int64)
        edge_spatial = torch.tensor(edge_spatial).long()

        G = geomData(x=torch.Tensor(node_embs),
                    edge_index=edge_spatial,
                    centroid=torch.Tensor(centroids))
        torch.save(G, save_path)


if __name__ == '__main__':
    main()