import argparse
import os
import pickle
import torch
from einops import rearrange
# import sys
# sys.path.append("../")
# from src.utils.utils import seed_torch, read_splits
# from src.training.main_embedding import build_datasets
import ipdb

'''
python convert_pkl_to_path.py \
    --pkl_dir /home/fywang/Documents/SPANTHER/src/splits/classification/NSCLC/embeddings \
    --out_dir /data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features
'''

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--pkl_dir", type=str, default="TCGA")
parser.add_argument("--pkl_file", type=str, default="uni_pt_patch_features_OT_embeddings_proto_16_allcat_eps_1.0")
parser.add_argument("--out_dir", type=str, default="TCGA")
# parser.add_argument("--split_dir", type=str, default="")
args = parser.parse_args()

def main():
    out_dir = f"{args.out_dir}/OT_embeddings_proto_16_allcat_eps_1.0"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{args.pkl_dir}/{args.pkl_file}.pkl", "rb") as f:
        data = pickle.load(f)
    
    for split in data.keys():
        split_vals = data[split]
        embs = split_vals["X"]
        slide_ids = split_vals["slide_ids"]
        for (s_id, emb) in zip(slide_ids, embs):
            emb = rearrange(emb, "(c d) -> c d", c=16, d=1024)
            torch.save(emb, f"{out_dir}/{s_id}.pt")    
        # os.makedirs(f"{out_dir}/{split}", exist_ok=True)
        # for k in data[split].keys():
        #     torch.save(data[split][k], f"{out_dir}/{split}/{k}.pt")
        # ipdb.set_trace()


if __name__ == "__main__":
    main()