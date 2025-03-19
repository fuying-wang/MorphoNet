from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import ipdb

'''
python create_dataset_csv.py --task classification --dataset RCC \
    --output_dir /home/fywang/Documents/SPANTHER/src/dataset_csv

python create_dataset_csv.py --task classification --dataset BRCA \
    --output_dir /home/fywang/Documents/SPANTHER/src/dataset_csv
'''

parser = ArgumentParser()
parser.add_argument('--task', type=str, default='survival')
parser.add_argument('--dataset', type=str, default='TCGA_BRCA')
parser.add_argument('--split_dir', type=str, 
                    default="/home/fywang/Documents/SPANTHER/src/splits")
parser.add_argument('--n_fold', type=int, default=5)
parser.add_argument('--output_dir', type=str, default=None)
args = parser.parse_args()


args.split_dir = os.path.join(args.split_dir, args.task)

def main():
    if args.task == 'survival':
        all_df = []

        for fold in range(args.n_fold):
            if args.dataset == "CPTAC_KIRC":
                df = pd.read_csv(os.path.join(args.split_dir, f"TCGA_KIRC_overall_survival_k={fold}", 'filtered_test_cptac.csv'))
            else:
                train_df = pd.read_csv(os.path.join(args.split_dir, f"{args.dataset}_overall_survival_k={fold}", 'train.csv'))
                if args.dataset == "TCGA_LUAD":
                    test_df = pd.read_csv(os.path.join(args.split_dir, f"{args.dataset}_overall_survival_k={fold}", 'test_tcgaluad.csv'))
                else:
                    test_df = pd.read_csv(os.path.join(args.split_dir, f"{args.dataset}_overall_survival_k={fold}", 'test.csv'))
                df = pd.concat([train_df, test_df], axis=0)
            df = df.reset_index(drop=True)
            all_df.append(df)

        all_df = pd.concat(all_df, axis=0)
        all_df.drop_duplicates(subset=['slide_id'], keep='first', inplace=True)
        all_df = all_df.reset_index(drop=True)

        save_dir = os.path.join(args.output_dir, args.dataset)
        os.makedirs(save_dir, exist_ok=True)
        all_df.to_csv(os.path.join(save_dir, 'survival.csv'), index=False)

    elif args.task == "classification":
        all_df = []

        for fold in range(args.n_fold):
            if args.dataset == "ebrains":
                train_df = pd.read_csv(os.path.join(args.split_dir, f"ebrains", 'train.csv'))
                val_df = pd.read_csv(os.path.join(args.split_dir, f"ebrains", 'val.csv'))
                test_df = pd.read_csv(os.path.join(args.split_dir, f"ebrains", 'test.csv'))
            elif args.dataset == "NSCLC":
                train_df = pd.read_csv(os.path.join(args.split_dir, f"NSCLC", 'train.csv'))
                val_df = pd.read_csv(os.path.join(args.split_dir, f"NSCLC", 'val.csv'))
                test_df = pd.read_csv(os.path.join(args.split_dir, f"NSCLC", 'test.csv'))
            elif args.dataset == "panda_wholesight":
                train_df = pd.read_csv(os.path.join(args.split_dir, f"panda_wholesight", 'train.csv'))
                val_df = pd.read_csv(os.path.join(args.split_dir, f"panda_wholesight", 'val.csv'))
                test_K_df = pd.read_csv(os.path.join(args.split_dir, f"panda_wholesight", 'test_K.csv'))
                test_R_df = pd.read_csv(os.path.join(args.split_dir, f"panda_wholesight", 'test_R.csv'))
                test_df = pd.concat([test_K_df, test_R_df], axis=0)
            else:
                train_df = pd.read_csv(os.path.join(args.split_dir, f"{args.dataset}", 'train.csv'))
                val_df = pd.read_csv(os.path.join(args.split_dir, f"{args.dataset}", 'val.csv'))
                test_df = pd.read_csv(os.path.join(args.split_dir, f"{args.dataset}", 'test.csv'))

            df = pd.concat([train_df, val_df, test_df], axis=0)
            df = df.reset_index(drop=True)
            all_df.append(df)

        all_df = pd.concat(all_df, axis=0)
        all_df.drop_duplicates(subset=['slide_id'], keep='first', inplace=True)
        all_df = all_df.reset_index(drop=True)

        save_dir = os.path.join(args.output_dir, args.dataset)
        os.makedirs(save_dir, exist_ok=True)
        all_df.to_csv(os.path.join(save_dir, 'classification.csv'), index=False)


if __name__ == '__main__':
    main()