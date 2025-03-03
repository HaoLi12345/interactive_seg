import json
import glob
import random
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='./')
parser.add_argument("--data_dir", type=str, default='/home/hao/Desktop/3D_train_npz_random_10percent_16G')
parser.add_argument("--selected_dataset", nargs='+', type=str,
                    default=['CT_LiverTumor', 'CT_AbdomenAtlas'], help='List of datasets')
parser.add_argument("--train_ratio", type=float, default=1)


def get_split(data_dir, selected_dataset, save_dir, train_ratio=1):
    npz_files = glob.glob(f"{data_dir}/**/*.npz", recursive=True)
    for dataset in selected_dataset:
        for file in npz_files:
            if 'CT_LiverTumor' in file:
                npz = np.load(file, allow_pickle=True)
                image = npz['imgs']
                gt = npz['gts']
                num_class = np.unique(gt)
                spacing = npz['spacing']
                box = npz['boxes']
                a = box[0]
                b = box[1]
                c = box[2]

                assert len(np.unique(gt)) - 1 == len(box)

                #print(num_class)





if __name__ == '__main__':
    args = parser.parse_args()
    get_split(args.data_dir, args.selected_dataset, args.save_dir)