import json
import glob
import random
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='./')
parser.add_argument("--data_dir", type=str, default='/home/hao/Desktop/3D_train_npz_random_10percent_16G')
parser.add_argument("--selected_dataset", nargs='+', type=str,
                    default=['CT_LiverTumor', 'CT_AbdomenAtlas'], help='List of datasets')
# parser.add_argument("--selected_dataset", nargs='+', type=str,
#                     default=['CT_LiverTumor'], help='List of datasets')
parser.add_argument("--train_ratio", type=float, default=1)



def get_split(data_dir, selected_dataset, save_dir, train_ratio=1):
    npz_files = glob.glob(f"{data_dir}/**/*.npz", recursive=True)
    split = {'train': [], 'val': []}
    split_full = {'train': [], 'val': []}
    count = 0
    for dataset in selected_dataset:
        tmp_list = []
        for i in range(0, len(npz_files), int(1/train_ratio)):
            npz_file = npz_files[i]
            if dataset in npz_file:
                # if count > 20:
                #     continue
                tmp_list.append(npz_file)
                count += 1
        random.shuffle(tmp_list)

        cutoff = len(tmp_list) // 8
        split['val'].extend(tmp_list[:cutoff])
        split['train'].extend(tmp_list[cutoff:])


    random.shuffle(npz_files)
    cutoff = len(npz_files) // 8
    split_full['val'] = npz_files[:cutoff]
    split_full['train'] = npz_files[cutoff:]
    print(len(npz_files))
    json_filename = os.path.join(save_dir, 'split_small.json')
    with open(json_filename, "w") as json_file:
        json.dump(split, json_file, indent=4)


    json_filename = os.path.join(save_dir, 'split_full.json')
    with open(json_filename, "w") as json_file:
        json.dump(split_full, json_file, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    get_split(args.data_dir, args.selected_dataset, args.save_dir, train_ratio=args.train_ratio)