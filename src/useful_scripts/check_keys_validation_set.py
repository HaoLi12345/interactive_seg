import os
import numpy as np

image_dir = '/home/hao/Desktop/3D_val_npz'
gt_dir = '/home/hao/Desktop/3D_val_gt_interactive_seg'
image_list = sorted(os.listdir(image_dir))
required_keys = {'imgs', 'spacing', 'boxes'}
required_keys_gt = {'gts', 'spacing', 'boxes'}

unique_keys_image, unique_keys_gt = set(), set()
for i in range(0, len(image_list)):
    image_name = image_list[i]
    file_path = os.path.join(image_dir, image_name)
    gt_path = os.path.join(gt_dir, image_name)
    if not os.path.isfile(gt_path):
        print('{} not exist'.format(gt_path))

    try:
        npz_file = np.load(file_path, allow_pickle=True)
        file_keys = set(npz_file.files)
        missing_keys = required_keys - file_keys
        unique_keys_image.update(npz_file.keys())


        npz_file_gt = np.load(gt_path, allow_pickle=True)
        file_keys_gt = set(npz_file_gt.files)
        missing_keys_gt = required_keys_gt - file_keys_gt
        unique_keys_gt.update(npz_file_gt.keys())

        if missing_keys_gt:
            print(f"GT: Skipping {npz_file_gt} (Missing keys: {missing_keys})")

        if missing_keys:
            print(f"Image: Skipping {file_path} (Missing keys: {missing_keys})")



    except Exception as e:
        print(f"Error loading {file_path}: {e}")
print("Unique keys across all NPZ files image:", unique_keys_image)
print("Unique keys across all NPZ files gt:", unique_keys_gt)

