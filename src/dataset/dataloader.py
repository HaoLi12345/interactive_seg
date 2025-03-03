from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import pickle
import random
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
import cc3d, math, json, scipy.ndimage
import matplotlib.pyplot as plt
import skimage.morphology as morph
from triton.language import dtype

required_keys_train = {'imgs', 'gts', 'spacing', 'boxes'}
required_keys_test = {'imgs', 'spacing', 'boxes'}


class Dataset_promise(Dataset):
    def __init__(self, split='train', image_size=128, args=None):
        self.args = args

        if split == 'test':
            self.data = sorted(os.listdir(args.data_dir))
            self.data_dir = args.data_dir
            self.label_dir = args.label_dir
        else:
            with open(args.split_path, "r") as json_file:
                data_split = json.load(json_file)
            self.data = data_split[split]

        self.image_size = (image_size, image_size, image_size)
        self.split = split


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.split == 'test':
            filepath = os.path.join(self.data_dir, self.data[index])
        else:
            filepath = self.data[index]

        npz_file = np.load(filepath, allow_pickle=True)
        file_keys = set(npz_file.files)

        required_keys = required_keys_test if self.split == 'test' else required_keys_train
        missing_keys = required_keys - file_keys
        if missing_keys:
            if self.split != 'test':
                filepath = self.data[0]
            else:
                filepath = os.path.join(self.data_dir, self.data[0])

        npz_file = np.load(filepath, allow_pickle=True)

        if self.split != 'test':
            image, gt, spacing, box = npz_file['imgs'], npz_file['gts'], npz_file['spacing'], npz_file['boxes']
            #assert len(np.unique(gt)) - 1 == len(box), 'class number is not matching between gt and boxes'
        else:
            image, spacing, box = npz_file['imgs'], npz_file['spacing'], npz_file['boxes']
            gt_path = os.path.join(self.label_dir, filepath.split('/')[-1])
            gt = np.load(gt_path, allow_pickle=True)['gts']

        box_array = np.array(
            [[b['z_min'], b['z_max'], b['z_mid_y_min'], b['z_mid_y_max'], b['z_mid_x_min'], b['z_mid_x_max']]
             for b in box], dtype=np.int32)

        data = {'image': image, 'gt': gt, 'spacing': spacing, 'box': box_array, 'path': filepath}
        return data


        ## FIXME you could use resample as below
        # resampled_image, resampled_gt, new_spacing, new_box = self.resample_to_128(image, gt, spacing, box)
        # resampled_image = np.expand_dims(resampled_image, 0).astype(np.float32)
        # resampled_gt = np.expand_dims(resampled_gt, 0).astype(np.float32)
        # # box for each row: 'z_min', 'z_max', 'z_mid', 'z_mid_x_min', 'z_mid_y_min', 'z_mid_x_max', 'z_mid_y_max'
        # data = {'image': resampled_image, 'gt': resampled_gt, 'spacing': new_spacing, 'box': new_box, 'path': filepath}
        # return data

        # sitk_image = sitk.GetImageFromArray(image)
        # sitk_image.SetSpacing(spacing)
        #
        # sitk_gt = sitk.GetImageFromArray(gt)
        # sitk_gt.SetSpacing(spacing)
        #
        # subject = tio.Subject(
        #     image=tio.ScalarImage.from_sitk(sitk_image),
        #     label=tio.LabelMap.from_sitk(sitk_gt),
        # )
        #
        # subject = self.transform(subject)
        # return subject.image.data.clone().detach().float(), subject.label.data.clone().detach().float(), filepath


    def resample_to_128(self, image, gt, spacing, boxes):
        target_shape = (128, 128, 128)
        #target_shape = (96, 128, 256)
        original_shape = image.shape  # (D, H, W)

        scale_factors = [target_dim / orig_dim for target_dim, orig_dim in zip(target_shape, original_shape)]
        resampled_image = scipy.ndimage.zoom(image, scale_factors, order=1)
        resampled_gt = scipy.ndimage.zoom(gt, scale_factors, order=0)
        new_spacing = tuple(orig_spacing / scale for orig_spacing, scale in zip(spacing, scale_factors))

        # Convert boxes to N×7 array
        new_boxes = np.array([
            [
                int(box['z_min'] * scale_factors[0]),
                int(box['z_mid_y_min'] * scale_factors[1]),
                int(box['z_mid_x_min'] * scale_factors[2]),
                int(box['z_max'] * scale_factors[0]),
                #int(box['z_mid'] * scale_factors[0]),
                int(box['z_mid_y_max'] * scale_factors[1]),
                int(box['z_mid_x_max'] * scale_factors[2])
            ]
            for box in boxes
        ], dtype=np.int32)

        if boxes.size > 1:
            select_index = random.randint(1, boxes.size)
        else:
            select_index = boxes.size
        selected_box = new_boxes[select_index - 1]
        resampled_gt = (resampled_gt == (select_index))
        return resampled_image, resampled_gt, new_spacing, selected_box


class Dataloader_promise(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def test_resample():
    original_shape = (158, 548, 245)
    binary_mask = np.zeros(original_shape, dtype=np.uint8)

    num_objects = 3
    boxes = []
    np.random.seed(42)

    for _ in range(num_objects):
        z_seed = np.random.randint(20, 140)
        y_seed = np.random.randint(50, 500)
        x_seed = np.random.randint(50, 200)

        blob_size = np.random.randint(20, 50)  # Larger blobs
        struct_element = morph.ball(blob_size // 2)  # Ensure a valid ball size

        z_end = min(z_seed + struct_element.shape[0], original_shape[0])
        y_end = min(y_seed + struct_element.shape[1], original_shape[1])
        x_end = min(x_seed + struct_element.shape[2], original_shape[2])

        binary_mask[z_seed:z_end, y_seed:y_end, x_seed:x_end] = struct_element[:z_end - z_seed, :y_end - y_seed,
                                                                :x_end - x_seed]

    labeled_mask, num_features = scipy.ndimage.label(binary_mask)

    for obj_id in range(1, num_features + 1):
        coords = np.array(np.where(labeled_mask == obj_id))

        z_min, y_min, x_min = coords.min(axis=1)
        z_max, y_max, x_max = coords.max(axis=1)

        z_mid = (z_min + z_max) // 2  # Middle slice in Z

        boxes.append({
            'z_min': z_min, 'z_max': z_max, 'z_mid': z_mid,
            'z_mid_x_min': x_min, 'z_mid_y_min': y_min,
            'z_mid_x_max': x_max, 'z_mid_y_max': y_max
        })

    target_shape = (128, 128, 128)
    scale_factors = [target_dim / orig_dim for target_dim, orig_dim in zip(target_shape, original_shape)]
    resampled_mask = scipy.ndimage.zoom(binary_mask, scale_factors, order=0)
    new_boxes = []
    for box in boxes:
        new_box = {
            'z_min': int(box['z_min'] * scale_factors[0]),
            'z_max': int(box['z_max'] * scale_factors[0]),
            'z_mid': int(box['z_mid'] * scale_factors[0]),
            'z_mid_x_min': int(box['z_mid_x_min'] * scale_factors[2]),
            'z_mid_y_min': int(box['z_mid_y_min'] * scale_factors[1]),
            'z_mid_x_max': int(box['z_mid_x_max'] * scale_factors[2]),
            'z_mid_y_max': int(box['z_mid_y_max'] * scale_factors[1])
        }
        new_boxes.append(new_box)

    fig, axes = plt.subplots(2, len(boxes), figsize=(10, 10))

    for idx, box in enumerate(boxes):
        z_mid = box['z_mid']
        slice_img = binary_mask[z_mid]  # Extracting the 2D slice

        axes[0, idx].imshow(slice_img, cmap='gray')
        rect = plt.Rectangle((box['z_mid_x_min'], box['z_mid_y_min']),
                             box['z_mid_x_max'] - box['z_mid_x_min'],
                             box['z_mid_y_max'] - box['z_mid_y_min'],
                             linewidth=2, edgecolor='blue', facecolor='none')
        axes[0, idx].add_patch(rect)
        axes[0, idx].set_title(f"Original Box {idx}: Slice {z_mid}")
        axes[0, idx].axis("off")

    for idx, box in enumerate(new_boxes):
        z_mid = box['z_mid']
        slice_img = resampled_mask[z_mid]  # Extracting the 2D slice

        axes[1, idx].imshow(slice_img, cmap='gray')
        rect = plt.Rectangle((box['z_mid_x_min'], box['z_mid_y_min']),
                             box['z_mid_x_max'] - box['z_mid_x_min'],
                             box['z_mid_y_max'] - box['z_mid_y_min'],
                             linewidth=2, edgecolor='red', facecolor='none')
        axes[1, idx].add_patch(rect)
        axes[1, idx].set_title(f"Transformed Box {idx}: Slice {z_mid}")
        axes[1, idx].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #test_resample()
    import time

    split_path = '../useful_scripts/split_small_backup.json'
    with open(split_path, "r") as json_file:
        data_split = json.load(json_file)
    train_split, val_split = data_split['train'], data_split['val']

    shape_list = []


    for file_path in train_split:
        start = time.time()
        npz_file = np.load(file_path, allow_pickle=True)
        image, gt, spacing, box = npz_file['imgs'], npz_file['gts'], npz_file['spacing'], npz_file['boxes']
        shape_list.append([image.shape[0], image.shape[1], image.shape[2]])
        print(f'spent time per file: {time.time() - start}')

    print(np.unique(shape_list))
    # print(f'unique heights: {np.unique(height_list)}')
    # print(f'unique width: {np.unique(width_list)}')

    #npz_file = np.load(filepath, allow_pickle=True)







