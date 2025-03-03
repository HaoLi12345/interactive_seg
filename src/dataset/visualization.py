import time
import numpy as np
import json
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy, random
def crop_zeros_3d(binary_array):
    nonzero_indices = np.argwhere(binary_array)
    min_indices = nonzero_indices.min(axis=0)
    max_indices = nonzero_indices.max(axis=0)  # +1 to include max index in slice
    slices = tuple(slice(min_idx, max_idx) for min_idx, max_idx in zip(min_indices, max_indices))
    return binary_array[slices], slices


def resample_to_128(image, gt, spacing, boxes):
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


split_path = '../useful_scripts/split_full.json'
with open(split_path, "r") as json_file:
    data_split = json.load(json_file)
train_split, val_split = data_split['train'], data_split['val']

shape_list = []
required_keys = {'imgs', 'gts', 'spacing', 'boxes'}
count = 0

file_list = ['MR_WMH_T1_Utrecht_20.npz', 'PET_autoPET_psma_2e5119d4ac37d41d_2016-08-05.npz', 'CT_totalseg-vertebrae_s1061.npz']
for file_path in train_split:
    file_name = file_path.split('/')[-1]
    if file_name in file_list:
        npz_file = np.load(file_path, allow_pickle=True)
        file_keys = set(npz_file.files)

        image_raw, gt_raw, spacing, box = npz_file['imgs'], npz_file['gts'], npz_file['spacing'], npz_file['boxes']
        print(f'image name: {file_path.split("/")[-1]}, with classes: {np.unique(gt_raw)}, with shape: {image_raw.shape}')

        sitk_image = sitk.GetImageFromArray(image_raw)
        sitk_gt = sitk.GetImageFromArray(gt_raw)
        print(spacing, image_raw.shape, np.unique(gt_raw))
        sitk.WriteImage(sitk_image, '123.nii.gz')
        sitk.WriteImage(sitk_gt, '123_gt.nii.gz')
        print(f'current file {file_name}')
        print(1)



# print(np.unique(shape_list))
#
# for i in range(0, len(box)):
#     if i >= 3:
#         gt = gt_raw.copy()
#         image = image_raw.copy()
#         gt[gt != i + 1] = 0
#         image = image * gt
#         # gt1, slices  = crop_zeros_3d(gt)
#
#         z_slices = slice(box[i]['z_min'], box[i]['z_max'])
#         y_slices = slice(box[i]['z_mid_y_min'], box[i]['z_mid_y_max'])
#         x_slices = slice(box[i]['z_mid_x_min'], box[i]['z_mid_x_max'])
#
#         slices_box = (z_slices, y_slices, x_slices)
#         image, gt = image[slices_box], gt[slices_box]
#
#         sitk_image = sitk.GetImageFromArray(image)
#         sitk_gt = sitk.GetImageFromArray(gt)
#         print(spacing, image.shape, np.unique(gt))
#         sitk.WriteImage(sitk_image, '123_box_crop.nii.gz')
#         sitk.WriteImage(sitk_gt, '123_gt_box_crop.nii.gz')
#
#         image, gt, _, _ = resample_to_128(image, gt, spacing, box)
#
#         sitk_image = sitk.GetImageFromArray(image)
#         sitk_gt = sitk.GetImageFromArray(gt.astype(int))
#         print(spacing, image.shape, np.unique(gt))
#         sitk.WriteImage(sitk_image, '123_box_crop_resample_128.nii.gz')
#         sitk.WriteImage(sitk_gt, '123_gt_box_crop_resample_128.nii.gz')
#
#         print(1)