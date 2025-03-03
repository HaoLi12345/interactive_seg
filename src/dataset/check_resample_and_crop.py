import numpy as np
import torch
import random
import SimpleITK as sitk
import torch.nn.functional as F

def crop_roi(image, gt, box):
        z_slices = slice(box['z_min'], box['z_max'])
        y_slices = slice(box['z_mid_y_min'], box['z_mid_y_max'])
        x_slices = slice(box['z_mid_x_min'], box['z_mid_x_max'])

        image_crop = image[:, z_slices, y_slices, x_slices]
        gt_crop = gt[:, z_slices, y_slices, x_slices]
        return image_crop, gt_crop


def put_back_roi(image, image_crop, gt_crop, box_class):
    blank_image, blank_gt = torch.zeros_like(image), torch.zeros_like(image)
    blank_image[:, box_class['z_min']: box_class['z_max'], box_class['z_mid_y_min']: box_class['z_mid_y_max'], box_class['z_mid_x_min']: box_class['z_mid_x_max']] = image_crop
    blank_gt[:, box_class['z_min']: box_class['z_max'], box_class['z_mid_y_min']: box_class['z_mid_y_max'], box_class['z_mid_x_min']: box_class['z_mid_x_max']] = gt_crop
    # sitk_image = sitk.GetImageFromArray(blank_image[0].cpu().numpy())
    # sitk_gt = sitk.GetImageFromArray(blank_gt[0].cpu().numpy())
    # sitk.WriteImage(sitk_image, '123_box_crop.nii.gz')
    # sitk.WriteImage(sitk_gt, '123_gt_box_crop.nii.gz')
    return blank_image, blank_gt


def resample_torch(image, gt, target_shape = (128, 128, 128)):
    original_dim = image.dim()
      # Desired shape
    if image.dim() == 4:  # (1, D, W, G)
        image = image.unsqueeze(1)  # Make it (1, 1, D, W, G)
        gt = gt.unsqueeze(1)
    resampled_image = F.interpolate(image, size=target_shape, mode='trilinear', align_corners=False)
    resampled_gt = F.interpolate(gt, size=target_shape, mode='nearest')
    if original_dim == 4:
        resampled_image, resampled_gt = resampled_image.squeeze(1), resampled_gt.squeeze(1)

    # sitk_image = sitk.GetImageFromArray(resampled_image[0].cpu().numpy())
    # sitk_gt = sitk.GetImageFromArray(resampled_gt[0].cpu().numpy())
    # sitk.WriteImage(sitk_image, '123_box_crop_resample_128.nii.gz')
    # sitk.WriteImage(sitk_gt, '123_gt_box_crop_resample_128.nii.gz')
    return resampled_image, resampled_gt



file_path = '/home/hao/Desktop/3D_train_npz_random_10percent_16G/CT/CT_TotalSeg-vertebrae/CT_totalseg-vertebrae_s1061.npz'
npz_file = np.load(file_path, allow_pickle=True)
image_raw, gt_raw, spacing, box = npz_file['imgs'], npz_file['gts'], npz_file['spacing'], npz_file['boxes']


image = torch.from_numpy(image_raw).unsqueeze(0)
gt = torch.from_numpy(gt_raw).unsqueeze(0)
# box = torch.from_numpy(box).unsqueeze(0)
a = torch.unique(gt)
total_class = torch.unique(gt) - 1


for i in range(0, len(total_class)):
    gt_class = (gt == (i+1)).float()
    print(torch.unique(gt_class))
    box_class = box[i]

    image_crop, gt_crop = crop_roi(image.float(), gt_class, box_class)
    image_resample, gt_resample = resample_torch(image_crop, gt_crop)
    image_resample_back, gt_resample = resample_torch(image_resample, gt_resample, target_shape=(gt_crop.size(1), gt_crop.size(2), gt_crop.size(3)))
    putback_image, putback_gt = put_back_roi(image, image_resample_back, gt_resample, box_class)




    print(1)
