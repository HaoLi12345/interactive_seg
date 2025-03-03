import logging
import os.path
import torch
from utils.util import setup_logger
from config.config_args import *
import numpy as np
from torch.backends import cudnn
from src.config.config_setup import build_model, get_dataloader
import time, random
import torch.nn.functional as F
from src.utils.util import _bbox_mask
import torchio as tio
import surface_distance
from surface_distance import metrics

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


class Tester(object):
    def __init__(self, args, logger, ckpt):
        self.args = args
        self.logger = logger

        self.val_data = get_dataloader(args, split='test')

        print('loading models and setting up')
        self.sam = build_model(args, checkpoint=ckpt)

        self.image_encoder = self.sam.image_encoder.to(args.device)
        self.prompt_encoder = self.sam.prompt_encoder.to(args.device)
        self.mask_decoder = self.sam.mask_decoder.to(args.device)

    def validate(self, epoch_num):
        self.image_encoder.eval()
        self.prompt_encoder.eval()
        self.mask_decoder.eval()

        loss = self.validater(epoch_num)
        return loss

    def validater(self, epoch_num):
        with torch.no_grad():
            dice_list, dice_initial_list = [], []

            for idx, data in enumerate(self.val_data):
                image, label, box = data['image'].to(self.args.device).float(), data['gt'].to(self.args.device).float(), data['box'].to(self.args.device).float()
                last_dice, initial_dice = self.interaction(self.sam, image, label, box)
                dice_initial_list.append(np.mean(initial_dice))
                dice_list.append(np.mean(last_dice))

                self.logger.info(' subject: ' + str(data['path'][0].split('/')[-1]) + ' init dice:' + str(round(np.mean(initial_dice), 2))
                                 + ' last dice:' + str(round(np.mean(last_dice), 2))
                                 + ' improvements:' + str(round(np.mean(last_dice) - np.mean(initial_dice), 2))
                                 + ' class num:' + str(len(last_dice))
                                 + ' total iteration:' + str(self.args.iter_nums)
                                 )

            self.logger.info("- all subjects: mean dice: " + str(np.mean(dice_list)) + "- mean initial dice: " + str(np.mean(dice_initial_list)))
        return np.mean(dice_list)

    def get_next_point(self, prev_seg, label, cube=False):  # prev_seg --> probability
        batch_points = []
        batch_labels = []

        pred_masks = (prev_seg > 0.5)
        true_masks = (label > 0)
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

        to_point_mask = torch.logical_or(fn_masks, fp_masks)

        points_list = [len(torch.argwhere(to_point_mask[i])) for i in range(to_point_mask.size(0))]
        points_min = min(points_list)
        num_clicks = self.args.num_clicks
        click_size = num_clicks
        dynamic_size = click_size

        for i in range(label.shape[0]):
            bp_list, bl_list = [], []
            points = torch.argwhere(to_point_mask[i])
            replace = False if len(points) > dynamic_size else True
            point_index = np.random.choice(len(points), size=dynamic_size, replace=replace)
            points_select = points[point_index]

            if not cube:
                for click_index in range(dynamic_size):
                    point = points_select[click_index]
                    if fn_masks[i, 0, point[1], point[2], point[3]]:
                        is_positive = True
                    else:
                        is_positive = False

                    bp = point[1:].clone().detach().reshape(1, 1, 3)
                    bl = torch.tensor([int(is_positive), ]).reshape(1, 1)
                    bp_list.append(bp)
                    bl_list.append(bl)
            else:
                B, _, D, H, W = label.shape  # Get volume dimensions

                for click_index in range(dynamic_size):
                    center_point = points_select[click_index][1:]  # Get (D, H, W) coordinates
                    is_positive = fn_masks[i, 0, center_point[0], center_point[1], center_point[2]]  # Assign label

                    # Generate 3x3x3 neighborhood around the center point
                    local_points = []
                    local_labels = []

                    for dz in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                neighbor = center_point + torch.tensor([dz, dy, dx]).to(label.device)

                                # Ensure neighbor is within bounds
                                if (0 <= neighbor[0] < D) and (0 <= neighbor[1] < H) and (0 <= neighbor[2] < W):
                                    local_points.append(neighbor.clone().detach().reshape(1, 3).to(label.device))
                                    local_labels.append(
                                        torch.tensor([int(is_positive)], dtype=torch.long).reshape(1).to(label.device))

                    # Stack local neighborhood points and labels
                    local_points = torch.cat(local_points, dim=0)  # Shape: (27, 3)
                    local_labels = torch.cat(local_labels, dim=0)  # Shape: (27,)

                    bp_list.append(local_points.unsqueeze(0))  # Add batch dimension
                    bl_list.append(local_labels.unsqueeze(0))

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

        return batch_points, batch_labels

    def get_points(self, prev_masks, label):
        batch_points, batch_labels = self.get_next_point(prev_masks, label)

        points_co = torch.cat(batch_points, dim=0).to(self.args.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_input = points_co
        labels_input = points_la

        bbox_coords = _bbox_mask(label[:, 0, :]).to(self.args.device) if self.args.use_box else None
        return points_input, labels_input, bbox_coords


    def batch_forward(self, sam_model, image_embedding, features, prev_masks, points=None, boxes=None):
        prev_masks = F.interpolate(prev_masks, scale_factor=0.25)
        features = [features[i].to(self.args.device) for i in range(0, len(features))]

        # sparse_embeddings --> (B, 2, embed_dim) 2 represents concat of coordination and its label
        # dense_embeddings --> (B, embed_dim, W, H, D), whd values are customized
        new_point_embedding, new_image_embedding = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=prev_masks,
            image_embeddings=image_embedding.to(self.args.device)
        )

        mask, pred_dice = sam_model.mask_decoder(
            prompt_embeddings=new_point_embedding,  # (B, 2, 256)
            image_embeddings=new_image_embedding,  # (B, 256, 64, 64)
            feature_list=features,
        )
        mask = F.interpolate(mask, scale_factor=2)
        return mask, pred_dice

    def interaction(self, sam_model, image, label, box, filename=None):
        total_class = torch.unique(label)
        subject_dice, subject_dice_initial = [], []


        for current_class in range(1, len(total_class)):
            label_class = (label == (current_class)).float()
            original_volume = torch.sum(label_class).item()

            box_class = box[0, current_class - 1]

            image_crop, label_crop = self.crop_roi(image.float(), label_class, box_class)
            image_crop = (image_crop - image_crop.mean()) / (image_crop.std() + 1e-8)
            image_resample, label_resample = self.resample_torch(image_crop, label_crop)

            image_resample, label_resample = image_resample.unsqueeze(0), label_resample.unsqueeze(0)
            image_embedding, feature_list = self.sam.image_encoder(image_resample)

            ## TODO: you can move these lists out of the loop, and set positive "prompts of other classes" as "negative prompt"
            self.click_points = []
            self.click_labels = []

            prev_masks = torch.zeros_like(image_resample, dtype=torch.float).to(label.device)
            last_dice, initial_dice = 0, 0
            for iter_num in range(self.args.iter_nums):

                prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

                if iter_num == 0:
                    points = None
                else:
                    points_input, labels_input, _ = self.get_points(prev_masks_sigmoid, label_resample)
                    points = [points_input, labels_input]
                seg_iter, _ = self.batch_forward(sam_model, image_embedding, feature_list, prev_masks_sigmoid, points=points, boxes=None)

                mask_refine, _ = self.sam.mask_decoder.refine(image_resample, seg_iter,
                                                              [self.click_points, self.click_labels], seg_iter.detach())
                prev_masks = mask_refine

                ## TODO: uncomment below if you want to save the prediction for each
                # pred = self.resample_torch_label((torch.sigmoid(mask_refine)>0.5).float(), target_shape=(label_crop.size(1), label_crop.size(2), label_crop.size(3)))
                # pred_putback = self.put_back_roi(image, pred, box_class)

                if iter_num == 0:
                    initial_dice = self.get_dice_score(torch.sigmoid(mask_refine), label_resample)
                    subject_dice_initial.append(round(initial_dice, 4))
                if iter_num == (self.args.iter_nums - 1):
                    last_dice = self.get_dice_score(torch.sigmoid(mask_refine), label_resample)
                    if not filename:
                        filename = 'unknown'
                    print(f'current class {current_class}, initial dice: {round(initial_dice, 2)}'
                          f' ---------> improvement: {round(last_dice - initial_dice, 2)},   volume: {original_volume}')
                    subject_dice.append(round(last_dice, 4))
            #print(f'subject last dice {subject_dice}')

        return subject_dice, subject_dice_initial

    def get_dice_score(self, prev_masks, label):
        def compute_dice(mask_pred, mask_gt):
            mask_pred = (mask_pred > 0.5)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum() + 0.000001
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum() + 0.000001
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (label > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()


    def crop_roi(self, image, gt, box):
        # box = torch.tensor(b['z_min'], b['z_max'], b['z_mid_y_min'], b['z_mid_y_max'], b['z_mid_x_min'], b['z_mid_x_max'])
        box = box.long()
        z_slices = slice(box[0], box[1]+1)
        y_slices = slice(box[2], box[3]+1)
        x_slices = slice(box[4], box[5]+1)

        image_crop = image[:, z_slices, y_slices, x_slices]
        gt_crop = gt[:, z_slices, y_slices, x_slices]
        return image_crop, gt_crop

    def put_back_roi(self, image, gt_crop, box_class):
        blank_gt = torch.zeros_like(image)

        blank_gt[:, box_class['z_min']: box_class['z_max']+1, box_class['z_mid_y_min']: box_class['z_mid_y_max']+1,
        box_class['z_mid_x_min']: box_class['z_mid_x_max']+1] = gt_crop
        return blank_gt


    def resample_torch(self, image, gt, target_shape=(128, 128, 128)):
        original_dim = image.dim()
        if image.dim() == 4:  # (1, D, W, G)
            image = image.unsqueeze(1)  # Make it (1, 1, D, W, G)
            gt = gt.unsqueeze(1)
        resampled_image = F.interpolate(image, size=target_shape, mode='trilinear', align_corners=False)
        resampled_gt = F.interpolate(gt, size=target_shape, mode='nearest')
        if original_dim == 4:
            resampled_image, resampled_gt = resampled_image.squeeze(1), resampled_gt.squeeze(1)
        return resampled_image, resampled_gt

    def resample_torch_label(self, gt, target_shape=(128, 128, 128)):
        original_dim = gt.dim()
        if gt.dim() == 4:  # (1, D, W, G)
            gt = gt.unsqueeze(1)
        resampled_gt = F.interpolate(gt, size=target_shape, mode='nearest')
        if original_dim == 4:
            resampled_gt = resampled_gt.squeeze(1)
        return resampled_gt



def main():
    init_seeds()
    args = parser.parse_args()
    check_and_setup_parser(args)

    log_name = 'test_' + args.save_name
    setup_logger(logger_name=log_name, root=args.save_dir, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    ckpt = os.path.join(args.save_dir, args.checkpoint + '.pth.tar')
    with torch.no_grad():
        tester = Tester(args, logger, ckpt)
        loss = tester.validate(epoch_num=0)

        print(loss)

    logger.info("- Test done")

if __name__ == "__main__":
    main()