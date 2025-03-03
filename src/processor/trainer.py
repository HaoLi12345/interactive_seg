import torch
import torch.nn.functional as F
import random
import numpy as np
from click.core import batch
from src.utils.util import _bbox_mask
from .trainer_basic import Trainer_basic


class Trainer(Trainer_basic):
    def __init__(self, args, logger):
        super().__init__(args, logger)

    def forward(self, sam_model, image, label, box, iter_nums, train=False, return_each_iter=False, filename=None):

        # image [B, D, W, H]
        total_class = torch.unique(label)

        subject_loss = 0
        subject_dice, subject_dice_initial = [],[]
        for current_class in range(1, len(total_class)):
            label_class = (label == (current_class)).float()
            box_class = box[0, current_class-1]

            original_volume = torch.sum(label_class).item()

            image_crop, label_crop = self.crop_roi(image.float(), label_class, box_class)
            image_crop = (image_crop - image_crop.mean()) / (image_crop.std() + 1e-8)
            image_resample, label_resample = self.resample_torch(image_crop, label_crop)

            image_resample, label_resample = image_resample.unsqueeze(0), label_resample.unsqueeze(0)
            image_embedding, feature_list = self.sam.image_encoder(image_resample)

            ## TODO: you can move these lists out of the loop, and set positive "prompts of other classes" as "negative prompt"
            self.click_points = []
            self.click_labels = []

            prev_masks = torch.zeros_like(image_resample, dtype=torch.float).to(label.device)
            return_loss, initial_dice = 0, 0
            for iter_num in range(iter_nums):
                loss = 0
                prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

                if iter_num == 0:
                    points = None
                else:
                    points_input, labels_input, _ = self.get_points(prev_masks_sigmoid, label_resample)
                    points = [points_input, labels_input]
                seg_iter, _ = self.iteration_forward(sam_model, image_embedding, feature_list, prev_masks_sigmoid, points=points, boxes=None)

                mask_refine, _ = self.sam.mask_decoder.refine(image_resample, seg_iter,
                                                          [self.click_points, self.click_labels], seg_iter.detach())
                if train:
                    if self.args.iter_weight:
                        loss = (iter_num + 1) * (self.bce(seg_iter, label_resample) + self.bce(mask_refine, label_resample))
                    else:
                        # loss = self.l1(seg_iter, label_resample) + self.l1(mask_refine, label_resample)
                        loss = self.bce(seg_iter, label_resample) + self.bce(mask_refine, label_resample)

                if iter_num == 0:
                    initial_dice = self.get_dice_score(torch.sigmoid(mask_refine), label_resample)
                    subject_dice_initial.append(round(initial_dice, 4))
                if iter_num == (iter_nums - 1):
                    last_dice = self.get_dice_score(torch.sigmoid(mask_refine), label_resample)
                    if not filename:
                        filename = 'unknown'
                    print(f'filename: {filename}:   current class {current_class}, initial dice: {round(initial_dice, 2)}'
                          f' ---------> improvement: {round(last_dice-initial_dice, 2)},   volume: {original_volume}')
                    subject_dice.append(round(last_dice, 4))

                return_loss += loss
                prev_masks = mask_refine
            return_loss = return_loss / iter_nums
            if train:
                self.scaler.scale(return_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.sam.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            subject_loss += return_loss
        if train:
            return subject_loss / (len(total_class)-1)
        else:
            print(f'subject init dice {subject_dice_initial}')
            print(f'subject last dice {subject_dice}')

            return np.mean(subject_dice), np.mean(subject_dice_initial)


    def get_points(self, prev_masks, label, train_mode=True):
        mode = 'train' if train_mode else 'validation'

        batch_points, batch_labels = self.get_next_point(prev_masks, label, cube=True)

        points_co = torch.cat(batch_points, dim=0).to(self.args.device) # b x num_clicks x 3
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device) # b x num_clicks x 1

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_input = points_co
        labels_input = points_la

        bbox_coords = _bbox_mask(label[:, 0, :], mode=mode, dynamic=self.args.dynamic_box).to(self.args.device) if self.args.use_box else None

        return points_input, labels_input, bbox_coords

    def get_next_point(self, prev_seg, label, cube=False): # prev_seg --> probability
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
                                    local_labels.append(torch.tensor([int(is_positive)], dtype=torch.long).reshape(1).to(label.device))

                    # Stack local neighborhood points and labels
                    local_points = torch.cat(local_points, dim=0)  # Shape: (27, 3)
                    local_labels = torch.cat(local_labels, dim=0)  # Shape: (27,)

                    bp_list.append(local_points.unsqueeze(0))  # Add batch dimension
                    bl_list.append(local_labels.unsqueeze(0))

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

        return batch_points, batch_labels

    def iteration_forward(self, sam_model, image_embedding, feature_list, prev_masks, points=None, boxes=None):
        prev_masks = F.interpolate(prev_masks, scale_factor=0.25)

        new_point_embedding, new_image_embedding = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=prev_masks,
            image_embeddings=image_embedding.to(self.args.device)
        )

        mask, dice_pred = sam_model.mask_decoder(
            prompt_embeddings=new_point_embedding,  # (B, 2, 256)
            image_embeddings=new_image_embedding,  # (B, 256, 64, 64)
            feature_list=feature_list
        )
        mask = F.interpolate(mask, scale_factor=2)

        return mask, dice_pred

    def create_binary_mask(self, input_tensor, boxes):
        B, _, D, H, W = input_tensor.size()
        mask = torch.zeros((B, 1, D, H, W), dtype=torch.uint8)
        # Ensure boxes shape is (B, 7)
        boxes = boxes.squeeze(1)
        for b in range(B):
            z_min, z_max = int(boxes[b, 0].item()), int(boxes[b, 3].item())
            y_min, y_max = int(boxes[b, 1].item()), int(boxes[b, 4].item())
            x_min, x_max = int(boxes[b, 2].item()), int(boxes[b, 5].item())

            mask[b, 0, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = 1
        return mask

    def crop_roi(self, image, gt, box):
        # box = torch.tensor(b['z_min'], b['z_max'], b['z_mid_y_min'], b['z_mid_y_max'], b['z_mid_x_min'], b['z_mid_x_max'])
        box = box.long()
        z_slices = slice(box[0], box[1]+1)
        y_slices = slice(box[2], box[3]+1)
        x_slices = slice(box[4], box[5]+1)

        image_crop = image[:, z_slices, y_slices, x_slices]
        gt_crop = gt[:, z_slices, y_slices, x_slices]
        return image_crop, gt_crop

    def put_back_roi(self, image, image_crop, gt_crop, box_class):
        blank_image, blank_gt = torch.zeros_like(image), torch.zeros_like(image)

        blank_image[:, box_class['z_min']: box_class['z_max']+1, box_class['z_mid_y_min']: box_class['z_mid_y_max']+1,
        box_class['z_mid_x_min']: box_class['z_mid_x_max']+1] = image_crop

        blank_gt[:, box_class['z_min']: box_class['z_max']+1, box_class['z_mid_y_min']: box_class['z_mid_y_max']+1,
        box_class['z_mid_x_min']: box_class['z_mid_x_max']+1] = gt_crop
        return blank_image, blank_gt


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








