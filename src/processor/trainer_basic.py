from abc import abstractmethod
import torch
import numpy as np
from torch.optim import AdamW, lr_scheduler
from src.config.config_setup import build_model, get_dataloader
from monai.losses import DiceCELoss, DiceLoss
import torch.nn as nn
from src.utils.util import save_checkpoint
import time
import os
import torch.distributed as dist
from torch.cuda import amp
import torchio as tio


class Trainer_basic(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        a = time.time()
        self.train_data, self.val_data = get_dataloader(args, split='train'), get_dataloader(args, split='val')
        self.sam = build_model(args)
        if self.args.ddp:
            self.sam = self.sam.module

        self.best_dice, self.best_epoch, self.start_epoch = 0, 0, 0
        self.setup()
        print('dataloaders are created, models are loaded, and others are set, spent {} for rank {}'
              .format(round(time.time() - a, 2), self.args.rank))


    def run(self):
        self.scaler = amp.GradScaler()
        for epoch_num in range(self.start_epoch, self.args.max_epoch):
            self.sam.train()
            if self.args.ddp:
                # dist.barrier() # set a barrier until all processes are at same point
                self.train_data.sampler.set_epoch(epoch_num)

            self.train(epoch_num)
            if self.args.ddp and self.args.rank == 0:
                print('doing validation on rank=0')
                current_mean_dice = self.validate(epoch_num)
            else:
                current_mean_dice = self.validate(epoch_num)
            # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
            # if self.args.ddp:
                # dist.barrier()
            self.save_model(current_mean_dice, epoch_num)

    @abstractmethod
    def forward(self, model, image, label, box, iter_nums, train, return_each_iter, filename):
        pass

    def train(self, epoch_num):
        loss_summary = []
        for idx, data in enumerate(self.train_data):
            self.optimizer.zero_grad()
            # increase speed based on gradient accumulation
            # my_context = self.sam.no_sync if self.args.rank != -1 and idx % self.args.accumulation_steps != 0 else nullcontext
            # with my_context():
            image, label, box = data['image'].to(self.args.device).float(), data['gt'].to(self.args.device).float(), data['box'].to(self.args.device).float()
            filename = data['path'][0].split('/')[-1]
            print(filename)
            with amp.autocast():
                loss = self.forward(self.sam, image, label, box, iter_nums=self.args.iter_nums, train=True, filename=filename)

            # self.scaler.scale(loss).backward()
            # self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.sam.parameters(), 1.0)
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            loss_summary.append(loss.detach().cpu().numpy())



            print('epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.train_data))
                  + ": loss:" + str(round(loss_summary[-1].flatten()[0], 4))
                  + ": rank:" + str(self.args.rank))
            self.logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.train_data))
                + ": loss:" + str(round(loss_summary[-1].flatten()[0], 4))
                + ": rank:" + str(self.args.rank))
        print('current lr: {}'.format(self.optimizer.param_groups[0]["lr"]))
        # If the first iteration creates NaN gradients (e.g. due to a high scaling factor and thus gradient overflow),
        # the optimizer.step() will be skipped and you might get this warning.
        self.update_lr(epoch_num, warm_up=self.args.warm_up)
        self.logger.info("- Train metrics: " + str(np.mean(loss_summary)))

    def validate(self, epoch_num):
        self.sam.eval()
        with torch.no_grad():
            dice_list, dice_list_initial = [], []
            for idx, data in enumerate(self.val_data):
                mean_dice = 0

                image, label, box = data['image'].to(self.args.device).float(), data['gt'].to(self.args.device).float(), data['box'].to(self.args.device).float()

                mean_dice_sub, initial_dice = self.forward(self.sam, image, label, box, iter_nums=self.args.iter_nums)
                dice_list_initial.append(initial_dice)
                mean_dice += mean_dice_sub
                dice_list.append(mean_dice)

                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    '   subject: ' + str(data['path'][0].split('/')[-1]) +
                    '   last:' + str(round(mean_dice, 4)) +
                    '   initial:' + str(round(initial_dice, 4)) +
                    '   last-initial:' + str(round(mean_dice, 4) - round(initial_dice, 4)))

            self.logger.info("- Val metrics (last-init) dice: " + str(np.mean(dice_list) - np.mean(dice_list_initial)))
            self.logger.info("- Val metrics mean dice: " + str(np.mean(dice_list)))
        return dice_list

    def get_dice_score(self, prev_masks, label, batch=False):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum() + 0.00001
            volume_intersect = (mask_gt & mask_pred).sum() + 0.00001
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (label > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        if batch:
            return dice_list
        else:
            return (sum(dice_list) / len(dice_list)).item()

    def save_model(self, current_dice, epoch_num):
        is_best = False
        if np.mean(current_dice) > self.best_dice:
            self.best_dice = np.mean(current_dice)
            self.best_epoch = epoch_num
            is_best = True

        if not self.args.ddp or (self.args.ddp and self.args.rank == 0):
            save_checkpoint({"epoch": epoch_num + 1,
                             "best_val_loss": self.best_dice,
                             "model_state_dict": self.sam.state_dict(),
                             "optimizer": self.optimizer.state_dict(),
                             "lr_scheduler": self.lr_scheduler.state_dict(),
                             },
                            is_best=is_best,
                            checkpoint=self.args.save_dir)
        self.logger.info("- Val metrics best mean dice: {} at epoch {} " .format(self.best_dice, self.best_epoch))

    def setup(self):
        self.setup_loss()
        self.setup_optimizier()
        self.setup_scheduler()

        if self.args.resume:
            if self.args.ddp:
                dist.barrier()
            checkpoint = 'best.pth.tar' if self.args.resume_best else 'last.pth.tar'
            ckpt = torch.load(os.path.join(self.args.save_dir, checkpoint))

            self.start_epoch = ckpt["epoch"]
            self.best_epoch = self.start_epoch
            self.best_dice = ckpt["best_val_loss"]
            self.sam.load_state_dict(ckpt["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.lr_scheduler_regular.load_state_dict(ckpt['lr_scheduler'])


            self.logger.info(f"Resume training from epoch {self.start_epoch}!")
            del ckpt
            torch.cuda.empty_cache()

    def setup_loss(self):
        self.loss_mse = nn.MSELoss()
        self.loss_segmentation = DiceCELoss(sigmoid=True)
        self.loss_Dice = DiceLoss(sigmoid=True)
        self.loss_validation = DiceLoss(reduction='none')
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def setup_optimizier(self):
        self.optimizer = AdamW([
            {'params': self.sam.image_encoder.parameters()},
            {'params': self.sam.prompt_encoder.parameters()},
            {'params': self.sam.mask_decoder.parameters()},
        ], lr=self.args.lr)

    def setup_scheduler(self):
        if self.args.lr_scheduler == 'linear':
            self.lr_scheduler_regular = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=500)
        else:
            self.lr_scheduler_regular = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        if self.args.warm_up:
            self.linear_warmup_scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)

    def update_lr(self, epoch, warmup_epoch=10, warm_up=False):
        if warm_up:
            if epoch < warmup_epoch:
                self.lr_scheduler = self.linear_warmup_scheduler
            else:
                self.lr_scheduler = self.lr_scheduler_regular
        else:
            self.lr_scheduler = self.lr_scheduler_regular
        self.lr_scheduler.step()










