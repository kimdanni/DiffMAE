# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched

from pathlib import Path
import os
import random

import matplotlib.pyplot as plt
from torchvision.transforms import Lambda
import matplotlib.pyplot as plt


            
def visualize_masked_patches(masked_patch, epoch, output_dir, is_pred=True):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
        
    p = 16 # NOTE : hard-coded for now
    masked_patch = masked_patch[0]
    N, num_patches, D = masked_patch.shape
    rows = cols = int(np.ceil(np.sqrt(num_patches)))
    
    sample_list = list(range(N))
    random.shuffle(sample_list)
    sample_list = sample_list[:4] if is_pred else [0] # visualize only the first sample for the original image
    unnormalize = Lambda(lambda t: (t + 1) / 2)
    masked_patch = unnormalize(masked_patch)
    masked_patch = masked_patch.clamp(0, 1)
    
    
    for i in sample_list:
        plt.figure(figsize=(20, 20))
        for j in range(rows * cols):
            plt.subplot(rows, cols, j + 1)
            if j < num_patches:
                # Plot the patch
                patch = masked_patch[i, j, :].detach().view(p, p, 3)
                patch = patch.cpu().numpy().astype(np.float32)
                plt.imshow(patch)
            else:
                # Fill the extra subplot with white color
                plt.imshow(np.ones((p, p, 3)))
            plt.axis('off')
        plt.tight_layout()
        name = "pred" if is_pred else "org"
        if not is_pred:
            plt.savefig(output_dir / f"Masked_Patches_epoch_{str(epoch)}")
        else:
            plt.savefig(output_dir / f"Predicted_Masked_Patches_for_sample_T_{i}_epoch_{str(epoch)}.png")

def visualize_result(model, samples, target, pred, output_dir, epoch, data_iter_step):
    visualize_masked_patches(pred, epoch, output_dir, is_pred=True)
    visualize_masked_patches(target, epoch, output_dir, is_pred=False)

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    visualize = args.visualize
    visualize_path = Path(args.output_dir) / "visualize"

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # TODO: visualize should not be performed in the pretrain engine
    if visualize:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(visualize_path):
            os.makedirs(visualize_path)
            
    data_samples = list(range(len(data_loader)))
    random_sample = random.choice(data_samples)

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        loss, target, pred = model(samples, mask_ratio=args.mask_ratio, kept_mask_ratio=args.kept_mask_ratio)
        
        loss_value = loss.item()
        if visualize and epoch % 50 == 0 and data_iter_step == random_sample:
            visualize_result(model, samples, target, pred, visualize_path, epoch, data_iter_step)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}