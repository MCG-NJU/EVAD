#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import time
import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter
from slowfast.utils.person_ap import PersonMeter


logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    if cfg.EVAL_ONLY:
        return
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, batch in enumerate(train_loader):
        inputs, labels, _, meta = zip(*batch)
        if writer is not None:
            writer.update_global_step(data_size * cur_epoch + cur_iter)
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr * param_group["lr_scale"]

        train_meter.data_toc()

        gt_instances = (meta, labels)

        if cfg.DETECTION.ENABLE:
            loss_dict = model(inputs, gt_instances)
        else:
            raise NotImplementedError

        # Compute the loss.
        loss = sum(loss_dict.values()) / len(loss_dict)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
                for k, v in loss_dict.items():
                    loss_dict[k] = du.all_reduce([v])[0]
            loss = loss.item()
            for k, v in loss_dict.items():
                loss_dict[k] = v.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr, loss_dict)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr,
                     "Train/loss_ce": loss_dict['loss_ce'],
                     "Train/loss_bce": loss_dict['loss_bce'],
                     "Train/loss_bbox": loss_dict['loss_bbox'],
                     "Train/loss_giou": loss_dict['loss_giou'],
                     "Max Mem": torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                     },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            raise NotImplementedError

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, person_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        person_meter (PersonMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    logger.info("Val samples = {}".format(len(val_loader)))
    start_time = time.time()

    for cur_iter, batch in enumerate(val_loader):
        inputs, labels, _, meta = zip(*batch)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            action_logits, ori_boxes, metadata, persons = model(inputs, meta)

            person_boxes = persons["boxes"]
            person_scores = persons["scores"]
            keyframe_filename = [m['keyframe_filename'] for m in meta]

            preds = action_logits

            check_msg = "{} vs {} vs {}".format(len(metadata), len(ori_boxes), len(preds))
            assert len(metadata) == len(ori_boxes) == len(preds), check_msg
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()
                person_boxes = person_boxes.cpu()
                person_scores = person_scores.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)
                person_boxes = torch.cat(du.all_gather_unaligned(person_boxes), dim=0)
                person_scores = torch.cat(du.all_gather_unaligned(person_scores), dim=0)

                keyframe_filename = du.all_gather_unaligned(keyframe_filename)
                keyframe_filename = [f for one_batch in keyframe_filename for f in one_batch]

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)
            person_meter.update_stats(person_boxes, person_scores, keyframe_filename)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

        if cfg.DEBUG:
            break

    logger.info("Inference is done in %f seconds." % (time.time() - start_time))
    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    person_meter.finalize_metrics()
    # write to tensorboard format if available.
    if writer is not None:
        if du.is_master_proc():
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map['AP50']}, global_step=cur_epoch
            )

    val_meter.reset()
    person_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    if cfg.NUM_GPUS > 1:
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer = optim.construct_optimizer(model_without_ddp, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        if cfg.TRAIN.DATASET == 'ava':
            train_meter = AVAMeter(len(train_loader), cfg, mode="train")
            val_meter = AVAMeter(len(val_loader), cfg, mode="val")
            person_meter = PersonMeter(len(val_loader), cfg, mode="val")
        else:
            raise ValueError('Do Not Support {}'.format(cfg.TRAIN.DATASET))
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc():
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of params: {} M'.format(n_parameters / 1e6))
    logger.info("Start epoch: {}".format(start_epoch + 1))

    # Evaluate the model on validation set and return.
    if cfg.EVAL_ONLY:
        eval_epoch(val_loader, model, val_meter, person_meter, start_epoch, cfg, writer)

        if writer is not None:
            writer.close()
        return

    # Perform the training loop.
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch)
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch)

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch and not cfg.EVAL_ONLY:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch or cur_epoch >= cfg.TRAIN.TEST_AFTER_EPOCH:
            eval_epoch(val_loader, model, val_meter, person_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
