#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
from slowfast.utils.person_ap import PersonMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, person_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        person_meter (PersonMeter): person testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    logger.info("Test samples = {}".format(len(test_loader)))

    for cur_iter, batch in enumerate(test_loader):
        inputs, labels, video_idx, meta = zip(*batch)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            action_logits, ori_boxes, metadata, persons = model(inputs, meta)

            person_boxes = persons["boxes"]
            person_scores = persons["scores"]
            keyframe_filename = [m["keyframe_filename"] for m in meta]

            preds = action_logits

            person_boxes = person_boxes.detach().cpu() if cfg.NUM_GPUS else person_boxes.detach()
            person_scores = person_scores.detach().cpu() if cfg.NUM_GPUS else person_scores.detach()
            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            check_msg = "{} vs {} vs {}".format(len(metadata), len(ori_boxes), len(preds))
            assert len(metadata) == len(ori_boxes) == len(preds), check_msg

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)
                person_boxes = torch.cat(du.all_gather_unaligned(person_boxes), dim=0)
                person_scores = torch.cat(du.all_gather_unaligned(person_scores), dim=0)

                keyframe_filename = du.all_gather_unaligned(keyframe_filename)
                keyframe_filename = [f for one_batch in keyframe_filename for f in one_batch]

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            person_meter.update_stats(person_boxes, person_scores, keyframe_filename)
            test_meter.log_iter_stats(None, cur_iter)
            if cfg.DEBUG:
                break
        else:
            raise NotImplementedError

        test_meter.iter_tic()
    if cfg.MODEL.SparseRCNN.PERSON_THRESHOLD >= 0.:  # use negative number to skip the action meter
        test_meter.log_epoch_stats(0)
    person_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform testing on the pretrained video model.
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
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create meters.
    if cfg.DETECTION.ENABLE:
        if cfg.TEST.DATASET == 'ava':
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
            person_meter = PersonMeter(len(test_loader), cfg, mode="test")
        else:
            raise ValueError('Do Not Support {}'.format(cfg.TEST.DATASET))
        if cfg.TEST.TENSOR_PATH != "":
            test_meter.finalize_metrics(skip_save=True)
            return 0

    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc():
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform testing on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, person_meter, cfg, writer)
    if writer is not None:
        writer.close()
