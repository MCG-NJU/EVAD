#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job, launch_job_slurm, launch_job_tencent
from slowfast.utils.parser import parse_args
from slowfast.config.defaults import get_cfg
import slowfast.utils.checkpoint as cu

from test_net import test
from train_net import train
from models import add_evad_config


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    if cfg.LAUNCHER == 'pytorch':
        launcher = launch_job
    elif cfg.LAUNCHER == 'slurm':
        launcher = launch_job_slurm
    elif cfg.LAUNCHER == 'tencent':
        launcher = launch_job_tencent
    else:
        raise NotImplementedError("Do Not Support {}".format(cfg.LAUNCHER))

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launcher(cfg=cfg, init_method=args.init_method, func=train, local_rank=args.local_rank)

    # Perform testing.
    if cfg.TEST.ENABLE:
        launcher(cfg=cfg, init_method=args.init_method, func=test, local_rank=args.local_rank)

    # Run demo.
    if cfg.DEMO.ENABLE:
        # demo(cfg)
        raise NotImplementedError


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    add_evad_config(cfg)
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


if __name__ == "__main__":
    main()
