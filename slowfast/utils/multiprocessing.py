#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multiprocessing helpers."""

import os
import torch


def run(
    local_rank,
    num_proc,
    func,
    init_method,
    shard_id,
    num_shards,
    backend,
    cfg,
    output_queue=None,
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        output_queue (queue): can optionally be used to return values from the
            master process.
    """
    # Initialize the process group.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        if init_method.startswith('tcp://'):
            # original slowfast dist
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
        elif init_method.startswith('env://'):
            os.environ['MASTER_PORT'] = str(cfg.PORT)
            os.environ['MASTER_ADDR'] = cfg.ADDR
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(rank)
            # init_method is assumed to be “env://”.
            torch.distributed.init_process_group(backend=backend)
        else:
            raise ValueError('init method {} is not supported'.format(init_method))
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    ret = func(cfg)
    if output_queue is not None and local_rank == 0:
        output_queue.put(ret)
