#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys
import simplejson
from iopath.common.file_io import g_pathmgr

import slowfast.utils.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = g_pathmgr.open(filename, "a", buffering=1024)
    atexit.register(io.close)
    return io


_FORMAT = "[%(filename)s: %(lineno)3d]: %(message)s"


def setup_logging(output_dir):
    """Sets up the logging."""
    # Enable logging only for the master process
    if du.is_master_proc():
        # Clear the root logger to prevent any existing logging config
        # (e.g. set by another module) from messing with our setup
        logging.root.handlers = []
        # Construct logging configuration
        logging_config = {"level": logging.INFO, "format": _FORMAT}
        # Log either to stdout or to a file
        if False and cfg.LOG_DEST == "stdout":
            logging_config["stream"] = sys.stdout
        else:
            logging_config["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(output_dir, "stdout.log"))
            ]
        # Configure logging
        logging.basicConfig(**logging_config)
    else:
        _suppress_print()


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.5f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
