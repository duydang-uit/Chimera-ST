#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:/kaggle/working/sample"
os.environ['PATH'] = os.getenv("PATH") + os.pathsep + "/kaggle/working/sample"
sys.path.append("/kaggle/working/sample")

from typing import Dict, Optional, Any, List, Tuple, Callable
import json
import random

import numpy as np
import torch
from hydra.core.config_store import ConfigStore

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from omegaconf import DictConfig, OmegaConf
from hydra.experimental import initialize
from fairseq.dataclass.initialize import register_hydra_cfg
from fairseq.trainer import Trainer
from fairseq.tasks.joint_task import JointTrainingTask


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: DictConfig) -> None:
    # print("Code train 1")
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    metrics.reset()
    # print("Code train 2")
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    random.seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)
    # print("Code train 3")
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    assert cfg.dataset.max_tokens is not None or \
        cfg.dataset.batch_size is not None or \
        isinstance(task, JointTrainingTask), \
        'Must specify batch size either with --max-tokens or --batch-size ' \
        ' if not training on joint-task'
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # print("Code train 4")
    for valid_sub_split in cfg.dataset.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    # print("Code train 5")
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {} ({})".format(cfg.task._name, task.__class__.__name__))
    logger.info("model: {} ({})".format(cfg.model._name, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(cfg.criterion._name, criterion.__class__.__name__)
    )
    logger.info("num. model params: {} (num. trained: {})".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # (optionally) Configure quantization
    # print("Code train 6")
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    # print("Code train 7")
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
        # print("Code train Con 1")
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
        # print("Code train Con 2")
    # print("Code train 8")
    logger.info('training on {} devices (GPUs/TPUs)'.format(cfg.distributed_training.distributed_world_size))
    logger.info('max tokens per GPU = {} and batch size per GPU = {}'.format(
        cfg.dataset.max_tokens,
        cfg.dataset.batch_size,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    # print("Code train 9")
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    try:
        # print("Code train Con 3")
        os.remove(os.path.join(
            cfg.checkpoint.save_dir, 'checkpoint_last.pt'))
    except Exception:
        pass
    # print("Code train 10")
    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    
    # print("Code train 11")
    while (
        lr > cfg.optimization.min_lr
        and epoch_itr.next_epoch_idx <= max_epoch
    ):
        # train for one epoch
            
        # print("Code train 12")
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
            
        # print("Code train 13")
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
            
        # print("Code train 14")
        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
            
        # print("Code train 15")
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(cfg.checkpoint.patience))
            return True
        else:
            return False


@metrics.aggregate("train")
def train(cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    
    # print("Code train 16")
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    
    # print("Code train 17")
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    
    # print("Code train 18")
    if getattr(cfg.common, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir if distributed_utils.is_master(cfg.distributed_training) else None
        ),
        default_log_format=('tqdm' if not cfg.common.no_progress_bar else 'simple'),
    )
    
    # print("Code train 19")
    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(',')
    should_stop = False
    num_updates = trainer.get_num_updates()
    if not os.path.exists(os.path.join(
            cfg.checkpoint.save_dir, 'checkpoint_last.pt')):
        logger.info("save checkpoint at the beginning")
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, None)
    
    # print("Code train 20")
    if cfg.dataset.validate_only:
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, True
        )
        return valid_losses, True
    
    # print("Code train 21")
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")
                if hasattr(task, 'log_private_metrics'):
                    task.log_private_metrics(
                        progress.log, 'train_inner',
                        num_updates, get_training_stats)

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )
        if should_stop:
            break

    # log end-of-epoch stats

    # print("Code train 23")
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    # print("Code train 24")
    metrics.reset_meters("train")

    if hasattr(task, 'log_private_metrics'):
        task.log_private_metrics(
            progress.print, 'train', num_updates, get_training_stats)
    
    return valid_losses, should_stop


def validate_and_save(cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, valid_subsets: List[str], end_of_epoch: bool) -> Tuple[List[Optional[float]], bool]:
    
    # print("Code train 25")
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or num_updates >= max_update
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or num_updates >= max_update
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation

    # Validate
    
    # print("Code train 26")
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    
    # print("Code train 27")
    should_stop = (
        should_stop_early(cfg, valid_losses[0])
        or num_updates >= max_update
        or (
            cfg.optimization.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > cfg.optimization.stop_time_hours
        )
    )

    # Save checkpoint
    
    # print("Code train 28")
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(cfg.checkpoint, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    
    # print("Code train 29")
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, subsets: List[str]) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""
    
    # print("Code train 30")
    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    # print("Code train 31")
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir if distributed_utils.is_master(cfg.distributed_training) else None
            ),
            default_log_format=('tqdm' if not cfg.common.no_progress_bar else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        # print("Code train 32")
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        # print("Code train 33")
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        # optionally log private stats
        # print("Code train 34")
        if hasattr(task, 'log_private_metrics'):
            def get_valid_stats2(stats):
                return get_valid_stats(cfg, trainer, stats)
            task.log_private_metrics(
                progress.print, 'valid', trainer.get_num_updates(),
                get_valid_stats2)
        # optionally dump visualization info
        # print("Code train 35")
        if hasattr(task, 'dump_features'):
            task.dump_features()

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]) -> Dict[str, Any]:
    # print("Code train 36")
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[cfg.checkpoint.best_checkpoint_metric]
        )
    return stats


def cli_main(modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> None:
    # print("Code train 37")
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)
    if args.save_task_config is not None:
        config_file = args.save_task_config[0]
        to_save = OmegaConf.to_container(cfg)
        for _key, _value in to_save.items():
            if isinstance(_value, argparse.Namespace):
                to_save[_key] = vars(_value)
        logger.info(f"dumping training config to {config_file}")
        with open(config_file, 'w') as f:
            json.dump(to_save, f, indent=4)
        exit()

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == '__main__':
    cs = ConfigStore.instance()
    register_hydra_cfg(cs)
    initialize(config_path="../config", strict=True)
    cli_main()
