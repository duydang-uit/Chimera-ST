# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from dataclasses import dataclass, field
from typing import List

import omegaconf
from fairseq.dataclass import FairseqDataclass
from omegaconf import II, DictConfig

from . import FairseqLRScheduler, register_lr_scheduler


@dataclass
class InverseSquareRootScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=4000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={
            "help": "initial learning rate during warmup phase; default is args.lr"
        },
    )
    # TODO common vars at parent class
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("inverse_sqrt", dataclass=InverseSquareRootScheduleConfig)
class InverseSquareRootSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, cfg: DictConfig, optimizer):
        super().__init__(cfg, optimizer)
        print(cfg.lr, type(cfg.lr), "Code LR")
        if isinstance(cfg.lr, omegaconf.listconfig.ListConfig) and len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with inverse_sqrt."
                " Consider --lr-scheduler=fixed instead."
            )
        warmup_end_lr = (
            cfg.lr[0]
            if isinstance(cfg.lr, omegaconf.listconfig.ListConfig)
            else cfg.lr
        )
        if cfg.warmup_init_lr < 0:
            cfg.warmup_init_lr = (
                0 if cfg.warmup_updates > 0 else warmup_end_lr
            )

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (
            warmup_end_lr - cfg.warmup_init_lr
        ) / cfg.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * cfg.warmup_updates ** 0.5

        # initial learning rate
        self.lr = cfg.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        print("code LR 2")
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        print("code LR 3")
        if num_updates < self.cfg.warmup_updates:
            self.lr = self.cfg.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates ** -0.5
        self.optimizer.set_lr(self.lr)
        return self.lr
