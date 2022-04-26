"""Learning rate scheduler."""

import numpy as np
import torch
from dotmap import DotMap
from fairseq.optim.fairseq_optimizer import FairseqOptimizer
from fairseq.optim.lr_scheduler import cosine_lr_scheduler
from torch import nn
from torch.optim.lr_scheduler import (
    LambdaLR, CosineAnnealingLR)
from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)


def clip_gradient(model, clip: float):
    nn.utils.clip_grad_norm_(model.parameters(), clip)


class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    """
    From Over9000
    https://github.com/mgrankin/over9000/blob/master/train.py
    """
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps,
                 pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        self.curr_epoch = 0
        super(ConcatLR, self).__init__(optimizer, last_epoch)

    def step(self):
        if self.curr_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        self.curr_epoch += 1
        super().step()

    def get_lr(self):
        if self.curr_epoch <= self.step_start:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()


class TradeoffAnnealer:
    def __init__(self, c, num_steps=None):
        """
        Anneal the tradeoff between label and augmentation loss according
            to some schedule.

        :param c: config
        :param num_steps: int, provide when loading from checkpoint to fast-
            forward to that tradeoff value.
        """
        self.c = c
        self.name = self.c.exp_tradeoff_annealing

        self.num_steps = 0
        self.init_tradeoff = self.c.exp_tradeoff
        self.curr_tradeoff = self.c.exp_tradeoff
        self.max_steps = self.get_max_steps()
        self.step_map = {
            'constant': self.constant_step,
            'cosine': self.cosine_step,
            'linear_decline': self.linear_decline_step}

        if self.name not in self.step_map.keys():
            raise NotImplementedError

        self.step = self.step_map[self.name]

        if num_steps > 0:
            # If we are loading a model from checkpoint,
            # should update the annealer to that number of steps.
            for _ in range(num_steps):
                self.step()

            print(f'Fast-forwarded tradeoff annealer to step {num_steps}.')

        print(
            f'Initialized "{self.name}" augmentation/label tradeoff annealer. '
            f'Annealing to minimum value in {self.max_steps} steps.')

    def get_max_steps(self):
        # If annealing proportion is set to -1,
        if self.c.exp_tradeoff_annealing_proportion == -1:
            # and the optimizer proportion is set, we use the optimizer
            # proportion to determine how long it takes for the tradeoff to
            # anneal to 0.
            if self.c.exp_optimizer_warmup_proportion != -1:
                return int(np.ceil(self.c.exp_optimizer_warmup_proportion
                                   * self.c.exp_num_total_steps))
            # and the optimizer proportion is not set,
            # we take all steps to anneal.
            else:
                return self.c.exp_num_total_steps

        if (self.c.exp_tradeoff_annealing_proportion < 0
                or self.c.exp_tradeoff_annealing_proportion > 1):
            raise Exception('Invalid tradeoff annealing proportion.')

        # Otherwise, we use the tradeoff annealing proportion to determine
        # for how long we anneal.
        return int(np.ceil(self.c.exp_tradeoff_annealing_proportion
                           * self.c.exp_num_total_steps))

    def constant_step(self):
        self.num_steps += 1
        return self.curr_tradeoff

    def linear_decline_step(self):
        curr = self.num_steps
        max_val = self.init_tradeoff

        if self.num_steps <= self.max_steps:
            self.curr_tradeoff = max_val - (curr / self.max_steps) * max_val
        else:
            self.curr_tradeoff = 0

        self.num_steps += 1

        return self.curr_tradeoff

    def cosine_step(self):
        if self.num_steps <= self.max_steps:
            self.curr_tradeoff = self.init_tradeoff * (1 / 2) * (
                np.cos(np.pi * (self.num_steps / self.max_steps)) + 1)
        else:
            self.curr_tradeoff = 0

        self.num_steps += 1

        return self.curr_tradeoff


class LRScheduler:
    def __init__(self, c, name, optimizer):
        self.c = c
        self.name = name
        self.optimizer = optimizer
        self.num_steps = 0

        self.construct_auto_scheduler()

        print(f'Initialized "{name}" learning rate scheduler.')

    def construct_auto_scheduler(self):
        total_steps = self.c.exp_num_total_steps

        if self.c.exp_optimizer_warmup_proportion >= 0:
            num_warmup_steps = (
                    total_steps * self.c.exp_optimizer_warmup_proportion)
        else:
            num_warmup_steps = self.c.exp_optimizer_warmup_fixed_n_steps

        print(f'Warming up for {num_warmup_steps}/{total_steps} steps.')

        if self.name == 'constant':
            self.scheduler = get_constant_schedule(optimizer=self.optimizer)
        elif self.name == 'linear_warmup':
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps)
        elif self.name == 'cosine_cyclic':
            args = dict(
                warmup_updates=num_warmup_steps,
                warmup_init_lr=1e-7,
                max_lr=self.c.exp_lr,
                lr=[1e-7],
                t_mult=2.,
                lr_period_updates=num_warmup_steps * 2,
                lr_shrink=0.5)
            optim = FairseqOptimizer(None)
            optim._optimizer = optim.optimizer = self.optimizer
            self.scheduler = cosine_lr_scheduler.CosineSchedule(
                optimizer=optim, args=DotMap(args))
        elif self.name == 'polynomial_decay_warmup':
            # Based on the fairseq implementation, which is based on BERT
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                lr_end=1e-7,
                power=1.0)
        elif self.name == 'flat_and_anneal':
            def d(x):
                return 1

            assert self.c.exp_optimizer_warmup_proportion >= 0

            # We use exp_optimizer_warmup_proportion to denote the
            # flat LR regime, prior to annealing
            dummy = LambdaLR(self.optimizer, d)
            cosine = CosineAnnealingLR(
                self.optimizer, int(total_steps * (
                    1 - self.c.exp_optimizer_warmup_proportion)))
            self.scheduler = ConcatLR(
                self.optimizer, dummy, cosine, total_steps,
                self.c.exp_optimizer_warmup_proportion)
        else:
            raise NotImplementedError

    def step(self):
        self.num_steps += 1
        c_lr = self.c.exp_lr
        num = self.num_steps
        tot = self.c.exp_num_total_steps

        if self.name == 'cosine_cyclic':
            self.scheduler.step_update(num_updates=num)
        else:
            self.scheduler.step()
