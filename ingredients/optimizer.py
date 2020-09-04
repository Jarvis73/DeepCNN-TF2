# Copyright 2019-2020 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================

import argparse
import tensorflow.keras as K


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument_group("Solver Arguments")
    # Epoch arguments
    parser.add_argument("--epochs", type=int, default=0,
                        help="Number of epochs for training")
    parser.add_argument("-e", "--total_epochs", type=int, default=0,
                        help="Number of total epochs for training")

    # Learning rate arguments
    parser.add_argument("-lr", "--lr", type=float, default=0.1,
                        help="Base learning rate for model training")
    parser.add_argument("-lrp", "--lrp", type=str, default="period_step",
                        help="Learning rate policy [custom_step/period_step/poly/plateau]. "
                             "Set empty string to disable it.")
    parser.add_argument("--lr_boundaries", type=int, nargs="*",
                        help="[custom_step] Use the specified lr at the given boundaries")
    parser.add_argument("--lr_values", type=float, nargs="+",
                        help="[custom_step] Use the specified lr at the given boundaries")
    parser.add_argument("-lr-step", "--lr_decay_step", type=int, default=10,
                        help="[period_step] Decay the base learning rate at a fixed step")
    parser.add_argument("-lr-rate", "--lr_decay_rate", type=float, default=0.1,
                        help="[period_step, plateau] Learning rate decay rate")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="[poly] Polynomial power")
    parser.add_argument("--lr_end", type=float, default=1e-6,
                        help="[poly, plateau] The minimal end learning rate")
    parser.add_argument("--lr_patience", type=int, default=30,
                        help="[plateau] Learning rate patience for decay")
    parser.add_argument("--lr_min_delta", type=float, default=1e-4,
                        help="[plateau] Minimum delta to indicate improvement")
    parser.add_argument("--cool_down", type=int, default=0,
                        help="[plateau]")
    parser.add_argument("--monitor", type=str, default="val_loss",
                        help="[plateau] Quantity to be monitored [val_loss/loss]")

    # Optimizer arguments
    parser.add_argument("-o", "--opt", type=str, default="sgd",
                        help="Optimizer for training [sgd/adam]")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="[adam] Parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.99,
                        help="[adam] Parameter")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="[adam] Parameter")
    parser.add_argument("--sgd_momentum", type=float, default=0.9,
                        help="[momentum] Parameter")
    parser.add_argument("-nesterov", "--sgd_nesterov", action="store_true",
                        help="[momentum] Parameter")


def train_epochs(start_epoch, cfg):
    if cfg.epochs > 0:
        return cfg.epochs + start_epoch
    else:
        return cfg.total_epochs


def lr_callback(cfg):
    if cfg.lrp == "plateau":
        callback = K.callbacks.ReduceLROnPlateau(monitor=cfg.monitor,
                                                 factor=cfg.lr_decay_rate,
                                                 patience=cfg.lr_patience,
                                                 mode='min',
                                                 min_delta=cfg.lr_min_delta,
                                                 cooldown=cfg.cool_down,
                                                 min_lr=cfg.lr_end)
    elif cfg.lrp == 'period_step':
        scheduler = K.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg.lr,
            decay_steps=cfg.lr_decay_step,
            decay_rate=cfg.lr_decay_rate,
            staircase=True)
        callback = K.callbacks.LearningRateScheduler(scheduler)
    elif cfg.lrp == "custom_step":
        scheduler = K.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=cfg.lr_boundaries,
            values=cfg.lr_values)
        callback = K.callbacks.LearningRateScheduler(scheduler)
    elif cfg.lrp == 'poly':
        scheduler = K.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=cfg.lr,
            decay_steps=cfg.total_epochs,
            end_learning_rate=cfg.lr_end,
            power=cfg.lr_power)
        callback = K.callbacks.LearningRateScheduler(scheduler)
    elif not cfg.lrp:
        callback = None
    else:
        raise ValueError('Not supported learning policy.')

    return callback

def get(cfg):
    if cfg.opt == "adam":
        optimizer_params = {"beta_1": cfg.adam_beta1, "beta_2": cfg.adam_beta2, "epsilon": cfg.adam_epsilon}
        optimizer = K.optimizers.Adam(cfg.lr, **optimizer_params)
    elif cfg.opt == "sgd":
        optimizer_params = {"momentum": cfg.sgd_momentum, "nesterov": cfg.sgd_nesterov}
        optimizer = K.optimizers.SGD(cfg.lr, **optimizer_params)
    else:
        raise ValueError("Not supported optimizer: " + cfg.opt)

    return optimizer
