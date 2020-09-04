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
import tensorflow.python.keras.callbacks as callbacks_module


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument_group("Callback Arguments")
    parser.add_argument("-c", "--ckpt", type=str, default="final",
                        help="File name of the real-time checkpoint. Set empty string to disable it.")
    parser.add_argument("-bc", "--best_ckpt", type=str, default="best",
                        help="File name of the best checkpoint. Set empty string to disable it.")
    parser.add_argument("-tb", "--tensorboard", type=str, default="logs",
                        help="Directory name of the tensorboard logs. Set empty string to disable it.")
    parser.add_argument("-v", "--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="0 = silent, 1 = progress bar, 2 = one line per epoch."
                             "Note that the progress bar is not particularly useful when"
                             "logged to a file, so verbose=2 is recommended when not running"
                             "interactively (eg, in a production environment).")


def get(base_dir, cfg, model, train_steps, **params):
    callbacks = [
        K.callbacks.TerminateOnNaN(),
    ]
    if cfg.ckpt:
        callbacks.append(K.callbacks.ModelCheckpoint(
            base_dir + f"/{cfg.tag}/ckpt/{cfg.ckpt}", save_weights_only=True, verbose=1))
    if cfg.best_ckpt:
        callbacks.append(K.callbacks.ModelCheckpoint(
            base_dir + f"/{cfg.tag}/ckpt/{cfg.best_ckpt}", save_best_only=True, save_weights_only=True, verbose=1))
    if cfg.tensorboard:
        callbacks.append(K.callbacks.TensorBoard(base_dir + f"/{cfg.tag}/{cfg.tensorboard}", write_graph=False))
    if cfg.lrp:
        from . import optimizer
        callbacks.append(optimizer.lr_callback(cfg))

    final_params = {
        "verbose": cfg.verbose,
        "epochs": cfg.total_epochs,
        "steps": train_steps
    }
    return callbacks_module.CallbackList(callbacks,
                                         add_history=True,
                                         add_progbar=cfg.verbose != 0,
                                         model=model,
                                         **params)


class CallBacks(object):
    def __init__(self, *args):
        self.call_backs = list(args)

    def on_batch_begin(self, batch, logs=None):
        for cb in self.call_backs:
            cb.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for cb in self.call_backs:
            cb.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.call_backs:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for cb in self.call_backs:
            cb.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        for cb in self.call_backs:
            cb.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        for cb in self.call_backs:
            cb.on_train_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        for cb in self.call_backs:
            cb.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for cb in self.call_backs:
            cb.on_train_end(logs)

    def on_test_begin(self, logs=None):
        for cb in self.call_backs:
            cb.on_test_begin(logs)

    def on_test_end(self, logs=None):
        for cb in self.call_backs:
            cb.on_test_end(logs)

    def set_model(self, model):
        for cb in self.call_backs:
            cb.set_model(model)

    def set_params(self, params):
        for cb in self.call_backs:
            cb.set_model(params)
