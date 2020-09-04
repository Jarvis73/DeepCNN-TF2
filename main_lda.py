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

import os
# Disable Tensorflow device information logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow as tf

from misc import config_gpu
from ingredients import callbacks, optimizer
from utils.logger import C

base_dir = "./model_dir/lda"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Global Arguments")
    parser.add_argument("mode", type=str, choices=["train", "test"])
    parser.add_argument("gpu", type=str)
    parser.add_argument("-t", "--tag", type=str, default="default")
    parser.add_argument("-b", "--bs", type=int, default=1000)
    parser.add_argument("-d", "--data", type=str, default="mnist", choices=["mnist", "cifar10", "stl10", "stl10_1k"])
    parser.add_argument("-tc", "--test_ckpt", type=str, default="best", choices=["best", "final"])

    optimizer.add_arguments(parser)
    callbacks.add_arguments(parser)

    args = parser.parse_args()
    return args


class Solver(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.get_ingredients()
        dataloader, cnn, input_shape = self.get_dataloader_and_cnn()
        self.model = cnn.create(cfg.bs, input_shape)
        self.c = C.c     # Used for colorful results

        self.train_loader, self.train_length = dataloader.train_set(cfg)
        self.test_loader, self.test_length = dataloader.test_set(cfg)

        if cfg.mode == "train":
            self.optimizer = optimizer.get(cfg)
            self.model.optimizer = self.optimizer
            self.cb = callbacks.get(base_dir, cfg, self.model, self.train_length)
            self.train_step = tf.function(self._train_step)
        self.test_step = tf.function(self._test_step)

    def get_ingredients(self):
        from ingredients import losses, metrics
        self.loss_obj = losses.LinearDiscriminativeLoss()
        self.lda = metrics.LDA(self.model, self.train_loader, self.test_loader, self.test_step,
                               verbose=True)

    def get_dataloader_and_cnn(self):
        if self.cfg.data == "mnist":
            from data_kits import dataloader_mnist as dataloader
            from networks import cnn7layers as cnn
            input_shape = (28, 28, 1)
        elif self.cfg.data == "cifar10":
            from data_kits import dataloader_cifar10 as dataloader
            from networks import cnn10layers as cnn
            input_shape = (32, 32, 3)
        elif self.cfg.data in ["stl10", "stl10_1k"]:
            from data_kits import dataloader_stl10 as dataloader
            from networks import cnn10layers as cnn
            input_shape = (96, 96, 3)
        return dataloader, cnn, input_shape

    def _train_step(self, X, y):
        with tf.GradientTape() as tape:
            output = self.model(X, training=True)
            loss = self.loss_obj(y, output)
            loss += tf.add_n(self.model.losses)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def _test_step(self, X, y):
        output = self.model(X, training=False)
        loss = self.loss_obj(y, output)
        return output, loss

    def train(self):
        c = self.c
        self.cb.on_train_begin()
        start_epoch = 0
        for epoch in range(start_epoch, self.cfg.total_epochs):
            print(c(f"Epoch: {epoch + 1}/{self.cfg.total_epochs}", C.UNDERLINE))
            self.cb.on_epoch_begin(epoch)
            # Training
            epoch_train_losses = []
            for i, batch in enumerate(self.train_loader):
                self.cb.on_batch_begin(i, {"size": int(batch[0].shape[0])})
                loss = self.train_step(*batch).numpy().item()
                self.cb.on_batch_end(i, {"loss": loss})
                epoch_train_losses.append(loss)
            train_loss = np.mean(epoch_train_losses)

            # Evaluate on validation set
            epoch_valid_losses = []
            epoch_valid_outputs = []
            epoch_val_ys = []
            for bid, batch in enumerate(self.test_loader):
                output, loss = self.test_step(*batch)
                epoch_valid_losses.append(loss.numpy().item())
                epoch_valid_outputs.append(output)
                epoch_val_ys.append(batch[1].numpy())
            val_loss = np.mean(epoch_valid_losses)
            val_output = tf.concat(epoch_valid_outputs, axis=0)
            val_y = np.concatenate(epoch_val_ys, axis=0)

            # Evaluate on training set
            epoch_train_outputs = []
            epoch_train_ys = []
            for bid, batch in enumerate(self.train_loader):
                output = self.model(batch[0], training=False)
                epoch_train_outputs.append(output)
                epoch_train_ys.append(batch[1].numpy())
            train_output = tf.concat(epoch_train_outputs, axis=0)
            train_y = np.concatenate(epoch_train_ys, axis=0)

            # Fit data in LDA
            self.lda.fit(train_output, train_y)
            acc = self.lda.test(train_output, train_y)
            val_acc = self.lda.test(val_output, val_y)

            logs = {"loss": train_loss, "accuracy": acc, "val_loss": val_loss, "val_accuracy": val_acc}
            print(" - ".join([f"{key}: {round(val, 5)}" for key, val in logs.items()]))
            self.cb.on_epoch_end(epoch, logs)
            if self.model.stop_training:
                break

        self.cb.on_train_end()
        return self.model.history

    def test(self):
        c = self.c
        ckpt_path = base_dir + f"/{self.cfg.tag}/ckpt/{self.cfg.test_ckpt}"
        print(f"Loading checkpoint from {ckpt_path} ... ", end="", flush=True)
        self.model.load_weights(ckpt_path).expect_partial()
        print(c("Loaded!", C.OKGREEN))

        print("Inference on training set ... ", end="", flush=True)
        outputs_tr, ys_tr = [], []
        for X, y in self.train_loader:
            outputs_tr.append(self.model(X, training=False))
            ys_tr.append(y.numpy())
        outputs_tr = tf.concat(outputs_tr, axis=0)
        ys_tr = np.concatenate(ys_tr, axis=0)
        print(c(f"{self.train_length * self.cfg.bs} samples finished!", C.OKGREEN))

        print("Inference on test set ... ", end="", flush=True)
        outputs_te, ys_te = [], []
        for X, y in self.test_loader:
            outputs_te.append(self.model(X, training=False))
            ys_te.append(y.numpy())
        outputs_te = tf.concat(outputs_te, axis=0)
        ys_te = np.concatenate(ys_te, axis=0)
        print(c(f"{self.test_length * self.cfg.bs} samples finished!", C.OKGREEN))

        # Computing LDA from network outputs
        print("Fitting LDA ... ", end="", flush=True)
        self.lda.fit(outputs_tr, ys_tr)
        print(c("Finished!", C.OKGREEN))

        acc_tr = self.lda.test(outputs_tr, ys_tr)
        acc_te = self.lda.test(outputs_te, ys_te)
        print(c("\ntrain acc: %.5f, test acc: %.5f" % (acc_tr, acc_te), C.OKBLUE))


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config_gpu()
    s = Solver(args)

    if args.mode == "train":
        s.train()
    elif args.mode == "test":
        s.test()
