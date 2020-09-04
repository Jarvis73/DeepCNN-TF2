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

import tqdm
import argparse
import tensorflow as tf
from tensorflow import keras as K

from data_kits import dataloader_mnist
from networks import minicnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "test"])
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, config):
        # Hyper parameters
        self.mode = config.mode
        self.total_epochs = 5
        self.batch_size = 32

        # Limit GPU memory usage first.
        self.config_gpu()

        self.model = minicnn.mini_cnn(name="MiniCNN")
        self.criterion = K.losses.SparseCategoricalCrossentropy()

        if self.mode == "train":
            self.tr_wrapper = dataloader_mnist.dataset("./", train_flag=True, batch_size=self.batch_size)
            self.optimizer = K.optimizers.Adam(learning_rate=0.001)
            # Define accumulators
            self.train_loss_accu = K.metrics.Mean(name="train_loss")
            self.train_acc_accu = K.metrics.SparseCategoricalAccuracy(name="train_acc")
        else:
            self.tr_wrapper = None
            self.optimizer = None
            self.train_loss_accu = None
            self.train_acc_accu = None

            # Construct test/validation dataset
        self.val_wrapper = dataloader_mnist.dataset("./", train_flag=False, batch_size=self.batch_size)
        self.val_loss_accu = K.metrics.Mean(name="val_loss")
        self.val_acc_accu = K.metrics.SparseCategoricalAccuracy(name="val_acc")

    @staticmethod
    def config_gpu():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def reset_accu(self):
        if self.mode == "train":
            self.train_loss_accu.reset_states()
            self.train_acc_accu.reset_states()
        self.val_loss_accu.reset_states()
        self.val_acc_accu.reset_states()

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            pred = self.model(images)
            loss = self.criterion(labels, pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss_accu(loss)
        self.train_acc_accu(labels, pred)

    @tf.function
    def test_step(self, images, labels):
        pred = self.model(images)
        loss = self.criterion(labels, pred)

        self.val_loss_accu(loss)
        self.val_acc_accu(labels, pred)

    def train_epoch(self, epoch):
        print("Run training set (Epoch {}/{}) ...".format(epoch, self.total_epochs))
        for images, labels in tqdm.tqdm(self.tr_wrapper["train"]["data"],
                                        total=self.tr_wrapper["train"]["size"] // self.batch_size):
            self.train_step(images, labels)
        print("Run validation set ...")
        for images, labels in tqdm.tqdm(self.val_wrapper["test"]["data"],
                                        total=self.val_wrapper["test"]["size"] // self.batch_size):
            self.test_step(images, labels)

        str_ = 'Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}, Test Loss: {:.3f}, Test Accuracy: {:.3f}'
        print(str_.format(epoch,
                          self.total_epochs,
                          self.train_loss_accu.result(),
                          self.train_acc_accu.result(),
                          self.val_loss_accu.result(),
                          self.val_acc_accu.result()))

    def train(self):
        for epoch in range(self.total_epochs):
            self.train_epoch(epoch + 1)
            self.reset_accu()
        self.model.save_weights("./model_dir/minicnn/model.ckpt", save_format="tf")

    def test(self):
        self.model.load_weights("./model_dir/minicnn/model.ckpt")
        print("Load model from ./model_dir/minicnn/model.ckpt")

        for images, labels in tqdm.tqdm(self.val_wrapper["test"]["data"],
                                        total=self.val_wrapper["test"]["size"] // self.batch_size):
            self.test_step(images, labels)

        str_ = 'Test Loss: {:.3f}, Test Accuracy: {:.3f}'
        print(str_.format(self.val_loss_accu.result(), self.val_acc_accu.result()))

    def exec(self):
        if self.mode == "train":
            self.train()
        else:
            self.test()


if __name__ == "__main__":
    args = parse_args()
    h = Trainer(args)
    h.exec()
