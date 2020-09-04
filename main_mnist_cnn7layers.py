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
from tensorflow import keras as K

from data_kits import dataloader_mnist as dataloader
from networks import cnn7layers as cnn
from misc import config_gpu

base_dir = "./model_dir/mnist_cnn7layers"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "test"])
    parser.add_argument("gpu", type=str)
    parser.add_argument("-t", "--tag", type=str, default="default")
    parser.add_argument("-b", "--bs", type=int, default=256)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    args = parser.parse_args()
    return args


def train(cfg):
    model = cnn.create(cfg.bs)
    model.compile(optimizer=K.optimizers.SGD(momentum=0.9, nesterov=True),
                  loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    train_loader = dataloader.train_set(cfg.bs)
    test_loader = dataloader.test_set(cfg.bs)

    history = model.fit(train_loader,
                        epochs=cfg.epochs,
                        validation_data=test_loader,
                        callbacks=[
                            K.callbacks.ModelCheckpoint(base_dir + f"/{cfg.tag}/ckpt/model",
                                                        save_weights_only=True,
                                                        verbose=1),
                            K.callbacks.ModelCheckpoint(base_dir + f"/{cfg.tag}/ckpt/best",
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        verbose=1),
                            K.callbacks.TensorBoard(base_dir + f"/{cfg.tag}/logs",
                                                    write_graph=False),
                            K.callbacks.LearningRateScheduler(
                                K.optimizers.schedules.ExponentialDecay(
                                    0.1, decay_steps=20, decay_rate=0.2, staircase=True
                                )
                            ),
                            K.callbacks.TerminateOnNaN()
                        ])


def test(cfg):
    model = cnn.create(cfg.bs)
    model.compile(loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.load_weights(base_dir + f"/{cfg.tag}/ckpt/best").expect_partial()

    test_loader = dataloader.test_set(cfg.bs)
    results = model.evaluate(test_loader)
    print("test loss: %.5f, test acc: %.5f" % tuple(results))


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config_gpu()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
