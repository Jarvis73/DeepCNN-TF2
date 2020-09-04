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

import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

DATA_DIR = str(Path(__file__).parent.parent / "data")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def train_set(bs, shuffle_buffer=1024):
    res = tfds.load(name='cifar10', split='train', data_dir=DATA_DIR, with_info=True)
    cifar: tf.data.Dataset = res[0]
    info: tfds.core.DatasetInfo = res[1]

    def map_fn(sample):
        image, label = sample["image"], sample["label"]
        image = tf.pad(image, ((4, 4), (4, 4), (0, 0)), mode='REFLECT')
        image = tf.image.random_crop(image, (32, 32, 3))
        image = tf.image.random_flip_left_right(image)
        # image = (tf.cast(image, tf.float32) - MEAN) / STD
        image = tf.cast(image, tf.float32) / 255.
        # return a tuple for model.fit api
        return image, label

    return (cifar
            .shuffle(shuffle_buffer)
            .map(map_fn, num_parallel_calls=min(bs, 4))
            .batch(bs, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE), info.splits["train"].num_examples // bs)


def test_set(bs):
    res = tfds.load(name='cifar10', split="test", data_dir=DATA_DIR, with_info=True)
    cifar: tf.data.Dataset = res[0]
    info: tfds.core.DatasetInfo = res[1]

    def map_fn(sample):
        image, label = sample["image"], sample["label"]
        # image = (tf.cast(image, tf.float32) - MEAN) / STD
        image = tf.cast(image, tf.float32) / 255.
        # return a tuple for model.fit api
        return image, label

    return (cifar
            .map(map_fn, num_parallel_calls=min(bs, 4))
            .batch(bs, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE), info.splits["test"].num_examples // bs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tf.random.set_seed(1234)
    dataset = train_set(1000, shuffle_buffer=10)
    x, y = next(iter(dataset))
    print(x.shape, y.shape)

    fig, ax = plt.subplots(2, 5)
    ax = ax.flat
    for i in range(10):
        ax[i].imshow(x[i])
    plt.show()
