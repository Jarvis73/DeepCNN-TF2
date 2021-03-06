# Copyright 2019 Jianwei Zhang All Right Reserved.
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


def train_set(bs, shuffle_buffer=50000):
    res = tfds.load(name='mnist', split='train', data_dir=DATA_DIR, with_info=True)
    mnist: tf.data.Dataset = res[0]
    info: tfds.core.DatasetInfo = res[1]

    def map_fn(sample):
        image, label = sample["image"], sample["label"]
        image = tf.cast(image, tf.float32) / 255.
        # return a tuple for model.fit api
        return image, label

    return (mnist
            .shuffle(shuffle_buffer)
            .map(map_fn, num_parallel_calls=min(bs, 4))
            .batch(bs, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE), info.splits["train"].num_examples // bs)


def test_set(bs):
    res = tfds.load(name='mnist', split="test", data_dir=DATA_DIR, with_info=True)
    mnist: tf.data.Dataset = res[0]
    info: tfds.core.DatasetInfo = res[1]

    def map_fn(sample):
        image, label = sample["image"], sample["label"]
        image = tf.cast(image, tf.float32) / 255.
        # return a tuple for model.fit api
        return image, label

    return (mnist
            .map(map_fn, num_parallel_calls=min(bs, 4))
            .batch(bs, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE), info.splits["test"].num_examples // bs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tf.random.set_seed(1234)
    dataset = train_set(10, shuffle_buffer=10)
    x, y = next(iter(dataset))
    print(x.shape, y.shape)

    fig, ax = plt.subplots(2, 5)
    ax = ax.flat
    for i in range(10):
        ax[i].imshow(x[i, ..., 0], cmap="gray")
    plt.show()
