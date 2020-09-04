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

import itertools
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

DATA_DIR = str(Path(__file__).parent.parent / "data")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def train_set(cfg):
    train_X_file = Path(__file__).parent.parent / "data/stl10/train_X.bin"
    train_y_file = Path(__file__).parent.parent / "data/stl10/train_y.bin"
    with train_X_file.open("rb") as f:
        images = np.frombuffer(f.read(), np.uint8)
        images = np.reshape(images, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
    with train_y_file.open("rb") as f:
        # Switch to zero-based indexing.
        labels = np.frombuffer(f.read(), np.uint8) - 1

    cls_idx = [np.where(labels == i)[0] for i in range(10)]
    if cfg.data == "stl10":
        max_id = 400
    elif cfg.data == "stl10_1k":
        max_id = 100
    else:
        raise ValueError(f"Unsupported data: {cfg.data}. [stl10/stl10_1k]")
    subset_idx = sorted(itertools.chain(*[x[:max_id] for x in cls_idx]))

    def map_fn(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32) / 255.
        return image, label

    return (tf.data.Dataset.from_tensor_slices((images[subset_idx], labels[subset_idx]))
            .shuffle(1000)
            .map(map_fn, num_parallel_calls=min(cfg.bs, 4))
            .batch(cfg.bs, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE), max_id * 10 // cfg.bs)


def test_set(cfg):
    train_X_file = Path(__file__).parent.parent / "data/stl10/train_X.bin"
    train_y_file = Path(__file__).parent.parent / "data/stl10/train_y.bin"
    with train_X_file.open("rb") as f:
        images = np.frombuffer(f.read(), np.uint8)
        images = np.reshape(images, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
    with train_y_file.open("rb") as f:
        # Switch to zero-based indexing.
        labels = np.frombuffer(f.read(), np.uint8) - 1

    cls_idx = [np.where(labels == i)[0] for i in range(10)]
    subset_idx = sorted(itertools.chain(*[x[400:] for x in cls_idx]))

    def map_fn(image, label):
        image = tf.cast(image, tf.float32) / 255.
        return image, label

    return (tf.data.Dataset.from_tensor_slices((images[subset_idx], labels[subset_idx]))
            .map(map_fn, num_parallel_calls=min(cfg.bs, 4))
            .batch(cfg.bs, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE), 1000 // cfg.bs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    tf.random.set_seed(1234)
    dataset, number = train_set(1000, shuffle_buffer=10)
    x, y = next(iter(dataset))
    print(x.shape, y.shape)
    print(y[:10])
    print(np.bincount(y))

    fig, ax = plt.subplots(2, 5)
    ax = ax.flat
    for i in range(10):
        ax[i].imshow(x[i])
    plt.show()
