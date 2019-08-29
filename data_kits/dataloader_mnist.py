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

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.keras.utils.data_utils import get_file

L = K.layers
mnist = K.datasets.mnist


def load_data(dataset_base_path, train=True):
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = get_file('mnist.npz',
                    origin=origin_folder + 'mnist.npz',
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45',
                    cache_subdir="data",
                    cache_dir=dataset_base_path)

    if train:
        with np.load(path) as f:
            x_data, y_data = f['x_train'], f['y_train']
        x_data = x_data / 255.
        # Add a channels dimension
        x_data = x_data[..., None]
    else:
        with np.load(path) as f:
            x_data, y_data = f['x_test'], f['y_test']
        x_data = x_data / 255.
        # Add a channels dimension
        x_data = x_data[..., None]

    return x_data, y_data


def dataset(dataset_base_path,
            train_flag=True,
            batch_size=1,
            prefetch_buffer_size=None):
    x_data, y_data = load_data(dataset_base_path, train_flag)
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.int32)

    def data_gen(x, y, shuffle=False):
        idx = np.arange(x.shape[0])
        if shuffle:
            np.random.shuffle(idx)
        for i in idx:
            yield x[i], y[i]

    ds = tf.data.Dataset.from_generator(lambda: data_gen(x_data, y_data, shuffle=train_flag),
                                        (tf.float32, tf.int32),
                                        (tf.TensorShape([28, 28, 1]), tf.TensorShape([])))
    ds = ds.batch(batch_size).prefetch(prefetch_buffer_size)

    tag = "train" if train_flag else "test"
    return {tag: {"data": ds, "size": x_data.shape[0]}}
