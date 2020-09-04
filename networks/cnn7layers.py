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

from tensorflow import keras

L = keras.layers


def create(batch_size, input_shape=(28, 28, 1), drop_rate=0.25, num_classes=10, l2=1e-4):
    kwargs = {
        "use_bias": False,
        "kernel_initializer": keras.initializers.he_normal(),
        "kernel_regularizer": keras.regularizers.l2(l2)
    }

    model = keras.Sequential([
        L.Conv2D(64, 3, padding="same",
                 input_shape=input_shape, batch_size=batch_size, **kwargs), L.BatchNormalization(), L.ReLU(),
        L.Conv2D(64, 3, padding="same", **kwargs), L.BatchNormalization(), L.ReLU(),
        L.MaxPool2D(), L.Dropout(drop_rate),

        L.Conv2D(96, 3, padding="same", **kwargs), L.BatchNormalization(), L.ReLU(),
        L.Conv2D(96, 3, padding="same", **kwargs), L.BatchNormalization(), L.ReLU(),
        L.MaxPool2D(), L.Dropout(drop_rate),

        L.Conv2D(256, 3, padding="valid", **kwargs), L.BatchNormalization(), L.ReLU(),
        L.Dropout(drop_rate * 2),
        L.Conv2D(256, 1, padding="valid", **kwargs), L.BatchNormalization(), L.ReLU(),
        L.Dropout(drop_rate * 2),
        L.Conv2D(num_classes, 1, padding="valid", **kwargs), L.BatchNormalization(), L.ReLU(),
        L.GlobalAvgPool2D()
    ])

    return model
