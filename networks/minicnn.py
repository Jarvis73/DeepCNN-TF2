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

from tensorflow import keras as K

L = K.layers


class MiniCNN(K.Model):
    def __init__(self, name=None):
        super(MiniCNN, self).__init__(name=name)
        self.conv = L.Conv2D(32, 3, activation='relu')
        self.flatten = L.Flatten()
        self.fc = L.Dense(128, activation='relu')
        self.out = L.Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.out(x)


mini_cnn = MiniCNN
