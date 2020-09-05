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


def inspect(ckpt, key):
    """ Inspect checkpoint  """
    import tensorflow as tf
    if key:
        print(tf.train.load_variable(ckpt, key))
    else:
        for x in tf.train.list_variables(ckpt):
            print(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="Path to the checkpoint file")
    parser.add_argument("-k", "--key", type=str, default="", help="Specify the variable key for acquiring values")
    args = parser.parse_args()

    inspect(args.ckpt, args.key)
