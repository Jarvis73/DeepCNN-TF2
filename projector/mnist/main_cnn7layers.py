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

import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from pathlib import Path

from networks import cnn7layers
from data_kits import dataloader_mnist as dataloader

BASE_DIR = Path(__file__).parent
SRC = BASE_DIR / "src_ckpt"
DST = BASE_DIR / "dst_ckpt"
IMG_SIZE = [28, 28]


def load_data():
    images, labels = [], []
    for image, label in dataloader.test_set(10000)[0]:
        images.append(image)
        labels.append(label)
    return images, labels


def get_final_features(images):
    # Load model
    model = cnn7layers.create(256)
    model.load_weights(str(SRC / "best"))

    features = []
    for image in images:
        features.append(model(image))
    return features


def create_sprite_image(images):
    rows, cols = IMG_SIZE

    # Number of rows
    sprite_dim = int(np.sqrt(images.shape[0]))
    # Empty image
    sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim), np.uint8)

    index = 0
    for i in range(sprite_dim):
        for j in range(sprite_dim):
            sprite_image[i * cols: (i + 1) * cols, j * rows: (j + 1) * rows] = \
                (1 - images[index].numpy().reshape(*IMG_SIZE))
            index += 1

    sprite_image[sprite_image > 0] = 255
    cv2.imwrite(str(DST / "sprite.png"), sprite_image.astype(np.uint8))


def save_projector(params):
    config = projector.ProjectorConfig()
    embedding_input = config.embeddings.add()
    embedding_input.tensor_name = "input/.ATTRIBUTES/VARIABLE_VALUE"
    embedding_input.metadata_path = str(DST / "metadata.tsv")
    if params.sprite:
        embedding_input.sprite.image_path = str(DST / "sprite.png")
        embedding_input.sprite.single_image_dim.extend(IMG_SIZE)

    embedding_feat = config.embeddings.add()
    embedding_feat.tensor_name = "feat/.ATTRIBUTES/VARIABLE_VALUE"
    embedding_feat.metadata_path = str(DST / "metadata.tsv")
    if params.sprite:
        embedding_feat.sprite.image_path = str(DST / "sprite.png")
        embedding_feat.sprite.single_image_dim.extend(IMG_SIZE)

    projector.visualize_embeddings(str(DST), config)


def main(args):
    images, labels = load_data()
    labels = tf.concat(labels, axis=0)
    image_t = tf.Variable(tf.reshape(tf.concat(images, axis=0), (labels.shape[0], -1)))
    features = get_final_features(images)
    feat_t = tf.Variable(tf.concat(features, axis=0))

    # Save to ckpt
    DST.mkdir(parents=True, exist_ok=True)
    ckpt = tf.train.Checkpoint(input=image_t, feat=feat_t)
    ckpt.save(str(DST / "embedding.ckpt"))
    # Save metadata
    metadata_file = DST / "metadata.tsv"
    with metadata_file.open("w") as f:
        f.write("\n".join([str(x.numpy().item()) for x in labels]))

    if args.sprite:
        create_sprite_image(image_t)
    save_projector(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sprite", action="store_true")

    args = parser.parse_args()
    main(args)
