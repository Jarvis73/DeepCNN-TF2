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
import collections
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from ingredients.losses import linear_discriminative_eigvals


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument_group("Metric Arguments")
    parser.add_argument("-m", "--metrics", type=str, default="acc", choices=["", "acc", "lda"],
                        help="Metircs to be computed in training and evaluation.")


class LDA(K.metrics.Metric):
    def __init__(self, model, train_loader, test_laoder, test_step,
                 lambda_val=1e-3, n_components=None, verbose=False, logger=None):
        super(LDA, self).__init__(name="acc,val_acc")
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_laoder
        self.test_step = test_step

        self.lambda_val = lambda_val
        self.n_components = n_components
        self.verbose = verbose
        self.logger = logger.info if logger is not None else print

    def update_state(self, y_true, y_pred, sample_weight=None):
        # LDA only compute metrics at the end of the epoch.
        pass

    def reset_states(self):
        pass

    def result(self):
        # Evaluate on validation set
        epoch_valid_losses = []
        epoch_valid_outputs = []
        epoch_val_ys = []
        for bid, batch in enumerate(self.test_loader):
            output, loss = self.test_step(*batch)
            epoch_valid_losses.append(loss.numpy().item())
            epoch_valid_outputs.append(output)
            epoch_val_ys.append(batch[1].numpy())
        val_loss = np.mean(epoch_valid_losses)
        val_output = tf.concat(epoch_valid_outputs, axis=0)
        val_y = np.concatenate(epoch_val_ys, axis=0)

        # Evaluate on training set
        epoch_train_outputs = []
        epoch_train_ys = []
        for bid, batch in enumerate(self.train_loader):
            output = self.model(batch[0], training=False)
            epoch_train_outputs.append(output)
            epoch_train_ys.append(batch[1].numpy())
        train_output = tf.concat(epoch_train_outputs, axis=0)
        train_y = np.concatenate(epoch_train_ys, axis=0)

        # Fit data in LDA
        self.fit(train_output, train_y)
        acc = self.test(train_output, train_y)
        val_acc = self.test(val_output, val_y)
        return {"acc": acc, "val_acc": val_acc}

    def fit(self, X, y):
        X = tf.convert_to_tensor(X, tf.float32)
        y = tf.convert_to_tensor(y, tf.int32)
        classes = tf.sort(tf.unique(y).y)

        if self.n_components is None:
            self.n_components = classes.shape[0] - 1

        means = []
        for i in classes:
            Xg = X[y == i]
            means.append(tf.reduce_mean(Xg, axis=0))
        self.means = tf.stack(means, axis=0)                                        # [cls, d]

        eigvals, eigvecs = linear_discriminative_eigvals(y, X, self.lambda_val, ret_vecs=True)
        eigvecs = tf.reverse(eigvecs, axis=[1])                                     # [d, cls]
        eigvecs = eigvecs / tf.linalg.norm(eigvecs, axis=0, keepdims=True)          # [d, cls]
        self.scaling = eigvecs.numpy()
        self.coef = tf.matmul(
            tf.matmul(self.means, eigvecs), tf.transpose(eigvecs, (1, 0)))           # [cls, d]
        self.intercept = -0.5 * tf.linalg.diag_part(
            tf.matmul(self.means, tf.transpose(self.coef, (1, 0))))                  # [cls]
        self.coef = self.coef.numpy()
        self.intercept = self.intercept.numpy()

        eigvals = eigvals.numpy()
        if self.verbose:
            top_k_evals = eigvals[-self.n_components + 1:]
            self.logger("\nLDA-Eigenvalues:", np.array_str(top_k_evals, precision=2, suppress_small=True))
            self.logger("Eigenvalues Ratio min/max: %.3f, Mean: %.3f" % (
                top_k_evals.min() / top_k_evals.max(), top_k_evals.mean()))

        return eigvals

    def prob(self, X):
        prob = np.dot(X, self.coef.T) + self.intercept                           # [N, cls]
        # prob_sigmoid = 1. / (np.exp(prob) + 1)                                      # [N, cls]
        # sigmoid = prob_sigmoid / np.sum(prob_sigmoid, axis=1, keepdims=True)                # [N, cls]
        # return sigmoid
        return prob

    def pred(self, X):
        return np.argmax(self.prob(X), axis=1)                                      # [N]

    def test(self, X, y):
        pred = self.pred(X)
        return np.sum(pred == y) / len(pred)

    def map(self, X):
        X_new = np.dot(X, self.scaling)                                             # [N, cls]
        return X_new[:, :self.n_components]                                         # [N, cls - 1]


class Metric(object):
    def __init__(self, cfg, **kwargs):
        self.metrics = []
        if cfg.metrics == "lda":
            self.metrics.append(LDA(**kwargs))
        elif cfg.metrics == "acc":
            self.metrics.append(K.metrics.SparseCategoricalAccuracy(name="accuracy"))
        else:
            raise ValueError(f"Unsupported metrics: {cfg.metrics}. [lda/acc]")

    def update_state(self, y_true, y_pred, sample_weight=None):
        for m in self.metrics:
            m.update_state(y_true, y_pred, sample_weight)

    def reset_states(self):
        for m in self.metrics:
            m.reset_states()

    def result(self, *args, **kwargs):
        all_res = {}
        for m in self.metrics:
            res = m.result()
            if isinstance(res, dict):
                all_res.update(res)
            else:
                all_res[m.name] = m.result()

        return all_res
