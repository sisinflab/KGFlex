"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import tensorflow as tf
import numpy as np
from tensorflow import keras
from scipy.sparse import csr_matrix

_RANDOM_SEED = 42

class KGFlex_Model(keras.Model):
    def __init__(self, learning_rate,
                 n_users,
                 users,
                 n_items,
                 n_features,
                 features_mapping,
                 embedding_size,
                 positive_items,
                 negative_items,
                 index_mask,
                 users_features,
                 name="VBPRMF",
                 random_seed=_RANDOM_SEED,
                 **kwargs):

        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)

        self._learning_rate = learning_rate
        self._n_users = n_users
        self._n_items = n_items

        self._n_features = n_features
        self._embedding_size = embedding_size

        self._positive_items = positive_items
        self.negative_items = negative_items

        self.feature_vecs = np.random.randn(self._n_features, self._embedding_size) / 10
        self.feature_bias = np.random.randn(self._n_features) / 10

        self.users_vecs = {user : np.zeros((self._n_features, self._embedding_size)) for user in users}
        for key, value in self.users_vecs.items():
            self.users_vecs[key][index_mask[key]] = np.random.randn(sum(index_mask[key]), self._embedding_size) / 10
            self.users_vecs[key] = csr_matrix(self.users_vecs[key])

        users_weights = dict()
        for u in users:
            users_weights[u] = csr_matrix(
                [users_features[u][feature] if feature in users_features[u] else 0 for feature in features_mapping])

        pass

    @tf.function
    def call(self, inputs, training=None):
        user, item = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = tf.squeeze(tf.nn.embedding_lookup(self.F, item))

        xui = beta_i + tf.reduce_sum((gamma_u * gamma_i), axis=1) + \
              tf.reduce_sum((theta_u * tf.matmul(feature_i, self.E)), axis=1) + \
              tf.squeeze(tf.matmul(feature_i, self.Bp))

        return xui, gamma_u, gamma_i, feature_i, theta_u, beta_i

    @tf.function
    def train_step(self, batch):
        user, pos, neg = batch
        with tf.GradientTape() as t:
            xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                self(inputs=(user, pos), training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(theta_u)]) \
                       + self.l_b * tf.nn.l2_loss(beta_pos) \
                       + self.l_b * tf.nn.l2_loss(beta_neg) / 10 \
                       + self.l_e * tf.reduce_sum([tf.nn.l2_loss(self.E), tf.nn.l2_loss(self.Bp)])

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

        return loss

    @tf.function
    def predict(self, start, stop):
        return self.Bi + tf.matmul(self.Gu[start:stop], self.Gi, transpose_b=True) \
               + tf.matmul(self.Tu[start:stop], tf.matmul(self.F, self.E), transpose_b=True) \
               + tf.squeeze(tf.matmul(self.F, self.Bp))

    def get_config(self):
        raise NotImplementedError

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
