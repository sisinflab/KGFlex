"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(0)

import random


class KGFlex_TFModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 num_features=100,
                 factors=10,
                 learning_rate=0.01,
                 name="KGFlex_TF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self._factors = factors

        self.initializer = tf.initializers.GlorotUniform()

        self.K = tf.Variable(self.initializer(shape=[self.num_users, self.num_features]), name='H', dtype=tf.float32,
                             trainable=False)
        self.H = tf.Variable(self.initializer(shape=[self.num_users, self._factors]), name='H', dtype=tf.float32)
        self.G = tf.Variable(self.initializer(shape=[self.num_features, self._factors]), name='G', dtype=tf.float32)

        # self.C = tf.sparse.from_dense(np.random.choice([True, False], size=(self.num_users, self.num_items, self.num_features)))
        self.C = tf.sparse.SparseTensor(indices=[[0, 0, 5], [1, 2, 10]], values=[1, 1], dense_shape=[self.num_users, self.num_items, self.num_features])

        self.loss = keras.losses.MeanSquaredError()
        self.optimizer = tf.optimizers.Adam(learning_rate)

    #@tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        h_u = tf.squeeze(tf.nn.embedding_lookup(self.H, user))
        z_u = h_u @ tf.transpose(self.G)  # num_features x 1
        k_u = tf.squeeze(tf.nn.embedding_lookup(self.K, user))  # num_features x 1
        a_u = k_u * z_u
        x_ui = tf.reduce_sum(list(map(lambda i, j: tf.squeeze(tf.gather(tf.gather(a_u, i), self.uifm[i.numpy()][j.numpy()])), user, item)))

        return x_ui

        # keys, values = tuple(zip(*self.new_map.items()))
        # init = tf.lookup.KeyValueTensorInitializer(keys, values)
        # self.paddingItems = tf.lookup.StaticHashTable(
        #     init,
        #     default_value=self.ent_total-1)

    #@tf.function
    def train_step(self, batch):
        user, pos, label = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            output = self(inputs=(user, pos), training=True)
            loss = self.loss(label, output)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    #@tf.function
    def predict(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        output = self.call(inputs=inputs, training=training)
        return output

    @tf.function
    def get_recs(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        user, item = inputs
        h_u = tf.squeeze(tf.nn.embedding_lookup(self.H, user))
        z_u = tf.reduce_sum(h_u * self.G, axis=1)  # num_features x 1
        k_u = tf.squeeze(tf.nn.embedding_lookup(self.K, user))  # num_features x 1
        c_ui = tf.nn.embedding_lookup(self.C, (user, item))  # 1 x num_features
        x_ui = tf.reduce_sum(c_ui * z_u * k_u)

        return x_ui

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
