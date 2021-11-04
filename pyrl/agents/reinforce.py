# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, print_function, division, unicode_literals
)

from builtins import *

# Scipy
import numpy as np

# Tensorflow
import tensorflow as tf
import tensorflow.keras as keras


################################################################################

class NormalReinforcePolicy(keras.Model):
    """"""

    def __init__(self, input_dim, output_dim, hidden_size=256, num_hidden=0):
        super().__init__(name="DefaultContinuousReinforcePolicy")
        self.dense_in = keras.Dense(hidden_size, input_dim=input_dim)
        self.dense_hidden = [keras.Dense(hidden_size, input_dim=hidden_size)
                             for _ in num_hidden]
        self.dense_out_mu = keras.Dense(output_dim, input_dim=hidden_size)
        self.dense_out_sigma = keras.Dense(output_dim, input_dim=hidden_size)

    def call(self, inputs):
        x = self.dense_in(inputs)
        for hidden_layer in self.dense_hidden:
            x = hidden_layer(x)
        mu = self.dense_out_mu(x)
        sigma = self.dense_out_sigma(x)

        return mu, sigma


class ReinforceContinuousAgent(object):
    """

    Entropy & Baseline
    https://fosterelli.co/entropy-loss-for-reinforcement-learning
    https://medium.com/@fork.tree.ai/understanding-baseline-techniques-for-reinforce-53a1e2279b57
    """

    def __init__(self, policy_net, optimizer):
        self.policy_net = policy_net
        self.optimizer = optimizer

        self.probs = []
        self.rewards = []
        self.entropies = []

    def compute_action(self, state):
        mu, sigma = self.policy_net(state)
        sigma = tf.math.softplus(sigma)

        eps = tf.random.normal(shape=mu.shape)
        action = (mu + sigma * eps)
        prob = normal_pdf(action, mu, sigma)
        entropy = -normal_entropy(sigma)

        self.probs.append(prob)
        self.entropies.append(entropy)

        return action

    def update(self, state, action, reward, next_state, done):
        if done:
            self.probs = []
            self.rewards = []
            self.entropies = []
        else:
            self.rewards.append(reward)

    def _update_policy(self):
        raise NotImplementedError()

################################################################################

_TF_PI = tf.constant([np.pi])


def normal_pdf(x, mu, sigma):
    """Computes the normal distributioin probability density function."""
    pi = tf.broadcast(_TF_PI, shape=sigma.shape)

    op1 = 1 / (sigma * tf.sqrt(2 * pi))
    op2_exp = -0.5 * tf.pow((x - mu) / sigma, 2)

    return op1 * tf.exp(op2_exp)


def normal_entropy(sigma):
    """"""
    pi = tf.broadcast(_TF_PI, shape=sigma.shape)
    return 0.5 * (1 + tf.log(2 * tf.pow(sigma, 2) * pi))
