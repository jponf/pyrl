# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import collections
from builtins import super

# SciPy
import numpy as np

# OpenAI
import gym

# Torch
import torch
import torch.nn as nn

from .utils import get_activation_layer


###############################################################################

class Critic(nn.Module):
    def __init__(self, state_size, action_size, output_size,
                 hidden_layers=3, hidden_size=512,
                 activation="relu",
                 layer_norm=False):
        assert hidden_layers > 0
        assert hidden_size > 0
        super().__init__()

        input_size = state_size + action_size
        layers = []
        for i in range(0, hidden_layers + 1):  # input + hidden
            layers.append(("linear{}".format(i),
                           nn.Linear(hidden_size if i > 0 else input_size,
                                     hidden_size)))
            if layer_norm:
                layers.append(("norm{}".format(i),
                               nn.LayerNorm(hidden_size)))
            layers.append(("{}{}".format(activation, i),
                           get_activation_layer(activation)))

        layers.append(("linear{}".format(i+1),
                       nn.Linear(hidden_size, output_size)))

        self.network = nn.Sequential(collections.OrderedDict(layers))

    def forward(self, states, actions):
        return self.network(torch.cat((states, actions), dim=1))


class Actor(nn.Module):

    def __init__(self, input_size, output_size,
                 hidden_layers=3, hidden_size=512,
                 activation="relu",
                 layer_norm=False):
        assert hidden_layers >= 0
        assert hidden_size > 0
        super().__init__()

        # action space for this actor
        self.action_space = gym.spaces.Box(
            low=np.repeat(-1.0, output_size).astype(np.float32, copy=False),
            high=np.repeat(1.0, output_size).astype(np.float32, copy=False),
            dtype=np.float32)

        layers = []
        for i in range(0, hidden_layers + 1):  # input + hidden
            layers.append(("linear{}".format(i),
                           nn.Linear(hidden_size if i > 0 else input_size,
                                     hidden_size)))
            if layer_norm:
                layers.append(("norm{}".format(i),
                               nn.LayerNorm(hidden_size)))
            layers.append(("{}{}".format(activation, i),
                           get_activation_layer(activation)))

        layers.append(("linear{}".format(i+1),
                       nn.Linear(hidden_size, output_size)))
        layers.append(("tanh{}".format(i+1), nn.Tanh()))

        self.network = nn.Sequential(collections.OrderedDict(layers))

    def forward(self, states):
        return self.network(states)

    def get_perturbable_parameters(self):
        return [x for x, _ in self.named_parameters()
                if 'norm' not in x]


###############################################################################

class HerCritic(Critic):
    """Utility wrapper to use the `Critic` class in Hindsigh
    Expirience Replay (HER) agents.

    The __init__ signature is the same as that of the `Critic`
    class, since all the arguments given to this class __init__
    are forwarded to the `Critic.__init__`.
    """

    def __init__(self, *args, **kwargs):
        super(HerCritic, self).__init__(*args, **kwargs)

    def forward(self, obs, goals, actions):
        return super(HerCritic, self).forward(
            states=torch.cat((obs, goals), dim=1), actions=actions)


class HerActor(Actor):
    """Utility wrapper to use the `Actor` class in Hindsigh
    Expirience Replay (HER) agents.

    The __init__ signature is the same as that of the `Actor`
    class, since all the arguments given to this class __init__
    are forwarded to the `Actor.__init__`.
    """

    def __init__(self, *args, **kwargs):
        super(HerActor, self).__init__(*args, **kwargs)

    def forward(self, obs, goals):
        return super(HerActor, self).forward(
            states=torch.cat((obs, goals), dim=1))


###############################################################################

class GaussianActorCritic(object):

    def __init__(self, actor_in_size, actor_out_size,
                 critic_in_size, critic_out_size,
                 actor_hidden_layers=3, actor_hidden_size=512,
                 actor_activation="relu", actor_layer_norm=False,
                 critic_hidden_layers=3, critic_hidden_size=512,
                 critic_activation="relu", critic_layer_norm=False):
        self.actor = Actor(input_size=actor_in_size,
                           output_size=actor_out_size,
                           hidden_layers=actor_hidden_layers,
                           hidden_size=actor_hidden_size,
                           activation=actor_activation,
                           layer_norm=actor_layer_norm)

        self.critic = Critic(input_size=critic_in_size,
                             output_size=critic_out_size,
                             hidden_layers=critic_hidden_layers,
                             hidden_size=critic_hidden_size,
                             activation=critic_activation,
                             layer_norm=critic_layer_norm)

    def forward(self, x):
        raise NotImplementedError()
