# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import collections

# SciPy
import numpy as np

# OpenAI
import gym

# Torch
import torch
import torch.nn as nn


###############################################################################

class CriticMLP(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_layers=3, hidden_size=256,
                 activation="relu", layer_norm=False):
        assert hidden_layers > 0
        assert hidden_size > 0
        super(CriticMLP, self).__init__()

        if len(observation_space.shape) > 1:
            raise ValueError('MLP observation space must have a single'
                             ' dimension')
        if len(action_space.shape) > 1:
            raise ValueError('MLP action space must have a single'
                             ' dimension')

        input_size = observation_space.shape[0] + action_space.shape[0]
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
                       nn.Linear(hidden_size, 1)))

        self.network = nn.Sequential(collections.OrderedDict(layers))

    def forward(self, states, actions):
        return self.network(torch.cat((states, actions), dim=1))


class ActorMLP(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_layers=3, hidden_size=256,
                 activation="relu",
                 layer_norm=False):
        assert hidden_layers >= 0
        assert hidden_size > 0
        super(ActorMLP, self).__init__()

        if len(observation_space.shape) > 1:
            raise ValueError('MLP observation space must have a single'
                             ' dimension')
        if len(action_space.shape) > 1:
            raise ValueError('MLP action space must have a single'
                             ' dimension')

        input_size = observation_space.shape[0]
        output_size = action_space.shape[0]

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

class HerCriticMLP(CriticMLP):
    """Utility wrapper to use the `Critic` class in Hindsigh
    Expirience Replay (HER) agents.

    The __init__ signature is the same as that of the `Critic`
    class, since all the arguments given to this class __init__
    are forwarded to the `Critic.__init__`.
    """

    def __init__(self, observation_space, action_space,
                 hidden_layers=3, hidden_size=512,
                 activation="relu", layer_norm=False,
                 *args, **kwargs):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("Hindsight Experience Replay critic expects a"
                            " gym.spaces.Dict observation space")

        obs_space = observation_space['observation']
        goal_space = observation_space['desired_goal']
        flat_obs_space = gym.spaces.Box(
            low=np.concatenate(obs_space.low, goal_space.low),
            hig=np.concatenate(obs_space.high, goal_space.high),
            dtype=obs_space.dtype)

        super(HerCriticMLP, self).__init__(flat_obs_space, action_space,
                                           hidden_layers, hidden_size,
                                           activation, layer_norm,
                                           *args, **kwargs)

    def forward(self, obs, goals, actions):
        return super(HerCriticMLP, self).forward(
            states=torch.cat((obs, goals), dim=1), actions=actions)


class HerActorMLP(ActorMLP):
    """Utility wrapper to use the `Actor` class in Hindsigh
    Expirience Replay (HER) agents.

    The __init__ signature is the same as that of the `Actor`
    class, since all the arguments given to this class __init__
    are forwarded to the `Actor.__init__`.
    """

    def __init__(self, observation_space, action_space,
                 hidden_layers=3, hidden_size=512,
                 activation="relu",
                 layer_norm=False,
                 *args, **kwargs):

        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("Hindsight Experience Replay actor expects a"
                            " gym.spaces.Dict observation space")

        obs_space = observation_space['observation']
        goal_space = observation_space['desired_goal']
        flat_obs_space = gym.spaces.Box(
            low=np.concatenate(obs_space.low, goal_space.low),
            hig=np.concatenate(obs_space.high, goal_space.high),
            dtype=obs_space.dtype)

        super(HerActorMLP, self).__init__(flat_obs_space, action_space,
                                          hidden_layers, hidden_size,
                                          activation, layer_norm,
                                          *args, **kwargs)

    def forward(self, obs, goals):
        return super(HerActorMLP, self).forward(
            states=torch.cat((obs, goals), dim=1))


###############################################################################

_ACTIVATIONS = {
    "leakyrelu": torch.nn.LeakyReLU,
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh
}


def get_activation_layer(name):
    """Get an activation layer given its name.

    :param name: Name of the activation layer, valid values are: leakyrelu,
        relu, sigmoid and tanh.
    """
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        msg = "invalid layer '{}', valid options are: {}"
        raise ValueError(
            msg.format(name, ", ".join(sorted(_ACTIVATIONS.keys()))))