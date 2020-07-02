# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

# SciPy
import numpy as np

# OpenAI
import gym

# PyTorch
import torch
import torch.nn as nn

import pyrl.agents.models_utils as models_utils


###############################################################################

class CriticMLP(nn.Module):
    """Critic implemented as a Multi-layer Perceptron."""

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
        hidden_layers_size = [hidden_size] * hidden_layers
        self.network = models_utils.create_mlp(
            input_size=input_size,
            output_size=1,
            hidden_layers=hidden_layers_size,
            layer_norm=layer_norm,
            activation=activation,
            last_activation=None)

    def forward(self, states, actions):
        return self.network(torch.cat((states, actions), dim=1))


class TwinnedCritic(nn.Module):
    """Wrapper module to encapsulate two critic networks."""

    def __init__(self, critic1, critic2):
        super(TwinnedCritic, self).__init__()
        self.c1 = critic1
        self.c2 = critic2

    def forward(self, states, actions):
        return self.c1(states, actions), self.c2(states, actions)


class ActorMLP(nn.Module):
    """Deterministic actor implemented as a Multi-layer Perceptron."""

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
        hidden_layers_size = [hidden_size] * hidden_layers

        # action space for this actor
        self.action_space = gym.spaces.Box(
            low=np.repeat(-1.0, output_size).astype(np.float32, copy=False),
            high=np.repeat(1.0, output_size).astype(np.float32, copy=False),
            dtype=np.float32)

        self.network = models_utils.create_mlp(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers_size,
            layer_norm=layer_norm,
            activation=activation,
            last_activation="tanh")

    def forward(self, states):
        return self.network(states)

    def get_perturbable_parameters(self):
        return [x for x, _ in self.named_parameters()
                if 'norm' not in x]


class GaussianActorMLP(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_layers=3, hidden_size=256,
                 activation="relu", layer_norm=False,
                 log_std_max=2, log_std_min=-20,
                 epsilon=1e-6):
        assert hidden_layers >= 0
        assert hidden_size > 0
        super(GaussianActorMLP, self).__init__()

        if len(observation_space.shape) > 1:
            raise ValueError('MLP observation space must have a single'
                             ' dimension')
        if len(action_space.shape) > 1:
            raise ValueError('MLP action space must have a single'
                             ' dimension')
        if log_std_max < log_std_min:
            raise ValueError("log_std_max must be <= than log_std_min")

        input_size = observation_space.shape[0]
        output_size = action_space.shape[0]
        hidden_layers_size = [hidden_size] * hidden_layers

        # action space for this actor
        self.action_space = gym.spaces.Box(
            low=np.repeat(-1.0, output_size).astype(np.float32, copy=False),
            high=np.repeat(1.0, output_size).astype(np.float32, copy=False),
            dtype=np.float32)

        self.network = models_utils.create_mlp(
            input_size=input_size,
            output_size=output_size * 2,
            hidden_layers=hidden_layers_size,
            layer_norm=layer_norm,
            activation=activation,
            last_activation=None)

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.epsilon = epsilon

    def forward(self, states):
        mean, log_std = torch.chunk(self.network(states), 2, dim=-1)
        log_std.clamp_(min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, states):
        """Sample elements from Gaussian distribution of (mean, std).

        :returns: A tuple with actions sampled form a normal distribution,
            their associated entropies and the actions equivalent to the
            means of the normal distribution.
        """
        means, log_stds = self(states)
        stds = log_stds.exp()
        normal = torch.distributions.Normal(means, stds)

        r_sample = normal.rsample()  # reparameterization trick
        rand_actions = torch.tanh(r_sample)

        # likelihood of the bounded actions (by tanh)
        log_prob = normal.log_prob(r_sample)
        log_prob -= torch.log(1 - rand_actions.pow(2) + self.epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return rand_actions, log_prob, torch.tanh(means)


###############################################################################

class HerCriticMLP(CriticMLP):
    """Utility wrapper to use the `Critic` class in Hindsigh
    Expirience Replay (HER) agents.

    The __init__ signature is the same as that of the `Critic`
    class, since all the arguments given to this class __init__
    are forwarded to the `Critic.__init__`.
    """

    def __init__(self, observation_space, action_space,
                 hidden_layers=3, hidden_size=256,
                 activation="relu", layer_norm=False,
                 *args, **kwargs):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("Hindsight Experience Replay critic expects a"
                            " gym.spaces.Dict observation space")

        obs_space = observation_space['observation']
        goal_space = observation_space['desired_goal']
        flat_obs_space = gym.spaces.Box(
            low=np.concatenate((obs_space.low, goal_space.low)),
            high=np.concatenate((obs_space.high, goal_space.high)),
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
                 hidden_layers=3, hidden_size=256,
                 activation="relu",
                 layer_norm=False,
                 *args, **kwargs):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("Hindsight Experience Replay actor expects a"
                            " gym.spaces.Dict observation space")

        obs_space = observation_space['observation']
        goal_space = observation_space['desired_goal']
        flat_obs_space = gym.spaces.Box(
            low=np.concatenate((obs_space.low, goal_space.low)),
            high=np.concatenate((obs_space.high, goal_space.high)),
            dtype=obs_space.dtype)

        super(HerActorMLP, self).__init__(flat_obs_space, action_space,
                                          hidden_layers, hidden_size,
                                          activation, layer_norm,
                                          *args, **kwargs)

    def forward(self, obs, goals):
        return super(HerActorMLP, self).forward(
            states=torch.cat((obs, goals), dim=1))
