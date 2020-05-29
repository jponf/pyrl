# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)


# SciPy
import numpy as np

# ...
import pyrl.util.ugym
from .models import ActorMLP, CriticMLP, HerActorMLP, HerCriticMLP
from .noise import NullActionNoise, NormalActionNoise, OUActionNoise
from .preprocessing import IdentityNormalizer, StandardNormalizer

###############################################################################


_ACTION_NOISES = {
    "ou": OUActionNoise,
    "normal": NormalActionNoise
}


def create_action_noise(name, action_space):
    """Creates an action noise from the given definition.

    :param name: Name and standard deviation in the format <name>_<stddev>,
        for example: ou_0.2 or normal_0.1. Additionally the special name
        "none" is supported to create a Null action noise.
    """
    if name.lower() == "none":
        return NullActionNoise(action_space.shape)

    noise, stddev = name.lower().split('_')
    try:
        stddev = float(stddev)
    except ValueError:
        raise ValueError("unable to parse standard deviation value,"
                         " expected <noise>_<stddev>")

    try:
        action_range = action_space.high - action_space.low
        return _ACTION_NOISES[noise](
            mu=np.zeros(action_space.shape),
            sigma=action_range * stddev)
    except KeyError:
        raise ValueError("unknown noise type '{}'".format(name))


_NORAMLIZERS = {
    "standard": StandardNormalizer,
}


def create_normalizer(name, shape, epsilon=1e-4, clip_range=float('inf')):
    """Create a Normalizer given its name, and its components

    :param name: Name of the normalizer. It can be "standard" or "none".
    :param shape: Shape of the arrays to normalize.
    """
    if name.lower() == "none":
        return IdentityNormalizer()

    try:
        return _NORAMLIZERS[name](
            shape=shape,
            epsilon=epsilon,
            clip_range=clip_range)
    except KeyError:
        raise ValueError("unknown normalizer type '{}'".format(name))


def dicts_mean(dicts):
    """Computes the mean of multiple dictionaries.
    """

    if not all(dicts[0].keys() == d.keys() for d in dicts):
        raise ValueError('All dictionaries must have the same keys')

    keys = dicts[0].keys()
    elems = len(dicts)
    return {k: sum([d[k] for d in dicts]) / elems for k in keys}


def create_actor(observation_space, action_space,
                 actor_cls=None, actor_kwargs=None):
    is_her = pyrl.util.ugym.is_her_space(observation_space)
    if actor_kwargs is None:
        actor_kwargs = {}

    actor_kwargs["observation_space"] = observation_space
    actor_kwargs["action_space"] = action_space

    if actor_cls is not None:
        actor = actor_cls(**actor_kwargs)
    else:
        obs_dim = len(observation_space.shape)
        act_dim = len(action_space.shape)

        if obs_dim == 1 and act_dim == 1:
            actor = (HerActorMLP(**actor_kwargs) if is_her
                     else ActorMLP(**actor_kwargs))
        else:
            # TODO: ActorConv
            raise ValueError("Unknown actor type for observation space with"
                             " shape {} and action space with shape {}"
                             .format(observation_space.shape,
                                     action_space.shape))

    return actor


def create_critic(observation_space, action_space,
                  critic_cls=None, critic_kwargs=None):
    is_her = pyrl.util.ugym.is_her_space(observation_space)
    if critic_kwargs is None:
        critic_kwargs = {}

    critic_kwargs["observation_space"] = observation_space
    critic_kwargs["action_space"] = action_space

    if critic_cls is not None:
        critic = critic_cls(**critic_kwargs)
    else:
        obs_dim = len(observation_space.shape)
        act_dim = len(action_space.shape)
        if obs_dim == 1 and act_dim == 1:
            critic = (HerCriticMLP(**critic_kwargs) if is_her else
                      CriticMLP(**critic_kwargs))
        else:
            # TODO: CriticConv
            raise ValueError("Unknown critic type for observation space with"
                             " shape {} and action space with shape {}"
                             .format(observation_space.shape,
                                     action_space.shape))

    return critic
