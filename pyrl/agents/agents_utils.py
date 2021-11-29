# -*- coding: utf-8 -*-

"""
Utility functions to initialize some components from string descriptions,
as well as functions with code required by more than agent.
"""

# SciPy
import numpy as np

# ...
import pyrl.util.ugym
from pyrl.models.models import (
    PolicyMLP,
    GaussianPolicyMLP,
    CriticMLP,
    HerPolicyMLP,
    HerGaussianPolicyMLP,
    HerCriticMLP,
)
from .noise import NullActionNoise, NormalActionNoise, OUActionNoise
from .preprocessing import IdentityNormalizer, StandardNormalizer
from .replay_buffer import HerReplayBuffer


###############################################################################


_ACTION_NOISES = {"ou": OUActionNoise, "normal": NormalActionNoise}


def create_action_noise(name, action_space):
    """Creates an action noise from the given definition.

    :param name: Name and standard deviation in the format <name>_<stddev>,
        for example: ou_0.2 or normal_0.1. Additionally the special name
        "none" is supported to create a Null action noise.
    """
    if name.lower() == "none":
        return NullActionNoise(action_space.shape)

    noise, stddev = name.lower().split("_")
    try:
        stddev = float(stddev)
    except ValueError:
        raise ValueError(
            "unable to parse standard deviation value," " expected <noise>_<stddev>"
        )

    try:
        action_range = action_space.high - action_space.low
        return _ACTION_NOISES[noise](
            mu=np.zeros(action_space.shape), sigma=action_range * stddev
        )
    except KeyError:
        raise ValueError("unknown noise type '{}'".format(name))


_NORAMLIZERS = {
    "standard": StandardNormalizer,
}


def create_normalizer(name, shape, epsilon=1e-4, clip_range=float("inf")):
    """Create a Normalizer given its name, and its components

    :param name: Name of the normalizer. It can be "standard" or "none".
    :param shape: Shape of the arrays to normalize.
    """
    if name.lower() == "none":
        return IdentityNormalizer(shape=shape)

    try:
        return _NORAMLIZERS[name](shape=shape, epsilon=epsilon, clip_range=clip_range)
    except KeyError:
        raise ValueError("unknown normalizer type '{}'".format(name))


def dicts_mean(dicts):
    """Computes the mean of multiple dictionaries."""

    if not all(dicts[0].keys() == d.keys() for d in dicts):
        raise ValueError("All dictionaries must have the same keys")

    keys = dicts[0].keys()
    elems = len(dicts)
    return {k: sum([d[k] for d in dicts]) / elems for k in keys}


def create_actor(
    observation_space,
    action_space,
    actor_cls=None,
    actor_kwargs=None,
    policy="deterministic",
):
    """Initializes an actor for the given observation and action
    space. If `actor_cls` is not given the created actor is
    automatically decided using the observation and action
    shapes, as well as the policy type."""
    is_her = pyrl.util.ugym.is_her_space(observation_space)
    if actor_kwargs is None:
        actor_kwargs = {}

    actor_kwargs["observation_space"] = observation_space
    actor_kwargs["action_space"] = action_space

    actor = None
    if actor_cls is not None:
        actor = actor_cls(**actor_kwargs)
    # HER actor
    elif is_her:
        obs_dim = len(observation_space["observation"].shape)
        goal_dim = len(observation_space["desired_goal"].shape)
        action_dim = len(action_space.shape)
        if obs_dim == 1 and goal_dim == 1 and action_dim == 1:
            if policy == "deterministic":
                actor = HerPolicyMLP(**actor_kwargs)
            elif policy == "gaussian":
                return HerGaussianPolicyMLP(**actor_kwargs)
    # Normal actor
    else:
        obs_dim = len(observation_space.shape)
        action_dim = len(action_space.shape)
        if obs_dim == 1 and action_dim == 1:
            if policy == "deterministic":
                actor = PolicyMLP(**actor_kwargs)
            elif policy == "gaussian":
                actor = GaussianPolicyMLP(**actor_kwargs)

    if actor is None:
        raise ValueError(
            "Unknown actor type for observation space"
            " {}, action space {} and policy {}".format(
                observation_space, action_space, policy
            )
        )

    return actor


def create_critic(observation_space, action_space, critic_cls=None, critic_kwargs=None):
    """Initializes a critic for the given observation and action
    space. If `critic_cls` is not given the created critic is
    automatically decided using the observation and action
    shapes."""

    is_her = pyrl.util.ugym.is_her_space(observation_space)
    if critic_kwargs is None:
        critic_kwargs = {}

    critic_kwargs["observation_space"] = observation_space
    critic_kwargs["action_space"] = action_space

    critic = None
    if critic_cls is not None:
        critic = critic_cls(**critic_kwargs)
    elif is_her:
        obs_dim = len(observation_space["observation"].shape)
        goal_dim = len(observation_space["desired_goal"].shape)
        action_dim = len(action_space.shape)
        if obs_dim == 1 and goal_dim == 1 and action_dim == 1:
            critic = HerCriticMLP(**critic_kwargs)
    else:
        obs_dim = len(observation_space.shape)
        action_dim = len(action_space.shape)
        if obs_dim == 1 and action_dim == 1:
            critic = CriticMLP(**critic_kwargs)

    if critic is None:
        raise ValueError(
            "Unknown critic type for observation space"
            " {} and action space {}".format(observation_space, action_space)
        )

    return critic


def load_her_demonstrations(demo_path, env, max_steps, action_fn):
    """Loads demonstrations from the file pointed by `demo_path`. The
    demonstrations are expected to use the state structure expected by HER
    and must be stored in a numpy file with fields 'obs', 'acs' and 'info'
    for the states, the actions and additional information respectivelys.
    """
    demos = np.load(demo_path, allow_pickle=True)
    d_obs, d_acs, d_info = demos["obs"], demos["acs"], demos["info"]
    num_episodes = min(len(d_obs), len(d_acs), len(d_info))

    buffer = HerReplayBuffer(
        obs_shape=env.observation_space["observation"].shape,
        goal_shape=env.observation_space["desired_goal"].shape,
        action_shape=env.action_space.shape,
        max_episodes=num_episodes,
        max_steps=max_steps,
    )

    for obs, acs, infos in zip(d_obs, d_acs, d_info):
        if len(acs) > buffer.max_steps:  # too many steps, ignore
            continue

        states, next_states = obs[:-1], obs[1:]
        transitions = zip(states, acs, next_states, infos)
        for state, action, next_state, info in transitions:
            reward = env.compute_reward(
                next_state["achieved_goal"], next_state["desired_goal"], info
            )

            if action_fn is not None:
                action = action_fn(action)

            buffer.add(
                obs=state["observation"],
                action=action,
                next_obs=next_state["observation"],
                reward=reward,
                terminal=info.get("is_success", False),
                goal=next_state["desired_goal"],
                achieved_goal=next_state["achieved_goal"],
            )
        buffer.save_episode()

    return buffer
