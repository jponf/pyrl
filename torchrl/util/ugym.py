# -*- coding: utf-8 -*-

import gym
import gym.wrappers


###############################################################################

def make_flat(env_name, unwrap=False, **kwargs):
    """Makes the gym environment and flattens its observation space if there
    is a known way to do it.

    :param env_name: Name of the environment to initialize.
    :param unwrap: Whether or not wrappers added by `gym.make()` must
        be removed before flattening the environment.
    :param **kwargs: Additional arguments passed to `gym.make()`.
    """
    env = gym.make(env_name, **kwargs)
    if unwrap and env.unwrapped is not None:
        env = env.unwrapped
    if is_her_env(env):
        return flatten_her_env(env)
    return env


def is_her_env(env):
    """Tests if the given environment is Hindsight Experience Replay
    compatible.
    """
    return (isinstance(env.observation_space, gym.spaces.Dict) and
            "observation" in env.observation_space.spaces and
            "desired_goal" in env.observation_space.spaces and
            "achieved_goal" in env.observation_space.spaces)


def flatten_her_env(env):
    """Flatten a Hindsight Experience Replay compatible environment."""
    if not isinstance(env.observation_space, gym.spaces.Dict):
        raise ValueError("HER environments observation space is of type {}"
                         " not {}".format(gym.spaces.Dict,
                                          type(env.observation_space)))
    if "observation" not in env.observation_space.spaces:
        raise ValueError("HER environment must have an 'observation' field")
    if "desired_goal" not in env.observation_space.spaces:
        raise ValueError("HER environment must have a 'desired_goal' field")
    if "achieved_goal" not in env.observation_space.spaces:
        raise ValueError("HER environment must have a 'achieved_goal' field")

    env = gym.wrappers.FilterObservation(env, filter_keys=("observation",
                                                           "desired_goal"))
    env = gym.wrappers.FlattenObservation(env)
    return env


def get_wrapped_env_by_class(env, env_class):
    """Traverses the wrapped environments until it
    finds the one from the given class.

    :return: The environment that is an instance of the given `env_class`.
    """
    while not isinstance(env, env_class):
        env = env.unwrapped
    return env
