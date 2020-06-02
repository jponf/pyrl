# -*- coding: utf-8 -*-

import random

import six

# SciPy Stack
import numpy as np

# Torch
import torch


###############################################################################

def initialize_seed(seed):
    """Initializes the seed of different PRNGs.

    :param seed: Value to initialize the PRNGs.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent, env, max_steps, render):
    """Evaluates the given agent on an environment.

    :return: A numpy array with the reward of each step taken by the agent.
    """
    rewards = []
    infos = []
    done = False

    state = env.reset()
    for _ in six.moves.range(max_steps):
        action = agent.compute_action(state)
        next_state, reward, done, info = env.step(action)
        if render:
            env.render()

        state = next_state
        rewards.append(reward)
        infos.append(info)
        if done:
            break

    return np.array(rewards), infos, done
