import gym
import numpy as np
import random
import torch
from typing import Optional

import pyrl.util.logging
from pyrl.agents.core import Agent

_LOG = pyrl.util.logging.get_logger()


def initialize_seed(seed: int, env: Optional[gym.Env]):
    """Initializes the seed of different PRNGs.

    :param seed: Value to initialize the PRNGs.
    :param env: If provided the environment seed is also initialized.
    """
    np.random.seed(seed)
    random.seed(seed)
    if env:
        env.seed(seed)
    torch.manual_seed(seed)


def cli_agent_evaluation(
    agent: Agent,
    env: gym.Env,
    num_episodes: int,
    pause: bool,
    is_her: bool,
):
    agent.set_eval_mode()

    _LOG.info("Agent trained for %d stes", agent.num_train_steps)
    _LOG.info("Action space size: %s", str(env.action_space))
    _LOG.info("Observation space size: %s", str(env.observation_space))

    all_rewards = []
    all_success = []
    for episode in range(num_episodes):
        _LOG.info("Running episode %d/%d", episode + 1, num_episodes)
        if pause:
            input("Press enter to start episode")
        rewards, success = evaluate_agent(
            agent=agent,
            env=env,
            num_evals=1,
            is_her=is_her,
            render=True,
        )
        all_rewards.extend(rewards)
        all_success.extend(success)
    env.close()

    sum_score = sum(x.sum() for x in all_rewards)
    sum_success = sum(all_success)
    _LOG.info("Avg sum score: %.5f", sum_score / len(all_rewards))

    if is_her:  # Only meaningful on HER environments
        _LOG.info(
            "Success %d of %d (%.2f)",
            sum_success,
            num_episodes,
            sum_success / num_episodes,
        )


def evaluate_agent(
    agent: Agent,
    env: gym.Env,
    num_evals: int,
    is_her: bool,
    render: bool,
):
    all_rewards = []
    all_success = []

    for _ in range(num_evals):
        rewards, infos, _ = _evaluate(
            agent=agent,
            env=env,
            max_steps=env.spec.max_episode_steps,
            render=render,
        )

        success = any(x["is_success"] for x in infos) if is_her else False
        all_rewards.append(rewards)
        all_success.append(success)

        if success:
            _LOG.info("[SUCCESS]")

        _LOG.info(
            "Last reward: %.5f, Sum reward: %.5f,"
            " Avg. reward: %.5f, Std. reward: %.5f",
            rewards[-1],
            np.sum(rewards),
            np.mean(rewards),
            np.std(rewards),
        )

    return all_rewards, all_success


def _evaluate(agent: Agent, env: gym.Env, max_steps: int, render: bool):
    """Utility to collect all steps of an evaluation.

    :return: A tuple of (rewards, infos, done) information. Where rewards
        and infos are as returned by
    """
    rewards = []
    infos = []
    done = False

    state = env.reset()
    for _ in range(max_steps):
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
