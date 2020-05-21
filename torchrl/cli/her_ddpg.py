# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

# Standard library
import random
import sys
import time

# SciPy stack
import numpy as np

# Torch
import torch

# OpenAI Gym
import gym

# Click (command line options)
import click

# ...
import torchrl.cli.util
import torchrl.agents.her_td3
import torchrl.util.logging


###############################################################################

click.disable_unicode_literals_warning = True

_LOG = torchrl.util.logging.get_logger()


###############################################################################

@click.command(name="her-ddpg-train")
@click.argument("environment", type=str)
@click.option("--num-epochs", type=int, default=200)
@click.option("--num-cycles", type=int, default=50)
@click.option("--num-episodes", type=int, default=16)
@click.option("--num-steps", type=int, default=50)
@click.option("--num-evals", type=int, default=20)
@click.option("--demo-path", type=click.Path(exists=True, file_okay=True),
              required=False, help="Path to the file with demonstration runs.")
@click.option("--replay-k", type=int, default=2,
              help="The ratio between HER replays and regular replays,"
                   " e.g. k = 4 -> 4 times as many HER replays as regular"
                   " replays.")
@click.option("--reward-scale", type=float, default=1.0)
@click.option("--action-noise", type=float, default=0.2)
@click.option("--replay-buffer", type=int, default=1000000)
@click.option("--actor-hidden-layers", type=int, default=3)
@click.option("--actor-hidden-size", type=int, default=256)
@click.option("--critic-hidden-layers", type=int, default=3)
@click.option("--critic-hidden-size", type=int, default=256)
@click.option("--normalize-obs/--no-normalize-obs", default=True)
@click.option("--render/--no-render", default=False)
@click.option("--load", type=click.Path(exists=True, dir_okay=True))
@click.option("--save", type=click.Path(), default="checkpoints/her_ddpg")
@click.option("--seed", type=int, default=1234)
def cli_her_ddpg_train(environment,
                       num_epochs,
                       num_cycles,
                       num_episodes,
                       num_rollout_steps,
                       num_train_steps,
                       num_eval_steps,
                       replay_k,
                       policy_delay,
                       reward_scale,
                       action_noise,
                       random_steps,
                       replay_buffer_episodes,
                       actor_hidden_layers,
                       actor_hidden_size,
                       critic_hidden_layers,
                       critic_hidden_size,
                       normalize_obs,
                       render,
                       load, save, seed):
    """Trains a HER + DDPG agent on an OpenAI's gym environment."""
    _LOG.info("Loading '%s'", environment)
    sys.stdout.flush()
    env = gym.make(environment).unwrapped
    torchrl.cli.util.initialize_seed(seed, env)

    _LOG.info("Action space: %s", str(env.action_space))
    _LOG.info("Observation space: %s", str(env.observation_space))

    if load:
        print("Loading agent")
        agent = torchrl.agents.her_ddpg.HERDDPG.load(
            path=load,
            reward_fn=lambda x, y: env.compute_reward(x, y, {}),
            train=True)
    else:
        print("Initializing new agent")
        agent = torchrl.agents.her_ddpg.HERDDPG(
            observation_space=env.observation_space,
            action_space=env.action_space,
            reward_fn=lambda x, y: env.compute_reward(x, y, {}),
            gamma=.95,
            tau=1e-3,
            replay_k=replay_k,
            batch_size=128,
            demo_batch_size=128,
            memory_size=100000,
            actor_lr=1e-4,
            critic_lr=1e-3,
            normalize_observations=True,
            normalize_observations_clip=5.0)

    # input("Press enter to start training")
    agent.set_train_mode()
    try:
        for epoch in range(1, num_epochs + 1):
            print("===== EPOCH: {}/{}".format(epoch, num_epochs))
            for cycle in range(1, num_cycles + 1):
                print("----- Cycle: {}/{}".format(cycle, num_cycles))

                for episode in range(1, num_episodes + 1):
                    print("E({})".format(episode, num_episodes),
                          end="", file=sys.stdout)

                    episode_rewards = []
                    start_time = time.time()
                    state = env.reset()

                    for _ in range(num_rollout_steps):
                        print(".", end="", file=sys.stdout)
                        sys.stdout.flush()

                        action = agent.compute_action(state)
                        next_state, reward, done, info = env.step(action)

                        agent.update(state, action, reward, next_state, done)
                        state = next_state
                        episode_rewards.append(reward)

                        # Episode finished before exhausting the # of steps
                        if done:
                            print("[DONE]", end="", file=sys.stdout)
                            break

                    # End rollouts
                    print("[{:.2f}s]".format(time.time() - start_time))
                    agent.reset()

                # End episodes [Train]
                for _ in range(num_train_steps):
                    print("+", end="", file=sys.stdout)
                    sys.stdout.flush()
                    agent.train()
                agent.update_target_networks()
                print("")

            # End cycles
            print("----- EVALUATING")
            agent.set_eval_mode()
            _evaluate(num_eval_steps, agent, env)
            agent.set_train_mode()
        # End epochs
    finally:
        print("Saving agent before exiting")
        agent.save(save)
        env.close()

    return 0


###############################################################################

@click.command("her-ddpg-test")
@click.argument("environment", type=str)
@click.argument("agent-path", type=click.Path(exists=True, dir_okay=True))
@click.option("--num-episodes", type=int, default=5)
@click.option("--num-steps", type=int, default=20)
@click.option("--seed", type=int, default=1234)
def cli_her_ddpg_test(environment, agent_path, num_episodes, num_steps, seed):
    print("Loading '{}'".format(environment), end=" ", file=sys.stdout)
    sys.stdout.flush()
    env = gym.make(environment, render=True)
    env = env.unwrapped
    print("... environment loaded", file=sys.stdout)
    _initialize_seed(seed, env)

    print("Loading agent from '{}'".format(agent_path))
    agent = torchrl.agents.her_ddpg.HERDDPG.load(
        agent_path,
        reward_fn=lambda x, y: env.compute_reward(x, y, {}),
        train=False)
    agent.set_eval_mode()

    print("Action space size:", agent.action_space)
    print("Observation space size:", agent.observation_space)

    for episode in range(num_episodes):
        print("Running episode {}/{}".format(episode + 1, num_episodes))
        raw_input("Press enter to start episode")
        _evaluate(num_steps, agent, env)

    env.close()
    return 0


###############################################################################

def _initialize_seed(seed, env):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


def _evaluate(max_steps, agent, env):
    rewards = []
    done = False

    state = env.reset()
    for _ in range(max_steps):
        print(".", end="", file=sys.stdout)
        sys.stdout.flush()

        action = agent.compute_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state

        rewards.append(reward)

        if done:
            print("[DONE]", end="", file=sys.stdout)
            break

    rewards = np.array(rewards)
    print("", file=sys.stdout)
    print("Last reward: {}, Avg. reward: {}, Std. rewards: {}"
          .format(rewards[-1], np.mean(rewards), np.std(rewards)))
