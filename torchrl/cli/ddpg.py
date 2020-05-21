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
import torchrl.agents.ddpg
import torchrl.util.logging


###############################################################################

click.disable_unicode_literals_warning = True

_LOG = torchrl.util.logging.get_logger()


###############################################################################

@click.command(name="ddpg-train")
@click.argument("environment", type=str)
@click.option("--num-epochs", type=int, default=500)
@click.option("--num-episodes", type=int, default=20)
@click.option("--num-rollout-steps", type=int, default=50)
@click.option("--num-evals", type=int, default=1)
@click.option("--num-eval-steps", type=int, default=50)
@click.option("--reward-scale", type=float, default=1.0)
@click.option("--action-penalty", type=float, default=1.0)
@click.option("--action-noise", type=float, default=0.2)
@click.option("--parameter-noise", type=float, default=0.0)
@click.option("--replay-buffer", type=int, default=1000000)
@click.option("--normalize-obs/--no-normalize-obs", default=True)
@click.option("--render/--no-render", default=False)
@click.option("--load", type=click.Path(exists=True, dir_okay=True))
@click.option("--save", type=click.Path(), default="checkpoints/ddpg")
@click.option("--seed", type=int, default=1234)
def cli_ddpg_train(environment,
                   num_epochs,
                   num_episodes,
                   num_rollout_steps,
                   num_evals,
                   num_eval_steps,
                   reward_scale,
                   action_penalty,
                   action_noise,
                   parameter_noise,
                   replay_buffer,
                   normalize_obs,
                   render,
                   load, save, seed):
    # Initialize environment
    print("Loading '{}'".format(environment), end=" ", file=sys.stdout)
    sys.stdout.flush()
    env = gym.make(environment)
    env = gym.wrappers.FlattenObservation(env.unwrapped)
    print("... environment loaded", file=sys.stdout)
    _initialize_seed(seed, env)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    if load:
        print("Loading agent")
        agent = torchrl.agents.ddpg.DDPG.load(load, replay_buffer=True)
    else:
        print("Initializing new agent")
        agent = torchrl.agents.ddpg.DDPG(
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=.99,
            tau=0.001,
            batch_size=128,
            reward_scale=reward_scale,
            replay_buffer_size=replay_buffer,
            actor_lr=0.001,
            critic_lr=0.001,
            action_penalty=action_penalty,
            normalize_observations=normalize_obs,
            action_noise=action_noise,
            parameter_noise=parameter_noise)

    print("Agent trained for", agent.num_episodes, "episodes")
    print("  - # steps:", agent.total_steps)
    print("  - Replay buffer:", len(agent.replay_buffer))

    _LOG.debug("Actor network\n%s", str(agent.actor))
    _LOG.debug("Critic network\n%s", str(agent.critic))

    if render:        # Some environments must be rendered
        env.render()  # before running

    agent.set_train_mode()
    try:
        total_episodes = 0
        for epoch in range(1, num_epochs + 1):
            print("===== EPOCH: {}/{}".format(epoch, num_epochs))
            for episode in range(1, num_episodes + 1):
                total_episodes += 1
                print("----- EPISODE: {}/{} [{}]".format(
                      episode, num_episodes, total_episodes))
                episode_rewards = []
                start_time = time.time()
                state = env.reset()
                agent.reset()

                for rollout_idx in range(num_rollout_steps):
                    print(".", end="", file=sys.stdout)
                    sys.stdout.flush()

                    action = agent.compute_action(state)
                    next_state, reward, done, info = env.step(action)
                    if render:
                        env.render()

                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    episode_rewards.append(reward)
                    # Episode finished before exhausting the # of steps
                    if done:
                        print("[DONE]", end="", file=sys.stdout)
                        break

                # End rollouts
                print("")
                episode_rewards = np.array(episode_rewards)

                # Train (train_steps == last num rollouts)
                for _ in range(rollout_idx + 1):
                    print("+", end="", file=sys.stdout)
                    sys.stdout.flush()
                    agent.train()
                print("")

                print("Elapsed: {:.2f}s, Sum reward: {}, Avg. reward: {},"
                      " Std. reward: {}"
                      .format(time.time() - start_time,
                              np.sum(episode_rewards),
                              np.mean(episode_rewards),
                              np.std(episode_rewards)))

                distance = agent.adapt_parameter_noise()
                if distance is not None:
                    print("Parameter noise distance:", distance)

            # End episodes
            print("----- EVALUATING")
            agent.set_eval_mode()
            for _ in range(num_evals):
                _evaluate(num_eval_steps, agent, env, render)
            agent.set_train_mode()
        # End epochs
    finally:
        print("Saving agent before exiting")
        agent.save(save)
        env.close()

    return 0


###############################################################################

@click.command("ddpg-test")
@click.argument("environment", type=str)
@click.argument("agent-path", type=click.Path(exists=True, dir_okay=True))
@click.option("--num-episodes", type=int, default=5)
@click.option("--num-steps", type=int, default=20)
@click.option("--seed", type=int, default=1234)
def cli_ddpg_test(environment, agent_path, num_episodes, num_steps, seed):
    print("Loading '{}'".format(environment), end=" ", file=sys.stdout)
    sys.stdout.flush()
    env = gym.make(environment)
    env = env.unwrapped
    print("... environment loaded", file=sys.stdout)
    _initialize_seed(seed, env)

    print("Loading agent from '{}'".format(agent_path))
    agent = torchrl.agents.ddpg.DDPG.load(agent_path, replay_buffer=False)
    agent.set_eval_mode()

    print("Agent trained for", agent.num_episodes, "episodes")
    print("  - # steps:", agent.total_steps)
    print("Action space size:", agent.action_space)
    print("Observation space size:", agent.observation_space)

    env.render()  # Some environments must be rendered before running
    all_rewards = []
    for episode in range(num_episodes):
        print("Running episode {}/{}".format(episode + 1, num_episodes))
        raw_input("Press enter to start episode")
        rewards = _evaluate(num_steps, agent, env, render=True)
        all_rewards.append(rewards)
    env.close()

    sum_score = sum(x.sum() for x in all_rewards)
    print("Average sum score over", len(all_rewards),
          "runs:", sum_score / len(all_rewards))

    return 0


###############################################################################

def _initialize_seed(seed, env):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


def _evaluate(max_steps, agent, env, render):
    rewards = []
    done = False

    state = env.reset()
    for _ in range(max_steps):
        action = agent.compute_action(state)
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()

        state = next_state
        rewards.append(reward)

        if done:
            print("[DONE]", end="", file=sys.stdout)
            break

    rewards = np.array(rewards)
    print("", file=sys.stdout)
    print("Last reward: {}, Sum reward: {}, Avg. reward: {},"
          " Std. reward: {}".format(rewards[-1], np.sum(rewards),
                                    np.mean(rewards), np.std(rewards)))
    return rewards
