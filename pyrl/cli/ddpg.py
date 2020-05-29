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
import pyrl.agents.ddpg
import pyrl.agents.replay_buffer
import pyrl.cli.util
import pyrl.util.logging
import pyrl.util.ugym


###############################################################################

click.disable_unicode_literals_warning = True

_LOG = pyrl.util.logging.get_logger()


###############################################################################

@click.command(name="ddpg-train")
@click.argument("environment", type=str)
@click.option("--num-epochs", type=int, default=500)
@click.option("--num-episodes", type=int, default=20)
@click.option("--num-evals", type=int, default=1)
@click.option("--gamma", type=float, default=.99, help="Discount factor")
@click.option("--tau", type=float, default=.001, help="Polyak averaging")
@click.option("--batch-size", type=int, default=128)
@click.option("--replay-buffer", type=int, default=1000000)
@click.option("--reward-scale", type=float, default=1.0)
@click.option("--action-noise", type=str, default="ou_0.2",
              help="Action noise, it can be 'none' or name_std, for example:"
                   " ou_0.2 or normal_0.1.")
@click.option("--parameter-noise", type=float, default=0.0)
@click.option("--obs-normalizer", type=click.Choice(["none", "standard"]),
              default="standard", help="If set to none, the observations "
                                       "won't be normalized")
@click.option("--obs-clip", type=float, default=5.0,
              help="Max/Min. value to clip the observations to if they are"
                   " being normalized.")
@click.option("--render/--no-render", default=False)
@click.option("--load", type=click.Path(exists=True, dir_okay=True))
@click.option("--save", type=click.Path(), default="checkpoints/ddpg")
@click.option("--seed", type=int, default=int(time.time()))
def cli_ddpg_train(environment,
                   num_epochs,
                   num_episodes,
                   num_evals,
                   gamma,
                   tau,
                   batch_size,
                   replay_buffer,
                   reward_scale,
                   action_noise,
                   parameter_noise,
                   obs_normalizer,
                   obs_clip,
                   render,
                   load, save, seed):
    # Initialize environment
    _LOG.info("Loading '%s'", environment)
    env = pyrl.util.ugym.make_flat(environment)

    pyrl.cli.util.initialize_seed(seed)
    env.seed(seed)

    _LOG.info("Action space: %s", str(env.action_space))
    _LOG.info("Observation space: %s", str(env.observation_space))

    if load:
        print("Loading agent")
        agent = pyrl.agents.ddpg.DDPG.load(load, replay_buffer=True)
    else:
        print("Initializing new agent")

        agent = pyrl.agents.ddpg.DDPG(
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            reward_scale=reward_scale,
            replay_buffer_size=replay_buffer,
            actor_lr=0.001,
            critic_lr=0.001,
            observation_normalizer=obs_normalizer,
            observation_clip=obs_clip,
            action_noise=action_noise,
            parameter_noise=parameter_noise)

    _LOG.info("Agent trained for %d episodes", agent.num_episodes)
    _LOG.info("  = # steps: %d", agent.total_steps)
    _LOG.info("  = Replay buffer: %d", len(agent.replay_buffer))
    _LOG.info("    = Max. Size: %d", agent.replay_buffer.max_size)

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
                agent.begin_episode()

                for rollout_idx in range(env.spec.max_episode_steps):
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
                agent.end_episode()
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
                _evaluate(env.spec.max_episode_steps, agent, env, render)
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
    agent = pyrl.agents.ddpg.DDPG.load(agent_path, replay_buffer=False)
    agent.set_eval_mode()

    print("Agent trained for", agent.num_episodes, "episodes")
    print("  - # steps:", agent.total_steps)
    print("Action space size:", agent.action_space)
    print("Observation space size:", agent.observation_space)

    env.render()  # Some environments must be rendered before running
    all_rewards = []
    for episode in range(num_episodes):
        print("Running episode {}/{}".format(episode + 1, num_episodes))
        input("Press enter to start episode")
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
