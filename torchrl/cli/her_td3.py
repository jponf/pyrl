# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

# Standard library
import math
import os
import sys
import time

# SciPy stack
import numpy as np

# OpenAI
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

@click.command(name="her-td3-train")
@click.argument("environment", type=str)
@click.option("--num-epochs", type=int, default=20)
@click.option("--num-cycles", type=int, default=50)
@click.option("--num-episodes", type=int, default=16)
@click.option("--num-steps", type=int, default=50)
@click.option("--num-evals", type=int, default=1)
@click.option("--demo-path", type=click.Path(exists=True, file_okay=True),
              required=False, help="Path to the file with demonstration runs.")
@click.option("--replay-k", type=int, default=4,
              help="The ratio between HER replays and regular replays,"
                   " e.g. k = 4 -> 4 times as many HER replays as regular"
                   " replays.")
@click.option("--policy-delay", type=int, default=2)
@click.option("--reward-scale", type=float, default=1.0)
@click.option("--action-noise", type=float, default=0.2)
@click.option("--replay-buffer", type=int, default=1000000)
@click.option("--actor-hidden-layers", type=int, default=3)
@click.option("--actor-hidden-size", type=int, default=256)
@click.option("--critic-hidden-layers", type=int, default=3)
@click.option("--critic-hidden-size", type=int, default=256)
@click.option("--q-filter/--no-q-filter", default=False)
@click.option("--normalize-obs/--no-normalize-obs", default=True)
@click.option("--render/--no-render", default=False)
@click.option("--load", type=click.Path(exists=True, dir_okay=True))
@click.option("--save", type=click.Path(), default="checkpoints/her-td3")
@click.option("--seed", type=int, default=1234)
def cli_her_td3_train(environment,
                      num_epochs,
                      num_cycles,
                      num_episodes,
                      num_steps,
                      num_evals,
                      demo_path,
                      replay_k,
                      policy_delay,
                      reward_scale,
                      action_noise,
                      replay_buffer,
                      actor_hidden_layers,
                      actor_hidden_size,
                      critic_hidden_layers,
                      critic_hidden_size,
                      q_filter,
                      normalize_obs,
                      render,
                      load, save, seed):
    """Trains a HER + TD3 agent on an OpenAI's gym environment."""
    _LOG.info("Loading '%s'", environment)
    env = gym.make(environment).unwrapped
    torchrl.cli.util.initialize_seed(seed, env)

    _LOG.info("Action space: %s", str(env.action_space))
    _LOG.info("Observation space: %s", str(env.observation_space))

    if load:
        _LOG.info("Loading agent")
        agent = torchrl.agents.her_td3.HerTD3.load(
            load, env, replay_buffer=True, log_dir=os.path.join(save, "log"))
    else:
        _LOG.info("Initializing new agent")
        agent = torchrl.agents.her_td3.HerTD3(
            env=env,
            eps_greedy=0.2,
            gamma=1.0 - 1.0 / num_steps,
            tau=0.005,
            replay_k=replay_k,
            batch_size=256,
            demo_batch_size=128,
            replay_buffer_episodes=int(math.ceil(replay_buffer / num_steps)),
            replay_buffer_steps=num_steps,
            policy_delay=policy_delay,
            reward_scale=reward_scale,
            actor_hidden_layers=actor_hidden_layers,
            actor_hidden_size=actor_hidden_size,
            actor_activation="relu",
            actor_lr=3e-4,
            critic_hidden_layers=critic_hidden_layers,
            critic_hidden_size=critic_hidden_size,
            critic_activation="relu",
            critic_lr=3e-4,
            q_filter=q_filter,
            normalize_observations=normalize_obs,
            normalize_observations_clip=5.0,
            action_noise=action_noise,
            log_dir=os.path.join(save, "log"))

    _LOG.info("Agent trained for %d episodes", agent.num_episodes)
    _LOG.info("  = # steps: %d", agent.total_steps)
    _LOG.info("  = Replay buffer: %d", agent.replay_buffer.count_steps())
    _LOG.info("    = Max. Episodes: %d", agent.replay_buffer.max_episodes)
    _LOG.info("    = Max. Steps: %d", agent.replay_buffer.max_steps)

    _LOG.debug("Actor network\n%s", str(agent.actor))
    _LOG.debug("Critic 1 network\n%s", str(agent.critic_1))
    _LOG.debug("Critic 2 network\n%s", str(agent.critic_2))

    if demo_path:
        _LOG.info("Loading demonstrations")
        agent.load_demonstrations(demo_path)

    if render:        # Some environments must be rendered
        env.render()  # before running

    agent.set_train_mode()
    try:
        for epoch in range(1, num_epochs + 1):
            _LOG.info("===== EPOCH: %d/%d", epoch, num_epochs)
            _run_train_epoch(agent, env, epoch,
                             num_cycles, num_episodes, num_steps,
                             save, render)

            # End episodes
            _LOG.info("----- EVALUATING")
            agent.set_eval_mode()
            _evaluate(agent, env, num_evals, num_steps, render)
            agent.set_train_mode()
        # End epochs
    except KeyboardInterrupt:
        _LOG.warn("Exiting due to keyboard interruption")
    finally:
        _LOG.info("Saving agent before exiting")
        agent.save(save, replay_buffer=True)
        env.close()

    return 0


def _run_train_epoch(agent, env, epoch,
                     num_cycles, num_episodes, num_steps,
                     save_path, render):
    for cycle in range(1, num_cycles + 1):
        _LOG.info("----- CYCLE: %d/%d [EPOCH: %d]", cycle, num_cycles, epoch)
        _run_train_cycle(agent, env, epoch, cycle,
                         num_episodes, num_steps, render)

        # Train (train_steps == last num rollouts)
        _LOG.info("Performing %d optimization steps", num_steps)
        train_time = time.time()
        agent.train(num_steps)
        _LOG.info("Elapsed: %.2fs", time.time() - train_time)

        save_time = time.time()
        agent.save(save_path, replay_buffer=True)
        _LOG.info("Agent saved [%.2fs]", time.time() - save_time)


def _run_train_cycle(agent, env, epoch, cycle,
                     num_episodes, num_steps, render):
    for episode in range(1, num_episodes + 1):
        _LOG.info("----- EPISODE: %d/%d [EPOCH: %d | CYCLE: %d]",
                  episode, num_episodes, epoch, cycle)
        start_time = time.time()
        rewards = _run_train_episode(agent, env, num_steps, render)

        _LOG.info("Elapsed: %.2fs", time.time() - start_time)
        _LOG.info("Last reward: %.5f, Sum reward: %.5f,"
                  " Avg. reward: %.5f, Std. reward: %.5f",
                  rewards[-1], np.sum(rewards),
                  np.mean(rewards), np.std(rewards))


def _run_train_episode(agent, env, num_steps, render):
    rewards = []
    state = env.reset()
    agent.reset()

    for _ in range(num_steps):
        print(".", end="", file=sys.stdout)
        sys.stdout.flush()

        action = agent.compute_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        rewards.append(reward)

        # Render environment?
        if render:
            env.render()

        # Episode finished before exhausting the # of steps
        if done:
            print("[DONE]", end="", file=sys.stdout)
            break

    agent.end_episode()
    print("")

    return np.array(rewards)


###############################################################################

@click.command("her-td3-optimize")
@click.argument("environment", type=str)
@click.argument("agent-path", type=click.Path(exists=True, dir_okay=True))
@click.option("--train-steps", type=int, required=True,
              help="Number of training steps to perform.")
@click.option("--demo-path", type=click.Path(exists=True, file_okay=True),
              required=False, help="Path to the file with demonstration runs.")
@click.option("--save", type=str, default="",
              help="Path to save the agent, if not given the agent will be"
                   "overwritten.")
def cli_her_td3_optimize(environment, agent_path,
                         train_steps, demo_path, save):
    """Loads an agent and performs optimization steps using the data that
    it has previously stored into its replay buffer.
    """
    if not save:
        _LOG.info("Alternative save path not given, the agent will be"
                  " overwritten")
        save = agent_path

    _LOG.info("Loading '%s'", environment)
    env = gym.make(environment).unwrapped

    _LOG.info("Loading agent from '%s'", agent_path)
    agent = torchrl.agents.her_td3.HerTD3.load(
        agent_path, env, replay_buffer=True,
        log_dir=os.path.join(save, "log"))
    agent.set_train_mode()

    _LOG.info("Agent trained for %d episodes", agent.num_episodes)
    _LOG.info("  - # steps: %d", agent.total_steps)
    _LOG.info("  - Replay buffer: %d", agent.replay_buffer.count_steps())

    if demo_path:
        _LOG.info("Loading demonstrations")
        agent.load_demonstrations(demo_path)

    try:
        _LOG.info("Performing %d optimization steps", train_steps)
        train_time = time.time()
        agent.train(train_steps, progress=True)
        _LOG.info("Elapsed: %.2fs", time.time() - train_time)
    except KeyboardInterrupt:
        _LOG.warn("Exiting due to keyboard interruption")
    finally:
        _LOG.info("Saving agent before exiting")
        agent.save(save, replay_buffer=True)
        env.close()

    sys.exit(0)


###############################################################################

@click.command("her-td3-test")
@click.argument("environment", type=str)
@click.argument("agent-path", type=click.Path(exists=True, dir_okay=True))
@click.option("--num-episodes", type=int, default=5)
@click.option("--num-steps", type=int, default=50)
@click.option("--pause/--no-pause", default=False,
              help="Pause (or not) before running an episode.")
@click.option("--seed", type=int, default=1234)
def cli_her_td3_test(environment, agent_path, num_episodes, num_steps,
                     pause, seed):
    """Runs a previosly trained HER+TD3 agent on a gym environment."""
    _LOG.info("Loading '%s'", environment)
    env = gym.make(environment).unwrapped
    torchrl.cli.util.initialize_seed(seed, env)

    _LOG.info("Loading agent from '%s'", agent_path)
    agent = torchrl.agents.her_td3.HerTD3.load(agent_path, env,
                                               replay_buffer=False)
    agent.set_eval_mode()

    _LOG.info("Agent trained for %d episodes", agent.num_episodes)
    _LOG.info("  - # steps: %d", agent.total_steps)
    _LOG.info("Action space size: %s", str(agent.env.action_space))
    _LOG.info("Observation space size: %s", str(agent.env.observation_space))

    all_rewards = []
    all_success = []
    for episode in range(num_episodes):
        _LOG.info("Running episode %d/%d", episode + 1, num_episodes)
        if pause:
            input("Press enter to start episode")
        rewards, success = _evaluate(agent, env, 1, num_steps, render=True)
        all_rewards.extend(rewards)
        all_success.extend(success)
    env.close()

    sum_score = sum(x.sum() for x in all_rewards)
    sum_success = sum(all_success)
    _LOG.info("Avg sum score: %.5f", sum_score / len(all_rewards))
    # _LOG.info("Last rewards: %s", ", ".join(str(x[-1]) for x in all_rewards))
    _LOG.info("Success %d of %d (%.2f)",
              sum_success, num_episodes, sum_success / num_episodes)

    return 0


###############################################################################

def _evaluate(agent, env, num_evals, num_steps, render):
    all_rewards = []
    all_success = []
    for _ in range(num_evals):
        rewards, infos, done = torchrl.cli.util.evaluate(agent, env, num_steps,
                                                         render)
        all_rewards.append(rewards)
        all_success.append(any(x.get("is_success", False) for x in infos))
        if done:
            _LOG.info("[DONE]")

        _LOG.info("Last reward: %.5f, Sum reward: %.5f,"
                  " Avg. reward: %.5f, Std. reward: %.5f",
                  rewards[-1], np.sum(rewards),
                  np.mean(rewards), np.std(rewards))

    return all_rewards, all_success
