# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

# Standard library
import os
import sys
import time

# SciPy stack
import numpy as np

# Click (command line options)
import click

# ...
import pyrl.agents.td3
import pyrl.cli.util
import pyrl.util.logging
import pyrl.util.ugym

###############################################################################

click.disable_unicode_literals_warning = True

_LOG = pyrl.util.logging.get_logger()

_EPISODE_SUMMARY_MSG = ("Last reward: {:.5f}, Sum reward: {:.5f},"
                        " Avg. reward: {:.5f}, Std. reward: {:.5f}")


###############################################################################

@click.command(name="td3-train")
@click.argument("environment", type=str)
@click.option("--num-epochs", type=int, default=500)
@click.option("--num-episodes", type=int, default=20)
@click.option("--num-rollout-steps", type=int, default=50)
@click.option("--num-evals", type=int, default=1)
@click.option("--num-eval-steps", type=int, default=50)
@click.option("--policy-delay", type=int, default=2)
@click.option("--reward-scale", type=float, default=1.0)
@click.option("--action-noise", type=float, default=0.2)
@click.option("--random-steps", type=int, default=2500)
@click.option("--replay-buffer", type=int, default=1000000)
@click.option("--actor-hidden-layers", type=int, default=3)
@click.option("--actor-hidden-size", type=int, default=256)
@click.option("--critic-hidden-layers", type=int, default=3)
@click.option("--critic-hidden-size", type=int, default=256)
@click.option("--normalize-obs/--no-normalize-obs", default=True)
@click.option("--render/--no-render", default=False)
@click.option("--load", type=click.Path(exists=True, dir_okay=True))
@click.option("--save", type=click.Path(), default="checkpoints/td3")
@click.option("--seed", type=int, default=1234)
def cli_td3_train(environment,
                  num_epochs,
                  num_episodes,
                  num_rollout_steps,
                  num_evals,
                  num_eval_steps,
                  policy_delay,
                  reward_scale,
                  action_noise,
                  random_steps,
                  replay_buffer,  # replay buffer size
                  actor_hidden_layers,
                  actor_hidden_size,
                  critic_hidden_layers,
                  critic_hidden_size,
                  normalize_obs,
                  render,
                  load, save, seed):
    """Trains a TD3 agent on an OpenAI's gym environment."""
    # Initialize environment
    _LOG.info("Loading '%s'", environment)
    env = pyrl.util.ugym.make_flat(environment, unwrap=True)
    pyrl.cli.util.initialize_seed(seed, env)

    _LOG.info("Action space: %s", str(env.action_space))
    _LOG.info("Observation space: %s", str(env.observation_space))

    if load:
        print("Loading agent")
        agent = pyrl.agents.td3.TD3.load(load,
                                         replay_buffer=True,
                                         log_dir=os.path.join(save, "log"))
    else:
        print("Initializing new agent")
        agent = pyrl.agents.td3.TD3(
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=.99,
            tau=0.005,
            batch_size=128,
            replay_buffer_size=replay_buffer,
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
            normalize_observations=normalize_obs,
            action_noise=action_noise,
            random_steps=random_steps,
            log_dir=os.path.join(save, "log"))

    _LOG.info("Agent trained for %d episodes", agent.num_episodes)
    _LOG.info("  - # steps: %d", agent.total_steps)
    _LOG.info("  - Replay buffer: %d", len(agent.replay_buffer))

    _LOG.debug("Actor network\n%s", str(agent.actor))
    _LOG.debug("Critic 1 network\n%s", str(agent.critic_1))
    _LOG.debug("Critic 2 network\n%s", str(agent.critic_2))

    if render:        # Some environments must be rendered
        env.render()  # before running

    agent.set_train_mode()
    try:
        total_episodes = 0
        for epoch in range(1, num_epochs + 1):
            print("===== EPOCH: {}/{} [Episodes so far: {}]".format(
                epoch, num_epochs, total_episodes))
            episodes = _run_train_epoch(agent, env, num_episodes,
                                        num_rollout_steps, render)
            total_episodes += episodes

            # End episodes
            start_time = time.time()
            agent.save(save, replay_buffer=True)
            print("Agent saved [{:.2f}s]".format(time.time() - start_time))

            print("----- EVALUATING")
            agent.set_eval_mode()
            _evaluate(agent, env, num_evals, num_eval_steps, render)
            agent.set_train_mode()
        # End epochs
    except KeyboardInterrupt:
        _LOG.warn("Exiting due to keyboard interruption")
    finally:
        print("Saving agent before exiting")
        agent.save(save, replay_buffer=True)
        env.close()

    return 0


def _run_train_epoch(agent, env, num_episodes, num_steps, render):
    for episode in range(1, num_episodes + 1):
        print("----- EPISODE: {}/{}".format(episode, num_episodes))
        start_time = time.time()

        rewards = _run_train_episode(agent, env, num_steps, render)

        # Train (train_steps == last num rollouts)
        agent.train(len(rewards))
        print("Elapsed: {:.2f}s".format(time.time() - start_time))
        print(_EPISODE_SUMMARY_MSG.format(rewards[-1],
                                          np.sum(rewards),
                                          np.mean(rewards),
                                          np.std(rewards)))

    return episode


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

    print("")
    return np.array(rewards)


###############################################################################

@click.command("td3-test")
@click.argument("environment", type=str)
@click.argument("agent-path", type=click.Path(exists=True, dir_okay=True))
@click.option("--num-episodes", type=int, default=5)
@click.option("--num-steps", type=int, default=20)
@click.option("--pause/--no-pause", default=False,
              help="Pause (or not) before running an episode.")
@click.option("--seed", type=int, default=1234)
def cli_td3_test(environment, agent_path, num_episodes, num_steps,
                 pause, seed):
    """Runs a previosly trained TD3 agent on an OpenAI's gym environment."""
    print("Loading '{}'".format(environment), end=" ", file=sys.stdout)
    sys.stdout.flush()
    env = pyrl.util.ugym.make_flat(environment, unwrap=True)
    print("... environment loaded", file=sys.stdout)
    pyrl.cli.util.initialize_seed(seed, env)

    print("Loading agent from '{}'".format(agent_path))
    agent = pyrl.agents.td3.TD3.load(agent_path, replay_buffer=False)
    agent.set_eval_mode()

    print("Agent trained for", agent.num_episodes, "episodes")
    print("  - # steps:", agent.total_steps)
    print("Action space size:", agent.action_space)
    print("Observation space size:", agent.observation_space)

    all_rewards = []
    for episode in range(num_episodes):
        print("Running episode {}/{}".format(episode + 1, num_episodes))
        if pause:
            raw_input("Press enter to start episode")
        rewards = _evaluate(agent, env, 1, num_steps, render=True)
        all_rewards.extend(rewards)
    env.close()

    sum_score = sum(x.sum() for x in all_rewards)
    print("Average sum score over", len(all_rewards),
          "runs:", sum_score / len(all_rewards))

    return 0


###############################################################################

def _evaluate(agent, env, num_evals, num_steps, render):
    all_rewards = []
    for _ in range(num_evals):
        rewards, done = pyrl.cli.util.evaluate(agent, env, num_steps,
                                                  render)
        all_rewards.append(rewards)

        if done:
            print("[DONE]")
        print(_EPISODE_SUMMARY_MSG.format(rewards[-1], np.sum(rewards),
                                          np.mean(rewards), np.std(rewards)))

    return all_rewards
