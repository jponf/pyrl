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
import pyrl.agents
import pyrl.cli.util
import pyrl.trainer
import pyrl.util.logging
import pyrl.util.ugym


###############################################################################

click.disable_unicode_literals_warning = True

_LOG = pyrl.util.logging.get_logger()


###############################################################################

@click.command(name="td3-train")
@click.argument("environment", type=str)
@click.option("--num-epochs", type=int, default=20)
@click.option("--num-episodes", type=int, default=20)
@click.option("--num-envs", type=int, default=1)
@click.option("--num-evals", type=int, default=1)
@click.option("--num-cpus", type=int, default=1)
@click.option("--gamma", type=float, default=.99, help="Discount factor")
@click.option("--tau", type=float, default=.001, help="Polyak averaging")
@click.option("--batch-size", type=int, default=128)
@click.option("--replay-buffer", type=int, default=1000000)
@click.option("--reward-scale", type=float, default=1.0)
@click.option("--policy-delay", type=int, default=2)
@click.option("--random-steps", type=int, default=1500)
@click.option("--action-noise", type=str, default="normal_0.2",
              help="Action noise, it can be 'none' or name_std, for example:"
                   " ou_0.2 or normal_0.1.")
@click.option("--obs-normalizer", type=click.Choice(["none", "standard"]),
              default="standard", help="If set to none, the observations "
                                       "won't be normalized")
@click.option("--obs-clip", type=float, default=5.0,
              help="Min/Max. value to clip the observations to if they are"
                   " being normalized.")
@click.option("--render/--no-render", default=False)
@click.option("--load", type=click.Path(exists=True, dir_okay=True))
@click.option("--save", type=click.Path(), default="checkpoints/td3")
@click.option("--seed", type=int, default=int(time.time()))
def cli_td3_train(environment,
                  num_epochs,
                  num_episodes,
                  num_envs,
                  num_evals,
                  num_cpus,
                  gamma,
                  tau,
                  batch_size,
                  replay_buffer,
                  reward_scale,
                  policy_delay,
                  random_steps,
                  action_noise,
                  obs_normalizer,
                  obs_clip,
                  render,
                  load, save, seed):
    """Trains a TD3 agent on an OpenAI's gym environment."""
    trainer = pyrl.trainer.AgentTrainer(
        agent_cls=pyrl.agents.TD3, env_name=environment,
        seed=seed, num_envs=num_envs, num_cpus=num_cpus,
        root_log_dir=os.path.join(save, "log"))
    pyrl.cli.util.initialize_seed(seed)
    trainer.env.seed(seed)

    if load:
        _LOG.info("Loading agent")
        trainer.initialize_agent(agent_path=load)
    else:
        _LOG.info("Initializing new agent")
        trainer.initialize_agent(
            agent_kwargs=dict(gamma=gamma,
                              tau=tau,
                              batch_size=batch_size,
                              reward_scale=reward_scale,
                              replay_buffer_size=replay_buffer,
                              policy_delay=policy_delay,
                              random_steps=random_steps,
                              actor_lr=1e-3,
                              critic_lr=1e-3,
                              observation_normalizer=obs_normalizer,
                              observation_clip=obs_clip,
                              action_noise=action_noise)
        )

    _LOG.info("Agent Data")
    _LOG.info("  = Train steps: %d", trainer.agent.num_train_steps)
    _LOG.info("  = Replay buffer: %d", len(trainer.agent.replay_buffer))
    _LOG.info("    = Max. Size: %d", trainer.agent.replay_buffer.max_size)

    _LOG.debug("Actor network\n%s", str(trainer.agent.actor))
    _LOG.debug("Critic 1 network\n%s", str(trainer.agent.critic_1))
    _LOG.debug("Critic 2 network\n%s", str(trainer.agent.critic_2))

    _LOG.info("Action space: %s", str(trainer.env.action_space))
    _LOG.info("Observation space: %s", str(trainer.env.observation_space))

    if render:        # Some environments must be rendered
        trainer.env.render()  # before running

    with trainer:
        _run_train(trainer, num_epochs, num_episodes, num_evals, save)

    sys.exit(0)


def _run_train(trainer, num_epochs, num_episodes, num_evals, save_path):
    try:
        for epoch in range(1, num_epochs + 1):
            _LOG.info("===== EPOCH: %d/%d", epoch, num_epochs)
            trainer.agent.set_train_mode()
            _run_train_epoch(trainer, epoch, num_episodes, save_path)

            save_start_time = time.time()
            trainer.agent.save(save_path, replay_buffer=True)
            _LOG.info("Agent saved [%.2fs]", time.time() - save_start_time)

            # End episodes
            _LOG.info("----- EVALUATING")
            trainer.agent.set_eval_mode()
            _evaluate(trainer.agent, trainer.env, num_evals, render=False)
        # End epochs
    except KeyboardInterrupt:
        _LOG.warn("Exiting due to keyboard interruption")
    finally:
        _LOG.info("Saving agent before exiting")
        trainer.agent.save(save_path, replay_buffer=True)
        trainer.env.close()


def _run_train_epoch(trainer, epoch, num_episodes, save_path):
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()
        _LOG.info("----- EPISODE: %d/%d [EPOCH: %d]",
                  episode, num_episodes, epoch)
        trainer.run(num_episodes=1, train_steps=0)
        _LOG.info("Elapsed: %.2fs", time.time() - episode_start_time)


###############################################################################

@click.command("td3-test")
@click.argument("environment", type=str)
@click.argument("agent-path", type=click.Path(exists=True, dir_okay=True))
@click.option("--num-episodes", type=int, default=5)
@click.option("--pause/--no-pause", default=False,
              help="Pause (or not) before running an episode.")
@click.option("--seed", type=int, default=int(time.time()))
def cli_td3_test(environment, agent_path, num_episodes, pause, seed):
    """Runs a previosly trained TD3 agent on an OpenAI's gym environment."""
    _LOG.info("Loading '%s'", environment)
    env = pyrl.util.ugym.make_flat(environment)
    pyrl.cli.util.initialize_seed(seed)
    env.seed(seed)

    _LOG.info("Loading agent from %s", agent_path)
    agent = pyrl.agents.TD3.load(agent_path, replay_buffer=False)
    agent.set_eval_mode()

    _LOG.info("Agent trained for %d stes", agent.num_train_steps)
    _LOG.info("Action space size: %s", str(agent.action_space))
    _LOG.info("Observation space size: %s", str(agent.observation_space))

    env.render()  # Some environments must be rendered before running
    all_rewards = []
    for episode in range(num_episodes):
        _LOG.info("Running episode %d/%d", episode + 1, num_episodes)
        rewards = _evaluate(agent, env, num_evals=1, render=True)
        all_rewards.extend(rewards)

    env.close()
    sum_score = sum(x.sum() for x in all_rewards)
    _LOG.info("Average sum score over %d runs: %d",
              len(all_rewards), sum_score / len(all_rewards))

    sys.exit(0)


###############################################################################

def _evaluate(agent, env, num_evals, render):
    all_rewards = []

    for _ in range(num_evals):
        rewards, infos, done = pyrl.cli.util.evaluate(
            agent, env, env.spec.max_episode_steps, render)

        all_rewards.append(rewards)
        if done:
            _LOG.info("[DONE]")

        _LOG.info("Last reward: %.5f, Sum reward: %.5f,"
                  " Avg. reward: %.5f, Std. reward: %.5f",
                  rewards[-1], np.sum(rewards),
                  np.mean(rewards), np.std(rewards))

    return all_rewards
