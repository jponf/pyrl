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

# OpenAI Gym
import gym

# Click (command line options)
import click

# ...
import pyrl.cli.util
import pyrl.agents
import pyrl.util.logging
import pyrl.trainer


###############################################################################

click.disable_unicode_literals_warning = True

_LOG = pyrl.util.logging.get_logger()


###############################################################################

@click.command(name="her-ddpg-train")
@click.argument("environment", type=str)
@click.option("--num-epochs", type=int, default=20)
@click.option("--num-cycles", type=int, default=50)
@click.option("--num-episodes", type=int, default=16)
@click.option("--num-envs", type=int, default=1)
@click.option("--num-evals", type=int, default=1)
@click.option("--num-cpus", type=int, default=1)
@click.option("--demo-path", type=click.Path(exists=True, file_okay=True),
              required=False, help="Path to the file with demonstration runs.")
@click.option("--eps-greedy", type=float, default=0.2)
@click.option("--reward-scale", type=float, default=1.0)
@click.option("--replay-buffer", type=int, default=1000000)
@click.option("--replay-k", type=int, default=4,
              help="The ratio between HER replays and regular replays,"
                   " e.g. k = 4 -> 4 times as many HER replays as regular"
                   " replays.")
@click.option("--q-filter/--no-q-filter", default=False)
@click.option("--action-noise", type=str, default="ou_0.2")
@click.option("--obs-normalizer", type=click.Choice(["none", "standard"]),
              default="standard", help="If set to none, the observations "
                                       "won't be normalized")
@click.option("--obs-clip", type=float, default=5.0,
              help="Min/Max. value to clip the observations to if they are"
                   " being normalized.")
@click.option("--render/--no-render", default=False)
@click.option("--load", type=click.Path(exists=True, dir_okay=True))
@click.option("--save", type=click.Path(), default="checkpoints/her-ddpg")
@click.option("--seed", type=int, default=int(time.time()))
def cli_her_ddpg_train(environment,
                       num_epochs,
                       num_cycles,
                       num_episodes,
                       num_envs,
                       num_evals,
                       num_cpus,
                       demo_path,
                       eps_greedy,
                       reward_scale,
                       replay_buffer,
                       replay_k,
                       q_filter,
                       action_noise,
                       obs_normalizer,
                       obs_clip,
                       render,
                       load, save, seed):
    """Trains a HER + DDPG agent on an OpenAI's gym environment."""
    trainer = pyrl.trainer.AgentTrainer(
        agent_cls=pyrl.agents.HerDDPG, env_name=environment,
        seed=seed, num_envs=num_envs, num_cpus=num_cpus,
        root_log_dir=os.path.join(save, "log"))

    pyrl.cli.util.initialize_seed(seed)
    trainer.env.seed(seed)

    if load:
        _LOG.info("Save path already exists, loading previously trained agent")
        trainer.initialize_agent(agent_path=load, demo_path=demo_path)
    else:
        _LOG.info("Initializing new agent")
        env = trainer.env
        agent_kwargs = dict(
            eps_greedy=eps_greedy,
            gamma=1.0 - 1.0 / env.spec.max_episode_steps,
            tau=0.005,
            batch_size=128,
            reward_scale=reward_scale,
            replay_buffer_episodes=int(math.ceil(replay_buffer /
                                                 env.spec.max_episode_steps)),
            replay_buffer_steps=env.spec.max_episode_steps,
            replay_k=replay_k,
            demo_batch_size=128,
            q_filter=q_filter,
            actor_lr=3e-4,
            critic_lr=3e-4,
            observation_normalizer=obs_normalizer,
            observation_clip=obs_clip,
            action_noise=action_noise)
        trainer.initialize_agent(agent_kwargs=agent_kwargs,
                                 demo_path=demo_path)

    agent = trainer.agent
    _LOG.info("Agent Data")
    _LOG.info("  = Train steps: %d", trainer.agent.num_train_steps)
    _LOG.info("  = Replay buffer")
    _LOG.info("    = Episodes: %d", agent.replay_buffer.num_episodes)
    _LOG.info("        = Max: %d", agent.replay_buffer.max_episodes)
    _LOG.info("    = Steps: %d", agent.replay_buffer.count_steps())
    _LOG.info("        = Max: %d", agent.replay_buffer.max_steps)

    _LOG.debug("Actor network\n%s", str(agent.actor))
    _LOG.debug("Critic network\n%s", str(agent.critic))

    _LOG.info("Action space: %s", str(trainer.env.action_space))
    _LOG.info("Observation space: %s", str(trainer.env.observation_space))

    with trainer:
        _run_train(trainer, num_epochs, num_cycles, num_episodes, num_evals,
                   save)

    sys.exit(0)


def _run_train(trainer, num_epochs, num_cycles, num_episodes, num_evals,
               save_path):
    try:
        for epoch in range(1, num_epochs + 1):
            _LOG.info("===== EPOCH: %d/%d", epoch, num_epochs)
            trainer.agent.set_train_mode()
            _run_train_epoch(trainer, epoch, num_cycles,
                             num_episodes, save_path)

            # End episodes
            _LOG.info("----- EVALUATING")
            trainer.agent.set_eval_mode()
            _evaluate(trainer.agent, num_evals, render=False)
        # End epochs
    except KeyboardInterrupt:
        _LOG.warn("Exiting due to keyboard interruption")
    finally:
        _LOG.info("Saving agent before exiting")
        trainer.agent.save(save_path, replay_buffer=True)
        trainer.agent.env.close()


def _run_train_epoch(trainer, epoch, num_cycles, num_episodes, save_path):
    for cycle in range(1, num_cycles + 1):
        cycle_start_time = time.time()
        _LOG.info("----- CYCLE: %d/%d [EPOCH: %d]", cycle, num_cycles, epoch)
        _LOG.info("Running %d episodes", num_episodes)
        trainer.run(num_episodes=num_episodes,
                    train_steps=trainer.agent.env.spec.max_episode_steps)
        _LOG.info("Elapsed: %.2fs", time.time() - cycle_start_time)

        save_start_time = time.time()
        trainer.agent.save(save_path, replay_buffer=True)
        _LOG.info("Agent saved [%.2fs]", time.time() - save_start_time)


###############################################################################

@click.command("her-ddpg-test")
@click.argument("environment", type=str)
@click.argument("agent-path", type=click.Path(exists=True, dir_okay=True))
@click.option("--num-episodes", type=int, default=5)
@click.option("--num-steps", type=int, default=20)
@click.option("--pause/--no-pause", default=False,
              help="Pause (or not) before running an episode.")
@click.option("--seed", type=int, default=1234)
def cli_her_ddpg_test(environment, agent_path, num_episodes, num_steps,
                      pause, seed):
    _LOG.info("Loading %s", environment)
    env = gym.make(environment).unwrapped
    pyrl.cli.util.initialize_seed(seed, env)

    _LOG.info("Loading agent from '%s'", agent_path)
    agent = pyrl.agents.her_ddpg.HERDDPG.load(
        agent_path, env=env, replay_buffer=False)
    agent.set_eval_mode()

    _LOG.info("Action space size: %s", str(env.action_space))
    _LOG.info("Observation space size: %s", str(env.observation_space))

    all_rewards = []
    for episode in range(num_episodes):
        _LOG.info("Running episode %d/%d", episode + 1, num_episodes)
        if pause:
            input("Press enter to start episode")
        rewards, success = _evaluate(agent, env, 1, num_steps, render=True)
        all_rewards.extend(rewards)
    env.close()

    sum_score = sum(x.sum() for x in all_rewards)
    _LOG.info("Avg sum score: %.5f", sum_score / len(all_rewards))
    _LOG.info("Last rewards: %s", ", ".join(str(x[-1]) for x in all_rewards))

    return 0


###############################################################################

def _evaluate(agent, env, num_evals, num_steps, render):
    all_rewards = []
    all_success = []
    for _ in range(num_evals):
        rewards, infos, done = pyrl.cli.util.evaluate(agent, env, num_steps,
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
