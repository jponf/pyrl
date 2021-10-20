# -*- coding: utf-8 -*-

# Standard library
import math
import os
import sys
import time

import six

# SciPy stack
import numpy as np

# OpenAI
import gym

# Click (command line options)
import click

# ...
import pyrl.cli.util
import pyrl.agents
import pyrl.util.logging
import pyrl.trainer


###############################################################################

_LOG = pyrl.util.logging.get_logger()


###############################################################################

@click.command(name="her-td3-train")
@click.argument("environment", type=str)
@click.option("--num-epochs", type=int, show_default=True, default=20)
@click.option("--num-cycles", type=int, show_default=True, default=50)
@click.option("--num-episodes", type=int, show_default=True, default=16)
@click.option("--num-envs", type=int, show_default=True, default=1)
@click.option("--num-evals", type=int, show_default=True, default=1)
@click.option("--num-cpus", type=int, show_default=True, default=1)
@click.option("--demo-path", type=click.Path(exists=True, file_okay=True),
              required=False, help="Path to the file with demonstration runs.")
@click.option("--eps-greedy", type=float, show_default=True, default=0.2)
@click.option("--reward-scale", type=float, show_default=True, default=1.0)
@click.option("--replay-buffer", type=int, show_default=True, default=1000000)
@click.option("--policy-delay", type=int, show_default=True, default=2)
@click.option("--random-steps", type=int, show_default=True, default=1500)
@click.option("--replay-k", type=int, show_default=True, default=4,
              help="The ratio between HER replays and regular replays,"
                   " e.g. k = 4 -> 4 times as many HER replays as regular"
                   " replays.")
@click.option("--q-filter/--no-q-filter", show_default=True, default=False)
@click.option("--action-noise", type=str, show_default=True, default="ou_0.2")
@click.option("--obs-normalizer", type=click.Choice(["none", "standard"]),
               show_default=True, default="standard",
               help="If set to none, the observations won't be normalized")
@click.option("--obs-clip", type=float, show_default=True, default=5.0,
              help="Min/Max. value to clip the observations to if they are"
                   " being normalized.")
@click.option("--render/--no-render", default=False)
@click.option("--load", type=click.Path(exists=True, dir_okay=True))
@click.option("--save", type=click.Path(), show_default=True,
              default="checkpoints/her-td3")
@click.option("--seed", type=int, default=int(time.time()))
def cli_her_td3_train(environment,
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
                      policy_delay,
                      random_steps,
                      replay_k,
                      q_filter,
                      action_noise,
                      obs_normalizer,
                      obs_clip,
                      render,
                      load, save, seed):
    """Trains a HER + TD3 agent on an OpenAI's gym environment."""
    trainer = pyrl.trainer.AgentTrainer(
        agent_cls=pyrl.agents.HerTD3, env_name=environment,
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
            policy_delay=policy_delay,
            random_steps=random_steps,
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
    _LOG.debug("Critic 1 network\n%s", str(agent.critic_1))
    _LOG.debug("Critic 2 network\n%s", str(agent.critic_2))

    _LOG.info("Action space: %s", str(trainer.env.action_space))
    _LOG.info("Observation space: %s", str(trainer.env.observation_space))

    if render:                # Some environments must be rendered
        trainer.env.render()  # before running

    with trainer:
        _run_train(trainer, num_epochs, num_cycles, num_episodes, num_evals,
                   save)

    sys.exit(0)


def _run_train(trainer, num_epochs, num_cycles, num_episodes, num_evals,
               save_path):
    try:
        for epoch in six.moves.range(1, num_epochs + 1):
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
        _LOG.warning("Exiting due to keyboard interruption")
    finally:
        _LOG.info("Saving agent before exiting")
        trainer.agent.save(save_path, replay_buffer=True)
        trainer.agent.env.close()


def _run_train_epoch(trainer, epoch, num_cycles, num_episodes, save_path):
    for cycle in six.moves.range(1, num_cycles + 1):
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

@click.command("her-td3-test")
@click.argument("environment", type=str)
@click.argument("agent-path", type=click.Path(exists=True, dir_okay=True))
@click.option("--num-episodes", type=int, default=5)
@click.option("--pause/--no-pause", default=False,
              help="Pause (or not) before running an episode.")
@click.option("--seed", type=int, default=int(time.time()))
def cli_her_td3_test(environment, agent_path, num_episodes, pause, seed):
    """Runs a previosly trained HER+TD3 agent on a gym environment."""
    _LOG.info("Loading '%s'", environment)
    env = gym.make(environment)
    pyrl.cli.util.initialize_seed(seed)
    env.seed(seed)

    _LOG.info("Loading agent from '%s'", agent_path)
    agent = pyrl.agents.HerTD3.load(agent_path, env,
                                    replay_buffer=False)
    agent.set_eval_mode()

    _LOG.info("Agent trained for %d stes", agent.num_train_steps)
    _LOG.info("Action space size: %s", str(agent.env.action_space))
    _LOG.info("Observation space size: %s", str(agent.env.observation_space))

    all_rewards = []
    all_success = []
    for episode in six.moves.range(num_episodes):
        _LOG.info("Running episode %d/%d", episode + 1, num_episodes)
        if pause:
            input("Press enter to start episode")
        rewards, success = _evaluate(agent, 1, render=True)
        all_rewards.extend(rewards)
        all_success.extend(success)
    env.close()

    sum_score = sum(x.sum() for x in all_rewards)
    sum_success = sum(all_success)
    _LOG.info("Avg sum score: %.5f", sum_score / len(all_rewards))
    # _LOG.info("Last rewards: %s", ", ".join(str(x[-1]) for x in all_rewards))
    _LOG.info("Success %d of %d (%.2f)",
              sum_success, num_episodes, sum_success / num_episodes)

    sys.exit(0)


###############################################################################

def _evaluate(agent, num_evals, render):
    all_rewards = []
    all_success = []

    for _ in six.moves.range(num_evals):
        rewards, infos, _ = pyrl.cli.util.evaluate(
            agent, agent.env, agent.max_episode_steps, render)

        success = any(x["is_success"] for x in infos)
        all_rewards.append(rewards)
        all_success.append(success)

        if success:
            _LOG.info("[SUCCESS]")

        _LOG.info("Last reward: %.5f, Sum reward: %.5f,"
                  " Avg. reward: %.5f, Std. reward: %.5f",
                  rewards[-1], np.sum(rewards),
                  np.mean(rewards), np.std(rewards))

    return all_rewards, all_success
