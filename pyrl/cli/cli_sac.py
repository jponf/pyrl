# -*- coding: utf-8 -*-

import numpy as np
import os
import six
import sys
import time
import typer
from pathlib import Path

import pyrl.agents
import pyrl.cli.util
import pyrl.trainer
import pyrl.util.logging
import pyrl.util.ugym
from pyrl.agents.agents_utils import ObservationNormalizer

###############################################################################

app = typer.Typer(
    name="sac",
    no_args_is_help=True,
    help="SAC agent CLI.",
)
_LOG = pyrl.util.logging.get_logger()


###############################################################################


@app.command(name="train", no_args_is_help=True, help="Train a SAC agent.")
def cli_sac_train(
    environment: str = typer.Argument(None, help="Gym's environment name"),
    num_epochs: int = typer.Option(
        20,
        help="Number of epochs to train the agent for. After each epoch the"
        + "agent state is saved",
    ),
    num_episodes: int = typer.Option(
        20,
        help="Number of episodes in an epoch",
    ),
    num_envs: int = typer.Option(
        1,
        help="Run the agent in this number of environments on each episode",
    ),
    num_evals: int = typer.Option(1),
    num_cpus: int = typer.Option(
        1,
        help="Number of CPUs avaliable to run environments in parallel",
    ),
    gamma: float = typer.Option(
        0.99,
        min=0.001,
        max=1.0,
        help="Discount factor",
    ),
    tau: float = typer.Option(0.001, help="Polyak averaging"),
    batch_size: int = typer.Option(
        default=128,
        min=8,
        help="Batch size used when training the agent's neural network",
    ),
    replay_buffer: int = typer.Option(
        default=1000000,
        min=10000,
        help="Number of transitions to keep on the replay buffer",
    ),
    reward_scale: float = typer.Option(
        default=1.0,
        help="Factor applied to each reward.",
    ),
    random_steps: int = typer.Option(
        default=1500,
        help="Number of steps taken completely at random before using the "
        + "actor's action + noise approach",
    ),
    obs_normalizer: ObservationNormalizer = typer.Option(
        ObservationNormalizer.STANDARD,
        help="Controls how observations will be normalized. "
        f"{ObservationNormalizer.NONE} disables observaion normalization",
    ),
    obs_clip: float = typer.Option(
        default=5.0,
        help="Min/Max. value to clip the observations to if they are being normalized",
    ),
    render: bool = typer.Option(
        default=False,
        help="Render gym's environment while training (slow)",
    ),
    load: Path = typer.Option(
        default=None,
        exists=True,
        file_okay=False,
        help="Path to a previously saved DDPG checkpoint to resume training",
    ),
    save: Path = typer.Option(
        default="checkpoints/ddpg",
        file_okay=False,
        help="Path to save the DDPG agent state",
    ),
    seed: int = typer.Option(0),
):
    """Trains a TD3 agent on an OpenAI's gym environment."""
    trainer = pyrl.trainer.AgentTrainer(
        agent_cls=pyrl.agents.SAC,
        env_name=environment,
        seed=seed,
        num_envs=num_envs,
        num_cpus=num_cpus,
        root_log_dir=os.path.join(save, "log"),
    )
    pyrl.cli.util.initialize_seed(seed)
    trainer.env.seed(seed)

    if load:
        _LOG.info("Loading agent")
        trainer.initialize_agent(agent_path=load)
    else:
        _LOG.info("Initializing new agent")
        trainer.initialize_agent(
            agent_kwargs=dict(
                gamma=gamma,
                tau=tau,
                batch_size=batch_size,
                reward_scale=reward_scale,
                replay_buffer_size=replay_buffer,
                random_steps=random_steps,
                actor_lr=1e-3,
                critic_lr=1e-3,
                observation_normalizer=obs_normalizer,
                observation_clip=obs_clip,
            ),
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

    if render:  # Some environments must be rendered
        trainer.env.render()  # before running

    with trainer:
        _run_train(trainer, num_epochs, num_episodes, num_evals, save)

    sys.exit(0)


def _run_train(trainer, num_epochs, num_episodes, num_evals, save_path):
    try:
        for epoch in six.moves.range(1, num_epochs + 1):
            _LOG.info("===== EPOCH: %d/%d", epoch, num_epochs)
            trainer.agent.set_train_mode()
            _run_train_epoch(trainer, epoch, num_episodes)

            save_start_time = time.time()
            trainer.agent.save(save_path, replay_buffer=True)
            _LOG.info("Agent saved [%.2fs]", time.time() - save_start_time)

            # End episodes
            _LOG.info("----- EVALUATING")
            trainer.agent.set_eval_mode()
            _evaluate(trainer.agent, trainer.env, num_evals, render=False)
        # End epochs
    except KeyboardInterrupt:
        _LOG.warning("Exiting due to keyboard interruption")
    finally:
        _LOG.info("Saving agent before exiting")
        trainer.agent.save(save_path, replay_buffer=True)
        trainer.env.close()


def _run_train_epoch(trainer, epoch, num_episodes):
    for episode in six.moves.range(1, num_episodes + 1):
        episode_start_time = time.time()
        _LOG.info("----- EPISODE: %d/%d [EPOCH: %d]", episode, num_episodes, epoch)
        trainer.run(num_episodes=1, train_steps=0)
        _LOG.info("Elapsed: %.2fs", time.time() - episode_start_time)


###############################################################################


@app.command("test", help="Test a SAC agent.")
def cli_sac_test(
    environment: str = typer.Argument(None, help="Gym's environment name"),
    agent_path: Path = typer.Argument(
        default=None,
        exists=True,
        file_okay=False,
        help="Path to a previously saved SAC agent checkpoint.",
    ),
    num_episodes: int = typer.Option(5, help="Number of episodes to run"),
    seed: int = typer.Option(0),
):
    """Runs a previosly trained TD3 agent on an OpenAI's gym environment."""
    _LOG.info("Loading '%s'", environment)
    env = pyrl.util.ugym.make_flat(environment)
    pyrl.cli.util.initialize_seed(seed)
    env.seed(seed)

    _LOG.info("Loading agent from %s", agent_path)
    agent = pyrl.agents.SAC.load(agent_path, replay_buffer=False)
    agent.set_eval_mode()

    _LOG.info("Agent trained for %d stes", agent.num_train_steps)
    _LOG.info("Action space size: %s", str(agent.action_space))
    _LOG.info("Observation space size: %s", str(agent.observation_space))

    env.render()  # Some environments must be rendered before running
    all_rewards = []
    for episode in six.moves.range(num_episodes):
        _LOG.info("Running episode %d/%d", episode + 1, num_episodes)
        rewards = _evaluate(agent, env, num_evals=1, render=True)
        all_rewards.extend(rewards)

    env.close()
    sum_score = sum(x.sum() for x in all_rewards)
    _LOG.info(
        "Average sum score over %d runs: %d",
        len(all_rewards),
        sum_score / len(all_rewards),
    )

    sys.exit(0)


###############################################################################


def _evaluate(agent, env, num_evals, render):
    all_rewards = []

    for _ in six.moves.range(num_evals):
        rewards, _, done = pyrl.cli.util.evaluate(
            agent,
            env,
            env.spec.max_episode_steps,
            render,
        )

        all_rewards.append(rewards)
        if done:
            _LOG.info("[DONE]")

        _LOG.info(
            "Last reward: %.5f, Sum reward: %.5f,"
            " Avg. reward: %.5f, Std. reward: %.5f",
            rewards[-1],
            np.sum(rewards),
            np.mean(rewards),
            np.std(rewards),
        )

    return all_rewards