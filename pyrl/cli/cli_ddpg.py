# -*- coding: utf-8 -*-

import os
import six
import sys
import time
import typer
from pathlib import Path
from typing import Optional

import pyrl.agents
import pyrl.trainer
import pyrl.util.logging
import pyrl.util.ugym
from pyrl.agents.agents_utils import ObservationNormalizer
from pyrl.agents.ddpg import DDPG
from pyrl.cli.common import cli_agent_evaluation, evaluate_agent, initialize_seed
from pyrl.trainer.trainer import AgentTrainer

###############################################################################

app = typer.Typer(
    name="ddpg",
    no_args_is_help=True,
    help="DDPG agent CLI.",
)
_LOG = pyrl.util.logging.get_logger()


###############################################################################


@app.command(name="train", no_args_is_help=True, help="Train a DDPG agent.")
def cli_ddpg_train(
    environment: str = typer.Argument(..., help="Gym's environment name"),
    num_epochs: int = typer.Option(
        default=20,
        help="Number of epochs to train the agent for. After each epoch the"
        + "agent state is saved.",
    ),
    num_episodes: int = typer.Option(
        default=20,
        help="Number of episodes in an epoch.",
    ),
    num_envs: int = typer.Option(
        default=1,
        help="Run the agent in this number of environments on each episode.",
    ),
    num_evals: int = typer.Option(1),
    num_cpus: int = typer.Option(
        default=1,
        help="Number of CPUs avaliable to run environments in parallel.",
    ),
    gamma: float = typer.Option(
        default=0.99,
        min=0.001,
        max=1.0,
        help="Discount factor.",
    ),
    tau: float = typer.Option(0.001, help="Polyak averaging."),
    batch_size: int = typer.Option(
        default=128,
        min=8,
        help="Batch size used when training the agent's neural network.",
    ),
    replay_buffer: int = typer.Option(
        default=1000000,
        min=10000,
        help="Number of transitions to keep on the replay buffer.",
    ),
    reward_scale: float = typer.Option(
        default=1.0,
        help="Factor applied to each reward.",
    ),
    action_noise: str = typer.Option(
        default="ou_0.2",
        help="Action noise, it can be 'none' or <name>_<std>, for example:"
        " ou_0.2 or normal_0.1.",
    ),
    parameter_noise: float = typer.Option(
        default=0.0,
        help="Adaptative parameter noise standard deviation.",
    ),
    obs_normalizer: ObservationNormalizer = typer.Option(
        ObservationNormalizer.STANDARD,
        help="Controls how observations will be normalized. "
        f"{ObservationNormalizer.NONE} disables observaion normalization.",
    ),
    obs_clip: float = typer.Option(
        default=5.0,
        help="Min/Max. value to clip the observations to if they are being normalized.",
    ),
    render: bool = typer.Option(
        default=False,
        help="Render gym's environment while training (slow).",
    ),
    load: Optional[Path] = typer.Option(
        default=None,
        exists=True,
        file_okay=False,
        help="Path to a previously saved DDPG checkpoint to resume training.",
    ),
    save: Path = typer.Option(
        default="checkpoints/ddpg",
        file_okay=False,
        help="Path to save the DDPG agent state.",
    ),
    seed: int = typer.Option(0),
):
    """Trains a DDPG agent on an OpenAI's gym environment."""
    trainer = pyrl.trainer.AgentTrainer(
        agent_cls=DDPG,
        env_name=environment,
        seed=seed,
        num_envs=num_envs,
        num_cpus=num_cpus,
        root_log_dir=os.path.join(save, "log"),
    )
    initialize_seed(seed, trainer.env)

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
                actor_lr=1e-3,
                critic_lr=1e-3,
                observation_normalizer=obs_normalizer,
                observation_clip=obs_clip,
                action_noise=action_noise,
                parameter_noise=parameter_noise,
            ),
        )

    _LOG.info("Agent Data")
    _LOG.info("  = Train steps: %d", trainer.agent.num_train_steps)
    _LOG.info("  = Replay buffer: %d", len(trainer.agent.replay_buffer))
    _LOG.info("    = Max. Size: %d", trainer.agent.replay_buffer.max_size)

    _LOG.debug("Actor network\n%s", str(trainer.agent.actor))
    _LOG.debug("Critic network\n%s", str(trainer.agent.critic))

    _LOG.info("Action space: %s", str(trainer.env.action_space))
    _LOG.info("Observation space: %s", str(trainer.env.observation_space))

    if render:  # Some environments must be rendered
        trainer.env.render()  # before running

    with trainer:
        _run_train(trainer, num_epochs, num_episodes, num_evals, save)

    sys.exit(0)


def _run_train(
    trainer: AgentTrainer,
    num_epochs: int,
    num_episodes: int,
    num_evals: int,
    save_path: Path,
):
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
            evaluate_agent(
                trainer.agent,
                trainer.env,
                num_evals,
                render=False,
                is_her=False,
            )
    except KeyboardInterrupt:
        _LOG.warn("Exiting due to keyboard interruption")
    finally:
        _LOG.info("Saving agent before exiting")
        trainer.agent.save(save_path, replay_buffer=True)
        trainer.env.close()


def _run_train_epoch(trainer: AgentTrainer, epoch: int, num_episodes: int):
    for episode in six.moves.range(1, num_episodes + 1):
        episode_start_time = time.time()
        _LOG.info("----- EPISODE: %d/%d [EPOCH: %d]", episode, num_episodes, epoch)
        trainer.run(num_episodes=1, train_steps=0)
        _LOG.info("Elapsed: %.2fs", time.time() - episode_start_time)


###############################################################################


@app.command("test", no_args_is_help=True, help="Test a DDPG agent.")
def cli_ddpg_test(
    environment: str = typer.Argument(..., help="Gym's environment name."),
    agent_path: Path = typer.Argument(
        default=...,
        exists=True,
        file_okay=False,
        help="Path to a previously saved DDPG agent checkpoint.",
    ),
    pause: bool = typer.Option(
        default=False,
        help="Whether the program should pause before running an episode.",
    ),
    num_episodes: int = typer.Option(5, help="Number of episodes to run."),
    seed: int = typer.Option(0),
):
    """Runs a previously trained DDPG agent on an OpenAI's gym environment."""
    _LOG.info("Loading '%s'", environment)
    env = pyrl.util.ugym.make_flat(environment)
    initialize_seed(seed, env)

    _LOG.info("Loading agent from %s", agent_path)
    agent = DDPG.load(agent_path, replay_buffer=False)

    cli_agent_evaluation(
        agent,
        env,
        num_episodes,
        pause=pause,
        is_her=False,
    )
    sys.exit(0)
