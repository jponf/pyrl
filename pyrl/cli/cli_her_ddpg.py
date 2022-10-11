# -*- coding: utf-8 -*-

import gym
import math
import os
import six
import sys
import time
import typer
from pathlib import Path
from typing import Optional

import pyrl.trainer
import pyrl.util.logging
from pyrl.agents.agents_utils import ObservationNormalizer
from pyrl.agents.her_ddpg import HerDDPG
from pyrl.cli.common import cli_agent_evaluation, evaluate_agent, initialize_seed

###############################################################################

app = typer.Typer(
    name="her-ddpg",
    no_args_is_help=True,
    help="HER+DDPG agent CLI.",
)
_LOG = pyrl.util.logging.get_logger()


###############################################################################


@app.command(name="train", no_args_is_help=True, help="Train a HER+DDPG agent")
def cli_her_ddpg_train(
    environment: str = typer.Argument(..., help="Gym's environment name."),
    num_epochs: int = typer.Option(
        default=20,
        help="Number of epochs to train the agent for. After each epoch the"
        + "agent state is saved.",
    ),
    num_cycles: int = typer.Option(
        default=50,
        help="Number of cycles in an epoch.",
    ),
    num_episodes: int = typer.Option(
        default=16,
        help="Number of episodes in a cycle.",
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
    demo_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        help="Path to the file with demonstration runs.",
    ),
    eps_greedy: float = typer.Option(
        default=0.2,
        help="Probability of picking a random action.",
    ),
    replay_buffer: int = typer.Option(
        default=1000000,
        min=10000,
        help="Number of transitions to keep on the replay buffer.",
    ),
    replay_k: int = typer.Option(
        default=4,
        help="The ratio between HER replays and regular replays,"
        + " e.g. k = 4 -> 4 times as many HER replays as regular replays.",
    ),
    q_filter: bool = typer.Option(
        default=False,
        help="Apply cloning loss only to states where the critic determines"
        + " that the demonstrator is better than the actor action.",
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
        help="Path to a previously saved HER+DDPG checkpoint to resume training.",
    ),
    save: Path = typer.Option(
        default="checkpoints/her-ddpg",
        file_okay=False,
        help="Path to save the HER+DDPG agent state.",
    ),
    seed: int = typer.Option(0),
):
    """Trains a HER + DDPG agent on an OpenAI's gym environment."""
    trainer = pyrl.trainer.AgentTrainer(
        agent_cls=HerDDPG,
        env_name=environment,
        seed=seed,
        num_envs=num_envs,
        num_cpus=num_cpus,
        root_log_dir=os.path.join(save, "log"),
    )
    initialize_seed(seed, trainer.env)

    if load:
        _LOG.info("Loading previously trained agent")
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
            replay_buffer_episodes=int(
                math.ceil(replay_buffer / env.spec.max_episode_steps),
            ),
            replay_buffer_steps=env.spec.max_episode_steps,
            replay_k=replay_k,
            demo_batch_size=128,
            q_filter=q_filter,
            actor_lr=3e-4,
            critic_lr=3e-4,
            observation_normalizer=obs_normalizer,
            observation_clip=obs_clip,
            action_noise=action_noise,
        )
        trainer.initialize_agent(agent_kwargs=agent_kwargs, demo_path=demo_path)

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

    if render:  # Some environments must be rendered
        trainer.env.render()  # before running

    with trainer:
        _run_train(trainer, num_epochs, num_cycles, num_episodes, num_evals, save)

    sys.exit(0)


def _run_train(trainer, num_epochs, num_cycles, num_episodes, num_evals, save_path):
    try:
        for epoch in six.moves.range(1, num_epochs + 1):
            _LOG.info("===== EPOCH: %d/%d", epoch, num_epochs)
            trainer.agent.set_train_mode()
            _run_train_epoch(trainer, epoch, num_cycles, num_episodes, save_path)

            # End episodes
            _LOG.info("----- EVALUATING")
            trainer.agent.set_eval_mode()
            evaluate_agent(
                trainer.agent,
                trainer.env,
                num_evals,
                render=False,
                is_her=True,
            )
    except KeyboardInterrupt:
        _LOG.warn("Exiting due to keyboard interruption")
    finally:
        _LOG.info("Saving agent before exiting")
        trainer.agent.save(save_path, replay_buffer=True)
        trainer.agent.env.close()


def _run_train_epoch(trainer, epoch, num_cycles, num_episodes, save_path):
    for cycle in six.moves.range(1, num_cycles + 1):
        cycle_start_time = time.time()
        _LOG.info("----- CYCLE: %d/%d [EPOCH: %d]", cycle, num_cycles, epoch)
        _LOG.info("Running %d episodes", num_episodes)
        trainer.run(
            num_episodes=num_episodes,
            train_steps=trainer.agent.env.spec.max_episode_steps,
        )
        _LOG.info("Elapsed: %.2fs", time.time() - cycle_start_time)

        save_start_time = time.time()
        trainer.agent.save(save_path, replay_buffer=True)
        _LOG.info("Agent saved [%.2fs]", time.time() - save_start_time)


###############################################################################


@app.command("test", no_args_is_help=True, help="Test a HER+DDPG agent")
def cli_her_ddpg_test(
    environment: str = typer.Argument(..., help="Gym's environment name."),
    agent_path: Path = typer.Argument(
        default=...,
        exists=True,
        file_okay=False,
        help="Path to a previously saved DDPG agent checkpoint.",
    ),
    num_episodes: int = typer.Option(5, help="Number of episodes to run"),
    pause: bool = typer.Option(
        default=False,
        help="Whether the program should pause before running an episode.",
    ),
    seed: int = typer.Option(0),
):
    """Runs a previosly trained HER+TD3 agent on a gym environment."""
    _LOG.info("Loading '%s'", environment)
    env = gym.make(environment)
    initialize_seed(seed, env)

    _LOG.info("Loading agent from '%s'", agent_path)
    agent = HerDDPG.load(agent_path, env, replay_buffer=False)

    cli_agent_evaluation(
        agent,
        env,
        num_episodes,
        pause=pause,
        is_her=True,
    )
    sys.exit(0)
