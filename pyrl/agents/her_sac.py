# -*- coding: utf-8 -*-

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

"""
Agent that implements the Hindsight Experience Replay combined with
the Soft Actor-Critic algorithm.
"""

import collections
import errno
import os
import pickle

# SciPy
import numpy as np

# PyTorch
import torch
import torch.optim as optim
import torch.nn.functional as F

# pyrl
import pyrl.util.logging
import pyrl.util.umath as umath
import pyrl.util.ugym

from pyrl.models.models_utils import soft_update
from .agents_utils import create_normalizer, dicts_mean, load_her_demonstrations
from .core import HerAgent
from .replay_buffer import HerBatch, HerReplayBuffer
from .sac import build_sac_ac


###############################################################################

_DEVICE = "cpu"
_LOG = pyrl.util.logging.get_logger()


###############################################################################


class HerSAC(HerAgent):
    """Hindsight Experience Replay Agent that uses Soft Actor Critic (SAC) as
    the off-policy RL algorithm.

    Introduced in the papers:
        HER: Hindsight Experience Replay.
        SAC: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
             Stochastic Actor.
    """

    def __init__(
        self,
        env,
        alpha=0.2,
        eps_greedy=0.2,
        gamma=0.95,
        tau=0.005,
        batch_size=128,
        reward_scale=1.0,
        replay_buffer_episodes=10000,
        replay_buffer_steps=100,
        random_steps=1000,
        replay_k=2,
        demo_batch_size=128,
        q_filter=False,
        action_penalty=1.0,
        prm_loss_weight=0.001,
        aux_loss_weight=None,
        actor_cls=None,
        actor_kwargs=None,
        actor_lr=0.001,
        critic_cls=None,
        critic_kwargs=None,
        critic_lr=0.001,
        tune_alpha=True,
        observation_normalizer="none",
        observation_clip=float("inf"),
    ):
        """
        :param env: OpenAI's GoalEnv instance.
        :param alpha: Initial entropy coefficient value.
        :param float eps_greedy: Probability of picking a random action
            in training mode.
        :param gamma: Bellman's discount rate.
        :type gamma: float
        :param tau: Used to perform "soft" updates (polyak averaging) of the
            weights from the actor/critic to their "target" counterparts.
        :type tau: float
        :param batch_size: Size of the sample used to train the actor and
            critic at each timestep.
        :type batch_size: int
        :param replay_buffer_episodes: Number of episodes to store in the
            replay buffer.
        :type replay_buffer_episodes: int
        :param replay_buffer_steps: Maximum number of steps per episode.
        :type replay_buffer_steps: int
        :param random_steps: Number of steps taken completely at random while
            training before using the actor action + noise.
        :type random_steps: int
                :param replay_k: Ratio between HER replays and regular replays,
            e.g: k = 4 -> 4 times as many HER replays as regular replays.
        :type replay_k: float
        :param demo_batch_size: Additional elements sampled from the demo
            replay buffer to train the actor and critic.
        type demo_batch_size: int
        :param action_penalty: Quadratic penalty on actions to avoid
            tanh saturation and vanishing gradients (0 disables the penalty).
        :type action_penalty: float
        :param prm_loss_weight: Weight corresponding to the primary loss.
        :type prm_loss_weight: float
        :param float aux_loss_weight: Weight corresponding to the auxiliary
            loss, also called the cloning loss.
        :type aux_loss_weight: float
        :param actor_cls: Actor network class.
        :type actor_cls: type
        :param actor_kwargs: Arguments to initialize the actor network.
        :type actor_kwargs: dict
        :param actor_lr: Learning rate for the actor network.
        :type actor_lr: float
        :param critic_cls: Critic network class.
        :type critic_cls: type
        :param actor_kwargs: Arguments to initialize the critic network.
        :type actor_kwargs: dict
        :param critic_lr: Learning rate for the critic network.
        :type critic_lr: float
        :param observation_normalizer: Normalize the environment observations
            according to a normalizer. observation_normalizer can either be
            "none" or "standard".
        :type observation_normalizer: str
        :param observation_clip: Range of the observations after being
            normalized. This parameter will only take effect when normalizer
            is not set to "none".
        :type observation_clip: float
        """
        super(HerSAC, self).__init__(env)
        self.eps_greedy = eps_greedy
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.replay_buffer = HerReplayBuffer(
            obs_shape=self.env.observation_space["observation"].shape,
            goal_shape=self.env.observation_space["desired_goal"].shape,
            action_shape=self.env.action_space.shape,
            max_episodes=replay_buffer_episodes,
            max_steps=replay_buffer_steps,
        )

        self.random_steps = random_steps

        self._replay_k = replay_k
        self._demo_batch_size = demo_batch_size
        self._q_filter = q_filter
        self._action_penalty = action_penalty
        self._prm_loss_weight = prm_loss_weight
        if aux_loss_weight is None:
            self._aux_loss_weight = 1.0 / demo_batch_size
        else:
            self._aux_loss_weight = aux_loss_weight

        # Build model (AC architecture)
        (
            self.actor,
            (self.critic_1, self.target_critic_1),
            (self.critic_2, self.target_critic_2),
        ) = build_sac_ac(
            self.env.observation_space,
            self.env.action_space,
            actor_cls,
            actor_kwargs,
            critic_cls,
            critic_kwargs,
        )

        self._actor_kwargs = actor_kwargs
        self._actor_lr = actor_lr
        self._critic_kwargs = critic_kwargs
        self._critic_lr = critic_lr

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # Autotune alpha
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
        self._log_alpha = torch.as_tensor(alpha, dtype=torch.float32).log()
        if tune_alpha:
            self._log_alpha.requires_grad_()
            self._alpha_optim = optim.Adam([self._log_alpha], lr=critic_lr)
        else:
            self._alpha_optim = None

        # Normalizer
        self._obs_normalizer_arg = observation_normalizer
        self.obs_normalizer = create_normalizer(
            observation_normalizer,
            self.env.observation_space["observation"].shape,
            clip_range=observation_clip,
        )
        self.goal_normalizer = create_normalizer(
            observation_normalizer,
            self.env.observation_space["desired_goal"].shape,
            clip_range=observation_clip,
        )

        # Demonstration replay buffer
        self._demo_replay_buffer = None

        # Other attributes
        self._total_steps = 0

    @property
    def alpha(self):
        """Relative importance of the entropy term against the reward."""
        with torch.no_grad():
            return self._log_alpha.exp()

    # HerAgent methods
    ##########################

    def load_demonstrations(self, demo_path):
        buffer = load_her_demonstrations(
            demo_path,
            env=self.env,
            action_fn=self._to_actor_space,
            max_steps=self.replay_buffer.max_steps,
        )

        _LOG.debug("(HER-SAC) Loaded Demonstrations")
        _LOG.debug("(HER-SAC)     = Num. Episodes: %d", buffer.num_episodes)
        _LOG.debug("(HER-SAC)     = Num. Steps: %d", buffer.count_steps())

        if buffer.count_steps() < self._demo_batch_size:
            raise ValueError(
                "demonstrations replay buffer has less than" " `demo_batch_size` steps"
            )

        self._demo_replay_buffer = buffer

    # BaseAgent methods
    ##########################

    def set_train_mode(self, mode=True):
        super(HerSAC, self).set_train_mode(mode)
        self.actor.train(mode=mode)
        self.critic_1.train(mode=mode)
        self.critic_2.train(mode=mode)

    def end_episode(self):
        super(HerSAC, self).end_episode()
        self.replay_buffer.save_episode()

    def update(self, state, action, reward, next_state, terminal):
        self._total_steps += 1

        self.obs_normalizer.update(state["observation"])
        self.goal_normalizer.update(state["desired_goal"])

        self.replay_buffer.add(
            obs=state["observation"],
            action=self._to_actor_space(action),
            next_obs=next_state["observation"],
            reward=reward,
            terminal=terminal,
            goal=next_state["desired_goal"],
            achieved_goal=next_state["achieved_goal"],
        )

    @torch.no_grad()
    def compute_action(self, state):
        # Random exploration
        if self._train_mode and (
            self._total_steps < self.random_steps
            or np.random.random_sample() < self.eps_greedy
        ):
            return self.env.action_space.sample()

        # Pre-process
        obs = torch.from_numpy(state["observation"]).float()
        obs = self.obs_normalizer.transform(obs).unsqueeze_(0).to(_DEVICE)
        goal = torch.from_numpy(state["desired_goal"]).float()
        goal = self.goal_normalizer.transform(goal).unsqueeze_(0).to(_DEVICE)

        # Compute action
        action, _ = self.actor(obs, goal, deterministic=not self._train_mode)
        action = action.squeeze_(0).cpu().numpy()

        return self._to_action_space(action)

    def train(self, steps, progress=False):
        if self.replay_buffer.count_steps() >= self.batch_size:
            super(HerSAC, self).train(steps, progress)

    def _train(self):
        batch = self._sample_batch()
        (obs, action, next_obs, reward, terminal, goal, _) = batch

        self._train_critic(obs, action, next_obs, goal, reward, terminal)
        self._train_policy(obs, action, goal)
        self._train_alpha(obs, goal)
        self._update_target_networks()

    def _train_critic(self, obs, action, next_obs, goal, reward, terminal):
        with torch.no_grad():
            next_action, next_log_pi = self.actor(next_obs, goal)
            next_log_pi[self.batch_size :] = 0.0

            next_q1 = self.target_critic_1(next_obs, goal, next_action)
            next_q2 = self.target_critic_2(next_obs, goal, next_action)

            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
            next_q *= (1 - terminal.int()) * self.gamma
            next_q += self.reward_scale * reward

        # Optimize critics
        curr_q1 = self.critic_1(obs, goal, action)
        loss_q1 = F.smooth_l1_loss(curr_q1, next_q)
        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_1_optimizer.step()

        curr_q2 = self.critic_2(obs, goal, action)
        loss_q2 = F.smooth_l1_loss(curr_q2, next_q)
        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_2_optimizer.step()

        with torch.no_grad():
            self._summary.add_scalars(
                "Q",
                {
                    "Mean_Q1": curr_q1.mean(),
                    "Mean_Q2": curr_q2.mean(),
                    "Mean_Target": next_q.mean(),
                },
                self._train_steps,
            )
            self._summary.add_scalar("Loss/Q1", loss_q1, self._train_steps)
            self._summary.add_scalar("Loss/Q2", loss_q2, self._train_steps)

    def _train_policy(self, obs, action, goal):
        actor_out, log_pi = self.actor(obs, goal)
        log_pi[self.batch_size :] = 0.0

        min_q = torch.min(
            self.critic_1(obs, goal, actor_out), self.critic_2(obs, goal, actor_out)
        )

        # Jπ = 𝔼st∼D,εt∼N[α * log π(f(εt;st)|st) − Q(st,f(εt;st))]
        pi_loss = (self.alpha * log_pi - min_q).mean()
        pi_loss += self._action_penalty * actor_out.pow(2).mean()
        if action.shape[0] > self.batch_size:  # has demo
            cloning_loss = actor_out[self.batch_size :] - action[self.batch_size :]
            if self._q_filter:
                cur_min_q = torch.min(
                    self.critic_1(
                        obs[self.batch_size :],
                        goal[self.batch_size :],
                        action[self.batch_size :],
                    ),
                    self.critic_2(
                        obs[self.batch_size :],
                        goal[self.batch_size :],
                        action[self.batch_size :],
                    ),
                )

                q_mask = (cur_min_q > min_q[self.batch_size :]).flatten()
                cloning_loss = cloning_loss[q_mask]

            prm_loss = self._prm_loss_weight * pi_loss
            aux_loss = self._aux_loss_weight * cloning_loss.pow(2).sum()
            pi_loss = prm_loss + aux_loss

            self._summary.add_scalars(
                "Loss",
                {"Actor_PRM": prm_loss, "Actor_AUX": aux_loss},
                self._train_steps,
            )

        self.actor_optimizer.zero_grad()
        pi_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        with torch.no_grad():
            self._summary.add_scalar("Loss/Policy", pi_loss, self._train_steps)
            self._summary.add_scalar(
                "Stats/LogProb", log_pi[: self.batch_size].mean(), self._train_steps
            )
            self._summary.add_scalar("Stats/Alpha", self.alpha, self._train_steps)

    def _train_alpha(self, obs, goal):
        if self._alpha_optim is not None:
            _, log_pi = self.actor(obs[: self.batch_size], goal[: self.batch_size])
            alpha_loss = (
                self._log_alpha * (-log_pi - self.target_entropy).detach()
            ).mean()

            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._summary.add_scalar(
                "Loss/Alpha", alpha_loss.detach(), self._train_steps
            )

    def _update_target_networks(self):
        soft_update(self.critic_1, self.target_critic_1, self.tau)
        soft_update(self.critic_2, self.target_critic_2, self.tau)

    def _sample_batch(self):
        def _sample_reward_fn(achieved_goals, goals):
            return self.env.compute_reward(achieved_goals, goals, None)

        has_demo = self._demo_replay_buffer is not None and self._demo_batch_size > 0

        batch = self.replay_buffer.sample_batch_torch(
            sample_size=self.batch_size,
            replay_k=self._replay_k,
            reward_fn=_sample_reward_fn,
            device=_DEVICE,
        )

        if has_demo:
            demo_batch = self._demo_replay_buffer.sample_batch_torch(
                sample_size=self._demo_batch_size,
                replay_k=0,
                reward_fn=_sample_reward_fn,
                device=_DEVICE,
            )
            batch = tuple(torch.cat((x, y), dim=0) for x, y in zip(batch, demo_batch))

        return self._normalize_batch(batch)

    def _normalize_batch(self, batch):
        (obs, action, next_obs, reward, terminal, goal, achieved_goal) = batch
        return HerBatch(
            obs=self.obs_normalizer.transform(obs),
            action=action,
            next_obs=self.obs_normalizer.transform(next_obs),
            reward=reward,
            terminal=terminal,
            goal=self.goal_normalizer.transform(goal),
            achieved_goal=self.goal_normalizer.transform(achieved_goal),
        )

    # Agent State
    ########################

    def state_dict(self):
        state = {
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "actor": self.actor.state_dict(),
            "log_alpha": self._log_alpha,
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "goal_normalizer": self.goal_normalizer.state_dict(),
            "train_steps": self._train_steps,
            "total_steps": self._total_steps,
        }

        return state

    def load_state_dict(self, state):
        self.critic_1.load_state_dict(state["critic1"])
        self.target_critic_1.load_state_dict(state["critic1"])
        self.critic_2.load_state_dict(state["critic2"])
        self.target_critic_2.load_state_dict(state["critic2"])
        self.actor.load_state_dict(state["actor"])

        with torch.no_grad():
            self._log_alpha.copy_(state["log_alpha"])

        self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        self.goal_normalizer.load_state_dict(state["goal_normalizer"])

        self._train_steps = state["train_steps"]
        self._total_steps = state["total_steps"]

    def aggregate_state_dicts(self, states):
        critic_1_state = dicts_mean([x["critic1"] for x in states])
        self.critic_1.load_state_dict(critic_1_state)
        self.target_critic_1.load_state_dict(critic_1_state)

        critic_2_state = dicts_mean([x["critic2"] for x in states])
        self.critic_2.load_state_dict(critic_2_state)
        self.target_critic_2.load_state_dict(critic_2_state)

        actor_state = dicts_mean([x["actor"] for x in states])
        self.actor.load_state_dict(actor_state)

        with torch.no_grad():
            self._log_alpha.copy_(sum(x["log_alpha"] for x in states))
            self._log_alpha.div_(len(states))

        self.obs_normalizer.load_state_dict([x["obs_normalizer"] for x in states])
        self.goal_normalizer.load_state_dict([x["goal_normalizer"] for x in states])

        self._train_steps = max(x["train_steps"] for x in states)
        self._total_steps = max(x["total_steps"] for x in states)

    def save(self, path, replay_buffer=True):
        try:
            os.makedirs(path)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

        args = collections.OrderedDict(
            [
                ("alpha", self.alpha.item()),
                ("eps_greedy", self.eps_greedy),
                ("gamma", self.gamma),
                ("tau", self.tau),
                ("batch_size", self.batch_size),
                ("reward_scale", self.reward_scale),
                ("replay_buffer_episodes", self.replay_buffer.max_episodes),
                ("replay_buffer_steps", self.replay_buffer.max_steps),
                ("random_steps", self.random_steps),
                # HER args
                ("replay_k", self._replay_k),
                ("demo_batch_size", self._demo_batch_size),
                ("q_filter", self._q_filter),
                ("action_penalty", self._action_penalty),
                ("prm_loss_weight", self._prm_loss_weight),
                ("aux_loss_weight", self._aux_loss_weight),
                # Actor-Critic
                ("actor_cls", type(self.actor)),
                ("actor_kwargs", self._actor_kwargs),
                ("actor_lr", self._actor_lr),
                ("critic_cls", type(self.critic_1)),
                ("critic_kwargs", self._critic_kwargs),
                ("critic_lr", self._critic_lr),
                ("tune_alpha", self._alpha_optim is not None),
                # Normalize
                ("observation_normalizer", self._obs_normalizer_arg),
                ("observation_clip", self.obs_normalizer.clip_range),
            ]
        )
        pickle.dump(args, open(os.path.join(path, "args.pkl"), "wb"))

        state = self.state_dict()
        pickle.dump(state, open(os.path.join(path, "state.pkl"), "wb"))

        if replay_buffer:
            self.replay_buffer.save(os.path.join(path, "replay_buffer.h5"))

    @classmethod
    def load(cls, path, env, *args, replay_buffer=True, **kwargs):
        if not os.path.isdir(path):
            raise ValueError("{} is not a directory".format(path))

        # Load and Override arguments used to build the instance
        with open(os.path.join(path, "args.pkl"), "rb") as rfh:
            _LOG.debug("(TD3) Loading agent arguments")
            args_values = pickle.load(rfh)
            args_values.update(kwargs)

            fmt_string = "    {{:>{}}}: {{}}".format(
                max(len(x) for x in args_values.keys())
            )
            for key, value in args_values.items():
                _LOG.debug(fmt_string.format(key, value))

        # Create instance and load the rest of the data
        instance = cls(env, **args_values)

        with open(os.path.join(path, "state.pkl"), "rb") as rfh:
            _LOG.debug("(TD3) Loading agent state")
            state = pickle.load(rfh)
            instance.load_state_dict(state)

        replay_buffer_path = os.path.join(path, "replay_buffer.h5")
        if replay_buffer and os.path.isfile(replay_buffer_path):
            _LOG.debug("(TD3) Loading replay buffer")
            instance.replay_buffer.load(replay_buffer_path)

        return instance

    # Utilities
    ########################

    def _to_actor_space(self, action):
        return umath.scale(
            x=action,
            min_x=self.env.action_space.low,
            max_x=self.env.action_space.high,
            min_out=self.actor.action_space.low,
            max_out=self.actor.action_space.high,
        )

    def _to_action_space(self, action):
        return umath.scale(
            x=action,
            min_x=self.actor.action_space.low,
            max_x=self.actor.action_space.high,
            min_out=self.env.action_space.low,
            max_out=self.env.action_space.high,
        )
