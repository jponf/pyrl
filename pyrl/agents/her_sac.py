# -*- coding: utf-8 -*-

import collections
import errno
import os
import pickle

# Scipy
import numpy as np

# Torch
import torch
import torch.optim as optim
import torch.nn.functional as F

# robotrl
import pyrl.util.logging
import pyrl.util.umath as umath
import pyrl.util.ugym

from .core import HerAgent
from .models_utils import soft_update
from .noise import NormalActionNoise
from .replay_buffer import HerReplayBuffer
from .utils import (create_action_noise, create_normalizer,
                    create_actor, create_critic, dicts_mean)

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

    def __init__(self,
                 env,
                 alpha=0.2,
                 gamma=.95,
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
                 observation_clip=float('inf')):
        """
        :param env: OpenAI's GoalEnv instance.
        :param alpha: Entropy coefficient, if None this value is auto-tuned.
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

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.replay_buffer = HerReplayBuffer(
            obs_shape=self.env.observation_space["observation"].shape,
            goal_shape=self.env.observation_space["desired_goal"].shape,
            action_shape=self.env.action_space.shape,
            max_episodes=replay_buffer_episodes,
            max_steps=replay_buffer_steps)

        self.random_steps = random_steps

        self._replay_k = replay_k
        self._demo_batch_size = demo_batch_size
        self._q_filter = True
        self._action_penalty = action_penalty
        self._prm_loss_weight = prm_loss_weight
        if aux_loss_weight is None:
            self._aux_loss_weight = 1.0 / demo_batch_size
        else:
            self._aux_loss_weight = aux_loss_weight

        # Build model (AC architecture)
        actors, critics_1, critics_2 = _build_ac(self.observation_space,
                                                 self.action_space,
                                                 actor_cls, actor_kwargs,
                                                 critic_cls, critic_kwargs)

        self.actor, self.target_actor = actors
        self.critic_1, self.target_critic_1 = critics_1
        self.critic_2, self.target_critic_2 = critics_2

        self._actor_kwargs = actor_kwargs
        self._actor_lr = actor_lr
        self._critic_kwargs = critic_kwargs
        self._critic_lr = critic_lr

        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),
                                             lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),
                                             lr=critic_lr)

        # Autotune alpha
        self.target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape)).item()
        self._log_alpha = torch.as_tensor(alpha, dtype=torch.float32).log()
        if tune_alpha:
            self._log_alpha.requires_grad_()
            self._alpha_optim = optim.Adam([self._log_alpha], lr=critic_lr)
        else:
            self._alpha_optim = None

        # Normalizer
        self._obs_normalizer_arg = observation_normalizer
        self.obs_normalizer = create_normalizer(observation_normalizer,
                                                self.observation_space.shape,
                                                clip_range=observation_clip)

        # Other attributes
        self._total_steps = 0


#
###############################################################################

def _build_ac(observation_space, action_space,
              actor_cls, actor_kwargs,
              critic_cls, critic_kwargs):
    actor = create_actor(observation_space, action_space,
                         actor_cls, actor_kwargs,
                         policy="gaussian").to(_DEVICE)
    target_actor = create_actor(observation_space, action_space,
                                actor_cls, actor_kwargs,
                                policy="gaussian").to(_DEVICE)
    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()

    critic_1 = create_critic(observation_space, action_space,
                             critic_cls, critic_kwargs).to(_DEVICE)
    target_critic_1 = create_critic(observation_space, action_space,
                                    critic_cls, critic_kwargs).to(_DEVICE)
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_1.eval()

    critic_2 = create_critic(observation_space, action_space,
                             critic_cls, critic_kwargs).to(_DEVICE)
    target_critic_2 = create_critic(observation_space, action_space,
                                    critic_cls, critic_kwargs).to(_DEVICE)
    target_critic_2.load_state_dict(critic_2.state_dict())
    target_critic_2.eval()

    return ((actor, target_actor),
            (critic_1, target_critic_1),
            (critic_2, target_critic_2))