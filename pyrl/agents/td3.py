# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import collections
import errno
import os
import six.moves.cPickle as pickle

import six

# Scipy
import numpy as np

# Torch
import torch
import torch.optim as optim
import torch.nn.functional as F

# ...
import pyrl.util.logging
import pyrl.util.umath as umath

from .core import Agent
from .noise import NormalActionNoise
from .replay_buffer import FlatReplayBuffer
from .utils import (create_action_noise, create_normalizer,
                    create_actor, create_critic, dicts_mean)


###############################################################################

_DEVICE = "cpu"
_LOG = pyrl.util.logging.get_logger()


###############################################################################

class TD3(Agent):
    """Twin Delayed Deep Deterministic Policy Gradient Algorithm.

    Introduced in the paper: Addressing Function Approximation Error in
    Actor-Critic Methods.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 gamma=.95,
                 tau=0.005,
                 batch_size=128,
                 reward_scale=1.0,
                 replay_buffer_size=1000000,
                 policy_delay=2,
                 random_steps=1000,
                 actor_cls=None,
                 actor_kwargs=None,
                 actor_lr=0.001,
                 critic_cls=None,
                 critic_kwargs=None,
                 critic_lr=0.001,
                 observation_normalizer="none",
                 observation_clip=float('inf'),
                 action_noise="normal_0.2"):
        """
        :param observation_space: Structure of the observations returned by
            the enviornment.
        :type observation_space: gym.Box
        :param action_space: Structure of the actions that can be taken in
            the environment.
        :type action_space: gym.Box
        :param gamma: Bellman's discount rate.
        :type gamma: float
        :param tau: Used to perform "soft" updates (polyak averaging) of the
            weights from the actor/critic to their "target" counterparts.
        :type tau: float
        :param batch_size: Size of the sample used to train the actor and
            critic at each timestep.
        :type batch_size: int
        :param replay_buffer_size: Number of transitions to store in the replay
            buffer.
        :type replay_buffer_size: int
        :param policy_delay: Number of times the critic networks are trained
            before training the policy network.
        :type policy_delay: int
        :param random_steps: Number of steps taken completely at random while
            training before using the actor action + noise.
        :type random_steps: int
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
        :param action_noise: Name and standard deviaton of the action noise
            expressed as name_std, i.e., ou_0.2 or normal_0.1. Use "none" to
            disable the use of action noise.
        :type action_noise: str
        """
        super(TD3, self).__init__(observation_space, action_space)
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.replay_buffer = FlatReplayBuffer(
            state_shape=self.observation_space.shape,
            action_shape=self.action_space.shape,
            max_size=replay_buffer_size)

        self.policy_delay = policy_delay
        self.random_steps = random_steps

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

        # Normalizer
        self._obs_normalizer_arg = observation_normalizer
        self.obs_normalizer = create_normalizer(observation_normalizer,
                                                self.observation_space.shape,
                                                clip_range=observation_clip)

        # Noise
        self._action_noise_arg = action_noise
        self.action_noise = create_action_noise(action_noise, action_space)

        actor_space_range = (self.actor.action_space.high -
                             self.actor.action_space.low)
        self.smoothing_noise = NormalActionNoise(
            mu=np.zeros(actor_space_range.shape),
            sigma=0.1 * actor_space_range,
            clip_min=0.15 * actor_space_range,
            clip_max=-0.15 * actor_space_range)

        # Other attributes
        self._total_steps = 0

    def set_train_mode(self, mode=True):
        """Sets the agent training mode."""
        super(TD3, self).set_train_mode(mode)
        self.actor.train(mode=mode)
        self.critic_1.train(mode=mode)
        self.critic_2.train(mode=mode)

    def begin_episode(self):
        self.action_noise.reset()

    def end_episode(self):
        pass

    def update(self, state, action, reward, next_state, terminal):
        self._total_steps += 1
        action = self._to_actor_space(action)  # re-scale action

        self.obs_normalizer.update(state)
        self.replay_buffer.add(state=state, action=action,
                               next_state=next_state,
                               reward=reward, terminal=terminal)

    @torch.no_grad()
    def compute_action(self, state):
        # Random exploration
        if self._train_mode and self._total_steps < self.random_steps:
            return self.action_space.sample()

        # Pre-process
        state = torch.from_numpy(state).float()
        state = self.obs_normalizer.transform(state).unsqueeze_(0).to(_DEVICE)

        # Compute action
        action = self.actor(state).squeeze_(0).cpu().numpy()

        # Post-process
        if self._train_mode:
            action = np.clip(action + self.action_noise(),
                             self.actor.action_space.low,
                             self.actor.action_space.high)

        return self._to_action_space(action)

    def train(self, steps, progress=False):
        if len(self.replay_buffer) >= self.batch_size:
            super(TD3, self).train(steps, progress)

    def _train(self):
        (state, action, next_state,
         reward, terminal) = self.replay_buffer.sample_batch_torch(
             self.batch_size, device=_DEVICE)

        next_state = self.obs_normalizer.transform(next_state)
        state = self.obs_normalizer.transform(state)

        # Compute critic loss (with smoothing noise)
        with torch.no_grad():
            next_action = self.target_actor(next_state).cpu().numpy()
            np.clip(next_action + self.smoothing_noise(),
                    self.actor.action_space.low,
                    self.actor.action_space.high,
                    out=next_action)
            next_action = torch.from_numpy(next_action).to(_DEVICE)

            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            target_q = (1 - terminal.int()) * self.gamma * min_target_q
            target_q += self.reward_scale * reward

        # Optimize critics
        current_q1 = self.critic_1(state, action)
        loss_q1 = F.smooth_l1_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_1_optimizer.step()

        current_q2 = self.critic_2(state, action)
        loss_q2 = F.smooth_l1_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_2_optimizer.step()

        self._summary.add_scalars("Q", {"Q1": current_q1.mean(),
                                        "Q2": current_q2.mean(),
                                        "Target": target_q.mean()},
                                  self._train_steps)
        self._summary.add_scalar("Loss/Critic1", loss_q1, self._train_steps)
        self._summary.add_scalar("Loss/Critic2", loss_q2, self._train_steps)

        # Delayed policy updates
        if ((self._train_steps + 1) % self.policy_delay) == 0:
            actor_out = self.actor(state)
            loss_a = -self.critic_1(state, actor_out).mean()

            self.actor_optimizer.zero_grad()
            loss_a.backward()
            self.actor_optimizer.step()

            self._summary.add_scalar("Loss/Actor", loss_a, self._train_steps)
            self._update_target_networks()

    def _update_target_networks(self):
        a_params = six.moves.zip(self.target_actor.parameters(),
                                 self.actor.parameters())
        c1_params = six.moves.zip(self.target_critic_1.parameters(),
                                  self.critic_1.parameters())
        c2_params = six.moves.zip(self.target_critic_2.parameters(),
                                  self.critic_2.parameters())
        for params in (a_params, c1_params, c2_params):
            for target_param, param in params:
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(param.data * self.tau)

    # Agent State
    ########################

    def state_dict(self):
        state = {"critic1": self.critic_1.state_dict(),
                 "critic2": self.critic_2.state_dict(),
                 "actor": self.actor.state_dict(),
                 "obs_normalizer": self.obs_normalizer.state_dict(),
                 "train_steps": self._train_steps,
                 "total_steps": self._total_steps}

        return state

    def load_state_dict(self, state):
        self.critic_1.load_state_dict(state['critic1'])
        self.target_critic_1.load_state_dict(state['critic1'])
        self.critic_2.load_state_dict(state['critic2'])
        self.target_critic_2.load_state_dict(state['critic2'])
        self.actor.load_state_dict(state["actor"])
        self.target_actor.load_state_dict(state["actor"])

        self.obs_normalizer.load_state_dict(state["obs_normalizer"])

        self._train_steps = state["train_steps"]
        self._total_steps = state["total_steps"]

    def aggregate_state_dicts(self, states):
        critic_1_state = dicts_mean([x['critic1'] for x in states])
        self.critic_1.load_state_dict(critic_1_state)
        self.target_critic_1.load_state_dict(critic_1_state)

        critic_2_state = dicts_mean([x['critic2'] for x in states])
        self.critic_2.load_state_dict(critic_2_state)
        self.target_critic_2.load_state_dict(critic_2_state)

        actor_state = dicts_mean([x['actor'] for x in states])
        self.actor.load_state_dict(actor_state)
        self.target_actor.load_state_dict(actor_state)

        self.obs_normalizer.load_state_dict([x['obs_normalizer']
                                             for x in states])

        self._train_steps = max(x["train_steps"] for x in states)
        self._total_steps = max(x["total_steps"] for x in states)

    # Save/Load Agent
    ########################

    def save(self, path, replay_buffer=True):
        try:
            os.makedirs(path)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

        args = collections.OrderedDict([
            ('observation_space', self.observation_space),
            ('action_space', self.action_space),
            ('gamma', self.gamma),
            ('tau', self.tau),
            ('batch_size', self.batch_size),
            ('reward_scale', self.reward_scale),
            ('replay_buffer_size', self.replay_buffer.max_size),
            ('policy_delay', self.policy_delay),
            ('random_steps', self.random_steps),
            ('actor_cls', type(self.actor)),
            ('actor_kwargs', self._actor_kwargs),
            ('actor_lr', self._actor_lr),
            ('critic_cls', type(self.critic_1)),
            ('critic_kwargs', self._critic_kwargs),
            ('critic_lr', self._critic_lr),
            ('observation_normalizer', self._obs_normalizer_arg),
            ('observation_clip', self.obs_normalizer.clip_range),
            ('action_noise', self._action_noise_arg)
        ])
        pickle.dump(args, open(os.path.join(path, "args.pkl"), 'wb'))

        state = self.state_dict()
        pickle.dump(state, open(os.path.join(path, "state.pkl"), "wb"))

        if replay_buffer:
            self.replay_buffer.save(os.path.join(path, 'replay_buffer.h5'))

    @classmethod
    def load(cls, path, replay_buffer=True, **kwargs):
        if not os.path.isdir(path):
            raise ValueError("{} is not a directory".format(path))

        # Load and Override arguments used to build the instance
        with open(os.path.join(path, "args.pkl"), "rb") as fh:
            _LOG.debug("(TD3) Loading agent arguments")
            args_values = pickle.load(fh)
            args_values.update(kwargs)

            fmt_string = "    {{:>{}}}: {{}}".format(
                max(len(x) for x in args_values.keys()))
            for key, value in args_values.items():
                _LOG.debug(fmt_string.format(key, value))

        # Create instance and load the rest of the data
        instance = cls(**args_values)

        with open(os.path.join(path, "state.pkl"), "rb") as fh:
            _LOG.debug("(TD3) Loading agent state")
            state = pickle.load(fh)
            instance.load_state_dict(state)

        replay_buffer_path = os.path.join(path, "replay_buffer.h5")
        if replay_buffer and os.path.isfile(replay_buffer_path):
            _LOG.debug("(TD3) Loading replay buffer")
            instance.replay_buffer.load(replay_buffer_path)

        return instance

    # Utilities
    ########################

    def _to_actor_space(self, action):
        return umath.scale(x=action,
                           min_x=self.action_space.low,
                           max_x=self.action_space.high,
                           min_out=self.actor.action_space.low,
                           max_out=self.actor.action_space.high)

    def _to_action_space(self, action):
        return umath.scale(x=action,
                           min_x=self.actor.action_space.low,
                           max_x=self.actor.action_space.high,
                           min_out=self.action_space.low,
                           max_out=self.action_space.high)


#
###############################################################################

def _build_ac(observation_space, action_space,
              actor_cls, actor_kwargs,
              critic_cls, critic_kwargs):
    actor = create_actor(observation_space, action_space,
                         actor_cls, actor_kwargs).to(_DEVICE)
    target_actor = create_actor(observation_space, action_space,
                                actor_cls, actor_kwargs).to(_DEVICE)
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
