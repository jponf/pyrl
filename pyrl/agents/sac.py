# -*- coding: utf-8 -*-

import collections
import errno
import os
import pickle
import statistics

# Scipy
# import numpy as np

# Torch
import torch
import torch.optim as optim
import torch.nn.functional as F

# ...
import pyrl.util.logging
import pyrl.util.umath as umath

from .core import Agent
from .models_utils import soft_update
from .replay_buffer import FlatReplayBuffer
from .utils import (create_normalizer,
                    create_actor, create_critic, dicts_mean)


###############################################################################

_DEVICE = "cpu"
_LOG = pyrl.util.logging.get_logger()


###############################################################################

class SAC(Agent):
    """Soft Actor Critic.

    Introduced in the paper: Off-Policy Maximum Entropy Deep Reinforcement
    Learning with a Stochastic Actor.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 alpha=0.2,
                 gamma=.95,
                 tau=0.005,
                 batch_size=128,
                 reward_scale=1.0,
                 replay_buffer_size=1000000,
                 random_steps=1000,
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
        :param observation_space: Structure of the observations returned by
            the enviornment.
        :type observation_space: gym.Box
        :param action_space: Structure of the actions that can be taken in
            the environment.
        :type action_space: gym.Box
        :param alpha: Entropy coefficient, if None this value is auto-tuned.
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
        """
        super(SAC, self).__init__(observation_space, action_space)

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.replay_buffer = FlatReplayBuffer(
            state_shape=self.observation_space.shape,
            action_shape=self.action_space.shape,
            max_size=replay_buffer_size)

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

        # Autotune alpha
        self.target_entropy = -torch.prod(
            torch.Tensor(action_space.shape)).item()
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

    @property
    def alpha(self):
        with torch.no_grad():
            return self._log_alpha.exp()

    def set_train_mode(self, mode=True):
        """Sets the agent training mode."""
        super(SAC, self).set_train_mode(mode)
        self.actor.train(mode=mode)
        self.critic_1.train(mode=mode)
        self.critic_2.train(mode=mode)

    def begin_episode(self):
        pass

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
        rand_action, _, mean_action = self.actor.sample(state)

        action = rand_action if self._train_mode else mean_action
        action = action.squeeze_(0).cpu().numpy()
        return self._to_action_space(action)

    def train(self, steps, progress=False):
        if len(self.replay_buffer) >= self.batch_size:
            super(SAC, self).train(steps, progress)

    def _train(self):
        (state, action, next_state,
         reward, terminal) = self.replay_buffer.sample_batch_torch(
             self.batch_size, device=_DEVICE)

        next_state = self.obs_normalizer.transform(next_state)
        state = self.obs_normalizer.transform(state)

        self._train_critic(state, action, next_state, reward, terminal)
        log_prob = self._train_policy(state)
        self._train_alpha(log_prob)
        self._update_target_networks()

    def _train_critic(self, state, action, next_state, reward, terminal):
        # Compute critic loss (with smoothing noise)
        with torch.no_grad():
            next_action, next_log_p, _ = self.target_actor.sample(next_state)

            next_q1 = self.target_critic_1(next_state, next_action)
            next_q2 = self.target_critic_2(next_state, next_action)

            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_p
            next_q *= (1 - terminal.int()) * self.gamma
            next_q += self.reward_scale * reward

        # Optimize critics
        current_q1 = self.critic_1(state, action)
        loss_q1 = F.smooth_l1_loss(current_q1, next_q)
        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_1_optimizer.step()

        current_q2 = self.critic_2(state, action)
        loss_q2 = F.smooth_l1_loss(current_q2, next_q)
        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_2_optimizer.step()

        with torch.no_grad():
            self._summary.add_scalars("Q", {"Mean_Q1": current_q1.mean(),
                                            "Mean_Q2": current_q2.mean(),
                                            "Mean_Target": next_q.mean()},
                                      self._train_steps)
            self._summary.add_scalar("Loss/Q1", loss_q1, self._train_steps)
            self._summary.add_scalar("Loss/Q2", loss_q2, self._train_steps)

    def _train_policy(self, state):
        actor_out, log_prob, _ = self.actor.sample(state)
        min_q = torch.min(self.critic_1(state, actor_out),
                          self.critic_2(state, actor_out))

        # Jπ = 𝔼st∼D,εt∼N[α * log π(f(εt;st)|st) − Q(st,f(εt;st))]
        loss_a = (self.alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad()
        loss_a.backward(retain_graph=True)
        self.actor_optimizer.step()

        with torch.no_grad():
            self._summary.add_scalar("Loss/Policy", loss_a, self._train_steps)
            self._summary.add_scalar("Stats/LogProb", log_prob.mean(),
                                     self._train_steps)
            self._summary.add_scalar("Stats/Alpha", self.alpha,
                                     self._train_steps)
        return log_prob

    def _train_alpha(self, log_prob):
        if self._alpha_optim is not None:
            alpha_loss = -(log_prob + self.target_entropy).detach().mean()
            alpha_loss *= self._log_alpha.exp()

            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._summary.add_scalar(
                'Loss/Alpha', alpha_loss.detach(), self._train_steps)

    def _update_target_networks(self):
        soft_update(self.actor, self.target_actor, self.tau)
        soft_update(self.critic_1, self.target_critic_1, self.tau)
        soft_update(self.critic_2, self.target_critic_2, self.tau)

    # Agent State
    ########################

    def state_dict(self):
        state = {"critic1": self.critic_1.state_dict(),
                 "critic2": self.critic_2.state_dict(),
                 "actor": self.actor.state_dict(),
                 "log_alpha": self._log_alpha,
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

        with torch.no_grad():
            self._log_alpha.copy_(state["log_alpha"])

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

        with torch.no_grad():
            self._log_alpha.copy_(sum(x["log_alpha"] for x in states))
            self._log_alpha.div_(len(states))

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
            ('alpha', self.alpha.item()),
            ('gamma', self.gamma),
            ('tau', self.tau),
            ('batch_size', self.batch_size),
            ('reward_scale', self.reward_scale),
            ('replay_buffer_size', self.replay_buffer.max_size),
            ('random_steps', self.random_steps),
            ('actor_cls', type(self.actor)),
            ('actor_kwargs', self._actor_kwargs),
            ('actor_lr', self._actor_lr),
            ('critic_cls', type(self.critic_1)),
            ('critic_kwargs', self._critic_kwargs),
            ('critic_lr', self._critic_lr),
            ('tune_alpha', self._alpha_optim is not None),
            ('observation_normalizer', self._obs_normalizer_arg),
            ('observation_clip', self.obs_normalizer.clip_range),
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
