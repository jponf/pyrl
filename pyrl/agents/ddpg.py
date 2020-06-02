# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

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

# ...
import pyrl.util.logging
import pyrl.util.umath as umath

from .core import Agent
from .noise import AdaptiveParamNoiseSpec
from .replay_buffer import FlatReplayBuffer
from .utils import (create_action_noise, create_normalizer,
                    create_actor, create_critic, dicts_mean)


###############################################################################

_DEVICE = "cpu"
_LOG = pyrl.util.logging.get_logger()


###############################################################################

class DDPG(Agent):
    """Deep Deterministic Policy Gradients.

    Introduced in the paper: Continuous Control With Deep Reinforcement
    Learning
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 gamma=.9,
                 tau=1e-3,
                 batch_size=128,
                 reward_scale=1.0,
                 replay_buffer_size=1000000,
                 actor_cls=None,
                 actor_kwargs=None,
                 actor_lr=0.001,
                 critic_cls=None,
                 critic_kwargs=None,
                 critic_lr=0.001,
                 observation_normalizer="none",
                 observation_clip=float('inf'),
                 action_noise="ou_0.2",
                 parameter_noise=0.0):
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
        :param parameter_noise: Whether parameter noise should be applied
            while training the agent.
        :type parameter_noise: bool
        """
        super(DDPG, self).__init__(observation_space, action_space)
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.replay_buffer = FlatReplayBuffer(
            state_shape=self.observation_space.shape,
            action_shape=self.action_space.shape,
            max_size=replay_buffer_size)

        # Build model (AC architecture)
        actors, critics = _build_ac(self.observation_space,
                                    self.action_space,
                                    actor_cls, actor_kwargs,
                                    critic_cls, critic_kwargs,
                                    parameter_noise)

        self.actor, self.target_actor = actors
        self.critic, self.target_critic = critics

        self._actor_kwargs = actor_kwargs
        self._actor_lr = actor_lr
        self._critic_kwargs = critic_kwargs
        self._critic_lr = critic_lr
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)

        # Normalizer
        self._obs_normalizer_arg = observation_normalizer
        self.obs_normalizer = create_normalizer(observation_normalizer,
                                                self.observation_space.shape,
                                                clip_range=observation_clip)

        # Noise
        self._action_noise_arg = action_noise
        self.action_noise = create_action_noise(action_noise, action_space)

        self.parameter_noise_arg = parameter_noise
        if parameter_noise > 0.0:
            self.param_noise = AdaptiveParamNoiseSpec(
                initial_stddev=parameter_noise,
                desired_stddev=parameter_noise)
            self.perturbed_actor = create_actor(
                observation_space, action_space, actor_cls, actor_kwargs)
            self.perturbed_actor.to(_DEVICE)
            self.perturbed_actor.eval()
            _perturb_actor(self.actor, self.perturbed_actor,
                           self.param_noise.current_stddev)
        else:
            self.param_noise = None
            self.perturbed_actor = None

    def set_train_mode(self, mode=True):
        """Sets the agent training mode."""
        super(DDPG, self).set_train_mode(mode)
        self.critic.train(mode=mode)
        self.actor.train(mode=mode)

    def begin_episode(self):
        self.action_noise.reset()
        if self.param_noise is not None:
            _perturb_actor(self.actor, self.perturbed_actor,
                           self.param_noise.current_stddev)

    def end_episode(self):
        pass

    def update(self, state, action, reward, next_state, terminal):
        action = self._to_actor_space(action)  # re-scale action

        self.obs_normalizer.update(torch.FloatTensor(state))
        self.replay_buffer.add(state=state, action=action,
                               next_state=next_state,
                               reward=reward, terminal=terminal)

    @torch.no_grad()
    def compute_action(self, state):
        # Pre-process
        state = torch.from_numpy(state).float()
        state = self.obs_normalizer.transform(state).unsqueeze_(0).to(_DEVICE)

        # Compute action (using appropriate net)
        if self._train_mode and self.perturbed_actor is not None:
            action = self.perturbed_actor(state)
        else:
            action = self.actor(state)

        # Post-process
        action = action.squeeze_(0).cpu().numpy()
        if self._train_mode:
            action = np.clip(action + self.action_noise(),
                             self.actor.action_space.low,
                             self.actor.action_space.high)

        return self._to_action_space(action)

    def train(self, steps, progress=False):
        if len(self.replay_buffer) >= self.batch_size:
            super(DDPG, self).train(steps, progress)

    def _train(self):
        (state, action, next_state,
         reward, terminal) = self.replay_buffer.sample_batch_torch(
             self.batch_size, device=_DEVICE)

        state = self.obs_normalizer.transform(state)
        next_state = self.obs_normalizer.transform(next_state)

        # Compute critic loss
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, next_action)
            target_q *= (1 - terminal.int()) * self.gamma
            target_q += self.reward_scale * reward

        # Optimize critic
        current_q = self.critic(state, action)
        loss_q = F.smooth_l1_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        self._summary.add_scalars("Q", {"Critic": current_q.detach().mean(),
                                        "Target": target_q.mean()},
                                  self._train_steps)
        self._summary.add_scalar("Loss/Critic", loss_q, self._train_steps)

        # Optimize actor
        actor_out = self.actor(state)
        loss_a = -self.critic(state, actor_out).mean()

        self.actor_optimizer.zero_grad()
        loss_a.backward()
        self.actor_optimizer.step()
        self._summary.add_scalar("Loss/Actor", loss_a, self._train_steps)

        # Update Target Networks
        self._update_target_networks()

    def _update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

    @torch.no_grad()
    def adapt_parameter_noise(self):
        """Perturbs a separate copy of the policy to adjust the scale
        for the next "real" perturbation.
        """
        if self.param_noise is not None:
            _perturb_actor(self.actor, self.perturbed_actor,
                           self.param_noise.current_stddev)
            (state, _, _, _, _) = self.replay_buffer.sample_batch_torch(
                self.batch_size, device=_DEVICE)

            state = self.obs_normalizer.transform(state)

            actor_r = self.actor(state)
            perturbed_r = self.perturbed_actor(state)
            distance = (actor_r - perturbed_r).pow_(2).mean().sqrt_().item()
            self.param_noise.adapt(distance)
            return distance
        return None

    # Agent State
    ########################

    def state_dict(self):
        state = {"critic": self.critic.state_dict(),
                 "actor": self.actor.state_dict(),
                 "obs_normalizer": self.obs_normalizer.state_dict(),
                 "train_steps": self._train_steps}

        return state

    def load_state_dict(self, state):
        self.critic.load_state_dict(state['critic'])
        self.target_critic.load_state_dict(state['critic'])
        self.actor.load_state_dict(state["actor"])
        self.target_actor.load_state_dict(state["actor"])

        self.obs_normalizer.load_state_dict(state["obs_normalizer"])

        self._train_steps = state["train_steps"]

    def aggregate_state_dicts(self, states):
        critic_state = dicts_mean([x['critic'] for x in states])
        self.critic.load_state_dict(critic_state)
        self.target_critic.load_state_dict(critic_state)

        actor_state = dicts_mean([x['actor'] for x in states])
        self.actor.load_state_dict(actor_state)
        self.target_actor.load_state_dict(actor_state)

        self.obs_normalizer.load_state_dict([x['obs_normalizer']
                                             for x in states])

        self._train_steps = max(x["train_steps"] for x in states)

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
            ('actor_cls', type(self.actor)),
            ('actor_kwargs', self._actor_kwargs),
            ('actor_lr', self._actor_lr),
            ('critic_cls', type(self.critic)),
            ('critic_kwargs', self._critic_kwargs),
            ('critic_lr', self._critic_lr),
            ('observation_normalizer', self._obs_normalizer_arg),
            ('observation_clip', self.obs_normalizer.clip_range),
            ('action_noise', self._action_noise_arg),
            ('parameter_noise', self.parameter_noise_arg)
        ])
        pickle.dump(args, open(os.path.join(path, "args.pkl"), 'wb'))

        state = self.state_dict()
        pickle.dump(state, open(os.path.join(path, "state.pkl"), "wb"))

        if replay_buffer:
            self.replay_buffer.save(os.path.join(path, 'replay_buffer.h5'))

        if self.param_noise:
            pickle.dump(self.param_noise,
                        open(os.path.join(path, "param_noise.pickle"), "wb"))

    @classmethod
    def load(cls, path, replay_buffer=True, **kwargs):
        if not os.path.isdir(path):
            raise ValueError("{} is not a directory".format(path))

        # Load and Override arguments used to build the instance
        with open(os.path.join(path, "args.pkl"), "rb") as fh:
            _LOG.debug("(DDPG) Loading agent arguments")
            args_values = pickle.load(fh)
            args_values.update(kwargs)

            fmt_string = "    {{:>{}}}: {{}}".format(
                max(len(x) for x in args_values.keys()))
            for key, value in args_values.items():
                _LOG.debug(fmt_string.format(key, value))

        # Create instance and load the rest of the data
        instance = cls(**args_values)

        with open(os.path.join(path, "state.pkl"), "rb") as fh:
            _LOG.debug("(DDPG) Loading agent state")
            state = pickle.load(fh)
            instance.load_state_dict(state)

        replay_buffer_path = os.path.join(path, "replay_buffer.h5")
        if replay_buffer and os.path.isfile(replay_buffer_path):
            _LOG.debug("(DDPG) Loading replay buffer")
            instance.replay_buffer.load(replay_buffer_path)

        if instance.param_noise:
            _LOG.debug("(DDPG) Loading parameter noise")
            instance.param_noise = pickle.load(
                open(os.path.join(path, "param_noise.pickle"), "rb"))

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

def _build_ac(observation_space, action_space, actor_cls, actor_kwargs,
              critic_cls, critic_kwargs, parameter_noise):
    if parameter_noise:
        actor_kwargs["layer_norm"] = True

    actor = create_actor(observation_space, action_space,
                         actor_cls, actor_kwargs).to(_DEVICE)
    target_actor = create_actor(observation_space, action_space,
                                actor_cls, actor_kwargs).to(_DEVICE)
    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()

    critic = create_critic(observation_space, action_space,
                           critic_cls, critic_kwargs).to(_DEVICE)
    target_critic = create_critic(observation_space, action_space,
                                  critic_cls, critic_kwargs).to(_DEVICE)
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()

    return (actor, target_actor), (critic, target_critic)


def _perturb_actor(actor, perturbed_actor, param_noise_std):
    perturbable_params = actor.get_perturbable_parameters()
    assert perturbable_params == perturbed_actor.get_perturbable_parameters()

    a_params = actor.named_parameters()
    p_params = perturbed_actor.named_parameters()
    for (a_name, a_param), (p_name, p_param) in zip(a_params, p_params):
        assert a_name == p_name
        if a_name in perturbable_params:
            noise = torch.normal(mean=0, std=param_noise_std,
                                 size=a_param.size()).to(p_param.data.device)
            p_param.data.copy_(a_param.data + noise)
        else:
            p_param.data.copy_(a_param.data)
