# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import collections
import errno
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Scipy
import numpy as np

# Torch
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard

# robotrl
import torchrl.util.logging
import torchrl.util.math as umath

from .models import Actor, Critic
from .noise import NormalActionNoise
from .preprocessing import StandardScaler
from .utils import ReplayBuffer


###############################################################################

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LOG = torchrl.util.logging.get_logger()


###############################################################################

class TD3(object):
    """Twin Delayed Deep Deterministic Policy Gradient Algorithm.

    Introduced in the paper: Addressing Function Approximation Error in
    Actor-Critic Methods.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 gamma=.9,
                 tau=0.005,
                 batch_size=128,
                 policy_delay=2,
                 reward_scale=1.0,
                 replay_buffer_size=500000,
                 actor_hidden_layers=3,
                 actor_hidden_size=256,
                 actor_activation="relu",
                 actor_lr=3e-4,
                 critic_hidden_layers=3,
                 critic_hidden_size=256,
                 critic_activation="relu",
                 critic_lr=3e-4,
                 normalize_observations=True,
                 action_noise=0.2,
                 random_steps=1000,
                 log_dir="td3_log"):
        """
        :param observation_space: Structure of the observations returned by
            the enviornment.
        :param action_space: Structure of the actions that can be taken in
            the environment.
        :param float gamma: Bellman's discount rate.
        :param float tau: Used to perform "soft" updates of the weights from
            the actor/critic to their "target" counterparts.
        :param int batch_size: Size of the sample used to train the actor and
            critic at each timestep.
        :param int replay_buffer_size: Size of the replay buffer.
        :param str actor_activation: Activation function used in the actor.
        :param float actor_lr: Learning rate for the actor network.
        :param str critic_activation: Activation function used in the critic.
        :param float critic_lr: Learning rate for the critic network.
        :param float action_noise: Standard deviation expressed as a fraction
            of the actions' range of values, a value in the range [0.0, 1.0].
            A value of 0 disables the use of action noise during training.
            tanh saturation and vanishing gradients (0 disables the penalty).
        :param bool normalize_observations: Whether or not observations should
            be normalized using the running mean and standard deviation before
            feeding them to the network.
        :param int random_steps: Initial number  of steps that will use a
            purely exploration policy (random). Afterwards, an off-policy
            exploration strategy with Gaussian noise will be used.
        :param str log_dir: Directory to output tensorboard logs.
        """
        self.observation_space = observation_space
        self.action_space = action_space

        state_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.reward_scale = reward_scale
        self.replay_buffer = ReplayBuffer(state_dim=state_dim,
                                          action_dim=action_dim,
                                          max_size=replay_buffer_size)

        # Build model (A2C architecture)
        self.actor = Actor(state_dim, action_dim,
                           hidden_layers=actor_hidden_layers,
                           hidden_size=actor_hidden_size,
                           activation=actor_activation).to(_DEVICE)
        self.target_actor = Actor(state_dim, action_dim,
                                  hidden_layers=actor_hidden_layers,
                                  hidden_size=actor_hidden_size,
                                  activation=actor_activation).to(_DEVICE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.critic_1 = Critic(state_dim, action_dim, 1,
                               hidden_layers=critic_hidden_layers,
                               hidden_size=critic_hidden_size,
                               activation=critic_activation).to(_DEVICE)
        self.target_critic_1 = Critic(state_dim, action_dim, 1,
                                      hidden_layers=critic_hidden_layers,
                                      hidden_size=critic_hidden_size,
                                      activation=critic_activation).to(_DEVICE)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_1.eval()

        self.critic_2 = Critic(state_dim, action_dim, 1,
                               hidden_layers=critic_hidden_layers,
                               hidden_size=critic_hidden_size,
                               activation=critic_activation).to(_DEVICE)
        self.target_critic_2 = Critic(state_dim, action_dim, 1,
                                      hidden_layers=critic_hidden_layers,
                                      hidden_size=critic_hidden_size,
                                      activation=critic_activation).to(_DEVICE)
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_critic_2.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),
                                             lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),
                                             lr=critic_lr)

        self._actor_hidden_layers = actor_hidden_layers
        self._actor_hidden_size = actor_hidden_size
        self._actor_activation = actor_activation
        self._actor_lr = actor_lr
        self._critic_hidden_layers = critic_hidden_layers
        self._critic_hidden_size = critic_hidden_size
        self._critic_activation = critic_activation
        self._critic_lr = critic_lr

        # Normalizer
        if normalize_observations:
            self.obs_normalizer = StandardScaler(
                n_features=state_dim,
                clip_range=5.0)
        else:
            self.obs_normalizer = None

        # Noise
        action_space_range = (self.actor.action_space.high -
                              self.actor.action_space.low)

        self.action_noise_arg = action_noise
        if action_noise > 0.0:
            self.action_noise = NormalActionNoise(
                mu=np.zeros(action_dim),
                sigma=action_noise * action_space_range)
        else:
            self.action_noise = None

        self.smoothing_noise = NormalActionNoise(
            mu=np.zeros(action_dim),
            sigma=0.1 * action_space_range,
            clip_min=np.repeat(-0.2, action_dim),
            clip_max=np.repeat(0.2, action_dim))

        self.random_steps = random_steps

        # Other training attributes
        self.total_steps = 0
        self.num_episodes = 0
        self.episode_steps = 0
        self.train_steps = 0
        self._train_mode = True
        self._summary_w = tensorboard.SummaryWriter(log_dir=log_dir)

    def set_eval_mode(self):
        """Sets the agent in evaluation mode."""
        self.set_train_mode(mode=False)

    def set_train_mode(self, mode=True):
        """Sets the agent training mode."""
        self.actor.train(mode=mode)
        self.critic_1.train(mode=mode)
        self.critic_2.train(mode=mode)
        self._train_mode = mode

    def reset(self):
        self.num_episodes += 1
        self.episode_steps = 0
        if self.action_noise is not None:
            self.action_noise.reset()

    def update(self, state, env_action, reward, next_state, done):
        self.total_steps += 1
        self.episode_steps += 1

        # register observation into normalizer
        if self.obs_normalizer:
            self.obs_normalizer.update(state)

        # re-scale action
        action = umath.scale(env_action,
                             self.action_space.low,
                             self.action_space.high,
                             self.actor.action_space.low,
                             self.actor.action_space.high)

        self.replay_buffer.add(state=state, action=action,
                               next_state=next_state,
                               reward=reward, terminal=done)

    def train(self, steps):
        """Trains the agent using the transitions stored during exploration.

        :param step: The number of training steps. It should be greater than
            `policy_delay` otherwise the policy will not be trained.
        """
        assert self._train_mode
        if len(self.replay_buffer) >= self.batch_size:
            for i in range(steps):
                self._train((i % self.policy_delay) == 0)
                self.train_steps += 1

    def _train(self, update_policy):
        (state, action, next_state,
         reward, terminal) = self.replay_buffer.sample_batch_torch(
             self.batch_size, device=_DEVICE)

        if self.obs_normalizer:
            next_state = self.obs_normalizer.transform(next_state)
            state = self.obs_normalizer.transform(state)

        # Compute critic loss
        with torch.no_grad():
            next_action = self.target_actor(next_state).cpu().numpy()
            next_action += self.smoothing_noise()
            np.clip(next_action,
                    self.actor.action_space.low,
                    self.actor.action_space.high,
                    out=next_action)
            next_action = torch.from_numpy(next_action).to(_DEVICE)

            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            target_q = (1 - terminal.int()) * self.gamma * min_target_q
            target_q += self.reward_scale * reward
            self._summary_w.add_scalars(
                "TargetQ",
                {"Q1": target_q1.mean(),
                 "Q2": target_q2.mean(),
                 "T": target_q.mean(),
                 "Min": min_target_q.mean()},
                self.train_steps)

        # Optimize critic
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

        self._summary_w.add_scalars("Q", {"Q1": current_q1.mean(),
                                          "Q2": current_q2.mean(),
                                          "Target": target_q.mean()},
                                    self.train_steps)
        self._summary_w.add_scalar("Loss/Critic1", loss_q1, self.train_steps)
        self._summary_w.add_scalar("Loss/Critic2", loss_q2, self.train_steps)

        # Delayed policy updates
        if update_policy:
            actor_out = self.actor(state)
            loss_a = -self.critic_1(state, actor_out).mean()

            self.actor_optimizer.zero_grad()
            loss_a.backward()
            self.actor_optimizer.step()

            self._update_target_networks()
            self._summary_w.add_scalar("Loss/Actor", loss_a, self.train_steps)

    def _update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

        for target_param, param in zip(self.target_critic_1.parameters(),
                                       self.critic_1.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

        for target_param, param in zip(self.target_critic_2.parameters(),
                                       self.critic_2.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

    @torch.no_grad()
    def compute_action(self, state):
        # Random exploration
        if self._train_mode and self.total_steps < self.random_steps:
            return self.action_space.sample()

        # Pre-process
        state = torch.from_numpy(state).float()
        if self.obs_normalizer:
            state = self.obs_normalizer.transform(state)
        state = state.unsqueeze_(0).to(_DEVICE)

        # Compute action
        action = self.actor(state)

        # Post-process
        action = action.squeeze_(0).cpu().numpy()
        if self._train_mode and self.action_noise is not None:
            noise = self.action_noise()
            action = np.clip(action + noise,
                             self.actor.action_space.low,
                             self.actor.action_space.high)

        env_action = umath.scale(action,
                                 self.actor.action_space.low,
                                 self.actor.action_space.high,
                                 self.action_space.low,
                                 self.action_space.high)
        return env_action

    def save(self, path, replay_buffer=True):
        """Saves the agent in the directory pointed by `path`.

        If the directory does not exist a new one will be created.
        """
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
            ('policy_delay', self.policy_delay),
            ('reward_scale', self.reward_scale),
            ('replay_buffer_size', self.replay_buffer.max_size),
            ('actor_hidden_layers', self._actor_hidden_layers),
            ('actor_hidden_size', self._actor_hidden_size),
            ('actor_activation', self._actor_activation),
            ('actor_lr', self._actor_lr),
            ('critic_hidden_layers', self._critic_hidden_layers),
            ('critic_hidden_size', self._critic_hidden_size),
            ('critic_activation', self._critic_activation),
            ('critic_lr', self._critic_lr),
            ('normalize_observations', self.obs_normalizer is not None),
            ('action_noise', self.action_noise_arg),
            ('random_steps', self.random_steps),
            ('log_dir', self._summary_w.log_dir)
        ])

        state = {
            'total_steps': self.total_steps,
            'num_episodes': self.num_episodes,
            'train_steps': self.train_steps
        }

        pickle.dump(args, open(os.path.join(path, "args.pickle"), 'wb'))
        pickle.dump(state, open(os.path.join(path, "state.pickle"), 'wb'))

        torch.save(self.actor.state_dict(),
                   os.path.join(path, 'actor.torch'))
        torch.save(self.critic_1.state_dict(),
                   os.path.join(path, 'critic_1.torch'))
        torch.save(self.critic_2.state_dict(),
                   os.path.join(path, 'critic_2.torch'))

        if replay_buffer:
            self.replay_buffer.save(os.path.join(path, 'replay_buffer.h5'))

        if self.obs_normalizer:
            self.obs_normalizer.save(os.path.join(path, 'obs_normalizer'))

    @classmethod
    def load(cls, path, replay_buffer=True, **kwargs):
        if not os.path.isdir(path):
            raise ValueError("{} is not a directory".format(path))

        # Load and Override arguments used to build the instance
        with open(os.path.join(path, "args.pickle"), "rb") as fh:
            _LOG.debug("(TD3) Loading agent arguments")
            args_values = pickle.load(fh)
            args_values.update(kwargs)

            fmt_string = "    {{:>{}}}: {{}}".format(
                max(len(x) for x in args_values.keys()))
            for key, value in args_values.items():
                _LOG.debug(fmt_string.format(key, value))

        # Create instance and load the rest of the data
        instance = cls(**args_values)

        with open(os.path.join(path, "state.pickle"), "rb") as fh:
            _LOG.debug("(TD3) Loading agent state")
            state = pickle.load(fh)
            instance.total_steps = state['total_steps']
            instance.num_episodes = state['num_episodes']
            instance.train_steps = state['train_steps']

        _LOG.debug("(TD3) Loading actor")
        actor_state = torch.load(os.path.join(path, "actor.torch"))
        instance.actor.load_state_dict(actor_state)
        instance.target_actor.load_state_dict(actor_state)
        _LOG.debug("(TD3) Loading critic 1")
        critic1_state = torch.load(os.path.join(path, "critic_1.torch"))
        instance.critic_1.load_state_dict(critic1_state)
        instance.target_critic_1.load_state_dict(critic1_state)
        _LOG.debug("(TD3) Loading critic 2")
        critic2_state = torch.load(os.path.join(path, "critic_2.torch"))
        instance.critic_2.load_state_dict(critic2_state)
        instance.target_critic_2.load_state_dict(critic2_state)

        replay_buffer_path = os.path.join(path, "replay_buffer.h5")
        if replay_buffer and os.path.isfile(replay_buffer_path):
            _LOG.debug("(TD3) Loading replay buffer")
            instance.replay_buffer.load(replay_buffer_path)

        if instance.obs_normalizer:
            _LOG.debug("(TD3) Loading observations normalizer")
            instance.obs_normalizer = StandardScaler.load(
                os.path.join(path, 'obs_normalizer'))

        return instance
