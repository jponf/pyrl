# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import errno
import os
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
from .noise import OUActionNoise, AdaptiveParamNoiseSpec
from .preprocessing import StandardScaler
from .utils import ReplayBuffer


###############################################################################

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LOG = torchrl.util.logging.get_logger()


###############################################################################

class DDPG(object):

    def __init__(self,
                 observation_space,
                 action_space,
                 gamma=.9,
                 tau=1e-3,
                 batch_size=128,
                 reward_scale=1.0,
                 replay_buffer_size=1000000,
                 actor_activation="relu",
                 actor_lr=3e-4,
                 critic_activation="relu",
                 critic_lr=3e-4,
                 action_penalty=1.0,
                 normalize_observations=True,
                 action_noise=0.2,
                 parameter_noise=0.0,
                 log_dir="ddpg_log"):
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
        :param bool normalize_observations: Whether or not observations should
            be normalized using the running mean and standard deviation before
            feeding them to the network.
        :param str actor_activation: Activation function used in the actor.
        :param float actor_lr: Learning rate for the actor network.
        :param str critic_activation: Activation function used in the critic.
        :param float critic_lr: Learning rate for the critic network.
        :param float action_penalty: Quadratic penalty on actions to avoid
            tanh saturation and vanishing gradients (0 disables the penalty).
        :param float action_noise: Standard deviation expressed as a fraction
            of the actions' range of values, a value in the range [0.0, 1.0].
            A value of 0 disables the use of action noise during training.
        :param bool parameter_noise: Whether parameter noise should be applied
            while training the agent.
        :param str log_dir: Directory to output tensorboard logs.
        """
        self.observation_space = observation_space
        self.action_space = action_space

        state_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.replay_buffer = ReplayBuffer(state_dim=state_dim,
                                          action_dim=action_dim,
                                          max_size=replay_buffer_size)

        # Build model (A2C architecture)
        self.actor = Actor(state_dim, action_dim,
                           activation=actor_activation,
                           layer_norm=parameter_noise).to(_DEVICE)
        self.critic = Critic(state_dim, action_dim, 1,
                             activation=critic_activation).to(_DEVICE)
        self.target_actor = Actor(state_dim, action_dim,
                                  activation=actor_activation,
                                  layer_norm=parameter_noise).to(_DEVICE)
        self.target_critic = Critic(state_dim, action_dim, 1,
                                    activation=critic_activation).to(_DEVICE)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.eval()
        self.target_actor.eval()

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self._action_penalty = action_penalty
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)

        # Normalizer
        if normalize_observations:
            self.obs_normalizer = StandardScaler(n_features=state_dim,
                                                 clip_range=5.0)
        else:
            self.obs_normalizer = None

        # Noise
        self.action_noise_arg = action_noise
        if action_noise > 0.0:
            self.action_noise = OUActionNoise(
                mu=np.zeros(action_dim),
                sigma=action_noise * (self.actor.action_space.high -
                                      self.actor.action_space.low))
        else:
            self.action_noise = None

        self.parameter_noise_arg = parameter_noise
        if parameter_noise > 0.0:
            self.param_noise = AdaptiveParamNoiseSpec(
                initial_stddev=parameter_noise,
                desired_stddev=parameter_noise)
            self.perturbed_actor = Actor(
                state_dim, action_dim, layer_norm=parameter_noise).to(_DEVICE)
            self.perturbed_actor.eval()
            _perturb_actor(self.actor, self.perturbed_actor,
                           self.param_noise.current_stddev)
        else:
            self.param_noise = None
            self.perturbed_actor = None

        # Other attributes
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
        self.critic.train(mode=mode)
        self.actor.train(mode=mode)
        self._train_mode = mode

    def reset(self):
        self.num_episodes += 1
        self.episode_steps = 0
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            _perturb_actor(self.actor, self.perturbed_actor,
                           self.param_noise.current_stddev)

    def update(self, state, env_action, reward, next_state, done):
        self.total_steps += 1
        self.episode_steps += 1

        # register observation into normalizer
        if self.obs_normalizer:
            self.obs_normalizer.update(torch.FloatTensor(state))

        # re-scale action
        action = umath.scale(env_action,
                             self.action_space.low,
                             self.action_space.high,
                             self.actor.action_space.low,
                             self.actor.action_space.high)

        # add a batch dimension and store
        self.replay_buffer.add(state=state, action=action,
                               next_state=next_state,
                               reward=reward, terminal=done)

    def train(self):
        """Trains the agent using the transitions stored during exploration.
        """
        assert self._train_mode
        self._train()
        self._update_target_networks()
        self.train_steps += 1

    def _train(self):
        (state, action, next_state,
         reward, terminal) = self.replay_buffer.sample_batch_torch(
             self.batch_size, device=_DEVICE)

        if self.obs_normalizer:
            next_state = self.obs_normalizer.transform(next_state)
            state = self.obs_normalizer.transform(state)

        # Compute critic loss
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, next_action)
            target_q *= (1 - terminal.int()) * self.gamma
            target_q += self.reward_scale * reward

        # Optimize critic
        current_q = self.critic(torch.cat((state, action), dim=1))
        loss_q = F.smooth_l1_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        self._summary_w.add_scalars("Q", {"Critic": current_q.detach().mean(),
                                          "Target": target_q.mean()},
                                    self.train_steps)
        self._summary_w.add_scalar("Loss/Critic", loss_q, self.train_steps)

        # Optimize actor
        actor_out = self.actor(state)
        loss_a = -self.critic(state, actor_out).mean()

        self.actor_optimizer.zero_grad()
        loss_a.backward()
        self.actor_optimizer.step()
        self._summary_w.add_scalar("Loss/Actor", loss_a, self.train_steps)

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

            if self.obs_normalizer:
                state = self.obs_normalizer.transform(state)

            actor_r = self.actor(state)
            perturbed_r = self.perturbed_actor(state)
            distance = (actor_r - perturbed_r).pow_(2).mean().sqrt_().item()
            self.param_noise.adapt(distance)
            return distance
        return None

    @torch.no_grad()
    def compute_action(self, state):
        # Pre-process
        state = torch.from_numpy(state).float()
        if self.obs_normalizer:
            state = self.obs_normalizer.transform(state)
        state = state.unsqueeze_(0).to(_DEVICE)

        # Compute action (using appropriate net)
        if self._train_mode and self.perturbed_actor is not None:
            action = self.perturbed_actor(state)
        else:
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
        try:
            os.makedirs(path)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

        args = {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'gamma': self.gamma,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'reward_scale': self.reward_scale,
            'replay_buffer_size': self.replay_buffer.max_size,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'action_penalty': self._action_penalty,
            'normalize_observations': self.obs_normalizer is not None,
            'action_noise': self.action_noise_arg,
            'parameter_noise': self.parameter_noise_arg
        }

        state = {
            'total_steps': self.total_steps,
            'num_episodes': self.num_episodes
        }

        pickle.dump(args, open(os.path.join(path, "args.pickle"), 'wb'))
        pickle.dump(state, open(os.path.join(path, "state.pickle"), 'wb'))

        torch.save(self.critic.state_dict(),
                   os.path.join(path, 'critic.torch'))
        torch.save(self.actor.state_dict(),
                   os.path.join(path, 'actor.torch'))

        if replay_buffer:
            self.replay_buffer.save(os.path.join(path, 'replay_buffer.h5'))

        if self.obs_normalizer:
            self.obs_normalizer.save(os.path.join(path, 'obs_normalizer'))
        if self.param_noise:
            pickle.dump(self.param_noise,
                        open(os.path.join(path, "param_noise.pickle"), "wb"))

    @classmethod
    def load(cls, path, replay_buffer=True, **kwargs):
        if not os.path.isdir(path):
            raise ValueError("{} is not a directory".format(path))

        # Load and Override arguments used to build the instance
        with open(os.path.join(path, "args.pickle"), "rb") as fh:
            _LOG.debug("(DDPG) Loading agent arguments")
            args_values = pickle.load(fh)
            args_values.update(kwargs)

            fmt_string = "    {{:>{}}}: {{}}".format(
                max(len(x) for x in args_values.keys()))
            for key, value in args_values.items():
                _LOG.debug(fmt_string.format(key, value))

        # Create instance and load the rest of the data
        instance = cls(**args_values)

        with open(os.path.join(path, "state.pickle"), "rb") as fh:
            _LOG.debug("(DDPG) Loading agent state")
            state = pickle.load(fh)
            instance.total_steps = state['total_steps']
            instance.num_episodes = state['num_episodes']

        _LOG.debug("(DDPG) Loading actor")
        actor_state = torch.load(os.path.join(path, "actor.torch"))
        instance.actor.load_state_dict(actor_state)
        instance.target_actor.load_state_dict(actor_state)
        _LOG.debug("(DDPG) Loading critic")
        critic_state = torch.load(os.path.join(path, "critic.torch"))
        instance.critic.load_state_dict(critic_state)
        instance.target_critic.load_state_dict(critic_state)

        replay_buffer_path = os.path.join(path, "replay_buffer.h5")
        if replay_buffer and os.path.isfile(replay_buffer_path):
            _LOG.debug("(DDPG) Loading replay buffer")
            instance.replay_buffer.load(replay_buffer_path)

        if instance.obs_normalizer:
            _LOG.debug("(DDPG) Loading observations normalizer")
            instance.obs_normalizer = StandardScaler.load(
                os.path.join(path, 'obs_normalizer'))

        if instance.param_noise:
            _LOG.debug("(DDPG) Loading parameter noise")
            instance.param_noise = pickle.load(
                open(os.path.join(path, "param_noise.pickle"), "rb"))

        return instance


# Utilities
###############################################################################

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
