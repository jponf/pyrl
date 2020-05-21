# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import collections
import errno
import os
import pickle

import tqdm

# Scipy
import numpy as np

# Torch
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard

# ...
import torchrl.util.logging
import torchrl.util.math as umath

from .models import HerActor, HerCritic
from .noise import NormalActionNoise
from .preprocessing import StandardScaler
from .utils import HerReplayBuffer


###############################################################################

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LOG = torchrl.util.logging.get_logger()


###############################################################################

class ReplayBuffer(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = []
        self._num_transitions = 0
        self._num_episodes = 0

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return self._num_episodes

    @property
    def num_transitions(self):
        return self._num_transitions

    def store_episode(self, transitions):
        # drop oldest
        while self.num_transitions + len(transitions) > self.maxlen:
            self._num_transitions -= len(self.buffer[0])
            self._num_episodes -= 1
            del self.buffer[0]

        self.buffer.append(transitions)
        self._num_transitions += len(transitions)
        self._num_episodes += 1


###############################################################################

class HERDDPG(object):
    """Hindsight Experience Replay Agent that uses DDPG as the
    off-policy RL algorithm.

    :param replay_k: The ratio between HER replays and regular replays,
        e.g. k = 4 -> 4 times as many HER replays as regular replays.
    :param memory_size: Number of episodes to store (best effort).
    """

    def __init__(self,
                 env,
                 eps_greedy=0.2,
                 gamma=.95,
                 tau=1e-3,
                 replay_k=2,
                 batch_size=256,
                 demo_batch_size=128,
                 reward_scale=1.0,
                 replay_buffer_episodes=10000,
                 replay_buffer_steps=100,
                 actor_hidden_layers=3,
                 actor_hidden_size=512,
                 actor_activation="relu",
                 actor_lr=3e-4,
                 critic_hidden_layers=3,
                 critic_hidden_size=512,
                 critic_activation="relu",
                 critic_lr=3e-4,
                 action_penalty=1.0,
                 prm_loss_weight=0.001,
                 aux_loss_weight=None,
                 normalize_observations=True,
                 normalize_observations_clip=5.0,
                 action_noise=0.2,
                 log_dir="her_ddpg_log"):
        self.env = env
        self.eps_greedy = eps_greedy
        self.gamma = gamma
        self.tau = tau

        action_dim = self.env.action_space.shape[0]
        goal_dim = self.env.observation_space["desired_goal"].shape[0]
        obs_dim = self.env.observation_space["observation"].shape[0]
        state_dim = obs_dim + goal_dim

        self.replay_k = replay_k
        self.batch_size = batch_size
        self.demo_batch_size = demo_batch_size
        self.reward_scale = reward_scale
        self.replay_buffer = HerReplayBuffer(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_episodes=replay_buffer_episodes,
            max_steps=replay_buffer_steps)

        # Build model
        # DDPG uses a simple actor critic (A2C) architecture
        self.actor = HerActor(
            state_dim, action_dim,
            hidden_layers=actor_hidden_layers,
            hidden_size=actor_hidden_size,
            activation=actor_activation).to(_DEVICE)
        self.target_actor = HerActor(
            state_dim, action_dim,
            hidden_layers=actor_hidden_layers,
            hidden_size=actor_hidden_size,
            activation=actor_activation).to(_DEVICE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.critic = HerCritic(
            state_dim, action_dim, 1,
            hidden_layers=critic_hidden_layers,
            hidden_size=critic_hidden_size,
            activation=critic_activation).to(_DEVICE)
        self.target_critic = HerCritic(
            state_dim, action_dim, 1,
            hidden_layers=critic_hidden_layers,
            hidden_size=critic_hidden_size,
            activation=critic_activation).to(_DEVICE)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)

        self._actor_hidden_layers = actor_hidden_layers
        self._actor_hidden_size = actor_hidden_size
        self._actor_activation = actor_activation
        self._actor_lr = actor_lr
        self._critic_hidden_layers = critic_hidden_layers
        self._critic_hidden_size = critic_hidden_size
        self._critic_activation = critic_activation
        self._critic_lr = critic_lr
        self._action_penalty = action_penalty
        self._prm_loss_weight = prm_loss_weight
        if aux_loss_weight is None:
            self._aux_loss_weight = 1.0 / demo_batch_size
        else:
            self._aux_loss_weight = aux_loss_weight

        # Normalizer
        self._normalize_observations_clip = normalize_observations_clip
        if normalize_observations:
            self.obs_normalizer = StandardScaler(
                n_features=obs_dim,
                clip_range=normalize_observations_clip)
            self.goal_normalizer = StandardScaler(
                n_features=goal_dim,
                clip_range=normalize_observations_clip)
        else:
            self.obs_normalizer = None
            self.goal_normalizer = None

        # Noise (normal noise with std equal to 5% of the total range)
        action_space_range = (self.actor.action_space.high -
                              self.actor.action_space.low)

        self.action_noise_arg = action_noise
        if action_noise > 0.0:
            self.action_noise = NormalActionNoise(
                mu=np.zeros(action_dim),
                sigma=action_noise * action_space_range)
        else:
            self.action_noise = None

        # Demonstration replay buffer
        self._demo_replay_buffer = None

        # Training state
        self.total_steps = 0
        self.num_episodes = 0
        self.episode_steps = 0
        self.train_steps = 0
        self._train_mode = True
        self._summary_w = tensorboard.SummaryWriter(log_dir=log_dir)

    def load_demonstrations(self, demo_path):
        """Loads a .npz file with 3 components 'acs', 'obs' and 'info'.

         + acs: are the actions taken by the agent as given to step(...).
         + obs: are the states returned by reset() and step(...).
         + info: are the info objects returne by step(...).

        Note: There should always be one more 'obs' than 'acs' and 'info'.

        :param demo_path: Path to the .npz file with the data to build the
            demonstration replay buffer.
        """
        demos = np.load(demo_path, allow_pickle=True)
        observations = demos["obs"]
        actions = demos["acs"]
        infos = demos["info"]
        num_episodes = min(len(observations), len(actions), len(infos))

        temp_rb = HerReplayBuffer(self.replay_buffer.obs_dim,
                                  self.replay_buffer.action_dim,
                                  self.replay_buffer.goal_dim,
                                  max_episodes=num_episodes,
                                  max_steps=self.replay_buffer.max_steps)

        for obs, acs, info in zip(observations, actions, infos):
            states = obs[:-1]
            next_states = obs[1:]

            # Ignore demonstrations with too many steps
            if len(acs) > temp_rb.max_steps:
                continue

            transitions = zip(states, acs, next_states, info)
            for state, action, next_state, info in transitions:
                reward = self.env.compute_reward(next_state["achieved_goal"],
                                                 next_state["desired_goal"],
                                                 info)
                action = self._to_actor_space(action)
                temp_rb.add(obs=state["observation"],
                            action=action,
                            next_obs=next_state["observation"],
                            reward=reward,
                            terminal=info["is_success"],
                            goal=next_state["desired_goal"],
                            achieved_goal=next_state["achieved_goal"])
            temp_rb.save_episode()

        _LOG.debug("(HER-TD3) Demonstratoin replay buffer size")
        _LOG.debug("(HER-TD3)     Num. Episodes: %d", temp_rb.num_episodes)
        _LOG.debug("(HER-TD3)     Num. Steps: %d", temp_rb.count_steps())
        if temp_rb.count_steps() < self.demo_batch_size:
            raise ValueError("demonstrations replay buffer has less steps than"
                             " `demo_batch_size`")
        self._demo_replay_buffer = temp_rb

    def set_eval_mode(self):
        """Sets the agent in evaluation mode."""
        self.set_train_mode(mode=False)

    def set_train_mode(self, mode=True):
        """Sets the agent training mode."""
        self.actor.train(mode=mode)
        self.critic.train(mode=mode)
        self._train_mode = mode

    def reset(self):
        """Resets the agent to run another episode.

        .. note:: Only necessary when calling update(...)
        """
        self.num_episodes += 1
        self.episode_steps = 0
        if self.action_noise is not None:
            self.action_noise.reset()

    def end_episode(self):
        """This function must be called at the end of a training episode
        to let the agent prepare for training.

        .. note:: Only necessary when calling update(...)
        """
        self.replay_buffer.save_episode()

    def update(self, state, action, reward, next_state, done):
        self.total_steps += 1
        self.episode_steps += 1

        # register observation into normalizer
        if self.obs_normalizer:
            self.obs_normalizer.update(state["observation"])
            self.goal_normalizer.update(state["desired_goal"])
            # self.goal_normalizer.update(state["achieved_goal"])

        # re-scale action
        return self._to_actor_space(action)

        self.replay_buffer.add(
            obs=state["observation"],
            action=action,
            next_obs=next_state["observation"],
            reward=reward,
            terminal=done,
            goal=next_state["desired_goal"],
            achieved_goal=next_state["achieved_goal"])

    def train(self, steps, progress=False):
        """Trains the agent using the transitions stored during exploration.

        :param step: The number of training steps. It should be greater than
            `policy_delay` otherwise the policy will not be trained.
        :param progress: If true, progress will be printed on the terminal.
        """
        assert self._train_mode
        if self.replay_buffer.count_steps() >= self.batch_size:
            if progress:
                t_steps = tqdm.trange(steps, desc="Step", dynamic_ncols=True)
            else:
                t_steps = range(steps)

            for i in t_steps:
                self._train()
                self.train_steps += 1

    @torch.no_grad()
    def compute_action(self, state):
        # Pre-process
        obs = torch.from_numpy(state["observation"]).float()
        goal = torch.from_numpy(state["desired_goal"]).float()
        if self.obs_normalizer:
            obs = self.obs_normalizer.transform(obs)
            goal = self.goal_normalizer.transform(goal)
        obs = obs.unsqueeze_(0).to(_DEVICE)
        goal = goal.unsqueeze_(0).to(_DEVICE)

        # Compute action
        if self._train_mode:
            if np.random.random_sample() < self.eps_greedy:
                action = np.random.uniform(low=self.actor.action_space.low,
                                           high=self.actor.action_space.high)
            else:
                action = self.actor(obs, goal).cpu().squeeze_(0).numpy()
                action = np.clip(action + self.action_noise(),
                                 self.actor.action_space.low,
                                 self.actor.action_space.high)
        else:
            action = self.actor(obs, goal).cpu().squeeze_(0).numpy()

        return self._to_action_space(action)

    # Agent training
    ##########################

    def _train(self):
        batch, demo_mask = self._sample_batch()
        (obs, action, next_obs, reward, terminal, goal, _) = batch

        if self.obs_normalizer:
            obs = self.obs_normalizer.transform(obs)
            next_obs = self.obs_normalizer.transform(next_obs)
            goal = self.goal_normalizer.transform(goal)

        # Compute critic loss
        with torch.no_grad():
            next_action = self.target_actor(next_obs, goal)
            target_q = self.target_critic(next_obs, goal, next_action)
            target_q = (1 - terminal.int()) * self.gamma + target_q
            target_q += self.reward_scale + reward

        # Optimize critic
        current_q = self.critic(obs, goal, action)
        loss_q = F.smooth_l1_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_out = self.actor(obs, goal)
        pi_loss = -self.critic(obs, goal, actor_out).mean()
        pi_loss += actor_out.pow(2).mean()
        if demo_mask.any():
            cloning_loss = (actor_out[demo_mask] - action[demo_mask])
            prm_loss = self._prm_loss_weight * pi_loss
            aux_loss = self._aux_loss_weight * cloning_loss.pow(2).sum()
            pi_loss = prm_loss + aux_loss

            self._summary_w.add_scalars("Loss", {"Actor_PRM": prm_loss,
                                                 "Actor_AUX": aux_loss},
                                        self.train_steps)

        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()

        self._update_target_networks()
        self._summary_w.add_scalar("Loss/Actor", pi_loss, self.train_steps)

    def _sample_batch(self):
        def _sample_reward_fn(achieved_goals, goals):
            info = {}
            return np.array([self.env.compute_reward(x, y, info)
                            for x, y in zip(achieved_goals, goals)])

        has_demo = self._demo_replay_buffer is not None
        exp_size = self.batch_size - (self.demo_batch_size * has_demo)

        batch = self.replay_buffer.sample_batch_torch(
            sample_size=exp_size, replay_k=self.replay_k,
            reward_fn=_sample_reward_fn, device=_DEVICE)
        if has_demo and self.demo_batch_size > 0:
            demo_batch = self._demo_replay_buffer.sample_batch_torch(
                sample_size=self.demo_batch_size, replay_k=0,
                reward_fn=_sample_reward_fn, device=_DEVICE)
            batch = tuple(torch.cat((x, y), dim=0)
                          for x, y in zip(batch, demo_batch))

        exp_mask = torch.zeros(exp_size, dtype=torch.bool)
        demo_mask = torch.ones(self.demo_batch_size, dtype=torch.bool)
        return batch, torch.cat((exp_mask, demo_mask), dim=0).to(_DEVICE)

    def _update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

    # Utilities
    ########################

    def _to_actor_space(self, action):
        return umath.scale(x=action,
                           min_x=self.env.action_space.low,
                           max_x=self.env.action_space.high,
                           min_out=self.actor.action_space.low,
                           max_out=self.actor.action_space.high)

    def _to_action_space(self, action):
        return umath.scale(x=action,
                           min_x=self.actor.action_space.low,
                           max_x=self.actor.action_space.high,
                           min_out=self.env.action_space.low,
                           max_out=self.env.action_space.high)

    # Save/Load Agent
    ########################

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
            ('eps_greedy', self.eps_greedy),
            ('gamma', self.gamma),
            ('tau', self.tau),
            ('replay_k', self.replay_k),
            ('batch_size', self.batch_size),
            ('demo_batch_size', self.demo_batch_size),
            ('reward_scale', self.reward_scale),
            ('replay_buffer_episodes', self.replay_buffer.max_episodes),
            ('replay_buffer_steps', self.replay_buffer.max_steps),
            ('actor_hidden_layers', self._actor_hidden_layers),
            ('actor_hidden_size', self._actor_hidden_size),
            ('actor_activation', self._actor_activation),
            ('actor_lr', self._actor_lr),
            ('critic_hidden_layers', self._critic_hidden_layers),
            ('critic_hidden_size', self._critic_hidden_size),
            ('critic_activation', self._critic_activation),
            ('critic_lr', self._critic_lr),
            ('action_penalty', self._action_penalty),
            ('prm_loss_weight', self._prm_loss_weight),
            ('aux_loss_weight', self._aux_loss_weight),
            ('normalize_observations', self.obs_normalizer is not None),
            ('normalize_observations_clip', self._normalize_observations_clip),
            ('action_noise', self.action_noise_arg),
            ('log_dir', self._summary_w.log_dir)
        ])

        state = {
            'total_steps': self.total_steps,
            'num_episodes': self.num_episodes,
            'train_steps': self.train_steps
        }

        pickle.dump(args, open(os.path.join(path, 'args.pickle'), 'wb'))
        pickle.dump(state, open(os.path.join(path, "state.pickle"), 'wb'))

        torch.save(self.actor.state_dict(),
                   os.path.join(path, 'actor.torch'))
        torch.save(self.actor_optimizer.state_dict(),
                   os.path.join(path, "actor_optimizer.torch"))

        torch.save(self.critic.state_dict(),
                   os.path.join(path, 'critic.torch'))
        torch.save(self.critic_optimizer.state_dict(),
                   os.path.join(path, 'critic_optimizer.torch'))

        if replay_buffer:
            self.replay_buffer.save(os.path.join(path, 'replay_buffer.h5'))

        if self.obs_normalizer is not None:
            self.obs_normalizer.save(os.path.join(path, 'obs_normalizer'))
            self.goal_normalizer.save(os.path.join(path, 'goal_normalizer'))

    @classmethod
    def load(cls, path, env, replay_buffer=True, **kwargs):
        if not os.path.isdir(path):
            raise ValueError("{} is not a directory".format(path))

        with open(os.path.join(path, "args.pickle"), "rb") as fh:
            _LOG.debug("(HER-DDPG) Loading agent arguments")
            args_values = pickle.load(fh)
            args_values.update(kwargs)

            fmt_string = "    {{:>{}}}: {{}}".format(
                max(len(x) for x in args_values.keys()))
            for key, value in args_values.items():
                _LOG.debug(fmt_string.format(key, value))

        # Create instance and load the rest of the data
        instance = cls(env, **args_values)

        with open(os.path.join(path, "state.pickle"), "rb") as fh:
            _LOG.debug("(HER-DDPG) Loading agent state")
            state = pickle.load(fh)
            instance.total_steps = state['total_steps']
            instance.num_episodes = state['num_episodes']
            instance.train_steps = state['train_steps']

        _LOG.debug("(HER-DDPG) Loading actor")
        actor_state = torch.load(os.path.join(path, "actor.torch"))
        instance.actor.load_state_dict(actor_state)
        instance.target_actor.load_state_dict(actor_state)
        instance.actor_optimizer.load_state_dict(
           torch.load(os.path.join(path, "actor_optimizer.torch")))

        _LOG.debug("(HER-DDPG) Loading critic")
        critic_state = torch.load(os.path.join(path, "critic.torch"))
        instance.critic.load_state_dict(critic_state)
        instance.target_critic.load_state_dict(critic_state)
        instance.critic_optimizer.load_state_dict(
           torch.load(os.path.join(path, "critic_optimizer.torch")))

        replay_buffer_path = os.path.join(path, "replay_buffer.h5")
        if replay_buffer and os.path.isfile(replay_buffer_path):
            _LOG.debug("(HER-DDPG) Loading replay buffer")
            instance.replay_buffer.load(replay_buffer_path)

        if instance.obs_normalizer is not None:
            _LOG.debug("(HER-DDPG) Loading observation normalizer")
            instance.obs_normalizer = StandardScaler.load(
                os.path.join(path, 'obs_normalizer'))
            _LOG.debug("(HER-DDPG) Loading goal normalizer")
            instance.goal_normalizer = StandardScaler.load(
                os.path.join(path, 'goal_normalizer'))

        return instance
