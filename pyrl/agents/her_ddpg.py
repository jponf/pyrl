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

from .core import HerAgent
from .replay_buffer import HerReplayBuffer
from .utils import (create_action_noise, create_normalizer,
                    create_actor, create_critic, dicts_mean)


###############################################################################

_DEVICE = "cpu"
_LOG = pyrl.util.logging.get_logger()


###############################################################################

class HerDDPG(HerAgent):
    """Hindsight Experience Replay Agent that uses Deep Deterministic Policy
    Gradient (DDPG) as the off-policy RL algorithm.

    Introduced in the papers:
        HER: Hindsight Experience Replay.
        DDPG: Continuous Control With Deep Reinforcement Learning.
    """

    def __init__(self,
                 env,
                 eps_greedy=0.2,
                 gamma=0.95,
                 tau=0.005,
                 batch_size=128,
                 reward_scale=1.0,
                 replay_buffer_episodes=10000,
                 replay_buffer_steps=100,
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
                 observation_normalizer="none",
                 observation_clip=float('inf'),
                 action_noise="normal_0.2"):
        """
        :param env: OpenAI's GoalEnv instance.
        :param float eps_greedy: Probability of picking a random action
            in training mode.
        param gamma: Bellman's discount rate.
        :type gamma: float
        :param tau: Used to perform "soft" updates (polyak averaging) of the
            weights from the actor/critic to their "target" counterparts.
        :type tau: float
        :param batch_size: Size of the sample used to train the actor and
            critic.
        :type batch_size: int
        :param replay_buffer_size: Number of transitions to store in the replay
            buffer.
        :type replay_buffer_size: int
        :param policy_delay: Number of times the critic networks are trained
            before training the policy network.
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
        :param action_noise: Name and standard deviaton of the action noise
            expressed as name_std, i.e., ou_0.2 or normal_0.1. Use "none" to
            disable the use of action nois
        """
        super(HerDDPG, self).__init__(env)
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
            max_steps=replay_buffer_steps)

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
        actors, critics = _build_ac(self.env.observation_space,
                                    self.env.action_space,
                                    actor_cls, actor_kwargs,
                                    critic_cls, critic_kwargs)
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
        self.obs_normalizer = create_normalizer(
            observation_normalizer,
            self.env.observation_space["observation"].shape,
            clip_range=observation_clip)
        self.goal_normalizer = create_normalizer(
            observation_normalizer,
            self.env.observation_space["desired_goal"].shape,
            clip_range=observation_clip)

        # Noise
        self._action_noise_arg = action_noise
        self.action_noise = create_action_noise(action_noise,
                                                self.env.action_space)

        # Demonstration replay buffer
        self._demo_replay_buffer = None

    # HerAgent methods
    ##########################

    def load_demonstrations(self, demo_path):
        demos = np.load(demo_path, allow_pickle=True)
        d_obs, d_acs, d_info = demos["obs"], demos["acs"], demos["info"]
        num_episodes = min(len(d_obs), len(d_acs), len(d_info))

        buffer = HerReplayBuffer(
            obs_shape=self.replay_buffer.obs_shape,
            goal_shape=self.replay_buffer.goal_shape,
            action_shape=self.replay_buffer.action_shape,
            max_episodes=num_episodes, max_steps=self.replay_buffer.max_steps)

        for obs, acs, info in six.moves.zip(d_obs, d_acs, d_info):
            if len(acs) > buffer.max_steps:  # too many steps, ignore
                continue

            states, next_states = obs[:-1], obs[1:]
            transitions = six.moves.zip(states, acs, next_states, info)
            for state, action, next_state, info in transitions:
                reward = self.env.compute_reward(next_state["achieved_goal"],
                                                 next_state["desired_goal"],
                                                 info)
                buffer.add(obs=state["observation"],
                           action=self._to_actor_space(action),
                           next_obs=next_state["observation"],
                           reward=reward,
                           terminal=info.get("is_success", False),
                           goal=next_state["desired_goal"],
                           achieved_goal=next_state["achieved_goal"])
            buffer.save_episode()

        _LOG.debug("(HER-DDPG) Loaded Demonstrations")
        _LOG.debug("(HER-DDPG)     = Num. Episodes: %d", buffer.num_episodes)
        _LOG.debug("(HER-DDPG)     = Num. Steps: %d", buffer.count_steps())

        if buffer.count_steps() < self._demo_batch_size:
            raise ValueError("demonstrations replay buffer has less than"
                             " `demo_batch_size` steps")

        self._demo_replay_buffer = buffer

    # BaseAgent methods
    ##########################

    def set_train_mode(self, mode=True):
        super(HerDDPG, self).set_train_mode(mode)
        self.actor.train(mode=mode)
        self.critic.train(mode=mode)
        self._train_mode = mode

    def begin_episode(self):
        self.action_noise.reset()

    def end_episode(self):
        self.replay_buffer.save_episode()

    def update(self, state, action, reward, next_state, terminal):
        self.obs_normalizer.update(state["observation"])
        self.goal_normalizer.update(state["desired_goal"])

        self.replay_buffer.add(obs=state["observation"],
                               action=self._to_actor_space(action),
                               next_obs=next_state["observation"],
                               reward=reward,
                               terminal=terminal,
                               goal=next_state["desired_goal"],
                               achieved_goal=next_state["achieved_goal"])

    @torch.no_grad()
    def compute_action(self, state):
        # Pre-process
        obs = torch.from_numpy(state["observation"]).float()
        goal = torch.from_numpy(state["desired_goal"]).float()
        obs = self.obs_normalizer.transform(obs).unsqueeze_(0).to(_DEVICE)
        goal = self.goal_normalizer.transform(goal).unsqueeze_(0).to(_DEVICE)

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

    def train(self, steps, progress=False):
        if self.replay_buffer.count_steps() >= self.batch_size:
            super(HerDDPG, self).train(steps, progress)

    def _train(self):
        batch, demo_mask = self._sample_batch()
        (obs, action, next_obs, reward, terminal, goal, _) = batch

        obs = self.obs_normalizer.transform(obs)
        next_obs = self.obs_normalizer.transform(next_obs)
        goal = self.goal_normalizer.transform(goal)

        with torch.no_grad():
            next_action = self.target_actor(next_obs, goal)
            target_q = self.target_critic(next_obs, goal, next_action)
            target_q = (1 - terminal.int()) * self.gamma * target_q
            target_q += self.reward_scale * reward

        # Optimize critic
        current_q = self.critic(obs, goal, action)
        loss_q = F.smooth_l1_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        self._summary.add_scalars("Q", {"Q": current_q.mean(),
                                        "Target": target_q.mean()},
                                  self._train_steps)
        self._summary.add_scalar("Loss/Critic", loss_q, self._train_steps)

        # Actor loss
        actor_out = self.actor(obs, goal)
        actor_q = self.critic(obs, goal, actor_out)

        pi_loss = -actor_q.mean()
        pi_loss += self._action_penalty * actor_out.pow(2).mean()
        if demo_mask.any():
            cloning_loss = (actor_out[demo_mask] - action[demo_mask])
            if self._q_filter:
                q_mask = (current_q[demo_mask] > actor_q[demo_mask]).flatten()
                cloning_loss = cloning_loss[q_mask]

            prm_loss = self._prm_loss_weight * pi_loss
            aux_loss = self._aux_loss_weight * cloning_loss.pow(2).sum()
            pi_loss = prm_loss + aux_loss

            self._summary.add_scalars("Loss", {"Actor_PRM": prm_loss,
                                               "Actor_AUX": aux_loss},
                                      self._train_steps)

        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()

        self._update_target_networks()
        self._summary.add_scalar("Loss/Actor", pi_loss, self._train_steps)

    def _sample_batch(self):
        def _sample_reward_fn(achieved_goals, goals):
            return self.env.compute_reward(achieved_goals, goals, None)

        has_demo = (self._demo_replay_buffer is not None and
                    self._demo_batch_size > 0)
        demo_batch_size = has_demo * self._demo_batch_size

        batch = self.replay_buffer.sample_batch_torch(
            sample_size=self.batch_size, replay_k=self._replay_k,
            reward_fn=_sample_reward_fn, device=_DEVICE)

        if has_demo:
            demo_batch = self._demo_replay_buffer.sample_batch_torch(
                sample_size=demo_batch_size, replay_k=0,
                reward_fn=_sample_reward_fn, device=_DEVICE)
            batch = tuple(torch.cat((x, y), dim=0)
                          for x, y in six.moves.zip(batch, demo_batch))

        exp_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        demo_mask = torch.ones(demo_batch_size, dtype=torch.bool)
        return batch, torch.cat((exp_mask, demo_mask), dim=0).to(_DEVICE)

    def _update_target_networks(self):
        a_params = six.moves.zip(self.target_actor.parameters(),
                                 self.actor.parameters())
        c_params = six.moves.zip(self.target_critic.parameters(),
                                 self.critic.parameters())

        for params in (a_params, c_params):
            for target_param, param in params:
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(param.data * self.tau)

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

    # Agent State
    ########################

    def state_dict(self):
        state = {"critic": self.critic.state_dict(),
                 "actor": self.actor.state_dict(),
                 "obs_normalizer": self.obs_normalizer.state_dict(),
                 "goal_normalizer": self.goal_normalizer.state_dict(),
                 "train_steps": self._train_steps}

        return state

    def load_state_dict(self, state):
        self.critic.load_state_dict(state['critic'])
        self.target_critic.load_state_dict(state['critic'])
        self.actor.load_state_dict(state["actor"])
        self.target_actor.load_state_dict(state["actor"])

        self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        self.goal_normalizer.load_state_dict(state["goal_normalizer"])

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
        self.goal_normalizer.load_state_dict([x['goal_normalizer']
                                              for x in states])

        self._train_steps = max(x["train_steps"] for x in states)

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
            ('batch_size', self.batch_size),
            ('reward_scale', self.reward_scale),
            ('replay_buffer_episodes', self.replay_buffer.max_episodes),
            ('replay_buffer_steps', self.replay_buffer.max_steps),

            # HER args
            ('replay_k', self._replay_k),
            ('demo_batch_size', self._demo_batch_size),
            ('q_filter', self._q_filter),
            ('action_penalty', self._action_penalty),
            ('prm_loss_weight', self._prm_loss_weight),
            ('aux_loss_weight', self._aux_loss_weight),

            # Actor-Critic
            ('actor_cls', type(self.actor)),
            ('actor_kwargs', self._actor_kwargs),
            ('actor_lr', self._actor_lr),
            ('critic_cls', type(self.critic)),
            ('critic_kwargs', self._critic_kwargs),
            ('critic_lr', self._critic_lr),

            # Normalize
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
    def load(cls, path, env, replay_buffer=True, **kwargs):
        if not os.path.isdir(path):
            raise ValueError("{} is not a directory".format(path))

        with open(os.path.join(path, "args.pkl"), "rb") as fh:
            _LOG.debug("(HER-DDPG) Loading agent arguments")
            args_values = pickle.load(fh)
            args_values.update(kwargs)

            fmt_string = "    {{:>{}}}: {{}}".format(
                max(len(x) for x in args_values.keys()))
            for key, value in args_values.items():
                _LOG.debug(fmt_string.format(key, value))

        # Create instance and load the rest of the data
        instance = cls(env, **args_values)

        with open(os.path.join(path, "state.pkl"), "rb") as fh:
            _LOG.debug("(HER-TD3) Loading agent state")
            state = pickle.load(fh)
            instance.load_state_dict(state)

        replay_buffer_path = os.path.join(path, "replay_buffer.h5")
        if replay_buffer and os.path.isfile(replay_buffer_path):
            _LOG.debug("(HER-TD3) Loading replay buffer")
            instance.replay_buffer.load(replay_buffer_path)

        return instance


###############################################################################

def _build_ac(observation_space, action_space, actor_cls, actor_kwargs,
              critic_cls, critic_kwargs):
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
