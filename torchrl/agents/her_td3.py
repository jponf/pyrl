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

# robotrl
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

class HerTD3(object):
    """Hindsight Experience Replay Agent that uses Twin Delayed Deep
    Deterministic Policy Gradient (TD3) as the off-policy RL algorithm.

    Introduced in the papers:
        HER: Hindsight Experience Replay.
        TD3: Addressing Function Approximation Error in Actor-Critic Methods.
    """

    def __init__(self,
                 env,
                 eps_greedy=0.2,
                 gamma=0.95,
                 tau=0.005,
                 n_step_q=1,
                 replay_k=2,
                 batch_size=256,
                 demo_batch_size=128,
                 policy_delay=2,
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
                 q_filter=False,
                 prm_loss_weight=0.001,
                 aux_loss_weight=None,
                 normalize_observations=True,
                 normalize_observations_clip=5.0,
                 action_noise=0.2,
                 smoothing_noise=True,
                 log_dir="her_td3_log"):
        """
        :param env: OpenAI's GoalEnv instance.
        :param float eps_greedy: Probability of picking a random action
            in training mode.
        :param float gamma: Bellman's discount rate.
        :param float tau: Used to perform "soft" updates of the weights from
            the actor/critic to their "target" counterparts.
        :param replay_k: The ratio between HER replays and regular replays,
            e.g. k = 4 -> 4 times as many HER replays as regular replays.
        :param int batch_size: Size of the sample, from the exploration buffer,
             used to train the agent.
        :param int demo_batch_size: Size of the sample, from the demonstration
            buffer, used to train the agent. This sample replaces entries
            from the batch to keep its size constant to `batch_size`.
        :param int replay_buffer_size: Size of the replay buffer.
        :param str actor_activation: Activation function used in the actor.
        :param float actor_lr: Learning rate for the actor network.
        :param str critic_activation: Activation function used in the critic.
        :param float critic_lr: Learning rate for the critic network.
        :param float action_noise: Standard deviation expressed as a fraction
            of the actions' range of values, a value in the range [0.0, 1.0].
            A value of 0 disables the use of action noise during training.
        :param float action_penalty: Quadratic penalty on actions to avoid
            tanh saturation and vanishing gradients (0 disables the penalty).
        :param float prm_loss_weight: Weight corresponding to the primary loss.
        :param float aux_loss_weight: Weight corresponding to the auxiliary
            loss, also called the cloning loss.
        :param bool normalize_observations: Whether or not observations should
            be normalized using the running mean and standard deviation before
            feeding them to the network.
        :param str log_dir: Directory to output tensorboard logs.
        """
        if not torchrl.util.ugym.is_her_env(env):
            raise ValueError("{} is not a valid HER environment".format(env))

        self.env = env
        self.eps_greedy = eps_greedy
        self.gamma = gamma
        self.tau = tau

        action_dim = self.env.action_space.shape[0]
        goal_dim = self.env.observation_space["desired_goal"].shape[0]
        obs_dim = self.env.observation_space["observation"].shape[0]
        state_dim = obs_dim + goal_dim

        self.n_step_q = n_step_q
        self.replay_k = replay_k
        self.batch_size = batch_size
        self.demo_batch_size = demo_batch_size
        self.policy_delay = policy_delay
        self.reward_scale = reward_scale
        self.replay_buffer = HerReplayBuffer(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_episodes=replay_buffer_episodes,
            max_steps=replay_buffer_steps)

        # Build model (A2C architecture)
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

        self.critic_1 = HerCritic(
            state_dim, action_dim, 1,
            hidden_layers=critic_hidden_layers,
            hidden_size=critic_hidden_size,
            activation=critic_activation).to(_DEVICE)
        self.target_critic_1 = HerCritic(
            state_dim, action_dim, 1,
            hidden_layers=critic_hidden_layers,
            hidden_size=critic_hidden_size,
            activation=critic_activation).to(_DEVICE)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_1.eval()

        self.critic_2 = HerCritic(
            state_dim, action_dim, 1,
            hidden_layers=critic_hidden_layers,
            hidden_size=critic_hidden_size,
            activation=critic_activation).to(_DEVICE)
        self.target_critic_2 = HerCritic(
            state_dim, action_dim, 1,
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
        self._action_penalty = action_penalty
        self._q_filter = q_filter
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

        if smoothing_noise:
            self.smoothing_noise = NormalActionNoise(
                mu=np.zeros(action_dim),
                sigma=0.1 * action_space_range,
                clip_min=np.repeat(-0.15, action_dim),
                clip_max=np.repeat(0.15, action_dim))
        else:
            self.smoothing_noise = None

        # Demonstration replay buffer
        self._demo_replay_buffer = None

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

    def reset(self):
        self.num_episodes += 1
        self.episode_steps = 0
        if self.action_noise is not None:
            self.action_noise.reset()

    def end_episode(self):
        """This function must be called at the end of a training episode
        to let the agent prepare for training.
        """
        self.replay_buffer.save_episode()

    def update(self, state, action, reward, next_state, done):
        self.total_steps += 1
        self.episode_steps += 1

        # register observation into normalizer
        if self.obs_normalizer:
            self.obs_normalizer.update(state["observation"])
            self.goal_normalizer.update(state["desired_goal"])

        # re-scale action
        action = self._to_actor_space(action)

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
                update_policy = (self.train_steps + 1) % self.policy_delay == 0
                self._train(update_policy)
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

    def _train(self, update_policy):
        batch, demo_mask = self._sample_batch()
        (obs, action, next_obs, reward, terminal, goal, _, gamma) = batch

        if self.obs_normalizer:
            obs = self.obs_normalizer.transform(obs)
            next_obs = self.obs_normalizer.transform(next_obs)
            goal = self.goal_normalizer.transform(goal)

        # Compute critic loss
        with torch.no_grad():
            next_action = self.target_actor(next_obs, goal).cpu().numpy()
            if self.smoothing_noise is not None:
                next_action += self.smoothing_noise()
                np.clip(next_action,
                        self.actor.action_space.low,
                        self.actor.action_space.high,
                        out=next_action)
            next_action = torch.from_numpy(next_action).to(_DEVICE)

            target_q1 = self.target_critic_1(next_obs, goal, next_action)
            target_q2 = self.target_critic_2(next_obs, goal, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            target_q = (1 - terminal.int()) * gamma * min_target_q
            target_q += self.reward_scale * reward

            self._summary_w.add_scalars(
                "TargetQ",
                {"Q1": target_q1.mean(),
                 "Q2": target_q2.mean(),
                 "T": target_q.mean(),
                 "Min": min_target_q.mean()},
                self.train_steps)

        # Optimize critic
        current_q1 = self.critic_1(obs, goal, action)
        loss_q1 = F.smooth_l1_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_1_optimizer.step()

        current_q2 = self.critic_2(obs, goal, action)
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
            actor_out = self.actor(obs, goal)
            actor_q1 = self.critic_1(obs, goal, actor_out)

            pi_loss = -actor_q1.mean() + actor_out.pow(2).mean()
            if demo_mask.any():
                cloning_loss = (actor_out[demo_mask] - action[demo_mask])
                if self._q_filter:  # where is the demonstation action better?
                    cloning_q1 = self.critic_1(obs, goal, action)

                    q_mask = cloning_q1[demo_mask] > actor_q1[demo_mask]
                    q_mask = q_mask.flatten()
                    cloning_loss = cloning_loss[q_mask]

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
            return self.env.compute_reward(achieved_goals, goals, None)

        has_demo = (self._demo_replay_buffer is not None and
                    self.demo_batch_size > 0)
        demo_size = self.demo_batch_size * has_demo
        exp_size = self.batch_size - demo_size

        batch = self.replay_buffer.sample_batch_torch(
            sample_size=exp_size, replay_k=self.replay_k,
            n_steps=self.n_step_q, gamma=self.gamma,
            reward_fn=_sample_reward_fn, device=_DEVICE)

        if has_demo:
            demo_batch = self._demo_replay_buffer.sample_batch_torch(
                sample_size=self.demo_batch_size, replay_k=0,
                n_steps=self.n_step_q, gamma=self.gamma,
                reward_fn=_sample_reward_fn, device=_DEVICE)
            batch = tuple(torch.cat((x, y), dim=0)
                          for x, y in zip(batch, demo_batch))

        exp_mask = torch.zeros(exp_size, dtype=torch.bool)
        demo_mask = torch.ones(demo_size, dtype=torch.bool)
        return batch, torch.cat((exp_mask, demo_mask), dim=0).to(_DEVICE)

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
            ('policy_delay', self.policy_delay),
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
            ('q_filter', self._q_filter),
            ('prm_loss_weight', self._prm_loss_weight),
            ('aux_loss_weight', self._aux_loss_weight),
            ('normalize_observations', self.obs_normalizer is not None),
            ('normalize_observations_clip', self._normalize_observations_clip),
            ('action_noise', self.action_noise_arg),
            ('smoothing_noise', self.smoothing_noise is not None),
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
        torch.save(self.actor_optimizer.state_dict(),
                   os.path.join(path, "actor_optimizer.torch"))

        torch.save(self.critic_1.state_dict(),
                   os.path.join(path, 'critic_1.torch'))
        torch.save(self.critic_1_optimizer.state_dict(),
                   os.path.join(path, 'critic_1_optimizer.torch'))

        torch.save(self.critic_2.state_dict(),
                   os.path.join(path, 'critic_2.torch'))
        torch.save(self.critic_2_optimizer.state_dict(),
                   os.path.join(path, 'critic_2_optimizer.torch'))

        if replay_buffer:
            self.replay_buffer.save(os.path.join(path, 'replay_buffer.h5'))

        if self.obs_normalizer:
            self.obs_normalizer.save(os.path.join(path, 'obs_normalizer'))
            self.goal_normalizer.save(os.path.join(path, 'goal_normalizer'))

    @classmethod
    def load(cls, path, env, replay_buffer=True, **kwargs):
        if not os.path.isdir(path):
            raise ValueError("{} is not a directory".format(path))

        # Load and Override arguments used to build the instance
        with open(os.path.join(path, "args.pickle"), "rb") as fh:
            _LOG.debug("(HER-TD3) Loading agent arguments")
            args_values = pickle.load(fh)
            args_values.update(kwargs)

            fmt_string = "    {{:>{}}}: {{}}".format(
                max(len(x) for x in args_values.keys()))
            for key, value in args_values.items():
                _LOG.debug(fmt_string.format(key, value))

        # Create instance and load the rest of the data
        instance = cls(env, **args_values)

        with open(os.path.join(path, "state.pickle"), "rb") as fh:
            _LOG.debug("(HER-TD3) Loading agent state")
            state = pickle.load(fh)
            instance.total_steps = state['total_steps']
            instance.num_episodes = state['num_episodes']
            instance.train_steps = state['train_steps']

        _LOG.debug("(HER-TD3) Loading actor")
        actor_state = torch.load(os.path.join(path, "actor.torch"))
        instance.actor.load_state_dict(actor_state)
        instance.target_actor.load_state_dict(actor_state)
        instance.actor_optimizer.load_state_dict(
           torch.load(os.path.join(path, "actor_optimizer.torch")))

        _LOG.debug("(HER-TD3) Loading critic 1")
        critic_1_state = torch.load(os.path.join(path, "critic_1.torch"))
        instance.critic_1.load_state_dict(critic_1_state)
        instance.target_critic_1.load_state_dict(critic_1_state)
        instance.critic_1_optimizer.load_state_dict(
           torch.load(os.path.join(path, "critic_1_optimizer.torch")))

        _LOG.debug("(HER-TD3) Loading critic 2")
        critic_2_state = torch.load(os.path.join(path, "critic_2.torch"))
        instance.critic_2.load_state_dict(critic_2_state)
        instance.target_critic_2.load_state_dict(critic_2_state)
        instance.critic_2_optimizer.load_state_dict(
           torch.load(os.path.join(path, "critic_2_optimizer.torch")))

        replay_buffer_path = os.path.join(path, "replay_buffer.h5")
        if replay_buffer and os.path.isfile(replay_buffer_path):
            _LOG.debug("(HER-TD3) Loading replay buffer")
            instance.replay_buffer.load(replay_buffer_path)

        if instance.obs_normalizer:
            _LOG.debug("(HER-TD3) Loading observation normalizer")
            instance.obs_normalizer = StandardScaler.load(
                os.path.join(path, 'obs_normalizer'))
            _LOG.debug("(HER-TD3) Loading goal normalizer")
            instance.goal_normalizer = StandardScaler.load(
                os.path.join(path, 'goal_normalizer'))

        return instance
