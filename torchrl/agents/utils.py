# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

# SciPy
import numpy as np

# HDF5
import h5py

# Torch
import torch

# ...
from .noise import NormalActionNoise, OUActionNoise


###############################################################################

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################

_ACTIVATIONS = {
    "leakyrelu": torch.nn.LeakyReLU,
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh
}


def get_activation_layer(name):
    """Get an activation layer given its name.

    :param name: Name of the activation layer, valid values are: leakyrelu,
        relu, sigmoid and tanh.
    """
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        msg = "invalid layer '{}', valid options are: {}"
        raise ValueError(
            msg.format(name, ", ".join(sorted(_ACTIVATIONS.keys()))))


_ACTION_NOISES = {
    "ou": OUActionNoise,
    "normal": NormalActionNoise
}


def get_action_noise(name, action_space):
    """Get an action noise given its name and standard deviation.

    :param name: Name and standard deviation in the format <name>_<stddev>,
        for example: ou_0.2 or normal_0.1.
    """
    noise, stddev = name.split('_')
    try:
        stddev = float(stddev)
    except ValueError:
        raise ValueError("unable to parse standard deviation value,"
                         " expected <noise>_<stddev>")

    try:
        action_range = action_space.high - action_space.low
        return _ACTION_NOISES[noise](
            mu=np.zeros(action_space.shape),
            sigma=action_range * stddev)
    except KeyError:
        raise ValueError("unknown noise type '{}'".format(name))


def make_tensor(value):
    """Creates a tensor from `value` if it is not one already.

    If `value` is not a Tensor, it uses `torch.from_numpy` when it is
    a numpy array and `torch.tensor` otherwise. If `value` is a Tensor
    `value` itself is returned.
    """
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif not isinstance(value, torch.Tensor):
        return torch.tensor(value)

    return value


def dicts_mean(dicts):
    """Computes the mean of multiple dictionaries.
    """

    if not all(dicts[0].keys() == d.keys() for d in dicts):
        raise ValueError('All dictionaries must have the same keys')

    keys = dicts[0].keys()
    elems = len(dicts)
    return {k: sum([d[k] for d in dicts]) / elems for k in keys}


###############################################################################

class ReplayBuffer(object):
    """Replay buffer that stores transitions from one state to another
    after taking an action, as well as the reward for performing the
    action and whether or not the resulting state was terminal or not.
    """

    def __init__(self, state_dim, action_dim,
                 max_size=500000,
                 dtype=np.float32,
                 rand=None):
        self.state = np.empty((max_size, state_dim), dtype=dtype)
        self.action = np.empty((max_size, action_dim), dtype=dtype)
        self.next_state = np.empty((max_size, state_dim), dtype=dtype)
        self.reward = np.empty((max_size, 1), dtype=dtype)
        self.terminal = np.empty((max_size, 1), dtype=np.bool)

        self._max_size = max_size
        self._index = 0
        self._size = 0

        if rand is not None:
            self.rand = rand
        else:
            self.rand = np.random.RandomState()

    def __getitem__(self, indices):
        if isinstance(indices, list):
            indices = np.array(indices)

        if isinstance(indices, np.ndarray):
            if (indices < 0).any() or (indices >= self._size).any():
                raise IndexError("indices out of replay buffer bounds")
        elif isinstance(indices, (int)):
            if 0 <= indices < self._size:
                raise IndexError("index out of replay buffer bounds")

        return (self.state[indices],
                self.action[indices],
                self.next_state[indices],
                self.reward[indices],
                self.terminal[indices])

    def __len__(self):
        return self._size

    def __repr__(self):
        fmt = ("ReplayBuffer(state_dim={!r}, action_dim={!r},"
               " max_size={!r}, dtype={!r})")
        return fmt.format(self.state.shape[1:], self.action.shape[1:],
                          self._max_size, self.state.dtype)

    def __str__(self):
        return repr(self)

    @property
    def max_size(self):
        return self._max_size

    def clear(self):
        """Removes all elements from the replay buffer."""
        self._index = 0
        self._size = 0

    def add(self, state, action, next_state, reward, terminal):
        """Adds the transition to the replay buffer."""
        self.state[self._index] = state
        self.action[self._index] = action
        self.next_state[self._index] = next_state
        self.reward[self._index] = reward
        self.terminal[self._index] = terminal

        self._index = (self._index + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample_batch(self, sample_size):
        """Samples a batch of size `sample_size`.

        :param sample_size: Size of the sampled batch.
        :return: A tuple containing the batch data as numpy
            arrays: (state, action, next_state, reward, terminal).
        """
        indices = self.rand.randint(low=0, high=self._size,
                                    size=sample_size)

        return (self.state[indices],
                self.action[indices],
                self.next_state[indices],
                self.reward[indices],
                self.terminal[indices])

    def sample_batch_torch(self, sample_size, device=None):
        """Samples a batch of size `sample_size`.

        :param sample_size: Size of the sampled batch.
        :param device: The device that must allocate the tensors.
        :return: A tuple containing the batch data as torch
            tensors: (state, action, next_state, reward, terminal).
        """
        device = device if device is not None else _DEVICE
        return (torch.from_numpy(x).to(device)
                for x in self.sample_batch(sample_size))

    def save(self, path):
        """Saves the replay buffer in hdf5 format into the file pointed
        by `path`.

        :param path: Path to the file where to save the replay buffer.
        """
        with h5py.File(path, "w") as h5f:
            meta = (self._max_size, self._index, self._size)
            h5f.create_dataset("meta", data=meta)
            h5f.create_dataset("state", data=self.state)
            h5f.create_dataset("action", data=self.action)
            h5f.create_dataset("next_state", data=self.next_state)
            h5f.create_dataset("reward", data=self.reward)
            h5f.create_dataset("terminal", data=self.terminal)

    def load(self, path):
        """Loads the replay buffer from the file pointed by `path`.

        :param path: Path to the file that contains a saved replay buffer.
        """
        with h5py.File(path, "r") as h5f:
            max_size, index, size = h5f["meta"]

            # compute copy size and copy "order"
            copy_size = min(size, self._max_size)
            if size > copy_size:           # old size is bigger than current
                index += size - copy_size  # capacity, correct index
            copy_indices = (np.arange(size) + index) % size
            copy_indices = copy_indices[:copy_size]

            # check that we copy into a view
            assert self.state[:copy_size].base is self.state
            assert self.action[:copy_size].base is self.action
            assert self.next_state[:copy_size].base is self.next_state
            assert self.reward[:copy_size].base is self.reward
            assert self.terminal[:copy_size].base is self.terminal

            # copy datasets into the numpy arrays
            temp = np.array(h5f["state"])
            np.copyto(self.state[:copy_size], temp[copy_indices])

            temp = np.array(h5f["action"])
            np.copyto(self.action[:copy_size], temp[copy_indices])

            temp = np.array(h5f["next_state"])
            np.copyto(self.next_state[:copy_size], temp[copy_indices])

            temp = np.array(h5f["reward"])
            np.copyto(self.reward[:copy_size], temp[copy_indices])

            temp = np.array(h5f["terminal"])
            np.copyto(self.terminal[:copy_size], temp[copy_indices])

            # set index and size accordingly
            self._index = copy_size % self._max_size
            self._size = copy_size


###############################################################################

class HerReplayBuffer(object):
    """Replay buffer structured and extended with additional functionallity
    required to implement the sampling techniques for Hindsight Experience
    Replay.

    :param obs_dim: Observation dimension.
    :param goal_dim: Goal dimension.
    :param action_dim: Action dimension.
    :param max_steps: Maximum number of steps per episode.
    :param max_episodes: Maximum number of episodes, if exceeded the oldest
        ones will be dropped.
    :param dtype: Type of the numerical buffers.
    :param rand: Numpy's random number generator.
    """

    def __init__(self, obs_dim, action_dim, goal_dim,
                 max_episodes, max_steps,
                 dtype=np.float32,
                 rand=None):
        assert max_steps > 0
        assert max_episodes > 0
        # Check memory size and warn?

        # Replay Buffer
        arr_dim = (max_episodes, max_steps)
        self.obs = np.empty(arr_dim + (obs_dim,), dtype=dtype)
        self.action = np.empty(arr_dim + (action_dim,), dtype=dtype)
        self.next_obs = np.empty(arr_dim + (obs_dim,), dtype=dtype)
        self.reward = np.empty(arr_dim + (1,), dtype=dtype)
        self.terminal = np.empty(arr_dim + (1,), dtype=np.bool)

        self.goal = np.empty(arr_dim + (goal_dim,), dtype=dtype)
        self.achieved_goal = np.empty(arr_dim + (goal_dim,), dtype=dtype)

        # Temporary buffer to store episode data
        self._obs = np.empty((max_steps, obs_dim), dtype=dtype)
        self._action = np.empty((max_steps, action_dim), dtype=dtype)
        self._next_obs = np.empty((max_steps, obs_dim), dtype=dtype)
        self._reward = np.empty((max_steps, 1), dtype=dtype)
        self._terminal = np.empty((max_steps, 1), dtype=np.bool)

        self._goal = np.empty((max_steps, goal_dim), dtype=dtype)
        self._achieved_goal = np.empty((max_steps, goal_dim), dtype=dtype)

        self._max_episodes = max_episodes
        self._max_steps = max_steps
        self._e_idx = 0
        self._s_idx = 0
        self._b_size = 0                                       # buffer size
        self._e_size = np.zeros(max_episodes, dtype=np.int32)  # episode size

        if rand is not None:
            self.rand = rand
        else:
            self.rand = np.random.RandomState()

    def __repr__(self):
        fmt = ("HerReplayBuffer(obs_dim={:!r}, action_dim={:!r},"
               " goal_dim={:!r}, max_episodes={:!r}, max_steps={:!r},"
               " dtype={:!r})")
        return fmt.format(self.obs.shape[1:], self.action.shape[1:],
                          self.goal.shape[1:], self._max_episodes,
                          self._max_steps, self.state.dtype)

    def __str__(self):
        return repr(self)

    @property
    def action_dim(self):
        return self.action.shape[-1]

    @property
    def goal_dim(self):
        return self.goal.shape[-1]

    @property
    def obs_dim(self):
        return self.obs.shape[-1]

    @property
    def max_episodes(self):
        return self._max_episodes

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def num_episodes(self):
        return self._b_size

    def count_steps(self):
        return self._e_size.sum()

    def clear(self):
        """Removes all elements from the replay buffer."""
        self._e_idx = 0
        self._s_idx = 0
        self._b_size = 0
        self._e_size = np.zeros(self._max_episodes)

    def add(self, obs, action, next_obs, reward, terminal,
            goal, achieved_goal):
        """Adds the transition to a temporary buffer.

        :raises IndexError: If the maximum number of steps per episode
            have already been added to the temporary buffer.
        """
        self._obs[self._s_idx] = obs
        self._action[self._s_idx] = action
        self._next_obs[self._s_idx] = next_obs
        self._reward[self._s_idx] = reward
        self._terminal[self._s_idx] = terminal

        self._goal[self._s_idx] = goal
        self._achieved_goal[self._s_idx] = achieved_goal

        self._s_idx += 1

    def get_episode(self, idx):
        """Gets all the transitions of an episode.

        :param idx: Episode index.

        :return: The transitions of an episode as a tuple of arrays
            (obs, actions, next_obs, rewards, terminal, goal, achieved_goal).
        """
        if idx < 0 or idx >= self._b_size:
            raise IndexError("index of out bounds")

        e_size = self._e_size[idx]
        return (self.obs[idx][:e_size],
                self.action[idx][:e_size],
                self.next_obs[idx][:e_size],
                self.reward[idx][:e_size],
                self.terminal[idx][:e_size],
                self.goal[idx][:e_size],
                self.achieved_goal[idx][:e_size])

    def save_episode(self):
        """Saves the temporary buffer into the replay buffer and clears it."""
        self.obs[self._e_idx] = self._obs
        self.action[self._e_idx] = self._action
        self.next_obs[self._e_idx] = self._next_obs
        self.reward[self._e_idx] = self._reward
        self.terminal[self._e_idx] = self._terminal

        self.goal[self._e_idx] = self._goal
        self.achieved_goal[self._e_idx] = self._achieved_goal

        self._e_size[self._e_idx] = self._s_idx
        self._b_size = min(self._b_size + 1, self._max_episodes)
        self._e_idx = (self._e_idx + 1) % self._max_episodes
        self._s_idx = 0

    def sample_batch(self, sample_size, replay_k, reward_fn, gamma, n_steps):
        """Samples a batch of size `sample_size`.

        There is only one HER sampling strategy, which is the 'future'
        strategy. This is the strategy that the authors reported has
        the best performance.

        :param sample_size: Size of the sampled batch.
        :param replay_k: Ratio between HER replays and regular replays,
            e.g: k = 4 -> 4 times as many HER replays as regular replays
        :return: A tuple containing the batch data as numpy
            arrays: (obs, action, next_obs, reward, terminal,
                     goal, achieved_goal).
        """
        future_p = 1.0 - (1.0 / (1.0 + replay_k))

        e_indices = self.rand.randint(low=0, high=self._b_size,
                                      size=sample_size)
        e_size = self._e_size[e_indices]
        s_indices = self.rand.random_sample(sample_size) * e_size
        s_indices = s_indices.astype(np.int32)

        # Create batch from episode and step indices
        obs = self.obs[e_indices, s_indices].copy()
        action = self.action[e_indices, s_indices].copy()
        next_obs = self.next_obs[e_indices, s_indices].copy()
        reward = self.reward[e_indices, s_indices].copy()
        terminal = self.terminal[e_indices, s_indices].copy()
        goal = self.goal[e_indices, s_indices].copy()
        achieved_goal = self.achieved_goal[e_indices, s_indices].copy()

        # Compute indices that will be replaced by HER goals
        her_indices = np.where(self.rand.random_sample(sample_size) < future_p)

        future_offset = (e_size - s_indices).astype(np.float64)
        future_offset *= self.rand.random_sample(sample_size)
        future_offset = future_offset.astype(np.int32)
        f_indices = (s_indices + future_offset)

        # Replace goal with achieved goal for previously-selected
        # HER transitions (her_indices). For the other transitions,
        # keep the original goal.
        f_achieved_goal = self.achieved_goal[e_indices[her_indices],
                                             f_indices[her_indices]]

        goal[her_indices] = f_achieved_goal

        # Recompute reward since we may have substituted the goal
        reward = reward_fn(achieved_goal, goal).reshape(reward.shape)

        # N-steps Q target
        gamma_ = np.tile(gamma, reward.shape).astype(self.reward.dtype)
        max_indices = e_size - 1
        max_indices[her_indices] = np.minimum(max_indices[her_indices],
                                              f_indices[her_indices])

        for i in range(1, n_steps):
            ns_indices = s_indices + i
            mask = ns_indices <= max_indices

            ns_indices = ns_indices[mask]
            ne_indices = e_indices[mask]
            n_reward = reward_fn(self.achieved_goal[ne_indices, ns_indices],
                                 goal[mask]).reshape(len(ne_indices), 1)

            reward[mask] += n_reward * gamma
            next_obs[mask] = self.next_obs[ne_indices, ns_indices]
            gamma_[mask] *= gamma

        return (obs, action, next_obs, reward, terminal,
                goal, achieved_goal, gamma_)

    def sample_batch_torch(self, sample_size, replay_k,
                           reward_fn, gamma, n_steps, device=None):
        """Samples a batch of size `sample_size` and transforms the
        values to torch tensors before returning them.

        For additional information see `sample_batch`.
        """
        device = device if device is not None else _DEVICE
        return tuple(torch.from_numpy(x).to(device)
                     for x in self.sample_batch(sample_size, replay_k,
                                                reward_fn, gamma, n_steps))

    def save(self, path):
        """Saves the replay buffer in hdf5 format into the file pointed
        by `path`.

        :param path: Path to the file where to save the replay buffer.
        """
        with h5py.File(path, "w") as h5f:
            meta = (self._max_episodes, self._max_steps,
                    self._e_idx, self._b_size)
            h5f.create_dataset("meta", data=meta)
            h5f.create_dataset("e_size", data=self._e_size)

            h5f.create_dataset("obs", data=self.obs)
            h5f.create_dataset("action", data=self.action)
            h5f.create_dataset("next_obs", data=self.next_obs)
            h5f.create_dataset("reward", data=self.reward)
            h5f.create_dataset("terminal", data=self.terminal)

            h5f.create_dataset("goal", data=self.goal)
            h5f.create_dataset("achieved_goal", data=self.achieved_goal)

    def load(self, path):
        """Loads the replay buffer from the file pointed by `path`.

        :param path: Path to the file that contains a saved replay buffer.
        """
        with h5py.File(path, "r") as h5f:
            max_episodes, max_steps, e_idx, b_size = h5f["meta"]
            e_size = np.array(h5f["e_size"])

            if self._max_steps < max_steps:
                raise ValueError("cannot load the replay buffer because it"
                                 " has too many steps per episode")

            # compute copy size and copy "order"
            copy_size = min(b_size, self._max_episodes)
            if b_size > copy_size:           # old size is bigger than current
                e_idx += b_size - copy_size  # capacity, correct index
            copy_indices = (np.arange(b_size) + e_idx) % b_size
            copy_indices = copy_indices[:copy_size]

            # check that we copy into a view
            assert self.obs[:copy_size].base is self.obs
            assert self.action[:copy_size].base is self.action
            assert self.next_obs[:copy_size].base is self.next_obs
            assert self.reward[:copy_size].base is self.reward
            assert self.terminal[:copy_size].base is self.terminal
            assert self.goal[:copy_size].base is self.goal
            assert self.achieved_goal[:copy_size].base is self.achieved_goal
            assert self._e_size[:copy_size].base is self._e_size

            # copy datasets into the numpy arrays
            temp = np.array(h5f["obs"])
            np.copyto(self.obs[:copy_size], temp[copy_indices])

            temp = np.array(h5f["action"])
            np.copyto(self.action[:copy_size], temp[copy_indices])

            temp = np.array(h5f["next_obs"])
            np.copyto(self.next_obs[:copy_size], temp[copy_indices])

            temp = np.array(h5f["reward"])
            np.copyto(self.reward[:copy_size], temp[copy_indices])

            temp = np.array(h5f["terminal"])
            np.copyto(self.terminal[:copy_size], temp[copy_indices])

            temp = np.array(h5f["goal"])
            np.copyto(self.goal[:copy_size], temp[copy_indices])

            temp = np.array(h5f["achieved_goal"])
            np.copyto(self.achieved_goal[:copy_size], temp[copy_indices])

            # set index and size accordingly
            self._e_idx = copy_size % self._max_episodes
            # self._s_idx = 0
            self._b_size = copy_size
            np.copyto(self._e_size[:copy_size], e_size[copy_indices])
