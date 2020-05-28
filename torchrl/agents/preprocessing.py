# -*- coding: utf-8 -*-

import errno
import os
import pickle

# Torch Stack
import torch

# HDF5
import h5py

# ...
import torchrl.agents.utils


###############################################################################
# Reference: https://github.com/openai/baselines/her/

class StandardScaler(object):
    """Standardize features by removing the mean and scaling to unit variance.

    The standrad score of a sample 'x' is calculated as:

    z = (x - u) / s

    where 'u' is the mean of the training samples, and 's' is the standard
    deviation of the samples seen so far.

    Standardization of a dataset is a common requirement for many machine
    learning estimators: they might behave badly if the individual features
    do not more or less look like standard normally distributed data (e.g.
    Gaussian with 0 mean and unit variance).
    """

    def __init__(self,
                 n_features,
                 epsilon=1e-4,
                 clip_range=float('inf')):
        """
        :param int n_features: Number of features in an individual sample.
        :param float epsilon: minimum std value.
        :param float clip_range: clip standardized values to be in
            range [-clip_range, clip_range] (default: inf tensor).
        """
        self.n_features = n_features
        self.epsilon = epsilon
        self.clip_range = torch.tensor(clip_range).float()

        self.sum = torch.zeros(self.n_features)
        self.sum_sq = torch.zeros(self.n_features)
        self.count = torch.zeros(1)
        self.mean = torch.zeros(self.n_features)
        self.std = torch.ones(self.n_features)

    def update(self, x):
        """Online computation of mean and std on 'x' for later scaling.

        :param x: The data used to compute the mean and standard deviation
            user for later scaling along the features axis.
        """
        x = torchrl.agents.utils.make_tensor(x).view(-1, self.n_features)

        self.sum += torch.sum(x, dim=0)
        self.sum_sq += torch.sum(x.pow(2), dim=0)
        self.count += x.shape[0]
        self._update_running_mean()

    def transform(self, x):
        """Perform standardization by centering and scaling."""
        mean = _reshape_broadcast(self.mean, x).to(x.device)
        std = _reshape_broadcast(self.std, x).to(x.device)
        return ((x - mean) / std).clamp(-self.clip_range, self.clip_range)

    def state_dict(self):
        """Generates a dictionary with the state of the normalizer."""
        return {"sum": self.sum, "sum_sq": self.sum_sq, "count": self.count}

    def load_state_dict(self, state):
        if isinstance(state, dict):
            self.sum = state["sum"]
            self.sum_sq = state["sum_sq"]
            self.count = state["count"]
        elif isinstance(state, list):
            self.sum = sum(o["sum"] for o in state) / len(state)
            self.sum_sq = sum(o["sum_sq"] for o in state) / len(state)
            self.count = sum(o["count"] for o in state) / len(state)
        else:
            raise TypeError('state must be either a dict or a list')

        self._update_running_mean()

    def _update_running_mean(self):
        if self.count > 0:
            self.mean = self.sum / self.count
            self.std = ((self.sum_sq / self.count) - self.mean.pow(2)).sqrt()
        else:
            self.mean = torch.zeros(self.n_features)
            self.std = torch.ones(self.n_features)

        # Avoid nan and zeros
        self.std[torch.isnan(self.std)] = self.epsilon
        self.std[self.std < self.epsilon] = self.epsilon

    def save(self, path):
        with h5py.File(path, "w") as h5f:
            h5f.create_dataset("n_features", data=self.n_features)
            h5f.create_dataset("epsilon", data=self.epsilon)
            h5f.create_dataset("clip_range", data=self.clip_range)
            h5f.create_dataset("sum", data=self.sum)
            h5f.create_dataset("sum_sq", data=self.sum_sq)
            h5f.create_dataset("count", data=self.count)

    @classmethod
    def load(cls, path):
        with h5py.File(path, "r") as h5f:
            instance = cls(n_features=h5f["n_features"][()],
                           epsilon=h5f["epsilon"][()],
                           clip_range=h5f["clip_range"][()])
            instance.load_state_dict(
                dict(sum=torch.as_tensor(h5f["sum"]),
                     sum_sq=torch.as_tensor(h5f["sum_sq"]),
                     count=torch.as_tensor(h5f["count"][()]))
            )
            return instance

    def __repr__(self):
        fmt = "StandardScaler(n_features={!r}, clip_range={!r})"
        return fmt.format(self.n_features, self.clip_range)

    def __str__(self):
        fmt = "StandardScaler {{\n    mean={}\n    std={}\n}}"
        return fmt.format(self.mean, self.std)


###############################################################################

def _reshape_broadcast(source, target):
    """Reshape the `source` tensor to make it compatible with `target`."""
    dim = len(target.shape)
    shape = ([1] * (dim - 1)) + [-1]
    return source.reshape(shape)
