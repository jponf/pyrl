# -*- coding: utf-8 -*-

import abc
import six

# Torch Stack
import torch

# HDF5
import h5py


###############################################################################

@six.add_metaclass(abc.ABCMeta)
class Normalizer(object):

    @abc.abstractmethod
    def transform(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        """Returns a dictionary containing the whole state of the normalizer.

        :return: A dictionary containing the whole state of the normalizer.
        :rtype: dict
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load_state_dict(self, state):
        """Copies the state into this agent. Any additional key in the
        dictionary is ignored.

        Unless you know what you are doing you should only pass dictionaries
        returned by `state_dict()`.

        :param state: A dict containing a valid agent state.

        :raise KeyError: If a required key is not in the dictionary.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, path):
        """Saves the normalizer in the given `path`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load(cls, path, *args, **kwargs):
        raise NotImplementedError()


class IdentityNormalizer(Normalizer):

    def transform(self, x):
        return x

    def update(self, x):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def save(self, path):
        pass

    def load(cls, path):
        return cls()


class StandardNormalizer(Normalizer):
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
                 shape,
                 epsilon=1e-4,
                 clip_range=float('inf')):
        """
        :param int n_features: Number of features in an individual sample.
        :param float epsilon: minimum std value.
        :param float clip_range: clip standardized values to be in
            range [-clip_range, clip_range] (default: inf tensor).
        """
        self.shape = shape
        self.epsilon = epsilon
        self.clip_range = torch.as_tensor(clip_range).float()

        self.sum = torch.zeros(shape)
        self.sum_sq = torch.zeros(shape)
        self.count = torch.zeros(1)
        self.mean = torch.zeros(shape)
        self.std = torch.ones(shape)

    def update(self, x):
        """Online computation of mean and std on 'x' for later scaling.

        :param x: The data used to compute the mean and standard deviation
            user for later scaling along the features axis.
        """
        x = torch.as_tensor(x)

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
            h5f.create_dataset("shape", data=self.shape)
            h5f.create_dataset("epsilon", data=self.epsilon)
            h5f.create_dataset("clip_range", data=self.clip_range)
            h5f.create_dataset("sum", data=self.sum)
            h5f.create_dataset("sum_sq", data=self.sum_sq)
            h5f.create_dataset("count", data=self.count)

    @classmethod
    def load(cls, path):
        with h5py.File(path, "r") as h5f:
            instance = cls(shape=tuple(h5f["shape"]),
                           epsilon=h5f["epsilon"][()],
                           clip_range=h5f["clip_range"][()])
            instance.load_state_dict(
                dict(sum=torch.as_tensor(h5f["sum"]),
                     sum_sq=torch.as_tensor(h5f["sum_sq"]),
                     count=torch.as_tensor(h5f["count"][()]))
            )
            return instance

    def __repr__(self):
        fmt = "StandardNormalizer(n_features={!r}, clip_range={!r})"
        return fmt.format(self.n_features, self.clip_range)

    def __str__(self):
        fmt = "StandardNormalizer {{\n    mean={}\n    std={}\n}}"
        return fmt.format(self.mean, self.std)


###############################################################################

def _reshape_broadcast(source, target):
    """Reshape the `source` tensor to make it compatible with `target`."""
    dim = len(target.shape)
    shape = ([1] * (dim - 1)) + [-1]
    return source.reshape(shape)
