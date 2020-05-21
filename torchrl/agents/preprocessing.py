# -*- coding: utf-8 -*-

import errno
import os
import pickle

# Torch Stack
import torch

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
        self.sumq = torch.zeros(self.n_features)
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
        self.sumq += torch.sum(x.pow(2), dim=0)
        self.count += x.shape[0]

        # Running mean and standard deviation
        self.mean = self.sum / self.count
        self.std = ((self.sumq / self.count) - self.mean.pow(2)).sqrt()

        # Avoid nan and zeros
        self.std[torch.isnan(self.std)] = self.epsilon
        self.std[self.std < self.epsilon] = self.epsilon

    def transform(self, x):
        """Perform standardization by centering and scaling."""
        mean = _reshape_broadcast(self.mean, x).to(x.device)
        std = _reshape_broadcast(self.std, x).to(x.device)
        return ((x - mean) / std).clamp(-self.clip_range, self.clip_range)

    def save(self, path):
        try:
            os.makedirs(path)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

        files = dict(mean=os.path.join(path, 'mean.pkl'),
                     std=os.path.join(path, 'std.pkl'),
                     sum=os.path.join(path, 'sum.pkl'),
                     sumq=os.path.join(path, 'sumq.pkl'),
                     count=os.path.join(path, 'count.pkl'),
                     args=os.path.join(path, 'args.pkl'))

        args = {
            'n_features': self.n_features,
            'clip_range': self.clip_range.item()
        }

        pickle.dump(self.mean, open(files['mean'], "wb"))
        pickle.dump(self.std, open(files['std'], "wb"))
        pickle.dump(self.sum, open(files['sum'], "wb"))
        pickle.dump(self.sumq, open(files['sumq'], "wb"))
        pickle.dump(self.count, open(files['count'], "wb"))
        pickle.dump(args, open(files['args'], "wb"))

    @classmethod
    def load(cls, path):
        files = dict(mean=os.path.join(path, 'mean.pkl'),
                     std=os.path.join(path, 'std.pkl'),
                     sum=os.path.join(path, 'sum.pkl'),
                     sumq=os.path.join(path, 'sumq.pkl'),
                     count=os.path.join(path, 'count.pkl'),
                     args=os.path.join(path, 'args.pkl'))

        args = pickle.load(open(files['args'], "rb"))
        instance = cls(**args)
        instance.mean = pickle.load(open(files['mean'], "rb"))
        instance.std = pickle.load(open(files['std'], "rb"))
        instance.sum = pickle.load(open(files['sum'], "rb"))
        instance.sumq = pickle.load(open(files['sumq'], "rb"))
        instance.count = pickle.load(open(files['count'], "rb"))

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
