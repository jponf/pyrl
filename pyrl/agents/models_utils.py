# -*- coding: utf-8 -*-

import collections

# PyTorch
import torch.nn as nn


###############################################################################

_ACTIVATIONS = {
    "leakyrelu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh
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


def soft_update(source, target, tau):
    """Moves the target network values slightly to the values of source."""
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - tau).add_(param.data * tau)


def create_mlp(input_size, output_size, hidden_layers,
               layer_norm=False, activation="relu", last_activation=None):
    """Creates a multi-layer perceptron network."""
    layers_sizes = [input_size]
    layers_sizes.extend(hidden_layers)
    if hidden_layers:
        layers_sizes.append(hidden_layers[-1])
    layers_sizes.append(output_size)

    layers = []
    for i in range(len(layers_sizes) - 1):
        layers.append(("linear{}".format(i),
                       nn.Linear(layers_sizes[i],
                                 layers_sizes[i + 1])))
        if i < len(layers_sizes) - 2:
            if layer_norm:
                layers.append(("layer_norm{}".format(i),
                               nn.LayerNorm(layers_sizes[i])))
            layers.append(("{}{}".format(activation, i),
                           get_activation_layer(activation)))
        elif last_activation is not None:
            layers.append(("{}{}".format(last_activation, i),
                           get_activation_layer(last_activation)))

    return nn.Sequential(collections.OrderedDict(layers))
