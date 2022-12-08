import torch
import torch.nn as nn
import torch.nn.functional as F

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class MLP(torch.nn.Module):
    def __init__(self,
                 in_sizes,
                 out_sizes,
                 activation,
                 output_activation='leaky_relu',
                 **kwargs
                 ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]
        layers = []
        for in_size, out_size in zip(in_sizes, out_sizes):
            layers.append(nn.Linear(in_size, out_size))
            # layers.append(nn.LayerNorm(out_size)) NOTE: empirically this does not work well, on small training set
            layers.append(nn.BatchNorm1d(out_size))
            layers.append(activation)

        if activation != output_activation:
            layers[-1] = output_activation  # replace last activation with output activation

        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)