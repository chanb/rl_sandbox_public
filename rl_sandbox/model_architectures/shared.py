import numpy as np
import torch
import torch.nn as nn

from rl_sandbox.model_architectures.utils import construct_conv2d_layers, construct_conv2dtranspose_layers


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Split(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.feature_dims = feature_dims

    def forward(self, x):
        features = []
        last_feature_idx = 0
        for feature_dim in self.feature_dims:
            features.append(x[..., last_feature_idx:last_feature_idx + feature_dim])
            last_feature_idx += feature_dim
        return features


class Fuse(nn.Module):
    def forward(self, features):
        return torch.cat(features, dim=-1)


class Conv2DEncoder(nn.Module):
    def __init__(self,
                 num_channels,
                 height,
                 width,
                 output_size,
                 layers,
                 activation=nn.Identity()):
        super().__init__()
        self._num_channels = num_channels
        self.output_size = output_size

        self._conv_layers, self._layers_dim = construct_conv2d_layers(layers=layers, in_dim=(height, width))

        print("Output Height: {}\tOutput Width: {}".format(*self._layers_dim[-1]))
        conv_output_dim = int(np.product(self._layers_dim[-1]))

        self._flatten = Flatten()
        self._linear_layer = nn.Linear(conv_output_dim * layers[-1][1], output_size)
        self._activation = activation

    def forward(self, x):
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        x = self._linear_layer(x)
        x = self._activation(x)
        return x

    @property
    def layers_dim(self):
        return self._layers_dim


class Conv2DDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 layers,
                 layers_dim):
        super().__init__()
        self._input_size = input_size
        
        self._in_channel = layers[0][1]
        self._layers_dim = layers_dim[::-1]
        self._in_dim = layers_dim[-1]

        self._linear_layer = nn.Linear(input_size, self._in_channel * int(np.product(self._in_dim)))
        self._relu = nn.ReLU()
        self._conv_transpose_layers = construct_conv2dtranspose_layers(layers)

    def forward(self, x):
        x = self._linear_layer(x)
        x = self._relu(x)

        x = x.reshape(x.shape[0], self._in_channel, *self._in_dim)
        layer_idx = 1
        for layer in self._conv_transpose_layers:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x, output_size=self._layers_dim[layer_idx])
                layer_idx += 1
            else:
                x = layer(x)
        return x
