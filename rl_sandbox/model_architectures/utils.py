import numpy as np
import torch
import torch.nn as nn

import rl_sandbox.constants as c


def default_weight_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.LSTM or type(m) == nn.GRU:
        torch.nn.init.xavier_uniform_(m.weight_ih_l0)
        torch.nn.init.orthogonal_(m.weight_hh_l0)
        if m.bias is not None:
            m.bias_ih_l0.data.fill_(0)
            m.bias_hh_l0.data.fill_(0)


def construct_linear_layers(layers):
    linear_layers = nn.ModuleList()
    for (in_dim, out_dim, activation, use_bias, dropout_p) in layers:
        linear_layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
        linear_layers.append(activation)
        if dropout_p > 0.:
            linear_layers.append(nn.Dropout(dropout_p))

    return linear_layers


def construct_conv2d_layers(layers, in_dim):
    conv_layers = nn.ModuleList()
    layers_dim = [in_dim]
    for (in_channels, out_channels, kernel_size, stride, padding, dilation, activation, use_bias, dropout_p, use_batch_norm) in layers:
        conv_layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=use_bias
        ))
        if use_batch_norm:
            conv_layers.append(nn.BatchNorm2d())
        conv_layers.append(activation)
        if dropout_p > 0.:
            conv_layers.append(nn.Dropout(dropout_p))

        layers_dim.append((
            int(np.floor(
                1 + float(layers_dim[-1][0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / float(stride[0]))),
            int(np.floor(
                1 + float(layers_dim[-1][1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / float(stride[1])))))

    return conv_layers, layers_dim


def construct_conv2dtranspose_layers(layers):
    conv_transpose_layers = nn.ModuleList()
    
    for (in_channels, out_channels, kernel_size, stride, padding, dilation, activation, use_bias, dropout_p, use_batch_norm) in layers:
        conv_transpose_layers.append(nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=use_bias
        ))

        if use_batch_norm:
            conv_transpose_layers.append(nn.BatchNorm2d(out_channels))
        conv_transpose_layers.append(activation)
        if dropout_p > 0.:
            conv_transpose_layers.append(nn.Dropout2d(p=dropout_p))
    
    return conv_transpose_layers


def construct_conv3d_layers(layers, in_depth, in_height, in_width):
    conv_layers = nn.ModuleList()
    depths = [in_depth]
    heights = [in_height]
    widths = [in_width]
    for (in_channels, out_channels, kernel_size, stride, padding, dilation, bias, activation, dropout_p, use_batch_norm) in layers:
        conv_layers.append(nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        ))
        if use_batch_norm:
            conv_layers.append(nn.BatchNorm3d())
        conv_layers.append(activation)
        if dropout_p > 0.:
            conv_layers.append(nn.Dropout(dropout_p))

        depths.append(int(np.floor(
            1 + float(depths[-1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / float(stride[0]))))
        heights.append(int(np.floor(
            1 + float(heights[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / float(stride[1]))))
        widths.append(int(np.floor(
            1 + float(widths[-1] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / float(stride[2]))))
    return conv_layers, depths, heights, widths


def make_model(model_cfg):
    return model_cfg[c.MODEL_ARCHITECTURE](**model_cfg[c.KWARGS])


def make_optimizer(parameters, optimizer_cfg):
    return optimizer_cfg[c.OPTIMIZER](parameters, **optimizer_cfg[c.KWARGS])


class RunningMeanStd():
    """ Modified from Baseline
    Assumes shape to be (number of inputs, input_shape)
    """

    def __init__(self, epsilon=1e-4, shape=(), norm_dim=(0,), a_min=-5., a_max=5.):
        assert epsilon > 0.
        self.shape = shape
        self.mean = torch.zeros(shape, dtype=torch.float)
        self.var = torch.ones(shape, dtype=torch.float)
        self.epsilon = epsilon
        self.count = epsilon
        self.a_min = a_min
        self.a_max = a_max
        self.norm_dim = norm_dim

    def update(self, x):
        batch_mean = torch.mean(x, dim=self.norm_dim)
        batch_var = torch.var(x, dim=self.norm_dim)
        batch_count = int(torch.prod(torch.tensor(
            [x.shape[dim] for dim in self.norm_dim])))
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, x):
        x_shape = x.shape
        x = x.reshape(-1, *self.shape)
        normalized_x = torch.clamp((x - self.mean) / torch.sqrt(self.var + self.epsilon),
                                   min=self.a_min,
                                   max=self.a_max)
        normalized_x[normalized_x != normalized_x] = 0.
        normalized_x = normalized_x.reshape(x_shape)
        return normalized_x

    def unnormalize(self, x):
        return x * torch.sqrt(self.var + self.epsilon) + self.mean
