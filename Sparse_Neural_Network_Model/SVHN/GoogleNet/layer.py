import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init


class Fully_Connect0(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Fully_Connect0, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.use_bias = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, input):
        output = input.mm(self.weights)
        if self.use_bias:
            output.add_(self.bias)
        return output