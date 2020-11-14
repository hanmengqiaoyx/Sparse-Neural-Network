import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F


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
        init.kaiming_normal_(self.weights)
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, input):
        output = input.mm(self.weights)
        if self.use_bias:
            output.add_(self.bias)
        return output


class Fully_Connect1(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Fully_Connect1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.dw_gate_weights0 = Parameter(torch.Tensor(8, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(8)
        self.dw_gate_weights1 = Parameter(torch.Tensor(16, 8, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_gate_weights2 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.dw_gate_weights3 = nn.Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(64)
        self.dw_gate_weights4 = Parameter(torch.Tensor(128, 64, 3, 3))
        self.dw_bn4 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gate_weights0 = Parameter(torch.Tensor(256, 128, 3, 3))
        self.bn0 = nn.BatchNorm2d(256)
        self.crop_gate_weights0 = Parameter(torch.Tensor(256, 256, 1, 16))
        self.crop_bn0 = nn.BatchNorm2d(256)
        self.crop_gate_weights1 = Parameter(torch.Tensor(128, 128, 1, 32))
        self.crop_bn1 = nn.BatchNorm2d(128)
        self.crop_gate_weights2 = Parameter(torch.Tensor(64, 64, 1, 64))
        self.crop_bn2 = nn.BatchNorm2d(64)
        self.crop_gate_weights3 = Parameter(torch.Tensor(32, 32, 1, 128))
        self.crop_bn3 = nn.BatchNorm2d(32)
        self.crop_gate_weights4 = Parameter(torch.Tensor(16, 16, 1, 256))
        self.crop_bn4 = nn.BatchNorm2d(16)
        self.crop_gate_weights5 = Parameter(torch.Tensor(8, 8, 1, 512))
        self.crop_bn5 = nn.BatchNorm2d(8)
        self.up_gate_weights0 = nn.Parameter(torch.Tensor(256, 128, 2, 1))
        self.rc_gate_weights0 = nn.Parameter(torch.Tensor(128, 256, 1, 1))
        self.rc_bn0 = nn.BatchNorm2d(128)
        self.up_gate_weights1 = nn.Parameter(torch.Tensor(128, 64, 2, 1))
        self.rc_gate_weights1 = nn.Parameter(torch.Tensor(64, 128, 1, 1))
        self.rc_bn1 = nn.BatchNorm2d(64)
        self.up_gate_weights2 = nn.Parameter(torch.Tensor(64, 32, 2, 1))
        self.rc_gate_weights2 = nn.Parameter(torch.Tensor(32, 64, 1, 1))
        self.rc_bn2 = nn.BatchNorm2d(32)
        self.up_gate_weights3 = nn.Parameter(torch.Tensor(32, 16, 2, 1))
        self.rc_gate_weights3 = nn.Parameter(torch.Tensor(16, 32, 1, 1))
        self.rc_bn3 = nn.BatchNorm2d(16)
        self.up_gate_weights4 = nn.Parameter(torch.Tensor(16, 8, 2, 1))
        self.rc_gate_weights4 = nn.Parameter(torch.Tensor(8, 16, 1, 1))
        self.rc_bn4 = nn.BatchNorm2d(8)
        self.gate_weights1 = Parameter(torch.Tensor(1, 8, 1, 1))
        self.bn1 = nn.BatchNorm2d(1)
        self.use_bias = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights4, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights4, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights5, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights4, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights4, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.gate_weights1, mode='fan_out', nonlinearity='relu')
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, input, epoch):
        if self.training:
            if epoch % 2 == 0:
                self.weights.requires_grad = True
                self.dw_gate_weights0.requires_grad = False
                self.dw_gate_weights1.requires_grad = False
                self.dw_gate_weights2.requires_grad = False
                self.dw_gate_weights3.requires_grad = False
                self.dw_gate_weights4.requires_grad = False
                self.gate_weights0.requires_grad = False
                self.crop_gate_weights0.requires_grad = False
                self.crop_gate_weights1.requires_grad = False
                self.crop_gate_weights2.requires_grad = False
                self.crop_gate_weights3.requires_grad = False
                self.crop_gate_weights4.requires_grad = False
                self.crop_gate_weights5.requires_grad = False
                self.up_gate_weights0.requires_grad = False
                self.up_gate_weights1.requires_grad = False
                self.up_gate_weights2.requires_grad = False
                self.up_gate_weights3.requires_grad = False
                self.up_gate_weights4.requires_grad = False
                self.rc_gate_weights0.requires_grad = False
                self.rc_gate_weights1.requires_grad = False
                self.rc_gate_weights2.requires_grad = False
                self.rc_gate_weights3.requires_grad = False
                self.rc_gate_weights4.requires_grad = False
                self.gate_weights1.requires_grad = False
                if self.use_bias:
                    self.bias.requires_grad = True
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                if self.use_bias:
                    layer1 = F.sigmoid((self.weights.mm(self.gate_weights0)).add_(self.gate_bias0))
                    layer_out = F.sigmoid((layer1.mm(self.gate_weights1)).add_(self.gate_bias1))
                else:
                    layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_gate_weights0, stride=1, padding=1, bias=None)))
                    layer01 = self.maxpool(layer00)
                    layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_gate_weights1, stride=1, padding=1, bias=None)))
                    layer11 = self.maxpool(layer10)
                    layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_gate_weights2, stride=1, padding=1, bias=None)))
                    layer21 = self.maxpool(layer20)
                    layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_gate_weights3, stride=1, padding=1, bias=None)))
                    layer31 = self.maxpool(layer30)
                    layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_gate_weights4, stride=1, padding=1, bias=None)))
                    layer41 = self.maxpool(layer40)
                    layer50 = F.relu(self.bn0(nn.functional.conv2d(layer41, self.gate_weights0, stride=1, padding=1, bias=None)))  # [256, 32, 16]
                    layer52 = F.relu(self.crop_bn0(nn.functional.conv2d(layer50, self.crop_gate_weights0, stride=1, bias=None)))  # [256, 32, 1]
                    layer60 = nn.functional.conv_transpose2d(layer52, self.up_gate_weights0, stride=(2, 1), bias=None)
                    layer42 = F.relu(self.crop_bn1(nn.functional.conv2d(layer40, self.crop_gate_weights1, stride=1, bias=None)))
                    layer61 = torch.cat((layer42, layer60), dim=1)  # [256, 64, 1]
                    layer62 = F.relu(self.rc_bn0(nn.functional.conv2d(layer61, self.rc_gate_weights0, stride=1, bias=None)))
                    layer70 = nn.functional.conv_transpose2d(layer62, self.up_gate_weights1, stride=(2, 1), bias=None)
                    layer32 = F.relu(self.crop_bn2(nn.functional.conv2d(layer30, self.crop_gate_weights2, stride=1, bias=None)))
                    layer71 = torch.cat((layer32, layer70), dim=1)  # [128, 128, 1]
                    layer72 = F.relu(self.rc_bn1(nn.functional.conv2d(layer71, self.rc_gate_weights1, stride=1, bias=None)))
                    layer80 = nn.functional.conv_transpose2d(layer72, self.up_gate_weights2, stride=(2, 1), bias=None)
                    layer22 = F.relu(self.crop_bn3(nn.functional.conv2d(layer20, self.crop_gate_weights3, stride=1, bias=None)))
                    layer81 = torch.cat((layer22, layer80), dim=1)  # [64, 256, 1]
                    layer82 = F.relu(self.rc_bn2(nn.functional.conv2d(layer81, self.rc_gate_weights2, stride=1, bias=None)))
                    layer90 = nn.functional.conv_transpose2d(layer82, self.up_gate_weights3, stride=(2, 1), bias=None)
                    layer12 = F.relu(self.crop_bn4(nn.functional.conv2d(layer10, self.crop_gate_weights4, stride=1, bias=None)))
                    layer91 = torch.cat((layer12, layer90), dim=1)  # [32, 512, 1]
                    layer92 = F.relu(self.rc_bn3(nn.functional.conv2d(layer91, self.rc_gate_weights3, stride=1, bias=None)))
                    layer100 = nn.functional.conv_transpose2d(layer92, self.up_gate_weights4, stride=(2, 1), bias=None)
                    layer02 = F.relu(self.crop_bn5(nn.functional.conv2d(layer00, self.crop_gate_weights5, stride=1, bias=None)))
                    layer101 = torch.cat((layer02, layer100), dim=1)  # [16, 1024, 1]
                    layer102 = F.relu(self.rc_bn4(nn.functional.conv2d(layer101, self.rc_gate_weights4, stride=1, bias=None)))  # [8, 1024, 1]
                    layer_out = F.relu(self.bn1(nn.functional.conv2d(layer102, self.gate_weights1, stride=1, bias=None)))  # [1, 1024, 1]
                gate = layer_out.view(1, self.in_features).expand(input.size(0), self.in_features)
                new_input = input.mul(gate)
                output = new_input.mm(self.weights)
            elif epoch % 2 != 0:
                self.weights.requires_grad = False
                self.dw_gate_weights0.requires_grad = True
                self.dw_gate_weights1.requires_grad = True
                self.dw_gate_weights2.requires_grad = True
                self.dw_gate_weights3.requires_grad = True
                self.dw_gate_weights4.requires_grad = True
                self.gate_weights0.requires_grad = True
                self.crop_gate_weights0.requires_grad = True
                self.crop_gate_weights1.requires_grad = True
                self.crop_gate_weights2.requires_grad = True
                self.crop_gate_weights3.requires_grad = True
                self.crop_gate_weights4.requires_grad = True
                self.crop_gate_weights5.requires_grad = True
                self.up_gate_weights0.requires_grad = True
                self.up_gate_weights1.requires_grad = True
                self.up_gate_weights2.requires_grad = True
                self.up_gate_weights3.requires_grad = True
                self.up_gate_weights4.requires_grad = True
                self.rc_gate_weights0.requires_grad =True
                self.rc_gate_weights1.requires_grad =True
                self.rc_gate_weights2.requires_grad =True
                self.rc_gate_weights3.requires_grad =True
                self.rc_gate_weights4.requires_grad =True
                self.gate_weights1.requires_grad = True
                if self.use_bias:
                    self.bias.requires_grad = False
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                if self.use_bias:
                    layer1 = F.sigmoid((self.weights.mm(self.gate_weights0)).add_(self.gate_bias0))
                    layer_out = F.sigmoid((layer1.mm(self.gate_weights1)).add_(self.gate_bias1))
                else:
                    layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_gate_weights0, stride=1, padding=1, bias=None)))
                    layer01 = self.maxpool(layer00)
                    layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_gate_weights1, stride=1, padding=1, bias=None)))
                    layer11 = self.maxpool(layer10)
                    layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_gate_weights2, stride=1, padding=1, bias=None)))
                    layer21 = self.maxpool(layer20)
                    layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_gate_weights3, stride=1, padding=1, bias=None)))
                    layer31 = self.maxpool(layer30)
                    layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_gate_weights4, stride=1, padding=1, bias=None)))
                    layer41 = self.maxpool(layer40)
                    layer50 = F.relu(self.bn0(nn.functional.conv2d(layer41, self.gate_weights0, stride=1, padding=1, bias=None)))  # [256, 32, 32]
                    layer52 = F.relu(self.crop_bn0(nn.functional.conv2d(layer50, self.crop_gate_weights0, stride=1, bias=None)))  # [256, 32, 1]
                    layer60 = nn.functional.conv_transpose2d(layer52, self.up_gate_weights0, stride=(2, 1), bias=None)
                    layer42 = F.relu(self.crop_bn1(nn.functional.conv2d(layer40, self.crop_gate_weights1, stride=1, bias=None)))
                    layer61 = torch.cat((layer42, layer60), dim=1)  # [256, 64, 1]
                    layer62 = F.relu(self.rc_bn0(nn.functional.conv2d(layer61, self.rc_gate_weights0, stride=1, bias=None)))
                    layer70 = nn.functional.conv_transpose2d(layer62, self.up_gate_weights1, stride=(2, 1), bias=None)
                    layer32 = F.relu(self.crop_bn2(nn.functional.conv2d(layer30, self.crop_gate_weights2, stride=1, bias=None)))
                    layer71 = torch.cat((layer32, layer70), dim=1)  # [128, 128, 1]
                    layer72 = F.relu(self.rc_bn1(nn.functional.conv2d(layer71, self.rc_gate_weights1, stride=1, bias=None)))
                    layer80 = nn.functional.conv_transpose2d(layer72, self.up_gate_weights2, stride=(2, 1), bias=None)
                    layer22 = F.relu(self.crop_bn3(nn.functional.conv2d(layer20, self.crop_gate_weights3, stride=1, bias=None)))
                    layer81 = torch.cat((layer22, layer80), dim=1)  # [64, 256, 1]
                    layer82 = F.relu(self.rc_bn2(nn.functional.conv2d(layer81, self.rc_gate_weights2, stride=1, bias=None)))
                    layer90 = nn.functional.conv_transpose2d(layer82, self.up_gate_weights3, stride=(2, 1), bias=None)
                    layer12 = F.relu(self.crop_bn4(nn.functional.conv2d(layer10, self.crop_gate_weights4, stride=1, bias=None)))
                    layer91 = torch.cat((layer12, layer90), dim=1)  # [32, 512, 1]
                    layer92 = F.relu(self.rc_bn3(nn.functional.conv2d(layer91, self.rc_gate_weights3, stride=1, bias=None)))
                    layer100 = nn.functional.conv_transpose2d(layer92, self.up_gate_weights4, stride=(2, 1), bias=None)
                    layer02 = F.relu(self.crop_bn5(nn.functional.conv2d(layer00, self.crop_gate_weights5, stride=1, bias=None)))
                    layer101 = torch.cat((layer02, layer100), dim=1)  # [16, 1024, 1]
                    layer102 = F.relu(self.rc_bn4(nn.functional.conv2d(layer101, self.rc_gate_weights4, stride=1, bias=None)))  # [8, 1024, 1]
                    layer_out = F.relu(self.bn1(nn.functional.conv2d(layer102, self.gate_weights1, stride=1, bias=None)))  # [1, 1024, 1]
                gate = layer_out.view(1, self.in_features).expand(input.size(0), self.in_features)
                new_input = input.mul(gate)
                output = new_input.mm(self.weights)
        elif not self.training:
            data = self.weights.view(1, 1, self.in_features, self.out_features)
            if self.use_bias:
                layer1 = F.sigmoid((self.weights.mm(self.gate_weights0)).add_(self.gate_bias0))
                layer_out = F.sigmoid((layer1.mm(self.gate_weights1)).add_(self.gate_bias1))
            else:
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_gate_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_gate_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_gate_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_gate_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_gate_weights4, stride=1, padding=1, bias=None)))
                layer41 = self.maxpool(layer40)
                layer50 = F.relu(self.bn0(nn.functional.conv2d(layer41, self.gate_weights0, stride=1, padding=1, bias=None)))  # [256, 32, 32]
                layer52 = F.relu(self.crop_bn0(nn.functional.conv2d(layer50, self.crop_gate_weights0, stride=1, bias=None)))  # [256, 32, 1]
                layer60 = nn.functional.conv_transpose2d(layer52, self.up_gate_weights0, stride=(2, 1), bias=None)
                layer42 = F.relu(self.crop_bn1(nn.functional.conv2d(layer40, self.crop_gate_weights1, stride=1, bias=None)))
                layer61 = torch.cat((layer42, layer60), dim=1)  # [256, 64, 1]
                layer62 = F.relu(self.rc_bn0(nn.functional.conv2d(layer61, self.rc_gate_weights0, stride=1, bias=None)))
                layer70 = nn.functional.conv_transpose2d(layer62, self.up_gate_weights1, stride=(2, 1), bias=None)
                layer32 = F.relu(self.crop_bn2(nn.functional.conv2d(layer30, self.crop_gate_weights2, stride=1, bias=None)))
                layer71 = torch.cat((layer32, layer70), dim=1)  # [128, 128, 1]
                layer72 = F.relu(self.rc_bn1(nn.functional.conv2d(layer71, self.rc_gate_weights1, stride=1, bias=None)))
                layer80 = nn.functional.conv_transpose2d(layer72, self.up_gate_weights2, stride=(2, 1), bias=None)
                layer22 = F.relu(self.crop_bn3(nn.functional.conv2d(layer20, self.crop_gate_weights3, stride=1, bias=None)))
                layer81 = torch.cat((layer22, layer80), dim=1)  # [64, 256, 1]
                layer82 = F.relu(self.rc_bn2(nn.functional.conv2d(layer81, self.rc_gate_weights2, stride=1, bias=None)))
                layer90 = nn.functional.conv_transpose2d(layer82, self.up_gate_weights3, stride=(2, 1), bias=None)
                layer12 = F.relu(self.crop_bn4(nn.functional.conv2d(layer10, self.crop_gate_weights4, stride=1, bias=None)))
                layer91 = torch.cat((layer12, layer90), dim=1)  # [32, 512, 1]
                layer92 = F.relu(self.rc_bn3(nn.functional.conv2d(layer91, self.rc_gate_weights3, stride=1, bias=None)))
                layer100 = nn.functional.conv_transpose2d(layer92, self.up_gate_weights4, stride=(2, 1), bias=None)
                layer02 = F.relu(self.crop_bn5(nn.functional.conv2d(layer00, self.crop_gate_weights5, stride=1, bias=None)))
                layer101 = torch.cat((layer02, layer100), dim=1)  # [16, 1024, 1]
                layer102 = F.relu(self.rc_bn4(nn.functional.conv2d(layer101, self.rc_gate_weights4, stride=1, bias=None)))  # [8, 1024, 1]
                layer_out = F.relu(self.bn1(nn.functional.conv2d(layer102, self.gate_weights1, stride=1, bias=None)))  # [1, 1024, 1]
            gate = layer_out.view(1, self.in_features).expand(input.size(0), self.in_features)
            new_input = input.mul(gate)
            output = new_input.mm(self.weights)
        if self.use_bias:
            output.add_(self.bias)
        return output


class Fully_Connect2(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Fully_Connect2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.dw_gate_weights0 = Parameter(torch.Tensor(8, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(8)
        self.dw_gate_weights1 = Parameter(torch.Tensor(16, 8, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_gate_weights2 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.dw_gate_weights3 = nn.Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(64)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=(2, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gate_weights0 = Parameter(torch.Tensor(128, 64, 3, 3))
        self.bn0 = nn.BatchNorm2d(128)
        self.crop_gate_weights0 = Parameter(torch.Tensor(128, 128, 1, 2))
        self.crop_bn0 = nn.BatchNorm2d(128)
        self.crop_gate_weights1 = Parameter(torch.Tensor(64, 64, 1, 4))
        self.crop_bn1 = nn.BatchNorm2d(64)
        self.crop_gate_weights2 = Parameter(torch.Tensor(32, 32, 1, 8))
        self.crop_bn2 = nn.BatchNorm2d(32)
        self.crop_gate_weights3 = Parameter(torch.Tensor(16, 16, 1, 9))
        self.crop_bn3 = nn.BatchNorm2d(16)
        self.crop_gate_weights4 = Parameter(torch.Tensor(8, 8, 1, 10))
        self.crop_bn4 = nn.BatchNorm2d(8)
        self.up_gate_weights0 = nn.Parameter(torch.Tensor(128, 64, 2, 1))
        self.rc_gate_weights0 = nn.Parameter(torch.Tensor(64, 128, 1, 1))
        self.rc_bn0 = nn.BatchNorm2d(64)
        self.up_gate_weights1 = nn.Parameter(torch.Tensor(64, 32, 2, 1))
        self.rc_gate_weights1 = nn.Parameter(torch.Tensor(32, 64, 1, 1))
        self.rc_bn1 = nn.BatchNorm2d(32)
        self.up_gate_weights2 = nn.Parameter(torch.Tensor(32, 16, 2, 1))
        self.rc_gate_weights2 = nn.Parameter(torch.Tensor(16, 32, 1, 1))
        self.rc_bn2 = nn.BatchNorm2d(16)
        self.up_gate_weights3 = nn.Parameter(torch.Tensor(16, 8, 2, 1))
        self.rc_gate_weights3 = nn.Parameter(torch.Tensor(8, 16, 1, 1))
        self.rc_bn3 = nn.BatchNorm2d(8)
        self.gate_weights1 = Parameter(torch.Tensor(1, 8, 1, 1))
        self.bn1 = nn.BatchNorm2d(1)
        self.use_bias = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_gate_weights3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.crop_gate_weights4, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_gate_weights3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.rc_gate_weights3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.gate_weights1, mode='fan_out', nonlinearity='relu')
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, input, epoch):
        if self.training:
            if epoch % 2 == 0:
                self.weights.requires_grad = True
                self.dw_gate_weights0.requires_grad = False
                self.dw_gate_weights1.requires_grad = False
                self.dw_gate_weights2.requires_grad = False
                self.dw_gate_weights3.requires_grad = False
                self.gate_weights0.requires_grad = False
                self.crop_gate_weights0.requires_grad = False
                self.crop_gate_weights1.requires_grad = False
                self.crop_gate_weights2.requires_grad = False
                self.crop_gate_weights3.requires_grad = False
                self.crop_gate_weights4.requires_grad = False
                self.up_gate_weights0.requires_grad = False
                self.up_gate_weights1.requires_grad = False
                self.up_gate_weights2.requires_grad = False
                self.up_gate_weights3.requires_grad = False
                self.rc_gate_weights0.requires_grad = False
                self.rc_gate_weights1.requires_grad = False
                self.rc_gate_weights2.requires_grad = False
                self.rc_gate_weights3.requires_grad = False
                self.gate_weights1.requires_grad = False
                if self.use_bias:
                    self.bias.requires_grad = True
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                if self.use_bias:
                    layer1 = F.sigmoid((self.weights.mm(self.gate_weights0)).add_(self.gate_bias0))
                    layer_out = F.sigmoid((layer1.mm(self.gate_weights1)).add_(self.gate_bias1))
                else:
                    layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_gate_weights0, stride=1, padding=1, bias=None)))
                    layer01 = self.maxpool0(layer00)
                    layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_gate_weights1, stride=1, padding=1, bias=None)))
                    layer11 = self.maxpool0(layer10)
                    layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_gate_weights2, stride=1, padding=1, bias=None)))
                    layer21 = self.maxpool1(layer20)
                    layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_gate_weights3, stride=1, padding=1, bias=None)))
                    layer31 = self.maxpool1(layer30)
                    layer40 = F.relu(self.bn0(nn.functional.conv2d(layer31, self.gate_weights0, stride=1, padding=1, bias=None)))  # [128, 32, 2]
                    layer42 = F.relu(self.crop_bn0(nn.functional.conv2d(layer40, self.crop_gate_weights0, stride=1, bias=None)))  # [128, 32, 1]
                    layer50 = nn.functional.conv_transpose2d(layer42, self.up_gate_weights0, stride=(2, 1), bias=None)
                    layer32 = F.relu(self.crop_bn1(nn.functional.conv2d(layer30, self.crop_gate_weights1, stride=1, bias=None)))
                    layer51 = torch.cat((layer32, layer50), dim=1)  # [128, 64, 1]
                    layer52 = F.relu(self.rc_bn0(nn.functional.conv2d(layer51, self.rc_gate_weights0, stride=1, bias=None)))
                    layer60 = nn.functional.conv_transpose2d(layer52, self.up_gate_weights1, stride=(2, 1), bias=None)
                    layer22 = F.relu(self.crop_bn2(nn.functional.conv2d(layer20, self.crop_gate_weights2, stride=1, bias=None)))
                    layer61 = torch.cat((layer22, layer60), dim=1)  # [64, 128, 1]
                    layer62 = F.relu(self.rc_bn1(nn.functional.conv2d(layer61, self.rc_gate_weights1, stride=1, bias=None)))
                    layer70 = nn.functional.conv_transpose2d(layer62, self.up_gate_weights2, stride=(2, 1), bias=None)
                    layer12 = F.relu(self.crop_bn3(nn.functional.conv2d(layer10, self.crop_gate_weights3, stride=1, bias=None)))
                    layer71 = torch.cat((layer12, layer70), dim=1)  # [32, 256, 1]
                    layer72 = F.relu(self.rc_bn2(nn.functional.conv2d(layer71, self.rc_gate_weights2, stride=1, bias=None)))
                    layer80 = nn.functional.conv_transpose2d(layer72, self.up_gate_weights3, stride=(2, 1), bias=None)
                    layer02 = F.relu(self.crop_bn4(nn.functional.conv2d(layer00, self.crop_gate_weights4, stride=1, bias=None)))
                    layer81 = torch.cat((layer02, layer80), dim=1)  # [16, 512, 1]
                    layer82 = F.relu(self.rc_bn3(nn.functional.conv2d(layer81, self.rc_gate_weights3, stride=1, bias=None)))  # [8, 512, 1]
                    layer_out = F.relu(self.bn1(nn.functional.conv2d(layer82, self.gate_weights1, stride=1, bias=None)))  # [1, 512, 1]
                gate = layer_out.view(1, self.in_features).expand(input.size(0), self.in_features)
                new_input = input.mul(gate)
                output = new_input.mm(self.weights)
            elif epoch % 2 != 0:
                self.weights.requires_grad = False
                self.dw_gate_weights0.requires_grad = True
                self.dw_gate_weights1.requires_grad = True
                self.dw_gate_weights2.requires_grad = True
                self.dw_gate_weights3.requires_grad = True
                self.gate_weights0.requires_grad = True
                self.crop_gate_weights0.requires_grad = True
                self.crop_gate_weights1.requires_grad = True
                self.crop_gate_weights2.requires_grad = True
                self.crop_gate_weights3.requires_grad = True
                self.crop_gate_weights4.requires_grad = True
                self.up_gate_weights0.requires_grad = True
                self.up_gate_weights1.requires_grad = True
                self.up_gate_weights2.requires_grad = True
                self.up_gate_weights3.requires_grad = True
                self.rc_gate_weights0.requires_grad =True
                self.rc_gate_weights1.requires_grad =True
                self.rc_gate_weights2.requires_grad =True
                self.rc_gate_weights3.requires_grad =True
                self.gate_weights1.requires_grad = True
                if self.use_bias:
                    self.bias.requires_grad = True
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                if self.use_bias:
                    layer1 = F.sigmoid((self.weights.mm(self.gate_weights0)).add_(self.gate_bias0))
                    layer_out = F.sigmoid((layer1.mm(self.gate_weights1)).add_(self.gate_bias1))
                else:
                    layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_gate_weights0, stride=1, padding=1, bias=None)))
                    layer01 = self.maxpool0(layer00)
                    layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_gate_weights1, stride=1, padding=1, bias=None)))
                    layer11 = self.maxpool0(layer10)
                    layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_gate_weights2, stride=1, padding=1, bias=None)))
                    layer21 = self.maxpool1(layer20)
                    layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_gate_weights3, stride=1, padding=1, bias=None)))
                    layer31 = self.maxpool1(layer30)
                    layer40 = F.relu(self.bn0(nn.functional.conv2d(layer31, self.gate_weights0, stride=1, padding=1, bias=None)))  # [128, 32, 2]
                    layer42 = F.relu(self.crop_bn0(nn.functional.conv2d(layer40, self.crop_gate_weights0, stride=1, bias=None)))  # [128, 32, 1]
                    layer50 = nn.functional.conv_transpose2d(layer42, self.up_gate_weights0, stride=(2, 1), bias=None)
                    layer32 = F.relu(self.crop_bn1(nn.functional.conv2d(layer30, self.crop_gate_weights1, stride=1, bias=None)))
                    layer51 = torch.cat((layer32, layer50), dim=1)  # [128, 64, 1]
                    layer52 = F.relu(self.rc_bn0(nn.functional.conv2d(layer51, self.rc_gate_weights0, stride=1, bias=None)))
                    layer60 = nn.functional.conv_transpose2d(layer52, self.up_gate_weights1, stride=(2, 1), bias=None)
                    layer22 = F.relu(self.crop_bn2(nn.functional.conv2d(layer20, self.crop_gate_weights2, stride=1, bias=None)))
                    layer61 = torch.cat((layer22, layer60), dim=1)  # [64, 128, 1]
                    layer62 = F.relu(self.rc_bn1(nn.functional.conv2d(layer61, self.rc_gate_weights1, stride=1, bias=None)))
                    layer70 = nn.functional.conv_transpose2d(layer62, self.up_gate_weights2, stride=(2, 1), bias=None)
                    layer12 = F.relu(self.crop_bn3(nn.functional.conv2d(layer10, self.crop_gate_weights3, stride=1, bias=None)))
                    layer71 = torch.cat((layer12, layer70), dim=1)  # [32, 256, 1]
                    layer72 = F.relu(self.rc_bn2(nn.functional.conv2d(layer71, self.rc_gate_weights2, stride=1, bias=None)))
                    layer80 = nn.functional.conv_transpose2d(layer72, self.up_gate_weights3, stride=(2, 1), bias=None)
                    layer02 = F.relu(self.crop_bn4(nn.functional.conv2d(layer00, self.crop_gate_weights4, stride=1, bias=None)))
                    layer81 = torch.cat((layer02, layer80), dim=1)  # [16, 512, 1]
                    layer82 = F.relu(self.rc_bn3(nn.functional.conv2d(layer81, self.rc_gate_weights3, stride=1, bias=None)))  # [8, 512, 1]
                    layer_out = F.relu(self.bn1(nn.functional.conv2d(layer82, self.gate_weights1, stride=1, bias=None)))  # [1, 512, 1]
                gate = layer_out.view(1, self.in_features).expand(input.size(0), self.in_features)
                new_input = input.mul(gate)
                output = new_input.mm(self.weights)
        elif not self.training:
            data = self.weights.view(1, 1, self.in_features, self.out_features)
            if self.use_bias:
                layer1 = F.sigmoid((self.weights.mm(self.gate_weights0)).add_(self.gate_bias0))
                layer_out = F.sigmoid((layer1.mm(self.gate_weights1)).add_(self.gate_bias1))
            else:
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_gate_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool0(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_gate_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool0(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_gate_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool1(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_gate_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool1(layer30)
                layer40 = F.relu(self.bn0(nn.functional.conv2d(layer31, self.gate_weights0, stride=1, padding=1, bias=None)))  # [128, 32, 2]
                layer42 = F.relu(self.crop_bn0(nn.functional.conv2d(layer40, self.crop_gate_weights0, stride=1, bias=None)))  # [128, 32, 1]
                layer50 = nn.functional.conv_transpose2d(layer42, self.up_gate_weights0, stride=(2, 1), bias=None)
                layer32 = F.relu(self.crop_bn1(nn.functional.conv2d(layer30, self.crop_gate_weights1, stride=1, bias=None)))
                layer51 = torch.cat((layer32, layer50), dim=1)  # [128, 64, 1]
                layer52 = F.relu(self.rc_bn0(nn.functional.conv2d(layer51, self.rc_gate_weights0, stride=1, bias=None)))
                layer60 = nn.functional.conv_transpose2d(layer52, self.up_gate_weights1, stride=(2, 1), bias=None)
                layer22 = F.relu(self.crop_bn2(nn.functional.conv2d(layer20, self.crop_gate_weights2, stride=1, bias=None)))
                layer61 = torch.cat((layer22, layer60), dim=1)  # [64, 128, 1]
                layer62 = F.relu(self.rc_bn1(nn.functional.conv2d(layer61, self.rc_gate_weights1, stride=1, bias=None)))
                layer70 = nn.functional.conv_transpose2d(layer62, self.up_gate_weights2, stride=(2, 1), bias=None)
                layer12 = F.relu(self.crop_bn3(nn.functional.conv2d(layer10, self.crop_gate_weights3, stride=1, bias=None)))
                layer71 = torch.cat((layer12, layer70), dim=1)  # [32, 256, 1]
                layer72 = F.relu(self.rc_bn2(nn.functional.conv2d(layer71, self.rc_gate_weights2, stride=1, bias=None)))
                layer80 = nn.functional.conv_transpose2d(layer72, self.up_gate_weights3, stride=(2, 1), bias=None)
                layer02 = F.relu(self.crop_bn4(nn.functional.conv2d(layer00, self.crop_gate_weights4, stride=1, bias=None)))
                layer81 = torch.cat((layer02, layer80), dim=1)  # [16, 512, 1]
                layer82 = F.relu(self.rc_bn3(nn.functional.conv2d(layer81, self.rc_gate_weights3, stride=1, bias=None)))  # [8, 512, 1]
                layer_out = F.relu(self.bn1(nn.functional.conv2d(layer82, self.gate_weights1, stride=1, bias=None)))  # [1, 512, 1]
            gate = layer_out.view(1, self.in_features).expand(input.size(0), self.in_features)
            new_input = input.mul(gate)
            output = new_input.mm(self.weights)
        if self.use_bias:
            output.add_(self.bias)
        return output