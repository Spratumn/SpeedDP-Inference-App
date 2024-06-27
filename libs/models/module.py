import torch
import torch.nn as nn


def get_module_by_name(model:nn.Module, module_name:str):
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None, None
    if hasattr(model, name_list[-1]):
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
    else:
        return None, None


def process_compiled_checkpoint(checkpoint):
    keys_list = list(checkpoint.keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            checkpoint[deal_key] = checkpoint[key]
            del checkpoint[key]

def fuse_state_dict(state_dict):
    """
    only support BaseConv or BaseConvBN
    """
    var = torch.sqrt(state_dict['bn.running_var'])
    conv_weights = (
        state_dict['bn.weight'] / var
        ).view([state_dict['conv.weight'].size()[0], 1, 1, 1]) * state_dict['conv.weight']

    if 'conv.bias' in state_dict:
        conv_bias = state_dict['conv.bias']
    else:
        conv_bias = torch.zeros_like(state_dict['bn.bias'])
    conv_bias = (
        state_dict['bn.weight'] * (conv_bias - state_dict['bn.running_mean'])
        ) / var + state_dict['bn.bias']

    return {'weight': conv_weights, 'bias': conv_bias}


def get_norm(name, out_channels):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    elif name == 'gn':
        module = nn.GroupNorm(out_channels)
    else:
        raise NotImplementedError
    return module



def get_activation(name='relu', inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == 'silu':
            module = SiLU()
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'relu6':
            module = nn.ReLU6(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=False)
        elif name == 'sigmoid':
            module = nn.Sigmoid()
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module

    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()
        self._mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self._mul.mul(x, self.act(x))



class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=-1, groups=1, bias=False, act='relu'):
        super().__init__()
        self.conv_kwargs = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': (kernel_size - 1) // 2 if padding == -1 else padding,
            'groups': groups,
            'bias' : bias
        }
        self.conv = nn.Conv2d(**self.conv_kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, kernel_size=kernel_size,
                              stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, kernel_size=1,
                              stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True,
                 expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels
        self._add = nn.quantized.FloatFunctional()

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add: y = self._add.add(y, x)
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, kernel_size=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid_channels, in_channels, kernel_size=3, stride=1, act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1,
                 shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act="silu"):
        super().__init__()
        self.conv_split = nn.Conv2d(in_channels, in_channels * 4, kernel_size=2, stride=2, bias=False)
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size, stride, act=act)
        if in_channels == 3:
            self.w = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32).view(12, 3, 2, 2)
        if in_channels == 1:
            self.w = torch.tensor([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=torch.float32).view(4, 1, 2, 2)
        self.conv_split.load_state_dict({'weight': self.w}, strict=False)
        self.conv_split.requires_grad_(False)

    def set_device(self, device):
        self.conv = self.conv.to(device)
        self.conv_split = self.conv_split.to(device)

    def forward(self, x):
        x = self.conv_split(x)
        return self.conv(x)
