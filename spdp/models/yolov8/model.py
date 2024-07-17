import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import create_model_config
from .loss import dist2bbox, make_anchors
from ..module import BaseConv, DWConv


class YOLOV8(nn.Module):
    def __init__(self, config, model_type, options=None):
        super().__init__()
        model_config, self.uid, self.model_config_str = create_model_config(config, model_type, options)
        self.phase = model_config.phase
        self.input_channel = model_config.input_channel
        self.catenum = model_config.get('catenum')
        self.task_type = model_config.task_type
        strides = [4, 8, 16, 32]
        self.stride = [strides[i] for i in model_config.out_idxes]
        fpn_feat_channels = [model_config.feat_channels[i] for i in model_config.out_idxes]
        self.backbone = BackboneV8(model_config.depth, model_config.width,
                                   model_config.feat_channels,
                                   out_idxes=model_config.out_idxes,
                                   depthwise=model_config.backbone.depthwise,
                                   act=model_config.backbone.act,
                                   input_channel=model_config.input_channel
                                   )

        self.fpn = FPNV8(model_config.depth,
                         model_config.width,
                         fpn_feat_channels,
                         depthwise=model_config.fpn.depthwise,
                         act=model_config.fpn.act
                         )
        if self.task_type == 'seg':
            self.head = Segment(model_config.width,
                                model_config.get('catenum'),
                                self.stride,
                                fpn_feat_channels,
                                depthwise=model_config.head.depthwise,
                                act=model_config.head.act)
        else:
            self.head = Detect(model_config.width,
                               model_config.get('catenum'),
                               self.stride,
                               fpn_feat_channels,
                               depthwise=model_config.head.depthwise,
                               act=model_config.head.act)
        self.head.phase = self.phase

    def set_phase(self, phase='train'):
        self.phase = phase
        self.head.phase = self.phase

    def set_device(self, device):
        self.device = device
        self.backbone = self.backbone.to(self.device)
        self.fpn = self.fpn.to(self.device)
        self.head = self.head.to(self.device)

    def forward(self, x, targets=None):
        preds = self.forward_once(x)
        return preds

    def forward_once(self, x):
        backbone_outputs = self.backbone(x)
        fpn_outputs = self.fpn(backbone_outputs)
        head_outputs = self.head(fpn_outputs)
        return head_outputs

class BackboneV8(nn.Module):
    def __init__(self, depth, width,
                 feat_channels=(128, 256, 512, 1024),
                 out_idxes=(1, 2, 3),
                 depthwise=False,
                 act="relu",
                 input_channel=3):
        super().__init__()
        self.last_layer = max(out_idxes)
        self.out_idxes = out_idxes
        nums = (3, 6, 6, 3)
        assert 0 <= self.last_layer < 4
        #                 0       1          2        3
        #                 4       8         16       32
        Conv = DWConv if depthwise else BaseConv
        in_ch, out_ch = input_channel, int(width * 64)
        self.stem = Conv(in_ch, out_ch, 3, stride=2, act=act)
        self.module_list = nn.ModuleList()
        for num, channel in zip(nums, feat_channels):
            in_ch, out_ch = out_ch, int(width * channel)
            n = max(round(num * depth), 1)
            feat_layer = nn.Sequential(
                Conv(in_ch, out_ch, 3, stride=2, act=act),
                C2f(out_ch, out_ch, n=n, shortcut=True, depthwise=depthwise, act=act)
            )
            self.module_list.append(feat_layer)
        self.sppf = SPPF(out_ch, out_ch, 5, depthwise=depthwise, act=act)

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for feat_layer in self.module_list:
            x = feat_layer(x)
            feats.append(x)
        feats[-1] = self.sppf(x)
        return [feats[i] for i in self.out_idxes]


class FPNV8(nn.Module):
    def __init__(self, depth, width,
                 feat_channels=(256, 512, 1024),
                 depthwise=False,
                 act="relu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        nums = (3, 3, 3)

        in_ch = int(width * feat_channels[1]) + int(width * feat_channels[2])
        out_ch = int(width * feat_channels[1])
        n = max(round(nums[1] * depth), 1)
        self.feat1_1 = C2f(in_ch, out_ch, n=n, shortcut=False, depthwise=depthwise, act=act)
        in_ch = int(width * feat_channels[0]) + int(width * feat_channels[1])
        out_ch = int(width * feat_channels[0])
        self.feat1_2 = C2f(in_ch, out_ch, n=n, shortcut=False, depthwise=depthwise, act=act)

        in_ch = int(width * feat_channels[0])
        out_ch = int(width * feat_channels[0])
        n = max(round(nums[0] * depth), 1)
        self.feat2_1 = Conv(in_ch, out_ch, kernel_size=3, stride=2, act=act)
        in_ch = int(width * feat_channels[0]) + int(width * feat_channels[1])
        out_ch = int(width * feat_channels[1])
        self.feat2_2 = C2f(in_ch, out_ch, n=n, shortcut=False, depthwise=depthwise, act=act)

        in_ch = int(width * feat_channels[1])
        out_ch = int(width * feat_channels[1])
        n = max(round(nums[2] * depth), 1)
        self.feat3_1 = Conv(in_ch, out_ch, kernel_size=3, stride=2, act=act)
        in_ch = int(width * feat_channels[1]) + int(width * feat_channels[2])
        out_ch = int(width * feat_channels[2])
        self.feat3_2 = C2f(in_ch, out_ch, n=n, shortcut=False, depthwise=depthwise, act=act)

    def forward(self, xs):
        x1, x2, x3 = xs
        x2 = self.feat1_1(torch.cat([x2, F.interpolate(x3, scale_factor=2, mode="nearest")], 1))
        x1 = self.feat1_2(torch.cat([x1, F.interpolate(x2, scale_factor=2, mode="nearest")], 1))
        x2 = self.feat2_2(torch.cat([x2, self.feat2_1(x1)], 1))
        x3 = self.feat3_2(torch.cat([x3, self.feat3_1(x2)], 1))
        return [x1, x2, x3]


class Detect(nn.Module):
    phase = 'train'
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, width, catenum, stride=(8, 16, 32), feat_channels=(256, 512, 1024), depthwise=False, act='silu'):  # detection layer
        super().__init__()
        self.catenum = catenum  # number of classes
        self.nl = len(feat_channels)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = catenum + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.Tensor(stride)  # strides computed during build
        Conv = DWConv if depthwise else BaseConv
        self.ch = [int(width * channel) for channel in feat_channels]
        c2, c3 = max((16, self.ch[0] // 4, self.reg_max * 4)), max(self.ch[0], min(self.catenum, 100))  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3, act=act),
                                               Conv(c2, c2, 3, act=act),
                                               nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in self.ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3, act=act),
                                               Conv(c3, c3, 3, act=act),
                                               nn.Conv2d(c3, self.catenum, 1)) for x in self.ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) # b, reg_max * 4 + cls, h, w
        return self.inference(x, shape) # b, 4 + cls, a

    def inference(self, preds, shape):
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors([pred.shape for pred in preds], self.stride, preds[0].dtype, preds[0].device, 0.5))
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in preds], 2) # b, reg_max * 4 + cls, a
        box, cls = x_cat.split((self.reg_max * 4, self.catenum), 1)
        b, _, a = box.shape
        box = box.view(b, 4, self.reg_max, a).transpose(2, 1).softmax(1)
        dbox = dist2bbox(self.dfl(box).view(b, 4, a), self.anchors.unsqueeze(0), xywh=True, dim=1)
        dbox = dbox * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""
    def __init__(self, width, catenum, stride=(8, 16, 32), feat_channels=(256, 512, 1024), depthwise=False, act='silu',
                 nm=32, npr=256):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(width, catenum, stride, feat_channels, depthwise, act)
        # self.no = catenum + self.reg_max * 4 +   # number of outputs per anchor
        self.nm = nm  # number of masks
        self.npr = int(width * npr)  # number of protos
        Conv = DWConv if depthwise else BaseConv
        self.proto = Proto(self.ch[0], self.npr, self.nm, depthwise, act)  # protos
        self.detect = Detect.forward

        c4 = max(self.ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3, act=act),
                                               Conv(c4, c4, 3, act=act),
                                               nn.Conv2d(c4, self.nm, 1)) for x in self.ch)

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # b, nm(32), a
        x = self.detect(self, x)
        #  b, 4 + cls + 32, a
        return torch.cat([x, mc], 1), p


class Proto(nn.Module):
    def __init__(self, c1, c_=256, c2=32, depthwise=False, act='silu'):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.cv1 = Conv(c1, c_, kernel_size=3, act=act)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, kernel_size=3, act=act)
        self.cv3 = Conv(c_, c2, kernel_size=1, act=act)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, depthwise=False, act='silu'):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        Conv = DWConv if depthwise else BaseConv
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, depthwise=False, act='silu', g=1, k=(3, 3), e=0.5): # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        Conv = DWConv if depthwise else BaseConv
        self.cv1 = Conv(c1, c_, k[0], 1, act=act)
        self.cv2 = Conv(c_, c2, k[1], 1, act=act)
        self.add = shortcut and c1 == c2
        self._add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        if self.add:
            return self._add.add(x, self.cv2(self.cv1(x)))
        else:
            return self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, depthwise=False, act='silu', g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = BaseConv(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = BaseConv((2 + n) * self.c, c2, 1, act=act)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, depthwise, act, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        return self.conv(x)


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor




