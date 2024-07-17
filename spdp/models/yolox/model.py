import torch
import torch.nn as nn
import torch.nn.functional as F


from .config import create_model_config
from ..module import BaseConv, Focus, CSPLayer, DWConv, SPPBottleneck



class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul,
                 feat_channels=(128, 256, 512, 1024),
                 out_idxes=(1, 2, 3),
                 depthwise=False,
                 act="relu",
                 input_channel=3):
        super().__init__()
        self.last_layer = max(out_idxes)
        assert 0 <= self.last_layer < 4
        #                 0       1          2        3
        #                 4       8         16       32
        out_names = ("dark2", "dark3", "dark4", "dark5")
        self.out_names = [out_names[idx] for idx in out_idxes]

        Conv = DWConv if depthwise else BaseConv
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # stem 64, 1/2
        in_ch, out_ch = input_channel, int(wid_mul * 64)
        self.stem = Focus(in_ch, out_ch, 3, act=act)

        # dark2 128, 1/4
        in_ch, out_ch = out_ch, int(wid_mul * feat_channels[0])
        self.dark2 = nn.Sequential(
            Conv(in_ch, out_ch, 3, 2, act=act),
            CSPLayer(out_ch, out_ch, n=base_depth, depthwise=depthwise, act=act)
        )

        # dark3 256, 1/8
        in_ch, out_ch = out_ch, int(wid_mul * feat_channels[1])
        self.dark3 = nn.Sequential(
            Conv(in_ch, out_ch, 3, 2, act=act),
            CSPLayer(out_ch, out_ch, n=base_depth * 3, depthwise=depthwise, act=act)
        )

        # dark4 512, 1/16
        in_ch, out_ch = out_ch, int(wid_mul * feat_channels[2])
        self.dark4 = nn.Sequential(
            Conv(in_ch, out_ch, 3, 2, act=act),
            CSPLayer(out_ch, out_ch, n=base_depth * 3, depthwise=depthwise, act=act)
        )

        # dark5 1024, 1/32
        in_ch, out_ch = out_ch, int(wid_mul * feat_channels[3])
        self.dark5 = nn.Sequential(
            Conv(in_ch, out_ch, 3, 2, act=act),
            SPPBottleneck(out_ch, out_ch, activation=act),
            CSPLayer(out_ch, out_ch, n=base_depth, shortcut=False, depthwise=depthwise, act=act)
        )

    def set_device(self, device):
        self.stem.set_device(device)
        self.stem = self.stem.to(device)
        self.dark2 = self.dark2.to(device)
        self.dark3 = self.dark3.to(device)
        self.dark4 = self.dark4.to(device)
        self.dark5 = self.dark5.to(device)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        if self.last_layer >= 1:
            x = self.dark3(x)
            outputs["dark3"] = x
        if self.last_layer >= 2:
            x = self.dark4(x)
            outputs["dark4"] = x
        if self.last_layer >= 3:
            x = self.dark5(x)
            outputs["dark5"] = x
        return [outputs[name] for name in self.out_names]


class YOLOX(nn.Module):
    def __init__(self, config, model_type, options=None):
        super().__init__()
        model_config, self.uid, self.model_config_str = create_model_config(config, model_type, options)
        self.phase = model_config.phase
        self.input_channel = model_config.input_channel
        stride = [4, 8, 16, 32]
        self.stride = [stride[idx] for idx in model_config.out_idxes]
        self.backbone = CSPDarknet(model_config.depth,
                                   model_config.width,
                                   feat_channels=model_config.feat_channels,
                                   out_idxes=model_config.out_idxes,
                                   depthwise=model_config.backbone.depthwise,
                                   act=model_config.backbone.act,
                                   input_channel=model_config.input_channel)
        feat_channels = [model_config.feat_channels[idx] for idx in model_config.out_idxes]
        if model_config.fpn.name == 'YOLOPAFPN':
            yolo_fpn = YOLOPAFPN
        else:
            assert model_config.fpn.name == 'YOLOPAN'
            yolo_fpn = YOLOPAN
        self.fpn = yolo_fpn(model_config.depth, model_config.width,
                           feat_channels=feat_channels,
                           depthwise=model_config.fpn.depthwise,
                           act=model_config.fpn.act)

        self.head = YOLOXHead(num_classes=model_config.catenum,
                              width=model_config.width,
                              stride=self.stride,
                              feat_channels=feat_channels,
                              depthwise=model_config.head.depthwise,
                              act=model_config.head.act,
                              phase=model_config.phase)

    def set_device(self, device):
        self.device = device
        self.backbone.set_device(device)
        self.backbone = self.backbone.to(device)
        self.fpn = self.fpn.to(device)
        self.head = self.head.to(device)

    def set_phase(self, phase='train'):
        self.phase = phase
        self.head.phase = self.phase

    def forward(self, x, targets=None):
        backbone_outs = self.backbone(x)
        fpn_outs = self.fpn(backbone_outs)
        outputs = self.head(fpn_outs)
        return outputs

    def forward_eval(self, x, targets, with_loss=True):
        backbone_outs = self.backbone(x)
        fpn_outs = self.fpn(backbone_outs)
        return self.head.forward_eval(fpn_outs, targets, with_loss)


class YOLOPAN(nn.Module):
    """
    feat_channels: [128, 256, 512, 1024]
    """
    def __init__(self, depth=1.0, width=1.0, feat_channels=(128, 256, 512, 1024), depthwise=False, act="silu"):
        super().__init__()
        self.feat_channels = feat_channels
        Conv = DWConv if depthwise else BaseConv
        self.iterNums = len(feat_channels) - 1

        self.up_bases = nn.ModuleList()
        self.up_csps = nn.ModuleList()
        for i in range(self.iterNums, 0, -1):
            self.up_bases.append(BaseConv(int(feat_channels[i] * width), int(feat_channels[i-1] * width), 1, 1, act=act))
            self.up_csps.append(CSPLayer(int(feat_channels[i] * width),
                                         int(feat_channels[i-1] * width),
                                         round(3 * depth),
                                         False,
                                         depthwise=depthwise,
                                         act=act))
        self.down_bases = nn.ModuleList()
        self.down_csps = nn.ModuleList()
        for i in range(self.iterNums):
            self.down_bases.append(Conv(int(feat_channels[i] * width), int(feat_channels[i] * width), 3, 2, act=act))
            self.down_csps.append(CSPLayer(int(feat_channels[i+1] * width),
                                           int(feat_channels[i+1] * width),
                                           round(3 * depth),
                                           False,
                                           depthwise=depthwise,
                                           act=act))

    def forward(self, input):
        assert len(input) == self.iterNums + 1
        #  1/4, 1/8, 1/16,1/32
        # [128, 256, 512, 1024]

        # up forward
        for i in range(self.iterNums):
            input[self.iterNums-i] = self.up_bases[i](input[self.iterNums-i])
            xup = F.interpolate(input[self.iterNums-i], scale_factor=2, mode="nearest")
            input[self.iterNums-i-1] = torch.cat([xup, input[self.iterNums-i-1]], 1)
            input[self.iterNums-i-1] = self.up_csps[i](input[self.iterNums-i-1])

        # down forward
        for i in range(self.iterNums):
            xdown = self.down_bases[i](input[i])
            input[i+1] = torch.cat([xdown, input[i+1]], 1)
            input[i+1] = self.down_csps[i](input[i+1])
        return input


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    feat_channels: [256, 512, 1024]
    """

    def __init__(self, depth=1.0, width=1.0, feat_channels=(256, 512, 1024), depthwise=False, act="silu"):
        super().__init__()
        self.feat_channels = feat_channels
        Conv = DWConv if depthwise else BaseConv

        self.lateral_conv0 = BaseConv(int(feat_channels[2] * width), int(feat_channels[1] * width), 1, 1, act=act)
        self.c3_p4 = CSPLayer(
            int(2 * feat_channels[1] * width),
            int(feat_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(int(feat_channels[1] * width), int(feat_channels[0] * width), 1, 1, act=act)
        self.c3_p3 = CSPLayer(
            int(2 * feat_channels[0] * width),
            int(feat_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(int(feat_channels[0] * width), int(feat_channels[0] * width), 3, 2, act=act)
        self.c3_n3 = CSPLayer(
            int(2 * feat_channels[0] * width),
            int(feat_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(int(feat_channels[1] * width), int(feat_channels[1] * width), 3, 2, act=act)
        self.c3_n4 = CSPLayer(
            int(2 * feat_channels[1] * width),
            int(feat_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        x2, x1, x0 = input

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = F.interpolate(fpn_out0, scale_factor=2, mode="nearest")  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.c3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = F.interpolate(fpn_out1, scale_factor=2, mode="nearest")  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.c3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.c3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.c3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs



###################################################################################################


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, stride=(8, 16, 32),
                 feat_channels=(256, 512, 1024), act="silu",
                 depthwise=False, phase='train'):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
            feat_channels: [[128, 256, 512, 1024]]
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.phase = phase
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(feat_channels)):
            self.stems.append(BaseConv(int(feat_channels[i] * width), int(256 * width), 1, stride=1, act=act))
            self.cls_convs.append(
                nn.Sequential(*[
                    Conv(int(256 * width), int(256 * width), 3, stride=1, act=act),
                    Conv(int(256 * width), int(256 * width), 3, stride=1, act=act)
                    ])
            )
            self.reg_convs.append(
                nn.Sequential(*[
                    Conv(int(256 * width), int(256 * width), 3, stride=1, act=act),
                    Conv(int(256 * width), int(256 * width), 3, stride=1, act=act),
                    ])
            )
            self.cls_preds.append(nn.Conv2d(int(256 * width), self.n_anchors * self.num_classes, 1, stride=1, padding=0))
            self.reg_preds.append(nn.Conv2d(int(256 * width), 4, 1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(int(256 * width), self.n_anchors, 1, stride=1, padding=0))

        self.stride = stride
        self.grids = [torch.zeros(1)] * len(feat_channels)
        self.expanded_strides = [None] * len(feat_channels)

    def forward(self, xin, targets=None):
        """
        Args:
            xin:(batch_size, anchor_attr, grid_w, grid_h)
            targets: Instance(indexes, bboxes, segments)
        Returns:
        """
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.stride, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.phase == 'train':
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        self.hw = [x.shape[-2:] for x in outputs]
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        return self.decode_outputs(outputs, dtype=xin[0].type())


    def get_output_and_grid(self, output, k, stride, dtype):
        """
        将输出的位置映射到原图尺度
        output: b, n_a, n_c
        grid: 1, n_a, 2
        """
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            if torch.__version__.split('.')[1] == '9':
                yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            else:
                yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1)

        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.stride):
            if torch.__version__.split('.')[1] == '9':
                yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            else:
                yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride, dtype=torch.int32))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
