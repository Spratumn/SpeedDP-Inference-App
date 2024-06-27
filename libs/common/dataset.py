
import numpy as np
import cv2
import copy
from functools import partial
import torch


def resample_segments(segments, n=1000):
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
    return segments


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i+1)
    return masks, index


class Instance:
    def __init__(self, image=None, bboxes=[], segments=None,
                 image_size=None, format="xyxy", normalized=False):
        if image is None:
            self.width, self.height = image_size
        else:
            self.width = image.shape[1]
            self.height = image.shape[0]
        self.image = image
        self.bboxes = bboxes
        self.segments = segments
        self.shape_type = 'rectangle' if segments is None else 'polygon'
        self.format = format
        self.normalized = normalized
        self.indexes = None

    def convert(self, format):
        if len(self.bboxes) == 0: return
        if self.format == format:
            return
        self.format = format
        stack_func = np.hstack if isinstance(self.bboxes, np.ndarray) else torch.hstack
        if format == "xyxy":
            self.bboxes[:, 1:] = stack_func((self.bboxes[:, 1:3] - self.bboxes[:, 3:5] / 2,
                                            self.bboxes[:, 1:3] + self.bboxes[:, 3:5] / 2))
        else:
            self.bboxes[:, 1:] = stack_func(((self.bboxes[:, 3:5] + self.bboxes[:, 1:3]) / 2,
                                            self.bboxes[:, 3:5] - self.bboxes[:, 1:3]))

    def resize(self, outsize, force_direction=0):
        """
        force_direction=-1: dst_size <= outsize
        """
        if self.height == outsize[1] and self.width == outsize[0]: return
        dst_width, dst_height = outsize
        h_scale = self.height / dst_height
        w_scale = self.width / dst_width
        if force_direction == 0:
            dst_size = (dst_width, dst_height)
        elif force_direction == 1:
            dst_size = (dst_width, int(self.height / w_scale))
        elif force_direction == 2:
            dst_size = (int(self.width / h_scale), dst_height)
        else:
            if w_scale < h_scale:
                dst_size = (int(self.width / h_scale), dst_height)
            else:
                dst_size = (dst_width, int(self.height / w_scale))
        scale_w, scale_h = (dst_size[0] / self.width, dst_size[1] / self.height)
        if self.image is not None:
            self.image = cv2.resize(self.image, dst_size, interpolation=cv2.INTER_LINEAR)
        self.width, self.height = dst_size
        if len(self.bboxes) == 0: return
        self.bboxes[:, 1:4:2] *= scale_w
        self.bboxes[:, 2:5:2] *= scale_h
        if self.segments is None: return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h

    def filter(self, min_size=1, max_scale=-1):
        if len(self.bboxes) == 0: return
        if self.format == "xyxy":
            x_filter = self.bboxes[:,3] - self.bboxes[:,1] >= min_size
            y_filter = self.bboxes[:,4] - self.bboxes[:,2] >= min_size
        else:
            x_filter = self.bboxes[:,3] >= min_size
            y_filter = self.bboxes[:,4] >= min_size
        size_filter = np.logical_and(x_filter, y_filter)
        if not all(size_filter):
            self.bboxes = self.bboxes[size_filter]
            if self.segments is not None: self.segments = self.segments[size_filter]
        if len(self.bboxes) == 0 or max_scale == -1:
            return
        min_scale = 1.0 / max_scale
        if self.format == "xyxy":
            rate = np.divide(self.bboxes[:,4] - self.bboxes[:,2], self.bboxes[:,3] - self.bboxes[:,1])
        else:
            rate = np.divide(self.bboxes[:,4], self.bboxes[:,3])
        rate_filter = np.logical_and(rate >= min_scale, rate <= max_scale)
        self.bboxes = self.bboxes[rate_filter]
        if self.segments is not None: self.segments = self.segments[rate_filter]

    def copy(self):
        if self.image is not None:
            return Instance(copy.deepcopy(self.image),
                            copy.deepcopy(self.bboxes),
                            copy.deepcopy(self.segments),
                            format=self.format,
                            normalized=self.normalized)
        else:
            return Instance(None,
                            copy.deepcopy(self.bboxes),
                            copy.deepcopy(self.segments),
                            image_size=(self.width, self.height),
                            format=self.format,
                            normalized=self.normalized)

    def to(self, device):
        if self.image is not None:
            self.image = self.image.to(device)
        self.indexes = self.indexes.to(device)
        self.bboxes = self.bboxes.to(device)
        if self.shape_type == 'polygon':
            self.segments = self.segments.to(device)

    @property
    def shape(self):
        shape_info = f'({self.height}, {self.width})'
        if self.image is not None:
            shape_info += f' - {self.image.shape}'
        return shape_info

    def __len__(self):
        return len(self.bboxes)


def create_augmenters(augmenter_dict):
    _a = InstanceAugmenter()
    augmenters = []
    for augmenter in augmenter_dict:
        for augmenter_name, params in augmenter.items():
            if not hasattr(_a, augmenter_name): continue
            elif augmenter_name == 'normalize_image':
                mean = np.array([float(v) for v in params['mean'].split(',')])[np.newaxis, np.newaxis, :]
                std = np.array([float(v) for v in params['std'].split(',')])[np.newaxis, np.newaxis, :]
                augmenters.append(partial(_a.normalize_image, mean=mean, std=std))
            else:
                augmenters.append(partial(getattr(_a, augmenter_name), **params))
    return augmenters


class InstanceAugmenter:
    @staticmethod
    def normalize_image(instance:Instance, mean=127, std=128):
        image = instance.image.astype(np.float32)
        image -= mean
        image /= std
        instance.image = image

    @staticmethod
    def resize(instance:Instance, outsize=(640, 512), min_size=1, max_scale=-1):
        instance.resize(outsize, force_direction=0)
        instance.filter(min_size, max_scale)

    @staticmethod
    def to_tensor(instance:Instance):
        instance.image = np.transpose(instance.image, (2, 0, 1)).astype(np.float32)
        instance.image = torch.from_numpy(instance.image)


