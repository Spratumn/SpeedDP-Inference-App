import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .loss import non_max_suppression
from .model import YOLOV8

from spdp.common.dataset import Instance, create_augmenters
from spdp.common.config import load_settings





class Predictor(object):
    def __init__(self, config, model_type, train_dir,
                 score_thresh=0.5, iou_thresh=0.5):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.inputsize = config.common.inputsize
        self.nc = config.common.catenum
        self.augmenter = create_augmenters(config.dataset.augmenters)
        train_settings = load_settings(os.path.join(train_dir, '.train'))
        self.task_type = config.common.task_type

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hf_model_path = os.path.join(train_dir, 'model.safetensors')
        pt_model_path = os.path.join(train_dir, 'model.pth')

        self.model = YOLOV8(config, model_type, options=train_settings['model_options'])
        self.model.set_device(self.device)
        if os.path.exists(hf_model_path):
            checkpoint = {}
            with safe_open(hf_model_path, framework="pt", device=0 if torch.cuda.is_available() else 'cpu') as f:
                for k in f.keys():
                    checkpoint[k] = f.get_tensor(k)
        else:
            if torch.cuda.is_available():
                checkpoint = torch.load(pt_model_path)
            else:
                checkpoint = torch.load(pt_model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.set_device(self.device)
        self.model.eval()
        self.model.set_phase('test')

    def predict(self, image):
        height, width = image.shape[:2]
        if len(image.shape) == 3 and image.shape[-1] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        instance = Instance(image)
        for aug in self.augmenter:
            aug(instance)
        image = torch.from_numpy(np.transpose(instance.image, (2, 0, 1)).astype(np.float32))
        image = image.to(self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image)
        return postprocess(outputs, self.inputsize, height, width,
                           self.score_thresh, self.iou_thresh,
                           task_type=self.task_type,
                           nc=self.nc)



def postprocess(outputs, model_size, height, width, score_thresh, iou_thresh, task_type='det', nc=0):
    if task_type == 'det':
        preds, proto = (outputs, None)
    else:
        preds, proto = outputs
    if torch.cuda.is_available(): preds = preds.detach()
    preds = non_max_suppression(preds, score_thresh, iou_thresh, nc=nc)
    results = []
    h_scale, w_scale = height / model_size[1], width / model_size[0]
    for i, pred in enumerate(preds):
        target_masks = [None] * pred.shape[0]
        if task_type == 'seg' and proto is not None:
            masks = process_mask(proto[i], pred[:, 6:], pred[:, :4], model_size[::-1])  # HWC
            if masks.shape[0]:
                masks = F.interpolate(masks[None], model_size[::-1], mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
                target_masks = masks.cpu().numpy()
        pred = pred.cpu().numpy()
        results_dict = {}
        for j in range(pred.shape[0]):
            x1, y1, x2, y2, score, label = pred[j][:6]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, model_size[0])
            y2 = min(y2, model_size[1])
            if x1 >= x2 or y1 >= y2: continue
            x1 = int(x1 * w_scale)
            y1 = int(y1 * h_scale)
            x2 = int(x2 * w_scale)
            y2 = int(y2 * h_scale)
            label = int(label)
            if label not in results_dict:
                results_dict[label] = [[x1, y1, x2, y2, score, target_masks[j]]]
            else:
                results_dict[label].append([x1, y1, x2, y2, score, target_masks[j]])
        results.append(results_dict)
    return results



def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))



def process_mask(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio
    return crop_mask(masks, downsampled_bboxes)  # CHW





