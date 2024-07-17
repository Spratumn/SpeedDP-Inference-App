import os
import cv2
import torch
import numpy as np
from torchvision.ops import nms
from safetensors import safe_open

from .model import YOLOX

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hf_model_path = os.path.join(train_dir, 'model.safetensors')
        pt_model_path = os.path.join(train_dir, 'model.pth')

        self.model = YOLOX(config, model_type, options=train_settings['model_options'])
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
        return postprocess(outputs, self.inputsize, height, width, self.score_thresh, self.iou_thresh)



def postprocess(prediction, model_size, height, width, score_thresh=0.7, iou_thresh=0.45, task_type='det', nc=0):
    if torch.cuda.is_available(): prediction = prediction.detach()
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    results = []

    h_scale, w_scale = height / model_size[1], width / model_size[0]

    for image_pred in prediction:
        results_dict = {}
        if image_pred.size(0) == 0: continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= score_thresh).squeeze()
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        keep_indexes = nms(boxes=detections[:, :4], scores=detections[:,4]*detections[:,5], iou_threshold=iou_thresh)
        for index in keep_indexes:
            det = detections[index].cpu().numpy()
            x1, y1, x2, y2 = det[:4]
            if not 0 <= x1 < x2 < model_size[0]: continue
            if not 0 <= y1 < y2 < model_size[1]: continue
            score, label = det[4] * det[5], int(det[6])
            x1 = int(x1 * w_scale)
            y1 = int(y1 * h_scale)
            x2 = int(x2 * w_scale)
            y2 = int(y2 * h_scale)
            if label not in results_dict:
                results_dict[label] = [[x1, y1, x2, y2, score]]
            else:
                results_dict[label].append([x1, y1, x2, y2, score])
        results.append(results_dict)
    return results


