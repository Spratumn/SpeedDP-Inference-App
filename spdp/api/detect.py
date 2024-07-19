import os
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
import json
from safetensors.torch import save_file as safe_save_file
from huggingface_hub import split_torch_state_dict_into_shards

from spdp.common.config import get_colormap
from spdp.models.creator import create_predictor
from spdp.common.config import make_project_config, load_settings




def draw_results(image, dets, catenames, colormaps, draw_label=False, draw_score=False, cate_ids=-1, rate=1):
    draw_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
    target_num = 0
    draw_mask = True if len(list(dets.keys())) and len(dets[list(dets.keys())[0]][0]) == 6 else False
    if draw_mask:
        height, width = image.shape[:2]
        mask_image = np.zeros((height, width, 3), np.uint8)
    for label in dets:
        if cate_ids != -1 and cate_ids and label not in cate_ids: continue
        target_num += len(dets[label])
        for i in range(len(dets[label])):
            x1, y1, x2, y2, score = dets[label][i][:5]
            x1, y1, x2, y2 = [int(v/rate) for v in [x1, y1, x2, y2]]
            catename = catenames[label]
            colormap = colormaps[label]
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), colormap, 2)
            mask = dets[label][i][5] if len(dets[label][i]) == 6 else None
            if draw_mask and mask is not None:
                image_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_image[image_mask>0] = colormap
            if draw_label or draw_score:
                if draw_label and draw_score:
                    text = '{}:{:.1f}%'.format(catename, score * 100)
                elif draw_label:
                    text = catename
                else:
                    text = '{:.1f}%'.format(score * 100)
                txt_color = (0, 0, 0) if np.mean(colormap) > 128 else (255, 255, 255)
                txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                txt_bg_color = [int(c*0.7) for c in colormap]
                cv2.rectangle(draw_image,
                            (x1 - 1, y1 - int(1.5*txt_size[1])),
                            (x1 + txt_size[0] + 1, y1),
                            txt_bg_color, -1)
                cv2.putText(draw_image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)
    if draw_mask:
        draw_image = cv2.addWeighted(draw_image, 1, mask_image, 0.3, 0)
    cv2.putText(draw_image, str(target_num), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=1)
    return draw_image



def save_state_dict(state_dict, save_directory: str):
    state_dict_split = split_torch_state_dict_into_shards(state_dict)
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        safe_save_file(
            shard,
            os.path.join(save_directory, filename),
            metadata={"format": "pt"},
        )
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(index, indent=2))


class Detector:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir): os.mkdir(cache_dir)

    def init(self, workdir, epxname, score_thresh=0.5, iou_thresh=0.5):
        settings = load_settings(os.path.join(workdir, '.prj'))
        train_settings = load_settings(os.path.join(workdir, epxname, '.train'))
        settings['model_type'] = train_settings['model_type']
        settings['inputsize'] = train_settings['inputsize']
        settings['augmenters'] = train_settings['augmenters']
        config = make_project_config(settings, phase='test')
        self.catenames = [catename for catename in settings['category_info']]
        self.colormaps = [get_colormap(i) for i in range(len(self.catenames))]
        self.predictor = create_predictor(config, settings['model_type'],
                                          os.path.join(workdir, epxname),
                                          score_thresh=score_thresh,
                                          iou_thresh=iou_thresh)
        if not os.path.exists(os.path.join(workdir, epxname, 'model.safetensors')):
            save_state_dict(self.predictor.model.state_dict(), os.path.join(workdir, epxname))

    def predict_image(self, image, draw_label=True, draw_score=True, cate_ids=-1):
        dets = self.predictor.predict(image)[0]
        return draw_results(image, dets, self.catenames, self.colormaps,
                            draw_label, draw_score, cate_ids)


    def predict_video(self, video, frame_num, draw_label=True, draw_score=True,
                      cate_ids=-1, rgb_input=False, result_path=''):
        vc = cv2.VideoCapture(video)
        frames = [vc.read()[1] for _ in range(int(vc.get(7)))]
        if frame_num > 0: frames = frames[:frame_num]
        for i, frame in enumerate(frames):
            if rgb_input: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = self.predictor.predict(frame)[0]
            frames[i] = cv2.cvtColor(draw_results(frame, dets, self.catenames, self.colormaps,
                                    draw_label, draw_score, cate_ids), cv2.COLOR_BGR2RGB)
        ImageSequenceClip(frames, vc.get(5)).write_videofile(result_path, codec='libx264', logger=None)
        vc.release()

