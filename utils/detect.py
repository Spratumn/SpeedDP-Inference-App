import os
import time
from io import BytesIO
from PIL import Image
import streamlit as st
import cv2
import numpy as np


from .main import TMP_DIR, PROJECT_DIR
from spdp.api.detect import Detector


def convert_image(image:Image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_image = buf.getvalue()
    return byte_image


def detect_image(prjname, expname, score_thresh, iou_thresh, draw_score, draw_label, rgb_input, image:Image):
    detector = Detector(TMP_DIR)
    detector.init(os.path.join(PROJECT_DIR, prjname), expname, score_thresh, iou_thresh)
    with st.spinner('Running Detection with the given image...'):
        image_np = np.asarray(image)
        if not rgb_input:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        result_image = detector.predict_image(image_np,
                                              draw_label=draw_label,
                                              draw_score=draw_score)
    return Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), 'done'


def detect_video(prjname, expname, score_thresh,
                 iou_thresh, draw_score,
                 draw_label, rgb_input,
                 det_num, video):
    time_uid = str(time.time())
    detector = Detector(TMP_DIR)
    detector.init(os.path.join(PROJECT_DIR, prjname), expname, score_thresh, iou_thresh)
    cur_video_dir = os.path.join(TMP_DIR, time_uid)
    if not os.path.exists(cur_video_dir): os.mkdir(cur_video_dir)
    video_path = os.path.join(cur_video_dir, video.name)
    byte_video = video.getvalue()
    with open(video_path, 'wb') as f:
        f.write(byte_video)

    with st.spinner('Running Detection with the given video...'):
        result_video_path = os.path.join(cur_video_dir, f'{video.name}-det.mp4')
        detector.predict_video(video_path, frame_num=det_num,
                               draw_label=draw_label,
                               draw_score=draw_score,
                               rgb_input=rgb_input,
                               result_path=result_video_path)
        return result_video_path, 'done'

