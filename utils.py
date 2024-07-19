import os
import sys
import time
from io import BytesIO
from PIL import Image
import streamlit as st
import cv2
import numpy as np

from spdp.api.detect import Detector
from spdp.common.config import load_settings

TMP_DIR = '.cache'
PROJECT_DIR = './Projects'


def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True


def get_project_info(project_name):
    prj_settings = load_settings(os.path.join(PROJECT_DIR, project_name, '.prj'))
    return {
        '任务类型': prj_settings['task_type'],
        '图像类型': prj_settings['image_type'],
        '目标类别': [catename for catename in prj_settings['category_info'].keys()]
    }



def get_exp_info(project_name, exp_name):
    exp_settings = load_settings(os.path.join(PROJECT_DIR, project_name, exp_name, '.train'))
    return {
        '模型类型': exp_settings['model_type'],
        '训练尺寸': f"[{exp_settings['inputsize'][0]}, {exp_settings['inputsize'][1]}]",
        '量化训练': '是' if exp_settings['qscheme'] else '否'
    }



def convert_image(image:Image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_image = buf.getvalue()
    return byte_image


def detect_images(prjname, expname, score_thresh, iou_thresh, draw_score, draw_label, rgb_input, images:Image):
    detector = Detector(TMP_DIR)
    detector.init(os.path.join(PROJECT_DIR, prjname), expname, score_thresh, iou_thresh)
    results = []
    for image in images:
        image_np = np.asarray(image)
        if not rgb_input:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        result_image = detector.predict_image(image_np,
                                            draw_label=draw_label,
                                            draw_score=draw_score)
        results.append(Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)))
    return results


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
    result_video_path = os.path.join(cur_video_dir, f'{video.name}-det.mp4')
    detector.predict_video(video_path, frame_num=det_num,
                            draw_label=draw_label,
                            draw_score=draw_score,
                            rgb_input=rgb_input,
                            result_path=result_video_path)
    return result_video_path

