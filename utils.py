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
        'task_type': prj_settings['task_type'],
        'image_type': prj_settings['image_type'],
        'catenames': [catename for catename in prj_settings['category_info'].keys()]
    }



def get_exp_info(project_name, exp_name):
    exp_settings = load_settings(os.path.join(PROJECT_DIR, project_name, exp_name, '.train'))
    return {
        'model_type': exp_settings['model_type'],
        'inputsize': f"[{exp_settings['inputsize'][0]}, {exp_settings['inputsize'][1]}]",
        'qat': 'True' if exp_settings['qscheme'] else 'False'
    }


def get_trainsize(project_name, exp_name):
    exp_settings = load_settings(os.path.join(PROJECT_DIR, project_name, exp_name, '.train'))
    return exp_settings['inputsize']



def convert_image(image:Image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_image = buf.getvalue()
    return byte_image


def detect_images(prjname, expname, inputsize,
                  score_thresh, iou_thresh,
                  draw_score, draw_label,
                  rgb_input,
                  images:Image):
    detector = Detector(TMP_DIR)
    detector.init(os.path.join(PROJECT_DIR, prjname), expname, score_thresh, iou_thresh, inputsize)
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


def detect_video(prjname, expname, inputsize,
                 score_thresh, iou_thresh,
                 draw_score, draw_label,
                 rgb_input,
                 det_num, video):
    time_uid = str(time.time())
    detector = Detector(TMP_DIR)
    detector.init(os.path.join(PROJECT_DIR, prjname), expname, score_thresh, iou_thresh, inputsize)
    cur_video_dir = os.path.join(TMP_DIR, time_uid)
    if not os.path.exists(cur_video_dir): os.mkdir(cur_video_dir)
    video_path = os.path.join(cur_video_dir, 'source.mp4')
    with open(video_path, 'wb') as f:
        f.write(video)
    result_video_path = os.path.join(cur_video_dir, f'result.mp4')
    detector.predict_video(video_path, frame_num=det_num,
                            draw_label=draw_label,
                            draw_score=draw_score,
                            rgb_input=rgb_input,
                            result_path=result_video_path)
    return result_video_path

