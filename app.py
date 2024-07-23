import os
import streamlit as st
from PIL import Image
import shutil

from utils import PROJECT_DIR, detect_images, detect_video, get_project_info, get_exp_info, get_trainsize


INFO_LINES = """
    ## User guide:
    1. select project from **Projects**;
    2. select experiment from **Experiments**;
    3. set **Score thresh**, **IOU thresh** and other settings;
    4. select and upload source file(s);
    5. click **Predict** button;
"""


st.set_page_config(
    page_title="SpeedDPHF",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.vizvision.com/',
        'Report a bug': "https://www.vizvision.com/",
        'About': "这是一个视觉算法工具集。"
    }
)


if 'video' not in st.session_state: st.session_state.video = None
if 'video_result' not in st.session_state: st.session_state.video_result = []
if 'images' not in st.session_state: st.session_state.images = []
if 'images_result' not in st.session_state: st.session_state.images_result = []
if 'image_idx' not in st.session_state: st.session_state.image_idx = 0
if 'config' not in st.session_state:
    prj = os.listdir(PROJECT_DIR)[0]
    exps = [
            exp for exp in os.listdir(os.path.join(PROJECT_DIR, prj))
            if os.path.isdir(os.path.join(PROJECT_DIR, prj, exp))
        ]
    st.session_state.config = {
        'prj': prj,
        'exps': exps,
        'prj_info': get_project_info(prj),
        'exp_info': get_exp_info(prj, exps[0])
    }
if 'inputsize_height' not in st.session_state:
    (st.session_state.inputsize_width, st.session_state.inputsize_height) = get_trainsize(prj, exps[0])


def project_changed():
    exps = [
        exp for exp in os.listdir(os.path.join(PROJECT_DIR, st.session_state.project_name))
        if os.path.isdir(os.path.join(PROJECT_DIR, st.session_state.project_name, exp))
    ]
    st.session_state.config['exps'] = exps
    st.session_state.config['prj'] = st.session_state.project_name
    st.session_state.config['prj_info'] = get_project_info(st.session_state.project_name)
    st.session_state.config['exp_info'] = get_exp_info(st.session_state.project_name, exps[0])
    (st.session_state.inputsize_width, st.session_state.inputsize_height) = get_trainsize(st.session_state.project_name,
                                                                                          exps[0])


def exp_changed():
    st.session_state.config['exp_info'] = get_exp_info(st.session_state.project_name,
                                                       st.session_state.exp_name)
    (st.session_state.inputsize_width, st.session_state.inputsize_height) = get_trainsize(st.session_state.project_name,
                                                                                          st.session_state.exp_name)


def image_idx_plus():
    st.session_state.image_idx += 1


def image_idx_minus():
    if st.session_state.image_idx > 0: st.session_state.image_idx -= 1


st.markdown(INFO_LINES)

st.subheader('Settings')
c1, c2 = st.columns(2)
with c1:
    c111, c112 = st.columns(2)
    with c111:
        project_name = st.selectbox('Projects:',
                                    tuple(os.listdir(PROJECT_DIR)),
                                    index=0,
                                    on_change=project_changed,
                                    key='project_name')
    with c112:
        exp_name = st.selectbox('Experiments:',
                                tuple(st.session_state.config['exps']),
                                index=0,
                                on_change=exp_changed,
                                key='exp_name')
    c121, c122, c123 = st.columns([1, 1, 2])
    with c121:
        inputsize_width = st.number_input('Inputsize width:', 320, 960,
                                          st.session_state.inputsize_width, 32)
    with c122:
        inputsize_height = st.number_input('Inputsize height:', 320, 960,
                                           st.session_state.inputsize_height, 32)
    with c123:
        score_thresh = st.slider('Score thresh:', 0.1, 1.0, 0.5, step=0.05)
    c131, c132, c133 = st.columns([1, 1, 2])
    with c131:
        line_width = st.number_input('Box line width:', max_value=5, min_value=1, value=2)
    with c133:
        iou_thresh = st.slider('IOU thresh:', 0.1, 1.0, 0.5, step=0.05)
    c141, c142, c143, c144 = st.columns(4)
    with c141:
        draw_label = st.checkbox('Draw class', True)
    with c142:
        draw_score = st.checkbox('Draw score', True)
    with c143:
        rgb_input = st.checkbox('RGB input', False)

with c2:
    c21, c22 = st.columns(2)
    with c21:
        prj_info = st.session_state.config['prj_info']
        st.write(f"Task Type: {prj_info['task_type']}")
        st.write(f"Image Type: {prj_info['image_type']}")
        st.write(f"Class Num: {len(prj_info['catenames'])}")
    with c22:
        exp_info = st.session_state.config['exp_info']
        st.write(f"Model Type: {exp_info['model_type']}")
        st.write(f"Train Insize: {exp_info['inputsize']}")
        st.write(f"Train Qat: {exp_info['qat']}")
    with st.expander("Class Names:"):
        st.write('#' + ' #'.join(prj_info['catenames']))


st.subheader('Predict')
tab_image, tab_video = st.tabs(["Image", "Video"])

with tab_image:
    image_col1, img_bt_col, image_col2 = st.columns([14, 2, 14])
    with st.form(key='tab_image', clear_on_submit=True):
        image_upload_c1, image_upload_c2 = st.columns([12, 1])
        with image_upload_c1:
            images_upload = st.file_uploader("Select image", type=["png", "jpg", "jpeg"],
                                            label_visibility='hidden',
                                            accept_multiple_files=True)
        with image_upload_c2:
            st.write('\n')
            st.write('\n')
            images_submitted = st.form_submit_button("Upload selected files")
        if len(images_upload) and images_submitted:
            st.session_state.image_idx = 0
            st.session_state.images.clear()
            st.session_state.images_result.clear()
            for image_upload in images_upload:
                if images_upload is None: continue
                st.session_state.images.append(Image.open(image_upload))

    if len(st.session_state.images):
        with img_bt_col:
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
        img_bt_col.button(label='last', on_click=image_idx_minus, use_container_width=True)
        img_bt_col.write('\n')
        img_bt_col.write('\n')
        img_bt_col.write('\n')
        img_bt_col.button(label='next', on_click=image_idx_plus, use_container_width=True)
        img_bt_c21, img_bt_c22 = st.columns([1, 12])
        if img_bt_c21.button("Predict", use_container_width=True):
            with img_bt_c22:
                with st.spinner('Running Detection with the given images...'):
                    st.session_state.images_result = detect_images(project_name, exp_name,
                                                                   (st.session_state.inputsize_width,
                                                                    st.session_state.inputsize_height),
                                                                   score_thresh, iou_thresh,
                                                                   draw_score, draw_label, line_width,
                                                                   rgb_input,
                                                                   st.session_state.images)

        st.session_state.image_idx = min(st.session_state.image_idx, len(st.session_state.images) - 1)
        image_col1.image(st.session_state.images[st.session_state.image_idx], width=800)
        if len(st.session_state.images_result):
            image_col2.image(st.session_state.images_result[st.session_state.image_idx], width=800)


with tab_video:
    video_col11, video_col12 = st.columns(2)
    with st.form(key='tab_video', clear_on_submit=True):
        video_upload_c1, video_upload_c2 = st.columns([12, 1])
        with video_upload_c1:
            video_upload = st.file_uploader("Select video", type=["mp4",], label_visibility='hidden')
        with video_upload_c2:
            st.write('\n')
            st.write('\n')
            video_submitted = st.form_submit_button("Upload selected file")
        if video_upload is not None and video_submitted:
            st.session_state.video = video_upload.read()
            st.session_state.video_result = None
    if st.session_state.video is not None:
        video_col11.video(st.session_state.video, autoplay=True)
        video_col21, video_col22, video_col23 = st.columns([1, 1, 10])
        with video_col21:
            frame_num = st.number_input('frame num:', max_value=5000, min_value=-1, value=50)
        with video_col22:
            st.write('\n')
            st.write('\n')
        if video_col22.button("Predict"):
            with video_col23:
                st.write('\n')
                st.write('\n')
                with st.spinner('Running Detection with the given video...'):
                    result_video_path = detect_video(project_name, exp_name,
                                                     (st.session_state.inputsize_width,
                                                      st.session_state.inputsize_height),
                                                     score_thresh, iou_thresh,
                                                     draw_score, draw_label, line_width,
                                                     rgb_input, frame_num,
                                                     st.session_state.video)
                if not os.path.exists(result_video_path):
                    st.error('Run detection error!')
                else:
                    with open(result_video_path, 'rb') as f:
                        st.session_state.video_result = f.read()
                    shutil.rmtree(os.path.dirname(result_video_path))
        if st.session_state.video_result is not None:
            video_col12.video(st.session_state.video_result, autoplay=True)


