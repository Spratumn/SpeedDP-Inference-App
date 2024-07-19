import os
import streamlit as st
from PIL import Image
import shutil

from utils import PROJECT_DIR, detect_images, detect_video, get_project_info, get_exp_info


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


if 'images' not in st.session_state: st.session_state.images = []
if 'results' not in st.session_state: st.session_state.results = []
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


def project_changed():
    exps = [
        exp for exp in os.listdir(os.path.join(PROJECT_DIR, st.session_state.project_name))
        if os.path.isdir(os.path.join(PROJECT_DIR, st.session_state.project_name, exp))
    ]
    st.session_state.config['exps'] = exps
    st.session_state.config['prj'] = st.session_state.project_name
    st.session_state.config['prj_info'] = get_project_info(st.session_state.project_name)
    st.session_state.config['exp_info'] = get_exp_info(st.session_state.project_name, exps[0])


def exp_changed():
    st.session_state.config['exp_info'] = get_exp_info(st.session_state.project_name,
                                                       st.session_state.exp_name)

def image_idx_plus():
    st.session_state.image_idx += 1


def image_idx_minus():
    if st.session_state.image_idx > 0: st.session_state.image_idx -= 1


st.subheader(':gear: 配置参数')
c11, c12 = st.columns(2)
with c11:
    c111, c112 = st.columns(2)
    with c111:
        project_name = st.selectbox('项目:',
                                    tuple(os.listdir(PROJECT_DIR)),
                                    index=0,
                                    on_change=project_changed,
                                    key='project_name')
    with c112:
        exp_name = st.selectbox('试验:',
                                tuple(st.session_state.config['exps']),
                                index=0,
                                on_change=exp_changed,
                                key='exp_name')
    c121, c122 = st.columns(2)
    with c121:
        score_thresh = st.slider('目标置信度:', 0.1, 1.0, 0.5, step=0.05)

    with c122:
        iou_thresh = st.slider('IOU阈值:', 0.1, 1.0, 0.5, step=0.05)
    c131, c132, c133, _ = st.columns(4)
    with c131:
        draw_label = st.checkbox('绘制目标标签', True)
    with c132:
        draw_score = st.checkbox('绘制置信度', True)
    with c133:
        rgb_input = st.checkbox('RGB input', False)

with c12:
    c121, c122 = st.columns(2)
    with c121:
        prj_info = st.session_state.config['prj_info']
        st.write(f"任务类型: {prj_info['任务类型']}")
        st.write(f"图像类型: {prj_info['图像类型']}")
        st.write(f"目标数量: {len(prj_info['目标类别'])}")
        with st.expander("目标类别信息:"):
            st.write('#' + ' #'.join(prj_info['目标类别']))
    with c122:
        exp_info = st.session_state.config['exp_info']
        st.write(f"模型类型: {exp_info['模型类型']}")
        st.write(f"训练尺寸: {exp_info['训练尺寸']}")
        st.write(f"量化训练: {exp_info['量化训练']}")


st.subheader(':gear: 运行推理')
tab_image, tab_video = st.tabs(["图像文件", "视频文件"])

with tab_image:
    image_col1, img_bt_col, image_col2 = st.columns([14, 2, 14])
    with st.form(key='tab_image', clear_on_submit=True):
        images_upload = st.file_uploader("上传图像", type=["png", "jpg", "jpeg"],
                                        label_visibility='hidden',
                                        accept_multiple_files=True)
        images_submitted = st.form_submit_button("上传选择的文件")
        if images_submitted and len(images_upload):
            st.session_state.image_idx = 0
            st.session_state.images.clear()
            st.session_state.results.clear()
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
        img_bt_col.button(label='上一张', on_click=image_idx_minus, use_container_width=True)
        img_bt_col.button(label='下一张', on_click=image_idx_plus, use_container_width=True)
        if img_bt_col.button("运行推理", use_container_width=True):
            with img_bt_col:
                with st.spinner('...'):
                    st.session_state.results = detect_images(project_name, exp_name,
                                                            score_thresh, iou_thresh,
                                                            draw_score, draw_label,
                                                            rgb_input,
                                                            st.session_state.images)

        st.session_state.image_idx = min(st.session_state.image_idx, len(st.session_state.images) - 1)
        image_col1.image(st.session_state.images[st.session_state.image_idx], width=800)
        if len(st.session_state.results):
            image_col2.image(st.session_state.results[st.session_state.image_idx], width=800)


with tab_video:
    video_col11, video_col12 = st.columns(2)
    video_upload = st.file_uploader("上传视频", type=["mp4",], label_visibility='hidden')
    if video_upload is not None:
        video_col11.video(video_upload.read(), autoplay=True)
        video_col21, video_col22, video_col23 = st.columns([1, 1, 10])
        with video_col21:
            frame_num = st.number_input('视频帧数 :', max_value=5000, min_value=-1, value=50)
        with video_col22:
            st.write('\n')
            st.write('\n')
        if video_col22.button("运行推理"):
            if video_upload is not None:
                with video_col23:
                    st.write('\n')
                    st.write('\n')
                    with st.spinner('Running Detection with the given video...'):
                        result_video_path = detect_video(project_name, exp_name,
                                                        score_thresh, iou_thresh,
                                                        draw_score, draw_label,
                                                        rgb_input, frame_num, video_upload)
                if not os.path.exists(result_video_path):
                    with video_col23:
                        st.error('Run detection error!')
                else:
                    with open(result_video_path, 'rb') as f:
                        video_data = f.read()
                    video_col12.video(video_data, autoplay=True)
                    shutil.rmtree(os.path.dirname(result_video_path))



