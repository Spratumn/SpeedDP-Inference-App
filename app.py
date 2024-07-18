import os
import streamlit as st
from PIL import Image
import shutil

from utils import PROJECT_DIR, detect_image, detect_video, get_project_info, get_exp_info


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


st.title(":mag_right: 目标检测与分割")
st.subheader(':gear: 参数配置')

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
        iou_thresh = st.slider('IOU阈值:', 0.3, 1.0, 0.5, step=0.05)
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


st.subheader(':film_frames: 单图像目标检测')
st.write(
    "选择并上传一张图像，运行目标检测任务."
)
detect_succeed = False
image_col1, image_col2 = st.columns(2)
image_upload = st.file_uploader("上传一张图像", type=["png", "jpg", "jpeg"], label_visibility='hidden')
if image_upload is not None:
    image = Image.open(image_upload)
    image_col1.write("原图:")
    image_col1.image(image)
else:
    image = None
image_c2, image_c3 = st.columns(2)
with image_c2:
    if st.button("图像检测 :point_left:"):
        if image_upload is not None:
            result_image = detect_image(project_name, exp_name,
                                        score_thresh, iou_thresh,
                                        draw_score, draw_label,
                                        rgb_input, image)
            if result_image is not None:
                image_col2.write("检测结果:")
                image_col2.image(result_image)
                detect_succeed = True
        else:
            st.error('请先上传一张图像，然后再点击 “运行单图像检测” 按钮')


st.subheader(':camera: 视频帧检测')
st.write(
    "选择并上传一个mp4格式的视频文件,然后逐帧做目标检测"
)
video_det_num = st.number_input('运行目标检测的视频帧数 :', max_value=5000, min_value=-1, value=-1)
video_col1, video_col2 = st.columns(2)
video_upload = st.file_uploader("上传视频", type=["mp4",], label_visibility='hidden')
if video_upload is not None:
    video_col1.write("原视频:")
    video_col1.video(video_upload.read())
video_data = None
video_c2, _, video_c3, _ = st.columns([2, 2, 2, 2])
with video_c2:
    if st.button("视频检测 :point_left:"):
        if video_upload is not None:
            result_video_path = detect_video(project_name, exp_name,
                                            score_thresh, iou_thresh,
                                            draw_score, draw_label,
                                            rgb_input, video_det_num,
                                            video_upload)
            if not os.path.exists(result_video_path):
                st.error('Run detection error!')
            else:
                with open(result_video_path, 'rb') as f:
                    video_data = f.read()
                video_col2.write("检测结果:")
                video_col2.video(video_data)
                shutil.rmtree(os.path.dirname(result_video_path))
        else:
            st.error('请先上传一个视频，然后再点击 “运行视频帧检测” 按钮')



