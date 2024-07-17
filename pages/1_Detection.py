import streamlit as st
from PIL import Image
import shutil
from utils.detect import *
from utils.main import *



if 'det_config' not in st.session_state:
    prjname = os.listdir(PROJECT_DIR)[0]
    exp_names = [
        exp for exp in os.listdir(os.path.join(PROJECT_DIR, prjname))
        if os.path.isdir(os.path.join(PROJECT_DIR, prjname, exp))
    ]
    st.session_state.det_config = {
        'project_name': prjname,
        'exp_name': exp_names[0],
        'exp_names': exp_names,
        'score_thresh': 0.5,
        'iou_thresh': 0.5,
        'rgb_input': False,
        'draw_label': True,
        'draw_score': True,
        'frame_num': -1
    }

def project_changed():
    st.session_state.det_config['exp_names'] = [
        exp for exp in os.listdir(os.path.join(PROJECT_DIR, st.session_state.project_name))
        if os.path.isdir(os.path.join(PROJECT_DIR, st.session_state.project_name, exp))
    ]
    st.session_state.det_config['project_name'] = st.session_state.project_name
    st.session_state.det_config['exp_name'] = st.session_state.det_config['exp_names'][0]



if check_password():
    st.title(":mag_right: 目标检测与分割")
    st.subheader(':gear: 参数配置')

    cfg_c11, cfg_c12, cfg_c13, cfg_c14 = st.columns(4)
    with cfg_c11:
        project_name = st.selectbox('项目:',
                                tuple(os.listdir(PROJECT_DIR)),
                                index=0,
                                on_change=project_changed,
                                key='project_name')
    with cfg_c12:
        exp_name = st.selectbox('试验:',
                                tuple(st.session_state.det_config['exp_names']),
                                index=0,
                                key='exp_name')
    with cfg_c13:
        score_thresh = st.slider('目标置信度:', 0.1, 1.0,
                                 st.session_state.det_config['score_thresh'],
                                 step=0.05)

    with cfg_c14:
        iou_thresh = st.slider('IOU阈值:', 0.3, 1.0,
                               st.session_state.det_config['iou_thresh'],
                               step=0.05)


    cfg_c21, cfg_c22, cfg_c23, cfg_c24 = st.columns(4)
    with cfg_c21:
        rgb_input = st.checkbox('RGB input', st.session_state.det_config['rgb_input'])
    with cfg_c22:
        draw_label = st.checkbox('绘制目标标签', st.session_state.det_config['draw_label'])
    with cfg_c23:
        draw_score = st.checkbox('绘制置信度', st.session_state.det_config['draw_score'])


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
                result_image, info = detect_image(project_name, exp_name,
                                                  score_thresh, iou_thresh,
                                                  draw_score, draw_label,
                                                  rgb_input, image)
                if result_image is not None:
                    image_col2.write("检测结果:")
                    image_col2.image(result_image)
                    detect_succeed = True
            else:
                st.error('请先上传一张图像，然后再点击 “运行单图像检测” 按钮')
    with image_c3:
        if detect_succeed:
            st.download_button("下载检测结果图像", convert_image(result_image),
                               f"det_reuslt_{image_upload.name}", "image/png")
    if detect_succeed:st.info(info)

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
                result_video_path, info = detect_video(project_name, exp_name,
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



    # st.subheader(':file_folder: 图像序列检测')
    # st.write("选择服务器上的一个图像文件夹，批量运行目标检测.")
    # with st.form("图像序列检测"):
    #     seq_c1, seq_c2, seq_c3 = st.columns(3)
    #     with seq_c1:
    #         image_dir = st.text_input('输入图像文件夹路径:')
    #     with seq_c2:
    #         image_det_num = st.number_input('运行检测的图像数:', max_value=5000, min_value=-1, value=-1)
    #     with seq_c3:
    #         save_to_txt = st.checkbox('保存txt', False)
    #     if st.form_submit_button("运行图像序列检测 :point_left:"):
    #         ret, info = detect_sequence(st.session_state.det_config, image_dir, det_num=image_det_num,
    #                                     save_to_txt=save_to_txt)
    #         if ret:st.info(info)



