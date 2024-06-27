import os
import gradio as gr
from libs.api.detect import Detector


PRJ_ROOT_DIR = './Projects'
INFO_LINES = """
    ## This App is running on a huggingface free server.
    ## user guide:
    1. select a project from **Projects**;
    2. select a experiment from **Experiments**;
    3. set **Score thresh** and **IOU thresh**;
    4. upload an image;
    5. click **Predict** button;
"""

detector = Detector()

def run_detection(prjname, expname, score_thresh, iou_thresh, image):
    if prjname is None:
        gr.Warning("Project is not selected!", duration=3)
        return
    if expname is None:
        gr.Warning("Experiment is not selected!", duration=3)
        return
    if image is None:
        gr.Warning("Input image is empty!", duration=3)
        return
    detector.init(os.path.join(PRJ_ROOT_DIR, prjname), expname, score_thresh, iou_thresh)
    return detector.predict_image(image)


def get_experiments(prjname):
    expnames = [
        expname for expname in os.listdir(os.path.join(PRJ_ROOT_DIR, prjname))
        if expname.startswith(('YOLOX', 'YOLOV8'))
        ]
    return gr.Dropdown(expnames, label="Experiments")


with gr.Blocks() as demo:
    gr.Markdown(INFO_LINES)
    with gr.Row():
        input_image = gr.Image(label="Input image")
        output_image = gr.Image(label="Result image", interactive=False, format='png')
    with gr.Row():
        prjname = gr.Dropdown(os.listdir(PRJ_ROOT_DIR), label="Projects")
        expname = gr.Dropdown(interactive=True, label="Experiments")
        prjname.change(get_experiments, inputs=prjname, outputs=expname)
    with gr.Row():
        score_thresh = gr.Slider(0.1, 1.0, value=0.5, label="score thresh")
        iou_thresh = gr.Slider(0.1, 1.0, value=0.5, label="iou thresh")
    submit_btn = gr.Button('Predict')
    submit_btn.click(run_detection, [prjname, expname, score_thresh, iou_thresh, input_image], output_image)

if __name__ == '__main__':
    demo.launch()