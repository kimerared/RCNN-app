# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1RJuq-k6UqiiawLttbKAD6tTkWAXG03nJ

## Requirements
Gradio, ultralytics
"""

"""## Libraries"""

import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Function to Load model and Make inference with YOLOv8
def yolov8_inference(img_path, model_id, conf_threshold, iou_threshold):
   # Load the model
    model = YOLO(model_id, conf_threshold, iou_threshold)

    # Perform inference
    results = model(img_path)
    for r in results:
      im_array = r.plot()  # plot a BGR numpy array of predictions
      im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
      im.show()
    return im


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(type="filepath", label="Photo")
                model_name = gr.Dropdown(
                    label="Model",
                    value='model/tank_weights.onnx',
                    choices=[
                        "model/tank_weights.onnx",
                        "model/yolov8m.pt",
                        "yolov8n-seg.pt",
                        "yolov8m-seg.pt"
                    ]
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.3
                )
                iou_threshold = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.3
                )
                yolo_infer = gr.Button(value="Inference")

            with gr.Column():
                output_numpy = gr.Image(type="numpy",label="Output")

        yolo_infer.click(
            fn=yolov8_inference,
            inputs=[
                img_path,
                model_name,
                conf_threshold,
                iou_threshold
            ],
            outputs=[output_numpy],
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv8: Object Detection & Segmentation
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
          For more information, please refer to
          <a href='https://kimerared.github.io/R-CNN/' target='_blank'>R-CNN GitHub</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=False, show_error=True, max_threads=1)
