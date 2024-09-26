import os
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

import spaces
import torch
import torchvision.models as models
import numpy as np
import gradio as gr
from gradio_imageslider import ImageSlider

from utils import preprocess_img, preprocess_img_from_path, postprocess_img
from vgg.vgg19 import VGG_19
from inference import inference

if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'
print('DEVICE:', device)
if device == 'cuda': print('CUDA DEVICE:', torch.cuda.get_device_name())

model = VGG_19().to(device).eval()
for param in model.parameters():
    param.requires_grad = False
segmentation_model = models.segmentation.deeplabv3_resnet101(
    weights='DEFAULT'
).to(device).eval()

style_files = os.listdir('./style_images')
style_options = {' '.join(style_file.split('.')[0].split('_')): f'./style_images/{style_file}' for style_file in style_files}
lrs = np.logspace(np.log10(0.001), np.log10(0.1), 10).tolist()
img_size = 512

cached_style_features = {}
for style_name, style_img_path in style_options.items():
    style_img = preprocess_img_from_path(style_img_path, img_size)[0].to(device)
    with torch.no_grad():
        style_features = model(style_img)
    cached_style_features[style_name] = style_features 

@spaces.GPU(duration=12)
def run(content_image, style_name, style_strength=5):
    yield [None] * 3
    content_img, original_size = preprocess_img(content_image, img_size)
    content_img = content_img.to(device)
    
    print('-'*15)
    print('DATETIME:', datetime.now(timezone.utc) - timedelta(hours=4)) # est
    print('STYLE:', style_name)
    print('CONTENT IMG SIZE:', original_size)
    print('STYLE STRENGTH:', style_strength, f'(lr={lrs[style_strength-1]:.3f})')

    style_features = cached_style_features[style_name]
    
    st = time.time()
    
    if device == 'cuda':
        stream_all = torch.cuda.Stream()
        stream_bg = torch.cuda.Stream()

    def run_inference_cuda(apply_to_background, stream):
        with torch.cuda.stream(stream):
            return run_inference(apply_to_background)
        
    def run_inference(apply_to_background):
        return inference(
            model=model,
            segmentation_model=segmentation_model,
            content_image=content_img,
            style_features=style_features,
            lr=lrs[style_strength-1],
            apply_to_background=apply_to_background
        )

    with ThreadPoolExecutor() as executor:
        if device == 'cuda':
            future_all = executor.submit(run_inference_cuda, False, stream_all)
            future_bg = executor.submit(run_inference_cuda, True, stream_bg)
        else:
            future_all = executor.submit(run_inference, False)
            future_bg = executor.submit(run_inference, True)
        generated_img_all, _ = future_all.result()
        generated_img_bg, bg_ratio = future_bg.result()

    et = time.time()
    print('TIME TAKEN:', et-st)
    
    yield (
        (content_image, postprocess_img(generated_img_all, original_size)),
        (content_image, postprocess_img(generated_img_bg, original_size)),
        f'{bg_ratio:.2f}'
    )

def set_slider(value):
    return gr.update(value=value)

css = """
#container {
    margin: 0 auto;
    max-width: 1200px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center; padding: 10px'>🖼️ Neural Style Transfer w/ Salient Object Detection")
    with gr.Row(elem_id='container'):
        with gr.Column():
            content_image = gr.Image(label='Content', type='pil', sources=['upload', 'webcam', 'clipboard'], format='jpg', show_download_button=False)
            style_dropdown = gr.Radio(choices=list(style_options.keys()), label='Style', value='Starry Night', type='value')
            with gr.Group():
                style_strength_slider = gr.Slider(label='Style Strength', minimum=1, maximum=10, step=1, value=5, info='Higher values add artistic flair, lower values add a realistic feel.')
            submit_button = gr.Button('Submit', variant='primary')
            
            examples = gr.Examples(
                examples=[
                    ['./content_images/Bridge.jpg', 'Starry Night', 6],
                    ['./content_images/GoldenRetriever.jpg', 'Great Wave', 5],
                    ['./content_images/CameraGirl.jpg', 'Bokeh', 10]
                ],
                inputs=[content_image, style_dropdown, style_strength_slider]
            )

        with gr.Column():
            output_image_all = ImageSlider(position=0.15, label='Styled Image', type='pil', interactive=False, show_download_button=False)
            download_button_1 = gr.DownloadButton(label='Download Styled Image', visible=False)
            with gr.Group():
                output_image_background = ImageSlider(position=0.15, label='Styled Background', type='pil', interactive=False, show_download_button=False)
                bg_ratio_label = gr.Label(label='Background Ratio')
            download_button_2 = gr.DownloadButton(label='Download Styled Background', visible=False)

    def save_image(img_tuple1, img_tuple2):
        filename1, filename2 = 'generated-all.jpg', 'generated-bg.jpg'
        img_tuple1[1].save(filename1)
        img_tuple2[1].save(filename2)
        return filename1, filename2
    
    submit_button.click(
        fn=lambda: [gr.update(visible=False) for _ in range(2)],
        outputs=[download_button_1, download_button_2]
    )
        
    submit_button.click(
        fn=run, 
        inputs=[content_image, style_dropdown, style_strength_slider], 
        outputs=[output_image_all, output_image_background, bg_ratio_label]
    ).then(
        fn=save_image,
        inputs=[output_image_all, output_image_background],
        outputs=[download_button_1, download_button_2]
    ).then(
        fn=lambda: [gr.update(visible=True) for _ in range(2)],
        outputs=[download_button_1, download_button_2]
    )

demo.queue = False
demo.config['queue'] = False
demo.launch(show_api=False)