import os
import time
from datetime import datetime, timezone, timedelta

import spaces
import torch
import torchvision.models as models
import numpy as np
import gradio as gr
from gradio_imageslider import ImageSlider

from utils import preprocess_img, preprocess_img_from_path, postprocess_img
from vgg19 import VGG_19
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

@spaces.GPU(duration=10)
def run(content_image, style_name, style_strength=5, apply_to_background=False, progress=gr.Progress(track_tqdm=True)):
    yield None
    content_img, original_size = preprocess_img(content_image, img_size)
    content_img = content_img.to(device)
    
    print('-'*15)
    print('DATETIME:', datetime.now(timezone.utc) - timedelta(hours=4)) # est
    print('STYLE:', style_name)
    print('CONTENT IMG SIZE:', original_size)
    print('STYLE STRENGTH:', style_strength, f'(lr={lrs[style_strength-1]})')

    style_features = cached_style_features[style_name]
    
    st = time.time()
    generated_img = inference(
        model=model,
        segmentation_model=segmentation_model,
        content_image=content_img,
        style_features=style_features,
        lr=lrs[style_strength-1],
        apply_to_background=apply_to_background
    )
    et = time.time()
    print('TIME TAKEN:', et-st)
    
    yield (content_image, postprocess_img(generated_img, original_size))

def set_slider(value):
    return gr.update(value=value)

css = """
#container {
    margin: 0 auto;
    max-width: 1100px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center; padding: 10px'>🖼️ Neural Style Transfer</h1>")
    with gr.Row(elem_id='container'):
        with gr.Column():
            content_image = gr.Image(label='Content', type='pil', sources=['upload', 'webcam', 'clipboard'], format='jpg', show_download_button=False)
            style_dropdown = gr.Radio(choices=list(style_options.keys()), label='Style', value='Starry Night', type='value')
            with gr.Group():
                style_strength_slider = gr.Slider(label='Style Strength', minimum=1, maximum=10, step=1, value=5, info='Higher values add artistic flair, lower values add a realistic feel.')
                apply_to_background = gr.Checkbox(label='Apply to background only')
            submit_button = gr.Button('Submit', variant='primary')
            
            examples = gr.Examples(
                examples=[
                    ['./content_images/Bridge.jpg', 'Starry Night'],
                    ['./content_images/GoldenRetriever.jpg', 'Great Wave'],
                    ['./content_images/CameraGirl.jpg', 'Bokeh']
                ],
                inputs=[content_image, style_dropdown]
            )

        with gr.Column():
            output_image = ImageSlider(position=0.15, label='Output', show_label=True, type='pil', interactive=False, show_download_button=False)
            download_button = gr.DownloadButton(label='Download Image', visible=False)

    def save_image(img_tuple):
        filename = 'generated.jpg'
        img_tuple[1].save(filename)
        return filename
    
    submit_button.click(
        fn=lambda: gr.update(visible=False),
        outputs=[download_button]
    )
        
    submit_button.click(
        fn=run, 
        inputs=[content_image, style_dropdown, style_strength_slider, apply_to_background], 
        outputs=[output_image]
    ).then(
        fn=save_image,
        inputs=[output_image],
        outputs=[download_button]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[download_button]
    )

demo.queue = False
demo.config['queue'] = False
demo.launch(show_api=False)