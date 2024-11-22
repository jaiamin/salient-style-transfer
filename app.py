import os
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

import spaces
import torch
import torch.optim as optim
import torchvision.models as models
import numpy as np
import gradio as gr
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from utils import preprocess_img, preprocess_img_from_path, postprocess_img
from vgg.vgg19 import VGG_19
from u2net.model import U2Net
from inference import inference

if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'
print('DEVICE:', device)
if device == 'cuda': print('CUDA DEVICE:', torch.cuda.get_device_name())

def load_model_without_module(model, model_path):
    state_dict = load_file(model_path, device=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
   
# load models 
model = VGG_19().to(device).eval()
for param in model.parameters():
    param.requires_grad = False
sod_model = U2Net().to(device).eval()
local_model_path = hf_hub_download(repo_id='jamino30/u2net-saliency', filename='u2net-duts-msra.safetensors')
load_model_without_module(sod_model, local_model_path)

style_files = os.listdir('./style_images')
style_options = {' '.join(style_file.split('.')[0].split('_')): f'./style_images/{style_file}' for style_file in style_files}
lrs = np.logspace(np.log10(0.001), np.log10(0.1), 10).tolist()
img_size = 512

# store style(s) features
cached_style_features = {}
for style_name, style_img_path in style_options.items():
    style_img = preprocess_img_from_path(style_img_path, img_size)[0].to(device)
    with torch.no_grad():
        style_features = model(style_img)
    cached_style_features[style_name] = style_features 

@spaces.GPU(duration=30)
def run(content_image, style_name, style_strength=10, optim_name='AdamW', apply_to_background=False):
    yield None
    content_img, original_size = preprocess_img(content_image, img_size)
    content_img_normalized, _ = preprocess_img(content_image, img_size, normalize=True)
    content_img, content_img_normalized = content_img.to(device), content_img_normalized.to(device)
    style_features = cached_style_features[style_name]
    
    if optim_name == 'Adam': 
        optim_caller = torch.optim.Adam
        iterations = 101
    elif optim_name == 'AdamW': 
        optim_caller = torch.optim.AdamW
        iterations = 101
    elif optim_name == 'L-BFGS': 
        optim_caller = torch.optim.LBFGS
        iterations = 5
    
    print('-'*15)
    print('DATETIME:', datetime.now(timezone.utc) - timedelta(hours=4)) # est
    print('STYLE:', style_name)
    print('CONTENT IMG SIZE:', original_size)
    print('STYLE STRENGTH:', style_strength, f'(lr={lrs[style_strength-1]:.3f})')
    
    st = time.time()
    generated_img = inference(
        model=model,
        sod_model=sod_model,
        content_image=content_img,
        content_image_norm=content_img_normalized,
        style_features=style_features,
        lr=lrs[style_strength-1],
        apply_to_background=apply_to_background,
        iterations=iterations,
        optim_caller=optim_caller,
    )
    et = time.time()
    print('TIME TAKEN:', et-st)
    
    yield postprocess_img(generated_img, original_size)

def set_slider(value):
    return gr.update(value=value)

css = """
#container {
    margin: 0 auto;
    max-width: 1200px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center; padding: 10px'>🖼️ Neural Style Transfer w/ Salient Region Preservation")
    with gr.Row(elem_id='container'):
        with gr.Column():
            content_image = gr.Image(label='Content', type='pil', sources=['upload', 'webcam', 'clipboard'], format='jpg', show_download_button=False)
            style_dropdown = gr.Radio(choices=list(style_options.keys()), label='Style', value='Starry Night', type='value')
            with gr.Group():
                style_strength_slider = gr.Slider(label='Style Strength', minimum=1, maximum=10, step=1, value=10, info='Higher values add artistic flair, lower values add a realistic feel.')
            with gr.Group():
                apply_to_background_checkbox = gr.Checkbox(label='Apply styling to background only', value=False)
            with gr.Accordion(label='Advanced Options', open=False):
                optim_dropdown = gr.Radio(choices=['Adam', 'AdamW', 'L-BFGS'], label='Optimizer', value='AdamW', type='value')
            submit_button = gr.Button('Submit', variant='primary')
            
            examples = gr.Examples(
                examples=[
                    ['./content_images/Surfer.jpg', 'Starry Night'],
                    ['./content_images/GoldenRetriever.jpg', 'Great Wave'],
                    ['./content_images/CameraGirl.jpg', 'Bokeh']
                ],
                inputs=[content_image, style_dropdown]
            )

        with gr.Column():
            output_image = gr.Image(label='Output', type='pil', interactive=False, show_download_button=False)
            download_button = gr.DownloadButton(label='Download Image', visible=False)

    def save_image(img):
        filename = 'generated.jpg'
        img.save(filename)
        return filename
    
    submit_button.click(
        fn=lambda: gr.update(visible=False),
        outputs=download_button
    )
        
    submit_button.click(
        fn=run, 
        inputs=[content_image, style_dropdown, style_strength_slider, optim_dropdown, apply_to_background_checkbox], 
        outputs=output_image
    ).then(
        fn=save_image,
        inputs=output_image,
        outputs=download_button
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=download_button
    )

demo.queue = False
demo.config['queue'] = False
demo.launch(show_api=False)
