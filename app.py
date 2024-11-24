import os
import time
from datetime import datetime, timezone, timedelta

import spaces
import torch
import torch.optim as optim
import numpy as np
import gradio as gr
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from utils import preprocess_img, postprocess_img
from vgg.vgg19 import VGG_19
from u2net.model import U2Net
from inference import inference

if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'
print('Device:', device)
if device == 'cuda': print('Name:', torch.cuda.get_device_name())

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
style_options = {
    'Starry Night': './style_images/Starry_Night.jpg',
    'Starry Night (v2)': './style_images/Starry_Night_v2.jpg',
    'Scream': './style_images/Scream.jpg',
    'Great Wave': './style_images/Great_Wave.jpg',
    'Oil Painting': './style_images/Oil_Painting.jpg',
    'Watercolor': './style_images/Watercolor.jpg',
    'Mosaic': './style_images/Mosaic.jpg',
    'Lego Bricks': './style_images/Lego_Bricks.jpg',
    'Bokeh': './style_images/Bokeh.jpg',
}
lrs = np.linspace(0.015, 0.075, 3).tolist()
img_size = 512

cached_style_features = {}
for style_name, style_img_path in style_options.items():
    style_img = preprocess_img(style_img_path, img_size)[0].to(device)
    with torch.no_grad():
        style_features = model(style_img)
    cached_style_features[style_name] = style_features 

@spaces.GPU(duration=30)
def run(content_image, style_name, style_strength=len(lrs), optim_name='AdamW', apply_to_background=False):
    yield None
    content_img, original_size = preprocess_img(content_image, img_size)
    content_img_normalized, _ = preprocess_img(content_image, img_size, normalize=True)
    content_img, content_img_normalized = content_img.to(device), content_img_normalized.to(device)
    style_features = cached_style_features[style_name]
    
    if optim_name == 'AdamW': 
        optim_caller = optim.AdamW
    elif optim_name == 'L-BFGS': 
        optim_caller = optim.LBFGS
    
    print('-'*30)
    print(datetime.now(timezone.utc) - timedelta(hours=5)) # EST
    
    st = time.time()
    generated_img = inference(
        model=model,
        sod_model=sod_model,
        content_image=content_img,
        content_image_norm=content_img_normalized,
        style_features=style_features,
        lr=lrs[style_strength-1],
        apply_to_background=apply_to_background,
        optim_caller=optim_caller,
    )
    et = time.time()
    print(f'{et-st:.2f}s')
    
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
            with gr.Group():
                content_image = gr.Image(label='Content', type='pil', sources=['upload', 'webcam', 'clipboard'], format='jpg', show_download_button=False)
            with gr.Group():
                style_dropdown = gr.Radio(choices=list(style_options.keys()), label='Style', value='Starry Night', type='value')
                style_strength_slider = gr.Slider(label='Style Strength', minimum=1, maximum=len(lrs), step=1, value=len(lrs))
                apply_to_background_checkbox = gr.Checkbox(label='Apply style transfer exclusively to the background', value=False)
            with gr.Accordion(label='Advanced Options', open=False):
                optim_dropdown = gr.Radio(choices=['AdamW', 'L-BFGS'], label='Optimizer', value='AdamW', type='value')
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
