import os
import time
from datetime import datetime, timezone, timedelta

import spaces
import torch
import gradio as gr

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

style_files = os.listdir('./style_images')
style_options = {' '.join(style_file.split('.')[0].split('_')): f'./style_images/{style_file}' for style_file in style_files}
optimal_settings = {
    'Starry Night': (100, False),
    'Lego Bricks': (100, False),
    'Mosaic': (100, False),
    'Oil Painting': (100, False),
    'Scream': (75, True),
    'Great Wave': (75, False),
    'Watercolor': (75, False),
}

cached_style_features = {}
for style_name, style_img_path in style_options.items():
    style_img_512 = preprocess_img_from_path(style_img_path, 512)[0].to(device)
    style_img_1024 = preprocess_img_from_path(style_img_path, 1024)[0].to(device)
    with torch.no_grad():
        style_features = (model(style_img_512), model(style_img_1024))
    cached_style_features[style_name] = style_features 

@spaces.GPU(duration=15)
def run(content_image, style_name, style_strength, output_quality, progress=gr.Progress(track_tqdm=True)):
    yield None
    img_size = 1024 if output_quality else 512
    content_img, original_size = preprocess_img(content_image, img_size)
    content_img = content_img.to(device)
    
    print('-'*15)
    print('DATETIME:', datetime.now(timezone.utc) - timedelta(hours=4)) # est
    print('STYLE:', style_name)
    print('CONTENT IMG SIZE:', original_size)
    print('STYLE STRENGTH:', style_strength)
    print('HIGH QUALITY:', output_quality)

    style_features = cached_style_features[style_name][0 if img_size == 512 else 1]
    converted_lr = 0.001 + (0.009 / 99) * (style_strength - 1) # [0.001, 0.01]
    
    st = time.time()
    generated_img = inference(
        model=model,
        content_image=content_img,
        style_features=style_features,
        lr=converted_lr
    )
    et = time.time()
    print('TIME TAKEN:', et-st)
    
    yield postprocess_img(generated_img, original_size)


def set_slider(value):
    return gr.update(value=value)

def update_settings(style):
    return optimal_settings.get(style, (100, False))

css = """
#container {
    margin: 0 auto;
    max-width: 550px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center; padding: 10px'>🖼️ Neural Style Transfer</h1>")
    with gr.Column(elem_id='container'):
        content_and_output = gr.Image(label='Content', show_label=False, type='pil', sources=['upload', 'webcam', 'clipboard'], format='jpg', show_download_button=False)
        style_dropdown = gr.Radio(choices=list(style_options.keys()), label='Style', info='Note: Adjustments automatically optimize for different styles.', value='Starry Night', type='value')
        
        with gr.Accordion('Adjustments', open=True):
            with gr.Group():
                style_strength_slider = gr.Slider(label='Style Strength', minimum=1, maximum=100, step=1, value=50)
                
                with gr.Row():
                    low_button = gr.Button('Low', size='sm').click(fn=lambda: set_slider(10), outputs=[style_strength_slider])
                    medium_button = gr.Button('Medium', size='sm').click(fn=lambda: set_slider(50), outputs=[style_strength_slider])
                    high_button = gr.Button('High', size='sm').click(fn=lambda: set_slider(100), outputs=[style_strength_slider])
            with gr.Group():
                output_quality = gr.Checkbox(label='More Realistic', info='Note: If unchecked, the resulting image will have a more artistic flair.')
        
        submit_button = gr.Button('Submit', variant='primary')
        download_button = gr.DownloadButton(label='Download Image', visible=False)

        def save_image(img):
            filename = 'generated.jpg'
            img.save(filename)
            return filename
        
        submit_button.click(
            fn=run, 
            inputs=[content_and_output, style_dropdown, style_strength_slider, output_quality], 
            outputs=[content_and_output]
        ).then(
            fn=save_image,
            inputs=[content_and_output],
            outputs=[download_button]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[download_button]
        )
        
        content_and_output.change(
            fn=lambda _: gr.update(visible=False),
            inputs=[content_and_output],
            outputs=[download_button]
        )
        
        style_dropdown.change(
            fn=lambda style: set_slider(update_settings(style)[0]), 
            inputs=[style_dropdown], 
            outputs=[style_strength_slider]
        )
        style_dropdown.change(
            fn=lambda style: gr.update(value=update_settings(style)[1]), 
            inputs=[style_dropdown], 
            outputs=[output_quality]
        )
        
        examples = gr.Examples(
            label='Example',
            examples=[['./content_images/Bridge.jpg', 'Starry Night', 100, False]],
            inputs=[content_and_output, style_dropdown, style_strength_slider, output_quality]
        )

demo.queue = False
demo.config['queue'] = False
demo.launch(show_api=False)