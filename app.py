import os
import time
import datetime
from tqdm import tqdm

import spaces
import torch
import torch.optim as optim
import gradio as gr

from utils import load_img, load_img_from_path, save_img
from vgg19 import VGG_19

if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'
print('DEVICE:', device)

model = VGG_19().to(device)
for param in model.parameters():
    param.requires_grad = False


style_files = os.listdir('./style_images')
style_options = {' '.join(style_file.split('.')[0].split('_')): f'./style_images/{style_file}' for style_file in style_files}

@spaces.GPU(duration=35)
def inference(content_image, style_image, style_strength, output_quality, progress=gr.Progress(track_tqdm=True)):
    yield None
    print('-'*15)
    print('DATETIME:', datetime.datetime.now())
    print('STYLE:', style_image)
    img_size = 1024 if output_quality else 512
    content_img, original_size = load_img(content_image, img_size)
    content_img = content_img.to(device)
    style_img = load_img_from_path(style_options[style_image], img_size)[0].to(device)
    
    print('CONTENT IMG SIZE:', original_size)
    print('STYLE STRENGTH:', style_strength)
    print('HIGH QUALITY:', output_quality)

    iters = 50
    # learning rate determined by input
    lr = 0.001 + (0.099 / 99) * (style_strength - 1)
    alpha = 1
    beta = 1

    st = time.time()
    generated_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([generated_img], lr=lr)
    
    for iter in tqdm(range(iters), desc='The magic is happening ✨'):
        generated_features = model(generated_img)
        content_features = model(content_img)
        style_features = model(style_img)
        
        content_loss = 0
        style_loss = 0
        
        for generated_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
            batch_size, n_feature_maps, height, width = generated_feature.size()
            
            content_loss += (torch.mean((generated_feature - content_feature) ** 2))
            
            G = torch.mm((generated_feature.view(batch_size * n_feature_maps, height * width)), (generated_feature.view(batch_size * n_feature_maps, height * width)).t())
            A = torch.mm((style_feature.view(batch_size * n_feature_maps, height * width)), (style_feature.view(batch_size * n_feature_maps, height * width)).t())
            
            E_l = ((G - A) ** 2)
            w_l = 1/5
            style_loss += torch.mean(w_l * E_l)
            
        total_loss = alpha * content_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    et = time.time()
    print('TIME TAKEN:', et-st)
    yield save_img(generated_img, original_size)


def set_slider(value):
    return gr.update(value=value)

css = """
#container {
    margin: 0 auto;
    max-width: 550px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center; padding: 10px'>🖼️ Neural Style Transfer</h1>")
    with gr.Column(elem_id='container'):
        content_and_output = gr.Image(show_label=False, type='pil', sources=['upload'], format='jpg')
        style_dropdown = gr.Radio(choices=list(style_options.keys()), label='Style', value='Starry Night', type='value')
        with gr.Accordion('Adjustments', open=False):
            with gr.Group():
                style_strength_slider = gr.Slider(label='Style Strength', minimum=1, maximum=100, step=1, value=50)
                with gr.Row():
                    low_button = gr.Button('Low').click(fn=lambda: set_slider(10), outputs=[style_strength_slider])
                    medium_button = gr.Button('Medium').click(fn=lambda: set_slider(50), outputs=[style_strength_slider])
                    high_button = gr.Button('High').click(fn=lambda: set_slider(100), outputs=[style_strength_slider])
            with gr.Group():
                output_quality = gr.Checkbox(label='More Realistic', info='Note: This takes longer, but improves output image quality')
        submit_button = gr.Button('Submit')
    
        submit_button.click(fn=inference, inputs=[content_and_output, style_dropdown, style_strength_slider, output_quality], outputs=[content_and_output])
        
        examples = gr.Examples(
            examples=[
                ['./content_images/TajMahal.jpg', 'Starry Night', 75, True],
                ['./content_images/GoldenRetriever.jpg', 'Lego Bricks', 50, True],
                ['./content_images/SeaTurtle.jpg', 'Mosaic', 100, True]
            ],
            inputs=[content_and_output, style_dropdown, style_strength_slider, output_quality]
        )

# disable queue
demo.queue = False
demo.config['queue'] = False
demo.launch(show_api=True, allowed_paths=['/tmp/gradio/'])