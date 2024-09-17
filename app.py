import time
from PIL import Image
from tqdm import tqdm

import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import gradio as gr

if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'
print('DEVICE:', device)

class VGG_19(nn.Module):
    def __init__(self):
        super(VGG_19, self).__init__()
        self.model = models.vgg19(pretrained=True).features[:30]
        
        for i, _ in enumerate(self.model):
            if i in [4, 9, 18, 27]:
                self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
                
    def forward(self, x):
        features = []
        
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in [0, 5, 10, 19, 28]:
                features.append(x)
        return features
    
model = VGG_19().to(device)
for param in model.parameters():
    param.requires_grad = False

def load_img(img: Image, img_size):
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    return img, original_size

def load_img_from_path(path_to_image, img_size):
    img = Image.open(path_to_image)
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    return img, original_size

def save_img(img, original_size):
    img = img.cpu().clone()
    img = img.squeeze(0)
    
    # address tensor value scaling and quantization
    img = torch.clamp(img, 0, 1)
    img = img.mul(255).byte()
    
    unloader = transforms.ToPILImage()
    img = unloader(img)
    
    img = img.resize(original_size, Image.Resampling.LANCZOS)
    
    return img


style_options = {
    # famous paintings
    'Starry Night': 'StarryNight.jpg',
    'Great Wave': 'GreatWave.jpg',
    'Scream': 'Scream.jpg',
    # styles
    'Lego Bricks': 'LegoBricks.jpg',
    'Oil Painting': 'OilPainting.jpg',
    'Mosaic': 'Mosaic.jpg',
    '8Bit': '8Bit.jpg',
}
style_options = {k: f'./style_images/{v}' for k, v in style_options.items()}

@spaces.GPU(duration=25)
def inference(content_image, style_image, style_strength, progress=gr.Progress(track_tqdm=True)):
    yield None
    print('-'*15)
    print('STYLE:', style_image)
    img_size = 512
    content_img, original_size = load_img(content_image, img_size)
    content_img = content_img.to(device)
    style_img = load_img_from_path(style_options[style_image], img_size)[0].to(device)
    
    print('CONTENT IMG SIZE:', original_size)

    iters = style_strength
    lr = 1e-1
    alpha = 1
    beta = 1

    st = time.time()
    generated_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([generated_img], lr=lr)
    
    for _ in tqdm(range(iters), desc='The magic is happening ✨'):
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
    
with gr.Blocks(title='🖼️ Neural Style Transfer') as demo:
    gr.HTML("<h1 style='text-align: center'>🖼️ Neural Style Transfer</h1>")
    with gr.Row():
        with gr.Column():
            content_image = gr.Image(label='Content', type='pil', sources=['upload'])
            style_dropdown = gr.Radio(choices=list(style_options.keys()), label='Style', value='Starry Night', type='value')
            with gr.Accordion('Adjustments', open=False):
                with gr.Group():
                    style_strength_slider = gr.Slider(label='Style Strength', minimum=0, maximum=100, step=5, value=50)
                    with gr.Row():
                        low_button = gr.Button('Low').click(fn=lambda: set_slider(10), outputs=[style_strength_slider])
                        medium_button = gr.Button('Medium').click(fn=lambda: set_slider(50), outputs=[style_strength_slider])
                        high_button = gr.Button('High').click(fn=lambda: set_slider(100), outputs=[style_strength_slider])
            submit_button = gr.Button('Submit')
        with gr.Column():
            output_image = gr.Image(label='Output', show_download_button=True, interactive=False)
    
    submit_button.click(fn=inference, inputs=[content_image, style_dropdown, style_strength_slider], outputs=[output_image])
    
    examples = gr.Examples(
        examples=[
            ['./content_images/TajMahal.jpg', 'Starry Night', 75],
            ['./content_images/GoldenRetriever.jpg', 'Lego Bricks', 50],
            ['./content_images/SeaTurtle.jpg', 'Mosaic', 100]
        ],
        inputs=[content_image, style_dropdown, style_strength_slider]
    )
    
demo.launch(show_api=True)