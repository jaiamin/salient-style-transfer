from PIL import Image

import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import gradio as gr

device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'
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


@spaces.GPU
def transfer_style(content_image):
    style_img_filename = 'StarryNight.jpg'
    img_size = 512
    content_img, original_size = load_img(content_image, img_size)
    content_img = content_img.to(device)
    style_img = load_img_from_path(f'./style_images/{style_img_filename}', img_size)[0].to(device)

    iters = 100
    lr = 1e-1
    alpha = 1
    beta = 1

    generated_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([generated_img], lr=lr)

    for iter in range(iters+1):
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

        yield save_img(generated_img, original_size), str(round(iter/iters*100))+'%'
        
    yield save_img(generated_img, original_size), str(round(iter/iters*100))+'%'


interface = gr.Interface(
    fn=transfer_style, 
    inputs=[gr.Image(label='Content', type='pil')], 
    outputs=[
        gr.Image(label='Output', show_download_button=True),
        gr.Label(label='Progress')
    ],
    title="Starry Night Style Transfer",
    api_name='style',
    allow_flagging='never',
).launch(inbrowser=True)