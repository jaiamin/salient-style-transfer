from PIL import Image

import torch
import torchvision.transforms as transforms

def preprocess_img_from_path(path_to_image, img_size):
    img = Image.open(path_to_image)
    return preprocess_img(img, img_size)

def preprocess_img(img: Image, img_size):
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    return img, original_size

def postprocess_img(img, original_size):
    img = img.detach().cpu().squeeze(0)
    
    # address tensor value scaling and quantization
    img = torch.clamp(img, 0, 1)
    img = img.mul(255).byte()
    
    unloader = transforms.ToPILImage()
    img = unloader(img)
    
    img = img.resize(original_size, Image.Resampling.LANCZOS)
    
    return img