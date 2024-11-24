from PIL import Image

import torch
import torchvision.transforms as transforms

def preprocess_img(img, img_size, normalize=False):
    if type(img) == str: img = Image.open(img)
    original_size = img.size
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    img = transform(img).unsqueeze(0)
    return img, original_size

def postprocess_img(img, original_size, normalize=False):
    img = img.detach().cpu().squeeze(0)
    
    # Denormalize the image
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
    img = torch.clamp(img, 0, 1)
    
    img = transforms.ToPILImage()(img)
    img = img.resize(original_size, Image.Resampling.LANCZOS)
    return img