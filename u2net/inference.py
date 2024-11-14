import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model import U2Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = preprocess(img).unsqueeze(0).to(device)
    return img

def run_inference(model, image_path, threshold=0.5):
    input_img = preprocess_image(image_path)
    with torch.no_grad():
        d1, *_ = model(input_img)
        pred = torch.sigmoid(d1)
        pred = pred[0, :, :].cpu().numpy()
    
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    if threshold is not None:
        pred = (pred > threshold).astype(np.uint8) * 255
    else:
        pred = (pred * 255).astype(np.uint8)
    return pred

def overlay_segmentation(original_image, binary_mask, alpha=0.5):
    original_image = Image.open(original_image).convert('RGB').resize((512, 512), Image.BILINEAR)
    original_image_np = np.array(original_image)
    overlay = np.zeros_like(original_image_np)
    overlay[:, :, 0] = binary_mask
    overlay_image = (1 - alpha) * original_image_np + alpha * overlay
    overlay_image = overlay_image.astype(np.uint8)
    return overlay_image


if __name__ == '__main__':
    # ---
    model_path = 'results/inter-u2net-duts.pt'
    image_path = 'images/ladies.jpg'
    # ---
    model = U2Net().to(device)
    model = nn.DataParallel(model)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    mask = run_inference(model, image_path, threshold=None)
    mask_with_threshold = run_inference(model, image_path, threshold=0.7)
    
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0, hspace=0)
    
    images = [
        Image.open(image_path).resize((512, 512)),
        mask,
        overlay_segmentation(image_path, mask_with_threshold),
        mask_with_threshold
    ]
    
    for i, img in enumerate(images):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(img, cmap='gray' if i % 2 != 0 else None)
        ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('inference-output.jpg', format='jpg', bbox_inches='tight', pad_inches=0)
