from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from data_loader import PASCALSDataset
from model import U2Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def load_model(model, model_path):
    state_dict = load_file(model_path, device=device.type)
    model.load_state_dict(state_dict)
    model.eval()

def eval(model, loader, criterion):
    model.eval()
    running_loss = 0.
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = sum([criterion(output, masks) for output in outputs])
            running_loss += loss.item()
    return running_loss / len(loader)


if __name__ == '__main__':
    batch_size = 1

    model_type = input('Model type [b,f]: ')
    model_name = 'best-u2net-duts-msra.safetensors' if model_type == 'b' else 'u2net-duts-msra.safetensors'
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    model = U2Net().to(device)
    model = nn.DataParallel(model)
    load_model(model, f'results/{model_name}')
    
    loader = DataLoader(PASCALSDataset(split='eval'), batch_size=batch_size, shuffle=False)

    loss = eval(model, loader, loss_fn)
    print(f'Loss: {loss:.4f}')