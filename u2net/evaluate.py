from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import PASCALSDataset
from model import U2Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def load_model(model, model_path):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
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
    batch_size = 40

    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    model = U2Net().to(device)
    model = nn.DataParallel(model)
    load_model(model, 'results/inter-u2net-duts.pt')
    
    loader = DataLoader(PASCALSDataset(split='eval'), batch_size=batch_size, shuffle=False)

    loss = eval(model, loader, loss_fn)
    print(f'Loss: {loss:.4f}')