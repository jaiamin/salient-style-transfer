import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from safetensors.torch import save_file

from data_loader import DUTSDataset, MSRADataset
from model import U2Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.
    for images, masks in tqdm(loader, desc='Training', leave=False):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = sum([criterion(output, masks) for output in outputs])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validating', leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(images)
            loss = sum([criterion(output, masks) for output in outputs])
            running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    return avg_loss

def save(model, model_name, losses):
    save_file(model.state_dict(), f'results/{model_name}.safetensors')
    with open('results/loss.txt', 'wb') as f:
        pickle.dump(losses, f)


if __name__ == '__main__':
    batch_size = 40
    valid_batch_size = 80
    epochs = 200
    
    lr = 1e-3
    loss_fn_bce = nn.BCEWithLogitsLoss(reduction='mean')
    loss_fn_dice = DiceLoss()
    alpha = 0.6
    loss_fn = lambda o, m: alpha * loss_fn_bce(o, m) + (1 - alpha) * loss_fn_dice(o, m)
    
    model_name = 'u2net-duts-msra'
    model = U2Net()
    model = torch.nn.parallel.DataParallel(model.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(
        ConcatDataset([DUTSDataset(split='train'), MSRADataset(split='train')]), 
        batch_size=batch_size, shuffle=True, pin_memory=True, 
        num_workers=8, persistent_workers=True
    )
    valid_loader = DataLoader(
        ConcatDataset([DUTSDataset(split='valid'), MSRADataset(split='valid')]), 
        batch_size=valid_batch_size, shuffle=False, pin_memory=True, 
        num_workers=8, persistent_workers=True
    )
    
    best_val_loss = float('inf')
    losses = {'train': [], 'val': []}
    
    # training loop
    try:
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_loss = validate(model, valid_loader, loss_fn)
            losses['train'].append(train_loss)
            losses['val'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_file(model.state_dict(), f'results/best-{model_name}.safetensors')
                
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})')
    finally:
        save(model, model_name, losses)