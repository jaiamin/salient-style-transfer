import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler

from data_loader import DUTSDataset, MSRADataset
from model import U2Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()

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


if __name__ == '__main__':
    batch_size = 40
    valid_batch_size = 80
    epochs = 100
    
    lr = 1e-4
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
    model_name = 'u2net-duts'
    model = U2Net()
    model = torch.nn.DataParallel(model.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(
        ConcatDataset([DUTSDataset(split='train'), MSRADataset(split='train')]), 
        batch_size=batch_size, shuffle=True, pin_memory=True, 
        num_workers=16, persistent_workers=True
    )
    valid_loader = DataLoader(
        ConcatDataset([DUTSDataset(split='valid'), MSRADataset(split='valid')]), 
        batch_size=valid_batch_size, shuffle=False, pin_memory=True, 
        num_workers=16, persistent_workers=True
    )
    
    losses = {'train': [], 'val': []}
    for epoch in tqdm(range(epochs), desc='Epochs'):
        torch.cuda.empty_cache()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = validate(model, valid_loader, loss_fn)
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'results/inter-{model_name}.pt')
            
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), f'results/{model_name}.pt')
    with open('results/loss.txt', 'wb') as f:
        pickle.dump(losses, f)