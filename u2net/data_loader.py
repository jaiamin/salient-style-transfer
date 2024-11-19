import os
import random
from PIL import Image

import torch
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
    

class SaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, split, img_size=512, val_split_ratio=0.05, subset_ratio=None):        
        self.img_size = img_size
        self.split = split
        self.image_dir, self.mask_dir = self.set_directories(split)
        
        all_images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        if split in ['train', 'valid']:
            train_imgs, val_imgs = train_test_split(all_images, test_size=val_split_ratio, random_state=42)
            self.images = train_imgs if split == 'train' else val_imgs
        else:
            self.images = all_images
            
        if subset_ratio: # subsampling
            total_samples = len(self.images)
            indices = np.random.choice(total_samples, int(total_samples * subset_ratio), replace=False)
            self.images = [self.images[i] for i in indices]
        
        self.img_resize = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_filename = img_filename.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if mask.mode != 'L': mask = mask.convert('L')
        mask = mask.point(lambda p: 255 if p > 128 else 0)
        
        img, mask = self.img_resize(img), self.mask_resize(mask)
        if self.split == 'train':
            img, mask = self.apply_augmentations(img, mask)
        
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        mask = transforms.ToTensor()(mask).squeeze(0)
        
        return img, mask
    
    def apply_augmentations(self, img, mask):
        if random.random() > 0.5: # horizontal flip
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
        
        if random.random() > 0.5: # random resized crop
            resized_crop = transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0))
            i, j, h, w = resized_crop.get_params(img, scale=(0.8, 1.0), ratio=(3/4, 4/3))
            img = transforms.functional.resized_crop(
                img, i, j, h, w, (self.img_size, self.img_size), 
                interpolation=transforms.InterpolationMode.BILINEAR
            )
            mask = transforms.functional.resized_crop(
                mask, i, j, h, w, (self.img_size, self.img_size), 
                interpolation=transforms.InterpolationMode.NEAREST
            )
        
        if random.random() > 0.5: # color jitter
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            img = color_jitter(img)
        
        return img, mask


class DUTSDataset(SaliencyDataset):
    def set_directories(self, split):
        train_or_test = 'train' if split in ['train', 'valid'] else 'test'
        image_dir = f'/data/duts_{train_or_test}_data/images'
        mask_dir = f'/data/duts_{train_or_test}_data/masks'
        return image_dir, mask_dir
    
    
class MSRADataset(SaliencyDataset):
    def set_directories(self, split):
        image_dir = '/data/msra_data/images'
        mask_dir = '/data/msra_data/masks'
        return image_dir, mask_dir
    

class PASCALSDataset(SaliencyDataset):
    def set_directories(self, split):
        image_dir = '/data/pascals_data/images'
        mask_dir = '/data/pascals_data/masks'
        return image_dir, mask_dir