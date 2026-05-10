import torch
import numpy as np
import torchvision.transforms as transforms
from dataclasses import dataclass, field
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torch.utils.data import Dataset, DataLoader
from typing import Dict

@dataclass
class LossValues:
    train_disc: float = np.inf
    train_disc_real: float = np.inf
    train_disc_gen: float = np.inf
    train_gen: float = np.inf
    train_gen_l1: float = np.inf
    val: float = np.inf
    best_epoch: int = 0

    train_disc_history: list = field(default_factory=list)
    train_gen_history: list = field(default_factory=list)
    val_history: list = field(default_factory=list)

class SEMColorizationDataset(Dataset):
    def __init__(self, file_paths, image_size, train=True):
    
        self.file_paths = file_paths
        self.train = train
        self.l_noise_prob = 0.5
        self.l_noise_std = 0.1

        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((image_size, image_size), Image.BICUBIC),
                transforms.RandomResizedCrop((image_size, image_size), scale=(0.65, 1.0), ratio=(1.0, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.3),
                transforms.RandomRotation(degrees=50),
            ])
        else:
            self.transforms = transforms.Resize((image_size, image_size),  Image.BICUBIC)

    
    def rgb_to_lab(self, img: np.ndarray) -> Dict[str, torch.Tensor]:

        img_lab = rgb2lab(np.array(img))
    
        # Reshape to (image_size, image_size, channels)
        img_lab = torch.from_numpy(img_lab).permute(2, 0, 1).float()
        img_lab = torch.unsqueeze(img_lab, 1)

        L = img_lab[0]
        ab = img_lab[1:]

        # Normalization: -1.0 <= x <= 1.0
        L = (L / 50.0) - 1.0
        ab = (ab + 128.0) / 255.0
        
        return {'L': L, 'ab': ab}

    def add_l_noise(self, L: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.l_noise_prob:
            return L

        noise = torch.randn_like(L) * self.l_noise_std
        return torch.clamp(L + noise, min=-1.0, max=1.0)
    

    def lab_to_rgb(self, L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of images in the Lab color space and converts them to RGB.
        """

        # Denormalize
        L = (L + 1.0) * 50.0
        ab = ab * 255.0 - 128.0
        Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        rgb_imgs = []

        for img in Lab:
            img_rgb = lab2rgb(img)
            rgb_imgs.append(img_rgb)

        return np.stack(rgb_imgs, axis=0)
    

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert('RGB')
        img = self.transforms(img)
        sample = self.rgb_to_lab(img)

        if self.train:
            sample['L'] = self.add_l_noise(sample['L'])

        return sample
    
    def __len__(self):
        return len(self.file_paths)
    

def get_dataloaders(batch_size: int=32, 
                    n_workers: int=4, 
                    pin_memory: bool=True, 
                    file_paths:str = '',
                    image_size: int=256,
                    train: bool=True) -> DataLoader:
    
    dataset = SEMColorizationDataset(file_paths=file_paths, 
                                     image_size=image_size,
                                     train=train)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=n_workers,
                            pin_memory=pin_memory)
    
    return dataloader