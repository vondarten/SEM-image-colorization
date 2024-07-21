import torch
import numpy as np
import torchvision.transforms as transforms
from dataclasses import dataclass, field
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab, lab2rgb

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

        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((image_size, image_size), Image.BICUBIC),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomPerspective(distortion_scale=0.1),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.transforms = transforms.Resize((image_size, image_size),  Image.BICUBIC)

    
    def rgb_to_lab(self, img: np.ndarray):

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
    

    def lab_to_rgb(self, L, ab):
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
        
        return self.rgb_to_lab(img)
    

    def __len__(self):
        return len(self.file_paths)
    

def get_dataloaders(batch_size=32, 
                    n_workers=4, 
                    pin_memory=True, 
                    file_paths=None,
                    image_size=256,
                    train=True):
    
    dataset = SEMColorizationDataset(file_paths=file_paths, 
                                     image_size=image_size,
                                     train=train)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=n_workers,
                            pin_memory=pin_memory)
    
    return dataloader