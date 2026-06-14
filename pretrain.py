import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import yaml
import scienceplots
import random
import pandas as pd
from build_model import build_model
from tqdm import tqdm
from fastai.vision.models.unet import DynamicUnet
from copy import deepcopy
from glob import glob
from datetime import datetime
from data_utils import get_dataloaders
from typing import Dict, List
# !pip install latex
# sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
plt.style.use('science')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[INFO] Device: {device} - Torch version: {torch.__version__}')

EXPERIMENT_NAME = 'unet-resnet18-just-mish-phage-best-384-weighted-detail-loss-v10'
SEED = 1337
IMAGE_SIZE = 384
BATCH_SIZE = 10
PATIENCE = 100
N_WORKERS = 1 
DATASET_PATH = './dataset_finetune_best_with_ai_aug'
BACKBONE = 'resnet18'
SELF_ATTENTION = False
ACTIVATION_FUNCTION = 'Mish'
ACTIVE_WEIGHT = 8.0
CHROMA_SCALE = 12.0
GRADIENT_WEIGHT = 0.2
CRITERION = None
OPTIM = torch.optim.Adam
LR = 5e-4
EPOCHS = 1000

print(f'[INFO] Starting experiment {EXPERIMENT_NAME}')

experiment_path = F'./experiments/{EXPERIMENT_NAME}-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
os.makedirs(experiment_path, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_paths = glob(f'{DATASET_PATH}/train/*.png')  \
            + glob(f'{DATASET_PATH}/train/*.jpeg') \
            + glob(f'{DATASET_PATH}/train/*.jpg') 

val_paths = glob(f'{DATASET_PATH}/val/*.png') \
          + glob(f'{DATASET_PATH}/val/*.jpeg') \
          + glob(f'{DATASET_PATH}/val/*.jpg') 

train_dl = get_dataloaders(batch_size=BATCH_SIZE, 
                           n_workers=N_WORKERS, 
                           file_paths=train_paths, 
                           image_size=IMAGE_SIZE)

val_dl = get_dataloaders(batch_size=BATCH_SIZE, 
                         n_workers=N_WORKERS, 
                         file_paths=val_paths, 
                         image_size=IMAGE_SIZE, 
                         train=False)

class ChromaWeightedL1Loss(torch.nn.Module):
    def __init__(self, neutral: float=128 / 255, active_weight: float=8.0, chroma_scale: float=12.0):
        super().__init__()
        self.neutral = neutral
        self.active_weight = active_weight
        self.chroma_scale = chroma_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        chroma = torch.sqrt(
            (target[:, 0:1] - self.neutral) ** 2 +
            (target[:, 1:2] - self.neutral) ** 2
        )
        weights = 1.0 + self.active_weight * torch.clamp(chroma * self.chroma_scale, 0.0, 1.0)
        return (torch.abs(pred - target) * weights).mean()


class ABGradientLoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        return torch.abs(pred_dx - target_dx).mean() + torch.abs(pred_dy - target_dy).mean()


class ColorizationDetailLoss(torch.nn.Module):
    def __init__(self, active_weight: float=8.0, chroma_scale: float=12.0, gradient_weight: float=0.2):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.chroma_l1 = ChromaWeightedL1Loss(active_weight=active_weight, chroma_scale=chroma_scale)
        self.gradient = ABGradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_2d = pred.squeeze(2) if pred.ndim == 5 else pred
        target_2d = target.squeeze(2) if target.ndim == 5 else target
        return self.chroma_l1(pred, target) + self.gradient_weight * self.gradient(pred_2d, target_2d)

def train_and_val(loss_values_train: List[float], 
                  loss_values_val: List[float], 
                  model: DynamicUnet, 
                  scaler: torch.cuda.amp.GradScaler,
                  train_dl: torch.utils.data.DataLoader, 
                  val_dl: torch.utils.data.DataLoader,
                  optim: torch.optim.Adam,
                  scheduler,
                  criterion: torch.nn.Module,  
                  epochs) -> Dict:

    torch.cuda.empty_cache()

    best_model = deepcopy(model.state_dict())

    min_val_loss = np.inf
    patience_counter = 0
    best_epoch = 0

    loss_train = np.inf
    loss_val = np.inf

    for i in range(epochs):

        loss_batch = 0.0
        total_samples = 0
        
        ### Train
        for data in tqdm(train_dl, desc = f"Epoch {i+1}/{epochs} | Loss Train: {loss_train:.6f}"):
            
            L, ab = data['L'].to(device), data['ab'].to(device)

            model.train()
            
            preds = model(L).unsqueeze(2)
            
            optim.zero_grad()

            # Casts operations to mixed precision
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                loss = criterion(preds, ab)

            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            loss_batch += loss.item() * L.size(0)  
            total_samples += L.size(0) 
        
        loss_train = loss_batch / total_samples
        loss_values_train.append(loss_train)

        loss_batch = 0.0
        total_samples = 0

        ### Validation
        for data in tqdm(val_dl, desc = f"Epoch {i+1}/{epochs} | Loss Val: {loss_val:.3f}"):
            L, ab = data['L'].to(device), data['ab'].to(device)
            
            model.eval()

            with torch.inference_mode():
                preds = model(L).unsqueeze(2)
                
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                loss = criterion(preds, ab)

            loss_batch += loss.item() * L.size(0)  
            total_samples += L.size(0) 
        
        loss_val = loss_batch / total_samples
        loss_values_val.append(loss_val)

        scheduler.step()

        current_lr = optim.param_groups[0]["lr"]
        print(f'[INFO] Current LR: {current_lr:.2e}')

        # Early stopping
        if loss_val < min_val_loss:
            best_model = deepcopy(model.state_dict())
            print(f'[INFO] New best model found [{loss_val:.4f}]. Updating state dict...')
            min_val_loss = loss_val
            patience_counter = 0 
            best_epoch = i
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[INFO] Early stopping triggered at epoch {best_epoch}, best validation loss: {min_val_loss:.4f}")
                break

    return best_model, best_epoch

def save_train_results(model: DynamicUnet, 
                       experiment_path: str, 
                       backbone: str, 
                       loss_values_train: List,
                       loss_values_val: List,
                       best_epoch: int=0
                       ) -> None:

    model_path = f"{experiment_path}/model.pth"
    training_results_path = f"{experiment_path}/results-{backbone}.csv"

    model = model.half() 
    
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': loss_values_val[-1]
    }, model_path)
    
    print(f'[INFO] Saved model weights to {model_path}.')

    df = pd.DataFrame({
        'train_loss': loss_values_train,
        'val_loss': loss_values_val
    })

    df.to_csv(training_results_path, index=False)
    print(f'[INFO] Saved training results to {model_path}.')

    with plt.style.context('science'):
        plt.figure(figsize=(4,3))
        plt.title('Losses from train and validation')
        plt.plot(df.index, df['train_loss'], label='Train')
        plt.plot(df.index, df['val_loss'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{experiment_path}/loss_plot.png')


model = build_model(backbone=BACKBONE, 
                    n_input=1, 
                    n_output=2, 
                    size=IMAGE_SIZE,
                    activation_function=ACTIVATION_FUNCTION,
                    self_attention=SELF_ATTENTION
                    ).to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'[INFO] Total number of parameters: {pytorch_total_params}')

### Automatic Mixed Precision
scaler = torch.cuda.amp.GradScaler()

optim = OPTIM(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optim,
    lr_lambda=lambda epoch: (
        1.0 if epoch < 70 else
        0.8 if epoch < 100 else
        0.5
    )
)

criterion = ColorizationDetailLoss(
    active_weight=ACTIVE_WEIGHT,
    chroma_scale=CHROMA_SCALE,
    gradient_weight=GRADIENT_WEIGHT,
)

loss_values_train = []
loss_values_val = []

hyperparams = {
    'EXPERIMENT_NAME': EXPERIMENT_NAME,
    'SEED': SEED,
    'IMAGE_SIZE': IMAGE_SIZE,
    'BATCH_SIZE': BATCH_SIZE,
    'PATIENCE': PATIENCE,
    'N_WORKERS': N_WORKERS,
    'DATASET_PATH': DATASET_PATH,
    'BACKBONE': BACKBONE,
    'SELF_ATTENTION': SELF_ATTENTION,
    'ACTIVATION_FUNCTION': ACTIVATION_FUNCTION,
    'CRITERION': criterion.__class__.__name__,
    'ACTIVE_WEIGHT': ACTIVE_WEIGHT,
    'CHROMA_SCALE': CHROMA_SCALE,
    'GRADIENT_WEIGHT': GRADIENT_WEIGHT,
    'OPTIM': optim.__class__.__name__,
    'LR': LR,
    'EPOCHS': EPOCHS
}

with open(f'{experiment_path}/config.yaml', 'w') as config_file:
    yaml.dump_all([hyperparams], config_file)


tic = time.time()
best_model, best_epoch = train_and_val(loss_values_train,
                                        loss_values_val,
                                        model, 
                                        scaler,
                                        train_dl, 
                                        val_dl,
                                        optim,
                                        scheduler,
                                        criterion, 
                                        EPOCHS)

model.load_state_dict(best_model)

print(f'Training took {round((time.time() - tic)/60, 2)} minutes.')

save_train_results(model,
                   experiment_path,
                   BACKBONE,
                   loss_values_train,
                   loss_values_val,
                   best_epoch
                    )

print(f'[INFO] Finished training.')