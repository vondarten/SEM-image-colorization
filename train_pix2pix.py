import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torchvision.models as models
import scienceplots
import yaml
from copy import deepcopy
from glob import glob
from datetime import datetime
from tqdm import tqdm
from torchvision import models 
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from data_utils import get_dataloaders, LossValues
from patch_gan import PatchGAN

plt.style.use('science')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'[INFO] Device: {device} - Torch version: {torch.__version__}')

EXPERIMENT_NAME = 'gan-unet-resnet18-just-mish-4-pt3'
PRETRAINED_EXP = "unet-resnet18-just-mish-2024-07-18-11:44:47"
SEED = 1337
IMAGE_SIZE = 384
BATCH_SIZE = 16
PATIENCE = 100
N_WORKERS = max(0, os.cpu_count() - 4)
DATASET_PATH = './'
BACKBONE = 'resnet18'
SELF_ATTENTION = False
ACTIVATION_FUNCTION = 'Mish'
OPTIM_D = torch.optim.Adam
OPTIM_G = torch.optim.Adam
CRITERION_D = torch.nn.BCEWithLogitsLoss 
CRITERION_G = torch.nn.L1Loss
ADAM_B1 = 0.5
ADAM_B2 = 0.999
LAMBDA = 50
LR_G = 3e-5
LR_D = 1e-4
EPOCHS = 1000

experiment_path = F'./experiments/{EXPERIMENT_NAME}-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
os.makedirs(experiment_path, exist_ok=True)

print(f'[INFO] Starting experiment {EXPERIMENT_NAME}')

hyperparams = {
    'EXPERIMENT_NAME': EXPERIMENT_NAME,
    'PRETRAINED_EXP': PRETRAINED_EXP,
    'SEED': SEED,
    'IMAGE_SIZE': IMAGE_SIZE,
    'BATCH_SIZE': BATCH_SIZE,
    'PATIENCE': PATIENCE,
    'N_WORKERS': N_WORKERS,
    'DATASET_PATH': DATASET_PATH,
    'BACKBONE': BACKBONE,
    'SELF_ATTENTION': SELF_ATTENTION,
    'ACTIVATION_FUNCTION': ACTIVATION_FUNCTION,
    'OPTIM_D': OPTIM_D.__name__,
    'OPTIM_G': OPTIM_G.__name__,
    'CRITERION_D': CRITERION_D.__name__,
    'CRITERION_G': CRITERION_G.__name__,
    'ADAM_B1': ADAM_B1,
    'ADAM_B2': ADAM_B2,
    'LAMBDA': LAMBDA,
    'LR_G': LR_G,
    'LR_D': LR_D,
    'EPOCHS': EPOCHS
}

with open(f'{experiment_path}/config.yaml', 'w') as config_file:
    yaml.dump_all([hyperparams], config_file)

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

def build_model(backbone, n_input=1, n_output=2, size=256):

    if backbone == 'resnet18':
        backbone = models.resnet18(weights='DEFAULT')
    
    elif backbone == 'resnet34':
        backbone = models.resnet34(weights='DEFAULT')
    
    elif backbone == 'convnext_tiny':
        backbone = models.convnext_tiny(weights='DEFAULT')

    elif backbone == 'efficientnet_v2_s':
        backbone = models.efficientnet_v2_s(weights='DEFAULT')

    elif backbone == 'shufflenet_v2_x0_5':
        backbone = models.shufflenet_v2_x2_0(weights='DEFAULT')

    elif backbone == 'shufflenet_v2_x1_0':
        backbone = models.shufflenet_v2_x2_0(weights='DEFAULT')

    elif backbone == 'shufflenet_v2_x1_5':
        backbone = models.shufflenet_v2_x2_0(weights='DEFAULT')

    elif backbone == 'shufflenet_v2_x2_0':
        backbone = models.shufflenet_v2_x2_0(weights='DEFAULT')
  
    body = create_body(backbone, n_in=n_input, pretrained=True, cut=-2) 

    if ACTIVATION_FUNCTION:
        model = DynamicUnet(body, n_output, (size, size), self_attention=SELF_ATTENTION, act_cls=torch.nn.Mish).to(device)
    else:
        model = DynamicUnet(body, n_output, (size, size), self_attention=SELF_ATTENTION).to(device)

    return model


discriminator = PatchGAN(custom_weights_init=True).to(device)

generator = build_model(backbone=BACKBONE, 
                        n_input=1, 
                        n_output=2, 
                        size=IMAGE_SIZE).to(device)

saved_model = torch.load(f"./experiments/{PRETRAINED_EXP}/model.pth")
generator.load_state_dict(saved_model['model_state_dict'])

criterion_adversarial = CRITERION_D() 
criterion_l1 = CRITERION_G()

lambda_coef = LAMBDA
g_optim = OPTIM_G(generator.parameters(),
                  lr=LR_G,
                  betas=(ADAM_B1, ADAM_B2))

d_optim = OPTIM_D(discriminator.parameters(),
                  lr=LR_D,
                  betas=(ADAM_B1, ADAM_B2))

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

def train_and_val(loss_values: LossValues, 
                  gen,
                  disc, 
                  train_dl, 
                  val_dl,
                  g_optim, 
                  g_scaler,
                  d_optim,
                  d_scaler,
                  criterion_adversarial,
                  criterion_l1,  
                  epochs,
                  lambda_coef,
                  device,
                  patience) -> DynamicUnet:

    torch.cuda.empty_cache()

    best_gen = deepcopy(gen.state_dict())
    min_val_loss = np.inf
    patience_counter = 0

    for i in range(epochs):
        loss_batch_disc = 0.0
        loss_batch_disc_real = 0.0
        loss_batch_disc_gen = 0.0
        loss_batch_gen = 0.0
        loss_batch_gen_l1 = 0.0
        total_samples = 0

        ### Training Stage
        for data in tqdm(train_dl, 
                         desc=f"Epoch {i+1}/{epochs} | Loss_D: {loss_values.train_disc:.5f} [Loss_D_Real: {loss_values.train_disc_real:.5f} | Loss_D_Gen: {loss_values.train_disc_gen:.5f}] Loss_G: {loss_values.train_gen:.5f} [Loss_G_L1: {loss_values.train_gen_l1:.5f}]"):

            L, ab = data['L'].to(device), data['ab'].squeeze(2).to(device)

            # i) Train the discriminator
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                gen_ab = gen(L)
                disc_real = disc(L, ab)
                loss_disc_real = criterion_adversarial(disc_real, torch.ones_like(disc_real))
                
                disc_gen = disc(L, gen_ab.detach())
                loss_disc_gen = criterion_adversarial(disc_gen, torch.zeros_like(disc_gen))
                
                disc_loss = (loss_disc_real + loss_disc_gen) / 2
            
            d_optim.zero_grad()
            d_scaler.scale(disc_loss).backward()
            d_scaler.step(d_optim)
            d_scaler.update()

            # ii) Train the genenerator
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                disc_gen = disc(L, gen_ab)
                loss_disc_gen = criterion_adversarial(disc_gen, torch.ones_like(disc_gen))
                l1_loss = criterion_l1(gen_ab, ab)
                gen_loss = loss_disc_gen + lambda_coef * l1_loss

            g_optim.zero_grad()
            g_scaler.scale(gen_loss).backward()
            g_scaler.step(g_optim)
            g_scaler.update()

            loss_batch_disc += disc_loss.item() * L.size(0)
            loss_batch_disc_real += loss_disc_real.item() * L.size(0)
            loss_batch_disc_gen += loss_disc_gen.item() * L.size(0)
            loss_batch_gen += gen_loss.item() * L.size(0)
            loss_batch_gen_l1 += l1_loss.item() * L.size(0)
            total_samples += L.size(0)

        loss_values.train_disc = loss_batch_disc / total_samples
        loss_values.train_disc_real = loss_batch_disc_real / total_samples
        loss_values.train_disc_gen = loss_batch_disc_gen / total_samples
        loss_values.train_gen = loss_batch_gen / total_samples
        loss_values.train_gen_l1 = loss_batch_gen_l1 / total_samples

        loss_values.train_disc_history.append(loss_values.train_disc)
        loss_values.train_gen_history.append(loss_values.train_gen)

        ### Validation Stage
        loss_batch = 0.0
        total_samples = 0

        for data in tqdm(val_dl, desc=f"Epoch {i+1}/{epochs} | Loss Val: {loss_values.val:.3f}"):
            L, ab = data['L'].to(device), data['ab'].to(device)
            
            gen.eval()

            with torch.inference_mode():
                preds = gen(L).unsqueeze(2)
                
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                loss = criterion_l1(preds, ab)

            loss_batch += loss.item() * L.size(0)
            total_samples += L.size(0)
        
        loss_values.val = loss_batch / total_samples
        loss_values.val_history.append(loss_values.val)

        # Early stopping
        if loss_values.val < min_val_loss:
            best_gen = deepcopy(gen.state_dict())
            print(f'[INFO] New best model found [{loss_values.val:5f}]. Updating state dict...\n')
            min_val_loss = loss_values.val
            patience_counter = 0
            loss_values.best_epoch = i
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {loss_values.best_epoch}, best validation loss: {min_val_loss:.4f}")
                break

    return best_gen

def save_train_results(model, 
                       experiment_path, 
                       backbone, 
                       loss_values
                       ):

    model_path = f"{experiment_path}/model.pth"
    training_results_path = f"{experiment_path}/results-{backbone}.csv"

    model = model.half() 
    
    torch.save({
        'epoch': loss_values.best_epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': loss_values.val
        }, model_path)
    
    print(f'[INFO] Saved model weights to {model_path}.')

    df = pd.DataFrame({
        'train_loss_discriminator': loss_values.train_disc_history,
        'train_loss_generator': loss_values.train_gen_history,
        'val_loss': loss_values.val_history
    })

    df.to_csv(training_results_path, index=False)
    print(f'[INFO] Saved training results to {training_results_path}.')

    # Plot 1: Generator vs Discriminator Loss
    plt.figure(figsize=(4,3))
    plt.plot(df.index, df['train_loss_discriminator'], label='Discriminator Train Loss')
    plt.plot(df.index, df['train_loss_generator'], label='Generator Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator vs Discriminator Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{experiment_path}/gen_vs_disc_loss.png')
    plt.close()
    
    # Plot 2: Train Discriminator vs Validation Loss
    plt.figure(figsize=(4,3))
    plt.plot(df.index, df['train_loss_discriminator'], label='Discriminator Train Loss')
    plt.plot(df.index, df['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Discriminator vs Validation Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{experiment_path}/train_disc_vs_val_loss.png')
    plt.close()

    # Plot 3: Train Generator vs Validation Loss
    plt.figure(figsize=(4,3))
    plt.plot(df.index, df['train_loss_generator'], label='Generator Train Loss')
    plt.plot(df.index, df['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Generator vs Validation Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{experiment_path}/train_gen_vs_val_loss.png')
    plt.close()


tic = time.time()

loss_values = LossValues()

best_generator = train_and_val(loss_values,
                               generator,
                               discriminator, 
                               train_dl,
                               val_dl,
                               g_optim, 
                               g_scaler,
                               d_optim,
                               d_scaler,
                               criterion_adversarial,
                               criterion_l1,  
                               EPOCHS,
                               LAMBDA,
                               device,
                               PATIENCE)

generator.load_state_dict(best_generator)

print(f'Training took {round((time.time() - tic)/60, 2)} minutes.')

save_train_results(generator,
                   experiment_path,
                   BACKBONE,
                   loss_values
                   )

print(f'[INFO] Finished training.')