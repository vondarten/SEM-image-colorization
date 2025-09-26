import logging
import numpy as np
import os
import time
import torch
from PIL import Image
from torchvision import models, transforms
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from api.utils import lab_to_rgb, rgb_to_lab

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColorizationController:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self) -> DynamicUnet:
        logger.info('Loading model...')
        tic = time.time()
        basemodel = models.resnet18(weights=None)
        body = create_body(basemodel, n_in=1, pretrained=False, cut=-2) 
        model = DynamicUnet(body, 2, (384, 384), self_attention=False, act_cls=torch.nn.Mish).to('cpu')
        experiment = 'gan-unet-resnet18-just-mish-4-2024-07-18-17:45:12'
        
        model_path = f"./experiments/{experiment}/model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        saved_model = torch.load(model_path, map_location='cpu')
        model.load_state_dict(saved_model['model_state_dict'])
        logger.info(f">>> Model loaded successfully in {time.time() - tic:.4f}s")
        return model

    def colorize_image(self, img: Image) -> Image:
        
        tic = time.time()

        orig_img = img.convert('L')
        width, height = orig_img.size
        
        # Convert the original image to LAB and keep the original L channel
        orig_lab = rgb_to_lab(img)
        orig_L = orig_lab['L']

        # Resize the original image and convert to LAB for model input
        resized_img = transforms.Resize((384, 384), Image.BILINEAR)(img)
        resized_lab = rgb_to_lab(resized_img)

        logger.info('Converted original and resized images to LAB color space')
        
        sample = {'L': resized_lab['L'].unsqueeze(0).to('cpu')}

        model_tic = time.time()

        self.model.eval()
        with torch.inference_mode():
            model_ab = self.model(sample['L'])
        
        logger.info(f'>>> Model inference done in {time.time() - model_tic:4f}s')

        # Upsample the generated ab channels to the original image size
        model_ab = torch.nn.functional.interpolate(model_ab, size=(height, width), mode='bilinear')

        # Combine the original L channel with the upsampled ab channels
        colorized_img_np = lab_to_rgb(orig_L.unsqueeze(0), model_ab).squeeze(0)

        logger.info(f'>>> Colorization completed in {time.time() - tic:.4f}s')

        return Image.fromarray((colorized_img_np * 255).astype(np.uint8))