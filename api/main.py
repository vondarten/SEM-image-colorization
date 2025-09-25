import os
import gc
import io
import base64
import logging
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from PIL import Image
from torchvision import models, transforms
from fastai.torch_core import TensorBase
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from api.schemas import ColorizeImageRequest, ColorizeImageResponse
from api.utils import lab_to_rgb, rgb_to_lab

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="SEM Image Colorizer", description="API for colorizing SEM images")

API_TOKEN = os.environ.get("API_TOKEN", "default")
api_key_header = APIKeyHeader(name="Token", auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key_header

class UNetColorizer(DynamicUnet):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: TensorBase) -> TensorBase:
        return super().forward(x)

def load_model() -> UNetColorizer:
    basemodel = models.resnet18(weights=None)
    body = create_body(basemodel, n_in=1, pretrained=False, cut=-2) 
    model = UNetColorizer(body, 2, (384, 384), self_attention=False, act_cls=torch.nn.Mish).to('cpu')
    experiment = 'gan-unet-resnet18-just-mish-4-2024-07-18-17:45:12'
    
    model_path = f"./experiments/{experiment}/model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    saved_model = torch.load(model_path, map_location='cpu')
    model.load_state_dict(saved_model['model_state_dict'])
    logger.info("Model loaded successfully.")
    return model

def colorize_image(model: UNetColorizer, img: Image) -> Image:
    orig_img = img.convert('L')
    width, height = orig_img.size
    
    # Convert the original image to LAB and keep the original L channel
    orig_lab = rgb_to_lab(img)
    orig_L = orig_lab['L']

    # Resize the original image and convert to LAB for model input
    resized_img = transforms.Resize((384, 384), Image.BICUBIC)(img)
    resized_lab = rgb_to_lab(resized_img)
    
    sample = {'L': resized_lab['L'].unsqueeze(0).to('cpu')}

    model.eval()
    with torch.inference_mode():
        model_ab = model(sample['L'])
    
    # Upsample the generated ab channels to the original image size
    model_ab = torch.nn.functional.interpolate(model_ab, size=(height, width), mode='bilinear')

    # Combine the original L channel with the upsampled ab channels
    colorized_img_np = lab_to_rgb(orig_L.unsqueeze(0), model_ab).squeeze(0)

    gc.collect()

    return Image.fromarray((colorized_img_np * 255).astype(np.uint8))

model = load_model()

@app.post("/colorize", status_code=200, response_model=ColorizeImageResponse)
async def colorize(request: ColorizeImageRequest, api_key: str = Depends(get_api_key)):
    logger.info("Received colorization request")
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(image_data))

        # Perform colorization
        colorized_img = colorize_image(model, img)
        
        # Encode the colorized image to base64
        buffer = io.BytesIO()
        colorized_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "image": img_str,
            "status": 200,
            "message": "Success"
        }
    
    except Exception as e:
        logger.critical(f'Critical - {e}')
        return {
            "status": 500,
            "message": "Internal error!"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)