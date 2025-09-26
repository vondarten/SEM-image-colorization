import os
import io
import base64
import logging
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from PIL import Image
from api.schemas import ColorizeImageRequest, ColorizeImageResponse
from api.controller import ColorizationController

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

controller = ColorizationController()

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

@app.post("/colorize", status_code=200, response_model=ColorizeImageResponse)
async def colorize(request: ColorizeImageRequest, api_key: str = Depends(get_api_key)):
    logger.info(f"Received colorization request")
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(image_data))

        # Perform colorization using the controller
        colorized_img = controller.colorize_image(img)
        
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