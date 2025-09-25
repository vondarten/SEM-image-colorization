from pydantic import BaseModel
from typing import Optional

class ColorizeImageRequest(BaseModel):
    image: str
    
class ColorizeImageResponse(BaseModel):
    image: Optional[str] = None 
    status: int
    message: str