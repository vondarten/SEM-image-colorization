import numpy as np
import torch
import warnings
from skimage.color import rgb2lab, lab2rgb
from typing import Dict
warnings.filterwarnings("ignore", category=UserWarning)

def lab_to_rgb(L, ab) -> np.ndarray:

    """
    Convert images from LAB color space to RGB color space.

    This function denormalizes the LAB images and converts them to RGB format.

    Args:
        L (torch.Tensor): A tensor of shape (N, 1, H, W) where N is the number of images,
                          H is the height, and W is the width. The tensor contains 
                          the L (luminance) channel of LAB images, normalized to the range [-1, 1].
        ab (torch.Tensor): A tensor of shape (N, 2, H, W) where N is the number of images,
                           H is the height, and W is the width. The tensor contains 
                           the a and b channels of LAB images, normalized to the range [-1, 1].

    Returns:
        np.ndarray: A NumPy array of shape (N, H, W, 3) containing the RGB images, 
                    where N is the number of images, H is the height, W is the width, 
                    and 3 represents the RGB channels.
    """

    # Denormalize
    L = (L + 1.0) * 50.0
    ab = ab * 255.0 - 128.0

    Lab = torch.cat([L, ab], dim=1).cpu().numpy()
    rgb_imgs = []

    for img in Lab:
        img = np.transpose(img, (1, 2, 0)) 
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def rgb_to_lab(img: np.ndarray) -> Dict:

    """
    Convert an RGB image to LAB color space.

    This function takes an RGB image, converts it to LAB color space, and normalizes
    the LAB channels to a specific range.

    Args:
        img (np.ndarray): A NumPy array representing the RGB image with shape 
                          (height, width, 3). The image can be in RGBA format, 
                          but it will be converted to RGB if necessary.

    Returns:
        Dict: A dictionary with two keys:
            - 'L': A tensor of shape (1, 1, height, width) representing the luminance 
                    channel of the LAB image, normalized to the range [-1, 1].
            - 'ab': A tensor of shape (2, 1, height, width) representing the a and b 
                    channels of the LAB image, normalized to the range [-1, 1].
    """

    img = img.convert('RGB')

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
