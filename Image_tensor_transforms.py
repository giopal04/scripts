import torch
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import ToPILImage

def PIL2Tensor(image: Image, is_img: bool = True) -> torch.Tensor:

    if is_img:
        return torch.tensor(np.array(image)).permute(2,0,1)
    
    else:
        return torch.tensor(np.array(image))

def Tensor2PIL(tensor: torch.Tensor) -> Image:
    to_image = ToPILImage()

    return to_image(tensor)
