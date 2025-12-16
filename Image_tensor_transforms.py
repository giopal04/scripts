import torch
from PIL import Image
from torchvision import transforms as T

def PIL2Tensor(image: Image) -> torch.Tensor:
    to_tensor = T.ToTensor()

    return to_tensor(image)

def Tensor2PIL(tensor: torch.Tensor) -> Image:
    to_image = T.ToPILImage()

    return to_image(tensor)
