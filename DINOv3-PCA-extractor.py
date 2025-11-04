#!/usr/bin/env python

import torch
from torchvision.transforms.functional import resize, normalize, to_tensor
from transformers import DINOv3ViTModel
from transformers.image_utils import load_image
import numpy as np

from sklearn.decomposition import PCA

from PIL import Image
from pathlib import Path
import argparse

DINOv3_MODELS = {
    'vit-small': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'vit-small-plus': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
    'vit-base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'vit-large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'vit-huge-plus': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
    'vit-7b': 'facebook/dinov3-vit7b16-pretrain-lvd1689m'
}
PATCH_SIZE = 16

def image_preprocess(image, img_dim, device, patch_size=16):
    if isinstance(img_dim, int):
        w, h = image.size
        h_patches = int(img_dim / patch_size)
        w_patches = int((w * img_dim) / (h * patch_size))

        img_dim = (h_patches * patch_size, w_patches * patch_size)

    image_resized = to_tensor(resize(image, img_dim))
    image_resized_norm = normalize(image_resized, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image_resized_norm = image_resized_norm.unsqueeze(0).to(device)

    return image_resized_norm

def PCA_extractor(patch_features, pca_dim):
    pca = PCA(n_components=3)
    pca.fit(patch_features[0])

    pca_features = pca.transform(patch_features[0])

    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    
    return pca_features.reshape(pca_dim).astype(np.uint8)

def main():
    #argparse and relative variables
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-dir',
                        required=True,
                        help='Directory containing the input images.')
    parser.add_argument('--output-dir',
                        required=True,
                        help='Directory where the PCA images will be saved.')
    parser.add_argument('--model',
                        choices=list(DINOv3_MODELS.keys()),
                        default='vit-small',
                        help='DINNO v3 model used to compute PCA. Default: "vit-small"')
    parser.add_argument('--img-dim',
                        required=True,
                        nargs='+',
                        type=int,
                        help='Dimention to which the input images are reshaped to. Affects the final dimention of the PCA images. Insert one or two values.')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        help='Setting the device to CPU')
    args = parser.parse_args()

    if len(args.img_dim) > 2:
        raise ValueError(f'"--img-dim" takes one or two arguments, {len(args.img_dim)} where given')
    for dim in args.img_dim:
        assert dim % PATCH_SIZE == 0, f'Image dimension ({dim}) must be multiple of {PATCH_SIZE}'

    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    MODEL_NAME = DINOv3_MODELS[args.model]
    IMG_DIM = args.img_dim[0] if len(args.img_dim) == 1 else tuple(args.img_dim)
    DEVICE = 'cuda:0' if not args.no_cuda else 'cpu'

    #loading model
    print(f'Using device {DEVICE}')
    print(f'Loading model {MODEL_NAME}')
    model = DINOv3ViTModel.from_pretrained(MODEL_NAME).to(DEVICE)

    #computing PCA
    print('Starting PCA computations')
    input_images = list(INPUT_DIR.iterdir())

    for i, image_path in enumerate(input_images):
        print(f'{i+1}/{len(input_images)}: Processing image {image_path}')
        
        image = load_image(image_path.as_posix())
        image_size = image.size
        image_res_norm = image_preprocess(image=image, img_dim=IMG_DIM, device=DEVICE, patch_size=PATCH_SIZE)
    
        pca_dim = tuple([int(x/PATCH_SIZE) for x in image_res_norm.shape[2:]] + [3])

        #extracting patch features
        with torch.no_grad():
            outputs = model(image_res_norm)
        patch_features = outputs.last_hidden_state[:,5:].cpu()

        pca_img = PCA_extractor(patch_features=patch_features, pca_dim=pca_dim)
        pca_img_name = f'{image_path.stem}_{image_size[1]}x{image_size[0]}_PCA_{pca_dim[0]}x{pca_dim[1]}{image_path.suffix}'

        #saving the PCA img
        print(f'{i+1}/{len(input_images)}: Saving PCA image to {OUTPUT_DIR/pca_img_name}')
        Image.fromarray(pca_img).save(OUTPUT_DIR/pca_img_name)


    

if __name__ == '__main__':
    main()
