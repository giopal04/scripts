#!/usr/bin/env python3

import os
import cv2
import glob
import numpy as np

from pathlib import Path
from functools import partial
from multiprocessing import Pool

def blend_images(image_path, mask_path, color_map, alpha=0.5):
    # Read the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Create an empty colorized mask
    colorized_mask = np.zeros_like(image)

    # Apply the color map to the mask
    for class_id, color in color_map.items():
        colorized_mask[mask == class_id] = color

    # Blend the original image with the colorized mask
    #blended_image = cv2.addWeighted(image, 1 - alpha, colorized_mask, alpha, 0)
    blended_image = cv2.addWeighted(image, 1, colorized_mask, alpha, 0)

    return blended_image

def process_image_pair(image_mask_pair, color_map, blend_dir, alpha=0.5):
    image_path, mask_path = image_mask_pair
    blended_image = blend_images(image_path, mask_path, color_map, alpha)

    # Save the blended image
    blended_image_path = blend_dir / Path(image_path).name.replace('img-', 'blended-img-')

    cv2.imwrite(str(blended_image_path), blended_image)

    print(f'Saved blended image: {blended_image_path}')

def main():
    # Directory containing the images and masks
    #directory = Path('/tmp/bio-36k')
    directory = Path('//mnt/btrfs-data/dataset/biodiversity/unreal-engine/ue5.2-rgb-mask-36k-1920x1080-20240926')
    imgs_dir  = directory / 'rgb'
    masks_dir = directory / 'masks'
    blend_dir = directory / 'blended'

    # Pattern to match image and mask files
    image_pattern = imgs_dir  / 'img-*.jpg'
    mask_pattern  = masks_dir / 'img-*.jpg'

    print(f'Processing images in {imgs_dir} and masks in {masks_dir}')

    # List of image and mask files
    image_files = sorted(glob.glob(str(image_pattern)))
    mask_files  = sorted(glob.glob(str(mask_pattern)))

    print(f'Found {len(image_files)} images and {len(mask_files)} masks')

    # Color map (class_id: [B, G, R])
    color_map = {
        0:   [  0,   0,   0], # Class 0 - background	(black)
        50:  [255,   0,   0], # Class 1 - grass		(blue)
        150: [  0, 255,   0], # Class 2 - trees		(green)
        200: [  0,   0, 255], # Class 3 - cube		(blue)
        # Add more classes as needed
    }

    # Alpha factor for blending
    alpha = 0.5

    ## Process each image and mask pair
    #for image_path, mask_path in zip(image_files, mask_files):
    #    blended_image = blend_images(image_path, mask_path, color_map, alpha)

    #    # Save the blended image
    #    blended_image_path = blend_dir / Path(image_path).name.replace('img-', 'blended-img-')

    #    cv2.imwrite(str(blended_image_path), blended_image)

    #    print(f'Saved blended image: {blended_image_path}')

    # Assuming image_files, mask_files, color_map, alpha, and blend_dir are defined
    image_mask_pairs = list(zip(image_files, mask_files))

    with Pool(processes=16) as pool:
        pool.map(partial(process_image_pair, color_map=color_map, blend_dir=blend_dir, alpha=alpha), image_mask_pairs)

if __name__ == '__main__':
    main()
