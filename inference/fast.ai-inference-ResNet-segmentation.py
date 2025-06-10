#!/usr/bin/env python3

# Run segmentation inference with:

# ./fast.ai-inference-ViT-segmentation.py --input_dir images --output_dir output --model_path models/surface-damage-segmentation-CH-UNet-vit_tiny_patch16_224-basic-data-aug-img_size-270-270-1b-2025-05-12_14.49.03-WD-0.0001-BS-4-LR-0.0001-0.001-epoch-3-valid_loss-0.1767.pth --batch_size 16 --img_size 224 --arch vit_tiny_patch16_224 --num_classes 2

# ./fast.ai-inference-ResNet-segmentation.py --model_path /mnt/raid1/repos/pothole-detection/models/pothole-segmentron-UNet-resnet101-basic-data-aug-img_size-540-540-1a-2025-05-22_18.35.22-WD-0.0001-BS-6-LR-0.0001-0.001-epoch-0-valid_loss-22.9389.pth --input_dir /tmp/potholes-train --output_dir /mnt/raid1/repos/pothole-detection/inference/bing-images-download/dataset/cracked-statue-inference --encoder ResNet-101 --batch_size 2 --device cuda:0 --num_classes 5 --img_size 540

# Run classification inference with:

# ./fast.ai-inference.py --input_dir /mnt/raid1/dataset/shrec-2025-protein-classification/v2-20250331/test-orig-renamed-labeled-screenshots/unk --output_dir /mnt/raid1/dataset/shrec-2025-protein-classification/v2-20250331/inference-test-set-labeled-screenshots --model_path /mnt/raid1/repos/shrec2025/Protein_Classification/notebooks/models/shrec-2025-protein-classification-resnet50--no-data-aug-img_size-320-320-1a-2025-04-04_12.15.05-BS-64-LR-0.0005-0.001-epoch-8-valid_loss-0.2962.pth --batch_size 64 --img_size 320 --device cuda:0 &> fast.ai-inference-on-test-set-`currdate`.txt

# test img: S8_jpg.rf.793e49c2ae65e25bd2d7a14bb534e5c5.jpg

import argparse
import shutil
from pathlib import Path
import dill
import timm
import torch
import torch.nn.functional as F
from fastai.vision.all import *

import cv2

from inference_utils import change_mask_color, overlay, get_img_and_patch_size, load_model, write_batch_log, process_batch

def parse_args():
    parser = argparse.ArgumentParser(description='Image Segmentation Inference witha a ResNet model')
    parser.add_argument('--task',		type=str, default='segmentation',	help='Only classification and segmentation are supported at the moment')
    parser.add_argument('--input_dir',		type=str, required=True,		help='Input  directory with images')
    parser.add_argument('--output_dir',		type=str, required=True,		help='Output directory for processed images')
    parser.add_argument('--model_path',		type=str, required=True,		help='Path to trained model .pth file')
    parser.add_argument('--batch_size',		type=int, default=4,			help='Inference batch size')
    parser.add_argument('--img_size',		type=int, default=518,			help='Inference image size (better if equal to training size)')
    parser.add_argument('--device',		type=str, default='cuda:0',		help='GPU to use')
    parser.add_argument('--encoder',		type=str, default='ResNet-50',		help='ResNet encoder to use')
    parser.add_argument('--num_classes',	type=int, default=2,			help='Number of classes in the classification/segmentation problem')

    return parser.parse_args()

def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Setup directories
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    learn = load_model(input_dir, args.model_path, device, encoder=args.encoder, img_size=args.img_size, bs=args.batch_size, num_classes=args.num_classes, args=args, task=args.task)
    learn.model.eval()
    
    # Create inference dataloader
    files = get_image_files(input_dir)
    dl    = learn.dls.test_dl(files, bs=args.batch_size, num_workers=48)
    
    # Process batches
    for i, batch in enumerate(dl):
        start = i * args.batch_size
        end = start + len(batch[0])
        batch_files = files[start:end]
        process_batch(batch[0], learn.model, batch_files, output_dir, device, args, debug=False)
        print(f"Processed batch {i+1} of {len(dl)}")

if __name__ == '__main__':
    main()
