#!/usr/bin/env python3

# Run with:
# ./fast.ai-inference.py --input_dir /mnt/raid1/dataset/shrec-2025-protein-classification/v2-20250331/test-orig-renamed-labeled-screenshots/unk --output_dir /mnt/raid1/dataset/shrec-2025-protein-classification/v2-20250331/inference-test-set-labeled-screenshots --model_path /mnt/raid1/repos/shrec2025/Protein_Classification/notebooks/models/shrec-2025-protein-classification-resnet50--no-data-aug-img_size-320-320-1a-2025-04-04_12.15.05-BS-64-LR-0.0005-0.001-epoch-8-valid_loss-0.2962.pth --batch_size 64 --img_size 320 --device cuda:0 &> fast.ai-inference-on-test-set-`currdate`.txt

import argparse
import shutil
from pathlib import Path
import dill
import timm
import torch
import torch.nn.functional as F
from fastai.vision.all import *

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('--input_dir',	type=str, required=True,				help='Input  directory with images')
    parser.add_argument('--output_dir',	type=str, required=True,				help='Output directory for processed images')
    parser.add_argument('--model_path',	type=str, required=True,				help='Path to trained model .pth file')
    parser.add_argument('--batch_size',	type=int, default=4,					help='Inference batch size')
    parser.add_argument('--img_size',	type=int, default=518,					help='Inference image size (better if equal to training size)')
    parser.add_argument('--device',	type=str, default='cuda:0',				help='GPU to use')
    parser.add_argument('--arch',	type=str, default='vit_giant_patch14_reg4_dinov2',	help='ViT architecture to use')

    return parser.parse_args()

def load_model(input_dir, model_path, device, arch, img_size):
	# Create dummy dataloaders to initialize model architecture
	dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
					   get_items=get_image_files,
					   get_y=parent_label,
					   item_tfms=Resize(img_size),
					   batch_tfms=Normalize.from_stats(*imagenet_stats))

	metrics = [
                accuracy,
                Precision(average='macro'),
                Recall(average='macro'),
                F1Score(average='macro'),
                Jaccard(average='macro'),
              ]    
	# Dummy dataloaders with 97 classes (matching your model output)
	dls = dblock.dataloaders(input_dir, bs=4, num_classes=97)
	
	# Create learner and load weights
	#learn = vision_learner(dls, resnet50, pretrained=False, n_out=97)
	print(f'Allocating a ViT Learner. Type: {arch}')
	model = timm.create_model(arch, pretrained=True, num_classes=97, dynamic_img_size=True)
	print(f'{model = }')
	learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=metrics, cbs=[MixedPrecision])
	#learn = vision_learner(dls, arch, pretrained=False, n_out=97, loss_func=CrossEntropyLossFlat(), metrics=metrics, cbs=[MixedPrecision])
	learn.model = learn.model.to(device)
	print(f'{learn.model}')
	#learn.model.load_state_dict(torch.load(model_path, map_location=device))
	#learn = load_learner(model_path, cpu=False, pickle_module=dill)
	#print(f'{type(learn)}')
	#print(f'{type(learn.model)}')
	#print(f'{learn}')
	#print(f'{learn.model}')

	model_path = Path(model_path).resolve().with_suffix('')

	print(f'Loading model: {model_path} to device: {device}')
	learn.load(model_path, device=device, weights_only=False)
	#learn.dls = dls
	return learn

def write_batch_log(log_entries, output_dir):
	"""Write batch results to a log file in the output directory."""
	log_file = output_dir / "inference_log.txt"
	with open(log_file, "a", encoding="utf-8") as f:
		for entry in log_entries:
			f.write(f"{entry}\n")

def process_batch(batch, model, files, output_dir, device, debug=False):
	with torch.no_grad():
		preds = model(batch.to(device))
		if debug:
			print(f'Predictions: {preds}')
		probs = F.softmax(preds, dim=1)
		if debug:
			print(f'Probabilities: {probs}')
		confs, clss = torch.max(probs, dim=1)
		if debug:
			print(f'Confidences: {confs} - Classes: {clss}')

	log_entries = []
	for file_path, cls, conf in zip(files, clss.cpu(), confs.cpu()):
		src_path = Path(file_path)
		file_id  = src_path.stem
		fid = file_id.split('-')[0]			# e.g. obj 1040
		sub_output_dir = output_dir / fid
		sub_output_dir.mkdir(parents=True, exist_ok=True)
		cls  = cls.item()
		conf = conf.item()
		new_stem = f"{file_path.stem}-cls-{cls:02d}-confidence-{conf:.3f}"
		new_path = sub_output_dir / f"{new_stem}{file_path.suffix}"
		#shutil.copy2(file_path, new_path)
		print(f'Hardlinking from {src_path} to {new_path}')
		#src_path.hardlink_to(new_path)
		new_path.hardlink_to(src_path)

		# Log entry formatting (class zero-padded to 2 digits)
		log_entries.append(f"{file_path.stem} - {cls:02d} - {conf:.3f}")

	# Write all entries for this batch
	write_batch_log(log_entries, output_dir)

def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Setup directories
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    learn = load_model(input_dir, args.model_path, device, arch=args.arch, img_size=args.img_size)
    learn.model.eval()
    
    # Create inference dataloader
    files = get_image_files(input_dir)
    dl    = learn.dls.test_dl(files, bs=args.batch_size, num_workers=48)
    
    # Process batches
    for i, batch in enumerate(dl):
        start = i * args.batch_size
        end = start + len(batch[0])
        batch_files = files[start:end]
        process_batch(batch[0], learn.model, batch_files, output_dir, device, debug=False)
        print(f"Processed batch {i+1} of {len(dl)}")

if __name__ == '__main__':
    main()
