#!/usr/bin/env python3

import random
import argparse

import numpy as np

import PIL
from PIL import Image
from pathlib import Path

import torch
torch.backends.cudnn.conv.fp32_precision = 'tf32'

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from sam3.model_builder import build_sam3_video_predictor

from sam3.agent.viz import visualize

import matplotlib
import matplotlib.pyplot as plt

from classes.maximally_distinct_colormap import hex_to_rgb_tuple, maximally_distinct_colormap

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",	type=str, default="/mnt/raid1/repos/sam3/models/sam3.pt",	help="Model checkpoint to load")
parser.add_argument("--image",		type=str, default="",						help="Image to segment")
parser.add_argument("--video",		type=str, default="",						help="Video to segment")
parser.add_argument("--text",		type=str, default="person",					help="Video to segment")
parser.add_argument("--output-dir",	type=str, default="/tmp",					help="Output directory")
parser.add_argument("--debug",		type=int, default=0,						help="Debug mode")
args = parser.parse_args()

# Run with:
# ---------
# sam3-predict-image-or-video.py --image damaged-statues/damaged-statues-Image_5_00001_.png --text 'statue OR monument OR column OR brick wall OR statue head OR capital column'

# Alternative version that modifies in-place (more memory efficient)
def zero_rgb_on_zero_alpha_inplace(arr):
	'''
	Sets R, G, B values to zero where the alpha channel is zero (in-place).
	Args:
		arr: numpy array of shape (..., 4) with dtype uint8
	Returns:
		The modified array (same object)
	'''
	# Create mask where alpha is 0
	mask = arr[..., 3] == 0
	# Set R, G, B to 0 where mask is True
	arr[mask, 0:3] = 0
	return arr

def overlay_masks_pil(image, masks, alpha=0.5, debug=False):
	'''
	Overlay masks on image using PIL for better blending.
	Args:
		image: PIL Image
		masks: Tensor/array of masks [N, H, W]
		alpha: Transparency of masks
	'''

	if isinstance(image, PIL.Image.Image):
		print(f'overlay_masks_pil() received: {type(image) = } - {image.size = } - {type(masks) = } - {masks.shape = } - {masks.dtype = }')
	elif isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
		print(f'overlay_masks_pil() received: {type(image) = } - {image.shape = } - {type(masks) = } - {masks.shape = } - {masks.dtype = }')
	else:
		raise TypeError

	# Convert masks to numpy
	if hasattr(masks, 'cpu'):
		masks = masks.permute(0, 2, 3, 1).cpu().to(torch.uint8).numpy()
	elif hasattr(masks, 'numpy'):
		masks = masks.permute(0, 2, 3, 1).to(torch.uint8).numpy()
	
	if isinstance(image, PIL.Image.Image):
		print(f'overlay_masks_pil() now has: {type(image) = } - {image.size = } - {type(masks) = } - {masks.shape = } - {masks.dtype = }')
	elif isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
		print(f'overlay_masks_pil() now has: {type(image) = } - {image.shape = } - {type(masks) = } - {masks.shape = } - {masks.dtype = }')
	else:
		raise TypeError

	# Convert image to RGBA
	image_rgba = image.convert("RGBA")
	
	# Generate colors
	n_masks = len(masks)

	rgb_masks	= Image.new('RGBA', image.size, 0)
	gray_masks	= Image.new('RGBA', image.size, 0)
	
	# Composite each mask
	for i, mask in enumerate(masks):
		if debug:
			print(f'{i+1}/{n_masks} - {mask.shape = } - {mask.dtype = }')
		if mask.shape[2] == 1:
			mask = mask[:, :, 0]
			if debug:
				print(f'{i+1}/{n_masks} - {mask.shape = } - {mask.dtype = }')

		# Get color
		color		= hex_to_rgb_tuple(maximally_distinct_colormap[i])
		gray_color	= (i+1,i+1,i+1)
		
		# Create colored overlay
		overlay		= Image.new("RGBA", image.size, color	   + (0,))
		gray_overlay	= Image.new("RGBA", image.size, gray_color + (0,))

		# Convert mask to alpha channel
		mask_img	= Image.fromarray((mask * 255).astype(np.uint8))

		overlay.putalpha     (mask_img.point(lambda p: int(p * alpha)))
		gray_overlay.putalpha(mask_img.point(lambda p: int(p * alpha)))

		# Count pixels
		img_data	= zero_rgb_on_zero_alpha_inplace(np.array(gray_overlay))[:, :, :3]
		px_in_mask	= int((img_data == gray_color).all(axis=-1).sum())
		print(f'{i+1}/{n_masks} - {color = }\t-\t{px_in_mask = }')
		
		# Composite
		image_rgba	= Image.alpha_composite(image_rgba, overlay)
		rgb_masks	= Image.alpha_composite(rgb_masks,  overlay)
		gray_masks	= Image.alpha_composite(gray_masks, gray_overlay)
	
	image_rgba.convert("RGB")
	return image_rgba, rgb_masks, gray_masks.convert("L")

'''
def build_sam3_image_model(
	bpe_path=None,
	device="cuda" if torch.cuda.is_available() else "cpu",
	eval_mode=True,
	checkpoint_path=None,
	load_from_HF=True,
	enable_segmentation=True,
	enable_inst_interactivity=False,
	compile=False,
'''

if args.image != "":
	#################################### For Image ####################################
	# Load the model
	model = build_sam3_image_model(load_from_HF=False, checkpoint_path=args.model_path)
	processor = Sam3Processor(model)
	# Load an image
	image = Image.open(args.image)
	inference_state = processor.set_image(image)
	# Prompt the model with text
	output = processor.set_text_prompt(state=inference_state, prompt=args.text)
	
	# Get the masks, bounding boxes, and scores
	masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
	if args.debug:
		print(f'{type(masks) = } - {type(boxes) = } - {type(scores) = }')
		print(f'{len(masks) = } - {len(boxes) = } - {len(scores) = }')
		print(f'{masks.shape = } - {boxes.shape = } - {scores.shape = }')
	print(f"Found {len(boxes)} masks")
	print(f"Scores: {scores}")
	# Visualize the masks
	blended, rgb_masks, gray_masks = overlay_masks_pil(image, masks, alpha=0.5) # .show()
	# Save the overlay image with PIL
	blended.save(Path(args.output_dir) / "blended.png")
	rgb_masks.save(Path(args.output_dir) / "instances.png")
	gray_masks.save(Path(args.output_dir) / "instances-grayscale.png")
elif args.video != "":
	###################################################################################
	############################## WARNING: TOTALLY UNTESTED ##########################
	###################################################################################
	#################################### For Video ####################################
	
	video_predictor = build_sam3_video_predictor(load_from_HF=False, checkpoint_path=args.model_path)
	video_path = args.video # a JPEG folder or an MP4 video file
	
	# Start a session
	response = video_predictor.handle_request(request=dict(type="start_session", resource_path=video_path,))
	response = video_predictor.handle_request(
	request=dict(type="add_prompt", session_id=response["session_id"], frame_index=0, text=args.text, ))
	output = response["outputs"]
else:
	print("Please specify either --image or --video")
	
