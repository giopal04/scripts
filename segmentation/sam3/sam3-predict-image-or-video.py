#!/usr/bin/env python3

import sys
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

# Get the absolute path of the directory 3 levels up (the repo root)
# .parent = sam3/ | .parent.parent = segmentation/ | .parent.parent.parent = root/
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

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


def abs_to_rel_coords(coords, img_width, img_height, coord_type="point"):
	"""Convert absolute coordinates to relative coordinates (0-1 range)

	Args:
		coords: List of coordinates
		coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
	"""
	if coord_type == "point":
		return [[x / img_width, y / img_height] for x, y in coords]
	elif coord_type == "box":
		print(f'{type(coords) = }')
		print(f'{coords.shape = }')
		print(f'abs_to_rel_coords() {coords = } - {img_width = } - {img_height = } - {coord_type = }')
		if len(coords) == 0:
			return []
		if len(coords.shape) == 1:
			x, y, w, h = coords
			return [x / img_width, y / img_height, w / img_width, h / img_height]					# just one bbox
		elif len(coords.shape) == 2:
			return [[x / img_width, y / img_height, w / img_width, h / img_height] for x, y, w, h in coords]	# list of bboxes
		else:
			raise ValueError(f"Got coords with unknown cardinality: {coords} - {coords.shape}")
	else:
		raise ValueError(f"Unknown coord_type: {coord_type}")

# Now write a function to convert relative coordinates to absolute coordinates
def rel_to_abs_coords(coords, img_width, img_height, coord_type="point"):
	"""Convert relative coordinates to absolute coordinates

	Args:
		coords: List of coordinates
		coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
	"""
	if coord_type == "point":
		return [[x * img_width, y * img_height] for x, y in coords]
	elif coord_type == "box":
		if len(coords) == 0:
			return []
		if len(coords.shape) == 1:
			x, y, w, h = coords
			return [int(x * img_width), int(y * img_height), int(w * img_width), int(h * img_height)]				# just one bbox
		elif len(coords.shape) == 2:
			return [[int(x * img_width), int(y * img_height), int(w * img_width), int(h * img_height)] for x, y, w, h in coords]	# list of bboxes
		else:
			raise ValueError(f"Got coords with unknown cardinality: {coords} - {coords.shape}")
	else:
		raise ValueError(f"Unknown coord_type: {coord_type}")

def propagate_in_video(predictor, session_id):
	# we will just propagate from frame 0 to the end of the video
	outputs_per_frame = {}
	for response in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id,)):
		outputs_per_frame[response["frame_index"]] = response["outputs"]
		# for sure this is the most inefficient and VRAM-hungry way to do this

	return outputs_per_frame


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
	
	#video_predictor = build_sam3_video_predictor(load_from_HF=False, checkpoint_path=args.model_path)
	if False:
		# this crashes with:
		# ValueError: Default process group has not been initialized, please make sure to call init_process_group.
		gpus_to_use     = range(torch.cuda.device_count())
	else:
		gpus_to_use     = [1]
	print(f"Using GPUs   : {gpus_to_use}")
	video_predictor = build_sam3_video_predictor(checkpoint_path=args.model_path, gpus_to_use=gpus_to_use)
	'''
	# https://github.com/facebookresearch/sam3/issues/169
	video_predictor = build_sam3_video_predictor(
		checkpoint_path=args.model_path,
		gpus_to_use=gpus_to_use, 
		offload_video_to_cpu=True,
		offload_state_to_cpu=True,
		async_loading_frames=True,
		forward_backbone_per_frame_for_eval=True,
		trim_past_non_cond_mem_for_eval=True,
		offload_output_to_cpu_for_eval=True,
	)
	'''

	video_path      = args.video # a JPEG folder or an MP4 video file
	print(f"Loading video: {video_path}")
	
	# Start a session
	response	= video_predictor.handle_request(request=dict(type="start_session", resource_path=video_path,))
	session_id	= response["session_id"]

	response	= video_predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=args.text, ))
	output		= response["outputs"]
	print(f"Found {len(output)} masks")
	print(f'{output = }')
	#if args.debug:
	#	print(f'{type(output[0]["masks"]) = } - {type(output[0]["boxes"]) = } - {type(output[0]["scores"]) = }')
	scores		= output["out_probs"]			# one per obj
	bboxes		= output["out_boxes_xywh"]		# one per obj
	masks		= output["out_binary_masks"]		# one per obj (e.g. masks.shape = (1, 1280, 720) if one single object)
	frame_stats	= output["frame_stats"]			# one per frame
	n_objs_tracked	= frame_stats['num_obj_tracked']
	n_objs_dropped	= frame_stats['num_obj_dropped']
	#print(f'{output["out_boxes_xywh"].shape = }')
	#print(f'{output["out_binary_masks"].shape = }')
	#print(f'{output["out_boxes_xywh"].shape = }')
	#print(f'{masks.shape = }')
	print(f'Starting to track {n_objs_tracked} objects (dropped: {n_objs_dropped}) with initial scores of: {scores}')
	if args.debug or True:
		for bbox, mask in zip(bboxes, masks):
			# bbox is an array with 4 elements
			print(f'{type(bbox)}')
			print(f'bbox: {bbox}')
			print(f'mask.shape: {mask.shape}')
			#bbox = abs_to_rel_coords(bbox, mask.shape[1], mask.shape[0], coord_type="box")	# masks.shape = (1280, 720) here because we're iterating over objects
			bbox = rel_to_abs_coords(bbox, mask.shape[1], mask.shape[0], coord_type="box")	# masks.shape = (1280, 720) here because we're iterating over objects
			print(f'Initial bbox area: {bbox[2] * bbox[3]} @ [x, y]: [{bbox[0]}, {bbox[1]}]')
			print(f'Initial mask size: {mask.shape} with total area: {mask.shape[0] * mask.shape[1]} (non-black px: {int(mask.sum())})')

	# this runs through the whole video
	outputs_per_frame = propagate_in_video(video_predictor, session_id=session_id)

	for frame_idx in range(0, len(outputs_per_frame)):
		print(f'Visualizing frame {frame_idx}')
		this_frame_preds= outputs_per_frame[frame_idx]
		print(f'{this_frame_preds = }')
		obj_ids		= this_frame_preds["out_obj_ids"]			# one per obj
		frame_stats	= this_frame_preds["frame_stats"]	# one per frame
		n_objs_tracked	= frame_stats['num_obj_tracked']
		n_objs_dropped	= frame_stats['num_obj_dropped']
		print(f'Frame [{frame_idx}] is tracking {n_objs_tracked} objects (dropped: {n_objs_dropped})')
		for obj_id in obj_ids:
			print(f'Processing obj_id: {obj_id}')
			score		= this_frame_preds["out_probs"]		# one per obj
			bbox		= this_frame_preds["out_boxes_xywh"]	# one per obj
			mask		= this_frame_preds["out_binary_masks"]	# one per obj (e.g. masks.shape = (1, 1280, 720) if one single object)
		sys.exit(0)
		#visualize_formatted_frame_output(frame_idx, video_frames_for_vis, outputs_list=[outputs_per_frame], titles=["SAM 3 Dense Tracking outputs"], figsize=(6, 4),)


	'''
	for i, out in enumerate(output):
		print(f'{out = }')
		masks  = out["out_binary_masks"]
		boxes  = out["out_boxes_xywh"]
		print(f"Found {len(boxes)} masks")
		print(f"Scores: {scores}")
		# Visualize the masks
		blended, rgb_masks, gray_masks = overlay_masks_pil(image, masks, alpha=0.5) # .show()
		# Save the overlay image with PIL
		blended.save(Path(args.output_dir) / f"blended-{i}.png")
		rgb_masks.save(Path(args.output_dir) / f"instances-{i}.png")
		gray_masks.save(Path(args.output_dir) / f"instances-grayscale-{i}.png")
	'''
else:
	print("Please specify either --image or --video")
	
