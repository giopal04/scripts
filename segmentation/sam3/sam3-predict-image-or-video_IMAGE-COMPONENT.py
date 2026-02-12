#!/usr/bin/env python3

import random
import argparse

import numpy as np

import PIL
from PIL import Image
from pathlib import Path

import torch
#torch.backends.cudnn.conv.fp32_precision = 'tf32'

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from sam3.model_builder import build_sam3_video_predictor

from sam3.agent.viz import visualize

import matplotlib
import matplotlib.pyplot as plt

from scripts.classes.maximally_distinct_colormap import hex_to_rgb_tuple, maximally_distinct_colormap

import json

import zlib
import base64

import colorsys

parser = argparse.ArgumentParser()
parser.add_argument("--model-path",	type=str, default="/mnt/raid1/repos/sam3/models/sam3.pt",	help="Model checkpoint to load")
parser.add_argument("--image-dir",	type=str, default="",						help="Director of images to segment")
parser.add_argument("--video",		type=str, default="",						help="Video to segment")
parser.add_argument('--prompts', 	type=str, default=["person"],	nargs='+',	help='Promts for the model')# ,
parser.add_argument("--output-dir",	type=str, default="/tmp",					help="Output directory")
parser.add_argument("--single-instances", action="store_true", help="For each image and each class, saves a mask containing all instances of a single class")
parser.add_argument("--save-blended", action="store_true", help="Save the instance segmentation mask blended with the image")
parser.add_argument("--no-semantic", action="store_false", help="Do not save semantic segmentation mask")
parser.add_argument("--no-instance", action="store_false", help="Do not save instance segmentation mask")
parser.add_argument("--debug",		action='store_true',						help="Debug mode")
args = parser.parse_args()

# Run with:
# ---------
# sam3-predict-image-or-video.py --image damaged-statues/damaged-statues-Image_5_00001_.png --text 'statue OR monument OR column OR brick wall OR statue head OR capital column'

# Alternative version that modifies in-place (more memory efficient)
#NOT USED ANYMORE
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

#NOT USED ANYMORE
def overlay_masks_pil(image, masks, alpha=0.5, debug=False):  
	'''
	Overlay masks on image using PIL for better blending.
	Args:
		image: PIL Image
		masks: Tensor/array of masks [N, H, W]
		alpha: Transparency of masks
	'''
	if debug: #NOTE: it was not here, is it necessary the code below?
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
	
	if debug: #NOTE: it was not here, is it necessary the code below?
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
		if debug:
			print(f'{i+1}/{n_masks} - {color = }\t-\t{px_in_mask = }')
		
		# Composite
		image_rgba	= Image.alpha_composite(image_rgba, overlay)
		rgb_masks	= Image.alpha_composite(rgb_masks,  overlay)
		gray_masks	= Image.alpha_composite(gray_masks, gray_overlay)
	
	image_rgba.convert("RGB")
	return image_rgba, rgb_masks, gray_masks.convert("L")

def preprocess_image(image, processor, prompt2id):
    all_instances = []

    inference_state = processor.set_image(image)
    for i, prompt in enumerate(prompt2id.keys()):
        class_id = i + 1
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks, scores, boxes = output['masks'], output['scores'], output['boxes']

        for j in range(len(scores)):
            all_instances.append({
                'class_id': class_id,
                'prompt': prompt,
                'mask': masks[j,0],
                'score': scores[j],
                'boxes': boxes[j],
                'instance_number': j
            })

	#This check to handle the possibility that SAM does not find anything	
    if not all_instances:
        W, H = image.size
        winner_indices = torch.zeros((H, W), dtype=torch.int64)
        max_scores = torch.zeros((H, W), dtype=torch.float32)
        return winner_indices, max_scores, []
        
    stacked_masks = torch.stack([inst['mask'] for inst in all_instances]) # [Total_Inst, H, W]
    stacked_scores = torch.stack([inst['score'] for inst in all_instances]).view(-1, 1, 1)
    
    weighted_masks = stacked_masks.float() * stacked_scores
    max_scores, winner_indices = torch.max(weighted_masks, dim=0) # [H, W]

    return winner_indices, max_scores, all_instances

def generate_masks(winner_indxs, max_scores, instances, prompt2id, id2color):
    H, W = winner_indxs.shape

    winner_indxs, max_scores = winner_indxs.cpu().numpy(), max_scores.cpu().numpy()

    gray_mask = np.zeros((H,W), dtype=np.uint8)
    rgb_mask = np.zeros((H,W,3), dtype=np.uint8)
    instance_images = {p: Image.new("RGBA", (W,H), (0,0,0,0)) for p in prompt2id.keys()}
    instance_segmentation = Image.new("RGBA", (W,H), (0,0,0,0))

    for indx, inst in enumerate(instances):
        winning_pixels = (winner_indxs == indx) & (max_scores > 0)

        if not np.any(winning_pixels):
            continue

        gray_mask[winning_pixels] = inst['class_id']

        class_color = id2color[str(inst['class_id'])][5]
        rgb_mask[winning_pixels] = class_color

        inst_color = id2color[str(inst['class_id'])][(inst['instance_number'] * 3) % 10]
        inst_mask_pil = Image.fromarray((winning_pixels * 255).astype(np.uint8))    
        inst_overlay = Image.new("RGBA", (W, H), inst_color)
        inst_overlay.putalpha(inst_mask_pil.point(lambda p: int(p * 1)))

        instance_images[inst['prompt']] = Image.alpha_composite(instance_images[inst['prompt']], inst_overlay)
        instance_segmentation = Image.alpha_composite(instance_segmentation, inst_overlay)

    rgb_mask_img = Image.fromarray(rgb_mask)
    gray_mask_img = Image.fromarray(gray_mask)

    overlay_mask = np.where(gray_mask>0, 255, 0).astype(np.uint8)

    return rgb_mask_img, gray_mask_img, instance_images, instance_segmentation, overlay_mask

def overlay_img(img1, img2, mask, alpha=0.5):
    img1 = img1.convert('RGBA')
    img2 = img2.convert('RGBA')

    mask_img = Image.fromarray(mask)
    img2.putalpha(mask_img.point(lambda p: int(p * alpha)))

    return Image.alpha_composite(img1, img2)

def loading_processor(checkpoint_path):
	model = build_sam3_image_model(load_from_HF=False, checkpoint_path=checkpoint_path)

	return Sam3Processor(model)

def encode_mask(mask):
	mask = mask.cpu().numpy()	
	
	data_type = mask.dtype
	shape = mask.shape
	compressed_mask = zlib.compress(mask.tobytes())
	
	return (base64.b64encode(compressed_mask).decode('utf-8'), shape, data_type)

#Here just for future use 
def decode_mask(encoded_string, shape, dtype):
	compressed_data = base64.b64decode(encoded_string)
	raw_bytes = zlib.decompress(compressed_data)
	
	return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

def get_color_scheme(number_of_classes):
    H_step = 1 / number_of_classes

    color_codes = {}
    for j in range(number_of_classes):
        h = j * H_step
        color_codes[str(j+1)] = []

        for i in range(10):
            l = 0.25 + i * 0.05
            s = 0.8

            r, g, b = colorsys.hls_to_rgb(h, l, s)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)

            block = np.zeros((3,50,50))
            block[0,:,:] = r
            block[1,:,:] = g
            block[2,:,:] = b

            color_codes[str(j+1)].append((r, g, b))
    
    return color_codes
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

if args.image_dir != "":
	#################################### For Image ####################################
	# Load the model
	print(f'Loading model...')
	processor = loading_processor(args.model_path)
	# Loading images
	print(f'Retriving images...')
	input_dir = Path(args.image_dir)
	if not input_dir.is_dir():
		raise FileNotFoundError(f'{input_dir} is not a directory')
	images_paths = list(input_dir.iterdir())

	# Preparing output dir
	out_dir = Path(args.output_dir)
	if not out_dir.exists():
		out_dir.mkdir()

	if args.no_semantic:
		print('Saving semantic segmentation masks.')
		seg_folder = out_dir/'segmentation'
		if not seg_folder.exists():
			seg_folder.mkdir()

	if args.no_instance:
		print('Saving instance segmentation masks')
		inst_folder = out_dir/'instance'
		if not inst_folder.exists():
			inst_folder.mkdir()
	
	if args.save_blended:
		print('Saving blended image.')
		blend_folder = out_dir/'blended'
		if not blend_folder.exists():
			blend_folder.mkdir()

	save_file = out_dir/'test_save.json'

	# Preapering prompt
	args.prompts.sort()
	prompt2id = {p: i+1 for i, p in enumerate(args.prompts)}
	id2colors = get_color_scheme(len(prompt2id))

	data = {
		'metadata': {},
		'images': {}
	}

	data['metadata']['prompt2id'] = prompt2id
	data['metadata']['id2colors'] = id2colors

	for i, path in enumerate(images_paths):
		print(f'{i+1}/{len(images_paths)}. Processing image {path.name}')
		data['images'][path.name] = []

		image = Image.open(path).convert('RGB')
		
		indices, scores, all_instances = preprocess_image(image, processor, prompt2id)
		rgb_segmentation, gray_segmentation, prompt_instances, instance_seg, overlay_mask = generate_masks(indices, scores, all_instances, prompt2id=prompt2id, id2color=id2colors)

		blended_image = overlay_img(image, instance_seg, overlay_mask)

		for inst in all_instances:
			encoded_mask, m_shape, m_dtype = encode_mask(inst['mask'])
			instance_data = {
				'prompt': inst['prompt'],
				'class_id': inst['class_id'],
				'mask': encoded_mask,
				'shape': list(m_shape),
				'dtype': str(m_dtype),
				'bbox': [float(x) for x in inst['boxes']],
				'score': float(inst['score'])
			}
			data['images'][path.name].append(instance_data)

		#Saving all
		if args.no_semantic:
			rgb_segmentation.save(seg_folder/f'{path.stem}-rgb.png')
			gray_segmentation.save(seg_folder/f'{path.stem}-gray.png')

		if args.no_instance:
			instance_seg.save(inst_folder/f'{path.stem}-inst.png')

			if args.save_blended:
				blended_image.save(blend_folder/f'{path.stem}-inst_blended.png')

			if args.single_instances:
				for prompt, inst in prompt_instances.items():
					inst.save(inst_folder/f'{path.stem}-{prompt}.png')
		


	save_file = out_dir/'test_save.json'
	with open(save_file, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

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
	
