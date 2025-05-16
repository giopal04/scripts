#!/usr/bin/env python3

import sys
import time

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import PIL
from PIL import Image, ImageDraw
import numpy as np

# Usage:
# 	coco-json-to-png-rgb-masks.py . --classes EJoint JTape Rust Crack ACrack --colors 0,0,255 0,0,255 255,128,0 0,255,0 0,255,0
#	(EJoint becomes blue, JTape becomes blue, Rust becomes orange, Crack becomes green, ACrack becomes green)
#
# Or:
#	coco-json-to-png-rgb-masks.py . --classes bridge§joint damaged§joint crack pothole --colors 0,0,255 255,128,0 0,255,0 255,0,0
#	(Use § special character as separator)
#
# Run the script, then you can double-check if all the masks have been generated...
#
# find . -type f -name '*.jpg' | sed 's:./train/::g' | sed 's:./valid/::g' | sed 's:./test/::g' | sed 's:.jpg$::g' | sort > /tmp/imgs.txt 
# find . -type f -name '*.png' | sed 's:./train/masks/::g' | sed 's:./valid/masks/::g' | sed 's:./test/masks/::g' | sed 's:\.png$::g' | sort > /tmp/masks.txt 

# Default colors (50 distinct colors from various sources)
DEFAULT_COLORS = [
	(128, 0, 0), (170, 110, 40), (128, 128, 0), (0, 128, 128), (0, 0, 128),
	(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
	(0, 255, 255), (128, 0, 128), (128, 128, 128), (139, 69, 19), (0, 100, 0),
	(184, 134, 11), (85, 107, 47), (72, 61, 139), (199, 21, 133), (25, 25, 112),
	(245, 222, 179), (220, 20, 60), (0, 139, 139), (47, 79, 79), (148, 0, 211),
	(255, 140, 0), (75, 0, 130), (240, 230, 140), (240, 128, 128), (230, 230, 250),
	(255, 240, 245), (127, 255, 0), (210, 105, 30), (255, 215, 0), (218, 112, 214),
	(50, 205, 50), (123, 104, 238), (0, 250, 154), (72, 209, 204), (255, 105, 180),
	(135, 206, 235), (221, 160, 221), (244, 164, 96), (250, 128, 114), (70, 130, 180),
	(255, 99, 71), (147, 112, 219), (143, 188, 143), (219, 112, 147), (238, 232, 170)
]

def find_split_dir(path: Path) -> Path:
	split_keywords = {'train', 'val', 'test', 'training', 'validation', 'valid', 'testing'}
	for parent in path.parents:
		dir_name = parent.name.lower()
		if any(kw in dir_name for kw in split_keywords):
			return parent
	return path.parent

def get_mask_path(image_path: Path, output_root: Path, debug=False) -> Path:
	split_dir = find_split_dir(image_path)
	relative_path = image_path.relative_to(output_root)
	if debug:
		print(f'get_mask_path() - {relative_path = }')
	#final_path = output_root / 'masks' / relative_path.with_suffix('.png')
	final_path = output_root / relative_path.parent / 'masks' / (relative_path.stem + '.png')
	if debug:
		print(f'get_mask_path() - {final_path = }')
	return final_path

def process_coco_file(json_path: Path, output_root: Path, class_colors: Dict[str, Tuple[int, int, int]], default_classes: List[str], debug=False, verbose_debug=False):
	colors = [class_colors.get(class_name, (0, 0, 0)) for class_name in default_classes]
	print(f'process_coco_file() received classes: {default_classes} - retrieved colors: {colors}')

	with open(json_path) as f:
		coco_data = json.load(f)

	if verbose_debug:
		print(f'COCO data is: {coco_data}')
	print(f'COCO data keys are: {coco_data.keys()}')

	categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
	images = {img['id']: img for img in coco_data.get('images', [])}

	annotations = coco_data.get('annotations', [])

	split = json_path.parent
	printable_split = split.name.upper()

	annotated_img_ids = []

	print(f'Found {len(images)} images, {len(categories)} categories and {len(coco_data.get("annotations", []))} annotations in {json_path}')
	print(f'Categories are: {categories}')

	if debug:
		print(f'Images: {images.keys()}')
	if verbose_debug:
		print(f'Annotations: {annotations}')
	'''
	if debug or True:
		print(f'Images[104]     : {images[104]}')
		print(f'Annotations[104]: {annotations[104]}')
	'''
	if verbose_debug:
		for idx,ann in enumerate(annotations):
			print(f'[{printable_split}] Processing annotation idx: {idx} - image id: {ann["image_id"]} - fn: {images.get(ann["image_id"])["file_name"]}')
	for idx,ann in enumerate(annotations):
		if 'image_id' not in ann or 'category_id' not in ann:
			print(f'[{printable_split}] Warning: skipping annotation without image_id or category_id: {ann}')
			continue
		img_id = ann["image_id"]

		'''
		if img_id == 103 and split.name == 'train':
			print(f'[{printable_split}] Processing image id: {ann["image_id"]} - {idx}')
			import pdb
			pdb.set_trace()
		'''
			
		image_info = images.get(ann['image_id'])
		if not image_info:
			print(f'[{printable_split}] Warning: skipping annotation without image info: {ann}')
			continue

		image_path = split / image_info['file_name']
		'''
		if (img_id >= 103 and img_id <= 105) and split.name == 'train':
			print(f'[{printable_split}] Processing annotation idx: {idx} - image id: {img_id} - file: {image_path}')
		'''
		if not image_path.exists():
			print(f'Warning: skipping missing image: {image_path}')
			continue

		class_name = categories.get(ann['category_id'], 'unknown')
		if class_name not in class_colors and class_name not in default_classes:
			print(f'[{printable_split}] Warning: skipping annotation with unknown class: {class_name} - {ann}')
			print(f'[{printable_split}] - categories are: {categories}')
			print(f'[{printable_split}] - default classes are: {default_classes}')
			print(f'[{printable_split}] - class colors are: {class_colors}')
			continue
		
		'''
		if debug or (img_id <= 105 and img_id >= 103) and split.name == 'train':
			print(f'[{printable_split}] Processing file: {image_path}')
		if '204_jpg.rf.13daae2309295adeb59e10259541430a' in str(image_path):
			import pdb
			pdb.set_trace()
		'''

		mask_path = get_mask_path(image_path, output_root)
		mask_path.parent.mkdir(parents=True, exist_ok=True)

		img  = Image.open(image_path)
		if img.size[0] >= 5000 and img.size[1] >= 5000:
			print(f'[{printable_split}] Processing XXL annotation idx: {idx} - image id: {img_id} - file: {image_path} with size: {img.size}')
		
		if not mask_path.exists():
			mask = Image.new('RGB', img.size, (0, 0, 0))
			#img.close()
		else:
			try:
				mask = Image.open(mask_path)
			except PIL.UnidentifiedImageError:
				print(f'[{printable_split}] Warning: rewriting corrupted mask: {mask_path}')
				#img  = Image.open(image_path)
				mask = Image.new('RGB', img.size, (0, 0, 0))
		img.close()
		
		draw  = ImageDraw.Draw(mask)
		color = class_colors.get(class_name, (0, 0, 0))
		if verbose_debug:
			print(f'[{printable_split}] - class_name is: {class_name} - color is: {color}')
		
		for seg in ann.get('segmentation', []):
			try:
				if isinstance(seg, list):
					if all(isinstance(s, (int, float)) for s in seg):
						polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
						draw.polygon(polygon, fill=color)
					else:
						for s in seg:
							polygon = [(s[i], s[i+1]) for i in range(0, len(s), 2)]
							draw.polygon(polygon, fill=color)
			except Exception as e:
				print(f"Error drawing polygon: {e}")
		
		mask.save(mask_path)
		mask.close()

		annotated_img_ids.append(img_id)

	for idx, img_id in enumerate(images):
		if img_id not in annotated_img_ids:
			print(f'[{printable_split}] Warning: missing annotation for image id: {img_id} - {images[img_id]}')

def process_single_image(image_path: Path, output_root: Path, class_colors: Dict[str, Tuple[int, int, int]]):
	json_path = image_path.with_suffix('.json')
	if not json_path.exists():
		print(f'Warning: skipping missing json: {json_path}')
		return

	with open(json_path) as f:
		ann_data = json.load(f)

	mask_path = get_mask_path(image_path, output_root)
	mask_path.parent.mkdir(parents=True, exist_ok=True)

	img = Image.open(image_path)
	mask = Image.new('RGB', img.size, (0, 0, 0))
	draw = ImageDraw.Draw(mask)
	
	for ann in ann_data.get('annotations', []):
		class_name = ann.get('category_name', 'unknown')
		color = class_colors.get(class_name, (0, 0, 0))
		
		for seg in ann.get('segmentation', []):
			try:
				polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
				draw.polygon(polygon, fill=color)
			except Exception as e:
				print(f"Error drawing polygon: {e}")
	
	mask.save(mask_path)
	img.close()
	mask.close()

def find_all_classes(input_path: Path) -> Dict[str, List[str]]:
	class_map = {}
	for json_path in input_path.rglob('*.json'):
		print(f'Reading {json_path}')
		with open(json_path) as f:
			try:
				data = json.load(f)
				if 'categories' in data:
					classes = [cat['name'] for cat in data['categories']]
					class_map[str(json_path)] = classes
				elif 'category_name' in data.get('annotations', [{}])[0]:
					classes = list(set(ann.get('category_name') for ann in data.get('annotations', [])))
					class_map[str(json_path)] = classes
			except Exception as e:
				print(f"Error reading {json_path}: {e}")
	return class_map

def countdown(msg, countdown=5):
	print(f'{msg}', end=' ', flush=True)
	for _ in range(countdown):
		time.sleep(1)
		print('.', end='', flush=True)
	print('\n', flush=True)

def main():
	parser = argparse.ArgumentParser(description='Convert COCO annotations to RGB masks')
	parser.add_argument('input_path', type=str, help='Root directory of the dataset')
	parser.add_argument('--output', type=str, help='Output directory (default: input_path)')
	parser.add_argument('--classes', nargs='+', help='List of classes to include (space-separated)')
	parser.add_argument('--colors', nargs='+', help='RGB colors for classes in RBG format (e.g., 255,0,0 is red)')
	parser.add_argument('--list-classes', action='store_true', help='List all available classes and exit')
	args = parser.parse_args()

	input_path = Path(args.input_path)
	output_root = Path(args.output) if args.output else input_path

	if args.list_classes:
		class_map = find_all_classes(input_path)
		for file_path, classes in class_map.items():
			print(f"{file_path}:")
			for cls in classes:
				print(f"  - {cls}")
		return

	class_colors = {}
	if args.classes:
		if args.colors:
			colors = [tuple(map(int, c.split(','))) for c in args.colors]
		else:
			colors = DEFAULT_COLORS
		
		for i, cls in enumerate(args.classes):
			class_colors[str(cls).replace('§', ' ')] = colors[i % len(colors)]
	else:
		class_colors = {cls: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] 
					  for i, cls in enumerate(find_all_classes(input_path).values())}

	'''
	def get_prio(x):
		return 0 if 'tr' in str(x) else 1 if 'val' in str(x) else 2 if 'test' in str(x) else 3
	'''

	# Process COCO-style annotations
	print(f'Looking for global COCO-style annotations in {input_path}')
	#for json_path in sorted(input_path.rglob('*.json'), key=lambda x: get_prio(x)):
	#for json_path in sorted(input_path.rglob('*.json'), key=lambda x: (return 0 if 'tr' in str(x) else 1 if 'val' in str(x) else 2 if 'test' in str(x) else 3, x)):
	'''
	for json_path in sorted(input_path.rglob('*.json'), key=lambda x: 0 if 'tr' in str(x) else 1 if 'val' in str(x) else 2 if 'test' in str(x) else 3):
		print(f'Looking for global COCO-style annotations in {json_path}')
	sys.exit(0)
	'''

	json_glob = sorted(input_path.rglob('*.json'), key=lambda x: 0 if 'tr' in str(x) else 1 if 'val' in str(x) else 2 if 'test' in str(x) else 3)
	json_lst  = [str(x) for x in list(json_glob)]
	if not json_lst or len(json_lst) == 0:
		print(f'No json files found in {input_path}')
		sys.exit(1)

	classes = [str(cl).replace('§', ' ') for cl in args.classes]
	countdown(msg=f'Found the following json files: {" - ".join(json_lst)} and the following classes: {" - ".join(classes)}, continue?', countdown=5)

	#for json_path in input_path.rglob('*.json'):
	#for json_path in sorted(input_path.rglob('*.json'), key=lambda x: 0 if 'tr' in str(x) else 1 if 'val' in str(x) else 2 if 'test' in str(x) else 3):
	if len(json_lst) == 1:
		for json_path in json_glob:
			print(f'Looking for global COCO-style annotations in {json_path}')
			if 'annotations' in json_path.name.lower() and json_path.stat().st_size > 0:
				process_coco_file(json_path, output_root, class_colors, classes)
	else:
		# Process per-image annotations
		print(f'Looking for per-image COCO-style annotations in {input_path}')
		for img_path in input_path.rglob('*.*'):
			if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.tif']:
				process_single_image(img_path, output_root, class_colors)

	print(f'Now you can run:\nfor i in train valid test ; do mkdir $i/images ; mv $i/*.jpg $i/images ; for j in images masks ; do ll $i/$j/*.jpg $i/$j/*.png 2>/dev/null | nl | tail -1 ; done ; done')

if __name__ == '__main__':
	main()
