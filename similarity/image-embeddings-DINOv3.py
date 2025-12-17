#!/usr/bin/env python3

'''
Image Embedding Extractor using DINOv3/DINOv2
Saves embeddings for efficient cosine similarity search later.
'''

# Run with:
# image-embeddings-DINOv3.py --model dinov3-small /tmp/damaged-statues

import sys
import pickle
import argparse
import numpy as np

import cv2
import torch

from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image
from torchvision import transforms

# Try to import DINOv2, with graceful fallback
try:
	import torchvision.transforms as T
	from transformers import AutoImageProcessor, AutoModel
	DINOV2_AVAILABLE = True
except ImportError:
	DINOV2_AVAILABLE = False
	print('Warning: DINOv2 dependencies not found. Install with:')
	print('pip install torch torchvision transformers')

try:
	from transformers import pipeline
	from transformers.image_utils import load_image
	DINOV3_AVAILABLE = True
except ImportError:
	DINOV3_AVAILABLE = False
	print('Warning: DINOv3 dependencies not found. Install with:')
	print('pip install torch torchvision transformers')
	



# Map model names to HuggingFace model IDs
# Sizes can be found here: https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-models
# Benchmark table can be found here: https://arxiv.org/html/2508.10104v1#S7.T14
model_map = {
	'dinov2-small'		:	'facebook/dinov2-small',
	'dinov2-base'		:	'facebook/dinov2-base',
	'dinov2-large'		:	'facebook/dinov2-large',

	'dinov3-small'		:	'facebook/dinov3-vits16-pretrain-lvd1689m',		# 21m
	'dinov3-small+'		:	'facebook/dinov3-vits16plus-pretrain-lvd1689m',		# 29m
	'dinov3-big'		:	'facebook/dinov3-vitb16-pretrain-lvd1689m',		# 86m
	'dinov3-large'		:	'facebook/dinov3-vitl16-pretrain-lvd1689m',		# 300m
	'dinov3-huge16+'	:	'facebook/dinov3-vith16plus-pretrain-lvd1689m',		# 840m
	'dinov3-7b'		:	'facebook/dinov3-vit7b16-pretrain-lvd1689m',		# this one is the original one from which all the others have been distilled
	}
ext_list = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


def setup_dinov2_model(model_name: str = 'facebook/dinov2-base'):
	'''Load DINOv2 model and processor'''
	if not DINOV2_AVAILABLE:
		raise ImportError('DINOv2 dependencies not installed')
	
	print(f'Loading {model_name} model...')
	processor	= AutoImageProcessor.from_pretrained(model_name)
	model		= AutoModel.from_pretrained(model_name)
	
	# Use GPU if available
	device		= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model		= model.to(device)
	model.eval()
	
	# Define preprocessing transforms
	transform	= transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	
	return model, processor, transform, device

def setup_dinov3_model(model_name: str = 'facebook/dinov3-vits16-pretrain-lvd1689m'):
	'''Load DINOv3 model and processor'''
	if not DINOV3_AVAILABLE:
		raise ImportError('DINOv3 dependencies not installed')

	print(f'Loading {model_name} model...')

	feature_extractor = pipeline(
	    model = model_name,
	    task  = 'image-feature-extraction', 
	)
	return feature_extractor

def extract_embeddings_dinov3(feature_extractor, image_paths: List[Path], debug=False) -> Tuple[List[str], np.ndarray]:
	'''
	url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg'
	image = load_image(url)
	
	features = feature_extractor(image)
	
	# Access the underlying model
	model = feature_extractor.model

	return features
	'''
	'''Extract embeddings using DINOv2'''
	filenames = []
	all_embeddings = []
	
	for img_path in image_paths:
		try:
			print(f'Processing: {img_path}')
			# Load and preprocess image
			image = load_image(str(img_path))

			features = feature_extractor(image)
			if debug:
				print(f'DINOv3 features type      : {type(features)}')
				print(f'DINOv3 features len       : {len(features)}')
				print(f'DINOv3 features[0] type   : {type(features[0])}')
				print(f'DINOv3 features[0] len    : {len(features[0])}')
				print(f'DINOv3 features[0][0] type: {type(features[0][0])}')
				print(f'DINOv3 features[0][0] len : {len(features[0][0])}')
				print(f'DINOv3 features shape  : {features.shape}')
			
			all_embeddings.append(np.array(features).squeeze()) # embedding.squeeze())
			filenames.append(str(img_path))
			
			print(f'Processed: {img_path.name}')
			
		except Exception as e:
			print(f'Error processing {img_path}: {e}')
	
	return filenames, np.array(all_embeddings)

def extract_embeddings_dinov2(
	model, 
	transform, 
	image_paths: List[Path]
) -> Tuple[List[str], np.ndarray]:
	'''Extract embeddings using DINOv2'''
	filenames = []
	all_embeddings = []
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	with torch.no_grad():
		for img_path in image_paths:
			try:
				print(f'Processing: {img_path}')
				# Load and preprocess image
				img = Image.open(img_path).convert('RGB')
				img_tensor = transform(img).unsqueeze(0).to(device)
				
				# Extract features
				outputs = model(img_tensor)
				# Use the [CLS] token representation
				embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
				
				# Normalize the embedding
				embedding = embedding / np.linalg.norm(embedding)
				
				all_embeddings.append(embedding.squeeze())
				filenames.append(str(img_path))
				
				print(f'Processed: {img_path.name}')
				
			except Exception as e:
				print(f'Error processing {img_path}: {e}')
	
	return filenames, np.array(all_embeddings)


def extract_embeddings_opencv(
	image_paths: List[Path]
) -> Tuple[List[str], np.ndarray]:
	'''
	Fallback method using OpenCV features.
	This is less powerful than DINOv2 but doesn't require extra dependencies.
	'''
	print('Using OpenCV ORB features (less powerful than DINOv2)')
	filenames = []
	all_descriptors = []
	
	# Initialize ORB detector
	orb = cv2.ORB_create(nfeatures=500)
	
	for img_path in image_paths:
		try:
			# Read image
			print(f'Processing: {img_path}')
			img = cv2.imread(str(img_path))
			if img is None:
				print(f'Could not read: {img_path}')
				continue
			
			# Convert to grayscale
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			# Detect and compute descriptors
			_, descriptors = orb.detectAndCompute(gray, None)
			
			if descriptors is not None:
				# Average pooling of descriptors
				avg_descriptor = np.mean(descriptors, axis=0)
				# Normalize
				norm = np.linalg.norm(avg_descriptor)
				if norm > 0:
					avg_descriptor = avg_descriptor / norm
				
				all_descriptors.append(avg_descriptor)
				filenames.append(str(img_path))
				print(f'Processed: {img_path.name}')
			else:
				print(f'No features found in: {img_path}')
				
		except Exception as e:
			print(f'Error processing {img_path}: {e}')
	
	return filenames, np.array(all_descriptors)


def save_embeddings(
	output_path: Path,
	filenames: List[str],
	embeddings: np.ndarray,
	metadata: Dict = None
):
	'''Save embeddings to a pickle file'''
	data = {
		'filenames': filenames,
		'embeddings': embeddings,
		'metadata': metadata or {}
	}
	
	with open(output_path, 'wb') as f:
		pickle.dump(data, f)
	
	print(f'\nSaved {len(filenames)} embeddings to: {output_path}')
	print(f'Embedding shape: {embeddings.shape}')
	print(f'Embedding dimension: {embeddings.shape[1] if len(embeddings.shape) > 1 else 1}')

def main():
	parser = argparse.ArgumentParser(
		description='Extract image embeddings for similarity search'
	)
	parser.add_argument('image_dir',	type=str, help='Directory containing images')
	parser.add_argument('--outfn', '-o',	type=str, default='image-embeddings.pkl', help='Output file for embeddings (default: image-embeddings.pkl in the same directory as source files)')
	#parser.add_argument('--model',		type=str, choices=['dinov2-base', 'dinov2-small', 'dinov2-large', 'opencv'], default='dinov2-base', help='Model to use for feature extraction')
	parser.add_argument('--model',		type=str, choices=list(model_map.keys()) + ['opencv'], default='dinov3-small', help='Model to use for feature extraction')
	parser.add_argument('--extensions',	type=str, nargs='+', default=ext_list, help='Image extensions to process')
	
	args = parser.parse_args()
	
	# Validate inputs
	image_dir = Path(args.image_dir)
	if not image_dir.exists() or not image_dir.is_dir():
		print(f'Error: Directory {image_dir} not found')
		sys.exit(1)
	
	# Find all image files
	image_paths = []
	for ext in args.extensions:
		image_paths.extend(image_dir.rglob(f'*{ext}'))
		image_paths.extend(image_dir.rglob(f'*{ext.upper()}'))
		#image_paths.extend(image_dir.glob(f'*{ext}'))
		#image_paths.extend(image_dir.glob(f'*{ext.upper()}'))
	
	if not image_paths:
		print(f'No images found in {image_dir} with extensions {args.extensions}')
		sys.exit(1)
	
	print(f'Found {len(image_paths)} images to process')
	
	# Extract embeddings based on model choice
	if args.model.startswith('dino'):
		if not DINOV2_AVAILABLE and not DINOV3_AVAILABLE:
			print('DINOv2 not available. Install with: pip install transformers')
			print('Falling back to OpenCV method...')
			filenames, embeddings = extract_embeddings_opencv(image_paths)
		else:
			model_id = model_map.get(args.model, 'facebook/dinov3-vits16-pretrain-lvd1689m')
			
			try:
				if args.model.startswith('dinov2'):
					model, processor, transform	= setup_dinov2_model(model_id)
					filenames, embeddings		= extract_embeddings_dinov2(model, transform, image_paths)
				elif args.model.startswith('dinov3'):
					feature_extractor_pipeline	= setup_dinov3_model(model_name=model_id)
					filenames, embeddings		= extract_embeddings_dinov3(feature_extractor_pipeline, image_paths)
					'''
					model, processor, transform, device = setup_dinov2_model(model_id)
					filenames, embeddings = extract_embeddings_dinov2(
						model, transform, device, image_paths
					)
					'''
			except Exception as e:
				print(f'Error with DINOv3/DINOv2: {e}')
				print('Falling back to OpenCV method...')
				filenames, embeddings = extract_embeddings_opencv(image_paths)
	else:
		# Use OpenCV method
		filenames, embeddings = extract_embeddings_opencv(image_paths)
	
	if len(filenames) == 0:
		print('No embeddings were extracted. Exiting.')
		sys.exit(1)
	
	# Prepare metadata
	metadata = {
		'model_used': args.model,
		'image_dir': str(image_dir),
		'total_images': len(image_paths),
		'processed_images': len(filenames),
	}
	
	# Save embeddings
	output_path = Path(args.outfn)
	save_embeddings(output_path, filenames, embeddings, metadata)
	# then run something like:
	# cosine_similarity(np.expand_dims(data['embeddings'][7].flatten(),0), np.expand_dims(data['embeddings'][8].flatten(),0))


if __name__ == '__main__':
	main()
	'''
	load a file named asdf.pkl with pickle
	with open('asdf.pkl', 'rb') as f:
		data = pickle.load(f)
		print(data)
	'''
