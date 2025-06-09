#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

# Color map (class_id: [B, G, R])
color_map = {
        0:   [  0,   0,   0], # Class 0 - background    (black)
        50:  [255,   0,   0], # Class 1 - grass         (blue)
        150: [  0, 255,   0], # Class 2 - trees         (green)
        200: [  0,   0, 255], # Class 3 - cube          (red)
         38: [  0,   0, 255], # Class 4 - crack         (red)
        255: [  0,   0, 255], # Class 4 - crack         (red)
         81: [  0,   0, 255], # Class 4 - crack         (red)
        # Add more classes as needed
}

def load_font(fn='Ubuntu-R.ttf', size=48, index=0, encoding="unic", use_pilfont=True):	# or lat1? https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
	if use_pilfont:
		ft = ImageFont.truetype(fn, size=size, index=index, encoding=encoding)
		print(f'Using PIL font: {fn}')
	else:
		# for some weird reason, try again to use OpenCV freetype module (usually unavailable for license reasons)
		ft = cv2.freetype.createFreeType2()
		ft.loadFontData(fontFileName=fn, id=0)
	return ft

def put_text(img, ft, txt, pt, c=(255, 255, 255, 0), sz="medium", thickness=2.0, use_pilfont=True):
	if use_pilfont:
		real_ft = ft[sz]
		# fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
		img_pil	= Image.fromarray(img)
		draw	= ImageDraw.Draw(img_pil)
		draw.text(pt, txt, font=real_ft, fill=c, font_size=sz, stroke_width=int(thickness), stroke_fill=c)
		img	= np.array(img_pil)
		return img
		# ImageDraw.text(xy, text, fill=None, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, stroke_fill=None, embedded_color=False, font_size=None)[source]
	else:
		ft.putText(img=img,
			   text=txt,
			   org=pt,
			   fontHeight=sz,
			   color=c,
			   thickness=thickness,
			   line_type=cv2.LINE_AA,
			   bottomLeftOrigin=False)
		return img

def resize_and_pad(img, target_width, target_height, resize=True, debug=False):
	"""Resize image while maintaining aspect ratio and pad to target dimensions."""
	if img is None:
		return np.zeros((target_height, target_width, 3), 0, 0)
	
	h, w = img.shape[:2]
	if h == 0 or w == 0:
		return np.zeros((target_height, target_width, 3), 0, 0)
	
	scale = min(target_width / w, target_height / h)
	if resize:
		new_w = int(w * scale)
		new_h = int(h * scale)
	else:
		new_w = w
		new_h = h

	pad_top    = (target_height - new_h) // 2
	pad_bottom =  target_height - new_h - pad_top
	pad_left   = (target_width  - new_w) // 2
	pad_right  =  target_width  - new_w - pad_left
	if debug:
		print(f'{new_w = }, {new_h = }, {pad_top = }, {pad_bottom = }, {pad_left = }, {pad_right = }')
	
	if resize:
		resized = cv2.resize(img, (new_w, new_h))
	else:
		resized = img
	
	padded = cv2.copyMakeBorder(
		resized, pad_top, pad_bottom, pad_left, pad_right, 
		cv2.BORDER_CONSTANT, value=0
	)
	return padded, pad_left, pad_top

def create_overlay(image, mask, alpha=0.5, debug=False):
	"""Create overlay of image and mask with transparency."""
	if image is None or mask is None:
		return None
	
	# Convert mask to color (red)
	if mask.ndim == 2:
		mask_color = np.zeros_like(image)
		mask_color[..., 2] = mask  # Set red channel
	else:
		mask_color = mask.copy()

	# Create an empty colorized mask
	colorized_mask = np.zeros_like(image)

	if debug:
		print(f'color_map = {color_map}')
		print(f'mask.shape = {mask.shape}')
		print(f'mask: {mask[mask > 0]}')
	print(f'non-null uniques in mask: {np.unique(mask[mask > 0])}')

	# Apply the color map to the mask
	for class_id, color in color_map.items():
		colorized_mask[mask == class_id] = color
	
	# Blend the original image with the colorized mask
	overlay = cv2.addWeighted(image, 1, colorized_mask, alpha, 0)
	return overlay

def find_pairs(root_dir):
	"""Find matching image/mask pairs in directory structure."""
	image_dirs = ['img',  'image', 'RGB', 'rgb', 'images']
	mask_dirs  = ['mask', 'masks', 'seg', 'segmentation']
	img_exts   = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
	
	image_files = {}
	mask_files = {}
	
	# Recursively search for image files
	for d in root_dir.rglob('*'):
		if d.is_dir() and d.name.lower() in image_dirs:
			for ext in img_exts:
				for img_path in d.glob(f'*{ext}'):
					key = img_path.stem
					if key not in image_files:
						image_files[key] = img_path
	
	# Recursively search for mask files
	for d in root_dir.rglob('*'):
		if d.is_dir() and d.name.lower() in mask_dirs:
			for ext in img_exts:
				for mask_path in d.glob(f'*{ext}'):
					key = mask_path.stem
					if key not in mask_files:
						mask_files[key.replace('_mask', '')] = mask_path
	
	# Create pairs where both exist
	pairs = []
	for key in set(image_files.keys()) & set(mask_files.keys()):
		pairs.append((image_files[key], mask_files[key]))
	
	return sorted(pairs, key=lambda x: x[0])

def main():
	parser = argparse.ArgumentParser(description='Segmentation Dataset Visualizer')
	parser.add_argument('root_dir', type=str, help='Root directory of dataset')
	args = parser.parse_args()
	
	root_path = Path(args.root_dir)
	if not root_path.exists():
		print(f"Error: Directory '{args.root_dir}' does not exist")
		return
	
	pairs = find_pairs(root_path)
	if not pairs:
		print("No matching image/mask pairs found")
		return
	
	print(f"Found {len(pairs)} image/mask pairs")
	
	# Display settings
	WIN_NAME = "Segmentation Visualizer"
	cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(WIN_NAME, 1920, 1080)
	
	# UI state
	current_idx = 0
	mode        = 0  # 0: overlay, 1: RGB+mask, 2: RGB+mask+overlay
	resize      = True
	alpha       = 0.5

	font_name = 'Ubuntu-R.ttf'
	ft = {}
	ft["large" ] = load_font(fn=font_name, size=48)
	ft["medium"] = load_font(fn=font_name, size=32)
	ft["small" ] = load_font(fn=font_name, size=16)
	ft["tiny"  ] = load_font(fn=font_name, size=8 )

	while True:
		img_path, mask_path = pairs[current_idx]
		image	= cv2.imread(str(img_path))
		mask	= cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
		
		# Create overlay
		overlay	= create_overlay(image, mask, alpha) if image is not None and mask is not None else None
		
		# Create display based on mode
		display	= np.zeros((1080, 1920, 3), dtype=np.uint8)
		
		if mode == 0:  # Overlay only
			if overlay is not None:
				if resize:
					disp_img, _, _ = resize_and_pad(overlay, 1920, 1080)
				else:
					disp_img, _, _ = resize_and_pad(overlay, 1920, 1080, resize=False)
				display = disp_img
		
		elif mode == 1:  # RGB + Mask
			if image is not None and mask is not None:
				# Convert mask to BGR for display
				mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask is not None else None
				
				img_disp, _, _ = resize_and_pad(image, 960, 1080)
				mask_disp, _, _ = resize_and_pad(mask_bgr, 960, 1080)
				display = np.hstack((img_disp, mask_disp))
		
		elif mode == 2:  # RGB + Mask + Overlay
			if image is not None and mask is not None:
				# Convert mask to BGR for display
				mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask is not None else None
				
				# Top row: RGB + Mask
				img_top, _, _ = resize_and_pad(image, 960, 540)
				mask_top, _, _ = resize_and_pad(mask_bgr, 960, 540)
				top_row = np.hstack((img_top, mask_top))
				
				# Bottom row: Overlay
				overlay_bottom, _, _ = resize_and_pad(overlay, 1920, 540)
				
				display = np.vstack((top_row, overlay_bottom))

		#print(cv2.getBuildInformation())

		# Add UI elements
		mode_text = f"Mode: {'Overlay' if mode == 0 else 'RGB+Mask' if mode == 1 else 'RGB+Mask+Overlay'}"
		#cv2.putText(display, mode_text, (20, 40  ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, lineType=cv2.LINE_AA)
		display = put_text(display, ft, mode_text, pt=(20, 40  ), c=(255, 128,  64), sz="medium", thickness=1.0)
		
		#file_info = f"{current_idx+1}/{len(pairs)}: {img_path.name} | {mask_path.name}"
		file_info1 = f"{current_idx+1}/{len(pairs)}"
		file_info2 = f"{image.shape[0]}x{image.shape[1]}"
		file_info3 = f"{img_path.name}"
		#cv2.putText(display, file_info, (20, 80  ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, lineType=cv2.LINE_AA)
		display = put_text(display, ft, file_info1, pt=(20,  80  ), c=(0  , 255, 255), sz="medium", thickness=1.0)
		display = put_text(display, ft, file_info2, pt=(20, 120  ), c=(0  , 128, 255), sz="medium", thickness=1.0)
		display = put_text(display, ft, file_info3, pt=(20, 160  ), c=(255, 128, 255), sz="medium", thickness=1.0)
		
		help_text = "Arrows: Navigate | 1-3: Change Mode | ESC: Exit | R: Resize"
		#cv2.putText(display, help_text, (20, 1040), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, lineType=cv2.LINE_AA)
		display = put_text(display, ft, help_text, pt=(20, 1040), c=(200, 200, 200), sz="small",  thickness=1.0)

		mode_colors= [(  0, 255,   0), (  0, 255, 255), (  0,   0, 255)]

		# Show mode selection boxes
		colors = [mode_colors[i] if mode == i else (100, 100, 100) for i in range(3)]
		for i, (label, color) in enumerate(zip(["Overlay", "RGB+Mask", "RGB+Mask+Overlay"], colors)):
			large_offset = 60 if i == 2 else 0
			cv2.rectangle(display, (1400 + i*130, 20), (1520 + i*130 + large_offset, 60), color, 2)
			#cv2.putText(display, label , (1510 + i*130, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, lineType=cv2.LINE_AA)
			#cv2.putText(display, str(i), (1580 + i*130, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)
			display = put_text(display, ft, label,    pt=(1430 + i*130, 30), c=color, sz="small", thickness=1.0)
			display = put_text(display, ft, str(i+1), pt=(1410 + i*130, 30), c=color, sz="small", thickness=1.0)
		
		# Show image
		cv2.imshow(WIN_NAME, display)
		
		# Handle keyboard input
		key = cv2.waitKey(10) & 0xFF
		if key == ord('1'):
			mode = 0
		elif key == ord('2'):
			mode = 1
		elif key == ord('3'):
			mode = 2
		elif key == ord('r') or key == ord('R'):
			resize = not resize 
		elif key == 81 or key == 82:  # Left arrow or Up arrow
			current_idx = max(0, current_idx - 1)
		elif key == 83 or key == 84:  # Right arrow or Down arrow
			current_idx = min(len(pairs) - 1, current_idx + 1)
		elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC
			break
	
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
