#!/usr/bin/env python

from PIL import Image
import numpy as np
from pathlib import Path


def clean_spurious_colors(fn, debug=False):
	if debug:
		#im = Image.open('e-20220203-112045-000048-000014.png-binarized.png')
		im = Image.open('e-20220203-095449-000063-000010.png-binarized.png')
	else:
		im = Image.open(fn)
	
	im = im.convert('RGBA')
	
	data = np.array(im)   # "data" is a height x width x 4 numpy array
	print(type(data.T), data.T.shape)
	red, green, blue, alpha = data.T # Temporarily unpack the bands for readability
	
	# Replace white with red... (leaves alpha values alone...)
	#white_areas = (red == 255) & (blue == 255) & (green == 255)
	red_areas = (red >= 200) 
	data[..., :-1][red_areas.T] = (255, 0, 0) # Transpose back needed
	
	green_areas = (green >= 200) 
	data[..., :-1][green_areas.T] = (0, 255, 0) # Transpose back needed

	black_areas = (red <= 50) & (blue <= 50) & (green <= 50)
	data[..., :-1][black_areas.T] = (0, 0, 0) # Transpose back needed

	if debug:
		print(data[..., :-1][red_areas.T][:10])
		print(data[..., :-1][green_areas.T][:10])
		print(data[..., :-1][black_areas.T][:10])
	
	im2 = Image.fromarray(data)
	if debug:
		im2.show()

	im2.save(f'{fn}-clean.png')

lst = Path('.').glob(f'*.png')
for fn in lst:
	print(fn)
	clean_spurious_colors(fn)

