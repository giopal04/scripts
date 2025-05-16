#!/usr/bin/env python3

import sys
import argparse

import numpy as np

from pathlib import Path

import cv2
import Dither

'''
@jit(nopython=True)
def floydDitherspeed(img_array):
    height, width, _ = img_array.shape
    colors = np.array([[255, 255, 255], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 128, 0], [253, 134, 18]])
    for y in range(0, height-1):
        for x in range(1, width-1):
            old_pixel = img_array[y, x, :]
            max_distance = 195075
            for color in colors:
                p_distances = old_pixel - color
                p_distances = np.power(p_distances, 2)
                distance = p_distances[0] + p_distances[1] + p_distances[2]
                if distance <= max_distance:
                    max_distance = distance
                    new_pixel = color
            img_array[y, x, :] = new_pixel
            quant_error = new_pixel - old_pixel
            img_array[y, x+1, :] =  img_array[y, x+1, :] + quant_error * 7/16
            img_array[y+1, x-1, :] =  img_array[y+1, x-1, :] + quant_error * 3/16
            img_array[y+1, x, :] =  img_array[y+1, x, :] + quant_error * 5/16
            img_array[y+1, x+1, :] =  img_array[y+1, x+1, :] + quant_error * 1/16
    return img_array
'''

def argument_parser():
	global args

	parser = argparse.ArgumentParser(description='Extract Segmentation Masks from Blended Imgs')

	parser.add_argument('--class1'	, required=True		, help='class 1 BGR <b1,g1,r1> color (e.g.   0,0,255 - red)')
	parser.add_argument('--class2'	, default=''		, help='class 2 BGR <b2,g2,r2> color (e.g.   255,0,0 - blue)')
	parser.add_argument('--class3'	, default=''		, help='class 3 BGR <b2,g2,r2> color (e.g. 0,255,255 - purple)')
	parser.add_argument('--dir'   	, required=True		, help='directory where images are located (e.g. /tmp/blended)')
	parser.add_argument('--out-dir'	, default='/tmp'	, help='directory where output masks will be written (e.g. /tmp)')
	parser.add_argument('--patt'	, required=True		, help='pattern with wildcards to be passed to Path().glob() for finding files')
	'''
	parser.add_argument('--model-name'			, help='the model to load for inference (can be both a .pkl or .pth model)')
	parser.add_argument('--classes', default=''		, help='classes of the problem (e.g. asphalt,pothole,crack)')
	parser.add_argument('--img', default=''			, help='the image to use for inference')
	parser.add_argument('--dir', default=''			, help='scan this directory and perform inference on all the images')
	parser.add_argument('--output-dir', default=''		, help='put all output (images/videos) in this directory')
	parser.add_argument('--video', default=''		, help='the video file to use for inference')
	parser.add_argument('--out-video', default=''		, help='write to this video file')
	parser.add_argument('--ffmpeg', default=''		, help='the ffmpeg command to use as source')
	define_boolean_argument(parser, *var2opt('rotatevideo')	, 'rotate the video 90° after inference (invert array idxs while saving)')
	define_boolean_argument(parser, *var2opt('test_set')	, 'don\'t use a test image, use the whole test set in --dataset-dir')
	define_boolean_argument(parser, *var2opt('show_frames')	, 'show frames in one or more OpenCV windows')
	parser.add_argument('--export-pkl-model', default=None	, help='if present, load the .pth model + create a DataLoader object for it from --dummy-dataset-dir and export a .pkl model')
	parser.add_argument('--dataset-dir', default=None	, help='if present, use this dataset to create a DataLoader object for the Learner')
	parser.add_argument('--crop-src', default=''		, help='crop source image(s)/video before inference (e.g. x,y,w,h -> 0,540,1920,540 = bottom half of 1920x1080 img)')
	parser.add_argument('--arch'				, help='create a custom architecture from module `arch` -- the module must be in the current directory and have the following API: `def create_model(device="cuda:0")` -- provide the module name without .py extension')
	parser.add_argument('--batch-size', type=int, default='32', help='batch size for inference')
	parser.add_argument('--just-benchmark-inference', type=int, default=0, help='don\'t perform actual inference on real images, videos or streams, just benchmark what your GPU can do in theory (BS=128)')
	define_boolean_argument(parser, *var2opt('save_image')	, 'save the resized image before feeding it to the network')
	'''

	args = parser.parse_args()
	print(f'argument_parser() received these arguments: {args}')
	return args

def mask_and_replace_color(img, color_lo, color_hi=None, color_rep=None, debug=False):
	if color_hi is None:
		color_hi = color_lo
	if color_rep is None:
		color_rep = color_lo
	print(f'Performing cv2.inRange() between {color_lo = } and {color_hi = }')
	#mask = cv2.inRange(img, color_lo, color_hi)			# a nice B/W mask of the selected color or range
	mask = cv2.inRange(img, (0,0,250), (0,0,255))			# a nice B/W mask of the selected color or range
	npix = mask[mask>0].shape[0]
	if debug:
		print(mask[mask>0].shape)				# e.g. (29409,) the map of white (monochrome) pixels inside the mask
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)			# BGR mask, but always B/W in content
	if debug:
		print(mask[mask>0].shape)				# e.g. (29409,) the map of white (BGR) pixels inside the mask (= x3 as before)
	mask[mask>0] = list(color_rep) * npix				# replace pixels according to the map by replicating the original tuple npix times
	return mask

if __name__ == '__main__':
	args = argument_parser()
	fns  = [fn for fn in Path(args.dir).rglob(f'{args.patt}')]
	print(f'Found {len(fns)} filenames: {list(map(lambda x: x.name, fns[:2]))} , ... , {list(map(lambda x: x.name, fns[-2:]))}')

	class1 = class2 = class3 = ''
	if args.class1 != '':
		class1 = args.class1.split(',')
		class1 = tuple(map(int, class1))
	if args.class2 != '':
		class2 = args.class2.split(',')
		class2 = tuple(map(int, class2))
	if args.class3 != '':
		class3 = args.class3.split(',')
		class3 = tuple(map(int, class3))

	print(f'Received the following classes as parameter: {class1} - {class2} - {class3}')

	debug = True
	if debug:
		delay = 10000
	else:
		delay =  1





	if len(fns) > 0:
		for fn in fns:
			masks = []
			print(f'Processing file: {fn.name}...')
			img = cv2.imread(str(fn))
			cv2.imshow('blended', img)
			print(f'{img.shape = }')
			'''
			diths = []
			for i in range(3):
				dith = Dither.dither(img[:,:,i], 'floyd-steinberg', resize=False)
				cv2.imshow('dithered'+(str(i)), dith)
				diths.append(dith)
			dith = np.stack((diths)).transpose(1,2,0)
			print(f'{dith.shape = }')
			cv2.imshow('stacked-dithered', dith)
			'''
			#mask = mask_and_replace_color(img, (0,0,255), color_rep=(0,255,0))
			if class1:
				print(f'{class1 = }')
				mask = mask_and_replace_color(img, tuple(class1), color_rep=tuple(class1))
				print(f'{mask.shape}')
				cv2.imshow('mask', mask)
				key = cv2.waitKey(delay)
				masks.append(mask)
			#mask = mask_and_replace_color(img, (255,0,255), color_rep=(0,0,255))
			if class2:
				print(f'{class2 = }')
				mask = mask_and_replace_color(img, tuple(class2), color_rep=tuple(class2))
				print(f'{mask.shape}')
				cv2.imshow('mask', mask)
				key = cv2.waitKey(delay)
				masks.append(mask)
			if class3:
				print(f'{class3 = }')
				mask = mask_and_replace_color(img, tuple(class3), color_rep=tuple(class3))
				print(f'{mask.shape}')
				cv2.imshow('mask', mask)
				key = cv2.waitKey(delay)
				masks.append(mask)
			#final_mask = masks[0] | masks[1] | masks[2]
			final_mask = masks[0]
			print(f'{final_mask.shape = }')
			for msk in masks:
				final_mask = cv2.bitwise_or(final_mask, msk)
				#final_mask |= msk
			cv2.imshow('mask', final_mask)
			#cv2.imshow('mask', mask)
			cv2.imwrite(str(Path(args.out_dir) / ('FINALMASK-'+fn.name)), final_mask)
			key = cv2.waitKey(delay)
			if key == 27:			# ESC
				break
