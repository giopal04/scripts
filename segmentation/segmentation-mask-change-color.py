#!/usr/bin/env python

import sys
import cv2
import argparse
import datetime
import numpy as np
from pathlib import Path

from classes.argument_parser import define_boolean_argument, var2opt

# E.g. conversion for surface-pattern-recognition dataset:
# /mnt/btrfs-data/venvs/ml-tutorials/repos/depthai-python/examples/custom-scripts/segmentation-mask-change-color.py -dir `pwd` -class1 255,255,255-0,0,0 -class2 255,0,0-1,1,1 -class3 0,0,255-2,2,2 -class4 0,255,0-3,3,3 -class5 44,0,0-4,4,4 -class6 185,26,255-5,5,5 -class7 0,211,255-6,6,6 -class8 0,88,0-7,7,7 --no-dry-run --to-grayscale

# ./segmentation-mask-change-color.py -dir /tmp/dataset-simple-shapes/dataset/labels -class1 1,1,1-255,0,0 -class2 2,2,2-0,255,0 -class3 3,3,3-0,0,255 -class4 4,4,4-255,255,0 -class5 5,5,5-255,0,255 --no-dry-run

parser = argparse.ArgumentParser()
parser.add_argument('-class1', nargs='?', help='class 1: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class2', nargs='?', help='class 2: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class3', nargs='?', help='class 3: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class4', nargs='?', help='class 4: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class5', nargs='?', help='class 5: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class6', nargs='?', help='class 6: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class7', nargs='?', help='class 7: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class8', nargs='?', help='class 8: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class9', nargs='?', help='class 9: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class10', nargs='?', help='class 10: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class11', nargs='?', help='class 11: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class12', nargs='?', help='class 12: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class13', nargs='?', help='class 13: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class14', nargs='?', help='class 14: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class15', nargs='?', help='class 15: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class16', nargs='?', help='class 16: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class17', nargs='?', help='class 17: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class18', nargs='?', help='class 18: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class19', nargs='?', help='class 19: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-class20', nargs='?', help='class 20: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-dir',   nargs='?', help="directory to search", default='.')
parser.add_argument('-ext',   nargs='?', help="file extension to search for", default='png')
#parser.add_argument('-dry-run', type=eval, choices=[True, False], default='True', help=)
define_boolean_argument(parser, *var2opt('to_grayscale'), 'convert to grayscale after color conversion, before writing to file'	, False)
define_boolean_argument(parser, *var2opt('dry_run')	, 'don\'t write anything, just show old and new masks'			, True)
define_boolean_argument(parser, *var2opt('show_images')	, 'show old and new masks'						, True)
args = parser.parse_args()

if not Path(args.dir).exists():
	print(f'Directory {args.dir} does not exist. Exiting...')
	sys.exit(1)

print(f'{args.dir = }')

search_path = Path(args.dir)

'''
colors1 = args.class1.split('-')
from1   = [int(i) for i in colors1[0].split(',')]
to1     = [int(i) for i in colors1[1].split(',')]
'''

from_to = []

for arg in vars(args):
	#print(arg, getattr(args, arg))
	val = getattr(args, arg)
	if 'class' in arg and val is not None:
		#print(val)
		colors = val.split('-')
		from_c = [int(i) for i in colors[0].split(',')]
		to_c   = [int(i) for i in colors[1].split(',')]
		from_to.append((from_c, to_c))

print(f'Converting class masks in this way: {from_to = }')

print(f'Dry run flag is: {args.dry_run} - show images flag is: {args.show_images}')

for fn in search_path.glob('*.' + args.ext):
	print(f'Reading image {str(fn)}...')
	img  = cv2.imread(str(fn))

	if args.show_images:
		cv2.imshow('Current mask', cv2.resize(img, (640, 360)))
		cv2.moveWindow('Current mask', 200, 200)
		cv2.waitKey(1)

	for from_c, to_c in from_to:
		#print(from_c)
		#print(to_c)
		mask_color_lo = np.array(from_c)
		mask_color_hi = np.array(from_c)

		mask = cv2.inRange(img, mask_color_lo, mask_color_hi)
		img[mask>0] = tuple(to_c)

	if args.show_images:
		cv2.imshow('New mask',  cv2.resize(img, (640, 360)))
		cv2.moveWindow('New mask', 840, 200)
		cv2.waitKey(1)

	if not args.dry_run:
		if args.to_grayscale:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		print(f'Writing {"grayscale " if args.to_grayscale else ""}image {str(fn)}...')
		cv2.imwrite(str(fn), img)
