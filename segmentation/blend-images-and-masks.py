#!/usr/bin/env python3

import sys
import argparse

from pathlib import Path

import cv2

def argument_parser():
	global args

	parser = argparse.ArgumentParser(description='Blend images and masks with the same name (except extension) found in two different directories')

	parser.add_argument('--imgs-dir' , required=True		, help='directory where images are located    (e.g. /tmp/images)')
	parser.add_argument('--masks-dir', required=True		, help='directory where RGB masks are located (e.g. /tmp/masks)')
	parser.add_argument('--out-dir'	 , default='/tmp'		, help='directory where output images will be written (e.g. /tmp)')

	args = parser.parse_args()
	print(f'argument_parser() received these arguments: {args}')
	return args


def overlay(frame, mask, alpha=0.5):
	return cv2.addWeighted(frame, 1, mask, alpha, 0)

if __name__ == '__main__':
	args = argument_parser()
	ifns  = [fn for fn in Path(args.imgs_dir).rglob(f'*.jpg')]
	mfns  = [fn for fn in Path(args.masks_dir).rglob(f'*.png')]
	ifns.sort()
	mfns.sort()
	print(f'Found {len(ifns)} filenames: {list(map(lambda x: x.name, ifns[:2]))} , ... , {list(map(lambda x: x.name, ifns[-2:]))}')
	print(f'Found {len(mfns)} filenames: {list(map(lambda x: x.name, mfns[:2]))} , ... , {list(map(lambda x: x.name, mfns[-2:]))}')

	debug = True
	if debug:
		delay = 10000
	else:
		delay =  1

	if len(ifns) > 0 and len(mfns) > 0 and len(ifns) == len(mfns):
		for ifn,mfn in zip(ifns,mfns):
			masks = []
			print(f'Processing files: {ifn.name}+{mfn.name}...')
			img = cv2.imread(str(ifn))
			msk = cv2.imread(str(mfn))
			blended = overlay(img,msk, alpha=0.5)
			blended = cv2.resize(blended, (1280, 720))
			cv2.imshow('blended', blended)
			cv2.imwrite(str(Path(args.out_dir) / ('blended-'+ifn.name)), blended)
			key = cv2.waitKey(delay)
			if key == 27:			# ESC
				break
