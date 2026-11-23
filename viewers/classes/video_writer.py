import shlex
import subprocess as sp

import sys
import cv2
import numpy as np

class video_writer:
	out_fn		= ''
	resolution	= (-1, -1)
	fps		= -1
	codec		= ''
	crf		= -1		# libx264 preset quality

	debug		= False

	nframes		= -1

	ffmpeg_prefix	= ''
	ffmpeg_suffix	= ''
	process		= None

	def __init__(self, out_fn, resolution, fps, codec='libx264', crf=24, debug=False):
		self.debug      = debug

		self.out_fn	= out_fn
		self.resolution	= resolution
		self.fps	= fps
		self.codec	= codec
		self.crf	= crf

		self.nframes	= 0

		if self.debug:
			print(f'Video writer initialized with the following parameters: {out_fn = } - {resolution = } - {fps = } - {codec = } - {crf = }')

		self.ffmpeg_prefix = f'ffmpeg -y -s '
		self.ffmpeg_suffix = f' -pixel_format bgr24 -f rawvideo -r {self.fps} -i pipe: -vcodec {self.codec} -pix_fmt yuv420p -crf {self.crf} {self.out_fn}'
		self.process = sp.Popen(shlex.split(f'{self.ffmpeg_prefix}{self.resolution[1]}x{self.resolution[0]}{self.ffmpeg_suffix}'), stdin=sp.PIPE, stdout=None, stderr=None)
	
		return
	
	def write(self, frame):
		if self.debug:
			print(f'Writing: {frame.shape = } - {frame[:100] = }')

		self.process.stdin.write(frame.tobytes())

		return frame
	
	def close(self):
		# Close and flush stdin
		self.process.stdin.close()
		# Wait for sub-process to finish
		self.process.wait()
		# Terminate the sub-process
		self.process.terminate()

		del self.process
