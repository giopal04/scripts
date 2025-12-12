import cv2
import math
import numpy as np
from datetime import datetime

def datetime_from_string(str_time):
	return datetime.strptime(str_time, '%Y-%m-%d-%H-%M-%S')

def compute_fps(curr_time, last_time, start_capture_time, dequeued_frames_dict):
	if curr_time != last_time:
		#print(f'{curr_time = }')
		last_time = curr_time
		curr_time = datetime.now()
		run_time = curr_time - datetime_from_string(start_capture_time)
		display_str = ''
		for stream, frames in dequeued_frames_dict.items():
			if run_time.total_seconds() == 0:
				break
			microseconds = run_time.seconds * 1000000 + run_time.microseconds
			fps = frames*1000000/microseconds
			if display_str != '':
				display_str += ' - '
			display_str += stream + ' ' + str(frames) + ' ' + f'{fps:.2f}'
		print(display_str)
	return last_time
