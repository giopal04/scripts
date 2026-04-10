#!/usr/bin/env python3

# Run with:

# ./comfyui-SAM3-video-segmentation.py --input-video /mnt/raid1/dataset/surface-damage-segmentation/sam3-video-dataset/statues/China-Tian-Tan-Buddha-Hl9TCsAIbdI-3840x2160-30fps-60s-1825.mp4 --prompt statue --custom-width 1080 --custom-height 1080 --chunk-size 50 --num-frames 150 --write-images --center-crop

# OR

# ./comfyui-SAM3-video-segmentation.py --input-video /mnt/raid1/dataset/surface-damage-segmentation/sam3-video-dataset/statues/Indonesia-Garuda-Wisnu-Kencana-dzC4FnMpcuk-1280x720-30fps-32s-985.mp4 --prompt statue --custom-width 1920 --custom-height 1080 --chunk-size 50 --num-frames 150 --write-images

# OR even better

# ./comfyui-SAM3-video-segmentation.py --input-video /mnt/raid1/dataset/surface-damage-segmentation/sam3-video-dataset/statues/China-Tian-Tan-Buddha-Hl9TCsAIbdI-3840x2160-30fps-60s-1825.mp4 --prompt "statue OR monument OR column" --custom-width 1920 --custom-height 1080 --chunk-size 100 --chunk-overlap 10 --num-frames 300 --write-images

# OR also

# ./comfyui-SAM3-video-segmentation.py --input-video /tmp/piazza-san-pietro.mp4 --prompt "column" --chunk-size 50 --num-frames 1400 --tile 0,0,1080,960 --tile-resize 1008x1008 --save-coco-json --one-every-n-frames 5
# ./comfyui-SAM3-video-segmentation.py --input-video /tmp/piazza-san-pietro.mp4 --prompt "column" --chunk-size 50 --num-frames 1400 --tile 0,960,1080,960 --tile-resize 1008x1008 --save-coco-json --one-every-n-frames 5

# OR even

# ./comfyui-SAM3-video-segmentation-claude.py --input-video /tmp/Track_A-Sphere.mp4 --prompt 'door double_door front_door gate' --chunk-size 20 --num-frames 500 --tile 0,0,3518,2440 --tile-resize 1008x1008 --gpu 0 --overlay-color green --overlay-alpha 0.9 --color-step 5

# ./comfyui-SAM3-video-segmentation-claude.py --input-video /tmp/Track_A-Sphere.mp4 --prompt 'door double_door front_door gate' --chunk-size 20 --num-frames 500 --tile 3518,0,3518,2440 --tile-resize 1008x1008 --gpu 1 --overlay-color green --overlay-alpha 0.9 --color-step 5




# Remember than AV1 is not well supported by OpenCV, you'll likely hit bug 11389 (https://github.com/opencv/opencv/issues/11389)
# Convert webm files to h264/h265 with ffmpeg first

# Also make sure to have plenty of space in /tmp because SAM3 requires several gigabytes of free space for its temporary files (e.g. /tmp/sam3_3be2b85c_fa088o3p/mmap_output/{frames.mmap, masks.mmap, vis.mmap})


# If you're getting CPU NMS warnings (Non-Maximum Suppression for bboxes running on CPU instead of GPU), run this:

# pip install https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cc_torch-latest/cc_torch-0.2+cu128torch2.9-cp313-cp313-linux_x86_64.whl
# pip install https://github.com/PozzettiAndrea/cuda-wheels/releases/download/torch_generic_nms-latest/torch_generic_nms-0.1%2Bcu128torch2.9-cp313-cp313-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl

# But it won't change your life, everything is still heavily CPU-bound.


import os

import sys
import types
import ctypes			# for ctypes.CDLL manual garbage collect
import argparse
import colorsys
import json
from datetime import datetime

import psutil
import numpy as np

from pathlib import Path

import gc
import cv2

import subprocess
import shutil, glob

#from memory_profiler import profile

parser = argparse.ArgumentParser()
parser.add_argument('--input-video',	type=str,	required=True,		help='Path to input video file. Note: ComfyUI doesn\'t allow writing output files outside its output directory.')
parser.add_argument('--model',		type=str,	default='/mnt/raid1/repos/sam3/models/sam3.pt',		help='SAM3 model to load')
parser.add_argument('--gpu',		type=str,	default='1',		help='GPU where to run the workflow')
parser.add_argument('--output-dir',	type=str,	default='/tmp',		help='Output directory (for both videos and images)')
parser.add_argument('--prompt',		type=str,	required=True,		help='Text prompt for SAM3')
parser.add_argument('--num-frames',	type=int,	default=10,		help='Total number of frames to process (0 = all)')
parser.add_argument('--start-frame',	type=int,	default=0,		help='Start frame (0 = beginning of video)')
parser.add_argument('--chunk-size',	type=int,	default=50,		help='Number of frames per chunk to avoid OOM')
parser.add_argument('--chunk-overlap',	type=int,	default=5,		help='Overlap between chunks for smooth transitions')
parser.add_argument('--custom-width',	type=int,	default=0,		help='Resize input video to this width before processing')
parser.add_argument('--custom-height',	type=int,	default=0,		help='Resize input video to this height before processing')
parser.add_argument('--overlay-color',	type=str,	default=None,		help='Master overlay color for segmentation (e.g. red, green, blue, yellow, cyan, magenta, #FF8800). All instances will be shades of this color. If omitted, each unique label in the prompt gets a color derived by hashing its text, so different prompts (e.g. "window" vs "zebra crossing") always produce distinct, reproducible colors.')
parser.add_argument('--overlay-alpha',	type=float,	default=0.5,		help='Transparency of overlay (0.0-1.0)')
parser.add_argument('--color-step',	type=int,	default=100,		help='How much to decrement the primary color channel per instance (default 10, e.g. red: 255→245→235…)')
parser.add_argument('--max-objects',	type=int,	default=100,		help='Maximum number of tracked objects to extract per chunk (default 10). Raise if you expect many instances.')
parser.add_argument('--max-num-objects', type=int,	default=200,		help='Hard cap on the number of objects SAM3 tracks internally during propagation (default 200). SAM3\'s own default is 10000, which causes O(N²) CUDA OOM on scenes with many instances (e.g. city windows). Lower this if you hit OOM during propagate_in_video.')
parser.add_argument('--one-every-n-frames', type=int,	default=1,		help='Process only every N-th source frame (e.g. 5 = keep 1, discard 4). All frame counts and chunk sizes are in kept-frame units; the skipping is applied transparently inside the frame reader.')
parser.add_argument('--center-crop',	action='store_true',			help='Center-crop input video instead of resizing')
parser.add_argument('--write-images',	action='store_true',			help='Also write images with videos (useful for creating standard image datasets)')
parser.add_argument('--offload-model',	action='store_true',			help='Offload model to CPU between chunks')
parser.add_argument('--debug',		action='store_true',			help='Show debug information')
parser.add_argument('--tile',		type=str,	default=None,		help='Process only a rectangular region of the input video: x,y,w,h (pixels, in original resolution). Output videos are written at the original resolution with zero-padding outside the tile.')
parser.add_argument('--tile-resize',	type=str,	default=None,		help='Resize the tile to WxH before sending to the GPU (e.g. 1008x1008). Masks are resized back to the original tile dimensions before being placed in the output canvas. Only meaningful when --tile is also set.')
parser.add_argument('--save-coco-json',	action='store_true',			help='Save per-frame, per-instance predictions as a COCO-compatible JSON file alongside the video outputs.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f'Using GPU: {args.gpu}')

import torch

# Get the absolute path of the directory 3 levels up (the repo root)
# .parent = sam3/ | .parent.parent = segmentation/ | .parent.parent.parent = root/
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from classes.tensor_utils	import center_crop
from classes.aux_utils		import check_pytorch_cuda_version, ensure_sam3_gpu_nms
from classes.comfy_utils	import nodeoutput_to_type, setup_comfyui_environment, import_sam3_package, load_sam3_model
from classes.ffmpeg_utils	import FFmpegCapture, start_ffmpeg_streaming, write_frame_to_ffmpeg, finalize_ffmpeg
from classes.coco_utils		import mask_to_polygons, mask_to_bbox_area

# Run checks immediately (before heavy imports) so the user sees them up front.
check_pytorch_cuda_version()

# ---------------------------------------------------------------------------

# Path to your ComfyUI installation
comfyui_base_dir  = Path('/mnt/raid1/repos/comfyui')

# Attempt to build GPU NMS extension before importing SAM3,
# so the compiled .so is on sys.path when SAM3 initializes.
ensure_sam3_gpu_nms(comfyui_base_dir)

# ---------------------------------------------------------------------------

# global placeholder for comfyui_sam3 module
comfyui_sam3_module	= None
sam3_nodes		= None

# Add ComfyUI base paths to import basic modules...
sys.path.append(str(comfyui_base_dir))
if args.debug:
	print(f'sys.path: {sys.path}')
os.environ['PYTHONPATH'] = str(comfyui_base_dir)
if args.debug:
	print(f'PYTHONPATH: {os.environ["PYTHONPATH"]}')
	
import nodes
import folder_paths
from comfy_api import latest

import app
import utils

from comfy_extras import nodes_mask

#@profile
def get_video_chunk(cap, start_frame, count, custom_width=0, custom_height=0,
					center_crop_param=False, tile=None, one_every_n_frames=1,
					overlap=0, master_stdin=None, is_last_chunk=False, debug=False):
	"""Fetches a specific range of frames from OpenCV VideoCapture.

	If *tile* is a (tx, ty, tw, th) tuple the crop is applied to the raw frame
	BEFORE any resize / center-crop so that the model only ever sees the tile
	region.

	*start_frame* and *count* are both in **kept-frame units** (i.e. after
	applying the one_every_n_frames subsampling).  The actual source-video seek
	position is ``start_frame * one_every_n_frames``; between each kept frame
	``one_every_n_frames - 1`` source frames are read and silently discarded so
	that SAM3 never sees them.
	"""
	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame * one_every_n_frames)
	frames = []
	# Native-resolution RGB frames (pre-tile-crop) that pass the left trim,
	# buffered here so we can apply the right trim only after knowing the
	# actual frame count (needed to handle the last chunk correctly).
	master_write_buffer = []  # list of (local_idx, frame)

	if debug:
		print(f'get_video_chunk() is about to read {count} frames from {start_frame} (kept-frame units) - is_last_chunk={is_last_chunk}')
	for idx in range(count):
		kept_local_idx  = idx
		kept_abs_idx    = start_frame + idx
		source_abs_idx  = (start_frame + idx) * one_every_n_frames
		frame_str = f'local={kept_local_idx} - abs={kept_abs_idx} - source={source_abs_idx}'

		ret, frame = cap.read()
		if debug:
			print(f'Extracted frame {frame_str} with size: {frame.shape}')
		if not ret:
			break
		# Discard the next (one_every_n_frames - 1) source frames so that only
		# every N-th source frame is forwarded to SAM3.
		for idx2 in range(one_every_n_frames - 1):
			if debug:
				print(f'get_video_chunk() is discarding frame {frame_str} (-> {source_abs_idx + idx2 + 1})')
			cap.read()	# return value intentionally ignored

		# Convert BGR to RGB and scale to 0-1 for the model
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Buffer master video frame candidates (native resolution, pre-tile-crop).
		# Right-trim cannot be determined yet because we don't know how many
		# frames will actually be read (EOF may cut the chunk short).  Only the
		# left trim is applied here; the right trim is resolved after the loop.
		if master_stdin is not None and one_every_n_frames > 1:
			if debug:
				print(f'get_video_chunk() is considering frame no. {frame_str} for writing to master video')

			left_trim = 0 if start_frame == 0 else overlap

			if left_trim <= idx:
				# frame is still full-resolution RGB here; subsequent tile/resize
				# operations only *reassign* the local variable, so this reference
				# stays valid and no copy is required.
				master_write_buffer.append((idx, frame))

		# --- tile crop (applied to the raw BGR frame, before any resize) ------
		if tile is not None:
			tx, ty, tw, th = tile
			frame = frame[ty:ty + th, tx:tx + tw]
			if debug:
				print(f'Tile-cropped frame to region x={tx} y={ty} w={tw} h={th}: {frame.shape}')
		# -----------------------------------------------------------------------
		if (custom_width > 0 or custom_height > 0) and not center_crop_param:
			if debug:
				orig_size = frame.shape
			frame = cv2.resize(frame, (custom_width, custom_height))
			if debug:
				print(f'Resized frame with size: {orig_size} to {frame.shape}')
		frames.append(frame)

	if not frames:
		return None

	# Flush master video frames now that we know the actual count.
	# right_trim is 0 when:
	#   - the caller explicitly flagged this as the last chunk, OR
	#   - fewer frames were read than requested (hit EOF mid-chunk), which
	#     means this is effectively the last chunk regardless of the flag.
	if master_write_buffer:
		actual_count = len(frames)
		right_trim = 0 if (is_last_chunk or actual_count < count) else overlap
		for buf_idx, buf_frame in master_write_buffer:
			if buf_idx < actual_count - right_trim:
				if debug:
					kept_abs_idx   = start_frame + buf_idx
					source_abs_idx = (start_frame + buf_idx) * one_every_n_frames
					frame_str = f'local={buf_idx} - abs={kept_abs_idx} - source={source_abs_idx}'
					print(f'get_video_chunk() is writing frame no. {frame_str} to master video')
				write_frame_to_ffmpeg(master_stdin, buf_frame)

	if debug:
		print(f'{torch.from_numpy(np.stack(frames)).shape = }')

	stacked    = np.stack(frames);
	del frames  # free the list of individual arrays

	tensor_out = torch.from_numpy(stacked).float() / 255.0
	del stacked                                  # numpy copy freed; float32 tensor is independent

	if center_crop_param:
		if debug:
			orig_size = tensor_out.shape
		tensor_out = center_crop(tensor_out, target_h=custom_height, target_w=custom_width)
		if debug:
			print(f'Center-cropped frame with size: {orig_size} to {tensor_out.shape}')
	return tensor_out

#@profile
def extract_per_object_masks_direct(masks_raw, extract_start, extract_end, max_objects=10, debug=False):
	"""
	Read per-object mask tensors DIRECTLY from the raw propagation dict.

	SAM3 can return masks in two formats:

	Format A — dict-of-dicts  (older versions):
	    {frame_idx: {obj_id: tensor(H, W)}}

	Format B — dict-of-arrays  (current observed behaviour):
	    {frame_idx: ndarray/tensor (N_instances, H, W)}
	    The instance axis (dim 0) indexes individual tracked objects.

	Both formats are handled.  Returned 'actual_obj_ids' are always
	0-based instance indices so downstream coloring code is consistent.

	IMPORTANT: extract_start / extract_end are chunk-position values (0-based
	within the loaded video chunk), and SAM3 stores masks under those same
	chunk-position frame keys.  We filter by VALUE (extract_start <= key <
	extract_end) rather than by list position, because propagation may not
	have produced a mask for every frame (e.g. the prompt frame at index 0 is
	always present even when start_frame > 0, creating a gap).  Missing frames
	within the target range are zero-padded so the output length always equals
	(extract_end - extract_start).

	Args:
		masks_raw:     dict returned by propagate_masks_chunk()
		extract_start: first chunk-position frame index to keep (inclusive)
		extract_end:   one-past-last chunk-position frame index to keep
		max_objects:   cap on how many object ids to process

	Returns:
		(per_obj, actual_obj_ids) where:
		  per_obj        — list of (slice_len, H, W) float32 tensors in [0,1], one per instance.
		  actual_obj_ids — list of 0-based int instance indices, same order as per_obj.
		Both are empty lists on failure.
	"""
	if not isinstance(masks_raw, dict) or len(masks_raw) == 0:
		print(f'[extract_direct] masks_raw is not a usable dict (type={type(masks_raw).__name__})')
		return [], []

	# Filter frame keys by VALUE, not position — avoids the gap caused by the
	# prompt frame (key=0) when start_frame > 0.
	slice_frames = sorted(fi for fi in masks_raw.keys()
	                      if extract_start <= fi < extract_end)
	target_len   = extract_end - extract_start   # expected number of output frames

	if debug:
		# masks_raw is a dict with the frame number as key and a numpy array of a boolean mask as value
		print(f'{masks_raw = }, {extract_start = }, {extract_end = }, {max_objects = }')
	print(f'[extract_direct] keys in range [{extract_start},{extract_end}): '
	      f'{len(slice_frames)} of {target_len} expected '
	      f'(total keys in dict: {len(masks_raw)})')

	# Peek at the first in-range frame to detect format; fall back to any frame
	# if the range is completely empty (will zero-pad everything below).
	first_key = slice_frames[0] if slice_frames else sorted(masks_raw.keys())[0]
	first_val = masks_raw[first_key]

	# ------------------------------------------------------------------ #
	# Format A: dict-of-dicts  {frame_idx: {obj_id: tensor(H,W)}}        #
	# ------------------------------------------------------------------ #
	if isinstance(first_val, dict):
		obj_ids = sorted(first_val.keys())
		if len(obj_ids) > max_objects:
			print(f'[extract_direct] Format A — capping {len(obj_ids)} SAM obj_ids → {max_objects}')
			obj_ids = obj_ids[:max_objects]
		print(f'[extract_direct] Format A — {len(obj_ids)} SAM obj_id(s) × {len(slice_frames)} frames '
		      f'(zero-padding {target_len - len(slice_frames)} missing frame(s))')

		per_obj        = []
		actual_obj_ids = []
		ref_h, ref_w   = None, None

		# Discover spatial dims from any available frame
		for fi in slice_frames:
			for v in masks_raw[fi].values():
				if hasattr(v, 'shape') and len(v.shape) >= 2:
					ref_h, ref_w = v.shape[-2], v.shape[-1]
					break
			if ref_h is not None:
				break

		for instance_idx, obj_id in enumerate(obj_ids):
			frame_tensors = []
			for fi in range(extract_start, extract_end):   # iterate all positions
				frame_dict = masks_raw.get(fi, {})
				m = frame_dict.get(obj_id) if isinstance(frame_dict, dict) else None
				if m is None:
					m = torch.zeros(ref_h, ref_w) if ref_h is not None else None
					if m is None:
						continue
				if isinstance(m, np.ndarray):
					m = torch.from_numpy(m)
				if m.ndim == 3:
					m = m.squeeze(0)
				if ref_h is None:
					ref_h, ref_w = m.shape[-2], m.shape[-1]
				frame_tensors.append(m.float())
			if frame_tensors:
				stacked = torch.stack(frame_tensors)
				if stacked.max() > 1.0:
					stacked = stacked / 255.0
				per_obj.append(stacked)
				actual_obj_ids.append(instance_idx)

		print(f'[extract_direct] Format A — extracted {len(per_obj)} instance tensor(s), '
		      f'each length {per_obj[0].shape[0] if per_obj else 0}')
		return per_obj, actual_obj_ids

	# ------------------------------------------------------------------ #
	# Format B: dict-of-arrays  {frame_idx: ndarray/tensor(N, H, W)}     #
	# ------------------------------------------------------------------ #
	if isinstance(first_val, (np.ndarray, torch.Tensor)):
		probe_tensor = torch.from_numpy(first_val) if isinstance(first_val, np.ndarray) else first_val
		if probe_tensor.ndim == 4:
			probe_tensor = probe_tensor.squeeze(0)

		n_instances = 1 if probe_tensor.ndim == 2 else probe_tensor.shape[0]
		n_instances = min(n_instances, max_objects)
		ref_h, ref_w = probe_tensor.shape[-2], probe_tensor.shape[-1]

		print(f'[extract_direct] Format B — {n_instances} instance(s) × {target_len} frames '
		      f'({len(slice_frames)} with masks, {target_len - len(slice_frames)} zero-padded)')

		# Build per-instance lists of (H, W) uint8 frame tensors instead of a
		# monolithic (target_len, N, H, W) tensor.  The old approach called
		# torch.stack() on a list of (N, H, W) frames which (a) held the full
		# N-instance block for every frame in memory simultaneously, (b)
		# torch.stack() created a second copy, doubling peak RAM before the
		# source list could be freed, and (c) the resulting views via
		# stacked_all[:, i, :, :] kept the whole (target_len, N, H, W) tensor
		# live for the duration of the call.
		#
		# Instead: distribute each frame's per-instance slices into n_instances
		# separate buckets immediately, then stack each bucket independently.
		# Peak overhead is now one extra (H, W) uint8 tensor per instance per
		# torch.stack() call — a factor of ~N×4 smaller than the old float32 approach.
		per_instance_frames = [[] for _ in range(n_instances)]

		for frame_idx in range(extract_start, extract_end):
			raw_frame = masks_raw.get(frame_idx)
			if raw_frame is None:
				# Zero-pad missing frames — one small (H, W) uint8 tensor per instance
				for instance_idx in range(n_instances):
					per_instance_frames[instance_idx].append(
						torch.zeros(ref_h, ref_w, dtype=torch.uint8)
					)
				continue

			# Zero-copy view for numpy (mmap-backed arrays from SAM3).
			# Only clone GPU tensors, which need to be detached from CUDA memory.
			if isinstance(raw_frame, np.ndarray):
				frame_masks_tensor = torch.from_numpy(raw_frame.copy())		# ← break mmap alias (otherwise: always frame 0 is inferenced->written to ffmpeg)
			else:
				frame_masks_tensor = raw_frame.clone() if raw_frame.is_cuda else raw_frame

			if frame_masks_tensor.ndim == 4:
				frame_masks_tensor = frame_masks_tensor.squeeze(0)
			if frame_masks_tensor.ndim == 2:
				frame_masks_tensor = frame_masks_tensor.unsqueeze(0)

			# Normalise to uint8 [0, 255] without ever going through float32.
			# SAM3 can produce either:
			#   • bool/uint8 in {0, 1}   → scale up by ×255
			#   • uint8 in {0, 255}      → already correct, keep as-is
			#   • float32 in [0.0, 1.0]  → scale up then cast (unavoidable, but rare)
			if frame_masks_tensor.dtype == torch.bool:
				frame_masks_u8 = frame_masks_tensor.to(torch.uint8) * 255
			elif frame_masks_tensor.dtype == torch.uint8:
				# Check whether values are already in [0,255] or binary [0,1].
				# A cheap proxy: if the max over the whole frame is ≤ 1, the mask
				# is binary 0/1 uint8 and needs scaling; otherwise it's already
				# in the [0,255] range.
				if frame_masks_tensor.max().item() <= 1:
					frame_masks_u8 = frame_masks_tensor * 255
				else:
					frame_masks_u8 = frame_masks_tensor   # zero-copy, already correct
			else:
				# Floating-point or other dtype — convert via scaling.
				# This path is uncommon; we accept the temporary float allocation here.
				frame_masks_u8 = (
					frame_masks_tensor.float().mul_(255).clamp_(0, 255).to(torch.uint8)
				)

			# Normalise instance count
			actual_n = frame_masks_u8.shape[0]
			if actual_n > n_instances:
				frame_masks_u8 = frame_masks_u8[:n_instances]
			elif actual_n < n_instances:
				padding = torch.zeros(n_instances - actual_n, ref_h, ref_w, dtype=torch.uint8)
				frame_masks_u8 = torch.cat([frame_masks_u8, padding], dim=0)

			# Distribute to per-instance buckets.
			# .contiguous() ensures each slice is its own allocation so that
			# frame_masks_u8 (and the underlying numpy mmap view) can be freed
			# immediately after this loop body — without it, the slices would
			# hold a reference to frame_masks_u8.
			for instance_idx in range(n_instances):
				per_instance_frames[instance_idx].append(
					frame_masks_u8[instance_idx].contiguous()
				)
			del frame_masks_tensor, frame_masks_u8  # release ASAP

		if not any(per_instance_frames):
			print('[extract_direct] Format B — no frames collected')
			return [], []

		# Stack per-instance: peak cost is 2× one (target_len, H, W) uint8 tensor at a time.
		per_obj = []
		for instance_idx in range(n_instances):
			instance_frame_list = per_instance_frames[instance_idx]
			if instance_frame_list:
				per_obj.append(torch.stack(instance_frame_list))  # (target_len, H, W) uint8
			per_instance_frames[instance_idx] = None  # release bucket immediately after stacking
		actual_obj_ids = list(range(len(per_obj)))

		print(f'[extract_direct] Format B — extracted {len(per_obj)} instance tensor(s) '
		      f'dtype={per_obj[0].dtype if per_obj else "n/a"}, '
		      f'each length {per_obj[0].shape[0] if per_obj else 0}')
		return per_obj, actual_obj_ids

	print(f'[extract_direct] Unrecognised format: {type(first_val).__name__}')
	return [], []


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

import hashlib

def color_name_to_hue(color_str):
	"""Convert a CSS color name or #RRGGBB / #RGB hex string to an HLS hue in [0, 1].
	Returns None if the color cannot be parsed.
	"""
	_NAMED = {
		'red':     (255, 0,   0),
		'green':   (0,   128, 0),
		'lime':    (0,   255, 0),
		'blue':    (0,   0,   255),
		'yellow':  (255, 255, 0),
		'cyan':    (0,   255, 255),
		'aqua':    (0,   255, 255),
		'magenta': (255, 0,   255),
		'fuchsia': (255, 0,   255),
		'orange':  (255, 165, 0),
		'purple':  (128, 0,   128),
		'pink':    (255, 192, 203),
		'teal':    (0,   128, 128),
		'maroon':  (128, 0,   0),
		'navy':    (0,   0,   128),
		'olive':   (128, 128, 0),
		'coral':   (255, 127, 80),
		'gold':    (255, 215, 0),
		'indigo':  (75,  0,   130),
		'violet':  (238, 130, 238),
		'brown':   (165, 42,  42),
		'crimson': (220, 20,  60),
	}
	s = color_str.strip().lower()
	if s in _NAMED:
		r, g, b = _NAMED[s]
	elif s.startswith('#') and len(s) in (4, 7):
		try:
			if len(s) == 4:   # #RGB shorthand
				r = int(s[1] * 2, 16)
				g = int(s[2] * 2, 16)
				b = int(s[3] * 2, 16)
			else:
				r = int(s[1:3], 16)
				g = int(s[3:5], 16)
				b = int(s[5:7], 16)
		except ValueError:
			return None
	else:
		return None
	h, _l, _s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
	return h


def label_text_to_hue(text):
	"""Derive a stable, well-distributed hue in [0, 1] from a label string.

	Uses the first 4 bytes of an MD5 digest so that the same label always
	produces the same hue across runs, and different labels (e.g. "window" vs
	"zebra crossing") produce visually distinct hues.
	"""
	digest = hashlib.md5(text.encode('utf-8', errors='replace')).digest()
	value  = int.from_bytes(digest[:4], 'big')
	return value / 0xFFFF_FFFF

#@profile
def build_label_color_map(obj_ids, video_state, color_step=10, master_hue=None, device='cpu'):
	"""
	Build a mapping {instance_idx: color_tensor} with label-aware shading.

	Strategy
	--------
	• Read ``video_state.prompts`` to discover how many distinct text labels
	  exist (e.g. "column" = 1 label, "statue, column" = 2 labels).
	• If *master_hue* is given (a float in [0, 1]):  ALL labels/instances use
	  that hue; only lightness varies per instance.
	  → ``--overlay-color red`` → every instance is a shade of red.
	• If *master_hue* is None:  each unique label gets a hue derived by
	  hashing its text (via ``label_text_to_hue``), so "window" always gets
	  the same color and "zebra crossing" always gets a different one.
	  → reproducible per-label colors without any manual configuration.
	• Within a hue group, lightness is stepped down by ``color_step/255``
	  per instance so individual objects are visually separable.
	• obj_ids are the 0-based instance indices returned by
	  ``extract_per_object_masks_direct``.

	Args:
		obj_ids:      0-based list of int instance indices.
		video_state:  SAM3VideoState with a ``.prompts`` attribute.
		color_step:   lightness decrement per instance within the same hue,
		              expressed in [0, 255] — matches ``--color-step``.
		master_hue:   HLS hue in [0, 1] to use for ALL instances, or None to
		              derive per-label hues from prompt text hashing.
		device:       torch device for the returned tensors.

	Returns:
		dict {instance_idx (int): tensor(3,) float32 in [0, 255]  (R, G, B)}
	"""
	# --- 0. Diagnostic: print raw prompts so the user can verify -----------
	raw_prompts = getattr(video_state, 'prompts', [])
	print(f'[color_map] video_state.prompts has {len(raw_prompts)} entry/entries:')
	for p in raw_prompts:
		print(f'  [color_map]   {vars(p) if hasattr(p, "__dict__") else repr(p)}')

	# --- 1. Collect unique label texts in prompt order ---------------------
	labels_seen = []   # ordered unique label strings
	for p in raw_prompts:
		# VideoPrompt stores text in .data (a tuple), not .text
		# Try .text first for forward-compat, then .data[0]
		text = (getattr(p, 'text',        None) or
		        getattr(p, 'label',       None) or
		        getattr(p, 'text_prompt', None) or
		        getattr(p, 'prompt',      None))
		if text is None:
			data = getattr(p, 'data', None)
			if isinstance(data, (list, tuple)) and len(data) > 0:
				text = data[0]
		if isinstance(text, str) and text.strip():
			label_text = text.strip()
			if label_text not in labels_seen:
				labels_seen.append(label_text)

	n_labels    = len(labels_seen)
	n_instances = len(obj_ids)

	print(f'[color_map] Detected {n_labels} label(s): {labels_seen}')
	print(f'[color_map] Assigning colors to {n_instances} instance(s) with color_step={color_step}')

	step_l = color_step / 255.0   # lightness step in [0, 1]

	# --- 2. No label info — hue-spread fallback ----------------------------
	if n_labels == 0:
		print('[color_map] No label text found — using evenly-spread hue fallback')
		n = max(n_instances, 1)
		result = {}
		for i, oid in enumerate(obj_ids):
			h = master_hue if master_hue is not None else (i / n)
			l = max(0.25, 0.70 - (i % max(n, 1)) * step_l)
			s = 0.90
			r, g, b = colorsys.hls_to_rgb(h, l, s)
			result[oid] = torch.tensor([r * 255, g * 255, b * 255],
			                           dtype=torch.float32, device=device)
			print(f'  [color_map] instance={i} HLS=({h:.2f},{l:.2f},{s:.2f}) '
			      f'RGB=({int(r*255)},{int(g*255)},{int(b*255)})')
		return result

	# --- 3. Assign hues: one base hue per label ----------------------------
	if master_hue is not None:
		# Explicit master color: all labels collapsed to a single hue.
		# Instances vary only in lightness → "40 shades of red", etc.
		label_hues = [master_hue] * n_labels
		print(f'[color_map] master_hue={master_hue:.3f} → all {n_labels} label(s) share this hue')
	else:
		# No master color: derive a stable per-label hue from the label text.
		# "window" will always produce the same hue; "zebra crossing" a different one.
		label_hues = [label_text_to_hue(lbl) for lbl in labels_seen]
		for lbl, h in zip(labels_seen, label_hues):
			print(f'  [color_map] label="{lbl}" → hash-derived hue={h:.3f}')

	# Distribute instances round-robin across labels so that, for example,
	# with 2 labels and 40 instances, instances 0,2,4… get label-0 hue and
	# instances 1,3,5… get label-1 hue.
	# With 1 label, ALL instances get the same hue (most common case).
	label_instance_counter = [0] * n_labels   # how many instances already assigned per label

	color_map = {}
	for pos, oid in enumerate(obj_ids):
		label_idx = pos % n_labels            # round-robin label assignment
		count     = label_instance_counter[label_idx]
		label_instance_counter[label_idx] += 1

		h = label_hues[label_idx]
		l = max(0.20, 0.70 - count * step_l)
		s = 0.90
		r, g, b = colorsys.hls_to_rgb(h, l, s)
		color_map[oid] = torch.tensor([r * 255, g * 255, b * 255],
		                              dtype=torch.float32, device=device)
		label_name = labels_seen[label_idx]
		print(f'  [color_map] instance={pos} label="{label_name}" count={count} '
		      f'HLS=({h:.2f},{l:.2f},{s:.2f}) '
		      f'RGB=({int(r*255)},{int(g*255)},{int(b*255)})')

	return color_map

#@profile
def get_instance_colors_torch(n_instances, n_classes=1, device='cpu'):
	"""
	Legacy helper — generates maximally distinct colors by spreading hues evenly.
	Kept for backwards compatibility; the streaming loop now uses
	``build_label_color_map`` instead.

	Returns:
		list of n_instances tensors, each shape (3,) float32 in [0, 255] (R, G, B).
	"""
	colors = []
	for i in range(max(n_instances, 1)):
		h = (i / max(n_instances, 1))
		l = 0.35 + (i % 5) * 0.07
		s = 0.85
		r, g, b = colorsys.hls_to_rgb(h, l, s)
		colors.append(torch.tensor([r * 255, g * 255, b * 255],
		                           dtype=torch.float32, device=device))
	return colors

#@profile
def generate_instance_overlay_torch(per_obj_masks, frames, colors,
                                    overlay_alpha=0.5, device='cpu'):
    B, H, W, _ = frames.shape

    # uint8 storage throughout: 4× less memory than float32.
    # Bring everything to CPU first — SAM3 has already saturated the GPU
    # budget; compositing is pure arithmetic that doesn't need the GPU.
    result        = frames.cpu().mul(255).clamp_(0, 255).to(torch.uint8)  # (B,H,W,3)
    colored_mask  = torch.zeros(B, H, W, 3, dtype=torch.uint8)
    combined_mask = torch.zeros(B, H, W,    dtype=torch.bool)

    # 7-bit fixed-point alpha: scale of 128 keeps every product within int16.
    # (max intermediate: 255 × 128 = 32640 < 32767 = int16 max)
    alpha_i = int(round(overlay_alpha * 128))   # e.g. 64 for α=0.5

    for instance_idx, obj_mask in enumerate(per_obj_masks):
        color_u8 = colors[instance_idx % len(colors)].cpu().to(torch.uint8)   # (3,) uint8

        # obj_mask arrives as uint8 [0, 255] from extract_per_object_masks_direct.
        # No mul(255) needed — that was the mirror of the now-removed float cast.
        # Accept float input defensively in case the fallback path sends it.
        if obj_mask.dtype == torch.uint8:
            mask_u8 = obj_mask.cpu()
        else:
            mask_u8 = obj_mask.cpu().mul(255).clamp_(0, 255).to(torch.uint8)

        mask_broadcast = mask_u8.unsqueeze(-1).to(torch.int16)   # (B,H,W,1)

        # Per-pixel weight in [0, alpha_i], inv_weight in [128-alpha_i, 128].
        # The pair always sums to 128, so the >> 7 rescale is exact.
        weight     = (mask_broadcast * alpha_i) >> 8                       # (B,H,W,1)
        inv_weight = 128 - weight                                          # (B,H,W,1)

        # int16 blend — no overflow, no float.
        result = (
            result.to(torch.int16) * inv_weight
            + color_u8.view(1, 1, 1, 3).to(torch.int16) * weight
        ).bitwise_right_shift_(7).clamp_(0, 255).to(torch.uint8)

        colored_mask = torch.where(
            mask_broadcast > 0,                    # broadcasts (B,H,W,1) over C
            color_u8.view(1, 1, 1, 3),             # broadcasts (1,1,1,3) over B,H,W
            colored_mask,
        )
        combined_mask.logical_or_(mask_u8 > 0)

    # Ship the final compact uint8 outputs to the target device once.
    return (result.to(device),
            combined_mask.to(torch.uint8).to(device),
            colored_mask.to(device))

def validate_output_paths_and_create_directories(debug, output_mask_path, output_overlay_path):
	"""
	Ensure output paths are provided (when not in debug mode) and that their
	parent directories exist on disk.  Raises ValueError on missing paths.
	"""
	# Validate output paths if not in debug mode
	if not debug:
		if output_mask_path is None or output_overlay_path is None:
			raise ValueError("output_mask_path and output_overlay_path required when debug=False")
		# Ensure output directories exist
		os.makedirs(os.path.dirname(os.path.abspath(output_mask_path)) or '.', exist_ok=True)
		os.makedirs(os.path.dirname(os.path.abspath(output_overlay_path)) or '.', exist_ok=True)


def resolve_master_hue_from_overlay_color(overlay_color):
	"""
	Convert the human-readable overlay_color string (e.g. 'red') to a hue
	float in [0, 1] used when tinting instance masks.

	Returns:
		master_hue (float | None): None means per-label prompt-hash coloring.
	"""
	# Resolve master hue once, before the chunk loop.
	# overlay_color=None  → per-label prompt-hash coloring
	# overlay_color='red' → all instances are shades of red
	if overlay_color is not None:
		master_hue = color_name_to_hue(overlay_color)
		if master_hue is None:
			print(f'[color] WARNING: could not parse overlay_color="{overlay_color}" '
			      f'— falling back to prompt-hash coloring')
		else:
			print(f'[color] overlay_color="{overlay_color}" → master_hue={master_hue:.3f}')
	else:
		master_hue = None
		print('[color] No --overlay-color specified — using prompt-hash coloring per label')
	return master_hue


def open_video_and_resolve_processing_dimensions(
		video_path, start_frame, num_frames, one_every_n_frames,
		custom_width, custom_height, tile, tile_resize, fps):
	"""
	Open the video capture, compute effective frame counts (accounting for
	subsampling and the num_frames limit), and derive the inference resolution
	(w × h) after applying tile / custom-size overrides.

	Returns:
		cap          : cv2.VideoCapture (caller must release)
		total_frames : int — how many kept-frames will be processed
		start_frame  : int — start offset in kept-frame units
		w, h         : int — inference width / height
		orig_w, orig_h : int — original video width / height (used for output)
		fps          : float — frames-per-second for the output video
	"""
	print(f'Opening input video: {video_path}')
	#cap = cv2.VideoCapture(video_path)
	cap = FFmpegCapture(video_path)
	if not cap.isOpened():
		print(f"Could not open video: {video_path}")
		return None, 0, 0, 0, 0, 0, 0, 0

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(f'Video has {total_frames} source frames')

	# Convert start_frame (source-frame units, from --start-frame) to
	# kept-frame units so all subsequent chunk arithmetic is consistent.
	if one_every_n_frames > 1:
		start_frame  = start_frame  // one_every_n_frames
		total_frames = total_frames // one_every_n_frames
		print(f'Frame subsampling: keeping 1 every {one_every_n_frames} source frames '
		      f'→ {total_frames} effective frames to process')

	if num_frames > 0:
		total_frames = min(num_frames, total_frames)
	print(f'About to process {total_frames} frames')

	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(f'Video resolution: {w}x{h}')

	# Original video resolution — always used for the output videos so that the
	# full frame is preserved (tile region + zero-padded surroundings).
	orig_w, orig_h = w, h

	if tile is not None:
		tx, ty, tw, th = tile
		# Validate tile bounds against the original video size
		if tx < 0 or ty < 0 or tx + tw > orig_w or ty + th > orig_h:
			raise ValueError(
				f'Tile ({tx},{ty},{tw},{th}) falls outside the video '
				f'dimensions {orig_w}x{orig_h}'
			)
		# Inference runs on the tile — ignore custom_width/height when tiling
		# (the tile already defines the region of interest).
		w, h = tw, th
		if tile_resize is not None:
			trw, trh = tile_resize
			w, h = trw, trh
			print(f'Tile active: x={tx} y={ty} w={tw} h={th}  →  resized to {w}x{h} for inference, '
			      f'masks resized back to {tw}x{th}, output at {orig_w}x{orig_h} (zero-padded)')
		else:
			print(f'Tile active: x={tx} y={ty} w={tw} h={th}  →  inference at {w}x{h}, '
			      f'output at {orig_w}x{orig_h} (zero-padded)')
	elif custom_width > 0 or custom_height > 0:
		if w > custom_width:
			w = custom_width
		if h > custom_height:
			h = custom_height
		print(f'Setting custom resolution: {w}x{h}')

	# Get actual FPS from source if not specified
	if fps is None or fps == 0:
		fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
	print(f'Video FPS: {fps}')

	return cap, total_frames, start_frame, w, h, orig_w, orig_h, fps


def initialize_coco_json_structure(coco_json_path, text_prompt, video_path):
	"""
	Build the skeleton COCO-compatible dict (info / licenses / categories /
	images / annotations) from the text_prompt's OR-separated labels, or
	return None when coco_json_path is not set.

	Returns:
		coco_data (dict | None)
	"""
	if coco_json_path is None:
		return None

	raw_labels = [lbl.strip() for lbl in text_prompt.split(' OR ') if lbl.strip()]
	categories = [
		{'id': idx + 1, 'name': lbl, 'supercategory': 'object'}
		for idx, lbl in enumerate(raw_labels or [text_prompt])
	]
	coco_data = {
		'info': {
			'description':  'SAM3 Video Segmentation',
			'version':      '1.0',
			'date_created': datetime.now().isoformat(),
			'video_path':   str(video_path),
			'prompt':       text_prompt,
		},
		'licenses':    [],
		'categories':  categories,
		'images':      [],
		'annotations': [],
	}
	print(f'[COCO] JSON output enabled → {coco_json_path}')
	print(f'[COCO] {len(categories)} category/categories: {[c["name"] for c in categories]}')
	return coco_data


def embed_frame_into_tile_padded_canvas(frame_tensor, tile, orig_w, orig_h):
	"""
	Place a tile-sized frame tensor into a zero-padded canvas that matches the
	original video resolution.

	Args:
		frame_tensor : (th, tw, 3) torch.Tensor
		tile         : (tx, ty, tw, th)
		orig_w, orig_h : full output resolution

	Returns:
		padded (orig_h, orig_w, 3) torch.Tensor of the same dtype
	"""
	tx, ty, tw, th = tile
	padded = torch.zeros(orig_h, orig_w, 3, dtype=frame_tensor.dtype)
	padded[ty:ty + th, tx:tx + tw] = frame_tensor
	return padded


def collect_coco_annotations_for_single_frame(
		coco_data, coco_ann_counter,
		per_obj_sliced, actual_obj_ids,
		abs_frame_idx, tile, tile_resize,
		orig_w, orig_h, w, h):
	"""
	Append one images entry and zero-or-more annotations entries to coco_data
	for a single output frame.  Handles tile-resize-back and zero-padding of
	per-instance masks before polygon / bbox extraction.

	Returns:
		coco_ann_counter (int) — updated running annotation ID.
	"""
	out_w = orig_w if tile is not None else w
	out_h = orig_h if tile is not None else h
	coco_data['images'].append({
		'id':        abs_frame_idx,
		'file_name': f'frame_{abs_frame_idx:06d}.png',
		'width':     out_w,
		'height':    out_h,
	})

	n_cats = len(coco_data['categories'])
	for inst_pos, (inst_mask_seq, obj_id) in enumerate(zip(per_obj_sliced, actual_obj_ids)):
		raw_inst = inst_mask_seq.cpu()  # (H_inf, W_inf)
		if raw_inst.dtype == torch.uint8:
			inst_u8 = raw_inst
		else:
			inst_u8 = (raw_inst.float() * 255).clamp_(0, 255).to(torch.uint8)

		if tile is not None and tile_resize is not None:
			tx, ty, tw, th = tile
			inst_4d = inst_u8.float().unsqueeze(0).unsqueeze(0)
			inst_4d = torch.nn.functional.interpolate(inst_4d, size=(th, tw), mode='nearest-exact')
			inst_u8 = inst_4d.squeeze().clamp_(0, 255).to(torch.uint8)

		if tile is not None:
			tx, ty, tw, th = tile
			padded_inst = torch.zeros(orig_h, orig_w, dtype=torch.uint8)
			padded_inst[ty:ty + th, tx:tx + tw] = inst_u8
			inst_u8 = padded_inst

		binary_np = (inst_u8.numpy() > 0).astype(np.uint8)
		if binary_np.max() == 0:
			continue
		polygons = mask_to_polygons(binary_np)
		if not polygons:
			continue
		bbox, area = mask_to_bbox_area(binary_np)
		cat_id = coco_data['categories'][inst_pos % n_cats]['id']
		coco_ann_counter += 1
		coco_data['annotations'].append({
			'id':           coco_ann_counter,
			'image_id':     abs_frame_idx,
			'category_id':  cat_id,
			'segmentation': polygons,
			'bbox':         bbox,
			'area':         area,
			'iscrowd':      0,
			'instance_id':  obj_id,
		})

	return coco_ann_counter


def cleanup_sam3_session_and_release_chunk_memory(
		sam3_model, video_state,
		chunk_frames, masks, per_obj_sliced, f_gpu,
		overlay, uint8_masks, colored_masks, curr_ch_frames, raw_mask_chunk,
		mask_cpu=None, overlay_cpu=None, colored_masks_cpu=None,
		debug=False):
	"""
	Close the SAM3 inference session for the current chunk, purge its
	temporary mmap directories, delete all large per-chunk tensors, and
	release GPU + system memory back to the OS.
	"""
	session_uuid = getattr(video_state, 'session_uuid', None)
	tmp_dir      = getattr(video_state, 'tmp_dir', None)

	print(f'Entering region with session_uuid: {session_uuid} - hasattr(sam3_model, "video_predictor")')
	print(f'[cleanup] session_uuid={session_uuid}')
	print(f'[cleanup] sam3_model type: {type(sam3_model).__name__}')
	print(f'[cleanup] sam3_model attrs: {[a for a in dir(sam3_model) if not a.startswith("__")]}')

	if session_uuid is not None:
		try:
			print(f'[cleanup] trying with sam3_model.close_session({session_uuid})')
			sam3_model.close_session(session_uuid)
			print(f'[cleanup] sam3_model.close_session({session_uuid}) OK')
		except Exception as e:
			print(f'[cleanup] close_session raised: {e} — trying manual purge')
			vp = getattr(sam3_model, '_video_predictor', None)
			if vp is not None:
				inference_state = getattr(vp, 'inference_state', None)
				if isinstance(inference_state, dict) and session_uuid in inference_state:
					del inference_state[session_uuid]
					print(f'[cleanup] Manually deleted from _video_predictor.inference_state')
				for attr in ('output_dict', 'maskmem_features', 'maskmem_pos_enc',
				             'output_dict_per_obj', 'temp_output_dict_per_obj',
				             'frames_already_tracked', 'cached_features'):
					if hasattr(vp, attr):
						old = getattr(vp, attr)
						setattr(vp, attr, {} if isinstance(old, dict) else None)
						print(f'[cleanup] Reset _video_predictor.{attr}')

	for d in glob.glob('/tmp/sam3_*'):
		try:
			shutil.rmtree(d)
			print(f'[cleanup] Removed mmap\'d tmpdir: {d}')
		except Exception as e:
			print(f'[cleanup] Could not remove {d}: {e}')

	del chunk_frames, video_state, masks, per_obj_sliced, f_gpu, overlay, uint8_masks, colored_masks, curr_ch_frames, raw_mask_chunk
	if not debug:
		del mask_cpu, overlay_cpu, colored_masks_cpu

	torch.cuda.empty_cache()
	gc.collect()
	try:
		ctypes.CDLL('libc.so.6').malloc_trim(0)
	except Exception:
		pass


def finalize_ffmpeg_outputs_and_save_coco_json(
		mask_stdin, mask_process, output_mask_path,
		overlay_stdin, overlay_process, output_overlay_path,
		coco_data, coco_json_path):
	"""
	Flush and close both FFmpeg streaming processes, then (if requested) write
	the accumulated COCO JSON to disk.  Safe to call even when the processes
	were never opened (None guards are in place).
	"""
	if mask_stdin is not None:
		finalize_ffmpeg(mask_stdin, mask_process)
		print(f"Mask video saved to: {output_mask_path}")
	if overlay_stdin is not None:
		finalize_ffmpeg(overlay_stdin, overlay_process)
		print(f"Overlay video saved to: {output_overlay_path}")

	if coco_data is not None and coco_json_path is not None:
		try:
			os.makedirs(os.path.dirname(os.path.abspath(coco_json_path)) or '.', exist_ok=True)
			with open(coco_json_path, 'w', encoding='utf-8') as _f:
				json.dump(coco_data, _f)
			print(f'[COCO] JSON saved to: {coco_json_path}  '
			      f'({len(coco_data["images"])} images, '
			      f'{len(coco_data["annotations"])} annotations)')
		except Exception as _e:
			print(f'[COCO] WARNING: could not write JSON to {coco_json_path}: {_e}')


#@profile
def process_video_streaming(video_path, sam3_model, text_prompt,
                            start_frame=0, num_frames=0,
                            custom_width=0, custom_height=0, center_crop=False,
                            chunk_size=50, overlap=5,
                            overlay_color='red', overlay_alpha=0.5,
                            max_objects=10,
                            base_path=None, output_mask_path=None, output_overlay_path=None,
                            fps=29.97, tile=None, tile_resize=None,
                            one_every_n_frames=1,
                            coco_json_path=None,
                            output_master_path=None,
                            debug=False):
	"""
	Process video with SAM3 and stream outputs to disk via FFmpeg.
	
	Args:
		video_path: Input video path
		sam3_model: SAM3 model instance
		text_prompt: Text prompt for segmentation
		num_frames: Limit processing to N *kept* frames (0 = all)
		chunk_size: Frames per chunk (in kept-frame units)
		overlap: Frame overlap between chunks (in kept-frame units)
		overlay_color: Color for mask overlay
		overlay_alpha: Alpha for overlay blending
		output_mask_path: Path for binary mask video output (required if debug=False)
		output_overlay_path: Path for overlay video output (required if debug=False)
		fps: Output video framerate
		one_every_n_frames: Keep 1 source frame, discard the next N-1.
		                    All other frame counts are in kept-frame units.
		                    start_frame is in *source*-frame units and is
		                    converted internally.
		coco_json_path: If set, per-frame per-instance predictions are written to
		                this path as a COCO-compatible JSON file.  Each frame
		                becomes an ``images`` entry; each non-empty instance mask
		                becomes an ``annotations`` entry with polygon segmentation,
		                bounding box, and area fields.
		output_master_path: When set (and one_every_n_frames > 1), write a
		                    "master" video that contains only the kept source
		                    frames at the original resolution with no SAM3
		                    processing applied.  Useful as the reference clip
		                    for downstream overlay / tile-reassembly scripts.
		debug: If True, accumulate masks/overlays in RAM and return them
		
	Returns:
		If debug=True: (final_masks, final_overlay) tensors
		If debug=False: None (outputs written to disk)
	"""
	global starting_free_gb

	validate_output_paths_and_create_directories(debug, output_mask_path, output_overlay_path)

	print(f'Opening input video: {video_path}')
	master_hue = resolve_master_hue_from_overlay_color(overlay_color)

	cap, total_frames, start_frame, w, h, orig_w, orig_h, fps = \
		open_video_and_resolve_processing_dimensions(
			video_path, start_frame, num_frames, one_every_n_frames,
			custom_width, custom_height, tile, tile_resize, fps)

	if cap is None:
		return (None, None) if debug else None

	coco_data        = initialize_coco_json_structure(coco_json_path, text_prompt, video_path)
	coco_ann_counter = 0

	# Pre-allocate debug tensors only if needed
	final_masks   = None
	final_overlay = None
	if debug:
		print("-= WARNING! DEBUG MODE ENABLED: Accumulating results in RAM, watch out for OOMs =-")
		# 1. Pre-allocate only the results (on CPU in uint8 to save RAM)
		# Mask: (B, H, W), Overlay: (B, H, W, 3)
		final_masks   = torch.zeros((total_frames, h, w),    dtype=torch.uint8, device='cpu')
		final_overlay = torch.zeros((total_frames, h, w, 3), dtype=torch.uint8, device='cpu')

	if center_crop:
		ffmpeg_out_w = custom_width
		ffmpeg_out_h = custom_height
		print(f"  Using cropped dimensions: {ffmpeg_out_w}x{ffmpeg_out_h}")
	else:
		ffmpeg_out_w = orig_w
		ffmpeg_out_h = orig_h
		print(f"  Using original dimensions: {ffmpeg_out_w}x{ffmpeg_out_h}")

	# Setup FFmpeg streaming outputs
	mask_stdin, mask_process       = None, None
	overlay_stdin, overlay_process = None, None
	mask_video_frame_count         = 0

	print(f"Starting FFmpeg streaming to:")
	print(f"  Masks: {output_mask_path}")
	print(f"  Overlay: {output_overlay_path}")
		
	# For mask video: grayscale, but we'll write RGB and convert or use gray8
	# Actually, let's use ffv1 for lossless mask storage, or just raw grayscale
	mask_stdin, mask_process = start_ffmpeg_streaming(
		output_mask_path, ffmpeg_out_w, ffmpeg_out_h, fps,
		#codec='ffv1', 		# Lossless for masks
		codec='libx265',
		pix_fmt='yuv420p',	# Single channel for masks
		#crf=0
		crf=23
	)
		
	overlay_stdin, overlay_process = start_ffmpeg_streaming(
		output_overlay_path, ffmpeg_out_w, ffmpeg_out_h, fps,
		codec='libx265',
		pix_fmt='yuv420p',
		crf=23
	)

	# Master video: original-resolution subsampled clip (no SAM3 processing).
	# Only created when the caller requests it AND subsampling is active.
	master_stdin       = None
	master_process     = None
	if output_master_path is not None and one_every_n_frames > 1:
			os.makedirs(os.path.dirname(os.path.abspath(output_master_path)) or '.', exist_ok=True)
			master_stdin, master_process = start_ffmpeg_streaming(
				output_master_path, ffmpeg_out_w, ffmpeg_out_h, fps,
				codec='libx265',
				pix_fmt='yuv420p',
				crf=23
			)
			print(f'  Master: {output_master_path}')

	num_chunks = (total_frames + chunk_size - 1) // chunk_size
	print(f'About to process {num_chunks} chunks of {chunk_size} frames each')

	if base_path is not None and args.write_images:
		masks_dir = Path(str(base_path)) / 'masks'
		rgb_dir   = Path(str(base_path)) / 'rgb'
		masks_dir.mkdir(parents=True, exist_ok=True)
		rgb_dir.mkdir(parents=True, exist_ok=True)

	try:
		for chunk_idx in range(num_chunks):
			actual_start = (chunk_idx * chunk_size) + start_frame
			actual_end   = min(actual_start + chunk_size, start_frame + total_frames)
			#print(f'{actual_start+chunk_size} vs. {total_frames}')
			print(f"{20 * '='} Processing chunk {chunk_idx + 1}/{num_chunks} ({actual_start} to {actual_end})")

			# Include overlap for SAM context
			init_start = max(0, actual_start - overlap) if chunk_idx > 0 else actual_start
			init_end   = min(actual_end + overlap, start_frame + total_frames)
			print(f'Loading frames {init_start} to {init_end}')
			load_count = init_end - init_start

			# 2. LOAD ONLY THE CHUNK
			chunk_frames = get_video_chunk(
				cap, init_start, load_count,
				custom_width=w, custom_height=h,
				center_crop_param=center_crop,
				tile=tile,
				one_every_n_frames=one_every_n_frames,
				overlap=overlap,
				master_stdin=master_stdin,
				is_last_chunk=(chunk_idx == num_chunks - 1))
			if chunk_frames is None:
				break
			print(f'Loaded {chunk_frames.shape[0]} frames with shape {chunk_frames.shape[1:]}')
			print(f'Center-crop is: {"enabled (= no resize)" if args.center_crop else "disabled (resize will be performed instead)"}')

			# 3. SEGMENTATION
			video_state = run_segmentation(chunk_frames, text_prompt)
			prop_start  = actual_start - init_start
			prop_end    = prop_start + (actual_end - actual_start)

			try:
				masks, video_state = propagate_masks_chunk(
					sam3_model, video_state,
					start_frame=prop_start, end_frame=prop_end,
					offload_model=False)
			except torch.cuda.OutOfMemoryError:
				torch.cuda.empty_cache()
				masks, video_state = propagate_masks_chunk(
					sam3_model, video_state,
					start_frame=prop_start, end_frame=prop_end,
					offload_model=True)

			# 4. OVERLAY CALCULATION (per-instance, pure torch)
			raw_mask_chunk, _ = get_sam3_outputs(masks, video_state)

			# Squeeze to the exact range we need (removing overlap context)
			extract_start  = actual_start - init_start
			extract_end    = extract_start + (actual_end - actual_start)
			curr_ch_frames = chunk_frames[extract_start:extract_end]
			f_gpu          = curr_ch_frames.to(sam3_model.device)

			# Read per-object masks DIRECTLY from the propagation dict — no extract() loop
			per_obj_sliced, actual_obj_ids = extract_per_object_masks_direct(
				masks, extract_start, extract_end, max_objects=max_objects)
			if not per_obj_sliced:
				# Fallback: treat the combined mask as a single object
				print('[WARN] Direct per-object extraction failed – falling back to single merged mask.')
				per_obj_sliced = [raw_mask_chunk[extract_start:extract_end].float()]
				actual_obj_ids = [1]

			# Assign a visually distinct color per instance, grouped by label
			print(f'[color] Building label-aware color map for {len(per_obj_sliced)} instance(s)...')
			color_map = build_label_color_map(
				actual_obj_ids, video_state,
				color_step=args.color_step,
				master_hue=master_hue,
				device=sam3_model.device)
			# Convert to an ordered list aligned with per_obj_sliced
			instance_colors = [color_map[oid] for oid in actual_obj_ids]

			with torch.no_grad():
				overlay, uint8_masks, colored_masks = generate_instance_overlay_torch(
					[m.to(sam3_model.device) for m in per_obj_sliced],
					f_gpu,
					colors=instance_colors,
					overlay_alpha=overlay_alpha,
					device=sam3_model.device)
				# overlay        : (B, H, W, 3) uint8  – blended frame  (RGB)
				# uint8_masks    : (B, H, W)    uint8  – binary combined mask
				# colored_masks  : (B, H, W, 3) uint8  – per-instance colored mask (RGB)

			# 5. STREAM OR STORE
			if debug:
				# Store in RAM (original behavior)
				final_masks[actual_start:actual_end]   = uint8_masks.cpu()
				final_overlay[actual_start:actual_end] = overlay.cpu()
			else:
				# Stream to FFmpeg (frame by frame to minimize memory)
				# Move to CPU once and iterate
				mask_cpu          = uint8_masks.cpu()   # (B, H, W)
				overlay_cpu       = overlay.cpu()       # (B, H, W, 3)
				colored_masks_cpu = colored_masks.cpu() # (B, H, W, 3)

				for i in range(overlay_cpu.shape[0]):
					# Write mask as grayscale (expand dims for ffmpeg gray format)
					frame         = curr_ch_frames[i] * 255 # because get_video_chunk() does torch.from_numpy() / 255
					frame         = cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_BGR2RGB)
					mask_frame    = colored_masks_cpu[i]  # (H, W, 3) RGB uint8
					overlay_frame = overlay_cpu[i]  # (H, W, 3)

					# tile-resize: scale masks back to original tile dimensions
					# This must happen BEFORE zero-padding so the canvas slice is the
					# right size.  nearest-exact is integer-safe for binary masks;
					# bilinear would soften edges but is not needed here.
					if tile is not None and tile_resize is not None:
						tx, ty, tw, th = tile
						def _resize_back(t):
							"""(H_inf, W_inf, 3) uint8  →  (th, tw, 3) uint8"""
							x = t.float().permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
							x = torch.nn.functional.interpolate(x, size=(th, tw), mode='nearest-exact')
							return x.squeeze(0).permute(1, 2, 0).to(torch.uint8)
						mask_frame    = _resize_back(mask_frame)
						overlay_frame = _resize_back(overlay_frame)

					# tile: embed into zero-padded canvas of original resolution
					if tile is not None:
						mask_frame    = embed_frame_into_tile_padded_canvas(mask_frame,    tile, orig_w, orig_h)
						overlay_frame = embed_frame_into_tile_padded_canvas(overlay_frame, tile, orig_w, orig_h)

					# COCO JSON annotation collection
					if coco_data is not None:
						abs_frame_idx    = actual_start + i
						per_obj_at_frame = [seq[i] for seq in per_obj_sliced]
						coco_ann_counter = collect_coco_annotations_for_single_frame(
							coco_data, coco_ann_counter,
							per_obj_at_frame, actual_obj_ids,
							abs_frame_idx, tile, tile_resize,
							orig_w, orig_h, w, h)

					if base_path is not None and args.write_images:
						# cv2.imwrite expects BGR — convert from our RGB tensor
						mask_frame_bgr = cv2.cvtColor(mask_frame.numpy(), cv2.COLOR_RGB2BGR)
						cv2.imwrite(f'{masks_dir}/{base_path.name}-mask-{mask_video_frame_count}.png', mask_frame_bgr)
						if debug:
							print(f'Saved {masks_dir}/{base_path.name}-mask-{mask_video_frame_count}.png')

					write_frame_to_ffmpeg(mask_stdin, mask_frame)
					mask_video_frame_count += 1
					if debug:
						print(f'{mask_frame.shape = }')
						print(f'{mask_frame.dtype = }')
					
					# Write overlay
					write_frame_to_ffmpeg(overlay_stdin, overlay_frame)
					if debug:
						print(f'{overlay_frame.shape = }')
						print(f'{overlay_frame.dtype = }')
					if base_path is not None and args.write_images:
						cv2.imwrite(f'{rgb_dir}/{base_path.name}-rgb-{mask_video_frame_count}.jpg',
						            frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
						if debug:
							print(f'Saved {rgb_dir}/{base_path.name}-rgb-{mask_video_frame_count}.jpg')

				print(f'{20 * "="} Produced {overlay_cpu.shape[0]} frames with shape {overlay_cpu.shape[1:]} and {mask_cpu.shape[0]} masks with shape {mask_cpu.shape[1:]}')
				
				#mask_stdin.flush()
				overlay_stdin.flush()

			# 6. CLEANUP
			cleanup_sam3_session_and_release_chunk_memory(
				sam3_model, video_state,
				chunk_frames, masks, per_obj_sliced, f_gpu,
				overlay, uint8_masks, colored_masks, curr_ch_frames, raw_mask_chunk,
				mask_cpu=(None if debug else mask_cpu),
				overlay_cpu=(None if debug else overlay_cpu),
				colored_masks_cpu=(None if debug else colored_masks_cpu),
				debug=debug)

			print(f"{20 * '='} Processed chunk {chunk_idx + 1}/{num_chunks} ({actual_start} to {actual_end})\n\n")

	finally:
		finalize_ffmpeg_outputs_and_save_coco_json(
			mask_stdin, mask_process, output_mask_path,
			overlay_stdin, overlay_process, output_overlay_path,
			coco_data, coco_json_path)
		if master_stdin is not None:
			finalize_ffmpeg(master_stdin, master_process)
			print(f'Master video saved to: {output_master_path}')
		cap.release()

	print(f'Written {mask_video_frame_count} frames to {output_mask_path}')

	if debug:
		return final_masks, final_overlay
	else:
		return None

def run_segmentation(video_frames, text_prompt):
	print(f"Initializing segmentation with prompt: '{text_prompt}'...")
	seg_node = sam3_nodes.SAM3VideoSegmentation()
	video_state = seg_node.segment(
		video_frames=video_frames,
		prompt_mode="text",
		text_prompt=text_prompt,
		frame_idx=0,
		score_threshold=0.3
	)
	return nodeoutput_to_type(video_state)

def propagate_masks_chunk(model, video_state, start_frame=0, end_frame=-1, offload_model=False):
	"""
	Propagate masks for a specific chunk of frames.
	
	Args:
		model: SAM3 model
		video_state: Current video state
		start_frame: Start frame index (relative to current state)
		end_frame: End frame index (-1 for all remaining)
		offload_model: Whether to offload model to CPU after propagation
	
	Returns:
		(masks, updated_video_state)
	"""
	print(f"  Propagating masks from frame {start_frame} to {end_frame}...")
	prop_node = sam3_nodes.SAM3Propagate()
	result = prop_node.propagate(
		sam3_model=model,
		video_state=video_state,
		start_frame=start_frame,
		end_frame=end_frame,
		direction="forward",
		offload_model=offload_model
	)
	# Returns (masks, scores, video_state)
	return result[0], result[2]

def get_sam3_outputs(masks, video_state):
    print("Generating mask images...")
    out_node = sam3_nodes.SAM3VideoOutput()
    result = out_node.extract(
        masks=masks,
        video_state=video_state,
        obj_id=-1,
        plot_all_masks=True
    )
    # Returns (masks, frames, visualization)
    return result[0], result[2]  # masks, visualization

# --- Main Execution ---
def patch_max_num_objects(model, limit, _seen=None):
	"""
	Recursively walk a SAM3 model object and lower every ``max_num_objects``
	attribute it finds to *limit*.

	SAM3's default is 10,000.  During propagation it computes a pairwise IoU
	matrix of shape [N, N, H*W] (bool) entirely on GPU; with N≈500+ objects
	(common for text prompts like "window" on city footage) that single
	allocation easily exceeds 19 GB and causes OOM.  This patch caps N before
	propagation starts, regardless of which sub-object in the hierarchy holds
	the attribute.

	Typical paths observed in practice (any/all may be present depending on
	SAM3 version):
	    model.max_num_objects
	    model.model.max_num_objects
	    model.video_predictor.max_num_objects
	    model.video_predictor.model.max_num_objects
	"""
	if _seen is None:
		_seen = set()
	obj_id = id(model)
	if obj_id in _seen:
		return
	_seen.add(obj_id)

	patched_paths = []
	# Patch the attribute on this object if present and above our limit
	if hasattr(model, 'max_num_objects'):
		old = model.max_num_objects
		if old != limit:
			model.max_num_objects = limit
			patched_paths.append(f'{type(model).__name__}.max_num_objects  {old} → {limit}')

	# Recurse into well-known child attributes (avoid iterating all parameters)
	for attr in ('model', 'video_predictor', 'image_encoder', 'mask_decoder'):
		child = getattr(model, attr, None)
		if child is not None and not isinstance(child, (int, float, str, bool, torch.Tensor)):
			patched_paths.extend(patch_max_num_objects(child, limit, _seen))

	return patched_paths


def main():
	# 1. Load SAM3 model
	sam3_model = load_sam3_model()

	# Patch SAM3's internal max_num_objects cap BEFORE any propagation.
	# The default (10000) causes an O(N²) GPU allocation in mask_iou() that
	# easily OOMs on scenes with many detected instances (e.g. city windows).
	patched = patch_max_num_objects(sam3_model, args.max_num_objects)
	if patched:
		print(f'[max_num_objects] Patched {len(patched)} location(s):')
		for p in patched:
			print(f'  {p}')
	else:
		print(f'[max_num_objects] WARNING: no max_num_objects attribute found on model — '
		      f'patch did not apply. OOM risk remains if SAM3 detects many objects.')
	
	# 2. Handle naming
	base_name		= Path(args.output_dir) / Path(args.input_video).stem

	# Parse optional tile and build a descriptive suffix for output filenames
	tile		= None
	tile_suffix	= ''
	if args.tile:
		try:
			tx, ty, tw, th = map(int, args.tile.split(','))
		except ValueError:
			raise ValueError(f'--tile must be four comma-separated integers: x,y,w,h  (got: "{args.tile}")')
		tile        = (tx, ty, tw, th)
		tile_suffix = f'-tile-x{tx}-y{ty}-w{tw}-h{th}'
		print(f'Tile mode: x={tx} y={ty} w={tw} h={th}')

	tile_resize = None
	if args.tile_resize:
		if tile is None:
			raise ValueError('--tile-resize requires --tile to be set')
		try:
			trw, trh = map(int, args.tile_resize.lower().split('x'))
		except ValueError:
			raise ValueError(f'--tile-resize must be WxH (e.g. 1008x1008)  (got: "{args.tile_resize}")')
		tile_resize  = (trw, trh)
		tile_suffix += f'-resize-{trw}x{trh}'
		print(f'Tile resize: {trw}x{trh}')

	mask_output_name	= f"{base_name}-mask{tile_suffix}-prompt-{args.prompt}.mkv"
	overlay_output_name	= f"{base_name}-overlay{tile_suffix}-prompt-{args.prompt}.mp4"
	coco_json_output_name	= (
		f"{base_name}-coco{tile_suffix}-prompt-{args.prompt}.json"
		if args.save_coco_json else None
	)

	# Master video: original-resolution clip subsampled by one_every_n_frames,
	# produced only when subsampling is active so it has actual value.
	master_output_name = None
	if args.one_every_n_frames > 1:
		master_output_name = (
			f"{base_name}-master{tile_suffix}"
			f"-1every{args.one_every_n_frames}frames.mp4"
		)

	one_every_n_frames_str	= f"-1every{args.one_every_n_frames}" if args.one_every_n_frames > 1 else ""
	base_path		= f"{base_name}-images{tile_suffix}-prompt-{args.prompt}{one_every_n_frames_str}"

	# 3. Process everything in one memory-managed loop
	process_video_streaming_output = process_video_streaming(
		video_path=args.input_video,
		sam3_model=sam3_model,
		text_prompt=args.prompt,
		start_frame=args.start_frame,
		num_frames=args.num_frames,
		custom_width=args.custom_width,
		custom_height=args.custom_height,
		center_crop=args.center_crop,
		chunk_size=args.chunk_size,
		overlap=args.chunk_overlap,
		overlay_color=args.overlay_color,
		overlay_alpha=args.overlay_alpha,
		max_objects=args.max_objects,
		base_path=base_path,
		output_mask_path=mask_output_name,
		output_overlay_path=overlay_output_name,
		fps=0,
		tile=tile,
		tile_resize=tile_resize,
		one_every_n_frames=args.one_every_n_frames,
		coco_json_path=coco_json_output_name,
		output_master_path=master_output_name,
		debug=args.debug
	)
	if args.debug and process_video_streaming_output is not None:
		final_mask_tensor, overlay_tensor = process_video_streaming_output
	
	print(f'\n')
	print(f'Finished!')
	print(f"Mask video   : {mask_output_name}")
	print(f"Overlay video: {overlay_output_name}")
	if master_output_name is not None:
		print(f"Master video : {master_output_name}")
	if coco_json_output_name is not None:
		print(f"COCO JSON    : {coco_json_output_name}")

if __name__ == "__main__":

	# Setup environment
	comfyui_nodes_dir = setup_comfyui_environment(comfyui_base_dir)

	# Import the class
	comfyui_sam3_module, sam3_nodes = import_sam3_package(comfyui_nodes_dir)
	
	main()

