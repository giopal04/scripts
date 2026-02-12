#!/usr/bin/env python3

# Run with:

# ./comfyui-SAM3-video-segmentation-gemini-v3.py --input-video /mnt/raid1/repos/scripts/segmentation/sam3/sam3-video-dataset/statues/China-Tian-Tan-Buddha-Hl9TCsAIbdI-3840x2160-30fps-60s-1825.mp4 --prompt statue --custom-width 1080 --custom-height 1080 --chunk-size 50 --num-frames 150 --write-images --center-crop

# OR

# ./comfyui-SAM3-video-segmentation-gemini-v3.py --input-video /mnt/raid1/repos/scripts/segmentation/sam3/sam3-video-dataset/statues/Indonesia-Garuda-Wisnu-Kencana-dzC4FnMpcuk-1280x720-30fps-32s-985.mp4 --prompt statue --custom-width 1920 --custom-height 1080 --chunk-size 50 --num-frames 150 --write-images

# OR even better

# ./comfyui-SAM3-video-segmentation-gemini-v3.py --input-video /mnt/raid1/repos/scripts/segmentation/sam3/sam3-video-dataset/statues/China-Tian-Tan-Buddha-Hl9TCsAIbdI-3840x2160-30fps-60s-1825.mp4 --prompt "statue OR monument OR column" --custom-width 1920 --custom-height 1080 --chunk-size 100 --chunk-overlap 10 --num-frames 300 --write-images




# Remember than AV1 is not well supported by OpenCV, you'll likely hit bug 11389 (https://github.com/opencv/opencv/issues/11389)
# Convert webm files to h264/h265 with ffmpeg first

# Also make sure to have plenty of space in /tmp because SAM3 requires several gigabytes of free space for its temporary files (e.g. /tmp/sam3_3be2b85c_fa088o3p/mmap_output/{frames.mmap, masks.mmap, vis.mmap})

import os

import sys
import types
import argparse

import psutil
import numpy as np

from pathlib import Path

import gc
import cv2

import subprocess

parser = argparse.ArgumentParser()
#parser.add_argument('--input_image',	type=str,				help='Input image path')
#parser.add_argument('--input_dir',	type=str,				help='Input image directory')
parser.add_argument('--input-video',	type=str,	required=True,		help='Path to input video file. Note: ComfyUI doesn\'t allow writing output files outside its output directory.')
parser.add_argument('--gpu',		type=str,	default='1',		help='GPU where to run the workflow')
#parser.add_argument('--output_image',	type=str,				help='Output image path')
parser.add_argument('--output-dir',	type=str,	default='/tmp',		help='Output directory (for both videos and images)')
parser.add_argument('--prompt',		type=str,	required=True,		help='Text prompt for SAM3')
parser.add_argument('--num-frames',	type=int,	default=10,		help='Total number of frames to process (0 = all)')
parser.add_argument('--chunk-size',	type=int,	default=50,		help='Number of frames per chunk to avoid OOM')
parser.add_argument('--chunk-overlap',	type=int,	default=5,		help='Overlap between chunks for smooth transitions')
parser.add_argument('--custom-width',	type=int,	default=0,		help='Resize input video to this width before processing')
parser.add_argument('--custom-height',	type=int,	default=0,		help='Resize input video to this height before processing')
parser.add_argument('--overlay-color',	type=str,	default='red',		help='Overlay color for segmentation (e.g. red, green, blue, yellow, cyan, magenta)')
parser.add_argument('--overlay-alpha',	type=float,	default=0.5,		help='Transparency of overlay (0.0-1.0)')
parser.add_argument('--center-crop',	action='store_true',			help='Center-crop input video instead of resizing')
parser.add_argument('--write-images',	action='store_true',			help='Also write images with videos (useful for creating standard image datasets)')
parser.add_argument('--offload-model',	action='store_true',			help='Offload model to CPU between chunks')
parser.add_argument('--debug',		action='store_true',			help='Show debug information')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f'Using GPU: {args.gpu}')

# Path to your ComfyUI installation
comfyui_base_dir  = Path('/mnt/raid1/repos/comfyui')
comfyui_nodes_dir = '' # let's just pretend we don't know where it is...

# global placeholder for comfyui_sam3 module
comfyui_sam3_module	= None
sam3_nodes		= None
# global placeholder for comfyui_vhs  module
comfyui_vhs_module	= None

import torch

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

def nodeoutput_to_type(some_nodeoutput, label="", debug=False):
	label=f' as {label}' if label != "" else ''
	if isinstance(some_nodeoutput, latest.io.NodeOutput):
		if debug:
			print(f'nodeoutput_to_type() received a NodeOutput{label}, getting result...')
		result = some_nodeoutput.result
		if isinstance(result, tuple):
			if debug:
				print(f'Result is a tuple, getting first element...')
			res_item = result[0]
			if isinstance(res_item, torch.Tensor):
				if debug:
					print(f'Result[0] is a Tensor with shape: {res_item.shape}')
			else:
				if debug:
					print(f'Result[0] is something else: {type(res_item)}')
			return res_item
		else:
			if debug:
				print(f'Result is not a tuple: {type(result)}')
			return result
	elif isinstance(some_nodeoutput, tuple):
		if debug:
			print(f'nodeoutput_to_type() received a tuple ({some_nodeoutput}){label}, getting first element...')
		res_item = some_nodeoutput[0]
		if isinstance(res_item, torch.Tensor):
			if debug:
				print(f'Result is a Tensor with shape: {res_item.shape}')
		else:
			if debug:
				print(f'Result is something else: {type(res_item)}')
		return res_item
	else:
		if debug:
			print(f'nodeoutput_to_type() received something else: {type(some_nodeoutput)}{label}')
		return some_nodeoutput

# --- Import Functions ---

import importlib.util

def setup_comfyui_environment(comfyui_base_dir):
	"""Setup environment similar to what ComfyUI does"""
	
	# Set up environment variables that some nodes might expect
	os.environ['COMFYUI_PATH'] = str(comfyui_base_dir)
	
	try:
		'''
		# why should we do that?!? It's already there, initialized and all...
		# the only problem is: we know where ComfyUI is installed or not?
		folder_paths.set_base_folder(comfyui_base_dir)
		'''
		comfyui_nodes_dir = Path(str(folder_paths.folder_names_and_paths["custom_nodes"][0][0]))
		sys.path.append(str(comfyui_nodes_dir))
		print(f'Found ComfyUI nodes dir at: {comfyui_nodes_dir}')
	except ImportError as e:
		print(f'Unable to import folder_paths, error: {e}')
	except ModuleNotFoundError as e:
		print(f'Unable to import folder_paths, error: {e}')
	except:
		print(f'Unable to import folder_paths, unknown error')
	return comfyui_nodes_dir

def import_sam3_package(comfyui_nodes_dir, debug=False):
	global comfyui_sam3_module
	global sam3_nodes

	"""Import SAM3Propagate class"""
	sam3_dir	= comfyui_nodes_dir / 'comfyui_sam3'
	sam3_nodes_dir	= sam3_dir / 'nodes'
	
	# Add the node directory to path
	if str(sam3_dir) not in sys.path:
		sys.path.append(str(sam3_dir))
	if str(sam3_dir) not in os.environ['PYTHONPATH']:
		os.environ['PYTHONPATH'] += ':' + str(sam3_dir)
	if str(sam3_nodes_dir) not in sys.path:
		sys.path.append(str(sam3_nodes_dir))
	if str(sam3_nodes_dir) not in os.environ['PYTHONPATH']:
		os.environ['PYTHONPATH'] += ':' + str(sam3_nodes_dir)

	if debug:
		print(f'sys.path  : {sys.path}')
		print(f'PYTHONPATH: {os.environ["PYTHONPATH"]}')
	
	# Try different import strategies
	try:
		# Strategy 1: Direct import
		import comfyui_sam3
		from comfyui_sam3.nodes import sam3_video_nodes

		'''
		# import comfyui_sam3 always throws something like this:

		Traceback (most recent call last):
		  File "/mnt/raid1/repos/comfyui/custom_nodes/comfyui_sam3/__init__.py", line 93, in <module>
		    from . import sam3_server
		  File "/mnt/raid1/repos/comfyui/custom_nodes/comfyui_sam3/sam3_server.py", line 17, in <module>
		    from server import PromptServer
		  File "/mnt/raid1/repos/comfyui/server.py", line 34, in <module>
		    from app.frontend_management import FrontendManager, parse_version
		ModuleNotFoundError: No module named 'app.frontend_management'
		'''

		print(f"Successfully imported: comfyui_sam3 (Strategy 1)")

		comfyui_sam3_module	= comfyui_sam3
		sam3_nodes		= sam3_video_nodes
		return

	except ModuleNotFoundError as e:
		print(f'Something happened while importing comfyui_sam3 or sam3_video_nodes, error: {e}')
	except ImportError:
		# Untested
		try:
			print(f"Failed to import SAM3Propagate with Strategy 1...")
			# Strategy 2: Import from comfyui_sam3 package
			import comfyui_sam3
			from comfyui_sam3.sam3_video_nodes import SAM3Propagate
			print(f"Successfully imported: {SAM3Propagate} (Strategy 2)")
			return SAM3Propagate
		except ImportError:
			print(f"Failed to import SAM3Propagate with Strategy 2...")
			import comfyui_sam3

			# Strategy 3: Import from file
			module_path = sam3_dir / 'nodes' / 'sam3_video_nodes.py'
			
			# Read the file to check for any relative imports
			with open(str(module_path), 'r') as f:
				content = f.read()
			
			# Create a custom module
			spec   = importlib.util.spec_from_file_location("sam3_video_nodes", module_path)
			module = importlib.util.module_from_spec(spec)
			
			# Execute the module with modified globals to handle relative imports
			module.__dict__['__file__']	= module_path
			module.__dict__['__package__'] = 'comfyui_sam3'
			
			spec.loader.exec_module(module)
			print(f"Successfully imported: {SAM3Propagate} (Strategy 3)")
			
			return module.SAM3Propagate
		except:
			print(f"Failed to import comfyui_sam3 with Strategy 3, giving up...")
			return None
	return

def ensure_namespace_exists(module_name, debug=False):
	"""
	Ensures all parent parts of a module path exist in sys.modules.
	e.g., if module_name is 'app.assets.api.routes', it ensures 
	'app', 'app.assets', and 'app.assets.api' are in sys.modules.
	"""
	parts = module_name.split('.')
	for i in range(1, len(parts)):
		parent_name = ".".join(parts[:i])
		if debug:
			print(f'Checking if namespace exists for: {parent_name}')
		if parent_name not in sys.modules:
			# Create a dummy module to act as a namespace package
			new_module = types.ModuleType(parent_name)
			# Namespace packages don't have a __file__, but they have a __path__
			new_module.__path__ = [] 
			sys.modules[parent_name] = new_module
			if debug:
				print(f"Created namespace placeholder: {parent_name}")

def load_python_module_from_filename(module_name, file_path, debug=False):
	# e.g.
	# module_name	= 'app.frontend_management'
	# file_path	= comfyui_base_dir / 'app/frontend_management.py'

	spec		= importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None:
		raise ImportError(f'Failed to import {module_name} from {file_path}')
	else:
		if debug:
			print(f'Imported {module_name} from {file_path}, obtained a ModuleSpec object with name: {spec.name}')
	module		= importlib.util.module_from_spec(spec)
	if debug:
		print(f'Imported {module_name} from {file_path}, obtained a Module object with name: {module.__name__}')
	sys.modules[module_name] = module			# this is fundamental for resolving dependencies, otherwise previous modules won't be found
	spec.loader.exec_module(module)
	ensure_namespace_exists(module_name, debug=debug)	# this is important to also create any parent namespaces that don't exist until now
	return module

def patch_prompt_server(server):
	from unittest.mock import MagicMock

	# 2. Assign it to the class attribute 'instance'
	if not hasattr(server.PromptServer, 'instance'):
		print("Patching PromptServer.instance...")
		server.PromptServer.instance = MagicMock()

def import_vhs_package(comfyui_nodes_dir, debug=False):
	global comfyui_vhs_module

	"""Import and return the VHS_LoadVideo function from Video Helper Suite nodes"""
	
	# Find Video Helper Suite (VHS) directory
	vhs_dir = None
	potential_vhs_dirs = [
		comfyui_nodes_dir / 'ComfyUI-VideoHelperSuite',
		comfyui_nodes_dir / 'VideoHelperSuite',
		comfyui_nodes_dir / 'vhsuite',
	]
	
	# Also check subdirectories
	for node_dir in comfyui_nodes_dir.iterdir():
		if node_dir.is_dir() and 'video' in node_dir.name.lower() and 'helper' in node_dir.name.lower():
			potential_vhs_dirs.append(node_dir)
	
	for dir_path in potential_vhs_dirs:
		if dir_path.exists():
			vhs_dir = dir_path
			print(f'Found VHS directory at: {vhs_dir}')
			break
	
	if not vhs_dir:
		print('Warning: Video Helper Suite not found in custom nodes directory')
		# Try to find it anywhere in sys.path
		for path in sys.path:
			path_obj = Path(path)
			if path_obj.exists() and 'video' in path_obj.name.lower() and 'helper' in path_obj.name.lower():
				vhs_dir = path_obj
				print(f'Found VHS directory in sys.path: {vhs_dir}')
				break
	
	if not vhs_dir:
		raise ImportError('Video Helper Suite not found. Please install it first.')
	
	# Add VHS directory to Python path
	if str(vhs_dir) not in sys.path:
		sys.path.append(str(vhs_dir))
	if debug:
		print(f'Added VHS directory to system path: {sys.path}')
	
	# Try to import VHS_LoadVideo
	VHS_LoadVideo = None

	module_name	= 'utils.install_util'
	file_path	= comfyui_base_dir / 'utils/install_util.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.logger'
	file_path	= comfyui_base_dir / 'app/logger.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.frontend_management'
	file_path	= comfyui_base_dir / 'app/frontend_management.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.database.db'
	file_path	= comfyui_base_dir / 'app/database/db.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.helpers'
	file_path	= comfyui_base_dir / 'app/assets/helpers.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.database.models'
	file_path	= comfyui_base_dir / 'app/database/models.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.database.models'
	file_path	= comfyui_base_dir / 'app/assets/database/models.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.database.tags'
	file_path	= comfyui_base_dir / 'app/assets/database/tags.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.database.bulk_ops'
	file_path	= comfyui_base_dir / 'app/assets/database/bulk_ops.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.scanner'
	file_path	= comfyui_base_dir / 'app/assets/scanner.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.api.schemas_in'
	file_path	= comfyui_base_dir / 'app/assets/api/schemas_in.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.api.schemas_out'
	file_path	= comfyui_base_dir / 'app/assets/api/schemas_out.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.database.queries'
	file_path	= comfyui_base_dir / 'app/assets/database/queries.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.manager'
	file_path	= comfyui_base_dir / 'app/assets/manager.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.app_settings'
	file_path	= comfyui_base_dir / 'app/app_settings.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.user_manager'
	file_path	= comfyui_base_dir / 'app/user_manager.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.assets.api.routes'
	file_path	= comfyui_base_dir / 'app/assets/api/routes.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.model_manager'
	file_path	= comfyui_base_dir / 'app/model_manager.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'utils.json_util'
	file_path	= comfyui_base_dir / 'utils/json_util.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.subgraph_manager'
	file_path	= comfyui_base_dir / 'app/subgraph_manager.py'
	load_python_module_from_filename(module_name, file_path)

	module_name	= 'app.custom_node_manager'
	file_path	= comfyui_base_dir / 'app/custom_node_manager.py'
	load_python_module_from_filename(module_name, file_path)

	import server
	from server import PromptServer

	# Call the patch to add .instance to PromptServer
	patch_prompt_server(server)

	module_name	= 'ComfyUI-VideoHelperSuite'
	file_path	= comfyui_nodes_dir / 'ComfyUI-VideoHelperSuite/__init__.py'
	vhs_module	= load_python_module_from_filename(module_name, file_path)
	# Strategy 1: Direct import from nodes module
	try:

		vhs_module.VHS_LoadVideo	= vhs_module.NODE_CLASS_MAPPINGS['VHS_LoadVideo']
		vhs_module.VHS_VideoInfo	= vhs_module.NODE_CLASS_MAPPINGS['VHS_VideoInfo']
		vhs_module.VHS_VideoCombine	= vhs_module.NODE_CLASS_MAPPINGS['VHS_VideoCombine']

		print(f'Executed {module_name} from {file_path}')
		comfyui_vhs_module = vhs_module
		return
	except ImportError as e:
		print(f'Failed to import ComfyUI_VideoHelperSuite, error (ImportError): {e}')
	except ModuleNotFoundError as e:
		print(f'Failed to import ComfyUI_VideoHelperSuite, error (ModuleNotFoundError): {e}')
	
	raise ImportError('Could not find or import Video Helper Suite module, giving up...')

def print_free_ram():
	"""Print total free RAM in human-readable format."""
	memory = psutil.virtual_memory()
	free_bytes = memory.available  # Actually usable memory (not just completely free)
	
	# Convert to GB for readability
	free_gb = free_bytes / (1024 ** 3)
	print(f"Free RAM: {free_gb:.2f} GB ({free_bytes:,} bytes)")
	return free_gb

# --- Workflow Functions ---

def load_sam3_model(model_name="sam3.pt"):
    print(f"Loading SAM3 model: {model_name}...")
    loader = comfyui_sam3_module.nodes.load_model.LoadSAM3Model()
    # Paths are typically relative to ComfyUI/models/sam3/
    model = loader.load_model(model_path=f"models/sam3/{model_name}")
    return nodeoutput_to_type(model)

def center_crop(tensor, target_h, target_w=None):
    """
    Center crop a tensor of shape [N, H, W, C] to [N, target_h, target_w, C].
    
    Args:
        tensor: Input tensor of shape [N, H, W, C]
        target_h: Target height
        target_w: Target width (if None, uses target_h for square crop)
    
    Returns:
        Cropped tensor of shape [N, target_h, target_w, C]
    """
    if target_w is None:
        target_w = target_h
    
    n, h, w, c = tensor.shape
    
    # Calculate crop start positions (centered)
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    # Ensure we don't go out of bounds
    start_h = max(0, start_h)
    start_w = max(0, start_w)
    
    # Perform the crop
    cropped = tensor[:, start_h:start_h + target_h, start_w:start_w + target_w, :]
    
    return cropped

'''
# Example usage
x = torch.randn(55, 1080, 1920, 3)
cropped = center_crop(x, 1080, 1080)
print(cropped.shape)  # torch.Size([55, 1080, 1080, 3])

# Verify: 1920 - 1080 = 840, so 420 removed from each side
print(f"Removed from left/right: {(1920 - 1080) // 2}")  # 420
'''


# replaces load_video()
def get_video_chunk(cap, start_frame, count, custom_width=0, custom_height=0, center_crop_param=False, debug=False):
	"""Fetches a specific range of frames from OpenCV VideoCapture."""
	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	frames = []
	for _ in range(count):
		ret, frame = cap.read()
		if not ret:
			break
		if (custom_width > 0 or custom_height > 0) and not center_crop_param:
			frame = cv2.resize(frame, (custom_width, custom_height))
		# Convert BGR to RGB and scale to 0-1 for the model
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(frame)
	
	if not frames:
		return None
		
	# Stack into [B, H, W, 3] and convert to float32 tensor
	#return torch.from_numpy(np.stack(frames)).float() / 255.0
	'''
		custom_nodes/comfyui_sam3/nodes/sam3_video_nodes.py:117 
		"tooltip": "Video frames as batch of images [N, H, W, C]"
	'''
	if debug:
		print(f'{torch.from_numpy(np.stack(frames)).shape = }')
	#tensor_out = torch.from_numpy(np.stack(frames))
	tensor_out = torch.from_numpy(np.stack(frames)).float() / 255.0
	if center_crop_param:
		tensor_out = center_crop(tensor_out, target_h=custom_height, target_w=custom_width)
	return tensor_out


def start_ffmpeg_streaming(output_path, width, height, fps, codec='libx265', pix_fmt='yuv420p', crf=23, debug=True):
	"""
	Start an FFmpeg process in streaming mode for writing video frames.
	Returns the stdin pipe and the process object.
	"""
	if codec == 'ffv1':
		crf = 0
		command = [
			'ffmpeg',
			'-y',  # Overwrite output file if it exists
			'-f', 'rawvideo',
			'-vcodec', 'rawvideo',
			'-s', f'{width}x{height}',  # size of one frame
			#'-pix_fmt', str(pix_fmt),
			'-pix_fmt', 'rgb24',
			'-r', str(fps),  # frames per second
			'-i', '-',  # The input comes from a pipe
			'-an',  # No audio
			'-c:v', codec,
			'-context', '0', '-level', '3',
			'-pix_fmt', pix_fmt,
			'-crf', str(crf),
			'-preset', 'fast',
			output_path
		]
	else:
		command = [
			'ffmpeg',
			'-y',  # Overwrite output file if it exists
			'-f', 'rawvideo',
			'-vcodec', 'rawvideo',
			'-s', f'{width}x{height}',  # size of one frame
			'-pix_fmt', 'rgb24',
			'-r', str(fps),  # frames per second
			'-i', '-',  # The input comes from a pipe
			'-an',  # No audio
			'-c:v', codec,
			'-pix_fmt', pix_fmt,
			'-crf', str(crf),
			'-preset', 'fast',
			output_path
		]

	print(f'{10 * "-"} Starting FFmpeg process with output: {output_path} (codec: {codec}, resolution: {width}x{height} @ {fps}fps - quality: {crf})')
	if debug:
		print(f'FFmpeg command: {command}')
	process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
	return process.stdin, process

def write_frame_to_ffmpeg(stdin, frame_tensor):
	"""
	Write a single frame (HWC uint8 tensor) to FFmpeg stdin.
	frame_tensor should be on CPU, uint8, shape (H, W, 3)
	"""
	# Ensure contiguous memory layout and convert to numpy
	if isinstance(frame_tensor, torch.Tensor):
		frame_np = frame_tensor.cpu().contiguous().numpy()
	else:
		frame_np = frame_tensor
	
	# Write raw bytes to FFmpeg
	stdin.write(frame_np.tobytes())
	stdin.flush()

def finalize_ffmpeg(stdin, process):
	"""
	Close stdin and wait for FFmpeg to finish.
	"""
	stdin.close()
	process.wait()
	if process.returncode != 0:
		stderr = process.stderr.read().decode()
		raise RuntimeError(f"FFmpeg failed with code {process.returncode}: {stderr}")

def process_video_streaming(video_path, sam3_model, text_prompt, num_frames=0,
							custom_width=0, custom_height=0, center_crop=False,
							chunk_size=50, overlap=5, 
							overlay_color='red', overlay_alpha=0.5,
							base_path=None, output_mask_path=None, output_overlay_path=None,
							fps=29.97, debug=False):
	"""
	Process video with SAM3 and stream outputs to disk via FFmpeg.
	
	Args:
		video_path: Input video path
		sam3_model: SAM3 model instance
		text_prompt: Text prompt for segmentation
		num_frames: Limit processing to N frames (0 = all)
		chunk_size: Frames per chunk
		overlap: Frame overlap between chunks
		overlay_color: Color for mask overlay
		overlay_alpha: Alpha for overlay blending
		output_mask_path: Path for binary mask video output (required if debug=False)
		output_overlay_path: Path for overlay video output (required if debug=False)
		fps: Output video framerate
		debug: If True, accumulate masks/overlays in RAM and return them
		
	Returns:
		If debug=True: (final_masks, final_overlay) tensors
		If debug=False: None (outputs written to disk)
	"""
	global starting_free_gb

	# Validate output paths if not in debug mode
	if not debug:
		if output_mask_path is None or output_overlay_path is None:
			raise ValueError("output_mask_path and output_overlay_path required when debug=False")
		# Ensure output directories exist
		os.makedirs(os.path.dirname(os.path.abspath(output_mask_path)) or '.', exist_ok=True)
		os.makedirs(os.path.dirname(os.path.abspath(output_overlay_path)) or '.', exist_ok=True)

	# OOM again, but remember that now here: custom_nodes/comfyui_sam3/nodes/sam3_video_nodes.py we have everything mmapped, vis_frames as well!

	print(f'Opening input video: {video_path}')
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Could not open video: {video_path}")
		return None if not debug else (None, None)
	
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(f'Video has {total_frames} frames')
	
	if num_frames > 0:
		total_frames = min(num_frames, total_frames)

	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(f'Video resolution: {w}x{h}')
	if custom_width > 0 or custom_height > 0:
		w = custom_width
		h = custom_height
		print(f'Setting custom resolution: {w}x{h}')

	# Get actual FPS from source if not specified
	if fps is None or fps == 0:
		fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
	print(f'Video FPS: {fps}')
	
	# Pre-allocate debug tensors only if needed
	final_masks	= None
	final_overlay	= None
	if debug:
		print("-= WARNING! DEBUG MODE ENABLED: Accumulating results in RAM, watch out for OOMs =-")
		# 1. Pre-allocate only the results (on CPU in uint8 to save RAM)
		# Mask: (B, H, W), Overlay: (B, H, W, 3)
		final_masks	= torch.zeros((total_frames, h, w), dtype=torch.uint8, device='cpu')
		final_overlay	= torch.zeros((total_frames, h, w, 3), dtype=torch.uint8, device='cpu')
	
	# Setup FFmpeg streaming outputs
	mask_stdin, mask_process	= None, None
	overlay_stdin, overlay_process	= None, None
	mask_video_frame_count		= 0
	
	print(f"Starting FFmpeg streaming to:")
	print(f"  Masks: {output_mask_path}")
	print(f"  Overlay: {output_overlay_path}")
		
	# For mask video: grayscale, but we'll write RGB and convert or use gray8
	# Actually, let's use ffv1 for lossless mask storage, or just raw grayscale
	mask_stdin, mask_process = start_ffmpeg_streaming(
		output_mask_path, w, h, fps, 
		#codec='ffv1', 		# Lossless for masks
		codec='libx265',
		#pix_fmt='gray',	# Single channel for masks
		pix_fmt='yuv420p',	# Single channel for masks
		#crf=0
		crf=23
	)
		
	overlay_stdin, overlay_process = start_ffmpeg_streaming(
		output_overlay_path, w, h, fps,
		codec='libx265',
		pix_fmt='yuv420p',
		crf=23
	)
	
	color_map	= {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1]}
	selected_color	= torch.tensor(color_map.get(overlay_color, [1, 0, 0])).view(1, 1, 1, 3)
	
	num_chunks = (total_frames + chunk_size - 1) // chunk_size

	if base_path is not None and args.write_images:
		masks_dir	= Path(str(base_path)+'-images') / 'masks'
		rgb_dir		= Path(str(base_path)+'-images') / 'rgb'
		masks_dir.mkdir(parents=True, exist_ok=True)
		rgb_dir.mkdir(parents=True, exist_ok=True)
	
	try:
		for chunk_idx in range(num_chunks):
			actual_start	= chunk_idx * chunk_size
			actual_end	= min(actual_start + chunk_size, total_frames)
			print(f"{20 * '='} Processing chunk {chunk_idx + 1}/{num_chunks} ({actual_start} to {actual_end})")
			
			# Include overlap for SAM context
			init_start	= max(0, actual_start - overlap) if chunk_idx > 0 else actual_start
			init_end	= min(actual_end + overlap, total_frames)
			load_count	= init_end - init_start
			
			# 2. LOAD ONLY THE CHUNK
			chunk_frames	= get_video_chunk(cap, init_start, load_count, custom_width=w, custom_height=h, center_crop_param=center_crop)
			if chunk_frames is None: 
				break
			print(f'Loaded {chunk_frames.shape[0]} frames with shape {chunk_frames.shape[1:]}')
			print(f'Center-crop is: {"enabled (= no resize)" if args.center_crop else "disabled (resize will be performed instead)"}')
			
			# 3. SEGMENTATION
			video_state	= run_segmentation(chunk_frames, text_prompt)
			prop_start	= actual_start - init_start
			prop_end	= prop_start + (actual_end - actual_start)
			
			try:
				masks, video_state = propagate_masks_chunk(
					sam3_model, video_state, 
					start_frame=prop_start, end_frame=prop_end, 
					offload_model=False
				)
			except torch.cuda.OutOfMemoryError:
				torch.cuda.empty_cache()
				masks, video_state = propagate_masks_chunk(
					sam3_model, video_state, 
					start_frame=prop_start, end_frame=prop_end, 
					offload_model=True
				)

			# 4. OVERLAY CALCULATION (In-place on GPU chunk)
			raw_mask_chunk,_= get_sam3_outputs(masks, video_state)
			
			# Squeeze to the exact range we need (removing overlap context)
			extract_start	= actual_start - init_start
			extract_end	= extract_start + (actual_end - actual_start)
			curr_ch_frames	= chunk_frames[extract_start:extract_end]
			
			m_gpu		= raw_mask_chunk[extract_start:extract_end].to(sam3_model.device).unsqueeze(-1)
			# Get the RGB frames for overlay (discarding overlap)
			f_gpu		= chunk_frames[extract_start:extract_end].to(sam3_model.device)
			
			with torch.no_grad():
				alpha_masks	= m_gpu * overlay_alpha
				colored_masks	= selected_color.to(sam3_model.device) * m_gpu
				# Result as uint8 (0-255) to save 75% system RAM when stored
				overlay		= ((f_gpu * (1 - alpha_masks)*255 + (colored_masks * alpha_masks)*255)).to(torch.uint8)
				uint8_masks	= (m_gpu.squeeze(-1) > 0).to(torch.uint8)

			# 5. STREAM OR STORE
			if debug:
				# Store in RAM (original behavior)
				final_masks[actual_start:actual_end]	= uint8_masks.cpu()
				final_overlay[actual_start:actual_end]	= overlay.cpu()
			else:
				# Stream to FFmpeg (frame by frame to minimize memory)
				# Move to CPU once and iterate
				mask_cpu				= uint8_masks.cpu()	# (B, H, W)
				overlay_cpu				= overlay.cpu()	# (B, H, W, 3)
				
				for i in range(colored_masks.shape[0]):
					# Write mask as grayscale (expand dims for ffmpeg gray format)
					frame		= curr_ch_frames[i] * 255	# because get_video_chunk() does torch.from_numpy() / 255
					frame		= cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_BGR2RGB)
					mask_frame	= colored_masks[i]  * 255	# (H, W)
					mask_frame	= mask_frame.to(torch.uint8)
					if base_path is not None and args.write_images:
						cv2.imwrite(f'{masks_dir}/{base_path.name}-mask-{mask_video_frame_count}.png', mask_frame.cpu().numpy())
						if debug:
							print(f'Saved {masks_dir}/{base_path.name}-mask-{mask_video_frame_count}.png')

					# For gray pix_fmt, we need to send H*W bytes
					# But our start_ffmpeg uses rgb24 input, so let's adapt or use proper format
					# Actually, let's send as-is and fix the ffmpeg command
					write_frame_to_ffmpeg(mask_stdin, mask_frame)
					mask_video_frame_count += 1
					if debug:
						print(f'{mask_frame.shape = }')
						print(f'{mask_frame.dtype = }')
					
					# Write overlay
					overlay_frame	= overlay_cpu[i]  # (H, W, 3)
					write_frame_to_ffmpeg(overlay_stdin, overlay_frame)
					if debug:
						print(f'{overlay_frame.shape = }')
						print(f'{overlay_frame.dtype = }')
					if base_path is not None and args.write_images:
						cv2.imwrite(f'{rgb_dir}/{base_path.name}-rgb-{mask_video_frame_count}.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
						if debug:
							print(f'Saved {rgb_dir}/{base_path.name}-rgb-{mask_video_frame_count}.jpg')

				print(f'{20 * "="} Produced {overlay_cpu.shape[0]} frames with shape {overlay_cpu.shape[1:]} and {mask_cpu.shape[0]} masks with shape {mask_cpu.shape[1:]}')
				
				#mask_stdin.flush()
				overlay_stdin.flush()

			# 6. CLEANUP
			# Manually clear the references to the large chunk frames
			del chunk_frames, video_state, masks, m_gpu, f_gpu, overlay, uint8_masks
			if not debug:
				del mask_cpu, overlay_cpu
			
			torch.cuda.empty_cache()
			gc.collect() 
			print(f"{20 * '='} Processed chunk {chunk_idx + 1}/{num_chunks} ({actual_start} to {actual_end})\n\n")

	finally:
		# Ensure FFmpeg processes are finalized even if error occurs
		if mask_stdin is not None:
			finalize_ffmpeg(mask_stdin, mask_process)
			print(f"Mask video saved to: {output_mask_path}")
			pass
		if overlay_stdin is not None:
			finalize_ffmpeg(overlay_stdin, overlay_process)
			print(f"Overlay video saved to: {output_overlay_path}")
		
		cap.release()
	print(f'Written {mask_video_frame_count} frames to {output_mask_path}')

	if debug:
		return final_masks, final_overlay
	else:
		return None

def load_video(video_path, custom_width=0, custom_height=0, force_rate=0, frame_load_cap=120, skip_first_frames=0, select_every_nth=1, warning_file_size=50 * 1024 * 1024, debug=True):
	print(f"Loading video: {video_path}...")
	file_size = Path(video_path).stat().st_size
	print(f"Video size is: {file_size}")
	if custom_width != 0 or custom_height != 0:
		print(f'Custom width: {custom_width} - custom height: {custom_height}')
	if file_size >= warning_file_size:
		print(f'Warning: video file is larger than {int(warning_file_size / 1024 / 1024)} MB, there\'s a risk of OOM...')

	if not Path(video_path).exists():
		#raise FileNotFoundError(f"Video file not found: {video_path}")
		print(f"Video file not found: {video_path}")
		return None, None
	# if is symlink, resolve it...
	if Path(video_path).is_symlink():
		# works beautifully if relative and in a subdirectory, untested in any other case
		old_video_path	= Path(video_path)
		video_path	= old_video_path.parent / Path(video_path).readlink()
		if debug:
			print(f"Input video file is symlink that resolves to: {video_path}")

	video_loader = comfyui_vhs_module.VHS_LoadVideo()
	result = video_loader.load_video(
		video=str(video_path),
		custom_width=custom_width, custom_height=custom_height,
		force_rate=force_rate,
		frame_load_cap=frame_load_cap,
		skip_first_frames=skip_first_frames,
		select_every_nth=select_every_nth
	)
	# VHS_LoadVideo returns (IMAGE, frame_count, audio, video_info)
	return result[0], result[3]

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
def main():
	# 1. Load SAM3 model
	sam3_model		= load_sam3_model()
	
	# 2. Handle naming
	base_name		= Path(args.output_dir) / Path(args.input_video).stem
	mask_output_name	= f"{base_name}-mask.mkv"
	overlay_output_name	= f"{base_name}-overlay.mp4"

	# 3. Process everything in one memory-managed loop
	process_video_streaming_output = process_video_streaming(
		video_path=args.input_video,
		sam3_model=sam3_model,
		text_prompt=args.prompt,
		num_frames=args.num_frames,
		custom_width=args.custom_width,
		custom_height=args.custom_height,
		center_crop=args.center_crop,
		chunk_size=args.chunk_size,
		overlap=args.chunk_overlap,
		overlay_color=args.overlay_color,
		overlay_alpha=args.overlay_alpha,
		base_path=base_name,
		output_mask_path=mask_output_name,
		output_overlay_path=overlay_output_name,
		fps=0,
		debug=args.debug
	)
	if args.debug and process_video_streaming_output is not None:
		final_mask_tensor, overlay_tensor = process_video_streaming_output
	
	print(f'\n')
	print(f'Finished!')
	print(f"Mask video   : {mask_output_name}")
	print(f"Overlay video: {overlay_output_name}")

if __name__ == "__main__":

	# Setup environment
	comfyui_nodes_dir = setup_comfyui_environment(comfyui_base_dir)
	
	# Import the class
	import_sam3_package(comfyui_nodes_dir)
	
	# Get VHS_LoadVideo function
	import_vhs_package(comfyui_nodes_dir)

	main()

