import os
import sys

import torch

from pathlib import Path

comfyui_nodes_dir = '' # let's just pretend we don't know where it is...

# global placeholders for comfyui_sam3 module
comfyui_sam3_module	= None
sam3_nodes		= None
latest_module		= None

# --- Workflow Functions ---

def patch_prompt_server(server):
	from unittest.mock import MagicMock

	# 2. Assign it to the class attribute 'instance'
	if not hasattr(server.PromptServer, 'instance'):
		print("Patching PromptServer.instance...")
		server.PromptServer.instance = MagicMock()

def load_sam3_model(model_name="sam3.pt"):
    print(f"Loading SAM3 model: {model_name}...")
    loader = comfyui_sam3_module.nodes.load_model.LoadSAM3Model()
    # Paths are typically relative to ComfyUI/models/sam3/
    model = loader.load_model(model_path=f"models/sam3/{model_name}")
    return nodeoutput_to_type(model)

def nodeoutput_to_type(some_nodeoutput, label="", debug=False):
	if latest_module is not None:
		latest = latest_module
	else:
		latest = None

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

def setup_comfyui_environment(comfyui_base_dir, debug=False):
	global comfyui_nodes_dir
	global latest_module

	# Add ComfyUI base paths to import basic modules...
	sys.path.append(str(comfyui_base_dir))
	if debug:
		print(f'sys.path: {sys.path}')
	os.environ['PYTHONPATH'] = str(comfyui_base_dir)
	if debug:
		print(f'PYTHONPATH: {os.environ["PYTHONPATH"]}')

	#import nodes
	import folder_paths
	from comfy_api import latest
	latest_module = latest

	#import app
	#import utils

	#from comfy_extras import nodes_mask

	
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
	sam3_dir	= Path(comfyui_nodes_dir) / 'comfyui_sam3'
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
		return comfyui_sam3_module, sam3_nodes

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

