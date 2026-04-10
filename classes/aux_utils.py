import os
import sys

import subprocess
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# GPU sanity checks — run before anything else so problems are surfaced early
# ---------------------------------------------------------------------------

def check_pytorch_cuda_version():
	"""Check if PyTorch was built with sufficient CUDA support (cu130+)."""
	cuda_version = torch.version.cuda
	if cuda_version is None:
		print('[WARN] PyTorch has NO CUDA support!')
		print('[WARN] Install with: pip install torch --index-url https://download.pytorch.org/whl/cu130')
		return False
	parts = cuda_version.split('.')
	major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
	cuda_int = major * 10 + minor  # e.g. 12.8 → 128, 13.0 → 130
	if cuda_int < 130:
		print(f'[WARN] PyTorch CUDA {cuda_version} detected — SAM3 needs 13.0+ for optimized ops.')
		print(f'[WARN] Upgrade with: pip install torch --index-url https://download.pytorch.org/whl/cu130')
		return False
	print(f'[OK] PyTorch CUDA version: {cuda_version}')
	return True

def ensure_sam3_gpu_nms(comfyui_base_dir, install_if_missing=False):
	"""
	Attempt to build the GPU-accelerated NMS CUDA extension for SAM3 if not already compiled.
	This eliminates the '5-10x slower' warning that causes tracking to run on CPU.
	"""
	sam3_node_dir  = Path(comfyui_base_dir) / 'custom_nodes' / 'comfyui_sam3'
	install_script = sam3_node_dir / 'install.py'

	if not install_script.exists():
		print(f'[NMS] install.py not found at {install_script}, skipping GPU NMS build.')
		return False

	# Check whether the compiled extension is already present
	try:
		import importlib.util as _ilu
		# SAM3 may name its compiled NMS extension differently depending on version;
		# try a few common candidates.
		for _candidate in ('nms_cuda', 'sam3_nms', '_nms_cuda'):
			if _ilu.find_spec(_candidate) is not None:
				print(f'[NMS] GPU-accelerated NMS already compiled ({_candidate}), skipping build.')
				return True
		# Also check for .so files directly inside the node dir
		import glob
		so_files = glob.glob(str(sam3_node_dir / '**' / '*nms*.so'), recursive=True)
		if so_files:
			print(f'[NMS] Found compiled NMS extension(s): {so_files}')
			return True
	except Exception:
		pass

	if not install_if_missing:
		print(f'[NMS] GPU-accelerated NMS extension not found — but skipping GPU NMS build because parameter --install-if-missing is False.')
		return False
	print(f'[NMS] GPU-accelerated NMS extension not found — attempting to build...')
	print(f'[NMS] Running: python {install_script}')
	try:
		result = subprocess.run(
			[sys.executable, str(install_script)],
			cwd=str(sam3_node_dir),
			capture_output=not args.debug,
			text=True,
			timeout=300  # 5 min max for compilation
		)
		# Build script should just install these two packages...
		# https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cc_torch-latest/cc_torch-0.2+cu128torch2.9-cp313-cp313-linux_x86_64.whl
		# https://github.com/PozzettiAndrea/cuda-wheels/releases/download/torch_generic_nms-latest/torch_generic_nms-0.1%2Bcu128torch2.9-cp313-cp313-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl
		if result.returncode == 0:
			print('[NMS] GPU NMS extension built successfully — tracking will now run on GPU.')
			if args.debug and result.stdout:
				print(result.stdout)
			return True
		else:
			print(f'[NMS] Build failed (exit code {result.returncode}).')
			if result.stderr:
				print(f'[NMS] stderr (tail):\n{result.stderr[-2000:]}')
			print(f'[NMS] Falling back to CPU NMS. Fix manually: cd {sam3_node_dir} && python install.py')
			return False
	except subprocess.TimeoutExpired:
		print('[NMS] Build timed out after 5 minutes.')
		return False
	except Exception as e:
		print(f'[NMS] Build error: {e}')
		return False

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

def print_free_ram():
	"""Print total free RAM in human-readable format."""
	memory = psutil.virtual_memory()
	free_bytes = memory.available  # Actually usable memory (not just completely free)
	
	# Convert to GB for readability
	free_gb = free_bytes / (1024 ** 3)
	print(f"Free RAM: {free_gb:.2f} GB ({free_bytes:,} bytes)")
	return free_gb

