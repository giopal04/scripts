#!/usr/bin/env python3

'''
Threshold-based Image Similarity Search and Deletion

Finds all images with similarity >= specified threshold to any other image,
then optionally deletes the duplicates (keeping one from each group).
'''

import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple

import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# Import from original script
try:
	from classes.argument_parser import define_boolean_argument, var2opt
except ImportError:
	# Fallback if the import fails
	def define_boolean_argument(parser, *args, **kwargs):
		parser.add_argument(f'--{args[0]}', action='store_true', 
						  help=kwargs.get('help', ''))
	def var2opt(var_name):
		return (var_name, var_name.replace('_', '-'))

def load_embeddings(embeddings_path: Path):
	'''Load embeddings from pickle file'''
	with open(embeddings_path, 'rb') as f:
		data = pickle.load(f)
	
	print(f'[DEBUG] Loaded {len(data["filenames"])} embeddings')
	print(f'[DEBUG] Model used: {data["metadata"].get("model_used", "Unknown")}')
	print(f'[DEBUG] Embedding dimension: {data["embeddings"].shape[1]}')
	
	return data

def lambda_print(x):
	'''Debug helper function'''
	print(f'[DEBUG] {type(x) = } - {x = }')
	return x

def find_duplicate_groups(
	embeddings_data: dict,
	threshold: float = 0.95,
	debug: bool = False
) -> List[List[str]]:
	'''
	Find groups of duplicate images based on similarity threshold.
	Returns list of groups, where each group contains similar filenames.
	'''
	n_images   = len(embeddings_data['filenames'])
	embeddings = embeddings_data['embeddings']
	filenames  = embeddings_data['filenames']
	
	print(f'[INFO] Computing pairwise similarities for {n_images} images...')
	
	# Compute full similarity matrix
	if debug:
		print(f'[DEBUG] Embeddings shape: {embeddings.shape}')
		print(f'[DEBUG] Computing cosine similarity matrix...')
	
	similarity_matrix = cosine_similarity(embeddings.reshape(n_images, -1))
	
	if debug:
		print(f'[DEBUG] Similarity matrix shape: {similarity_matrix.shape}')
		print(f'[DEBUG] Diagonal values (self-similarity): {similarity_matrix.diagonal()}')
	
	# Zero out diagonal (self-similarity) and lower triangle to avoid duplicates
	np.fill_diagonal(similarity_matrix, 0)
	
	# Find pairs above threshold
	above_threshold = np.where(similarity_matrix >= threshold)
	pairs = list(zip(above_threshold[0], above_threshold[1]))
	
	if debug:
		print(f'[DEBUG] Found {len(pairs)} pairs above threshold {threshold}')
		if pairs:
			print(f'[DEBUG] First few pairs: {pairs[:5]}')
	
	# Build adjacency list for connected components (groups of similar images)
	adjacency = defaultdict(list)
	for i, j in pairs:
		adjacency[filenames[i]].append(filenames[j])
		adjacency[filenames[j]].append(filenames[i])  # Undirected graph
	
	# Find connected components (groups of similar images)
	visited = set()
	groups = []
	
	for filename in filenames:
		if filename not in visited:
			# BFS to find connected component
			queue = deque([filename])
			component = []
			
			while queue:
				current = queue.popleft()
				if current not in visited:
					visited.add(current)
					component.append(current)
					queue.extend(adjacency[current])
			
			if len(component) > 1:  # Only keep groups with duplicates
				groups.append(component)
	
	print(f'[INFO] Found {len(groups)} groups of similar images (threshold >= {threshold})')
	
	# Sort groups by size (largest first)
	groups.sort(key=len, reverse=True)
	
	# Print group statistics
	total_duplicates = sum(len(g) - 1 for g in groups)
	print(f'[INFO] Total duplicate images found: {total_duplicates}')
	
	return groups

def display_group(
	group: List[str],
	group_idx: int,
	keep_first: bool = True
):
	'''
	Display images in a similarity group.
	Returns list of files to delete (all except the first).
	'''
	print(f'\n[GROUP {group_idx + 1}] {len(group)} similar images:')
	
	# Display first image (the one we'll keep)
	keep_file = group[0]
	fn = Path(keep_file).name
	print(f'  KEEP: {fn} (reference)')
	
	if Path(keep_file).exists():
		img   = cv2.imread(keep_file)
		sz    = f'{img.shape[1]}x{img.shape[0]}'
		wname = f'Group {group_idx+1} - KEEP - {sz} - {fn}'
		if img is not None:
			cv2.namedWindow	(wname, cv2.WINDOW_NORMAL)
			cv2.moveWindow	(wname, 0, 100)
			cv2.resizeWindow(wname, 640, 640)
			cv2.imshow	(wname, img)
	
	# Display duplicates (to be deleted)
	files_to_delete = []
	for i, duplicate in enumerate(group[1:], 1):
		fn = Path(duplicate).name
		print(f'  DELETE ({i}): {fn}')
		files_to_delete.append(duplicate)
		
		if Path(duplicate).exists():
			img   = cv2.imread(duplicate)
			sz    = f'{img.shape[1]}x{img.shape[0]}'
			wname = f'Group {group_idx+1} - DELETE {i} - {sz} - {fn}'
			if img is not None:
				cv2.namedWindow	(wname, cv2.WINDOW_NORMAL)
				cv2.moveWindow	(wname, 670 * i, 100)
				cv2.resizeWindow(wname, 640, 640)
				cv2.imshow	(wname, img)
	
	return files_to_delete

def confirm_deletion(files: List[str], no_prompt: bool = False) -> bool:
	'''Ask for confirmation before deletion'''
	if no_prompt:
		return True
	
	print(f'\n[CONFIRMATION] About to delete {len(files)} files:')
	for f in files[:5]:  # Show first 5 files
		print(f'  {Path(f).name}')
	if len(files) > 5:
		print(f'  ... and {len(files) - 5} more')
	
	response = input(f'\nProceed with deletion? [y/N]: ').strip().lower()
	return response == 'y'

def delete_files(files: List[str], dry_run: bool = False):
	'''Delete files with error handling'''
	deleted_count = 0
	error_count = 0
	
	for file_path in files:
		path = Path(file_path)
		if dry_run:
			print(f'[DRY RUN] Would delete: {path.name}')
			deleted_count += 1
		else:
			try:
				if path.exists():
					path.unlink()  # Delete file
					print(f'[DELETED] {path.name}')
					deleted_count += 1
				else:
					print(f'[WARNING] File not found: {path.name}')
					error_count += 1
			except Exception as e:
				print(f'[ERROR] Failed to delete {path.name}: {e}')
				error_count += 1
	
	print(f'\n[SUMMARY] {"(not) " if dry_run else ""}Deleted {deleted_count} files, {error_count} errors')

def main():
	parser = argparse.ArgumentParser(
		description='Find and delete duplicate images using similarity threshold'
	)
	parser.add_argument('--embeddings_file','-e', type=str,		default='image-embeddings.pkl',	help='Pickle file containing embeddings')
	parser.add_argument('--threshold',	'-t', type=float,	default=0.95,			help='Similarity threshold (default: 0.95)')
	parser.add_argument('--max-groups',	'-m', type=int,		default=None,			help='Maximum number of groups to process (for testing)')

	define_boolean_argument(parser, *var2opt('no_prompt'),		'Delete without confirmation',		False)
	define_boolean_argument(parser, *var2opt('dry_run'),		'Don\'t actually delete anything',	True)
	define_boolean_argument(parser, *var2opt('debug'),		'Show debug information',		False)
	define_boolean_argument(parser, *var2opt('show_images'),	'Show similar images before deleting',	True)
	
	args = parser.parse_args()
	
	# Debug prints
	print(f'[DEBUG] Arguments: {args}')
	print(f'[DEBUG] Threshold: {args.threshold}')
	print(f'[DEBUG] No prompt: {args.no_prompt}')
	print(f'[DEBUG] Dry run  : {args.dry_run}')
	
	# Load embeddings
	embeddings_path = Path(args.embeddings_file)
	if not embeddings_path.exists():
		print(f'[ERROR] Embeddings file {embeddings_path} not found')
		sys.exit(1)
	
	data = load_embeddings(embeddings_path)
	
	if args.debug:
		print(f'[DEBUG] Embedings contains the following keys: {list(data.keys())}')
		print(f'[DEBUG] First few filenames: {data["filenames"][:3]}')
	
	# Find duplicate groups
	groups = find_duplicate_groups(data, args.threshold, debug=args.debug)
	
	if not groups:
		print('[INFO] No duplicate groups found above threshold')
		return
	
	# Limit groups if specified
	if args.max_groups:
		groups = groups[:args.max_groups]
		print(f'[INFO] Limiting to first {len(groups)} groups')
	
	# Process each group
	all_files_to_delete = []
	
	for i, group in enumerate(groups):
		if args.debug:
			print(f'[DEBUG] Processing group {i}: {len(group)} images')
		
		files_to_delete = []
		
		if args.show_images:
			files_to_delete = display_group(group, i, keep_first=True)
			print('\nPress any key to continue to next group...')
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		else:
			# Without display, just mark all but first for deletion
			files_to_delete = group[1:]
			print(f'\n[GROUP {i + 1}] {len(group)} images')
			print(f'  KEEP: {Path(group[0]).name}')
			for duplicate in files_to_delete:
				print(f'  DELETE: {Path(duplicate).name}')
		
		all_files_to_delete.extend(files_to_delete)
	
	# Summary
	print(f'\n[SUMMARY] Found {len(groups)} groups')
	print(f'[SUMMARY] Total duplicates marked for deletion: {len(all_files_to_delete)}')
	
	if args.dry_run:
		print('[INFO] Dry run mode - no files will be deleted')
	
	# Delete files
	if all_files_to_delete and (args.no_prompt or args.dry_run or confirm_deletion(all_files_to_delete, args.no_prompt)):
		delete_files(all_files_to_delete, args.dry_run)
	else:
		print('[INFO] Deletion cancelled or no files to delete')

if __name__ == '__main__':
	main()
