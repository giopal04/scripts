#!/usr/bin/env python3

'''
Image Similarity Search using stored embeddings

Use image-embeddings-DINOv3.py to generate embeddings from images
'''

# Launch with:
# cosine-similarity-search.py --query /tmp/similarity-test/instances-grayscale_colorization_attempt.png

import sys
import pickle
import argparse

from pathlib import Path

from classes.argument_parser import define_boolean_argument, var2opt

import numpy as np
import cv2

from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embeddings_path: Path):
	'''Load embeddings from pickle file'''
	with open(embeddings_path, 'rb') as f:
		data = pickle.load(f)
	
	print(f'Loaded {len(data["filenames"])} embeddings')
	print(f'Model used: {data["metadata"].get("model_used", "Unknown")}')
	print(f'Embedding dimension: {data["embeddings"].shape[1]}')
	
	return data

def lambda_print(x):
	print(f'{type(x) = } - {x = }')
	return x

def find_similar_images(
	query_filename: str,
	embeddings_data: dict,
	top_k: int = 5,
	debug: bool = False
):
	'''Find similar images using cosine similarity'''
	# Find query index
	try:
		query_idx = embeddings_data["filenames"].index(query_filename)
	except ValueError:
		print(f'Error: {query_filename} not found in embeddings')
		return []
	
	# Calculate similarities
	query_embedding   = embeddings_data["embeddings"][query_idx].reshape(1, -1)
	paired_embeddings = list(zip(embeddings_data["embeddings"], embeddings_data["filenames"]))	# list of tuples (embedding, filename)
	if debug:
		print(f'{paired_embeddings = }')
		print(f'{type(paired_embeddings[0]) = }')
		print(f'{paired_embeddings[0] = }')
		print(f'{paired_embeddings[0][0].shape = }')

		print(f'\nSearching for {query_filename}...')
		print(f'{query_embedding.shape = }')

	similarities = [
			[embedding_pair[0], embedding_pair[1], cosine_similarity(query_embedding, np.expand_dims(embedding_pair[0].flatten(),0))[0]]
				for embedding_pair in paired_embeddings
			] # list of lists (embedding, filename, similarity)
	if debug:
		print(f'{similarities = }')
		similarities = sorted(similarities, key=lambda x: lambda_print(x[2]), reverse=True)
	else:
		similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
	if debug:
		print(f'Sorted {similarities = }')
	
	# Get top K matches (excluding the query itself)
	results = []
	for idx, (emb, fn, sim) in enumerate(similarities):
		if debug:
			print(f'{idx = } - {sim = } - {emb = } - {fn = }')
		if fn == query_filename:
			continue
		results.append({
			'filename': fn,
			'similarity': float(sim[0]),
			'embedding': emb,
			'rank': len(results) + 1
		})
		if len(results) >= top_k:
			break
	
	return results

def main():
	parser = argparse.ArgumentParser(
		description='Search for similar images using stored embeddings'
	)
	parser.add_argument('--embeddings_file', '-e',	type=str, default='image-embeddings.pkl', help='Pickle file containing embeddings')
	parser.add_argument('--query', '-q',		type=str, required=True, help='Query image filename (must be in the embeddings)')
	parser.add_argument('--top-k', '-k',		type=int, default=5, help='Number of similar images to return (default: 5)')
	define_boolean_argument(parser, *var2opt('debug'),		'show debug information',	False)
	define_boolean_argument(parser, *var2opt('show_images'),	'show retrieved images',	True)
	
	args = parser.parse_args()
	
	# Load embeddings
	embeddings_path = Path(args.embeddings_file)
	if not embeddings_path.exists():
		print(f'Error: Embeddings file {embeddings_path} not found')
		sys.exit(1)
	
	data = load_embeddings(embeddings_path)
	print(f'Loaded {len(data["filenames"])} embeddings')
	print(f'Embedings contains the following keys		: {data.keys()}')
	if len(data['filenames']) <= 100:
		print(f'Embedings contains the following filenames	: {[Path(fn).name for fn in data["filenames"]]}')
	else:
		print(f'Embedings contains the following filenames	: {[Path(fn).name for fn in data["filenames"][:5]]} ... {[Path(fn).name for fn in data["filenames"][-5:]]}')
	print(f'Embedings contains the following metadata	: {data["metadata"]}')
	
	# Find similar images
	results = find_similar_images(args.query, data, args.top_k, debug=args.debug)
	
	if not results:
		print('No similar images found.')
		return
	
	# Display results
	qfn   = args.query
	qfnn  = Path(args.query).name
	image = cv2.imread(qfn)
	print(f'\nTop {len(results)} similar images to\n{args.query} ({image.shape[0]}x{image.shape[1]}):')
	print('-' * 60)

	if args.show_images:
		cv2.namedWindow	(qfnn, cv2.WINDOW_NORMAL)
		cv2.moveWindow	(qfnn, 0, 320)
		cv2.resizeWindow(qfnn, 320, 320)
		cv2.imshow	(qfnn, image)

	for idx, result in enumerate(results):
		fn    = result["filename"]
		fnn   = Path(result["filename"]).name
		image = cv2.imread(fn)
		print(f'{result["rank"]}. Filename  : {Path(result["filename"]).name}')
		print(f'   Size      : {image.shape[0]}x{image.shape[1]}')
		print(f'   Similarity: {result["similarity"]:.4f}')
		print(f'   Path      : {result["filename"]}')
		print()
		if args.show_images:
			cv2.namedWindow	(fnn, cv2.WINDOW_NORMAL)
			cv2.moveWindow	(fnn, 0+320*(idx+1), 320)
			cv2.resizeWindow(fnn, 320, 320)
			cv2.imshow	(fnn, image)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()
