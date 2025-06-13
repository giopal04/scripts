#!/usr/bin/env python

import sys
import json
import re
from io import StringIO
import pandas as pd

def format_time(time_str):
	parts = time_str.split(':')
	if len(parts) == 3:  # HH:MM:SS format
		return f"{int(parts[0])}:{parts[1]}"
	elif len(parts) == 2:  # MM:SS format
		return f"0:{parts[0].zfill(2)}"
	return time_str  # fallback

def extract_tables(notebook_path):
	with open(notebook_path, 'r', encoding='utf-8') as f:
		nb = json.load(f)
	
	tables = []
	for cell in nb['cells']:
		if cell['cell_type'] == 'code' and 'outputs' in cell:
			for output in cell['outputs']:
				if 'data' in output and 'text/html' in output['data']:
					html_content = ''.join(output['data']['text/html'])
					if '<table' in html_content and 'dice_multi' in html_content:
						tables.append(html_content)
						if len(tables) == 2:
							return tables
	return tables

def process_table(html_table):
	df = pd.read_html(StringIO(html_table))[0]
	best_idx = df['dice_multi'].idxmax()
	best_row = df.loc[best_idx]
	
	# Format values
	train_loss = f"{best_row['train_loss']:.3f}"
	valid_loss = f"{best_row['valid_loss']:.3f}"
	miou = f"{best_row['mIoU']:.3f}"
	dice = f"{best_row['dice_multi']:.3f}"
	jaccard = f"{best_row['jaccard_coeff_multi']:.3f}"
	time = format_time(str(best_row['time']))
	
	return f"{train_loss} & {valid_loss} & {miou} & {dice} & {jaccard} & {time}"

if __name__ == "__main__":
	debug = False

	if len(sys.argv) != 2:
		print("Usage: fastai-extract-best-run-from-tables.py <notebook.ipynb>")
		sys.exit(1)
	
	notebook_file = sys.argv[1]
	tables = extract_tables(notebook_file)
	print(f'Found {len(tables)} tables in the notebook')

	if debug:
		print(f'Tables: {tables}')

	if len(tables) == 0:
		print(f"Error: No valid tables found in the notebook")
		sys.exit(1)
	
	for i, table in enumerate(tables):
		result = process_table(table)
		print(f"Run {i}: {result}")

# Example output:
# Run 1: 0.021 & 0.024 & 0.637 & 0.854 & 0.773 & 5:33
# Run 2: ... (formatted results for second run)

sys.exit(0)

