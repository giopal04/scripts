#!/usr/bin/env python3

import json
import os
from glob import glob

def default_categories():
	# Define COCO categories based on the provided class information
	categories = [
		{"id":  1,	"name": "Crack",		"supercategory": "damage"},
		{"id":  2,	"name": "ACrack",		"supercategory": "damage"},	# Alligator Crack
		{"id":  3,	"name": "Wetspot",		"supercategory": "damage"},
		{"id":  4,	"name": "Efflorescence",	"supercategory": "damage"},
		{"id":  5,	"name": "Rust",			"supercategory": "damage"},
		{"id":  6,	"name": "Rockpocket",		"supercategory": "damage"},
		{"id":  7,	"name": "Hollowareas",		"supercategory": "damage"},
		{"id":  8,	"name": "Cavity",		"supercategory": "damage"},
		{"id":  9,	"name": "Spalling",		"supercategory": "damage"},
		{"id": 10,	"name": "Graffiti",		"supercategory": "damage"},
		{"id": 11,	"name": "Weathering",		"supercategory": "damage"},
		{"id": 12,	"name": "Restformwork",		"supercategory": "damage"},
		{"id": 13,	"name": "ExposedRebars",	"supercategory": "damage"},	# Exposed Rebars
		{"id": 14,	"name": "Bearing",		"supercategory": "object"},
		{"id": 15,	"name": "EJoint",		"supercategory": "object"},	# Expansion Joint
		{"id": 16,	"name": "Drainage",		"supercategory": "object"},
		{"id": 17,	"name": "PEquipment",		"supercategory": "object"},	# Protective Equipment
		{"id": 18,	"name": "JTape",		"supercategory": "object"},	# Joint Tape
		{"id": 19,	"name": "WConccor",		"supercategory": "object"},	# Washouts/Concrete Corrosion
	]
	category_name_to_id = {cat["name"]: cat["id"] for cat in categories}

	return categories, category_name_to_id

def get_categories(category_list):
	categories = []
	for idx,cat in enumerate(category_list):
		cat = cat.strip()
		categories.append({"id": idx + 1, "name": cat, "supercategory": "damage"})
	category_name_to_id = {cat["name"]: cat["id"] for cat in categories}

	return categories, category_name_to_id

def polygon_area(points):
	"""Calculate the area of a polygon using the shoelace formula."""
	area = 0.0
	n = len(points)
	for i in range(n):
		j = (i + 1) % n
		area += points[i][0] * points[j][1]
		area -= points[j][0] * points[i][1]
	return abs(area) / 2.0

def convert_to_coco(input_files, output_path, categories, category_name_to_id, debug=False):
	coco_data = {
		"info": {},
		"licenses": [],
		"categories": categories,
		"images": [],
		"annotations": []
	}

	image_id = 1
	annotation_id = 1

	for input_file in input_files:

		if 'coco_annotations.json' in str(input_file):
			print(f"Skipping output file: {input_file}")
			continue
		if input_file.endswith(".json") == False:
			print(f"Skipping non-JSON file: {input_file}")	
			continue

		with open(input_file, "r") as f:
			data = json.load(f)

		#print(f'Processing {input_file} - data: {data}')
		if debug:
			print(f'Processing {input_file}')
			for key,val in data.items():
				if key != 'imageData':
					print(f'data[{key}]: {val}')

		# Add image information
		image_info = {
			"id": image_id,
			"file_name": data["imageName"] if 'imageName' in data else data["imagePath"],
			"height": data["imageHeight"],
			"width": data["imageWidth"],
		}
		coco_data["images"].append(image_info)

		# Process each annotation (polygon)
		for shape in data["shapes"]:
			label = shape["label"]
			if label not in category_name_to_id:
				print(f"Skipping unknown label '{label}' in {input_file}")
				continue

			points = shape["points"]
			segmentation = [[coord for point in points for coord in point]]
			xs = [p[0] for p in points]
			ys = [p[1] for p in points]
			x_min, x_max = min(xs), max(xs)
			y_min, y_max = min(ys), max(ys)
			bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
			area = polygon_area(points)

			annotation = {
				"id": annotation_id,
				"image_id": image_id,
				"category_id": category_name_to_id[label],
				"segmentation": segmentation,
				"area": area,
				"bbox": bbox,
				"iscrowd": 0,
			}
			coco_data["annotations"].append(annotation)
			annotation_id += 1

		image_id += 1

	with open(output_path, "w") as f:
		json.dump(coco_data, f, indent=2)

	print(f'Converted {len(coco_data["images"])} images and {len(coco_data["annotations"])} annotations into {output_path}')


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Convert dacl10k custom JSON annotations to COCO format.")
	parser.add_argument("--input_dir", required=True, help="Directory containing input dacl10k JSON files")
	parser.add_argument("--output_file", default="coco_annotations.json", help="Output COCO JSON file path")
	parser.add_argument("--categories", default="", nargs='+', help="Space-separated list to override default dacl10k categories (e.g. 'Crack, ACrack, Rust, EJoint, ...')")
	args = parser.parse_args()

	input_files = glob(os.path.join(args.input_dir, "*.json"))
	if not input_files:
		raise ValueError(f"No JSON files found in {args.input_dir}")

	if args.categories:
		#category_list = args.categories.split(" ")
		print(f'Overriding default categories with: {args.categories}')
		category_list = [cat.replace("§"," ") for cat in args.categories]
		categories, category_name_to_id = get_categories(category_list)
		print(f"Found {len(categories)} categories: {categories}")
	else:
		categories, category_name_to_id = default_categories()

	convert_to_coco(input_files, args.output_file, categories, category_name_to_id)
	#print(f"Converted {len(input_files)} files to COCO format. Output saved to {args.output_file}")

if __name__ == '__main__':
	main()
