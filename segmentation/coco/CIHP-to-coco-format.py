#!/usr/bin/env python3

import argparse
import base64
import json
import zlib
from pathlib import Path
import numpy as np
from pycocotools import mask as coco_mask_utils
import supervisely as sly # For robust bitmap decoding

def decode_bitmap_sly(encoded_bitmap_str: str, origin_yx: tuple) -> tuple[np.ndarray, tuple]:
    """
    Decodes Supervisely's base64 encoded zlib compressed bitmap.
    Returns the boolean numpy array of the sub-mask and its origin.
    """
    bitmap_np = sly.Bitmap.base64_2_data(encoded_bitmap_str) # Returns a boolean np.ndarray
    return bitmap_np, origin_yx

def main():
    parser = argparse.ArgumentParser(
        description="Convert individual annotation JSONs to a single COCO-style JSON file."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing individual .jpg.json annotation files.",
    )
    parser.add_argument(
        "--meta_file",
        type=Path,
        required=True,
        help="Path to the meta.json file containing class definitions.",
    )
    parser.add_argument(
        "--output_coco_file",
        type=Path,
        required=True,
        help="Path to save the aggregated COCO JSON file.",
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default=".jpg",
        help="Comma-separated list of the extension(s) of the original image files (e.g., .jpg,.jpeg,.png)."
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: Input directory {args.input_dir} not found.")
        return
    if not args.meta_file.is_file():
        print(f"Error: Meta file {args.meta_file} not found.")
        return

    args.output_coco_file.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Categories from meta.json ---
    with open(args.meta_file, "r") as f:
        meta_data = json.load(f)

    categories = []
    # classId from individual JSONs maps to 'id' in meta.json
    # COCO category_id should be the 'id' from meta.json
    class_id_to_coco_category_id = {}
    class_id_to_title = {}

    for cat_info in meta_data["classes"]:
        categories.append({
            "id": cat_info["id"], # Use meta.json's id as COCO category_id
            "name": cat_info["title"],
            "supercategory": "object", # Or derive if available
        })
        class_id_to_coco_category_id[cat_info["id"]] = cat_info["id"]
        class_id_to_title[cat_info["id"]] = cat_info["title"]


    coco_output = {
        "info": {
            "description": "Aggregated COCO dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "DatasetNinja User",
            "date_created": "", # Will be set later
        },
        "licenses": [{"url": "http://creativecommons.org/licenses/by/2.0/", "id": 1, "name": "Attribution License"}],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id_counter = 0
    annotation_id_counter = 0

    annotation_files = []
    for ext in args.image_extensions.split(","):
        print(f'Looking for "*{ext}.json" files in {args.input_dir}')
        found_files = sorted(list(args.input_dir.glob(f"*{ext}.json")))
        annotation_files.extend(found_files)
        if not found_files:
            print(f"No '*{ext}.json' files found in {args.input_dir}")

    print(f"Found {len(annotation_files)} annotation files.")
    if len(annotation_files) == 0:
        return

    for ann_file_path in annotation_files:
        print(f"Processing {ann_file_path.name}...")
        with open(ann_file_path, "r") as f:
            ann_data = json.load(f)

        image_filename = ann_file_path.stem # e.g., "0000006.jpg"
        image_height = ann_data["size"]["height"]
        image_width = ann_data["size"]["width"]

        image_id_counter += 1
        current_image_id = image_id_counter

        coco_output["images"].append({
            "id": current_image_id,
            "width": image_width,
            "height": image_height,
            "file_name": image_filename, # Assumes original image has same stem
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": "",
        })

        for obj in ann_data["objects"]:
            if obj["geometryType"] != "bitmap":
                print(f"Skipping non-bitmap geometry: {obj['geometryType']} in {ann_file_path.name}")
                continue

            class_id = obj["classId"]
            if class_id not in class_id_to_coco_category_id:
                print(f"Warning: classId {class_id} not found in meta.json. Skipping object in {ann_file_path.name}")
                continue

            coco_category_id = class_id_to_coco_category_id[class_id]
            # origin is [col, row] in the example, but Supervisely uses [row, col] / [y,x]
            # The example shows origin: [212, 265] which seems like [x,y] or [col, row]
            # Supervisely Bitmap origin is [top, left] which is [row, col] or [y,x]
            # Let's assume the provided format is [x,y] and convert if sly expects [y,x]
            # The provided JSON's origin [212, 265] seems to be [col, row] i.e. [x, y]
            # sly.Bitmap.base64_2_data does not need origin. Origin is for placing.
            # sly.Bitmap expects origin as (top, left) i.e. (y, x)

            # The example JSON seems to have origin as [x, y]
            # origin_x, origin_y = obj["bitmap"]["origin"]
            # Let's stick to the structure of the example: obj["bitmap"]["origin"] is [col, row]
            # For numpy slicing, we need [row, col]
            origin_col, origin_row = obj["bitmap"]["origin"] # [x,y] from file

            sub_mask_bool, _ = decode_bitmap_sly(obj["bitmap"]["data"], (origin_row, origin_col))
            sub_mask_uint8 = sub_mask_bool.astype(np.uint8) # Convert boolean to 0/1
            
            full_image_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            
            mask_h, mask_w = sub_mask_uint8.shape
            
            # Ensure the sub-mask fits within the full image dimensions
            end_row = min(origin_row + mask_h, image_height)
            end_col = min(origin_col + mask_w, image_width)
            
            # Calculate the actual height and width of the part of sub_mask to place
            actual_h = end_row - origin_row
            actual_w = end_col - origin_col

            if actual_h <= 0 or actual_w <= 0:
                print(f"Warning: Sub-mask for object {obj['id']} in {ann_file_path.name} is outside image bounds or zero-sized. Origin: ({origin_row},{origin_col}), Sub-mask shape: ({mask_h},{mask_w}), Image shape: ({image_height},{image_width}). Skipping.")
                continue

            try:
                full_image_mask[origin_row:end_row, origin_col:end_col] = sub_mask_uint8[:actual_h, :actual_w]
            except Exception as e:
                print(f"Error placing sub_mask for object {obj['id']} in {ann_file_path.name}: {e}")
                print(f"  Full mask shape: {full_image_mask.shape}")
                print(f"  Sub_mask_uint8 shape: {sub_mask_uint8.shape}")
                print(f"  Origin: (row={origin_row}, col={origin_col})")
                print(f"  Target slice: [{origin_row}:{end_row}, {origin_col}:{end_col}]")
                print(f"  Source slice: [:{actual_h}, :{actual_w}]")
                continue


            if np.sum(full_image_mask) == 0: # Check if mask is empty after placement
                # print(f"Warning: Empty mask for object {obj['id']} in {ann_file_path.name} (class: {class_id_to_title[class_id]}). Skipping.")
                continue

            # Convert to FORTRAN-contiguous array for pycocotools
            fortran_mask = np.asfortranarray(full_image_mask)
            rle = coco_mask_utils.encode(fortran_mask)
            
            # pycocotools RLE 'counts' can be bytes, json.dump needs str
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            
            area = coco_mask_utils.area(rle)
            bbox = coco_mask_utils.toBbox(rle) # [x,y,width,height]

            annotation_id_counter += 1
            coco_output["annotations"].append({
                "id": annotation_id_counter,
                "image_id": current_image_id,
                "category_id": coco_category_id,
                "segmentation": rle,
                "area": float(area), # Ensure area is float
                "bbox": bbox.tolist(), # Ensure bbox is list of floats/ints
                "iscrowd": 0,
            })

    # Update date_created
    from datetime import datetime
    coco_output["info"]["date_created"] = datetime.utcnow().isoformat()

    with open(args.output_coco_file, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"\nSuccessfully converted {len(annotation_files)} annotation files.")
    print(f"Aggregated COCO data saved to {args.output_coco_file}")
    print(f"Total images: {image_id_counter}, Total annotations: {annotation_id_counter}")

if __name__ == "__main__":
    main()
