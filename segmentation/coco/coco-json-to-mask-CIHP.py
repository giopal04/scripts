#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as coco_mask_utils
from PIL import Image # For saving masks
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description="Extract composite segmentation masks from an aggregated COCO JSON file."
    )
    parser.add_argument(
        "--coco_file",
        type=Path,
        required=True,
        help="Path to the aggregated COCO JSON file.",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=Path,
        required=True,
        help="Directory to save the extracted composite mask images and legend.",
    )
    args = parser.parse_args()

    if not args.coco_file.is_file():
        print(f"Error: COCO file {args.coco_file} not found.")
        return

    args.output_mask_dir.mkdir(parents=True, exist_ok=True)

    with open(args.coco_file, "r") as f:
        coco_data = json.load(f)

    # --- 1. Create Category Mapping (Original COCO ID to Compact ID) ---
    # Compact ID 0 will be background.
    # Compact IDs 1, 2, ... N will be for the actual classes.
    coco_cat_id_to_compact_id = {}
    compact_id_to_class_info = {
        0: {"name": "background", "original_coco_id": None} # Background class
    }
    
    # Sort categories by their original ID to ensure somewhat consistent mapping if order matters
    # (though for semantic masks, the exact compact ID value is less important than the mapping itself)
    sorted_categories = sorted(coco_data['categories'], key=lambda c: c['id'])

    for i, category_info in enumerate(sorted_categories):
        compact_id = i + 1 # Start compact IDs from 1
        original_coco_id = category_info['id']
        coco_cat_id_to_compact_id[original_coco_id] = compact_id
        compact_id_to_class_info[compact_id] = {
            "name": category_info['name'],
            "original_coco_id": original_coco_id
        }

    # Save the legend/mapping
    legend_file_path = args.output_mask_dir / "class_legend.json"
    with open(legend_file_path, "w") as f:
        json.dump(compact_id_to_class_info, f, indent=4)
    print(f"Saved class legend to: {legend_file_path}")
    
    # --- 2. Group Annotations by Image ID ---
    image_id_to_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)

    print(f"Processing {len(coco_data['images'])} images...")
    
    extracted_count = 0
    for i, image_info in enumerate(coco_data['images']):
        if (i+1) % 10 == 0 or i == len(coco_data['images']) -1 :
            print(f"Processing image {i+1}/{len(coco_data['images'])}: {image_info['file_name']}")

        image_id = image_info['id']
        image_height = image_info['height']
        image_width = image_info['width']
        
        # Initialize composite mask with background class (compact ID 0)
        composite_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        annotations_for_image = image_id_to_annotations.get(image_id, [])
        
        # Sort annotations, e.g., by area descending, so larger objects are drawn first.
        # This can help if there are overlaps, though the last drawn will still win.
        # If specific overlap handling is needed, it's more complex.
        # For now, we'll just use the order they appear or sort by annotation ID.
        annotations_for_image.sort(key=lambda a: a.get('area', 0), reverse=False) # Smaller areas first, larger ones can overwrite

        for ann in annotations_for_image:
            original_coco_category_id = ann['category_id']
            compact_class_id = coco_cat_id_to_compact_id.get(original_coco_category_id)

            if compact_class_id is None:
                print(f"Warning: Annotation {ann['id']} has category_id {original_coco_category_id} which is not in the legend. Skipping this annotation.")
                continue

            rle = ann['segmentation']
            if isinstance(rle['counts'], str): # Ensure counts is bytes for pycocotools
                rle['counts'] = rle['counts'].encode('utf-8')

            # Ensure RLE has 'size' if not already present
            if 'size' not in rle:
                 rle['size'] = [image_height, image_width]

            try:
                binary_instance_mask = coco_mask_utils.decode(rle) # HxW numpy array with 0s and 1s
            except Exception as e:
                print(f"  Error decoding RLE for annotation {ann['id']} (image: {image_info['file_name']}): {e}")
                continue

            if binary_instance_mask.sum() == 0:
                # print(f"  Warning: Decoded mask is empty for annotation {ann['id']}. Skipping.")
                continue
                
            # Update composite_mask: where binary_instance_mask is 1, set to compact_class_id
            # This means later annotations will overwrite earlier ones in case of overlap.
            composite_mask[binary_instance_mask == 1] = compact_class_id
            
        # --- 3. Save the Composite Mask ---
        image_filename_stem = Path(image_info['file_name']).stem
        output_mask_filename = f"{image_filename_stem}_mask.png"
        output_path = args.output_mask_dir / output_mask_filename
        
        try:
            mask_pil = Image.fromarray(composite_mask, mode='P') # 'P' for palette mode, good for class IDs
            
            # Create a simple palette: index N -> color (N, N, N) for grayscale visualization
            # Max compact_id will be len(coco_data['categories'])
            # If you have specific colors for classes, you'd define them here.
            # For now, simple grayscale based on class index.
            # Pillow 'P' mode needs a flat list of R,G,B values.
            num_palette_entries = len(compact_id_to_class_info) # max_compact_id + 1
            palette = []
            for k in range(num_palette_entries):
                # Simple grayscale: value k -> color (k,k,k)
                # This will look black for background (0,0,0) and increasingly lighter gray for classes.
                # Adjust if you have more than 255 classes, though uint8 mask implies < 256.
                val = min(k * (255 // max(1, num_palette_entries-1) if num_palette_entries > 1 else 10) , 255) if k > 0 else 0 # Scale for better visualization
                if k == 0: # Background
                    palette.extend([0,0,0]) # Black
                else: # Classes
                    # A simple distinct color generation
                    r = (k * 50) % 256
                    g = (k * 90) % 256
                    b = (k * 120) % 256
                    palette.extend([r, g, b])

            # Fill the rest of the palette if it's shorter than 768 (256*3)
            if len(palette) < 256 * 3:
                palette.extend([0] * (256 * 3 - len(palette)))
            
            mask_pil.putpalette(palette)
            mask_pil.save(output_path)
            extracted_count += 1
        except Exception as e:
            print(f"  Error saving composite mask {output_path}: {e}")

    print(f"\nSuccessfully generated and saved {extracted_count} composite masks to {args.output_mask_dir}")

if __name__ == "__main__":
    main()