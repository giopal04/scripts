#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/giorgio/venvs/StatueSeg')

from fastai.vision.all import * #TODO: Change to proper imports
import albumentations as alb
from fasttransform import ItemTransform
from pycocotools.coco import COCO 
import pycocotools.mask as cocomask
from tqdm import tqdm
import base64, zlib
import colorsys
import argparse
import cv2

from pprint import pprint

class SegmentationAlbumentationsTransform(ItemTransform):
    def __init__(self, aug):
        self.aug = aug

    def encodes(self, x):
        if not isinstance(x, (tuple, list)): #NOTE: this is necessary because TargetMaskConvertTransform gets trapped into the PKL even just doing learn.export()
            print('ERROR')
            return x
        
        img, msk = x
        aug = self.aug(image=np.array(img), mask=np.array(msk))
        return PILImage.create(aug['image']), PILMask.create(aug['mask'])

    def __repr__(self):
        return f'SegmentationAlbumentationsTransform()\naug={repr(self.aug)}'
    
    def __call__(self, x): 
        return self.encodes(x)

# Get inputs

def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-folder", type=str, help="Path to the folder containing the dataset")
    parser.add_argument("--data", type=str, help="COCO json file containing the output of the SAM3 analysis")
    parser.add_argument("--crack-dir", type=str, default="/mnt/dataset/surface-damage-segmentation/omnicrack30k/omnicrack30k/train", help="Path to omnicrack train set (the folder shold contain two sub directories named imags and masks)")
    parser.add_argument("--scale-factor", type=float, help="Scale factor for crack overlay")
    parser.add_argument("--threshold", type=int, default=50, help="Number of pixel under which a crack is not printed on the image")
    parser.add_argument("--max-cracks", type=int, default=10, help="Max number of crack on the same image")
    parser.add_argument("--enable-transforms", action="store_true", help="Applay rotations and flip to the crack image")
    parser.add_argument("--output-dir", type=str, default="/tmp", help="Saving directory for the final json file")
    parser.add_argument("--debug", action="store_true", help="Enables debug mode")
    return parser.parse_args()

def get_mask(path):
    return path.parents[1]/'masks'/f'{path.stem}.png'

# Preprocess and utilities

def prepare_crack_images(root_path):
    all_images = list(root_path.iterdir())

    return [(path, get_mask(path)) for path in tqdm(all_images)]

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return x1, y1, x2, y2 

def mask_to_polygons(mask, tolerance=1.0):  
    polygons = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if tolerance > 0:
            epsilon = tolerance * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(contour) >= 3:
            poly = contour.reshape(-1).tolist()
            polygons.append(poly)
            
    return polygons

# OLD
def encode_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()	

    data_type = mask.dtype
    shape = mask.shape
    compressed_mask = zlib.compress(mask.tobytes())
	
    return (base64.b64encode(compressed_mask).decode('utf-8'), shape, data_type)

# OLD
def decode_mask(encoded_string, shape, dtype):
	compressed_data = base64.b64decode(encoded_string)
	raw_bytes = zlib.decompress(compressed_data)
	
	return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

# Filtering instances

# OLD
def mIoU(mask1, mask2):
    intersection = np.sum((mask1 & mask2).astype(np.float32))
    union = np.sum((mask1 | mask2).astype(np.float32))
    
    if union == 0:
        return 0.0
    
    return intersection/union

# OLD
def filter_by_mask_iou(instances, iou_threshold=0.5):
    instances = sorted(instances, key=lambda x: x['score'], reverse=True)
    
    keep = []
    
    while len(instances) > 0:
        best_instance = instances.pop(0)
        best_mask = decode_mask(encoded_string=best_instance['mask'], shape=best_instance['shape'], dtype=best_instance['dtype'])
        keep.append(best_instance)
        
        remaining_instances = []
        for remaining in instances:
            remaining_mask = decode_mask(encoded_string=remaining['mask'], shape=remaining['shape'], dtype=remaining['dtype'])
            iou = mIoU(best_mask, remaining_mask)
            
            if iou < iou_threshold:
                remaining_instances.append(remaining)
        
        instances = remaining_instances
        
    return keep

# Color generator

# OLD
def get_color_scheme(number_of_classes):
    H_step = 1 / number_of_classes

    color_codes = {}
    for j in range(number_of_classes):
        h = j * H_step
        color_codes[str(j+1)] = []

        for i in range(10):
            l = 0.25 + i * 0.05
            s = 0.8

            r, g, b = colorsys.hls_to_rgb(h, l, s)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)

            block = np.zeros((3,50,50))
            block[0,:,:] = r
            block[1,:,:] = g
            block[2,:,:] = b

            color_codes[str(j+1)].append((r, g, b))
    
    return color_codes

# crack Overlay pipeline

def load_random_image(source_files):
    indx = random.randint(0, len(source_files)-1)
    s_path, s_mask_path = source_files[indx]


    return PILImage.create(s_path), PILMask.create(s_mask_path)

def get_dynamic_pipeline(img1_shape, img2_shape, scale_factor, transforms):
    H1, W1 = img1_shape[:2]
    W2, H2 = img2_shape #Size from an Image
    
    final_pipeline = []
    
    if H2 > H1 and W2 > W1:
        final_pipeline.append(alb.CenterCrop(height=H1, width=W1, p=1.0))
    else:
        final_pipeline.append(alb.Resize(height=H1, width=W1, p=1.0))
    
    final_pipeline.extend(transforms)
    
    final_h, final_w = max(1, int(H1 * scale_factor)), max(1, int(W1 * scale_factor))
    final_pipeline.append(alb.Resize(height=final_h, width=final_w, p=1.0))
    
    return SegmentationAlbumentationsTransform(alb.Compose(final_pipeline))

def overlay_crack(img1, msk1, img2, msk2):
    #NOTE: Currently the cracks mask are the invers of what one may expects.
    # Meaning that the relevant class, the crack, has the code 0 while the background the code 1

    #NOTE: The img2 and msk2 are already transformed.

    #Ensuring that if there is not any crack it exit the function
    if not np.any(msk2) or not np.any(msk1):
        return img1, np.zeros_like(msk1)

    #Images are all array
    H1, W1 = img1.shape[:-1]
    H2, W2 = img2.shape[:-1]

    #Defining the anchor point and crack baricenter
    baricenter = np.mean(np.transpose(np.nonzero(msk2)), axis=0).astype(np.int32)
    yB, xB = baricenter

    mask_points = np.transpose(np.nonzero(msk1))
    anchor_point = mask_points[random.randint(0, mask_points.shape[0]-1)]
    yA, xA = anchor_point

    #Defining the crop boundaries
    top, left = yA - yB, xA - xB

    s_top = max(0, -top)
    s_bottom = min(H2, H1 - top)
    s_left = max(0, -left)
    s_right = min(W2, W1 - left)

    d_top = max(0, top)
    d_bottom = min(H1, H2 + top)
    d_left = max(0, left)
    d_right = min(W1, W2 + left)
    
    #Creating a canvas
    canvas, canvas_mask = np.zeros_like(img1), np.zeros_like(msk1)
    canvas[d_top:d_bottom, d_left:d_right, :] = img2[s_top:s_bottom, s_left:s_right, :]
    canvas_mask[d_top:d_bottom, d_left:d_right] = msk2[s_top:s_bottom, s_left:s_right]

    #canvas, canvas_mask = canvas, canvas_mask

    #Overlay
    condition = (msk1 != 0) * (canvas_mask == 1)        
    img1_over = np.where(np.stack([condition]*3, axis=2), canvas, img1)
    """img1_over = img1.copy()
    img1_over[condition] = canvas[condition]"""
    #msk1_over = np.where(condition, 1, 0)
    msk1_over = condition.astype(np.uint8)

    return img1_over, msk1_over

def main():

    args = get_argparse()
    

    #Input directories and files
    crack_dir = Path(args.crack_dir)
    dataset_dir = Path(args.dataset_folder)

    '''
    image_dir = Path(args.input_images)
    '''
    
    if args.debug:
        print(f'Taking data from {args.data}')
    uncracked_dataset = COCO(args.data)

    '''
    with open(args.data, 'r') as f:
        json_data = json.load(f)
    '''
    #Saving folder
    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    
    if not (out_dir/'images').exists():
        (out_dir/'images').mkdir()

    '''

    #Creating the final json
    final_data = {
        'metadata': {},
        'images': {}
    }

    """classes = [key for key in json_data['metadata']['prompt2id'].keys()]
    classes.sort()
    classes = classes + ['crack']"""
    prompt2id = {
        'crack': 1,
        'statue': 3,
        'monument': 5,
        'column': 2,
        'pedestal': 6,
    }

    color_scheme = get_color_scheme(6)

    #final_data['metadata']['prompt2id'] = prompt2id
    #final_data['metadata']['id2colors'] = color_scheme

    '''

    #Apply crack to images

    #Getting crack images
    print('Loading crack images...')
    source_files = prepare_crack_images(crack_dir/'images')

    #Preparing transform
    if args.enable_transforms:
        crack_tsfm = [alb.RandomRotate90(),
                      alb.HorizontalFlip(),
                      alb.VerticalFlip()]
    else: 
        crack_tsfm = []

    # Adding the crack category
    num_original_cats = len(uncracked_dataset.cats)
    uncracked_dataset.dataset['categories'].append(
        {
            'id': num_original_cats, #NOTE: to be checked
            'name': 'crack',
            'supercategory': 'defect'
        }
    )

    # Number of anns
    num_original_anns = len(uncracked_dataset.anns)

    # -----------------------------------------
    # --------------- Hold Code ---------------
    # -----------------------------------------

    print('Cleaning the original file and adding crack...')
    for k in tqdm(uncracked_dataset.imgs):
        # Retrieving image info
        img_info = uncracked_dataset.loadImgs(ids=k)[0]
        img_path = Path(img_info['file_name'])
        img_id = img_info['id']
        img_h, img_w = img_info['height'], img_info['width']

        if args.debug:
            print(f"Starting image: {img_path.name}")
        
        # Load the img annotations
        img_anns = uncracked_dataset.getAnnIds(imgIds=img_id)
        original_instances = uncracked_dataset.loadAnns(ids=img_anns)
        if args.debug:
            print(f"\tFound {len(original_instances)} instances")

        #Loading the image
        img_rgb = PILImage.create(dataset_dir/img_path) #BUG: Check  that the image path is correct
        img = np.array(img_rgb)
        
        instances_to_add = []

        '''
        #Cleaning the original instances 
        cleaned_instances = filter_by_mask_iou(instances=original_instances)
        '''

        #Checking if there is anything in the image
        if not original_instances:
            img_save_path = out_dir/img_path
            img_save_path.parent.mkdir(parents=True, exist_ok=True)
            img_rgb.save(img_save_path)
            continue
          
        #Adding the cracks
        iterations = np.random.randint(1, args.max_cracks)
        if args.debug:
            print(f"\tAdding {iterations} cracks")
        for _ in range(iterations):
            indx = np.random.randint(0, len(original_instances))
            selected_instance = original_instances[indx]

            # Decoding the mask
            rle_msk = cocomask.frPyObjects(selected_instance['segmentation'], img_h, img_w)
            msk_decoded = cocomask.decode(rle_msk)
            
            if args.debug:
                print(f'{msk_decoded.shape = }')

            #NOTE: The decoded mask can have multiple components
            if msk_decoded.shape[-1] > 1:
                for i in range(msk_decoded.shape[-1]):
                    msk = msk_decoded[...,i]
                msk = np.where(msk > 1, 1, msk)
            else:
                msk = msk_decoded.squeeze(axis=-1)
                
            msk = msk.astype(np.uint8)
            if args.debug:
                #print(f"\t\tAdding crack to instance of class {selected_instance['prompt']}")
                print(f"\t\tTotal mask size {msk.sum()}")

            img_c, msk_c = load_random_image(source_files=source_files)
            final_pipeline = get_dynamic_pipeline(img.shape, img_c.size, scale_factor=args.scale_factor, transforms=crack_tsfm)
            img_c_t, msk_c_t = final_pipeline((img_c, msk_c))
            img_c_t_a, msk_to_convert = np.array(img_c_t), np.array(msk_c_t)

            msk_c_t_a = (msk_to_convert == 0).astype(np.uint8)
            
            if args.debug:
                print(f"\t\tNumber of pixels of the crack: {msk_c_t_a.sum()}")

            tmp, tmp_msk = overlay_crack(img1=img, msk1=msk, img2=img_c_t_a, msk2=msk_c_t_a)
            
            if args.debug:
                print(f"\t\tNumber of pixels of the added crack: {tmp_msk.sum()}")

            if tmp_msk.sum() > args.threshold:                
                encoded_mask = mask_to_polygons(tmp_msk, tolerance=0.003)
                box = list(get_bbox(tmp_msk))
                ann_id = num_original_anns + 1

                instances_to_add.append(
                    {
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': num_original_cats + 1,
                        'segmentation': encoded_mask,
                        'iscrowd': 0,
                        'area': int(np.sum(tmp_msk)),
                        'bbox': [int(x) for x in box]
                    }
                )
                
                img = tmp
            
            else:
                continue
        
        for instance in instances_to_add:
            uncracked_dataset.dataset['annotations'].append(instance)

        if args.debug:
            print(f"\tAdded {len(instances_to_add)} cracks")

        final_image = Image.fromarray(img, mode='RGB')
        img_save_path = out_dir/img_path
        img_save_path.parent.mkdir(parents=True, exist_ok=True)
        final_image.save(img_save_path)
    
    save_file = out_dir/'cracked_statue.json'
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(uncracked_dataset.dataset, f, ensure_ascii=False, indent=None)
    
if __name__ == '__main__':
    main()
