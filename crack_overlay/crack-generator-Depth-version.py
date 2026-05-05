#!/usr/bin/env python3

# The code is a mess, it would be nice to rewrite it entirely with numpy

from depth_anything_3.api import DepthAnything3
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
import pycocotools.mask as cocomask
import argparse
from datetime import datetime
import time
import cv2
import json
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/giorgio/venvs/Depth-Anything-3')
from format import inspect, content

# Managing inputs 
def get_argparse():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", type=str, help="Path to the dataset COCO json")
    parser.add_argument("--dataset-folder", type=str, help="Path to the dataste folder")
    parser.add_argument("--output-dir", type=str, help="Final directory where the new dataset is saved")
    parser.add_argument("--crack-directory", type=str, default="/mnt/dataset/surface-damage-segmentation/omnicrack30k/omnicrack30k/train/images", help="Path to the folder containing the crack images and masks (it must have two subfolders, called images and masks respectively)")
    parser.add_argument("--max-cracks", type=int, default=5, help="Max number of crack to applay in each image")
    parser.add_argument("--threshold", type=int, default=100, help="Minimum number of pixels of the crack to get it add to the image")
    parser.add_argument("--resolution", type=int, default=504, help="Image resolution to compute the depth map")
    parser.add_argument("--max-angles", type=float, nargs=3, default=[np.pi/2, 0.1, 0.1], help="Maximum angles defining the random interval in which a rotation of the projector is choosen (the first one is a planar rotation the other two are vertical rotation)")
    parser.add_argument("--rel-depth-range", type=float, nargs=2, default=[-0.5,0.25], help="Intervall in which the projection distance is sampled (normal distribution)")
    parser.add_argument("--feather-radius", type=int, default=5, help="Parameter to controll the Gaussian blur when attaching images")
    parser.add_argument("--depth-influence", type=float, default=0.3, help="Parameter to determine the influence of the depth map in determining the blending")
    parser.add_argument("--on-cpu", action="store_true", help="Moves everything to cpu")
    parser.add_argument("--debug", action="store_true", help="Print useful information")

    return parser.parse_args()

#Utilities
def get_bbox(mask):
    mask = convert_to_array(mask)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return x1, y1, x2, y2

def convert_to_radiants(ang):
    return np.pi*ang/180

def convert_to_degrees(ang):
    return 180*ang/np.pi

def convert_to_tensor(array, device='cuda:0'):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        return array.to(device=device)
    else:
        return None

def convert_to_array(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        return None

def mask_to_polygons(mask, tolerance=1.0):
    mask = convert_to_array(mask)  
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


# Crack dataset related function
def get_mask(path):
    return path.parents[1]/'masks'/f'{path.stem}.png'

def prepare_crack_images(root_path):                                                                                                                                                                               
    all_images = list(root_path.iterdir())
    
    return [(path, get_mask(path)) for path in tqdm(all_images)]

def load_random_image(sources):
    indx = np.random.randint(0, len(sources)-1)
    s_path, s_msk_path = sources[indx]

    return Image.open(s_path).convert('RGB'), Image.open(s_msk_path).convert('L')

# Depth anythin inference
def load_model(debug=False):
    print('Loading Depth Anytihng V3 model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE", log=debug)
    model = model.to(device)
    model.eval()

    print(f'Model loaded on {device}')

    return model

def model_inference(imgs, model, resolution=504, debug=False):
    # Run inference
    # NOTE: It's better to make inference on upscaled images
    # assert len(imgs) == 1, f'Works only for one image at a time, {len(imgs)} where given'

    prediction = model.inference(
        image=imgs,
        process_res=resolution,
        process_res_method="upper_bound_resize",
        export_dir=None,
        export_format="glb",        
    )
    
    if debug:
        print(f"Depth shape: {prediction.depth.shape}")
        print(f"Extrinsics: {prediction.extrinsics.shape if prediction.extrinsics is not None else 'None'}")
        print(f"Intrinsics: {prediction.intrinsics.shape if prediction.intrinsics is not None else 'None'}")

    return convert_to_tensor(prediction.depth[0]), convert_to_tensor(prediction.intrinsics[0]), convert_to_tensor(prediction.extrinsics[0])

def rescale_output(depth, intrinsics, original_shape, device='cuda:0'):
    *_, H, W = depth.shape
    H_or, W_or = original_shape

    Is = torch.Tensor([
        [W_or/W, 0, 0],
        [0, H_or/H, 0],
        [0, 0, 1]]
        ).to(device=device)
    
    rescaled_depth = F.interpolate(
        depth.unsqueeze(dim=0).unsqueeze(dim=0),
        size=original_shape,
        mode='bilinear',
        align_corners=False
    )

    rescaled_K = torch.matmul(Is, intrinsics)

    return rescaled_depth, rescaled_K

# Surface reconstruction and projection
def random_mask_point(mask, is_crack):
    mh, mw = mask.shape
    ys, xs = np.where(mask > 0)

    if len(xs) > 1:
        i = np.random.randint(0, len(xs)-1)
        return xs[i], ys[i]
    
    if is_crack:
        return mw // 2, mh // 2
    
    return None, None

def reconstruct_surface(depth, intrinsics, extrinsics, device='cuda:0', debug=False):
    #depth: [H,W]
    #intrinsics: [3,3]
    #extrinsics: [3,4]

    #this matricies must be reshaped down to the original image dimention
    rotation, translation = extrinsics[:,:3], extrinsics[:,3]
    
    height, width = depth.shape

    v, u = torch.meshgrid(
        torch.arange(height, device=device), 
        torch.arange(width, device=device), 
        indexing='ij'
    )
    ones = torch.ones((height, width), device=device)

    pixels = torch.stack([u, v, ones], dim=-1).unsqueeze(dim=-1)

    if debug:
        inspect(depth, 'depth')
        inspect(intrinsics, 'intrinscs')
        inspect(rotation, 'rotation')
        inspect(translation, 'translation')
        inspect(pixels, 'pixels')


    points = depth.unsqueeze(dim=-1)*torch.matmul(torch.inverse(intrinsics), pixels).squeeze(dim=-1)
    final_surface = (torch.matmul(rotation, points.unsqueeze(dim=-1)).squeeze(dim=-1) + translation)

    if debug:
        inspect(final_surface, 'final_surface')

    return final_surface   

def create_projector(fov, scale, img_shape, center, r=None, t=None, debug=False, device='cuda:0'):
    #img_shape: [H,W,3] or [H,W]
    #the fov here is in radiant 

    if r is None:
        r = torch.from_numpy(np.eye(3)).to(dtype=torch.float32, device=device)
    else:
        r = convert_to_tensor(r)
    
    if t is None:
        t = torch.from_numpy(np.array([0,0,0])).to(dtype=torch.float32, device=device)
    else:
        t = convert_to_tensor(t)

    if len(img_shape) == 3:
        h, w = img_shape[:-1]
    else:
        h, w = img_shape

    fx, fy = (w * scale)/(2*torch.tan(0.5*fov)), (h * scale)/(2*torch.tan(0.5*fov))
    cx, cy = center

    projector_intrinsics = torch.Tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]).to(dtype=torch.float32, device=device)

    if debug:
        content(projector_intrinsics, 'intrinsics')
        content(r, 'rotation')
        content(t, 'translation')
    
    return [projector_intrinsics, r, t]

def get_off_set(surface, anchor, pdepth):
    xanchor, yanchor = anchor

    anchor3D = surface[yanchor, xanchor].clone()
    anchor3D[-1] = torch.tensor(pdepth)

    return anchor3D

def get_rotation_matrix(theta_z=0, phi_x=0, phi_y=0):
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    rotz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    cx, sx = np.cos(phi_x), np.sin(phi_x)
    rotx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=np.float32)

    cy, sy = np.cos(phi_y), np.sin(phi_y)
    roty = np.array([
        [cy, 0, -sy],
        [0, 1, 0],
        [sy, 0, cy]
    ], dtype=np.float32)

    return rotz @ rotx @ roty

def get_random_extrinsics_parameters(max_angles, min_depth, rel_depth_range):
    theta_m, phi_x_m, phi_y_m = max_angles
    a, b = rel_depth_range

    theta = 2*theta_m*np.random.rand() - theta_m
    phi_x = 2*phi_x_m*np.random.rand() - phi_x_m
    phi_y = 2*phi_y_m*np.random.rand() - phi_y_m

    min_depth = min_depth.cpu().float()
    depth = np.random.normal(loc=(b+a)*min_depth/2, scale=(b-a)*min_depth/6)

    return [theta, phi_x, phi_y], depth

def get_original_fov(intrinsics, original_shape):
    H_or, W_or = original_shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]

    fovx, fovy = 2*torch.arctan(W_or/(2*fx)), 2 * torch.arctan(H_or/(2*fy))

    return (fovx+fovy)/2

def projects_points(surface, projector_component, debug=False):
    #projector_componen: list of projector component with intrinsics, rotation and translation

    k, r, t = projector_component

    if debug:
        inspect(surface, 'surface')
        inspect(r, 'rotation')
        inspect(t, 'translation')

    repositioned_surface = torch.matmul(r, surface.unsqueeze(dim=-1)).squeeze(dim=-1) + torch.matmul(r, t)
    
    if debug:
        inspect(repositioned_surface, 'repositioned_surface')
        inspect(k, 'k')
    
    projected_surface = torch.matmul(k, repositioned_surface.unsqueeze(dim=-1)).squeeze(dim=-1)
    z = projected_surface[:,:,-1].unsqueeze(dim=-1)
    projected_surface = projected_surface / torch.clamp(z, 1e-3)
    projected_surface = projected_surface.to(dtype=torch.int16)

    if debug:
        inspect(projected_surface, 'projected_surface')

    return projected_surface

def isolate_crack(ppoints, target_image, target_mask, source_image, source_mask, device='cuda:0', debug=False):
    '''if isinstance(target_mask, np.ndarray):
        valid_pixels = torch.from_numpy(target_mask).to(device=device)
    else:
        valid_pixels = torch.zeros_like(target_mask)
        valid_pixels = torch.where(target_mask, 1, 0).to(device=device)'''
    
    target_mask_t = convert_to_tensor(target_mask, device=device)
    source_mask_t = convert_to_tensor(source_mask, device=device)
    source_image_t = convert_to_tensor(source_image, device=device)
    target_image_t = convert_to_tensor(target_image, device=device)

    valid_pixels = target_mask_t.bool()

    hs, ws = source_mask.shape
    
    inx = (ppoints[:,:,0] >= 0) & (ppoints[:,:,0] < ws)
    iny = (ppoints[:,:,1] >= 0) & (ppoints[:,:,1] < hs)
    in_source = inx & iny
    
    if debug:
        content(in_source, 'in_source')

    valid_target_pixels = valid_pixels * in_source
    valid_pixels = torch.where(valid_target_pixels, 1, 0)
    
    if debug:
        content(valid_pixels, 'valid_pixels')

    x_coords = ppoints[valid_pixels>0][...,0].long()
    y_coords = ppoints[valid_pixels>0][...,1].long()

    valid_pixels[valid_pixels > 0] += source_mask_t[y_coords, x_coords]
    isolated_crack = torch.zeros_like(target_image_t).to(device=device)
    isolated_crack[valid_pixels > 0] = source_image_t[y_coords, x_coords, :]

    crack_mask = torch.zeros_like(target_mask_t)
    crack_mask = torch.where(valid_pixels > 1, 1, 0)
    
    return isolated_crack, crack_mask

def depth_aware_blend(target_img, source_image, mask_array, depth, feather_radius=5, depth_influence=0.3, debug=False):
    mask_array = convert_to_array(mask_array)
    depth = convert_to_array(depth)
    target_img = convert_to_array(target_img)
    source_image = convert_to_array(source_image)

    if mask_array.sum() == 0:
        return target_img

    target_img = Image.fromarray(target_img.astype(np.uint8), mode='RGB')
    source_layer = Image.fromarray(source_image.astype(np.uint8), mode='RGB')

    binary_mask = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    
    if debug:
        content(mask_array, 'mask_array')

    if depth_influence > 0:
        d_min, d_max = np.min(depth[mask_array > 0]), np.max(depth[mask_array > 0])
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-7)
        depth_factor = 1.0 - (depth_norm * depth_influence)
    
        binary_mask = (binary_mask * depth_factor).astype(np.uint8)

    pil_mask = Image.fromarray(binary_mask, mode='L')

    if feather_radius > 0:
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    blended_img = Image.composite(source_layer, target_img, pil_mask)

    return np.array(blended_img)

# Main Function
def main():
    # Getting the inputs 
    args = get_argparse()
    
    DEBUG = args.debug

    # Getting all the paths
    COCO_JSON_PATH = Path(args.dataset)
    DATASET_FOLDER = Path(args.dataset_folder)
    
    CRACK_DIR = Path(args.crack_directory)
    
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Retrieving other global parameters
    RESOLUTION = args.resolution
    MAX_ITERATIONS = args.max_cracks
    THRESHOLD = args.threshold
    MAX_ANGLES = args.max_angles
    REL_DEPTH_RANGE = args.rel_depth_range
    FEATHER_RADIUS = args.feather_radius
    DEPTH_INFLUENCE = args.depth_influence
    DEVICE = 'cuda:0' if not args.on_cpu else 'cpu'

    print(f'Adding cracks on the datatset:\n"{COCO_JSON_PATH}"')
    print(f'Final data is saved to:\n"{OUTPUT_DIR}"')
    
    # Loading the COCO files
    coco_dataset = COCO(COCO_JSON_PATH)

    # Loading the crack dataset
    print('Loading the crack images...')
    print(f'Getting data from:\n{CRACK_DIR}')
    sources = prepare_crack_images(CRACK_DIR)
    
    if DEBUG:
        print(sources)

    # Loading the Depth Anything model    
    start_loading = time.time()
    model = load_model(debug=DEBUG)
    print(f'Done ({(time.time()-start_loading):0.2f}s)')

    # Adding the crack category
    original_cat = coco_dataset.cats

    crack_id = list(original_cat.keys())[-1] + 1
    coco_dataset.dataset['categories'].append(
        {
            'id': crack_id,
            'name': 'crack',
            'supercategory': 'defect'
        }
    )
    
    # Getting the total number of annotations
    ann_indexing = len(coco_dataset.anns) + 1

    # Adding the cracks
    print('Starting applaying cracks...')
    for img_info in tqdm(coco_dataset.imgs.values()):
        # Getting image info
        img_path = Path(img_info['file_name'])
        img_id = img_info['id']
        img_h, img_w = img_info['height'], img_info['width']

        if DEBUG:
            print(f'Starting manipulation of image: {img_path.name}, id: {img_id}')
        
        # Load image annotations
        anns_ids = coco_dataset.getAnnIds(imgIds=img_id)
        orig_anns = coco_dataset.loadAnns(ids=anns_ids)
        
        if DEBUG:
            print(f"\tFound {len(orig_anns)} annotations")

        new_img_anns = []

        # Loading the image
        img_rgb = Image.open(DATASET_FOLDER/img_path).convert('RGB')
        img = np.array(img_rgb)

        if DEBUG:
            inspect(img, 'img')

        # Checking if the image has any annotation
        if not orig_anns:
            img_save_path = OUTPUT_DIR/img_path
            img_save_path.parent.mkdir(parents=True, exist_ok=True) 
            img_rgb.save(img_save_path)
            continue

        # Getting the depth map and surface
        D_model, K_model, Rtc = model_inference(imgs=[img], model=model, resolution=RESOLUTION)

        Dc, Kc = rescale_output(D_model, K_model, (img_h, img_w), device=DEVICE)
        Dc = Dc.squeeze(0).squeeze(0)

        surface = reconstruct_surface(Dc, Kc, Rtc, device=DEVICE)

        # Adding cracks
        iterations = np.random.randint(1, MAX_ITERATIONS+1)
        
        if DEBUG:
            print(f'Adding {iterations} cracks')

        for _ in range(iterations):
            # Selecting a random annotation
            indx = np.random.randint(0, len(orig_anns))
            selected_instance = orig_anns[indx]

            poly_msk = selected_instance['segmentation']
            
            # Retrieve and decode the mask
            rle_msk = cocomask.frPyObjects(poly_msk, img_h, img_w)
            msk_decoded = cocomask.decode(rle_msk)

            if msk_decoded.shape[-1] > 1:
                msk = np.zeros(msk_decoded.shape[:-1])
                for i in range(msk_decoded.shape[-1]):
                    msk += msk_decoded[...,i]
                msk = np.where(msk > 1, 1, msk)
            else:
                msk = msk_decoded.squeeze(-1)

            msk = msk.astype(np.uint8)
            
            if DEBUG:
                print(f'Selected the annotation with id: {selected_instance["id"]}')
                print(f'Total mask size: {msk.sum()}')
            
            # Getting the crack image
            cimg_pil, cmsk_pil = load_random_image(sources)
            cimg, cmsk = np.array(cimg_pil), np.array(cmsk_pil)
            cmsk = 1-cmsk//255

            if DEBUG:
                print('-'*50)
                content(cmsk, 'cmsk')
                print(f'There are a total of {cmsk.sum()} crack pixels')

            anchor_point = random_mask_point(msk, is_crack=False)
            if anchor_point[0] is None and anchor_point[1] is None:
                continue

            # Defining a projector
            min_depth = Dc[msk>0].min()
            angles, depth = get_random_extrinsics_parameters(MAX_ANGLES, min_depth, REL_DEPTH_RANGE)

            Rp = get_rotation_matrix(*angles)
            tp = get_off_set(surface, anchor_point, depth)

            fov = get_original_fov(Kc, (img_h, img_w))
            cx, cy = random_mask_point(cmsk, is_crack=True)
            
            projector = create_projector(fov=fov,
                                         scale=1,
                                         img_shape=cimg.shape,
                                         center=(cx, cy),
                                         r=Rp,
                                         t=tp,
                                         device=DEVICE)
            
            ppoints = projects_points(surface, projector)
            isolated_crack, tmp_msk = isolate_crack(ppoints=ppoints,
                                                    target_image=img,
                                                    target_mask=msk,
                                                    source_image=cimg,
                                                    source_mask=cmsk,
                                                    device=DEVICE)
            
            tmp = depth_aware_blend(target_img=img,
                                    source_image=isolated_crack,
                                    mask_array=tmp_msk,
                                    depth=Dc,
                                    feather_radius=FEATHER_RADIUS,
                                    depth_influence=DEPTH_INFLUENCE,
                                    debug=DEBUG)
            
            if DEBUG:
                print(f"\t\tNumber of pixels of the added crack: {tmp_msk.sum()}")
            
            if tmp_msk.sum() > THRESHOLD:                
                encoded_mask = mask_to_polygons(tmp_msk, tolerance=0.003)
                box = list(get_bbox(tmp_msk))
                ann_id = ann_indexing

                new_img_anns.append(
                    {
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': crack_id,
                        'segmentation': encoded_mask,
                        'iscrowd': 0,
                        'area': int(np.sum(tmp_msk.cpu().numpy())),
                        'bbox': [int(x) for x in box]
                    }
                )
                
                ann_indexing += 1
                img = tmp
            
            else:
                continue
        
        for ann in new_img_anns:
                coco_dataset.dataset['annotations'].append(ann)
            
        if DEBUG:
            print(f"\tAdded {len(new_img_anns)} cracks")
        
        final_image = Image.fromarray(img, mode='RGB')
        img_save_path = OUTPUT_DIR/img_path
        img_save_path.parent.mkdir(parents=True, exist_ok=True)
        final_image.save(img_save_path)
    
    save_file = OUTPUT_DIR/'cracked_statue.json'
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(coco_dataset.dataset, f, ensure_ascii=False, indent=None)
    
    print(f'New dataset file saved:\n{save_file}')

if __name__ == '__main__':
    main()
