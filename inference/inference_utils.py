from pathlib import Path
import dill
import timm
import torch
import torch.nn.functional as F
from fastai.vision.all import *
from inspect import getmembers, isfunction
import torchvision.transforms as T

import cv2

from format import Text

def item_transform(img_size):
	# This is a transform, there's no image!
	return Resize(img_size, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)

def change_mask_color(img, from_c, to_c, debug=False):
	if len(img.shape) < 3 or img.shape[2] == 1:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	mask_color_lo = np.array(from_c)
	mask_color_hi = np.array(from_c)
	if debug:
		print(f'mask_color_lo: {mask_color_lo}')
		print(f'mask_color_hi: {mask_color_hi}')
	to_c = np.array(to_c)

	if debug:
		print(f'to_c: {to_c}')

	mask = cv2.inRange(img, mask_color_lo, mask_color_hi)
	img[mask>0] = tuple(to_c)
	return img			# a mask, actually

def center_crop(img, new_width=None, new_height=None):
	# center-crop a PIL Image - https://stackoverflow.com/a/32385865/1396334
	width  = img.shape[1]
	height = img.shape[0]

	if new_width is None:
		new_width = min(width, height)

	if new_height is None:
		new_height = min(width, height)

	left = int(np.ceil((width - new_width) / 2))
	right = width - int(np.floor((width - new_width) / 2))

	top = int(np.ceil((height - new_height) / 2))
	bottom = height - int(np.floor((height - new_height) / 2))

	if len(img.shape) == 2:
		center_cropped_img = img[top:bottom, left:right]
	else:
		center_cropped_img = img[top:bottom, left:right, ...]

	return center_cropped_img

def overlay(frame, mask, alpha=0.5):
	if len(mask.shape) < 3 or mask.shape[2] == 1:
		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	if mask.shape != frame.shape:
		print(f'mask.shape != frame.shape: {mask.shape} != {frame.shape}')
		mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
	return cv2.addWeighted(frame, 1, mask, alpha, 0)

def get_img_and_patch_size(arch_str):
	import re
	pattern			= r"_patch[0-9][0-9]_"
	patch_size		= re.findall(pattern, arch_str)[0]
	patch_size		= int(patch_size.replace('patch', '').replace('_', ''))
	if 'vit_giant_patch14_reg4_dinov2' in arch_str or 'vit_large_patch14_reg4_dinov2' in arch_str:	# https://huggingface.co/timm/vit_giant_patch14_reg4_dinov2.lvd142m
		vit_img_size	= 518
	else:
		pattern		= '_[0-9][0-9][0-9]$'
		vit_img_size	= re.findall(pattern, arch_str)[0]
		vit_img_size	= int(vit_img_size.replace('_', ''))		# final ViT size (e.g. 224), fixed
	return vit_img_size, patch_size

def find_backbone(backbone):
    arch_lst = getmembers(models, isfunction)
    print(f'Searching for backbone: {backbone}')
    for arch in arch_lst:
        if arch[0] == backbone:
            print(f'Found arch: {arch[0]} - func: {arch[1]}')
            return arch[1]

def load_model(input_dir, model_path, device, encoder, img_size, bs, num_classes, args, task='segmentation', decoder=None):
	# Create dummy dataloaders to initialize model architecture
	dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
					   get_items=get_image_files,
					   get_y=parent_label,
					   item_tfms=item_transform(img_size),
					   batch_tfms=Normalize.from_stats(*imagenet_stats))

	metrics = [
                accuracy,
                Precision(average='macro'),
                Recall(average='macro'),
                F1Score(average='macro'),
                Jaccard(average='macro'),
              ]    
	# Dummy dataloaders with 97 classes (matching your model output)
	#dls = dblock.dataloaders(input_dir, bs=4, num_classes=97)
	print(f'Creating dummy dataloaders with {num_classes} classes from directory: {Path(input_dir).name}')
	dls = dblock.dataloaders(input_dir, bs=bs, num_classes=num_classes)

	if 'resnet' in encoder.lower() or 'convnext' in encoder.lower() or 'efficientnet' in encoder.lower():
		arch_func = find_backbone(encoder.lower().replace('-', ''))

		if arch_func is None:
			print(f'Unable to find {encoder} within Fast.ai standard models, retrying with Timm...')
			# we're forced to use len(env['codes']) because there's no dls.c inside a segmentation DataLoader
			arch_func = partial(timm.create_model, model_name=encoder, pretrained=True, num_classes=num_classes)

		print(f'Allocating a ResNet Learner. Type: {encoder} - {arch_func = }')
		learn     = unet_learner(dls=dls, arch=arch_func, loss_func=CrossEntropyLossFlat(axis=1), metrics=metrics, self_attention=True, n_out=num_classes) #.to_fp16()
	elif 'ViT' in encoder or 'vit' in encoder:
		# Create learner and load weights
		#learn = vision_learner(dls, resnet50, pretrained=False, n_out=97)
		print(f'Allocating a ViT Learner. Type: {encoder}')
		if task == 'classification':
			model = timm.create_model(encoder, pretrained=True, num_classes=num_classes, dynamic_img_size=True)
			print(f'{model = }')
			learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=metrics, cbs=[MixedPrecision])
		elif task == 'segmentation':
			vit_img_size, patch_size = get_img_and_patch_size(encoder)
			if args.segmenter_version == '1':
				from segmenter import Segmenter
				# model, cls, img_dim, patch_size, cut_enc=-3, cut_in_dec=4, cut_out_dec=-3
				segmenter = Segmenter(model=encoder, cls=num_classes, img_dim = vit_img_size, patch_size=patch_size)
			elif args.segmenter_version == '2':
				from segmenter_v2 import Segmenter
				# def __init__(self, backbone, cls, img_dim, patch_size, decoder=None, cut_enc=-3, cut_in_dec=4, cut_out_dec=-3)
				# encoder: vit_large_patch16_224 - decoder: vit_small_patch16_224
				segmenter = Segmenter(backbone=encoder, cls=num_classes, img_dim = vit_img_size, patch_size=patch_size, decoder=decoder)
			else:
				raise ValueError(f"segmenter version '{args.segmenter_version}' not supported")
			learn     = Learner(dls=dls, model=segmenter, loss_func=CrossEntropyLossFlat(axis=1), metrics=metrics, wd=1e-4).to_fp16()
		else:
			raise ValueError(f"task '{task}' not supported")
			#learn = vision_learner(dls, encoder, pretrained=False, n_out=97, loss_func=CrossEntropyLossFlat(), metrics=metrics, cbs=[MixedPrecision])
	else:
		print(f'Unknown encoder: {encoder}')
		return

	#learn = vision_learner(dls, encoder, pretrained=False, n_out=97, loss_func=CrossEntropyLossFlat(), metrics=metrics, cbs=[MixedPrecision])
	learn.model = learn.model.to(device)
	print(f'{learn.model}')
	#learn.model.load_state_dict(torch.load(model_path, map_location=device))
	#learn = load_learner(model_path, cpu=False, pickle_module=dill)
	#print(f'{type(learn)}')
	#print(f'{type(learn.model)}')
	#print(f'{learn}')
	#print(f'{learn.model}')

	model_path = Path(model_path).resolve().with_suffix('')

	print(f'Loading model: {model_path} to device: {device}')
	learn.load(model_path, device=device, weights_only=False)
	#learn.dls = dls
	return learn

def write_batch_log(log_entries, output_dir):
	"""Write batch results to a log file in the output directory."""
	log_file = output_dir / "inference_log.txt"
	with open(log_file, "a", encoding="utf-8") as f:
		for entry in log_entries:
			f.write(f"{entry}\n")


def resize_mask_and_blend_with_orig_img(orig_img, mask, img_size=540):
	#orig_img = np.array(Image.open(file_path))
	orig_h, orig_w	= orig_img.shape[:2]
	
	# Compute padding parameters
	ratio		= min(img_size / orig_h, img_size / orig_w)
	new_h, new_w	= int(orig_h * ratio), int(orig_w * ratio)
	pad_top		= (img_size - new_h) // 2
	pad_left	= (img_size - new_w) // 2

	# Process mask: crop padding -> resize to original
	#mask		= clss[i].cpu().numpy()
	cropped_mask	= mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
	resized_mask	= cv2.resize(cropped_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
	
	# Overlay mask on original image
	blend		= overlay(orig_img, resized_mask, alpha=0.5)
	#save_path	= output_dir / Path(file_path).name
	#cv2.imwrite(str(save_path), blend[..., ::-1])			# RGB to BGR for OpenCV
	return blend, resized_mask

def process_batch(batch, model, files, output_dir, device, args, debug=False):
	with torch.no_grad():
		print(f'{Text(batch, "batch"):inspect}')
		preds = model(batch.to(device))
		if debug:
			#print(f'Predictions: {preds}')
			print(f'{Text(preds, "preds"):inspect}')
		probs = F.softmax(preds, dim=1)
		if debug:
			#print(f'Probabilities: {probs}')
			print(f'{Text(probs, "probs"):inspect}')
		confs, clss = torch.max(probs, dim=1)
		if debug:
			#print(f'Confidences: {confs} - Classes: {clss}')
			print(f'{Text(confs, "confs"):content}')
			print(f'{Text(clss,  "clss"):content}')
			print(f'{clss[clss != 0].shape = }')

	# Unnormalize batch images (if needed for visualization)
	norm = Normalize.from_stats(*imagenet_stats)
	batch_imgs = norm.decode(batch).permute(0, 2, 3, 1).cpu().numpy()  # [batch, H, W, 3]
	#show_image(norm_apply_denorm(timg, f, nrm)[0]);

	log_entries = []
	#for file_path, cls, conf, batch_itm in zip(files, clss.cpu(), confs.cpu(), batch.cpu()):
	for file_path, cls, conf, batch_itm in zip(files, clss.cpu(), confs.cpu(), batch_imgs):
		src_path = Path(file_path)
		file_id  = src_path.stem
		fid = file_id.split('-')[0]			# e.g. obj 1040
		if args.task == 'classification':
			sub_output_dir = output_dir / fid
			sub_output_dir.mkdir(parents=True, exist_ok=True)
			cls      = cls.item()
			conf     = conf.item()
			new_stem = f"{file_path.stem}-cls-{cls:02d}-confidence-{conf:.3f}"
			# Log entry formatting (class zero-padded to 2 digits)
			log_entries.append(f"{file_path.stem} - {cls:02d} - {conf:.3f}")
			new_path = sub_output_dir / f"{new_stem}{file_path.suffix}"
			#shutil.copy2(file_path, new_path)
			print(f'Hardlinking from {src_path} to {new_path}')
			#src_path.hardlink_to(new_path)
			new_path.hardlink_to(src_path)
		elif args.task == 'segmentation':
			if cls.shape[0] % args.img_size != 0:
				print(f'cls.shape  = {cls.shape} is not divisible by {args.img_size}')
			if conf.shape[0] % args.img_size != 0:
				print(f'conf.shape = {conf.shape} is not divisible by {args.img_size}')
			new_blend_stem	= f"{file_path.stem}-blend-{cls.shape[0]}-{cls.shape[1]}"
			new_mask_stem	= f"{file_path.stem}-mask-{cls.shape[0]}-{cls.shape[1]}"
			log_entries.append(f"{file_path.stem} - {cls.shape} - {conf.shape}")
			new_blend_path	= output_dir / f"{new_blend_stem}{file_path.suffix}"
			new_blend_spath	= output_dir / f"{new_blend_stem}-scaled{file_path.suffix}"
			new_blend_opath	= output_dir / f"{new_blend_stem}-orig{file_path.suffix}"
			new_mask_path	= output_dir / f"{new_mask_stem}.png"
			mask		= cls.cpu().numpy().astype(np.uint8)
			mask		= change_mask_color(mask, (1, 1, 1), (  0,   0, 255))		# TODO: hardcoded! write an argparse entry for this!
			mask		= change_mask_color(mask, (2, 2, 2), (  0, 255,   0))		# TODO: hardcoded! write an argparse entry for this!
			mask		= change_mask_color(mask, (3, 3, 3), (255,   0,   0))		# TODO: hardcoded! write an argparse entry for this!
			mask		= change_mask_color(mask, (4, 4, 4), (  0, 255, 255))		# TODO: hardcoded! write an argparse entry for this!
			cv2.imwrite(new_mask_path, mask)
			img		= cv2.imread(file_path)
			blend, rsz_mask = resize_mask_and_blend_with_orig_img(img, mask, img_size=args.img_size)
			cv2.imwrite(new_blend_path, blend)
			itm_tfm		= item_transform(args.img_size)
			img		= itm_tfm(Image.fromarray(img))					# Fast.ai Resize() works only on PIL Images...
			img		= np.array(img)
			#cv2.imwrite(f'/tmp/asdf-{new_blend_stem}.jpg', img)
			#blend		= overlay(img, mask)
			#blend		= overlay(img, mask)
			#cv2.imwrite(new_blend_path, blend)
			if debug:
				print(f'{Text(batch_itm, "batch_itm"):content}')
			#batch_img	= np.asarray(to_image(batch_itm))
			#batch_img	= np.asarray(to_image(batch_itm))
			#print(f'{Text(batch_img, "batch_img"):inspect}')
			scaled_img	= cv2.resize(center_crop(img), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
			blend_scaled	= overlay(scaled_img, mask)
			cv2.imwrite(new_blend_spath, blend_scaled)
			tfm2img		= T.ToPILImage()
			#batch_img	= np.asarray(tfm2img(batch_itm).convert('RGB'))

			# Inside the batch loop (use instead of original image blending)
			batch_img	= (batch_itm * 255).astype(np.uint8)
			blend_padded	= overlay(batch_img[..., ::-1], mask)
			cv2.imwrite(new_blend_opath, blend_padded)

			if debug:
				print(f'{Text(batch_img, "batch_img"):content}')
			#blend_orig	= overlay(batch_img, mask)
			#cv2.imwrite(new_blend_opath, blend_orig)

	# Write all entries for this batch
	write_batch_log(log_entries, output_dir)

