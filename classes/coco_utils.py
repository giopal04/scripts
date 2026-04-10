import cv2
import numpy as np

def mask_to_polygons(mask_np):
	"""Convert a binary uint8 mask (H, W) to a list of COCO polygon coordinate lists.

	Each contour with at least 3 points is returned as a flat list
	[x1, y1, x2, y2, ...], matching the COCO ``segmentation`` field format.
	Contours with fewer than 3 points are silently dropped.

	Args:
		mask_np: (H, W) uint8 array with non-zero pixels marking the object.

	Returns:
		List of flat int lists — one per connected component.  May be empty
		if the mask is entirely zero or all contours are degenerate.
	"""
	contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	polygons = []
	for c in contours:
		if c.size >= 6:   # need at least 3 (x, y) pairs
			polygons.append(c.flatten().tolist())
	return polygons


def mask_to_bbox_area(mask_np):
	"""Compute the COCO bounding box and pixel area of a binary uint8 mask.

	Args:
		mask_np: (H, W) uint8 array with non-zero pixels marking the object.

	Returns:
		Tuple (bbox, area) where *bbox* is [x, y, width, height] (COCO format,
		all ints) and *area* is the integer number of non-zero pixels.
		Returns ([0, 0, 0, 0], 0) for an empty mask.
	"""
	rows = np.any(mask_np, axis=1)
	cols = np.any(mask_np, axis=0)
	if not rows.any():
		return [0, 0, 0, 0], 0
	rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
	cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
	bbox = [cmin, rmin, cmax - cmin + 1, rmax - rmin + 1]
	area = int(mask_np.astype(bool).sum())
	return bbox, area

