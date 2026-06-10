#!/usr/bin/env python3
'''
COCO Dataset Viewer using OpenCV + Qt.

Run with:
---------

./coco-viewer-qt.py -a /tmp/tiny/__tiny-sam3-video-dataset-v2-phase-2-deduplicated-coco.json -i /tmp/tiny

OR

./coco-viewer-qt.py -a /tmp/tiny/__tiny-sam3-video-dataset-v2-phase-2-deduplicated-coco.json -i /tmp/tiny/ --output-video /tmp/out.mp4 --video-fps 25

OR

coco-viewer-qt.py -a . --format ade20k --default-show-classes 'statue,monument,column,pedestal,crack'
coco-viewer-qt.py -a . --format ade20k --default-show-classes 'statue,sculpture,monument,column,pedestal,crack'


Features
--------
* OpenCV-based image loading, masks, bounding boxes, labels, and saving.
* PySide6 Qt GUI instead of Tkinter.
* Fast keyboard jumps for large datasets.
* Parent-directory selector and parent-group navigation.
* Auto-detected plain/compressed COCO annotations: .json, .json.gz, .json.bz2, .json.xz.
* Optional headless video export via --output-video using ffmpeg_utils.py.
* --default-show-classes: comma- or dash-separated list of category names shown by default.
* Delete key: removes current image & annotations.
  - ADE20K mode: moves files (image + JSON + segmentation PNG) into __deleted__/ on disk.
  - COCO mode:  moves image file into __deleted__/ on disk; annotations stay in memory.
    Press W to flush kept and deleted annotation files to disk.

Install runtime deps, for example:
	pip install opencv-python numpy PySide6

Video export also requires ffmpeg/ffprobe on PATH.
'''

from __future__ import annotations

import argparse
import colorsys
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import cv2
import numpy as np

try:
	from classes.ffmpeg_utils import finalize_ffmpeg, start_ffmpeg_streaming_v2, write_frame_to_ffmpeg
except ImportError as exc:
	# Get the absolute path of the directory 2 levels up (the repo root)
	root_dir = Path(__file__).resolve().parent.parent
	sys.path.append(str(root_dir))
	try:
		from classes.ffmpeg_utils import finalize_ffmpeg, start_ffmpeg_streaming_v2, write_frame_to_ffmpeg
	except ImportError:
		def start_ffmpeg_streaming_v2(*_, **__):
			raise RuntimeError("Video export requires classes.ffmpeg_utils.py on PYTHONPATH.")

		def write_frame_to_ffmpeg(*_, **__):
			raise RuntimeError("Video export requires classes.ffmpeg_utils.py on PYTHONPATH.")

		def finalize_ffmpeg(*_, **__):
			return None

try:
	from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - runtime guidance only
	raise SystemExit(
		"PySide6 is required for the GUI. Install it with: pip install PySide6\n"
		"For headless video export, PySide6 is still imported by this script."
	) from exc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass(frozen=True)
class RenderOptions:
	bboxes_on: bool = True
	labels_on: bool = True
	masks_on: bool = True
	object_based_coloring: bool = False
	bbox_width: int = 3
	mask_alpha: int = 128
	label_size: int = 15
	ignore: tuple[int, ...] = ()


class Data:
	"""COCO data access and iteration.

	Internally every supported input format is normalized to the small COCO-like
	structure used by the renderer: images, annotations, categories.
	"""

	def __init__(self, image_dir: str | Path, annotations_file: str | Path, dataset_format: str = "coco"):
		self.image_dir = Path(image_dir)
		self.dataset_format = dataset_format
		self.annotations_file = Path(annotations_file)
		if dataset_format == "ade20k":
			self.instances, images, self.categories = parse_ade20k_tree(annotations_file, self.image_dir)
		else:
			self.instances, images, self.categories = parse_coco(annotations_file)

		if self.instances is None or images is None or self.categories is None:
			self.images			= None
			self.annotations_by_image	= None
			self.current_image		= None, None
			self.categories			= None
			return

		# In COCO mode we keep a separate list of annotations moved out by the delete action.
		# These are flushed to disk only on Key_W.
		self.deleted_annos: list[dict] = []
		self.deleted_image_ids: set[int] = set()

		self.annotations_by_image = group_annotations_by_image(self.instances.get("annotations", []))
		self.images = ImageList(images)
		logging.info(
			"Parsed %d images, %d annotations, %d categories.",
			len(self.images),
			len(self.instances.get("annotations", [])),
			len(self.categories),
		)
		self.current_image = self.images.next()

	def prepare_image(self, object_based_coloring: bool = False):
		img_id, img_name = self.current_image
		full_path = self.image_dir / img_name
		objects = self.annotations_by_image.get(img_id, [])
		obj_category_ids = [obj["category_id"] for obj in objects]
		img_categories = sorted(set(obj_category_ids))
		names_colors = [self.categories[i] for i in obj_category_ids]

		if object_based_coloring:
			obj_colors = prepare_colors(len(objects))
			names_colors = [[names_colors[i][0], obj_colors[i]] for i in range(len(objects))]

		return full_path, objects, names_colors, obj_category_ids, img_categories

	def next_image(self):
		self.current_image = self.images.next()

	def previous_image(self):
		self.current_image = self.images.prev()

	def jump_images(self, offset: int):
		self.current_image = self.images.jump(offset)

	def set_image_index(self, index: int):
		self.current_image = self.images.set_index(index)


def parse_coco(annotations_file: str | Path) -> tuple[dict, list[tuple[int, str]], dict[int, list]]:
	instances = load_annotations(annotations_file)
	if instances is None:
		return None, None, None
	return instances, get_images(instances), get_categories(instances)


def parse_ade20k_tree(annotations_path: str | Path, image_root: str | Path) -> tuple[dict, list[tuple[int, str]], dict[int, list]]:
	"""Read ADE20K-style per-image JSON files from a file or directory tree.

	Each JSON is expected to contain an ``annotation`` object with a filename and an
	``object`` list. Object polygons are converted to COCO polygon segmentation and
	bounding boxes so the existing viewer code can draw them unchanged.
	"""
	annotations_path = Path(annotations_path)
	image_root = Path(image_root)
	json_files = list_annotation_files(annotations_path)
	if not json_files:
		raise ValueError(f"No JSON annotation files found under: {annotations_path}")

	images: list[dict[str, Any]] = []
	annotations: list[dict[str, Any]] = []
	category_name_to_id: dict[str, int] = {}
	category_id_to_name: dict[int, str] = {}
	image_id = 1
	annotation_id = 1

	for json_path in json_files:
		try:
			payload = load_annotations(json_path)
		except Exception as exc:
			logging.warning("Skipping unreadable JSON %s: %s", json_path, exc)
			continue

		ade = payload.get("annotation", payload)
		if not isinstance(ade, dict) or "object" not in ade:
			logging.debug("Skipping non-ADE JSON: %s", json_path)
			continue

		filename = str(ade.get("filename") or json_path.with_suffix(".jpg").name)
		image_path = resolve_ade_image_path(json_path, annotations_path, image_root, ade, filename)
		file_name = path_for_image_list(image_path, image_root)
		height, width = ade_image_size(ade)
		images.append({"id": image_id, "file_name": file_name, "height": height, "width": width})

		objects = ade.get("object", [])
		if isinstance(objects, dict):
			objects = [objects]
		for obj in objects or []:
			if not isinstance(obj, dict):
				continue
			polygon = ade_polygon_to_flat_points(obj.get("polygon"))
			if len(polygon) < 6:
				continue
			name = str(obj.get("name") or obj.get("raw_name") or "object").strip() or "object"
			category_id = ade_category_id(obj, name, category_name_to_id, category_id_to_name)
			bbox = bbox_from_polygon(polygon)
			annotations.append({
				"id": annotation_id,
				"image_id": image_id,
				"category_id": category_id,
				"bbox": bbox,
				"segmentation": [polygon],
				"area": polygon_area(polygon),
				"iscrowd": 0,
				"ade_object": obj,
			})
			annotation_id += 1
		image_id += 1

	categories = [{"id": cat_id, "name": name} for cat_id, name in sorted(category_id_to_name.items())]
	instances = {
		"info": {"description": f"ADE20K-style per-image annotations from {annotations_path}"},
		"images": images,
		"annotations": annotations,
		"categories": categories,
	}
	return instances, get_images(instances), get_categories(instances)


def list_annotation_files(path: str | Path) -> list[Path]:
	path = Path(path)
	if path.is_file():
		return [path]
	patterns = ("*.json", "*.json.gz", "*.json.bz2", "*.json.xz")
	files: list[Path] = []
	for pattern in patterns:
		print(f'Scanning for JSON files in {path} with pattern: {pattern}')
		tmpfiles = path.rglob(pattern)
		# Skip anything inside a __deleted__ directory.
		files.extend(f for f in tmpfiles if "__deleted__" not in f.parts)
	return sorted(set(files))


def ade_image_size(ade: dict) -> tuple[int | None, int | None]:
	imsize = ade.get("imsize") or []
	if len(imsize) >= 2:
		return int(imsize[0]), int(imsize[1])
	return None, None


def resolve_ade_image_path(json_path: Path, annotations_root: Path, image_root: Path, ade: dict, filename: str) -> Path:
	candidates: list[Path] = []
	candidates.append(json_path.with_name(filename))
	candidates.append(image_root / filename)

	folder = ade.get("folder")
	if folder:
		folder_path = Path(str(folder))
		candidates.append(image_root / folder_path / filename)
		candidates.append(folder_path / filename)

	try:
		rel_json_parent = json_path.parent.relative_to(annotations_root if annotations_root.is_dir() else annotations_root.parent)
		candidates.append(image_root / rel_json_parent / filename)
	except ValueError:
		pass

	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()
	return candidates[0]


def path_for_image_list(image_path: Path, image_root: Path) -> str:
	try:
		return str(image_path.resolve().relative_to(image_root.resolve()))
	except Exception:
		return str(image_path)


def ade_polygon_to_flat_points(polygon: Any) -> list[float]:
	if not isinstance(polygon, dict):
		return []
	xs = polygon.get("x") or []
	ys = polygon.get("y") or []
	points: list[float] = []
	for x, y in zip(xs, ys):
		try:
			points.extend([float(x), float(y)])
		except (TypeError, ValueError):
			continue
	return points


def bbox_from_polygon(flat_points: list[float]) -> list[float]:
	xs = flat_points[0::2]
	ys = flat_points[1::2]
	min_x, max_x = min(xs), max(xs)
	min_y, max_y = min(ys), max(ys)
	return [min_x, min_y, max_x - min_x, max_y - min_y]


def polygon_area(flat_points: list[float]) -> float:
	points = np.asarray(flat_points, dtype=np.float32).reshape(-1, 2)
	if len(points) < 3:
		return 0.0
	return float(abs(cv2.contourArea(points)))


def ade_category_id(
	obj: dict,
	name: str,
	category_name_to_id: dict[str, int],
	category_id_to_name: dict[int, str],
) -> int:
	name_ndx = obj.get("name_ndx")
	try:
		candidate = int(name_ndx)
	except (TypeError, ValueError):
		candidate = None

	if candidate is not None and candidate not in category_id_to_name:
		category_id_to_name[candidate] = name
		category_name_to_id[name] = candidate
		return candidate
	if candidate is not None and category_id_to_name.get(candidate) == name:
		return candidate
	if name in category_name_to_id:
		return category_name_to_id[name]

	next_id = max(category_id_to_name.keys(), default=0) + 1
	while next_id in category_id_to_name:
		next_id += 1
	category_id_to_name[next_id] = name
	category_name_to_id[name] = next_id
	return next_id


def detect_annotation_compression(path: str | Path) -> str | None:
	"""Return the compression type for a COCO annotation file, or None for plain JSON.

	Detection prefers file magic bytes, then falls back to the final suffix. This keeps
	standard .json/.json.gz/.json.bz2/.json.xz paths transparent while also handling
	compressed files whose extension is missing or non-standard.
	"""
	path = Path(path)
	with open(path, "rb") as f:
		header = f.read(6)

	if header.startswith(b"\x1f\x8b"):
		return "gzip"
	if header.startswith(b"BZh"):
		return "bzip2"
	if header.startswith(b"\xfd7zXZ\x00"):
		return "xz"

	suffix = path.suffix.lower()
	if suffix == ".gz":
		return "gzip"
	if suffix == ".bz2":
		return "bzip2"
	if suffix == ".xz":
		return "xz"
	return None


def open_annotation_text(path: str | Path):
	"""Open a plain or compressed annotation file as UTF-8 text.

	The decompressor modules are imported lazily so that plain .json startup stays
	identical to before and optional compression support is only loaded on demand.
	"""
	path = Path(path)
	compression = detect_annotation_compression(path)
	if compression == "gzip":
		import gzip

		return gzip.open(path, "rt", encoding="utf-8")
	if compression == "bzip2":
		import bz2

		return bz2.open(path, "rt", encoding="utf-8")
	if compression == "xz":
		import lzma

		return lzma.open(path, "rt", encoding="utf-8")
	return path.open("r", encoding="utf-8")


def load_annotations(fname: str | Path) -> dict:
	if not Path(fname).is_file():
		print(f'Annotations path should be a file while in COCO mode (chose ADE20K mode with --format=ade20k if you have multiple JSON files scattered through directories...)')
		return None
	compression = detect_annotation_compression(fname)
	detail = "plain JSON" if compression is None else f"{compression}-compressed JSON"
	logging.info("Parsing %s (%s)...", fname, detail)
	with open_annotation_text(fname) as f:
		return json.load(f)


def get_images(instances: dict) -> list[tuple[int, str]]:
	return [(image["id"], image["file_name"]) for image in instances.get("images", [])]


def group_annotations_by_image(annotations: Iterable[dict]) -> dict[int, list[dict]]:
	grouped: dict[int, list[dict]] = {}
	for ann in annotations:
		grouped.setdefault(ann["image_id"], []).append(ann)
	return grouped


def prepare_colors(n_objects: int, shuffle: bool = True) -> list[tuple[int, int, int]]:
	if n_objects <= 0:
		return []
	hsv_tuples = [(x / n_objects, 1.0, 1.0) for x in range(n_objects)]
	colors = [tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_tuples]
	if shuffle:
		random.seed(42)
		random.shuffle(colors)
		random.seed(None)
	return colors


def get_categories(instances: dict) -> dict[int, list]:
	categories = instances.get("categories", [])
	colors = prepare_colors(max(1, len(categories)), shuffle=True)
	return {cat["id"]: [cat["name"], colors[i % len(colors)]] for i, cat in enumerate(categories)}


def load_rgb_image(path: str | Path) -> np.ndarray:
	bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if bgr is None:
		#raise FileNotFoundError(f"Could not read image: {path}")
		print(f"Could not read image: {path}")
		# create a 1008x1008 px black image
		bgr = np.zeros((1008, 1008, 3), dtype=np.uint8)
	return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rle_to_mask(rle, height: int, width: int) -> np.ndarray:
	"""Decode the uncompressed COCO RLE format used by the original script."""
	counts = np.asarray(rle, dtype=np.int64).reshape(-1, 2)
	flat = np.zeros(height * width, dtype=np.uint8)
	offset = 0
	for index, length in counts:
		offset += int(index)
		flat[offset : offset + int(length)] = 255
		offset += int(length)
	return flat.reshape((width, height)).T


def draw_masks_cv2(
	image_rgb: np.ndarray,
	objects: list[dict],
	obj_categories: list[list],
	ignore: set[int],
	alpha: int,
) -> np.ndarray:
	if alpha <= 0:
		return image_rgb

	overlay = image_rgb.copy()
	h, w = image_rgb.shape[:2]
	blend = float(np.clip(alpha, 0, 255)) / 255.0

	for i, (category, obj) in enumerate(zip(obj_categories, objects)):
		if i in ignore:
			continue
		color = tuple(int(v) for v in category[-1])
		segmentation = obj.get("segmentation")

		if isinstance(segmentation, list):
			for polygon in segmentation:
				if not polygon:
					continue
				points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
				points = np.round(points).astype(np.int32)
				if len(points) >= 3:
					cv2.fillPoly(overlay, [points], color=color)
		elif isinstance(segmentation, dict) and obj.get("iscrowd"):
			counts = segmentation.get("counts")
			if isinstance(counts, list):
				mask = rle_to_mask(counts, segmentation["size"][0], segmentation["size"][1])
				if mask.shape[:2] != (h, w):
					mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
				overlay[mask > 0] = color

	return cv2.addWeighted(overlay, blend, image_rgb, 1.0 - blend, 0)


def draw_bboxes_cv2(
	image_rgb: np.ndarray,
	objects: list[dict],
	labels: bool,
	obj_categories: list[list],
	ignore: set[int],
	width: int,
	label_size: int,
) -> np.ndarray:
	if width <= 0:
		return image_rgb

	img = image_rgb.copy()
	h, w = img.shape[:2]
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = max(label_size, 1) / 35.0
	text_thickness = max(1, round(width / 2))

	for i, (category, obj) in enumerate(zip(obj_categories, objects)):
		if i in ignore:
			continue
		x, y, bw, bh = obj.get("bbox", [0, 0, 0, 0])
		x0 = int(round(x))
		y0 = int(round(y))
		x1 = int(round(x + bw))
		y1 = int(round(y + bh))
		x0, y0 = max(0, x0), max(0, y0)
		x1, y1 = min(w - 1, x1), min(h - 1, y1)
		color = tuple(int(v) for v in category[-1])

		cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=width, lineType=cv2.LINE_AA)

		if labels:
			text = str(category[0])
			(tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
			label_y0 = max(0, y0 - th - baseline - 4)
			label_y1 = min(h - 1, label_y0 + th + baseline + 4)
			label_x0 = x0
			label_x1 = min(w - 1, x0 + tw + 6)
			cv2.rectangle(img, (label_x0, label_y0), (label_x1, label_y1), color, thickness=-1)
			cv2.putText(
				img,
				text,
				(label_x0 + 3, label_y1 - baseline - 2),
				font,
				font_scale,
				(255, 255, 255),
				text_thickness,
				cv2.LINE_AA,
			)

	return img


def compose_image(data: Data, options: RenderOptions) -> tuple[np.ndarray, list[int], list[int]]:
	full_path, objects, names_colors, img_obj_categories, img_categories = data.prepare_image(
		options.object_based_coloring
	)
	image = load_rgb_image(full_path)
	ignore = set(options.ignore)
	if options.masks_on:
		image = draw_masks_cv2(image, objects, names_colors, ignore, options.mask_alpha)
	if options.bboxes_on:
		image = draw_bboxes_cv2(
			image,
			objects,
			options.labels_on,
			names_colors,
			ignore,
			options.bbox_width,
			options.label_size,
		)
	return image, img_obj_categories, img_categories


def compose_rgb_mask_bbox_headless(data: Data, options: RenderOptions) -> np.ndarray:
	"""Return a side-by-side RGB | mask | bbox panel for video export."""
	full_path, objects, names_colors, _, _ = data.prepare_image(options.object_based_coloring)
	rgb = load_rgb_image(full_path)
	ignore = set(options.ignore)
	overlay = draw_masks_cv2(rgb, objects, names_colors, ignore, options.mask_alpha)
	overlay = draw_bboxes_cv2(overlay, objects, options.labels_on, names_colors, ignore, options.bbox_width, options.label_size)
	return np.ascontiguousarray(overlay)


class ImageList:
	def __init__(self, images: list[tuple[int, str]]):
		self.image_list = images or []
		self.n = -1
		self.max = len(self.image_list)
		if not self.image_list:
			raise ValueError("No images found in annotation file.")

	def next(self):
		self.n = (self.n + 1) % self.max
		return self.image_list[self.n]

	def prev(self):
		self.n = (self.n - 1) % self.max
		return self.image_list[self.n]

	def jump(self, offset: int):
		self.n = (self.n + offset) % self.max
		return self.image_list[self.n]

	def set_index(self, index: int):
		self.n = index % self.max
		return self.image_list[self.n]

	def __len__(self):
		return self.max


class ImageViewer(QtWidgets.QMainWindow):
	def __init__(self, data: Data, default_show_classes: set[str] | None = None):
		super().__init__()
		self.data = data
		self.default_show_classes: set[str] = default_show_classes or set()
		self.current_image_rgb: np.ndarray | None = None
		self.current_img_obj_categories: list[int] = []
		self.current_img_categories: list[int] = []
		self.selected_cats: set[int] | None = None
		self.selected_objs: set[int] | None = None
		self.parent_depth = 0
		self.source_pixmap: QtGui.QPixmap | None = None
		# Cache of image dicts for images deleted in COCO mode (for the __deleted__ output file).
		self._deleted_images_cache: list[dict] = []

		self.setWindowTitle("COCO Viewer - OpenCV + Qt")
		self.resize(1200, 800)
		self._build_ui()
		self._bind_shortcuts()
		self.showMaximized()
		app = QtWidgets.QApplication.instance()
		if app is not None:
			app.installEventFilter(self)
		self.update_image(local=False)

	def _build_ui(self):
		central = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(central)
		body = QtWidgets.QHBoxLayout()
		layout.addLayout(body, stretch=1)

		self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
		self.image_label.setBackgroundRole(QtGui.QPalette.Dark)
		self.image_label.setMinimumSize(1, 1)
		self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
		self.scroll = QtWidgets.QScrollArea(widgetResizable=True)
		self.scroll.setAlignment(QtCore.Qt.AlignCenter)
		self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.scroll.setWidget(self.image_label)
		self.scroll.viewport().installEventFilter(self)
		body.addWidget(self.scroll, stretch=1)

		side = QtWidgets.QWidget()
		side_layout = QtWidgets.QVBoxLayout(side)
		side_layout.addWidget(QtWidgets.QLabel("categories"))
		self.category_list = QtWidgets.QListWidget(selectionMode=QtWidgets.QAbstractItemView.ExtendedSelection)
		side_layout.addWidget(self.category_list, stretch=1)
		side_layout.addWidget(QtWidgets.QLabel("objects"))
		self.object_list = QtWidgets.QListWidget(selectionMode=QtWidgets.QAbstractItemView.ExtendedSelection)
		side_layout.addWidget(self.object_list, stretch=1)
		side_layout.addWidget(QtWidgets.QLabel("parent selector"))
		self.parent_combo = QtWidgets.QComboBox()
		self.parent_combo.setToolTip(
			"J/Ctrl+J changes the selected parent directory. "
			"N/Ctrl+N jumps to the next/previous image with a different selected parent."
		)
		side_layout.addWidget(self.parent_combo)
		#self.parent_path_label = QtWidgets.QLabel()
		#self.parent_path_label.setWordWrap(False)
		#self.parent_path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
		#side_layout.addWidget(self.parent_path_label)
		body.addWidget(side, stretch=0)

		box = QtWidgets.QGroupBox("parent")
		box_layout = QtWidgets.QVBoxLayout(box)
		slider_layout = QtWidgets.QHBoxLayout()
		self.parent_path_label = QtWidgets.QLabel()
		self.parent_path_label.setWordWrap(False)
		self.parent_path_label.setMinimumWidth(1)
		self.parent_path_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
		self.parent_path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
		#layout.addLayout(slider_layout)
		self.bbox_slider	= self._make_slider("bbox",   0,  25,   3, slider_layout)
		self.label_slider	= self._make_slider("label", 10, 100,  15, slider_layout)
		self.mask_slider	= self._make_slider("mask",   0, 255, 128, slider_layout)
		box_layout.addWidget(self.parent_path_label)
		#box_layout.addWidget(slider_layout)
		box_layout.addLayout(slider_layout)
		layout.addWidget(box)

		self.status		= self.statusBar()
		self.file_label		= QtWidgets.QLabel()
		self.parent_label	= QtWidgets.QLabel()
		self.count_label	= QtWidgets.QLabel()
		self.object_label	= QtWidgets.QLabel()
		self.category_label	= QtWidgets.QLabel()
		for widget in (self.file_label, self.parent_label, self.object_label, self.category_label, self.count_label):
			self.status.addPermanentWidget(widget)

		self.bboxes_on = QtGui.QAction("BBoxes", self, checkable=True, checked=True)
		self.labels_on = QtGui.QAction("Labels", self, checkable=True, checked=True)
		self.masks_on = QtGui.QAction("Masks", self, checkable=True, checked=True)
		self.object_coloring = QtGui.QAction("Object colors", self, checkable=True, checked=False)

		file_menu = self.menuBar().addMenu("File")
		save_action = file_menu.addAction("Save")
		save_action.setShortcut("Ctrl+S")
		save_action.triggered.connect(self.save_image)
		quit_action = file_menu.addAction("Exit")
		quit_action.setShortcut("Ctrl+Q")
		quit_action.triggered.connect(self.close)

		view_menu = self.menuBar().addMenu("View")
		for action in (self.bboxes_on, self.labels_on, self.masks_on, self.object_coloring):
			view_menu.addAction(action)
			action.triggered.connect(self.update_image)

		self.category_list.itemSelectionChanged.connect(self.select_category)
		self.object_list.itemSelectionChanged.connect(self.select_object)
		self.parent_combo.currentIndexChanged.connect(self.select_parent_depth)
		for slider in (self.bbox_slider, self.label_slider, self.mask_slider):
			slider.valueChanged.connect(self.update_image)

		self.setCentralWidget(central)

	def _make_slider(self, label: str, minimum: int, maximum: int, value: int, layout: QtWidgets.QHBoxLayout):
		box = QtWidgets.QGroupBox(label)
		box_layout = QtWidgets.QVBoxLayout(box)
		slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=minimum, maximum=maximum, value=value)
		box_layout.addWidget(slider)
		layout.addWidget(box)
		return slider

	def _bind_shortcuts(self):
		bindings = {
			QtGui.QKeySequence(QtCore.Qt.Key_Right): self.next_img,
			QtGui.QKeySequence(QtCore.Qt.Key_Left): self.prev_img,
			QtGui.QKeySequence(QtCore.Qt.Key_K): self.prev_img,
			QtGui.QKeySequence(QtCore.Qt.Key_B): self.toggle_bboxes,
			QtGui.QKeySequence(QtCore.Qt.Key_L): self.toggle_labels,
			QtGui.QKeySequence(QtCore.Qt.Key_M): self.toggle_masks,
			QtGui.QKeySequence(QtCore.Qt.Key_Space): self.toggle_all,
			QtGui.QKeySequence(QtCore.Qt.Key_Delete): self.delete_current,
			QtGui.QKeySequence(QtCore.Qt.Key_W): self.save_annotations,
		}
		for sequence, callback in bindings.items():
			shortcut = QtGui.QShortcut(sequence, self)
			shortcut.activated.connect(callback)

	def eventFilter(self, obj, event):
		key_press_type = QtCore.QEvent.Type.KeyPress if hasattr(QtCore.QEvent, "Type") else QtCore.QEvent.KeyPress
		resize_type = QtCore.QEvent.Type.Resize if hasattr(QtCore.QEvent, "Type") else QtCore.QEvent.Resize
		if obj is self.scroll.viewport() and event.type() == resize_type:
			QtCore.QTimer.singleShot(0, self._fit_image_to_viewport)
		if self.isActiveWindow() and event.type() == key_press_type:
			if self._handle_navigation_key(event):
				return True
		return super().eventFilter(obj, event)

	def _handle_navigation_key(self, event) -> bool:
		modifiers = event.modifiers()
		if modifiers & (QtCore.Qt.AltModifier | QtCore.Qt.MetaModifier | QtCore.Qt.ShiftModifier):
			return False
		direction = -1 if modifiers & QtCore.Qt.ControlModifier else 1
		key = event.key()

		if key == QtCore.Qt.Key_1:
			self.jump_images(direction * 10)
		if key == QtCore.Qt.Key_2:
			self.jump_images(direction * 100)
		elif key == QtCore.Qt.Key_3:
			self.jump_images(direction * 1000)
		elif key == QtCore.Qt.Key_4:
			self.jump_images(direction * 10000)
		elif key == QtCore.Qt.Key_J:
			self.change_parent_selector(direction)
		elif key == QtCore.Qt.Key_N:
			self.jump_to_different_parent(direction)
		else:
			return False

		event.accept()
		return True

	def _full_path_for_image(self, image: tuple[int, str]) -> Path:
		return self.data.image_dir / image[-1]

	def _current_full_path(self) -> Path:
		return self._full_path_for_image(self.data.current_image)

	def _parent_dirs_for_path(self, path: Path) -> list[Path]:
		return [parent for parent in (path.parent, *path.parent.parents) if parent.name]

	def _current_parent_dirs(self) -> list[Path]:
		return self._parent_dirs_for_path(self._current_full_path())

	def _parent_key_at_index(self, index: int, depth: int) -> str | None:
		parent_dirs = self._parent_dirs_for_path(self._full_path_for_image(self.data.images.image_list[index]))
		if depth < 0 or depth >= len(parent_dirs):
			return None
		return os.path.normpath(str(parent_dirs[depth]))

	def _clamp_parent_depth(self) -> list[Path]:
		parent_dirs = self._current_parent_dirs()
		if not parent_dirs:
			self.parent_depth = 0
		elif self.parent_depth >= len(parent_dirs):
			self.parent_depth = len(parent_dirs) - 1
		elif self.parent_depth < 0:
			self.parent_depth = 0
		return parent_dirs

	def _update_parent_selector(self):
		parent_dirs = self._clamp_parent_depth()
		self.parent_combo.blockSignals(True)
		self.parent_combo.clear()
		if not parent_dirs:
			self.parent_combo.addItem("no parent directories")
			self.parent_combo.setEnabled(False)
			self.parent_path_label.setText("parent: n/a")
			self.parent_path_label.setToolTip("")
			self.parent_label.setText("parent: n/a")
			self.parent_label.setToolTip("")
		else:
			for depth, parent in enumerate(parent_dirs):
				self.parent_combo.addItem(f"{depth + 1}: {parent.name}", str(parent))
			self.parent_combo.setCurrentIndex(self.parent_depth)
			self.parent_combo.setEnabled(True)
			self._update_parent_labels(parent_dirs)
		self.parent_combo.blockSignals(False)

	def _update_parent_labels(self, parent_dirs: list[Path] | None = None):
		if parent_dirs is None:
			parent_dirs = self._clamp_parent_depth()
		if not parent_dirs:
			self.parent_path_label.setText("parent: n/a")
			self.parent_path_label.setToolTip("")
			self.parent_label.setText("parent: n/a")
			self.parent_label.setToolTip("")
			return
		selected_parent = parent_dirs[self.parent_depth]
		self.parent_path_label.setText(str(selected_parent))
		self.parent_path_label.setToolTip(str(selected_parent))
		self.parent_label.setText(f"parent: {selected_parent.name}")
		self.parent_label.setToolTip(str(selected_parent))

	def render_options(self, local: bool = True) -> RenderOptions:
		# Build the ignore set fresh from the live data so this method is safe to call
		# before self.current_img_obj_categories has been updated for the new image.
		img_id, _ = self.data.current_image
		objects = self.data.annotations_by_image.get(img_id, [])
		live_obj_category_ids = [obj["category_id"] for obj in objects]
		n_objs = len(live_obj_category_ids)
		print(f'Found {n_objs} objects with the following IDs: {live_obj_category_ids}')

		if self.selected_objs is not None:
			ignore: tuple[int, ...] = tuple(i for i in range(n_objs) if i not in self.selected_objs)
		elif self.default_show_classes:
			# No explicit user selection, but a default class filter is active.
			# Build the set of object indices whose category name is in the default set.
			visible_objs: set[int] = set()
			for i, category_id in enumerate(live_obj_category_ids):
				name = self.data.categories[category_id][0]
				#print(f'Matching found {i} - {name} with provided --default-show-classes {self.default_show_classes}')
				logging.debug(f'Matching found {i} - {name} with provided --default-show-classes {self.default_show_classes}')
				for class_name in self.default_show_classes:
					if class_name in name:
						logging.debug(f'MATCH!!! {i} - {name}')
						visible_objs.add(i)
			ignore = tuple(i for i in range(n_objs) if i not in visible_objs)
		else:
			ignore = ()
		return RenderOptions(
			bboxes_on=self.bboxes_on.isChecked(),
			labels_on=self.labels_on.isChecked(),
			masks_on=self.masks_on.isChecked(),
			object_based_coloring=self.object_coloring.isChecked(),
			bbox_width=self.bbox_slider.value(),
			mask_alpha=self.mask_slider.value(),
			label_size=self.label_slider.value(),
			ignore=ignore,
		)

	def update_image(self, *_, local: bool = True):
		self.bbox_slider.setEnabled(self.bboxes_on.isChecked())
		self.label_slider.setEnabled(self.labels_on.isChecked())
		self.mask_slider.setEnabled(self.masks_on.isChecked())
		self.current_image_rgb, self.current_img_obj_categories, self.current_img_categories = compose_image(
			self.data, self.render_options(local=local)
		)
		self._show_rgb(self.current_image_rgb)
		self._update_parent_selector()
		self._update_status()
		self._update_category_list()
		self._update_object_list()

	def _show_rgb(self, image_rgb: np.ndarray):
		h, w, ch = image_rgb.shape
		qimage = QtGui.QImage(image_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
		self.source_pixmap = QtGui.QPixmap.fromImage(qimage)
		self.image_label.setToolTip(f"{w}x{h}")
		self._fit_image_to_viewport()
		QtCore.QTimer.singleShot(0, self._fit_image_to_viewport)

	def _fit_image_to_viewport(self):
		if self.source_pixmap is None or self.source_pixmap.isNull():
			return

		available = self.scroll.viewport().size()
		if available.width() <= 1 or available.height() <= 1:
			return

		scaled = self.source_pixmap.scaled(
			available,
			QtCore.Qt.KeepAspectRatio,
			QtCore.Qt.SmoothTransformation,
		)
		self.image_label.setPixmap(scaled)
		self.image_label.resize(available)

	def _update_status(self):
		description = self.data.instances.get("info", {}).get("description", "")
		self.status.showMessage(description)
		self.file_label.setText(self.data.current_image[-1])
		self.file_label.setToolTip(str(self._current_full_path()))
		self._update_parent_labels()
		self.object_label.setText(f"objects: {len(self.current_img_obj_categories)}")
		self.category_label.setText(f"categories: {len(self.current_img_categories)}")
		self.count_label.setText(f"{self.data.images.n + 1}/{self.data.images.max}")

	def _update_category_list(self):
		self.category_list.blockSignals(True)
		self.category_list.clear()
		for category_id in self.current_img_categories:
			name = self.data.categories[category_id][0]
			self.category_list.addItem(f"{category_id} {name}")
		if self.selected_cats is not None:
			for i in self.selected_cats:
				item = self.category_list.item(i)
				if item:
					item.setSelected(True)
		elif self.default_show_classes:
			# Pre-select only the categories whose names match --default-show-classes.
			# selected_cats stays None — render_options handles filtering independently.
			for i, category_id in enumerate(self.current_img_categories):
				name = self.data.categories[category_id][0]
				if name in self.default_show_classes:
					item = self.category_list.item(i)
					if item:
						item.setSelected(True)
		else:
			self.category_list.selectAll()
		self.category_list.blockSignals(False)

	def _update_object_list(self):
		self.object_list.blockSignals(True)
		self.object_list.clear()
		for i, category_id in enumerate(self.current_img_obj_categories):
			name = self.data.categories[category_id][0]
			self.object_list.addItem(f"{i} {name}")
		if self.selected_objs is not None:
			for i in self.selected_objs:
				item = self.object_list.item(i)
				if item:
					item.setSelected(True)
		elif self.default_show_classes:
			# Highlight only objects whose category name is in the default set.
			for i, category_id in enumerate(self.current_img_obj_categories):
				name = self.data.categories[category_id][0]
				if name in self.default_show_classes:
					item = self.object_list.item(i)
					if item:
						item.setSelected(True)
		else:
			self.object_list.selectAll()
		self.object_list.blockSignals(False)

	def select_category(self):
		self.selected_cats = {idx.row() for idx in self.category_list.selectedIndexes()}
		selected_objs = set()
		for category_index in self.selected_cats:
			if category_index >= len(self.current_img_categories):
				continue
			selected_category_id = self.current_img_categories[category_index]
			for i, category_id in enumerate(self.current_img_obj_categories):
				if category_id == selected_category_id:
					selected_objs.add(i)
		self.selected_objs = selected_objs
		self.update_image()

	def select_object(self):
		self.selected_objs = {idx.row() for idx in self.object_list.selectedIndexes()}
		selected_cats = set()
		for object_index in self.selected_objs:
			if object_index >= len(self.current_img_obj_categories):
				continue
			object_category_id = self.current_img_obj_categories[object_index]
			for i, category_id in enumerate(self.current_img_categories):
				if category_id == object_category_id:
					selected_cats.add(i)
		self.selected_cats = selected_cats
		self.update_image()

	def select_parent_depth(self, index: int):
		parent_dirs = self._current_parent_dirs()
		if index < 0 or index >= len(parent_dirs):
			return
		self.parent_depth = index
		self._update_parent_labels(parent_dirs)
		self.status.showMessage(f"Selected parent: {parent_dirs[self.parent_depth]}", 3000)

	def _clear_selection(self):
		self.selected_cats = None
		self.selected_objs = None

	def jump_images(self, offset: int):
		self.data.jump_images(offset)
		self._clear_selection()
		self.update_image(local=False)
		self.status.showMessage(f"Jumped {offset:+d} images.", 3000)

	def change_parent_selector(self, direction: int = 1):
		parent_dirs = self._current_parent_dirs()
		if not parent_dirs:
			self.status.showMessage("No parent directories for the current image.", 3000)
			return
		self.parent_depth = (self.parent_depth + direction) % len(parent_dirs)
		self._update_parent_selector()
		self._update_status()
		self.status.showMessage(f"Selected parent: {parent_dirs[self.parent_depth]}", 3000)

	def jump_to_different_parent(self, direction: int = 1):
		parent_dirs = self._clamp_parent_depth()
		if not parent_dirs:
			self.status.showMessage("No parent directories for the current image.", 3000)
			return

		current_index = self.data.images.n
		current_parent = self._parent_key_at_index(current_index, self.parent_depth)
		if current_parent is None:
			self.status.showMessage("No selected parent directory at this depth.", 3000)
			return

		for offset in range(1, self.data.images.max):
			candidate_index = (current_index + direction * offset) % self.data.images.max
			candidate_parent = self._parent_key_at_index(candidate_index, self.parent_depth)
			if candidate_parent is not None and candidate_parent != current_parent:
				self.data.set_image_index(candidate_index)
				self._clear_selection()
				self.update_image(local=False)
				jump_name = "previous" if direction < 0 else "next"
				self.status.showMessage(f"Jumped to {jump_name} parent: {candidate_parent}", 3000)
				return

		self.status.showMessage(
			f"No different parent directory found at selector depth {self.parent_depth + 1}.",
			3000,
		)

	def next_img(self):
		self.data.next_image()
		self._clear_selection()
		self.update_image(local=False)

	def prev_img(self):
		self.data.previous_image()
		self._clear_selection()
		self.update_image(local=False)

	def save_image(self):
		if self.current_image_rgb is None:
			return
		stem = Path(self.data.current_image[-1]).stem
		path, _ = QtWidgets.QFileDialog.getSaveFileName(
			self,
			"Save image",
			f"{stem}.png",
			"PNG files (*.png);;JPEG files (*.jpg *.jpeg);;All files (*)",
		)
		if path:
			bgr = cv2.cvtColor(self.current_image_rgb, cv2.COLOR_RGB2BGR)
			cv2.imwrite(path, bgr)

	# ------------------------------------------------------------------
	# Delete / save helpers
	# ------------------------------------------------------------------

	def delete_current(self):
		"""Delete the currently displayed image (and its annotations).

		ADE20K mode: moves the image file and its per-image JSON into a
		``__deleted__`` sub-directory relative to the image root.

		COCO mode: moves the image file into a ``__deleted__`` sub-directory,
		transfers its annotations from the live list into ``data.deleted_annos``
		(in memory only), and removes the image entry from ``data.images``.
		Nothing is written to disk until the user presses W.
		"""
		img_id, img_name = self.data.current_image
		if img_name is None:
			return

		full_path = self.data.image_dir / img_name

		if self.data.dataset_format == "ade20k":
			self._delete_ade20k(img_id, img_name, full_path)
		else:
			self._delete_coco(img_id, img_name, full_path)

	def _move_to_deleted(self, src: Path, deleted_root: Path) -> Path:
		"""Move *src* into *deleted_root*, preserving its sub-path relative to
		``self.data.image_dir``.  Returns the destination path."""
		try:
			rel = src.resolve().relative_to(self.data.image_dir.resolve())
		except ValueError:
			rel = Path(src.name)
		dst = deleted_root / rel
		dst.parent.mkdir(parents=True, exist_ok=True)
		shutil.move(str(src), str(dst))
		return dst

	def _delete_ade20k(self, img_id: int, img_name: str, full_path: Path):
		deleted_root = self.data.image_dir / "__deleted__"
		deleted_root.mkdir(parents=True, exist_ok=True)
		moved: list[str] = []

		# Glob everything in the same directory whose name starts with this image's stem.
		# This catches the image itself, companion JSONs, segmentation PNGs (_seg.png,
		# _parts_1.png, _parts_2.png, ...), and the per-image subdirectory if present.
		stem = full_path.stem
		parent_dir = full_path.parent
		candidates = sorted(parent_dir.iterdir())
		for entry in candidates:
			if entry.name.startswith(stem) and "__deleted__" not in entry.parts:
				dst = self._move_to_deleted(entry, deleted_root)
				moved.append(str(dst))
				logging.info("  moved %s -> %s", entry, dst)

		self._remove_image_from_list(img_id)
		self.status.showMessage(
			f"[ADE20K] Moved {len(moved)} item(s) to __deleted__: {img_name}", 5000
		)
		logging.info("Deleted (ADE20K) %s -> __deleted__ (%d items)", img_name, len(moved))
		self._advance_after_delete()

	def _delete_coco(self, img_id: int, img_name: str, full_path: Path):
		deleted_root = self.data.image_dir / "__deleted__"

		# Move the image file on disk.
		if full_path.exists():
			self._move_to_deleted(full_path, deleted_root)

		# Transfer annotations from live dict to deleted list (in memory).
		annos = self.data.annotations_by_image.pop(img_id, [])
		self.data.deleted_annos.extend(annos)
		self.data.deleted_image_ids.add(img_id)

		# Capture the image dict before we remove it so we can write the deleted file later.
		for img_dict in self.data.instances.get("images", []):
			if img_dict["id"] == img_id:
				self._deleted_images_cache.append(img_dict)
				break

		# Remove from the instances dict too so Key_W writes a clean file.
		self.data.instances["images"] = [
			img for img in self.data.instances.get("images", []) if img["id"] != img_id
		]
		self.data.instances["annotations"] = [
			ann for ann in self.data.instances.get("annotations", []) if ann["image_id"] != img_id
		]

		self._remove_image_from_list(img_id)
		self.status.showMessage(
			f"[COCO] Deleted {img_name} ({len(annos)} annos moved to memory). Press W to save.", 5000
		)
		logging.info("Deleted (COCO) image_id=%d %s — %d annotations pending flush", img_id, img_name, len(annos))
		self._advance_after_delete()

	def _remove_image_from_list(self, img_id: int):
		"""Remove the image with *img_id* from the ImageList."""
		il = self.data.images
		il.image_list = [(iid, name) for iid, name in il.image_list if iid != img_id]
		il.max = len(il.image_list)
		if il.max == 0:
			self.status.showMessage("No more images.", 5000)
			return
		# Clamp the index so it stays in bounds.
		il.n = min(il.n, il.max - 1)

	def _advance_after_delete(self):
		"""Move to the next image after deletion (or wrap to previous if at end)."""
		il = self.data.images
		if il.max == 0:
			self.data.current_image = (None, None)
			self.update_image(local=False)
			return
		# il.n is already clamped; just load whatever is there now.
		self.data.current_image = il.image_list[il.n]
		self._clear_selection()
		self.update_image(local=False)

	def save_annotations(self):
		"""Write annotation state to disk (COCO mode only, triggered by Key_W).

		Writes two files next to the original annotation file:
		  - ``<stem>.json``            – kept (surviving) annotations.
		  - ``<stem>__deleted__.json`` – annotations for deleted images.

		ADE20K mode has no single annotation file to rewrite, so this is a no-op
		(individual files are already moved on delete).
		"""
		if self.data.dataset_format == "ade20k":
			self.status.showMessage("ADE20K mode: files are moved on delete. Nothing extra to save.", 4000)
			return

		ann_path = self.data.annotations_file
		# Always write plain .json regardless of original compression.
		stem = ann_path.name.split(".")[0]
		kept_path = ann_path.parent / (stem + ".json")
		deleted_path = ann_path.parent / (stem + "__deleted__.json")

		kept_instances = dict(self.data.instances)
		deleted_instances = {
			"info": kept_instances.get("info", {}),
			"licenses": kept_instances.get("licenses", []),
			"images": self._deleted_images_cache,
			"annotations": self.data.deleted_annos,
			"categories": kept_instances.get("categories", []),
		}

		with open(kept_path, "w", encoding="utf-8") as f:
			json.dump(kept_instances, f)
		with open(deleted_path, "w", encoding="utf-8") as f:
			json.dump(deleted_instances, f)

		n_kept = len(kept_instances.get("annotations", []))
		n_del = len(self.data.deleted_annos)
		self.status.showMessage(
			f"Saved: {kept_path.name} ({n_kept} annos kept), "
			f"{deleted_path.name} ({n_del} annos deleted).",
			6000,
		)
		logging.info("Saved kept annotations → %s", kept_path)
		logging.info("Saved deleted annotations → %s", deleted_path)

	# ------------------------------------------------------------------
	# Rendering toggles
	# ------------------------------------------------------------------

	def toggle_bboxes(self):
		self.bboxes_on.setChecked(not self.bboxes_on.isChecked())
		self.update_image()

	def toggle_labels(self):
		self.labels_on.setChecked(not self.labels_on.isChecked())
		self.update_image()

	def toggle_masks(self):
		self.masks_on.setChecked(not self.masks_on.isChecked())
		self.update_image()

	def toggle_all(self):
		any_on = self.bboxes_on.isChecked() or self.labels_on.isChecked() or self.masks_on.isChecked()
		for action in (self.bboxes_on, self.labels_on, self.masks_on):
			action.setChecked(not any_on)
		self.update_image()


def export_video(data: Data, output_path: str | Path, options: RenderOptions, fps: float, codec: str, crf: int):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	original_index = data.images.n
	data.images.n = -1
	data.current_image = data.images.next()
	first_frame = compose_rgb_mask_bbox_headless(data, options)
	height, width = first_frame.shape[:2]
	print(f'Opening ffmpeg video with size: {width}x{height}')
	stdin, process = start_ffmpeg_streaming_v2(output_path, width, height, fps, codec=codec, crf=crf)
	try:
		write_frame_to_ffmpeg(stdin, first_frame)
		for idx in range(1, data.images.max):
			data.current_image = data.images.next()
			frame = compose_rgb_mask_bbox_headless(data, options)
			if frame.shape[:2] != (height, width):
				frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
			write_frame_to_ffmpeg(stdin, np.ascontiguousarray(frame))
			if idx % 100 == 0:
				logging.info("Wrote %d/%d frames...", idx + 1, data.images.max)
	finally:
		finalize_ffmpeg(stdin, process)
		data.images.n = original_index
		data.current_image = data.images.image_list[original_index]
	logging.info("Wrote video: %s", output_path)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="View images with bboxes/masks from COCO or ADE20K-style annotations")
	parser.add_argument("-i", "--images", default="", type=str, metavar="PATH", help="image root folder; in ADE20K mode this may be the dataset root or the image subtree")
	parser.add_argument("-a", "--annotations", default="", type=str, metavar="PATH", help="COCO annotation file, ADE JSON file, or ADE JSON root directory")
	parser.add_argument("--format", choices=("coco", "ade20k"), default="coco", help="annotation format to load")
	parser.add_argument("--output-video", default="", type=str, metavar="PATH", help="write RGB|mask|bbox video and exit")
	parser.add_argument("--video-fps", default=2.0, type=float, help="FPS for --output-video")
	parser.add_argument("--video-codec", default="libx265", type=str, help="FFmpeg codec for --output-video")
	parser.add_argument("--video-crf", default=18, type=int, help="CRF for --output-video")
	parser.add_argument("--object-colors", action="store_true", help="color instances rather than categories")
	parser.add_argument("--no-labels", action="store_true", help="disable labels for GUI startup and video export")
	parser.add_argument("--no-bboxes", action="store_true", help="disable boxes for GUI startup and video export")
	parser.add_argument("--no-masks", action="store_true", help="disable masks for GUI startup and video export")
	parser.add_argument("--bbox-width", default=3, type=int, help="bbox line width")
	parser.add_argument("--label-size", default=15, type=int, help="label text size")
	parser.add_argument("--mask-alpha", default=128, type=int, help="mask alpha, 0-255")
	parser.add_argument(
		"--default-show-classes",
		default="",
		type=str,
		metavar="CLASSES",
		help=(
			"Comma- or dash-separated list of category names to show by default "
			"(e.g. 'statue,monument' or 'statue-monument'). "
			"All other categories are hidden on startup; Shift+Click/Ctrl+Click still work normally."
		),
	)
	return parser


def options_from_args(args: argparse.Namespace) -> RenderOptions:
	return RenderOptions(
		bboxes_on=not args.no_bboxes,
		labels_on=not args.no_labels,
		masks_on=not args.no_masks,
		object_based_coloring=args.object_colors,
		bbox_width=args.bbox_width,
		mask_alpha=args.mask_alpha,
		label_size=args.label_size,
	)


def main() -> int:
	args = build_parser().parse_args()
	if not args.annotations:
		logging.error("Please specify at least --annotations (also --images if they're in a different root directory than --annotations). In ADE20K mode it may be a directory containing many per-image JSON files.")
		return 2
	if not args.images:
		annotations_path = Path(args.annotations)
		images = annotations_path if annotations_path.is_dir() else annotations_path.parent
	else:
		images = args.images

	data = Data(images, args.annotations, dataset_format=args.format)

	if data.images is None or data.annotations_by_image is None or data.categories is None:
		print(f'Failed to create a data object from annotations in {args.annotations} and images in {args.images if args.images else "<None>"} ')
		return 1

	options = options_from_args(args)

	if args.output_video:
		export_video(data, args.output_video, options, args.video_fps, args.video_codec, args.video_crf)
		return 0

	# Parse --default-show-classes: accept comma- or dash-separated names.
	default_show_classes: set[str] = set()
	if args.default_show_classes.strip():
		raw = args.default_show_classes.strip()
		# Support both "a,b,c" and "a-b-c" (but not mixed); prefer comma split first.
		if "," in raw:
			default_show_classes = {s.strip() for s in raw.split(",") if s.strip()}
		else:
			default_show_classes = {s.strip() for s in raw.split("-") if s.strip()}

	app = QtWidgets.QApplication(sys.argv)
	viewer = ImageViewer(data, default_show_classes=default_show_classes)
	viewer.bboxes_on.setChecked(options.bboxes_on)
	viewer.labels_on.setChecked(options.labels_on)
	viewer.masks_on.setChecked(options.masks_on)
	viewer.object_coloring.setChecked(options.object_based_coloring)
	viewer.bbox_slider.setValue(options.bbox_width)
	viewer.label_slider.setValue(options.label_size)
	viewer.mask_slider.setValue(options.mask_alpha)
	viewer.update_image(local=False)
	viewer.show()
	return app.exec()


if __name__ == "__main__":
	raise SystemExit(main())
