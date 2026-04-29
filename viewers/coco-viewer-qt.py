#!/usr/bin/env python3
'''
COCO Dataset Viewer using OpenCV + Qt.

Run with:
---------

./coco-viewer-qt.py -a /tmp/tiny/__tiny-sam3-video-dataset-v2-phase-2-deduplicated-coco.json -i /tmp/tiny

OR

./coco-viewer-qt.py -a /tmp/tiny/__tiny-sam3-video-dataset-v2-phase-2-deduplicated-coco.json -i /tmp/tiny/ --output-video /tmp/out.mp4 --video-fps 25

Features
--------
* OpenCV-based image loading, masks, bounding boxes, labels, and saving.
* PySide6 Qt GUI instead of Tkinter.
* Fast keyboard jumps for large datasets.
* Parent-directory selector and parent-group navigation.
* Auto-detected plain/compressed COCO annotations: .json, .json.gz, .json.bz2, .json.xz.
* Optional headless video export via --output-video using ffmpeg_utils.py.

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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
	from classes.ffmpeg_utils import finalize_ffmpeg, start_ffmpeg_streaming_v2, write_frame_to_ffmpeg
except ImportError as exc:
	# Get the absolute path of the directory 2 levels up (the repo root)
	root_dir = Path(__file__).resolve().parent.parent
	sys.path.append(str(root_dir))

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
	"""COCO data access and iteration."""

	def __init__(self, image_dir: str | Path, annotations_file: str | Path):
		self.image_dir = Path(image_dir)
		self.instances, images, self.categories = parse_coco(annotations_file)
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
	return instances, get_images(instances), get_categories(instances)


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
	def __init__(self, data: Data):
		super().__init__()
		self.data = data
		self.current_image_rgb: np.ndarray | None = None
		self.current_img_obj_categories: list[int] = []
		self.current_img_categories: list[int] = []
		self.selected_cats: set[int] | None = None
		self.selected_objs: set[int] | None = None
		self.parent_depth = 0
		self.source_pixmap: QtGui.QPixmap | None = None

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
		if self.selected_objs is None:
			ignore: tuple[int, ...] = ()
		else:
			ignore = tuple(i for i in range(len(self.current_img_obj_categories)) if i not in self.selected_objs)
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
		if self.selected_cats is None:
			self.category_list.selectAll()
		else:
			for i in self.selected_cats:
				item = self.category_list.item(i)
				if item:
					item.setSelected(True)
		self.category_list.blockSignals(False)

	def _update_object_list(self):
		self.object_list.blockSignals(True)
		self.object_list.clear()
		for i, category_id in enumerate(self.current_img_obj_categories):
			name = self.data.categories[category_id][0]
			self.object_list.addItem(f"{i} {name}")
		if self.selected_objs is None:
			self.object_list.selectAll()
		else:
			for i in self.selected_objs:
				item = self.object_list.item(i)
				if item:
					item.setSelected(True)
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
	parser = argparse.ArgumentParser(description="View images with bboxes from a COCO dataset")
	parser.add_argument("-i", "--images", default="", type=str, metavar="PATH", help="path to images folder (`basename of annotations` if not specified)")
	parser.add_argument("-a", "--annotations", default="", type=str, metavar="PATH", help="path to annotations file (.json, .json.gz, .json.bz2, .json.xz)")
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
		logging.error("Please specify at least --annotations (also --images if they're in a different root directory than --annotations).")
		return 2
	if not args.images:
		images = Path(args.annotations).parent
	else:
		images = args.images

	data = Data(images, args.annotations)
	options = options_from_args(args)

	if args.output_video:
		export_video(data, args.output_video, options, args.video_fps, args.video_codec, args.video_crf)
		return 0

	app = QtWidgets.QApplication(sys.argv)
	viewer = ImageViewer(data)
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
