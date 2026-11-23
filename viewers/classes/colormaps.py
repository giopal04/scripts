#!/usr/bin/env python3

import cv2

def apply_colormap(frame, cmap=0):
	if cmap == 0 or cmap > 21:
		return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
	if cmap == 1:
		return cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
	if cmap == 2:
		return cv2.applyColorMap(frame, cv2.COLORMAP_AUTUMN)
	if cmap == 3:
		return cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)
	if cmap == 4:
		return cv2.applyColorMap(frame, cv2.COLORMAP_RAINBOW)
	if cmap == 5:
		return cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
	if cmap == 6:
		return cv2.applyColorMap(frame, cv2.COLORMAP_SUMMER)
	if cmap == 7:
		return cv2.applyColorMap(frame, cv2.COLORMAP_SPRING)
	if cmap == 8:
		return cv2.applyColorMap(frame, cv2.COLORMAP_COOL)
	if cmap == 9:
		return cv2.applyColorMap(frame, cv2.COLORMAP_HSV)
	if cmap == 10:
		return cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
	if cmap == 11:
		return cv2.applyColorMap(frame, cv2.COLORMAP_PINK)
	if cmap == 12:
		return cv2.applyColorMap(frame, cv2.COLORMAP_PARULA)
	if cmap == 13:
		return cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)
	if cmap == 14:
		return cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
	if cmap == 15:
		return cv2.applyColorMap(frame, cv2.COLORMAP_PLASMA)
	if cmap == 16:
		return cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
	if cmap == 17:
		return cv2.applyColorMap(frame, cv2.COLORMAP_CIVIDIS)
	if cmap == 18:
		return cv2.applyColorMap(frame, cv2.COLORMAP_TWILIGHT)
	if cmap == 19:
		return cv2.applyColorMap(frame, cv2.COLORMAP_TWILIGHT_SHIFTED)
	if cmap == 20:
		return cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)
	if cmap == 21:
		return cv2.applyColorMap(frame, cv2.COLORMAP_DEEPGREEN)

