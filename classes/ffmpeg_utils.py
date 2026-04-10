import json

import torch
import subprocess
import numpy as np

from pathlib import Path
from typing import Tuple

# ─── ffprobe ──────────────────────────────────────────────────────────────────────

def ffprobe_video(path: Path) -> dict:
    """Return {'width', 'height', 'fps', 'nb_frames'} for the first video stream."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', '-show_format',
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    for stream in info.get('streams', []):
        if stream.get('codec_type') != 'video':
            continue
        w = int(stream['width'])
        h = int(stream['height'])
        fps_str = stream.get('r_frame_rate', '25/1')
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
        # nb_frames is the most reliable count; fall back to duration * fps
        nb_frames = int(stream.get('nb_frames', 0))
        if nb_frames == 0:
            duration = float(stream.get('duration', 0) or
                             info.get('format', {}).get('duration', 0))
            nb_frames = max(1, round(duration * fps))
        return {'width': w, 'height': h, 'fps': fps, 'nb_frames': nb_frames}
    raise ValueError(f'No video stream found in {path}')


# ─── ffmpeg ───────────────────────────────────────────────────────────────────────────────────────────────────────

def start_ffmpeg_streaming_v2(output_path: Path, width: int, height: int, fps: float,
			codec: str = 'libx265', pix_fmt: str = 'yuv420p', crf: int = 18, debug: bool = False) -> Tuple:
    """
    Launch an FFmpeg process that reads rawvideo rgb24 frames from stdin.

    Handles ffv1 (lossless, no crf/preset) and standard codecs (libx265 etc.)
    correctly. Returns (stdin_pipe, process).
    """
    input_args = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',         # we always feed RGB24 from numpy
        '-r', str(fps),
        '-i', '-',
        '-an',                        # no audio
    ]
    if codec == 'ffv1':
        # ffv1 is lossless and does not accept -crf or -preset.
        # rgb24 output pix_fmt keeps the mask colours bit-exact.
        enc_args = [
            '-c:v', 'ffv1',
            '-level', '3',
            '-coder', '1',
            '-context', '1',
            '-g', '1',
            '-slices', '24',
            '-pix_fmt', 'rgb24',
        ]
    else:
        enc_args = [
            '-c:v', codec,
            '-pix_fmt', pix_fmt,
            '-crf', str(crf),
            '-preset', 'fast',
        ]
    cmd = input_args + enc_args + [str(output_path)]
    sep = '─' * 10
    print(f'{sep} Starting FFmpeg → {output_path.name}  '
          f'({codec}, {width}×{height} @ {fps:.3f} fps)')
    if debug:
        print(f'    cmd: {" ".join(str(c) for c in cmd)}')
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdin, proc

def start_ffmpeg_streaming(output_path, width, height, fps, codec='libx265', pix_fmt='yuv420p', crf=23, debug=True):
	"""
	Start an FFmpeg process in streaming mode for writing video frames.
	Returns the stdin pipe and the process object.
	"""
	if codec == 'ffv1':
		crf = 0
		command = [
			'ffmpeg',
			'-y',  # Overwrite output file if it exists
			'-f', 'rawvideo',
			'-vcodec', 'rawvideo',
			'-s', f'{width}x{height}',  # size of one frame
			#'-pix_fmt', str(pix_fmt),
			'-pix_fmt', 'rgb24',
			'-r', str(fps),  # frames per second
			'-i', '-',  # The input comes from a pipe
			'-an',  # No audio
			'-c:v', codec,
			'-context', '0', '-level', '3',
			'-pix_fmt', pix_fmt,
			'-crf', str(crf),
			'-preset', 'fast',
			output_path
		]
	else:
		command = [
			'ffmpeg',
			'-y',  # Overwrite output file if it exists
			'-f', 'rawvideo',
			'-vcodec', 'rawvideo',
			'-s', f'{width}x{height}',  # size of one frame
			'-pix_fmt', 'rgb24',
			'-r', str(fps),  # frames per second
			'-i', '-',  # The input comes from a pipe
			'-an',  # No audio
			'-c:v', codec,
			'-pix_fmt', pix_fmt,
			'-crf', str(crf),
			'-preset', 'fast',
			output_path
		]

	print(f'{10 * "-"} Starting FFmpeg process with output: {output_path} (codec: {codec}, resolution: {width}x{height} @ {fps}fps - quality: {crf})')
	if debug:
		print(f'FFmpeg command: {command}')
	process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
	return process.stdin, process

def write_frame_to_ffmpeg(stdin, frame_tensor):
	"""
	Write a single frame (HWC uint8 tensor) to FFmpeg stdin.
	frame_tensor should be on CPU, uint8, shape (H, W, 3)
	"""
	# Ensure contiguous memory layout and convert to numpy
	if isinstance(frame_tensor, torch.Tensor):
		#frame_np = frame_tensor.cpu().contiguous().numpy()
		frame_np = frame_tensor.cpu().numpy()
	else:
		frame_np = frame_tensor
	
	# Write raw bytes to FFmpeg
	stdin.write(frame_np.tobytes())
	stdin.flush()

def finalize_ffmpeg(stdin, process):
	"""
	Close stdin and wait for FFmpeg to finish.
	"""
	stdin.close()
	process.wait()
	if process.returncode != 0:
		stderr = process.stderr.read().decode()
		raise RuntimeError(f"FFmpeg failed with code {process.returncode}: {stderr}")

# ─── FFmpegCapture ─ drop-in replacement for cv2.VideoCapture ─────────────────────────────────────────────────────

class FFmpegCapture:
    """
    Drop-in replacement for cv2.VideoCapture that uses FFmpeg for decoding.
    Supports any codec FFmpeg understands (AV1, HEVC, VP9, …).

    Usage:
        cap = FFmpegCapture("video.mkv")          # instead of cv2.VideoCapture(...)
        cap = FFmpegCapture("video.mkv", hw="cuda") # optional HW accel
    """

    # ------------------------------------------------------------------ #
    #  construction / destruction
    # ------------------------------------------------------------------ #

    def __init__(self, path: str, hw: str | None = None):
        """
        Parameters
        ----------
        path : str
            Path to the video file.
        hw : str | None
            Optional FFmpeg hwaccel name ('cuda', 'videotoolbox', …).
            Leave None for pure-software decoding (always works).
        """
        self.path = path
        self.hw   = hw

        self._proc:      subprocess.Popen | None = None
        self._pos_frame: int   = 0          # next frame index we will return

        # populated by _probe()
        self._fps:         float = 0.0
        self._frame_count: int   = 0
        self._width:       int   = 0
        self._height:      int   = 0
        self._duration:    float = 0.0      # seconds

        self._probe()

    # ------------------------------------------------------------------ #
    #  public OpenCV-compatible API
    # ------------------------------------------------------------------ #

    def isOpened(self) -> bool:
        return self._width > 0 and self._height > 0

    def get(self, prop: int) -> float:
        import cv2  # still needed for constants (CAP_PROP_*) and color cvt elsewhere
        mapping = {
            cv2.CAP_PROP_FPS:          self._fps,
            cv2.CAP_PROP_FRAME_COUNT:  float(self._frame_count),
            cv2.CAP_PROP_FRAME_WIDTH:  float(self._width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self._height),
            cv2.CAP_PROP_POS_FRAMES:   float(self._pos_frame),
            cv2.CAP_PROP_POS_MSEC:     self._pos_frame / self._fps * 1000
                                       if self._fps else 0.0,
        }
        return mapping.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        import cv2  # still needed for constants (CAP_PROP_*) and color cvt elsewhere
        if prop == cv2.CAP_PROP_POS_FRAMES:
            target = int(value)
            if target != self._pos_frame:
                self._restart_at(target)
            return True
        if prop == cv2.CAP_PROP_POS_MSEC:
            if self._fps:
                self._restart_at(int(value / 1000 * self._fps))
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return (True, BGR-frame) or (False, None) at EOF / error."""
        if self._proc is None:
            self._restart_at(self._pos_frame)

        frame_bytes = self._width * self._height * 3
        raw = self._read_exactly(frame_bytes)
        if raw is None:
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )
        self._pos_frame += 1
        return True, frame

    def release(self):
        self._kill_proc()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        self.release()

    # ------------------------------------------------------------------ #
    #  internal helpers
    # ------------------------------------------------------------------ #

    def _probe(self):
        """Use ffprobe to fill resolution / fps / duration metadata."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            self.path,
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "ffprobe not found or failed — make sure FFmpeg is installed."
            ) from exc

        info        = json.loads(out)
        vid_stream  = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if vid_stream is None:
            raise RuntimeError(f"No video stream found in {self.path!r}")

        # resolution
        self._width  = int(vid_stream["width"])
        self._height = int(vid_stream["height"])

        # fps  (r_frame_rate is the most reliable field)
        num, den     = map(int, vid_stream["r_frame_rate"].split("/"))
        self._fps    = num / den if den else 0.0

        # duration / frame count
        self._duration = float(
            vid_stream.get("duration")
            or info.get("format", {}).get("duration", 0)
        )
        # prefer the explicit frame count; fall back to fps × duration
        if "nb_frames" in vid_stream and int(vid_stream["nb_frames"]) > 0:
            self._frame_count = int(vid_stream["nb_frames"])
        elif self._fps and self._duration:
            self._frame_count = int(round(self._fps * self._duration))

    def _build_ffmpeg_cmd(self, start_frame: int) -> list[str]:
        """
        Build an FFmpeg command that outputs raw BGR24 frames starting at
        *start_frame*.  Fast input-side seek is used (accurate to the nearest
        keyframe); FFmpeg then decodes forward from there.
        """
        cmd = ["ffmpeg"]

        # optional hardware acceleration
        if self.hw:
            cmd += ["-hwaccel", self.hw]

        # fast seek: put -ss *before* -i so FFmpeg seeks at the demuxer level.
        # This lands on the nearest keyframe ≤ target timestamp and is orders
        # of magnitude faster than decoding from the beginning.
        if start_frame > 0 and self._fps:
            timestamp = start_frame / self._fps
            cmd += ["-ss", f"{timestamp:.6f}"]

        cmd += [
            "-i",    self.path,
            "-f",    "rawvideo",
            "-pix_fmt", "bgr24",   # keep BGR so downstream cv2 code stays unchanged
            "-an",                 # drop audio — we only want video frames
            "pipe:1",
        ]
        return cmd

    def _restart_at(self, frame_num: int):
        """Kill any running FFmpeg process and launch a fresh one at *frame_num*."""
        self._kill_proc()
        self._pos_frame = frame_num

        cmd = self._build_ffmpeg_cmd(frame_num)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._width * self._height * 3 * 8,  # ~8-frame buffer
        )

    def _kill_proc(self):
        if self._proc is not None:
            try:
                self._proc.stdout.close()
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def _read_exactly(self, n: int) -> bytes | None:
        """Read exactly *n* bytes from the FFmpeg pipe, or None on EOF/error."""
        buf = bytearray()
        remaining = n
        while remaining > 0:
            chunk = self._proc.stdout.read(remaining)
            if not chunk:
                return None          # EOF / process exited
            buf += chunk
            remaining -= len(chunk)
        return bytes(buf)
