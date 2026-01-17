import math

import numpy as np
from typing import List, Optional, Tuple, Literal
from pathlib import Path
import cv2
import subprocess
import json
import torch

try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False


def _extract_frames_ffmpeg(video_path: str, frame_indices: list[int]) -> np.ndarray:
    """Extract specific frames using ffmpeg."""
    frames = []

    for idx in frame_indices:
        # Use ffmpeg to extract a specific frame
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"select=eq(n\\,{idx})",
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-",
        ]

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)

            # Check if output is empty (frame doesn't exist)
            if len(output) == 0:
                raise subprocess.CalledProcessError(1, cmd)

            # Get frame dimensions by probing first
            if len(frames) == 0:
                info_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    video_path,
                ]
                info_output = subprocess.check_output(info_cmd).decode("utf-8")
                info_data = json.loads(info_output)
                width = info_data["streams"][0]["width"]
                height = info_data["streams"][0]["height"]

            # Decode raw RGB data
            frame_data = np.frombuffer(output, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3))
            frames.append(frame)

        except subprocess.CalledProcessError:
            # Frame might not exist, create a black frame
            if len(frames) > 0:
                frames.append(np.zeros_like(frames[0]))
            else:
                # Default fallback frame
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

    return np.array(frames)


def get_frames_by_indices(
    video_path: Path,
    indices: List[int] | np.ndarray,
    video_backend: Optional[Literal['decord', 'torchvision_av', 'torchcodec']] = "ffmpeg",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        assert DECORD_AVAILABLE, "decord is not available."
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        assert TORCHCODEC_AVAILABLE, "torchcodec is not available."
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_at(indices=indices).data.numpy()
    elif video_backend == "ffmpeg":
        return _extract_frames_ffmpeg(video_path, list(indices))
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path,** (video_backend_kwargs if video_backend_kwargs is not None else {}))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    else:
        raise NotImplementedError

# Copy from gr00t
def get_accumulate_timestamp_idxs(
    timestamps: List[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: Optional[int] = 0,
    allow_negative=False,
) -> Tuple[List[int], List[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx
