from __future__ import annotations

import os

import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image


def get_sorted_image_filenames(
    image_dir: str,
    image_extension: str = 'jpg',
    extra_start: int = 0,
    extra_end: int = 0,
) -> list[str]:
    fns = sorted(
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if filename.endswith(image_extension)
    )
    fns = [fns[0]] * extra_start + fns + [fns[-1]] * extra_end
    return fns


def save_gif(image_dir, output_filename: str, extra_start: int = 0, extra_end: int = 0):
    image_filenames = get_sorted_image_filenames(
        image_dir,
        extra_start=extra_start,
        extra_end=extra_end,
    )
    image_filenames = image_filenames + []
    images = [Image.open(x) for x in image_filenames]
    images[0].save(output_filename, save_all=True, append_images=images[1:])


def save_mp4(image_dir, output_filename: str, fps: int, extra_start: int = 0, extra_end: int = 0):
    assert 0 < fps <= 60
    image_filenames = get_sorted_image_filenames(
        image_dir,
        extra_start=extra_start,
        extra_end=extra_end,
    )
    clip = ImageSequenceClip(image_filenames, fps=fps)
    clip.write_videofile(output_filename)


def slerp_np(t, v0, v1, dot_threshold=0.9995):
    """Spherically interpolate two arrays v0 and v1"""
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > dot_threshold:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    return v2


def slerp(t, v0, v1, dot_threshold=0.9995):
    input_device = v0.device
    v0 = v0.cpu().numpy()
    v1 = v1.cpu().numpy()
    v2 = slerp_np(t, v0, v1, dot_threshold)
    return torch.from_numpy(v2).to(input_device)


def preprocess_image(image: Image, resize: bool = False):
    w, h = image.size
    if w % 32 != 0 or h % 32 != 0:
        if not resize:
            raise ValueError(
                'image dimensions must be divisible by 32 - resize=True to automatically fix this'
            )
        else:
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
            image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0
