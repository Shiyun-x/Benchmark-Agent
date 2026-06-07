import os
from pathlib import Path
from typing import Union, List, Sequence, Optional
import io
import random
import os
import io
from pathlib import Path
from typing import Union, List, Sequence, Optional, Dict, Any
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = str(_PROJECT_ROOT / "workspace" / "output_images")

ImagePath = Union[str, Path]
BBox = Sequence[int]



def _to_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _from_list(orig, lst):
    if isinstance(orig, (list, tuple)):
        return lst
    return lst[0]


def _ensure_out_dir(out_dir: Optional[ImagePath]) -> Path:
    if out_dir is None:
        out_dir = OUT_ROOT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _ensure_out_path(in_path: ImagePath, suffix: str, out_dir: Optional[ImagePath]) -> str:
    out_dir = _ensure_out_dir(out_dir)
    in_path = Path(in_path)
    stem = in_path.stem
    ext = in_path.suffix or ".png"
    out_name = f"{stem}{suffix}{ext}"
    return str(out_dir / out_name)


def _open_image(path: ImagePath) -> Image.Image:
    return Image.open(path).convert("RGB")


def _normalize_severity(severity: int, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(int(severity), hi))



# ===========================degradation functions===========================
def _apply_random_ops(
    image_paths: Union[ImagePath, List[ImagePath]],
    ops,
    suffix: str,
    severity: int,
    output_dir: Optional[ImagePath],
) -> Union[str, List[str]]:
    """
    Generic degradation executor:
    - ops: a list of functions, each op: (img: PIL.Image, severity: int) -> PIL.Image
    - randomly pick one op and apply it to each image
    """
    img_list = _to_list(image_paths)
    severity = _normalize_severity(severity)

    outputs = []
    for path in img_list:
        img = _open_image(path)
        op = random.choice(ops)
        out_img = op(img, severity)
        out_path = _ensure_out_path(path, suffix=suffix, out_dir=output_dir)
        out_img.save(out_path)
        outputs.append(out_path)

    return _from_list(image_paths, outputs)

def degrade_spatial(
    image_paths: Union[ImagePath, List[ImagePath]],
    severity: int = 1,
    output_dir: Optional[ImagePath] = None,
) -> Union[str, List[str]]:
    """
    Spatial degradation: Gaussian blur / downsample+upsample / pixelation.
    """
    def _gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
        radius = 1.0 * severity
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def _downsample_upsample(img: Image.Image, severity: int) -> Image.Image:
        w, h = img.size
        scale = 1.0 / (1.5 + 0.5 * severity)  # higher severity -> smaller scale
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        small = img.resize((new_w, new_h), Image.BICUBIC)
        back = small.resize((w, h), Image.BICUBIC)
        return back

    def _pixelate(img: Image.Image, severity: int) -> Image.Image:
        w, h = img.size
        block = max(4, int(min(w, h) / (32 / severity)))  # higher severity -> larger block size
        small_w = max(1, w // block)
        small_h = max(1, h // block)
        small = img.resize((small_w, small_h), Image.NEAREST)
        back = small.resize((w, h), Image.NEAREST)
        return back

    ops = [_gaussian_blur, _downsample_upsample, _pixelate]
    return _apply_random_ops(image_paths, ops, suffix="_deg_spatial", severity=severity, output_dir=output_dir)

def degrade_photometric(
    image_paths: Union[ImagePath, List[ImagePath]],
    severity: int = 1,
    output_dir: Optional[ImagePath] = None,
) -> Union[str, List[str]]:
    """
    Light / color degradation: darkening, overexposure, contrast shift, color jitter.
    """
    def _low_light(img: Image.Image, severity: int) -> Image.Image:
        factor = max(0.1, 1.0 - 0.25 * severity)
        return ImageEnhance.Brightness(img).enhance(factor)

    def _over_expose(img: Image.Image, severity: int) -> Image.Image:
        factor = 1.0 + 0.3 * severity
        return ImageEnhance.Brightness(img).enhance(factor)

    def _contrast_shift(img: Image.Image, severity: int) -> Image.Image:
        factor = 1.0 + 0.4 * (severity if random.random() < 0.5 else -severity)
        return ImageEnhance.Contrast(img).enhance(factor)

    def _color_jitter(img: Image.Image, severity: int) -> Image.Image:
        # mild saturation + hue shift
        sat_factor = 1.0 + 0.4 * (severity if random.random() < 0.5 else -severity)
        img2 = ImageEnhance.Color(img).enhance(sat_factor)
        # simple hue shift via HSV
        arr = np.array(img2.convert("HSV"))
        h = arr[:, :, 0].astype(np.int32)
        shift = int(15 * severity) * (1 if random.random() < 0.5 else -1)
        h = (h + shift) % 256
        arr[:, :, 0] = h.astype(np.uint8)
        return Image.fromarray(arr, mode="HSV").convert("RGB")

    ops = [_low_light, _over_expose, _contrast_shift, _color_jitter]
    return _apply_random_ops(image_paths, ops, suffix="_deg_photometric", severity=severity, output_dir=output_dir)

def degrade_compression(
    image_paths: Union[ImagePath, List[ImagePath]],
    severity: int = 1,
    output_dir: Optional[ImagePath] = None,
) -> Union[str, List[str]]:
    """
    Compression artifacts such as those introduced by JPEG compression.
    """
    def _jpeg_compress(img: Image.Image, severity: int) -> Image.Image:
        quality = max(5, 70 - severity * 12)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    ops = [_jpeg_compress]  # only one op for now; still routed through _apply_random_ops
    return _apply_random_ops(image_paths, ops, suffix="_deg_compress", severity=severity, output_dir=output_dir)

def degrade_occlusion(
    image_paths: Union[ImagePath, List[ImagePath]],
    severity: int = 1,
    output_dir: Optional[ImagePath] = None,
) -> Union[str, List[str]]:
    """
    Occlusion / missing info: random block occlusion.
    """
    def _random_blocks(img: Image.Image, severity: int) -> Image.Image:
        w, h = img.size
        arr = np.array(img)
        num_blocks = 1 + severity
        for _ in range(num_blocks):
            block_w = int(w * random.uniform(0.1, 0.2 * severity))
            block_h = int(h * random.uniform(0.1, 0.2 * severity))
            x0 = random.randint(0, max(0, w - block_w))
            y0 = random.randint(0, max(0, h - block_h))
            x1 = x0 + block_w
            y1 = y0 + block_h

            mode = random.choice(["black", "white", "gray"])
            if mode == "black":
                color = (0, 0, 0)
            elif mode == "white":
                color = (255, 255, 255)
            else:
                gray = random.randint(64, 192)
                color = (gray, gray, gray)

            arr[y0:y1, x0:x1, :] = np.array(color, dtype=np.uint8)

        return Image.fromarray(arr)

    ops = [_random_blocks]
    return _apply_random_ops(image_paths, ops, suffix="_deg_occlusion", severity=severity, output_dir=output_dir)

def degrade_noise(
    image_paths: Union[ImagePath, List[ImagePath]],
    severity: int = 1,
    output_dir: Optional[ImagePath] = None,
) -> Union[str, List[str]]:
    """
    Noise / Sensor degradation: Gaussian noise or salt-and-pepper noise.
    """
    def _gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        sigma = 8 * severity
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _salt_pepper(img: Image.Image, severity: int) -> Image.Image:
        arr = np.array(img)
        h, w, c = arr.shape
        amount = 0.01 * severity
        num = int(h * w * amount)

        coords = (np.random.randint(0, h, num), np.random.randint(0, w, num))
        arr[coords] = 255
        coords = (np.random.randint(0, h, num), np.random.randint(0, w, num))
        arr[coords] = 0

        return Image.fromarray(arr)

    ops = [_gaussian_noise, _salt_pepper]
    return _apply_random_ops(image_paths, ops, suffix="_deg_noise", severity=severity, output_dir=output_dir)

# unit-tested
def apply_image_degradation(
    image_paths: Union[ImagePath, List[ImagePath]],
    mode: str = "random",
    severity: int = 3,
    output_dir: Optional[ImagePath] = OUT_ROOT,
) -> Union[str, List[str]]:
    """
    Unified image degradation interface.

    Parameters
    ----------
    image_paths : str | Path | List[str | Path]
        Image path(s) (single path or list).
    mode : str
        Degradation mode:
            - "spatial"       spatial degradation (blur / downsample / pixelation)
            - "photometric"   lighting / color degradation
            - "compression"   compression artifacts
            - "occlusion"     occlusion / missing information
            - "noise"         sensor noise
            - "random"        pick one of the five modes above at random per image
    severity : int
        Degradation strength, roughly 1–5.
    output_dir : optional
        Output directory; defaults to global OUT_ROOT when omitted.

    Returns
    -------
    Output image path(s) matching the input shape (single path or list).
    """
    img_list = _to_list(image_paths)
    severity = max(1, min(int(severity), 5))

    # map to the five degradation function families
    dispatch = {
        "spatial": degrade_spatial,
        "photometric": degrade_photometric,
        "compression": degrade_compression,
        "occlusion": degrade_occlusion,
        "noise": degrade_noise,
    }

    # non-random: apply the same degradation to the whole batch
    if mode != "random":
        if mode not in dispatch:
            raise ValueError(f"Unsupported degradation mode: {mode}")
        fn = dispatch[mode]
        # these functions already support single/multi-image input
        return fn(image_paths=img_list, severity=severity, output_dir=output_dir)

    # mode == "random": pick a degradation mode independently per image
    modes = list(dispatch.keys())
    outputs: List[str] = []
    for p in img_list:
        m = random.choice(modes)
        fn = dispatch[m]
        out_path = fn(image_paths=p, severity=severity, output_dir=output_dir)
        # sub-functions return str for a single image; collect directly
        outputs.append(out_path)

    return _from_list(image_paths, outputs)



def image2text(
    image_paths: Union[ImagePath, List[ImagePath]],
    prompt: str,
    model: str = "gpt-5.1",
    max_tokens: int = 512,
) -> Union[str, List[str]]:
    """
    Generic image2text tool:
    - input: image(s) + prompt
    - output: text related to the image(s)
      (caption, description, dialogue opener, character thoughts, etc.; controlled by prompt)
    """
    img_list = _to_list(image_paths)

    from utils.llm_caller import llm_call_json

    outputs = []
    for path in img_list:
        text = llm_call_json(
            model=model,
            prompt=prompt,
            images=[path],
            max_tokens=max_tokens,
        )
        outputs.append(text)

    return _from_list(image_paths, outputs)


def mask_or_crop_to_bbox(
    image_paths: Union[ImagePath, List[ImagePath]],
    bboxes: Union[BBox, List[BBox]],
    mode: str = "crop",
    output_dir: Optional[ImagePath] = None,
) -> Union[str, List[str]]:
    """
    crop_or_mask_to_bbox
    - mode: "crop" : only output the bbox region (small image).
    - mode: "mask" : keep the bbox region, black out the rest.
    """
    img_list = _to_list(image_paths)
    bbox_list = _to_list(bboxes)

    if len(bbox_list) == 1 and len(img_list) > 1:
        bbox_list = bbox_list * len(img_list)

    if len(img_list) != len(bbox_list):
        raise ValueError("image_paths and bboxes length mismatch")

    mode = mode.lower()
    if mode not in {"crop", "mask"}:
        raise ValueError("mode must be 'crop' or 'mask'")

    outputs = []
    for path, bbox in zip(img_list, bbox_list):
        img = _open_image(path)
        w, h = img.size
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, min(int(x_min), w))
        x_max = max(0, min(int(x_max), w))
        y_min = max(0, min(int(y_min), h))
        y_max = max(0, min(int(y_max), h))

        if mode == "crop":
            patched = img.crop((x_min, y_min, x_max, y_max))
            suffix = "_crop_bbox"
        else:
            arr = np.array(img)
            mask_arr = np.zeros_like(arr)
            mask_arr[y_min:y_max, x_min:x_max, :] = arr[y_min:y_max, x_min:x_max, :]
            patched = Image.fromarray(mask_arr)
            suffix = "_mask_bbox"

        out_path = _ensure_out_path(path, suffix=suffix, out_dir=output_dir)
        patched.save(out_path)
        outputs.append(out_path)

    return _from_list(image_paths, outputs)

def ocr(
    image_paths: Union[ImagePath, List[ImagePath]],
    *args, **kwargs
) -> Union[str, List[str]]:
    """
    Extract text from image(s) via OCR.

    parameters
    ----------
        image_paths: path(s) to input image(s)
    returns
        extracted text string per image (single image → str, list → List[str])
    """
    pass


def bbox_to_point(
    image_paths: Union[ImagePath, List[ImagePath]],
    bboxes: Union[BBox, List[BBox]],
    radius: int = 5,
    label: Optional[str] = None,
    output_dir: Optional[ImagePath] = None,
) -> Union[str, List[str]]:
    """
    Draw a point at the center of each bbox, with an optional label.

    - image_paths: single path or list
    - bboxes: single bbox or a list aligned with image_paths
    """
    img_list = _to_list(image_paths)
    bbox_list = _to_list(bboxes)

    if len(bbox_list) == 1 and len(img_list) > 1:
        bbox_list = bbox_list * len(img_list)

    if len(img_list) != len(bbox_list):
        raise ValueError("image_paths and bboxes length mismatch")

    outputs = []
    for path, bbox in zip(img_list, bbox_list):
        img = _open_image(path)
        draw = ImageDraw.Draw(img)

        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        r = max(1, int(radius))
        left = cx - r
        top = cy - r
        right = cx + r
        bottom = cy + r

        # draw the point in red
        draw.ellipse((left, top, right, bottom), fill=(255, 0, 0))

        # optional label
        if label is not None:
            text_pos = (cx + r + 2, cy - r - 2)
            draw.text(text_pos, str(label), fill=(255, 0, 0))

        out_path = _ensure_out_path(path, suffix="_bbox_point", out_dir=output_dir)
        img.save(out_path)
        outputs.append(out_path)

    return _from_list(image_paths, outputs)


def image2objects(
    image_paths: Union[ImagePath, List[ImagePath]],
    *args, **kwargs,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Detect objects in image(s) and return bounding boxes with category labels.

    parameters
    ----------
        image_paths: path(s) to input image(s)
    returns
        per-image list of detections, each as:
            {"category": str, "category_id": int, "bbox": [x_min, y_min, x_max, y_max], "score": float}
        single image → List[dict], multiple images → List[List[dict]]
    """
    pass

