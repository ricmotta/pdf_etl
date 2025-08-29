"""
OpenCV-based preprocessing for receipt OCR.

Pipeline (in order):
1) Convert to grayscale
2) Background normalization (illumination correction)
3) Adaptive threshold (binary image)
4) Deskew (estimate text angle and rotate)
5) Denoise (median)
6) Sharpen (unsharp mask)
7) Upscale to a minimum width (improves small fonts)
8) Optional auto-crop the biggest text area (largest contour)

Returns a PIL.Image ready for OCR. Also supports debug artifact saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessConfig:
    min_width: int = 1800              # upscale receipts to at least this width
    do_autocrop: bool = True           # try to crop away table/background
    deskew_max_angle: float = 8.0      # ignore angles beyond this (deg)
    debug_dir: Optional[Path] = None   # if set, write intermediate artifacts here


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _illumination_correction(gray: np.ndarray) -> np.ndarray:
    """Normalize uneven lighting using large Gaussian blur as background model."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    # Prevent divide-by-zero by maxing background
    bg = np.clip(bg, 1, 255)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def _adaptive_binarize(img: np.ndarray) -> np.ndarray:
    """Adaptive threshold; Gaussian method works well on receipts."""
    # OpenCV expects 8-bit single channel
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    bin_img = cv2.adaptiveThreshold(
        img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10
    )
    return bin_img


def _estimate_skew_angle(bin_img: np.ndarray) -> float:
    """
    Estimate skew by Hough lines on the binary image.
    Returns angle in degrees. Positive = rotate counter-clockwise.
    """
    edges = cv2.Canny(bin_img, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=120)
    if lines is None:
        return 0.0

    angles = []
    for rho_theta in lines[:200]:  # limit for speed
        rho, theta = rho_theta[0]
        # Convert to degrees and shift around 0 relative to horizontal
        deg = (theta * 180.0 / np.pi) - 90.0
        # Keep near-horizontal lines only
        if -45 <= deg <= 45:
            angles.append(deg)

    if not angles:
        return 0.0

    # Use median to be robust against outliers
    return float(np.median(angles))


def _rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image keeping whole content (like imutils.rotate_bound)."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderValue=255)


def _deskew(bin_img: np.ndarray, gray_for_rotate: np.ndarray, max_angle: float) -> np.ndarray:
    angle = _estimate_skew_angle(bin_img)
    if abs(angle) < 0.1 or abs(angle) > max_angle:
        return gray_for_rotate
    return _rotate_bound(gray_for_rotate, -angle)  # rotate opposite to correct skew


def _unsharp(img: np.ndarray, radius: float = 1.5, amount: float = 1.5) -> np.ndarray:
    """Unsharp mask using Gaussian blur."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
    # result = (1 + amount) * img - amount * blurred
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype("uint8")


def _auto_crop(bin_img: np.ndarray, pad: int = 8) -> Tuple[int, int, int, int]:
    """
    Find largest contour and return crop box (x,y,w,h).
    If nothing reasonable is found, return full image box.
    """
    h, w = bin_img.shape[:2]
    contours, _ = cv2.findContours(255 - bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h

    # pick largest area contour
    cnt = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(cnt)

    # sanity: avoid cropping to a tiny region
    if cw * ch < 0.2 * w * h:
        return 0, 0, w, h

    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + cw + pad, w); y1 = min(y + ch + pad, h)
    return x0, y0, x1 - x0, y1 - y0


def preprocess_receipt(
    img_input: Image.Image | np.ndarray | str,
    cfg: PreprocessConfig | None = None
) -> Image.Image:
    """
    High-level preprocess entry point.
    - Accepts PIL.Image, numpy array (BGR/GRAY), or file path.
    - Returns a PIL.Image (binary/sharpened/upscaled) ideal for OCR.
    """
    cfg = cfg or PreprocessConfig()

    # --- Load & grayscale ---
    if isinstance(img_input, str):
        cv_img = cv2.imread(img_input, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise FileNotFoundError(f"Could not read image: {img_input}")
    elif isinstance(img_input, Image.Image):
        cv_img = cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
    else:
        cv_img = img_input.copy()

    gray = _ensure_gray(cv_img)

    # Optional debug dir
    if cfg.debug_dir:
        Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(cfg.debug_dir, "01_gray.png")), gray)

    # --- Illumination correction ---
    norm = _illumination_correction(gray)
    if cfg.debug_dir:
        cv2.imwrite(str(Path(cfg.debug_dir, "02_norm.png")), norm)

    # --- Adaptive threshold (binary) ---
    bin_img = _adaptive_binarize(norm)
    if cfg.debug_dir:
        cv2.imwrite(str(Path(cfg.debug_dir, "03_bin.png")), bin_img)

    # --- Deskew (estimate angle on binary, rotate normalized gray) ---
    rotated_gray = _deskew(bin_img, norm, cfg.deskew_max_angle)
    if cfg.debug_dir:
        cv2.imwrite(str(Path(cfg.debug_dir, "04_rotated.png")), rotated_gray)

    # --- Re-binarize after rotation (keeps edges crisp) ---
    bin_after = _adaptive_binarize(rotated_gray)
    if cfg.debug_dir:
        cv2.imwrite(str(Path(cfg.debug_dir, "05_bin_after.png")), bin_after)

    # --- Optional auto-crop largest text area ---
    if cfg.do_autocrop:
        x, y, w, h = _auto_crop(bin_after)
        bin_after = bin_after[y:y+h, x:x+w]
        if cfg.debug_dir:
            cv2.imwrite(str(Path(cfg.debug_dir, "06_cropped.png")), bin_after)

    # --- Denoise & Sharpen ---
    den = cv2.medianBlur(bin_after, 3)
    sharp = _unsharp(den, radius=1.5, amount=1.2)
    if cfg.debug_dir:
        cv2.imwrite(str(Path(cfg.debug_dir, "07_sharp.png")), sharp)

    # --- Upscale to min width ---
    h, w = sharp.shape[:2]
    if w < cfg.min_width:
        scale = cfg.min_width / float(w)
        sharp = cv2.resize(sharp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        if cfg.debug_dir:
            cv2.imwrite(str(Path(cfg.debug_dir, "08_upscaled.png")), sharp)

    # Convert back to PIL (mode 'L')
    pil = Image.fromarray(sharp).convert("L")
    return pil
