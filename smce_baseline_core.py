"""OCR + prediction helpers for SMCE baseline (notebook + Streamlit)."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from smce_brand_rules import extract_brand_product, extract_product

REPO_ROOT = Path(__file__).resolve().parent

PRIVATE_ROOT_CANDIDATES = [
    REPO_ROOT / "data" / "private_test",
    REPO_ROOT / "private_test",
    Path(os.environ.get("SMCE_PRIVATE_TEST_DIR", "")),
]


def preprocess(img: Image.Image, max_dim: int = 1280) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    return img.filter(ImageFilter.SHARPEN)


def postprocess_ocr(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if not tokens:
        return ""
    deduped = [tokens[0]]
    for tok in tokens[1:]:
        if tok.lower() != deduped[-1].lower():
            deduped.append(tok)
    return " ".join(deduped)


def run_ocr_on_image(img: Image.Image, reader, min_conf: float = 0.35) -> str:
    img = preprocess(img.convert("RGB"))
    try:
        results = reader.readtext(np.array(img), detail=1, paragraph=False)
        results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
        lines = [r[1] for r in results if r[2] > min_conf]
        return postprocess_ocr(" ".join(lines))
    except Exception:
        return ""


def predict_public(ocr_text: str, product_fn: Callable[[str], str] | None = None) -> str:
    fn = product_fn or extract_product
    return fn(ocr_text or "")


def predict_private(
    ocr_text: str,
    product_fn: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    brand, product = extract_brand_product(ocr_text or "")
    if product_fn and not brand and not product:
        product = product_fn(ocr_text or "")
    return brand, product


def _has_private_images(root: Path) -> bool:
    for sub in ("images", "images_sample"):
        d = root / sub
        if d.is_dir() and any(d.glob("priv_*.jpg")):
            return True
    return False


def find_private_root() -> Path | None:
    for root in PRIVATE_ROOT_CANDIDATES:
        if not root or not Path(root).exists():
            continue
        root = Path(root).resolve()
        test_csv = root / "private_test.csv"
        if test_csv.is_file() and _has_private_images(root):
            return root
        if _has_private_images(root):
            return root
    return None


def private_images_dir(root: Path) -> Path:
    """Prefer full ``images/``; fall back to bundled ``images_sample/``."""
    full = root / "images"
    if full.is_dir() and any(full.glob("priv_*.jpg")):
        return full
    sample = root / "images_sample"
    if sample.is_dir() and any(sample.glob("priv_*.jpg")):
        return sample
    return full


def load_private_catalog(root: Path) -> pd.DataFrame:
    test_csv = root / "private_test.csv"
    if test_csv.is_file():
        return pd.read_csv(test_csv, keep_default_na=False)
    img_dir = private_images_dir(root)
    ids = sorted(p.stem for p in img_dir.glob("*.jpg"))
    return pd.DataFrame({"image_id": ids})


def load_private_solution(root: Path) -> pd.DataFrame | None:
    for name in ("solution_private.csv", "solution_private_eval.csv"):
        path = root / name
        if path.is_file():
            return pd.read_csv(path, keep_default_na=False)
    return None


def load_train_labels() -> pd.DataFrame | None:
    for path in [
        REPO_ROOT / "data" / "train_labels.csv",
        REPO_ROOT / "train_labels.csv",
    ]:
        if path.is_file():
            return pd.read_csv(path, keep_default_na=False)
    return None


def build_product_predictor(train_labels: pd.DataFrame):
    from smce_product_model import ProductPredictor

    model = ProductPredictor(min_class_count=3, prob_threshold=0.60, max_features=3000)
    model.fit(train_labels, extract_product)
    return model


def setup_full_private_images(source_dir: Path, dest_dir: Path | None = None) -> int:
    """Copy ``source_dir/images/*.jpg`` into ``data/private_test/images/``."""
    dest = dest_dir or (REPO_ROOT / "data" / "private_test" / "images")
    src = source_dir / "images"
    if not src.is_dir():
        raise FileNotFoundError(f"Missing {src}")
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for jpg in src.glob("*.jpg"):
        shutil.copy2(jpg, dest / jpg.name)
        n += 1
    return n
