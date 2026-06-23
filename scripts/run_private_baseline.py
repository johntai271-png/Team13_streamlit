#!/usr/bin/env python3
"""Batch OCR on available private_test images → submission_private.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smce_baseline_core import (  # noqa: E402
    build_product_predictor,
    find_private_root,
    load_private_catalog,
    load_train_labels,
    predict_private,
    private_images_dir,
    run_ocr_on_image,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all on disk)")
    parser.add_argument("-o", "--output", default="submission_private.csv")
    args = parser.parse_args()

    root = find_private_root()
    if root is None:
        raise SystemExit("private_test not found under data/private_test/")

    import easyocr

    reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    catalog = load_private_catalog(root)
    img_dir = private_images_dir(root)
    available = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if args.limit:
        available = available[: args.limit]

    labels = load_train_labels()
    product_fn = None
    if labels is not None:
        product_fn = build_product_predictor(labels).predict

    rows = []
    for iid in available:
        path = img_dir / f"{iid}.jpg"
        from PIL import Image

        ocr_text = run_ocr_on_image(Image.open(path), reader)
        brand, product = predict_private(ocr_text, product_fn)
        rows.append(
            {
                "image_id": iid,
                "ocr_text": ocr_text or " ",
                "brand_name": brand or " ",
                "product_name": product or " ",
            }
        )
        print(f"  {iid}: ocr={len(ocr_text)} brand={brand!r}")

    # pad missing IDs from catalog with empty placeholders
    done = {r["image_id"] for r in rows}
    for iid in catalog["image_id"]:
        if iid not in done:
            rows.append(
                {
                    "image_id": iid,
                    "ocr_text": " ",
                    "brand_name": " ",
                    "product_name": " ",
                }
            )

    out = pd.DataFrame(rows).sort_values("image_id")
    out.to_csv(args.output, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print(f"Wrote {args.output} ({len(out)} rows, {len(available)} OCR'd)")


if __name__ == "__main__":
    main()
