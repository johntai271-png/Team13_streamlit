#!/usr/bin/env python3
"""Streamlit demo — SMCE baseline on private_test format."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from smce_baseline_core import (
    build_product_predictor,
    find_private_root,
    load_private_catalog,
    load_private_solution,
    load_train_labels,
    predict_private,
    predict_public,
    private_images_dir,
    run_ocr_on_image,
)

st.set_page_config(page_title="SMCE Baseline Demo", page_icon="🛒", layout="wide")

st.title("SMCE Baseline — Private Test Demo")
st.caption(
    "EasyOCR (CPU) + brand rules + optional sklearn product head. "
    "Dataset: `data/private_test/` (sample images included; run `scripts/setup_private_images.py` for full set)."
)


@st.cache_resource(show_spinner="Loading EasyOCR (vi + en, CPU)...")
def get_ocr_reader():
    import easyocr

    return easyocr.Reader(["vi", "en"], gpu=False, verbose=False)


@st.cache_resource(show_spinner="Training lightweight product head...")
def get_product_predictor():
    labels = load_train_labels()
    if labels is None:
        return None
    return build_product_predictor(labels)


def _score_private(gt: pd.DataFrame, pred: pd.DataFrame) -> float | None:
    metric_py = Path(__file__).resolve().parent / "private_test" / "metric.py"
    if not metric_py.is_file():
        return None
    sys.path.insert(0, str(metric_py.parent))
    try:
        from metric import score  # noqa: WPS433

        cols = ["image_id", "ocr_text", "brand_name", "product_name"]
        return float(score(gt[cols], pred[cols], "image_id"))
    except Exception as exc:
        st.warning(f"Private score unavailable: {exc}")
        return None


with st.sidebar:
    st.header("Settings")
    fmt = st.radio("Format", ["Private (4 columns)", "Public (3 columns)"], index=0)
    private_mode = fmt.startswith("Private")
    use_train_head = st.checkbox("Use train product head", value=True)
    min_conf = st.slider("OCR min confidence", 0.1, 0.9, 0.35, 0.05)
    priv_root = find_private_root()
    if priv_root:
        img_dir = private_images_dir(priv_root)
        n_local = len(list(img_dir.glob("*.jpg")))
        st.caption(f"Images: `{img_dir.name}/` ({n_local:,} jpg)")

predictor = get_product_predictor() if use_train_head else None
product_fn = predictor.predict if predictor else None

tab_upload, tab_private, tab_text = st.tabs(["Upload image", "Private catalog", "Text only"])

with tab_upload:
    uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png"])
    image_id = st.text_input("image_id", value="demo_0001")
    if uploaded:
        img = Image.open(uploaded)
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, use_container_width=True)
        if st.button("Run OCR", type="primary"):
            ocr_text = run_ocr_on_image(img, get_ocr_reader(), min_conf)
            st.session_state["ocr_text"] = ocr_text
            if private_mode:
                b, p = predict_private(ocr_text, product_fn)
                st.session_state.update(brand_name=b, product_name=p)
            else:
                st.session_state["product_name"] = predict_public(ocr_text, product_fn)
        with c2:
            ocr_text = st.text_area("ocr_text", st.session_state.get("ocr_text", ""), height=140)
            if private_mode:
                st.text_input("brand_name", st.session_state.get("brand_name", ""))
                st.text_input("product_name", st.session_state.get("product_name", ""))
            else:
                st.text_input("product_name", st.session_state.get("product_name", ""))

with tab_private:
    if priv_root is None:
        st.warning(
            "No `data/private_test/` found. Clone repo and ensure `private_test.csv` + `images_sample/` exist."
        )
    else:
        catalog = load_private_catalog(priv_root)
        solution = load_private_solution(priv_root)
        images_dir = private_images_dir(priv_root)
        available = {p.stem for p in images_dir.glob("*.jpg")}
        st.caption(f"{len(catalog):,} IDs in CSV · {len(available):,} jpg on disk")

        defaults = [i for i in ("priv_h_0006", "priv_d_0002", "priv_d_0003") if i in available]
        selected = st.multiselect(
            "image_id",
            [i for i in catalog["image_id"] if i in available],
            default=defaults or list(available)[:3],
            max_selections=20,
        )

        if st.button("Run OCR", type="primary") and selected:
            reader = get_ocr_reader()
            rows = []
            for iid in selected:
                ocr_text = run_ocr_on_image(Image.open(images_dir / f"{iid}.jpg"), reader, min_conf)
                brand, product = predict_private(ocr_text, product_fn)
                rows.append(
                    {
                        "image_id": iid,
                        "ocr_text": ocr_text,
                        "brand_name": brand,
                        "product_name": product,
                    }
                )
            st.session_state["private_rows"] = rows

        if "private_rows" in st.session_state:
            pred_df = pd.DataFrame(st.session_state["private_rows"])
            st.dataframe(pred_df, use_container_width=True)
            if solution is not None:
                gt = solution[solution["image_id"].isin(pred_df["image_id"])]
                score = _score_private(gt, pred_df)
                if score is not None:
                    st.metric("Private score (subset)", f"{score:.4f}")

with tab_text:
    ocr_in = st.text_area("ocr_text", "Dove Smoothie tẩy da chết", height=100)
    if private_mode:
        b, p = predict_private(ocr_in, product_fn)
        st.json({"ocr_text": ocr_in, "brand_name": b, "product_name": p})
    else:
        st.json({"ocr_text": ocr_in, "product_name": predict_public(ocr_in, product_fn)})
