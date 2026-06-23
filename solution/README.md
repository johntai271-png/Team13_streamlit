# Team solution folder

**Replace the Python modules in this folder with your team's pipeline.**  
The Streamlit demo and batch script call a stable API — keep the entry points below working.

## Required API

Implement in [`pipeline.py`](./pipeline.py):

```python
def predict_from_image(img: PIL.Image.Image, min_conf: float = 0.35) -> dict[str, str]:
    """
    Returns:
        {
            "ocr_text": str,
            "brand_name": str,
            "product_name": str,
        }
    """
```

Optional:

```python
def predict_from_text(ocr_text: str) -> tuple[str, str]:
    """Return (brand_name, product_name) from OCR text only."""
```

## Files in this folder

| File | Role | Team action |
|------|------|-------------|
| [`pipeline.py`](./pipeline.py) | OCR + brand/product orchestration | **Replace / enhance** |
| [`brand_rules.py`](./brand_rules.py) | Regex brand dictionary (baseline) | Extend or swap |
| [`product_model.py`](./product_model.py) | Sklearn product head (baseline) | Replace with your model |
| [`baseline_notebook.ipynb`](./baseline_notebook.ipynb) | Reference EDA + experiments | Replace with your notebook |

## Baseline stack (included)

1. EasyOCR (`vi` + `en`, CPU)
2. Image preprocess (resize, contrast, sharpen)
3. Regex brand rules → `brand_rules.py`
4. Optional TF-IDF + LogisticRegression product head → `product_model.py`

## Do not modify

- [`../shared/`](../shared/) — dataset path helpers
- [`../private_test/metric.py`](../private_test/metric.py) — official scoring
- [`../team_config.py`](../team_config.py) — branding (edit instead of forking pipeline imports)

## Quick test

```bash
python scripts/run_submission.py --limit 6
# → outputs/submission_private.csv
```

```bash
streamlit run streamlit_app.py
```

See [README.md](../README.md#team-setup) for the full fork checklist.
