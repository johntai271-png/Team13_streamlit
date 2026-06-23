<div align="center">

# SMCE Baseline Starter

**CPU-friendly OCR + brand/product extraction for the SMCE private test (1,202 images)**

<p>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/OCR-EasyOCR-orange" alt="EasyOCR"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-red" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"/>
</p>

</div>

## Overview

Reference starter for **The 2nd URA Hackathon 2026** private round. Teams get:

- `smce_baseline.ipynb` — end-to-end pipeline (OCR → `brand_name` + `product_name`)
- **Streamlit** demo for quick iteration on single images
- **`data/private_test/`** — IDs + sample images (full JPEG set installed separately)
- **`private_test/metric.py`** — same scoring formula as Kaggle hidden round

**Private score:**

`0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product`

## Repository layout

```
smce-baseline-starter/
├── smce_baseline.ipynb      # Main notebook
├── streamlit_app.py         # Interactive demo
├── smce_baseline_core.py    # OCR + predict helpers
├── smce_brand_rules.py      # Regex brand dictionary (0 MB)
├── smce_product_model.py    # Sklearn product head (~few MB)
├── data/
│   ├── train_labels.csv     # Weak labels for product head
│   └── private_test/
│       ├── private_test.csv
│       ├── sample_submission_private.csv
│       ├── images_sample/   # 6 demo images (in repo)
│       └── images/          # Full set (gitignored, ~100 MB)
├── private_test/metric.py   # Local scoring
└── scripts/
    ├── setup_private_images.py
    └── run_private_baseline.py
```

## Quickstart

### Prerequisites

- Python **3.11+**
- ~**1 GB RAM** (EasyOCR downloads ~200 MB weights on first run)

### Install

```bash
git clone https://github.com/YOUR_ORG/smce-baseline-starter.git
cd smce-baseline-starter
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Full private images (recommended)

The repo ships **6 sample JPEGs**. For the full **1,202** images:

```bash
python scripts/setup_private_images.py /path/to/private_test
```

If you have the hackathon monorepo checked out:

```bash
python scripts/setup_private_images.py ../private_test
```

### Streamlit demo

```bash
streamlit run streamlit_app.py
```

Tabs: upload image · pick from private catalog · text-only rule tuning.

### Notebook

```bash
jupyter notebook smce_baseline.ipynb
```

Run cells in order. Cell 9 is the **private live test** (same logic as Streamlit).

### Batch submission (CLI)

```bash
python scripts/run_private_baseline.py --limit 20   # smoke test on 20 jpg
python scripts/run_private_baseline.py              # all images on disk
```

Output: `submission_private.csv` (4 columns, Kaggle-ready placeholders).

### Local scoring (optional)

Copy `solution_private.csv` (BTC only) to `data/private_test/` — file is **gitignored**.

```python
import pandas as pd
import sys
sys.path.insert(0, "private_test")
from metric import score

sol = pd.read_csv("data/private_test/solution_private.csv", keep_default_na=False)
sub = pd.read_csv("submission_private.csv", keep_default_na=False)
print(score(sol, sub, "image_id"))
```

## Submission format (private)

| Column | Description |
|--------|-------------|
| `image_id` | `priv_h_*` or `priv_d_*` |
| `ocr_text` | Raw OCR |
| `brand_name` | Brand entity |
| `product_name` | Product entity |

Empty fields → single space `" "` in CSV.

## Deploy Streamlit (Cloud)

1. Push this repo to GitHub (public).
2. [share.streamlit.io](https://share.streamlit.io) → New app → `streamlit_app.py`.
3. For full private catalog, attach images via your own storage or use **Upload** tab only.

## Customize

| Goal | Where |
|------|--------|
| Add brands | `smce_brand_rules.py` → `BRAND_RULES` |
| Train product head | `data/train_labels.csv` + Cell 3b |
| Swap OCR | Replace EasyOCR in notebook / `smce_baseline_core.py` |
| Private metric weights | `private_test/metric.py` |

## Status

| Component | Status |
|-----------|--------|
| Private test IDs + sample images | Included |
| Full 1,202 images | Install via script |
| Streamlit demo | Ready |
| Notebook private cell | Ready |
| Ground truth solution | Not public (BTC) |

## License

MIT — see [LICENSE](LICENSE).

---

SMCE Challenge 2026 · RAISE Lab, HCMUT
