# Private test data (SMCE hidden round)

| File | In repo | Notes |
|------|---------|--------|
| `private_test.csv` | Yes | 1,202 `image_id` rows |
| `sample_submission_private.csv` | Yes | Empty template (4 columns) |
| `images_sample/` | Yes | 6 demo JPEGs for quick smoke test |
| `images/` | **No** | Full 1,202 images — install locally |

## Install full images

From the hackathon BTC bundle or monorepo:

```bash
python scripts/setup_private_images.py /path/to/private_test
# e.g. from this repo's parent:
python scripts/setup_private_images.py ../private_test
```

## Local evaluation (optional)

Place `solution_private.csv` here (gitignored) to score in notebook / Streamlit.
Do **not** publish ground truth publicly.
