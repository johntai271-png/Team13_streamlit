"""Team-facing configuration — edit this file after forking the template."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# Team identity (required after fork)

TEAM_NAME = "Team 13 - TANAHI"
TEAM_MEMBERS = "Trần Đức Hiếu, Phan Tấn Tài, Trịnh Hữu Trí, Nguyễn Hoàng Ngân"
GITHUB_REPO = "https://github.com/johntai271-png/Team13_streamlit.git"
OTHER_RESOURCE = "https://canva.link/z9023igdtcn2nfy"
STREAMLIT_APP_URL = ""  # e.g. "https://ura-team-abc.streamlit.app" after deploy


# Streamlit page copy

SUBTITLE = (
    "OCR & Product Name Extraction from Social Media Images "
    "by HCMUT URA Research Group"
)
PAGE_TITLE = f"The 2nd URA Hackathon - {TEAM_NAME}"
BROWSER_TITLE = PAGE_TITLE


# Branding assets (replace files under assets/ if needed)

ASSETS_DIR = REPO_ROOT / "assets"
FAVICON = ASSETS_DIR / "kaggle_144224_logos_thumb76_76.png"
LOGO = ASSETS_DIR / "bk_name_en.png"
LOGO_WIDTH = 280


# UI theme

THEME_PRIMARY = "#1565C0"
THEME_PRIMARY_DARK = "#0D47A1"
THEME_BG = "#FFFFFF"
THEME_TEXT = "#1A2B4A"
THEME_MUTED = "#5C6B8A"


# Default inference settings (override inside solution/pipeline.py if needed)

DEFAULT_MIN_CONF = 0.35


# Model footprint (edit when you change OCR / models — benchmark layer reads this)

MODEL_PROFILE: dict[str, str | float | None] = {
    "pipeline": (
        "PaddleOCR PP-OCRv4 (det only) + VietOCR (vgg_seq2seq) "
        "+ 46 regex brand rules + sklearn (TF-IDF + LogReg) + char-ngram KNN"
    ),
    "runtime_device": "CPU",
    "product_head_mb": None,  # auto-estimate when None
    "ocr_backend_note": (
        "Paddle det + VietOCR vgg_seq2seq weights downloaded once; "
        "OCR text = VietOCR đọc từng crop (không dùng latin rec của Paddle)"
    ),
    "lightweight_notes": (
        "Trích xuất brand/product chạy rules + sklearn (vài MB) trên CPU. "
        "Pipeline layout (khung đỏ + prominence) chỉ bù khi ML trống hoặc "
        "brand/product hiếm — không ghi đè dự đoán ML đã có support."
    ),
}
