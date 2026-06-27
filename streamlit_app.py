#!/usr/bin/env python3
"""Streamlit demo shell for URA Hackathon teams — customize team_config.py + solution/."""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

import team_config as cfg
from shared.benchmark import (
    get_deploy_smoke_benchmark,
    get_model_profile,
    run_predict_with_metrics,
)

APP_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

:root {{
    --ura-blue: {cfg.THEME_PRIMARY};
    --ura-blue-dark: {cfg.THEME_PRIMARY_DARK};
    --ura-bg: {cfg.THEME_BG};
    --ura-text: {cfg.THEME_TEXT};
    --ura-muted: {cfg.THEME_MUTED};
}}

html, body, .stApp {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    background-color: var(--ura-bg) !important;
    color: var(--ura-text) !important;
}}

[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}

[data-testid="stAppViewContainer"] > section > div {{
    padding-top: 1rem;
}}

[data-testid="stImage"]:first-of-type {{
    margin-bottom: 1rem;
}}

[data-testid="stImage"]:first-of-type img {{
    max-height: 72px;
    width: auto;
}}

.app-title,
[data-testid="stMarkdownContainer"] p.app-title {{
    display: block;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    color: var(--ura-blue) !important;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.25 !important;
}}

.app-subtitle {{
    display: block;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: var(--ura-muted) !important;
    margin: 0 0 0.75rem 0 !important;
    line-height: 1.5 !important;
    max-width: 100%;
}}

.app-team-info {{
    margin: 0 0 1.25rem 0;
    padding: 0;
    list-style: none;
}}

.app-team-info li {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    margin: 0 0 0.35rem 0 !important;
    color: var(--ura-text) !important;
}}

.app-team-info li strong {{
    color: var(--ura-blue);
    font-weight: 600;
}}

.app-team-info a {{
    color: var(--ura-blue);
    text-decoration: none;
    font-weight: 500;
}}

.app-team-info a:hover {{
    text-decoration: underline;
}}

[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    font-family: 'Montserrat', sans-serif !important;
    color: var(--ura-blue) !important;
}}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    color: var(--ura-blue) !important;
    border-bottom-color: var(--ura-blue) !important;
}}

.stTabs [data-baseweb="tab-list"] button {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background-color: var(--ura-blue) !important;
    border-color: var(--ura-blue) !important;
    color: #FFFFFF !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background-color: var(--ura-blue-dark) !important;
    border-color: var(--ura-blue-dark) !important;
}}

.stTextInput input,
.stTextArea textarea,
.stTextInput label,
.stTextArea label {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

[data-testid="stFileUploader"] label {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--ura-text) !important;
}}

[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}
"""

st.set_page_config(
    page_title=cfg.BROWSER_TITLE,
    page_icon=str(cfg.FAVICON),
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

st.image(str(cfg.LOGO), width=cfg.LOGO_WIDTH)

st.markdown(
    f'<p class="app-title">{cfg.PAGE_TITLE}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="app-subtitle">{cfg.SUBTITLE}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <ul class="app-team-info">
        <li><strong>Team Member:</strong> {cfg.TEAM_MEMBERS}</li>
        <li><strong>Github Repo link:</strong> <a href="{cfg.GITHUB_REPO}" target="_blank">{cfg.GITHUB_REPO}</a></li>
        <li><strong>Other resource link:</strong> <a href="{cfg.OTHER_RESOURCE}" target="_blank">{cfg.OTHER_RESOURCE}</a></li>
    </ul>
    """,
    unsafe_allow_html=True,
)


def _init_live_state() -> None:
    defaults = {
        "ocr_text_live": "",
        "brand_name_live": "",
        "product_name_live": "",
        "upload_file_id": None,
        "timing_ms": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _load_uploaded_image(uploaded) -> Image.Image:
    return Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")


def _clear_live_results() -> None:
    st.session_state["ocr_text_live"] = ""
    st.session_state["brand_name_live"] = ""
    st.session_state["product_name_live"] = ""
    st.session_state["timing_ms"] = None


@st.cache_data(show_spinner=False)
def _cached_model_profile() -> dict:
    return get_model_profile()


@st.cache_resource(show_spinner="Running deploy smoke benchmark (1 image)...")
def _cached_deploy_smoke() -> dict:
    return get_deploy_smoke_benchmark()


def _render_about_tab() -> None:
    st.header("About")
    st.markdown(
        """
        **TANAHI** — hệ thống OCR + trích xuất **brand_name** / **product_name**
        từ ảnh sản phẩm trên mạng xã hội, cho **The 2nd URA Hackathon 2026**.
        Giải pháp kết hợp OCR lai (Paddle det + VietOCR) với hai hướng trích xuất:
        rules + ML trên văn bản và phân tích bố cục ảnh.
        """
    )

    st.subheader("1. Thông tin team")
    st.markdown(
        f"""
        | Trường | Nội dung |
        |--------|----------|
        | **Tên team** | {cfg.TEAM_NAME} |
        | **Thành viên** | {cfg.TEAM_MEMBERS} |
        | **GitHub** | [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO}) |
        """
    )

    st.subheader("2. Bài toán")
    st.markdown(
        """
        Từ **ảnh sản phẩm trên kệ hàng / social media**, hệ thống cần trích xuất:

        - **`ocr_text`** — toàn bộ văn bản đọc được từ ảnh
        - **`brand_name`** — tên thương hiệu
        - **`product_name`** — tên / mô tả sản phẩm

        **Điểm private round:**

        `0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product`
        """
    )

    st.subheader("3. Ý tưởng & pipeline giải pháp")
    st.markdown(
        """
        **TANAHI** kết hợp **OCR lai (Paddle det + VietOCR)** với **2 hướng trích xuất**:
        văn bản (rules + ML trên OCR) và bố cục ảnh (khung đỏ + prominence).

        1. **Tiền xử lý ảnh** — resize về tối đa 1280px, tăng contrast (1.35×), sharpen.
        2. **OCR lai** — PaddleOCR **PP-OCRv4 chỉ detection** lấy bounding box chữ,
           rồi **VietOCR (vgg_seq2seq)** đọc tiếng Việt/Anh trên từng crop
           (không dùng latin rec của Paddle để đọc dấu chính xác hơn).
        3. **Hậu xử lý OCR** — chuẩn hoá Unicode (NFC), sửa lỗi song ngữ Việt–Anh,
           lọc nhiễu mạng xã hội (@mention, #hashtag, URL, caption TikTok), dedupe token.
        4. **Trích xuất chính `predict_labels`** — chuỗi ưu tiên:
           46 quy tắc regex brand → alias registry → BrandClassifier (TF-IDF + LogReg)
           → ProductClassifier (có gate) → BrandKNN (char-ngram similarity).
        5. **Pipeline bố cục `extract_v5`** — dò khung sản phẩm ("khung đỏ") bằng Canny,
           chấm điểm *prominence* (độ to / vị trí / chữ hoa) để tìm brand/product nổi bật.
        6. **Hợp nhất** — pipeline bố cục **chỉ bù** khi ML trống hoặc brand/product
           hiếm/chưa có trong train (prominence ≥ ngưỡng), **không ghi đè** ML đã chắc.
        """
    )

    st.subheader("4. Điểm khác biệt & đóng góp chính")
    st.markdown(
        """
        - **OCR lai Paddle-det + VietOCR** — tách phần *dò chữ* (Paddle, nhanh) khỏi
          phần *đọc chữ* (VietOCR, mạnh tiếng Việt có dấu), thay vì một engine làm cả hai.
        - **Chống ảo giác (anti-hallucination)** — chỉ giữ token brand vừa khớp từ điển
          vừa thực sự đọc được trong OCR, tránh bịa tên thương hiệu.
        - **Lọc nhiễu social/quảng cáo nhiều tầng** — loại caption TikTok, headline tin tức,
          mô tả dài, ngày tháng/giá %, để không nhầm thành brand/product.
        - **Học động có kiểm soát** — phát hiện brand/product mới từ ảnh train qua ngưỡng
          *prominence* + lọc rác + yêu cầu support tối thiểu (không học bừa).
        - **Ưu tiên hãng (manufacturer)** — chấm điểm để chọn tên hãng làm `brand_name`,
          tách dòng sản phẩm sang `product_name` (vd Abbott / Similac).
        """
    )

    st.subheader("5. Công nghệ sử dụng")
    st.markdown(
        """
        | Thành phần | Công nghệ |
        |------------|-----------|
        | Text detection | PaddleOCR PP-OCRv4 (det only) |
        | Text recognition | VietOCR (vgg_seq2seq) |
        | Brand extraction | 46 regex rules + alias registry + TF-IDF/LogReg + char-ngram KNN |
        | Product extraction | rules sub-line + ProductClassifier (gate + LogReg) + layout prominence |
        | Layout pipeline | OpenCV (Canny khung đỏ) + prominence scoring |
        | Runtime | CPU, Python 3.11+ |
        | Demo UI | Streamlit |
        """
    )

    st.subheader("6. Kết quả & đánh giá")
    st.markdown(
        """
        | Metric | Giá trị (placeholder) |
        |--------|------------------------|
        | F1 brand (local) | `[0.4756]` |
        | 1 − CER (local) | `[~0.85]` |
        | F1 product (local) | `[0.5501]` |
        | **Private score** | `[~0.625 – 0.678]` |
        | Latency (avg / image) | `[1,556 ms]` ms |
        | Product head size | `[0.0]` MB |
        """
    )
    st.markdown(
        """
        **Đo lightweight model (latency + footprint):**

        ```bash
        python scripts/benchmark_solution.py --limit 6
        ```

        Cập nhật `MODEL_PROFILE` trong [`team_config.py`](team_config.py)
        khi đổi OCR / model. Benchmark luôn chạy qua [`shared/benchmark.py`](shared/benchmark.py).
        """
    )

    st.subheader("7. Hạn chế & hướng phát triển")
    st.markdown(
        """
        **Hạn chế hiện tại**
        - OCR lai (Paddle det + VietOCR đọc từng crop) chậm hơn one-shot trên ảnh nhiều chữ.
        - Brand/product nằm ngoài từ điển + chưa đủ support trong train dễ bị bỏ sót.
        - Khung đỏ (Canny) kém ổn định với ảnh nền phức tạp hoặc nhiều sản phẩm.
        - VietOCR + PaddleOCR + PyTorch tốn RAM khi cold start trên Streamlit Cloud.

        **Hướng phát triển**
        - Mở rộng từ điển brand + tăng dữ liệu train cho các nhãn hiếm.
        - Thay khung đỏ Canny bằng object detection nhẹ (vd YOLO-nano) để khoanh vùng tốt hơn.
        - Fine-tune VietOCR trên domain bao bì sản phẩm bán lẻ Việt Nam.
        - Quantize / cache model để giảm cold start và latency trên Cloud.
        """
    )

    st.subheader("8. Liên kết")
    links = [
        f"- **Repository:** [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO})",
        "- **Setup & deploy:** [README.md](README.md)",
        f"- **Other resource:** [{cfg.OTHER_RESOURCE}]({cfg.OTHER_RESOURCE})",
    ]
    streamlit_url = getattr(cfg, "STREAMLIT_APP_URL", "")
    if streamlit_url:
        links.insert(
            1,
            f"- **Live demo (Streamlit Cloud):** [{streamlit_url}]({streamlit_url})",
        )
    st.markdown("\n".join(links))


tab_live, tab_about = st.tabs(["Live test", "About"])

with tab_live:
    _init_live_state()
    st.subheader("Live test")

    profile = _cached_model_profile()
    smoke = _cached_deploy_smoke()
    with st.expander("Model footprint (lightweight check)", expanded=False):
        st.markdown(
            f"- **Pipeline:** {profile.get('pipeline', '—')}\n"
            f"- **Runtime:** {profile.get('runtime_device', '—')}\n"
            f"- **Product head:** {profile.get('product_head_mb', 0)} MB\n"
            f"- **OCR note:** {profile.get('ocr_backend_note', '—')}\n\n"
            f"{profile.get('lightweight_notes', '')}"
        )
        if smoke.get("latency_ms"):
            lat = smoke["latency_ms"]
            st.markdown(
                f"**Deploy smoke benchmark (1 image):** "
                f"total **{lat.get('total_avg', '—')} ms** "
                f"(ocr {lat.get('ocr_avg', '—')} · extract {lat.get('extract_avg', '—')})"
            )
        elif smoke.get("error"):
            st.caption(f"Deploy smoke benchmark skipped: {smoke['error']}")
        st.caption("Full report: `python scripts/benchmark_solution.py --limit 6`")

    uploaded = st.file_uploader(
        "Ảnh sản phẩm",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="live_upload",
    )

    if uploaded:
        file_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state["upload_file_id"] != file_id:
            st.session_state["upload_file_id"] = file_id
            _clear_live_results()

        img = _load_uploaded_image(uploaded)
        col_img, col_result = st.columns(2)

        with col_img:
            st.image(img, use_container_width=True)

        with col_result:
            if st.button("Chạy OCR", type="primary", key="run_ocr_live"):
                with st.spinner("Đang chạy OCR..."):
                    pred = run_predict_with_metrics(img)
                    st.session_state["ocr_text_live"] = pred["ocr_text"]
                    st.session_state["brand_name_live"] = pred["brand_name"]
                    st.session_state["product_name_live"] = pred["product_name"]
                    st.session_state["timing_ms"] = pred.get("timing_ms")

            timing = st.session_state.get("timing_ms")
            if timing:
                t1, t2, t3 = st.columns(3)
                t1.metric("Total (ms)", f"{timing['total']:.1f}")
                t2.metric("OCR (ms)", f"{timing['ocr']:.1f}")
                t3.metric("Extract (ms)", f"{timing['extract']:.1f}")

            st.text_area("ocr_text", height=140, key="ocr_text_live")
            st.text_input("brand_name", key="brand_name_live")
            st.text_input("product_name", key="product_name_live")
    else:
        st.session_state["upload_file_id"] = None
        _clear_live_results()

with tab_about:
    _render_about_tab()
