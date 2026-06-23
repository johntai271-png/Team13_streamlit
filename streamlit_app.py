#!/usr/bin/env python3
"""Streamlit demo shell for URA Hackathon teams — customize team_config.py + solution/."""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

import team_config as cfg
from solution import predict_from_image

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
    for key in ("ocr_text_live", "brand_name_live", "product_name_live", "upload_file_id"):
        if key not in st.session_state:
            st.session_state[key] = "" if key != "upload_file_id" else None


def _load_uploaded_image(uploaded) -> Image.Image:
    return Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")


def _clear_live_results() -> None:
    st.session_state["ocr_text_live"] = ""
    st.session_state["brand_name_live"] = ""
    st.session_state["product_name_live"] = ""


def _render_about_tab() -> None:
    st.header("About")
    st.markdown(
        """
        Tab này dành cho **mỗi team** trình bày giải pháp OCR + trích xuất
        **brand_name** và **product_name** cho cuộc thi. Hãy thay các placeholder
        bên dưới bằng nội dung thật của team bạn (hoặc chỉnh trực tiếp trong
        [`streamlit_app.py`](streamlit_app.py) hàm `_render_about_tab`).
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
        > **Placeholder — mô tả pipeline của team**

        1. **Tiền xử lý ảnh** — `[ví dụ: resize, tăng contrast, sharpen, …]`
        2. **OCR** — `[ví dụ: EasyOCR vi+en, PaddleOCR, custom model, …]`
        3. **Hậu xử lý OCR** — `[ví dụ: dedupe token, chuẩn hóa Unicode, …]`
        4. **Trích xuất brand** — `[ví dụ: regex dictionary, NER, fuzzy match, …]`
        5. **Trích xuất product** — `[ví dụ: rule-based, sklearn, LLM, …]`
        6. **Hậu kiểm / ensemble** — `[nếu có]`
        """
    )

    st.subheader("4. Điểm khác biệt & đóng góp chính")
    st.markdown(
        """
        - `[Điểm mạnh 1]`
        - `[Điểm mạnh 2]`
        - `[Điểm mạnh 3]`
        """
    )

    st.subheader("5. Công nghệ sử dụng")
    st.markdown(
        """
        | Thành phần | Công nghệ (placeholder) |
        |------------|-------------------------|
        | OCR | `[EasyOCR / …]` |
        | Brand extraction | `[Regex rules / …]` |
        | Product extraction | `[Sklearn / …]` |
        | Runtime | `[CPU / GPU, Python 3.11+]` |
        | Demo UI | `Streamlit` |
        """
    )

    st.subheader("6. Kết quả & đánh giá")
    st.markdown(
        """
        | Metric | Giá trị (placeholder) |
        |--------|------------------------|
        | F1 brand (local) | `[—]` |
        | 1 − CER (local) | `[—]` |
        | F1 product (local) | `[—]` |
        | **Private score** | `[—]` |
        | Thời gian inference / ảnh | `[—]` |
        """
    )

    st.subheader("7. Hạn chế & hướng phát triển")
    st.markdown(
        """
        **Hạn chế hiện tại**
        - `[ví dụ: brand mới chưa có trong từ điển]`

        **Hướng phát triển**
        - `[ví dụ: fine-tune OCR trên domain retail VN]`
        """
    )

    st.subheader("8. Liên kết")
    links = [
        f"- **Repository:** [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO})",
        "- **Setup & deploy:** [README.md](README.md)",
        f"- **Other resource:** [{cfg.OTHER_RESOURCE}]({cfg.OTHER_RESOURCE})",
    ]
    if cfg.STREAMLIT_APP_URL:
        links.insert(1, f"- **Live demo (Streamlit Cloud):** [{cfg.STREAMLIT_APP_URL}]({cfg.STREAMLIT_APP_URL})")
    st.markdown("\n".join(links))


tab_live, tab_about = st.tabs(["Live test", "About"])

with tab_live:
    _init_live_state()
    st.subheader("Live test")

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
                    pred = predict_from_image(img)
                    st.session_state["ocr_text_live"] = pred["ocr_text"]
                    st.session_state["brand_name_live"] = pred["brand_name"]
                    st.session_state["product_name_live"] = pred["product_name"]

            st.text_area("ocr_text", height=140, key="ocr_text_live")
            st.text_input("brand_name", key="brand_name_live")
            st.text_input("product_name", key="product_name_live")
    else:
        st.session_state["upload_file_id"] = None
        _clear_live_results()

with tab_about:
    _render_about_tab()
