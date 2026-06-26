"""
Team solution pipeline — replace this module (and siblings) with your own approach.

The Streamlit demo and submission script import:
    predict_from_image(img) -> {"ocr_text", "brand_name", "product_name", "timing_ms"?}
    get_model_profile() -> see shared/benchmark.py (template-owned)
"""

# ============================================================
# CELL 5 — HUẤN LUYỆN MÔ HÌNH + HỌC ĐỘNG (prominence > DYNAMIC_PROMINENCE_THRESHOLD)
# Học từ train_labels.csv: BrandClassifier + ProductClassifier + BrandKNN.
# Thu thập brand/product MỚI qua pipeline (KHÔNG học ML đơn thuần):
# chỉ đăng ký khi prominence > DYNAMIC_PROMINENCE_THRESHOLD + lọc rác.
# ============================================================
import re
import unicodedata
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os
import warnings
import numpy as np
import time
import cv2
from PIL import Image
from typing import Any

warnings.filterwarnings('ignore')

# ---- Cấu hình chính ----
RUNNING_ON_KAGGLE      = os.path.exists('/kaggle/input')
OCR_BOX_MIN_AREA_RATIO     = 0.002   # lọc box det < 0.2% — bắt chữ nhỏ hơn
OCR_PREPROCESS_MAX_DIM     = 1400    # resize trước OCR (cao hơn = chữ rõ hơn)
OCR_PREPROCESS_CONTRAST    = 1.45    # tăng contrast tiền xử lý
OCR_READING_ORDER_ROW_TOL  = 0.55    # ngưỡng gom dòng khi sắp box đọc
MEDIA_FRAME_SAMPLES        = 11      # mẫu frame GIF/video (cell 7b)
PIPELINE_LAYOUT_MIN_PROMINENCE = 30.0  # layout chỉ bù ML khi prominence >= ngưỡng này
PIPELINE_LAYOUT_FALLBACK_BRAND = 22.0  # ngưỡng fallback brand trong extract_v5
DYNAMIC_PROMINENCE_THRESHOLD = 28.0  # học động: đăng ký brand/product mới khi prominence > ngưỡng
SKIP_TRAIN_DISCOVERY       = False   # True = bỏ qua bước học brand mới từ ảnh train
TRAIN_DISCOVERY_SAMPLE     = 100     # số ảnh train tối đa dùng để khám phá brand mới

CPU_THREADS = max(1, os.cpu_count() or 2)
NUM_WORKERS = max(1, min(CPU_THREADS, 4))
for _env in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env, str(CPU_THREADS))

def _find_first(root, names):
    """Tìm đệ quy file theo danh sách tên ưu tiên (vd private_test.csv trước test.csv)."""
    if root is None or not Path(root).exists():
        return None
    root = Path(root)
    for nm in names:
        hits = sorted(root.rglob(nm))
        if hits:
            return hits[0]
    return None

def _find_images_dir(root, prefer=('test',)):
    """Trả về thư mục chứa nhiều ảnh nhất (đệ quy). Ưu tiên thư mục có từ khoá
    trong `prefer` để tách biệt ảnh test và ảnh train."""
    if root is None or not Path(root).exists():
        return None
    root = Path(root)
    items = []
    cands = [root] + [p for p in root.rglob('*') if p.is_dir()]
    for d in cands:
        try:
            n = sum(1 for p in d.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        except Exception:
            n = 0
        if n > 0:
            items.append((d, n))
    if not items:
        return None
    pref = [t for t in items if any(k in str(t[0]).lower() for k in prefer)]
    pool = pref if pref else items
    return max(pool, key=lambda t: t[1])[0]

if RUNNING_ON_KAGGLE:
    _roots = [
        Path('/kaggle/input/competitions/the-2nd-ura-hackathon'),
        Path('/kaggle/input/the-2nd-ura-hackathon'),
    ]
    COMP_ROOT = next((p for p in _roots if p.exists()), _roots[0])
    WORK_DIR  = Path('/kaggle/working')
else:
    COMP_ROOT = Path(os.environ.get(
        'TANAHI_COMP_ROOT',
        r'C:\Users\LENOVO\Downloads\Urahackathon2026\the-2nd-ura-hackathon'))
    WORK_DIR  = Path('.')

# Khi import như thư viện (Streamlit / run_submission) KHÔNG chạy batch OCR + self-test;
# chỉ bật khi chạy trong notebook/script qua biến môi trường TANAHI_RUN_BATCH=1.
_RUN_BATCH = os.environ.get('TANAHI_RUN_BATCH', '0') == '1'

TEST_CSV   = _find_first(COMP_ROOT, ['private_test.csv', 'test_private.csv',
                                     'public_test.csv', 'test.csv'])
SAMPLE_CSV = _find_first(COMP_ROOT, ['sample_submission_private.csv', 'sample_submission.csv',
                                     'sample_submission_public.csv'])
TRAIN_CSV  = _find_first(COMP_ROOT, ['train_labels.csv', 'train.csv', 'train_label.csv'])

# Ảnh test: ưu tiên cùng nhánh với file test (tránh nhầm ảnh train)
_test_base = TEST_CSV.parent if TEST_CSV is not None else COMP_ROOT
IMAGES_DIR = (_find_images_dir(_test_base, prefer=('test',))
              or _find_images_dir(COMP_ROOT, prefer=('test',)) or COMP_ROOT)
# Ảnh train: phục vụ học động (discovery)
TRAIN_IMAGES_DIR = ((_find_images_dir(TRAIN_CSV.parent, prefer=('train',)) if TRAIN_CSV is not None else None)
                    or _find_images_dir(COMP_ROOT, prefer=('train',)))

OUTPUT_CSV     = WORK_DIR / 'submission.csv'
CHECKPOINT_CSV = WORK_DIR / 'checkpoint.csv'

if TEST_CSV is not None and TEST_CSV.exists():
    test_df = pd.read_csv(TEST_CSV)
else:
    # Không có dataset test (vd deploy Streamlit) — chỉ phục vụ predict_from_image/text.
    test_df = pd.DataFrame({'image_id': []})

def _detect_col(df, *cands):
    if df is None:
        return None
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

# Chuẩn hoá cột id của test về 'image_id'
_TEST_ID = _detect_col(test_df, 'image_id', 'id', 'img_id', 'image', 'filename', 'file_name')
if _TEST_ID and _TEST_ID != 'image_id':
    test_df = test_df.rename(columns={_TEST_ID: 'image_id'})

# Lập chỉ mục đường dẫn media test (ảnh tĩnh + GIF/WebP động + video)
STATIC_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
ANIMATED_IMAGE_EXTS = ('.gif', '.webp', '.apng')
VIDEO_EXTS = ('.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v', '.mpeg', '.mpg')
ALL_TEST_MEDIA_EXTS = STATIC_IMAGE_EXTS + ANIMATED_IMAGE_EXTS + VIDEO_EXTS

IMAGE_PATH_INDEX = {}
if IMAGES_DIR and Path(IMAGES_DIR).is_dir():
    for p in Path(IMAGES_DIR).glob('*'):
        if p.suffix.lower() in ALL_TEST_MEDIA_EXTS:
            IMAGE_PATH_INDEX[p.stem] = str(p)
            IMAGE_PATH_INDEX[p.name] = str(p)

from solution.brand_rules import (
    train_labels_df,
    OCR_COL,
    HAS_BRAND_COL,
    BRAND_COL,
    PRODUCT_COL,
    MIN_BRAND_SUPPORT,
    MIN_BRAND_ALIAS_LEN,
    BRAND_REGISTRY,
    _fold_ascii,
    _is_generic_product,
    _register_brand,
    _build_product_canon_map,
    _is_plausible_brand_name,
    _is_plausible_product_name,
    normalize_brand_name,
    normalize_product_name,
    split_brand_product,
    detect_brand_in_ocr,
    build_brand_knowledge_from_train,
    _PRODUCT_CANON_MAP,
    _is_ocr_noise_only,
    _is_informational_noise,
    clean_social_ocr,
    reconcile_brand_product,
    _is_social_caption,
    _is_glued_product_brand,
    _is_description_prose,
    extract_by_rules,
    post_process_prediction,
    guess_product_from_ocr,
    SOCIAL_NOISE_WORDS,
    apply_ocr_typo_map,
    correct_ocr_from_train_catalog,
)

from solution.product_model import (
    BrandClassifier,
    ProductClassifier,
    BrandKNN,
    brand_clf,
    product_clf,
    brand_knn,
    _train_frame,
    _BRAND_SUPPORT_COUNTS,
    _PRODUCT_SUPPORT_COUNTS,
    _brand_support_count,
    _product_support_count,
    _is_rare_or_new_brand,
    _is_rare_or_new_product,
    _register_discovered_product,
    retrain_predictors,
)

# ============================================================
# CELL 4 — LÕI PIPELINE (BASELINE / Ý TƯỞNG CHỦ ĐẠO)
# Phát hiện khung vật thể (đỏ) + khung chữ, đọc bằng VietOCR (cell 7),
# hậu xử lý song ngữ, rồi trích xuất brand/product theo bố cục.
# Import cv2: nếu lệch ABI với numpy hiện tại -> cài lại opencv-headless (không đổi numpy).
# ============================================================
import sys, subprocess

def _import_cv2_safe():
    try:
        import cv2 as _cv2
        return _cv2
    except (ImportError, RuntimeError, OSError):
        print("  OpenCV lệch ABI numpy %s -> cài lại opencv-python-headless..." % (
            __import__('numpy').__version__))
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y",
             "opencv-python", "opencv-contrib-python", "opencv-python-headless"],
            check=False,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
             "--force-reinstall", "opencv-python-headless>=4.10.0"],
            check=False,
        )
        for _m in list(sys.modules):
            if _m == "cv2" or _m.startswith("cv2."):
                del sys.modules[_m]
        import cv2 as _cv2
        return _cv2

cv2 = _import_cv2_safe()
import difflib

# ---------- Hậu xử lý chính tả song ngữ (Việt - Anh) ----------
ENGLISH_PRODUCT_WORDS = {
    "setup","install","gpu","cpu","model","data","image","file",
    "download","link","process","python","import","from","train",
    "test","run","error","api","version","config","github","project",
    "maybelline","mascara","hyper","curl","waterproof","gel","volume",
    "matte","cream","lotion","serum","cleanser","shampoo","conditioner",
    "eyeliner","eyebrow","lipstick","lipbalm","foundation","powder",
    "blush","skincare","active","formula","natural","organic","extract",
    "original","premium","classic","style","design","brand","product",
    "made","in","vietnam","usa","korea","japan","quality","best",
    "super","ultra","mega","extreme","pro","max","smart","light","dark",
    "color","black","white","red","blue","green","ml","g","kg","size",
    "pack","set","limit","edition","free","soft","smooth","fresh","cool",
    "dry","wet","hot","warm","sunblock","sunscreen","uv","protect","moisture",
}

def remove_vietnamese_accents(input_str):
    nfd_form = unicodedata.normalize('NFD', input_str)
    only_ascii = "".join([c for c in nfd_form if not unicodedata.combining(c)])
    return only_ascii.replace('\u0110', 'D').replace('\u0111', 'd')

def clean_bilingual_word(word):
    """Sửa lỗi từ ngữ dựa trên bộ từ điển bao bì thu nhỏ."""
    cleaned = word
    clean_word = re.sub(r'^\W+|\W+$', '', word)
    if re.match(r'^[a-zA-Z0-9]+$', clean_word):
        temp = clean_word.replace('0', 'o').replace('1', 'i')
        if temp.lower() in ENGLISH_PRODUCT_WORDS:
            return word.replace(clean_word, temp)
    if any(char in '\u00e0\u00e1\u1ea3\u00e3\u1ea1\u1eb1\u1eaf\u1eb3\u1eb5\u1eb7\u1ea7\u1ea5\u1ea9\u1eab\u1ead\u00e8\u00e9\u1ebb\u1ebd\u1eb9\u1ec1\u1ebf\u1ec3\u1ec5\u1ec7\u00ec\u00ed\u1ec9\u0129\u1ecb\u00f2\u00f3\u1ecf\u00f5\u1ecd\u1ed3\u1ed1\u1ed5\u1ed7\u1ed9\u1edd\u1edb\u1edf\u1ee1\u1ee3\u00f9\u00fa\u1ee7\u0169\u1ee5\u1eeb\u1ee9\u1eed\u1eef\u1ef1\u1ef3\u00fd\u1ef7\u1ef9\u1ef5\u0111' for char in clean_word.lower()):
        no_accent_word = remove_vietnamese_accents(clean_word)
        if no_accent_word.lower() in ENGLISH_PRODUCT_WORDS:
            if clean_word.istitle():
                final_word = no_accent_word.capitalize()
            elif clean_word.isupper():
                final_word = no_accent_word.upper()
            else:
                final_word = no_accent_word.lower()
            return word.replace(clean_word, final_word)
    return cleaned

def bilingual_post_processor(text_list):
    processed_list = []
    for text in text_list:
        if not text:
            processed_list.append("")
            continue
        text = unicodedata.normalize('NFC', str(text).strip())
        text = apply_ocr_typo_map(text)
        words_in_line = text.split()
        cleaned_words = [clean_bilingual_word(w) for w in words_in_line]
        processed_line = " ".join(cleaned_words)
        processed_line = re.sub(r'\s+([.,!?;:])', r'\1', processed_line)
        processed_line = re.sub(r'\s+', ' ', processed_line).strip()
        processed_list.append(processed_line)
    return processed_list


# ---------- Từ điển & mẫu hỗ trợ trích xuất bố cục ----------
BRAND_NORMALIZATION_MAP = {
    "ha long": "Ha Long Canfoco",
    "canfoco": "Ha Long Canfoco",
    "\u0111\u1ed3 h\u1ed9p h\u1ea1 long": "Ha Long Canfoco",
    "dove": "Dove",
    "vinamilk": "Vinamilk",
    "vissan": "Vissan",
    "nestle": "Nestl\u00e9",
    "nescafe": "Nescafe",
    "omo": "Omo",
    "knorr": "Knorr",
    "maybelline": "Maybelline",
}

PROMO_KEYWORDS = {"ch\u00ednh h\u00e3ng", "\u0111\u1ed9c quy\u1ec1n", "freeship", "cam k\u1ebft", "uy t\u00edn", "qu\u00e0 t\u1eb7ng", "m\u1edbi", "new", "official"}

DESCRIPTION_INDICATORS = [
    "\u0111\u1ea7u c\u1ecd", "cho mi", "d\u1ea1ng gel", "c\u00f4ng th\u1ee9c", "ph\u1ee7 hi\u1ec7u", "h\u00e0ng mi",
    "th\u1ea5m n\u01b0\u1edbc", "ch\u1ea3i t\u1eebng", "nh\u1eb9 m\u01b0\u1ee3t", "ch\u1ed1ng tr\u00f4i", "cong v\u00fat", "su\u1ed1t nhi\u1ec1u",
    "si\u00eau m\u1ecbn", "b\u00e0o m\u00f2n", "kh\u00f4ng lo", "m\u1ec1m m\u1ecbn", "l\u00e0n da", "d\u01b0\u1ee1ng \u1ea9m",
]

VARIANT_PATTERN = r'\d+\s*(g|ml|kg|l|gr|gx|h\u1ed9p|g\u00f3i|chai|l\u1ed1c|lon|c\u00e1i|vi\u00ean|h\u0169)\s*(x\s*\d+)?|\b(combo|l\u1ed1c|v\u1ec9|set)\s*\d+\s*(h\u1ed9p|g\u00f3i|chai|l\u1ed1c|lon|c\u00e1i|vi\u00ean|h\u0169)?'
PROMO_PATTERN = r'\b(gi\u1ea3m|giam)\s*\d+%\b|\b\d+%\s*(off|gi\u1ea3m|giam)?\b|\bmua\s*\d+\s*t\u1eb7ng\s*\d+\b'


def fuzzy_match_brand(ocr_text, normalization_map, threshold=0.75):
    ocr_clean = re.sub(r'\W+', '', ocr_text.lower())
    for variant, canonical in normalization_map.items():
        v_clean = re.sub(r'\W+', '', variant.lower())
        if v_clean in ocr_clean:
            return True, canonical
    words = ocr_text.lower().split()
    for word in words:
        clean_word = re.sub(r'\W+', '', word)
        for variant, canonical in normalization_map.items():
            v_clean = re.sub(r'\W+', '', variant.lower())
            ratio = difflib.SequenceMatcher(None, clean_word, v_clean).ratio()
            if ratio >= threshold:
                return True, canonical
    return False, None


def is_negative_line(text):
    text_lower = text.lower()
    patterns = [
        r'\d{2,4}[-/\.]\d{2}[-/\.]\d{2,4}',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'www\.|http[s]?://|\.com|\.vn',
        r'^\d+$',
    ]
    if any(re.search(p, text) for p in patterns):
        return True
    neg_keywords = {"ng\u00e0y s\u1ea3n xu\u1ea5t", "th\u00e0nh ph\u1ea7n", "ingredients", "ch\u1ec9 ti\u00eau", "nh\u00e2n qu\u1ea3", "di s\u1ea3n", "nh\u1eafn nh\u1ee7"}
    if any(kw in text_lower for kw in neg_keywords):
        return True
    return False


def clean_title_noise(text):
    cleaned = re.sub(r'^[A-Za-z]\s+|\s+[A-Za-z]$', '', text)
    return cleaned.strip()


def is_brand_duplicate(text, brand_name):
    if not brand_name:
        return False
    t_clean = re.sub(r'\W+', '', text.lower())
    b_clean = re.sub(r'\W+', '', brand_name.lower())
    if t_clean == b_clean:
        return True
    brand_words = [re.sub(r'\W+', '', w) for w in brand_name.lower().split()]
    if t_clean in brand_words and len(t_clean) > 1:
        return True
    return False


def is_description_line(text, y_center, max_y):
    text_lower = text.lower()
    if any(indicator in text_lower for indicator in DESCRIPTION_INDICATORS):
        return True
    if any(text_lower.startswith(word) for word in ["\u0111\u1ea7u", "d\u1ea1ng", "c\u00f4ng", "cho", "th\u1ea5m", "ph\u1ee7", "h\u00e0ng", "l\u00e0m", "gi\u00fap", "h\u1ea1t", "kh\u00f4ng"]):
        return True
    if max_y > 0 and y_center > (max_y * 0.35):
        if "smoothie" not in text_lower and "mascara" not in text_lower:
            return True
    return False


# ---------- Chấm điểm ứng viên & trích xuất thông tin ----------
def check_inside_red_box(box_coords, red_box):
    if not red_box: return 0.0
    xs, ys = [p[0] for p in box_coords], [p[1] for p in box_coords]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    ixmin, iymin = max(xmin, red_box[0]), max(ymin, red_box[1])
    ixmax, iymax = min(xmax, red_box[2]), min(ymax, red_box[3])
    inter_area = max(0, ixmax - ixmin) * max(0, iymax - iymin)
    box_area = (xmax - xmin) * (ymax - ymin)
    return inter_area / (box_area + 1e-5)

def compute_candidate_score(box_coords, text, max_y, red_box, img_area):
    xs, ys = [p[0] for p in box_coords], [p[1] for p in box_coords]
    height, width = max(ys) - min(ys), max(xs) - min(xs)
    y_center = min(ys) + (height / 2)
    box_area = width * height
    aspect_ratio = width / (height + 1e-5)
    position_weight = 1.0 - (y_center / max_y) if max_y > 0 else 0.5
    base_score = height * aspect_ratio * position_weight
    area_ratio = box_area / img_area if img_area > 0 else 0
    overlap_ratio = check_inside_red_box(box_coords, red_box)
    multiplier = 1.0
    if overlap_ratio > 0.5: multiplier *= 1.5
    else:
        if area_ratio > 0.03: multiplier *= 2.0
        else: multiplier *= 0.5
    return base_score * multiplier

def compute_brand_prominence_score(line, max_y):
    text, height, y_center = line["text"], line["height"], line["y_center"]
    capital_multiplier = 2.0 if text.isupper() else (1.5 if text and text[0].isupper() else 1.0)
    word_count = len(text.split())
    length_multiplier = 2.0 if 1 <= word_count <= 3 else (0.3 if word_count > 6 else 1.0)
    y_factor = (y_center / max_y) if max_y > 0 else 0.5
    position_weight = (1.0 - y_factor) if y_center < (max_y * 0.65) else 0.05
    return height * capital_multiplier * length_multiplier * position_weight


def extract_universal_product_info_v5(boxes, texts, img_shape=None, red_box=None):
    lines_info = []
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        ys = [p[1] for p in box]
        height = max(ys) - min(ys)
        y_center = min(ys) + (height / 2)
        lines_info.append({"text": text, "y_center": y_center, "height": height,
                           "is_brand": False, "is_promo": False, "is_variant": False, "box": box})

    if not lines_info: return {}

    if img_shape is not None:
        img_area = img_shape[0] * img_shape[1]
        max_box_area = max((max(p[0] for p in b) - min(p[0] for p in b)) * (max(p[1] for p in b) - min(p[1] for p in b)) for b in boxes)
        if (max_box_area / img_area) < 0.010:
            return {"brand_name": " ", "product_name": " ", "variant": "Kh\u00f4ng c\u00f3", "description": ["\u1ea2nh r\u00e1c/Ch\u1eef qu\u00e1 nh\u1ecf"]}

    max_y = max(line["y_center"] for line in lines_info)
    brand_name, product_name = None, None
    product_name_parts = []

    variant_specs = []
    for line in lines_info:
        t_low = line["text"].lower()
        if (_is_social_caption(line["text"]) or _is_description_prose(line["text"])
                or _is_informational_noise(line["text"])
                or is_negative_line(line["text"])
                or any(k in t_low for k in PROMO_KEYWORDS) or re.search(PROMO_PATTERN, t_low)):
            line["is_promo"] = True
        if re.search(VARIANT_PATTERN, t_low):
            line["is_variant"] = True
            variant_specs.append(line["text"])

    valid_lines = [l for l in lines_info if not l['is_promo'] and not l['is_variant'] and len(l['text']) >= 3]
    repeating_groups = []

    for l in valid_lines:
        matched = False
        t_clean = re.sub(r'\W+', '', l['text'].lower())
        for g in repeating_groups:
            if difflib.SequenceMatcher(None, t_clean, g['clean_rep']).ratio() > 0.75:
                g['lines'].append(l)
                curr_text = l['text']
                rep_text = g['rep']
                curr_score = (len(curr_text.replace(" ", "")), -curr_text.count(" "))
                rep_score = (len(rep_text.replace(" ", "")), -rep_text.count(" "))
                if curr_score > rep_score:
                    g['rep'] = curr_text
                    g['clean_rep'] = t_clean
                matched = True
                break
        if not matched:
            repeating_groups.append({'rep': l['text'], 'clean_rep': t_clean, 'lines': [l]})

    multi_groups = [g for g in repeating_groups if len(g['lines']) >= 2]

    if multi_groups:
        for g in multi_groups:
            g['score'] = sum(compute_brand_prominence_score(l, max_y) for l in g['lines']) / len(g['lines'])
        multi_groups.sort(key=lambda x: x['score'], reverse=True)
        if len(multi_groups) == 1:
            brand_name = multi_groups[0]['rep']
            for l in multi_groups[0]['lines']: l['is_brand'] = True
        else:
            g1, g2 = multi_groups[0], multi_groups[1]
            if len(g1['rep']) > len(g2['rep']):
                product_name_parts.append(g1['rep'])
                brand_name = g2['rep']
            else:
                product_name_parts.append(g2['rep'])
                brand_name = g1['rep']
            for l in g1['lines'] + g2['lines']: l['is_brand'] = True

    if not brand_name:
        for line in lines_info:
            matched, m_brand = fuzzy_match_brand(line["text"], BRAND_NORMALIZATION_MAP)
            if matched:
                brand_name = m_brand
                line["is_brand"] = True
                break
        if not brand_name:
            for line in lines_info: line["prominence_score"] = compute_brand_prominence_score(line, max_y)
            fallback_brand = max(valid_lines, key=lambda x: x.get("prominence_score", 0), default=None)
            if fallback_brand and fallback_brand.get("prominence_score", 0) >= PIPELINE_LAYOUT_FALLBACK_BRAND:
                brand_name = fallback_brand["text"]
                fallback_brand["is_brand"] = True

    img_area = img_shape[0] * img_shape[1] if img_shape else 1.0
    product_candidates = [l for l in valid_lines if not l["is_brand"] and not is_brand_duplicate(l["text"], brand_name)]

    if not product_name_parts and product_candidates:
        for c in product_candidates:
            c["score"] = compute_candidate_score(c["box"], c["text"], max_y, red_box, img_area)
        max_score = max(c["score"] for c in product_candidates)
        for c in product_candidates:
            if c["score"] >= max_score * 0.35 and not is_description_line(c["text"], c["y_center"], max_y):
                product_name_parts.append(clean_title_noise(c["text"]))

    description_parts = [l["text"] for l in valid_lines if not l["is_brand"] and clean_title_noise(l["text"]) not in product_name_parts]
    descriptions, current_desc = [], ""
    for desc in description_parts:
        if current_desc and (desc[0].islower() or desc.startswith("kh\u00f4ng") or desc.startswith("h\u01b0\u01a1ng")):
            current_desc += " " + desc
        else:
            if current_desc: descriptions.append(current_desc)
            current_desc = desc
    if current_desc: descriptions.append(current_desc)

    return {
        "brand_name": brand_name if brand_name else " ",
        "product_name": " ".join(product_name_parts) if product_name_parts else " ",
        "variant": ", ".join(variant_specs) if variant_specs else "Kh\u00f4ng c\u00f3",
        "description": descriptions
    }


# ---------- Cắt khung, tìm khung đỏ, đọc VietOCR, gói pipeline ----------
def crop_box(pil_img, box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    xmin, xmax = int(min(xs)), int(max(xs))
    ymin, ymax = int(min(ys)), int(max(ys))
    w, h = pil_img.size
    xmin = max(0, xmin - 3); ymin = max(0, ymin - 3)
    xmax = min(w, xmax + 3); ymax = min(h, ymax + 3)
    return np.array(pil_img.crop((xmin, ymin, xmax, ymax)))


def find_red_box(cv_img, sorted_boxes):
    """Tìm khung vật thể sản phẩm (khung đỏ) bằng Canny + dilate + chấm điểm
    (số khung chữ bên trong, độ ở giữa, độ lớn). Giữ nguyên ý tưởng pipeline."""
    img_h, img_w = cv_img.shape[:2]
    img_area = img_h * img_w
    img_center = (img_w / 2, img_h / 2)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_box = None
    best_score = -1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < img_area * 0.05 or area > img_area * 0.65:
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.15 or aspect_ratio > 3.0:
            continue
        obj_center = (x + w / 2, y + h / 2)
        dist = ((obj_center[0] - img_center[0]) ** 2 + (obj_center[1] - img_center[1]) ** 2) ** 0.5
        max_dist = (img_center[0] ** 2 + img_center[1] ** 2) ** 0.5
        centricity = 1.0 - (dist / max_dist)
        text_inside = 0
        for box in sorted_boxes:
            bx = sum(p[0] for p in box) / 4
            by = sum(p[1] for p in box) / 4
            if x <= bx <= x + w and y <= by <= y + h:
                text_inside += 1
        score = (text_inside * 2.0) + (centricity * 1.5) + (area / img_area)
        if score > best_score:
            best_score = score
            red_box = [x, y, x + w, y + h]
    return red_box


def recognize_vietocr(crop_np):
    """Đọc 1 crop bằng VietOCR (predictor khởi tạo ở cell 7)."""
    try:
        return vietocr_predictor.predict(Image.fromarray(crop_np))
    except Exception:
        return ''


# ---------- Ngưỡng tự tin động + lọc rác (tổng quát) ----------
def _is_valid_pipeline_label(text, field='brand'):
    """Chặn từ khóa rác/quảng cáo/negative — không cho gán vào brand/product mới."""
    if not text or str(text).strip() in ('', ' '):
        return False
    t = unicodedata.normalize('NFC', str(text).strip())
    t_low = t.lower()
    if is_negative_line(t):
        return False
    if any(k in t_low for k in PROMO_KEYWORDS):
        return False
    if re.search(PROMO_PATTERN, t_low):
        return False
    # Bổ sung: % giảm / off (word boundary Unicode hay fail với tiếng Việt)
    if re.search(r'(?:gi[\u1ea3a]m|off|sale|discount)\s*\d+\s*%', t_low):
        return False
    if re.search(r'\d+\s*%\s*(?:off|gi[\u1ea3a]m|sale)?', t_low):
        return False
    if _is_ocr_noise_only(t):
        return False
  # token mạng xã hội đơn lẻ
    words = re.findall(r'[^\W\d_]+', t_low, flags=re.UNICODE)
    if words and all(w in SOCIAL_NOISE_WORDS for w in words):
        return False
    if field == 'brand' and _is_generic_product(t) and len(words) <= 1:
        return False
    if field == 'brand' and not _is_plausible_brand_name(t):
        return False
    if _is_informational_noise(t):
        return False
    if field == 'product' and _is_generic_product(t):
        return False
    if _is_social_caption(t):
        return False
    if _is_description_prose(t):
        return False
    if _is_glued_product_brand(t) and field == 'brand':
        return False
    if field == 'product' and not _is_plausible_product_name(t):
        return False
    return True


def _compute_max_prominence(boxes, texts, img_shape, target_text=None):
    """Tính prominence score cao nhất từ bố cục pipeline (ngưỡng học động / layout bù)."""
    if not boxes or not texts:
        return 0.0
    lines = []
    max_y = 0.0
    for box, text in zip(boxes, texts):
        t = str(text or '').strip()
        if len(t) < 2:
            continue
        ys = [p[1] for p in box]
        height = max(ys) - min(ys)
        y_center = min(ys) + height / 2
        max_y = max(max_y, y_center)
        lines.append({'text': t, 'y_center': y_center, 'height': height})
    if not lines:
        return 0.0
    if max_y <= 0:
        max_y = 1.0
    scores = []
    for ln in lines:
        if is_negative_line(ln['text']):
            continue
        if target_text:
            t_clean = re.sub(r'\W+', '', ln['text'].lower())
            tgt = re.sub(r'\W+', '', target_text.lower())
            if tgt and t_clean != tgt:
                ratio = difflib.SequenceMatcher(None, t_clean, tgt).ratio()
                if ratio < 0.55 and tgt not in t_clean and t_clean not in tgt:
                    continue
        scores.append(compute_brand_prominence_score(ln, max_y))
    return max(scores) if scores else 0.0


def pipeline_predict_with_metrics(pil_img, cv_img, sorted_boxes):
    """ĐỌC TRỰC TIẾP pipeline + trả thêm prominence score (chất lượng bố cục ảnh).
    Trả về (brand_name, product_name, prominence_score)."""
    if not sorted_boxes:
        return '', '', 0.0
    red_box = find_red_box(cv_img, sorted_boxes)
    crops = [crop_box(pil_img, b) for b in sorted_boxes]
    raw_texts = [recognize_vietocr(c) for c in crops]
    cleaned = bilingual_post_processor(raw_texts)
    info = extract_universal_product_info_v5(sorted_boxes, cleaned, cv_img.shape, red_box=red_box)
    brand = str(info.get('brand_name', '') or '').strip()
    product = str(info.get('product_name', '') or '').strip()
    if brand in ('', ' '): brand = ''
    if product in ('', ' '): product = ''
    if brand and not _is_valid_pipeline_label(brand, 'brand'):
        brand = ''
    if product and not _is_valid_pipeline_label(product, 'product'):
        product = ''
    prominence = _compute_max_prominence(sorted_boxes, cleaned, cv_img.shape, brand or None)
    if prominence == 0.0:
        prominence = _compute_max_prominence(sorted_boxes, cleaned, cv_img.shape, None)
    if brand:
        brand = normalize_brand_name(brand)
    if product:
        product = normalize_product_name(product)
        product = _strip_brand_prefix(brand, product)
        if _is_generic_product(product):
            product = ''
    return brand, product, float(prominence)


def pipeline_predict_labels(pil_img, cv_img, sorted_boxes):
    """ĐỌC TRỰC TIẾP theo pipeline (wrapper không cần prominence)."""
    b, p, _ = pipeline_predict_with_metrics(pil_img, cv_img, sorted_boxes)
    return b, p


# Gộp các thương hiệu trong BRAND_NORMALIZATION_MAP của pipeline vào registry
for _c in set(BRAND_NORMALIZATION_MAP.values()):
    _register_brand(_c)

print(f'Lõi pipeline sẵn sàng | registry hiện có {len(BRAND_REGISTRY)} alias thương hiệu.')
print("✓ Đã chạy xong cell 4")


def discover_brands_from_train_images(max_n=None):
    """HỌC ĐỘNG (KHÔNG phải học ML đơn thuần): chạy pipeline trên ảnh train,
    chỉ đăng ký brand/product MỚI khi:
      - prominence score pipeline > DYNAMIC_PROMINENCE_THRESHOLD
      - có bằng chứng trong OCR
      - KHÔNG phải từ khóa rác/quảng cáo (negative filtering)
      - lặp >= MIN_BRAND_SUPPORT lần
  Hàm này cần engine OCR (cell 7) nên được gọi ở cell 7."""
    import random
    if max_n is None:
        max_n = TRAIN_DISCOVERY_SAMPLE
    if SKIP_TRAIN_DISCOVERY or TRAIN_IMAGES_DIR is None:
        print('  Bỏ qua học động (đã tắt hoặc không có thư mục ảnh train).')
        return 0
    paths = [p for p in Path(TRAIN_IMAGES_DIR).glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not paths:
        print('  Không tìm thấy ảnh train -> bỏ qua học động.')
        return 0
    random.seed(42); random.shuffle(paths)
    paths = paths[:max_n]
    brand_cand = Counter()
    product_cand = Counter()
    n_high_prom = 0
    _build_product_canon_map()
    for p in tqdm(paths, desc='Học động (ảnh train)'):
        try:
            pil = preprocess(Image.open(str(p)).convert('RGB'))
            cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            boxes = _pipeline_sorted_boxes(cv_img, ocr_det)
            if not boxes:
                continue
            txt, conf = _vietocr_pipeline_ocr(pil, boxes)
            if not str(txt).strip():
                continue
            b, pr, prominence = pipeline_predict_with_metrics(pil, cv_img, boxes)
            if prominence < DYNAMIC_PROMINENCE_THRESHOLD:
                continue
            n_high_prom += 1
            folded_txt = _fold_ascii(txt)
            if b and _is_valid_pipeline_label(b, 'brand'):
                bf = _fold_ascii(b)
                known = bf in BRAND_REGISTRY or bf.replace(' ', '') in BRAND_REGISTRY
                if not known and len(bf.replace(' ', '')) >= MIN_BRAND_ALIAS_LEN:
                    if bf.split()[0] in folded_txt:
                        brand_cand[b] += 1
            if pr and _is_valid_pipeline_label(pr, 'product'):
                pk = _fold_ascii(normalize_product_name(pr))
                canon_map = _build_product_canon_map()
                if pk not in canon_map and _fold_ascii(pr.split()[0]) in folded_txt:
                    product_cand[pr] += 1
        except Exception:
            continue
    added_b, added_p = 0, 0
    for b, c in brand_cand.items():
        if c >= MIN_BRAND_SUPPORT:
            _register_brand(b)
            added_b += 1
    for pr, c in product_cand.items():
        if c >= MIN_BRAND_SUPPORT and _register_discovered_product(pr):
            added_p += 1
    print(f'  Học động: ảnh prominence>{DYNAMIC_PROMINENCE_THRESHOLD}: {n_high_prom}')
    print(f'  Đăng ký mới: +{added_b} brand, +{added_p} product (ứng viên brand={len(brand_cand)}, product={len(product_cand)}).')
    return added_b + added_p


print("✓ Đã chạy xong cell 5")

# ============================================================
# CELL 6 — DỰ ĐOÁN (have_train-first) + EVAL + KIỂM THỬ
# predict_labels: rules -> classifier -> KNN (giống have_train).
# predict_image: ML là nguồn chính; pipeline layout chỉ bù khi prominence cao.
# ============================================================

def _strip_brand_prefix(brand, product):
    """Bỏ token brand bị lặp ở đầu product (vd brand 'Vinamilk', product 'Vinamilk Dielac')."""
    if not (brand and product):
        return product
    bf = _fold_ascii(brand)
    pf = _fold_ascii(product)
    if pf == bf:
        return ''
    if pf.startswith(bf + ' '):
        return product[len(brand):].strip()
    return product


def predict_labels(ocr_text, image_id=None):
    """Dự đoán brand/product từ văn bản OCR (rules -> classifier -> KNN)."""
    ocr_text = str(ocr_text or '').strip()
    if _is_ocr_noise_only(ocr_text) or _is_informational_noise(ocr_text):
        return '', ''

    brand = product = ''

    merged = extract_by_rules(ocr_text)
    if merged:
        merged = post_process_prediction(merged, ocr_text)
        brand, product = split_brand_product(merged)

    if not brand:
        brand = detect_brand_in_ocr(ocr_text)

    if not brand and brand_clf is not None:
        brand = brand_clf.predict(ocr_text)

    if not product:
        product = guess_product_from_ocr(ocr_text, brand)

    if not product and product_clf is not None:
        product = product_clf.predict(ocr_text)

    if (not brand or not product) and brand_knn is not None:
        kb, kp = brand_knn.predict(ocr_text)
        if not brand and kb:
            brand = normalize_brand_name(kb)
        if not product and kp and not _is_generic_product(kp):
            product = normalize_product_name(kp)

    brand, product = reconcile_brand_product(ocr_text, brand, product)
    return brand, product


def _normalize_pair(brand, product):
    brand = normalize_brand_name(brand) if brand else ''
    if product:
        product = normalize_product_name(product)
        product = _strip_brand_prefix(brand, product)
        if _is_generic_product(product) or _is_description_prose(product):
            product = ''
    if _is_glued_product_brand(brand) or _is_description_prose(brand):
        brand = ''
    return brand, product


def predict_image(image_id, ocr_text, ocr_conf, boxes=None):
    """have_train-first: predict_labels trên OCR VietOCR là nguồn chính.
    Pipeline layout chỉ bù khi prominence >= PIPELINE_LAYOUT_MIN_PROMINENCE
    và ML trống hoặc brand/product hiếm/chưa có đủ support trong train."""
    cleaned = clean_social_ocr(ocr_text)
    if _is_ocr_noise_only(cleaned) or _is_informational_noise(cleaned):
        return '', '', False

    brand, product = predict_labels(cleaned, image_id)

    pipe_brand, pipe_product, prominence = '', '', 0.0
    used_pipeline = False

    if boxes:
        try:
            pil_img = load_image(image_id)
            if pil_img is not None:
                pil_img = preprocess(pil_img)
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                pipe_brand, pipe_product, prominence = pipeline_predict_with_metrics(
                    pil_img, cv_img, boxes)
                used_pipeline = True
        except Exception:
            pipe_brand, pipe_product, prominence = '', '', 0.0

    if used_pipeline and prominence >= PIPELINE_LAYOUT_MIN_PROMINENCE:
        pb_ok = pipe_brand and _is_valid_pipeline_label(pipe_brand, 'brand')
        pp_ok = pipe_product and _is_valid_pipeline_label(pipe_product, 'product')

        if pb_ok and not _is_glued_product_brand(pipe_brand) and not _is_description_prose(pipe_brand):
            if not brand or _is_rare_or_new_brand(brand):
                brand = pipe_brand
        if pp_ok and _is_plausible_product_name(pipe_product):
            if (not product or _is_social_caption(product)
                    or _is_rare_or_new_product(product) or _is_generic_product(product)):
                product = pipe_product

    brand, product = reconcile_brand_product(cleaned, brand, product)
    brand, product = _normalize_pair(brand, product)
    return brand, product, used_pipeline


def _token_f1(pred, gt):
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    pt = _fold_ascii(pred).split()
    gt_ = _fold_ascii(gt).split()
    tp = sum(1 for t in pt if t in gt_)
    if tp == 0:
        return 0.0
    p = tp / len(pt)
    r = tp / len(gt_)
    return 2 * p * r / (p + r)


def evaluate_pipeline(df=None, n=500, assumed_cer=0.25):
    if df is None:
        df = _train_frame
    if df is None or not len(df):
        return None
    if len(df) > n:
        df = df.sample(n, random_state=42)
    sb = sp = exact_b = exact_p = ev = 0
    for _, row in df.iterrows():
        ocr = str(row['_ocr']).strip()
        gt_b = str(row['_brand']).strip()
        gt_p = str(row['_product']).strip()
        pb, pp = predict_labels(ocr)
        sb += _token_f1(pb, gt_b)
        sp += _token_f1(pp, gt_p)
        if _fold_ascii(pb) == _fold_ascii(gt_b):
            exact_b += 1
        if _fold_ascii(pp) == _fold_ascii(gt_p):
            exact_p += 1
        ev += 1
    f1b = sb / ev if ev else 0
    f1p = sp / ev if ev else 0
    return {
        'F1_brand': f1b, 'F1_product': f1p,
        'ExactBrand': exact_b / ev if ev else 0,
        'ExactProduct': exact_p / ev if ev else 0,
        'N': ev, 'AssumedCER': assumed_cer,
    }


if _train_frame is not None and len(_train_frame):
    print('Eval %d mẫu train (predict_labels trên OCR train)...' % min(500, len(_train_frame)))
    m = evaluate_pipeline(_train_frame, 500)
    print('F1_brand=%.4f  F1_product=%.4f  ExactBrand=%.4f  ExactProduct=%.4f  N=%d' % (
        m['F1_brand'], m['F1_product'], m['ExactBrand'], m['ExactProduct'], m['N']))
    for _cer in (0.0, 0.15, 0.25):
        _s = 0.4 * m['F1_brand'] + 0.35 * (1 - _cer) + 0.25 * m['F1_product']
        print('  Est.Score (CER=%.2f): %.4f' % (_cer, _s))

_register_brand('Dove')
_P = [
    ('HALONG CANFOCO Pate C\u1ed9t \u0110\u00e8n', 'Ha Long Canfoco', 'Pate'),
    ('ate Jate Cotcen 130 tan thit lon',           None,             'Pate C\u1ed9t \u0110\u00e8n'),
    ('HIGHLANDS COFFEE tra sen vang',              'Highlands Coffee', None),
    ('tiktok viral capcut fyp news',               '',               ''),
    ('HiPP Combiotic organic milk',                'HiPP',           None),
    ('Nestl\u00e9 NAN OPTIpro 0-6',                'Nestl\u00e9',    'NAN'),
    ('Dove Smoothie t\u1ea9y da ch\u1ebft',        'Dove',           None),
    ('Nescafe Gold Blend huong chat',               'Nescafe',        None),
    ('Panasonic GH5 mirrorless camera',            'Panasonic',      None),
    ('La Roche Posay Effaclar gel',                'La Roche-Posay', None),
    ('Vinamilk Dielac so 1',                        'Vinamilk',       'Dielac'),
    ('Trả lời hình luận của Thúy nêng chốt liền rẻ quá Abbott Ped Complete Ion Pediasure Úc 850g',
                                                   'Abbott',         'PediaSure'),
    ('Similac Totalladongsua cong thuc cao cap cua Abbott, duoc phat trien nhâm ho tro',
                                                   'Abbott',         'Similac'),
    ('1964 Không chi là Quốc Tế Thiếu Nhi Vitamin A EM SỮA 14 THÁNG các mom đừng quên ngày mai',
                                                   '',               ''),
]
print('Kiểm thử nhanh predict_labels:')
ok_n = 0
for txt, eb, ec in _P:
    b, p = predict_labels(txt)
    combined = (b + ' ' + p).strip()
    ok = True
    if eb is not None:
        ok = ok and ((eb == '' and b == '') or (eb != '' and eb.lower() in b.lower()))
    if ec is not None:
        ok = ok and ((ec == '' and combined == '') or (ec != '' and ec.lower() in combined.lower()))
    if p and _is_generic_product(p):
        ok = False
    if ok:
        ok_n += 1
    print('  [%s] %-40s => brand=%-16s product=%s' % (
        'OK' if ok else 'FAIL', txt[:40], repr(b)[:16], repr(p)[:24]))
print('  Kết quả: %d/%d' % (ok_n, len(_P)))
print("✓ Đã chạy xong cell 6")

# ============================================================
# CELL 7 — ENGINE OCR: Paddle det ONLY + VietOCR (không latin rec)
# Bound box: PP-OCRv4 det. Đọc chữ: VietOCR từng crop.
# ============================================================
import os
from pathlib import Path
import re
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

def _patch_numpy_for_imgaug():
    """PaddleOCR 2.7.3 -> imgaug dùng np.sctypes (đã bỏ ở numpy 2.x)."""
    import numpy as _np
    if not hasattr(_np, 'sctypes'):
        _np.sctypes = {
            'float': [_np.float16, _np.float32, _np.float64],
            'int': [_np.int8, _np.int16, _np.int32, _np.int64],
            'uint': [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
            'complex': [_np.complex64, _np.complex128],
            'others': [bool, _np.bytes_, _np.str_, _np.object_],
        }

_patch_numpy_for_imgaug()

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

_pp_det = None


def _det_model_dir():
    return Path.home() / '.paddleocr' / 'whl' / 'det' / 'ch' / 'ch_PP-OCRv4_det_infer'


def _paddle_model_ready(d):
    d = Path(d)
    return (d / 'inference.pdmodel').is_file() and (d / 'inference.pdiparams').is_file()


def _get_maybe_download():
    """ppocr chỉ có sau khi paddleocr được import — thử nhiều đường dẫn."""
    import importlib
    import paddleocr  # noqa: F401 — đưa ppocr vào sys.path trên một số bản cài
    for mod_name in ('paddleocr.ppocr.utils.network', 'ppocr.utils.network'):
        try:
            return importlib.import_module(mod_name).maybe_download
        except ModuleNotFoundError:
            continue
    raise RuntimeError(
        'Không tìm thấy ppocr.utils.network. Chạy lại cell 1 (paddleocr==2.7.3) rồi Restart session.'
    )


def _prefetch_det_model():
    det_dir = _det_model_dir()
    url = 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar'
    if not _paddle_model_ready(det_dir):
        print('Tải trước model det PP-OCRv4...')
        _get_maybe_download()(str(det_dir), url)
        print('  đã tải det')
    return det_dir


def _make_paddle_det():
    """Det PP-OCRv4 — chỉ text_detector, không rec latin."""
    _patch_numpy_for_imgaug()
    # Tắt IR-optim của Paddle inference (PaddleOCR hard-code bật ở utility.py).
    # Trên CPU không có AVX (vd Streamlit Cloud), pass SelfAttentionFusePass khi
    # tối ưu đồ thị gây 'Illegal instruction' (SIGILL) làm chết tiến trình NGAY khi
    # khởi tạo predictor. Tắt ir_optim bỏ qua các fuse-pass đó; OCR vẫn chạy bình
    # thường. Kaggle/máy có AVX không bị ảnh hưởng.
    try:
        from paddle import inference as _pi
        if not getattr(_pi.create_predictor, '_tanahi_noiropt', False):
            _orig_cp = _pi.create_predictor

            def _cp_noiropt(config, *a, **kw):
                try:
                    config.switch_ir_optim(False)
                except Exception:
                    pass
                return _orig_cp(config, *a, **kw)

            _cp_noiropt._tanahi_noiropt = True
            _pi.create_predictor = _cp_noiropt
    except Exception:
        pass
    import paddleocr  # phải import trước prefetch / ppocr
    det_dir = None
    try:
        det_dir = _prefetch_det_model()
    except Exception as e:
        print(f'  Prefetch det bỏ qua ({e}) — PaddleOCR sẽ tự tải model khi khởi tạo.')

    from paddleocr import PaddleOCR
    # Cấu hình rõ ràng chỉ sử dụng Detector, bỏ qua Rec và Cls để tránh tải thêm file thừa
    return PaddleOCR(
        ocr_version='PP-OCRv4',
        det=True,
        rec=False,         # Tắt bộ nhận diện của Paddle
        cls=False,         # Tắt bộ phân loại góc của Paddle
        use_angle_cls=False,
        det_model_dir=str(det_dir) if det_dir else None,  # Chỉ định thư mục chứa det model đã tải
        lang='ch',         # Đặt 'ch' khớp với thư mục mô hình det tiếng Trung/Đa ngôn ngữ của PP-OCRv4
        use_gpu=False,
        show_log=False,
        enable_mkldnn=False,
    )


def preprocess(img, max_dim=None, contrast=None):
    """Tiền xử lý ảnh trước det/VietOCR — resize + contrast + sharpness."""
    max_dim = max_dim if max_dim is not None else OCR_PREPROCESS_MAX_DIM
    contrast = contrast if contrast is not None else OCR_PREPROCESS_CONTRAST
    img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(float(contrast))
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img.filter(ImageFilter.SHARPEN)


def _sort_boxes_reading_order(boxes):
    """Sắp box theo dòng (top->bottom) rồi trái->phải — ocr_text đọc tự nhiên hơn."""
    if not boxes:
        return []
    row_tol = float(OCR_READING_ORDER_ROW_TOL)
    items = []
    for box in boxes:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        cy = sum(ys) / len(ys)
        cx = sum(xs) / len(xs)
        h = max(ys) - min(ys) or 1.0
        items.append((box, cy, cx, h))
    items.sort(key=lambda t: t[1])
    rows = []
    for box, cy, cx, h in items:
        tol = max(10.0, h * row_tol)
        placed = False
        for row in rows:
            if abs(cy - row['cy']) <= tol:
                row['items'].append((cx, box))
                row['cy'] = (row['cy'] + cy) / 2.0
                placed = True
                break
        if not placed:
            rows.append({'cy': cy, 'items': [(cx, box)]})
    rows.sort(key=lambda r: r['cy'])
    ordered = []
    for row in rows:
        row['items'].sort(key=lambda t: t[0])
        ordered.extend(b for _, b in row['items'])
    return ordered


def postprocess_ocr(text):
    """Hậu xử lý ocr_text: chuẩn Unicode, typo map, train catalog, dedup token."""
    text = str(text or '').strip()
    if not text:
        return ''
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text)
    text = apply_ocr_typo_map(text)
    text = correct_ocr_from_train_catalog(text)
    tokens = text.split()
    if not tokens:
        return ''
    deduped = [tokens[0]]
    for tok in tokens[1:]:
        if tok.lower() != deduped[-1].lower():
            deduped.append(tok)
    return ' '.join(deduped)


def _pipeline_sorted_boxes(cv_img, det_engine):
    """Bound khung chữ: text_detector + lọc theo OCR_BOX_MIN_AREA_RATIO."""
    try:
        dt_boxes, _ = det_engine.text_detector(cv_img)
        raw_boxes = dt_boxes.tolist() if dt_boxes is not None else []
    except Exception:
        return []
    img_h, img_w = cv_img.shape[:2]
    img_area = img_h * img_w
    min_ratio = OCR_BOX_MIN_AREA_RATIO
    valid_boxes = [
        box for box in raw_boxes
        if ((max(p[0] for p in box) - min(p[0] for p in box))
            * (max(p[1] for p in box) - min(p[1] for p in box)) / img_area) >= min_ratio
    ]
    return _sort_boxes_reading_order(valid_boxes)


def _vietocr_pipeline_ocr(pil_img, boxes):
    """VietOCR từng crop + bilingual_post_processor — nguồn OCR chính cho predict_labels."""
    if not boxes:
        return '', 0.0
    raw_texts = []
    for box in boxes:
        crop_np = crop_box(pil_img, box)
        if crop_np.size == 0:
            raw_texts.append('')
            continue
        raw_texts.append(recognize_vietocr(crop_np))
    cleaned = bilingual_post_processor(raw_texts)
    ocr_text = postprocess_ocr(' '.join(t for t in cleaned if str(t).strip()))
    return ocr_text, (0.85 if ocr_text.strip() else 0.0)


def _process_one_image(image_id):
    """Det (Paddle) + VietOCR trên luồng chính."""
    path = IMAGE_PATH_INDEX.get(str(image_id))
    if not path or not os.path.isfile(path):
        return image_id, '', 0.0, []
    try:
        pil = preprocess(Image.open(path).convert('RGB'))
    except Exception:
        return image_id, '', 0.0, []
    cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    boxes = _pipeline_sorted_boxes(cv_img, ocr_det)
    ocr_text, conf = _vietocr_pipeline_ocr(pil, boxes)
    return image_id, ocr_text, conf, boxes


def _init_ocr_process():
    global _pp_det
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    _patch_numpy_for_imgaug()
    _pp_det = _make_paddle_det()


def _det_image_process(image_id, image_path):
    global _pp_det
    if not image_path or not os.path.isfile(image_path):
        return image_id, []
    try:
        pil = preprocess(Image.open(image_path).convert('RGB'))
    except Exception:
        return image_id, []
    cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    boxes = _pipeline_sorted_boxes(cv_img, _pp_det)
    return image_id, boxes


def load_image(image_id):
    path = IMAGE_PATH_INDEX.get(str(image_id))
    if path is None or not Path(path).exists():
        return None
    try:
        return Image.open(str(path)).convert('RGB')
    except Exception:
        return None


print('Đang nạp Paddle det PP-OCRv4 (chỉ bound box, không latin rec)...')
ocr_det = _make_paddle_det()
print('Det sẵn sàng | OCR text: VietOCR trên crop.')

print('Đang nạp VietOCR (vgg_seq2seq, CPU)...')
torch.set_num_threads(max(1, CPU_THREADS // 2))
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
_vcfg = Cfg.load_config_from_name('vgg_seq2seq')
_vcfg['device'] = 'cpu'
vietocr_predictor = Predictor(_vcfg)
print('VietOCR sẵn sàng.')

if _RUN_BATCH:
    print('\nChạy học động (thu thập brand mới từ ảnh train)...')
    _n_new = discover_brands_from_train_images()
    if _n_new:
        print('Huấn luyện lại mô hình sau học động...')
        retrain_predictors()

    print('\nKiểm thử nhanh trên ảnh test đầu tiên:')
    _fid = test_df['image_id'].iloc[0] if len(test_df) else None
    _img = load_image(_fid) if _fid is not None else None
    if _img is not None:
        _pil = preprocess(_img)
        _cv = cv2.cvtColor(np.array(_pil), cv2.COLOR_RGB2BGR)
        _boxes = _pipeline_sorted_boxes(_cv, ocr_det)
        _txt, _conf = _vietocr_pipeline_ocr(_pil, _boxes)
        _b, _p = predict_labels(clean_social_ocr(_txt))
        print(f'  image_id : {_fid}')
        print(f'  khung    : {len(_boxes)} box (Paddle det)')
        print(f'  ocr_text : {_txt[:80]}{"..." if len(_txt) > 80 else ""}')
        print(f'  predict  : brand={repr(_b)[:20]} product={repr(_p)[:30]}')
    else:
        print(f'  Cảnh báo: không tìm thấy ảnh {_fid}')
    print("✓ Đã chạy xong cell 7")


# ============================================================
# CELL 7b — MEDIA: GIF / VIDEO / ẢNH ĐỘNG → 1 FRAME CHO OCR
# Chạy sau cell 7. Ảnh tĩnh giữ nguyên; GIF/WebP nhiều frame / video
# chỉ lấy 1 frame nét nhất (Laplacian) trong vài mẫu rải đều.
# ============================================================

import numpy as np

MEDIA_FRAME_SAMPLES = int(globals().get('MEDIA_FRAME_SAMPLES', 11))   # mẫu frame GIF/video
_frame_cache = {}         # path -> PIL.Image RGB (tránh decode lại video)


def _media_ext(path):
    return Path(str(path)).suffix.lower()


def is_video_path(path):
    exts = globals().get('VIDEO_EXTS', ('.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v', '.mpeg', '.mpg'))
    return _media_ext(path) in exts


def is_animated_image_path(path):
    exts = globals().get('ANIMATED_IMAGE_EXTS', ('.gif', '.webp', '.apng'))
    return _media_ext(path) in exts


def _pil_n_frames(path):
    try:
        with Image.open(str(path)) as im:
            return int(getattr(im, 'n_frames', 1) or 1)
    except Exception:
        return 1


def needs_single_frame_extraction(path):
    """True nếu file là video hoặc ảnh có >1 frame."""
    path = str(path)
    if is_video_path(path):
        return True
    if is_animated_image_path(path) or _pil_n_frames(path) > 1:
        return True
    return False


def _sharpness_score(pil_rgb):
    """Điểm độ nét (variance of Laplacian) — frame cao hơn thường đọc OCR tốt hơn."""
    gray = np.asarray(pil_rgb.convert('L'), dtype=np.float32)
    if gray.size < 16:
        return 0.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return float(gx * gx + gy * gy)


def _pick_best_frame(frames):
    if not frames:
        return None
    return max(frames, key=lambda t: t[0])[1]


def _sample_indices(n, k):
    if n <= 1:
        return [0]
    k = max(1, min(int(k), n))
    if k == 1:
        return [n // 2]
    step = max(1, (n - 1) // (k - 1))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return sorted(set(idx))[:k]


def _frame_from_image_sequence(path):
    """GIF / WebP / PNG động — chọn 1 frame nét nhất."""
    path = str(path)
    candidates = []
    with Image.open(path) as im:
        n = int(getattr(im, 'n_frames', 1) or 1)
        for i in _sample_indices(n, MEDIA_FRAME_SAMPLES):
            try:
                im.seek(i)
            except EOFError:
                break
            frame = im.convert('RGB')
            candidates.append((_sharpness_score(frame), frame.copy()))
    best = _pick_best_frame(candidates)
    if best is not None:
        return best
    return Image.open(path).convert('RGB')


def _frame_from_video(path):
    """Video mp4/webm/... — lấy mẫu vài frame, giữ frame nét nhất."""
    path = str(path)
    candidates = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f'Không mở được video: {path}')
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            n = 1
        for i in _sample_indices(n, MEDIA_FRAME_SAMPLES):
            if n > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            candidates.append((_sharpness_score(pil), pil))
    finally:
        cap.release()
    best = _pick_best_frame(candidates)
    if best is not None:
        return best
    raise OSError(f'Không đọc được frame từ video: {path}')


def extract_single_frame(path):
    """Trích đúng 1 frame RGB từ ảnh động hoặc video."""
    path = str(path)
    if path in _frame_cache:
        return _frame_cache[path].copy()
    if is_video_path(path):
        img = _frame_from_video(path)
    elif needs_single_frame_extraction(path):
        img = _frame_from_image_sequence(path)
    else:
        img = Image.open(path).convert('RGB')
    _frame_cache[path] = img
    return img.copy()


def open_media_for_ocr(path):
    """Mở media test: ảnh tĩnh hoặc 1 frame đại diện nếu động/video."""
    path = str(path)
    if needs_single_frame_extraction(path):
        return extract_single_frame(path)
    return Image.open(path).convert('RGB')


def _preprocess_path_for_ocr(path):
    return preprocess(open_media_for_ocr(path))


# --- Gắn vào pipeline OCR (cell 7) ---
def _process_one_image(image_id):
    """Det (Paddle) + VietOCR — dùng 1 frame nếu GIF/video."""
    path = IMAGE_PATH_INDEX.get(str(image_id))
    if not path or not os.path.isfile(path):
        return image_id, '', 0.0, []
    try:
        pil = _preprocess_path_for_ocr(path)
    except Exception:
        return image_id, '', 0.0, []
    cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    boxes = _pipeline_sorted_boxes(cv_img, ocr_det)
    ocr_text, conf = _vietocr_pipeline_ocr(pil, boxes)
    return image_id, ocr_text, conf, boxes


def _det_image_process(image_id, image_path):
    global _pp_det
    if not image_path or not os.path.isfile(image_path):
        return image_id, []
    try:
        pil = _preprocess_path_for_ocr(image_path)
    except Exception:
        return image_id, []
    cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    boxes = _pipeline_sorted_boxes(cv_img, _pp_det)
    return image_id, boxes


def load_image(image_id):
    path = IMAGE_PATH_INDEX.get(str(image_id))
    if path is None or not Path(path).exists():
        return None
    try:
        return open_media_for_ocr(path)
    except Exception:
        return None


# Thống kê nhanh
_n_media = sum(1 for p in IMAGE_PATH_INDEX.values() if needs_single_frame_extraction(p))
print(f'Media động/video trong test: {_n_media:,} file (OCR dùng 1 frame / file)')
print("✓ Đã chạy xong cell 7b — pipeline OCR dùng open_media_for_ocr()")

# ============================================================
# CELL 8 — CHẠY OCR THEO LÔ + DỰ ĐOÁN LAI + LƯU CHECKPOINT
# Det + VietOCR trên luồng chính (ocr_det cell 7). Loky worker det lỗi im lặng trên Kaggle.
# ============================================================

SAVE_EVERY = 50
RESET_CHECKPOINT = False   # True = xóa checkpoint cũ, chạy lại từ đầu; xong thì đổi False

if _RUN_BATCH:
    if RESET_CHECKPOINT and CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()
        print(f'Đã xóa checkpoint cũ: {CHECKPOINT_CSV}')

    # Khôi phục từ checkpoint nếu có
    results, done_ids = [], set()
    if CHECKPOINT_CSV.exists():
        ckpt = pd.read_csv(CHECKPOINT_CSV, keep_default_na=False)
        done_ids = set(ckpt['image_id'])
        results = ckpt.to_dict('records')
        print(f'Tiếp tục từ checkpoint: {len(done_ids):,} ảnh đã xong.')
    else:
        print('Bắt đầu chạy mới.')

    pending = [i for i in test_df['image_id'] if i not in done_ids]
    print(f'Còn lại: {len(pending):,} ảnh | det+VietOCR tuần tự (luồng chính)')

    errors = 0
    n_pipeline = 0
    n_viet_read = 0
    t0 = time.perf_counter()

    for batch_start in tqdm(range(0, len(pending), SAVE_EVERY), desc='OCR theo lô'):
        batch = pending[batch_start: batch_start + SAVE_EVERY]

        for image_id in batch:
            try:
                image_id, ocr_text, conf, boxes = _process_one_image(image_id)
                if ocr_text.strip():
                    n_viet_read += 1
                brand, product, used = predict_image(image_id, ocr_text, conf, boxes)
                if used:
                    n_pipeline += 1
                results.append({
                    'image_id': image_id,
                    'ocr_text': ocr_text,
                    'brand_name': brand,
                    'product_name': product,
                })
            except Exception:
                results.append({'image_id': image_id, 'ocr_text': '',
                                'brand_name': '', 'product_name': ''})
                errors += 1

        pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8')

    elapsed = time.perf_counter() - t0
    n_ocr   = sum(1 for r in results if str(r.get('ocr_text', '')).strip())
    n_brand = sum(1 for r in results if str(r.get('brand_name', '')).strip())
    n_prod  = sum(1 for r in results if str(r.get('product_name', '')).strip())
    print(f'\nHoàn tất {len(results)} ảnh trong {elapsed/60:.1f} phút ({elapsed/max(len(results),1):.2f}s/ảnh)')
    print(f'OCR có chữ: {n_ocr} | Có brand: {n_brand} | Có product: {n_prod}')
    print(f'VietOCR có chữ: {n_viet_read} ảnh | Có box det (layout bù): {n_pipeline} | Lỗi: {errors}')
    print("✓ Đã chạy xong cell 8")


def predict_from_text(ocr_text: str) -> tuple[str, str]:
    """Extract brand + product from raw OCR text (no image)."""
    cleaned = clean_social_ocr(ocr_text)
    brand, product = predict_labels(cleaned)
    brand, product = _normalize_pair(brand, product)
    return brand, product


def predict_from_image(
    img: Image.Image,
    min_conf: float = 0.35,
    *,
    include_timing: bool = True,
) -> dict[str, Any]:
    """
    Main entry point for Streamlit + batch submission.
    """
    t0 = time.perf_counter()

    t_ocr = time.perf_counter()
    # Preprocess image
    pil_img = preprocess(img)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # Det boxes
    boxes = _pipeline_sorted_boxes(cv_img, ocr_det)
    # Rec text
    ocr_text, conf = _vietocr_pipeline_ocr(pil_img, boxes)
    ocr_ms = (time.perf_counter() - t_ocr) * 1000

    t_extract = time.perf_counter()
    # Predict brand & product
    cleaned = clean_social_ocr(ocr_text)
    if _is_ocr_noise_only(cleaned) or _is_informational_noise(cleaned):
        brand, product = '', ''
    else:
        brand, product = predict_labels(cleaned)

        pipe_brand, pipe_product, prominence = '', '', 0.0
        used_pipeline = False

        if boxes:
            try:
                pipe_brand, pipe_product, prominence = pipeline_predict_with_metrics(
                    pil_img, cv_img, boxes)
                used_pipeline = True
            except Exception:
                pipe_brand, pipe_product, prominence = '', '', 0.0

        if used_pipeline and prominence >= PIPELINE_LAYOUT_MIN_PROMINENCE:
            pb_ok = pipe_brand and _is_valid_pipeline_label(pipe_brand, 'brand')
            pp_ok = pipe_product and _is_valid_pipeline_label(pipe_product, 'product')

            if pb_ok and not _is_glued_product_brand(pipe_brand) and not _is_description_prose(pipe_brand):
                if not brand or _is_rare_or_new_brand(brand):
                    brand = pipe_brand
            if pp_ok and _is_plausible_product_name(pipe_product):
                if (not product or _is_social_caption(product)
                        or _is_rare_or_new_product(product) or _is_generic_product(product)):
                    product = pipe_product

        brand, product = reconcile_brand_product(cleaned, brand, product)
        brand, product = _normalize_pair(brand, product)

    extract_ms = (time.perf_counter() - t_extract) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    result = {
        "ocr_text": ocr_text,
        "brand_name": brand,
        "product_name": product,
    }
    if include_timing:
        result["timing_ms"] = {
            "ocr": round(ocr_ms, 1),
            "extract": round(extract_ms, 1),
            "total": round(total_ms, 1),
        }
    return result
