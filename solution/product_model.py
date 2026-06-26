from __future__ import annotations

import re
import unicodedata
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Callable
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from solution.brand_rules import (
    train_labels_df,
    OCR_COL,
    HAS_BRAND_COL,
    BRAND_COL,
    PRODUCT_COL,
    MIN_BRAND_SUPPORT,
    MIN_BRAND_ALIAS_LEN,
    BRAND_REGISTRY,
    _PRODUCT_CANON_MAP,
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
    _brand_support_count,
    _product_support_count,
    _BRAND_SUPPORT_COUNTS,
    _PRODUCT_SUPPORT_COUNTS,
    _build_training_frame,
    _train_frame,
)

# ---------- COMPATIBILITY ProductPredictor FOR BENCHMARK ----------
def _clean(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


class ProductPredictor:
    def __init__(
        self,
        min_class_count: int = 3,
        prob_threshold: float = 0.60,
        max_features: int = 3000,
    ):
        self.min_class_count = min_class_count
        self.prob_threshold = prob_threshold
        self.max_features = max_features
        self._has_clf: Pipeline | None = None
        self._prod_clf: Pipeline | None = None
        self._n_train = 0
        self._n_classes = 0

    def fit(
        self,
        train_labels: pd.DataFrame,
        rule_fn: Callable[[str], str],
    ) -> "ProductPredictor":
        df = train_labels.copy()
        df["ocr_text"] = df["ocr_text"].map(_clean)
        df["product_name"] = df["product_name"].map(_clean)

        self._has_clf = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 4),
                        max_features=self.max_features,
                        min_df=2,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(max_iter=400, class_weight="balanced"),
                ),
            ]
        )
        has_label = (df["product_name"] != "").astype(int)
        self._has_clf.fit(df["ocr_text"], has_label)

        pos = df[(df["ocr_text"] != "") & (df["product_name"] != "")]
        counts = pos["product_name"].value_counts()
        keep = counts[counts >= self.min_class_count].index
        pos = pos[pos["product_name"].isin(keep)]

        self._prod_clf = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 4),
                        max_features=self.max_features,
                        min_df=2,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(max_iter=400, class_weight="balanced"),
                ),
            ]
        )
        if len(pos):
            self._prod_clf.fit(pos["ocr_text"], pos["product_name"])

        self._rule_fn = rule_fn
        self._n_train = len(df)
        self._n_classes = pos["product_name"].nunique() if len(pos) else 0
        return self

    def predict(self, ocr_text: str) -> str:
        ocr_text = _clean(ocr_text)
        if not ocr_text:
            return ""

        ruled = self._rule_fn(ocr_text)
        if ruled:
            return ruled

        if self._has_clf is None or self._prod_clf is None:
            return ""

        proba = self._has_clf.predict_proba([ocr_text])[0]
        has_idx = list(self._has_clf.classes_).index(1) if 1 in self._has_clf.classes_ else -1
        if has_idx < 0 or proba[has_idx] < self.prob_threshold:
            return ""

        return str(self._prod_clf.predict([ocr_text])[0])

    def model_size_mb(self) -> float:
        total = 0
        for clf in (self._has_clf, self._prod_clf):
            if clf is not None:
                total += len(pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL))
        return total / (1024 * 1024)

    def summary(self) -> str:
        return (
            f"ProductPredictor(train={self._n_train}, classes={self._n_classes}, "
            f"features<={self.max_features}, size≈{self.model_size_mb():.2f}MB, "
            f"prob_threshold={self.prob_threshold})"
        )


# ---------- NEW MODEL CLASSIFIERS FROM TANAHIDEMO VER 2 ----------

def _is_valid_pipeline_label(name, label_type):
    if label_type == 'brand':
        return _is_plausible_brand_name(name)
    elif label_type == 'product':
        return _is_plausible_product_name(name)
    return False



def _pipe(mf=6000, C=1.5):
    return Pipeline([
        ('tf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=mf, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=800, class_weight='balanced', C=C, solver='lbfgs')),
    ])


class BrandClassifier:
    """Brand nặng 0.4 -> ưu tiên evidence (alias trong OCR), classifier là fallback bảo thủ."""
    def __init__(self, min_cls=MIN_BRAND_SUPPORT, thr=0.62):
        self.min_cls = min_cls; self.thr = thr; self._clf = None; self.fitted = False
    def fit(self, df):
        pos = df[df['_brand'] != '']
        vc = pos['_brand'].value_counts()
        keep = vc[vc >= self.min_cls].index
        pos = pos[pos['_brand'].isin(keep)]
        if pos['_brand'].nunique() >= 2:
            self._clf = _pipe(mf=5000).fit(pos['_ocr'], pos['_brand'])
            self.fitted = True
        print('  BrandClassifier: %d dòng, %d brand (support>=%d).' % (len(pos), len(keep), self.min_cls))
        return self
    def predict(self, ocr_text):
        ocr_text = str(ocr_text or '').strip()
        if not ocr_text:
            return ''
        b = detect_brand_in_ocr(ocr_text)
        if b:
            return b
        if self.fitted and self._clf is not None:
            try:
                proba = self._clf.predict_proba([ocr_text])[0]
                i = int(proba.argmax())
                if proba[i] >= self.thr:
                    return normalize_brand_name(str(self._clf.classes_[i]))
            except Exception:
                pass
        return ''


class ProductClassifier:
    """Product nặng 0.25 -> cho phép đoán khi đủ tin cậy, có chặn token generic."""
    def __init__(self, min_cls=2, thr=0.45, gate_thr=0.40):
        self.min_cls = min_cls; self.thr = thr; self.gate_thr = gate_thr
        self._gate = None; self._clf = None; self.fitted = False
    def fit(self, df):
        yg = (df['_product'] != '').astype(int)
        if yg.nunique() >= 2:
            self._gate = _pipe(mf=6000).fit(df['_ocr'], yg)
        pos = df[df['_product'] != '']
        vc = pos['_product'].value_counts()
        pos = pos[pos['_product'].isin(vc[vc >= self.min_cls].index)]
        if pos['_product'].nunique() >= 2:
            self._clf = _pipe(mf=6000).fit(pos['_ocr'], pos['_product'])
            self.fitted = True
        print('  ProductClassifier: %d positive, %d product (support>=%d).' % (
            int(yg.sum()), pos['_product'].nunique(), self.min_cls))
        return self
    def predict(self, ocr_text):
        ocr_text = str(ocr_text or '').strip()
        if not ocr_text or not self.fitted:
            return ''
        try:
            if self._gate is not None:
                gp = self._gate.predict_proba([ocr_text])[0]
                cl = list(self._gate.classes_)
                if 1 in cl and gp[cl.index(1)] < self.gate_thr:
                    return ''
            proba = self._clf.predict_proba([ocr_text])[0]
            i = int(proba.argmax())
            if proba[i] < self.thr:
                return ''
            p = normalize_product_name(str(self._clf.classes_[i]))
            return '' if _is_generic_product(p) else p
        except Exception:
            return ''


class BrandKNN:
    """Fallback similarity: bỏ phiếu brand & product từ các mẫu train gần nhất."""
    def __init__(self, k=5, thr=0.55):
        self.k = k; self.thr = thr; self._tf = None; self._mat = None
        self._brands = []; self._products = []; self.fitted = False
    def fit(self, df):
        self._tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=6000, sublinear_tf=True)
        self._mat = self._tf.fit_transform(df['_ocr'])
        self._brands = df['_brand'].tolist()
        self._products = df['_product'].tolist()
        self.fitted = True
        print('  BrandKNN: %d vector.' % len(self._brands))
        return self
    def _vote(self, labels, idx, sims):
        votes = [labels[i] for i in idx if sims[i] >= self.thr and labels[i]]
        return Counter(votes).most_common(1)[0][0] if votes else ''
    def predict(self, ocr_text):
        if not self.fitted or not ocr_text or len(str(ocr_text).strip()) < 5:
            return '', ''
        try:
            sv = self._tf.transform([ocr_text])
            sims = cosine_similarity(sv, self._mat).flatten()
            idx = sims.argsort()[::-1][:self.k]
            return self._vote(self._brands, idx, sims), self._vote(self._products, idx, sims)
        except Exception:
            return '', ''


# ---- Huấn luyện cơ bản từ nhãn CSV ----
brand_clf = product_clf = brand_knn = None


def _is_rare_or_new_brand(brand):
    """Brand xuất hiện ít hoặc chưa từng thấy trong train (< MIN_BRAND_SUPPORT)."""
    if not brand:
        return False
    b = normalize_brand_name(brand)
    if not b:
        return False
    f = _fold_ascii(b)
    if f in BRAND_REGISTRY or f.replace(' ', '') in BRAND_REGISTRY:
        if _brand_support_count(b) >= MIN_BRAND_SUPPORT:
            return False
    return _brand_support_count(b) < MIN_BRAND_SUPPORT


def _is_rare_or_new_product(product):
    if not product:
        return False
    p = normalize_product_name(product)
    return _product_support_count(p) < MIN_BRAND_SUPPORT


def _register_discovered_product(name):
    """Ghi product mới vào bản đồ chuẩn hoá (sau khi qua ngưỡng prominence + lọc rác)."""
    global _PRODUCT_CANON_MAP
    name = unicodedata.normalize('NFC', str(name).strip())
    if not name or not _is_valid_pipeline_label(name, 'product'):
        return False
    _build_product_canon_map()
    key = _fold_ascii(name)
    if key not in _PRODUCT_CANON_MAP:
        _PRODUCT_CANON_MAP[key] = name
        return True
    return False

if _train_frame is not None and len(_train_frame):
    _added = build_brand_knowledge_from_train()
    print('Học từ nhãn CSV: +%d brand từ train (registry=%d alias).' % (_added, len(BRAND_REGISTRY)))
    print('Đang huấn luyện BrandClassifier...')
    brand_clf = BrandClassifier().fit(_train_frame)
    print('Đang huấn luyện ProductClassifier...')
    product_clf = ProductClassifier().fit(_train_frame)
    print('Đang huấn luyện BrandKNN...')
    brand_knn = BrandKNN().fit(_train_frame)
    print('Bộ dự đoán đã sẵn sàng.')
else:
    print('Không có train data -> tắt classifier (chỉ dùng rules).')


def retrain_predictors():
    """Huấn luyện lại 3 mô hình sau khi registry được mở rộng (học động)."""
    global brand_clf, product_clf, brand_knn, _train_frame
    _train_frame = _build_training_frame()
    if _train_frame is None or not len(_train_frame):
        return
    brand_clf = BrandClassifier().fit(_train_frame)
    product_clf = ProductClassifier().fit(_train_frame)
    brand_knn = BrandKNN().fit(_train_frame)
