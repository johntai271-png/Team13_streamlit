from __future__ import annotations

import pickle
from typing import Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


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
