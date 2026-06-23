"""
SMCE Private Test Metric (The 2nd URA Hackathon 2026)

Host notes:
- Private set: 1,202 rows (443 TikTok hard + 759 Private DS).
- Submission must have columns: image_id, ocr_text, brand_name, product_name.
- Empty fields in CSV should be a single space " " (stripped before scoring).
- Usage column (if present on solution) is ignored by Kaggle before score() is called.

Score = 0.4 * F1_brand + 0.35 * (1 - CER) + 0.25 * F1_product

Constraints (Kaggle):
- Function named score(solution, submission, row_id_column_name) -> float
- First three arguments unchanged; all arguments type-annotated
- Returns a single finite non-null float
"""

import pandas as pd

W_BRAND = 0.4
W_OCR = 0.35
W_PRODUCT = 0.25


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def _clean(val) -> str:
    return "" if pd.isna(val) else str(val).strip()


def _token_f1(gt: str, pred: str) -> float:
    gt = _clean(gt)
    pred = _clean(pred)
    if not gt and not pred:
        return 1.0
    gt_tokens = set(gt.lower().split())
    pred_tokens = set(pred.lower().split())
    if not gt_tokens or not pred_tokens:
        return 0.0
    common = gt_tokens & pred_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _cer(gt: str, pred: str) -> float:
    gt = _clean(gt)
    pred = _clean(pred)
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    m, n = len(gt), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if gt[i - 1] == pred[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return min(dp[n] / len(gt), 1.0)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Private composite score for OCR + brand + product extraction.

    Combines token-level F1 on brand_name and product_name (case-insensitive) with
    character error rate on ocr_text (case-sensitive). Higher is better; range [0, 1].

    Weights: brand 40%, OCR 35%, product 25%.

    >>> import pandas as pd
    >>> sol = pd.DataFrame({
    ...     'image_id': ['img_0001'],
    ...     'ocr_text': ['Nestle NAN OPTIPRO'],
    ...     'brand_name': ['Nestle'],
    ...     'product_name': ['Nestle NAN OPTIPRO'],
    ... })
    >>> score(sol.copy(), sol.copy(), 'image_id')
    1.0
    >>> empty = pd.DataFrame({
    ...     'image_id': ['img_0001'],
    ...     'ocr_text': [''],
    ...     'brand_name': [''],
    ...     'product_name': [''],
    ... })
    >>> score(empty.copy(), empty.copy(), 'image_id')
    1.0
    """
    required = {"ocr_text", "brand_name", "product_name"}

    if row_id_column_name not in solution.columns:
        raise ParticipantVisibleError(f"Solution missing id column '{row_id_column_name}'.")
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission missing id column '{row_id_column_name}'.")
    if not required.issubset(solution.columns):
        raise ParticipantVisibleError(f"Solution must contain columns: {required}")
    if not required.issubset(submission.columns):
        raise ParticipantVisibleError(
            f"Submission must contain columns: {required}. Got: {list(submission.columns)}"
        )

    for col in required:
        if not pd.api.types.is_string_dtype(solution[col]) and not pd.api.types.is_object_dtype(solution[col]):
            raise ParticipantVisibleError(f"Solution column {col} must be text.")
        if not pd.api.types.is_string_dtype(submission[col]) and not pd.api.types.is_object_dtype(submission[col]):
            raise ParticipantVisibleError(f"Submission column {col} must be text.")

    if submission[row_id_column_name].duplicated().any():
        raise ParticipantVisibleError("Duplicate image_id in submission.")

    sol_ids = set(solution[row_id_column_name])
    sub_ids = set(submission[row_id_column_name])
    if sub_ids != sol_ids:
        missing = sol_ids - sub_ids
        extra = sub_ids - sol_ids
        raise ParticipantVisibleError(
            f"Submission IDs must match solution exactly. Missing {len(missing)}, extra {len(extra)}."
        )

    merged = solution.merge(
        submission,
        on=row_id_column_name,
        suffixes=("_gt", "_pred"),
        how="inner",
    )
    if merged.empty:
        raise ParticipantVisibleError(
            f"No matching rows found. Check that '{row_id_column_name}' values match."
        )

    brand_f1 = merged.apply(
        lambda r: _token_f1(r["brand_name_gt"], r["brand_name_pred"]), axis=1
    ).mean()
    product_f1 = merged.apply(
        lambda r: _token_f1(r["product_name_gt"], r["product_name_pred"]), axis=1
    ).mean()
    avg_cer = merged.apply(
        lambda r: _cer(r["ocr_text_gt"], r["ocr_text_pred"]), axis=1
    ).mean()

    final = float(
        W_BRAND * brand_f1 + W_OCR * (1.0 - avg_cer) + W_PRODUCT * product_f1
    )
    if not (final == final and abs(final) != float("inf")):
        raise ParticipantVisibleError("Metric returned a non-finite value.")
    return round(final, 4)
