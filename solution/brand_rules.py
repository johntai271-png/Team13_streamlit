# ============================================================
# CELL 3 — LỚP QUY TẮC / REGEX THƯƠNG HIỆU (have_train)
# Gồm: chuẩn hoá Unicode, lọc nhiễu mạng xã hội, 46 quy tắc regex
# thương hiệu, tách brand/product, registry thương hiệu (mở rộng động),
# và chuẩn hoá tên brand/product. Đây là phần "chuẩn cuộc thi".
# ============================================================

# ==================== TIỆN ÍCH CHUẨN HOÁ ====================
import re
import unicodedata
from collections import Counter
import pandas as pd
from shared.data_utils import load_train_labels

train_labels_df = load_train_labels()

def _detect_col(df, *cands):
    if df is None:
        return None
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

BRAND_COL     = _detect_col(train_labels_df, 'brand_name', 'brand', 'thuong_hieu')
PRODUCT_COL   = _detect_col(train_labels_df, 'product_name', 'product', 'san_pham')
OCR_COL       = _detect_col(train_labels_df, 'ocr_text', 'ocr', 'text')
HAS_BRAND_COL = BRAND_COL is not None

_BRAND_SUPPORT_COUNTS = Counter()
_PRODUCT_SUPPORT_COUNTS = Counter()

def _brand_support_count(brand):
    if not brand:
        return 0
    return _BRAND_SUPPORT_COUNTS.get(_fold_ascii(normalize_brand_name(brand)), 0)

def _product_support_count(product):
    if not product:
        return 0
    return _PRODUCT_SUPPORT_COUNTS.get(_fold_ascii(normalize_product_name(product)), 0)

def _fold_ascii(s):
    s = str(s).replace('\u0111','d').replace('\u0110','D').casefold()
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c)!='Mn')

def strip_diacritics(text):
    return _fold_ascii(text)


# ==================== LỌC NHIỄU MẠNG XÃ HỘI ====================
SOCIAL_NOISE_WORDS = frozenset({
    'tiktok','capcut','instagram','reels','facebook','youtube',
    'follow','like','share','comment','subscribe','livestream',
    'fyp','duet','stitch','viral','trending','video','clip','news',
})

OCR_TYPO_MAP = (
    (r'canfuc([o0])', r'canfoc\1'),
    (r'vinamill\b', 'vinamilk'),
    (r'vinamik\b',  'vinamilk'),
    (r'vinamil\b',  'vinamilk'),
    (r'halong\b',   'ha long'),
    (r'pat\u00ea',  'pate'),
    (r'pediasure',   'PediaSure'),
    (r'similac',     'Similac'),
    (r'ensure\b',   'Ensure'),
    (r'glucerna',    'Glucerna'),
    (r'dielac',      'Dielac'),
    (r'dutchlady',   'Dutch Lady'),
    (r'dutch\s*lady','Dutch Lady'),
    (r'nescafe',     'Nescafe'),
    (r'nestle',      'Nestl\u00e9'),
    (r'nestl[e\u00e9]',   'Nestl\u00e9'),
    (r'optipro',     'OPTIpro'),
    (r'highlands?\s*coffee', 'Highlands Coffee'),
    (r'larocheposay|la\s*roche[\s-]*posay', 'La Roche-Posay'),
    (r"paulaschoice|paula['\u2019]?s\s*choice", "Paula's Choice"),
    (r'pediasure',   'PediaSure'),
    (r'profutura',   'Profutura'),
    (r'growplus|grow\s*plus', 'GrowPlus'),
    (r'cotc[e]n',    'c\u1ed9t \u0111\u00e8n'),
    (r'cot\s*den',  'c\u1ed9t \u0111\u00e8n'),
)

_TRAIN_OCR_CANON = None


def _build_train_ocr_canon():
    """Map fold(ocr_train) -> ocr chu\u1ea9n t\u1eeb train_labels (s\u1eeda OCR g\u1ea7n GT)."""
    global _TRAIN_OCR_CANON
    if _TRAIN_OCR_CANON is not None:
        return _TRAIN_OCR_CANON
    canon = {}
    df = train_labels_df
    col = OCR_COL
    if df is not None and col and col in df.columns:
        for val in df[col].dropna().astype(str):
            v = unicodedata.normalize('NFC', val.strip())
            if len(v) < 6:
                continue
            k = _fold_ascii(v)
            if k and k not in canon:
                canon[k] = v
    _TRAIN_OCR_CANON = canon
    return canon


def apply_ocr_typo_map(text):
    """\u00c1p OCR_TYPO_MAP l\u00ean chu\u1ed7i OCR."""
    if not text:
        return ''
    t = unicodedata.normalize('NFC', str(text))
    for pat, repl in OCR_TYPO_MAP:
        if callable(repl):
            continue
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', t).strip()


def correct_ocr_from_train_catalog(text, cutoff=0.92):
    """N\u1ebfu OCR g\u1ea7n kh\u1edbp m\u1eabu train -> tr\u1ea3 b\u1ea3n chu\u1ea9n (ch\u1ec9 chu\u1ed7i ng\u1eafn)."""
    if not text or len(text) > 120:
        return text
    canon = _build_train_ocr_canon()
    if not canon:
        return text
    key = _fold_ascii(text)
    if key in canon:
        return canon[key]
    try:
        from difflib import get_close_matches
        hit = get_close_matches(key, list(canon.keys()), n=1, cutoff=cutoff)
        if hit:
            return canon[hit[0]]
    except Exception:
        pass
    return text


SOCIAL_CAPTION_RES = (
    r'(?i)\btr[aả]\s*l[oời]\s*h[iì]nh\s*lu[aậ]n\b',
    r'(?i)\bch[oố]t\s*li[eề]n\b',
    r'(?i)\br[eẻ]\s*qu[aá]\b',
    r'(?i)\bc[aầ]m\s*đ[uự]ợc\s*tr[eê]n\s*tay\b',
    r'(?i)\binbox\b|\bib\b|\bzalo\b',
    r'(?i)\bfreeship\b|\bship\s*cod\b',
    r'(?i)\bcomment\b|\breply\b',
)

_SOCIAL_CAPTION_WORDS = frozenset({
    'tra', 'loi', 'trả', 'lời', 'hình', 'luận', 'chốt', 'liền', 'rẻ', 'quá',
    'cầm', 'tay', 'nêng', 'thúy', 'thuy', 'comment', 'reply', 'inbox', 'zalo',
})

# Tin tức / chiến dịch y tế / bài viết mẹ bé — không phải nhãn sản phẩm
INFORMATIONAL_NOISE_RES = (
    r'(?i)\bvitamin\s*a\b',
    r'(?i)\bquoc\s*te\s*thieu\s*nhi\b',
    r'(?i)\bthi\s*luc\b|\bthị\s*lực\b',
    r'(?i)\bcac\s*mom\b|\bcác\s*mom\b',
    r'(?i)\bdung\s*quen\b|\bđừng\s*quên\b',
    r'(?i)\bngay\s*mai\b|\bngày\s*mai\b',
    r'(?i)\bco\s*so\s*y\s*te\b|\bcơ\s*sở\s*y\s*tế\b',
    r'(?i)\bmien\s*phi\b|\bmiễn\s*phí\b',
    r'(?i)\bem\s*sua\s*\d+\s*thang\b|\bEM\s*SỮA\s*\d+',
    r'(?i)\btram\s*trinh\b|\btrạm\s*trình\b',
    r'(?i)\buong\s*nghen\b|\buống\s*nghen\b',
    r'(?i)\bphat\s*trien\s*xuong\b|\bphát\s*triển\s*xương\b',
    r'(?i)\b6\s*-\s*36\s*thang\b',
    r'(?i)\btang\s*cuong\b|\btăng\s*cường\b',
    r'(?i)\bcho\s*cac\s*be\b|\bcho\s*các\s*bé\b',
    r'(?i)\bsap\s*xep\s*thoi\s*gian\b|\bsắp\s*xếp\s*thời\s*gian\b',
)

_INFORMATIONAL_NOISE_WORDS = frozenset({
    'vitamin', 'mom', 'moms', 'nghen', 'thieu', 'nhi', 'quoc', 'te',
    'mien', 'phi', 'thang', 'uong', 'ngay', 'mai', 'quen', 'dung',
    'yte', 'tế', 'sua', 'em', 'be', 'bé', 'cac', 'các', 'tang', 'cuong',
    'cường', 'xuong', 'xương', 'thiluc', 'thịlực', 'trinh', 'tram',
})

# Token ngắn / từ thường gặp — không match sub-line product trong BRAND_RULES
_PRODUCT_SUBLINE_BLOCKLIST = frozenset({
    'mom', 'em', 'be', 'cac', 'lon', 'can', 'hu', 'new', 'gold', 'plus',
    'den', 'sen', 'do', 'ha', 'long', 'cot', 'hop', 'sua', 'ml', 'g', 'kg',
})


def _is_informational_noise(text):
    """Tin tức / chiến dịch / caption bài viết — không phải brand/product."""
    if not text or not str(text).strip():
        return False
    raw = unicodedata.normalize('NFC', str(text).strip())
    if extract_by_rules(raw) or detect_brand_in_ocr(raw):
        return False
    t = _fold_ascii(raw)
    if any(re.search(pat, t) for pat, _ in _PRODUCT_LINE_PATTERNS):
        return False
    t_low = raw.lower()
    if any(re.search(p, t_low) for p in SOCIAL_CAPTION_RES):
        return True
    t = _fold_ascii(raw)
    if any(re.search(p, t_low) or re.search(p, t) for p in INFORMATIONAL_NOISE_RES):
        return True
    words = re.findall(r'[^\W\d_]+', t_low, flags=re.UNICODE)
    if len(words) >= 4:
        hits = sum(1 for w in words if w in _INFORMATIONAL_NOISE_WORDS)
        if hits >= 2:
            return True
    if re.search(r'(?i)\b(19|20)\d{2}\b', raw) and re.search(r'(?i)thieu\s*nhi|quoc\s*te', t):
        return True
    if len(words) >= 10:
        has_brand = bool(extract_by_rules(raw) or detect_brand_in_ocr(raw))
        has_prod = any(re.search(pat, t) for pat, _ in _PRODUCT_LINE_PATTERNS)
        if not has_brand and not has_prod:
            if _is_description_prose(raw) or hits >= 1:
                return True
    return False


def _is_plausible_brand_name(text):
    """Brand hợp lệ — loại headline tin tức / chiến dịch."""
    if not text or _is_informational_noise(text):
        return False
    if _is_description_prose(text) or _is_glued_product_brand(text):
        return False
    words = str(text).split()
    if len(words) > 6:
        return False
    folded = _fold_ascii(text)
    if extract_by_rules(text):
        return True
    if detect_brand_in_ocr(text):
        return True
    for brand in KNOWN_BRANDS:
        if _fold_ascii(brand) == folded:
            return True
    if folded in BRAND_REGISTRY:
        return True
    try:
        if _brand_support_count(text) >= MIN_BRAND_SUPPORT:
            return True
    except Exception:
        pass
    if re.search(r'(?i)\b(19|20)\d{2}\b', text):
        return False
    if len(words) <= 2:
        return True
    return False


_PRODUCT_LINE_PATTERNS = [
    (r'(?i)\bion\s+pediasure(?:\s+uc|\s+úc)?\b', 'Ion Pediasure Úc'),
    (r'(?i)\bpediasure\b', 'PediaSure'),
    (r'(?i)\bsimilac\b', 'Similac'),
    (r'(?i)\bglucerna\b', 'Glucerna'),
    (r'(?i)\bensure(?:\s+gold)?\b', 'Ensure'),
    (r'(?i)\bprofutura\b', 'Profutura'),
    (r'(?i)\bdielac\b', 'Dielac'),
    (r'(?i)\bnan\s+optipro\b', 'NAN OPTIpro'),
    (r'(?i)\bnan\b', 'NAN'),
    (r'(?i)\bcombiotic\b', 'Combiotic'),
    (r'(?i)\bgrowplus\b', 'GrowPlus'),
    (r'(?i)\btra\s+sen\s+vang\b', 'Trà Sen Vàng'),
    (r'(?i)\bpate\s+c[ộo]t\s*d[eè]n\b', 'Pate Cột Đèn'),
    (r'(?i)\bnescafe\b', 'Nescafe'),
    (r'(?i)\bgrow\s*plus\b', 'GrowPlus'),

    # --- Nhóm mỹ phẩm ---
    (r'(?i)\bkem\s+ch[ôo]ng\s+n[ăa]ng\b', 'Kem chống nắng'),
    (r'(?i)\bs[ữu]a\s+r[ửu]a\s+m[ặa]t\b', 'Sữa rửa mặt'),
    (r'(?i)\bserum\b', 'Serum'),
    (r'(?i)\bretinol\b', 'Retinol'),
    (r'(?i)\btoner\b', 'Toner'),
    (r'(?i)\bm[ặa]t\s+n[ạa]\s+gi[ấa]y\b', 'Mặt nạ giấy'),
    (r'(?i)\bkem\s+dư[õo]ng\s+[âa]m\b', 'Kem dưỡng ẩm'),
    (r'(?i)\bson\s+m[ôo]i\b', 'Son môi'),
    (r'(?i)\bt[ẩa]y\s+t[ếe]\s+b[àa]o\s+ch[ếe]t\b', 'Tẩy tế bào chết'),
    (r'(?i)\bx[ịi]t\s+kho[áa]ng\b', 'Xịt khoáng'),
    (r'(?i)\bkem\s+n[ềe]n\s+cushion\b', 'Kem nền cushion'),
    (r'(?i)\bph[ấa]n\s+ph[ủu]\b', 'Phấn phủ'),
    (r'(?i)\bm[áa]\s+h[ôo]ng\b', 'Má hồng'),
    (r'(?i)\bmascara\b', 'Mascara'),
    (r'(?i)\bkem\s+che\s+khuy[ếe]t\s+đi[ểe]m\b', 'Kem che khuyết điểm'),
    (r'(?i)\bcollagen\b', 'Collagen'),
    (r'(?i)\bglutathione\b', 'Glutathione'),
    (r'(?i)\bvitamin\s+[cC]\b', 'Vitamin C'),

    # --- Nhóm sữa & dinh dưỡng ---
    (r'(?i)\bs[ữu]a\s+b[ộo]t\b', 'Sữa bột'),
    (r'(?i)\bcolos(?:\s*baby|baby|bab|i[ạa]?b?)\b', 'Colos Baby'),
    (r'(?i)\bs[ữu]a\s+t[ăa]ng\s+c[âa]n\b', 'Sữa tăng cân'),
    (r'(?i)\bs[ữu]a\s+pha\s+s[ẵa]n\b', 'Sữa pha sẵn'),
    (r'(?i)\bb[ỉi]m\b', 'Bỉm'),
    (r'(?i)\bt[ãa]\b', 'Tã'),
    (r'(?i)\bd3k2\b', 'D3K2'),
    (r'(?i)\bvitamin\s+d3\b', 'Vitamin D3'),
    (r'(?i)\bsiro\s+[ăa]n\s+ngon\b', 'Siro ăn ngon'),
    (r'(?i)\bs[ữu]a\s+chua\s+u[ôo]ng\b', 'Sữa chua uống'),
    (r'(?i)\blactoferrin\b', 'Lactoferrin'),
    (r'(?i)\bcanxi\b', 'Canxi'),
    (r'(?i)\bdha\b', 'DHA'),
    (r'(?i)\bmen\s+vi\s+sinh\b', 'Men vi sinh'),

    # --- Nhóm đồ ăn vặt ---
    (r'(?i)\bb[áa]nh\s+tr[áa]ng\s+tr[ộo]n\b', 'Bánh tráng trộn'),
    (r'(?i)\btr[àa]\s+s[ữu]a\b', 'Trà sữa'),
    (r'(?i)\bkem\s+tươi\b', 'Kem tươi'),
    (r'(?i)\bkem\s+[ôo]c\s+qu[ếe]\b', 'Kem ốc quế'),
    (r'(?i)\bb[áa]nh\s+bao\b', 'Bánh bao'),
    (r'(?i)\bch[âa]n\s+g[àa]\s+s[ảa]\s+t[ắa]c\b', 'Chân gà sả tắc'),
    (r'(?i)\bm[ìi]\s+cay\b', 'Mì cay'),
    (r'(?i)\bx[úu]c\s+x[íi]ch\b', 'Xúc xích'),
    (r'(?i)\bb[áa]nh\s+tr[áa]ng\s+nư[ớo]ng\b', 'Bánh tráng nướng'),
    (r'(?i)\bsnack\b', 'Snack'),
    (r'(?i)\bb[áa]nh\s+kem\b', 'Bánh kem'),
    (r'(?i)\bb[áa]nh\s+cu[ôo]n\b', 'Bánh cuốn'),
    (r'(?i)\bch[èe]\b', 'Chè'),
    (r'(?i)\bpudding\b', 'Pudding'),

    # --- Nhóm thời trang ---
    (r'(?i)\b[áa]o\s+thun\s+basic\b', 'Áo thun basic'),
    (r'(?i)\bqu[ầa]n\s+jeans\b', 'Quần jeans'),
    (r'(?i)\b[ôo]ng\s+r[ộo]ng\b', 'Ống rộng'),
    (r'(?i)\bđ[ầa]m\b', 'Đầm'),
    (r'(?i)\bv[áa]y\b', 'Váy'),
    (r'(?i)\b[áa]o\s+kho[áa]c\b', 'Áo khoác'),
    (r'(?i)\bđ[ồo]\s+b[ộo]\s+[ởo]\s+nh[àa]\b', 'Đồ bộ ở nhà'),
    (r'(?i)\b[áa]o\s+d[àa]i\s+c[áa]ch\s+t[âa]n\b', 'Áo dài cách tân'),
    (r'(?i)\bqu[ầa]n\s+jogger\b', 'Quần jogger'),
    (r'(?i)\b[áa]o\s+polo\b', 'Áo polo'),
    (r'(?i)\bch[âa]n\s+v[áa]y\s+x[ếe]p\s+ly\b', 'Chân váy xếp ly'),
    (r'(?i)\bt[úu]i\s+x[áa]ch\b', 'Túi xách'),
    (r'(?i)\bv[íi]\s+da\b', 'Ví da'),
    (r'(?i)\bk[íi]nh\s+m[áa]t\b', 'Kính mát'),
    (r'(?i)\bmũ\s+lư[ỡo]i\s+trai\b', 'Mũ lưỡi trai'),

    # --- Nhóm thiết bị & đồ gia dụng ---
    (r'(?i)\bs[ạa]c\s+d[ựu]\s+ph[òo]ng\b', 'Sạc dự phòng'),
    (r'(?i)\btai\s+nghe\s+bluetooth\b', 'Tai nghe bluetooth'),
    (r'(?i)\bloa\s+mini\b', 'Loa mini'),
    (r'(?i)\bđ[èe]n\s+led\b', 'Đèn LED'),
    (r'(?i)\bcamera\s+an\s+ninh\b', 'Camera an ninh'),
    (r'(?i)\bqu[ạa]t\s+mini\b', 'Quạt mini'),
    (r'(?i)\bm[áa]y\s+h[úu]t\s+b[ụu]i\s+mini\b', 'Máy hút bụi mini'),
    (r'(?i)\bn[ôo]i\s+chi[êe]n\s+kh[ôo]ng\s+d[ầa]u\b', 'Nồi chiên không dầu'),
    (r'(?i)\bm[áa]y\s+xay\s+sinh\s+t[ôo]\b', 'Máy xay sinh tố'),
    (r'(?i)\bch[ảa]o\s+ch[ôo]ng\s+d[íi]nh\b', 'Chảo chống dính'),
    (r'(?i)\b[âa]m\s+si[êe]u\s+t[ôo]c\b', 'Ấm siêu tốc'),
    (r'(?i)\bh[ôo]p\s+đ[ựu]ng\s+th[ựu]c\s+ph[âa]m\b', 'Hộp đựng thực phẩm'),
    (r'(?i)\bb[ìi]nh\s+gi[ữu]\s+nhi[ệe]t\b', 'Bình giữ nhiệt'),

    # --- Nhóm thực phẩm & đặc sản ---
    (r'(?i)\bs[ầa]u\s+ri[êe]ng\b', 'Sầu riêng'),
    (r'(?i)\bxo[àa]i\b', 'Xoài'),
    (r'(?i)\bthanh\s+long\b', 'Thanh long'),
    (r'(?i)\bbư[ởơ]i\b', 'Bưởi'),
    (r'(?i)\bg[ạa]o\s+st25\b', 'Gạo ST25'),
    (r'(?i)\bm[ăa]ng\s+c[ụu]t\b', 'Măng cụt'),
    (r'(?i)\bpate\b', 'Pate'),
    (r'(?i)\bnư[ớơ]c\s+m[ắa]m\b', 'Nước mắm'),
    (r'(?i)\btư[ơ]ng\s+[ớo]t\b', 'Tương ớt'),
    (r'(?i)\bhat\s+n[êe]m\b', 'Hạt nêm'),
    (r'(?i)\bd[ầa]u\s+[ăa]n\b', 'Dầu ăn'),
    (r'(?i)\bs[ôo]t\s+mayonnaise\b', 'Sốt mayonnaise'),

    # --- Nhóm vệ sinh & chăm sóc cơ thể ---
    (r'(?i)\bs[ữu]a\s+t[ắa]m\b', 'Sữa tắm'),
    (r'(?i)\bd[ầa]u\s+g[ộo]i\b', 'Dầu gội'),
    (r'(?i)\bd[ầa]u\s+x[ảa]\b', 'Dầu xả'),
    (r'(?i)\bs[ữu]a\s+dư[ỡo]ng\s+th[ểe]\b', 'Sữa dưỡng thể'),
    (r'(?i)\bkem\s+body\b', 'Kem body'),
    (r'(?i)\bx[ịi]t\s+kh[ửu]\s+m[ùu]i\b', 'Xịt khử mùi'),
    (r'(?i)\bnư[ớơ]c\s+hoa\b', 'Nước hoa'),
    (r'(?i)\bnư[ớơ]c\s+lau\s+s[àa]n\b', 'Nước lau sàn'),
    (r'(?i)\bnư[ớơ]c\s+r[ửu]a\s+ch[ée]n\b', 'Nước rửa chén'),
    (r'(?i)\bb[ộo]t\s+gi[ạa]t\b', 'Bột giặt'),
    (r'(?i)\bnư[ớơ]c\s+x[ảa]\s+v[ảa]i\b', 'Nước xả vải'),
    (r'(?i)\bgi[âa]y\s+v[ệe]\s+sinh\b', 'Giấy vệ sinh'),
]


def _is_social_caption(text):
    """Caption/reply TikTok — không phải brand/product."""
    if not text or not str(text).strip():
        return False
    t_low = unicodedata.normalize('NFC', str(text).strip()).lower()
    if any(re.search(p, t_low) for p in SOCIAL_CAPTION_RES):
        return True
    words = re.findall(r'[^\W\d_]+', t_low, flags=re.UNICODE)
    if len(words) >= 5:
        hits = sum(1 for w in words if w in _SOCIAL_CAPTION_WORDS)
        if hits >= 2:
            return True
    if len(words) > 8:
        prod_hint = r'pediasure|similac|ensure|nan|dielac|vinamilk|milo|aptamil|hipp|friso|pate'
        if not re.search(prod_hint, t_low, re.I):
            return True
    return False


def _strip_social_caption_segments(text):
    """Cắt đoạn caption/reply trong OCR trước predict."""
    t = str(text or '').strip()
    if not t:
        return t
    anchors = (
        r'abbott|vinamilk|nestl|pediasure|similac|ensure|aptamil|hipp|friso|nan|'
        r'dielac|canfoco|halong|highlands|pate|vinamill'
    )
    t = re.sub(
        rf'(?i)^(?:tr[aả]\s*l[oời]|reply|comment)[^.]*?(?=\b(?:{anchors})\b)',
        ' ', t)
    t = re.sub(
        r'(?i)\btr[aả]\s*l[oời]\s*h[iì]nh\s*lu[aậ]n\b[^.]*?(?=\b(?:abbott|vinamilk|pediasure|similac)\b)',
        ' ', t)
    t = re.sub(
        r'(?i)\bch[oố]t\s+li[eề]n\b[^.]*?(?=\b(?:pediasure|similac|ensure|abbott)\b)',
        ' ', t)
    t = re.sub(r'(?i)\bc[aầ]m\s+đ[uự]ợc\s+tr[eê]n\s+tay\b', ' ', t)
    t = re.sub(r'(?i)\br[eẻ]\s+qu[aá]\b', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def clean_social_ocr(text):
    text = re.sub(r'@[\w\.]+|#\w+|https?://\S+|www\.\S+', ' ', str(text))
    text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', ' ', text)
    text = re.sub(r'(\d\s*){5,}', ' ', text)
    for pat, repl in OCR_TYPO_MAP:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    text = _strip_social_caption_segments(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


PRODUCT_MIN_OCR_CHARS = 4

def _is_ocr_noise_only(text):
    text = str(text or '').strip()
    if len(text) < PRODUCT_MIN_OCR_CHARS: return True
    tokens = re.sub(
        r'[^\w\s\+\u00c0-\u1ef9\u0111\u0110]', ' ', text.lower()
    ).split()
    if not tokens: return True
    meaningful = [
        t for t in tokens
        if len(t)>=3 and t not in SOCIAL_NOISE_WORDS
        and not t.startswith('@') and 'www' not in t
    ]
    return len(meaningful)==0


# ==================== QUY TẮC THƯƠNG HIỆU v4.0 ====================
# Format: (regex, canonical_brand, [product_line_keywords])
BRAND_RULES = [
    (r'ha\s*long\s*canf[ou]c[o0].*pate.*c[\u1ed9o]t|c[\u1ed9o]t\s*[d\u0111][\u00e8e]n.*ha\s*long\s*canf[ou]c[o0]',
     'Ha Long Canfoco Pate C\u1ed9t \u0110\u00e8n', []),
    (r'ha\s*long\s*canf[ou]c[o0].*pate|canf[ou]c[o0].*pate\s*c[\u1ed9o]t|pate\s*c[\u1ed9o]t.*canf[ou]c[o0]',
     'Ha Long Canfoco Pate', []),
    (r'ha\s*long\s*canf[ou]c[o0]|halong\s*canf[ou]c[o0]|canfuc[o0]|canfood|\bcanf[ou]c[o0]\b',
     'Ha Long Canfoco', []),
    (r'halong\s*canf\b|ha\s*long\s*canf\b', 'Ha Long Canfoco', []),

    (r'[d\u0111][\u1ed3o]\s*h[\u1ed9o]p\s*h[\u1ea1a]\s*long|do\s*hop\s*ha\s*long',
     '\u0110\u1ed3 H\u1ed9p H\u1ea1 Long', []),
    (r'hophalong|hop\s*ha\s*long', '\u0110\u1ed3 H\u1ed9p H\u1ea1 Long', []),

    (r'pate\s*c[\u1ed9o]t\s*[d\u0111][\u00e8e]n|pate\s*cot\s*den|c[\u1ed9o]t\s*[d\u0111][\u00e8e]n\s*h[\u1ea3a]i\s*ph[\u00f2o]ng',
     'Pate C\u1ed9t \u0110\u00e8n H\u1ea3i Ph\u00f2ng', []),
    (r'cotc[e]n.*hai|cot\s*c[e]n.*hai|cotd.*hai\s*ph[\u00f2o]ng',
     'Pate C\u1ed9t \u0110\u00e8n H\u1ea3i Ph\u00f2ng', []),
    (r'[\[fjh]?at[e\u00ea].*cot.*d[e\u00ea\u00e8n]|cotc[e]n|cot\s*c[e]n\b|\bcotd[e\u00ea]?\b',
     'Pate C\u1ed9t \u0110\u00e8n', []),

    (r'pate\s*h[\u1ea1a]\s*long|h[\u1ea1a]\s*long\s*pate', 'Ha Long Canfoco Pate', []),
    (r'vinamilk|vinamill|vinamik', 'Vinamilk',
     ['flex','adm gold','sure','canxi','colosbaby','colos baby','ong tho','\u00f4ng th\u1ecd','dielac','grow']),
    (r'th\s*true|thtrue|th\s*true\s*milk', 'TH True Milk',
     ['true yogurt','grow','school milk','true butter']),
    (r'dutch\s*lady|c\u00f4\s*g\u00e1i|dutchlady', 'Dutch Lady',
     ['grow','complete','canxi','mom']),
    (r'nutifood|\bnuti\b', 'Nutifood', ['growplus','grow plus','pedia','iq']),
    (r'\bensure\b', 'Abbott Ensure', ['gold','original','plus']),
    (r'pediasure', 'Abbott PediaSure', []),
    (r'similac', 'Abbott Similac', []),
    (r'glucerna', 'Abbott Glucerna', []),
    (r'\bmilo\b', 'Nestl\u00e9 Milo', []),
    (r'nestle|nestl\u00e9', 'Nestl\u00e9',
     ['milo','coffee mate','carnation','nestea','nan','s\u1eefa b\u1ed9t']),
    (r'aptamil', 'Aptamil', ['first infant formula','profutura','gold']),
    (r'\bhipp\b', 'HiPP', ['combiotic','organic']),
    (r'\bfriso\b', 'Friso', ['gold','comfort','prestige']),
    (r'\bmeiji\b', 'Meiji', ['growing','step']),
    (r'nan\s*(optipro|opti\s*pro|infinipro|infini\s*pro|supremepro|supreme\s*pro|a2|ha)\b',
     'Nestl\u00e9 NAN', []),
    (r'sua\s*nan\b|nestle\s*nan|nestl\u00e9\s*nan', 'Nestl\u00e9 NAN', []),
    (r'\bdulac\b|\bdielac\b', 'Vinamilk Dielac', []),
    (r'ba\s*vi\b|bavi\b|ba\s*v\u00ec', 'Ba V\u00ec', ['gold']),
    (r'lothamilk', 'Lothamilk', ['canxi']),
    (r'yomost', 'Yomost', []),
    (r'dalat\s*milk|\u0111\u00e0\s*l\u1ea1t', '\u0110\u00e0 L\u1ea1t Milk', []),
    (r'\bkun\b|kun\s*milk', 'Kun', ['chocolate','strawberry']),
    (r'\bfami\b', 'Fami', ['canxi','kid']),
    (r'anlene', 'Anlene', ['gold','concentrate']),
    (r'\banchor\b', 'Anchor', ['butter','cream']),
    (r'vissan', 'Vissan',
     ['pate heo','pate ga','pate g\u00e0','xuc xich','x\u00fac x\u00edch','lap xuong','l\u1ea1p x\u01b0\u1edfng']),
    (r'\bhafi\b', 'Hafi', ['pate']),
    (r'ba\s*huan|ba\s*hu\u00e2n', 'Ba Hu\u00e2n', ['pate']),
    (r'san\s*ha\b|san\s*h\u00e0', 'San H\u00e0', ['pate']),
    (r'\bcp\b|c\.p\.', 'CP', ['pate','x\u00fac x\u00edch']),
    (r'long\s*bien|long\s*bi\u00ean', 'Long Bi\u00ean', ['pate']),
    (r'highlands?\s*coffee', 'Highlands Coffee',
     ['tra sen vang','tra vai','banh mi que','americano']),
    (r'the\s*coffee\s*house', 'The Coffee House', ['tra phuc kien']),
    (r'nhan\s*hoa\s*foods?', 'Nh\u00e2n H\u00f2a Foods', ['pate']),
    (r'\bacnes\b', 'Acnes', ['vitamin cleanser']),

    # --- Nh\u00f3m S\u1eefa & Dinh d\u01b0\u1ee1ng (m\u1edf r\u1ed9ng, kh\u00f4ng VitaDairy/Morinaga/Colos Baby) ---
    (r'\babbott\b', 'Abbott', ['pediasure','similac','ensure','glucerna']),
    (r'grow\s*plus|growplus', 'GrowPlus', []),

    # --- Skincare & M\u1ef9 ph\u1ea9m ---
    (r'\bdove\b', 'Dove', ['smoothie','men+care','go fresh']),
    (r'\bolay\b', 'Olay', ['total effects','regenerist']),
    (r"l['\u2019]?oreal|loreal", "L'Oreal", []),
    (r'maybelline', 'Maybelline', []),
    (r'\bnivea\b', 'Nivea', []),
    (r'cetaphil', 'Cetaphil', []),
    (r'bioderma', 'Bioderma', ['sensibio']),
    (r'la\s*roche[\s-]*posay|larocheposay', 'La Roche-Posay', []),
    (r'skin\s*1004|skin1004', 'Skin1004', []),
    (r'\bklairs\b', 'Klairs', []),
    (r"paula['\u2019]?s\s*choice", "Paula's Choice", []),
    (r'\bgarnier\b', 'Garnier', []),
    (r'the\s*whoo', 'The Whoo', []),
    (r'\blaneige\b', 'Laneige', []),
    (r'hada\s*labo', 'Hada Labo', []),
    (r'\beveline\b', 'Eveline', []),
    (r'the\s*body\s*shop', 'The Body Shop', []),
    (r'hazeline', 'Hazeline', []),
    (r'\bbotanika\b', 'Botanika', []),
    (r'\bziaja\b', 'ZIAJA', []),

    # --- Th\u1ef1c ph\u1ea9m & \u0110\u1ed3 u\u1ed1ng ---
    (r'nescafe|nescaf[e\u00e9]', 'Nescafe', ['gold','classic','latte']),

    # --- M\u00e1y m\u00f3c & \u0110i\u1ec7n t\u1eed ---
    (r'panasonic', 'Panasonic', []),
    (r'\bdell\b', 'Dell', []),
    (r'\blenovo\b', 'Lenovo', []),
    (r'\bgodox\b', 'Godox', []),
    (r'\bcanon\b', 'Canon', []),
    (r'\bedifier\b', 'Edifier', []),
    (r'\banker\b', 'Anker', []),
    (r'\btefal\b', 'Tefal', []),
    (r'\bnidec\b', 'Nidec', []),
    (r'vinfast', 'VinFast', []),
    (r'mitsubishi', 'Mitsubishi', []),

    # --- Ch\u01b0a x\u00e1c minh \u0111\u1ea7y \u0111\u1ee7 ---
    (r'\btololo\b', 'Tololo', []),
    (r'\byamia\b', 'Yamia', []),
    (r'lanbena', 'Lanbena', []),
    (r'\boggi\b', 'Oggi', []),
    (r'\bbere\b', 'BERE', []),
    (r'suanon', 'SUANON', []),
    (r'\bnitol\b', 'Nitol', []),
    (r'\bpate\b|pat\u00ea', 'Pate', []),
]

_SUB_NAMES = {
    'flex':'Flex','adm gold':'ADM Gold','sure':'Sure','canxi':'Canxi',
    'colosbaby':'ColosBaby','colos baby':'Colos Baby',
    'ong tho':'\u00d4ng Th\u1ecd','\u00f4ng th\u1ecd':'\u00d4ng Th\u1ecd',
    'dielac':'Dielac','grow':'Grow','true yogurt':'True Yogurt',
    'school milk':'School Milk','true butter':'True Butter',
    'growplus':'GrowPlus+','grow plus':'GrowPlus+','pedia':'Pedia','iq':'IQ',
    'gold':'Gold','original':'Original','plus':'Plus',
    'milo':'Milo','coffee mate':'Coffee Mate','nan':'NAN','s\u1eefa b\u1ed9t':'S\u1eefa B\u1ed9t',
    'first infant formula':'First Infant Formula','profutura':'Profutura',
    'combiotic':'Combiotic','organic':'Organic',
    'growing':'Growing','step':'Step',
    'comfort':'Comfort','prestige':'Prestige',
    'pate heo':'Pate Heo','pate ga':'Pate G\u00e0','pate g\u00e0':'Pate G\u00e0',
    'xuc xich':'X\u00fac X\u00edch','x\u00fac x\u00edch':'X\u00fac X\u00edch',
    'lap xuong':'L\u1ea1p X\u01b0\u1edfng','l\u1ea1p x\u01b0\u1edfng':'L\u1ea1p X\u01b0\u1edfng',
    'pate':'Pate',
    'tra sen vang':'Tr\u00e0 Sen V\u00e0ng','tra vai':'Tr\u00e0 V\u1ea3i',
    'banh mi que':'B\u00e1nh M\u00ec Que','americano':'Americano',
    'tra phuc kien':'Tr\u00e0 Ph\u00fac Ki\u1ebfn',
    'vitamin cleanser':'Vitamin Cleanser','concentrate':'Concentrate',
    'butter':'Butter','cream':'Cream','chocolate':'Chocolate','strawberry':'Strawberry',
    'kid':'Kid','kid+':'Kid+',
}


def _normalize_for_rules(text):
    tl = text.lower().replace('pat\u00ea','pate')
    tl = re.sub(r'vina[jilm1]{0,1}milk', 'vinamilk', tl, flags=re.IGNORECASE)
    tl = re.sub(r'canfuc([o0])', r'canfoc\1', tl, flags=re.IGNORECASE)
    tl = re.sub(r'halong\b', 'ha long', tl, flags=re.IGNORECASE)
    return tl


# --- Bộ lọc token chống ảo giác (anti-hallucination) ---
def _build_brand_vocab():
    vocab = set()
    for _pat, brand, lines in BRAND_RULES:
        for w in brand.split(): vocab.add(w.casefold())
        for line in lines:
            for w in line.title().split(): vocab.add(w.casefold())
    return frozenset(vocab)

_BRAND_VOCAB = _build_brand_vocab()
_BRAND_VOCAB_FOLDED = frozenset(_fold_ascii(t) for t in _BRAND_VOCAB)


def _ocr_word_set(text):
    tl = _normalize_for_rules(text)
    return {_fold_ascii(w) for w in re.findall(r'[^\W\d_]+', tl, flags=re.UNICODE) if w}


def _token_in_ocr(tok, ocr_text, ocr_words):
    tf = _fold_ascii(tok)
    if tf in ocr_words: return True
    compact = _fold_ascii(ocr_text).replace(' ','')
    if len(tf)>=2 and tf in compact: return True
    return any(len(tf)>=2 and tf in ow for ow in ocr_words)


def _assign_readable_brand_tokens(ocr_text, candidate):
    """Chỉ giữ token thuộc brand list VÀ đọc được từ ocr_text (chống ảo giác)."""
    if not candidate or not candidate.strip(): return ''
    ocr_words = _ocr_word_set(ocr_text)
    kept = []
    for tok in candidate.split():
        if _fold_ascii(tok) not in _BRAND_VOCAB_FOLDED: continue
        if _token_in_ocr(tok, ocr_text, ocr_words): kept.append(tok)
    return ' '.join(kept)


def extract_by_rules(text):
    # Rule neo theo regex trên OCR nên TIN CẬY -> trả về nhãn đầy đủ.
    if not text: return ''
    tl = _normalize_for_rules(text)
    matched = ''
    for pattern, brand, lines in BRAND_RULES:
        if re.search(pattern, tl, re.IGNORECASE):
            for line in lines:
                if re.search(line, tl, re.IGNORECASE):
                    sub = _SUB_NAMES.get(line, line.title())
                    matched = f'{brand} {sub}'.strip()
                    break
            if not matched:
                matched = brand
            break
    return matched


def post_process_prediction(pred, ocr_text):
    """Nâng cấp nhãn 'Pate' chung chung -> cụ thể hơn bằng cách quét lại OCR."""
    if not pred: return pred
    nd = _fold_ascii(ocr_text) if ocr_text else ''
    if pred == 'Pate':
        if re.search(r'hai\s*ph[o]ng', nd):
            if re.search(r'cot|cotcen|cotd|col\s*d', nd):
                return 'Pate C\u1ed9t \u0110\u00e8n H\u1ea3i Ph\u00f2ng'
        if any(re.search(p, nd) for p in [r'cot\s*d[e]n',r'\bcotd[e]?\b',r'cotcen',r'cot\s*cen']):
            return 'Pate C\u1ed9t \u0110\u00e8n'
        if re.search(r'ha\s*long|halong', nd):
            return '\u0110\u1ed3 H\u1ed9p H\u1ea1 Long'
    if pred == 'Nestl\u00e9':
        if re.search(r'\bnan\b|nan\s*opti|nan\s*infini|nan\s*supreme', nd):
            return 'Nestl\u00e9 NAN'
        if re.search(r'\bmilo\b', nd):
            return 'Nestl\u00e9 Milo'
    if pred == 'Vinamilk':
        if re.search(r'dulac|dielac', nd):
            return 'Vinamilk Dielac'
    return pred


# ==================== CHUẨN HOÁ TÊN SẢN PHẨM ====================
_PRODUCT_CANON_MAP = None

def _build_product_canon_map():
    """Gộp các biến thể chính tả/dấu câu của cùng 1 sản phẩm bằng fold-key,
    chọn spelling phổ biến nhất trong train data làm chính tắc."""
    global _PRODUCT_CANON_MAP
    if _PRODUCT_CANON_MAP is not None:
        return _PRODUCT_CANON_MAP
    _PRODUCT_CANON_MAP = {}
    try:
        if train_labels_df is not None:
            _pcol = PRODUCT_COL if ('PRODUCT_COL' in globals() and PRODUCT_COL) else 'product_name'
            if _pcol is not None and _pcol in train_labels_df.columns:
                names = train_labels_df[_pcol].astype(str).str.strip()
                names = names[names != '']
                counts = Counter(unicodedata.normalize('NFC', n) for n in names)
                groups = {}
                for name, cnt in counts.items():
                    key = _fold_ascii(name)
                    groups.setdefault(key, []).append((cnt, name))
                for key, variants in groups.items():
                    variants.sort(key=lambda x: (-x[0], -len(x[1])))
                    _, best_name = variants[0]
                    _PRODUCT_CANON_MAP[key] = best_name
    except Exception:
        pass
    return _PRODUCT_CANON_MAP


def _is_plausible_product_name(text):
    """Tên product hợp lệ — loại caption dài / generic."""
    if not text or _is_social_caption(text) or _is_informational_noise(text):
        return False
    if _is_generic_product(text):
        return False
    words = str(text).split()
    if len(words) > 8:
        return False
    folded = _fold_ascii(text)
    if any(re.search(pat, folded) for pat, _ in _PRODUCT_LINE_PATTERNS):
        return True
    canon = _build_product_canon_map()
    if folded in canon:
        return True
    try:
        if _product_support_count(text) >= MIN_BRAND_SUPPORT:
            return True
    except Exception:
        pass
    if _is_description_prose(text):
        return False
    if len(words) <= 3 and any(re.search(pat, folded) for pat, _ in _PRODUCT_LINE_PATTERNS):
        return True
    return False


def guess_product_from_ocr(ocr_text, brand=''):
    """Trích product từ OCR sau khi đã có brand (catalog + pattern + sub-lines rules)."""
    ocr_text = str(ocr_text or '').strip()
    if not ocr_text or _is_ocr_noise_only(ocr_text):
        return ''
    if _is_description_prose(ocr_text):
        ocr_text = clean_social_ocr(ocr_text)
    if not ocr_text:
        return ''
    tl = _normalize_for_rules(ocr_text)
    folded = _fold_ascii(tl)
    compact = folded.replace(' ', '')

    for pat, label in _PRODUCT_LINE_PATTERNS:
        if re.search(pat, tl, re.IGNORECASE):
            return label

    canon = _build_product_canon_map()
    for key, pname in sorted(canon.items(), key=lambda x: -len(x[0])):
        if len(key) < 4 or key in _PRODUCT_SUBLINE_BLOCKLIST:
            continue
        if len(key) <= 5:
            if not re.search(r'(?<!\w)' + re.escape(key) + r'(?!\w)', folded):
                continue
        elif key not in folded and key.replace(' ', '') not in compact:
            continue
        if not _is_social_caption(pname) and not _is_informational_noise(pname):
            return pname

    bf = _fold_ascii(brand) if brand else ''
    for _pat, _bname, lines in BRAND_RULES:
        if bf and bf not in _fold_ascii(_bname) and _fold_ascii(_bname) not in bf:
            if not re.search(_pat, tl, re.IGNORECASE):
                continue
        for line in lines:
            if len(line) < 4 or line.lower() in _PRODUCT_SUBLINE_BLOCKLIST:
                continue
            if not re.search(r'(?<!\w)' + re.escape(line) + r'(?!\w)', tl, re.IGNORECASE):
                continue
            sub = _SUB_NAMES.get(line, line.title())
            if not _is_generic_product(sub) and not _is_informational_noise(sub):
                return sub
    return ''


def normalize_product_name(name):
    """Chuẩn hoá nhãn sản phẩm: NFC + map catalog train; không .title() chuỗi lạ."""
    if not name:
        return ''
    name = unicodedata.normalize('NFC', str(name).strip())
    name = re.sub(r'\s+', ' ', name)
    if not name or _is_social_caption(name):
        return ''
    canon_map = _build_product_canon_map()
    key = _fold_ascii(name)
    if key in canon_map:
        return canon_map[key]
    return name


# ==================== TÁCH BRAND / PRODUCT ====================
KNOWN_BRANDS = [
    'Ha Long Canfoco', 'Pate C\u1ed9t \u0110\u00e8n', 'TH True Milk', 'Dutch Lady',
    'Highlands Coffee', 'The Coffee House', 'Nh\u00e2n H\u00f2a Foods', 'H\u1ea1 Long',
    'Abbott', 'Nestl\u00e9', 'Vinamilk', 'Nutifood', 'Aptamil', 'HiPP', 'Friso',
    'Meiji', 'Ba V\u00ec', 'Lothamilk', 'Yomost', '\u0110\u00e0 L\u1ea1t Milk', 'Kun',
    'Fami', 'Anlene', 'Anchor', 'Vissan', 'Hafi', 'Ba Hu\u00e2n', 'San H\u00e0',
    'CP', 'Long Bi\u00ean', 'Acnes',
    'GrowPlus',
    'Dove', 'Olay', "L'Oreal", 'Maybelline', 'Nivea', 'Cetaphil', 'Bioderma',
    'La Roche-Posay', 'Skin1004', 'Klairs', "Paula's Choice", 'Garnier', 'The Whoo',
    'Laneige', 'Hada Labo', 'Eveline', 'The Body Shop', 'Hazeline', 'Botanika', 'ZIAJA',
    'Nescafe', 'Panasonic', 'Dell', 'Lenovo', 'Godox', 'Canon', 'Edifier', 'Anker',
    'Tefal', 'Nidec', 'VinFast', 'Mitsubishi',
    'Tololo', 'Yamia', 'Lanbena', 'Oggi', 'BERE', 'SUANON', 'Nitol',
]

# Hãng (manufacturer) vs dòng sản phẩm — ưu tiên hãng làm brand_name
_MANUFACTURER_CANON = frozenset({
    'Abbott', 'Nestlé', 'Vinamilk', 'Nutifood', 'Aptamil', 'HiPP', 'Friso', 'Meiji',
    'Ha Long Canfoco', 'TH True Milk', 'Dutch Lady', 'Highlands Coffee', 'The Coffee House',
    'Vissan', 'Dove', 'Nhân Hòa Foods', 'Đà Lạt Milk', 'Ba Vì', 'Lothamilk', 'Yomost',
    'Fami', 'Anlene', 'Anchor', 'Hafi', 'Ba Huân', 'San Hà', 'CP', 'Long Biên', 'Acnes',
    'Hạ Long', 'Pate Cột Đèn',
})
_MANUFACTURER_FOLDED = frozenset(_fold_ascii(b) for b in _MANUFACTURER_CANON)

# (product_regex_on_folded, manufacturer_regex, brand_out, product_out)
_BRAND_PRODUCT_OCR_PAIRS = [
    (r'similac', r'abbott', 'Abbott', 'Similac'),
    (r'pediasure', r'abbott', 'Abbott', 'PediaSure'),
    (r'glucerna', r'abbott', 'Abbott', 'Glucerna'),
    (r'ensure', r'abbott', 'Abbott', 'Ensure'),
    (r'\bnan\b', r'nestl', 'Nestlé', 'NAN'),
    (r'dielac', r'vinamilk', 'Vinamilk', 'Dielac'),
    (r'profutura', r'aptamil', 'Aptamil', 'Profutura'),
]

_DESCRIPTION_PROSE_MARKERS = (
    'cong thuc', 'duoc phat trien', 'được phát triển', 'nham ho tro', 'nhằm hỗ trợ',
    'tieu hoa', 'tiêu hóa', 'phat trien', 'phát triển', 'cao cap', 'cao cấp',
    'cua abbott', 'của abbott', 'hemiendich', 'tieuhoa', 'tren hoi', 'protection',
    'resentional', 'duoc phat', 'tro he', 'choi tren',
)


def _is_description_prose(text):
    """Đoạn mô tả/quảng cáo dài — không phải product_name."""
    if not text or not str(text).strip():
        return False
    raw = str(text).strip()
    t = _fold_ascii(raw)
    if _is_social_caption(raw):
        return True
    words = raw.split()
    if len(words) > 6:
        hits = sum(1 for m in _DESCRIPTION_PROSE_MARKERS if m in t or m in raw.lower())
        if hits >= 1:
            return True
        vn = ('duoc', 'được', 'nham', 'nhằm', 'cong', 'công', 'cao', 'cap', 'cấp',
              'cua', 'của', 'tren', 'trên', 'phat', 'phát', 'trien', 'triển', 'ho tro')
        if sum(1 for w in words if w.lower() in vn) >= 2:
            return True
    if ',' in raw and len(words) > 4:
        return True
    if len(raw) > 42:
        if not any(re.search(p, t) for p, _ in _PRODUCT_LINE_PATTERNS):
            return True
        desc_hits = sum(1 for m in _DESCRIPTION_PROSE_MARKERS if m in t)
        if desc_hits >= 1:
            return True
    return False


def _is_glued_product_brand(name):
    """Brand nhìn như dòng SP dính chữ OCR (vd Similac Totalladongsữa)."""
    if not name:
        return False
    f = _fold_ascii(name).replace(' ', '')
    for key in ('similac', 'pediasure', 'ensure', 'glucerna', 'profutura', 'dielac'):
        if f.startswith(key) and len(f) > len(key) + 3:
            return True
    return False


def _score_brand_candidate(canonical):
    """Điểm alias brand: ưu tiên hãng, phạt tên dài/dính rác."""
    if not canonical:
        return -999
    score = 0
    cf = _fold_ascii(canonical)
    if cf in _MANUFACTURER_FOLDED or canonical in _MANUFACTURER_CANON:
        score += 200
    if any(cf.startswith(_fold_ascii(m)) for m in _MANUFACTURER_CANON):
        score += 150
    score -= max(0, len(canonical) - 18)
    score -= max(0, len(canonical.split()) - 3) * 25
    if _is_glued_product_brand(canonical):
        score -= 300
    if _is_description_prose(canonical):
        score -= 400
    return score


def reconcile_brand_product(ocr_text, brand, product):
    """Chuẩn hoá cặp brand/product từ OCR — Abbott+Similac, bỏ mô tả dài."""
    ocr_text = str(ocr_text or '').strip()
    brand = str(brand or '').strip()
    product = str(product or '').strip()
    if not ocr_text:
        return brand, product

    folded = _fold_ascii(_normalize_for_rules(ocr_text))

    for prod_pat, manu_pat, b_out, p_out in _BRAND_PRODUCT_OCR_PAIRS:
        if re.search(prod_pat, folded, re.I) and re.search(manu_pat, folded, re.I):
            return b_out, p_out

    line_prod = guess_product_from_ocr(ocr_text, brand) or ''
    manu = ''
    if re.search(r'abbott', folded, re.I):
        manu = 'Abbott'
    elif re.search(r'nestl', folded, re.I):
        manu = 'Nestlé'
    elif re.search(r'vinamilk|vinamill', folded, re.I):
        manu = 'Vinamilk'
    elif re.search(r'aptamil', folded, re.I):
        manu = 'Aptamil'

    if _is_glued_product_brand(brand) or _is_description_prose(brand):
        brand = ''
    if _is_description_prose(product):
        product = ''
    if product and _fold_ascii(brand) == _fold_ascii(product):
        product = ''

    if not brand and manu:
        brand = manu
    if not product and line_prod:
        product = line_prod

    if brand and not product and line_prod:
        product = line_prod
    if product and not brand and manu:
        brand = manu

    if _is_glued_product_brand(brand) and line_prod:
        if manu:
            brand = manu
        product = line_prod

    brand = normalize_brand_name(brand) if brand else ''
    if product:
        product = normalize_product_name(product)
        if brand:
            bf, pf = _fold_ascii(brand), _fold_ascii(product)
            if pf.startswith(bf + ' '):
                product = product[len(brand):].strip()
        if _is_generic_product(product) or _is_description_prose(product):
            product = ''
    return brand, product


MERGED_SPLIT = {
    'Ha Long Canfoco Pate C\u1ed9t \u0110\u00e8n': ('Ha Long Canfoco', 'Pate C\u1ed9t \u0110\u00e8n'),
    'Ha Long Canfoco Pate':                       ('Ha Long Canfoco', 'Pate'),
    'Ha Long Canfoco':                            ('Ha Long Canfoco', ''),
    '\u0110\u1ed3 H\u1ed9p H\u1ea1 Long':         ('H\u1ea1 Long', '\u0110\u1ed3 H\u1ed9p'),
    'Pate C\u1ed9t \u0110\u00e8n H\u1ea3i Ph\u00f2ng': ('Pate C\u1ed9t \u0110\u00e8n', 'H\u1ea3i Ph\u00f2ng'),
    'Pate C\u1ed9t \u0110\u00e8n':                 ('Pate C\u1ed9t \u0110\u00e8n', ''),
    'Pate':                                       ('', 'Pate'),
    'Vinamilk Dielac':                            ('Vinamilk', 'Dielac'),
    'Nestl\u00e9 NAN':                            ('Nestl\u00e9', 'NAN'),
    'Nestl\u00e9 Milo':                           ('Nestl\u00e9', 'Milo'),
}

GENERIC_PRODUCT_TOKENS = frozenset({
    'cot','hop','milk','sua','gold','plus','new','lon','can','hu','ml','g','kg',
    'den','sen','do','ha','long',
})

MIN_BRAND_SUPPORT   = 2   # brand mới từ train cần >= số mẫu này
MIN_BRAND_ALIAS_LEN = 4   # alias ngắn hơn -> bỏ qua (tránh dùng 'cp','ps')

BRAND_REGISTRY = {}
def _register_brand(canonical):
    canonical = unicodedata.normalize('NFC', str(canonical).strip())
    if not canonical: return
    aliases = set()
    f = _fold_ascii(canonical)
    aliases.add(f)
    aliases.add(f.replace(' ', ''))
    for a in list(aliases):
        if len(a.replace(' ','')) >= MIN_BRAND_ALIAS_LEN or ' ' in a:
            BRAND_REGISTRY[a] = canonical

for _b in KNOWN_BRANDS:
    _register_brand(_b)

_BRAND_CANON_FOLD = {}   # fold(brand) -> canonical casing (ưu tiên train)

def build_brand_knowledge_from_train():
    """Học brand mới (vd 'Dove') từ cột brand_name của train, có kiểm soát
    support tối thiểu. Không bịa: chỉ đưa brand có trong train."""
    if train_labels_df is None or BRAND_COL is None:
        return 0
    s = train_labels_df[BRAND_COL].astype(str).map(lambda x: unicodedata.normalize('NFC', x.strip()))
    s = s[s != '']
    vc = s.value_counts()
    added = 0
    for brand, cnt in vc.items():
        _BRAND_CANON_FOLD.setdefault(_fold_ascii(brand), brand)
        if cnt >= MIN_BRAND_SUPPORT:
            _register_brand(brand)
            added += 1
    return added


def normalize_brand_name(name):
    if not name: return ''
    name = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(name).strip()))
    if not name: return ''
    f = _fold_ascii(name)
    if f in _BRAND_CANON_FOLD: return _BRAND_CANON_FOLD[f]
    if f in BRAND_REGISTRY:    return BRAND_REGISTRY[f]
    fc = f.replace(' ', '')
    if fc in BRAND_REGISTRY:   return BRAND_REGISTRY[fc]
    return name


def detect_brand_in_ocr(ocr_text):
    """Quét alias brand trong OCR — ưu tiên hãng (manufacturer), phạt tên dính OCR."""
    if not ocr_text:
        return ''
    folded = _fold_ascii(_normalize_for_rules(ocr_text))
    words = set(folded.split())
    compact = folded.replace(' ', '')
    best, best_score = '', -10**9
    for alias, canonical in BRAND_REGISTRY.items():
        a = alias.replace(' ', '')
        if len(a) < MIN_BRAND_ALIAS_LEN:
            continue
        if ' ' in alias:
            hit = alias in folded
        else:
            hit = (a in words) or (len(a) >= 6 and a in compact)
        if not hit or _is_description_prose(canonical):
            continue
        sc = _score_brand_candidate(canonical)
        if sc > best_score:
            best_score, best = sc, canonical
    return best


def _is_generic_product(product):
    if not product: return True
    f = _fold_ascii(product).strip()
    if len(f) < 2: return True
    toks = f.split()
    if len(toks) == 1 and toks[0] in GENERIC_PRODUCT_TOKENS: return True
    return False


def split_brand_product(merged):
    """Tách nhãn gộp -> (brand_name, product_name)."""
    if not merged: return '', ''
    nm = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(merged).strip()))
    if nm in MERGED_SPLIT:
        b, p = MERGED_SPLIT[nm]
        return normalize_brand_name(b), p
    nf = _fold_ascii(nm)
    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):
        bf = _fold_ascii(brand)
        if nf == bf:
            return normalize_brand_name(brand), ''
        if nf.startswith(bf + ' '):
            return normalize_brand_name(brand), nm[len(brand):].strip()
    return '', nm


def _build_training_frame():
    """Chuẩn hoá train -> DataFrame với cột _ocr / _brand / _product (auto schema)."""
    if train_labels_df is None:
        return None
    df = train_labels_df.copy()
    oc = OCR_COL if (OCR_COL and OCR_COL in df.columns) else ('ocr_text' if 'ocr_text' in df.columns else None)
    df['_ocr'] = df[oc].astype(str).str.strip() if oc else ''
    if HAS_BRAND_COL:
        df['_brand'] = df[BRAND_COL].astype(str).str.strip().map(normalize_brand_name)
        if PRODUCT_COL is not None:
            df['_product'] = df[PRODUCT_COL].astype(str).str.strip().map(normalize_product_name)
        else:
            df['_product'] = ''
    else:
        merged = df[PRODUCT_COL].astype(str).str.strip() if PRODUCT_COL is not None else pd.Series([''] * len(df))
        bp = merged.map(split_brand_product)
        df['_brand'] = bp.map(lambda t: normalize_brand_name(t[0]))
        df['_product'] = bp.map(lambda t: normalize_product_name(t[1]))
    return df[df['_ocr'] != '']


_train_frame = _build_training_frame()
if _train_frame is not None and len(_train_frame):
    for b in _train_frame['_brand']:
        if b:
            _BRAND_SUPPORT_COUNTS[_fold_ascii(normalize_brand_name(b))] += 1
    for p in _train_frame['_product']:
        if p:
            _PRODUCT_SUPPORT_COUNTS[_fold_ascii(normalize_product_name(p))] += 1


# --- Kiểm thử nhanh quy tắc ---
_rule_tests = [
    ('HA LONG CANFOCO Pate C\u1ed9t \u0110\u00e8n', 'Ha Long Canfoco'),
    ('HALONG CANFUCO j TIkTok',                    'Ha Long Canfoco'),
    ('Vinamilk Flex 180ml',                        'Vinamilk'),
    ('MILO Nestle chocolate 3 in 1',               'Nestl\u00e9'),
    ('tiktok capcut viral',                        ''),
]
_ok = sum(1 for t, e in _rule_tests
          if (e == '' and extract_by_rules(t) == '') or (e != '' and e.lower() in extract_by_rules(t).lower()))
print(f'Lớp quy tắc: {len(BRAND_RULES)} regex | Kiểm thử nhanh: {_ok}/{len(_rule_tests)} đạt')
print("✓ Đã chạy xong cell 3")