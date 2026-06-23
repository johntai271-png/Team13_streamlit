"""Brand rules baseline extracted from smce_baseline.ipynb."""

from __future__ import annotations

import re

BRAND_RULES = [
    # PATE / HA LONG (dominant in test set)
    (r"ha\s*long\s*canfoco.*pate.*c[ộo]t|c[ộo]t\s*đ[èe]n.*ha\s*long\s*canfoco", "Ha Long Canfoco Pate Cột Đèn", []),
    (r"ha\s*long\s*canfoco.*pate|canfoco.*pate\s*c[ộo]t|pate\s*c[ộo]t\s*đ[èe]n.*canfoco", "Ha Long Canfoco Pate", []),
    (r"ha\s*long\s*canfoco|halong\s*canfoco|canfood|canfoco", "Ha Long Canfoco", []),
    (r"đ[ồo]\s*h[ộo]p\s*h[ạa]\s*long|do\s*hop\s*ha\s*long", "Đồ Hộp Hạ Long", []),
    (r"pate\s*c[ộo]t\s*đ[èe]n|pate\s*cot\s*den|c[ộo]t\s*đ[èe]n\s*h[ảa]i\s*ph[òo]ng", "Pate Cột Đèn Hải Phòng", []),
    (r"h[ạa]\s*long\s*pate|pate\s*h[ạa]\s*long", "Ha Long Canfoco Pate", []),
    # MILK / DAIRY
    (r"vinamilk", "Vinamilk", ["flex", "adm gold", "sure", "canxi",
                                 "colosbaby", "colos baby", "ong tho", "ông thọ", "dielac", "grow"]),
    (r"th true|thtrue", "TH True Milk", ["true yogurt", "grow", "school milk", "true butter"]),
    (r"dutch lady|cô gái", "Dutch Lady", ["grow", "complete", "canxi", "mom"]),
    (r"nutifood|nuti\b", "Nutifood", ["growplus", "grow plus", "pedia", "iq"]),
    (r"ensure\b", "Abbott Ensure", ["gold", "original", "plus"]),
    (r"pediasure", "Abbott PediaSure", []),
    (r"similac", "Abbott Similac", []),
    (r"glucerna", "Abbott Glucerna", []),
    (r"milo\b", "Nestlé Milo", []),
    (r"nestle|nestlé", "Nestlé", ["milo", "coffee mate", "carnation", "nestea", "nan", "sữa bột"]),
    (r"aptamil", "Aptamil", []),
    (r"friso\b", "Friso", ["gold", "comfort", "prestige"]),
    (r"meiji\b", "Meiji", ["growing", "step"]),
    (r"ba vi\b|bavi\b|ba vì", "Ba Vì", ["gold"]),
    (r"lothamilk", "Lothamilk", ["canxi"]),
    (r"yomost", "Yomost", []),
    (r"dalat milk|đà lạt", "Đà Lạt Milk", []),
    (r"kun\b|kun milk", "Kun", ["chocolate", "strawberry"]),
    (r"fami\b", "Fami", ["canxi", "kid"]),
    (r"anlene", "Anlene", ["gold", "concentrate"]),
    (r"anchor\b", "Anchor", ["butter", "cream"]),
    # PATE / CANNED MEAT (other brands)
    (r"vissan", "Vissan", ["pate heo", "pate ga", "pate gà",
                           "xuc xich", "xúc xích", "lap xuong", "lạp xưởng"]),
    (r"hafi\b", "Hafi", ["pate"]),
    (r"ba huan|ba huân", "Ba Huân", ["pate"]),
    (r"san ha\b|san hà", "San Hà", ["pate"]),
    (r"\bcp\b|c\.p\.", "CP", ["pate", "xúc xích"]),
    (r"long bien|long biên", "Long Biên", ["pate"]),
    (r"\bpate\b|patê", "Pate", []),
    # ADD YOUR OWN BELOW
    # (r"regex", "Brand", ["line1", "line2"]),
]

def extract_product(text: str) -> str:
    """Return 'Brand ProductLine', 'Brand', or '' if no match."""
    if not text or not text.strip():
        return ""
    tl = text.lower().replace("patê", "pate")
    for pattern, brand, lines in BRAND_RULES:
        if re.search(pattern, tl, re.IGNORECASE):
            for line in lines:
                if re.search(line, tl, re.IGNORECASE):
                    return f"{brand} {line.replace('+', '+').title()}"
            return brand
    return ""


def extract_brand_product(ocr_text: str) -> tuple[str, str]:
    """Return (brand_name, product_name) for private submission format."""
    ocr_text = "" if ocr_text is None else str(ocr_text).strip()
    if not ocr_text:
        return "", ""
    tl = ocr_text.lower().replace("patê", "pate")
    for pattern, brand, lines in BRAND_RULES:
        if re.search(pattern, tl, re.IGNORECASE):
            for line in lines:
                if re.search(line, tl, re.IGNORECASE):
                    return brand, f"{brand} {line.replace('+', '+').title()}"
            return brand, brand
    product = extract_product(ocr_text)
    return "", product
