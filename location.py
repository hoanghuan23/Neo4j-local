import unicodedata
import re

from photon_api import geocode_with_hints

def normalize_vi(s: str) -> str:
    """Bỏ dấu, lowercase, strip để so khớp text."""
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"[\u0300-\u036f]", "", s)  # xóa dấu thanh
    s = s.replace("đ", "d").replace("Đ", "D")
    return s.strip().lower()

def match_score(query: str, result: dict) -> float:
    """Chỉ chấp nhận name khớp toàn bộ query sau khi chuẩn hóa."""
    q = normalize_vi(query)
    name = normalize_vi(str(result.get("name") or ""))
    return 1.0 if q and name == q else 0.0

def geocode(query: str):
    results = geocode_with_hints(query)
    scored = []
    for r in results:
        if not isinstance(r.get("address"), dict):
            continue
        s = match_score(query, r)
        if s > 0:
            scored.append((s, r.get("importance", 0), r))

    scored.sort(key=lambda x: (x[1] or 0, -x[0]))
    return [r["address"] for _, _, r in scored]

if __name__ == "__main__":
    import json, sys

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        try:
            query = input("Nhập địa danh cần tra cứu: ").strip()
        except EOFError:
            query = ""
    if not query:
        raise SystemExit("Usage: python3 location.py <location>")

    results = geocode_with_hints(query)
    debug = [
        {
            "score": match_score(query, r),
            "name": r.get("name"),
            "importance": r.get("importance"),
            "address": r.get("address"),
        }
        for r in results if isinstance(r.get("address"), dict)
    ]
    debug = [r for r in debug if r["score"] > 0]
    debug.sort(key=lambda x: (x["importance"] or 0, -x["score"]))
    print(json.dumps(debug, ensure_ascii=False, indent=2))
