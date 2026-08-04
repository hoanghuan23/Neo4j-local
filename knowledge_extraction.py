import json
import math
import re
import unicodedata
from functools import lru_cache

import requests

from knowledge_settings import (
    CONFIDENCE_LEVELS,
    COUNTRY_NAME_FALLBACKS,
    ENTITY_TYPES,
    EVENT_NAME_PATTERN,
    GENERIC_ENTITY_EXACT,
    GENERIC_PERSON_OR_GROUP_SUFFIXES,
    KNOWLEDGE_SCHEMA,
    LOCATION_NAME_PATTERN,
    LOGGER,
    NULL_STRINGS,
    OLLAMA_LOG_PREVIEW_CHARS,
    OLLAMA_MAX_ATTEMPTS,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    ORGANIZATION_NAME_PATTERN,
)


def normalize_name(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.strip().casefold().split()))


def make_search_name(value: str) -> str:
    normalized = unicodedata.normalize("NFD", normalize_name(value))
    without_accents = "".join(
        character for character in normalized if unicodedata.category(character) != "Mn"
    )
    return without_accents.replace("đ", "d")


def normalize_null(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.casefold() in NULL_STRINGS:
            return None
        return stripped
    if isinstance(value, list):
        return [normalize_null(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_null(item) for key, item in value.items()}
    return value


def _clean_text(value) -> str:
    normalized = normalize_null(value)
    if normalized is None:
        return ""
    return " ".join(str(normalized).split())


def _enum_value(value, allowed: set[str]) -> str | None:
    text = _clean_text(value).upper()
    return text if text in allowed else None


def _valid_confidence(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        return None
    return number


def _normalized_source_text(value: str) -> str:
    return normalize_name(re.sub(r"\s+", " ", value))


def _evidence_in_content(evidence: str, content: str) -> bool:
    return _normalized_source_text(evidence) in _normalized_source_text(content)


@lru_cache(maxsize=1)
def _country_names() -> frozenset[str]:
    names = set(COUNTRY_NAME_FALLBACKS)
    try:
        with open(
            "/usr/share/iso-codes/json/iso_3166-1.json",
            encoding="utf-8",
        ) as country_file:
            countries = json.load(country_file).get("3166-1", [])
        for country in countries:
            for key in ("name", "official_name", "common_name"):
                value = country.get(key)
                if value:
                    names.add(normalize_name(value))
    except (OSError, TypeError, ValueError):
        LOGGER.debug("Không đọc được danh mục ISO country; dùng fallback tích hợp")
    return frozenset(names)


def classify_entity_type(entity: dict) -> str | None:
    entity_type = _enum_value(entity.get("type"), ENTITY_TYPES)
    if entity_type is None:
        return None
    name = _clean_text(entity.get("canonical_name")) or _clean_text(entity.get("name"))
    normalized_name = normalize_name(name)
    if normalized_name in _country_names() or LOCATION_NAME_PATTERN.search(name):
        return "LOCATION"
    if ORGANIZATION_NAME_PATTERN.search(name):
        return "ORGANIZATION"
    return entity_type


def is_generic_entity(entity: dict) -> bool:
    name = _clean_text(entity.get("name"))
    canonical_name = _clean_text(entity.get("canonical_name")) or name
    entity_type = classify_entity_type(entity)
    candidate = normalize_name(canonical_name)

    if not candidate or candidate in GENERIC_ENTITY_EXACT:
        return True
    if candidate.startswith(("a ", "an ")):
        return True
    if candidate.startswith("the "):
        remaining = candidate[4:]
        if remaining in GENERIC_PERSON_OR_GROUP_SUFFIXES:
            return True
    words = candidate.split()
    if len(words) == 2 and words[-1] in GENERIC_PERSON_OR_GROUP_SUFFIXES:
        return True
    if entity_type in {"PERSON", "ORGANIZATION"} and EVENT_NAME_PATTERN.search(
        candidate
    ):
        return True
    return False


def parse_ollama_payload(payload: dict) -> dict:
    raw_response = payload.get("response", "")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("Ollama trả về trường response rỗng")

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as error:
        preview = raw_response[:OLLAMA_LOG_PREVIEW_CHARS]
        LOGGER.error(
            "Không parse được JSON từ Ollama | line=%s column=%s position=%s "
            "| raw_response_preview=%r",
            error.lineno,
            error.colno,
            error.pos,
            preview,
        )
        raise ValueError("Ollama trả về nội dung không phải JSON hợp lệ") from error


def call_ollama(prompt: str, output_schema: dict) -> dict:
    request_body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "format": output_schema,
        "options": {"temperature": 0},
        "prompt": prompt,
    }
    last_error = None
    last_payload = {}

    for attempt in range(1, OLLAMA_MAX_ATTEMPTS + 1):
        response = requests.post(
            OLLAMA_URL,
            json=request_body,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        LOGGER.info(
            "Ollama HTTP %s | model=%s | attempt=%s/%s | response_bytes=%s",
            response.status_code,
            OLLAMA_MODEL,
            attempt,
            OLLAMA_MAX_ATTEMPTS,
            len(response.content),
        )
        response.raise_for_status()

        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as error:
            preview = response.text[:OLLAMA_LOG_PREVIEW_CHARS]
            LOGGER.error("Ollama trả về HTTP body không phải JSON: %r", preview)
            last_error = ValueError("Ollama trả về HTTP body không phải JSON")
            LOGGER.warning(
                "%s | attempt=%s/%s",
                last_error,
                attempt,
                OLLAMA_MAX_ATTEMPTS,
            )
            continue

        last_payload = payload
        LOGGER.info(
            "Ollama hoàn tất | done=%s | reason=%s | prompt_tokens=%s | "
            "output_tokens=%s",
            payload.get("done"),
            payload.get("done_reason"),
            payload.get("prompt_eval_count"),
            payload.get("eval_count"),
        )

        try:
            return parse_ollama_payload(payload)
        except ValueError as error:
            last_error = error
            LOGGER.warning(
                "%s | attempt=%s/%s | done=%s | reason=%s | thinking_chars=%s",
                error,
                attempt,
                OLLAMA_MAX_ATTEMPTS,
                payload.get("done"),
                payload.get("done_reason"),
                len(payload.get("thinking") or ""),
            )

    message = (
        f"Ollama không trả về JSON hợp lệ sau {OLLAMA_MAX_ATTEMPTS} lần thử: "
        f"{last_error}"
    )
    LOGGER.error(
        "%s | done=%s | reason=%s | thinking_chars=%s",
        message,
        last_payload.get("done"),
        last_payload.get("done_reason"),
        len(last_payload.get("thinking") or ""),
    )
    raise ValueError(message) from last_error


def extract_knowledge(content: str, call_model=None) -> dict:
    prompt = f"""
Bạn trích xuất tri thức trực tiếp từ văn bản theo đúng JSON schema.

BƯỚC 1 - ENTITY CÓ TÊN
- Chỉ trả PERSON, ORGANIZATION, LOCATION có tên riêng và nhận diện toàn cục.
- Quốc gia, bang/tỉnh, thành phố, quận và địa danh là LOCATION.
- Công ty, cơ quan, ủy ban có tên riêng, câu lạc bộ và đội thể thao là ORGANIZATION.
- Tên giải đấu hoặc sự kiện không phải Entity.
- Không đưa mô tả chung vào entities: "a man", "Maryland man", "the victim",
  "a House panel", "Italian community". Chúng chỉ có thể là participant_text.
- Không dịch tên, không tạo tên không có trong văn bản, không lấy hashtag/handle.
- Mỗi Entity có local_id e1, e2... duy nhất. Alias cùng chủ thể dùng cùng
  canonical_name và type. Chỉ HIGH khi phân giải chắc chắn.

BƯỚC 2 - EVENT CÓ HÀNH ĐỘNG
- Không mặc định mỗi câu là Event. Không tạo Event chỉ cho thời gian, địa điểm,
  sự hiện diện hoặc bối cảnh. Mỗi Event phải có hành động rõ ràng.
- evidence_text là đoạn nguyên văn ngắn nhất trong văn bản chứng minh hành động.
- MEETING chỉ là gặp/họp. Nói, cảnh báo, phủ nhận, khuyến nghị là STATEMENT.
- Đẩy, đánh, tấn công là ASSAULT. Chết đuối là một DROWNING, không thêm DEATH
  trùng. Chỉ dùng RESIGNATION hoặc TRANSFER khi nói trực tiếp từ chức/chuyển giao.
- Thi đấu/giải đấu là SPORTS_EVENT. Loại Event trùng trong cùng Post.
- Taxonomy duy nhất: STATEMENT, MEETING, APPOINTMENT, APPROVAL, ELECTION,
  RESIGNATION, ARREST, ASSAULT, ACCIDENT, DEATH, DROWNING, INVESTIGATION,
  PROTEST, SPORTS_EVENT, TRANSFER, OTHER.
- Status: PLANNED nếu được lên lịch/dự định; ONGOING nếu đang diễn ra; COMPLETED
  nếu đã xảy ra/kết thúc rõ; REPORTED nếu nguồn thuật lại và không có trạng thái
  mạnh hơn; ALLEGED nếu là cáo buộc/chưa xác thực; UNKNOWN nếu thiếu thông tin.

BƯỚC 3 - PARTICIPANT
- Mỗi participant có đúng một trong entity_id hoặc participant_text; trường còn
  lại là null. Người/nhóm/cơ quan không tên dùng participant_text, không tạo Entity.
- Tên của chính giải đấu/sự kiện không phải participant và không được gán role LOCATION.
- Role duy nhất: ACTOR, TARGET, VICTIM, SPEAKER, SUBJECT, LOCATION,
  ORGANIZATION, PARTICIPANT. Chỉ dùng PARTICIPANT khi không xác định cụ thể hơn.

BƯỚC 4 - QUAN HỆ EVENT
- Chỉ APPROVES, CAUSES, ENABLES, PRECEDES, RELATED_TO và chỉ khi evidence_text
  nói trực tiếp quan hệ. Không suy ra nhân quả từ thứ tự câu hoặc đồng xuất hiện.
- Mọi reference phải trỏ tới local_id trong cùng JSON. Không có thì trả mảng rỗng.

Ví dụ sửa lỗi: "A Maryland man pushed another man ... The man drowned" tạo
ASSAULT và DROWNING với anonymous participants; khuyến nghị của House panel và
bình luận về Cubs là STATEMENT; một vụ seaplane crash lặp lại chỉ là một ACCIDENT.

Chỉ trả JSON đúng schema, không giải thích.

Văn bản:
```text
{content}
```
    """.strip()

    if call_model is None:
        call_model = call_ollama
    result = call_model(prompt, KNOWLEDGE_SCHEMA)
    return {
        "entities": result.get("entities", []),
        "events": result.get("events", []),
        "event_relations": result.get("event_relations", []),
    }


def extract_entities(content: str) -> list[dict]:
    """Compatibility wrapper for callers that only need named entities."""
    return extract_knowledge(content)["entities"]


def prepare_entity(entity: dict) -> dict | None:
    name = _clean_text(entity.get("name"))
    canonical_name = _clean_text(entity.get("canonical_name"))
    entity_type = classify_entity_type(entity)
    confidence = _enum_value(entity.get("resolution_confidence"), CONFIDENCE_LEVELS)

    if (
        not name
        or entity_type is None
        or "#" in name
        or "@" in name
        or "#" in canonical_name
        or "@" in canonical_name
    ):
        return None
    if confidence is None:
        confidence = "LOW"

    is_canonical = confidence == "HIGH" and bool(canonical_name)
    display_name = canonical_name if is_canonical else name
    normalized_name = normalize_name(display_name)
    if not normalized_name:
        return None

    return {
        "name": name,
        "display_name": display_name,
        "normalized_name": normalized_name,
        "search_name": make_search_name(normalized_name),
        "entity_type": entity_type,
        "confidence": confidence,
        "is_canonical": is_canonical,
    }
