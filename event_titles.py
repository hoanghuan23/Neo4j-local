import re
from collections.abc import Callable

from knowledge_settings import EVENT_TITLE_SCHEMA, LOGGER


TITLE_MIN_WORDS = 10
TITLE_MAX_WORDS = 25


def normalize_event_title(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def event_title_word_count(value: object) -> int:
    title = normalize_event_title(value)
    return len(title.split(" ")) if title else 0


def is_valid_event_title(value: object) -> bool:
    count = event_title_word_count(value)
    return TITLE_MIN_WORDS <= count <= TITLE_MAX_WORDS


def _repair_prompt(description: str) -> str:
    return f"""
Chỉ trả JSON đúng schema với một title tiếng Việt cho Event.

Quy tắc bắt buộc:
- Title dài từ {TITLE_MIN_WORDS} đến {TITLE_MAX_WORDS} từ.
- Ưu tiên cấu trúc: [chủ thể] + [hành động chính] + [đối tượng]
  + [địa điểm nếu có] + [thời gian nếu có].
- Chỉ sử dụng thông tin được hỗ trợ trực tiếp bởi description.
- Không thêm thời gian hoặc địa điểm nếu description không xác định rõ.
- Không đưa chi tiết phụ, nguyên nhân, bình luận hoặc trạng thái điều tra vào title.
- Không suy diễn.

description:
{description}
    """.strip()


def resolve_event_title(
    description: object,
    proposed_title: object,
    call_model: Callable,
) -> tuple[str, bool]:
    """Return a valid title, or description fallback marked for backfill."""
    clean_description = normalize_event_title(description)
    clean_title = normalize_event_title(proposed_title)
    if is_valid_event_title(clean_title):
        return clean_title, False

    try:
        repaired = call_model(
            _repair_prompt(clean_description),
            EVENT_TITLE_SCHEMA,
        )
        clean_title = normalize_event_title(repaired.get("title"))
        if is_valid_event_title(clean_title):
            return clean_title, False
    except Exception:
        LOGGER.exception("Không thể sửa title Event không hợp lệ")

    return clean_description, True


def generate_event_title(
    description: object,
    call_model: Callable,
) -> tuple[str, bool]:
    """Generate a title with one retry before falling back to description."""
    clean_description = normalize_event_title(description)
    for _attempt in range(2):
        try:
            generated = call_model(
                _repair_prompt(clean_description),
                EVENT_TITLE_SCHEMA,
            )
            title = normalize_event_title(generated.get("title"))
            if is_valid_event_title(title):
                return title, False
        except Exception:
            LOGGER.exception("Không thể tạo title Event")
    return clean_description, True
