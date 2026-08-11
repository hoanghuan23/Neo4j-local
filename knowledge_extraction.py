import json
import math
import re
import unicodedata
from functools import lru_cache

import requests

from knowledge_settings import (
    CONFIDENCE_LEVELS,
    COUNTRY_ENTITY_ALIASES,
    COUNTRY_NAME_FALLBACKS,
    ENTITY_TYPES,
    EVENT_NAME_PATTERN,
    GENERIC_ENTITY_EXACT,
    GENERIC_PERSON_OR_GROUP_SUFFIXES,
    KNOWLEDGE_SCHEMA,
    LOCATION_NAME_PATTERN,
    LOGGER,
    MAX_EVENTS_PER_POST,
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


def _source_contains_name(content: str, name: str) -> bool:
    """Match a complete, explicitly written name after text normalization."""
    return (
        re.search(
            rf"(?<!\w){re.escape(normalize_name(name))}(?!\w)",
            normalize_name(content),
        )
        is not None
    )


def recover_explicit_country_entities(content: str, result: dict) -> dict:
    """Add configured countries that the model omitted despite direct evidence."""
    entities = result.get("entities")
    if not isinstance(entities, list):
        entities = []
        result["entities"] = entities

    known_names = {
        make_search_name(candidate)
        for entity in entities
        if isinstance(entity, dict)
        for candidate in (
            _clean_text(entity.get("name")),
            _clean_text(entity.get("canonical_name")),
        )
        if candidate
    }
    used_ids = {
        _clean_text(item.get("local_id"))
        for section in ("entities", "events")
        for item in result.get(section, [])
        if isinstance(item, dict) and _clean_text(item.get("local_id"))
    }

    next_id = 1
    for source_name, canonical_name in COUNTRY_ENTITY_ALIASES.items():
        identity = make_search_name(canonical_name)
        if identity in known_names or not _source_contains_name(content, source_name):
            continue
        while f"e{next_id}" in used_ids:
            next_id += 1
        local_id = f"e{next_id}"
        entities.append(
            {
                "local_id": local_id,
                "name": canonical_name,
                "canonical_name": canonical_name,
                "type": "LOCATION",
                "resolution_confidence": "HIGH",
            }
        )
        used_ids.add(local_id)
        known_names.add(identity)

    return result


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
- Với mỗi cụm từ, trước tiên phải xác nhận đó thực sự là tên riêng của một người,
  tổ chức hoặc địa điểm. Nếu không chắc cụm từ thuộc một trong ba nhóm này thì
  bỏ qua; tuyệt đối không chọn LOCATION chỉ để khớp JSON schema.
- Không tạo Entity cho ngày, tháng, thứ, năm hoặc biểu thức thời gian, ví dụ:
  "ngày 7", "tháng 8", "thứ 6", "hôm nay", "sáng 02/8". Chúng chỉ có thể
  xuất hiện trong time_expression của một Event thực tế. Nếu bài chỉ thông báo
  ngày tháng mà không có Event thực tế thì trả entities và events đều là [].
- Không tạo Entity cho giá cả, hàng hóa, chỉ số kinh tế hoặc chủ đề tin tức
  chung, ví dụ: "giá dầu", "giá vàng", "giá xăng", "tỷ giá", "lãi suất",
  "chứng khoán". Đây không phải PERSON, ORGANIZATION hay LOCATION.
- Ví dụ "Đọc nhanh 7-8: Giá dầu tăng mạnh; giá vàng biến động ra sao?" không
  chứa Entity có tên, vì vậy entities là []. Ví dụ "Hôm nay là thứ 6, ngày 7,
  tháng 8" cũng phải có entities là [] và events là [].
- Trước khi tạo Event, rà lần lượt từng câu trong toàn bộ văn bản để lấy đủ mọi
  PERSON, ORGANIZATION và LOCATION có tên riêng, kể cả Entity chỉ xuất hiện một
  lần hoặc không tham gia Event. Ưu tiên không bỏ sót Entity có bằng chứng trực
  tiếp trong văn bản.
- Quốc gia, bang/tỉnh, thành phố, quận và địa danh là LOCATION.
- LOCATION vẫn phải được lấy khi nó chỉ hướng di chuyển, nơi xuất phát, điểm đến
  hoặc tuyến đường của Event. Ví dụ câu "vận chuyển từ nước ngoài vào Việt Nam
  qua khu vực biên giới Hoành Mô (Quảng Ninh)" bắt buộc có đủ ba LOCATION riêng:
  "Việt Nam", "Hoành Mô" và "Quảng Ninh"; không được chỉ lấy địa điểm gần động
  từ nhất.
- Công ty, cơ quan, ủy ban có tên riêng, câu lạc bộ và đội thể thao là ORGANIZATION.
- Khi một câu chứa tên lồng nhau, vẫn lấy từng tổ chức hoặc địa điểm có tên riêng.
  Ví dụ "Đội X (Cục Y, Bộ Z)" có thể chứa ba ORGANIZATION; "xã A (tỉnh B)"
  chứa hai LOCATION. Không bỏ qua đơn vị cấp trên, cấp dưới hoặc địa danh nằm
  trong ngoặc.
- Tên giải đấu hoặc sự kiện không phải Entity.
- Tên người đi kèm kính ngữ/chức danh vẫn bắt buộc là PERSON, kể cả chỉ xuất
  hiện một lần. Ví dụ "ông Đoàn Bảo Châu", "ông Nguyễn Văn Nhỏ", "bà Nhật Kim Anh" phải tạo Entity có name và
  canonical_name là "Đoàn Bảo Châu"; không đưa "ông/bà" vào tên chuẩn.
- Một người có tên xuất hiện nhiều lần chỉ tạo một Entity. Mọi lần người đó tham
  gia Event đều phải tham chiếu cùng local_id của Entity này.
- Không đưa mô tả chung vào entities: "a man", "Maryland man", "the victim",
  "a House panel", "Italian community". Chúng chỉ có thể là participant_text.
- Không dịch tên, không tạo tên không có trong văn bản, không lấy hashtag/handle.
- Mỗi Entity có local_id e1, e2... duy nhất. Alias cùng chủ thể dùng cùng
  canonical_name và type. Chỉ HIGH khi phân giải chắc chắn.

BƯỚC 2 - EVENT CÓ HÀNH ĐỘNG
- Event là một sự kiện thực tế đã xảy ra, đang xảy ra, hoặc được dự kiến sẽ xảy
  ra trong thế giới thực, có hành động hoặc thay đổi trạng thái rõ ràng.
- KHÔNG tạo Event cho cảm xúc, mong muốn, sở thích; câu hỏi, yêu cầu hoặc gợi ý;
  việc đề cập chung đến một người/vật; hay hành động hội thoại như hỏi, mong chờ,
  nhắc đến. Với nội dung chỉ thuộc các nhóm này, trả events là [].
- Không mặc định mỗi câu là Event. Không tạo Event chỉ cho thời gian, địa điểm,
  sự hiện diện hoặc bối cảnh.
- Tối đa {MAX_EVENTS_PER_POST} Event cho toàn bộ văn bản. Không tạo nhiều Event
  cho cùng một câu. Nếu nhiều mô tả cùng nói về một hành động thì gộp thành một
  Event.
- Phải nhận diện hành động kể cả từ bị chèn ký tự để né kiểm duyệt, 
  ví dụ "đ/ánh" = "đánh", "b/ắn" = "bắn".
- Các cụm như "xác minh video", "video ghi lại", "hình ảnh cho thấy" không làm mất sự kiện được mô tả bên trong nội dung
- Khi nội dung có chủ thể thực hiện hành động và đối tượng chịu tác động, bắt buộc phải tạo Event, kể cả khi chủ thể không có tên riêng.
- description phải là bản tóm tắt tự đầy đủ của Event. Ngoài hành động chính,
  phải giữ các chi tiết quan trọng được nói trực tiếp như số tiền, mức phạt, số
  điểm, số lượng, khoảng cách, thời hạn và hậu quả. Không bịa hoặc suy diễn chi tiết.
- Nếu câu kế tiếp bổ sung số tiền, số điểm, số lượng, khoảng cách hoặc hậu quả
  cho hành động ở câu trước, hãy gộp chi tiết đó vào cùng Event; không tạo Event
  riêng chỉ cho câu bổ sung và không bỏ chi tiết vì câu đó lược chủ ngữ.
- evidence_text là đoạn nguyên văn ngắn nhất chứng minh cả hành động và các chi
  tiết quan trọng trong description. Có thể lấy nhiều câu liền kề khi thông tin
  của cùng Event nằm ở các câu đó.
- MEETING chỉ là gặp/họp. Nói, cảnh báo, phủ nhận, khuyến nghị là STATEMENT.
- Đẩy, đánh, tấn công, hất/tạt/ném vào người là ASSAULT. Chết đuối là một
  DROWNING, không thêm DEATH trùng. Chỉ dùng RESIGNATION hoặc TRANSFER khi nói
  trực tiếp từ chức/chuyển giao.
- Thi đấu/giải đấu là SPORTS_EVENT. Loại Event trùng trong cùng Post.
- Taxonomy duy nhất: STATEMENT, MEETING, APPOINTMENT, APPROVAL, ELECTION,
  RESIGNATION, ARREST, ASSAULT, ACCIDENT, DEATH, DROWNING, INVESTIGATION,
  PROTEST, SPORTS_EVENT, TRANSFER, OTHER.
- Status: PLANNED nếu được lên lịch/dự định; ONGOING nếu đang diễn ra; COMPLETED
  nếu đã xảy ra/kết thúc rõ; REPORTED nếu nguồn thuật lại và không có trạng thái
  mạnh hơn; ALLEGED nếu là cáo buộc/chưa xác thực; UNKNOWN nếu thiếu thông tin.

BƯỚC 3 - PARTICIPANT
- Mỗi participant có đúng một trong entity_id hoặc participant_text; trường còn
  lại là null.
- Mọi entity_id trong participants bắt buộc phải trùng với local_id của một Entity
  đã tồn tại trong mảng entities. Tuyệt đối không tạo entity_id giả hoặc tham chiếu tới Entity không tồn tại.
- Người/nhóm/cơ quan không tên dùng participant_text và không tạo Entity. Ví dụ: "nữ tài xế", "cụ bà", "người bán",
  "vị khách", "nghi phạm", "lực lượng tìm kiếm", "tổ công tác"...
- Với participant không có tên riêng, bắt buộc đặt:
  entity_id = null
  participant_text = nguyên văn cụm mô tả xuất hiện trong bài
- Nếu participant là người có tên riêng, bắt buộc dùng entity_id trỏ tới Entity PERSON
  và đặt participant_text = null. Không bao giờ đặt họ tên đầy đủ như
  "ông Đoàn Bảo Châu" vào participant_text.
- Phải đưa đầy đủ các chủ thể chính của hành động vào participants
- Không gộp hai chủ thể khác nhau vào một participant. Nếu một tổ chức thực hiện
  hành động đối với một người không tên, tạo riêng participant dùng entity_id
  cho tổ chức và participant dùng participant_text cho người không tên.
- Role phản ánh vai trò trong hành động, không chỉ dựa vào loại đối tượng
- Tên của chính giải đấu hoặc sự kiện không phải participant và không được gán role LOCATION.
- Role duy nhất: ACTOR, TARGET, VICTIM, SPEAKER, SUBJECT, LOCATION,
  ORGANIZATION, PARTICIPANT. Chỉ dùng PARTICIPANT khi không xác định cụ thể hơn.

BƯỚC 4 - QUAN HỆ EVENT
- Chỉ APPROVES, CAUSES, ENABLES, PRECEDES, RELATED_TO và chỉ khi evidence_text
  nói trực tiếp quan hệ. Không suy ra nhân quả từ thứ tự câu hoặc đồng xuất hiện.
- Mọi reference phải trỏ tới local_id trong cùng JSON. Không có thì trả mảng rỗng.

BƯỚC 5 - KIỂM TRA ID TRƯỚC KHI TRẢ JSON
- Lập danh sách toàn bộ local_id thực sự có trong entities. Kiểm tra lại từng
  entity_id của participants; không được dùng bất kỳ ID nào ngoài danh sách đó.
- Nếu participant là cụm không có tên riêng như "lực lượng tìm kiếm" hoặc
  "tổ công tác", luôn sửa thành entity_id = null và participant_text = cụm
  nguyên văn. Không gán ID của một Entity khác chỉ vì Entity đó có trong bài.
- Ví dụ: nếu entities chỉ có e1 là "Mặt trận Dân tộc Giải phóng Miền Nam Việt Nam"
  nhưng actor là "lực lượng tìm kiếm", actor phải dùng entity_id = null,
  participant_text = "lực lượng tìm kiếm"; tuyệt đối không tạo tham chiếu e2.

Ví dụ sửa lỗi: "A Maryland man pushed another man ... The man drowned" tạo
ASSAULT và DROWNING với anonymous participants; khuyến nghị của House panel và
bình luận về Cubs là STATEMENT; một vụ seaplane crash lặp lại chỉ là một ACCIDENT.
Ví dụ: "Cơ quan A đã xử phạt một nữ tài xế. Mức xử phạt là 35 triệu đồng và
trừ 10 điểm giấy phép lái xe." phải tạo một Event có description giữ đủ mức phạt
35 triệu đồng và trừ 10 điểm; participants gồm Cơ quan A là ACTOR và
"nữ tài xế" là TARGET; evidence_text gồm hai câu liền kề này.
"Bộ phim mà mình cực mong chờ phần 2 mà chưa thấy, bác nào biết phim tương tự k ạ"
không có sự kiện thực tế, vì vậy trả events là [].

Chỉ trả JSON đúng schema, không giải thích.

Văn bản:
```text
{content}
```
    """.strip()

    if call_model is None:
        call_model = call_ollama
    result = call_model(prompt, KNOWLEDGE_SCHEMA)
    knowledge = {
        "entities": result.get("entities", []),
        "events": result.get("events", []),
        "event_relations": result.get("event_relations", []),
    }
    return recover_explicit_country_entities(content, knowledge)


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

    identity_names = list(
        dict.fromkeys(
            candidate
            for value in (name, canonical_name, display_name)
            if (candidate := normalize_name(value))
        )
    )

    return {
        "name": name,
        "display_name": display_name,
        "normalized_name": normalized_name,
        "identity_names": identity_names,
        "search_name": make_search_name(normalized_name),
        "entity_type": entity_type,
        "confidence": confidence,
        "is_canonical": is_canonical,
    }
