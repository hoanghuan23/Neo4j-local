import json
import math
import re
import unicodedata
from functools import lru_cache
from groq import Groq
import requests

from knowledge_settings import (
    CONFIDENCE_LEVELS,
    COUNTRY_ENTITY_ALIASES,
    COUNTRY_NAME_FALLBACKS,
    ENTITY_TYPES,
    EVENT_NAME_PATTERN,
    GENERIC_ENTITY_EXACT,
    GENERIC_PERSON_OR_GROUP_SUFFIXES,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    GROQ_MAX_ATTEMPTS,
    KNOWLEDGE_SCHEMA,
    LOCATION_NAME_PATTERN,
    LOGGER,
    MAX_EVENTS_PER_POST,
    NULL_STRINGS,
    OLLAMA_CONTEXT_TOKENS,
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
        "keep_alive": 0,
        "think": False,
        "format": output_schema,
        "options": {
            "temperature": 0,
            "num_ctx": OLLAMA_CONTEXT_TOKENS,
        },
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
            "Ollama HTTP %s | model=%s | attempt=%s/%s | context_tokens=%s "
            "| prompt_chars=%s | response_bytes=%s",
            response.status_code,
            OLLAMA_MODEL,
            attempt,
            OLLAMA_MAX_ATTEMPTS,
            OLLAMA_CONTEXT_TOKENS,
            len(prompt),
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

def call_groq(prompt: str, output_schema: dict) -> dict:
    if not GROQ_API_KEY:
        raise ValueError("Chưa cấu hình GROQ_API_KEY trong .env")

    client = Groq(
        api_key = GROQ_API_KEY,
        timeout = GROQ_TIMEOUT_SECONDS,
    )

    last_error = None

    for attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                reasoning_effort="low",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "knowledge_extraction",
                        "strict": True,
                        "schema": output_schema,
                    },
                },
            )

            raw_response = response.choices[0].message.content
            if not raw_response:
                raise ValueError("Groq trả về content rỗng")

            LOGGER.info(
                "Groq hoàn tất | model=%s | attempt=%s/%s | "
                "prompt_tokens=%s | output_tokens=%s",

                GROQ_MODEL,
                attempt,
                GROQ_MAX_ATTEMPTS,
                getattr(response.usage, "prompt_tokens", None),
                getattr(response.usage, "completion_tokens", None)
            )

            return json.loads(raw_response)
        except Exception as error:
            last_error = error

            LOGGER.warning(
                "Groq lỗi | model=%s | attempt=%s/%s | error=%s",
                GROQ_MODEL,
                attempt,
                GROQ_MAX_ATTEMPTS,
                error
            )
    raise ValueError(
        f"Groq không trả về kết quả hợp lệ sau "
        f"{GROQ_MAX_ATTEMPTS} lần thử: {last_error}"
    ) from last_error

def extract_knowledge(content: str, call_model=None) -> dict:
    prompt = f"""
    Bạn trích xuất tri thức trực tiếp từ văn bản và CHỈ trả JSON đúng schema được yêu cầu.

MỤC TIÊU ƯU TIÊN
- Ưu tiên độ chính xác (precision) hơn số lượng (recall).
- Không bắt buộc phải tạo Entity hoặc Event.
- Output rỗng là kết quả hoàn toàn hợp lệ và thường xuyên xảy ra với caption mạng xã hội.
- Nếu không chắc một Entity/Event có hợp lệ hay không → BỎ QUA.
- Không suy diễn thông tin không được văn bản trực tiếp hỗ trợ.
- Không ép bất kỳ cụm từ nào vào Entity type hoặc Event type chỉ để làm đầy JSON.

======================================================================
BƯỚC 0 - HARD GATE: PHÂN BIỆT CHỦ ĐỀ / MÔ TẢ VỚI SỰ VIỆC CỤ THỂ
======================================================================

Trước khi trích xuất, đọc toàn bộ văn bản và xác định:

A. Văn bản có trực tiếp khẳng định/tường thuật/thông báo một sự việc cụ thể
   đã xảy ra, đang xảy ra hoặc đã được lên kế hoạch/dự kiến xảy ra không?

B. Sự việc đó có thể diễn đạt dưới dạng:
   "[ai/cái gì] đã/đang/sẽ [làm gì hoặc xảy ra chuyện gì]"
   mà KHÔNG cần tự suy diễn thêm thông tin không?

C. Văn bản có đang tường thuật sự việc đó, thay vì chỉ:
   - nêu chủ đề;
   - đặt tiêu đề/hook;
   - mô tả đặc điểm;
   - mô tả trạng thái/tình cảm/quan hệ;
   - nêu cảm xúc/nhận xét;
   - đặt câu hỏi;
   - đưa lời chúc;
   - đưa slogan;
   - ví von/giả định;
   - nói chung chung về một hành động hay khái niệm không?

Chỉ khi A = CÓ, B = CÓ và C = CÓ thì mới được tạo Event ở BƯỚC 2.

Nếu một trong A/B/C là KHÔNG hoặc KHÔNG CHẮC:
- không tạo Event từ nội dung đó;
- events có thể là [].

QUAN TRỌNG:
- Có động từ KHÔNG đồng nghĩa với có Event.
- Có cấu trúc CHỦ THỂ + ĐỘNG TỪ + ĐỐI TƯỢNG cũng KHÔNG đủ để tạo Event.
- Caption ngắn KHÔNG mặc định là Event.
- Tiêu đề/hook mô tả một chủ đề KHÔNG phải Event nếu không khẳng định occurrence cụ thể.
- Một trạng thái, đặc điểm hoặc quan hệ kéo dài không tự động là Event.

Ví dụ KHÔNG có Event:
- "Lợi ích của việc xinh xắn" → events = [].
- "Chị gái phũ miệng nhưng thương em" → events = [].
- "Khi bạn có một người bạn như thế này" → events = [].
- "Cái kết của việc ngủ quên" → nếu chỉ là hook và không kể sự việc cụ thể → events = [].
- "Con gái khi yêu" → events = [].
- "Một ngày làm nhân viên văn phòng" → nếu chỉ là chủ đề/caption chung → events = [].
- "Những lợi ích không ai nói cho bạn biết" → events = [].
- "Đúng là cuộc sống" → events = [].
- "Anh em thấy thế nào?" → events = [].
- "Bộ phim hay nhất mình từng xem" → events = [].
- "Bộ phim mà mình cực mong chờ phần 2 mà chưa thấy, bác nào biết phim tương tự k ạ"
  → events = [].

Ví dụ CÓ Event:
- "Nam thanh niên lao xe xuống mương sáng nay"
  → một sự việc cụ thể được tường thuật → tạo ACCIDENT nếu phù hợp.
- "Cô gái tát bạn trai giữa quán cà phê"
  → một hành vi cụ thể → tạo ASSAULT.
- "Apple ra mắt iPhone 18 hôm nay"
  → một hành động cụ thể → tạo Event phù hợp.
- "Công an Hà Nội bắt giữ hai nghi phạm"
  → một hành động bắt giữ cụ thể → ARREST.
- "Manchester United đánh bại Arsenal 2-1"
  → một kết quả thi đấu cụ thể → SPORTS_EVENT.

======================================================================
BƯỚC 1 - ENTITY CÓ TÊN
======================================================================

Chỉ tạo Entity cho một ĐỐI TƯỢNG CỤ THỂ có tên riêng hoặc tên/định danh rõ ràng
và thuộc đúng một trong các type:

PERSON, ORGANIZATION, LOCATION, PRODUCT, SOFTWARE, EVENT, MEDIA, VEHICLE.

HARD GATE CHO ENTITY:

Trước khi tạo MỖI Entity, bắt buộc kiểm tra:

1. Cụm từ này có chỉ một đối tượng cụ thể có danh tính/định danh riêng không?
2. Nó có thực sự thuộc một trong tám Entity type được phép không?

Nếu câu trả lời cho một trong hai là KHÔNG hoặc KHÔNG CHẮC → bỏ qua.

Tuyệt đối không chọn type "gần nhất" chỉ để giữ một cụm từ trong entities.
Không có type phù hợp → KHÔNG tạo Entity.

Các khái niệm/danh từ chung/trạng thái/đặc điểm/chủ đề sau KHÔNG tự động là Entity:
- lợi ích;
- vẻ đẹp;
- xinh xắn;
- tình yêu;
- tình bạn;
- cuộc sống;
- công việc;
- hạnh phúc;
- nỗi buồn;
- sức khỏe;
- kiến thức;
- kinh nghiệm;
- bí quyết;
- lý do;
- kết quả;
- xu hướng;
- drama;
- câu chuyện;
- khoảnh khắc;
- giá dầu;
- giá vàng;
- lãi suất;
- chứng khoán;
- trí tuệ nhân tạo.

Danh sách trên chỉ là ví dụ, không phải blacklist đầy đủ.
Mọi khái niệm tương tự cũng phải bị loại nếu không có danh tính riêng.

Ví dụ:
"Lợi ích của việc xinh xắn"
→ "Lợi ích" KHÔNG phải Entity.
→ "xinh xắn" KHÔNG phải Entity.
→ entities = [].

-------------------------
PERSON
-------------------------
Người có tên riêng hoặc danh tính xác định.

Ví dụ:
- "Donald Trump"
- "Lionel Messi"
- "Nguyễn Văn A"

Tên đi kèm kính ngữ/chức danh vẫn là PERSON nhưng bỏ kính ngữ/chức danh khỏi
name và canonical_name.

Ví dụ:
"ông Đoàn Bảo Châu" → "Đoàn Bảo Châu".

Không tạo PERSON cho:
- "một người đàn ông";
- "nữ tài xế";
- "nạn nhân";
- "nghi phạm";
- "chị gái";
- "em trai";
nếu không có danh tính riêng.

Nếu các chủ thể vô danh này tham gia một Event hợp lệ thì xử lý ở participant_text.

-------------------------
ORGANIZATION
-------------------------
Tổ chức, công ty, cơ quan, ủy ban, trường học, bệnh viện, câu lạc bộ,
đội thể thao hoặc đơn vị có tên riêng.

Ví dụ:
- "Apple"
- "FIFA"
- "Manchester United"
- "Bộ Công an"

Danh từ chung như:
- "tòa án";
- "cảnh sát";
- "cơ quan chức năng";
- "lực lượng tìm kiếm"
không phải Entity nếu không có tên/định danh riêng.

-------------------------
LOCATION
-------------------------
Quốc gia, vùng lãnh thổ, bang/tỉnh, thành phố, quận/huyện, xã/phường,
địa danh hoặc địa điểm địa lý có tên riêng.

Ví dụ:
- "Việt Nam"
- "Hà Nội"
- "Quảng Ninh"
- "Hoành Mô"

LOCATION vẫn phải được lấy khi là nơi xuất phát, điểm đến, tuyến đường
hoặc nơi Event xảy ra.

Không tạo LOCATION cho mô tả chung như:
- "quán cà phê";
- "bệnh viện";
- "ngoài đường";
nếu không có tên riêng.

-------------------------
PRODUCT
-------------------------
Sản phẩm hoặc dòng sản phẩm có tên riêng/model xác định.

Ví dụ:
- "iPhone 16 Pro"
- "PlayStation 5"
- "Galaxy S26"

Không tạo PRODUCT cho danh từ chung như:
- "điện thoại";
- "máy tính";
- "tai nghe";
- "xe"
nếu không có tên/model cụ thể.

-------------------------
SOFTWARE
-------------------------
Phần mềm, hệ điều hành, ứng dụng, nền tảng phần mềm hoặc phiên bản phần mềm
có tên định danh rõ ràng.

Ví dụ:
- "iOS"
- "iOS 27"
- "Android 16"
- "Windows 11"
- "Photoshop"

Phiên bản cụ thể có thể là Entity riêng nếu văn bản thực sự nói trực tiếp
về phiên bản đó.

Ví dụ "iOS" và "iOS 27" chỉ tạo riêng khi chúng thực sự được đề cập như
hai đối tượng khác nhau.

Không tự suy diễn phiên bản không xuất hiện trong văn bản.

-------------------------
EVENT ENTITY
-------------------------
EVENT Entity CHỈ là TÊN/ĐỊNH DANH RIÊNG của một sự kiện, giải đấu,
chiến dịch, hội nghị hoặc chương trình có danh tính tồn tại độc lập
và có thể được nhắc lại giữa nhiều bài viết.

Ví dụ hợp lệ:
- "World Cup 2026"
- "SEA Games 33"
- "WWDC 2026"
- "Olympic Games"

EVENT Entity KHÔNG phải là mọi hành động/sự việc được trích xuất ở BƯỚC 2.

Phân biệt bắt buộc:

EVENT Entity
= tên của một sự kiện có danh tính riêng.

Event ở BƯỚC 2
= một occurrence/hành động/sự việc cụ thể được văn bản tường thuật.

Ví dụ:
"Lợi ích"
→ KHÔNG phải EVENT Entity.

"tai nạn"
→ KHÔNG tự động là EVENT Entity.

"vụ bắt giữ"
→ KHÔNG tự động là EVENT Entity.

"World Cup 2026"
→ có thể là EVENT Entity.

TUYỆT ĐỐI không dùng EVENT như type dự phòng khi không biết gán Entity type nào.

-------------------------
MEDIA
-------------------------
Tác phẩm truyền thông có tên riêng như phim, series, chương trình truyền hình,
sách, bài hát, album, trò chơi điện tử hoặc tác phẩm tương tự.

Ví dụ:
- "Squid Game"
- "Avengers: Endgame"
- "Grand Theft Auto VI"

Không tạo MEDIA cho cụm chung như:
- "bộ phim";
- "bài hát";
- "cuốn sách".

-------------------------
VEHICLE
-------------------------
Phương tiện hoặc mẫu phương tiện có tên/model xác định.

Ví dụ:
- "Tesla Model 3"
- "Boeing 787"
- "Honda SH 160i"

Không tạo VEHICLE cho mô tả chung như:
- "chiếc xe";
- "xe máy";
- "máy bay";
- "ô tô".

-------------------------
QUY TẮC ENTITY CHUNG
-------------------------

- Không tạo Entity cho ngày, tháng, năm, thứ hoặc biểu thức thời gian.
  Ví dụ: "ngày 7", "tháng 8", "hôm nay", "sáng 02/8".
  Thông tin thời gian chỉ đưa vào time_expression của Event khi phù hợp.

- Không tạo Entity cho số tiền, giá cả, số lượng, tỷ lệ hoặc chỉ số.
  Ví dụ: "35 triệu đồng", "10 điểm", "5%", "giá dầu tăng 8%".

- Không đưa mô tả chung vào entities.
  Ví dụ:
  "a man", "Maryland man", "the victim", "nữ tài xế",
  "nghi phạm", "tòa án", "lực lượng chức năng".

- Không lấy hashtag hoặc handle làm Entity chỉ vì chúng xuất hiện trong bài.

- Không dịch tên Entity.

- Không tạo tên không xuất hiện hoặc không thể suy ra chắc chắn từ văn bản.

- Khi một câu chứa nhiều Entity lồng nhau, lấy từng Entity có tên riêng.
  Ví dụ:
  "Đội X (Cục Y, Bộ Z)" → 3 ORGANIZATION nếu cả ba là tên xác định.
  "xã A (tỉnh B)" → 2 LOCATION.

- Một chủ thể xuất hiện nhiều lần chỉ tạo một Entity.
  Alias/cách viết khác của cùng chủ thể dùng cùng canonical_name,
  type và cùng local_id khi có thể phân giải chắc chắn.

- Không gộp hai đối tượng khác nhau chỉ vì tên giống nhau hoặc có quan hệ
  phiên bản/sản phẩm.

Ví dụ:
"Apple" → ORGANIZATION
"iPhone 16" → PRODUCT
"iOS" → SOFTWARE
"iOS 27" → SOFTWARE
"World Cup 2026" → EVENT

Đây là các Entity khác nhau.

- Mỗi Entity có local_id duy nhất.
- Khuyến nghị namespace Entity là e1, e2, e3...
- Chỉ đặt resolution_confidence = HIGH khi việc nhận diện và phân giải
  chủ thể thực sự chắc chắn.
- resolution_confidence = HIGH KHÔNG thể biến một Entity không hợp lệ
  thành Entity hợp lệ.

Trước khi sang BƯỚC 2, rà toàn bộ văn bản để lấy đủ các Entity có tên
thuộc tám type trên, kể cả Entity chỉ xuất hiện một lần hoặc không tham gia Event.

======================================================================
BƯỚC 2 - EVENT / OCCURRENCE
======================================================================

Event ở bước này là một HÀNH ĐỘNG, BIẾN CỐ hoặc THAY ĐỔI TRẠNG THÁI CỤ THỂ
được văn bản trực tiếp tường thuật/khẳng định/thông báo.

Mặc định ưu tiên:
events = []

Chỉ tạo Event khi có bằng chứng rõ ràng rằng một sự việc cụ thể:
- đã xảy ra;
- đang xảy ra;
- hoặc đã được lên kế hoạch/dự kiến xảy ra.

Trước khi tạo MỖI Event, bắt buộc áp dụng lại EVENT HARD GATE:

1. Có occurrence cụ thể không?
2. Văn bản có trực tiếp tường thuật/khẳng định occurrence đó không?
3. Evidence có đủ để mô tả occurrence mà không suy diễn không?

Nếu bất kỳ câu nào = KHÔNG hoặc KHÔNG CHẮC → KHÔNG tạo Event.

KHÔNG tạo Event nếu hành động/sự việc chỉ xuất hiện trong:
- câu nói mang tính cảm xúc, suy ngẫm hoặc hồi tưởng chung;
- câu trích dẫn không nhằm thông báo một sự việc cụ thể;
- thành ngữ;
- ví von;
- giả định;
- ví dụ;
- mong muốn;
- sở thích;
- câu hỏi;
- lời chúc;
- slogan;
- hook/caption chung;
- mô tả đặc điểm;
- trạng thái/tình cảm/quan hệ;
- mô tả chung không xác định occurrence cụ thể;
- thông tin nền chỉ dùng để bổ nghĩa cho chủ thể.

Không loại Event chỉ vì nội dung là tiêu đề hoặc caption ngắn.
Nếu caption trực tiếp tường thuật một occurrence cụ thể thì vẫn tạo Event.

Ví dụ:
"Chủ cửa hàng tạp hoá đổ nước vào chai cũ bán trước cổng bệnh viện"
→ trực tiếp tường thuật một hành vi cụ thể.
→ tạo Event OTHER.
→ "chủ cửa hàng tạp hoá" là anonymous participant ACTOR.

Ví dụ:
"Hoàng Thuỳ Linh: 'Ngày con sinh ra đời là ngày mẹ vất vả biết bao'"
→ lời nói/cảm xúc, không phải bài viết đang tường thuật một ca sinh cụ thể.
→ không tạo Event sinh nở.

Một câu có thể chứa nhiều Event nếu có nhiều hành động độc lập.

Không tách thành nhiều Event nếu nhiều câu/mô tả chỉ bổ sung chi tiết
cho cùng một hành động.

Tối đa {MAX_EVENTS_PER_POST} Event cho toàn bộ văn bản.

Nhận diện cả từ bị chèn ký tự để né kiểm duyệt:
- "đ/ánh" = "đánh"
- "b/ắn" = "bắn"

Các cụm như:
- "video ghi lại";
- "hình ảnh cho thấy";
- "xác minh video"
không làm mất Event thực sự được mô tả bên trong.

Việc có chủ thể thực hiện hành động và đối tượng chịu tác động
KHÔNG phải điều kiện đủ để tạo Event.

Chỉ SAU KHI xác định câu thực sự mô tả Event thì chủ thể không có tên riêng
mới được giữ bằng participant_text.

description:
- phải tự đầy đủ;
- giữ các chi tiết quan trọng được nói trực tiếp;
- có thể giữ số tiền, mức phạt, số điểm, số lượng, khoảng cách,
  thời hạn, kết quả và hậu quả;
- không suy diễn.

Nếu câu kế tiếp chỉ bổ sung chi tiết cho Event trước → gộp vào cùng Event.

evidence_text:
- là đoạn nguyên văn ngắn nhất đủ chứng minh Event;
- phải chứa bằng chứng trực tiếp về hành động/biến cố;
- có thể dùng nhiều câu liền kề nếu cần;
- không được chọn một cụm chỉ mô tả chủ đề rồi suy diễn thành Event.

Taxonomy Event duy nhất:

STATEMENT, MEETING, APPOINTMENT, APPROVAL, ELECTION, RESIGNATION,
ARREST, ASSAULT, ACCIDENT, DEATH, DROWNING, INVESTIGATION,
PROTEST, SPORTS_EVENT, TRANSFER, OTHER.

Quy tắc:
- MEETING chỉ dùng cho gặp/họp.
- Đánh, đẩy, tấn công, hất/tạt/ném vào người → ASSAULT.
- Chết đuối → DROWNING; không tạo thêm DEATH cho cùng hành động.
- Thi đấu, trận đấu hoặc diễn biến/kết quả của giải đấu → SPORTS_EVENT.
- RESIGNATION và TRANSFER chỉ dùng khi văn bản trực tiếp nói về
  từ chức hoặc chuyển giao/chuyển nhượng.
- OTHER chỉ dùng khi occurrence đã vượt qua EVENT HARD GATE
  nhưng không phù hợp taxonomy cụ thể hơn.
- TUYỆT ĐỐI không dùng OTHER để biến một caption/chủ đề không phải Event
  thành Event.

Status phản ánh trạng thái Event:
- PLANNED: chưa xảy ra nhưng đã được lên lịch/dự kiến.
- ONGOING: đang diễn ra.
- COMPLETED: đã xảy ra hoặc kết thúc.
- ALLEGED: chỉ là cáo buộc/chưa xác thực.
- REPORTED: chỉ dùng khi văn bản nói sự việc được báo cáo nhưng không xác định
  được trạng thái mạnh hơn.
- UNKNOWN: không đủ thông tin.

======================================================================
BƯỚC 3 - PARTICIPANT
======================================================================

Chỉ tạo participant cho một Event đã vượt qua EVENT HARD GATE.

Mỗi participant có đúng một trong:
- entity_id
hoặc
- participant_text.

Trường còn lại phải là null.

- entity_id chỉ được tham chiếu local_id có thật trong entities.
- Participant có tên riêng phải dùng entity_id và participant_scope = null.
- Participant không có tên riêng phải dùng:
  entity_id = null
  participant_text = nguyên văn cụm mô tả trong bài
  participant_scope = GLOBAL_ROLE hoặc POST_LOCAL.

GLOBAL_ROLE:
Chỉ dùng khi participant_text biểu thị một vai trò/chức danh/nhóm chức năng
chung có thể được dùng lại giữa nhiều bài viết và không đại diện cho
danh tính của một cá nhân cụ thể.

Ví dụ:
- "Đại biểu quốc hội"
- "lực lượng chức năng"
- "cơ quan chức năng"
- "công an"
- "lực lượng công an"

Tên cơ quan xác định như "Công an Hà Nội" vẫn là Entity ORGANIZATION.

POST_LOCAL:
Dùng khi participant_text nói tới một người hoặc nhóm vô danh cụ thể
trong ngữ cảnh bài hiện tại.

Ví dụ:
- "một người đàn ông"
- "nữ tài xế"
- "nạn nhân"
- "nghi phạm"
- "cụ bà 89 tuổi"

Khi không chắc có thể dùng chung giữa nhiều bài → luôn chọn POST_LOCAL.

Không gộp hai chủ thể khác nhau vào một participant.

Phải đưa đủ các chủ thể chính của Event vào participants khi văn bản có nêu.

Role phản ánh vai trò trong Event, không phản ánh Entity.type.

Role duy nhất:
ACTOR, TARGET, VICTIM, SPEAKER, SUBJECT, LOCATION, PARTICIPANT.

Ý nghĩa:
- ACTOR: chủ thể thực hiện hành động.
- TARGET: đối tượng mà hành động hướng tới.
- VICTIM: người/nhóm chịu thiệt hại, thương tích hoặc hành vi gây hại.
- SPEAKER: người/tổ chức phát ngôn trong STATEMENT.
- SUBJECT: chủ đề/chủ thể được nói tới.
- LOCATION: địa điểm thực nơi Event xảy ra, bắt đầu, kết thúc hoặc đi qua.
- PARTICIPANT: chỉ dùng khi không xác định được role cụ thể hơn.

Tên của chính giải đấu hoặc EVENT Entity không phải LOCATION participant.

======================================================================
BƯỚC 4 - EVENT RELATION
======================================================================

Chỉ dùng:
APPROVES, CAUSES, ENABLES, PRECEDES, RELATED_TO.

- APPROVES:
  Event nguồn trực tiếp thể hiện sự phê duyệt Event đích.

- CAUSES:
  Văn bản trực tiếp nói Event nguồn gây ra Event đích.

- ENABLES:
  Event nguồn tạo điều kiện cho Event đích xảy ra.

- PRECEDES:
  Văn bản nói rõ Event nguồn xảy ra trước Event đích.

- RELATED_TO:
  Văn bản nói rõ hai Event có liên hệ nhưng không phù hợp loại trên.

Chỉ tạo relation khi evidence_text trực tiếp chứng minh quan hệ.

Không suy ra relation chỉ vì hai Event:
- cùng nằm trong một Post;
- cùng Entity;
- cùng chủ đề;
- xuất hiện gần nhau trong văn bản.

Mọi source_event_id và target_event_id phải tham chiếu Event tồn tại trong JSON.

Nếu schema cho phép, nên dùng namespace Event riêng như:
ev1, ev2, ev3...
để tránh nhầm với Entity e1, e2, e3.

Nếu schema hiện tại bắt buộc format local_id khác thì tuân theo schema hiện tại,
nhưng mọi ID vẫn phải tham chiếu đúng loại đối tượng.

======================================================================
BƯỚC 5 - SEMANTIC VALIDATION
======================================================================

Trước schema validation, bắt buộc kiểm tra NGỮ NGHĨA.

ENTITY VALIDATION:

Với từng Entity:
1. Nó có tên/định danh riêng không?
2. Nó có thực sự thuộc Entity type đã gán không?
3. Nếu type = EVENT:
   - nó có phải TÊN của một sự kiện có danh tính độc lập không?

Nếu không → XÓA Entity đó.

Đặc biệt loại:
- danh từ chung;
- khái niệm;
- đặc điểm;
- cảm xúc;
- trạng thái;
- quan hệ;
- chủ đề;
- cụm hook/caption;
nếu chúng không phải đối tượng có tên/định danh riêng.

Ví dụ:
{{
  "name": "Lợi ích",
  "type": "EVENT"
}}
→ SAI → phải loại.

resolution_confidence = HIGH không phải lý do để giữ một Entity sai.

EVENT VALIDATION:

Với từng Event:
1. evidence_text có trực tiếp chứng minh occurrence không?
2. description có mô tả một occurrence cụ thể không?
3. Event có vượt qua A/B/C của EVENT HARD GATE không?
4. Có phải model chỉ biến chủ đề/caption thành Event OTHER không?

Nếu bất kỳ điều nào không đạt → XÓA Event.

Nếu Event bị xóa:
- xóa các event_relation tham chiếu Event đó;
- không giữ participant chỉ tồn tại cho Event đã bị xóa.

OTHER không phải fallback cho nội dung không phải Event.

Ví dụ sai:
description = "Lợi ích của việc xinh xắn"
type = OTHER
→ đây là chủ đề/khái niệm, không phải occurrence.
→ XÓA Event.

======================================================================
BƯỚC 6 - STRUCTURAL / SCHEMA VALIDATION
======================================================================

Trước khi trả JSON, kiểm tra:

1. Mọi Entity local_id là duy nhất.
2. Mọi Event local_id là duy nhất.
3. Không để ID gây tham chiếu nhầm giữa Entity và Event.
4. Mọi participant.entity_id tồn tại trong entities.
5. Participant không tên riêng luôn dùng participant_text và có
   participant_scope phù hợp.
6. Participant dùng entity_id phải có participant_scope = null.
7. Không dùng Entity khác thay cho anonymous participant.
8. Tên giải đấu/sự kiện có tên riêng có thể là Entity type EVENT,
   nhưng không được dùng làm LOCATION participant chỉ vì nó là Event.
9. Mọi event_relation chỉ tham chiếu Event tồn tại.
10. Không có field ngoài JSON schema.
11. Không tạo object rỗng/placeholder chỉ để đủ cấu trúc.
12. Nếu không có Entity → entities = [].
13. Nếu không có Event → events = [].
14. Nếu không có relation → event_relations = [].

======================================================================
VÍ DỤ TỔNG HỢP
======================================================================

INPUT:
"Lợi ích của việc xinh xắn"

OUTPUT:
{{
  "entities": [],
  "events": [],
  "event_relations": []
}}

Lý do nội bộ:
- "Lợi ích" là khái niệm chung, không phải Entity.
- "xinh xắn" là đặc điểm, không phải Entity.
- không có occurrence cụ thể.
KHÔNG đưa phần lý do này vào JSON output.

---

INPUT:
"Chị gái phũ miệng nhưng thương em #muahenamay #vfc #trend"

OUTPUT:
{{
  "entities": [],
  "events": [],
  "event_relations": []
}}

Lý do nội bộ:
- "chị gái" và "em" không có tên riêng;
- "phũ miệng" là đặc điểm;
- "thương" trong ngữ cảnh này biểu thị trạng thái/tình cảm/quan hệ,
  không phải occurrence cụ thể được tường thuật;
- hashtag không tự động là Entity.
KHÔNG đưa phần lý do này vào JSON output.

---

INPUT:
"Cơ quan A đã xử phạt một nữ tài xế. Mức xử phạt là 35 triệu đồng và trừ 10 điểm giấy phép lái xe."

→ một Event nếu "Cơ quan A" là tên/định danh tổ chức hợp lệ theo ngữ cảnh/schema;
→ "nữ tài xế" = anonymous participant TARGET;
→ description giữ 35 triệu đồng và trừ 10 điểm;
→ không tạo Entity cho "35 triệu đồng" hoặc "10 điểm".

---

INPUT:
"A Maryland man pushed another man ... The man drowned."

→ ASSAULT và DROWNING nếu văn bản trực tiếp tường thuật hai occurrence;
→ anonymous participants;
→ không tạo PERSON Entity chỉ từ "a Maryland man" hoặc "another man".

======================================================================
QUY TẮC CUỐI CÙNG
======================================================================

Trước khi trả kết quả, tự hỏi:

"Nếu bỏ schema sang một bên, văn bản có THỰC SỰ nói rằng sự việc này
đã/đang/sẽ xảy ra không?"

Nếu KHÔNG hoặc KHÔNG CHẮC → không tạo Event.

Và với mỗi Entity, tự hỏi:

"Đây có THỰC SỰ là một đối tượng có tên/định danh riêng không?"

Nếu KHÔNG hoặc KHÔNG CHẮC → không tạo Entity.

Ưu tiên bỏ sót một trường hợp mơ hồ hơn là tạo Entity/Event sai.

Chỉ trả JSON đúng schema, không giải thích, không markdown.
Văn bản:
``` text
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
