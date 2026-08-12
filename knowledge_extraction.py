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
Bạn trích xuất tri thức trực tiếp từ văn bản và chỉ trả JSON đúng schema.

BƯỚC 1 - ENTITY CÓ TÊN

- Chỉ tạo Entity cho đối tượng có tên riêng hoặc tên định danh rõ ràng và thuộc
  đúng một trong các type:
  PERSON, ORGANIZATION, LOCATION, PRODUCT, SOFTWARE, EVENT, MEDIA, VEHICLE.

- Trước khi tạo Entity, phải xác định type dựa trên bản chất của đối tượng.
  Tuyệt đối không ép một cụm từ vào type gần nhất chỉ để phù hợp JSON schema.
  Nếu không xác định hợp lý thuộc một trong tám type trên thì bỏ qua.

- PERSON:
  Người có tên riêng hoặc danh tính xác định.
  Ví dụ: "Donald Trump", "Lionel Messi", "Nguyễn Văn A".
  Tên đi kèm kính ngữ/chức danh vẫn là PERSON nhưng bỏ kính ngữ/chức danh khỏi
  name và canonical_name.
  Ví dụ: "ông Đoàn Bảo Châu" → "Đoàn Bảo Châu".

- ORGANIZATION:
  Tổ chức, công ty, cơ quan, ủy ban, trường học, bệnh viện, câu lạc bộ,
  đội thể thao hoặc đơn vị có tên riêng.
  Ví dụ: "Apple", "FIFA", "Manchester United", "Bộ Công an".
  Danh từ chung như "tòa án", "cảnh sát", "cơ quan chức năng",
  "lực lượng tìm kiếm" không phải Entity nếu không có tên riêng.

- LOCATION:
  Quốc gia, vùng lãnh thổ, bang/tỉnh, thành phố, quận/huyện, xã/phường,
  địa danh hoặc địa điểm địa lý có tên riêng.
  Ví dụ: "Việt Nam", "Hà Nội", "Quảng Ninh", "Hoành Mô".
  LOCATION vẫn phải được lấy khi là nơi xuất phát, điểm đến, tuyến đường
  hoặc nơi Event xảy ra.

- PRODUCT:
  Sản phẩm hoặc dòng sản phẩm có tên riêng.
  Ví dụ: "iPhone 16 Pro", "PlayStation 5", "Galaxy S26".
  Không tạo PRODUCT cho danh từ chung như "điện thoại", "máy tính",
  "tai nghe", "xe" nếu không có tên sản phẩm cụ thể.

- SOFTWARE:
  Phần mềm, hệ điều hành, ứng dụng, nền tảng phần mềm hoặc phiên bản phần mềm
  có tên định danh rõ ràng.
  Ví dụ: "iOS", "iOS 27", "Android 16", "Windows 11", "Photoshop".
  Phiên bản cụ thể có thể là Entity riêng nếu văn bản đang nói trực tiếp về
  phiên bản đó.
  Ví dụ "iOS" và "iOS 27" chỉ tạo riêng khi chúng thực sự được đề cập như
  hai đối tượng khác nhau; không tự suy diễn phiên bản không có trong văn bản.

- EVENT:
  Sự kiện, giải đấu, chiến dịch, hội nghị hoặc chương trình có tên riêng,
  có danh tính tồn tại độc lập và có thể được nhắc lại giữa nhiều bài viết.
  Ví dụ: "World Cup 2026", "SEA Games 33", "WWDC 2026", "Olympic Games".
  Không tạo EVENT Entity cho mọi hành động được trích xuất ở BƯỚC 2.
  Event Entity là TÊN của một sự kiện; Event ở BƯỚC 2 là một hành động/sự việc
  cụ thể được mô tả trong bài. Hai khái niệm này phải được phân biệt.

- MEDIA:
  Tác phẩm truyền thông có tên riêng như phim, series, chương trình truyền hình,
  sách, bài hát, album, trò chơi điện tử hoặc tác phẩm tương tự.
  Ví dụ: "Squid Game", "Avengers: Endgame", "Grand Theft Auto VI".
  Không tạo MEDIA cho cụm chung như "bộ phim", "bài hát", "cuốn sách".

- VEHICLE:
  Phương tiện hoặc mẫu phương tiện có tên/model xác định.
  Ví dụ: "Tesla Model 3", "Boeing 787", "Honda SH 160i".
  Không tạo VEHICLE cho mô tả chung như "chiếc xe", "xe máy",
  "máy bay", "ô tô".

- Không tạo Entity cho ngày, tháng, năm, thứ hoặc biểu thức thời gian.
  Ví dụ: "ngày 7", "tháng 8", "hôm nay", "sáng 02/8".
  Thông tin này chỉ được đưa vào time_expression của Event khi phù hợp.

- Không tạo Entity cho số tiền, giá cả, số lượng, tỷ lệ hoặc chỉ số.
  Ví dụ: "35 triệu đồng", "10 điểm", "5%", "giá dầu tăng 8%".

- Không tạo Entity cho chủ đề hoặc khái niệm chung nếu không phải một đối tượng
  có tên/định danh riêng.
  Ví dụ: "giá dầu", "giá vàng", "lãi suất", "chứng khoán",
  "trí tuệ nhân tạo" không tự động là Entity.

- Không đưa mô tả chung vào entities.
  Ví dụ: "a man", "Maryland man", "the victim", "nữ tài xế",
  "nghi phạm", "tòa án", "lực lượng chức năng".
  Nếu chúng thực sự tham gia Event thì xử lý bằng participant_text.

- Khi một câu chứa nhiều Entity lồng nhau, lấy từng Entity có tên riêng.
  Ví dụ:
  "Đội X (Cục Y, Bộ Z)" → 3 ORGANIZATION nếu cả ba là tên xác định.
  "xã A (tỉnh B)" → 2 LOCATION.

- Không dịch tên Entity.
- Không tạo tên không xuất hiện hoặc không thể suy ra chắc chắn từ văn bản.
- Không lấy hashtag hoặc handle làm Entity chỉ vì chúng xuất hiện trong bài.

- Một chủ thể xuất hiện nhiều lần chỉ tạo một Entity.
  Alias hoặc cách viết khác của cùng chủ thể phải dùng cùng canonical_name,
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

- Mỗi Entity có local_id e1, e2... duy nhất.
- Chỉ đặt resolution_confidence = HIGH khi việc nhận diện và phân giải chủ thể
  là chắc chắn.

- Trước khi tạo Event ở BƯỚC 2, rà toàn bộ văn bản để lấy đủ mọi Entity có tên
  thuộc tám type trên, kể cả Entity chỉ xuất hiện một lần hoặc không tham gia
  Event.

BƯỚC 2 - EVENT

- Event là hành động hoặc thay đổi trạng thái thực tế đã xảy ra, đang xảy ra
  hoặc được dự kiến sẽ xảy ra.
- Không tạo Event chỉ cho thời gian, địa điểm, sự hiện diện, bối cảnh,
  cảm xúc, sở thích, mong muốn hoặc câu hỏi/gợi ý của người đăng.
- STATEMENT chỉ tạo khi một chủ thể phát biểu, tuyên bố, cảnh báo, phủ nhận,
  xác nhận, khuyến nghị hoặc đưa ra quan điểm có nội dung thông tin đáng lưu.
- Một câu có thể chứa nhiều Event nếu có nhiều hành động độc lập.
  Không tách Event nếu nhiều câu hoặc nhiều mô tả chỉ bổ sung chi tiết cho cùng
  một hành động.
- Tối đa {MAX_EVENTS_PER_POST} Event cho toàn bộ văn bản.
- Nhận diện cả từ bị chèn ký tự để né kiểm duyệt:
  "đ/ánh" = "đánh", "b/ắn" = "bắn".
- Các cụm như "video ghi lại", "hình ảnh cho thấy", "xác minh video"
  không làm mất Event được mô tả bên trong.
- Nếu có chủ thể thực hiện hành động và đối tượng chịu tác động, vẫn tạo Event
  kể cả khi một hoặc cả hai không có tên riêng.
- description phải tự đầy đủ và giữ các chi tiết quan trọng được nói trực tiếp:
  số tiền, mức phạt, số điểm, số lượng, khoảng cách, thời hạn, kết quả và hậu quả.
  Không suy diễn.
- Nếu câu kế tiếp bổ sung chi tiết cho Event trước, gộp vào cùng Event.
- evidence_text là đoạn nguyên văn ngắn nhất đủ chứng minh hành động và các
  chi tiết quan trọng. Có thể dùng nhiều câu liền kề nếu cần.

Taxonomy Event duy nhất:
STATEMENT, MEETING, APPOINTMENT, APPROVAL, ELECTION, RESIGNATION,
ARREST, ASSAULT, ACCIDENT, DEATH, DROWNING, INVESTIGATION,
PROTEST, SPORTS_EVENT, TRANSFER, OTHER.

Quy tắc:
- MEETING chỉ dùng cho gặp/họp.
- Đánh, đẩy, tấn công, hất/tạt/ném vào người → ASSAULT.
- Chết đuối → DROWNING; không tạo thêm DEATH cho cùng hành động.
- Thi đấu, trận đấu hoặc diễn biến của giải đấu → SPORTS_EVENT.
- RESIGNATION và TRANSFER chỉ dùng khi văn bản nói trực tiếp về từ chức
  hoặc chuyển giao/chuyển nhượng.

Status phản ánh trạng thái của Event:
- PLANNED: chưa xảy ra nhưng đã được lên lịch/dự kiến.
- ONGOING: đang diễn ra.
- COMPLETED: đã xảy ra hoặc kết thúc.
- ALLEGED: chỉ là cáo buộc/chưa xác thực.
- REPORTED: chỉ dùng khi văn bản nói một sự việc được báo cáo nhưng không xác
  định được trạng thái mạnh hơn.
- UNKNOWN: không đủ thông tin.

BƯỚC 3 - PARTICIPANT

- Mỗi participant có đúng một trong:
  entity_id hoặc participant_text. Trường còn lại phải là null.
- entity_id chỉ được tham chiếu local_id có thật trong entities.
- Participant có tên riêng phải dùng entity_id và participant_scope = null.
- Participant không có tên riêng phải dùng:
  entity_id = null
  participant_text = nguyên văn cụm mô tả trong bài.
  participant_scope = GLOBAL_ROLE hoặc POST_LOCAL.
- Chỉ dùng GLOBAL_ROLE khi participant_text biểu thị một vai trò, chức danh
  hoặc nhóm chức năng chung có thể được dùng lại giữa nhiều bài viết, không
  đại diện cho danh tính của một cá nhân cụ thể. Ví dụ: "Đại biểu quốc hội",
  "lực lượng chức năng", "cơ quan chức năng", "công an",
  "lực lượng công an". Các tên cơ quan xác định như "Công an Hà Nội" vẫn là
  Entity ORGANIZATION, không phải anonymous participant.
- Dùng POST_LOCAL khi participant_text nói tới một người hoặc một nhóm vô danh
  cụ thể trong ngữ cảnh bài hiện tại. Ví dụ: "một người đàn ông", "nữ tài xế",
  "nạn nhân", "nghi phạm", "cụ bà 89 tuổi".
- Khi không chắc có thể dùng chung giữa nhiều bài, luôn chọn POST_LOCAL.
- Không gộp hai chủ thể khác nhau vào một participant.
- Phải đưa đủ các chủ thể chính của hành động vào participants.
- Role phản ánh vai trò trong Event, không phản ánh Entity.type.

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

Tên của chính giải đấu hoặc Event không phải participant và tuyệt đối không
được gán role LOCATION.

BƯỚC 4 - EVENT RELATION

Chỉ dùng:
APPROVES, CAUSES, ENABLES, PRECEDES, RELATED_TO.

- APPROVES: Event nguồn trực tiếp thể hiện sự phê duyệt Event đích.
- CAUSES: văn bản nói Event nguồn trực tiếp gây ra Event đích.
- ENABLES: Event nguồn tạo điều kiện cho Event đích xảy ra.
- PRECEDES: văn bản nói rõ Event nguồn xảy ra trước Event đích.
- RELATED_TO: văn bản nói rõ hai Event có liên hệ nhưng không phù hợp loại trên.

Chỉ tạo relation khi evidence_text trực tiếp chứng minh quan hệ.
Không suy ra relation chỉ vì hai Event:
- cùng nằm trong một Post,
- cùng Entity,
- cùng chủ đề,
- hoặc xuất hiện gần nhau trong văn bản.

Mọi source_event_id và target_event_id phải tham chiếu Event tồn tại trong JSON.

BƯỚC 5 - VALIDATION

Trước khi trả JSON, kiểm tra:
1. Mọi local_id là duy nhất.
2. Mọi participant.entity_id tồn tại trong entities.
3. Participant không tên riêng luôn dùng participant_text và có
   participant_scope phù hợp; participant dùng entity_id có
   participant_scope = null.
4. Không dùng Entity khác thay cho participant anonymous.
5. Tên giải đấu/sự kiện có tên riêng được tạo thành Entity type EVENT, nhưng không được dùng làm LOCATION participant.
6. Mọi event_relation chỉ tham chiếu Event tồn tại.
7. Không có field ngoài JSON schema.

Ví dụ:
- "Cơ quan A đã xử phạt một nữ tài xế. Mức xử phạt là 35 triệu đồng và
  trừ 10 điểm giấy phép lái xe."
  → một Event; Cơ quan A = ACTOR, "nữ tài xế" = TARGET;
  description giữ 35 triệu đồng và trừ 10 điểm.

- "A Maryland man pushed another man ... The man drowned."
  → ASSAULT và DROWNING với anonymous participants.

- "Bộ phim mà mình cực mong chờ phần 2 mà chưa thấy, bác nào biết phim tương tự k ạ"
  → events = [].

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
