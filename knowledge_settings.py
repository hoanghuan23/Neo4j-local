import logging
import os
import re

logging.getLogger("neo4j").setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "huanhoang"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("KNOWLEDGE_MODEL", "gemma4:e2b")
OLLAMA_TIMEOUT_SECONDS = 600
OLLAMA_MAX_ATTEMPTS = 2
OLLAMA_LOG_PREVIEW_CHARS = 2_000
POST_LIMIT = int(os.getenv("KNOWLEDGE_POST_LIMIT", "100"))
KNOWLEDGE_WORKERS = max(1, int(os.getenv("KNOWLEDGE_WORKERS", "3")))
KNOWLEDGE_MAX_RETRIES = int(os.getenv("KNOWLEDGE_MAX_RETRIES", "3"))
KNOWLEDGE_PROMPT_VERSION = "knowledge-v7"
KNOWLEDGE_PIPELINE_ENABLED = os.getenv(
    "KNOWLEDGE_PIPELINE_ENABLED", "true"
).strip().casefold() not in {"0", "false", "no", "off"}

ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION"}
EVENT_TYPES = {
    "STATEMENT",
    "MEETING",
    "APPOINTMENT",
    "APPROVAL",
    "ELECTION",
    "RESIGNATION",
    "ARREST",
    "ASSAULT",
    "ACCIDENT",
    "DEATH",
    "DROWNING",
    "INVESTIGATION",
    "PROTEST",
    "SPORTS_EVENT",
    "TRANSFER",
    "OTHER",
}
EVENT_STATUSES = {
    "PLANNED",
    "ONGOING",
    "COMPLETED",
    "REPORTED",
    "ALLEGED",
    "UNKNOWN",
}
EVENT_ROLES = {
    "ACTOR",
    "TARGET",
    "VICTIM",
    "SPEAKER",
    "SUBJECT",
    "LOCATION",
    "ORGANIZATION",
    "PARTICIPANT",
}
EVENT_RELATION_TYPES = {
    "APPROVES",
    "CAUSES",
    "ENABLES",
    "PRECEDES",
    "RELATED_TO",
}
CONFIDENCE_LEVELS = {"HIGH", "MEDIUM", "LOW"}
MAX_EVENTS_PER_POST = 5


def _strict_object(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


ENTITY_ITEM_SCHEMA = _strict_object(
    {
        "local_id": {"type": "string"},
        "name": {"type": "string"},
        "canonical_name": {"type": "string"},
        "type": {"type": "string", "enum": sorted(ENTITY_TYPES)},
        "resolution_confidence": {
            "type": "string",
            "enum": sorted(CONFIDENCE_LEVELS),
        },
    },
    [
        "local_id",
        "name",
        "canonical_name",
        "type",
        "resolution_confidence",
    ],
)

PARTICIPANT_ITEM_SCHEMA = _strict_object(
    {
        "entity_id": {"type": ["string", "null"]},
        "participant_text": {"type": ["string", "null"]},
        "role": {"type": "string", "enum": sorted(EVENT_ROLES)},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    ["entity_id", "participant_text", "role", "confidence"],
)

EVENT_ITEM_SCHEMA = _strict_object(
    {
        "local_id": {"type": "string"},
        "type": {"type": "string", "enum": sorted(EVENT_TYPES)},
        "description": {"type": "string"},
        "evidence_text": {"type": "string"},
        "status": {"type": "string", "enum": sorted(EVENT_STATUSES)},
        "time_expression": {"type": ["string", "null"]},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "participants": {
            "type": "array",
            "items": PARTICIPANT_ITEM_SCHEMA,
        },
    },
    [
        "local_id",
        "type",
        "description",
        "evidence_text",
        "status",
        "time_expression",
        "confidence",
        "participants",
    ],
)

EVENT_RELATION_ITEM_SCHEMA = _strict_object(
    {
        "source_event_id": {"type": "string"},
        "type": {"type": "string", "enum": sorted(EVENT_RELATION_TYPES)},
        "target_event_id": {"type": "string"},
        "evidence_text": {"type": "string"},
    },
    ["source_event_id", "type", "target_event_id", "evidence_text"],
)

KNOWLEDGE_SCHEMA = _strict_object(
    {
        "entities": {"type": "array", "items": ENTITY_ITEM_SCHEMA},
        "events": {
            "type": "array",
            "items": EVENT_ITEM_SCHEMA,
            "maxItems": MAX_EVENTS_PER_POST,
        },
        "event_relations": {
            "type": "array",
            "items": EVENT_RELATION_ITEM_SCHEMA,
        },
    },
    ["entities", "events", "event_relations"],
)

# Temporary compatibility alias for existing callers and tests.
ENTITY_SCHEMA = KNOWLEDGE_SCHEMA

NULL_STRINGS = {"", "null", "none", "nil", "n/a"}
GENERIC_ENTITY_EXACT = {
    "a man",
    "a woman",
    "a person",
    "a boy",
    "a girl",
    "a driver"
    "an individual",
    "the man",
    "the woman",
    "the person",
    "the individual",
    "the victim",
    "the suspect",
    "the defendant",
    "the president",
    "the government",
    "a house panel",
    "house panel",
    "italian community",
    "người đàn ông",
    "nam thanh niên",
    "nữ thanh niên",
    "bé trai",
    "bé gái",
    "trẻ em",
    "người già",
    "bị cáo",
    "người lái xe",
    "công nhân",
    "học sinh",
    "sinh viên",
    "người phụ nữ",
    "người dân",
    "người mua",
    "người bán",
    "khách hàng",
    "nạn nhân",
    "nghi phạm",
    "tài xế",
    "tài xế taxi",
    "xe máy",
    "xe oto",
    "camera",
    "camera an ninh",
    "cảnh sát",
    "lực lượng tìm kiếm",
    "lực lượng chức năng",
    "tổ công tác",
}
GENERIC_PERSON_OR_GROUP_SUFFIXES = {
    "man",
    "woman",
    "person",
    "individual",
    "victim",
    "suspect",
    "resident",
    "official",
    "officials",
}

# Used only to recover a participant when the model emits a dangling entity_id
# without participant_text. Recovery is deliberately conservative: validation
# accepts it only when one unique anonymous description occurs in the event.
ANONYMOUS_PARTICIPANT_PATTERN = re.compile(
    r"(?<!\w)(?:"
    r"(?:nam|nữ)\s+tài\s+xế(?:\s+taxi)?|"
    r"tài\s+xế(?:\s+taxi)?|"
    r"người\s+(?:đàn\s+ông|phụ\s+nữ|dân|mua|bán)|"
    r"cụ\s+bà(?:\s+\d+\s+tuổi)?|"
    r"vị\s+khách|khách\s+hàng|nạn\s+nhân|nghi\s+phạm|"
    r"lực\s+lượng\s+(?:chức\s+năng|tìm\s+kiếm)|"
    r"(?:tổ\s+công\s+tác|đội\s+tìm\s+kiếm|nhóm\s+tìm\s+kiếm)|"
    r"cảnh\s+sát|"
    r"(?:a|an|the|another)\s+(?:man|woman|person|individual|"
    r"victim|suspect|resident|official|witness)"
    r")(?!\w)",
    re.IGNORECASE,
)
EVENT_NAME_PATTERN = re.compile(
    r"\b(championships?|tournaments?|grand prix|olympics|games|world cup)\b$",
    re.IGNORECASE,
)
ORGANIZATION_NAME_PATTERN = re.compile(
    r"\b(company|corporation|corp|inc|ltd|llc|agency|authority|department|"
    r"ministry|committee|council|embassy|university|club|association|"
    r"administration)\b",
    re.IGNORECASE,
)
LOCATION_NAME_PATTERN = re.compile(
    r"\b(harbou?r|lake|river|island)\b$",
    re.IGNORECASE,
)
COUNTRY_NAME_FALLBACKS = {
    "bahamas",
    "china",
    "france",
    "germany",
    "india",
    "iran",
    "italy",
    "japan",
    "laos",
    "north korea",
    "russia",
    "south korea",
    "syria",
    "taiwan",
    "uk",
    "u.k.",
    "ukraine",
    "united kingdom",
    "united states",
    "united states of america",
    "us",
    "u.s.",
    "usa",
    "u.s.a.",
    "vietnam",
    "viet nam",
}

EVENT_ACTION_TRIGGERS = {
    "STATEMENT": {
        "said",
        "says",
        "spoke",
        "speaks",
        "stated",
        "announced",
        "warned",
        "warns",
        "denied",
        "denies",
        "recommended",
        "claimed",
        "told",
        "expressed",
        "expresses",
        "commented",
        "comments",
        "reported",
        "nói",
        "phát biểu",
        "cảnh báo",
        "phủ nhận",
        "khuyến nghị",
        "tuyên bố",
        "cho biết",
        "cho hay",
        "tiết lộ",
        "thừa nhận",
        "khẳng định",
        "nhấn mạnh",
        "bày tỏ",
        "kêu gọi",
        "yêu cầu",
        "giải thích",
        "chia sẻ",
        "nhận định"
    },
    "MEETING": {
        "met",
        "meet",
        "meeting",
        "held talks",
        "convened",
        "discussed",
        "talked",
        "gặp",
        "cuộc họp",
        "họp",
        "hội đàm",
        "hội nghị",
        "làm việc với",
        "trao đổi",
        "tiếp xúc"
        "đàm phán"
    },
    "APPOINTMENT": {
        "appointed",
        "named",
        "nominated",
        "selected",
        "bổ nhiệm",
        "đề cử",
        "được bổ nhiệm",
        "chỉ định",
        "phân công",
        "bầu làm",
        "thăng chức",
        "nhậm chức"
    },
    "APPROVAL": {
        "approved",
        "approval",
        "authorized",
        "authorization",
        "adopted",
        "passed",
        "phê duyệt",
        "chấp thuận",
        "thông qua",
        "đồng ý",
        "đồng thuận",
        "được phê duyệt",
        "được thông qua",
        "phê chuẩn",
        "cấp phép"
    },
    "ELECTION": {
        "elected",
        "election",
        "voted",
        "won the vote",
        "bầu cử",
        "đắc cử",
        "bỏ phiếu",
        "tái cử",
        "tranh cử",
        "ứng cử"
    },
    "RESIGNATION": {
        "resigned",
        "resigns",
        "stepped down",
        "quit office",
        "từ chức",
        "rời chức",
    },
    "ARREST": {
        "arrested",
        "detained",
        "taken into custody",
        "bắt giữ",
        "tạm giữ",
        "bị tạm giữ",
        "tạm giam",
        "bị tạm giam",
        "bị bắt",
        "bị bắt giữ",
        "khống chế"
    },
    "ASSAULT": {
        "attacked",
        "assaulted",
        "pushed",
        "hit",
        "struck",
        "beat",
        "bombed",
        "bombing",
        "stabbed",
        "shot",
        "tấn công",
        "hành hung",
        "đẩy",
        "đánh",
        "đâm",
        "bắn",
        "hất",
        "tạt",
        "ném",
        "đá",
        "chém",
        "tát",
        "xô xát",
        "ẩu đả",
        "nổ súng"
    },
    "ACCIDENT": {
        "crashed",
        "collision",
        "accident",
        "derailed",
        "wrecked",
        "tai nạn",
        "va chạm",
        "rơi",
        "lao vào",
        "lật",
        "đâm vào",
        "tông",
        "đụng",
        "ngã",
        "rơi xuống",
        "chìm",
        "cháy"
        "phát nổ"
    },
    "DEATH": {
        "died",
        "dead",
        "killed",
        "death",
        "qua đời",
        "thiệt mạng",
        "tử vong",
        "chết",
        "hy sinh",
        "không qua khỏi",
        "mất mạng"
    },
    "DROWNING": {
        "drowned",
        "drowning",
        "chết đuối",
        "đuối nước",
        "bị đuối nước",
        "nước cuốn"
    },
    "INVESTIGATION": {
        "investigated",
        "investigation",
        "probe",
        "inquiry",
        "điều tra",
        "tiến hành điều tra",
        "xác minh",
        "đang xác minh",
        "làm rõ",
        "khởi tố điều tra"
    },
    "PROTEST": {
        "protested",
        "protest",
        "demonstrated",
        "demonstration",
        "rallied"
        "rally",
        "biểu tình",
        "tuần hành",
        "tụ tập",
        "đình công",
        "bãi công"
    },
    "SPORTS_EVENT": {
        "competed",
        "competition",
        "championship",
        "match",
        "game",
        "tournament",
        "played",
        "won",
        "lost",
        "thi đấu",
        "trận đấu",
        "giải đấu",
        "ghi bàn",
        "lập công"
        "kiến tạo"
        "đối đầu"
    },
    "TRANSFER": {
        "transferred",
        "transfer",
        "handed over",
        "moved to",
        "chuyển giao",
        "chuyển nhượng",
        "điều chuyển",
        "gia nhập",
        "đầu quân",
        "mượn",
        "chiêu mộ",
        "chuyển sang"
    },
    "OTHER": {
        "created",
        "made",
        "opened",
        "closed",
        "launched",
        "signed",
        "filed",
        "banned",
        "blocked",
        "released",
        "discovered",
        "found",
        "received",
        "damaged",
        "became",
        "looked",
        "stole",
        "rescued",
        "built",
        "destroyed",
        "tạo",
        "mở",
        "đóng",
        "ký",
        "cấm",
        "phát hiện",
        "giải cứu",
        "phá hủy",
        "huy động",
        "tìm kiếm",
        "tham gia",
        "xét xử",
        "mở phiên tòa"
    },
}

RELATION_EVIDENCE_MARKERS = {
    "APPROVES": {
        "approve",
        "approved",
        "approves",
        "authorize",
        "authorized",
        "phê duyệt",
        "chấp thuận",
        "thông qua",
    },
    "CAUSES": {
        "cause",
        "caused",
        "because",
        "resulted in",
        "led to",
        "due to",
        "gây",
        "khiến",
        "dẫn đến",
        "do",
    },
    "ENABLES": {
        "enable",
        "enabled",
        "allows",
        "allowed",
        "made possible",
        "cho phép",
        "tạo điều kiện",
    },
    "PRECEDES": {
        "before",
        "prior to",
        "earlier than",
        "preceded",
        "trước",
    },
}
