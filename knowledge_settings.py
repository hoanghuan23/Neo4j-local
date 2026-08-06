import logging
import os
import re

logging.getLogger("neo4j").setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "huanhoang"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("KNOWLEDGE_MODEL", "qwen3.5:4b")
OLLAMA_TIMEOUT_SECONDS = 600
OLLAMA_MAX_ATTEMPTS = 2
OLLAMA_LOG_PREVIEW_CHARS = 2_000
POST_LIMIT = int(os.getenv("KNOWLEDGE_POST_LIMIT", "100"))
KNOWLEDGE_WORKERS = max(1, int(os.getenv("KNOWLEDGE_WORKERS", "3")))
KNOWLEDGE_MAX_RETRIES = int(os.getenv("KNOWLEDGE_MAX_RETRIES", "3"))
KNOWLEDGE_PROMPT_VERSION = "knowledge-v3"
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
        "start_year": {"type": ["integer", "null"]},
        "end_year": {"type": ["integer", "null"]},
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
        "start_year",
        "end_year",
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
    },
    "MEETING": {
        "met",
        "meet",
        "meeting",
        "held talks",
        "convened",
        "gặp",
        "cuộc họp",
        "họp",
        "hội đàm",
    },
    "APPOINTMENT": {
        "appointed",
        "named",
        "nominated",
        "selected",
        "bổ nhiệm",
        "đề cử",
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
    },
    "ELECTION": {
        "elected",
        "election",
        "voted",
        "won the vote",
        "bầu cử",
        "đắc cử",
        "bỏ phiếu",
    },
    "RESIGNATION": {
        "resigned",
        "resigns",
        "stepped down",
        "quit office",
        "từ chức",
    },
    "ARREST": {
        "arrested",
        "detained",
        "taken into custody",
        "bắt giữ",
        "tạm giữ",
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
        "ném"
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
    },
    "DEATH": {
        "died",
        "dead",
        "killed",
        "death",
        "qua đời",
        "thiệt mạng",
        "tử vong",
    },
    "DROWNING": {
        "drowned",
        "drowning",
        "chết đuối",
        "đuối nước",
    },
    "INVESTIGATION": {
        "investigated",
        "investigation",
        "probe",
        "inquiry",
        "điều tra",
    },
    "PROTEST": {
        "protested",
        "protest",
        "demonstrated",
        "rally",
        "biểu tình",
        "tuần hành",
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
    },
    "TRANSFER": {
        "transferred",
        "transfer",
        "handed over",
        "moved to",
        "chuyển giao",
        "chuyển nhượng",
        "điều chuyển",
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
