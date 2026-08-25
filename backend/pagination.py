import base64
import binascii

from pydantic import ValidationError

from backend.models import EventSearchCursor


MAX_CURSOR_LENGTH = 4_096


def encode_event_cursor(cursor: EventSearchCursor) -> str:
    payload = cursor.model_dump_json().encode("utf-8")
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def decode_event_cursor(value: str) -> EventSearchCursor:
    if not value or len(value) > MAX_CURSOR_LENGTH:
        raise ValueError("Cursor không hợp lệ")

    try:
        encoded = value.encode("ascii")
        padding = b"=" * (-len(encoded) % 4)
        payload = base64.b64decode(
            encoded + padding,
            altchars=b"-_",
            validate=True,
        )
        return EventSearchCursor.model_validate_json(payload)
    except (UnicodeEncodeError, binascii.Error, ValidationError, ValueError):
        raise ValueError("Cursor không hợp lệ") from None
