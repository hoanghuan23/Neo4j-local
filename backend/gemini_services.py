import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, Field

from backend.models import EventResult, ParsedQuestion
from backend.question_parser import (
    RuleBasedQuestionParser,
    has_explicit_duration,
    normalize_entity_for_search,
    normalize_location_for_search,
)


LOGGER = logging.getLogger(__name__)
TOKENS_PER_MILLION = Decimal("1000000")


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)


def _price(value: str | Decimal) -> Decimal:
    try:
        price = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Giá Gemini phải là số USD hợp lệ") from exc
    if price < 0:
        raise ValueError("Giá Gemini không được âm")
    return price


def _log_usage(
    response: Any,
    *,
    stage: str,
    model: str,
    input_price_per_million_usd: Decimal,
    output_price_per_million_usd: Decimal,
) -> None:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        LOGGER.warning(
            "Gemini usage unavailable | stage=%s | model=%s",
            stage,
            model,
        )
        return

    input_tokens = int(getattr(metadata, "prompt_token_count", None) or 0)
    output_tokens = int(
        getattr(metadata, "candidates_token_count", None) or 0
    )
    thinking_tokens = int(
        getattr(metadata, "thoughts_token_count", None) or 0
    )
    billable_output_tokens = output_tokens + thinking_tokens
    total_tokens = int(
        getattr(metadata, "total_token_count", None)
        or input_tokens + billable_output_tokens
    )
    input_cost = (
        Decimal(input_tokens)
        * input_price_per_million_usd
        / TOKENS_PER_MILLION
    )
    output_cost = (
        Decimal(billable_output_tokens)
        * output_price_per_million_usd
        / TOKENS_PER_MILLION
    )

    LOGGER.info(
        "Gemini usage | stage=%s | model=%s | input_tokens=%d | "
        "output_tokens=%d | thinking_tokens=%d | total_tokens=%d | "
        "input_cost_usd=%.8f | output_cost_usd=%.8f | total_cost_usd=%.8f",
        stage,
        model,
        input_tokens,
        output_tokens,
        thinking_tokens,
        total_tokens,
        input_cost,
        output_cost,
        input_cost + output_cost,
    )


def _validate_structured_response(
    response: Any,
    schema: type[BaseModel],
) -> BaseModel:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    if parsed is not None:
        return schema.model_validate(parsed)

    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini trả về nội dung rỗng")
    return schema.model_validate_json(text)


class GeminiQuestionParser:
    def __init__(
        self,
        *,
        client: Any,
        types_module: Any,
        model: str,
        default_hours: int = 24,
        input_price_per_million_usd: str | Decimal = "0.25",
        output_price_per_million_usd: str | Decimal = "1.50",
    ):
        self.client = client
        self.types = types_module
        self.model = model
        self.default_hours = default_hours
        self.input_price_per_million_usd = _price(
            input_price_per_million_usd
        )
        self.output_price_per_million_usd = _price(
            output_price_per_million_usd
        )

    def parse(self, question: str) -> ParsedQuestion:
        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                "Phân tích câu hỏi tìm kiếm sự kiện tiếng Việt dưới đây. "
                "Quy tắc:\n"
                "1. location:\n"
                "- Chứa một địa điểm chính được yêu cầu hoặc null.\n"
                "- Dùng tên địa lý ngắn gọn, đủ phân biệt để tìm kiếm bao quát.\n"
                "- Bỏ các tiền tố hành chính như 'tỉnh', 'thành phố', 'TP.'.\n"
                "- Ví dụ: 'thành phố Lạng Sơn', 'tỉnh Lạng Sơn' -> 'Lạng Sơn'.\n"
                "- Không đặt tên người hoặc tổ chức vào location.\n\n"

                "2. entity:\n"
                "- Chứa người, tổ chức, chủ thể CỤ THỂ hoặc CHỦ ĐỀ mà người dùng "
                "trực tiếp muốn tìm sự kiện liên quan.\n"
                "- Với chủ đề rộng, vẫn giữ nguyên cụm chủ đề trong entity để dùng làm "
                "điều kiện tìm kiếm văn bản; không có nghĩa chủ đề đó là một node Entity.\n"
                "- Ví dụ: 'bóng đá Việt Nam', 'thể thao Việt Nam', 'kinh tế Việt Nam', "
                "'giáo dục', 'giao thông' phải được giữ trong entity, không trả null.\n"
                "- Chỉ trả entity=null khi câu hỏi thực sự không có chủ thể hoặc chủ đề "
                "cần lọc.\n"
                "- Nếu có nhiều chủ thể cụ thể, giữ đầy đủ tên và giữ nguyên quan hệ "
                "'và' hoặc 'hoặc' theo câu hỏi.\n\n"
                "- Với cấu trúc 'giữa A và B', chỉ trả 'A và B'; không đưa từ "
                "'giữa' vào entity.\n\n"

                "3. time:\n"
                "- Quy đổi khoảng thời gian sang hours.\n"
                "- Số ngày = số ngày * 24.\n"
                "- Số tuần = số tuần * 168.\n"
                "- Số tháng = số tháng * 720 (quy ước 30 ngày).\n"
                f"- Nếu không nêu khoảng thời gian, dùng hours={self.default_hours}.\n"
                "- Riêng 'tuần trước' tạm dùng hours=168.\n"
                "- Giới hạn hours trong khoảng 1 đến 720.\n\n"

                "4. posted_date:\n"
                "- Nếu câu hỏi chỉ định đúng một ngày theo dạng 'ngày D tháng M' "
                "hoặc 'ngày D tháng M năm YYYY', trả ngày ISO tương ứng.\n"
                "- Nếu không có ngày lịch chính xác, trả null.\n\n"

                "5. Intent luôn là 'search_events'.\n\n"
                f"Câu hỏi: {question}"
            ),
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ParsedQuestion,
                temperature=0,
                automatic_function_calling=(
                    self.types.AutomaticFunctionCallingConfig(disable=True)
                ),
            ),
        )
        _log_usage(
            response,
            stage="question_parser",
            model=self.model,
            input_price_per_million_usd=self.input_price_per_million_usd,
            output_price_per_million_usd=self.output_price_per_million_usd,
        )
        parsed = ParsedQuestion.model_validate(
            _validate_structured_response(response, ParsedQuestion)
        )
        rule_parsed = RuleBasedQuestionParser(
            default_hours=self.default_hours
        ).parse(question)
        location = normalize_location_for_search(parsed.location)
        entity = normalize_entity_for_search(parsed.entity)
        posted_date = rule_parsed.posted_date
        hours = (
            rule_parsed.hours
            if has_explicit_duration(question)
            else parsed.hours
        )
        if posted_date is not None and rule_parsed.location is not None:
            location = rule_parsed.location
        if location is None and entity is None:
            # An unconstrained repository query means "latest events".  If the
            # model misses a broad topic, using the original text as a search
            # condition is safer than returning unrelated recent events.
            entity = " ".join(question.strip().strip("?.!,;:").split()) or None
        return parsed.model_copy(
            update={
                "location": location,
                "entity": entity,
                "hours": hours,
                "posted_date": posted_date,
            }
        )


class FallbackQuestionParser:
    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback

    def parse(self, question: str) -> ParsedQuestion:
        try:
            return self.primary.parse(question)
        except Exception as exc:
            LOGGER.warning(
                "Gemini question parsing failed; using rule-based fallback: %s",
                type(exc).__name__,
            )
            return self.fallback.parse(question)


class GeminiAnswerGenerator:
    def __init__(
        self,
        *,
        client: Any,
        types_module: Any,
        model: str,
        input_price_per_million_usd: str | Decimal = "0.25",
        output_price_per_million_usd: str | Decimal = "1.50",
    ):
        self.client = client
        self.types = types_module
        self.model = model
        self.input_price_per_million_usd = _price(
            input_price_per_million_usd
        )
        self.output_price_per_million_usd = _price(
            output_price_per_million_usd
        )

    def generate(
        self,
        *,
        question: str,
        parsed: ParsedQuestion,
        events: list[EventResult],
    ) -> str:
        payload = {
            "question": question,
            "query": parsed.model_dump(mode="json"),
            # Keep answer-generation behavior based on the full description.
            # The title is a separate structured API field for the frontend.
            "events": [
                event.model_dump(mode="json", exclude={"title"})
                for event in events
            ],
        }
        response = self.client.models.generate_content(
            model=self.model,
            contents=(
            "Bạn là bộ tạo câu trả lời từ dữ liệu sự kiện đã truy xuất.\n\n"

            "QUY TẮC BẮT BUỘC:\n"
            "1. Mảng `events` chứa bao nhiêu phần tử thì câu trả lời PHẢI có đúng "
            "bấy nhiêu mục sự kiện.\n"
            "mỗi sự kiện thành một mục riêng; không được bỏ sót mục nào.\n"
            "2. Mỗi phần tử của `events` tương ứng với CHÍNH XÁC MỘT mục riêng biệt.\n"
            "3. Tuyệt đối KHÔNG gộp hai hay nhiều phần tử `events` vào cùng một câu, "
            "đoạn văn hoặc mục.\n"
            "Nói cách khác: không gộp nhiều sự kiện vào cùng một mục.\n"
            "4. Không được tạo đoạn mở đầu kiểu 'Trong 2 tuần qua...' hoặc đoạn tổng kết.\n"
            "5. Không được tự suy luận rằng nhiều event giống nhau là cùng một sự kiện. "
            "Việc gộp event đã được xử lý trước khi dữ liệu đến đây.\n"
            "6. Giữ nguyên thứ tự của mảng events.\n"
            "7. Chỉ sử dụng thông tin có trong từng event tương ứng. "
            "Không lấy thông tin của event khác để bổ sung vào event hiện tại.\n"
            "8. Không suy đoán, không tra cứu thông tin bên ngoài.\n"
            "9. Nếu một event có nhiều nguồn thì chỉ gắn các nguồn thuộc event đó.\n\n"
            "10. `sources` là danh sách đầy đủ các bài viết nguồn đã khử "
            "trùng. Mỗi phần tử có `source`, `posted_at` và `url`. "
            "`post` chỉ là bài đại diện và không phải một nguồn bổ sung.\n\n"
            "11. Dùng description làm nội dung; không thêm hoặc thay bằng field title.\n\n"

            "ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:\n"
            "- Mỗi event phải nằm trên một block Markdown riêng.\n"
            "- Mỗi block bắt đầu bằng: `Sự kiện N`\n"
            "- Dòng tiếp theo là nội dung tóm tắt ngắn gọn của CHÍNH event đó.\n"
            "- Sau đó mới liệt kê nguồn của event đó nếu có.\n"
            "- Giữa hai event phải có một dòng trống.\n\n"

            "Ví dụ nếu `events` có 3 phần tử, đầu ra phải có đúng dạng:\n\n"
            "Sự kiện 1\n"
            "<tóm tắt chỉ event[0]>\n"
            "<nguồn của event[0]>\n\n"
            "Sự kiện 2\n"
            "<tóm tắt chỉ event[1]>\n"
            "<nguồn của event[1]>\n\n"
            "ự kiện 3\n"
            "<tóm tắt chỉ event[2]>\n"
            "<nguồn của event[2]>\n\n"

            "Không được thay định dạng trên bằng một đoạn văn đánh số 1, 2, 3.\n\n"
            "Không dùng danh sách Markdown đánh số thay cho các block Sự kiện N.\n\n"

            "DỮ LIỆU:\n"
            + json.dumps(payload, ensure_ascii=False)
        ),
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeneratedAnswer,
                temperature=0,
                automatic_function_calling=(
                    self.types.AutomaticFunctionCallingConfig(disable=True)
                ),
            ),
        )
        _log_usage(
            response,
            stage="answer_generator",
            model=self.model,
            input_price_per_million_usd=self.input_price_per_million_usd,
            output_price_per_million_usd=self.output_price_per_million_usd,
        )
        result = _validate_structured_response(response, GeneratedAnswer)
        answer = GeneratedAnswer.model_validate(result).answer.strip()
        if not answer:
            raise ValueError("Gemini trả về câu trả lời rỗng")
        return answer


class FallbackAnswerGenerator:
    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback

    def generate(
        self,
        *,
        question: str,
        parsed: ParsedQuestion,
        events: list[EventResult],
    ) -> str:
        if not events:
            return self.fallback.generate(
                question=question,
                parsed=parsed,
                events=events,
            )
        try:
            return self.primary.generate(
                question=question,
                parsed=parsed,
                events=events,
            )
        except Exception as exc:
            LOGGER.warning(
                "Gemini answer generation failed; using template fallback: %s",
                type(exc).__name__,
            )
            return self.fallback.generate(
                question=question,
                parsed=parsed,
                events=events,
            )
