"""Kiểm tra OpenAI-compatible API key qua VC gateway."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    import openai
except ImportError:
    print(
        'Thiếu thư viện "openai". Hãy cài bằng: pip install openai',
        file=sys.stderr,
    )
    raise SystemExit(2)


DEFAULT_BASE_URL = "http://ai-gateway.vcadm.vn/proxyllm"
DEFAULT_MODEL = "openai/gpt-4-turbo-2024-04-09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible API key")
    parser.add_argument(
        "--prompt",
        default="Trả lời chính xác một từ: OK",
        help="Nội dung gửi tới model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    if not api_key:
        print(
            "Thiếu OPENAI_API_KEY trong file .env.",
            file=sys.stderr,
        )
        return 2

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    print(f"Đang kiểm tra model: {model}")
    print(f"Gateway: {base_url}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=20,
        )
    except openai.AuthenticationError as exc:
        print(f"API key không hợp lệ: {exc}", file=sys.stderr)
        return 1
    except openai.APIStatusError as exc:
        print(
            f"Gateway trả về HTTP {exc.status_code}: {exc}",
            file=sys.stderr,
        )
        return 1
    except openai.APIConnectionError as exc:
        print(f"Không kết nối được tới gateway: {exc}", file=sys.stderr)
        return 1

    content = response.choices[0].message.content
    print("API key hoạt động.")
    print(f"Phản hồi: {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
