"""LangSmith instrumentation shared by the knowledge extraction pipeline."""

import json
from typing import Any, Callable

from langsmith import get_current_run_tree, traceable


def _llm_inputs(inputs: dict) -> dict:
    """Render the project's prompt argument as a chat message in LangSmith."""
    prompt = inputs.get("prompt", "")
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": str(prompt)}],
            }
        ]
    }


def _llm_outputs(output: Any) -> dict:
    """Render structured extraction output as an assistant message."""
    content = json.dumps(output, ensure_ascii=False, default=str)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


def trace_llm(
    *,
    name: str,
    provider: str,
    model: str,
) -> Callable:
    """Create a trace decorator for a directly-invoked LLM provider."""
    return traceable(
        name=name,
        run_type="llm",
        metadata={
            "ls_provider": provider,
            "ls_model_name": model,
            "ls_temperature": 0,
        },
        process_inputs=_llm_inputs,
        process_outputs=_llm_outputs,
    )


def set_langsmith_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
) -> None:
    """Attach provider token counts without changing application output."""
    run = get_current_run_tree()
    if run is None:
        return

    total_output_tokens = output_tokens + reasoning_tokens
    usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": input_tokens + total_output_tokens,
    }
    if reasoning_tokens:
        usage_metadata["output_token_details"] = {
            "text": output_tokens,
            "reasoning": reasoning_tokens,
        }
    run.set(usage_metadata=usage_metadata)


def set_langsmith_model(model: str) -> None:
    """Override static model metadata for callers configured at runtime."""
    run = get_current_run_tree()
    if run is not None:
        run.metadata["ls_model_name"] = model
