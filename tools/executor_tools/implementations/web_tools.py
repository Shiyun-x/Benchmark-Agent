from typing import Any, Dict, List, Optional
from pathlib import Path
import os
import re
import sys

# Web search + tool execution can exceed a single chat completion; cap only this path.
# Overrides llm_caller's DEFAULT_LLM_REQUEST_TIMEOUT_S for web_search calls only.
_WEB_SEARCH_TIMEOUT_S = int(os.getenv("WEB_SEARCH_REQUEST_TIMEOUT_S", "180"))

try:
    from utils.llm_caller import llm_call_json
except ModuleNotFoundError:
    # Allow direct execution: python tools/executor_tools/implementations/web_tools.py
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from utils.llm_caller import llm_call_json


def _compact_answer_text(text: str, max_chars: int) -> str:
    """Normalize whitespace and truncate at a line boundary when possible."""
    if not text:
        return text
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    cap = max(1, int(max_chars))
    if len(text) <= cap:
        return text
    cut = text[:cap]
    # Prefer breaking after a bullet or newline, not mid-sentence
    for sep in ("\n\n", "\n- ", "\n* ", "\n1.", "\n"):
        idx = cut.rfind(sep)
        if idx > cap * 0.55:
            cut = cut[:idx]
            break
    cut = cut.rstrip()
    if len(cut) < len(text):
        cut = cut + "\n...(truncated)"
    return cut


def web_search(
    query: str,
    model: str = "openai/responses/gpt-5.4",
    force_search: bool = True,
    search_context_size: str = "medium",
    image_paths: Optional[List[str]] = None,
    max_output_chars: int = 3000,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if search_context_size not in {"low", "medium", "high"}:
        search_context_size = "medium"

    tool_choice = "required" if force_search else "auto"

    ret = llm_call_json(
        model=model,
        system_prompt=(
            "You are a careful web research assistant. "
            "Use web search. "
            "For complex questions, perform multiple focused searches if necessary. "
            "Synthesize into a SHORT, structured factual summary in JSON — not a long essay. "
            "Format the answer as Markdown: optional one-line bold title, then bullet lines "
            "('- ' or '* ') for distinct facts; at most about 8–10 bullets; one short sentence per bullet; "
            "avoid long paragraphs and repetition. "
            "[IMPORTANT] If reliable information cannot be found, say so in one or two bullets. "
            "Do not fabricate missing facts. "
            "This is a single-turn tool call. "
            "Do not ask follow-up questions. "
            "Do not suggest continuing the conversation. "
        ),
        user_prompt=(
            f"{query.strip()}\n\n"
            "Return a JSON object with key:\n"
            "- answer: string — Markdown as specified (bullets, tight; no wall of text).\n"
            "The answer must be the direct final result for this single request."
        ),
        images=image_paths or None,
        extra_create_params={
            "timeout": _WEB_SEARCH_TIMEOUT_S,
            "request_timeout": _WEB_SEARCH_TIMEOUT_S,
            "tools": [
                {
                    "type": "web_search",
                    "search_context_size": search_context_size,
                }
            ],
            "tool_choice": tool_choice,
        },
    )

    if isinstance(ret, list):
        ret = ret[0] if ret else {}
    if not isinstance(ret, dict):
        raise ValueError(f"web_search got invalid llm response type: {type(ret).__name__}")
    if ret.get("ok") is False:
        raise RuntimeError(ret.get("error") or "web_search call failed")

    payload = ret.get("json")
    if not isinstance(payload, dict):
        payload = {}

    answer = str(payload.get("answer") or ret.get("raw_text") or "").strip()
    cap = max(200, int(max_output_chars))
    answer = _compact_answer_text(answer, cap)
    return {
        "answer": answer,
    }