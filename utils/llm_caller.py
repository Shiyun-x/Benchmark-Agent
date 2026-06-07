# tools/llm_caller.py
# -----------------------------------------------------------------------------
# A lightweight, generic LLM caller as a plain tool.
# - Supports text / JSON two modes.
# - Uses litellm.completion under the hood.
# -----------------------------------------------------------------------------

from typing import Dict, Any, List, Optional, Union
import os
import json
import re
import ast
import base64
import mimetypes
from litellm import completion
import litellm

# Configure litellm
litellm.drop_params = True

try:
    from utils.model_config import get_tool_model, get_api_key, get_api_base_url
    DEFAULT_MODEL = get_tool_model("default")
except Exception:
    DEFAULT_MODEL = "gpt-5.1"
    def get_api_key(_config_path=None):
        return os.getenv("LLM_API_KEY", "")
    def get_api_base_url(_config_path=None):
        return None

# ===== Safety / size limits =====
MAX_USER_JSON_CHARS = 16000
MAX_SYS_PROMPT_CHARS = 8000
MAX_USER_PROMPT_CHARS = 12000
MAX_TOKENS = 12000
DEFAULT_LLM_REQUEST_TIMEOUT_S = int(os.getenv("LLM_REQUEST_TIMEOUT_S", "900"))
# Type aliases
JSONType = Union[Dict[str, Any], List[Any]]

def _extract_fenced_blocks(text: str) -> List[str]:
    """
    Extract ```json ... ``` or ``` ... ``` fenced blocks that look like JSON object/array.
    """
    if not text:
        return []
    blocks = []
    # Capture either {...} or [...] inside fences
    for m in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        s = m.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            blocks.append(s)
    return blocks

def _fix_common_json_issues(snippet: str) -> str:
    """
    Attempt light repairs on common non-JSON issues:
      - single quotes -> double quotes
      - True/False/None -> true/false/null
      - trailing commas before ] or }
    """
    if not isinstance(snippet, str):
        return snippet
    fixed = snippet

    # Replace single quotes with double quotes (naive but effective for many cases)
    fixed = fixed.replace("'", '"')

    # Python literals -> JSON
    fixed = re.sub(r'(?<!")\bTrue\b(?!")', 'true', fixed)
    fixed = re.sub(r'(?<!")\bFalse\b(?!")', 'false', fixed)
    fixed = re.sub(r'(?<!")\bNone\b(?!")', 'null', fixed)

    # Remove trailing commas before } or ]
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

    return fixed

def _literal_eval_fallback(snippet: str) -> Optional[JSONType]:
    """
    Fallback using Python's ast.literal_eval for dict/list-like strings.
    """
    try:
        pyobj = ast.literal_eval(snippet)
        if isinstance(pyobj, (dict, list)):
            return pyobj
    except Exception:
        pass
    return None

def _json5_fallback(text: str) -> Optional[JSONType]:
    """
    Optional fallback via json5 if installed.
    """
    try:
        import json5  # type: ignore
        obj = json5.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    return None

def _safe_json_loads(text: str) -> Optional[JSONType]:
    """
    Strong JSON extractor (top-level dict OR list):
      1) direct json.loads
      2) fenced blocks ```json ... ```
      3) outermost {...} slice (raw) OR outermost [...] slice (raw)
      4) slice + auto-fixes (quotes, literals, trailing commas)
      5) ast.literal_eval
      6) json5 (optional)
    """
    if not text or not isinstance(text, str):
        return None

    # 1) direct
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    # 2) fenced blocks
    for block in _extract_fenced_blocks(text):
        try:
            obj = json.loads(block)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            fixed = _fix_common_json_issues(block)
            try:
                obj = json.loads(fixed)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                obj = _literal_eval_fallback(block)
                if obj is not None:
                    return obj
                obj = _json5_fallback(block)
                if obj is not None:
                    return obj

    # 3) outermost slice: prefer array if prompt expects it, but here just pick the wider valid span
    obj_start, obj_end = text.find("{"), text.rfind("}")
    arr_start, arr_end = text.find("["), text.rfind("]")

    candidates = []
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(text[obj_start:obj_end+1])
    if arr_start >= 0 and arr_end > arr_start:
        candidates.append(text[arr_start:arr_end+1])

    # try longer snippet first
    candidates.sort(key=len, reverse=True)

    for snippet in candidates:
        # 3a) raw
        try:
            obj = json.loads(snippet)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
        # 4) fixes
        fixed = _fix_common_json_issues(snippet)
        try:
            obj = json.loads(fixed)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
        # 5) literal eval
        obj = _literal_eval_fallback(snippet)
        if obj is not None:
            return obj
        # 6) json5
        obj = _json5_fallback(snippet)
        if obj is not None:
            return obj

    # last resort: try json5 on full text
    obj = _json5_fallback(text)
    if obj is not None:
        return obj

    return None

def _build_messages(
    system_prompt: str,
    user_prompt: str,
    user_json: Optional[Dict[str, Any]],
    images: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-style messages (system + user).

    - If no images and no user_json: content is a simple string.
    - If images or user_json present: use OpenAI Vision format,
      user.content is a list of text/image_url items.
    - images: Can be remote URLs or local file paths.
      Local paths are read and converted to base64 data URLs.
    """
    sys_p = system_prompt
    usr_p = user_prompt

    messages: List[Dict[str, Any]] = []

    if sys_p:
        messages.append({"role": "system", "content": sys_p})

    if not images and user_json is None:
        if usr_p:
            messages.append({"role": "user", "content": usr_p})
        return messages

    # ---- multimodal mode: OpenAI Vision style ----
    user_content: List[Dict[str, Any]] = []

    if usr_p:
        user_content.append({
            "type": "text",
            "text": usr_p,
        })

    if user_json is not None:
        dumped = json.dumps(user_json, ensure_ascii=False)
        user_content.append({
            "type": "text",
            "text": f"```json\n{dumped}\n```"
        })

    # Process images: remote URLs used directly, local paths converted to base64 data URLs
    if images:
        for img_path in images:
            if not img_path:
                continue

            if re.match(r"^https?://", img_path):
                # Remote URL, use directly
                url = img_path
            else:
                # Local file path: read file and convert to base64 data URL
                abs_path = os.path.abspath(img_path)
                try:
                    with open(abs_path, "rb") as f:
                        data = f.read()
                    b64 = base64.b64encode(data).decode("utf-8")

                    mime, _ = mimetypes.guess_type(abs_path)
                    if mime is None:
                        mime = "image/png"  # Default fallback

                    url = f"data:{mime};base64,{b64}"
                except Exception as e:
                    # If reading fails, add a warning text instead of failing the entire call
                    user_content.append({
                        "type": "text",
                        "text": f"[WARN] failed to load image '{abs_path}': {e}"
                    })
                    continue

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            })

    if user_content:
        messages.append({
            "role": "user",
            "content": user_content
        })

    return messages

def llm_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic LLM call tool.

    Args:
        payload: Dictionary containing:
            - mode: "text" | "json" (default "text")
            - model: str (default from config or "gpt-5.1")
            - system_prompt: str (optional)
            - user_prompt: str (optional)
            - user_json: dict (optional; appended as fenced block)
            - images: List[str] (optional; image URLs or local paths)
            - temperature: float (optional, default 0.2)
            - max_tokens: int (optional, default None → model decides)
            - extra_create_params: dict (optional; pass-through to litellm)

    Returns:
        Dictionary with:
            - raw_text: str - The raw response text
            - json: dict or list - Parsed JSON (empty dict if mode is "text" or parsing failed)
            - ok: bool - Whether the call succeeded
            - error: str or None - Error message if failed
    """
    mode: str = payload.get("mode", "text")
    model: str = payload.get("model", DEFAULT_MODEL)
    system_prompt: str = payload.get("system_prompt", "")
    user_prompt: str = payload.get("user_prompt", "")
    user_json: Optional[Dict[str, Any]] = payload.get("user_json")
    images: Optional[List[str]] = payload.get("images")
    temperature: float = float(payload.get("temperature", 0.2))
    max_tokens: Optional[int] = payload.get("max_tokens")
    extra_create_params: Dict[str, Any] = payload.get("extra_create_params", {}) or {}
    # response_format is not universally supported by OpenAI-compatible gateways.
    # Default to OFF; callers can explicitly enable when their provider supports it.
    use_response_format: bool = bool(payload.get("use_response_format", False))

    # JSON mode enforcement: use stricter settings for better JSON output reliability
    json_mode_original_prompt = system_prompt
    if mode == "json":
        # For JSON mode, default to temperature=0 for more deterministic output
        # unless explicitly set by user
        if "temperature" not in payload:
            temperature = 0.0
        # Add JSON enforcement prompt
        system_prompt += """\n\n\n[IMPORTANT] Return ONLY valid JSON (a single top-level JSON object OR a JSON array).\n\nUse double quotes for all keys/strings. No trailing commas. [IMPORTANT]"""

    messages = _build_messages(system_prompt, user_prompt, user_json, images)

    # API from config (models.yaml) or env; optional model_config_path in payload for flow-specific config
    _config_path = payload.get("model_config_path")
    _api_key = get_api_key(_config_path)
    _base_url = get_api_base_url(_config_path)

    base_create_params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    # Avoid unbounded LiteLLM/provider waits. Callers can override these through
    # extra_create_params when they need a shorter or longer request timeout.
    if DEFAULT_LLM_REQUEST_TIMEOUT_S > 0:
        base_create_params["timeout"] = DEFAULT_LLM_REQUEST_TIMEOUT_S
        base_create_params["request_timeout"] = DEFAULT_LLM_REQUEST_TIMEOUT_S
    if _base_url:
        base_create_params["base_url"] = _base_url
    if _api_key:
        base_create_params["api_key"] = _api_key
    # if max_tokens is not None:
    #     base_create_params["max_tokens"] = int(max_tokens)
    max_tokens = MAX_TOKENS
    base_create_params["max_tokens"] = max_tokens
    # Optional: response_format for JSON mode when provider supports it
    if mode == "json" and use_response_format:
        base_create_params["response_format"] = {"type": "json_object"}
    
    # Apply extra params last, allowing override of response_format if needed
    if extra_create_params:
        base_create_params.update(extra_create_params)

    last_error = None
    last_content = ""
    # Increase retries for JSON mode to improve reliability
    max_attempts = 3 if mode == "json" else 2

    for attempt in range(1, max_attempts + 1):
        try:
            create_params = dict(base_create_params)
            
            # For JSON mode: use temperature=0 if not explicitly set by user, otherwise use user's value
            # For text mode: gradually reduce temperature on retries
            if mode == "json":
                # Only override if user didn't explicitly set temperature
                create_params["temperature"] = 0.3
            
            
            resp = completion(**create_params)
            content = resp.choices[0].message.content or ""

            if mode == "text":
                return {
                    "raw_text": content,
                    "json": {},
                    "ok": True,
                    "error": None,
                }

            # JSON mode: try to parse
            parsed = _safe_json_loads(content)
            if isinstance(parsed, (dict, list)):
                return {
                    "raw_text": content,
                    "json": parsed,
                    "ok": True,
                    "error": None,
                }

            # If parsing failed, keep content for caller to use (e.g. extract from raw text)
            last_content = content
            # If parsing failed, strengthen the prompt for next attempt
            if attempt < max_attempts and mode == "json":
                # Rebuild messages with stronger prompt for next attempt
                stronger_prompt = json_mode_original_prompt + f"""\n\n\n[IMPORTANT - RETRY {attempt + 1}] Return ONLY valid JSON (a single top-level JSON object OR a JSON array). Use double quotes for all keys/strings. No trailing commas. The previous response was invalid. [IMPORTANT]"""
                base_create_params["messages"] = _build_messages(
                    stronger_prompt, user_prompt, user_json, images
                )
            
            last_error = f"Failed to parse JSON from model response. content_len={len(content)} \n Raw content: {content[:200]}..."

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

    # All retries failed; still return last_content so caller can try to use it
    return {
        "raw_text": last_content,
        "json": {},
        "ok": False,
        "error": f"LLM call failed after {max_attempts} attempts. Last error: {last_error}",
    }

# -------------------------- Convenience wrappers ------------------------------

def llm_call_text(
    system_prompt: str,
    user_prompt: str = "",
    user_json: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Simple wrapper for text mode."""
    if model is None:
        model = DEFAULT_MODEL
    return llm_call({
        "mode": "text",
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "user_json": user_json,
        **kwargs
    })

def llm_call_json(
    system_prompt: str,
    user_prompt: str = "",
    user_json: Optional[Dict[str, Any]] = None,
    images: Optional[List[str]] = None,
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper for JSON mode with enhanced reliability.
    
    This function ensures JSON output by:
    1. Setting temperature=0 by default for deterministic output
    2. Using stronger prompts and multiple retry attempts
    3. Robust JSON parsing with fallback strategies
    4. (Optional) Using response_format when explicitly enabled and supported by provider
    
    Note: While this significantly improves JSON output reliability,
    it cannot guarantee 100% success if the model doesn't support
    structured outputs or if the prompt is fundamentally incompatible.
    
    Args:
        system_prompt: System prompt (will be enhanced with JSON enforcement)
        user_prompt: User prompt
        user_json: Optional JSON to include in the request
        images: Optional list of image URLs or paths
        model: Model name (defaults to DEFAULT_MODEL)
        **kwargs: Additional parameters passed to llm_call (e.g., use_response_format=True)
        
    Returns:
        Dictionary with 'ok', 'json', 'raw_text', and 'error' fields.
        If 'ok' is False, check 'error' for details.
    """
    if model is None:
        model = DEFAULT_MODEL
    return llm_call({
        "mode": "json",
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "user_json": user_json,
        "images": images,
        **kwargs
    })

