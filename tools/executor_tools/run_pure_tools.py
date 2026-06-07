import sys
import importlib
import os
from typing import Any, Dict

from cv2 import add

from utils.llm_caller import llm_call_json
import json

_LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "90"))

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

@dataclass
class PureToolSpec:
    name: str
    category: str
    description: str
    io_inputs: List[str]
    io_outputs: List[str]
    behavior: List[str]
    params_doc: Dict[str, str]           
    typical_uses: List[str]

    param_schema: Dict[str, Any]         # Per-parameter structure description
    return_schema: Dict[str, Any]        # Tool return-value structure description

    backend: Callable[..., Dict[str, Any]]

    raw_cfg: Dict[str, Any]

# ==============================
# LLM Prompt Templates
# ==============================
# NOTE: We intentionally define the prompt as a module-level template (like Stage2 prompts),
# and inject the per-call payload via a placeholder replacement to avoid fragile string concat.
PURE_TOOL_PLANNER_PROMPT = '''
You receive:
- ONE pure tool specification (what the tool does, parameters, return values, behavior, typical uses),
- ONE step configuration (how this tool is supposed to be used in the tool plan),
- ONE resources object:
  - It contains ONLY the fields listed in step.resource_fields,
    expanded into nested JSON structure (e.g., original.*, tool_memory.*, interim.*, final.*).

Your job:
1. Decide CONCRETE values for the tool's parameters ("tool_args").
2. DO NOT fabricate the tool's return values. You are not executing the tool.

Default memory model:
- PURE tool target_fields identify a stable memory base path, usually:
  `tool_memory.<tool_name>.<operation_key>`.
- The executor, not you, writes:
  - `<memory_base>.input`  = tool_args
  - `<memory_base>.output` = backend tool result
- Do NOT plan writes to interim.* or final.*. Later LLM steps read tool_memory through explicit resource_fields and write any needed interim/final fields themselves.

Important:
- You MUST return STRICT JSON.
- Return a SINGLE top-level JSON object. No extra text. No markdown fences.
- You MUST follow the tool's parameter schema and return schema EXACTLY.
- You MUST use realistic values for tool parameters, based on the provided resources, not placeholders.
- Output JSON fields: tool_name, tool_args, notes.
- tool_name MUST equal tool_spec.name.
- tool_args MUST match tool_spec.param_schema (required fields present, types match).
- Do NOT invent hidden/backend parameters that are not in tool_spec.param_schema.
- Do NOT output write_back. It is ignored by the main executor.

Special rule for web_search:
- The upper-level transformability plan should not contain concrete params; generate them here from this sample's resources.
- The only exposed arguments are `query` and optional `image_paths`.
- `query` means the concrete web search request for THIS tool call: it should state exactly what **factual** information to retrieve now, using available resource values or attached images as anchors. It is not a generic restatement of the subtask or tool name.
- **Use `image_paths` whenever this sample has usable image file paths in resources** and the search would be more accurate or grounded by sending those pixels (identity, category, place/object, artwork, chart/diagram, satellite/medical-style imagery for public-reference facts, etc.). Prefer image+text over text-only when both are available and facts depend on what is shown. Omit `image_paths` only when the search is strictly textual (e.g., spelling of a string already fully given in text) or no image paths exist in resources.
- Do NOT output force_search, search_context_size, max_output_chars, max_hops, return_trace, or any other backend-only argument.
- The tool returns answer, sources, and insufficient_info. The executor stores the concrete query/image_paths under `.input` and the returned answer under `.output`.

Example output (illustration only; choose values based on resources):
"""
{
  "tool_name": "text2speech",
  "tool_args": {
    "dialog": [
      {"text": "Ewww! Oh! It's the Mattress King!", "speaker": "Speaker 1", "language": "en"},
      {"text": "Booo!!", "speaker": "Speaker 2", "language": "en"}
    ]
  },
  "notes": "Use interim.dialog (or original.input.raw_text) to build dialog turns; executor will retain dialog and merged_audio_path in tool_memory."
}
"""

Below is the input JSON for planning this pure tool call:
"""
<<USER_PAYLOAD_JSON>>
"""

Return ONLY one JSON object with fields: tool_name, tool_args, notes.
'''.strip()

# ==============================
# Pure Tool Spec Builders
# ==============================
def _build_spec_text2speech(tool_cfg: Dict[str, Any]) -> PureToolSpec:
    from tools.executor_tools.implementations.audio_tools import text2speech
    io_cfg = tool_cfg.get("io") or {}
    params_list = tool_cfg.get("params") or []

    # YAML params is a list of {name: doc}
    params_doc: Dict[str, str] = {}
    for p in params_list:
        if isinstance(p, dict):
            for k, v in p.items():
                params_doc[k] = str(v)

    # Structured param_schema for the LLM; keep descriptions concise
    param_schema = {
        "dialog": {
        "type": "list",
        "required": True,
        "description": "Ordered list of dialog turns. IMPORTANT: each item MUST correspond to exactly ONE language.",
        "item_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "required": True,
                    "description": (
                        "Utterance content. MUST be in a single dominant language. "
                        "Must be natural language, which is considered as valid human utterances."
                        "DO NOT include any unnecessary characters like [laughs], [good], etc."
                        "DO NOT include structure text like speaker ID."
                        "If a speaker talks in multiple languages, SPLIT into multiple items."
                        "MUST be less than 200 characters, if exceeds, split into multiple items but keep the same speaker."
                    ),
                },
                "speaker": {
                    "type": "string",
                    "required": False,
                    "description": (
                        "Speaker ID or name. Required only when multiple speakers are needed. "
                        "If splitting a mixed-language utterance, reuse the SAME speaker value."
                    ),
                },
                "language": {
                    "type": "string",
                    "required": False,
                    "description": (
                        "Language code for THIS item only, only supported language codes in ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']. "
                    ),
                },
                "gender": {
                    "type": "string",
                    "required": False,
                    "enum": ["male", "female"],
                    "description": "Preferred voice gender if supported.",
                },
            },
        },
    }

    }

    # Return schema: single merged audio file path only
    return_schema = {
        "merged_audio_path": {
            "type": "string",
            "description": "Real system path to single audio file for the whole dialog concatenated in order."
        }
    }

    spec = PureToolSpec(
        name=tool_cfg.get("name", "text2speech"),
        category=tool_cfg.get("category", ""),
        description=tool_cfg.get("description", ""),
        io_inputs=list(io_cfg.get("inputs") or []),
        io_outputs=list(io_cfg.get("outputs") or []),
        behavior=list(tool_cfg.get("behavior") or []),
        params_doc=params_doc,
        typical_uses=list(tool_cfg.get("typical_uses") or []),
        param_schema=param_schema,
        return_schema=return_schema,
        backend=text2speech,  
        raw_cfg=tool_cfg,
    )
    return spec

def _build_spec_add_environmental_noise(tool_cfg: Dict[str, Any]) -> PureToolSpec:
    from tools.executor_tools.implementations.audio_tools import add_environmental_noise
    io_cfg = tool_cfg.get("io") or {}
    params_list = tool_cfg.get("params") or []

    # YAML params is a list of {name: doc}
    params_doc: Dict[str, str] = {}
    for p in params_list:
        if isinstance(p, dict):
            for k, v in p.items():
                params_doc[k] = str(v)
    TUT_NOISE_TYPES = [
    "beach",
    "bus",
    "cafe/restaurant",
    "car",
    "city_center",
    "forest_path",
    "grocery_store",
    "home",
    "library",
    "metro_station",
    "office",
    "park",
    "residential_area",
    "train",
    "tram",
]
    # -------------------------------
    # param_schema for the LLM
    # -------------------------------
    param_schema = {
        "audio_path": {
            "type": "string",
            "required": True,
            "description": "Real system path to clean audio file."
        },
        "noise_type": {
            "type": "string",
            "required": True,
            "description": (
                "Scene noise type. Valid types from TUT noise dataset: "
                + ", ".join(TUT_NOISE_TYPES)
            ),
        },
        "intensity": {
            "type": "string",
            "required": False,
            "enum": ["low", "medium", "high"],
            "default": "medium",
            "description": "Noise intensity mapped to SNR levels."
        },
        "seed": {
            "type": "integer",
            "required": False,
            "description": "Random seed for selecting noise file."
        },
    }

    # -------------------------------
    # return_schema for the LLM
    # -------------------------------
    return_schema = {
        "noisy_audio_path": {
            "type": "string",
            "description": "Real system path to noisy audio file."
        }
    }

    spec = PureToolSpec(
        name=tool_cfg.get("name", "add_environmental_noise"),
        category=tool_cfg.get("category", ""),
        description=tool_cfg.get("description", ""),
        io_inputs=list(io_cfg.get("inputs") or []),
        io_outputs=list(io_cfg.get("outputs") or []),
        behavior=list(tool_cfg.get("behavior") or []),
        params_doc=params_doc,
        typical_uses=list(tool_cfg.get("typical_uses") or []),
        param_schema=param_schema,
        return_schema=return_schema,
        backend=add_environmental_noise,
        raw_cfg=tool_cfg,
    )

    return spec

def _build_spec_translate(tool_cfg: Dict[str, Any]) -> PureToolSpec:
    from tools.executor_tools.implementations.text_tools import translate
    io_cfg = tool_cfg.get("io") or {}
    params_list = tool_cfg.get("params") or []

    params_doc: Dict[str, str] = {}
    for p in params_list:
        if isinstance(p, dict):
            for k, v in p.items():
                params_doc[k] = str(v)

    param_schema = {
        "translate_items": {
            "type": "list",
            "required": True,
            "description": "List of translation units. Each item is an object with fields: text (string, required), target_language (string, required).",
            "item_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "required": True,
                        "description": "Source text to translate."
                    },
                    "target_language": {
                        "type": "string",
                        "required": True,
                        "description": "Target language code, only supported language codes in ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']."
                    },
                },
            },
        }
    }

    return_schema = {
        "translated_items": {
            "type": "list",
            "description": (
                "List of translated results. "
                "Each item has: source_text, target_language, translated_text."
            )
        }
    }

    spec = PureToolSpec(
        name=tool_cfg.get("name", "translate"),
        category=tool_cfg.get("category", ""),
        description=tool_cfg.get("description", ""),
        io_inputs=list(io_cfg.get("inputs") or []),
        io_outputs=list(io_cfg.get("outputs") or []),
        behavior=list(tool_cfg.get("behavior") or []),
        params_doc=params_doc,
        typical_uses=list(tool_cfg.get("typical_uses") or []),
        param_schema=param_schema,
        return_schema=return_schema,
        backend=translate,
        raw_cfg=tool_cfg,
    )
    return spec


def _build_spec_web_search(tool_cfg: Dict[str, Any]) -> PureToolSpec:
    from tools.executor_tools.implementations.web_tools import web_search
    io_cfg = tool_cfg.get("io") or {}
    params_list = tool_cfg.get("params") or []

    params_doc: Dict[str, str] = {}
    for p in params_list:
        if isinstance(p, dict):
            for k, v in p.items():
                params_doc[k] = str(v)

    param_schema = {
        "query": {
            "type": "string",
            "required": True,
            "description": "Web search query text.",
        },
        "image_paths": {
            "type": "list",
            "required": False,
            "description": "Optional list of local image paths or image URLs used as visual context for search.",
            "item_schema": {"type": "string"},
        },
    }

    return_schema = {
        "answer": {"type": "string", "description": "Search-grounded answer text."},
    }

    spec = PureToolSpec(
        name=tool_cfg.get("name", "web_search"),
        category=tool_cfg.get("category", ""),
        description=tool_cfg.get("description", ""),
        io_inputs=list(io_cfg.get("inputs") or []),
        io_outputs=list(io_cfg.get("outputs") or []),
        behavior=list(tool_cfg.get("behavior") or []),
        params_doc=params_doc,
        typical_uses=list(tool_cfg.get("typical_uses") or []),
        param_schema=param_schema,
        return_schema=return_schema,
        backend=web_search,
        raw_cfg=tool_cfg,
    )
    return spec

def _build_spec_cleanup_interim_fields(tool_cfg: Dict[str, Any]) -> PureToolSpec:
    io_cfg = tool_cfg.get("io") or {}
    params_list = tool_cfg.get("params") or []

    params_doc: Dict[str, str] = {}
    for p in params_list:
        if isinstance(p, dict):
            for k, v in p.items():
                params_doc[k] = str(v)

    def cleanup_interim_fields() -> Dict[str, Any]:
        # No-op pure tool: actual field deletion is handled by deleted_interim_fields
        # in the transformation executor after each step.
        return {"status": "ok"}

    spec = PureToolSpec(
        name=tool_cfg.get("name", "cleanup_interim_fields"),
        category=tool_cfg.get("category", ""),
        description=tool_cfg.get("description", ""),
        io_inputs=list(io_cfg.get("inputs") or []),
        io_outputs=list(io_cfg.get("outputs") or []),
        behavior=list(tool_cfg.get("behavior") or []),
        params_doc=params_doc,
        typical_uses=list(tool_cfg.get("typical_uses") or []),
        param_schema={},
        return_schema={
            "status": {
                "type": "str",
                "description": "always 'ok' for no-op cleanup marker tool",
            }
        },
        backend=cleanup_interim_fields,
        raw_cfg=tool_cfg,
    )
    return spec

# ==============================
# Build Pure Tool Registry from Context
# =============================`
def build_pure_tool_registry(tools_cfg: Dict[str, Any]) -> Dict[str, PureToolSpec]:
    registry: Dict[str, PureToolSpec] = {}

    for t in tools_cfg:
        name = t.get("name")
        if name == "text2speech":
            spec = _build_spec_text2speech(t)
        elif name == "translate":
            spec = _build_spec_translate(t)
        elif name == "web_search":
            spec = _build_spec_web_search(t)
        elif name == "add_environmental_noise":
            spec = _build_spec_add_environmental_noise(t)
        elif name == "cleanup_interim_fields":
            spec = _build_spec_cleanup_interim_fields(t)
        else:
            continue  # Unknown pure tool; skip
        registry[spec.name] = spec

    return registry


def _llm_plan_pure_call(
    subtask: Dict[str, Any],
    resources: Dict[str, Any],
    step: Dict[str, Any],
    tool_spec: Dict[str, Any],
    model: str = "gpt-5.1",
) -> Dict[str, Any]:

    # Defensive: this should ALWAYS be a dict built from PureToolSpec.
    # If it's None/invalid, fail fast with a clear error (otherwise debugging is confusing).
    if not isinstance(tool_spec, dict) or not tool_spec:
        raise ValueError(
            f"pure_tool_spec_missing_or_invalid: {type(tool_spec).__name__} value={tool_spec}"
        )

    user_payload = {
        "subtask": subtask,
        "resources": resources,   # Subset corresponding to resource_fields only
        "step": step,
        "tool_spec": tool_spec,
    }

    user_payload_json = json.dumps(user_payload, ensure_ascii=False, indent=2)
    user_prompt = PURE_TOOL_PLANNER_PROMPT.replace("<<USER_PAYLOAD_JSON>>", user_payload_json)

    resp = llm_call_json(
        system_prompt="You are a meticulous and precise tool invocation planner. Always follow the instructions exactly.",
        user_prompt=user_prompt,
        model=model,
        extra_create_params={
            "timeout": _LLM_TIMEOUT_S,
            "request_timeout": _LLM_TIMEOUT_S,
        },
    )

    if not isinstance(resp, dict):
        raise ValueError(f"LLM pure-tool planner did not return a JSON object: {resp}")

    # llm_call_json returns: {raw_text, json, ok, error}
    if resp.get("ok") is False:
        raise ValueError(
            f"LLM pure-tool planner call failed: {resp.get('error')}"
        )

    plan = resp.get("json")
    if isinstance(plan, list):
        plan = plan[0] if plan else {}
    if not isinstance(plan, dict):
        raise ValueError(f"LLM pure-tool planner returned non-dict json: {type(plan).__name__} value={plan}")

    return plan

from typing import Tuple
import copy

def _get_by_path(obj: Dict[str, Any], path: str) -> Any:
    if not path:
        return obj
    cur: Any = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

def _set_by_path(obj: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    if not path:
        raise ValueError("path cannot be empty")
    parts = path.split(".")
    cur = obj
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value
    return obj


def _is_effectively_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _safe_memory_key(value: Any, fallback: str) -> str:
    raw = str(value or fallback or "memory").strip()
    chars: List[str] = []
    prev_underscore = False
    for ch in raw:
        if ch.isalnum():
            chars.append(ch.lower())
            prev_underscore = False
        else:
            if not prev_underscore:
                chars.append("_")
                prev_underscore = True
    key = "".join(chars).strip("_")
    return key or str(fallback or "memory")


def _tool_memory_base_path(step: Dict[str, Any], tool_name: str) -> str:
    target_fields = step.get("target_fields") or []
    for path in target_fields:
        if isinstance(path, str) and path.startswith("tool_memory."):
            return path.rstrip(".")

    # Cleanup-only tools are allowed to have no memory target.
    if str(step.get("step_name") or "") == "cleanup_interim_fields":
        return ""

    step_key = _safe_memory_key(step.get("step_name") or step.get("operation"), "step")
    tool_key = _safe_memory_key(tool_name, "tool")
    return f"tool_memory.{tool_key}.{step_key}"


def _build_tool_memory_record(
    *,
    tool_spec: PureToolSpec,
    step: Dict[str, Any],
    tool_args: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    target_base = _tool_memory_base_path(step, tool_spec.name)
    if not target_base:
        return {}

    return {
        "target_base": target_base,
        "tool_name": tool_spec.name,
        "step_name": step.get("step_name") or tool_spec.name,
        "step_index": step.get("step_index", -1),
        "kind": step.get("kind"),
        "input": copy.deepcopy(tool_args if isinstance(tool_args, dict) else {}),
        "output": copy.deepcopy(result if isinstance(result, dict) else {}),
    }


def _build_tool_registry(tool_spec) -> Dict[str, Any]:
    return {
        "name": tool_spec.name,
        "description": tool_spec.description,
        "param_schema": tool_spec.param_schema,
        "return_schema": tool_spec.return_schema,
    }

def run_pure_step(
    subtask: Dict[str, Any],
    resources: Dict[str, Any],
    step: Dict[str, Any],
    tool_registry: Dict[str, PureToolSpec],
    dataset_json_path: str = None,
    dataset_id: str = None,
) -> Tuple[bool, Dict[str, Any], str, Dict[str, Any]]:
    """
    Execute ONE pure tool step for ONE sample.
    Returns:
      ok: whether execution succeeded
      new_current: updated current state
      note: text note (LLM notes or error reason)
    """
    step_name = step.get("step_name")
    if not step_name:
        return False, {}, "step_name_missing", {}

    tool_spec = tool_registry.get(step_name)
    if tool_spec is None:
        return False, {}, f"pure_tool_not_registered[{step_name}]", {}

    # 1) LLM planning
    try:
        plan = _llm_plan_pure_call(
            subtask=subtask,
            resources=resources,
            step=step,
            tool_spec=_build_tool_registry(tool_spec),
        )
        # print(f"Pure tool plan for step {step_name}")
    except Exception as e:
        return False, {}, f"llm_plan_error: {e}", {}

    try:
        tool_name = plan.get("tool_name") or step_name
    except Exception as e:
        return False, {}, f"plan_tool_name_missing[{step_name}]", {}
    note = plan.get("notes") or ""
    tool_args = plan.get("tool_args") or {}
    target_fields = [p for p in (step.get("target_fields") or []) if isinstance(p, str) and p]
    if step_name != "cleanup_interim_fields" and any(
        not p.startswith("tool_memory.") for p in target_fields
    ):
        return (
            False,
            {},
            f"pure_tool_target_must_be_tool_memory[{step_name}]: {target_fields}",
            {},
        )
    if step_name != "cleanup_interim_fields" and len(target_fields) > 1:
        return (
            False,
            {},
            f"pure_tool_target_must_be_single_tool_memory_base[{step_name}]: {target_fields}",
            {},
        )

    # Resolve image_paths from basename-only values when planner emits web/image tool args.
    if isinstance(tool_args, dict) and isinstance(tool_args.get("image_paths"), list):
        try:
            from tools.shared.media_paths import resolve_image_paths
            resolved = resolve_image_paths(
                tool_args.get("image_paths") or [],
                dataset_json_path=dataset_json_path,
                dataset_id=dataset_id,
            )
            if resolved:
                tool_args["image_paths"] = resolved
        except Exception:
            pass

    # 1.5) Validate + auto-fill required args (defensive against planner omissions)
    try:
        param_schema = tool_spec.param_schema or {}
        required_keys = [
            k for k, info in param_schema.items()
            if isinstance(info, dict) and bool(info.get("required"))
        ]
    except Exception:
        required_keys = []

    missing_required = [
        k for k in required_keys
        if (not isinstance(tool_args, dict)) or (k not in tool_args) or _is_effectively_empty(tool_args.get(k))
    ]

    if missing_required:
        if step_name == "text2speech" and "dialog" in missing_required:
            return (
                False,
                {},
                "pure_tool_missing_required_args[text2speech]: ['dialog']. "
                "Planner did not provide required tool_args.dialog. "
                "Fix Stage1 planning: create a structured dialog list (e.g., interim.dialog/turn_list/dialogue_turns) "
                "and include that field in step.resource_fields so the pure-tool planner can use it.",
                {},
            )
        return (
            False,
            {},
            f"pure_tool_missing_required_args[{step_name}]: {missing_required}. "
            f"Ensure step.resource_fields includes the needed inputs (e.g., interim.dialog or original.input.raw_text).",
            {},
        )

    # 2) Execute backend
    backend_fn = tool_spec.backend
    
    try:
        result = backend_fn(**tool_args)
    except Exception as e:
        return False, {}, f"tool_runtime_error[{step_name}]: {e}", {}

    if not isinstance(result, dict):
        return False, {}, f"tool_result_not_dict[{step_name}]", {}

    # 3) PURE tools do not write interim/final directly. The transform executor
    # stores the memory record under tool_memory.<tool>.<step>.
    delta_current: Dict[str, Any] = {}
    additional_messages: Dict[str, Any] = {}
    if step_name != "cleanup_interim_fields":
        additional_messages[f"{tool_name}_tool_args"] = tool_args
        if tool_name == "web_search" or step_name == "web_search":
            additional_messages["web_search_result"] = result
        memory_record = _build_tool_memory_record(
            tool_spec=tool_spec,
            step=step,
            tool_args=tool_args,
            result=result,
        )
        if memory_record:
            additional_messages["_tool_memory_record"] = memory_record

    return True, delta_current, note, additional_messages