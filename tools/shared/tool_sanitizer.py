from typing import Any, Callable, Dict, List, Set, Tuple
import json
import os

from utils.llm_caller import llm_call_json

_LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "90"))

_FIX_PURE_TOOL_PROMPT = r"""
You are a post-processor that must FIX ONE PURE TOOL OPERATION.

You receive:
- tools_json: a JSON list of PURE tools, each with a "name" field.
- op_json: ONE operation object from a transformability plan.

Your task:
- Ensure the PURE tool name field is EXACTLY one of the tool names in tools_json.
  Some callers use op_json["operation"]; Stage1 next_step objects use op_json["step_name"].
- If the current value does not match any tool name, choose the most appropriate tool
  based on "notes", "target_fields" and other fields.
- NEVER invent new tool names.
- ONLY modify "operation" and/or "step_name"; keep other fields as-is.

Return ONLY the fixed operation JSON object (no extra text).
tools_json:
{tools_json}

op_json:
{op_json}
"""


def _extract_pure_tool_names(tool_list_json: List[Dict[str, Any]]) -> Set[str]:
    names: Set[str] = set()
    for t in tool_list_json:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _get_pure_tool_name_field(op: Dict[str, Any]) -> str:
    return str(op.get("operation") or op.get("step_name") or "").strip()


def _sync_pure_tool_name_fields(op: Dict[str, Any], name: str) -> None:
    if "operation" in op or "step_name" not in op:
        op["operation"] = name
    if "step_name" in op:
        op["step_name"] = name


def _llm_fix_pure_tool_op(
    op: Dict[str, Any],
    tool_list_json: List[Dict[str, Any]],
    pure_tool_names: Set[str],
    model: str,
) -> Dict[str, Any]:
    try:
        prompt = _FIX_PURE_TOOL_PROMPT.format(
            tools_json=json.dumps(tool_list_json, ensure_ascii=False, indent=2),
            op_json=json.dumps(op, ensure_ascii=False, indent=2),
        )
        resp = llm_call_json(
            system_prompt="",
            user_prompt=prompt,
            model=model,
            extra_create_params={
                "timeout": _LLM_TIMEOUT_S,
                "request_timeout": _LLM_TIMEOUT_S,
            },
        )
        if not resp.get("ok"):
            return op
        fixed = resp.get("json")
        if not isinstance(fixed, dict):
            return op
        new_name = str(fixed.get("operation") or fixed.get("step_name") or "").strip()
        if new_name in pure_tool_names:
            _sync_pure_tool_name_fields(op, new_name)
        return op
    except Exception:
        return op


def _registry_to_tool_list_and_names(
    pure_tool_registry: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    reg = pure_tool_registry or {}
    pure_tool_names: Set[str] = set()
    tool_list_json: List[Dict[str, Any]] = []
    for name, spec in reg.items():
        if not isinstance(name, str) or not name.strip():
            continue
        pure_tool_names.add(name.strip())
        if hasattr(spec, "raw_cfg") and isinstance(getattr(spec, "raw_cfg"), dict):
            tool_list_json.append(getattr(spec, "raw_cfg"))
        elif isinstance(spec, dict) and spec:
            tool_list_json.append(spec)
        else:
            tool_list_json.append({"name": name})
    return tool_list_json, pure_tool_names


def _sanitize_pure_tool_op(
    plan: Any,
    pure_tool_registry: Dict[str, Any],
    model: str,
) -> Any:
    if not isinstance(plan, dict):
        return plan

    tool_list_json, pure_tool_names = _registry_to_tool_list_and_names(pure_tool_registry)
    if not pure_tool_names:
        return plan

    tool_type = str(plan.get("tool_type") or "").upper()
    if tool_type != "PURE":
        return plan

    op_name = _get_pure_tool_name_field(plan)
    if op_name in pure_tool_names:
        _sync_pure_tool_name_fields(plan, op_name)
        return plan

    notes = str(plan.get("notes") or "")
    matched_names = [n for n in pure_tool_names if n in notes]
    if len(matched_names) == 1:
        _sync_pure_tool_name_fields(plan, matched_names[0])
        return plan

    print(f"[sanitize_pure_tool_op] fixing PURE tool op via LLM")
    return _llm_fix_pure_tool_op(
        op=plan,
        tool_list_json=tool_list_json,
        pure_tool_names=pure_tool_names,
        model=model,
    )


def _sanitize_pure_tool_ops(
    plan: Any,
    tool_list_json: List[Dict[str, Any]],
    model: str,
) -> Any:
    if isinstance(plan, dict):
        pseudo_reg = {
            str(t.get("name") or "").strip(): t
            for t in (tool_list_json or [])
            if isinstance(t, dict) and (str(t.get("name") or "").strip())
        }
        return _sanitize_pure_tool_op(plan, pseudo_reg, model)
    if not isinstance(plan, list):
        return plan

    pure_tool_names = _extract_pure_tool_names(tool_list_json)
    if not pure_tool_names:
        return plan

    new_plan: List[Any] = []
    for op in plan:
        if not isinstance(op, dict):
            new_plan.append(op)
            continue

        tool_type = str(op.get("tool_type") or "").upper()
        if tool_type != "PURE":
            new_plan.append(op)
            continue

        op_name = _get_pure_tool_name_field(op)
        if op_name in pure_tool_names:
            _sync_pure_tool_name_fields(op, op_name)
            new_plan.append(op)
            continue

        notes = str(op.get("notes") or "")
        matched_names = [n for n in pure_tool_names if n in notes]
        if len(matched_names) == 1:
            _sync_pure_tool_name_fields(op, matched_names[0])
            new_plan.append(op)
            continue

        print(f"[sanitize_pure_tool_ops] fixing PURE tool op via LLM")
        fixed_op = _llm_fix_pure_tool_op(
            op=op,
            tool_list_json=tool_list_json,
            pure_tool_names=pure_tool_names,
            model=model,
        )
        new_plan.append(fixed_op)

    return new_plan


# ---------------------------------------------------------------------------
# Modality hard validators
# ---------------------------------------------------------------------------

def validate_audio(sample_final: Dict[str, Any]) -> Tuple[bool, str]:
    """Hard rule: final.input.audio_url must exist and be non-empty."""
    final_input = sample_final.get("input") or {}
    audio_url = final_input.get("audio_url")
    if not audio_url:
        return False, "audio modality requires `final.input.audio_url` but it is missing or empty."
    return True, ""


MODAL_VALIDATORS: Dict[str, Callable[[Dict[str, Any]], Tuple[bool, str]]] = {
    "audio": validate_audio,
}


def run_modal_validators(
    sample_final: Dict[str, Any],
    validators: List[Callable[[Dict[str, Any]], Tuple[bool, str]]],
) -> Tuple[bool, str]:
    """Run all modality validators in order; return False on the first failure."""
    for validator in validators:
        ok, reason = validator(sample_final)
        if not ok:
            return False, reason
    return True, ""
