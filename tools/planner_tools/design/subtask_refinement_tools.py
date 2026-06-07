# tools/subtask_refinement_tools.py
# -----------------------------------------------------------------------------
# Subtask refinement tools used by the Design Agent: Revise one subtask,
# Replan full subtask set, Discard one subtask. Shared helper _upsert_subtask
# is used by Grounding Agent as well.
# -----------------------------------------------------------------------------
from typing import Dict, Any, List, Optional
import json

from utils.registry import register_tool
from utils.agent_utils import Result
from utils.llm_caller import llm_call_json
from utils.model_config import get_tool_model


def _get_refinement_model(model_config_path=None):
    """Model for revise/replan (same config key as former analyst_model)."""
    try:
        return get_tool_model("analyst_model", model_config_path)
    except Exception:
        return "gpt-5.1"


def _as_list_str(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _get_subtasks_list(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    subtasks = ctx.get("subtasks") or []
    sts = []
    for st in subtasks:
        sts.append({
            "id": st.get("id"),
            "name": st.get("name"),
            "description": st.get("description"),
            "answer_type": st.get("answer_type"),
            "modalities": st.get("modalities", []),
            "sample_schema": st.get("sample_schema"),
            "keywords": st.get("keywords"),
        })
    return sts if isinstance(sts, list) else []


def _upsert_subtask(subtasks: List[Dict[str, Any]], updated: Dict[str, Any]) -> List[Dict[str, Any]]:
    sid = (updated.get("id") or "").strip()
    if not sid:
        subtasks.append(updated)
        return subtasks
    for i, st in enumerate(subtasks):
        if (st.get("id") or "").strip() == sid:
            subtasks[i] = updated
            return subtasks
    subtasks.append(updated)
    return subtasks


def _normalize_field_spec(fs: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(fs, dict):
        out["dtype"] = str(fs.get("dtype") or "").strip().lower()
        out["subtype"] = str(fs.get("subtype") or "").strip().lower()
        out["type"] = str(fs.get("type") or "").strip().lower()
    else:
        out["dtype"] = out["subtype"] = ""
        out["type"] = "str"
    if out["dtype"] not in ("audio", "text", "image"):
        out["dtype"] = out["dtype"] or ""
    if out["type"] not in ("str", "list"):
        out["type"] = "str"
    return out


def _normalize_sample_schema(ss: Any) -> Dict[str, Any]:
    if not isinstance(ss, dict):
        return {"input": {"fields": {}}, "output": {"fields": {}}}
    inp = ss.get("input") if isinstance(ss.get("input"), dict) else {}
    outp = ss.get("output") if isinstance(ss.get("output"), dict) else {}
    in_fields = inp.get("fields") if isinstance(inp.get("fields"), dict) else {}
    out_fields = outp.get("fields") if isinstance(outp.get("fields"), dict) else {}
    allowed_input_fields = {"question", "context", "image_url", "audio_url"}
    norm_in = {
        str(k).strip(): _normalize_field_spec(v)
        for k, v in in_fields.items()
        if str(k).strip() in allowed_input_fields
    }
    norm_out = {"answer": _normalize_field_spec(out_fields.get("answer", {}))}
    return {"input": {"fields": norm_in}, "output": {"fields": norm_out}}


_ALLOWED_ANSWER_TYPES = {"binary", "choice", "label", "span"}


def _normalize_subtask(st: Dict[str, Any], global_modalities: List = None) -> Dict[str, Any]:
    st = dict(st or {})
    st.setdefault("id", "st_unknown")
    st.setdefault("name", None)
    st.setdefault("description", "No description.")
    st.setdefault("sample_schema", {"input": {"fields": {}}, "output": {"fields": {}}})
    st.setdefault("keywords", [])
    st.setdefault("modalities", global_modalities or [])
    st.setdefault("answer_type", "choice")
    st["id"] = (str(st["id"]).strip() or "st_unknown").lower().replace(" ", "_")
    st["name"] = str(st["name"] or "").strip() or st["id"]
    st["description"] = str(st["description"]).strip() or "No description."
    at = str(st.get("answer_type") or "").strip().lower()
    if at not in _ALLOWED_ANSWER_TYPES:
        at = "choice"
    st["answer_type"] = at
    st["modalities"] = _as_list_str(st.get("modalities"))
    st["keywords"] = _as_list_str(st.get("keywords"))
    st["sample_schema"] = _normalize_sample_schema(st.get("sample_schema"))
    in_fields = st["sample_schema"]["input"]["fields"]
    if "question" not in in_fields:
        in_fields["question"] = {"dtype": "text", "subtype": "question", "type": "str"}
    out_fields = st["sample_schema"]["output"]["fields"]
    if "answer" not in out_fields:
        out_fields["answer"] = {"dtype": "text", "subtype": "", "type": "str"}
    ans = out_fields["answer"]
    if not ans.get("subtype"):
        ans["subtype"] = "yes_no" if at == "binary" else "multiple_choice" if at == "choice" else "multi_class" if at == "label" else "multiple_choice"
    return st


_REVISE_PROMPT = r"""
You are revising ONE benchmark subtask. The **designer's instruction** below is the main directive—apply it to the given subtask. Your job is to produce the revised subtask that matches what the designer asked for.

Designer's guidance:
{guidance}

Current subtask (JSON):
{subtask_json}

Context: allowed modalities for this benchmark: {global_modalities}; retrieval keywords: {keywords}. You may use these to inform the revision only if the guidance implies it.

Rules:
- Output a single JSON object with exactly these fields: "id", "name", "description", "answer_type", "modalities", "sample_schema", "keywords".
- Keep "id" unchanged. Preserve the evaluation intent and use answer_type in {{binary, choice, label, span}}.
- Keep the sample schema simple: input is "question" plus optional "context" and/or one modality URL ("audio_url", "image_url"); output must contain only "answer".
- Prefer answer_type="choice" when it fits. Put choices inside the question or compact context, not in a separate "candidates" or "options" field.
- Do not add fields such as "evidence", "supporting_turns", "answer_id", "rationale", "label", or "explanation".
- Change only what the guidance asks for; leave the rest intact unless the guidance implies otherwise.
- Return ONLY the JSON object; no extra text or markdown.
"""

@register_tool("llm_revise_one_subtask")
def llm_revise_one_subtask(
    decision_rationale: str,
    subtask_id: str,
    guidance: str,
    context_variables: Optional[dict] = None,
) -> Result:
    """Revise one existing subtask using the provided guidance while keeping the same id.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    - subtask_id: The ID of the subtask to revise.
    - guidance: The guidance for the better proposal.
    """
    model_config_path = (context_variables or {}).get("model_config_path")
    model = _get_refinement_model(model_config_path)
    ctx = dict(context_variables or {})
    subtasks = _get_subtasks_list(ctx)
    full_subtasks = ctx.get("subtasks", []) or []
    global_modalities = ctx.get("modalities")
    keywords = ctx.get("keywords") or []
    target = next((s for s in subtasks if (s.get("id") or "").strip() == (subtask_id or "").strip()), None)
    if target is None:
        raise ValueError(f"[llm_revise_one_subtask] subtask_id not found: {subtask_id}")

    st = _normalize_subtask(target, global_modalities=global_modalities)
    subtask_id = st["id"]
    prompt = _REVISE_PROMPT.format(
        subtask_json=json.dumps(st, ensure_ascii=False, indent=2),
        global_modalities=json.dumps(_as_list_str(global_modalities), ensure_ascii=False),
        keywords=json.dumps(_as_list_str(keywords), ensure_ascii=False),
        guidance=(guidance or "").strip() or "Make the subtask easier to match real datasets without changing the intent.",
    )
    resp = llm_call_json(system_prompt="", user_prompt=prompt, model=model)
    if not resp.get("ok"):
        raise RuntimeError(f"[llm_revise_one_subtask] JSON parse failed: {resp.get('error')}")
    revised_raw = resp.get("json") or {}
    revised = _normalize_subtask(revised_raw, global_modalities=global_modalities)
    old = next((s for s in full_subtasks if (s.get("id") or "").strip() == subtask_id), None)
    merged = dict(old or {})
    merged.update(revised)
    merged["retrieval_result"] = []
    merged["scored_status"] = "no"
    merged["scored_candidates"] = {}
    merged["dataset_preference"] = {}
    merged["transformability"] = {}
    merged["selected_candidate_ids"] = []
    merged["notes"] = "revised; needs re-grounding"
    ctx["subtasks"] = _upsert_subtask(full_subtasks, merged)
    msg = f'revised subtask "{(st.get("name") or subtask_id)}" to "{(revised.get("name") or subtask_id)}"'
    return Result(value=msg, context_variables=ctx)


@register_tool("discard_subtask")
def discard_subtask(
    decision_rationale: str,
    subtask_id: str,
    context_variables: Optional[dict] = None,
) -> Result:
    """Remove one subtask from the current working set.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    - subtask_id: The ID of the subtask to discard.
    """
    ctx = dict(context_variables or {})
    subtasks = ctx.get("subtasks", []) or []
    new_subtasks = []
    found = False
    for sub in subtasks:
        if (sub.get("id") or "").strip() == (subtask_id or "").strip():
            found = True
            continue
        new_subtasks.append(sub)
    if not found:
        raise ValueError(f"[discard_subtask] subtask_id not found: {subtask_id}")
    ctx["subtasks"] = new_subtasks
    msg = f'discarded subtask "{subtask_id}"'
    return Result(value=msg, context_variables=ctx)
