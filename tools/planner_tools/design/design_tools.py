# tools/design_tools.py
# -----------------------------------------------------------------------------
# Tools for the Design Agent: Proposer (propose candidates only), process_proposed_subtasks
# (select into working set), Revise, Discard, case_resolved.
# -----------------------------------------------------------------------------
import copy
import json
from typing import Dict, Any, List, Optional

from utils.registry import register_tool
from utils.agent_utils import Result
from tools.planner_tools.design.subtasks_parser import parse_topic_to_subtasks
from utils.model_config import get_tool_model
from utils.constant import GROUNDING_STAGE_KEYS


def _subtask_design_signature(st: Dict[str, Any]) -> str:
    """Stable signature over design-defining fields only."""
    design_keys = (
        "id", "name", "description", "answer_type",
        "modalities", "sample_schema", "keywords",
    )
    payload = {k: st.get(k) for k in design_keys}
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(payload)


def _carry_forward_grounding_state_for_unchanged_subtasks(
    old_subtasks: List[Dict[str, Any]],
    new_subtasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Preserve grounding progress only for unchanged subtasks across design backtracking."""
    old_by_id: Dict[str, Dict[str, Any]] = {}
    for st in old_subtasks or []:
        if not isinstance(st, dict):
            continue
        sid = str(st.get("id") or "").strip()
        if sid:
            old_by_id[sid] = st

    merged: List[Dict[str, Any]] = []
    for st in new_subtasks or []:
        if not isinstance(st, dict):
            continue
        sid = str(st.get("id") or "").strip()
        old = old_by_id.get(sid)
        if not old:
            merged.append(st)
            continue
        if _subtask_design_signature(old) != _subtask_design_signature(st):
            merged.append(st)
            continue
        st_new = dict(st)
        for k in GROUNDING_STAGE_KEYS:
            if k in old:
                st_new[k] = copy.deepcopy(old[k])
        merged.append(st_new)
    return merged


def _as_list_str(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _init_subtask_for_grounding(st: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure subtask has fields needed by downstream Grounding Agent."""
    st = dict(st or {})
    st.setdefault("retrieval_result", [])
    st.setdefault("scored_status", "no")
    st.setdefault("scored_candidates", {})
    st.setdefault("dataset_preference", {})
    st.setdefault("transformability", {})
    st.setdefault("notes", "proposed; needs grounding")
    return st


def _summarize_working_subtasks_for_proposer(subtasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    summary: List[Dict[str, str]] = []
    for st in subtasks or []:
        summary.append({
            "id": str(st.get("id") or "").strip(),
            "name": str(st.get("name") or "").strip(),
            "description": str(st.get("description") or "").strip(),
        })
    return summary


@register_tool("proposer")
def proposer(
    decision_rationale: str,
    guidance: str = "",
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Generate candidate subtasks under the current benchmark scope that follow the expected benchmark schema and formatting constraints and store them in `proposed_subtasks`.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    - guidance: Optional steering for a refreshed proposal, e.g. what to emphasize, avoid, or change.
    """
    ctx = dict(context_variables or {})
    task_id = str(ctx.get("task_id") or "benchmark").strip()
    description = str(ctx.get("description") or "").strip()
    if not description:
        raise ValueError("[proposer] context must contain 'description' (user requirement).")
    target_size = int(ctx.get("target_size") or 3000)
    modalities = _as_list_str(ctx.get("modalities"))
    keywords = _as_list_str(ctx.get("keywords"))
    short_topic = str(ctx.get("short_topic") or "benchmark").strip()
    working_subtasks_summary = _summarize_working_subtasks_for_proposer(ctx.get("subtasks") or [])

    model_config_path = ctx.get("model_config_path")
    model = get_tool_model("parse_subtasks", model_config_path)

    draft = parse_topic_to_subtasks(
        task_id=task_id,
        description=description,
        target_size=target_size,
        model=model,
        modalities=modalities if modalities else None,
        keywords=keywords if keywords else None,
        short_topic=short_topic if short_topic != "benchmark" else None,
        proposal_guidance=guidance.strip() or None,
        existing_working_subtasks=working_subtasks_summary or None,
    )

    proposed = [
        _init_subtask_for_grounding(st)
        for st in (draft.get("subtasks") or [])
    ]

    ctx["proposed_subtasks"] = proposed
    msg = (
        f"proposer generated {len(proposed)} candidate subtasks (in proposed_subtasks): "
        f"{[st.get('name') or st.get('id') for st in proposed]}"
    )
    if guidance.strip():
        msg += " [used guidance]"

    return Result(
        value=msg,
        context_variables=ctx,
    )


@register_tool("process_proposed_subtasks")
def process_proposed_subtasks(
    decision_rationale: str,
    accepted_subtask_ids: List[str],
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """Resolve the current `proposed_subtasks` set by accepting a subset or rejecting it entirely.
    Accepted subtasks are merged (upserted) into the current working set. An empty accepted_subtask_ids list rejects the proposal and only clears `proposed_subtasks` without touching the working set.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    - accepted_subtask_ids: The IDs of the subtasks to accept. An empty list means reject this proposal and clear `proposed_subtasks`.
    """
    ctx = dict(context_variables or {})
    proposed = ctx.get("proposed_subtasks") or []
    if not proposed:
        raise ValueError(
            "[process_proposed_subtasks] No proposed_subtasks in context. Call proposer() first."
        )

    ids_set = {str(sid).strip() for sid in (accepted_subtask_ids or []) if str(sid).strip()}
    if not ids_set:
        ctx["proposed_subtasks"] = []
        return Result(
            value="rejected current proposed_subtasks and cleared the proposal set",
            context_variables=ctx,
        )

    selected = [
        _init_subtask_for_grounding(st)
        for st in proposed
        if (st.get("id") or "").strip() in ids_set
    ]
    if len(selected) != len(ids_set):
        found_ids = {(st.get("id") or "").strip() for st in selected}
        missing = ids_set - found_ids
        raise ValueError(
            f"[process_proposed_subtasks] Some IDs not in proposed_subtasks: {missing}"
        )

    working = list(ctx.get("subtasks") or [])
    working_index = {(st.get("id") or "").strip(): i for i, st in enumerate(working)}
    added, updated = [], []
    for st in selected:
        sid = (st.get("id") or "").strip()
        if sid in working_index:
            working[working_index[sid]] = st
            updated.append(st.get("name") or sid)
        else:
            working.append(st)
            added.append(st.get("name") or sid)

    ctx["subtasks"] = working
    ctx["proposed_subtasks"] = []

    parts = []
    if added:
        parts.append(f"added {len(added)}: {added}")
    if updated:
        parts.append(f"updated {len(updated)}: {updated}")
    msg = f"process_proposed_subtasks: {'; '.join(parts)}. Working set now has {len(working)} subtasks."

    return Result(
        value=msg,
        context_variables=ctx,
    )


def _do_design_stabilized(summary: str, context_variables: Optional[Dict[str, Any]]) -> Result:
    """Shared logic: set design_result and return. Used by case_resolved (tool name for core to exit)."""
    ctx = dict(context_variables or {})
    subtasks = ctx.get("subtasks", []) or []
    if not subtasks:
        raise ValueError(
            "Working set (subtasks) is empty. Add or accept at least one subtask before stabilizing."
        )
    payload = {
        "status": "stabilized",
        "summary": (summary or "").strip() or "Design stabilized; ready for grounding.",
        "subtask_count": len(subtasks),
    }
    ctx["proposed_subtasks"] = []
    ctx["design_result"] = payload
    return Result(
        value=payload.get("summary", ""),
        context_variables=ctx,
    )


@register_tool("case_resolved")
def case_resolved(
    decision_rationale: str,
    summary: str,
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """Finalize the current subtask set and mark design as ready for grounding.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    - summary: A brief summary of the current subtask set.
    """
    return _do_design_stabilized(summary, context_variables)
