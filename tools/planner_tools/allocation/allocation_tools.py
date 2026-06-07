from calendar import c
import re
from typing import Dict, List, Any, Optional, Tuple
import math
from utils.llm_caller import llm_call_json
from regex import D, R
from utils.agent_utils import Result
from utils.registry import register_tool
import json


def _ensure_change_log(ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    safe_ctx = dict(ctx or {})
    safe_ctx.setdefault("change_log", [])
    return safe_ctx


DIAGNOSE_STAGE1_PROMPT = r"""
You are the Allocation Diagnoser (Stage 1: Failure Analysis).

Goal:
Analyze why the latest allocation attempt failed.
Focus ONLY on identifying the reasons and data characteristics behind the failure.
Do NOT propose any fixes or improvements in this stage.

Allocation logic (for reference):
- Allocation runs in two phases from scratch.
- Phase 1: greedy allocation with per-dataset cap (floor(quota * cap_ratio)).
    - cap_ratio defaults is 1/length(dataset_allocation_order).
- Phase 2: top-up allocation using remaining dataset capacity, ignoring the cap.
- Phase 2 represents a best-effort attempt and consumes all usable remaining capacity.

Context:
This stage is invoked only after allocation has failed.

==================== Inputs ====================

Last allocation result:
{last_allocation_json}

Subtask configuration:
{subtasks_json}

Dataset configuration:
{datasets_json}

Global subtask allocation order:
{subtask_allocation_order}

===============================================

What to analyze:
1) Which subtasks are unfilled and how much quota is unmet.
2) For each unfilled subtask, identify ONE dominant failure cause:
   - DATASET_EXHAUSTED    (key datasets ran out of capacity)
   - ORDER_STARVATION     (earlier subtasks consumed shared datasets)
   - INSUFFICIENT_SUPPLY  (overall supply seems insufficient)
3) Identify datasets that are exhausted or nearly exhausted and used by multiple subtasks.

Return ONLY valid JSON:

{{
  "unfilled_subtasks": ["A", "..."],
  "unmet_total": <int>,
  "per_subtask": {{
    "A": {{
      "quota_left": <int>,
      "cause": "DATASET_EXHAUSTED | ORDER_STARVATION | INSUFFICIENT_SUPPLY",
      "key_datasets": ["D1", "D2"]
    }}
  }},
  "hot_datasets": [
    {{
      "dataset_id": "D1",
      "remaining": <int>,
      "used_by": ["A", "B"]
    }}
  ]
}}

Rules:
- Use only information present in the inputs.
- Do NOT suggest changes or actions.
- Do NOT include opinions or value judgments.
- Keep everything concise and factual.
- Do NOT output anything outside the JSON.
"""

DIAGNOSE_STAGE2_PROMPT = r"""You are the Allocation Diagnoser (Stage 2: Improvement Recommendations).

Your task:
Based on the failure analysis, provide high-level, non-executable recommendations
about how the allocation configuration could be improved in the next attempt.

These recommendations should express judgment and direction,
NOT specific actions or parameter settings.

You MUST NOT propose executable actions or tool calls.
You MUST NOT specify exact parameter values.

==================== Inputs ====================

Failure analysis result:
{analysis_json}

Last allocation result:
{last_allocation_json}

Subtask configuration:
{subtasks_json}

Dataset configuration:
{datasets_json}

Global subtask allocation order:
{subtask_allocation_order}

===============================================

What to recommend:
- Which subtasks should be protected or deprioritized.
- Which datasets appear overused or should be yielded to others.
- Whether allocation order likely contributed to starvation.
- Whether quota expectations appear unrealistic given dataset supply.

Return ONLY valid JSON:

{{
  "recommendations": [
    {{
      "focus": "dataset_usage | subtask_order | quota_expectation",
      "scope": {{
        "subtask_id": "optional",
        "dataset_id": "optional"
      }},
      "opinion": "one concise sentence describing what could be improved",
      "rationale": "one concise sentence explaining why, grounded in the analysis"
    }}
  ],
  "note": "one short summary sentence"
}}

Rules:
- Provide 2 to 5 recommendations.
- Each recommendation must be grounded in the analysis and inputs.
- Avoid speculative or redundant statements.
- Do NOT imply specific parameter values or actions.
- Keep language concise and neutral.
- Do NOT output anything outside the JSON.
"""

@register_tool("attempt_allocation")
def attempt_allocation(
    decision_rationale: str,
    context_variables: Optional[dict] = None,
) -> Result:
    """
    Deterministic allocation attempt with two phases:
    - decision_rationale: A brief explanation of why attempt_allocation is called at this moment.

    Phase 1 (hard cap):
      - Outer: subtask_allocation_order
      - Inner: subtask.dataset_allocation_order
      - Respect per-dataset cap = floor(quota * cap_ratio)

    Phase 2 (top-up):
      - For unmet subtasks only
      - Same orders
      - Ignore per-dataset cap, only respect dataset remaining and quota_left
      - Does NOT steal allocations from other subtasks
    """
    ctx = _ensure_change_log(context_variables)
    allocation_config = ctx.get("allocation_config") or {}
    subtasks = allocation_config.get("subtasks", {}) or {}
    datasets = allocation_config.get("datasets", {}) or {}
    subtask_allocation_order = allocation_config.get("subtask_allocation_order", []) or []
    # ---- initialize dataset state ----
    dataset_state = {}
    for did, d in datasets.items():
        cap = int(d.get("num_samples", 0) or 0)
        dataset_state[did] = {
            "capacity": cap,
            "used": 0,
            "remaining": cap,
            "used_by": {}
        }

    # ---- initialize subtask state ----
    subtask_state = {}
    for sid, st in subtasks.items():
        quota = int(st.get("quota", 0) or 0)
        subtask_state[sid] = {
            "quota": quota,
            "allocated": 0,
            "quota_left": quota,
            "cap_ratio": float(st.get("cap_ratio", 1.0)),
            "alloc": {},
            # bookkeeping (optional but useful)
            "allocated_phase1": 0,
            "allocated_topup": 0,
        }

    # -------------------------
    # Phase 1: hard-cap greedy
    # -------------------------
    for sid in subtask_allocation_order:
        if sid not in subtasks or sid not in subtask_state:
            continue

        st_conf = subtasks[sid]
        st_state = subtask_state[sid]
        if st_state["quota_left"] <= 0:
            continue

        quota = st_state["quota"]
        cap_ratio = st_state["cap_ratio"]
        per_dataset_cap = math.floor(quota * cap_ratio)

        dataset_order = st_conf.get("dataset_allocation_order", []) or []

        for did in dataset_order:
            if st_state["quota_left"] <= 0:
                break
            if did not in dataset_state:
                continue

            ds_state = dataset_state[did]
            if ds_state["remaining"] <= 0:
                continue

            already_used = st_state["alloc"].get(did, 0)
            cap_left = per_dataset_cap - already_used
            if cap_left <= 0:
                continue

            alloc = min(st_state["quota_left"], ds_state["remaining"], cap_left)
            if alloc <= 0:
                continue

            # apply
            st_state["alloc"][did] = already_used + alloc
            st_state["allocated"] += alloc
            st_state["allocated_phase1"] += alloc
            st_state["quota_left"] -= alloc

            ds_state["used"] += alloc
            ds_state["remaining"] -= alloc
            ds_state["used_by"][sid] = ds_state["used_by"].get(sid, 0) + alloc

    # -------------------------
    # Phase 2: top-up (salvage)
    # -------------------------
    topup_total = 0
    for sid in subtask_allocation_order:
        if sid not in subtasks or sid not in subtask_state:
            continue

        st_conf = subtasks[sid]
        st_state = subtask_state[sid]
        if st_state["quota_left"] <= 0:
            continue

        dataset_order = st_conf.get("dataset_allocation_order", []) or []

        for did in dataset_order:
            if st_state["quota_left"] <= 0:
                break
            if did not in dataset_state:
                continue

            ds_state = dataset_state[did]
            if ds_state["remaining"] <= 0:
                continue

            alloc = min(st_state["quota_left"], ds_state["remaining"])
            if alloc <= 0:
                continue

            # apply (no cap check here)
            already_used = st_state["alloc"].get(did, 0)
            st_state["alloc"][did] = already_used + alloc
            st_state["allocated"] += alloc
            st_state["allocated_topup"] += alloc
            st_state["quota_left"] -= alloc

            ds_state["used"] += alloc
            ds_state["remaining"] -= alloc
            ds_state["used_by"][sid] = ds_state["used_by"].get(sid, 0) + alloc

            topup_total += alloc

    # ---- build change log ----
    total_quota = sum(st["quota"] for st in subtask_state.values())
    total_allocated = sum(st["allocated"] for st in subtask_state.values())
    unfilled = [sid for sid, st in subtask_state.items() if st["quota_left"] > 0]

    change_log = f"Attempted allocation: total_quota={total_quota}, total_allocated={total_allocated}, topup_allocated={topup_total}, unfilled_subtasks={len(unfilled)}"
    ctx["change_log"].append(change_log)
    # ---- build output ----
    return Result(
        value=change_log,
        context_variables={
            "change_log": ctx["change_log"],
            "last_allocation": {
                "subtasks": subtask_state,
                "datasets": dataset_state,
                "ok": len(unfilled) == 0,
                "unmet_total": sum(subtask_state[sid]["quota_left"] for sid in unfilled),
            }
        }
    )

@register_tool("diagnose_allocation")
def diagnose_allocation(
    decision_rationale: str,
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Two-stage allocation diagnosis:
    - Stage 1: analysis (what happened and why)
    - Stage 2: suggestions (what the planner could try)
    - decision_rationale: A brief explanation of why diagnose_allocation is called at this moment.
    """

    ctx = _ensure_change_log(context_variables)
    allocation_config = ctx.get("allocation_config", {})
    last_allocation = ctx.get("last_allocation", {})
    subtask_order = allocation_config.get("subtask_allocation_order", [])
    subtasks_conf = allocation_config.get("subtasks", {})
    datasets_conf = allocation_config.get("datasets", {})

    # ---------- Stage 1: Analysis ----------
    stage1_input = {
        "last_allocation": last_allocation,
        "subtask_allocation_order": subtask_order,
        "subtasks": subtasks_conf,
        "datasets": datasets_conf,
    }

    prompt = DIAGNOSE_STAGE1_PROMPT.format(
        last_allocation_json=json.dumps(stage1_input["last_allocation"]),
        subtask_allocation_order=json.dumps(stage1_input["subtask_allocation_order"]),
        subtasks_json=json.dumps(stage1_input["subtasks"]),
        datasets_json=json.dumps(stage1_input["datasets"])   
    )


    stage1_res = llm_call_json(
        system_prompt="",
        user_prompt=prompt,
    )

    analysis = stage1_res.get("json") if stage1_res.get("ok") else {
        "ok": bool(last_allocation.get("ok", False)),
        "unfilled_subtasks": [],
        "unmet_total": int(last_allocation.get("unmet_total", 0) or 0),
        "hot_datasets": [],
        "per_subtask": {},
        "global_observations": [
            f"Stage1 error: {stage1_res.get('error', 'unknown')}"
        ],
    }

    # ---------- Stage 2: Suggestions ----------
    prompt = DIAGNOSE_STAGE2_PROMPT.format(
        analysis_json=json.dumps(analysis),
        last_allocation_json=json.dumps(stage1_input["last_allocation"]),
        subtask_allocation_order=json.dumps(stage1_input["subtask_allocation_order"]),
        subtasks_json=json.dumps(stage1_input["subtasks"]),
        datasets_json=json.dumps(stage1_input["datasets"])   
    )
    stage2_res = llm_call_json(
        system_prompt="",
        user_prompt=prompt,
    )

    suggestions = stage2_res.get("json") if stage2_res.get("ok") else {
        "suggestions": [],
        "note": f"Stage2 error: {stage2_res.get('error', 'unknown')}",
    }

    # ---------- persist ----------
    ctx["last_diagnosis"] = {
        "analysis": analysis,
        "suggestions": suggestions,
    }

    ctx["change_log"].append(f"Diagnosis completed, suggestions: {len(suggestions.get('recommendations', []))} items")

    return Result(
        value=ctx["change_log"][-1],
        context_variables=ctx,
    )

@register_tool("set_subtask_dataset_allocation_order")
def set_subtask_dataset_allocation_order(
    decision_rationale: str,
    subtask_id: str,
    dataset_order: List[str],
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Set dataset allocation order for a specific subtask.
    Behavior: Directly updates `dataset_allocation_order` for the subtask.
    Input:
        - decision_rationale: A brief explanation of why this reordering is needed.
        - subtask_id: ID of the subtask to
        - dataset_order: list of dataset IDs in desired order
    """

    ctx = _ensure_change_log(context_variables)
    allocation_config = ctx.get("allocation_config", {})
    subtasks = allocation_config.get("subtasks", {})

    if subtask_id not in subtasks:
        return Result(
            value=f"error: subtask {subtask_id} not found",
            context_variables=ctx,
        )

    st = subtasks[subtask_id]
    old_order = list(st.get("dataset_allocation_order", []))

    # keep only valid datasets that exist in original order
    desired = [d for d in dataset_order if d in old_order]

    # append remaining datasets in original relative order
    remaining = [d for d in old_order if d not in desired]

    new_order = desired + remaining

    st["dataset_allocation_order"] = new_order

    ctx["change_log"].append(f"set_subtask_dataset_allocation_order: subtask_id={subtask_id}, old_order={old_order}, new_order={new_order}")

    return Result(
        value=ctx["change_log"][-1],
        context_variables=ctx,
    )

@register_tool("refine_quota")
def refine_quota(
    decision_rationale: str,
    quota_for_subtask: Dict[str, int],
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Directly set quota for a specific subtask.
    Input:
        - decision_rationale: A brief explanation of why quotas are being refined.
        - quota_for_subtask: dict of {subtask_id: new_quota}
        - sum of new_quota should be = sum of original quotas
    """

    ctx = _ensure_change_log(context_variables)
    allocation_config = ctx.get("allocation_config", {})
    subtasks = allocation_config.get("subtasks", {})

    total_old = sum(st.get("quota", 0) for st in subtasks.values())
    # Count unchanged subtasks' quotas too; agent may pass only the subtasks it wants to change.
    total_new = sum(
        quota_for_subtask.get(sid, st.get("quota", 0))
        for sid, st in subtasks.items()
    )
    if total_old != total_new:
        return Result(
            value=f"error: total quota mismatch, old_total={total_old}, new_total={total_new}",
            context_variables=ctx,
        )
    old_quota_list = {}
    for sid,st in subtasks.items():
        if sid in quota_for_subtask:
            old_quota_list[sid] = st.get("quota", 0)
            st["quota"] = quota_for_subtask[sid]

    ctx["allocation_config"]["subtasks"] = subtasks
    ctx["change_log"].append(f"refine_quota: changes={quota_for_subtask}, old_quotas={old_quota_list}")

    return Result(
        value=ctx["change_log"][-1],
        context_variables=ctx,
    )

@register_tool("set_subtask_allocation_order")
def set_subtask_allocation_order(
    decision_rationale: str,
    order: List[str],
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Set global subtask allocation order.
    Input:
        - decision_rationale: A brief explanation of why the subtask order is being changed.
        - order: list of subtask IDs in desired order
    """

    ctx = _ensure_change_log(context_variables)
    allocation_config = ctx.get("allocation_config", {})    
    subtasks = allocation_config.get("subtasks", {})

    # basic validation
    for sid in order:
        if sid not in subtasks:
            return Result(
                value=f"error: subtask {sid} not found",
                context_variables=ctx,
            )

    old = allocation_config.get("subtask_allocation_order", [])
    # Append any subtasks not listed in order so none are silently dropped from allocation.
    remaining = [sid for sid in old if sid not in set(order)]
    new_order = list(order) + remaining
    allocation_config["subtask_allocation_order"] = new_order
    ctx["allocation_config"] = allocation_config
    ctx["change_log"].append(f"set_subtask_allocation_order: old_order={old}, new_order={new_order}")

    return Result(
        value=ctx["change_log"][-1],
        context_variables=ctx,
    )


def _check_allocation_gap(
    metadata: Any,
    context_variables: Dict[str, Any],
    unmet_ratio_threshold: float = 0.0,
) -> Tuple[bool, str]:
    """Check if allocation gap is acceptable. Returns (accepted, feedback_message)."""
    allocation = context_variables.get("allocation", {})
    ok = allocation.get("ok", False)
    unmet_total = int(allocation.get("unmet_total", 0) or 0)
    target_size = getattr(metadata, "target_size", 0) or context_variables.get("target_size", 0)
    subtasks_alloc = allocation.get("subtasks", {}) or {}

    if ok:
        return True, ""
    if unmet_ratio_threshold > 0 and target_size > 0:
        if unmet_total <= target_size * unmet_ratio_threshold:
            return True, ""

    gaps = []
    subtask_ids_with_gap = []
    for sid, st in subtasks_alloc.items():
        quota_left = int(st.get("quota_left", 0) or 0)
        if quota_left > 0:
            quota = int(st.get("quota", 0) or 0)
            gaps.append(f"{sid}: quota={quota}, gap(quota_left)={quota_left}")
            subtask_ids_with_gap.append(sid)
    feedback = (
        f"Allocation gap too large: unmet_total={unmet_total} (target_size={target_size}). "
        f"Subtasks with gap (revise ONLY these; leave other subtasks unchanged): {'; '.join(gaps)}. "
        f"Subtask IDs to revise: {subtask_ids_with_gap}. "
        "Revise only the above subtask(s)—e.g. reduce scope, relax constraints, or lower quota expectation—so allocation can be satisfied. Do not modify other subtasks."
    )
    return False, feedback


def _build_allocation_orders(ctx: dict) -> dict:
    """Build allocation_config from subtasks and id2card in context.

    Returns a dict with subtask_allocation_order, per-subtask dataset_allocation_order,
    and dataset sample counts. Mutates nothing; caller assigns the result.
    """
    subtasks = ctx.get("subtasks", []) or []
    allocation_config: Dict[str, Any] = {}

    allocation_config["subtask_allocation_order"] = [
        st["id"]
        for st in sorted(subtasks, key=lambda st: float(st.get("importance", 0.0)), reverse=True)
        if "id" in st
    ]

    allocation_config["subtasks"] = {}
    for st in subtasks:
        scored = st.get("scored_candidates") or {}
        n_candidates = len(scored)
        cap_ratio = round(min(1, 1 / n_candidates), 2) if n_candidates > 0 else 1.0
        st_config = {
            "quota": st.get("quota", 0),
            "cap_ratio": cap_ratio,
            "dataset_allocation_order": [
                dataset_id
                for dataset_id, _ in sorted(
                    scored.items(),
                    key=lambda item: float(item[1].get("scores", {}).get("overall", 0.0)),
                    reverse=True,
                )
            ],
        }
        st_id = st.get("id")
        if st_id:
            allocation_config["subtasks"][st_id] = st_config

    allocation_config["datasets"] = {}
    seen_datasets: set = set()
    for st in subtasks:
        for dataset_id in (st.get("scored_candidates") or {}):
            seen_datasets.add(dataset_id)
    for dataset_id in seen_datasets:
        allocation_config["datasets"][dataset_id] = {
            "num_samples": int(ctx.get("id2card", {}).get(dataset_id, {}).get("size_samples", 0)),
        }

    return allocation_config
