from typing import Any, Dict, List, Optional, Tuple
import json
from utils.registry import register_tool
from utils.llm_caller import llm_call_json
from tools.shared.primitives import _clamp01, _safe_int, _subtasks_brief_task_only

_QUOTA_PROMPT_TASK_ONLY = r"""
You are a Planner Tool. Create an importance + quota plan for benchmark subtasks.

INPUTS:
- user_goal (short_topic): {short_topic}
- target_size (total benchmark items): {target_size}
- subtasks (JSON list):
{subtasks_json}

TASK:
For each subtask, output:
- importance: float in [0,1] indicating how critical this subtask is to evaluate the user_goal.
- quota: integer number of evaluation items assigned to this subtask.

RULES (STRICT):
1) Do NOT create/delete subtasks.
2) Quotas are non-negative integers.
3) Sum of quotas MUST equal target_size EXACTLY.
4) Importance must be in [0,1].
5) Prefer allocating more quota to subtasks that:
   - directly test core abilities implied by the user_goal,
   - cover broad/central capability,
   - are required for end-to-end coverage.
6)  quota_plan must include exactly one entry for each input subtask id.

OUTPUT (STRICT JSON ONLY):
{{
  "quota_plan": [
    {{"id":"...", "importance":0.0, "quota":0}}
  ],
  "quota_notes":"1-3 short sentences"
}}
No extra keys. No markdown. No extra text.
"""

@register_tool("llm_plan_quota")
def llm_plan_quota(
    model: str = "gpt-5-mini",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    subtasks: List[Dict[str, Any]] = None,
    short_topic: str = "",
    target_size: int = 0,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Don't pass any direct arguments.
    """
    brief = _subtasks_brief_task_only(subtasks)
    prompt = _QUOTA_PROMPT_TASK_ONLY.format(
        short_topic=short_topic,
        target_size=target_size,
        subtasks_json=json.dumps(brief, ensure_ascii=False, indent=2),
    )

    resp = llm_call_json(
        system_prompt="",
        user_prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not resp.get("ok"):
        raise RuntimeError(
            f"[llm_plan_quota] JSON parse failed: {resp.get('error')}\n"
            f"raw={resp.get('raw_text','')[:800]}"
        )

    data = resp.get("json") or {}
    plan = data.get("quota_plan", [])
    if not isinstance(plan, list):
        raise ValueError("[llm_plan_quota] quota_plan is not a list")

    plan_map: Dict[str, Dict[str, Any]] = {}
    for row in plan:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "").strip()
        if not sid:
            continue
        plan_map[sid] = {
            "importance": _clamp01(row.get("importance", 0.5)),
            "quota": max(0, _safe_int(row.get("quota", 0), 0)),
        }

    sids = []
    for st in subtasks:
        sid = str(st.get("id") or "").strip()
        if not sid:
            continue
        sids.append(sid)
        if sid not in plan_map:
            plan_map[sid] = {"importance": 0.5, "quota": 0}

    # enforce exact sum
    total = sum(plan_map[sid]["quota"] for sid in sids)
    diff = target_size - total

    if sids and diff != 0:
        sids_sorted = sorted(sids, key=lambda x: plan_map[x]["importance"], reverse=True)
        n = len(sids_sorted)

        if diff > 0:
            add_each, rem = divmod(diff, n)
            for sid in sids_sorted:
                plan_map[sid]["quota"] += add_each
            for sid in sids_sorted[:rem]:
                plan_map[sid]["quota"] += 1
        else:
            # need to remove -diff items without going below 0
            need = -diff
            # iterate in rounds but bounded: remove from lowest importance first
            sids_low = list(reversed(sids_sorted))
            idx = 0
            while need > 0:
                sid = sids_low[idx % n]
                if plan_map[sid]["quota"] > 0:
                    plan_map[sid]["quota"] -= 1
                    need -= 1
                idx += 1
                # optional: add a hard safety cap
                if idx > target_size + n + 1000:
                    break

    quota_notes = data.get("quota_notes", "")
    for st in subtasks:
        sid = str(st.get("id", "")).strip()
        if sid in plan_map:
            st["importance"] = plan_map[sid]["importance"]
            st["quota"] = plan_map[sid]["quota"]

    return subtasks, quota_notes,