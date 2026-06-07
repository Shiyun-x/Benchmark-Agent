# agents/allocation_agent.py
# -----------------------------------------------------------------------------
# Allocation Agent: allocates dataset samples to subtasks. Iterative loop:
# Allocating → Diagnoses → Adjustment. Can feed back to Design Agent when
# allocation constraints suggest design changes.
# -----------------------------------------------------------------------------
from typing import Dict, Any, List, Optional
import json

from utils.registry import register_agent
from utils.agent_utils import Agent, Result

from tools.planner_tools.allocation.allocation_tools import (
    attempt_allocation,
    diagnose_allocation,
    set_subtask_dataset_allocation_order,
    set_subtask_allocation_order,
    refine_quota,
)


def _format_allocation_agent_history(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "None yet."
    lines = []
    for entry in entries[-10:]:
        tool = str(entry.get("tool") or "").strip() or "unknown_tool"
        decision_rationale = str(entry.get("decision_rationale") or "").strip() or "Not recorded."
        args = str(entry.get("args") or "{}").strip()
        result = str(entry.get("result") or "").strip() or "No result recorded."
        lines.append(
            f"- tool: {tool}\n"
            f"  decision_rationale: {decision_rationale}\n"
            f"  args: {args}\n"
            f"  result: {result}"
        )
    return "\n".join(lines)


def case_resolved(decision_rationale: str, summary: str) -> Result:
    """Call when allocation succeeds. Does not overwrite last_allocation (set by attempt_allocation).
    - decision_rationale: A brief explanation of why allocation is considered complete at this point.
    - summary: A brief summary of the allocation outcome.
    """
    payload = {"summary": summary}
    return Result(
        value=json.dumps(payload, ensure_ascii=False, indent=2),
        context_variables={"allocation_result": payload},
    )


@register_agent("get_allocation_agent")
def get_allocation_agent(model: str, **kwargs) -> Agent:

    def instructions(ctx: Dict[str, Any]) -> str:
        allocation_config = ctx.get("allocation_config", {}) or {}
        subtasks = allocation_config.get("subtasks", {}) or {}
        datasets = allocation_config.get("datasets", {}) or {}
        last_diagnosis = ctx.get("last_diagnosis", {}) or {}
        allocation_agent_history = ctx.get("allocation_agent_history", []) or []
        allocation_agent_history_str = _format_allocation_agent_history(allocation_agent_history)
        allocation_result = ctx.get("allocation_result", {}) or {}

        return f"""
You are the **Allocation Agent** in a benchmark construction system.

Your responsibility is to **allocate dataset samples to subtasks** given the grounded
instantiations from the Grounding Agent. If allocation fails, diagnose the failure,
adjust configuration (subtask order, dataset order, quotas), and retry.

Your goal is to reach the best possible allocation. If repeated reasonable attempts
still fail, return the best effort result. In persistent failure cases, you may
signal feedback for the Design Agent (e.g. quota or coverage constraints).

--------------------------------------------------
Available tools:

- attempt_allocation()
- diagnose_allocation()
- case_resolved(decision_rationale, summary)

Configuration adjustment tools (use selectively):
- set_subtask_dataset_allocation_order(subtask_id, dataset_order)
- set_subtask_allocation_order(order)
- refine_quota(quota_for_subtask)

--------------------------------------------------
Operating loop:

1) Attempt allocation.
2) If allocation succeeds, stop (unfilled_subtasks=0) and call case_resolved(decision_rationale, summary).
3) If allocation fails:
   - Call diagnose_allocation to understand the failure and obtain recommendations.
   - Apply a small number of targeted changes (order or quota).
   - Retry allocation.

Repeat a limited number of times. Avoid oscillating settings.

Every tool call must include a brief `decision_rationale`. It should include:
- your short assessment of the current allocation state,
- the problem or opportunity you see,
- and why this tool is the best next action now.

Keep it concise. One to three sentences is enough.

--------------------------------------------------
## How to read history

The conversation history already contains your previous tool calls and the corresponding tool results.

In addition, `allocation_agent_history` below is a compact memory from earlier completed Allocation Agent runs. Use it to remember what you previously tried across retries or backtracking.

When a previous tool call includes `decision_rationale`, interpret it as a brief visible analysis of the current state and why that tool was the best next action at that moment.

When a tool message returns content, interpret it as the **observed result** of that action.

Use the history to understand what you already tried, what changed, and what happened after each action. Avoid repeating the same unsuccessful action unless the current state or feedback gives you a new reason.

--------------------------------------------------
## Current State (from ctx)

- subtasks: {json.dumps(subtasks, ensure_ascii=False)}
- datasets: {json.dumps(datasets, ensure_ascii=False)}
- diagnosis and recommendations: {json.dumps(last_diagnosis, ensure_ascii=False)}

## Current allocation result (if any)

{json.dumps(allocation_result, ensure_ascii=False) if allocation_result else "None yet."}

## Previous Allocation Agent Runs (if any)

This is your compact memory from earlier completed Allocation Agent runs. Use it to keep continuity across retries or backtracking.

{allocation_agent_history_str}
"""

    tools = [
        attempt_allocation,
        diagnose_allocation,
        set_subtask_dataset_allocation_order,
        set_subtask_allocation_order,
        refine_quota,
        case_resolved,
    ]

    return Agent(
        name="Allocation Agent",
        model=model,
        instructions=instructions,
        functions=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )
