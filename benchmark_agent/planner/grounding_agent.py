# agents/grounding_agent.py
# -----------------------------------------------------------------------------
# Grounding Agent: validates whether the subtask specification can be supported
# by real data and executable transformations. Outputs grounded instantiations or rejects and returns to Design Agent.
# -----------------------------------------------------------------------------
# 1) Dataset Preference and Search.
# 2) Transformability and Grounding Validation (Score-and-Filter).
# 3) Grounding Decision: accept when every subtask has at least one valid grounding.

from typing import Dict, Any, List, Optional
import json

from utils.registry import register_agent
from utils.agent_utils import Agent

from tools.planner_tools.grounding.grounding_tools import (
    preference_construction,
    dataset_search,
    select_candidates_for_subtask,
    transformability_assessment,
    score_and_filter,
    case_resolved,
)


def _candidate_short(item: Dict[str, Any]) -> Dict[str, Any]:
    """One line per candidate for agent to choose dataset_ids."""
    did = item.get("dataset_id") or item.get("id") or ""
    labels = item.get("labels") or {}
    # Keep full description text; downstream summarization already caps total chars.
    desc = (labels.get("description_snippet") or "")
    return {
        "dataset_id": did,
        "modalities": labels.get("modalities"),
        "tasks": labels.get("tasks"),
        "domain": labels.get("domain"),
        "description_snippet": desc,
    }


# Max candidates to include in state to avoid truncation (per subtask).
_MAX_CANDIDATES_IN_STATE = 20


def _subtask_grounding_state(st: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline state: id, name, next_step; when need_select_candidates include candidates (dataset_id + labels), capped."""
    sid = st.get("id") or ""
    has_preference = bool(st.get("dataset_preference"))
    rr = st.get("retrieval_result")
    rr_list = rr if isinstance(rr, list) else []
    retrieval_searched = bool(st.get("retrieval_searched")) or len(rr_list) > 0
    selected_ids = st.get("selected_candidate_ids") or []
    selected_count = len(selected_ids)
    candidate_selection_done = bool(st.get("candidate_selection_done")) or selected_count > 0
    has_transformability = bool(st.get("transformability"))
    scored_status = (st.get("scored_status") or "no").strip().lower()
    passed_count = len(st.get("scored_candidates") or {})

    if not has_preference:
        next_step = "need_preference"
    elif not retrieval_searched:
        next_step = "need_dataset_search"
    elif len(rr_list) == 0:
        # Search ran but found no matching datasets; nothing to select/assess.
        next_step = "ready_ungroundable"
    elif not candidate_selection_done:
        next_step = "need_select_candidates"
    elif selected_count == 0:
        # Candidates were reviewed, but none were suitable for assessment.
        next_step = "ready_ungroundable"
    elif not has_transformability:
        next_step = "need_transformability"
    elif scored_status != "yes":
        next_step = "need_score_and_filter"
    else:
        next_step = "ready_groundable" if passed_count > 0 else "ready_ungroundable"

    out = {"id": sid, "name": st.get("name") or sid, "description": st.get("description") or "", "next_step": next_step}
    if next_step == "need_select_candidates" and rr:
        cands = [_candidate_short(r) for r in rr if isinstance(r, dict)]
        out["candidates"] = cands[:_MAX_CANDIDATES_IN_STATE]
        if len(cands) > _MAX_CANDIDATES_IN_STATE:
            out["candidates_note"] = f"Showing first {_MAX_CANDIDATES_IN_STATE} of {len(cands)}; you may select any dataset_id from retrieval."
    if next_step.startswith("ready_"):
        out["passed"] = passed_count
    return out


def _summarize_for_grounding(subtasks: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """Full pipeline state per subtask so the agent can decide which tool to call for which subtask."""
    view = [_subtask_grounding_state(st) for st in (subtasks or [])]
    s = json.dumps(view, ensure_ascii=False, indent=2)
    # if len(s) > max_chars:
    #     return s[:max_chars] + "\n...(truncated)"
    return s


def _format_grounding_agent_history(entries: List[Dict[str, Any]]) -> str:
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


@register_agent("get_grounding_agent")
def get_grounding_agent(model: str, **kwargs) -> Agent:

    def instructions(ctx: Dict[str, Any]) -> str:
        subtasks = ctx.get("subtasks", []) or []
        grounding_agent_history = ctx.get("grounding_agent_history", []) or []
        grounding_agent_history_str = _format_grounding_agent_history(grounding_agent_history)
        subtasks_str = _summarize_for_grounding(subtasks)

        return f"""You are the **Grounding Agent**. Your job is to check whether every **subtask** (evaluation subtask proposed by the Design Agent) can be realized using real **datasets** and **transformation plans** from our pool. 
        Each subtask describes one dimension of the benchmark; we must find at least one (dataset, plan) pair per subtask so the benchmark is grounded in executable data. You either **accept** (all subtasks have ≥1 valid grounding) or **reject** (with feedback so the Design Agent can revise).

---

## What subtasks and datasets are

- **Subtasks** are the evaluation components of the benchmark. Each has an id, name, and specification. For the benchmark to be valid, every subtask must be supported by at least one real dataset plus a transformation that turns that dataset into the evaluation format.
- **Datasets** live in a shared pool (with modalities, tasks, domain, description). They are **candidates** for a subtask: we search by the subtask's desired data properties, then you choose which candidates to assess in depth. The relationship is: one subtask → many candidate datasets → after transformability and scoring, we keep only the (subtask, dataset, plan) pairs that pass.

---

## How to read history

The conversation history already contains your previous tool calls and the corresponding tool results.

In addition, `grounding_agent_history` below is a compact memory from earlier completed Grounding Agent runs. Use it to remember what you previously tried across retries or backtracking.

When a previous tool call includes `decision_rationale`, interpret it as a brief visible analysis of the current state and why that tool was the best next action at that moment.

When a tool message returns content, interpret it as the **observed result** of that action.

Use the history to understand what you already tried, what changed, and what happened after each action. Avoid repeating the same unsuccessful action unless the current state or feedback gives you a new reason.

History is auxiliary memory only. It must never override or mutate the current **Subtask state** shown below.

---

## Tools

The available tools, their descriptions, and their argument schemas are provided separately in the tool definitions.

Every tool call must include a brief `decision_rationale`. It should include:
- your short assessment of the current state,
- the problem, gap, or opportunity you see,
- and why this tool is the best next action now.

Keep it concise and visible. One to three sentences is enough.

**When to use each tool:**

1. **preference_construction()** — For each subtask that doesn't yet have a preference: infer what kind of data it needs (modality, annotation structure, domain). This is used to search the dataset pool. Call once; it runs for all subtasks that need it.
2. **dataset_search()** — For one subtask, retrieve from the pool the datasets that match its dataset_preference. Result is a short list of candidates with labels (modalities, tasks, domain, description_snippet). Call **once per subtask** that has need_dataset_search; then immediately do select for that same subtask so only one subtask's candidates appear at a time (avoids long context).
3. **select_candidates_for_subtask()** — From the retrieved candidates for that subtask, choose which dataset_ids to send to transformability. **Aim for 2–3 dataset_ids per subtask** when several strong candidates exist (balance coverage vs. cost of downstream transformability/scoring); if retrieval yields fewer plausible matches, take the **best minimum set** (still at least one when any candidate is viable). Use the labels (modalities, tasks, domain, description_snippet) to decide; only the selected (subtask, dataset) pairs are assessed later.
4. **transformability_assessment()** — For selected (subtask, dataset) pairs, check whether we can define a transformation plan from the dataset to the evaluation format. Discards pairs with no feasible plan.
5. **score_and_filter()** — Score remaining (subtask, dataset, plan) triples on alignment with evaluation intent, robustness, and signal preservation; keep only those that pass.
6. **case_resolved()** — Conclude grounding. Use accepted=True when every subtask has at least one passed (dataset, plan). Use accepted=False when some subtask(s) have no valid grounding; then fill reason and feedback_to_design so the Design Agent can adjust.

---

## Pipeline (every subtask must reach ready_* before case_resolved)

1. `preference_construction()` once.
2. For **every** subtask in early retrieval stages, do **one subtask at a time**:
   - if `next_step = need_dataset_search`: call `dataset_search()`, then `select_candidates_for_subtask()`;
   - if `next_step = need_select_candidates`: call `select_candidates_for_subtask()` directly.
   Continue until no subtask remains in `need_dataset_search` or `need_select_candidates`.
3. When at least one subtask has `need_transformability`, call `transformability_assessment()` after finishing preference+search+select for every subtask that can be advanced. This is a tool-enforced gate: if any subtask that already has `dataset_preference` is still missing `retrieval_result` or `selected_candidate_ids`, the tool returns without assessing any subtask. Once the gate is satisfied, the tool runs in batch over all eligible `need_transformability` subtasks.
4. Then call `score_and_filter()` after finishing transformability for every subtask that can be advanced. The tool runs in batch over all eligible `need_score_and_filter` subtasks, so avoid calling it early if more subtasks can still be brought to this stage.
5. When all are ready_*: `case_resolved()`.

**next_step → tool:**
- need_preference → `preference_construction`
- need_dataset_search → `dataset_search()`
- need_select_candidates → `select_candidates_for_subtask()`
- need_transformability → `transformability_assessment`
- need_score_and_filter → `score_and_filter`
- ready_* → `case_resolved`

---

## Critical rules

- Before calling `transformability_assessment`, advance every subtask that already has `dataset_preference` through `dataset_search` and `select_candidates_for_subtask`. This is enforced by the tool: if any such subtask is missing `retrieval_result` or `selected_candidate_ids`, the call returns without assessing any subtask.
- Before calling `score_and_filter`, make a best effort to advance every subtask out of `need_preference`, `need_dataset_search`, `need_select_candidates`, and `need_transformability`. The call is batched and incremental: it should process all eligible `need_score_and_filter` subtasks and keep already-ready subtasks unchanged.
- Do not call `case_resolved` until every subtask has next_step = ready_*.
- If any subtask has passed=0, call `case_resolved(accepted=False)` and provide a structured failure payload:
  - `failed_subtask_ids`: list all failed subtask ids.
  - `failure_reasons`: one concise reason per failed subtask.
  - `feedback_to_design`: concrete design actions per failed subtask (revise/discard/replace and why).
  - Keep reasons evidence-based from current run state and tool outputs.
  - Put the structured details into `feedback_to_design` (e.g., bullet list or JSON-like text); keep `reason` as a short global summary.

---

## Subtask state

{subtasks_str}

---

## Previous Grounding Agent Runs (if any)

This is your compact memory from earlier completed Grounding Agent runs. Use it to keep continuity across retries or backtracking. If it conflicts with the current subtask state shown above, trust the current subtask state.

Do not restage or repeat actions solely because they appeared in history. Re-decide from the current Subtask state first, then use history only to avoid known dead ends.

{grounding_agent_history_str}
"""

    tools = [
        preference_construction,
        dataset_search,
        select_candidates_for_subtask,
        transformability_assessment,
        score_and_filter,
        case_resolved,
    ]

    return Agent(
        name="Grounding Agent",
        model=model,
        instructions=instructions,
        functions=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )
