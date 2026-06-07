# tools/grounding_tools.py
# -----------------------------------------------------------------------------
# Tools for the Grounding Agent: Preference Construction, Dataset Search,
# Transformability Assessment, Score-and-Filter, Grounding Accept/Reject.
# -----------------------------------------------------------------------------
from typing import Dict, Any, List, Optional
import json
import os
import re

from utils.registry import register_tool
from utils.agent_utils import Result
from tools.planner_tools.grounding.dataset_preference_tools import plan_dataset_preference
from tools.planner_tools.grounding.dataset_retriever_tools import retrieve_candidates
from tools.planner_tools.grounding.dataset_scorer_tools import transformability_check, scoring
from tools.planner_tools.design.subtask_refinement_tools import _upsert_subtask


# ---------- 1) Dataset Preference and Search ----------

def _safe_id(text: str, fallback: str = "unknown") -> str:
    """Sanitize an identifier (e.g., subtask_id/topic_id) for filenames."""
    t = (text or "").strip() or fallback
    t = re.sub(r"[\/\\:\*?\"<>\|\s]", "_", t)
    return re.sub(r"[^\w._-]", "_", t) or fallback


def _labels_from_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """Build a short label dict for the agent to decide which datasets to select."""
    if not card or not isinstance(card, dict):
        return {}
    # Keep full description text to avoid losing important details for selection.
    desc = (card.get("description") or card.get("card_text") or "")
    return {
        "modalities": card.get("modalities") or [],
        "tasks": card.get("tasks") or [],
        "domain": card.get("domain") or "",
        "description_snippet": desc.strip(),
    }


def _safe_subtask_name(st: Dict[str, Any], fallback_id: str = "") -> str:
    """Sanitize subtask name for use in cache filename."""
    name = (st.get("name") or st.get("id") or fallback_id) or "unknown"
    name = re.sub(r"[\/\\:\*?\"<>\|\s]", "_", name)
    return re.sub(r"[^\w._-]", "_", name)


def _use_grounding_stage_cache(ctx: Dict[str, Any]) -> bool:
    """
    Whether to read per-stage grounding cache files (preference/transformability/scoring).
    """
    return bool(ctx.get("_use_grounding_stage_cache", True))


@register_tool("preference_construction")
def preference_construction(
    decision_rationale: str,
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Generate dataset preferences for **all subtasks that do not yet have** dataset_preference.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    """
    ctx = dict(context_variables or {})
    full_subtasks = list(ctx.get("subtasks", []) or [])
    stage = "preference_construction"
    updated = 0
    use_stage_cache = _use_grounding_stage_cache(ctx)
    for st in full_subtasks:
        if st.get("dataset_preference"):
            continue
        sid = (st.get("id") or "").strip()
        subtask_name_safe = _safe_subtask_name(st, sid)
        cache_path = _grounding_cache_path(ctx, stage, sid, subtask_name_safe)
        legacy_cache_path = _legacy_grounding_cache_path(ctx, stage, subtask_name_safe)
        if use_stage_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    ds_pref = json.load(f)
                st["dataset_preference"] = ds_pref
                updated += 1
                continue
            except (json.JSONDecodeError, IOError):
                os.remove(cache_path)
        elif use_stage_cache and os.path.exists(legacy_cache_path):
            # Backward-compat: load old filename (no subtask_id prefix), and migrate to new name.
            try:
                with open(legacy_cache_path, "r", encoding="utf-8") as f:
                    ds_pref = json.load(f)
                st["dataset_preference"] = ds_pref
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(ds_pref, f, ensure_ascii=False, indent=2)
                updated += 1
                continue
            except (json.JSONDecodeError, IOError):
                try:
                    os.remove(legacy_cache_path)
                except OSError:
                    pass
        ds_pref = plan_dataset_preference(subtask=st)
        st["dataset_preference"] = ds_pref
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(ds_pref, f, ensure_ascii=False, indent=2)
        updated += 1
    ctx["subtasks"] = full_subtasks

    return Result(
        value=f"Dataset preferences set for {updated} subtask(s).",
        context_variables=ctx,
    )


@register_tool("dataset_search")
def dataset_search(
    decision_rationale: str,
    subtask_id: str,
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Search the dataset pool for **one subtask** that has dataset_preference but no retrieval yet.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    - subtask_id: the ID of the subtask to search for datasets
    """
    ctx = dict(context_variables or {})
    dataset_cards = ctx.get("dataset_cards", []) or []
    id2card = ctx.get("id2card") or {}
    full_subtasks = list(ctx.get("subtasks", []) or [])
    subtask = next((s for s in full_subtasks if (s.get("id") or "").strip() == (subtask_id or "").strip()), None)
    if subtask is None:
        raise ValueError(f"[dataset_search] subtask_id not found: {subtask_id}")
    if not subtask.get("dataset_preference"):
        raise ValueError(f"[dataset_search] subtask {subtask_id} has no dataset_preference; run preference_construction first.")
    existing_retrieval = subtask.get("retrieval_result")
    if subtask.get("retrieval_searched") or (isinstance(existing_retrieval, list) and len(existing_retrieval) > 0):
        return Result(
            value=f"Subtask {subtask_id} already has retrieval_result; use select_candidates_for_subtask.",
            context_variables=ctx,
        )

    sid = (subtask_id or "").strip()
    subtask_name_safe = _safe_subtask_name(subtask, sid)
    stage = "dataset_search"
    cache_path = _grounding_cache_path(ctx, stage, sid, subtask_name_safe)
    legacy_cache_path = _legacy_grounding_cache_path(ctx, stage, subtask_name_safe)
    use_stage_cache = _use_grounding_stage_cache(ctx)

    if use_stage_cache and (os.path.exists(cache_path) or os.path.exists(legacy_cache_path)):
        load_path = cache_path if os.path.exists(cache_path) else legacy_cache_path
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and isinstance(cached.get("retrieval_result"), list):
                retrieval_result = cached.get("retrieval_result") or []
            elif isinstance(cached, list):
                # Backward-compat with possible raw-list dumps.
                retrieval_result = cached
            else:
                retrieval_result = []
            subtask["retrieval_result"] = retrieval_result
            subtask["retrieval_searched"] = True
            subtask["notes"] = (
                f"loaded {len(retrieval_result)} candidates from dataset_search stage cache; "
                "select via select_candidates_for_subtask"
            )
            ctx["subtasks"] = full_subtasks
            if load_path == legacy_cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "subtask_id": sid,
                            "subtask_name": subtask.get("name") or "",
                            "retrieval_result": retrieval_result,
                            "n_retrieved": len(retrieval_result),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            return Result(
                value=f"Dataset search cache hit for {subtask_id}: {len(retrieval_result)} candidates loaded.",
                context_variables=ctx,
            )
        except (json.JSONDecodeError, IOError):
            try:
                os.remove(load_path)
            except OSError:
                pass

    ds_pref = subtask.get("dataset_preference") or {}
    raw_result = retrieve_candidates(subtask, dataset_cards, ds_pref)
    retrieval_result = []
    for r in raw_result:
        did = r.get("dataset_id") or r.get("id") or ""
        item = dict(r)
        item["labels"] = _labels_from_card(id2card.get(did, {}))
        retrieval_result.append(item)
    subtask["retrieval_result"] = retrieval_result
    subtask["retrieval_searched"] = True
    subtask["notes"] = f"retrieved {len(retrieval_result)} candidates (with labels); select via select_candidates_for_subtask"
    ctx["subtasks"] = full_subtasks
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "subtask_id": sid,
                "subtask_name": subtask.get("name") or "",
                "retrieval_result": retrieval_result,
                "n_retrieved": len(retrieval_result),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return Result(
        value=f"Dataset search for {subtask_id}: {len(retrieval_result)} candidates with labels. Call select_candidates_for_subtask({subtask_id!r}, dataset_ids) next.",
        context_variables=ctx,
    )


@register_tool("select_candidates_for_subtask")
def select_candidates_for_subtask(
    decision_rationale: str,
    subtask_id: str,
    dataset_ids: List[str],
    reason: str = "",
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Select which retrieved datasets for this subtask (only for the subtask with subtask_id) should go to transformability assessment.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    - subtask_id: the ID of the subtask to select datasets for
    - dataset_ids: the list of dataset IDs to select. Prefer 2–3 IDs when multiple strong retrieval matches exist; use fewer only when quality candidates are scarce.
    - reason: the reason for selecting the datasets for the subtask
    """
    ctx = dict(context_variables or {})
    subtasks = ctx.get("subtasks", []) or []
    subtask = next((s for s in subtasks if (s.get("id") or "").strip() == (subtask_id or "").strip()), None)
    if subtask is None:
        raise ValueError(f"[select_candidates_for_subtask] subtask_id not found: {subtask_id}")

    sid = (subtask_id or "").strip()
    subtask_name_safe = _safe_subtask_name(subtask, sid)
    stage = "select_candidates"
    cache_path = _grounding_cache_path(ctx, stage, sid, subtask_name_safe)
    legacy_cache_path = _legacy_grounding_cache_path(ctx, stage, subtask_name_safe)
    use_stage_cache = _use_grounding_stage_cache(ctx)

    if use_stage_cache and (os.path.exists(cache_path) or os.path.exists(legacy_cache_path)):
        load_path = cache_path if os.path.exists(cache_path) else legacy_cache_path
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_selected = cached.get("selected_candidate_ids") if isinstance(cached, dict) else None
            if isinstance(cached_selected, list):
                retrieval = subtask.get("retrieval_result") or []
                available_ids = {
                    str(r.get("dataset_id") or r.get("id") or "").strip()
                    for r in retrieval
                    if isinstance(r, dict)
                }
                requested = [str(did).strip() for did in cached_selected if str(did).strip()]
                requested = [did for did in requested if did in available_ids]

                full_subtasks = list(ctx.get("subtasks", []) or [])
                for s in full_subtasks:
                    if (s.get("id") or "").strip() == sid:
                        s["selected_candidate_ids"] = requested
                        s["candidate_selection_done"] = True
                        break
                ctx["subtasks"] = full_subtasks

                if load_path == legacy_cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cached, f, ensure_ascii=False, indent=2)
                return Result(
                    value=f"Selection cache hit for {subtask_id}: {len(requested)} dataset(s) loaded.",
                    context_variables=ctx,
                )
        except (json.JSONDecodeError, IOError):
            try:
                os.remove(load_path)
            except OSError:
                pass

    retrieval = subtask.get("retrieval_result") or []
    available_ids = {str(r.get("dataset_id") or r.get("id") or "").strip() for r in retrieval if isinstance(r, dict)}
    requested = [str(did).strip() for did in (dataset_ids or []) if str(did).strip()]
    invalid = set(requested) - available_ids
    if invalid:
        raise ValueError(
            f"[select_candidates_for_subtask] IDs not in retrieval_result: {invalid}. "
            f"Available: {available_ids}"
        )

    full_subtasks = list(ctx.get("subtasks", []) or [])
    for s in full_subtasks:
        if (s.get("id") or "").strip() == (subtask_id or "").strip():
            s["selected_candidate_ids"] = requested
            s["candidate_selection_done"] = True
            break
    ctx["subtasks"] = full_subtasks

    # Cache selection decision for easier inspection / recovery.
    selected_set = set(requested)
    not_selected = sorted([did for did in available_ids if did and did not in selected_set])
    payload = {
        "subtask_id": sid,
        "subtask_name": subtask.get("name") or "",
        "selected_candidate_ids": requested,
        "not_selected_candidate_ids": not_selected,
        "reason": (reason or "").strip(),
        "n_retrieved": len(retrieval) if isinstance(retrieval, list) else 0,
        "n_selected": len(requested),
        "n_not_selected": len(not_selected),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # If legacy exists, keep it (do not delete) but ensure new cache is written.

    return Result(
        value=f"Selected {len(requested)} dataset(s) for {subtask_id} to send to transformability.",
        context_variables=ctx,
    )


@register_tool("add_candidates_for_subtask")
def add_candidates_for_subtask(
    decision_rationale: str,
    subtask_id: str,
    dataset_ids: List[str],
    reason: str = "",
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Add extra dataset(s) from the pool to a subtask's candidates for transformability assessment.
    Use this when the user wants to manually add datasets (e.g. in interactive mode) that were
    not in the retrieval_result. Each dataset_id must exist in id2card (dataset pool).
    Appends to retrieval_result and selected_candidate_ids.
    """
    ctx = dict(context_variables or {})
    id2card = ctx.get("id2card") or {}
    full_subtasks = list(ctx.get("subtasks", []) or [])
    subtask = next((s for s in full_subtasks if (s.get("id") or "").strip() == (subtask_id or "").strip()), None)
    if subtask is None:
        raise ValueError(f"[add_candidates_for_subtask] subtask_id not found: {subtask_id}")

    requested = [str(did).strip() for did in (dataset_ids or []) if str(did).strip()]
    if not requested:
        return Result(value="No dataset_ids provided.", context_variables=ctx)

    missing = [did for did in requested if did not in id2card]
    if missing:
        raise ValueError(
            f"[add_candidates_for_subtask] dataset_id(s) not in pool (id2card): {missing}. "
            f"Available pool has {len(id2card)} datasets."
        )

    retrieval = list(subtask.get("retrieval_result") or [])
    existing_ids = {str(r.get("dataset_id") or r.get("id") or "").strip() for r in retrieval if isinstance(r, dict)}
    selected = list(subtask.get("selected_candidate_ids") or [])

    added = 0
    for did in requested:
        if did in existing_ids:
            if did not in selected:
                selected.append(did)
                added += 1
            continue
        card = id2card.get(did, {})
        labels = _labels_from_card(card)
        retrieval.append({
            "dataset_id": did,
            "id": did,
            "labels": labels,
        })
        selected.append(did)
        existing_ids.add(did)
        added += 1

    for s in full_subtasks:
        if (s.get("id") or "").strip() == (subtask_id or "").strip():
            s["retrieval_result"] = retrieval
            s["retrieval_searched"] = True
            s["selected_candidate_ids"] = selected
            s["candidate_selection_done"] = True
            break
    ctx["subtasks"] = full_subtasks

    # Cache manual additions for easier inspection / recovery.
    sid = (subtask_id or "").strip()
    subtask_name_safe = _safe_subtask_name(subtask, sid)
    stage = "add_candidates"
    cache_path = _grounding_cache_path(ctx, stage, sid, subtask_name_safe)
    final_selected_set = set(str(d).strip() for d in (selected or []) if str(d).strip())
    # "Not selected" is relative to the current retrieval_result (which may include manually added items).
    retrieval_ids = {str(r.get("dataset_id") or r.get("id") or "").strip() for r in retrieval if isinstance(r, dict)}
    not_selected = sorted([did for did in retrieval_ids if did and did not in final_selected_set])
    payload = {
        "subtask_id": sid,
        "subtask_name": subtask.get("name") or "",
        "added_dataset_ids": requested,
        "final_selected_candidate_ids": list(final_selected_set),
        "not_selected_candidate_ids": not_selected,
        "reason": (reason or "").strip(),
        "n_retrieved_total": len(retrieval) if isinstance(retrieval, list) else 0,
        "n_added": len(requested),
        "n_selected_final": len(final_selected_set),
        "n_not_selected": len(not_selected),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return Result(
        value=f"Added {added} dataset(s) to {subtask_id}; total selected for transformability: {len(selected)}.",
        context_variables=ctx,
    )


# ---------- 2) Transformability and Score-and-Filter ----------

def _grounding_cache_path(ctx: Dict, stage: str, subtask_id: str, subtask_name_safe: str) -> str:
    cache_root = ctx.get("cache_root") or ctx.get("cache_path") or "cache"
    stage_safe = _safe_id(stage, "stage")
    sid_safe = _safe_id(subtask_id, "unknown_subtask")
    return os.path.join(cache_root, "score_cache", stage_safe, f"{sid_safe}_{stage_safe}_{subtask_name_safe}.json")


def _legacy_grounding_cache_path(ctx: Dict, stage: str, subtask_name_safe: str) -> str:
    """Previous cache filename format (no subtask_id prefix). Kept for backward compat."""
    cache_root = ctx.get("cache_root") or ctx.get("cache_path") or "cache"
    stage_safe = _safe_id(stage, "stage")
    return os.path.join(cache_root, "score_cache", stage_safe, f"{stage_safe}_{subtask_name_safe}.json")


def _filter_retrieval_by_selected(subtask: Dict[str, Any]) -> tuple:
    """If subtask has selected_candidate_ids, return filtered retrieval_result and original for restore."""
    retrieval = subtask.get("retrieval_result") or []
    selected = subtask.get("selected_candidate_ids")
    if not selected:
        return retrieval, None
    selected_set = set(str(did).strip() for did in selected)
    filtered = [r for r in retrieval if isinstance(r, dict) and (r.get("dataset_id") or r.get("id") or "").strip() in selected_set]
    return filtered, retrieval


def _run_transformability_for_subtask(subtask_id: str, ctx: Dict[str, Any]) -> int:
    """Run transformability for one subtask; return number of feasible plans. Mutates ctx."""
    subtasks = ctx.get("subtasks", []) or []
    subtask = next((s for s in subtasks if (s.get("id") or "").strip() == (subtask_id or "").strip()), None)
    if subtask is None:
        raise ValueError(f"[transformability_assessment] subtask_id not found: {subtask_id}")

    filtered, original = _filter_retrieval_by_selected(subtask)
    if original is not None:
        subtask["retrieval_result"] = filtered

    subtask_name = re.sub(r"[\/\\:\*?\"<>\|\s]", "_", (subtask.get("name") or subtask_id))
    subtask_name = re.sub(r"[^\w._-]", "_", subtask_name)
    cache_path = _grounding_cache_path(ctx, "transformability_check", subtask_id, subtask_name)
    legacy_cache_path = _legacy_grounding_cache_path(ctx, "transformability_check", subtask_name)
    model_config_path = ctx.get("model_config_path")
    use_stage_cache = _use_grounding_stage_cache(ctx)

    try:
        if use_stage_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    transformability = json.load(f)
            except (json.JSONDecodeError, IOError):
                os.remove(cache_path)
                ctx, transformability = transformability_check(
                    subtask_id=subtask_id, ctx=ctx, model_config_path=model_config_path
                )
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(transformability, f, ensure_ascii=False, indent=2)
        elif use_stage_cache and os.path.exists(legacy_cache_path):
            # Backward-compat: migrate old filename to new filename.
            try:
                with open(legacy_cache_path, "r", encoding="utf-8") as f:
                    transformability = json.load(f)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(transformability, f, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, IOError):
                try:
                    os.remove(legacy_cache_path)
                except OSError:
                    pass
                ctx, transformability = transformability_check(
                    subtask_id=subtask_id, ctx=ctx, model_config_path=model_config_path
                )
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(transformability, f, ensure_ascii=False, indent=2)
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            ctx, transformability = transformability_check(
                subtask_id=subtask_id, ctx=ctx, model_config_path=model_config_path
            )
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(transformability, f, ensure_ascii=False, indent=2)

        full_subtasks = list(ctx.get("subtasks", []) or [])
        for i, s in enumerate(full_subtasks):
            if (s.get("id") or "").strip() == (subtask_id or "").strip():
                full_subtasks[i]["transformability"] = transformability
                ctx["subtasks"] = full_subtasks
                break

        n_plans = sum(1 for v in (transformability or {}).values() if isinstance(v, dict) and str(v.get("transformable") or "").strip().lower() == "yes")
        return n_plans
    finally:
        if original is not None:
            subtask["retrieval_result"] = original


@register_tool("transformability_assessment")
def transformability_assessment(
    decision_rationale: str,
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Assess transformability for (subtask, dataset) pairs that the agent selected.
    It will automatically run for every subtask that has completed search+select and not yet been assessed.
    - decision_rationale: A brief thought-provoking explanation of why this tool is called at this moment.
    """
    ctx = dict(context_variables or {})
    full_subtasks = list(ctx.get("subtasks", []) or [])

    # Require every subtask with preference to complete search+select before transformability.
    # Empty placeholder lists from Design are not enough; explicit status flags mark completed stages.
    # Exception: searched subtasks with 0 results or reviewed subtasks with 0 selected candidates are
    # already ungroundable and must not block assessment for other subtasks.
    not_ready = []
    for s in full_subtasks:
        sid = (s.get("id") or "").strip()
        if not sid:
            continue
        if not s.get("dataset_preference"):
            continue
        retrieval_result = s.get("retrieval_result")
        selected_ids = s.get("selected_candidate_ids") or []
        retrieval_searched = bool(s.get("retrieval_searched")) or (
            isinstance(retrieval_result, list) and len(retrieval_result) > 0
        )
        candidate_selection_done = bool(s.get("candidate_selection_done")) or bool(selected_ids)
        if retrieval_searched and isinstance(retrieval_result, list) and len(retrieval_result) == 0:
            continue
        if candidate_selection_done and len(selected_ids) == 0:
            continue
        if not retrieval_searched or not candidate_selection_done:
            not_ready.append(sid)
    if not_ready:
        return Result(
            value=(
                f"Cannot run transformability_assessment() yet: the following subtask(s) have not completed "
                f"dataset_search and select_candidates_for_subtask: {not_ready}. "
                "Complete search+select for every subtask first, then call transformability_assessment()."
            ),
            context_variables=ctx,
        )

    ids_to_run = []
    for s in full_subtasks:
        sid = (s.get("id") or "").strip()
        if not sid:
            continue
        if not (s.get("retrieval_result") and s.get("selected_candidate_ids")):
            continue
        # NOTE: some upstream tools initialize transformability as {}.
        # Treat empty dict as "not yet assessed".
        if s.get("transformability"):
            continue
        ids_to_run.append(sid)

    if not ids_to_run:
        return Result(
            value="No subtasks need transformability assessment (all done or none ready).",
            context_variables=ctx,
        )

    total_plans = 0
    for sid in ids_to_run:
        n = _run_transformability_for_subtask(sid, ctx)
        total_plans += n

    return Result(
        value=f"Transformability assessed for {len(ids_to_run)} subtask(s); total {total_plans} feasible plans.",
        context_variables=ctx,
    )


def _run_score_and_filter_for_subtask(subtask_id: str, ctx: Dict[str, Any]) -> int:
    """Run score_and_filter for one subtask; return number of passed candidates. Mutates ctx."""
    subtasks = ctx.get("subtasks", []) or []
    subtask = next((s for s in subtasks if (s.get("id") or "").strip() == (subtask_id or "").strip()), None)
    if subtask is None:
        raise ValueError(f"[score_and_filter] subtask_id not found: {subtask_id}")

    filtered, original = _filter_retrieval_by_selected(subtask)
    if original is not None:
        subtask["retrieval_result"] = filtered

    try:
        subtask_name = re.sub(r"[\/\\:\*?\"<>\|\s]", "_", (subtask.get("name") or subtask_id))
        subtask_name = re.sub(r"[^\w._-]", "_", subtask_name)
        cache_path = _grounding_cache_path(ctx, "scoring", subtask_id, subtask_name)
        legacy_cache_path = _legacy_grounding_cache_path(ctx, "scoring", subtask_name)
        model_config_path = ctx.get("model_config_path")
        use_stage_cache = _use_grounding_stage_cache(ctx)

        if use_stage_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    scored_candidates = json.load(f)
            except (json.JSONDecodeError, IOError):
                os.remove(cache_path)
                ctx, scored_candidates = scoring(
                    subtask_id=subtask_id, ctx=ctx, model_config_path=model_config_path
                )
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(scored_candidates, f, ensure_ascii=False, indent=2)
        elif use_stage_cache and os.path.exists(legacy_cache_path):
            # Backward-compat: migrate old filename to new filename.
            try:
                with open(legacy_cache_path, "r", encoding="utf-8") as f:
                    scored_candidates = json.load(f)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(scored_candidates, f, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, IOError):
                try:
                    os.remove(legacy_cache_path)
                except OSError:
                    pass
                ctx, scored_candidates = scoring(
                    subtask_id=subtask_id, ctx=ctx, model_config_path=model_config_path
                )
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(scored_candidates, f, ensure_ascii=False, indent=2)
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            ctx, scored_candidates = scoring(
                subtask_id=subtask_id, ctx=ctx, model_config_path=model_config_path
            )
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(scored_candidates, f, ensure_ascii=False, indent=2)

        full_subtasks = list(ctx.get("subtasks", []) or [])
        for i, s in enumerate(full_subtasks):
            if (s.get("id") or "").strip() == (subtask_id or "").strip():
                full_subtasks[i]["scored_candidates"] = scored_candidates
                full_subtasks[i]["scored_status"] = "yes"
                ctx["subtasks"] = full_subtasks
                break

        return len(scored_candidates) if isinstance(scored_candidates, dict) else 0
    finally:
        if original is not None:
            subtask["retrieval_result"] = original


@register_tool("score_and_filter")
def score_and_filter(
    decision_rationale: str,
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Score and filter candidate (subtask, dataset, plan) pairs. 
    It will automatically run for every subtask that has transformability done and not yet been scored.
    - decision_rationale: A brief explanation of why this tool is called at this moment.
    """
    ctx = dict(context_variables or {})
    full_subtasks = list(ctx.get("subtasks", []) or [])

    ids_to_run = []
    for s in full_subtasks:
        sid = (s.get("id") or "").strip()
        if not sid or not s.get("transformability"):
            continue
        if (s.get("scored_status") or "").strip().lower() == "yes":
            continue
        ids_to_run.append(sid)

    if not ids_to_run:
        return Result(
            value="No subtasks need score_and_filter (all done or none ready).",
            context_variables=ctx,
        )

    total_passed = 0
    for sid in ids_to_run:
        n = _run_score_and_filter_for_subtask(sid, ctx)
        total_passed += n

    return Result(
        value=f"Score and filter done for {len(ids_to_run)} subtask(s); total {total_passed} valid groundings.",
        context_variables=ctx,
    )


# ---------- 3) Grounding Decision (case_resolved) ----------

@register_tool("case_resolved")
def case_resolved(
    decision_rationale: str,
    accepted: bool,
    summary: str = "",
    reason: str = "",
    failed_subtask_ids: Optional[List[str]] = None,
    failure_reasons: Optional[Dict[str, str]] = None,
    feedback_to_design: str = "",
    context_variables: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Call when grounding is done. Use accepted=True when every subtask has at least one
    validated (dataset, plan) pair; use accepted=False when some subtask(s) cannot be grounded.
    - decision_rationale: A brief explanation of why this tool is called at this moment.
    - accepted=True: provide summary; passes grounded instantiations to allocation.
    - accepted=False: provide reason, failed_subtask_ids (list of failed subtask ids),
      failure_reasons (dict of subtask_id -> concise reason), and feedback_to_design
      (concrete design actions per failed subtask) for the Design Agent.
    """
    ctx = dict(context_variables or {})
    if accepted:
        subtasks = ctx.get("subtasks", []) or []
        ungrounded = [
            (st.get("id"), st.get("name"))
            for st in subtasks
            if not (st.get("scored_candidates") and len(st.get("scored_candidates", {})) > 0)
        ]
        if ungrounded:
            raise ValueError(
                f"[case_resolved] accepted=True but subtasks without valid groundings: {ungrounded}. "
                "Use accepted=False and provide reason/feedback_to_design instead."
            )
        payload = {
            "status": "accepted",
            "summary": (summary or "").strip() or "All subtasks grounded; ready for allocation.",
        }
        ctx["grounding_result"] = payload
        return Result(value=payload.get("summary", ""), context_variables=ctx)
    payload = {
        "status": "rejected",
        "reason": (reason or "").strip() or "One or more subtasks could not be grounded.",
        "failed_subtask_ids": list(failed_subtask_ids or []),
        "failure_reasons": dict(failure_reasons or {}),
        "feedback_to_design": (feedback_to_design or "").strip(),
    }
    ctx["grounding_result"] = payload
    ctx["grounding_feedback"] = payload.get("feedback_to_design") or payload.get("reason", "")
    return Result(value=payload.get("reason", ""), context_variables=ctx)
