from ast import List
import json
from utils.registry import register_tool
from utils.agent_utils import Result
from typing import Any, Dict, Optional, List
import os
from utils.llm_caller import llm_call_json
from tools.executor_tools.transformation_tools_rollback import transform_samples_iterative

_TRANSFORM_BATCH_FNAME = "transform_log/transform_batch/debug_transform_batch_{subtask_id}.json"


def _save_transform_cache(cache_path: str, context_variables: Dict[str, Any], subtask_id: str) -> None:
    """Write compact transform output to debug_transform_batch_{subtask_id}.json."""
    ctx_to_dump = {
        "task_id": context_variables.get("task_id"),
        "topic_id": context_variables.get("topic_id"),
        "subtask_order": context_variables.get("subtask_order"),
        "current_subtask_id": context_variables.get("current_subtask_id"),
        "transformed_buffer": context_variables.get("transformed_buffer", {}),
    }
    path = os.path.join(cache_path, _TRANSFORM_BATCH_FNAME.format(subtask_id=subtask_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ctx_to_dump, f, ensure_ascii=False, indent=2)
    print(f"[BenchmarkFlow] Transform batch cache saved to: {path}")


def _load_transformed_buffer_from_file(cache_path: str, subtask_id: str) -> Dict[str, Any]:
    """Load transformed_buffer from a saved debug_transform_batch JSON. Returns {} on missing/invalid."""
    path = os.path.join(cache_path, _TRANSFORM_BATCH_FNAME.format(subtask_id=subtask_id))
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("transformed_buffer") or {}
    except Exception:
        return {}


@register_tool("iterative_transform_batch")
def iterative_transform_batch(
    max_workers: int = 24,
    context_variables: Optional[Dict[str, Any]] = None,
    cache_root: Optional[str] = None,
) -> Result:
    """
    Iterative two-stage orchestrator:
    - Process multiple samples concurrently with a thread pool
    - Each worker runs the full per-sample loop (Stage1 plan -> Stage2 execute -> evaluate -> retry)
    - Up to 3 retries per step
    - Each sample completes all steps independently

    Expected ctx fields:
    - current_subtask_id: str
    - subtasks: List[Dict]
    - current_pairs: List[Dict]
    - dataset_cards: Dict[str -> dataset_card]
    - transformed_buffer: Dict[subtask_id -> List[ {dataset_id, idx, sample} ] ]
    - tools_list: Dict[str, Any]
    - model_config_path: Optional[str]

    Returns:
    - status: ok / partial / no_op
    - Also writes back to ctx
    """
    ctx = dict(context_variables or {})
    
    subtask_id = str(ctx.get("current_subtask_id") or "").strip()
    current_pairs = ctx.get("current_pairs") or []
    subtasks = ctx.get("subtasks") or []
    dataset_cards = ctx.get("dataset_cards") or {}
    transformed_buffer = ctx.get("transformed_buffer") or {}
    tools_list = ctx.get("tools_list") or {}
    
    if not subtask_id or not isinstance(current_pairs, list):
        print(f"[iterative_transform_batch] no_op: missing current_subtask_id or current_pairs")
        return ctx

    subtask = None
    for st in subtasks:
        if str(st.get("id") or "").strip() == subtask_id:
            subtask = st
            break

    if not subtask:
        print(f"[iterative_transform_batch] no_op: subtask not found: {subtask_id}")
        return ctx

    if not isinstance(dataset_cards, dict):
        print(f"[iterative_transform_batch] no_op: missing dataset_cards")
        return ctx

    # --- Iterative transform ---
    model_config_path = ctx.get("model_config_path")
    current_pairs, transformed_buffer, payload = transform_samples_iterative(
        subtask=subtask,
        current_pairs=current_pairs,
        dataset_cards=dataset_cards,
        transformed_buffer=transformed_buffer,
        subtask_id=subtask_id,
        max_workers=max_workers,
        tools_list=tools_list,
        cache_root=cache_root,
        model_config_path=model_config_path,
    )

    print(f"{subtask_id} Iterative transformation done:", payload)
    
    # --- Write back to ctx ---
    ctx["current_pairs"] = current_pairs
    ctx["transformed_buffer"] = transformed_buffer

    # --- Overall status ---
    status = payload.get("status", "unknown")

    if status == "ok":
        overall = "ok"
    elif status == "no_op":
        overall = "no_op"
    else:
        overall = "unknown"

    result_payload = {
        "status": overall,
        "details": payload,
    }
    
    return ctx
    