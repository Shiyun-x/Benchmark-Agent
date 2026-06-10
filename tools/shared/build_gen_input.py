from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import yaml
from typing import Any, Dict, List, Optional, Tuple


def _load_dataset_cards(dataset_card_dir: str) -> List[Dict[str, Any]]:
    """Load all dataset card JSON files from a directory."""
    cards: List[Dict[str, Any]] = []
    if not dataset_card_dir or not os.path.isdir(dataset_card_dir):
        return cards
    for p in glob.glob(os.path.join(dataset_card_dir, "*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    cards.extend([c for c in data if isinstance(c, dict)])
                elif isinstance(data, dict):
                    cards.append(data)
        except Exception:
            continue
    return cards


def _load_dataset_cards_from_config(config_path: str) -> List[Dict[str, Any]]:
    """Load dataset cards from a YAML config (dataset_cards.yaml).

    The config specifies dataset_card_dir and an optional datasets list of IDs to include.
    Paths in the config are resolved relative to the project root (two levels above utils/resources/).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    card_dir = cfg.get("dataset_card_dir", "")
    if card_dir and not os.path.isabs(card_dir):
        project_root = os.path.abspath(os.path.join(os.path.dirname(config_path), "..", ".."))
        card_dir = os.path.join(project_root, card_dir)
    cards = _load_dataset_cards(card_dir)
    allowed_ids = cfg.get("datasets")
    if allowed_ids:
        allowed_set = {str(i) for i in allowed_ids}
        cards = [c for c in cards if str(c.get("dataset_id", "")) in allowed_set]
    return cards


def _build_id2card(dataset_cards: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build mapping: dataset_id -> dataset_card."""
    id2card = {}
    for card in dataset_cards or []:
        ds_id = card.get("dataset_id")
        if isinstance(ds_id, str) and ds_id:
            id2card[ds_id] = card
    return id2card


def _compute_gap_per_subtask(
    subtask_order: List[str],
    verified: Dict[str, Dict[str, Any]],
    allocation_subtasks: Dict[str, Any],
) -> Dict[str, int]:
    """Return per-subtask quota deficit. Only includes subtasks with gap > 0."""
    gap_per_subtask: Dict[str, int] = {}
    for stid in subtask_order:
        quota = int(allocation_subtasks.get(stid, {}).get("quota", 0) or 0)
        vbuf = verified.get(stid, {}).get("verified_buffer") or {}
        verified_count = sum(len(items) for items in vbuf.values() if isinstance(items, list))
        gap = max(0, quota - verified_count)
        if gap > 0:
            gap_per_subtask[stid] = gap
    return gap_per_subtask


def _refill_idx_todo_for_replenishment(
    context_variables: Dict[str, Any],
    gap_per_subtask: Dict[str, int],
    allocation_subtasks: Dict[str, Any],
) -> int:
    """Pull extra indices from dataset_index_pool into pairs' idx_todo for replenishment.

    Mutates context_variables in place. Returns total number of indices added.
    """
    dataset_index_pool = context_variables.get("dataset_index_pool") or {}
    pairs_by_subtask = context_variables.get("pairs_by_subtask") or {}
    total_added = 0
    for stid, gap in gap_per_subtask.items():
        if gap <= 0:
            continue
        pairs = pairs_by_subtask.get(stid) or []
        if not pairs:
            continue
        need_per_pair = (gap + len(pairs) - 1) // len(pairs)
        for pair in pairs:
            did = str(pair.get("dataset_id") or "").strip()
            if not did:
                continue
            pool = dataset_index_pool.get(did)
            if not isinstance(pool, list):
                continue
            take = min(need_per_pair, len(pool))
            if take <= 0:
                continue
            add_idx = pool[:take]
            dataset_index_pool[did] = pool[take:]
            pair.setdefault("idx_todo", [])
            if not isinstance(pair["idx_todo"], list):
                pair["idx_todo"] = []
            pair["idx_todo"].extend(add_idx)
            total_added += take
    return total_added


def _stable_seed(*parts: str) -> int:
    """
    Stable deterministic seed from multiple string parts.
    """
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def group_pairs_by_subtask(
    execution_pairs: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert flat execution_pairs -> {subtask_id: [pair, ...]}.
    Keeps original pair dict objects (mutations reflect globally).
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in execution_pairs or []:
        sid = str(p.get("subtask_id") or "").strip()
        if not sid:
            continue
        out.setdefault(sid, []).append(p)
    return out


def build_generator_inputs(
    benchmark_task_id: str,
    subtasks: List[Dict[str, Any]],
    id2card: Dict[str, Dict[str, Any]],
    allocation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Preparation-stage ONLY (JSON-serializable, no locks, no refill/reserve):

    Builds:
      - execution_pairs: (subtask,dataset) pairs with initial idx_todo filled (global no-dup per dataset)
      - pairs_by_subtask: {subtask_id: [pair, ...]}
      - dataset_index_pool: remaining idx per dataset (NOT consumed in prepare stage)
      - subtask_order: subtask ids ordered by importance DESC (only those that exist in pairs_by_subtask)

    Pair schema (minimal, execution-stage may add more fields later):
      {
        "subtask_id": str,
        "dataset_id": str,
        "need_n": int,
        "tool_plan": list,
        "idx_todo": list[int]
      }

    Expectations:
      - allocation["subtasks"][stid]["alloc"] == {did: need_n}
      - subtasks[*]["transformability"][did]["plan"] (optional)
      - id2card[did]["size_samples"] == N
    """
    allocation = allocation or {}
    alloc_subtasks: Dict[str, Any] = allocation.get("subtasks") or {}

    # Sort subtasks by importance DESC (deterministic), do NOT mutate input list.
    subtasks_sorted = sorted(
        (subtasks or []),
        key=lambda st: (-float(st.get("importance", 0.0) or 0.0), str(st.get("id", ""))),
    )

    execution_pairs: List[Dict[str, Any]] = []
    per_dataset_reqs: Dict[str, List[Tuple[int, str, int]]] = {}   # did -> [(order, stid, need_n)]
    pair_entry_map: Dict[Tuple[str, str], Dict[str, Any]] = {}     # (stid, did) -> pair
    order_counter = 0

    # 1) Build (subtask,dataset) pairs and collect per-dataset requirements
    for st in subtasks_sorted:
        stid = str(st.get("id") or "").strip()
        if not stid:
            continue

        st_alloc_info = alloc_subtasks.get(stid) or {}
        alloc_map = st_alloc_info.get("alloc") or {}
        if not isinstance(alloc_map, dict) or not alloc_map:
            continue

        transformability = st.get("transformability") or {}

        for did_raw, need_raw in alloc_map.items():
            did = str(did_raw or "").strip()
            if not did:
                continue

            try:
                need_n = int(need_raw)
            except Exception:
                need_n = 0
            if need_n <= 0:
                continue

            tp = transformability.get(did) or {}
            tool_plan = (tp.get("plan") or []) if isinstance(tp, dict) else []

            # Guard against duplicate (stid,did) by merging (shouldn't happen, but safe)
            key = (stid, did)
            pair = pair_entry_map.get(key)
            if pair is None:
                pair = {
                    "subtask_id": stid,
                    "dataset_id": did,
                    "need_n": need_n,
                    "tool_plan": tool_plan,
                    "idx_todo": [],  # filled later by per-dataset slicing
                }
                pair_entry_map[key] = pair
                execution_pairs.append(pair)
            else:
                pair["need_n"] = int(pair.get("need_n", 0) or 0) + need_n
                if not pair.get("tool_plan") and tool_plan:
                    pair["tool_plan"] = tool_plan

            per_dataset_reqs.setdefault(did, []).append((order_counter, stid, need_n))
            order_counter += 1

    # 2) Assign global no-dup indices per dataset, then slice to each pair's idx_todo
    dataset_index_pool: Dict[str, List[int]] = {}

    for did, reqs in per_dataset_reqs.items():
        card = id2card.get(did) or {}
        try:
            n_total = int(card.get("size_samples", 0) or 0)
        except Exception:
            n_total = 0

        if n_total <= 0:
            dataset_index_pool[did] = []
            continue

        total_need = sum(int(x[2]) for x in reqs)
        if total_need > n_total:
            raise ValueError(
                f"Dataset {did}: total allocated {total_need} > size_samples {n_total}"
            )

        # Stable shuffle for reproducibility
        full = list(range(n_total))
        rng = random.Random(_stable_seed(benchmark_task_id, did))
        rng.shuffle(full)

        reqs_sorted = sorted(reqs, key=lambda x: x[0])
        cursor = 0

        for _, stid, need_n in reqs_sorted:
            if need_n <= 0:
                continue

            take = full[cursor: cursor + need_n]
            if len(take) != need_n:
                raise RuntimeError(
                    f"Dataset {did}: insufficient idxs when slicing (need {need_n}, got {len(take)})"
                )
            cursor += need_n

            pair = pair_entry_map.get((stid, did))
            if pair is not None:
                pair["idx_todo"].extend(take)

        dataset_index_pool[did] = full[cursor:]  # remaining idxs (do NOT consume in prepare stage)

    pairs_by_subtask = group_pairs_by_subtask(execution_pairs)

    # 3) subtask order (only keep subtasks that actually have pairs)
    subtask_order: List[str] = []
    for st in subtasks_sorted:
        stid = str(st.get("id") or "").strip()
        if stid and stid in pairs_by_subtask:
            subtask_order.append(stid)

    return {
        "pairs_by_subtask": pairs_by_subtask,
        "dataset_index_pool": dataset_index_pool,
        "subtask_order": subtask_order,
    }


def _load_transformation_tools(ctx: dict, path: str) -> List[Dict[str, Any]]:
    """Load and filter transformation tools from a YAML file by the active modalities in ctx."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "tools" not in data:
        raise ValueError("Invalid tool yaml format: missing `tools`")
    tools = data["tools"]
    if not isinstance(tools, list):
        raise ValueError("`tools` must be a list")
    modalities = ctx.get("modalities", []) or []
    return [
        tool for tool in tools
        if any(m in modalities for m in (tool.get("modalities") or []))
    ]
