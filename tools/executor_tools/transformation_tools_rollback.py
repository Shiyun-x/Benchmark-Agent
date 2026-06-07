# tools/transformation_tools_rollback.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from tqdm import tqdm
from utils.registry import register_tool
import copy
from utils.llm_caller import llm_call_json
from tools.executor_tools.run_pure_tools import run_pure_step, build_pure_tool_registry
from utils.model_config import get_tool_model
from tools.shared.choice_question import stable_seed, normalize_and_shuffle_choice_question
from tools.shared.media_paths import resolve_image_paths
from tools.shared.primitives import _safe_int
from tools.shared.tool_sanitizer import _sanitize_pure_tool_op

SAMPLE_BRIEF_CACHE_LOCK = threading.Lock()
SAMPLE_ITERATIVE_CACHE_LOCK = threading.Lock()


def _get_stage1_model(model_config_path=None):
    """Get stage1 model from config, with fallback to default."""
    try:
        return get_tool_model("transform_stage1", model_config_path)
    except Exception:
        return "gpt-5.1"  # Fallback default

def _get_stage2_model(model_config_path=None):
    """Get stage2 model from config, with fallback to default."""
    try:
        return get_tool_model("transform_stage2", model_config_path)
    except Exception:
        return "gpt-5.1"  # Fallback default

def _get_stage0_model(model_config_path=None):
    """Get sample-brief model from config, with fallback to stage1/default."""
    try:
        return get_tool_model("transform_stage0", model_config_path)
    except Exception:
        return _get_stage1_model(model_config_path)

STAGE1_MODEL = _get_stage1_model()
STAGE2_MODEL = _get_stage2_model()

IMAGE_KEY_PATTERNS = ["image_file", "image_files", "image", "img", "image_list"]
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

# LLM call timeout (seconds). Prevents a single hung request from stalling the whole pipeline.
# You can override via env var: LLM_TIMEOUT_S=180
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "90"))

# Kill the process if a transform batch makes no sample-level progress for too long.
# This is intentionally process-level because a hung provider request inside a worker
# thread cannot be safely interrupted from the parent thread on Windows.
NO_PROGRESS_TIMEOUT_S = int(os.getenv("TRANSFORM_NO_PROGRESS_TIMEOUT_S", "480"))

# Diagnostics: print per-sample timing + heartbeat for stuck futures.
# Enable via env var: TRANSFORM_DIAG=1
# works even if the module was imported before env vars were set (common in IDE debug sessions).
TRANSFORM_DIAG = str(os.getenv("TRANSFORM_DIAG", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
TRANSFORM_DIAG_HEARTBEAT_S = int(os.getenv("TRANSFORM_DIAG_HEARTBEAT_S", "30"))

# Step progress printing (per-sample, per-step).
# Default OFF. Enable via env: TRANSFORM_TRACE=1
TRANSFORM_TRACE = str(os.getenv("TRANSFORM_TRACE", "0")).strip().lower() not in ("0", "false", "no", "n", "off")
TRANSFORM_TRACE_HEARTBEAT_S = int(os.getenv("TRANSFORM_TRACE_HEARTBEAT_S", "15"))

def _jd(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)

def _build_subtask_json(subtask: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": subtask.get("id"),
        "name": subtask.get("name"),
        "description": subtask.get("description"),
        "io_schema": subtask.get("sample_schema"),
        "answer_type": subtask.get("answer_type"),
    }

def _build_subtask_json_for_stage2_llm(subtask: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": subtask.get("id"),
        "name": subtask.get("name"),
        "description": subtask.get("description"),
        "answer_type": subtask.get("answer_type"),
    }

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_transformation_buffer_cache(
    cache_root: Optional[str],
    *,
    subtask_id: str,
    dataset_id: str,
) -> List[Dict[str, Any]]:
    """
    Load the persisted transformation buffer for a (subtask_id, dataset_id) pair.
    This cache is the authoritative "already transformed" output; when hit, we should
    skip Stage1/Stage2 for those indices.

    Expected file schema:
      {
        "subtask_id": "...",
        "dataset_id": "...",
        "transformed_buffer": [
          {"dataset_id": "...", "idx": 12, "sample": {...}},
          ...
        ]
      }
    """
    if not cache_root or not isinstance(cache_root, str):
        return []
    if not subtask_id or not dataset_id:
        return []
    try:
        transformed_cache_path = os.path.join(
            cache_root,
            "transform_log",
            "transformation_buffer",
            f"{subtask_id}_{dataset_id}_transformed.json",
        )
        if not os.path.exists(transformed_cache_path):
            return []
        cached_data = _load_json(transformed_cache_path)
        if not isinstance(cached_data, dict):
            return []
        items = cached_data.get("transformed_buffer") or []
        if not isinstance(items, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            idx = it.get("idx")
            sample = it.get("sample")
            did = it.get("dataset_id") or dataset_id
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if not isinstance(sample, dict):
                continue
            out.append({"dataset_id": str(did), "idx": idx_int, "sample": sample})
        return out
    except Exception:
        return []


def _sample_design_brief_cache_path(
    cache_root: Optional[str],
    *,
    subtask_id: Optional[str],
    dataset_id: Optional[str],
    idx_int: int,
) -> Optional[str]:
    if not cache_root or not isinstance(cache_root, str):
        return None
    if not subtask_id:
        return None
    safe_subtask = str(subtask_id).replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    return os.path.join(
        cache_root,
        "transform_log",
        "sample_design_briefs",
        f"{safe_subtask}.json",
    )


def _sample_design_brief_cache_key(dataset_id: Optional[str], idx_int: int) -> str:
    safe_dataset = str(dataset_id or "").replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    return f"{safe_dataset}::{idx_int}"


def _sample_iterative_cache_path(
    cache_root: Optional[str],
    *,
    subtask_id: Optional[str],
    dataset_id: Optional[str] = None,
    idx_int: int = 0,
) -> Optional[str]:
    if not cache_root or not isinstance(cache_root, str):
        return None
    if not subtask_id:
        return None
    safe_subtask = str(subtask_id).replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    return os.path.join(
        cache_root,
        "transform_log",
        "sample_iterative",
        f"{safe_subtask}.json",
    )


def _load_sample_iterative_state(
    cache_path: Optional[str],
    *,
    dataset_id: Optional[str],
    idx_int: int,
) -> Dict[str, Any]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with SAMPLE_ITERATIVE_CACHE_LOCK:
            data = _load_json(cache_path)
        if not isinstance(data, dict):
            return {}
        bucket = data.get("sample_iterative_states")
        if not isinstance(bucket, dict):
            return {}
        cached = bucket.get(_sample_design_brief_cache_key(dataset_id, idx_int))
        return cached if isinstance(cached, dict) else {}
    except Exception:
        return {}


def _save_sample_iterative_state(
    cache_path: Optional[str],
    *,
    subtask_id: Optional[str],
    dataset_id: Optional[str],
    idx_int: int,
    state: Dict[str, Any],
) -> None:
    if not cache_path or not isinstance(state, dict) or not state:
        return
    try:
        with SAMPLE_ITERATIVE_CACHE_LOCK:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            existing: Dict[str, Any] = {}
            if os.path.exists(cache_path):
                try:
                    existing = _load_json(cache_path)
                except Exception:
                    existing = {}
            if not isinstance(existing, dict):
                existing = {}

            bucket = existing.get("sample_iterative_states")
            if not isinstance(bucket, dict):
                bucket = {}

            bucket[_sample_design_brief_cache_key(dataset_id, idx_int)] = state
            existing["sample_iterative_states"] = bucket
            existing.setdefault("subtask_id", subtask_id)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_sample_design_brief_cache(
    cache_path: Optional[str],
    *,
    dataset_id: Optional[str],
    idx_int: int,
) -> Dict[str, Any]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with SAMPLE_BRIEF_CACHE_LOCK:
            data = _load_json(cache_path)
        if not isinstance(data, dict):
            return {}
        bucket = data.get("sample_design_briefs")
        if not isinstance(bucket, dict):
            return {}
        brief = bucket.get(_sample_design_brief_cache_key(dataset_id, idx_int))
        return _normalize_sample_design_brief(brief) if isinstance(brief, dict) else {}
    except Exception:
        return {}


def _save_sample_design_brief_cache(
    cache_path: Optional[str],
    *,
    subtask_id: Optional[str],
    dataset_id: Optional[str],
    idx_int: int,
    sample_design_brief: Dict[str, Any],
) -> None:
    if not cache_path or not isinstance(sample_design_brief, dict) or not sample_design_brief:
        return
    sample_design_brief = _normalize_sample_design_brief(sample_design_brief)
    if not sample_design_brief:
        return
    try:
        with SAMPLE_BRIEF_CACHE_LOCK:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            existing = {}
            if os.path.exists(cache_path):
                try:
                    existing = _load_json(cache_path)
                except Exception:
                    existing = {}
            if not isinstance(existing, dict):
                existing = {}
            bucket = existing.get("sample_design_briefs")
            if not isinstance(bucket, dict):
                bucket = {}
            bucket[_sample_design_brief_cache_key(dataset_id, idx_int)] = sample_design_brief
            payload = {
                "subtask_id": subtask_id,
                "sample_design_briefs": bucket,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _looks_like_image_path(v):
    if not isinstance(v, str):
        return False
    v = v.lower()
    return any(v.endswith(ext) for ext in IMAGE_EXTS)

def _collect_image_files_with_keys(sample: dict):
    """
    Return [(path, key_path_str), ...]
    key_path_str is like "input.image_files[0]", giving the LLM semantic location hints.
    """
    results = []

    inp = sample.get("input")
    if isinstance(inp, dict):
        for k, v in inp.items():
            kk = str(k).lower()
            if any(p in kk for p in IMAGE_KEY_PATTERNS):
                if isinstance(v, str) and _looks_like_image_path(v):
                    results.append((v, f"input.{k}"))
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, str) and _looks_like_image_path(item):
                            results.append((item, f"input.{k}[{i}]"))

    return results


def _collect_image_paths_from_resources(obj: Any) -> List[str]:
    """
    Recursively collect all string values that look like image paths from resources
    (or any nested dict/list). Used in Stage2: pass image paths from step resource_fields
    resources to the vision model. Returns a deduplicated path list (first-seen order preserved).
    """
    out: List[str] = []

    def _walk(v: Any) -> None:
        if isinstance(v, str):
            if _looks_like_image_path(v):
                out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, list):
            for x in v:
                _walk(x)

    _walk(obj)
    return list(dict.fromkeys(out))

def _pick_samples_by_indices(samples: List[Any], indices: List[int]) -> List[Any]:
    picked = []
    for idx in indices:
        if idx < 0 or idx >= len(samples):
            continue
        picked.append(samples[idx])
    return picked

# =========================
# Helper functions for field path operations
# =========================

def _get_by_path(obj: Dict[str, Any], path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def _set_by_path(obj: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    *parents, last = parts
    cur = obj
    for p in parents:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[last] = value

def _del_by_path(obj: Dict[str, Any], path: str) -> None:
    parts = path.split(".")
    *parents, last = parts
    cur = obj
    for p in parents:
        if not isinstance(cur, dict) or p not in cur:
            return
        cur = cur[p]
    if isinstance(cur, dict) and last in cur:
        del cur[last]

def _deep_merge(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def _build_resources_for_step(sample: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
    resources: Dict[str, Any] = {}
    resource_fields = step.get("resource_fields") or []
    original_root = sample.get("original") or {}
    current_root = sample.get("current") or {}
    brief_top = sample.get("sample_design_brief")

    for path in resource_fields:
        if not isinstance(path, str) or not path:
            continue

        if path.startswith("original."):
            inner_path = path[len("original.") :]
            value = _get_by_path(original_root, inner_path)
        elif path == "interim.sample_design_brief" and isinstance(brief_top, dict) and brief_top:
            value = brief_top
        else:
            # e.g. "tool_memory.web_search.identify_artist.output.answer" /
            # "interim.turn_list" / "final.input.question"
            value = _get_by_path(current_root, path)

        if value is None:
            continue
        _set_by_path(resources, path, value)

    return resources


def _preview_for_prompt(value: Any, max_chars: int) -> str:
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max(0, max_chars - 3)] + "..."
    return text


def _iter_memory_leaf_paths(value: Any, prefix: str) -> List[Tuple[str, Any]]:
    if isinstance(value, dict):
        leaves: List[Tuple[str, Any]] = []
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                continue
            leaves.extend(_iter_memory_leaf_paths(child, f"{prefix}.{key}"))
        return leaves
    return [(prefix, value)]


def _iter_tool_memory_records(node: Any, base_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    if not isinstance(node, dict):
        return []
    if isinstance(node.get("input"), dict) or isinstance(node.get("output"), dict):
        return [(base_path, node)]

    records: List[Tuple[str, Dict[str, Any]]] = []
    for key, child in node.items():
        if not isinstance(key, str) or not key or key.startswith("_"):
            continue
        if key in {"calls", "latest_by_tool"}:
            continue
        records.extend(_iter_tool_memory_records(child, f"{base_path}.{key}"))
    return records


def _summarize_tool_memory(tool_memory: Any) -> Dict[str, Any]:
    if not isinstance(tool_memory, dict):
        return {}

    summary_records: Dict[str, Any] = {}
    for base_path, record in _iter_tool_memory_records(tool_memory, "tool_memory"):
        max_chars = 500
        try:
            max_chars = int(max_chars)
        except Exception:
            max_chars = 500
        max_chars = max(80, max_chars)

        resource_paths: List[str] = []
        preview: Dict[str, str] = {}
        for section in ("input", "output"):
            section_value = record.get(section)
            if section_value is None:
                continue
            section_base = f"{base_path}.{section}"
            for path, value in _iter_memory_leaf_paths(section_value, section_base):
                if value is None:
                    continue
                resource_paths.append(path)
                preview[path] = _preview_for_prompt(value, max_chars)

        if not resource_paths:
            continue
        summary_records[base_path] = {
            "resource_paths": resource_paths,
            "preview": preview,
        }

    if not summary_records:
        return {}
    return {
        "summary_only": True,
        "usage": (
            "Use these stable resource_paths when an LLM step needs retained PURE tool input/output. "
            "Do not copy tool_memory into interim only for memory."
        ),
        "records": summary_records,
    }


def _current_state_for_stage1_prompt(current_state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(current_state or {})
    summary = _summarize_tool_memory(state.get("tool_memory"))
    if summary:
        state["tool_memory"] = summary
    else:
        state.pop("tool_memory", None)
    return state


def _store_tool_memory_record(
    current_state: Dict[str, Any],
    memory_record: Dict[str, Any],
    *,
    iteration: int,
) -> None:
    if not isinstance(current_state, dict) or not isinstance(memory_record, dict) or not memory_record:
        return
    target_base = memory_record.get("target_base")
    if not isinstance(target_base, str) or not target_base.startswith("tool_memory."):
        return

    payload = {
        "input": copy.deepcopy(memory_record.get("input") or {}),
        "output": copy.deepcopy(memory_record.get("output") or {}),
    }

    _set_by_path(current_state, target_base, payload)


def _apply_step_delta(
    current: Dict[str, Any],
    step: Dict[str, Any],
    delta_current: Dict[str, Any],
) -> Dict[str, Any]:
    new_current = copy.deepcopy(current or {})

    for top_key in ("interim", "final"):
        part = delta_current.get(top_key)
        if isinstance(part, dict):
            if not isinstance(new_current.get(top_key), dict):
                new_current[top_key] = {}
            new_current[top_key] = _deep_merge(new_current[top_key], part)

    for p in step.get("deleted_interim_fields") or []:
        if isinstance(p, str) and p.startswith("interim."):
            _del_by_path(new_current, p)

    return new_current


def _enforce_deleted_fields(
    current: Dict[str, Any],
    step: Dict[str, Any],
) -> List[str]:
    """
    Defensive cleanup: if deleted_interim_fields still exist after applying a step,
    force-delete them again and report any residual paths that still exist.
    """
    residual: List[str] = []
    for p in step.get("deleted_interim_fields") or []:
        if not isinstance(p, str) or not p.startswith("interim."):
            continue
        # Best-effort second pass deletion.
        if _get_by_path(current, p) is not None:
            _del_by_path(current, p)
        if _get_by_path(current, p) is not None:
            residual.append(p)
    return residual

def _check_target_fields_filled(
    new_current: Dict[str, Any],
    step: Dict[str, Any],
) -> List[str]:
    target_fields = step.get("target_fields") or []
    tool_type = str(step.get("tool_type") or "").upper()
    if tool_type == "PURE" and not target_fields:
        return []

    missing: List[str] = []
    for tpath in target_fields:
        if not isinstance(tpath, str) or not tpath:
            continue
        if tool_type != "PURE" and tpath.startswith("tool_memory."):
            missing.append(tpath)
            continue
        if tool_type == "PURE" and tpath and not tpath.startswith("tool_memory."):
            missing.append(tpath)
            continue
        if tpath.startswith("original."):
            continue
        if _get_by_path(new_current, tpath) is None:
            missing.append(tpath)
    return missing

# =========================
# Stage0: Sample-level design brief
# =========================
STAGE0_SAMPLE_BRIEF_PROMPT = r"""
====================================================
YOUR ROLE
====================================================
You are the Sample-Level Design Brief Writer for a benchmark construction system.

Your job is to inspect ONE raw sample and produce a compact design brief that helps later stages construct a high-quality benchmark item for the given subtask.

You do NOT execute transformations.
You do NOT call tools or retrieve external evidence yourself.
You do NOT produce a rigid step-by-step execution plan.
You DO identify how this specific sample can be shaped into a difficult, faithful, and novel evaluation item.

====================================================
INPUTS
====================================================
- subtask_json: target evaluation subtask
{subtask_json}

- tool_plan_template_json: dataset-level transformation sketch. Treat it as a loose skeleton, not a binding sample-level plan.
{template_ops_json}

- later_available_tools_json: PURE tools that later stages may insert when the sample needs extra evidence or modality conversion. Treat these as planning affordances only; you cannot call them.
{later_available_tools_json}

- raw_sample_json: original sample to diagnose
{raw_sample_json}

====================================================
WHAT TO PRODUCE
====================================================
Write a compact working contract for Stage1:

1. fit
- "good" if the sample naturally supports the subtask.
- "weak" if it can work but needs careful restructuring or light conflict injection.
- "skip" if it is unsuitable.

2. core_facts
- 3-5 atomic facts that MUST remain true.
- Use concrete sample-specific entities, roles, dates, values, or relations.

3. design_goal
- One sentence: what final benchmark item this sample should become and what capability it should test.

4. reasoning_hops
- 2-4 evidence-dependent hops the model under test must perform to solve the final item.
- These are solver-side reasoning requirements, not construction steps.
- **What counts as a "hop":** any step where the final answer is **not** trivially readable in one shot from the presented input. This is **not** limited to symbolic math-style chains. Examples: (i) read a non-obvious signal from an image/audio/table, **then** apply a predicate or combine with another cue; (ii) integrate two independent constraints; (iii) combine modal evidence with text. Perceptual extraction ("what is shown / said / measured here?") **is** a hop when the answer is not directly printed on the surface and further use of that information is required.
- Each hop must name a concrete evidence operation (e.g. compare two claims, reconcile a timeline, discount a biased source, combine image/audio/text cues, identify a depicted entity or style then map it to a factual claim).
- If a hop needs external **factual** knowledge (entity, historical, geographic, cultural, artwork, scientific, etc.) that is absent from the raw fields, you may describe it as evidence to be retrieved by a later tool such as `web_search`. **When the sample includes images,** describe retrieval that can use **the same image** (image-grounded search) to anchor facts about what is depicted—do not assume only text queries. Combining input-grounded perception (including pixels) with such facts is allowed when the final judgment still **depends on what is in the input** (the item must not be answerable from world knowledge alone while ignoring the input).
- Do NOT invent retrieved facts or cite sources that are not in the raw sample; only mark the evidence gap and how later retrieved evidence should be used.
- Avoid generic hops like "understand the context" or "choose the best answer".

5. difficulty_plan
- One sentence describing how to make the final item hard/new: e.g. confusable distractors, source weighting, temporal/causal reconciliation, cross-modal evidence, etc.
- It MUST say how wrong answers remain locally plausible but are falsified by evidence.
- It MUST say how to avoid single-cue solving or keyword matching.

6. leakage_guards
- 2-4 concrete things Stage1/Stage2 must not reveal in final.input.
- **Filenames are out of scope:** the evaluated model only sees native multimodal content (pixels, waveform, etc.), not local asset file names or path strings. Do **not** list "hide the filename" as a leakage risk as if the model could read it; leakage_guards address content in final.input that the model actually receives.

7. first_focus
- One sentence: the first construction focus for Stage1.

====================================================
CONCISENESS + ANTI-TEMPLATE RULES
====================================================
- Keep the brief concise and sample-specific.
- Avoid generic wording that could apply to any news/event sample.
- Use concrete entities/signals from this sample (names, roles, timeline anchors, concrete fields), but do not copy long passages.
- Prefer short bullets/phrases over long prose.
- Do NOT output repeated boilerplate across fields.
- Think divergently BEFORE writing: consider multiple possible sample-reconstruction routes.
- Then converge to ONE best direction and encode only that single direction in the final JSON.
- Favor difficulty from evidence dependencies, not from obscure wording or longer context.
- The final item should require at least TWO independent evidence checks whenever the sample supports it.
- Do not mark a sample as "skip" only because it lacks external facts if later_available_tools_json includes a suitable retrieval tool and the sample has concrete anchors for that retrieval.

Hard size limits:
- `core_facts`: at most 5 items.
- Other list fields: at most 4 items.
- Each item: at most 1 short sentence.
- `design_goal`, `difficulty_plan`, `first_focus`: keep each under ~220 characters.
- Total JSON should stay compact (target <= 1200 chars when possible).

====================================================
OUTPUT FORMAT
====================================================
Return EXACTLY ONE JSON object:

{{
  "fit": "good" | "weak" | "skip",
  "core_facts": ["..."],
  "design_goal": "...",
  "reasoning_hops": ["..."],
  "difficulty_plan": "...",
  "leakage_guards": ["..."],
  "first_focus": "..."
}}
"""


def _cap_list(values: Any, max_items: int = 4, max_item_len: int = 220) -> List[Any]:
    if not isinstance(values, list):
        return []
    out: List[Any] = []
    seen = set()
    for v in values:
        if len(out) >= max_items:
            break
        if isinstance(v, str):
            vv = " ".join(v.split()).strip()
            if not vv:
                continue
            vv = vv[:max_item_len]
            key = vv.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(vv)
        else:
            out.append(v)
    return out


def _cap_text(value: Any, max_len: int = 220) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()[:max_len]


def _normalize_sample_design_brief(brief: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact and de-template Stage0 output to reduce prompt bloat for Stage1.
    Keeps only the compact Stage0 contract. Also maps older verbose brief caches.
    """
    if not isinstance(brief, dict):
        return {}

    fit_raw = brief.get("fit")
    if isinstance(fit_raw, dict):
        status = str(fit_raw.get("status") or "weak").strip().lower()
    else:
        status = str(fit_raw or "weak").strip().lower()
    if status not in ("good", "weak", "skip"):
        status = "weak"

    old_fit = fit_raw if isinstance(fit_raw, dict) else {}
    old_design = brief.get("design_intent") if isinstance(brief.get("design_intent"), dict) else {}
    old_difficulty = brief.get("difficulty_strategy") if isinstance(brief.get("difficulty_strategy"), dict) else {}
    old_constraints = brief.get("constraints") if isinstance(brief.get("constraints"), dict) else {}

    core_facts = brief.get("core_facts")
    if not isinstance(core_facts, list):
        core_facts = old_constraints.get("must_preserve") or old_fit.get("usable_signals") or []

    design_goal = brief.get("design_goal")
    if not isinstance(design_goal, str) or not design_goal.strip():
        parts = [
            _cap_text(old_design.get("item_shape"), 120),
            _cap_text(old_design.get("target_capability"), 120),
        ]
        design_goal = " ".join(p for p in parts if p).strip()

    reasoning_hops = brief.get("reasoning_hops")
    if not isinstance(reasoning_hops, list):
        reasoning_hops = old_difficulty.get("reasoning_hops") or old_design.get("what_model_must_reason_over") or []

    difficulty_plan = brief.get("difficulty_plan")
    if not isinstance(difficulty_plan, str) or not difficulty_plan.strip():
        parts = [
            _cap_text(old_difficulty.get("distractor_strategy"), 120),
            _cap_text(old_difficulty.get("novelty_strategy"), 100),
        ]
        difficulty_plan = " ".join(p for p in parts if p).strip()

    leakage_guards = brief.get("leakage_guards")
    if not isinstance(leakage_guards, list):
        leakage_guards = old_constraints.get("must_not_leak") or old_fit.get("leakage_risks") or old_difficulty.get("avoid_shortcuts") or []

    return {
        "fit": status,
        "core_facts": _cap_list(core_facts, 5, 180),
        "design_goal": _cap_text(design_goal, 220),
        "reasoning_hops": _cap_list(reasoning_hops, 4, 180),
        "difficulty_plan": _cap_text(difficulty_plan, 220),
        "leakage_guards": _cap_list(leakage_guards, 4, 180),
        "first_focus": _cap_text(brief.get("first_focus"), 220),
    }


def _build_later_available_tools_for_stage0(pure_tool_registry: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expose only compact planning affordances to Stage0; tools still execute later."""
    tools: List[Dict[str, Any]] = []
    for name, spec in (pure_tool_registry or {}).items():
        name_str = str(name)
        if not (name_str.startswith("web_") or name_str.endswith("_evidence")):
            continue
        tools.append({
            "name": name_str,
            "description": getattr(spec, "description", ""),
            "inputs": list((getattr(spec, "param_schema", {}) or {}).keys()),
            "planning_note": (
                "Later stages may insert this PURE tool when the sample has concrete "
                "anchors and the subtask needs external evidence; Stage0 must not call it."
            ),
        })
    return tools


def _build_stage0_prompt(
    subtask: Dict[str, Any],
    template_ops: List[Dict[str, Any]],
    sample_for_prompt: Dict[str, Any],
    later_available_tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    return STAGE0_SAMPLE_BRIEF_PROMPT.format(
        subtask_json=_jd(_build_subtask_json(subtask)),
        template_ops_json=_jd(template_ops),
        later_available_tools_json=_jd(later_available_tools or []),
        raw_sample_json=_jd(sample_for_prompt),
    )


def _run_stage0_sample_brief(
    subtask: Dict[str, Any],
    template_ops: List[Dict[str, Any]],
    sample_for_prompt: Dict[str, Any],
    model: str,
    image_files: Optional[List[str]] = None,
    later_available_tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a sample-level design brief. Fallback is conservative and non-blocking."""
    prompt = _build_stage0_prompt(
        subtask=subtask,
        template_ops=template_ops,
        sample_for_prompt=sample_for_prompt,
        later_available_tools=later_available_tools,
    )
    try:
        resp = llm_call_json(
            system_prompt="You write concise sample-level design briefs for benchmark construction.",
            user_prompt=prompt,
            images=image_files if image_files else None,
            model=model,
            extra_create_params={
                "custom_llm_provider": "openai",
                "timeout": LLM_TIMEOUT_S,
                "request_timeout": LLM_TIMEOUT_S,
            },
        )
        if not resp.get("ok"):
            raise ValueError(str(resp.get("error")))
        brief = resp.get("json")
        if isinstance(brief, list):
            brief = brief[0] if brief else {}
        if isinstance(brief, dict) and brief:
            return _normalize_sample_design_brief(brief)
    except Exception as e:
        return _normalize_sample_design_brief({
            "fit": "weak",
            "core_facts": ["Preserve factual invariants from the original sample."],
            "design_goal": str(subtask.get("description") or subtask.get("name") or "Construct a schema-aligned benchmark sample."),
            "reasoning_hops": ["Identify relevant evidence.", "Apply the subtask predicate."],
            "difficulty_plan": "Keep distractors plausible and evidence-falsifiable when applicable.",
            "leakage_guards": ["Do not leak final.output.answer in final.input."],
            "first_focus": "Extract the minimal evidence needed for the subtask.",
        })

# =========================
# Stage1: Plan next step
# =========================
# Stage1: Adaptive Step Controller
# =========================
STAGE1_NEXT_STEP_PROMPT = r"""
====================================================
YOUR ROLE
====================================================
You are Stage1, the Adaptive Step Controller in a benchmark construction system.

Stage0 already wrote a sample-level design brief. Do NOT redesign the sample from scratch.
Your job is to compare the brief with the current working state, choose exactly ONE next executable step, and handle retry/done/skip decisions.

Stage2 is the executor. You only plan the next step.

====================================================
INPUTS
====================================================
- subtask_json
{subtask_json}

- sample_design_brief_json: sample-level design intent from Stage0. Treat it as guidance, not a rigid step list.
{sample_design_brief_json}

- template_ops_json: remaining dataset-level template steps. They are a loose skeleton.
{template_ops_json}

- tail_ops_json: allowed finishing/repair operations after template work is complete.
{tail_ops_json}

- available_injected_tools_json: optional dynamic PURE tools that may be inserted between template steps. If this list is empty, do NOT invent injected tools.
{available_injected_tools_json}

- original_sample_json
{original_sample_json}

- current_state_json: cumulative state (tool_memory.*, interim.*, final.*). If the last step failed/incomplete, failed changes were rolled back.
{current_state_json}

  current_state_json may include tool_memory.summary_only=true. This is the planning view of retained PURE tool resources. It shows stable memory base paths and concrete resource_paths like `tool_memory.web_search.identify_artist.output.answer` that later LLM steps can copy into resource_fields. PURE outputs are preserved in tool_memory; do not duplicate them into interim just to remember them.

- executed_steps_json: template steps already executed, excluding the last successful step when it is also shown as last_step_json.
{executed_steps_json}

- last_step_json
{last_step_json}

- last_step_result_json
{last_step_result_json}

- last_step_retry_count
{last_step_retry_count}

====================================================
CONTROLLER POLICY
====================================================
1. Align with the brief
- Identify what parts of sample_design_brief are already satisfied by current_state_json.
- Identify the most important missing piece blocking a valid final sample.
- Preserve the brief's core_facts, design_goal, reasoning_hops, difficulty_plan, and leakage_guards unless execution evidence shows they are impossible.
- Every planned step should either build evidence needed by a reasoning_hop, construct confusable-but-falsifiable answer choices, or repair leakage/schema issues.
- If a step would make the final item answerable by one obvious cue, lexical overlap, or a single perspective, revise the step before using it.
- In brief_alignment, satisfied/missing/risks are diagnostic notes. **Non-empty missing or risks does NOT by itself require another step:** if the hard gates in section 5 already pass (schema, answer_type, no leakage, one correct choice when applicable, item is evidence-grounded), you may declare done even when missing lists residual polish or risks list hypothetical edge cases. Reserve further steps for **concrete** defects (wrong answer, leakage, schema break, trivially text-solvable choice set, etc.), not for exhaustive perfection.

2. Retry policy
- Retry the previous step only when last_step_result_json is "fail" or "incomplete", or when a successful result violates the brief/subtask constraints.
- Retry only if last_step_retry_count < 3.
- Retry guidance must name failed_or_missing_paths, root_cause, and concrete_fix.

3. Step selection
- Prefer the next useful template step when template_ops_json is non-empty.
- Tail ops are for final repair/formatting after template steps are complete.
- You may choose kind="injected" only if available_injected_tools_json contains the tool name and the brief/current_state shows a concrete evidence gap. Injected PURE steps must use step_index=-1 and target one stable memory base path: `tool_memory.<tool_name>.<operation_key>`.
- If no injected tools are available, do not output kind="injected".
- For a web_search PURE step, params.pure_tool_param should briefly describe the concrete search intent for THIS sample: what **factual** information needs to be retrieved, which resource values **and/or which sample image path(s)** anchor the search (when images exist and visual grounding improves fact retrieval, **require** image-based anchoring in the intent), and how the evidence will be used. Do not write a generic task description.
- For any PURE step, target_fields must be `["tool_memory.<tool_name>.<operation_key>"]` except cleanup-only steps. Do NOT use interim.* or final.* as PURE targets.
- If a template PURE step already declares a `tool_memory.*` target_fields path, execute it as written.
- For later LLM steps, include the exact `tool_memory.*.input.*` or `tool_memory.*.output.*` resource path when retained PURE tool input/output is needed. Use the exact resource_paths shown in current_state_json.tool_memory.records.
- If an LLM step's notes or guidance rely on a PURE output but resource_fields lacks the corresponding tool_memory path, add that path before executing the LLM step.

4. Guidance requirements
- For LLM steps, params.guidance must be concrete and state-aware: cite exact resource paths, target paths, and the intended transformation for THIS sample.
- Keep large source content out of guidance. Refer to paths/indices/short identifiers rather than pasting long passages.
- The guidance should explain how the step advances the Stage0 brief.
- When constructing final.input/final.output or options, guidance must name which reasoning_hops the evaluated model will need to solve the final item.
- reasoning_hops are INTERNAL design constraints, not user-facing text. Do NOT ask Stage2 to serialize hops into the question as explicit instructions (e.g., "Step 1 ... Step 2 ...", "first infer ... then choose ...").
- For final.input.question, prefer one concise neutral interrogative sentence that tests the intended capability without exposing the intended reasoning chain.
- For choice tasks, require wrong options to be locally plausible and differ by small evidence-dependent changes, not absurd distractors.

5. Final acceptance
- Set done=true only when final.input and final.output satisfy the subtask schema, answer_type, no-leakage rules, and the brief's difficulty/multi-hop intent (multi-hop = answer not trivially direct-read; includes perception + further reasoning, not only formal chains).
- final.input must not reveal final.output.answer.
- For multimodal items, the question must not restate decisive visual/audio/structured evidence.
- For choice items, exactly one option must be correct and distractors must be plausible but evidence-falsifiable.
- Do NOT declare done if the item can be solved from question/options alone by surface plausibility, keyword matching, or one evidence cue.
- Prefer at least two meaningful evidence checks when the subtask allows; if you are **just short** of that bar but the item is already non-trivial, grounded, and passes leakage/schema/choice correctness, you **may** declare done instead of opening another repair loop. Schedule tail repair only when a **clear** quality gate still fails (e.g. obvious text-only solvability, broken schema, or decisive leakage)—not to chase minor wording asymmetry or hypothetical reviewer nitpicks already noted under brief_alignment.risks.

Answer-type constraints:
{answer_type_specific_rules}

====================================================
OUTPUT FORMAT
====================================================
Return EXACTLY ONE JSON object:

{{
  "idx": <int>,
  "decision": "use" | "skip" | "done",
  "reason": "short explanation",
  "brief_alignment": {{
    "satisfied": ["..."],
    "missing": ["..."],
    "risks": ["..."]
  }},
  "done": true | false,
  "retry": true | false,
  "next_step": {{
    "step_name": "<step name>",
    "step_index": <int>,
    "kind": "template" | "tail" | "injected",
    "tool_type": "LLM" | "PURE",
    "resource_fields": ["..."],
    "deleted_interim_fields": ["..."],
    "target_fields": ["..."],   // PURE uses ["tool_memory.<tool_name>.<operation_key>"]; cleanup-only PURE may use []
    "params": {{
      "guidance": "..."          // for LLM steps
      // OR
      "pure_tool_param": {{}}    // for PURE steps
    }}
  }} | null
}}

Strict rules:
- If decision="done": done=true, retry=false, next_step=null.
- If decision="skip": done=false, retry=false, next_step=null.
- If retry=true: decision="use", done=false, next_step must retry the previous logical step.
- If decision="use" and retry=false: next_step must be non-null.
- For template steps, step_index must be the _original_index from template_ops_json.
- For tail or injected steps, step_index must be -1.
- For PURE steps, target_fields must be one stable `tool_memory.<tool_name>.<operation_key>` base path, except cleanup-only operations with no resources/targets. The executor stores tool_args under `.input` and the backend result under `.output`; later LLM steps should read those values via tool_memory resource_fields and write any needed user-visible artifacts to final.*.
- brief_alignment.satisfied may still be rich while missing/risks list small or speculative items; that is OK for decision="done" when section 5 is satisfied. Keep missing/risks honest but short—do not pad them to justify unnecessary steps.
"""

STAGE1_ANSWER_TYPE_CHOICE_RULES = r"""
- For answer_type == "choice", enforce all of the following:
  **(A) Correctness under the question predicate:**
  - Determine correctness using the predicate implied by subtask_json.description and final.input.*.
  - Exactly one option must be semantically correct; wrong options must be strictly incorrect but close enough to be confusable.
  - The predicate should require non-trivial use of the provided input (causal/temporal/multi-source/constraint integration, cross-modal combination, or perceptual readout + further judgment). "Multi-hop" means the answer is not obtainable in one trivial glance; it does **not** require a formal step-by-step proof. Using **factual** world knowledge together with input-grounded evidence is allowed when the correct option still hinges on the input, not on external facts alone.

  **(B) Text-only non-solvability (CRITICAL):**
  - A strong text-only model that sees only final.input.question + options should NOT reliably exceed near-random.
  - If simple lexical overlap, named-entity uniqueness, or style/length asymmetry makes one option noticeably easier to pick, you MUST revise stem/options before proceeding.
  - Avoid options where one is globally "correct-sounding" without checking evidence.
  - Avoid specific personal names, exact years, or narrow domain labels that make one option stand out without evidence.
  - If question+options alone already make one option clearly more plausible, you MUST revise stem/options (usually: shorten and neutralize stem, reduce option deltas, remove uniquely identifying labels).

  **(C) Gold hypothesis + minimal perturbations:**
  - First derive one gold hypothesis H for THIS sample from evidence.
  - Then create exactly 3 minimally perturbed variants of H:
    - each wrong option changes only one or two atomic details,
    - all options remain globally plausible in isolation,
    - all options stay in the same semantic frame and similar abstraction level.
  - Differences should be subtle and evidence-dependent, not four unrelated stories.
  - Wrong options should be partially plausible in isolation but decisively falsifiable by concrete evidence checks.

  **(D) Option surface form (ATOMIC / SHORT / NON-EXPLANATORY):**
  - Use either:
    1) label mode (short category/label), or
    2) value mode (short value/expression).
  - Keep options brief and symmetric (prefer <= 8-12 words when textual).
  - Do NOT use explanatory connectors ("because", "so that", "in order to", "which means").
  - Do NOT embed long causal chains, historical background, or step-by-step reasoning into option text.
  - Avoid options that stand out by length, style, or specificity in a way that leaks correctness.

  **(E) Mandatory guidance payload in params.guidance:**
  - You MUST explicitly include:
    - the gold hypothesis H,
    - 3 minimal perturbations (which atomic detail is changed in each wrong option),
    - option mode (label/value) and exact text for A/B/C/D,
    - which option letter is correct,
    - concrete evidence checks required to eliminate each wrong option
      (with dot-paths / indices / timestamps / rows / regions where applicable),
    - why question+options alone should remain near-random for text-only guessing.
  - Guidance should make "mutual confusability + strict incorrectness" explicit for each wrong option.

  **(F) Grounding discipline:**
  - Labels/categories may be used, but must be grounded in actual fields from current_state_json/original_sample_json.
  - Do NOT invent unsupported labels or distinctions not derivable from the provided evidence.

  **(G) Examples of good surface shape (style only):**
  - "Major third" / "Diminished fifth" / "Minor seventh" / "Diminished sixth"
  - "0" / "0.2142" / "0.3571" / "0.5"
  - "Greedy strategy" / "Dynamic programming" / "Backtracking search" / "Random sampling"

  **(H) Global finishing behavior for choice tasks:**
  - In the finishing phase, re-check that exactly one option is semantically correct under the predicate.
  - If text-only non-solvability or option confusability **clearly** fails (e.g. one option obviously stands out without evidence, or stem leaks the answer), schedule tail op(s) to repair stem/options before declaring done. If the set is already plausibly confusable and only minor asymmetry remains, you may declare done without another tail pass.
"""

STAGE1_ANSWER_TYPE_NON_CHOICE_RULES = r"""
- For answer_type != "choice":
  **(A) Output-type strictness:**
  - Do NOT create or reason about A/B/C/D options.
  - Focus on strict alignment with the requested output type:
    - binary -> yes/no (or equivalent binary form),
    - label -> one concise label,
    - span -> short extractive span from provided context (<= 5 words when required),
    - structured -> conform to the required schema.

  **(B) Capability fidelity and evidence dependence (CRITICAL):**
  - The final question must test the capability defined in subtask_json, not a shortcut.
  - The answer must not be solvable from **external factual knowledge alone** while ignoring the input. The evaluated model **may** combine input (including perception over image/audio/structured fields) with **factual** world knowledge; the benchmark is invalid only if the labeled answer follows from general knowledge without using the provided input modalities/content.
  - If image/audio/table/code/log fields are present, do NOT restate decisive evidence in the stem.

  **(C) Non-trivial reasoning requirement:**
  - Preserve or increase reasoning difficulty; avoid reducing the task to single-cue lookup.
  - Require at least two evidence-dependent checks or constraints when the subtask permits (e.g. perceptual readout + application of a rule, or two non-redundant cues). Avoid **only** one trivial surface read that already states the answer. Perceptual identification plus a separate factual or relational step counts as non-trivial.
  - "Multi-hop" here means the solution path is not a single obvious read-off; it is **not** restricted to arithmetic or formal deduction. When the subtask implies multi-hop reasoning, the question should require combining multiple clues/constraints or perception-plus-reasoning.
  - Avoid stems that leak the conclusion through wording, labels, or premise framing.

  **(D) Leakage control and stem minimalism:**
  - final.input may contain only information needed to ask the question; it must not encode or imply final.output.answer.
  - Keep the question stem short, neutral, and interrogative.
  - Do NOT include decisive details that directly reveal the answer.

  **(E) Mandatory guidance payload in params.guidance:**
  - Include concrete read paths and write paths (dot paths),
  - deterministic transformation procedure for THIS sample,
  - explicit mapping from evidence to expected output field shape,
  - verification notes for alignment (schema + answer_type + no leakage).
  - Guidance must be state-aware (paths, indices, counts, concrete values), not generic prose.

  **(F) Global finishing behavior for non-choice tasks:**
  - In the finishing phase, re-check strict answer_type conformance before declaring done:
    - binary -> exactly one binary-form answer,
    - label -> one concise label only,
    - span -> extractive short span (no paraphrased explanation),
    - structured -> valid schema-compatible object.
  - If output shape, answer form, or **decisive** leakage is not compliant, schedule tail op(s) to repair final.input/final.output before declaring done. Tiny formatting nits alone do not require another tail if the sample already meets the subtask and section 5.
"""


def _get_stage1_answer_type_rules(subtask: Dict[str, Any]) -> str:
    answer_type = str((subtask or {}).get("answer_type") or "").strip().lower()
    if answer_type == "choice":
        return STAGE1_ANSWER_TYPE_CHOICE_RULES
    return STAGE1_ANSWER_TYPE_NON_CHOICE_RULES


def _build_stage1_prompt(
    subtask: Dict[str, Any],
    sample_design_brief: Dict[str, Any],
    remaining_template_ops: List[Dict[str, Any]],
    tail_ops_allowed: List[str],
    available_injected_tools: List[Dict[str, Any]],
    sample_for_prompt: Dict[str, Any],
    current_state: Dict[str, Any],
    executed_steps_list: List[Dict[str, Any]],
    last_step_json: Optional[Dict[str, Any]],
    last_step_result: Optional[Dict[str, Any]],
    last_step_retry_count: int,
) -> str:
    return STAGE1_NEXT_STEP_PROMPT.format(
        subtask_json=_jd(_build_subtask_json(subtask)),
        sample_design_brief_json=_jd(sample_design_brief),
        template_ops_json=_jd(remaining_template_ops),
        tail_ops_json=_jd(tail_ops_allowed),
        available_injected_tools_json=_jd(available_injected_tools),
        original_sample_json=_jd(sample_for_prompt),
        current_state_json=_jd(current_state),
        executed_steps_json=_jd(executed_steps_list),
        last_step_json=_jd(last_step_json) if last_step_json else "null",
        last_step_result_json=_jd(last_step_result) if last_step_result else "null",
        last_step_retry_count=last_step_retry_count,
        answer_type_specific_rules=_get_stage1_answer_type_rules(subtask),
    )


# =========================
# Stage2: Execute single step
# =========================

STAGE2_SAMPLE_PROMPT = r"""
You are the Sample-Level Executor (Stage2). 
You apply exactly ONE transformation step to a working sample.

You receive:
- subtask_json: high-level description of the evaluation subtask, which defines the evaluation target.
- step_json: specification of this single operation, including:
  • step_name / operation
  • params
  • resource_fields: the list of field paths you may read
  • target_fields: the list of field paths you must update
- resources: JSON object containing ONLY the fields listed in resource_fields
  (expanded into nested structure).
  It may include tool_memory.* only when step.resource_fields explicitly requests retained PURE tool memory.

**YOUR OUTPUT MUST be raw JSON only (no markdown/code fences/explanations), and the field "this_step_result" MUST be present**

Inputs:

subtask_json: {subtask_json}
step_json: {step_json}
resources: {resources}

====================================================
Input field semantics (if contained in resources)
====================================================
1. **original.**
   - Ground truth reference
   - Provides factual constraints

2. **interim.**
    - Internal scaffolding fields
    - used to store intermediate representations
    - PURE tool outputs should not appear here. Use tool_memory.* for retained tool input/output.
    - You may use retrieved evidence to construct factual context, choices, labels, answers, or other fields required by this step, but only to the extent supported by that evidence.
    - If retrieved evidence is insufficient, conflicting, or uncertain, keep the output conservative and avoid unsupported claims.

3. **tool_memory.**
   - Hidden retained PURE tool memory. The evaluated model never sees it.
   - It preserves PURE tool inputs and outputs as stable resources, such as `tool_memory.web_search.identify_artist.input.query` and `tool_memory.web_search.identify_artist.output.answer`.
   - Use it only when included in resources through an explicit `tool_memory.*` resource_fields path. If it is not in resources, you must not assume or invent it.
   - Treat it like original.* for write permissions: read-only in Stage2.
   - If the final benchmark needs a tool artifact visible to the evaluated model (for example an audio/image path), map or rewrite the relevant tool_memory value into final.input.* in this step.
   - Do not copy hidden evidence into final.input unless the subtask explicitly requires that content to be visible to the evaluated model.
   - Treat retained web_search results as retrieved evidence, not hidden ground truth; do not add facts beyond what the retained evidence supports.

4. **final.** semantics (evaluation sample)
- final.input.*  = the ONLY content the **model under test** will see as INPUT. It may contain ONLY the information **necessary to ask the question**; it must NOT leak the answer. The model never sees final.output.
  **Multimodal asset names:** local file names, path strings, or URL basename metadata for attached image/audio/video are **not** shown to the evaluated model as hints; they receive the native modalities only. Do not treat obscuring filenames as an anti-leakage requirement—the model cannot rely on filenames.
  **External factual knowledge:** if the task requires combining what is **in** the input (including what is seen/heard/read in multimodal fields) with **factual** world knowledge, that is allowed. The sample is invalid only if the answer could be chosen correctly using external facts **without** using the provided input content.
  **The question (final.input.question or any field that poses the query) must ONLY ask — it must NOT state, echo, or imply the answer** (e.g. forbidden: "…and the answer is X", "Who is Y? — It is Z", "the correct one is A").
  **Do NOT expose the reasoning procedure in the question**: avoid explicit chain-of-thought scaffolding such as "Step 1/Step 2", "first ... then ...", "reason through these hops", or other staged instructions that guide solving.
  **Multi-hop intent from sample_design_brief.reasoning_hops (when that object appears in resources) must be implemented implicitly in evidence/option design, not written verbatim into final.input.question.**
  Prefer short, direct stems (typically one sentence) unless the subtask schema explicitly requires longer context.
  **Nothing else in final.input must reveal or hint at final.output.answer** — no marking the correct choice, no implying by order or wording, no encoding the answer elsewhere. For choice-type: at most 5 options.
  **Audio-specific rule (IMPORTANT):** if final.input contains an audio field (e.g. audio_url), do NOT restate or summarize the audio/dialogue in the question stem. Keep the stem short and refer to "the audio/recording". The model should use the audio itself to answer; long paraphrases often leak the answer.
- final.output.* = ground-truth ANSWER used for evaluation (not visible to the model under test).
- Follow the guidance in step_json.params closely. 

Dot-path semantics for fields, e.g. "final.input.question" means: nested keys:
{{
    "final": {{
        "input": {{
            "question": ...
        }}
    }}
}}
====================================================
Single-step execution semantics
====================================================
To apply this step:

1. Collect resources from resource_fields
2. Apply transformation according to params and step_name, you should:
   - Follow all instructions in params closely.
   - When consuming `web_search` evidence in resources, including under tool_memory.*, use it as retrieved evidence, not hidden ground truth; do not add external facts beyond what the evidence supports.
   - When consuming tool_memory artifacts, decide whether they remain hidden reasoning/evidence or must be transformed into visible final.input fields. Only write visible fields when required by target_fields and the subtask schema.
3. Write outputs to target_fields in this_step_result
   - May create new final.X or interim.X fields
   - Must NOT create or update tool_memory.X fields.
4. Before returning, run a strict self-check:
   - Every path in target_fields is present in this_step_result with correct nested structure.
   - this_step_result contains only interim/final branches that are needed for target_fields.
   - Final output obeys leakage constraints defined above.
   - If this step writes final.input/final.output or answer choices, the result still requires the reasoning_hops described in any provided sample_design_brief (when included in resources).
   - The step must not collapse a multi-hop item into a one-hop lookup, a keyword match, or an obvious option.
====================================================
Clarifying transformation freedoms (IMPORTANT)
====================================================
- Preserving factual invariants does NOT require preserving the surface narrative, wording, etc. The resource sample provides factual and meta constraints, not a target representation.
- Subtask-facing representations may:
    - compress, expand, delete, or reorganize surface content,
    - scenarize events,
    - introduce roles, perspectives, or latent structure,
    - promote events or entities as evaluation targets,
    - insert evaluation interfaces (choice, label, span),
    - add reasoning or contrastive framing,
    as long as factual invariants remain true and no unsupported external facts are introduced.
- External facts are allowed ONLY when they are present in original data/meta or in resources as retrieved evidence from a prior `web_search` step.
- Synthetic additions are ALLOWED; hallucination is narrowly defined as altering factual invariants or importing unsupported external facts.
- Operations are atomic in semantics, not in surface footprint. A single operation may substantially rewrite or restructure the surface to satisfy the subtask.
- Deletion and Synthetic internal scaffolding is allowed (meta is a generative substrate).
====================================================
DIFFICULTY PRESERVATION
====================================================
- If resources include interim.sample_design_brief (the Stage0 brief), follow its core_facts, reasoning_hops, difficulty_plan, and leakage_guards.
- Make difficulty come from evidence integration, source reliability, temporal/causal consistency, cross-modal combination, or subtle option falsification.
- Do NOT make the final question harder by using vague wording, missing information, or unsupported facts.
- Do NOT explain the full reasoning chain in final.input. The model under test must infer it.
- For multiple-choice outputs, wrong options should be close variants of the correct hypothesis and falsifiable by specific evidence, not obviously silly or unrelated.
====================================================
Length and expansion policy (IMPORTANT)
====================================================
Default policy: stay concise and only write the minimum content needed for this step.
Only expand when step_json.params explicitly requests "long_context" or equivalent expansion guidance.

1) Controlled expansion[Very important]:
   - Expansion of resource content is ALLOWED but must be moderate and purposeful.
   - Expansion factor should generally be ≤ 3x the resource content length. 
   - Never exceed 4x even under expansion-related params. 

2) Default size targets (soft constraints):
   - Narrative / factual context: usually concise; use ~100-500 words only when required by params.
   - Dialog / multi-turn context: usually concise; use ~4-8 short turns (total ~100-500 words) only when required by params.

====================================================
EXECUTION PURPOSE (CRITICAL)
====================================================
Stage2 is not editing or repairing the resource dataset sample.
It is progressively constructing a new QA **evaluation sample** for the target subtask: the **model under test will see only final.input**; anything in final.input that leaks final.output.answer — including **stating or implying the answer in the question text** (e.g. "…and the answer is X", "Who is Y? — It is Z") — invalidates the evaluation.
====================================================
Output format (STRICT JSON)
====================================================

Return format (only JSON object):

{{
  "this_step_result": {{ 
    "interim": {{ ...updated fields... }},
    "final":   {{ ...updated fields... }}   
  }}
}}

Rules[IMPORTANT]:
- Inside "this_step_result", you MUST cover ALL paths listed in "target_fields" using nested structure.
- Inside "this_step_result", you MUST NOT add any other keys beyond "interim" and/or "final".
- If target_fields contains tool_memory.*, that step is invalid for Stage2; Stage2 can only write interim.* and final.*.
- At the top level of the JSON, there MUST be exactly one key: "this_step_result".
- **YOUR OUTPUT MUST be raw JSON only (no markdown/code fences/explanations), and "this_step_result" MUST be present**

Minimal valid output examples:
1) only final path update:
{{
  "this_step_result": {{
    "final": {{
      "input": {{
        "question": "..."
      }}
    }}
  }}
}}
2) interim + final update:
{{
  "this_step_result": {{
    "interim": {{
      "evidence_summary": "..."
    }},
    "final": {{
      "output": {{
        "answer": "..."
      }}
    }}
  }}
}}
"""


def _run_llm_step(
    subtask: Dict[str, Any],
    step: Dict[str, Any],
    resources: Dict[str, Any],
    model: str = "gpt-5.1",
    dataset_json_path: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any], str]:
    """Execute a single LLM step. Detect image paths in resources and pass them as multimodal input."""
    user_prompt = STAGE2_SAMPLE_PROMPT.format(
        subtask_json=_jd(_build_subtask_json_for_stage2_llm(subtask)),
        step_json=_jd(step),
        resources=_jd(resources),
    )

    image_files = _collect_image_paths_from_resources(resources)
    if image_files:
        image_files = resolve_image_paths(
            image_files,
            dataset_json_path=dataset_json_path,
            dataset_id=dataset_id,
        )

    retry_times = 3
    last_raw_text_json: Any = {}

    for attempt in range(1, retry_times + 1):
        try:
            resp = llm_call_json(
                system_prompt="",
                user_prompt=user_prompt,
                model=model,
                images=image_files if image_files else None,
                extra_create_params={
                    "custom_llm_provider": "openai",
                    # LiteLLM forwards this to underlying client (OpenAI etc.).
                    # We include both keys for compatibility across providers.
                    "timeout": LLM_TIMEOUT_S,
                    "request_timeout": LLM_TIMEOUT_S,
                },
            )
            if not resp.get("ok"):
                last_raw_text_json = {}
                # Retry transient failures instead of failing immediately.
                # Small backoff to avoid hammering the endpoint.
                time.sleep(min(0.5 * attempt, 2.0))
                continue

            out = resp.get("json") or {}
            note = f"llm_attempt={attempt}/{retry_times}"

            this_step_result = out.get("this_step_result")
            if not isinstance(this_step_result, dict):
                raw_text = resp.get("raw_text") or ""
                try:
                    last_raw_text_json = json.loads(raw_text) if raw_text else {}
                except Exception:
                    last_raw_text_json = {}

                if isinstance(last_raw_text_json, dict):
                    this_step_result = last_raw_text_json.get("this_step_result")

            if not isinstance(this_step_result, dict):
                continue

            return True, this_step_result, note

        except Exception:
            continue

    return (
        False,
        {},
        f"this_step_result parse failed, last_raw_json={last_raw_text_json}",
    )


def _normalize_final_export_for_sample(
    subtask: Dict[str, Any],
    dataset_id: Optional[str],
    idx_int: int,
    final_obj: Any,
) -> Dict[str, Any]:
    """Deep-copy ``final`` and apply deterministic choice shuffle when ``answer_type`` is choice."""
    final_obj = copy.deepcopy(final_obj) if isinstance(final_obj, dict) else {}
    try:
        answer_type = str(subtask.get("answer_type") or "").lower()
        if answer_type == "choice":
            inp = final_obj.get("input") or {}
            outp = final_obj.get("output") or {}
            q = inp.get("question")
            a = outp.get("answer")
            has_audio = isinstance(inp, dict) and isinstance(inp.get("audio_url"), str) and bool(inp.get("audio_url"))
            if isinstance(q, str) and isinstance(a, str):
                seed = stable_seed(subtask.get("id"), dataset_id, idx_int, "choice_shuffle")
                patched = normalize_and_shuffle_choice_question(q, a, seed, has_audio=has_audio)
                if patched:
                    inp["question"] = patched["question"]
                    outp["answer"] = patched["answer"]
                    final_obj["input"] = inp
                    final_obj["output"] = outp
    except Exception:
        pass
    return final_obj


def _final_has_body_for_export(final_obj: Any) -> bool:
    if not isinstance(final_obj, dict):
        return False
    return bool(final_obj.get("input")) or bool(final_obj.get("output"))


def _build_partial_export_sample(
    *,
    original_sample: Dict[str, Any],
    subtask: Dict[str, Any],
    dataset_id: Optional[str],
    idx_int: int,
    current_state: Dict[str, Any],
    pure_tool_messages: List[Any],
    exit_reason: str,
) -> Optional[Dict[str, Any]]:
    """
    Same outer shape as a successful transform sample (original + current.final),
    for downstream verify / debug_transform_batch when the iterative loop exits without done.
    """
    final_src = (current_state or {}).get("final") or {}
    if not _final_has_body_for_export(final_src):
        return None
    final_export = _normalize_final_export_for_sample(subtask, dataset_id, idx_int, final_src)
    sw: Dict[str, Any] = {
        "original": original_sample,
        "current": {"final": final_export},
        "_transform_pipeline_status": "partial",
        "_transform_exit_reason": exit_reason,
    }
    if pure_tool_messages:
        sw["_pure_tool_messages"] = list(pure_tool_messages)
    return sw


def _extract_stage1_trace_fields(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Keep key Stage1 planner fields for step_history debugging."""
    if not isinstance(plan, dict):
        return {}
    brief_alignment = plan.get("brief_alignment")
    if isinstance(brief_alignment, dict):
        brief_alignment = copy.deepcopy(brief_alignment)
    else:
        brief_alignment = {}
    return {
        "stage1_idx": plan.get("idx"),
        "stage1_decision": plan.get("decision"),
        "stage1_reason": plan.get("reason") or "",
        "stage1_done": bool(plan.get("done", False)),
        "stage1_retry": bool(plan.get("retry", False)),
        "stage1_brief_alignment": brief_alignment,
    }


# =========================
# Full per-sample processing flow (with retry support)
# =========================

def _process_one_sample_iterative(
    idx_int: int,
    raw_sample: Dict[str, Any],
    subtask: Dict[str, Any],
    template_ops: List[Dict[str, Any]],
    tail_ops_allowed: List[str],
    pure_tool_registry: Dict[str, Any],
    stage0_model: str,
    stage1_model: str,
    stage2_model: str,
    image_files: List[str],
    dataset_json_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    sample_design_brief_cache_path: Optional[str] = None,
    subtask_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    trace_cb: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Full iterative pipeline for a single sample:
    - Loop: Stage1 plan -> Stage2 execute -> Stage1 evaluate -> decide whether to retry
    - Retry support (up to 3 attempts per step)
    """
    # Initialize state
    original_sample = copy.deepcopy(raw_sample)
    current_state: Dict[str, Any] = {"interim": {}, "final": {}}
    executed_steps: List[int] = []  # Executed step indices (in execution order)
    step_history: List[Dict[str, Any]] = []  # Step execution history

    if trace_cb:
        try:
            trace_cb(idx_int, f"start subtask={subtask_id} dataset={dataset_id}")
        except Exception:
            pass

    # Retry-related state (may be overwritten by cache)
    last_step_plan: Optional[Dict[str, Any]] = None
    last_step_result: Optional[Dict[str, Any]] = None
    last_step_retry_count: int = 0
    sample_design_brief: Dict[str, Any] = {}
    
    # Collect pure tool messages
    pure_tool_messages: List[Dict[str, Any]] = []

    # Load cache if present: sample_iterative/{subtask}.json -> sample_iterative_states[dataset::idx]
    if cache_path:
        cached = _load_sample_iterative_state(
            cache_path,
            dataset_id=dataset_id,
            idx_int=idx_int,
        )
        if cached:
            try:
                current_state = cached.get("current_state", current_state)
                executed_steps = cached.get("executed_steps", executed_steps)
                step_history = cached.get("step_history", step_history)
                sample_design_brief = cached.get("sample_design_brief", sample_design_brief)
                last_step_plan = cached.get("last_step_plan", last_step_plan)
                last_step_result = cached.get("last_step_result", last_step_result)
                last_step_retry_count = cached.get("last_step_retry_count", last_step_retry_count)
                pure_tool_messages = cached.get("pure_tool_messages", pure_tool_messages)
                if not isinstance(pure_tool_messages, list):
                    pure_tool_messages = []
            except Exception:
                pass  # Cache load failed; continue normal flow

    sample_for_brief = copy.deepcopy(original_sample)
    sample_for_brief["idx"] = idx_int
    later_available_tools_for_stage0 = _build_later_available_tools_for_stage0(pure_tool_registry)
    if not sample_design_brief:
        sample_design_brief = _load_sample_design_brief_cache(
            sample_design_brief_cache_path,
            dataset_id=dataset_id,
            idx_int=idx_int,
        )
        if sample_design_brief and trace_cb:
            try:
                trace_cb(idx_int, "stage0 loaded sample_design_brief cache")
            except Exception:
                pass
    if not sample_design_brief:
        if trace_cb:
            try:
                trace_cb(idx_int, "stage0 building sample_design_brief")
            except Exception:
                pass
        sample_design_brief = _run_stage0_sample_brief(
            subtask=subtask,
            template_ops=template_ops,
            sample_for_prompt=sample_for_brief,
            model=stage0_model,
            image_files=image_files,
            later_available_tools=later_available_tools_for_stage0,
        )
        _save_sample_design_brief_cache(
            sample_design_brief_cache_path,
            subtask_id=subtask_id,
            dataset_id=dataset_id,
            idx_int=idx_int,
            sample_design_brief=sample_design_brief,
        )

    # Brief is fixed for the sample; keep it only at top-level (cache + Stage1 prompt), not mirrored in interim.
    intr = current_state.get("interim")
    if isinstance(intr, dict) and "sample_design_brief" in intr:
        legacy = intr.get("sample_design_brief")
        if not sample_design_brief and isinstance(legacy, dict) and legacy:
            sample_design_brief = _normalize_sample_design_brief(legacy)
        intr.pop("sample_design_brief", None)

    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Stage1: Plan next step (or retry)
        if trace_cb:
            try:
                trace_cb(
                    idx_int,
                    f"stage1 planning iter={iteration} retry_count={last_step_retry_count} last_step={((last_step_plan or {}).get('step_name'))}",
                )
            except Exception:
                pass
        sample_for_prompt = copy.deepcopy(original_sample)
        sample_for_prompt["idx"] = idx_int

        # Build last_step_json (full info from the last attempt, including params; may be success or failure)
        last_step_json = None
        if last_step_plan:
            last_step_json = {
                "step_index": last_step_plan.get("step_index"),
                "step_name": last_step_plan.get("step_name"),
                "kind": last_step_plan.get("kind", "template"),
                "tool_type": last_step_plan.get("tool_type"),
                "resource_fields": last_step_plan.get("resource_fields", []),
                "deleted_interim_fields": last_step_plan.get("deleted_interim_fields", []),
                "target_fields": last_step_plan.get("target_fields", []),
                "params": last_step_plan.get("params", {}),
            }

        # Build executed-steps list (no params, in execution order)
        # Exclude from executed_steps_list only when the last attempt is the last successful execution
        # Otherwise (e.g. most recent attempt failed), do not drop the last successful step from the prompt.
        executed_steps_list = []
        executed_step_indices_set = set(executed_steps)
        last_success_step_index = executed_steps[-1] if executed_steps else None
        exclude_last_success = (
            last_success_step_index is not None
            and isinstance(last_step_plan, dict)
            and last_step_plan.get("step_index") == last_success_step_index
        )
        
        for step_idx in executed_steps:
            if exclude_last_success and step_idx == last_success_step_index:
                continue  # avoid duplication with last_step_json (successful last step)
            if step_idx < len(template_ops):
                step = template_ops[step_idx]
                # Full step info (excluding params)
                executed_step_info = {
                    "step_index": step_idx,
                    "step_name": step.get("step_name") or step.get("operation"),
                    "tool_type": step.get("tool_type"),
                    "resource_fields": step.get("resource_fields", []),
                    "deleted_interim_fields": step.get("deleted_interim_fields", []),
                    "target_fields": step.get("target_fields", []),
                }
                executed_steps_list.append(executed_step_info)

        # Build remaining template_ops (exclude executed), with original indices
        remaining_template_ops = []
        for idx, step in enumerate(template_ops):
            if idx not in executed_step_indices_set:
                step_with_index = copy.deepcopy(step)
                step_with_index["_original_index"] = idx  # Attach original index
                remaining_template_ops.append(step_with_index)

        # Build prompt
        available_injected_tools = [
            {
                "name": name,
                "description": getattr(spec, "description", ""),
                "param_schema": getattr(spec, "param_schema", {}),
                "return_schema": getattr(spec, "return_schema", {}),
            }
            for name, spec in (pure_tool_registry or {}).items()
            if str(name).startswith("web_") or str(name).endswith("_evidence")
        ]
        current_state_for_prompt = _current_state_for_stage1_prompt(current_state)
        user_prompt = _build_stage1_prompt(
            subtask=subtask,
            sample_design_brief=sample_design_brief,
            remaining_template_ops=remaining_template_ops,
            tail_ops_allowed=tail_ops_allowed,
            available_injected_tools=available_injected_tools,
            sample_for_prompt=sample_for_prompt,
            current_state=current_state_for_prompt,
            executed_steps_list=executed_steps_list,
            last_step_json=last_step_json,
            last_step_result=last_step_result,
            last_step_retry_count=last_step_retry_count,
        )

        # Call Stage1 LLM
        if image_files:
            resp = llm_call_json(
                system_prompt="You are the Sample-Level Planner (Iterative Mode) in a benchmark construction system.",
                user_prompt=user_prompt,
                images=image_files,
                model=stage1_model,
                extra_create_params={
                    "custom_llm_provider": "openai",
                    "timeout": LLM_TIMEOUT_S,
                    "request_timeout": LLM_TIMEOUT_S,
                },
            )
        else:
            resp = llm_call_json(
                system_prompt="You are the Sample-Level Planner (Iterative Mode) in a benchmark construction system.",
                user_prompt=user_prompt,
                model=stage1_model,
                extra_create_params={
                    "custom_llm_provider": "openai",
                    "timeout": LLM_TIMEOUT_S,
                    "request_timeout": LLM_TIMEOUT_S,
                },
            )

        if not resp.get("ok"):
            return {
                "idx": idx_int,
                "status": "fail",
                "reason": f"stage1_planning_failed: {resp.get('error')}",
                "sample": None,
            }

        plan = resp.get("json")

        if isinstance(plan, list):
            plan = plan[0] if plan else {}
        if not isinstance(plan, dict):
            return {
                "idx": idx_int,
                "status": "fail",
                "reason": "stage1_invalid_plan",
                "sample": None,
            }



        decision = (plan.get("decision") or "use").lower()
        retry_flag = plan.get("retry", False)
        done_flag = plan.get("done", False)
        stage1_trace_fields = _extract_stage1_trace_fields(plan)

        if trace_cb:
            try:
                ns = plan.get("next_step") if isinstance(plan, dict) else None
                ns_name = ns.get("step_name") if isinstance(ns, dict) else None
                trace_cb(
                    idx_int,
                    f"stage1 plan iter={iteration} decision={decision} retry={bool(retry_flag)} done={bool(done_flag)} next={ns_name}",
                )
            except Exception:
                pass
        
        if decision == "skip":
            return {
                "idx": idx_int,
                "status": "skip",
                "reason": plan.get("reason") or "",
                "sample": None,
            }

        if done_flag:
            # Keep only original, current.final, _pure_tool_messages
            final_obj = _normalize_final_export_for_sample(
                subtask, dataset_id, idx_int, current_state.get("final", {}) or {}
            )
            current_state["final"] = final_obj

            sample_wrapper = {
                "original": original_sample,
                "current": {
                    "final": final_obj,
                },
            }
            if pure_tool_messages:
                sample_wrapper["_pure_tool_messages"] = pure_tool_messages
            return {
                "idx": idx_int,
                "status": "ok",
                "reason": "",
                "sample": sample_wrapper,
            }

        # -----------------------
        # Decide: retry vs new step
        # -----------------------
        is_retry = plan.get("retry", False)
        actually_retry_step = False  # Set True only when entering the retry branch
        retry_base_state = "current"

        if is_retry and last_step_retry_count < 3:
            actually_retry_step = True

            # Stage1 chose to retry the previous step: read from last_step_json (full info including params)
            # If last_step_json is empty (first attempt failed), fall back to last_step_plan
            if last_step_json:
                # Retry step info from last_step_json (full info)
                last_executed_step = last_step_json
                
                # Roll back to previous step state
                if step_history:
                    last_history = step_history[-1]
                    # Roll back current_state only when retrying after Stage2 did not persist successfully
                    # (fail / incomplete): restart from that step's state_before.
                    # If the previous round already applied delta and Stage1 only wants a quality retry,
                    # keep state_after; otherwise tail/template writes to final are lost,
                    # and the next step's state_before log looks like nothing was ever changed.
                    prev_applied_ok = (
                        last_step_result is not None
                        and last_step_result.get("status") == "ok"
                    )
                    if prev_applied_ok:
                        retry_base_state = "state_after"
                        current_state = copy.deepcopy(
                            last_history.get("state_after", current_state)
                        )
                    else:
                        retry_base_state = "state_before"
                        current_state = copy.deepcopy(
                            last_history.get("state_before", current_state)
                        )
                    # Keep the prior history entry's execution semantics; only mark it superseded by retry
                    if step_history:
                        step_history[-1]["superseded_by_retry"] = True
                        step_history[-1]["retry_requested_after_status"] = (
                            last_step_result or {}
                        ).get("status")
                    # Remove from executed_steps (but keep in step_history)
                    if executed_steps and executed_steps[-1] == last_history.get("step_index"):
                        executed_steps.pop()
                else:
                    # Should not reach here in normal flow
                    current_state = {"interim": {}, "final": {}}
                
                # Use improved params from plan if present; otherwise keep original
                next_step = {
                    "step_name": last_executed_step.get("step_name"),
                    "step_index": last_executed_step.get("step_index"),
                    "kind": last_executed_step.get("kind", "template"),
                    "tool_type": last_executed_step.get("tool_type"),
                    "resource_fields": last_executed_step.get("resource_fields", []),
                    "deleted_interim_fields": last_executed_step.get("deleted_interim_fields", []),
                    "target_fields": last_executed_step.get("target_fields", []),
                    "params": plan.get("next_step", {}).get("params", last_executed_step.get("params", {})),
                }
            else:
                # last_step_json is derived from last_step_plan, so both are always
                # None or non-None together — this branch is a safety fallback only.
                return {
                    "idx": idx_int,
                    "status": "fail",
                    "reason": "retry_but_no_step_info",
                    "sample": None,
                }
            
            # If last_step_result is ok, Stage1 voluntarily retries (quality unsatisfied); bump retry count.
            # If last_step_result is a failure, count was already incremented on failure.
            if last_step_result and last_step_result.get("status") == "ok":
                last_step_retry_count += 1
        else:
            # New step (not a retry)
            if is_retry:
                # Stage1 wanted retry but limit reached; proceed to next step
                # Clear last_step_result since we are moving on
                last_step_result = None
            
            next_step = plan.get("next_step")
            if not isinstance(next_step, dict):
                return {
                    "idx": idx_int,
                    "status": "fail",
                    "reason": "plan_missing_next_step",
                    "sample": None,
                }
            if next_step.get("kind") == "injected":
                injected_name = str(next_step.get("step_name") or "").strip()
                if injected_name not in (pure_tool_registry or {}):
                    return {
                        "idx": idx_int,
                        "status": "fail",
                        "reason": f"injected_tool_not_registered[{injected_name}]",
                        "sample": None,
                    }
                next_step["step_index"] = -1
            next_step = _sanitize_pure_tool_op(plan=next_step, pure_tool_registry=pure_tool_registry, model=stage1_model)
            # If Stage1 did not provide step_index, try matching from remaining_template_ops
            if next_step.get("step_index") is None and next_step.get("kind") == "template":
                step_name = next_step.get("step_name")
                # Find matching step in remaining_template_ops
                for rem_step in remaining_template_ops:
                    if rem_step.get("_original_index") is not None:
                        rem_step_name = rem_step.get("step_name") or rem_step.get("operation")
                        if rem_step_name == step_name:
                            next_step["step_index"] = rem_step.get("_original_index")
                            break
                # If still not found, match by step content against original template_ops
                if next_step.get("step_index") is None:
                    for idx, orig_step in enumerate(template_ops):
                        if idx not in executed_step_indices_set:
                            orig_step_name = orig_step.get("step_name") or orig_step.get("operation")
                            if orig_step_name == step_name:
                                # Check other fields for a match
                                if (orig_step.get("tool_type") == next_step.get("tool_type") and
                                    orig_step.get("resource_fields") == next_step.get("resource_fields")):
                                    next_step["step_index"] = idx
                                    break
            
            # New step: reset retry count and last_step_result
            last_step_retry_count = 0
            last_step_result = None

        # Mark retry step (True only when entering the retry branch)
        is_retry_step = actually_retry_step

        # Save current state (for rollback on retry)
        state_before = copy.deepcopy(current_state)

        sample_wrapper = {
            "original": original_sample,
            "current": current_state,
            "sample_design_brief": sample_design_brief,
        }

        step = {
            "kind": next_step.get("kind") or "template",
            "step_index": next_step.get("step_index", -1),
            "step_name": next_step.get("step_name"),
            "tool_type": next_step.get("tool_type") or "LLM",
            "resource_fields": next_step.get("resource_fields") or [],
            "deleted_interim_fields": next_step.get("deleted_interim_fields") or [],
            "target_fields": next_step.get("target_fields") or [],
            "params": next_step.get("params") or {},
        }

        tool_type = (step.get("tool_type") or "LLM").upper()
        resources = _build_resources_for_step(sample_wrapper, step)

        # Stage2: Execute step
        if trace_cb:
            try:
                trace_cb(
                    idx_int,
                    f"stage2 exec iter={iteration} step={step.get('step_name')} tool={tool_type} retrying={bool(is_retry_step)} retry_count={last_step_retry_count}",
                )
            except Exception:
                pass

        if tool_type == "PURE":
            step_name = step.get("step_name")
            tool_spec = pure_tool_registry.get(step_name)
            if tool_spec is None:
                tool_type = "LLM"
        
        pending_tool_memory_record: Dict[str, Any] = {}

        if tool_type == "LLM":
            ok, delta_current, note = _run_llm_step(
                subtask=subtask,
                step=step,
                resources=resources,
                model=stage2_model,
                dataset_json_path=dataset_json_path,
                dataset_id=dataset_id,
            )
        elif tool_type == "PURE":
            ok, delta_current, note, additional_messages = run_pure_step(
                subtask=_build_subtask_json(subtask),
                resources=resources,
                step=step,
                tool_registry=pure_tool_registry,
                dataset_json_path=dataset_json_path,
                dataset_id=dataset_id,
            )
            # Collect pure tool messages
            if additional_messages:
                pending_tool_memory_record = additional_messages.pop("_tool_memory_record", {}) or {}
                pure_tool_messages.append(additional_messages)
        else:
            return {
                "idx": idx_int,
                "status": "fail",
                "reason": f"unknown_tool_type[{tool_type}]",
                "sample": None,
            }
        
        if not ok:
            # Execution failed; retry if under retry limit
            if last_step_retry_count < 3:
                # Increment retry count (Stage2 execution failure)
                last_step_retry_count += 1
                last_step_result = {
                    "status": "fail",
                    "error": note,
                    "delta": delta_current,
                }
                last_step_plan = next_step

                if trace_cb:
                    try:
                        trace_cb(
                            idx_int,
                            f"stage2 failed -> retry step={step.get('step_name')} retry_count={last_step_retry_count} err={note}",
                        )
                    except Exception:
                        pass

                try:
                    step_history.append({
                        "step_index": next_step.get("step_index"),
                        "step_name": step.get("step_name"),
                        "tool_type": step.get("tool_type"),
                        "kind": next_step.get("kind"),
                        "resource_fields": step.get("resource_fields", []),
                        "deleted_interim_fields": step.get("deleted_interim_fields", []),
                        "target_fields": step.get("target_fields", []),
                        "params": step.get("params", {}),
                        "delta": copy.deepcopy(delta_current),
                        "state_before": state_before,
                        "state_after": copy.deepcopy(state_before),
                        "retry_count": last_step_retry_count,
                        "is_retry": is_retry_step,
                        "retry_base_state": retry_base_state,
                        "status": "fail",
                        "error": note,
                        **stage1_trace_fields,
                    })
                except Exception:
                    pass

                # Roll back state (if partial results were applied)
                current_state = state_before
                continue  # Back to Stage1 to plan retry
            else:
                return {
                    "idx": idx_int,
                    "status": "fail",
                    "reason": f"step_execution_failed_after_retries: {note}",
                    "sample": None,
                }

        # Apply step result
        current_state = _apply_step_delta(current_state, step, delta_current)
        residual_deleted = _enforce_deleted_fields(current_state, step)
        if pending_tool_memory_record:
            _store_tool_memory_record(
                current_state,
                pending_tool_memory_record,
                iteration=iteration,
            )
        if residual_deleted and trace_cb:
            try:
                trace_cb(
                    idx_int,
                    f"cleanup guard: force delete failed for paths={residual_deleted}",
                )
            except Exception:
                pass

        # Check that all target_fields are filled
        missing = _check_target_fields_filled(current_state, step)
        if missing:
            # Missing fields; retry if under retry limit
            if last_step_retry_count < 3:
                # Increment retry count (missing fields)
                last_step_retry_count += 1
                last_step_result = {
                    "status": "incomplete",
                    "missing_fields": missing,
                    "delta": delta_current,
                }
                last_step_plan = next_step

                if trace_cb:
                    try:
                        trace_cb(
                            idx_int,
                            f"stage2 incomplete -> retry step={step.get('step_name')} retry_count={last_step_retry_count} missing={missing}",
                        )
                    except Exception:
                        pass

                # Log incomplete attempt to debug retry loops / stalls
                try:
                    step_history.append({
                        "step_index": next_step.get("step_index"),
                        "step_name": step.get("step_name"),
                        "tool_type": step.get("tool_type"),
                        "kind": next_step.get("kind"),
                        "resource_fields": step.get("resource_fields", []),
                        "deleted_interim_fields": step.get("deleted_interim_fields", []),
                        "target_fields": step.get("target_fields", []),
                        "params": step.get("params", {}),
                        "delta": copy.deepcopy(delta_current),
                        "state_before": state_before,
                        "state_after": copy.deepcopy(state_before),
                        "retry_count": last_step_retry_count,
                        "is_retry": is_retry_step,
                        "retry_base_state": retry_base_state,
                        "status": "incomplete",
                        "missing_fields": missing,
                        **stage1_trace_fields,
                    })
                except Exception:
                    pass

                # Roll back state
                current_state = state_before
                continue  # Back to Stage1 to plan retry
            else:
                return {
                    "idx": idx_int,
                    "status": "fail",
                    "reason": f"missing_target_fields_after_retries: {missing}",
                    "sample": None,
                }

        # Step executed successfully
        if trace_cb:
            try:
                trace_cb(
                    idx_int,
                    f"stage2 ok iter={iteration} step={step.get('step_name')} tool={tool_type}",
                )
            except Exception:
                pass
        step_index = next_step.get("step_index")
        # For template steps, ensure step_index is set and validated
        if next_step.get("kind") == "template":
            if step_index is None or step_index < 0:
                # If Stage1 did not provide step_index, try matching from remaining_template_ops
                step_name = next_step.get("step_name")
                tool_type_for_match = next_step.get("tool_type")
                resource_fields = next_step.get("resource_fields", [])
                
                for rem_step in remaining_template_ops:
                    if rem_step.get("_original_index") is not None:
                        rem_step_name = rem_step.get("step_name") or rem_step.get("operation")
                        rem_tool_type = str(rem_step.get("tool_type") or "").upper()
                        rem_resource_fields = rem_step.get("resource_fields", [])
                        tool_type_upper = str(tool_type_for_match or "").upper()
                        rem_resource_fields_set = set(rem_resource_fields) if isinstance(rem_resource_fields, list) else set()
                        resource_fields_set = set(resource_fields) if isinstance(resource_fields, list) else set()
                        if (rem_step_name == step_name and 
                            rem_tool_type == tool_type_upper and
                            rem_resource_fields_set == resource_fields_set):
                            step_index = rem_step.get("_original_index")
                            next_step["step_index"] = step_index
                            break
                # If still not found, match by step content against original template_ops
                if step_index is None or step_index < 0:
                    tool_type_upper = str(tool_type_for_match or "").upper()
                    resource_fields_set = set(resource_fields) if isinstance(resource_fields, list) else set()
                    for idx, orig_step in enumerate(template_ops):
                        if idx not in executed_step_indices_set:
                            orig_step_name = orig_step.get("step_name") or orig_step.get("operation")
                            orig_tool_type = str(orig_step.get("tool_type") or "").upper()
                            orig_resource_fields = orig_step.get("resource_fields", [])
                            orig_resource_fields_set = set(orig_resource_fields) if isinstance(orig_resource_fields, list) else set()
                            if (orig_step_name == step_name and
                                orig_tool_type == tool_type_upper and
                                orig_resource_fields_set == resource_fields_set):
                                step_index = idx
                                next_step["step_index"] = step_index
                                break
            
            # Validate step_index with soft correction (prefer re-resolve; fail only if unresolvable)
            if step_index is not None and step_index >= 0:
                # Hard constraint: out-of-range index must error
                if step_index >= len(template_ops):
                    return {
                        "idx": idx_int,
                        "status": "fail",
                        "reason": f"step_index_out_of_range: {step_index} >= {len(template_ops)}",
                        "sample": None,
                    }
                orig_step = template_ops[step_index]
                orig_step_name = orig_step.get("step_name") or orig_step.get("operation")
                orig_tool_type = str(orig_step.get("tool_type") or "").upper()
                orig_resource_fields = orig_step.get("resource_fields", [])
                next_step_name = next_step.get("step_name") or ""
                next_tool_type = str(next_step.get("tool_type") or "").upper()
                next_resource_fields = next_step.get("resource_fields", [])
                
                orig_resource_fields_set = set(orig_resource_fields) if isinstance(orig_resource_fields, list) else set()
                next_resource_fields_set = set(next_resource_fields) if isinstance(next_resource_fields, list) else set()
                
                # Hard constraint: step_name + tool_type must match
                if not (orig_step_name == next_step_name and orig_tool_type == next_tool_type):
                    return {
                        "idx": idx_int,
                        "status": "fail",
                        "reason": f"step_index_mismatch: step_index={step_index}, expected step_name={orig_step_name}, got {next_step_name}",
                        "sample": None,
                    }
                # Soft check on resource_fields: template fields unused in current step only warn
                if not orig_resource_fields_set.issubset(next_resource_fields_set):
                    missing_rf = orig_resource_fields_set - next_resource_fields_set
                    pass  # Silently ignore warning
        
        # Append step_index to executed_steps (template step with valid index)
        if step_index is not None and step_index >= 0:
            if step_index not in executed_steps:
                # Do not sort; preserve execution order
                executed_steps.append(step_index)

        # Record step history (full info for later plan assembly)
        step_history.append({
            "step_index": step_index,
            "step_name": step.get("step_name"),
            "tool_type": step.get("tool_type"),
            "kind": next_step.get("kind"),
            "resource_fields": step.get("resource_fields", []),
            "deleted_interim_fields": step.get("deleted_interim_fields", []),
            "target_fields": step.get("target_fields", []),
            "params": step.get("params", {}),
            "delta": copy.deepcopy(delta_current),  # Save delta for history
            "state_before": state_before,
            "state_after": copy.deepcopy(current_state),
            "retry_count": last_step_retry_count,
            "is_retry": is_retry_step,
            "retry_base_state": retry_base_state,
            **stage1_trace_fields,
        })

        # Update last_step_result / last_step_plan to successful execution state
        last_step_result = {
            "status": "ok",
            "delta": delta_current,
        }
        last_step_plan = next_step
        # Only reset retry counter when this was a genuinely new step.
        # If this was a voluntary quality retry (is_retry_step=True), preserve the
        # accumulated count so Stage1 cannot loop indefinitely on the same step.
        if not is_retry_step:
            last_step_retry_count = 0

        # Save cache (after last_step_* update so cache matches in-memory state)
        if cache_path:
            to_save = {
                "idx": idx_int,
                "sample_design_brief": sample_design_brief,
                "current_state": current_state,
                "executed_steps": executed_steps,
                "step_history": step_history,
                "last_step_plan": last_step_plan,
                "last_step_result": last_step_result,
                "last_step_retry_count": last_step_retry_count,
                "pure_tool_messages": pure_tool_messages,
            }
            _save_sample_iterative_state(
                cache_path,
                subtask_id=subtask_id,
                dataset_id=dataset_id,
                idx_int=idx_int,
                state=to_save,
            )

    # Max iterations reached: still export last final for debug_transform_batch and verify repair
    exit_reason = f"max_iterations_reached: {max_iterations}"
    partial_sample = _build_partial_export_sample(
        original_sample=original_sample,
        subtask=subtask,
        dataset_id=dataset_id,
        idx_int=idx_int,
        current_state=current_state,
        pure_tool_messages=pure_tool_messages,
        exit_reason=exit_reason,
    )
    if partial_sample is not None:
        return {
            "idx": idx_int,
            "status": "partial",
            "reason": exit_reason,
            "sample": partial_sample,
        }
    return {
        "idx": idx_int,
        "status": "fail",
        "reason": exit_reason,
        "sample": None,
    }

# =========================
# Main entry: multi-threaded processing of multiple samples
# =========================

@register_tool("transform_samples_iterative")
def transform_samples_iterative(
    subtask: Dict[str, Any],
    current_pairs: List[Dict[str, Any]],
    dataset_cards: Dict[str, Any],
    transformed_buffer: Dict[str, List[Dict[str, Any]]],
    subtask_id: str,
    max_workers: int = 20,
    tools_list: Optional[Dict[str, Any]] = None,
    cache_root: Optional[str] = None,
    model_config_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Iterative transform entry point:

    - Group by pair first (each pair is a fixed dataset_id + tool_plan + idx_todo)
    - For each pair:
        - Load raw_samples, template_ops, idx_todo
        - Run the full iterative pipeline per sample within the pair
        - Samples within a pair may run in parallel (sample-level parallelism)
    - Pairs run strictly in sequence without interleaving
    """
    stage1_model = _get_stage1_model(model_config_path)
    stage2_model = _get_stage2_model(model_config_path)
    stage0_model = _get_stage0_model(model_config_path)
    pure_tool_registry = build_pure_tool_registry(tools_list)
    
    if not isinstance(transformed_buffer, dict):
        transformed_buffer = {}
    if not isinstance(transformed_buffer.get(subtask_id), list):
        transformed_buffer[subtask_id] = []

    tail_ops_allowed = [
        "answer_format_adjust",
        "final_question_rephrase",
        "final_answer_rephrase",
        "final_context_rephrase",
        "rationale_optional_add",
        "remove_noise_texts",
        "remove_interim_fields",
    ]

    # ============== Build per-pair sample lists grouped by pair ==============
    pair_jobs: List[Dict[str, Any]] = []
    total_samples = 0

    for pair in current_pairs:
        if not isinstance(pair, dict):
            continue

        did = str(pair.get("dataset_id") or "").strip()
        if not did:
            continue

        dataset_card = dataset_cards.get(did, {})
        if not dataset_card:
            continue

        src = (dataset_card.get("raw_meta") or {}).get("source_json")
        if not isinstance(src, str) or not src.strip() or not os.path.exists(src):
            continue

        # Build todo list early so we can skip reading the raw dataset if everything is cached.
        idx_todo = pair.get("idx_todo") or []
        if not isinstance(idx_todo, list):
            continue

        template_ops = pair.get("tool_plan_template") or pair.get("tool_plan") or []
        if not isinstance(template_ops, list) or not template_ops:
            continue

        # --- Read transformation_buffer cache FIRST ---
        # If a sample is already in transformation_buffer, it is fully processed and must NOT
        # re-enter the Stage1/Stage2 iterative pipeline.
        cached_items = _load_transformation_buffer_cache(
            cache_root,
            subtask_id=subtask_id,
            dataset_id=did,
        )
        cached_idx_set = {it.get("idx") for it in cached_items if isinstance(it, dict)}
        cached_idx_set = {i for i in cached_idx_set if isinstance(i, int)}

        # Merge cached samples into the in-memory transformed_buffer (dedupe by dataset_id+idx).
        try:
            existing_keys = {
                (str(x.get("dataset_id")), int(x.get("idx")))
                for x in (transformed_buffer.get(subtask_id) or [])
                if isinstance(x, dict) and x.get("dataset_id") is not None and x.get("idx") is not None
            }
        except Exception:
            existing_keys = set()
        for it in cached_items:
            if not isinstance(it, dict):
                continue
            try:
                k = (str(it.get("dataset_id") or did), int(it.get("idx")))
            except Exception:
                continue
            if k in existing_keys:
                continue
            transformed_buffer[subtask_id].append({
                "dataset_id": k[0],
                "idx": k[1],
                "sample": it.get("sample"),
            })
            existing_keys.add(k)

        # Filter out cached indices from processing list
        if cached_idx_set:
            idx_todo = [x for x in idx_todo if _safe_int(x, -1) not in cached_idx_set]
        if not idx_todo:
            # All requested samples already transformed (cache hit); no need to read raw dataset.
            continue

        raw_obj = _load_json(src)
        raw_samples = raw_obj.get("data") if isinstance(raw_obj, dict) else None
        if not isinstance(raw_samples, list) or not raw_samples:
            continue

        # Build this pair's sample list
        sample_args: List[Tuple[int, Dict[str, Any]]] = []
        for idx in idx_todo:
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if idx_int < 0 or idx_int >= len(raw_samples):
                continue
            raw_sample = raw_samples[idx_int]
            sample_args.append((idx_int, raw_sample))

        if not sample_args:
            continue

        pair_jobs.append({
            "pair": pair,
            "dataset_id": did,
            "template_ops": template_ops,
            "sample_args": sample_args,  # list of (idx_int, raw_sample)
        })

        total_samples += len(sample_args)

    if not pair_jobs:
        payload = {"status": "no_op", "reason": "no_samples_to_process"}
        return current_pairs, transformed_buffer, payload

    # Global progress bar: sum of sample counts across all pairs
    progress_bar = tqdm(total=total_samples, desc=f"Transform[{subtask_id}]", dynamic_ncols=True)
    results: List[Tuple[Dict[str, Any], str, Dict[str, Any]]] = []

    # Runtime config (re-read env for debug sessions)
    trace_enabled = str(os.getenv("TRANSFORM_TRACE", "0")).strip().lower() not in ("0", "false", "no", "n", "off")
    trace_heartbeat_s = int(os.getenv("TRANSFORM_TRACE_HEARTBEAT_S", str(TRANSFORM_TRACE_HEARTBEAT_S)))
    trace_enabled = trace_enabled or TRANSFORM_TRACE

    def _trace(msg: str) -> None:
        if not trace_enabled:
            return
        try:
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)

    # Re-read diagnostic env vars at runtime (IDE debug sessions may set env after import)
    diag_enabled = str(os.getenv("TRANSFORM_DIAG", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
    diag_heartbeat_s = int(os.getenv("TRANSFORM_DIAG_HEARTBEAT_S", str(TRANSFORM_DIAG_HEARTBEAT_S)))
    diag_enabled = diag_enabled or TRANSFORM_DIAG

    def _diag(msg: str) -> None:
        if not diag_enabled:
            return
        try:
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)

    # ============== Process all pair samples concurrently (flatten across pairs into one pool) ==============
    # Flatten all (pair, sample) jobs from all pairs so max_workers are shared globally,
    # allowing samples from different datasets to run concurrently instead of pair-by-pair.
    global_status_lock = threading.Lock()
    global_status_by_key: Dict[Tuple[str, int], str] = {}  # keyed by (did, idx)

    def _process_one_global(
        pair: Dict[str, Any],
        did: str,
        template_ops: List[Any],
        source_json: Optional[str],
        idx_int: int,
        raw_sample: Dict[str, Any],
    ):
        def _cb(i: int, s: str) -> None:
            if not trace_enabled:
                return
            with global_status_lock:
                global_status_by_key[(did, i)] = s
            _trace(f"[TransformTrace] subtask={subtask_id} dataset={did} idx={i} {s}")

        if trace_enabled:
            with global_status_lock:
                global_status_by_key[(did, idx_int)] = "init"

        image_entries = _collect_image_files_with_keys(raw_sample)
        image_files = [p for p, _ in image_entries]
        if image_files:
            image_files = resolve_image_paths(
                image_files,
                dataset_json_path=source_json,
                dataset_id=did,
            )

        result = _process_one_sample_iterative(
            idx_int=idx_int,
            raw_sample=raw_sample,
            subtask=subtask,
            template_ops=template_ops,
            tail_ops_allowed=tail_ops_allowed,
            pure_tool_registry=pure_tool_registry,
            stage0_model=stage0_model,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            image_files=image_files,
            dataset_json_path=source_json,
            cache_path=_sample_iterative_cache_path(
                cache_root,
                subtask_id=subtask_id,
                dataset_id=did,
                idx_int=idx_int,
            ),
            sample_design_brief_cache_path=_sample_design_brief_cache_path(
                cache_root,
                subtask_id=subtask_id,
                dataset_id=did,
                idx_int=idx_int,
            ),
            subtask_id=subtask_id,
            dataset_id=did,
            trace_cb=_cb if trace_enabled else None,
        )
        return pair, did, result

    # Build flat job list across all pairs
    all_sample_jobs: List[Tuple] = []
    for job in pair_jobs:
        _pair = job["pair"]
        _did = job["dataset_id"]
        _template_ops = job["template_ops"]
        _source_json = dataset_cards.get(_did, {}).get("raw_meta", {}).get("source_json", None)
        for idx_int, raw_sample in job["sample_args"]:
            all_sample_jobs.append((_pair, _did, _template_ops, _source_json, idx_int, raw_sample))

    # Pre-load each dataset's existing cache so we can append incrementally without re-reading on every write.
    did_cache_state: Dict[str, Dict] = {}
    for job in pair_jobs:
        _dc_did = job["dataset_id"]
        if _dc_did in did_cache_state:
            continue
        _dc_path = os.path.join(cache_root, "transform_log", "transformation_buffer", f"{subtask_id}_{_dc_did}_transformed.json") if cache_root else None
        if _dc_path and os.path.exists(_dc_path):
            try:
                did_cache_state[_dc_did] = _load_json(_dc_path)
            except Exception:
                did_cache_state[_dc_did] = {"subtask_id": subtask_id, "dataset_id": _dc_did, "transformed_buffer": []}
        elif _dc_path:
            did_cache_state[_dc_did] = {"subtask_id": subtask_id, "dataset_id": _dc_did, "transformed_buffer": []}

    actual_workers = min(max_workers, len(all_sample_jobs)) or 1
    if diag_enabled:
        _diag(
            f"[TransformDiag] submit subtask={subtask_id} total_samples={len(all_sample_jobs)} "
            f"workers={actual_workers} pairs={len(pair_jobs)} "
            f"datasets={[j['dataset_id'] for j in pair_jobs]}"
        )

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_info: Dict[Any, Tuple[Dict, str, int]] = {
            executor.submit(_process_one_global, pair, did, tmpl, src, idx, raw): (pair, did, idx)
            for pair, did, tmpl, src, idx, raw in all_sample_jobs
        }
        start_ts_by_key: Dict[Tuple[str, int], float] = {
            (did, idx): time.time()
            for _, did, _, _, idx, _ in all_sample_jobs
        }
        last_progress_ts = time.time()
        pending = set(future_to_info.keys())

        while pending:
            wait_timeout_s = 30
            if trace_enabled or diag_enabled:
                wait_timeout_s = min(trace_heartbeat_s, diag_heartbeat_s)
            if NO_PROGRESS_TIMEOUT_S > 0:
                wait_timeout_s = min(wait_timeout_s, NO_PROGRESS_TIMEOUT_S)

            done, pending = wait(pending, timeout=wait_timeout_s, return_when=FIRST_COMPLETED)

            if not done:
                now = time.time()
                pending_keys = [future_to_info[pf][1:] for pf in pending if pf in future_to_info]
                stalled_s = now - last_progress_ts
                if NO_PROGRESS_TIMEOUT_S > 0 and stalled_s >= NO_PROGRESS_TIMEOUT_S:
                    msg = (
                        f"[TransformTimeout] no completed samples for {stalled_s:.1f}s "
                        f"(limit={NO_PROGRESS_TIMEOUT_S}s); terminating process to stop hung LLM/API calls. "
                        f"subtask={subtask_id} pending={len(pending_keys)}"
                    )
                    try:
                        tqdm.write(msg)
                    except Exception:
                        print(msg, flush=True)
                    try:
                        import sys
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception:
                        pass
                    os._exit(124)

                if not (trace_enabled or diag_enabled):
                    continue

                pending_keys_sorted = sorted(pending_keys)[:50]
                with global_status_lock:
                    parts = [
                        f"{d}/{i}:{global_status_by_key.get((d, i), 'unknown')}"
                        for d, i in pending_keys_sorted
                    ]
                longest_key = None
                longest_elapsed = -1.0
                for d, i in pending_keys:
                    elapsed = now - start_ts_by_key.get((d, i), now)
                    if elapsed > longest_elapsed:
                        longest_elapsed = elapsed
                        longest_key = (d, i)
                if trace_enabled:
                    _trace(
                        f"[TransformTrace] heartbeat subtask={subtask_id} pending={len(pending_keys)} "
                        f"longest={longest_key} longest_elapsed_s={longest_elapsed:.1f} "
                        + " | ".join(parts[:30])
                        + (" | ...truncated" if len(parts) > 30 else "")
                    )
                if diag_enabled:
                    _diag(
                        f"[TransformDiag] heartbeat subtask={subtask_id} waiting={len(pending_keys)} "
                        f"longest={longest_key} longest_elapsed_s={longest_elapsed:.1f}"
                    )
                continue

            for f in done:
                last_progress_ts = time.time()
                pair_fallback, did_fallback, idx_fallback = future_to_info.get(f, ({}, "unknown", None))
                t0 = start_ts_by_key.get((did_fallback, idx_fallback), None)
                try:
                    pair_ret, did_ret, result = f.result()
                except Exception as e:
                    pair_ret, did_ret, result = pair_fallback, did_fallback, {
                        "idx": idx_fallback,
                        "status": "fail",
                        "reason": f"worker_exception: {type(e).__name__}: {e}",
                        "sample": None,
                    }
                elapsed_s = (time.time() - t0) if (t0 is not None) else None
                if trace_enabled and idx_fallback is not None:
                    with global_status_lock:
                        global_status_by_key[(did_fallback, idx_fallback)] = f"done status={result.get('status')}"
                    _trace(
                        f"[TransformTrace] done subtask={subtask_id} dataset={did_ret} idx={result.get('idx')} "
                        f"status={result.get('status')} elapsed_s={(elapsed_s if elapsed_s is not None else 'na')}"
                    )
                if diag_enabled:
                    _diag(
                        f"[TransformDiag] done subtask={subtask_id} dataset={did_ret} idx={result.get('idx')} "
                        f"status={result.get('status')}"
                    )
                results.append((pair_ret, did_ret, result))
                progress_bar.update(1)

                # Incremental cache write: persist ok and partial samples immediately.
                if result.get("status") in ("ok", "partial") and isinstance(result.get("sample"), dict):
                    _ic_did = did_ret
                    _ic_idx = result.get("idx")
                    _ic_sample = result.get("sample")
                    _ic_state = did_cache_state.get(_ic_did)
                    _ic_path = os.path.join(cache_root, "transform_log", "transformation_buffer", f"{subtask_id}_{_ic_did}_transformed.json") if cache_root and _ic_state is not None else None
                    if _ic_state is not None and _ic_path:
                        _ic_existing = {item.get("idx") for item in _ic_state.get("transformed_buffer", [])}
                        if _ic_idx not in _ic_existing:
                            _ic_state.setdefault("transformed_buffer", []).append({
                                "dataset_id": _ic_did,
                                "idx": _ic_idx,
                                "sample": _ic_sample,
                            })
                        try:
                            os.makedirs(os.path.dirname(_ic_path), exist_ok=True)
                            with open(_ic_path, "w", encoding="utf-8") as _ic_f:
                                json.dump(_ic_state, _ic_f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass

    # Update in-memory transformed_buffer from collected results (disk already saved incrementally above).
    for pair_ret, did_ret, result in results:
        idx_int = result.get("idx")
        status = result.get("status")
        sample = result.get("sample")
        if status in ("ok", "partial") and isinstance(sample, dict):
            transformed_buffer[subtask_id].append({
                "dataset_id": did_ret,
                "idx": idx_int,
                "sample": sample,
            })

    progress_bar.close()

    # Compute overall status
    any_transformed = any(
        transformed_buffer.get(subtask_id, [])
    )

    payload = {"status": "ok" if any_transformed else "no_op"}
    return current_pairs, transformed_buffer, payload
