# tools/dataset_scorer_tool.py
from __future__ import annotations

from logging import config
from typing import List, Dict, Any, Optional,Tuple, Set
import json

from regex import T
from tools.shared.tool_sanitizer import _sanitize_pure_tool_ops
from utils.registry import register_tool
from utils.llm_caller import llm_call_json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import time
import random


def _sleep_backoff(attempt: int, base: float = 0.8, cap: float = 6.0):
    t = min(cap, base * (2 ** attempt))
    t *= (0.8 + 0.4 * random.random())
    time.sleep(t)

def run_batches_concurrent(
    batches: List[List[str]],
    process_batch_fn,
    *,
    max_workers: int = 8,
    max_retries: int = 2,
    error_prefix: str = "batch failed",
    collect_errors: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns:
      - merged_result: Dict
      - errors: List[{
            batch: [...],
            attempt: int,
            error: str,
            traceback: str
        }]
    """
    if not batches:
        return {}, []

    merged: Dict[str, Any] = {}
    errors: List[Dict[str, Any]] = []

    mw = max(1, min(int(max_workers), len(batches)))

    with ThreadPoolExecutor(max_workers=mw) as ex:
        future_to_state = {
            ex.submit(process_batch_fn, b): {
                "batch": b,
                "attempt": 0,
            }
            for b in batches
        }

        while future_to_state:
            for fut in as_completed(list(future_to_state.keys())):
                state = future_to_state.pop(fut)
                batch = state["batch"]
                attempt = state["attempt"]

                try:
                    part = fut.result()
                    if part:
                        merged.update(part)

                except Exception as e:
                    err_info = {
                        "batch": batch,
                        "attempt": attempt + 1,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

                    if attempt < max_retries:
                        _sleep_backoff(attempt)
                        new_fut = ex.submit(process_batch_fn, batch)
                        future_to_state[new_fut] = {
                            "batch": batch,
                            "attempt": attempt + 1,
                        }
                    else:
                        if collect_errors:
                            errors.append(err_info)
                        else:
                            raise RuntimeError(
                                f"{error_prefix} (batch={batch})\n{err_info['error']}"
                            ) from e

    return merged, errors
# Helpers
def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

def _chunk(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def _get_dataset_id_from_retrieval_item(r: Dict[str, Any]) -> str:
    return str(r.get("dataset_id") or r.get("id") or "").strip()


def _is_interim_path(path: Any) -> bool:
    return isinstance(path, str) and path.startswith("interim.")


def _as_str_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    return [v for v in x if isinstance(v, str) and v]


def _recalibrate_interim_deletions(plan: Any) -> List[Dict[str, Any]]:
    """
    Deterministically recalibrate deleted_interim_fields timing:
    - collect all interim.* paths referenced in the plan
    - remove existing deleted_interim_fields placements
    - place each deletion right after its last usage step
      (or in a final cleanup-only op when last usage is the final step)
    """
    if not isinstance(plan, list):
        return []

    ops: List[Dict[str, Any]] = []
    for op in plan:
        if isinstance(op, dict):
            ops.append(dict(op))
    if not ops:
        return []

    # Keep only valid string paths on each op and clear deletion placement for recalibration.
    for op in ops:
        op["resource_fields"] = _as_str_list(op.get("resource_fields"))
        op["target_fields"] = _as_str_list(op.get("target_fields"))
        op["deleted_interim_fields"] = []

    n = len(ops)
    interim_last_use: Dict[str, int] = {}
    seen_interim: Set[str] = set()

    for i, op in enumerate(ops):
        referenced = set(op.get("resource_fields", [])) | set(op.get("target_fields", []))
        for p in referenced:
            if _is_interim_path(p):
                seen_interim.add(p)
                interim_last_use[p] = i

    if not seen_interim:
        return ops

    # Bucket deletions by target operation index.
    delete_bucket: Dict[int, List[str]] = {}
    need_final_cleanup: List[str] = []
    for p in sorted(seen_interim):
        last_i = interim_last_use.get(p)
        if last_i is None:
            continue
        delete_i = last_i + 1
        if delete_i < n:
            delete_bucket.setdefault(delete_i, []).append(p)
        else:
            need_final_cleanup.append(p)

    for i, fields in delete_bucket.items():
        # "after step i-1" means "in step i's deleted_interim_fields"
        existing = set(_as_str_list(ops[i].get("deleted_interim_fields")))
        existing.update(fields)
        # Guardrail: same op cannot both write and delete the same path
        target_set = set(_as_str_list(ops[i].get("target_fields")))
        cleaned = sorted([p for p in existing if p not in target_set])
        ops[i]["deleted_interim_fields"] = cleaned

    if need_final_cleanup:
        last_op = ops[-1]
        is_cleanup_only = (
            not _as_str_list(last_op.get("resource_fields"))
            and not _as_str_list(last_op.get("target_fields"))
        )
        if is_cleanup_only:
            existing = set(_as_str_list(last_op.get("deleted_interim_fields")))
            existing.update(need_final_cleanup)
            last_op["deleted_interim_fields"] = sorted(existing)
        else:
            ops.append({
                "operation": "cleanup_interim_fields",
                "tool_type": "PURE",
                "resource_fields": [],
                "target_fields": [],
                "deleted_interim_fields": sorted(set(need_final_cleanup)),
                "dimensions": ["cleanup"],
                "notes": "Deterministic cleanup of interim fields after their final use.",
            })

    return ops

_TRANSFORMABILITY_CHECK_BATCH_PROMPT = r"""
You are the Transformability Assessor.
Your job is to judge whether retrieved datasets can be transformed into the target subtask's QA format.

[CRITICAL CONTEXT]
- A **subtask** is an **evaluation task** used to assess model capabilities.
- You are designing a transformation pipeline to create evaluation data for it.
- Do NOT rewrite samples or generate QA. Only analyze feasibility and required operations.

[Your Task]
For each candidate dataset, determine:
- Can this dataset be transformed into the format required by the evaluation subtask?
- What transformation operations are needed?
- Is the transformation feasible and grounded?

==================== Inputs ====================
Subtask (Evaluation Task, JSON):
{subtask_json}

Candidate datasets (JSON list, size={n_cards}):
{dataset_cards_json}

Pure tools(Only these can be used as PURE operations):
{tools_json}

==================== Target Format ====================
Final format: QA (question → structured answer)

[Path Convention]
The path format `final.input.xxx` and `final.output.xxx` corresponds to the subtask's sample_schema:
- `final.input.xxx` maps to `sample_schema.input.fields.xxx`
- `final.output.xxx` maps to `sample_schema.output.fields.xxx`

Example: If sample_schema has:
  - input.fields: {{"question": ..., "context": ..., "audio_url": ...}}
  - output.fields: {{"answer": ...}}
Then the final paths are:
  - final.input.question
  - final.input.context
  - final.input.audio_url
  - final.output.answer

Structured answer types: {{binary, choice, label, span}}
Open-ended free text is discouraged.

Constraints:
- Final fields MUST match the subtask sample_schema exactly.
  **Important**: This means you CANNOT add fields that don't exist in sample_schema.
  However, you CAN merge required information into existing fields.
  Example: If subtask expects multiple-choice but sample_schema has no "choices" field,
  you can embed the choices within the "question" field (e.g., "Question: ... Options: A) ... B) ...").
- final.input.* and final.output.* may ONLY write to fields defined in the sample_schema.
  You may create interim.* fields freely, but you must NOT create schema-external final.* fields.
- final.input.* MUST NOT leak final.output.* answers.
- The subtask defines answer_type. A feasible pipeline MUST produce that type.
- If answer_type mismatch, add grounded alignment operations by merging into existing schema fields.
- You may synthesize questions, contexts, choices, labels, scenarios, narrators, conflicts, omissions, and interfaces. The ground-truth final.output.* answer may come from original data/metadata, reliable reasoning over them, or a controlled synthetic construction defined by the transformation pipeline.
- Synthetic additions are allowed as long as they do NOT contradict original data, original metadata, or preserved objective facts. The final.output.* answer must be consistent with, and verifiable from, the constructed final input and the operation plan.
- Treat factual meta as hard constraints, not as a complete script. LLM operations may enrich, adapt, reframe, or instantiate new content around those constraints, but must preserve every factual invariant expressed by the dataset meta.

==================== Dataset Types ====================
• Type-A (QA-like): Already contains question-answer structures, may require rewriting or adaptation.
• Type-B (Meta-rich): Contains factual/meta/structural information from which grounded QA can be synthesized.
  Meta defines factual invariants. Representation is flexible (structure, perspective, modality, discourse, style, etc.).
  Evaluate feasibility in representation space, not surface text space.

A dataset may be classified as: "A", "B", or "either".

==================== Transformability Principles ====================
High recall is preferred. Prefer direct derivation from original data or metadata, but also accept transformations that create new evaluation information through controlled synthesis (e.g., new perspectives, disagreements, choice labels, scenarios, or queried claims) when the synthesis is compatible with the dataset meta and does not contradict preserved facts. The dataset meta provides factual invariants and usable resources; it does not need to already contain the exact target task format. The answer does not need to be explicitly stored in the source dataset if the transformation pipeline can define it in a controlled, self-consistent, and verifiable way. **External knowledge:** whenever **factual** world or domain knowledge (often specialized or professional) would improve grounding, disambiguate entities, enrich a transformed sample, or yield clearer evaluation signal, **favor** including `web_search` as a grounded retrieval step—treat it as a normal complement to the dataset, not a last resort—before marking a candidate infeasible. Use it only for **verifiable facts** from credible sources (not opinions, speculation, or subjective takes). Retrieved fact evidence is acceptable grounding for objective answers (e.g., entity, artwork, historical, cultural, geographic, temporal, scientific, biographical, clinical or technical definitions). **Image-grounded search:** when the sample modality includes **images**, treat **vision-first `web_search`** (same image passed as `image_paths` at execution, plus a precise text `query`) as a normal, encouraged way to pull **factual** evidence (identity, taxonomy, location, era, equipment, diagnosis-related public facts, etc.)—do not default to text-only anchors if the picture is the strongest signal. Mark no only when even a controlled synthetic construction and available tool use (including web_search) would be unsupported, contradictory to the data/meta, impossible for the available modality/fields, or too subjective/brittle to verify.

==================== Tools ====================
[LLM Operations - Very Flexible]
LLM operations have **very loose constraints**. You can create, rewrite, or delete most content, with only two types of facts that MUST NOT be violated:

**Two types of facts that MUST be preserved (cannot be altered):**
1. **Objective facts**: Entities, events, outcomes, relationships that are objectively true in the real world.
2. **Factual meta from original data**: Facts explicitly stated in the original dataset's meta information.

**Everything else is flexible - you can:**
• Create, rewrite, or delete content
• Restructure, expand, scenarize, or synthesize representations
• Create synthetic structure (roles, targets, tasks, perspectives, indexing)
• Generate synthetic scenarios or scenes
• Build synthetic evaluation interfaces (choice sets, spans, labels)
• Add synthetic reasoning layers
• Construct synthetic targets
• Perform synthetic alignment operations
• Introduce additional context, background, or framing as needed
• Modify or remove non-factual content
• Create dialogue turns from text content, or merge text content into dialogue turns.
• When supported, convert image modality to text: write an image caption/description, extract salient entities/actions/relations, or generate scene-grounded dialogue turns for depicted characters (e.g., when the image is a dialogue scene).
• Introduce background knowledge or context when the subtask requires it. It may shape the constructed scenario or interface, but it must not contradict original data, original metadata, or preserved objective facts.

[PURE Operations]
Pure tools operate on concrete assets (audio/image/etc).
- PURE tool names MUST match existing Pure tools exactly (cannot invent new names).
- Each PURE tool can only be used once per pipeline.
- Transformability is a dataset-level planning stage: DO NOT generate sample-level `params` for any operation here.
- PURE tools are trusted resource producers. Their target_fields MUST contain exactly one stable memory base path:
  `tool_memory.<pure_tool_name>.<operation_key>`
  where operation_key is a short, stable snake_case name describing this tool use (for example:
  `tool_memory.web_search.identify_artist`, `tool_memory.text2speech.dialog_audio`).
- PURE tools MUST NOT target interim.* or final.*. The executor will write:
  - `<memory_base>.input`  = concrete tool_args used at sample execution time
  - `<memory_base>.output` = the backend tool result
- Concrete tool_args (query text, image_paths, etc.) are planned later at **sample** execution time, but **dataset-level** `resource_fields` is still REQUIRED now: it must list every `original.*` (and later `tool_memory.*` / `interim.*`) path the pure-tool planner will need to read from the sample to build those args. Never use `resource_fields: []` for a real PURE tool step (cleanup-only is the only exception).

[External Knowledge via web_search]
- **Encouraged use:** Include the PURE tool `web_search` whenever **external factual** knowledge—typically verifiable, often **domain-specialized or professional** (medicine, STEM, law, finance, arts catalogues, standards, named entities, dates, taxonomy, etc.)—would strengthen the pipeline, unlock new **insights for sample transformation** (richer questions, correct labels, disambiguation, standard terminology), or fill gaps not fully spelled out in the dataset. You do not need to wait for a hard "blocker"; if credible search-backed facts would clearly help, plan `web_search`.
- **Fact-only constraint:** Retrieved content must be used as **objective, checkable fact**. Do not rely on `web_search` for opinions, rumors, taste, or other non-factual material. Prefer sources and queries that yield stable professional or encyclopedic facts.
- Treat `web_search` as a reliable retrieval tool for **fact-type** evidence when the query can be grounded in sample resources (text labels, entities, titles, metadata, **and/or images**) and the later LLM step can cite/consume the retrieved evidence.
- **Use images well (`image_paths`):** whenever the pipeline reads **sample images** (`original.input.*` image paths), **prefer** planning `web_search` so execution can attach those same paths as `image_paths` alongside a focused `query`—use the pixels to retrieve **facts** (what/where/when/who/which standard term), not only when text metadata is missing. Visual search is encouraged for species, landmarks, artworks, products, instruments, charts/screenshots, satellite or medical-style imagery (public-reference-level facts), etc., as long as results stay **verifiable factual** evidence tied to the depicted content.
- `web_search` supports image inputs: at sample level, the pure-tool planner **should** pass `image_paths` when any **factual** lookup would benefit from the actual image (identity, attribution, fine-grained category, geographic or temporal anchoring from visuals), not only for "artwork identity" edge cases.
- For image datasets, do not mark a candidate "no" merely because the dataset lacks text metadata if the image itself can be supplied to `web_search` and the subtask can be grounded by retrieved evidence plus preserved sample resources.
- **Avoid redundancy:** Do NOT use `web_search` when the needed information is already fully and unambiguously present in original fields/meta with nothing factual left to add.
- A `web_search` operation should write to a stable memory base such as
  `tool_memory.web_search.identify_artist` or `tool_memory.web_search.retrieve_biography_fact`.
- Later LLM operations that consume web_search evidence MUST include the concrete memory output path in resource_fields, for example:
  `tool_memory.web_search.identify_artist.output.answer`.
  Mentioning tool_memory in notes is NOT sufficient; if an LLM step depends on PURE output, the corresponding `tool_memory.*.input.*` or `tool_memory.*.output.*` path MUST appear in resource_fields.
- Do NOT include `params` for `web_search` in this transformability plan. At sample level, the pure-tool planner will generate the concrete `query` and optional `image_paths` (whenever image-grounded fact retrieval helps).
- **INVALID:** `web_search` with `resource_fields: []`. You must list at least one anchor from the dataset card `meta_structure`, e.g. `original.input.<path>` for the primary image field (when the task is image-grounded) and any text fields (title, artist name, caption) that the search should ground on—use real paths from `meta_structure[role].paths`, prefixed with `original.` as in the schema section below. **If the sample uses an image, include that image path in `resource_fields`** so the executor can pass it as `image_paths` for fact search, unless the search is strictly text-only and the image adds no factual anchor.


==================== Operation Design ====================
- Operations are minimal representation edits, each modifying one dimension.
- Operations form ordered pipelines. Shortest feasible grounded pipeline preferred.
- Multi-step synthesis for Type-B is normal and expected.

==================== Required Operation Schema (UPDATED) ====================
Each required operation must declare:

• resource_fields:
    MUST use REAL dataset paths resolved to:
        original.input.xxx
        original.output.xxx
    For every PURE operation **except** `cleanup_interim_fields`, `resource_fields` MUST be non-empty: list all sample fields the tool will read to build `tool_args` (anchors for retrieval, audio/image paths, text to convert, etc.). Empty `[]` is only allowed for cleanup-only ops.
    Do NOT use role names such as "context", "entity", "question", "answer", use **real paths** from dataset meta instead.
    To find real paths: look up meta_structure[role]["paths"] in the dataset card.
    Example: if meta_structure["question"]["paths"] = ["input.question"], the valid resource_field is original.input.question.
    interim.xxx created by a prior LLM operation in this pipeline may also appear here as input.
    tool_memory.xxx paths created by a prior PURE operation may also appear here as input.
    interim.xxx fields are temporary and must not persist to the final output; tool_memory.xxx fields are persistent hidden tool resources.

• target_fields:
    For LLM operations, MUST write into:
        interim.xxx (temporary fields)
        final.input.xxx (must match sample_schema.input.fields.xxx)
        final.output.xxx (must match sample_schema.output.fields.xxx)
    Note: final.input.xxx and final.output.xxx paths must correspond to fields defined in the subtask's sample_schema.
    interim.xxx paths may be created as needed. final.input.xxx and final.output.xxx paths must already exist in sample_schema.
    LLM operations MUST NOT write tool_memory.*.
    For PURE operations, target_fields MUST contain one `tool_memory.<tool_name>.<operation_key>` base path.
    Cleanup-only PURE operations are the only exception: they use target_fields=[].

• deleted_interim_fields:
    - List of interim fields to delete **after this step**.
    - You MAY keep most interim deletions in a dedicated final cleanup step.
    - A deterministic post-checker will recalibrate deletion timing to right-after-last-use.
    - Ensure every temporary interim field is eventually deleted by pipeline end.

• dimensions:
    - Short descriptive tags for what representation aspect this operation changes.
    - These are not fixed enums; use concise labels such as schema_alignment, question_synthesis,
      answer_extraction, choice_construction, modality_conversion, context_rewriting,
      label_mapping, or span_extraction.

IMPORTANT CONSISTENCY RULES:
- original.xxx is immutable and MUST NOT be deleted or rewritten.
- interim.xxx fields must be created by an earlier LLM operation before they can appear in later resource_fields.
- tool_memory.xxx fields must be created by an earlier PURE operation before they can appear in later resource_fields.
- Operation order must satisfy all field dependencies.
- Within a single operation, the **same path MUST NOT appear in both target_fields and deleted_interim_fields**.
  If you need to first use a field and then delete it, do this in **two separate operations**:
    Bad (same op):
    target_fields: ["interim.ctx"], deleted_interim_fields: ["interim.ctx"]

  Good (two ops):
    op1: target_fields: ["interim.ctx"]  # write
    op2: deleted_interim_fields: ["interim.ctx"]  # then delete

- if you need to overwrite an interim or final field, just list it in target_fields; no need to delete first.
- Once an interim.xxx field is deleted, it MUST NOT appear in any later resource_fields or target_fields.
- A cleanup-only operation must have tool_type = "PURE", resource_fields = [], and target_fields = []; it should only list deleted_interim_fields.
- If an LLM step's notes rely on information retrieved/created by a PURE step, that LLM step's resource_fields MUST include the relevant `tool_memory.<tool>.<operation_key>.input.*` or `.output.*` path.

At pipeline end, ONLY:
    final.input.*
    final.output.*
may remain in the exported benchmark sample. tool_memory.* may remain as hidden construction memory during execution but is never exported; no interim.* fields may remain.
==================== Judgment ====================
For each dataset, decide:
• transformable: "yes" | "no"
  - "yes": Feasible grounded pipeline exists, including pipelines that create answers through controlled synthesis compatible with dataset meta and preserved facts
  - "no": No executable grounded or controlled-synthetic path exists, expected usable sample count is clearly insufficient, or dataset modality/fields are fundamentally incompatible with the subtask requirements
• dataset_type: "A" | "B" | "either" | "none"
• risk: "low" | "medium" | "high"
  High risk ≠ impossible. Mark "yes" if a feasible grounded or controlled-synthetic pipeline exists, even if the transformation is somewhat risky.

==================== Output Format (STRICT JSON) ====================
Output must be LIST of dataset transformability results.
Output list size = number of datasets, same order.

Each item:
{{
  "dataset_id": "...",
  "transformable": "yes" | "no",
  "dataset_type": "A" | "B" | "either" | "none",
  "required_operations": [
    {{
      "operation": "<short_action_name_or_pure_tool_name>",
      "tool_type": "LLM" | "PURE",
      "resource_fields": [...],
      "deleted_interim_fields": [...],
      "target_fields": [...],
      "dimensions": [...],
      "notes": "<short note>"
    }}
  ],
  "risk": "low" | "medium" | "high",
  "overall_notes": "...",
  "confidence": 0.0-1.0
}}

[Output Constraints]
- PURE tool names must match existing Pure tools exactly (cannot invent new names)
- Each PURE tool can only be used once per pipeline
- PURE steps (except cleanup_interim_fields) MUST have non-empty `resource_fields` listing real `original.*` / `tool_memory.*` / `interim.*` paths as required by the schema above; never emit `resource_fields: []` for `web_search`.
- Do NOT include `params`, `tool_args`, or concrete pure-tool invocation payloads in required_operations.
- LLM operation names must be short (3-6 words or snake_case)
- Operations must form a coherent pipeline
- overall_notes is REQUIRED for every item regardless of transformable value:
    - transformable = "yes": describe the source of the ground-truth answer and the key synthesis steps
    - transformable = "no": state the blocker that makes transformation infeasible
- If no grounded path exists → transformable = "no", dataset_type = "none", required_operations = []
- Do not use placeholder or pseudo-operations to make an infeasible dataset look transformable
- You MUST output a list of EXACTLY {n_cards} items
- You MUST only output valid JSON (no extra explanations or text outside JSON)
"""

# --------------------------------------------------------------
# Step1: Transformability Check for subtask candidates (batch)
# --------------------------------------------------------------

def _build_transformability_subtask_json(sub: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(sub.get("id") or "").strip(),
        "name": str(sub.get("name") or "").strip(),
        "description": str(sub.get("description") or "").strip(),
        "task": str(sub.get("task") or "").strip(),
        "answer_type": str(sub.get("answer_type") or "").strip(),
        "sample_schema": sub.get("sample_schema") if isinstance(sub.get("sample_schema"), dict) else {},
        
    }

def _build_tools_list_json(tools_payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools_payload, list):
        return []
    out = []
    for t in tools_payload:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name") or "").strip()
        description = str(t.get("description") or "").strip()
        params = str(t.get("params") or "")
        # typical_uses = str(t.get("typical_uses") or "").strip()
        if name:
            out.append({"name": name, "description": description, "params": params})
    return out


def _clean_transformability_plan(plan: Any) -> Any:
    """
    Remove extra fields that LLMs sometimes add outside the required operation schema.
    """
    if not isinstance(plan, list):
        return plan

    cleaned = []
    for op in plan:
        if not isinstance(op, dict):
            cleaned.append(op)
            continue

        item = dict(op)
        item.pop("pure_tool_call", None)
        item.pop("params", None)
        item.pop("tool_args", None)
        cleaned.append(item)

    return cleaned


def _card_path_to_resource_field(path: str) -> str:
    """Map a meta_structure paths entry like 'input.image_file' to 'original.input.image_file'."""
    if not isinstance(path, str):
        return ""
    p = path.strip()
    if not p:
        return ""
    if p.startswith("original."):
        return p
    if p.startswith("input.") or p.startswith("output."):
        return f"original.{p}"
    return f"original.input.{p}"


def _infer_web_search_resource_fields_from_card(card_json: Dict[str, Any]) -> List[str]:
    """
    When the LLM omits resource_fields for web_search, infer anchors from meta_structure
    (image / audio / video modalities first, then other paths).
    """
    structure = card_json.get("meta_structure") if isinstance(card_json, dict) else None
    if not isinstance(structure, dict):
        return []

    modality_first: List[str] = []
    rest: List[str] = []
    for _role, info in structure.items():
        if not isinstance(info, dict):
            continue
        modality = str(info.get("modality") or "").lower()
        paths = info.get("paths") or []
        if not isinstance(paths, list):
            continue
        for raw in paths:
            full = _card_path_to_resource_field(raw)
            if not full:
                continue
            low = full.lower()
            if any(
                x in modality
                for x in ("image", "audio", "video", "speech", "sound", "visual")
            ) or any(
                x in low
                for x in (
                    "image",
                    "photo",
                    "picture",
                    "pic",
                    "audio",
                    "video",
                    "speech",
                    "wav",
                    "mp3",
                    "png",
                    "jpg",
                    "jpeg",
                )
            ):
                modality_first.append(full)
            else:
                rest.append(full)

    ordered = list(dict.fromkeys(modality_first + rest))
    return ordered[:12]


def _ensure_pure_resource_fields(plan: Any, card_json: Dict[str, Any]) -> Any:
    """Fill empty resource_fields for PURE tools (esp. web_search) when the model omits them."""
    if not isinstance(plan, list):
        return plan
    for op in plan:
        if not isinstance(op, dict):
            continue
        if str(op.get("tool_type") or "").upper() != "PURE":
            continue
        name = str(op.get("operation") or op.get("step_name") or "").strip().lower()
        if name == "cleanup_interim_fields":
            continue
        rf = op.get("resource_fields")
        if isinstance(rf, list) and any(isinstance(x, str) and x.strip() for x in rf):
            continue
        if name == "web_search":
            inferred = _infer_web_search_resource_fields_from_card(card_json)
            if inferred:
                op["resource_fields"] = inferred
    return plan


def _truncate_example(example: Any, max_chars: int = 100) -> Dict[str, Any]:
    """
    Truncate example sample safely and mark as truncated.
    Keeps structure but limits total string length.
    """
    try:
        raw = json.dumps(example, ensure_ascii=False)
    except Exception:
        return {
            "_truncated": True,
            "_reason": "example_not_serializable",
            "preview": str(example)[:max_chars],
        }

    if len(raw) <= max_chars:
        return example

    # re-wrap after truncation
    return {
        "_truncated": True,
        "_reason": "example_too_long",
        "_preview": raw[:max_chars] + "...",
    }

def _build_transformability_card_json(dataset_id: str, card: Dict[str, Any]) -> Dict[str, Any]:
    meta = card.get("meta", {}) or {}
    fields = meta.get("fields", {}) or {}
    meta_desc = meta.get("meta_structure_description", "") or ""

    structure = {}
    for role, info in fields.items():
        if not isinstance(info, dict):
            continue
        path = info.get("paths") or []
        content_extent = info.get("content_extent") or ""
        modality = info.get("modality") or ""
        style = info.get("style") or ""

        structure[role] = {
            "paths": path if isinstance(path, list) else [],
            "modality": str(modality).strip(),
            "content_extent": str(content_extent).strip(),
            "style": str(style).strip(),
        }
    tasks = card.get("tasks", []) or []
    return {
        "dataset_id": dataset_id,
        "name": str(card.get("name") or "").strip(),
        "description": str(card.get("description") or "").strip(),
        "tasks": tasks,
        "meta_desc": meta_desc,
        "meta_structure": structure,
    }




@register_tool("transformability_check")
def transformability_check(
    subtask_id: str,
    batch_size: int = 1,
    model: str = None,
    model_config_path: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4069,
    max_workers: int = 16,
    ctx: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Step1: Transformability Check for ALL dataset candidates of a subtask (batch calls).
    """
    from utils.model_config import get_tool_model
    if model is None:
        model = get_tool_model("transformability_check", model_config_path)
    
    subtasks = ctx.get("subtasks") or []
    subtask = next((st for st in subtasks if str(st.get("id") or "") == str(subtask_id)), None)
    if subtask is None:
        raise ValueError(f"[transformability_check] subtask_id not found: {subtask_id}")
    id2card = ctx.get("id2card")
    retrieval = subtask.get("retrieval_result") or []
    # domain_match = subtask.get("domain_match") or {}
    # tools placeholder (you said keep empty for now)
    # Later you can replace with a structured list of tools.
    tools_payload = ctx.get("tools_list")  or []

    # candidates (stable order, dedup)
    cand_ids: List[str] = []
    seen = set()
    for r in retrieval:
        if not isinstance(r, dict):
            continue
        did = _get_dataset_id_from_retrieval_item(r)
        if not did or did in seen:
            continue
        seen.add(did)
        cand_ids.append(did)

    # filter by domain_match pass
    # def _domain_pass(did: str) -> bool:
    #     m = domain_match.get(did)
    #     return isinstance(m, dict) and bool(m.get("pass"))

    # pass_ids = [did for did in cand_ids if retrieval(did)]

    present_cand_ids = [did for did in cand_ids if did in id2card]
    st_json = _build_transformability_subtask_json(subtask)
    tool_list_json = _build_tools_list_json(tools_payload)

    def _process_batch(id_batch: List[str]) -> Dict[str, Dict[str, Any]]:

        cards_json = [_build_transformability_card_json(did, id2card[did]) for did in id_batch]

        prompt = _TRANSFORMABILITY_CHECK_BATCH_PROMPT.format(
            subtask_json=json.dumps(st_json, ensure_ascii=False, indent=2),
            dataset_cards_json=json.dumps(cards_json, ensure_ascii=False, indent=2),
            n_cards=len(cards_json),
            tools_json=json.dumps(tool_list_json, ensure_ascii=False, indent=2),
        )
        assert len(cards_json) == len(id_batch)

        resp = llm_call_json(
            system_prompt="",
            user_prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not resp.get("ok"):
            raise RuntimeError(f"[llm_transformability_check_subtask_candidates_batch] JSON parse failed: {resp.get('error')}")

        results = resp.get("json") or []
        if not isinstance(results, list) or len(results) != len(cards_json):
            raise ValueError("[llm_transformability_check_subtask_candidates_batch] invalid results length")

        # bind by order; warn if LLM returned a mismatched dataset_id
        part: Dict[str, Dict[str, Any]] = {}
        for i, r in enumerate(results):
            expected_id = id_batch[i]
            if not isinstance(r, dict):
                r = {}
            returned_id = str(r.get("dataset_id") or "").strip()
            if returned_id and returned_id != expected_id:
                print(
                    f"[transformability_check] dataset_id mismatch at position {i}: "
                    f"expected {expected_id!r}, got {returned_id!r}; binding by order"
                )

            can = r.get("transformable")
            plan = r.get("required_operations")

            # Pure tool name check
            plan = _sanitize_pure_tool_ops(
                plan=plan,
                tool_list_json=tool_list_json,
                model=model
            )
            plan = _ensure_pure_resource_fields(plan, cards_json[i])
            plan = _recalibrate_interim_deletions(plan)
            plan = _clean_transformability_plan(plan)

            dataset_type = str(r.get("dataset_type") or r.get("path") or "").strip().lower()
            risk = str(r.get("risk") or "").strip().lower()
            overall_notes = str(r.get("overall_notes") or "").strip()
            confidence = _clamp01(r.get("confidence"))
           

            part[expected_id] = {
                "transformable": can,
                "plan": plan,
                "dataset_type": dataset_type,
                "risk": risk,
                "overall_notes": overall_notes,
                "confidence": confidence,
            }
        return part
    batches = list(_chunk(present_cand_ids, batch_size))
    # _process_batch(batches[0])  # for debug

    transformability, errors = run_batches_concurrent(
        batches,
        _process_batch,
        max_workers=max_workers,
        error_prefix="[transformability_check] batch failed]",
    )
    if errors:
        raise RuntimeError(f"[transformability_check] some batches failed: {errors}")
    
    # Update transformability field in the subtask
    # Note: subtask is a reference to an element in subtasks list, so modifying it
    # directly will update the list element. We update explicitly to ensure consistency.
    subtask["transformability"] = transformability
    
    # Explicitly update the subtask in the list to ensure all fields are preserved
    # This ensures we don't accidentally lose other fields when updating
    subtasks = ctx.get("subtasks") or []
    for i, st in enumerate(subtasks):
        if str(st.get("id") or "") == str(subtask_id):
            # Only update transformability field, preserving all other fields
            subtasks[i].update({
                "transformability": transformability
            })
            break
    ctx["subtasks"] = subtasks
    return ctx, transformability
# --------------------------------------------------------------
# Step2: Scoring for subtask candidates (batch)
# --------------------------------------------------------------
_SCORING_BATCH_PROMPT = r"""
You are the **Dataset Scorer**.

Task: For ONE subtask and a list of (dataset + tool_plan) items, score EACH item as an evaluation candidate.

IMPORTANT:
- You are scoring the COMBINATION: dataset + tool_plan (not the dataset alone).
- Assume the tool_plan will be executed as written.

==================== Inputs ====================

Subtask (JSON, minimal):
{subtask_json}

Candidates (JSON list, size={n_items}):
{items_json}

Each item has:
- dataset_id
- dataset (minimal card info)
- tool_plan (list of steps)
- transformability_notes: summary from the prior transformability assessment — use this to inform your risk and task_alignment scores.

==================== Scoring criteria ====================

For EACH item, output scores in [0,1]:

1) task_alignment:
   **After applying the tool_plan**, how well does the resulting data match the subtask intent?

2) data_coverage:
   Does the dataset content likely cover the subtask domain/scene and variation needed (based on the minimal info given)?

3) transform_simplicity:
   How simple/cheap is the tool_plan? Higher score = simpler pipeline. (more steps, heavier transforms → lower score)
   This is a secondary factor with low influence. Complexity alone should not disqualify a grounded plan.

4) risk:
   Risk that transformation will break labels, change semantics, or introduce artifacts that invalidate evaluation.
   (Higher risk → higher risk score; this dimension is INVERTED — higher is worse)
   This is a secondary factor with low influence. Controlled, meta-consistent synthesis is not risky by default.

5) overall:
   Your overall suitability score for this dataset+plan combination.
   task_alignment and data_coverage are the dominant factors.
   transform_simplicity and risk are weaker factors.

Rules:
- Do NOT compare items to each other. Score each independently.
- Keep notes short and concrete.
- Refer to transformability_notes when assessing risk and task_alignment.
- Do not penalize a plan merely because it requires reliable reasoning over dataset-supported facts; penalize only if the reasoning is brittle, subjective, or unsupported.
- A plan using `web_search` can be high-quality when **factual** external knowledge—often specialized or professional—complements the dataset, adds grounded insight for sample transformation, or fills factual gaps the card does not spell out.
- Treat credible `web_search` evidence as strong grounding for **objective fact-type** use (entities, artwork, history/culture, geography/time, biographies, relationships, clinical or technical definitions, standards, taxonomy, and other verifiable information). Downgrade if search is used for non-factual or speculative content.
- `web_search` supports optional image inputs (`image_paths`) at sample execution time. **Reward** plans that attach **sample images** when retrieval of **factual** evidence is plausibly better with pixels (identity, fine-grained class, place/object recognition, chart/diagram reading supported by public facts). Penalize text-only `web_search` when the card clearly provides images that would materially improve grounded fact retrieval but the plan omits them from the evidence path.
- Penalize `web_search` when it is redundant with dataset fields/meta, vague or unverifiable, or used to smuggle non-factual or subjective content instead of stable facts.
- Prefer plans where `web_search` targets `tool_memory.web_search.<operation_key>` and later LLM steps consume `tool_memory.*.output.*` (or needed `.input.*`) under leakage constraints.

==================== Output ====================
Return ONLY valid JSON:(LIST)

Each item:

{{
    "dataset_id": "string (copy from input item)",
    "scores": {{
    "task_alignment": 0.0-1.0,
    "data_coverage": 0.0-1.0,
    "transform_simplicity": 0.0-1.0,
    "risk": 0.0-1.0,
    "overall": 0.0-1.0
    }},
    "notes": "short explanation"
}}


Constraints:
- Output exactly {n_items} items in the SAME ORDER as input.
- dataset_id MUST match the corresponding input item.
No extra text.
"""

def _build_scoring_subtask_json(sub: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(sub.get("id") or "").strip(),
        "name": str(sub.get("name") or "").strip(),
        "description": str(sub.get("description") or "").strip(),
        "answer_type": str(sub.get("answer_type") or "").strip(),
        "sample_schema": sub.get("sample_schema") if isinstance(sub.get("sample_schema"), dict) else {},
    }

def _build_scoring_card_json(dataset_id: str, card: Dict[str, Any]) -> Dict[str, Any]:
    example = {}
    data_source = card.get("raw_meta", {}).get("source_json") if isinstance(card.get("raw_meta"), dict) else None
    if data_source:
        try:
            with open(data_source, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = raw.get("data")
            example = random.choice(data) if isinstance(data, list) and len(data) > 0 else {}
        except Exception:
            example = {}
    example = _truncate_example(example, max_chars=800)

    return {
        "name": str(card.get("name") or "").strip(),
        "io_schemas": card.get("io_schemas") if isinstance(card.get("io_schemas"), list) else [],
        "description": str(card.get("description") or "").strip(),
        "example_sample": example,
        "capability": card.get("capability") if isinstance(card.get("capability"), dict) else {},
    }

def _normalize_scores(obj: Any) -> Dict[str, float]:
    obj = obj if isinstance(obj, dict) else {}
    transform_simplicity = obj.get("transform_simplicity") if "transform_simplicity" in obj else obj.get("transform_cost")
    return {
        "task_alignment": _clamp01(obj.get("task_alignment")),
        "data_coverage": _clamp01(obj.get("data_coverage")),
        "transform_simplicity": _clamp01(transform_simplicity),
        "risk": _clamp01(obj.get("risk")),
        "overall": _clamp01(obj.get("overall")),
    }


def _recompute_overall_with_high_recall_bias(scores: Dict[str, float]) -> float:
    """
    Downweight simplicity/risk so high-recall, meta-consistent plans are not over-filtered.
    """
    task_alignment = _clamp01(scores.get("task_alignment"))
    data_coverage = _clamp01(scores.get("data_coverage"))
    transform_simplicity = _clamp01(scores.get("transform_simplicity"))
    risk = _clamp01(scores.get("risk"))
    return _clamp01(
        0.45 * task_alignment
        + 0.35 * data_coverage
        + 0.10 * transform_simplicity
        + 0.10 * (1.0 - risk)
    )

@register_tool("scoring")
def scoring(
    subtask_id: str,
    batch_size: int = 1,
    model: str = None,
    model_config_path: str = None,
    temperature: float = 0.0,
    max_tokens: int = 5000,
    max_workers: int = 16,
    therehold: float = 0.53,
    ctx: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Step2: Scoring (LLM #2) for ALL candidates of a subtask, scoring dataset + tool_plan.
    """
    from utils.model_config import get_tool_model
    if model is None:
        model = get_tool_model("scoring", model_config_path)
    
    subtasks = ctx.get("subtasks") or []
    subtask = next((st for st in subtasks if str(st.get("id") or "") == str(subtask_id)), None)
    if subtask is None:
        raise ValueError(f"[scoring] subtask_id not found: {subtask_id}")
   
    id2card = ctx.get("id2card") or {}
    retrieval = subtask.get("retrieval_result") or []
    transformability = subtask.get("transformability") or {}
   
    # collect candidate ids (stable order, dedup)
    cand_ids: List[str] = []
    seen = set()
    for r in retrieval:
        if not isinstance(r, dict):
            continue
        did = _get_dataset_id_from_retrieval_item(r)
        if not did or did in seen:
            continue
        seen.add(did)
        cand_ids.append(did)

    def _can_transform(did: str) -> bool:
        t = transformability.get(did)
        return isinstance(t, dict) and str(t.get("transformable") or "").strip().lower() == "yes"

    # filter: domain pass AND can_transform
    eligible_ids = [did for did in cand_ids if _can_transform(did) and did in id2card]
    st_json = _build_scoring_subtask_json(subtask)

    def _process_batch(id_batch: List[str]) -> Dict[str, Dict[str, Any]]:
        items = []
        for did in id_batch:
            card_json = _build_scoring_card_json(did, id2card[did])
            plan = transformability.get(did, {}).get("plan")
            transformability_notes = transformability.get(did, {}).get("overall_notes") or ""
            if not isinstance(plan, list):
                plan = []
            items.append({
                "dataset_id": did,
                "dataset": card_json,
                "tool_plan": plan,
                "transformability_notes": transformability_notes,
            })

        prompt = _SCORING_BATCH_PROMPT.format(
            subtask_json=json.dumps(st_json, ensure_ascii=False, indent=2),
            items_json=json.dumps(items, ensure_ascii=False, indent=2),
            n_items=len(items),
        )

        resp = llm_call_json(
            system_prompt="",
            user_prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not resp.get("ok"):
            raise RuntimeError(f"[scoring] JSON parse failed: {resp.get('error')}")

        results = resp.get("json") or []
        if not isinstance(results, list) or len(results) != len(items):
            raise ValueError("[scoring] invalid results length")
        part: Dict[str, Dict[str, Any]] = {}
        for i, r in enumerate(results):
            did = id_batch[i]
            if not isinstance(r, dict):
                r = {}
            returned_id = str(r.get("dataset_id") or "").strip()
            if returned_id and returned_id != did:
                print(
                    f"[scoring] dataset_id mismatch at position {i}: "
                    f"expected {did!r}, got {returned_id!r}; binding by order"
                )
            scores = _normalize_scores(r.get("scores"))
            llm_overall = scores.get("overall", 0.0)
            weighted_overall = _recompute_overall_with_high_recall_bias(scores)
            # Keep model judgment while enforcing high-recall weighting policy.
            scores["overall"] = _clamp01(0.3 * llm_overall + 0.7 * weighted_overall)
            notes = str(r.get("notes") or "").strip()
            part[did] = {"scores": scores, "notes": notes}
        # print("[scoring] processed batch:", id_batch)
        return part
    
    if not eligible_ids:
        print(f"[scoring] subtask {subtask_id}: no eligible candidates after transformability filter, skipping scoring.")

    batches = list(_chunk(eligible_ids, batch_size))
    score_dict, error = run_batches_concurrent(
        batches,
        _process_batch,
        max_workers=max_workers,
        error_prefix="[scoring] batch failed]",
    )
    if error:
        raise RuntimeError(f"[scoring] some batches failed: {error}")
    qualify_candidates = {}
    for did, score_info in score_dict.items():
        overall_score = score_info.get("scores", {}).get("overall", 0.0)
        if overall_score >= therehold:
            qualify_candidates[did] = score_info

    subtask["scoring"] = score_dict
    subtask["scored_candidates"] = qualify_candidates
    subtask["scored_status"] = "yes"
    ctx["subtasks"] = subtasks
    return ctx, qualify_candidates