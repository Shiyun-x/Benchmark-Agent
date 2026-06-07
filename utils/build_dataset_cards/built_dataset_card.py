# benchmark_agent/pipelines/build_dataset_card.py

import asyncio
import json
import os
import hashlib
from pathlib import Path
import re
import argparse
from typing import Dict, Any, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import os
import tqdm
import litellm

from utils.schema.dataset_card import DatasetCard, IOSchema
from utils.build_dataset_cards.dataset_card_tools import (
    load_closed_sets,
    read_dataset_json,
    infer_modalities,
    infer_io_schema,
    sample_text_pairs,
    count_samples,
)
from utils.llm_caller import llm_call_json


DEFAULT_CONFIG_PATH = Path(__file__).with_name("built_dataset_card.config.json")


def detect_type(v, detect_list_item_type=True):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        if detect_list_item_type and len(v) > 0:
            first_item_type = detect_type(v[0], detect_list_item_type=False)
            return f"list<{first_item_type}>"
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


_index_pattern = re.compile(r"^(?P<key>[^\[\]]+)\[(?P<idx>\d+)\]$")
_empty_bracket_pattern = re.compile(r"^(?P<key>[^\[\]]+)\[\]$")


def get_value_by_path(sample, path_str):
    cur = sample
    tokens = path_str.split(".")
    
    for i, token in enumerate(tokens):
        if cur is None:
            return None

        # numeric list index token, e.g. '0'
        if isinstance(cur, list) and token.isdigit():
            idx = int(token)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
            continue

        m_empty = _empty_bracket_pattern.match(token)
        if m_empty:
            key = m_empty.group("key")
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
            if not isinstance(cur, list):
                return None
            if i < len(tokens) - 1 and len(cur) > 0:
                cur = cur[0]
            else:
                return cur
            continue

        m = _index_pattern.match(token)
        if m:
            key = m.group("key")
            idx = int(m.group("idx"))
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
            if not isinstance(cur, list) or not (0 <= idx < len(cur)):
                return None
            cur = cur[idx]
            continue

        if isinstance(cur, dict):
            cur = cur.get(token)
        else:
            return None

    return cur


def load_one_sample_from_raw(raw):
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list) and raw["data"]:
        return raw["data"][0]
    if isinstance(raw, list) and raw:
        return raw[0]
    raise ValueError("Cannot find sample in raw data")


def add_path_types_to_card(card: DatasetCard, raw: Dict[str, Any]) -> DatasetCard:
    meta = card.meta or {}
    fields = meta.get("fields")
    if not isinstance(fields, dict):
        return card

    try:
        sample = load_one_sample_from_raw(raw)
    except Exception as e:
        print(f"  Warning: cannot load sample for type detection: {e}")
        return card

    for field_name, field_cfg in fields.items():
        paths = field_cfg.get("paths")
        if not isinstance(paths, list):
            continue

        path_types = []
        for p in paths:
            if not isinstance(p, str):
                path_types.append("invalid_path")
                continue
            value = get_value_by_path(sample, p)
            t = detect_type(value, detect_list_item_type=True)
            path_types.append(t)

        field_cfg["path_types"] = path_types

    card.meta = meta
    return card 


def resolve_source_path(raw_path: str, source_root: Optional[str]) -> str:
    """
    If `raw_meta.source_json` is relative, resolve it against `source_root`.
    """
    if not source_root:
        return raw_path
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(source_root, raw_path)


def update_path_types_for_cards(
    card_dir: str,
    source_root: Optional[str] = None,
) -> None:
    """
    Batch-update `meta.fields[*].path_types` for existing `*_card.json` files.
    """
    updated = 0

    if not os.path.isdir(card_dir):
        raise NotADirectoryError(f"card_dir not found: {card_dir}")

    card_names = sorted(os.listdir(card_dir))
    for name in card_names:
        if not name.endswith("_card.json"):
            continue

        card_path = os.path.join(card_dir, name)
        if not os.path.isfile(card_path):
            continue

        try:
            with open(card_path, "r", encoding="utf-8") as f:
                card_payload = json.load(f)
        except Exception as e:
            print(f"Skip {name}: cannot load json: {e}")
            continue

        meta = card_payload.get("meta") or {}
        raw_meta = card_payload.get("raw_meta") or {}
        fields = meta.get("fields")
        if not isinstance(fields, dict):
            continue

        source_json_path = raw_meta.get("source_json")
        if not source_json_path or not isinstance(source_json_path, str):
            continue

        resolved_source_path = resolve_source_path(source_json_path, source_root)
        if not os.path.exists(resolved_source_path):
            print(f"Skip {name}: source_json not found: {resolved_source_path}")
            continue

        try:
            with open(resolved_source_path, "r", encoding="utf-8") as f:
                raw_payload = json.load(f)
            sample = load_one_sample_from_raw(raw_payload)
        except Exception as e:
            print(f"Skip {name}: cannot load sample from {resolved_source_path}: {e}")
            continue

        # Update each field's path_types based on the same sample.
        for _, field_cfg in fields.items():
            if not isinstance(field_cfg, dict):
                continue

            paths = field_cfg.get("paths")
            if not isinstance(paths, list):
                continue

            path_types: List[str] = []
            for p in paths:
                if not isinstance(p, str):
                    path_types.append("invalid_path")
                    continue
                value = get_value_by_path(sample, p)
                path_types.append(detect_type(value, detect_list_item_type=True))

            field_cfg["path_types"] = path_types

        card_payload["meta"] = meta
        try:
            with open(card_path, "w", encoding="utf-8") as f:
                json.dump(card_payload, f, ensure_ascii=False, indent=2)
            updated += 1
            print(f"Updated path_types: {name}")
        except Exception as e:
            print(f"Skip {name}: cannot write file: {e}")

    print(f"update_path_types_for_cards done. Updated {updated} cards in {card_dir}")


# ================================
# dataset_id
# ================================

def make_dataset_id(path: str) -> str:
    """
    Generate a dataset_id based on the last 4 parts of the file path + hash.
    """
    p = Path(path)
    parts = p.parts
    suffix = "/".join(parts[-4:]) if len(parts) > 4 else str(p)
    suffix = suffix.replace("General-Bench-Openset/", "")
    if "comprehension/" in suffix:
        suffix = suffix.replace("comprehension/", "")
    if "json/" in suffix:
        suffix = suffix.replace("json/", "")
    print(suffix)
    hash_code = hashlib.md5(suffix.encode("utf-8")).hexdigest()[:8]
    return f"{hash_code}"


# ================================
# LLM prompts
# ================================

DATASET_CARD_SYSTEM_PROMPT = """
You are a dataset card author for a benchmark-construction system.

Your job is to output a structured dataset card annotation in STRICT JSON.

You will be given:
- basic dataset info (id, name, modalities, io_schemas, size, other tags)
- a few raw sample items (as in the original JSON)
- a closed set of task labels (optional)

From these, you must:
1) Describe what the dataset is about (description, card_text).
2) Optionally assign task labels (if they clearly fit the samples).
3) Assign a short free-text domain tag.
4) Describe, in detail, the META STRUCTURE of the samples:
   - what canonical components exist (context, question, answer, dialog, options, bbox...),
   - where they live in the JSON (field paths),
   - what they semantically look like (content_extent, style, content type).
"""
STYLE_SCHEMA = {
        "text": {
            "narrative": "stories, events unfolding in time, personal experiences",
            "factual": "encyclopedic, expository, knowledge-style descriptions",
            "dialog": "multi-turn conversational content",
            "instruction": "explicit instructions, prompts, commands, or tasks",
            "structured": "structured data, tables, lists, or other organized formats",
            "mixed": "a mixture of the above",
            "other": "none of the above clearly fits",
        },
        "image":{
            "scenery": "natural landscapes, outdoor scenes, cityscapes",
            "objects": "everyday objects, items, or products",
            "screenshot-UI": "user interfaces, app screens, or website snapshots",
            "illustration": "drawings, cartoons, or artistic illustrations",
            "diagram": "technical diagrams, charts, graphs, or schematics",
            "mixed": "a mixture of the above",
            "other": "none of the above clearly fits",
        },
        "audio": {
            "speech": "spoken language, conversations, or monologues",
            "music": "musical pieces, songs, or instrumental tracks",
            "environmental": "ambient sounds, nature sounds, or urban soundscapes",
            "mixed": "a mixture of the above",
            "other": "none of the above clearly fits",
        },
}

CONTENT_EXTENT_SCHEMA = {
    "labels": ["short", "medium", "long", "mixed"],
    "by_modality": {
        "text": {
            "short": "short phrases or single sentences (roughly < 50 tokens)",
            "medium": "a few sentences or one short paragraph (roughly 50-200 tokens)",
            "long": "multi-paragraph or article-level content (roughly > 200 tokens)",
            "mixed": "strongly varying text lengths across samples",
        },
        "audio": {
            "short": "very short clips or single utterances",
            "medium": "short dialogs or clips with several utterances",
            "long": "long recordings, meetings, or extended conversations",
            "mixed": "strongly varying durations across samples",
        },
        "image": {
            "short": "simple content (single object, very simple scene)",
            "medium": "moderately complex scenes with several elements",
            "long": "very complex scenes with many elements or sequences",
            "mixed": "strongly varying visual complexity across samples",
        },
    },
}

def build_user_prompt_for_card(ctx: Dict[str, Any], modelity: str) -> str:
    ds_id = ctx.get("dataset_id", "")
    ds_name = ctx.get("dataset_name", "")
    modalities = ctx.get("modalities", []) or []
    io_schemas = ctx.get("io_schemas", None)
    size_samples = ctx.get("size_samples", 0)
    samples = ctx.get("sample_items", [])
    existing = ctx.get("existing_tags", {})
    closed_tasks = ctx.get("closed_tasks", []) or []

    canonical_roles = [
        "context", "question", "answer", "rationale",
        "dialog", "options",
        "image", "audio", "category",
        "bbox", "mask", "point",
        "entity", "span", "table",
        "other_aux",
    ]
    # ==== 1) compute relevant modalities for this batch (text is always included) ====
    #  "image" / "audio" / "text"
    main_modality = modelity or (modalities[0] if modalities else "text")

    modalities_set = set()
    modalities_set.add("text")  # always include text schema
    for m in modalities:
        if isinstance(m, str):
            modalities_set.add(m)
    if isinstance(main_modality, str):
        modalities_set.add(main_modality)

    # shrink schema to relevant modalities only (text always kept)
    style_schema_for_prompt = {m: STYLE_SCHEMA[m]
                               for m in modalities_set
                               if m in STYLE_SCHEMA}

    # content_extent
    extent_by_modality = {
        m: CONTENT_EXTENT_SCHEMA["by_modality"][m]
        for m in modalities_set
        if m in CONTENT_EXTENT_SCHEMA["by_modality"]
    }
    extent_schema_for_prompt = {
        "labels": CONTENT_EXTENT_SCHEMA["labels"],
        "by_modality": extent_by_modality,
    }

    return f"""
====================
DATASET CONTEXT
====================
- dataset_id: {ds_id}
- name: {ds_name}
- size_samples: {size_samples}
- modalities: {modalities}
- io_schemas: {io_schemas}
- existing_tags (may be noisy): {existing}

A small random sample of items (raw examples, as in the JSON file):
{json.dumps(samples[:2], ensure_ascii=False, indent=2)}

====================
CLOSED TASK SET (OPTIONAL)
====================
You MAY assign zero or more tasks from this closed set if they clearly fit:
CLOSED TASKS: {closed_tasks}

If no task matches well, you can return an empty list.

====================
META FIELDS SCHEMA
====================
We use a CLOSED SET of canonical roles (keys in the "fields" object):

canonical roles: {canonical_roles}

Their meaning:
- "context": main input content the model reads (passage, story, article, prompt, document, etc.)
- "question": query/instruction that asks about the context
- "answer": target label or text answer
- "rationale": explanation / reasoning text for the answer
- "dialog": multi-turn conversation (sequence of utterances, often with speaker roles)
- "options": multiple-choice options
- "image": image reference (path/id/url)
- "audio": audio reference (path/id/url)
- "category": category labels or tags
- "bbox": bounding boxes for visual objects
- "mask": segmentation masks for visual regions
- "point": point annotations for visual content
- "entity": structured entities or spans with types
- "span": explicit span indices or extracted spans used for answers/annotations
- "table": tabular data or structured tables
- "<other_aux>": other auxiliary fields not covered above but sometimes useful, please name them yourself.

DO NOT include:
    - ids, sample_id, uuid, index
    - split indicators (train/val/test)
    - timestamps
    - version, filename, data source
    - keys without semantic meaning
    Ignore such fields completely and DO NOT explain them.

For each role you include, you MUST provide:
- "paths": list of dot-separated field paths in the sample JSON that consistently play this role.
- "modality": one of "text", "image", "audio", "mixed", "unknown".
- "content_extent": one of {CONTENT_EXTENT_SCHEMA["labels"]}, whose meaning depends on the field's modality.
  Use the following guidance:
  {json.dumps(extent_schema_for_prompt, ensure_ascii=False, indent=2)}

- "style": a style label chosen from the STYLE_SCHEMA for this field's modality.
  The available style schemas are:
  {json.dumps(style_schema_for_prompt, ensure_ascii=False, indent=2)}

- "semantics": 1-3 English sentences describing:
    * what information this field contains,
    * what it is about,
    * how it relates to other fields.

IMPORTANT:
- If a field has modality "text", you MUST use the "text" styles.
- If a field has modality "image" or "audio", you MUST use the corresponding modality-specific styles.

IMPORTANT for "context":
- If there is any field that behaves as the main context, you MUST include a "context" role.

====================
OUTPUT JSON SCHEMA
====================
You MUST respond with a SINGLE JSON object of the exact form:

{{
  "description": "<1-3 concise English sentences describing what the dataset is about and what it evaluates>",
  "card_text": "<50-200 English words, more detailed description (modality, data style, typical content, assumptions)>",

  "tasks": ["<task_from_closed_tasks>", ...],

  "domain": "<short free-text domain tag, e.g. 'medical', 'legal', 'everyday life stories', 'news', 'programming'>",

  "meta": {{
    "fields": {{
      "context": {{
        "paths": ["context", "raw_text"],
        "modality": "text",
        "content_extent": "long",
        "style": "narrative",
        "semantics": "..."
      }},
      "question": {{
        "paths": ["question"],
        "modality": "text",
        "content_extent": "short",
        "style": "instruction",
        "semantics": "..."
      }},
      "answer": {{
        "paths": ["answer"],
        "modality": "text",
        "content_extent": "short",
        "style": "other",
        "semantics": "..."
      }}
      // etc. Only include roles that truly exist.
    }},
    "meta_structure_description": "<ONE short paragraph that ONLY describes which components exist in each sample
                                   (context/question/answer/dialog/options/bbox/mask/...),
                                   what information they carry,
                                   and how they relate to each other.
                                   Do NOT talk about training or benchmarks here.>"
  }}
}}

====================
HARD RULES
====================
- "tasks" MUST be chosen from the CLOSED TASKS set, or [] if nothing fits.
- "domain" is a short free-text tag summarizing the main domain (e.g. 'medical QA', 'Wikipedia QA', 'multi-domain dialog').
- In "meta.fields":
  - keys MUST be canonical roles,
  - "paths" MUST correspond to real fields in the sample JSON (no fake paths),
  - you MUST give a meaningful "semantics" text, especially for "context".
- The response MUST be valid JSON; NO extra commentary, NO trailing commas.
"""


def call_llm_for_dataset_card(ctx: Dict[str, Any], model: str = "gpt-5-mini", modality: str = "text") -> Dict[str, Any]:
    system_prompt = DATASET_CARD_SYSTEM_PROMPT.strip()
    user_prompt = build_user_prompt_for_card(ctx, modality)


    result = llm_call_json(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    if isinstance(result, str):
        result = json.loads(result)
    return result


# ================================
# Build a single DatasetCard
# ================================

def build_single_card(
    dataset_id: str,
    json_path: str,
    tasks_yaml: str,
    scenario_yaml: str, 
    model: str = "gpt-5-mini",
    modality = "text",
) -> DatasetCard:
    raw = read_dataset_json(json_path)

    filename = Path(json_path).stem
    dataset_name = raw.get("name") or filename.replace("_", " ").title()
    if dataset_name.lower() == "annotation" or dataset_name.lower() == "annotation v2" or dataset_name.lower() == "annotation v1" or dataset_name.lower() == "annotations":
        dataset_name = Path(json_path).parent.stem.replace("_", " ").title()
    if dataset_name.lower() == "json":
        # fall back to parent directory name
        dataset_name = Path(json_path).parent.parent.stem.replace("_", " ").title()
    modalities = infer_modalities(raw, main_modality=modality)
    io_schemas = infer_io_schema(raw, main_modality=modality)
    samples = sample_text_pairs(raw, k=12)
    size = count_samples(raw)
    closed = load_closed_sets(tasks_yaml, scenario_yaml)
    closed_tasks = (closed.get("tasks") or {}).get("canonical", []) or []

    existing_tags = {
        "task": raw.get("task"),
        "type": raw.get("type"),
        "skill": raw.get("skill"),
        "domain": raw.get("domain"),
        "general_capability": raw.get("general_capability"),
        "set_type": raw.get("set_type"),
    }
    # TODO: ctx fields vary by modality (e.g. audio adds description; image adds skill, domain, general_capability)
    ctx = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "modalities": modalities,
        "io_schemas": io_schemas,
        "size_samples": size,
        "sample_items": samples,
        "existing_tags": existing_tags,
        "closed_tasks": closed_tasks,
    }

    result = call_llm_for_dataset_card(ctx, model=model,modality = modality,)
    result = result.get("json")
    meta = result.get("meta") or {}
    if "fields" not in meta:
        meta["fields"] = {}
    if "meta_structure_description" not in meta:
        meta["meta_structure_description"] = ""

    card = DatasetCard(
        dataset_id=ctx["dataset_id"],
        name=ctx["dataset_name"],
        modalities=modalities,
        io_schemas=[IOSchema(**io) for io in io_schemas] if io_schemas else None,
        size_samples=size,

        description=(result.get("description", "") or "").strip(),
        card_text=(result.get("card_text", "") or "").strip(),
        tasks=result.get("tasks", []) or [],
        domain=result.get("domain", "") or "",

        meta=meta,

        raw_meta={
            "source_json": str(Path(json_path)),
            "existing_tags": {k: v for k, v in existing_tags.items() if v},
            "original_raw_domain": raw.get("domain"),
        },
    )

    # attach type info to each path in meta.fields
    card = add_path_types_to_card(card, raw)

    return card


# ================================
# File traversal & main
# ================================

def find_json_files(
    root_dir: str,
    include_patterns=(".json",),
    exclude_dirs=(".cache", "hf_cache", "dataset_cards"),
) -> List[str]:
    root = Path(root_dir)
    files: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in include_patterns:
            if any(part in exclude_dirs for part in p.parts):
                continue
            files.append(str(p))
    return files


def _resolve_config_file_path(raw_path: Optional[str], config_dir: Path) -> Optional[str]:
    if raw_path is None:
        return None
    candidate = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if candidate.is_absolute():
        return str(candidate)
    return str((config_dir / candidate).resolve())


def load_runtime_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")

    build_cfg = payload.get("build")
    if not isinstance(build_cfg, dict):
        raise ValueError("Config file must contain a 'build' object.")

    path_types_cfg = payload.get("path_types") or {}
    if not isinstance(path_types_cfg, dict):
        raise ValueError("'path_types' must be a JSON object.")

    required_keys = ["root_dir", "modality", "out_dir", "tasks_yaml", "scenario_yaml"]
    missing_keys = [key for key in required_keys if not build_cfg.get(key)]
    if missing_keys:
        raise ValueError(f"Missing required build config keys: {missing_keys}")

    config_dir = config_file.parent
    out_dir = _resolve_config_file_path(build_cfg["out_dir"], config_dir)
    max_workers = int(build_cfg.get("max_workers", 16))
    if max_workers <= 0:
        raise ValueError("'max_workers' must be a positive integer.")

    return {
        "build": {
            "root_dir": _resolve_config_file_path(build_cfg["root_dir"], config_dir),
            "modality": build_cfg["modality"],
            "out_dir": out_dir,
            "tasks_yaml": _resolve_config_file_path(build_cfg["tasks_yaml"], config_dir),
            "scenario_yaml": _resolve_config_file_path(build_cfg["scenario_yaml"], config_dir),
            "model": build_cfg.get("model", "gpt-5.1"),
            "max_workers": max_workers,
        },
        "path_types": {
            "card_dir": _resolve_config_file_path(path_types_cfg.get("card_dir"), config_dir) or out_dir,
            "source_root": _resolve_config_file_path(path_types_cfg.get("source_root"), config_dir),
        },
    }


def _worker(json_path, out_dir, tasks_yaml, scenario_yaml, modality, model):
    try:
        dataset_id = make_dataset_id(json_path)
        target_path = os.path.join(out_dir, f"{dataset_id}_card.json")

        card = build_single_card(
            dataset_id=dataset_id,
            json_path=json_path,
            tasks_yaml=tasks_yaml,
            scenario_yaml=scenario_yaml,
            model=model,
            modality=modality,
        )

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(card.model_dump_json(indent=2, ensure_ascii=False))

        return True

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False
        
def main_build(build_config: Dict[str, Any]):
    root_dir = build_config["root_dir"]
    modality = build_config["modality"]
    out_dir = build_config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tasks_yaml = build_config["tasks_yaml"]
    scenario_yaml = build_config["scenario_yaml"]
    model = build_config["model"]

    json_files = find_json_files(root_dir)
    json_files = json_files

    max_workers = build_config["max_workers"]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _worker,
                json_path,
                out_dir,
                tasks_yaml,
                scenario_yaml,
                modality,
                model,
            )
            for json_path in json_files
        ]

        for _ in tqdm.tqdm(as_completed(futures), total=len(futures)):
            pass

    # close litellm async clients once after all threads finish
    if hasattr(litellm, "close_litellm_async_clients"):
        litellm.close_litellm_async_clients()


def main():
    """
    Default behavior:
    1) build dataset cards (slow, uses LLM)
    2) update meta.fields[*].path_types for all existing *_card.json after build

    Notes:
    - Runtime paths are read from the config file next to this script by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the JSON config file.",
    )
    args = parser.parse_args()
    runtime_config = load_runtime_config(args.config)

    # 1) build cards
    main_build(runtime_config["build"])

    # 2) update types as a stage after build
    update_path_types_for_cards(
        card_dir=runtime_config["path_types"]["card_dir"],
        source_root=runtime_config["path_types"]["source_root"],
    )

if __name__ == "__main__":
    main()
