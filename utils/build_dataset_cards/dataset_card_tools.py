# benchmark_agent/tools/dataset_card_tools.py
# -----------------------------------------------------------------------------
# This module provides utility functions for processing dataset cards,
# including loading closed sets, canonicalizing values, reading dataset JSON files,
# inferring modalities and I/O schemas, sampling text pairs, and counting samples.
# -----------------------------------------------------------------------------
import json, random, re
from typing import Dict, Any, List, Tuple, Optional
import yaml

ASCII_RE = re.compile(r'^[\x00-\x7F]+$')

def load_closed_sets(tasks_yaml: str, scenario_yaml: str) -> Dict[str, Dict[str, Any]]:
    with open(tasks_yaml, "r", encoding="utf-8") as f:
        tasks = yaml.safe_load(f)
    with open(scenario_yaml, "r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)
    return {"tasks": tasks, "scenario": scenario}

def canonicalize(value: str, closed: Dict[str, Any]) -> Optional[str]:

    if not value:
        return None
    v = value.strip().lower()

    # exact match first
    for c in closed.get("canonical", []):
        if v == c.lower():
            return c

    # match aliases (values are lists or sets)
    alias_map = closed.get("aliases", {}) or {}
    for canon, alset in alias_map.items():
        if v in [a.lower() for a in alset]:
            return canon

    return None


def read_dataset_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
from typing import Dict, Any, List, Optional

ALLOWED_MODALITIES = {"text", "image", "audio"}

MODALITY_ALIASES = {
    # text
    "text": "text",
    "texts": "text",
    "question": "text",
    "questions": "text",
    "caption": "text",
    "captions": "text",
    "prompt": "text",
    "prompts": "text",

    # image / vision
    "image": "image",
    "images": "image",
    "image_list": "image",
    "image_lists": "image",
    "image_file": "image",
    "image_files": "image",
    "frame": "image",
    "frames": "image",
    "video": "image",  

    # audio
    "audio": "audio",
    "audios": "audio",
    "audio_file": "audio",
    "audio_files": "audio",
    "wav": "audio",
    "flac": "audio",
    "mp3": "audio",

}

def _normalize_modalities(raw_list: List[Any]) -> List[str]:
    out = set()
    for x in raw_list:
        if not x:
            continue
        x_str = str(x).lower()
        base = MODALITY_ALIASES.get(x_str, x_str)
        if base in ALLOWED_MODALITIES:
            out.add(base)
    return sorted(out)



def infer_modalities(example: Dict[str, Any],
                     main_modality: Optional[str] = None) -> List[str]:
    # 1. only rely on data itself
    raw_tokens: List[Any] = []

    m = example.get("modality")
    if isinstance(m, dict):
        ins = m.get("in") or m.get("in_") or []
        outs = m.get("out") or []
        if not isinstance(ins, (list, tuple)):
            ins = [ins]
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        raw_tokens.extend(ins)
        raw_tokens.extend(outs)
    else:
        io = example.get("data") or example.get("samples") or []
        if io:
            inp = (io[0].get("input") or {})
            raw_tokens.extend(list(inp.keys()))
            if any(k in inp for k in ["image_file", "image_files", "images", "image_list"]):
                raw_tokens.append("image")
            if any(k in inp for k in ["audio_file", "audio_files"]):
                raw_tokens.append("audio")
            if any(k in inp for k in ["text", "prompt", "question", "caption"]):
                raw_tokens.append("text")

    mods = _normalize_modalities(raw_tokens)

    # 2. main_modality as the "lowest guarantee": only add, not delete
    if main_modality:
        extra = _normalize_modalities([main_modality])
        for m in extra:
            if m not in mods:
                mods.append(m)

    return sorted(set(mods))


def infer_io_schema(example: Dict[str, Any],
                    main_modality: Optional[str] = None,
                    default_output: str = "text"
                    ) -> Optional[List[Dict[str, List[str]]]]:
    m = example.get("modality")

    if isinstance(m, dict):
        ins = m.get("in") or m.get("in_") or []
        outs = m.get("out") or []
        if not isinstance(ins, (list, tuple)):
            ins = [ins]
        if not isinstance(outs, (list, tuple)):
            outs = [outs]

        ins_norm = _normalize_modalities(list(ins))
        outs_norm = _normalize_modalities(list(outs))

        # main_modality as the lowest guarantee for input
        if main_modality:
            extra_in = _normalize_modalities([main_modality])
            for mm in extra_in:
                if mm not in ins_norm:
                    ins_norm.append(mm)

        if not ins_norm and not outs_norm:
            return None

        if not outs_norm:
            outs_norm = _normalize_modalities([default_output]) or ["text"]

        return [{"in": ins_norm, "out": outs_norm}]

    # no explicit modality, use infer_modalities + default output
    mods = infer_modalities(example, main_modality=main_modality)
    if not mods:
        return None

    outs_norm = _normalize_modalities([default_output]) or ["text"]
    return [{"in": mods, "out": outs_norm}]


def sample_text_pairs(example: Dict[str, Any], k: int = 40) -> List[Tuple[str, str]]:
    data = example.get("data", [])
    if not isinstance(data, list): return []
    n = min(k, len(data))
    picks = random.sample(data, n) if len(data) > n else data[:n]

    return picks

def count_samples(example: Dict[str, Any]) -> int:
    data = example.get("data", [])
    return len(data) if isinstance(data, list) else 0