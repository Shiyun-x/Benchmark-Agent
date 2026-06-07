# tools/dataset_preference_tools.py

from typing import Dict, Any, List
import json

from utils.llm_caller import llm_call_json  

CANONICAL_ROLES: List[str] = [
    "context", "question", "answer", "rationale",
    "dialog", "options",
    "image", "audio", "category",
    "bbox", "mask", "point",
    "entity", "span", "table"]

PREFERENCE_MODEL = "gpt-5.1"
# =============================
# System prompt
# =============================
DATASET_PREFERENCE_SYSTEM_PROMPT = r"""
You are assisting a benchmark construction system.

Given a subtask, your job is to describe what kinds of dataset characteristics would make a dataset POTENTIALLY useful for building evaluation QA samples for that subtask.

Here, a subtask is a specific evaluation objective within the benchmark: it defines the capability or behavior to test, the intended input/output schema, and the answer type. It is NOT necessarily the same as a dataset's native task.
A useful dataset may provide raw material, annotations, context, or structure that can be transformed into QA samples for this objective.

====================
SUBTASK CONTEXT
====================
- subtask_id: {sid}
- name: {name}
- description: {desc}
- keywords: {keywords}


Sample schema (the structure of input/output for this subtask):
{sample_schema_str}

Answer type:
{answer_type}

Based on this subtask, propose dataset PREFERENCES (not hard constraints) following the instructions.

IMPORTANT PRINCIPLES:
- This is NOT a strict requirement list.
- Do NOT try to perfectly match the final benchmark task.
- Your job is to describe dataset potentials and affordances.
- Favor HIGH RECALL: include anything that could yield useful data after transformation.
- Do NOT filter out datasets just because their native QA format does not match.

You should reason about two types of usefulness:

Type A (native usefulness):
  Datasets whose existing QA samples are directly helpful with light transformation.

Type B (meta usefulness):
  Datasets whose meta structure (e.g., narrative context, dialog, bbox, segmentation,
  multi-turn conversation, options, etc.) could provide high-quality raw material
  for creating new QA, even if the native QA is not suitable.

You must express your preferences in terms of:
- domains (high-level topics)
- field roles (from a closed set)
- modality
- structural affordances
- semantic affordances

CLOSED SET OF FIELD ROLES
You MUST choose roles ONLY from this closed set (keys in "preferred_roles"):

{canonical_roles}

where:
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

====================
ROLE PREFERENCES:
- You DO NOT need to include all roles.
- Include a role only if having that role in a dataset could be helpful.
- Each role can have:
  - a natural language "reason"
  - style and content_extent preferences (as soft preferences)
  - style means the general tone/format of the content, different modalities may have different styles(e.g., text style: narrative, factual, dialog, instruction; image style: scenery, objects, diagram, etc; audio style: speech, music, environmental, etc.)
  - content_extent means the length/size/complexity of the content (e.g., text length: short, medium, long, mixed; image size: small, medium, large; audio duration: short, medium, long, mixed)
  - an optional "semantics_hint"
  - a "weight" in [0, 1] (how important this role is; 0.0-0.3 = minor, 0.4-0.7 = useful, 0.8-1.0 = very important)
- We have tools to transform between some roles semantically (e.g., "rationale" <-> "answer", "context" <-> "dialog"..), and some modality conversions (e.g., "text" -> "audio" via tts).
- So DO NOT be overly restrictive in role selection.

DOMAINS:
- "preferred_domains" is a list of free-text domain tags that would be especially useful.
- These are soft preferences, NOT hard filters.

RULES:
- Do NOT specify MUST/ONLY/REQUIRED conditions.
- Roles MUST come only from the closed set.
- All fields are soft preferences.
- Do NOT output commentary outside JSON.


====================
OUTPUT FORMAT (STRICT JSON)
====================
You MUST output a SINGLE JSON object of the form:

{{
  "preferred_roles": {{
    "<role>": {{
      "reason": "<why this role helps for this subtask>",
      "style_preferences": ["narrative", "factual", "dialog", "instruction", "other", etc.],
      "content_extent_preferences": ["short", "medium", "long", "mixed"],
      "semantics_hint": "<optional natural-language hint (1-2 sentences)>",
      "weight": 0.0
    }}
  }},

  "preferred_domains": ["<domain>", "..."]

}}
"""


# =============================
# build user prompt from subtask
# =============================

def build_dataset_preference_user_prompt(subtask: Dict[str, Any]) -> str:
    sid = subtask.get("id", "")
    name = subtask.get("name", "") or ""
    desc = subtask.get("description", "") or ""
    keywords = " ".join(subtask.get("keywords", []) or [])
    sample_schema = subtask.get("sample_schema", {}) or {}
    answer_type = subtask.get("answer_type", "") or ""

    sample_schema_str = json.dumps(sample_schema, ensure_ascii=False, indent=2)
    return {
        "sid": sid,
        "name": name,
        "desc": desc,
        "keywords": keywords,
        "sample_schema_str": sample_schema_str,
        "answer_type": answer_type,
    }

# =============================
# main entry: plan_dataset_preference
# =============================

def plan_dataset_preference(
    subtask: Dict[str, Any],
    temperature: float = 0.1,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """
    Given a subtask, call LLM to generate dataset_preference.

    Return a dict, at least containing these keys:
      - preferred_roles: Dict[str, Dict]
      - preferred_domains: List[str]
    """
    model = PREFERENCE_MODEL
    subtask_json = build_dataset_preference_user_prompt(subtask)
    user_prompt = DATASET_PREFERENCE_SYSTEM_PROMPT.format(
        sid=subtask_json["sid"],
        name=subtask_json["name"],
        desc=subtask_json["desc"],
        keywords=subtask_json["keywords"],
        sample_schema_str=subtask_json["sample_schema_str"],
        answer_type=subtask_json["answer_type"],
        canonical_roles=", ".join(CANONICAL_ROLES),
    )
    if model is None:
        raise ValueError("model must be specified for plan_dataset_preference")
    raw = llm_call_json(
        model=model,
        system_prompt="You are a helpful assistant that produces dataset preference JSON based on the given subtask.",
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if isinstance(raw, str):
        try:
            pref = json.loads(raw)
        except Exception:
            pref = {}
    else:
        pref = raw or {}

    preferred_roles = pref.get("preferred_roles") or {}
    if not isinstance(preferred_roles, dict):
        preferred_roles = {}

    cleaned_roles: Dict[str, Any] = {}
    allowed = set(CANONICAL_ROLES)
    for role, cfg in preferred_roles.items():
        if role not in allowed:
            continue
        if not isinstance(cfg, dict):
            continue
        w = cfg.get("weight", 0.5)
        try:
            w = float(w)
        except Exception:
            w = 0.5
        w = max(0.0, min(1.0, w))
        cfg["weight"] = w
        cleaned_roles[role] = cfg

    pref["preferred_roles"] = cleaned_roles

    key = "preferred_domains"
    v = pref.get(key)
    if not isinstance(v, list):
        pref[key] = []
    else:
        pref[key] = [str(x) for x in v]

    return pref.get("json")
