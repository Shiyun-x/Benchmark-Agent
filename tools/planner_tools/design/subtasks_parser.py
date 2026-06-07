# tools/subtask_parser.py
# -----------------------------------------------------------------------------
# Analyst Parser tool: parse free-form benchmark description into structured subtasks.
# This tool is an LLM-based parser that transforms user requirements into a set of benchmark subtasks.
# Used as the first step in the benchmark construction pipeline.
# -----------------------------------------------------------------------------
import json
from typing import Dict, Any, List, Optional
from utils.llm_caller import llm_call_json


PROMPT_TEMPLATE = r"""
You are the Subtask Proposer in a benchmark construction system.

Input:
- task_id: {task_id}
- target_size: {target_size}
- raw_user_description:
\"\"\"{description}\"\"\"

==================== Your role ====================

Your job is to propose a strong candidate set of benchmark subtasks.

You are not finalizing the benchmark. A downstream Design Agent will review, select, revise, or discard your proposals.

Be creatively helpful:
- surface meaningful evaluation angles,
- propose subtasks that are non-redundant and useful together,
- and aim to give the downstream Design Agent a strong candidate set to choose from.

But stay disciplined:
- every subtask must fit the required QA schema,
- every subtask must have a structured and evaluable answer,
- and the final set must remain compact, concrete, and realistically executable.

We are decomposing the user's benchmark goal, not listing a generic capability taxonomy.

==================== What is a "subtask" in this system? ====================

A subtask is a structured QA-style evaluation unit derived from the user's benchmark request.

Each subtask must:
1. Be grounded in the user's evaluation intent.
2. Represent one coherent and understandable evaluation angle.
3. Preserve the user's core requested capability, while varying the concrete evaluation direction.
4. Be non-redundant relative to other subtasks.
5. Be useful in combination with the other subtasks for covering the benchmark goal.
6. Be practical enough to implement from existing or transformable data sources.
7. Produce structured, evaluable answers rather than open-ended free-form outputs.
8. Have a unified sample schema that downstream transformation and generation can use.

==================== Key design principles ====================

Subtasks should be:
- **Relevant**: tied to what the user actually wants to evaluate.
- **Concrete**: names and descriptions should be understandable and actionable.
- **Executable**: downstream should plausibly be able to generate or transform data into this form.
- **Jointly sufficient**: together they should cover the evaluation intent well.
- **Non-redundant**: avoid duplicates, while allowing slight overlap when it helps reflect the benchmark goal naturally.
- **Compact**: propose 1-3 subtasks.

Subtasks must not be overly fragmented. Every subtask must directly evaluate the user's core requested capability; differences between subtasks should mainly come from additional abilities, evidence constraints, or concrete implementation conditions.

Every proposed subtask should be comprehensive enough to cover the user's overall core requirement directly. Other subtasks may emphasize narrower variants or stress conditions, but they should not become isolated micro-skills detached from the main benchmark goal.

Prefer subtasks that collectively reveal meaningful differences in failure modes, evidence requirements, or reasoning demands, without forcing an artificial split.

Do not create separate subtasks solely because of:
- minor wording differences,
- minor difficulty differences,
- individual low-level operations that only test a small fragment of the user's requested capability,
- answer format differences alone,
- or superficial domain/topic variations.

Slight overlap across subtasks is acceptable when the benchmark theme naturally requires it. The goal is not perfect separation; the goal is a set that is useful, non-redundant, and jointly representative of the user's demand.

==================== Scope guidance ====================

If benchmark scope is provided separately, treat it as guidance rather than a rigid boundary.

Use it to stay aligned with the benchmark topic and likely modalities, but still propose the best candidate subtasks for the user's real evaluation goal.

Do not invent wildly unrelated subtasks, but do not be overly constrained if a slightly refined decomposition better serves the benchmark intent.

==================== Answer format requirement ====================

All subtasks must use QA with **structured, evaluable answers** such as:
- binary (yes/no)
- choice (multiple-choice)
- label (multi-class)
- span (span extraction)

Avoid open-ended free-text answers unless the user clearly requires them.
Prefer `choice` whenever it fits the capability being tested. Multiple-choice answers are easier to score and usually work well for generated benchmark samples.

Important: do not decompose mainly by answer format. First decide the evaluation distinction, then choose the most suitable structured answer type.

==================== Sample schema rules ====================

All subtasks must use QA format:
    evaluation_unit = "QA"

Keep every sample schema minimal. The benchmark instance should look like a simple question-answer problem, not a complex annotation task.

Input fields (fixed family only):
    - "question" (required, exactly one)
    - "context" (optional, 0 or 1)
    - "image_url" (optional, 0 or 1)
    - "audio_url" (optional, 0 or 1)

If a modality is not used, do not include that field.

Text-only tasks should usually use:
    question + optional context

Audio tasks should usually use:
    audio_url + question
Use context only if essential background is not already contained in the audio.

Image tasks should usually use:
    image_url + question
Use context only if essential background is not already contained in the image.

For multimodal tasks, if the needed information is already in the modality input, do not add redundant context.

Do NOT add fields such as "candidates", "options", "evidence", "supporting_turns", "answer_id", "rationale", "label", or "explanation".
For multiple-choice tasks, include the answer choices inside the question text or compact context, not as a separate schema field.

Modalities:
    dtype ∈ {{text, image, audio}}

Type:
    type ∈ {{str, list}}

Output fields:
- exactly one field: "answer" (required; structured/evaluable; subtype must match answer_type)

Examples:
- answer_type="binary" → subtype="yes_no"
- answer_type="choice" → subtype="multiple_choice"
- answer_type="label" → subtype="multi_class"
- answer_type="span" → subtype="extractive_span"

==================== Decomposition criteria ====================

Subtasks should be decomposed along evaluation-relevant distinctions that matter to the user goal, such as:
- what semantic distinction is being tested,
- what evidence or information must be used,
- what reasoning condition is required,
- what comparison or decision operator is involved,
- what abstraction level or granularity matters,
- what modality meaningfully changes the evaluation,
- and what kind of structured answer best fits the task after the distinction is chosen.

These dimensions are open-ended. Do not rely on a fixed taxonomy.

Keep the decomposition at the benchmark-goal level. Do not split the user's requested capability into prerequisite micro-operations unless those operations are themselves meaningful evaluation directions for the full benchmark goal.

==================== Retrieval metadata per subtask ====================

Each subtask in your output must specify:
- modalities: the modalities used by that subtask
- keywords: 3-6 short search cues for dataset retrieval

==================== Output format (STRICT JSON) ====================

{{
  "subtasks": [
    {{
      "id": "st_01",
      "name": "string",
      "description": "2-5 sentences explaining what capability is tested, what evidence/input matters, what answer is expected, and why this subtask matters for the benchmark goal",
      "answer_type": "binary | choice | label | span",
      "modalities": [...],
      "sample_schema": {{
        "input": {{
          "fields": {{
            "question": {{ "dtype": "...", "subtype": "...", "type": "str" }},
            "context": {{ "dtype": "...", "subtype": "...", "type": "str" }},
            "image_url": {{ "dtype": "...", "subtype": "...", "type": "list|str" }},
            "audio_url": {{ "dtype": "...", "subtype": "...", "type": "list|str" }}
          }}
        }},
        "output": {{
          "fields": {{
            "answer": {{ "dtype": "text", "subtype": "...", "type": "str"}}
          }}
        }}
      }},
      "keywords": ["..."]
    }}
  ]
}}

Strict rules:
- propose 1-3 subtasks
- make the subtasks jointly strong candidates for the user goal
- ensure every subtask directly covers the user's core requested capability
- keep differences mainly in extra abilities and concrete implementation conditions
- avoid redundancy
- prefer choice-style answers when appropriate
- keep input/output simple: input is question plus optional context/modality URL; output is only answer
- return valid JSON only
- return ONLY the JSON object with key "subtasks"
"""

SCOPE_PROMPT_TEMPLATE = r"""
You are the **Benchmark Scope Planner**, the first step in an automated benchmark construction pipeline.

A user wants to build an evaluation benchmark for large language/multimodal models. They have described the evaluation target of benchmark in natural language. 

Your task is to extract the benchmark-level scope from the description.

Input:
- task_id: {task_id}
- target_size: {target_size}
- raw_user_description:
\"\"\"{description}\"\"\"

Return STRICT JSON only, with exactly these keys:
{{
  "short_topic": "string",
  "modalities": ["text"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}
Rules:
1. "short_topic" must be a concise benchmark topic name, not a sentence.
2. "short_topic" should describe the overall evaluation focus.
3. "modalities" must be a JSON array of unique lowercase strings.
4. Allowed modality values are: "text", "image", "audio".
5. Choose the minimal set of modalities clearly required by the user description.
6. "keywords" must contain 3-8 short retrieval-oriented phrases.
7. Keywords should help retrieve relevant datasets or benchmark resources.
8. Prefer concrete capability/task phrases over vague words.
9. If the description is underspecified, make the most conservative reasonable interpretation.
Output requirements:
- Return only one JSON object.
- No markdown.
- No explanations.
- No extra keys.
"""
from typing import Any, Dict, List, Optional

def _as_list_str(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]

def _normalize_field_spec(fs: Any) -> Dict[str, Any]:
    """
    Field spec must be {"dtype": "...", "subtype": "...", type:"..."}.
    """
    out: Dict[str, Any] = {}
    if isinstance(fs, dict):
        out["dtype"] = str(fs.get("dtype") or "").strip().lower()
        out["subtype"] = str(fs.get("subtype") or "").strip().lower()
        out["type"] = str(fs.get("type") or "").strip().lower()
    else:
        out["dtype"] = ""
        out["subtype"] = ""
        out["type"] = "str"

    # basic sanitize
    if out["dtype"] not in ("audio", "text", "image"):
        # keep empty; do not invent
        out["dtype"] = out["dtype"] or ""
    if out["type"] not in ("str", "list"):
        out["type"] = "str"
    out["subtype"] = out["subtype"] or ""
    return out

def _normalize_sample_schema(ss: Any) -> Dict[str, Any]:
    """
    Ensure a simple QA schema shape:
    {
      "input": {"fields": {question, optional context/modality_url}},
      "output": {"fields": {answer}}
    }
    """
    if not isinstance(ss, dict):
        return {"input": {"fields": {}}, "output": {"fields": {}}}

    inp = ss.get("input") if isinstance(ss.get("input"), dict) else {}
    outp = ss.get("output") if isinstance(ss.get("output"), dict) else {}

    in_fields = inp.get("fields") if isinstance(inp.get("fields"), dict) else {}
    out_fields = outp.get("fields") if isinstance(outp.get("fields"), dict) else {}

    norm_in: Dict[str, Any] = {}
    allowed_input_fields = {"question", "context", "image_url", "audio_url"}
    for k, v in in_fields.items():
        key = str(k).strip()
        if not key or key not in allowed_input_fields:
            continue
        norm_in[key] = _normalize_field_spec(v)

    answer_spec = out_fields.get("answer", {}) if isinstance(out_fields, dict) else {}
    norm_out: Dict[str, Any] = {"answer": _normalize_field_spec(answer_spec)}

    return {"input": {"fields": norm_in}, "output": {"fields": norm_out}}

# ---- key normalization logic below ----

_ALLOWED_ANSWER_TYPES = {"binary", "choice", "label", "span"}

def _normalize_subtask(st: Dict[str, Any]) -> Dict[str, Any]:
    st = dict(st or {})

    # basic fields
    st.setdefault("id", "st_unknown")
    st.setdefault("name", None)
    st.setdefault("description", "No description.")
    st.setdefault("sample_schema", {"input": {"fields": {}}, "output": {"fields": {}}})
    st.setdefault("keywords", [])
    st.setdefault("modalities", [])
    st.setdefault("answer_type", "choice")

    # normalize id / name / description
    st["id"] = (str(st["id"]).strip() or "st_unknown").lower().replace(" ", "_")
    st["name"] = str(st["name"] or "").strip() or st["id"]
    st["description"] = str(st["description"]).strip() or "No description."

    # normalize answer_type
    at = str(st.get("answer_type") or "").strip().lower()
    if at not in _ALLOWED_ANSWER_TYPES:
        at = "choice"
    st["answer_type"] = at

    # normalize modalities / keywords
    st["modalities"] = _as_list_str(st.get("modalities"))
    st["keywords"] = _as_list_str(st.get("keywords"))

    # normalize sample_schema
    st["sample_schema"] = _normalize_sample_schema(st.get("sample_schema"))
    in_fields = st["sample_schema"]["input"]["fields"]
    if "question" not in in_fields:
        in_fields["question"] = {"dtype": "text", "subtype": "question", "type": "str"}
    out_fields = st["sample_schema"]["output"]["fields"]

    # Safety: ensure output has answer field
    if "answer" not in out_fields:
        out_fields["answer"] = {"dtype": "text", "subtype": "", "type": "str"}

    # set a more appropriate subtype from answer_type (avoid free_text)
    ans = out_fields["answer"]
    if not ans.get("subtype"):
        if at == "binary":
            ans["subtype"] = "yes_no"
        elif at == "choice":
            ans["subtype"] = "multiple_choice"
        elif at == "label":
            ans["subtype"] = "multi_class"
        else:
            ans["subtype"] = "span"

    return st

def plan_benchmark_scope(
    task_id: str,
    description: str,
    target_size: int = 450,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """
    Plan benchmark-level scope only: short_topic, modalities, keywords.
    Used before Design Agent; proposer then receives these as parameters and only proposes subtasks.
    """
    user_prompt = SCOPE_PROMPT_TEMPLATE.format(
        task_id=task_id, description=description, target_size=target_size
    )
    resp = llm_call_json(
        system_prompt="",
        user_prompt=user_prompt,
        model=model or None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not resp.get("ok"):
        raise RuntimeError(
            f"[plan_benchmark_scope] JSON parse failed: {resp.get('error')}\n"
            f"raw={resp.get('raw_text','')[:600]}"
        )
    data: Dict[str, Any] = resp["json"] or {}
    out = {
        "short_topic": str(data.get("short_topic") or "benchmark").strip(),
        "modalities": _as_list_str(data.get("modalities")),
        "keywords": _as_list_str(data.get("keywords")),
    }
    return out


def parse_topic_to_subtasks(
    task_id: str,
    description: str,
    target_size: int = 3000,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 3000,
    modalities: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    short_topic: Optional[str] = None,
    proposal_guidance: Optional[str] = None,
    existing_working_subtasks: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Decompose user requirement into subtasks. When modalities/keywords/short_topic are provided
    (e.g. from plan_benchmark_scope), they are injected into the prompt as benchmark-level guidance.
    proposal_guidance can be supplied by the Design Agent to steer a fresh proposal after reviewing
    an earlier candidate set. existing_working_subtasks can be supplied so the proposer knows what
    is already in the current working set and can avoid naive duplication.
    """
    user_prompt = PROMPT_TEMPLATE.format(
        task_id=task_id, description=description, target_size=target_size
    )
    if modalities is not None or keywords is not None or short_topic is not None:
        user_prompt += "\n\n[Benchmark scope guidance]\n"
        user_prompt += "Use this scope as guidance when proposing subtasks. Stay aligned with it, but optimize for a strong candidate set for the user's benchmark goal.\n"
        user_prompt += f"- short_topic: {short_topic or 'benchmark'}\n"
        user_prompt += f"- modalities: {modalities or []}\n"
        user_prompt += f"- keywords: {keywords or []}\n"
    if proposal_guidance is not None and str(proposal_guidance).strip():
        user_prompt += "\n\n[Design guidance for this proposal]\n"
        user_prompt += str(proposal_guidance).strip() + "\n"
    if existing_working_subtasks:
        user_prompt += "\n\n[Current working set summary]\n"
        user_prompt += (
            "These subtasks are already in the current working set. "
            "Use them as context so your proposal can complement, improve, replace, or rethink the current set rather than naively repeating it.\n"
        )
        user_prompt += json.dumps(existing_working_subtasks, ensure_ascii=False, indent=2) + "\n"
    resp = llm_call_json(
        system_prompt="",
        user_prompt=user_prompt,
        model=model or None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not resp.get("ok"):
        raise RuntimeError(
            f"[subtask_parser] JSON parse failed: {resp.get('error')}\n"
            f"raw={resp.get('raw_text','')[:600]}"
        )

    data: Dict[str, Any] = resp["json"] or {}
    data.setdefault("subtasks", [])
    if not isinstance(data.get("subtasks"), list):
        data["subtasks"] = []

    if short_topic is not None:
        data["short_topic"] = str(short_topic).strip()
    else:
        data["short_topic"] = str(data.get("short_topic") or "benchmark").strip()
    if modalities is not None:
        data["modalities"] = _as_list_str(modalities)
    else:
        data["modalities"] = _as_list_str(data.get("modalities"))
    if keywords is not None:
        data["keywords"] = _as_list_str(keywords)
    else:
        data["keywords"] = _as_list_str(data.get("keywords"))

    data["subtasks"] = [_normalize_subtask(st) for st in data["subtasks"]]
    # Keep proposer output compact and aligned with prompt contract.
    data["subtasks"] = data["subtasks"][:3]
    return data
