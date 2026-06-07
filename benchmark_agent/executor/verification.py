import os
import json
import copy
from tqdm import tqdm
import concurrent.futures
from utils.llm_caller import llm_call_json
from typing import Dict, Any, List, Set
from collections import defaultdict
from threading import Lock
from typing import Optional
from tools.shared.tool_sanitizer import MODAL_VALIDATORS, run_modal_validators
from utils.model_config import get_tool_model
from tools.shared.choice_question import stable_seed, normalize_and_shuffle_choice_question
from tools.shared.media_paths import resolve_image_paths

def _get_verify_model(model_config_path=None):
    """Get verify model from config, with fallback to default."""
    try:
        return get_tool_model("verify", model_config_path)
    except Exception:
        return "gpt-5.1"  # Fallback default

VERIFY_MODEL = _get_verify_model()  # Default for backward compatibility
VERIFY_SHARED_CORE = """
You are part of a 3-agent QA verification pipeline in a benchmark construction system.

Goal:
Determine whether the transformed QA sample in `current` is already a valid benchmark sample for the target subtask, or whether it has only minimally fixable issues, or must be rejected.

Note: `current` is the transformed sample being evaluated. It is the direct target of all evaluation and repair. The final output of this pipeline is a corrected version of `current` that conforms to `sample_schema`.

====================
Topic / user evaluation intent (highest-level goal)
====================
{topic_user_intent_section}

====================
Subtask Specification
====================
The subtask defines:
- a sample_schema (input/output fields), which is the final instance contract, not sample content
- an answer_type in [binary, choice, span, label]
- a natural-language description explaining what the subtask is evaluating

Important:
- `current.output` must contain actual answer content, not schema metadata or placeholders.
- `current` must contain all semantically required elements of a valid QA instance, not just approximately matching field structure.

answer_type = {answer_type}

sample_schema:
{sample_schema}

Subtask name: {subtask_name}
Subtask description: {subtask_desc}

====================
Sample Schema Contract
====================
Treat `sample_schema` as the exact contract for the final returned benchmark instance:
- The final sample must contain exactly the top-level sections declared by `sample_schema` (normally `input` and `output`).
- Inside each section, the final sample must contain exactly the fields declared under that section's `fields`.
- Do not preserve extra fields just because they were present in `current` or `original`.
- Do not drop declared fields unless the sample is rejected.
- Do not copy schema metadata such as `fields`, `dtype`, `subtype`, or `type` into the final sample. The final sample must contain real instance values only.

Schema mismatches are not automatically rejection reasons.
- If all required semantic content already exists somewhere in `current`, schema mismatch is a fixable `schema_recoverability` issue and MUST be repaired.
- Examples of fixable schema-only issues: wrong field name, wrong nesting, extra field whose content can be moved into a declared field, answer format that can be normalized without changing meaning, or media path stored under an equivalent field name.
- Reject only when required semantic content is absent from `current`, the content cannot be placed into any schema field without changing meaning, or repair would require inventing facts or substantial rewriting.

For `answer_type=choice`, the final sample has a stricter packaging contract:
- A discrete option set must be visible to the evaluated model.
- If `sample_schema.input.fields` includes an `options` field, options must be stored in `input.options` and must not be duplicated in `input.question`.
- If `sample_schema.input.fields` does NOT include an `options` field, options must be embedded in `input.question`, and `input.options` or any other non-schema options field must be removed.
- `output.answer` must be exactly one option label such as "A", "B", "C", or "D". Full option text, "A. text", "A) text", and explanatory answer strings are invalid final answers.
- Non-canonical choice answers are fixable when they map uniquely to one option label. They are rejection-worthy only when no unique option mapping exists.
- If no discrete option set exists anywhere in `current`, the sample is not repairable as a choice item. Do not invent options during verification.

====================
Shared Constraints
====================
- `original` and `additional info` are read-only references.
- Any actual modification may only be applied to `current`.
- `current` must satisfy the verification priority rules below (**hard constraints first**; then **user/topic evaluation intent** when provided — this outranks the subtask description/name; then subtask alignment as a tie-breaker / refinement).
- The final repaired output must strictly follow `sample_schema`. (Note: `current` may not yet conform to `sample_schema` on entry; schema-only mismatches with recoverable content are repair obligations, not immediate rejection grounds.)
- Do not introduce unsupported facts.
- Do not contradict source facts from `original` that are still being used in `current`, but do not require `current` to preserve the original task, original options, or original answer label.

Verification priority (apply in order; higher items cannot be waived by lower items):
1) **Hard constraints (non-negotiable):** `sample_schema` + declared `modalities` + `answer_type` correctness, missing-answer rules, choice uniqueness, leakage rules, and factual grounding supported by verifier-only evidence (`additional info`, including `_pure_tool_messages` when present). Neither user intent nor subtask text can override these.
2) **User / topic evaluation intent (dominant behavioral spec when provided):** If `Topic / user evaluation intent` is non-empty, treat it as the benchmark's primary semantic contract and as **higher priority than the subtask `description` and subtask `name`**. When (2) conflicts with subtask wording, follow (2) as long as (1) still passes.
3) **Subtask alignment (secondary coverage guide):** Subtasks organize coverage and quotas; they are not allowed to veto a sample that clearly satisfies the overall topic/user intent. When (2) is non-empty, use the subtask spec mainly to enforce hard packaging/modality/answer-type constraints and to catch obvious wrong-modality mistakes. A sample may be only loosely aligned with the subtask and still pass if it serves the topic intent.

Subtask mismatch policy:
- If the sample clearly serves the topic/user evaluation intent and satisfies hard constraints, do not reject solely because it is not a perfect fit for this subtask.
- Treat such mismatch as a non-fatal `subtask_alignment` note at most.
- Reject for subtask alignment only when the sample is clearly unrelated to the topic/user intent, breaks hard constraints, or has unsupported/incorrect gold labeling under `current`.

Construction-aligned expectations (leakage & answerability; match `transformation_tools_rollback.py` / Stage2 policy for `final.input`):
- **Answerability:** reject only if the labeled answer is correct **without meaningfully using** the evidence the subtask is supposed to test (e.g. world knowledge alone while ignoring required image/audio/table/text in `current.input`). Combining **input-grounded** perception or reading with **general factual world knowledge** is allowed when the subtask requires it; do not demand that every factual premise appear verbatim in `current.input` if the item is designed to use multimodal input plus legitimate background knowledge.
- **Verifier evidence vs benchmark input:** distinguish the evaluated model's input from the verifier's evidence. `current.input` is what the evaluated model receives. `additional info` / `_pure_tool_messages` are verifier-only evidence used to check that the intended gold chain is objectively supported. Do **not** reject merely because a required external fact, entity attribute, relationship, or world-knowledge premise is absent from `current.input`; those facts may be intentionally latent knowledge being tested. Use verifier-only evidence to confirm the gold answer and distractors are supported, unique, and non-contradictory.
- **External-fact benchmark design:** Some valid subtasks intentionally require the evaluated model to combine input-grounded evidence (e.g. recognizing an entity, event, object, speaker, style, table pattern, code behavior, or document cue) with external factual knowledge not printed in the prompt. This is a valid multi-hop design when the final answer still depends on the provided input. Reject only when the answer can be chosen while ignoring the input, or when verifier-only evidence fails to establish a stable gold chain from the input-grounded cue to the labeled answer.
- **Filenames / media paths:** local paths or URL strings for `image_url` / `audio_url` (and similar) are how media is delivered; the evaluated model sees the **modality**, not a privileged filename channel. Do **not** treat leakage as present **solely** because a path or basename looks informative.
- **Leakage beyond answer echo:** `current.input` must not **state, echo, or imply** `current.output.answer`, must not mark or trivially cue the correct option, and must not embed **explicit chain-of-thought or staged hop instructions** in the question (e.g. "Step 1 / Step 2", "first … then …", "reason through these hops") that replace implicit multi-hop evidence design — those collapse the intended challenge the same way answer echo does.
- **Multimodal stems:** generic pointers ("Based on this image…", "According to the audio…") remain acceptable; **concrete** restatement of visual/audio content in text remains leakage when the subtask expects perception of the media.

Important interpretation rules for `original`:
- `original` is provenance/source material for diagnosis. You cannot modify it.
- `original` is NOT the target benchmark sample. The transformed `current` may intentionally change the question, options, label space, and gold answer.
- Do NOT treat `original.output.answer` as the expected answer for `current`, unless `current` clearly preserves the exact original question and option set.
- A mismatch between `current.output.answer` and `original.output.answer` is not itself an error. Judge the answer against `current.input`, the current option set, the subtask, and verifier-only evidence.
- `original` must NOT be treated as hidden evidence that can supply missing required content to `current`.
- In particular, if a required answer is missing from `current`, it is considered missing even if `original` contains an answer.
- Missing semantically required content is not a minor structural defect.

Important interpretation rules for `additional info`:
- `additional info` is read-only and cannot be modified.
- It contains supplementary context provided exclusively for the verifier — for example, a text transcript of an audio file, or OCR/caption of an image. It does NOT appear in the final benchmark sample and is NOT seen by the evaluated model.
- The evaluated model receives only the fields in `current.input` (e.g., the audio via `audio_url`, plus any text fields such as `question` and `context`). It cannot access `additional info`.
- When `current` contains a multimodal field (e.g., audio_url, image_url): if this request includes **attached images** (vision inputs), use them as the actual pixels for those fields when paths could be resolved or are remote URLs. Otherwise use `additional info` (e.g., OCR/caption) when present. The verifier cannot play audio directly—use transcript-style content in `additional info` for audio when provided.
- This distinction matters for leakage detection: if `question` or `context` describes the concrete content of the audio/image in text, the evaluated model can read that description and bypass the need for actual perception — this is leakage even if the verifier finds it helpful.
- If `additional info` is empty, the sample has no supplementary context and verification must rely solely on `current` and `original`.

Common structured keys in `additional info` (when present; aligns with transform-stage semantics in `transformation_tools_rollback.py`):
- `_pure_tool_messages`: array of per-PURE-step debug records from the transform pipeline (e.g. `web_search_tool_args` with `query` / optional `image_paths`, and `web_search_result` with retrieved content such as `answer`). Verifier-only; not part of the benchmark shown to the evaluated model. Treat `web_search_result` as retrieved external evidence from that pipeline — use it to check whether factual Q/A claims stay within evidence supported by retrieval plus `original`/`current`, not as unconstrained world knowledge. It may support latent external facts that are intentionally tested and therefore absent from `current.input`. Do not use it to treat a missing required answer in `current.output` as present (`missing_answer` rules still apply). Same epistemic stance as the transform executor: retrieved evidence, not hidden ground truth to extend beyond what the evidence states.
- Transcript, dialog-turn lists, merged dialogue text, or similar: use only as a readable substitute for what was said in audio (or analogous) when verifying multimodal design — not as a source to fill gaps missing from `current`. Construction policy for audio tasks discourages restating or summarizing the audio/dialogue in the question stem so the model must use the recording; if `current` still embeds long dialogue-like paraphrase in text while audio is present, treat that as leakage risk.

====================
About context/meta in current
====================
Some fields in `current` may contain:
- evidence
- background
- metadata
- dialogue/OCR
- or mixed QA-bearing content

We call these "context/meta fields" regardless of field name.

Important: all fields inside `input` are presented together to the model as a combined input, with `question` always last. For example, if `input` has both `context` and `question`, the model sees: `[context] [question]`. The combined text must read naturally in this order — `context` sets the background, and `question` follows coherently from it. Each field must carry distinct, non-redundant information. If two fields substantially repeat the same content, the overlap should be removed from whichever field is less semantically central to it.

Context/meta fields such as `context` are optional supplements — they are only needed when they provide information not already present in other input fields (e.g., `question`, `audio_url`). If all required information is already conveyed by the other fields, `context` should be empty or absent. A non-empty `context` that adds nothing beyond what the `question` already states is itself a defect.

Policy:
- Context/meta in `current` MAY be minimally edited when necessary.
- Such edits must remain faithful to `original`, stay close to `current`, preserve core evidential meaning, and must not introduce unsupported facts.

Allowed minimal edits include:
  * small clarification
  * leakage removal
  * minimal disambiguation
  * light restructuring
  * separating QA-bearing text from evidence-bearing text
  * normalizing field placement when the required semantic content already exists in `current`

Not allowed:
  * contradicting `original`
  * changing core facts
  * substantial rewriting
  * replacing evidence with a summary
  * drifting far from `current`
  * inventing or filling in a missing answer
  * adding a new required semantic element that is absent from `current`

Important boundary:
- "Structurally recoverable" means the sample already contains the required semantic content, but its field names, nesting, placement, or formatting do not yet perfectly match `sample_schema`.
- "Structurally recoverable" does NOT include cases where a required semantic element is absent.
- Missing answer content is therefore NOT a recoverable schema-only problem.

Verifier role boundary:
- Verification repair is packaging/normalization, not another transformation pass.
- Prefer preserving the original wording of `current.input.question`, option texts, context/evidence, and `current.output` content.
- Safe repairs are limited to moving existing content into schema fields, removing redundant duplicate placement, canonicalizing answer format when the mapping is unique, and very small formatting edits needed for schema validity.
- Do not rewrite the question to make a weak sample stronger, more explicit, more aligned with the subtask, or easier to answer. Those are generator/design responsibilities, not verifier responsibilities.
- Do not lengthen `question` or `context` with extra reasoning instructions, credibility criteria, bias/vantage-point analysis, step hints, or answer-selection guidance. More detailed questions are often easier and may create leakage.
- Do not rewrite distractors, replace options, add new options, change the gold label, or make the task more aligned with the subtask.
- If validity would require changing the substance of the question, options, evidence, or gold answer, reject the sample instead of repairing it.

If valid repair would require substantial rewriting of context/meta, changing core facts, changing QA substance, or adding a missing required semantic element, the sample should be rejected.
"""

VERIFY_SAMPLE_BLOCK = """
====================
Sample to Process
====================
original:
{original_sample}

current:
{current_sample}

additional info:
{additional_info}
"""

INSPECTOR_RULES = """
You are Agent 1: Inspector.

Role:
- Primary responsibility: judge whether the `current` sample is a valid instance for the subtask's evaluation target.
- Concretely, decide:
  * whether `current` matches what the subtask intends to evaluate (semantic alignment)
  * whether the sample is structurally and semantically valid as-is
  * if invalid, whether it is fixable with minimal safe changes, or should be rejected
- Only diagnose issues and decide pass/reject/fixable.
- Do not propose edits.
- Do not apply edits.
- Do not assume missing required content can be supplied later unless that content already exists somewhere in `current`.

Shared rules:
{shared_core}

====================
Inspection Scope
====================
Inspect `current` against the target subtask and determine whether the sample is:
- already valid,
- fixable with minimal safe changes,
- or rejectable.

Check the following dimensions.

(0) Subtask Alignment
- Determine whether the QA in `current` serves the overall Topic / user evaluation intent first, then whether it fits this subtask as a secondary coverage guide.
- If topic/user intent is provided, the sample does NOT need to perfectly ask the exact kind of thing described by the subtask, as long as it still serves the topic intent and satisfies hard constraints.
- Honor the subtask description exactly. If the description explicitly lists example valid task forms (e.g., "such as", "e.g.", "including", or a colon-separated list), any current sample matching one of those listed forms should be treated as subtask-aligned unless another concrete requirement fails. Do not reject a listed example form by imposing a narrower interpretation of a broader label in the subtask name.
- Do not require every named capability in a compound subtask label to appear in one sample when the description presents them as alternatives. A sample may instantiate one allowed branch of the subtask (for example one listed relation type, evidence type, or reasoning target) and still be aligned.
- Also reconcile alignment with **Topic / user evaluation intent** (see Shared Constraints verification priority):
  * If user intent is provided and the sample clearly serves that intent, treat minor mismatches against **subtask `description` / `name`** as **non-fatal** `subtask_alignment` tension (severity medium/low, fixable=true) unless they collide with hard constraints (schema/modality/answer-type), contradict source facts used in `current`, or make the gold answer unsupported.
  * If user intent conflicts with the subtask wording in a way that cannot satisfy both without rewriting the core question, prefer **user intent over the subtask `description`** when deciding what the benchmark is trying to test — but you still cannot waive hard constraints.
- If the sample differs from this subtask but still serves the overall topic/user intent, do not mark it unfixable. Mark it fatal only if it is clearly unrelated to both the topic/user intent and the subtask, or if hard constraints/gold support fail.

(1) Answerability from current
- Determine whether the question/prompt is answerable **using the topic/subtask-intended evidence in `current.input`** (including multimodal fields the schema exposes), not merely whether a human could guess an answer.
- The sample is **defective on answerability** if the labeled answer is reachable **without** meaningfully using that intended input (e.g. external/world knowledge alone while ignoring required image/audio/table content). This matches the construction rule: invalid only if solvable **without** using the provided input content.
- `original` may be used only for diagnosis, not as hidden evidence for the final sample.
- Use `additional info` as verifier-only evidence. For multimodal content in `current`, it may be a readable substitute (e.g., transcript/OCR/caption) so you can judge what the model is supposed to perceive. For external retrieved facts (e.g., web search results), use it to verify that the intended gold identity/fact/relation is supported and that distractors are false or uniquely weaker. Do not require those external facts to appear in `current.input`; they may be the knowledge being tested.
- For multi-hop items, count "input-grounded cue/entity/event/pattern recognition -> external fact or relation -> answer" as answerable when verifier-only evidence supports the chain. Reject only if the chain does not depend on the input, the input-grounded cue is not stably supported by the verifier evidence, or the external fact/relation is unsupported, contradictory, or non-unique.
- If answerability is missing, check whether minimal safe clarification/disambiguation inside `current` could fix it.
- Do not treat missing answer content in `output` as answerable merely because a human inspector can infer the answer from context. Missing required answer content is handled separately under missing-answer / structure rules.

(2) Semantic Correctness
- If `current` provides an answer, check whether the answer actually answers the question.
- Check whether the answer is grounded in `current` **together with** what the subtask allows (including multimodal input and, when appropriate, external factual premises supported by verifier-only evidence — not every premise must be repeated verbatim in text if the design uses input evidence plus legitimate latent knowledge).
- Check whether the answer contradicts context/meta in `current`.
- For transformed samples, judge correctness against `current.input` and the current option set/answer space. Do not reject merely because `current.output.answer` differs from `original.output.answer`; the transformation may have intentionally changed labels or options.

(3) Answer-Type Alignment
Check according to answer_type:

- binary:
  * there should be a clear yes/no question
  * the final answer must be normalizable to exactly "yes" or "no"

- choice:
  * there should be a clear question and a discrete option set visible in `current.input`
  * if `sample_schema` defines a dedicated options field, options must be in that field
  * if `sample_schema` does not define an options field, options must be embedded in the `question` field
  * options already present but stored in a separate non-schema field are a fixable `schema_recoverability` issue when they can be losslessly moved into the schema-required location
  * if no discrete option set exists anywhere in `current`, record an unfixable high-severity `answer_type` issue; do not treat option creation as a safe repair
  * the answer must correspond to exactly one option
  * if the answer is full option text, "A. text", "A) text", or a sentence, check whether it maps uniquely to one option label; if yes, record a fixable `answer_type` issue
  * exactly one option may be correct
  * the final answer must be normalizable to exactly "A" or "B" or "C" or "D", etc.; a non-canonical answer is not valid as-is

- span:
  * the answer should be extractive from current-supported evidence
  * the answer should be short and exact-match evaluable
  * preferably no more than 5 words

- label:
  * if a label set is explicitly enumerated in `current` or `sample_schema`, the answer must belong to it
  * if no label set is defined, verify only that the answer is a single semantically coherent label consistent with the subtask description; do not require it to match a closed set

(4) Ambiguity and Choice Uniqueness
- For choice samples: determine whether EXACTLY ONE option is correct. If multiple options are correct, or none is clearly correct, record a choice_uniqueness issue. If the question wording is so under-specified that the correct option cannot be determined, record ambiguity.
- For label samples: do not flag a question as ambiguous merely because it is open-ended by design (e.g., "what is the underlying intention?"). Only record ambiguity if two or more mutually contradictory labels are equally valid given the evidence, making it impossible to determine a single correct answer.
- For binary/span samples: record ambiguity if the question wording is so vague that a reasonable answerer cannot determine what is being asked.

(5) Answer Leakage in current
- Check whether the question, options, or context/meta reveals the answer directly or indirectly in a way that collapses the intended reasoning challenge.
- This includes:
  * wording that mirrors only the correct option
  * clues present only for the correct option
  * context that explicitly states the answer while the sample pretends to ask for inference
  * explicit chain-of-thought or staged hop instructions in the question that replace evidence-based inference (see Shared Constraints / construction-aligned expectations)
- For multiple-choice, also check whether distractors are too weak, whether option phrasing makes the answer trivial, or whether one option stands out by length/style/specificity in a way that leaks correctness.

(6) Multimodal Field Presence and Leakage
This check applies when the subtask requires multimodal input (image or audio).

Step 1 — Field presence:
- Check whether `current` contains the required multimodal field (e.g., audio_url, audio_file, image_url, image_file).
- Only confirm the field exists. Do not validate whether the path or URL is accessible.
- If a field with equivalent semantics exists under a different name (e.g., audio_path instead of audio_url), treat it as a fixable `schema_recoverability` issue, not as a missing field.
- If no multimodal field of any kind is present and the subtask requires one, record a `missing_multimodal_field` issue (always unfixable and fatal).

Step 2 — Content leakage:
- If a multimodal field is present, check whether the question or context describes the concrete visual/audio content in a way that bypasses the need to actually perceive the media.
- If **attached images** are provided with this request, use them as the ground-truth pixels for the corresponding image fields in `current` / `original` when judging concrete visual leakage vs generic stems.
- Generic references are acceptable:
  * "Based on this image..."
  * "Looking at this image..."
  * "According to the audio..."
- Concrete content description (e.g., "the man in the red shirt says...") should be flagged as leakage.
- Do **not** treat the mere presence of a file path or URL in a media field as multimodal leakage; flag text that **describes** the same content the model should get from pixels or waveform.

(7) Schema and Required-Content Recoverability
- Check whether `current` can be normalized to the target `sample_schema` without losing required content, inventing new content, or adding missing semantic content.
- Do NOT reject merely because field names, nesting, placement, or formatting do not yet match `sample_schema`.
- Do NOT pass a sample as already valid when it has schema-only mismatches.
- Every recoverable schema mismatch MUST be reported as `schema_recoverability` with `fixable=true`.

You must check:
- whether any required semantic element is missing
- whether any unexpected extra field not defined by `sample_schema` is present
- whether every declared `sample_schema` field has an instance value or equivalent content in `current`
- whether `current` accidentally contains schema metadata (`fields`, `dtype`, `subtype`, `type`) instead of real sample values

Required semantic elements include, as applicable:
- a usable question/prompt
- required evidence/context if the subtask needs it
- a valid answer in `current.output`
- for choice tasks, an existing discrete option set

Rules:
- If a required semantic element is absent, do not classify it as a minor structure problem.
- If `current` contains a field not defined by `sample_schema`, record a `schema_recoverability` issue.
- An extra-field issue is fixable only if the extra field can be safely removed or its content can be losslessly merged into an allowed field without changing meaning.
- If an extra field contains important content that has no valid place in the schema and cannot be safely normalized, mark it unfixable.
- If a declared schema field is missing but equivalent content exists elsewhere in `current`, record a fixable `schema_recoverability` issue.
- If a declared schema field is missing and no equivalent content exists in `current`, record the appropriate missing-content issue; do not call it schema-only.
- For choice tasks with no schema-declared `options` field, an extra `input.options` field is fixable when its options can be appended to `input.question`; the issue should be `schema_recoverability` with `fixable=true`.
- For choice tasks, if the option set is absent from both `input.question` and any current input field such as `input.options`, this is missing required semantic content and is unfixable.
- For choice tasks, an answer like "A. hard worker", "A) hard worker", or exact option text is fixable when it maps uniquely to one option; the issue should be `answer_type` with `fixable=true`.

Priority rule:
- Missing required semantic content takes precedence over schema recoverability.
- A sample is recoverable only if the required semantic content already exists somewhere in `current`.

Special rule for missing answer:
- If `current` does not contain answer content required for the sample, record a `missing_answer` issue.
- `missing_answer` is always unfixable.
- `missing_answer` is always fatal.
- Any sample with `missing_answer` must be rejected.
- Do not treat adding an answer as a minimal repair, even if the answer seems inferable from context or visible in `original`.

(8) Input Field Redundancy
This check applies only when `sample_schema` defines multiple input fields (e.g., both `question` and `context`).
- Verify that each input field carries distinct, non-redundant information.
- `question` should state what is being asked; `context` should provide background or evidence that the question draws on — not restate the question or re-describe the task.
- If `context` (or another supplementary field) repeats content already present in `question` or other fields, record a redundancy issue.
- Fix by removing the repeated content from the supplementary field. If after removing the overlap the supplementary field becomes empty or carries no meaningful information, remove it entirely (or leave it empty if the schema requires the field to exist).
- Do not preserve a `context` field solely to fill the field — an empty or absent context is correct when the question already contains all necessary information.

(9) Repair Feasibility
Mark an issue unfixable if solving it would require any of:
- contradicting `original`
- introducing unsupported facts
- substantial rewriting of context/meta
- substantial rewriting of the question, options, evidence, or answer
- replacing distractors or inventing new choices
- changing the gold answer except for equivalent canonical formatting
- changing the task into a different one
- drifting too far from `current`
- adding a missing required semantic element that is absent from `current`

====================
Issue Classification
====================
For each issue, output:
- id: short machine-friendly issue id
- category: one of
  "subtask_alignment"
  "answerability"
  "semantic_correctness"
  "answer_type"
  "ambiguity"
  "choice_uniqueness"
  "leakage"
  "image_audio_leakage"
  "missing_multimodal_field"
  "input_field_redundancy"
  "schema_recoverability"
  "context_meta_drift_risk"
  "missing_answer"
  "other"
- problem: concise description of what is wrong
- fixable: true | false
- severity: "high" | "medium" | "low"
- evidence: concise basis for the judgment
- requires_context_edit: true | false

Issue labeling rules:
- Use `missing_answer` only when required answer content is absent from `current`.
- Do not downgrade a missing answer into `schema_recoverability`.
- Use `missing_multimodal_field` when a required image/audio field (e.g., audio_url, image_url) is entirely absent from `current`. This is always unfixable and fatal.
- Use `image_audio_leakage` when the multimodal field exists but the question/context reveals its concrete content.
- Use `input_field_redundancy` when two or more input fields substantially repeat the same content. This is always fixable.
- Use `schema_recoverability` only for structure/placement/format mismatches where the required semantic content already exists in `current`.
- Recoverable schema mismatch or uniquely mappable non-canonical choice answer means decision="pass", can_fix=true.
- For `subtask_alignment`, reserve **fatal** (`fixable=false`, severity high) for cases that also violate hard constraints, contradict source facts used in `current`, make the gold answer unsupported, or are clearly unrelated to the provided user/topic intent. If user/topic intent is non-empty and the sample serves it well, do not reject for subtask mismatch alone.

====================
Decision Policy
====================
Definitions:
- A fatal issue is any unfixable issue that prevents the sample from becoming a valid benchmark instance under the allowed repair scope.
- All `missing_answer` issues are fatal by definition.
- `subtask_alignment` is **not automatically fatal**: when user/topic intent is provided, it becomes fatal only when the sample fails that intent, has an unfixable hard-constraint failure, or has unsupported/incorrect gold labeling. Subtask mismatch by itself is non-fatal.

Decision rules:
- decision="pass" means there is no fatal issue.
- decision="reject" means there exists at least one fatal issue.
- can_fix=true means:
  * decision="pass", and
  * there is at least one fixable issue.
- can_fix=false means:
  * there are no issues, or
  * the sample is rejected.

Therefore:
- If there are no issues at all:
  * decision="pass"
  * can_fix=false
  * issues=[]
- If there are only fixable issues:
  * decision="pass"
  * can_fix=true
- If there is any fatal issue:
  * decision="reject"
  * can_fix=false

Priority rule:
- Missing required semantic content takes precedence over schema recoverability.
- A sample cannot be passed as fixable if a required answer is absent.
- A recoverable schema mismatch cannot be passed as already valid; it must be passed with can_fix=true.
- A choice sample with no discrete option set in `current` cannot be repaired.

{sample_block}

Return STRICT JSON:

{{
  "decision": "pass" | "reject",
  "can_fix": true | false,
  "issues": [
    {{
      "id": "short_issue_id",
      "category": <category>,
      "problem": "what is wrong",
      "fixable": true | false,
      "severity": "high" | "medium" | "low",
      "evidence": "brief supporting basis",
      "requires_context_edit": true | false
    }}
  ],
  "reason": "required when decision=reject"
}}

Requirements:
- Do not output a modified sample.
- Do not output a repair plan.
- Do not omit `reason` when decision="reject".
- `missing_answer` must produce decision="reject".
- Keep issues precise and non-redundant.
"""
REPAIR_PLANNER_RULES = """
You are Agent 2: Repair Planner.

Role:
- Only produce a repair plan.
- Never rewrite the sample directly.
- Never output a repaired sample.
- Convert Inspector findings into minimal, concrete, executable repair actions.

Shared rules:
{shared_core}

====================
Planning Goal
====================
Produce an ordered repair plan that:
- fixes all fixable issues,
- uses the minimum necessary changes,
- stays faithful to source facts already used by `current`,
- stays close to `current`,
- preserves evidential meaning,
- and prepares the sample for final normalization without continuing the transformation.

====================
Planning Principles
====================
1. Only plan repairs for issues marked fixable=true.
2. If Inspector decision="reject" and can_fix=false, return plan_status="not_fixable".
3. Prefer the smallest possible repair.
4. Do not propose actions that:
   - modify `original`
   - introduce unsupported facts
   - substantially rewrite context/meta
   - substantially rewrite the question
   - lengthen the question or context with extra reasoning guidance
   - add bias/vantage/credibility instructions that were not already present
   - rewrite, replace, or invent answer options
   - change the gold answer except to canonicalize an equivalent format
   - change the target task
5. If a repair touches context/meta, it must be explicitly minimal:
   - clarification
   - leakage removal
   - minimal disambiguation
   - light restructuring
   - separation of QA-bearing content from context/meta
6. Final schema conformity should be handled at the end of the action plan, not first.
7. Include `map_fields_to_schema` for fixable schema issues, and `extract_choice_label` or `normalize_answer` for fixable non-canonical choice answers.
8. If the only way to pass is to improve the QA content rather than normalize existing content, return plan_status="not_fixable".

====================
Allowed Action Types
====================
Use only the following `action` values:

- "rewrite_question" (only for punctuation/format cleanup or removing accidental answer echo; never for elaboration)
- "normalize_answer"
- "extract_choice_label"
- "rewrite_options" (only for formatting existing options; never for changing option substance)
- "replace_option" (discouraged; use only for removing an unsupported accidental duplicate without changing the gold task)
- "delete_option" (only for duplicate/invalid extra option cleanup when the remaining option set is still valid)
- "reorder_options"
- "add_minimal_disambiguation"
- "light_edit_context_meta"
- "separate_qa_from_context_meta"
- "normalize_answer_type"
- "map_fields_to_schema"
- "other_minimal_safe_fix"

====================
Action Design Rules
====================
Each action must be:
- ordered
- specific
- executable by the fixer
- minimal
- justified by an Inspector issue

Each action step must include:
- step: integer
- target: exact field or approximate location
- action: one allowed action type
- instruction: concrete instruction
- expected_result: what should become true after the action
- issue_ids: which Inspector issue ids this action addresses

====================
Recommended Action Mapping
====================
Use these mappings when appropriate:

- question wording leaks answer
  -> not_fixable unless a tiny deletion removes an accidental answer echo without changing or lengthening the question's substance

- answer format not canonical
  -> normalize_answer
  -> extract_choice_label (if choice answer is full text)

- choice option placement violates schema
  -> map_fields_to_schema (move existing options to the schema-required location; never invent options)

- multiple correct options
  -> not_fixable unless caused only by a duplicate option that can be removed without changing the intended task

- missing role/entity cue causing under-specification
  -> not_fixable unless the cue already exists in `current` and only needs field placement/format cleanup

- QA text mixed into context/meta
  -> separate_qa_from_context_meta

- minor leakage or ambiguity inside context/meta
  -> light_edit_context_meta only for small deletion/formatting; reject if it requires rewriting or lengthening evidence

- final answer-type cleanup
  -> normalize_answer_type

- final structural cleanup
  -> map_fields_to_schema

For choice tasks, if the option set is absent from `current`, return plan_status="not_fixable"; never plan to invent options.

Question/options rewrite boundary:
- Do not use rewrite-style actions to improve quality, add reasoning, create better distractors, or align the sample with the subtask.
- Use rewrite-style actions only when the original semantic content remains unchanged and the edit is limited to formatting, de-duplication, or removing an accidental answer echo.
- Prefer shorter repairs. If a proposed repair makes the question longer or more instructive, return plan_status="not_fixable".
- When in doubt, return plan_status="not_fixable".

====================
Context/Meta Planning Rules
====================
Use `light_edit_context_meta` ONLY when necessary and ONLY for minimal safe edits.
Examples of acceptable planner instructions:
- remove a directly revealing phrase
- add a missing speaker/role cue strongly supported by current/additional info
- separate evidence from embedded option list
- slightly restructure for clarity without changing meaning

Do NOT use it for:
- rewriting the whole passage
- summarizing evidence
- adding unsupported evidence
- changing substantive factual content

====================
Output Policy
====================
- If all fixable issues are addressed by the action_plan, return plan_status="ready" with the action_plan.
- If safe repair is not possible, return plan_status="not_fixable".

{sample_block}

Inspector output:
{inspector_output}

Return STRICT JSON:

{{
  "plan_status": "ready" | "not_fixable",
  "action_plan": [
    {{
      "step": 1,
      "target": "input.question | input.context | output.answer | inferred field name | etc",
      "action": "rewrite_question | normalize_answer | extract_choice_label | rewrite_options | replace_option | delete_option | reorder_options | add_minimal_disambiguation | light_edit_context_meta | separate_qa_from_context_meta | normalize_answer_type | map_fields_to_schema | other_minimal_safe_fix",
      "instruction": "specific and executable instruction",
      "expected_result": "what should become true",
      "issue_ids": ["issue_1", "issue_2"]
    }}
  ],
  "reason": "required when plan_status=not_fixable"
}}

Requirements:
- action_plan must be ordered.
- action_plan must be minimal.
- Do not include speculative or redundant actions.
- Do not output the repaired sample.
"""
FIXER_RULES = """
You are Agent 3: Fixer.

Role:
- Apply the Planner's action plan to `current`.
- Return the final repaired sample.
- Be conservative.
- Only make changes justified by the plan.

Shared rules:
{shared_core}

====================
Execution Rules
====================
1. Only modify `current`.
2. Follow the action plan in order.
3. Do not invent extra major edits outside the plan.
4. Small incidental edits are allowed only if they are necessary to complete a planned action safely and do not change meaning.
5. Any edit to context/meta must remain:
   - faithful to source facts already used by `current`
   - close to `current`
   - minimal
   - evidence-preserving
   - non-contradictory
6. Do not introduce unsupported facts.
7. Do not substantially rewrite context/meta.
8. Do not substantially rewrite the question, options, evidence, or answer. The verifier fixes packaging and normalization; it does not continue the generator's transformation.
9. Do not change the benchmark's declared `subtask_id` / evaluation packaging, and do not rewrite the item into a different benchmark goal. If the sample needs semantic rewriting to align with the topic or subtask, reject it.
10. Do not make the question or context longer, more explicit, or more instructive. Do not add bias/vantage/credibility framing, step hints, or reasoning criteria.
11. Safe edits are limited to schema field mapping, formatting, de-duplication, removing accidental answer echoes, and canonical answer-label extraction when the mapping is unique.

====================
Required Final Validity Checks
====================
Before returning, ensure all of the following are true.

(A) Subtask Alignment
- The final sample evaluates the intended benchmark goal under the Shared Constraints verification priority (**hard constraints**, then **user/topic intent when provided (outranks subtask `description`)**, then subtask wording as secondary refinement).
- If user intent is provided and the repaired sample clearly serves it, do not reject solely because the subtask `description` match is imperfect — unless that imperfection breaks hard constraints or makes the gold answer unsupported.

(B) Answerability
- The final question is answerable through the subtask's intended chain: use final `current.input` for the evaluated model's visible evidence, and use `additional info` as verifier-only evidence to confirm any latent external facts, identities, relationships, or world-knowledge premises that the benchmark is intentionally testing.
- The labeled answer must not be correct **without** meaningfully using that intended input (including multimodal); world knowledge alone while ignoring required media/text is a defect, consistent with Shared Constraints. But do not require tested external facts to be printed in `current.input` when verifier-only evidence supports them.

(C) Semantic Correctness
- The final answer correctly answers the final question.
- The final answer is grounded in the final `current.input` together with factual premises allowed by **user/topic intent (when provided)** and the subtask design, supported by verifier-only evidence.
- The final question/options/answer must preserve the substance of `current`; do not make semantic improvements beyond the repair plan.
- The final question should not be longer or more guiding than the input question unless the only added text is moving an existing option set into the schema-required question field.

(D) Answer-Type Alignment
Enforce according to answer_type:

- binary:
  * final answer must be exactly "yes" or "no"

- choice:
  * enforce the choice packaging contract from Shared rules
  * use only options already present in `current`; never invent options
  * final answer must be one bare option label and exactly one option may be correct

- span:
  * final answer must be extractive
  * final answer should be short and exact-match evaluable
  * aim for no more than 5 words; this is a soft guideline and must not be used as a rejection criterion

- label:
  * final answer must be one valid label only
  * if a label set is explicitly defined, the answer must belong to it; if no label set is defined, ensure the answer is a single coherent phrase consistent with the subtask description

(E) Leakage Removal
- No answer leakage should remain.
- For image/audio tasks, the `current.input` must not contain concrete visual/audio **narrative** that replaces perception; `image_url` / `audio_url` path strings alone are not leakage.

(F) Final Normalization
- Normalize answer type.
- Then normalize structure to `sample_schema`.
- The returned `sample` must be the final benchmark instance itself, not wrapped in `current`, `final`, `sample`, or any metadata object.
- The returned `sample` must contain exactly the schema-declared top-level keys and section fields, with no extra fields or schema metadata keys.
- For choice tasks, move existing options to the schema-required location and convert uniquely mappable answers to bare labels.
- Do not create new options, replace distractors, rewrite the stem, or change the gold answer beyond canonical label extraction.
- Do not add extra explanation, reasoning instructions, bias/vantage framing, or credibility analysis to the question/context.
- Field names and placement in the returned sample must strictly conform to the Sample Schema Contract.

====================
Status Policy
====================
- Return status="ok" if the action_plan is empty and current is already valid (no changes were needed).

- Return status="fixed" if:
  * one or more actions were applied successfully
  * and final sample is valid

- Return status="rejected" if:
  * the plan cannot be applied safely
  * or the final sample still cannot be made valid after safe execution

{sample_block}

Inspector output:
{inspector_output}

Planner output:
{planner_output}

Return STRICT JSON:

{{
  "status": "ok" | "fixed" | "rejected",
  "sample": <final schema-valid sample — include only when status is "ok" or "fixed", omit entirely when status is "rejected">,
  "reason": "required when status is rejected; optional but encouraged when status is fixed"
}}

Requirements:
- If status is ok/fixed, include a final schema-valid sample.
- If status is rejected, do not include a sample.
- Be conservative: prefer rejection over unsafe repair.
"""



def _load_checkpoint(subtask_id: str, dataset_id: str, checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """
    Restore from local disk if a verify result already exists for this (subtask_id, dataset_id).
    """
    path = os.path.join(checkpoint_dir, subtask_id, f"{dataset_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_checkpoint(subtask_id: str, dataset_id: str, data: Dict[str, Any], checkpoint_dir: str) -> None:
    """
    Write the verify result for this (subtask_id, dataset_id) to a local JSON file.
    """
    os.makedirs(os.path.join(checkpoint_dir, subtask_id), exist_ok=True)
    path = os.path.join(checkpoint_dir, subtask_id, f"{dataset_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _verify_cache_key(entry: Dict[str, Any], default_dataset_id: str) -> Optional[tuple]:
    if not isinstance(entry, dict):
        return None
    dataset_id = entry.get("dataset_id") or default_dataset_id
    idx = entry.get("idx")
    if idx is None:
        return None
    return (str(dataset_id), str(idx))


def _filter_cached_verify_items(
    items: List[Dict[str, Any]],
    *,
    dataset_id: str,
    current_keys: set,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items or []:
        key = _verify_cache_key(item, dataset_id)
        if key is None or key not in current_keys or key in seen:
            continue
        out.append(item)
        seen.add(key)
    return out


def _build_verify_stats(verified: List[Dict[str, Any]], rejected: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"total": len(verified) + len(rejected), "ok": 0, "fixed": 0, "rejected": len(rejected)}
    for item in verified:
        sample = item.get("sample") if isinstance(item, dict) else {}
        status = sample.get("status") if isinstance(sample, dict) else None
        if status == "fixed":
            stats["fixed"] += 1
        else:
            stats["ok"] += 1
    return stats


# ==================== Single subtask entry point ====================
def build_additional_info(sample: Dict[str, Any]) -> str:
    """
    Build the additional_info string.
    """
    additional_parts = []
    for key, value in sample.items():
        if key not in ["original", "current"]:
            dumped = json.dumps(value, ensure_ascii=False, indent=2)
            additional_parts.append(f"{key}:\n{dumped}")
    return "\n\n".join(additional_parts)


def _format_topic_user_intent_section(
    *,
    short_topic: Optional[str],
    topic_user_requirements: Optional[str],
) -> str:
    st = str(short_topic or "").strip()
    ur = str(topic_user_requirements or "").strip()
    if not st and not ur:
        return (
            "(No topic-level user evaluation intent was provided to the verifier. "
            "In that case, treat the subtask specification as the primary behavioral contract.)"
        )
    parts: List[str] = []
    if st:
        parts.append(f"short_topic / topic label:\n{st}")
    if ur:
        parts.append(f"user / topic evaluation intent (verbatim):\n{ur}")
    parts.append(
        "How to use this section:\n"
        "- This is the user's benchmark-building goal. It is not part of `current.input` and is not shown to the evaluated model.\n"
        "- When non-empty, this section outranks the subtask `description` / `name` for semantic intent (but never outranks hard constraints like schema/modality/answer-type/leakage/missing answer).\n"
        "- Prefer rejecting only if the sample fails (1) hard constraints or clearly fails this stated intent.\n"
        "- If the sample clearly advances this intent but is only a partial lexical match to the subtask description, do not treat that as fatal `subtask_alignment` by itself."
    )
    return "\n\n".join(parts)


def _build_shared_core_block(
    subtask_name: str,
    subtask_desc: str,
    answer_type: str,
    schema_str: str,
    *,
    short_topic: Optional[str] = None,
    topic_user_requirements: Optional[str] = None,
) -> str:
    topic_user_intent_section = _format_topic_user_intent_section(
        short_topic=short_topic,
        topic_user_requirements=topic_user_requirements,
    )
    return VERIFY_SHARED_CORE.format(
        topic_user_intent_section=topic_user_intent_section,
        subtask_name=subtask_name,
        subtask_desc=subtask_desc,
        answer_type=answer_type,
        sample_schema=schema_str,
    )


def _build_sample_block(sample_view: Dict[str, Any], additional_info: str) -> str:
    return VERIFY_SAMPLE_BLOCK.format(
        original_sample=json.dumps(sample_view.get("original"), ensure_ascii=False, indent=2),
        current_sample=json.dumps(sample_view.get("current"), ensure_ascii=False, indent=2),
        additional_info=additional_info,
    )


def _normalize_current_by_schema_and_type(
    current_sample: Any,
    answer_type: str,
    sample_schema: Dict[str, Any],
) -> Any:
    """
    Lightweight post-normalization after Fixer:
    - keep only schema top-level keys if schema provides dict keys
    - binary answer normalize to yes/no when possible
    """
    if not isinstance(current_sample, dict):
        return current_sample

    normalized = copy.deepcopy(current_sample)
    if isinstance(sample_schema, dict) and sample_schema:
        allowed_keys = set(sample_schema.keys())
        if allowed_keys:
            normalized = {k: v for k, v in normalized.items() if k in allowed_keys}

    at = str(answer_type or "").lower()
    if at == "binary":
        output_obj = normalized.get("output")
        if isinstance(output_obj, dict):
            answer_val = output_obj.get("answer")
            if isinstance(answer_val, str):
                ans = answer_val.strip().lower()
                yes_set = {"yes", "y", "true", "1", "affirmative", "yeah", "yep"}
                no_set = {"no", "n", "false", "0", "negative", "nope"}
                if ans in yes_set:
                    output_obj["answer"] = "yes"
                elif ans in no_set:
                    output_obj["answer"] = "no"
        normalized["output"] = output_obj
    return normalized


def _has_real_output(sample: Any) -> bool:
    if not isinstance(sample, dict):
        return False
    output_obj = sample.get("output")
    if not isinstance(output_obj, dict) or not output_obj:
        return False
    if "fields" in output_obj and "answer" not in output_obj:
        return False
    return True


def _schema_input_field_keys(sample_schema: Dict[str, Any]) -> Set[str]:
    """Match _strict_schema_check: input field names from sample_schema.input."""
    if not isinstance(sample_schema, dict):
        return set()
    sec_schema = sample_schema.get("input")
    if not isinstance(sec_schema, dict):
        return set()
    if isinstance(sec_schema.get("fields"), dict):
        return set(sec_schema["fields"].keys())
    return {k for k in sec_schema.keys() if k != "fields"}


_ORIGINAL_IMAGE_SINGLE_KEYS = ("image", "image_file")
_ORIGINAL_IMAGE_LIST_KEYS = ("images", "image_list", "image_lists", "image_files")


def _input_image_url_effectively_missing(inp: Dict[str, Any]) -> bool:
    if "image_url" not in inp:
        return True
    v = inp.get("image_url")
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, list) and len(v) == 0:
        return True
    return False


def _extract_image_url_value_from_original(orig_input: Dict[str, Any]) -> Optional[Any]:
    """
    Single-image fields -> str for image_url; list-style fields -> list.
    List keys take precedence when any yields a non-empty list.
    """
    if not isinstance(orig_input, dict):
        return None

    for k in _ORIGINAL_IMAGE_LIST_KEYS:
        v = orig_input.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            out = [x for x in v if x is not None and str(x).strip()]
            if out:
                return out
        elif isinstance(v, str) and v.strip():
            return [v.strip()]

    for k in _ORIGINAL_IMAGE_SINGLE_KEYS:
        v = orig_input.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            non_empty = [x for x in v if x is not None and str(x).strip()]
            if not non_empty:
                continue
            if len(non_empty) == 1:
                return str(non_empty[0])
            return non_empty
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _backfill_image_url_from_original_for_verify(
    whole_sample: Dict[str, Any],
    current_final: Dict[str, Any],
    sample_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    If schema requires input.image_url but final.input lacks it, copy from original.input
    (image/image_file -> str; images/image_list/image_lists/image_files -> list).
    Mutates current_final and whole_sample['current']['final'] when applied.
    Returns possibly updated current_final.
    """
    if not isinstance(current_final, dict):
        return current_final
    schema_in = _schema_input_field_keys(sample_schema)
    if "image_url" not in schema_in:
        return current_final

    inp = current_final.get("input")
    if not isinstance(inp, dict):
        inp = {}
        current_final["input"] = inp
    if not _input_image_url_effectively_missing(inp):
        return current_final

    original = whole_sample.get("original")
    if not isinstance(original, dict):
        return current_final
    orig_in = original.get("input")
    if not isinstance(orig_in, dict):
        return current_final

    val = _extract_image_url_value_from_original(orig_in)
    if val is None:
        return current_final

    current_final = copy.deepcopy(current_final)
    current_final.setdefault("input", {})
    current_final["input"]["image_url"] = val
    whole_sample.setdefault("current", {})
    whole_sample["current"]["final"] = current_final
    return current_final


_VERIFY_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")


def _verify_looks_like_image_ref(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    low = t.lower()
    if low.startswith("http://") or low.startswith("https://"):
        return True
    return any(low.endswith(ext) for ext in _VERIFY_IMAGE_EXTS)


def _verify_collect_input_image_refs(inp: Any) -> List[str]:
    """Paths/URLs from original or current ``input`` for verifier vision attachments."""
    if not isinstance(inp, dict):
        return []
    keys = (
        "image_url",
        "image",
        "image_file",
        "image_files",
        "images",
        "image_list",
        "image_lists",
    )
    out: List[str] = []
    for k in keys:
        v = inp.get(k)
        if isinstance(v, str) and _verify_looks_like_image_ref(v):
            out.append(v.strip())
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str) and _verify_looks_like_image_ref(x):
                    out.append(x.strip())
    return out


def _verify_build_llm_image_list(
    sample_view: Dict[str, Any],
    *,
    dataset_json_path: Optional[str],
    dataset_id: str,
    max_images: int = 16,
) -> Optional[List[str]]:
    """
    Resolve local image refs (basename / relative) like the transform pipeline, keep http(s) URLs as-is.
    Order: ``current`` input refs first, then ``original`` (deduped).
    """
    raw: List[str] = []
    for sec in ("current", "original"):
        block = sample_view.get(sec)
        if not isinstance(block, dict):
            continue
        inp = block.get("input")
        raw.extend(_verify_collect_input_image_refs(inp))
    # stable dedupe on raw strings
    raw = list(dict.fromkeys(raw))
    if not raw:
        return None

    urls: List[str] = []
    locals_paths: List[str] = []
    for p in raw:
        low = p.lower()
        if low.startswith("http://") or low.startswith("https://"):
            urls.append(p)
        else:
            locals_paths.append(p)

    resolved = resolve_image_paths(
        locals_paths,
        dataset_json_path=dataset_json_path,
        dataset_id=dataset_id or None,
    )
    merged: List[str] = []
    seen: Set[str] = set()
    for p in urls + resolved:
        if p not in seen:
            seen.add(p)
            merged.append(p)
    if not merged:
        return None
    if len(merged) > max_images:
        merged = merged[:max_images]
    return merged


def _strict_schema_check(sample: Any, sample_schema: Dict[str, Any]) -> tuple[bool, str]:
    """
    Enforce exact schema match:
    - no extra keys and no missing keys
    - checks top-level keys and the immediate field keys under input/output

    Expected sample shape (instance):
      {"input": {...}, "output": {...}}
    Schema shape (description):
      {"input": {"fields": {...}}, "output": {"fields": {...}}}
    """
    if not isinstance(sample_schema, dict) or not sample_schema:
        return True, ""
    if not isinstance(sample, dict):
        return False, "sample_not_object"

    expected_top = set(sample_schema.keys())
    actual_top = set(sample.keys())
    if actual_top != expected_top:
        missing = sorted(expected_top - actual_top)
        extra = sorted(actual_top - expected_top)
        return False, f"top_level_mismatch missing={missing} extra={extra}"

    for sec in expected_top:
        sec_schema = sample_schema.get(sec)
        sec_val = sample.get(sec)
        if not isinstance(sec_val, dict):
            return False, f"{sec}_not_object"
        # Prefer schema["fields"] if present, otherwise fall back to dict keys
        if isinstance(sec_schema, dict) and isinstance(sec_schema.get("fields"), dict):
            expected_fields = set(sec_schema["fields"].keys())
        elif isinstance(sec_schema, dict):
            expected_fields = set(sec_schema.keys())
        else:
            continue
        actual_fields = set(sec_val.keys())
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            return False, f"{sec}_fields_mismatch missing={missing} extra={extra}"

    return True, ""


def verify_one_subtask(
    subtask: Dict[str, Any],
    transformed_items: List[Dict[str, Any]],
    max_workers: int = 2,
    sample_workers: int = 10,
    topic: str = "default_topic",
    checkpoint_dir: Optional[str] = None,
    model_config_path: Optional[str] = None,
    *,
    topic_user_requirements: Optional[str] = None,
    topic_short_topic: Optional[str] = None,
    dataset_cards: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """

    Returns:
        {
          "verified_buffer": {dataset_id: [items...]},
          "rejected_buffer": {dataset_id: [rejected_entries...]},
          "stats": {"total":..., "ok":..., "fixed":..., "rejected":...}
        }
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("cache", topic, "verify_log")

    llm_model = _get_verify_model(model_config_path)
    # 1) Group by dataset_id
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in transformed_items:
        did = item.get("dataset_id") or "unknown_dataset"
        grouped[did].append(item)

    subtask_id = subtask.get("id") or subtask.get("subtask_id") or ""
    subtask_name = subtask.get("name") or subtask.get("title") or ""
    subtask_desc = subtask.get("description") or ""
    sample_schema = subtask.get("sample_schema") or {}
    answer_type = subtask.get("answer_type") or ""
    schema_str = json.dumps(sample_schema, ensure_ascii=False, indent=2)

    shared_core = _build_shared_core_block(
        subtask_name=subtask_name,
        subtask_desc=subtask_desc,
        answer_type=answer_type,
        schema_str=schema_str,
        short_topic=topic_short_topic,
        topic_user_requirements=topic_user_requirements,
    )


    # ====== Modal hard rule: audio modality requires audio_url in input ======
    modalities = subtask.get("modalities") or []
    modalities_norm = {str(m).lower() for m in modalities}

    active_validators = [
        MODAL_VALIDATORS[m]
        for m in modalities_norm
        if m in MODAL_VALIDATORS
    ]    
    # 2) Count total samples (all items passed in this run)
    total_samples = len(transformed_items)

    # 3) Pre-load checkpoints per dataset and count already-completed samples
    dataset_to_cached: Dict[str, Optional[Dict[str, Any]]] = {}
    already_done = 0

    for dataset_id, items in grouped.items():
        cached = _load_checkpoint(subtask_id, dataset_id, checkpoint_dir)
        dataset_to_cached[dataset_id] = cached
        if cached is not None:
            current_keys = {
                key for key in (_verify_cache_key(item, dataset_id) for item in items)
                if key is not None
            }
            cached_verified = _filter_cached_verify_items(
                cached.get("verified", []),
                dataset_id=dataset_id,
                current_keys=current_keys,
            )
            cached_rejected = _filter_cached_verify_items(
                cached.get("rejected", []),
                dataset_id=dataset_id,
                current_keys=current_keys,
            )
            cached_done = len(cached_verified) + len(cached_rejected)
            already_done += cached_done
            remaining = max(len(items) - cached_done, 0)
            print(
                f"[Checkpoint] Restored dataset {dataset_id}: cached={cached_done}, "
                f"remaining={remaining}, total={len(items)}"
            )

    # 4) Initialize sample-level progress bar
    progress = tqdm(
        total=total_samples,
        desc=f"Verifying samples for {subtask_id}",
    )
    # Advance progress bar by already-completed sample count
    if already_done > 0:
        progress.update(min(already_done, total_samples))

    progress_lock = Lock()

    # 5) Define per-dataset execution logic (updates progress bar per sample)
    def _run_one_dataset(dataset_id: str, items: List[Dict[str, Any]]):
        cached = dataset_to_cached.get(dataset_id)
        current_keys = {
            key for key in (_verify_cache_key(item, dataset_id) for item in items)
            if key is not None
        }
        verified: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        done_keys = set()

        if cached is not None:
            verified = _filter_cached_verify_items(
                cached.get("verified", []),
                dataset_id=dataset_id,
                current_keys=current_keys,
            )
            rejected = _filter_cached_verify_items(
                cached.get("rejected", []),
                dataset_id=dataset_id,
                current_keys=current_keys,
            )
            done_keys = {
                key for key in (
                    _verify_cache_key(item, dataset_id) for item in (verified + rejected)
                )
                if key is not None
            }

        # ---- per-item worker: all captured vars are read-only, safe for parallel execution ----
        def _process_item(item):
            whole_sample = item.get("sample") or {}
            current_obj = whole_sample.get("current") or {}
            raw_final = current_obj.get("final")
            current_final = raw_final if isinstance(raw_final, dict) else {}
            sample_view = {
                "original": whole_sample.get("original"),
                "current": current_final,
            }

            current_final = _backfill_image_url_from_original_for_verify(
                whole_sample, current_final, sample_schema
            )
            sample_view["current"] = current_final

            # Deterministic pre-normalization for choice questions:
            # - shuffle options (and update answer) to avoid "always A"
            # - trim overly-leading stems for audio tasks
            try:
                answer_type_norm = str(answer_type or "").lower()
                if answer_type_norm == "choice" and isinstance(current_final, dict):
                    inp = current_final.get("input") if isinstance(current_final.get("input"), dict) else {}
                    outp = current_final.get("output") if isinstance(current_final.get("output"), dict) else {}
                    q = inp.get("question")
                    a = outp.get("answer")
                    has_audio = isinstance(inp.get("audio_url"), str) and bool(inp.get("audio_url"))
                    if isinstance(q, str) and isinstance(a, str):
                        seed = stable_seed(subtask_id, dataset_id, item.get("idx"), "verify_choice_shuffle")
                        patched = normalize_and_shuffle_choice_question(q, a, seed, has_audio=has_audio)
                        if patched:
                            current_final = copy.deepcopy(current_final)
                            current_final.setdefault("input", {})
                            current_final.setdefault("output", {})
                            current_final["input"]["question"] = patched["question"]
                            current_final["output"]["answer"] = patched["answer"]
                            # Persist into whole_sample so returned "ok" uses normalized version
                            whole_sample.setdefault("current", {})
                            whole_sample["current"]["final"] = current_final
                            sample_view["current"] = current_final
            except Exception:
                pass

            additional_info = build_additional_info(whole_sample)
            sample_block = _build_sample_block(sample_view=sample_view, additional_info=additional_info)

            source_json: Optional[str] = None
            if isinstance(dataset_cards, dict):
                card = dataset_cards.get(dataset_id)
                if isinstance(card, dict):
                    sj = (card.get("raw_meta") or {}).get("source_json")
                    if isinstance(sj, str) and sj.strip():
                        source_json = sj.strip()

            verify_images = _verify_build_llm_image_list(
                sample_view,
                dataset_json_path=source_json,
                dataset_id=dataset_id,
            )

            def _rej(reason: str):
                return ("rejected", {
                    "dataset_id": item.get("dataset_id", dataset_id),
                    "idx": item.get("idx"),
                    "errors": [reason],
                    "sample": whole_sample,
                })

            # -------- Agent 1: Inspector (check only) --------
            inspector_prompt = INSPECTOR_RULES.format(
                shared_core=shared_core,
                sample_block=sample_block,
            )
            inspector_result = llm_call_json(
                system_prompt="You are a strict QA inspector. Inspect only.",
                user_prompt=inspector_prompt,
                model=llm_model,
                images=verify_images,
                extra_create_params={
                    "custom_llm_provider": "openai",

                },
            )
            inspector_resp = inspector_result.get("json", {}) if isinstance(inspector_result, dict) else {}
            inspector_decision = inspector_resp.get("decision", "reject")
            inspector_can_fix = bool(inspector_resp.get("can_fix", False))
            inspector_issues = inspector_resp.get("issues", [])
            has_inspector_issue = isinstance(inspector_issues, list) and len(inspector_issues) > 0

            # Short-circuit: Inspector already says sample passes and no repair is needed.
            if inspector_decision == "pass" and (not inspector_can_fix or not has_inspector_issue):
                new_item = copy.deepcopy(item)
                if not _has_real_output(current_final):
                    return _rej("missing_output")
                ok_schema, schema_reason = _strict_schema_check(current_final, sample_schema)
                if not ok_schema:
                    return _rej(f"schema_mismatch:{schema_reason}")
                final_sample = _normalize_current_by_schema_and_type(
                    current_final, answer_type=answer_type, sample_schema=sample_schema
                )
                new_item["sample"]["status"] = "ok"
                new_item["sample"]["current"] = final_sample
                return ("verified", new_item)

            # Reject directly if Inspector says unfixable.
            if inspector_decision == "reject":
                return _rej(inspector_resp.get("reason", "inspector_rejected"))

            # Defensive fallback: if Inspector does not reject but also says no fix, skip
            # Planner/Fixer to avoid unnecessary calls and keep sample as-is.
            if not inspector_can_fix:
                new_item = copy.deepcopy(item)
                if not _has_real_output(current_final):
                    return _rej("missing_output")
                ok_schema, schema_reason = _strict_schema_check(current_final, sample_schema)
                if not ok_schema:
                    return _rej(f"schema_mismatch:{schema_reason}")
                final_sample = _normalize_current_by_schema_and_type(
                    current_final, answer_type=answer_type, sample_schema=sample_schema
                )
                new_item["sample"]["status"] = "ok"
                new_item["sample"]["current"] = final_sample
                return ("verified", new_item)

            # -------- Agent 2: Repair Planner (plan only) --------
            planner_prompt = REPAIR_PLANNER_RULES.format(
                shared_core=shared_core,
                sample_block=sample_block,
                inspector_output=json.dumps(
                    {
                        "decision": inspector_decision,
                        "can_fix": inspector_can_fix,
                        "issues": inspector_issues,
                        "reason": inspector_resp.get("reason"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            planner_result = llm_call_json(
                system_prompt="You are a repair planner. Plan only.",
                user_prompt=planner_prompt,
                model=llm_model,
                images=verify_images,
                extra_create_params={
                    "custom_llm_provider": "openai",
                },
            )
            planner_resp = planner_result.get("json", {}) if isinstance(planner_result, dict) else {}
            plan_status = planner_resp.get("plan_status", "not_fixable")
            action_plan = planner_resp.get("action_plan", [])

            if plan_status == "not_fixable":
                return _rej(planner_resp.get("reason", "planner_not_fixable"))

            # -------- Agent 3: Fixer (apply plan) --------
            fixer_prompt = FIXER_RULES.format(
                shared_core=shared_core,
                sample_block=sample_block,
                inspector_output=json.dumps(inspector_resp, ensure_ascii=False, indent=2),
                planner_output=json.dumps(planner_resp, ensure_ascii=False, indent=2),
            )
            fixer_result = llm_call_json(
                system_prompt="You are a precise fixer. Apply plan only.",
                user_prompt=fixer_prompt,
                model=llm_model,
                images=verify_images,
                extra_create_params={
                    "custom_llm_provider": "openai",
                },
            )
            resp = fixer_result.get("json", {}) if isinstance(fixer_result, dict) else {}
            status = resp.get("status", "rejected")

            if status == "rejected":
                return _rej(resp.get("reason", "unknown"))

            # Keep additional fields on the item unchanged; only update the sample field
            new_item = copy.deepcopy(item)
            fixed_sample = resp.get("sample")
            if fixed_sample is None:
                fixed_sample = current_final
            if not _has_real_output(fixed_sample):
                return _rej("missing_output")
            ok_schema, schema_reason = _strict_schema_check(fixed_sample, sample_schema)
            if not ok_schema:
                return _rej(f"schema_mismatch:{schema_reason}")
            fixed_sample = _normalize_current_by_schema_and_type(
                fixed_sample, answer_type=answer_type, sample_schema=sample_schema
            )

            if status == "fixed":
                new_item["sample"]["status"] = "fixed"
                new_item["sample"]["current"] = fixed_sample
                # Already the final version; no need to wrap in another 'final' layer
                new_item["sample"]["fix_reason"] = resp.get("reason", "unspecified")
            else:
                has_action = isinstance(action_plan, list) and len(action_plan) > 0
                has_issue = isinstance(inspector_issues, list) and len(inspector_issues) > 0
                if has_action or has_issue:
                    new_item["sample"]["status"] = "fixed"
                    new_item["sample"]["current"] = fixed_sample
                    new_item["sample"]["fix_reason"] = "planner/fixer applied or validated issue-oriented normalization"
                else:
                    new_item["sample"]["status"] = "ok"
                    new_item["sample"]["current"] = fixed_sample
            return ("verified", new_item)

        # Submit all uncached items to inner thread pool; collect results without shared mutable state.
        items_todo = [item for item in items if _verify_cache_key(item, dataset_id) not in done_keys]
        _inner_workers = min(sample_workers, len(items_todo)) if items_todo else 1
        new_verified: List[Dict[str, Any]] = []
        new_rejected: List[Dict[str, Any]] = []

        def _collect_one(kind, data):
            (new_verified if kind == "verified" else new_rejected).append(data)
            # Incremental checkpoint after every sample: a crash loses at most 1 sample's work.
            _save_checkpoint(subtask_id, dataset_id, {
                "verified": verified + new_verified,
                "rejected": rejected + new_rejected,
                "stats": _build_verify_stats(verified + new_verified, rejected + new_rejected),
            }, checkpoint_dir)

        if _inner_workers <= 1:
            for item in items_todo:
                kind, data = _process_item(item)
                _collect_one(kind, data)
                with progress_lock:
                    progress.update(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=_inner_workers) as inner_ex:
                future_to_item = {inner_ex.submit(_process_item, item): item for item in items_todo}
                for fut in concurrent.futures.as_completed(future_to_item):
                    try:
                        kind, data = fut.result()
                    except Exception as exc:
                        orig = future_to_item[fut]
                        kind, data = "rejected", {
                            "dataset_id": orig.get("dataset_id", dataset_id),
                            "idx": orig.get("idx"),
                            "errors": [f"worker_exception:{type(exc).__name__}:{exc}"],
                            "sample": orig.get("sample"),
                        }
                    _collect_one(kind, data)
                    with progress_lock:
                        progress.update(1)

        verified = verified + new_verified
        rejected = rejected + new_rejected

        result = {
            "verified": verified,
            "rejected": rejected,
            "stats": _build_verify_stats(verified, rejected),
        }

        _save_checkpoint(subtask_id, dataset_id, result, checkpoint_dir)
        return dataset_id, result

    # 6) Run all datasets concurrently
    verified_buffer: Dict[str, List[Dict[str, Any]]] = {}
    rejected_buffer: Dict[str, List[Dict[str, Any]]] = {}
    global_stats = {"total": 0, "ok": 0, "fixed": 0, "rejected": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_run_one_dataset, did, items)
            for did, items in grouped.items()
        ]

        for fut in concurrent.futures.as_completed(futures):
            dataset_id, data = fut.result()

            # Aggregate global stats (newly run + restored from checkpoint)
            for k in global_stats:
                global_stats[k] += data["stats"].get(k, 0)

            if data["verified"]:
                verified_buffer[dataset_id] = data["verified"]
            if data["rejected"]:
                rejected_buffer[dataset_id] = data["rejected"]

    progress.close()

    return {
        "verified_buffer": verified_buffer,
        "rejected_buffer": rejected_buffer,
        "stats": global_stats,
    }
