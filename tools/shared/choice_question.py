from typing import Any, Dict, List, Optional, Tuple
import hashlib
import random
import re


_CHOICE_LABELS = ("A", "B", "C", "D", "E")


def stable_seed(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _split_stem_and_options(question: str) -> Optional[Dict[str, Any]]:
    if not isinstance(question, str):
        return None
    q = question.strip()
    if not q:
        return None

    lines = [ln.rstrip() for ln in q.splitlines()]
    opt_line_re = re.compile(r"^\s*([A-E])\s*[\)\.\:]\s*(.+?)\s*$")
    opt_starts = []
    parsed_opts: List[tuple[str, str]] = []

    for i, ln in enumerate(lines):
        m = opt_line_re.match(ln)
        if not m:
            continue
        opt_starts.append(i)
        parsed_opts.append((m.group(1), m.group(2)))

    if len(parsed_opts) >= 3:
        first_idx = opt_starts[0]
        stem = "\n".join(lines[:first_idx]).strip()
        opts: List[tuple[str, str]] = []
        expected_idx = 0
        for i in range(first_idx, len(lines)):
            m = opt_line_re.match(lines[i])
            if not m:
                break
            label, text = m.group(1), m.group(2)
            if expected_idx < len(_CHOICE_LABELS) and label != _CHOICE_LABELS[expected_idx]:
                pass
            opts.append((label, text))
            expected_idx += 1
        if len(opts) >= 3:
            return {"stem": stem, "options": opts}

    inline_re = re.compile(r"(?is)\boptions\s*[:：]\s*(.+)$")
    m = inline_re.search(q)
    if not m:
        return None
    after = m.group(1).strip()
    parts = re.split(r"(?i)\b([A-E])\s*[\)\.\:]\s*", after)
    if len(parts) < 5:
        return None

    opts2: List[tuple[str, str]] = []
    it = iter(parts[1:])
    for label, text in zip(it, it):
        lab = label.strip().upper()
        txt = text.strip()
        if not lab or lab not in _CHOICE_LABELS or not txt:
            continue
        opts2.append((lab, txt))
    if len(opts2) < 3:
        return None
    stem = q[: m.start()].strip()
    return {"stem": stem, "options": opts2}


def _is_standalone_options_header_line(line: str) -> bool:
    return bool(re.fullmatch(r"(?i)options\s*[:：]?\s*", (line or "").strip()))


def _last_nonempty_line_is_options_header(text: str) -> bool:
    if not (text or "").strip():
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return _is_standalone_options_header_line(lines[-1])


def _collapse_consecutive_trailing_options_headers(stem: str) -> str:
    s = (stem or "").rstrip()
    if not s:
        return ""
    lines = s.splitlines()
    while len(lines) >= 2:
        if _is_standalone_options_header_line(lines[-1]) and _is_standalone_options_header_line(lines[-2]):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def _rebuild_question(stem: str, options: List[tuple[str, str]]) -> str:
    stem = _collapse_consecutive_trailing_options_headers((stem or "").strip())
    out_lines: List[str] = []
    if stem:
        out_lines.append(stem)
    if not _last_nonempty_line_is_options_header(stem):
        out_lines.append("Options:")
    for label, text in options:
        out_lines.append(f"{label}. {text}".strip())
    return "\n".join(out_lines).strip()


def _trim_stem_keep_last_question_sentence(stem: str) -> str:
    s = (stem or "").strip()
    if not s:
        return s
    if len(s) <= 220:
        return s
    qpos = s.rfind("?")
    if qpos != -1:
        start = max(s.rfind("\n", 0, qpos), s.rfind(". ", 0, qpos), s.rfind("。", 0, qpos))
        start = 0 if start == -1 else start + 1
        cand = s[start: qpos + 1].strip()
        if cand:
            return cand
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else s


def normalize_and_shuffle_choice_question(
    question: str,
    answer: str,
    seed: int,
    *,
    has_audio: bool = False,
) -> Optional[Dict[str, str]]:
    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    ans = answer.strip().upper()
    if ans not in _CHOICE_LABELS:
        return None

    parsed = _split_stem_and_options(question)
    if not parsed:
        return None

    stem = str(parsed["stem"] or "")
    options_in: List[tuple[str, str]] = list(parsed["options"] or [])
    if len(options_in) < 3:
        return None

    label_to_text: Dict[str, str] = {}
    for lab, txt in options_in:
        lab_u = str(lab).strip().upper()
        if lab_u in _CHOICE_LABELS and lab_u not in label_to_text:
            label_to_text[lab_u] = str(txt).strip()

    if ans not in label_to_text:
        return None

    original_order = [lab for lab, _ in options_in if lab in label_to_text]
    unique_labs = list(dict.fromkeys(original_order))
    option_texts = [(lab, label_to_text[lab]) for lab in unique_labs if label_to_text.get(lab)]
    if len(option_texts) < 3:
        return None

    if has_audio:
        stem = _trim_stem_keep_last_question_sentence(stem)
        if stem and "audio" not in stem.lower() and "recording" not in stem.lower():
            stem = f"Based on the audio, {stem[0].lower() + stem[1:] if len(stem) > 1 else stem}"

    rng = random.Random(seed)
    shuffled = option_texts[:]
    rng.shuffle(shuffled)

    new_options: List[tuple[str, str]] = []
    old_label_to_new: Dict[str, str] = {}
    for i, (old_lab, txt) in enumerate(shuffled):
        if i >= len(_CHOICE_LABELS):
            break
        new_lab = _CHOICE_LABELS[i]
        new_options.append((new_lab, txt))
        old_label_to_new[old_lab] = new_lab

    new_answer = old_label_to_new.get(ans)
    if not new_answer:
        return None

    new_question = _rebuild_question(stem, new_options)
    return {"question": new_question, "answer": new_answer}
