from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from uuid import uuid4
import random
import json
import argparse
import multiprocessing as mp
import math
import os

import torch
from pydub import AudioSegment
from TTS.api import TTS
from tqdm import tqdm


# ================= Basic configuration =================

_TTS_INSTANCE: Optional[TTS] = None
_AVAILABLE_SPEAKERS: List[str] = []

FEMALE_SPEAKER_NAMES = {
    "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema",
    "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara",
    "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
    "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Nova Hogarth",
    "Maja Ruoho", "Uta Obando", "Lidiya Szekeres", "Szofi Granger",
    "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick",
    "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa",
    "Alma María", "Rosemary Okafor"
}


# ================= TTS initialization =================

def _get_tts() -> TTS:
    global _TTS_INSTANCE, _AVAILABLE_SPEAKERS

    if _TTS_INSTANCE is not None:
        return _TTS_INSTANCE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO][PID {mp.current_process().pid}] Initializing XTTS on device: {device}")

    model_path = os.environ.get("XTTS_MODEL_PATH", "/model/xtts-v2")
    config_path = os.path.join(model_path, "config.json")
    tts = TTS(model_path=model_path, config_path=config_path).to("cuda:0")
    _TTS_INSTANCE = tts

    if hasattr(tts, "speakers") and tts.speakers:
        _AVAILABLE_SPEAKERS = list(tts.speakers)
        print(f"[INFO][PID {mp.current_process().pid}] Speakers loaded: {len(_AVAILABLE_SPEAKERS)}")
    else:
        _AVAILABLE_SPEAKERS = []
        print(f"[INFO][PID {mp.current_process().pid}] No speakers attribute; using 'default'")

    return tts


def _pick_voice_for_gender(gender: Optional[str]) -> str:
    tts = _get_tts()
    global _AVAILABLE_SPEAKERS
    speakers = _AVAILABLE_SPEAKERS or getattr(tts, "speakers", None) or []

    if not speakers:
        return "default"

    g = (gender or "auto").lower()

    if g == "female":
        pool = [s for s in speakers if s in FEMALE_SPEAKER_NAMES]
    elif g == "male":
        pool = [s for s in speakers if s not in FEMALE_SPEAKER_NAMES]
    else:
        pool = speakers

    if not pool:
        pool = speakers

    return random.choice(pool)


def _synthesize_segment(text: str, language: str, speaker_name: str, output_path: str) -> str:
    tts = _get_tts()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tts.tts_to_file(
        text=text,
        speaker=speaker_name,
        language=language,
        file_path=str(out),
    )
    return str(out)


# ================= Single-dialog synthesis =================

def text2speech_single(
    dialog: List[Dict[str, Any]],
    output_path: str,
    default_language: str = "en",
) -> Dict[str, Any]:
    """
    Synthesize one dialog to a single wav at output_path.
    Skips if the file already exists and is larger than 1 KB.
    Returns {"merged_audio_path": str, "status": "exists"|"generated"|"invalid"}.
    """
    final = Path(output_path)
    try:
        if final.exists():
            size = final.stat().st_size
            if size > 1024:
                return {"merged_audio_path": str(final.resolve()), "status": "exists"}
            else:
                print(f"[WARN][PID {mp.current_process().pid}] small/broken wav ({size} bytes), re-generate: {final}")

        if not dialog or not isinstance(dialog, list):
            print(f"[WARN][PID {mp.current_process().pid}] empty/invalid dialog, skipped")
            return {"merged_audio_path": str(final.resolve()), "status": "invalid"}

        final.parent.mkdir(parents=True, exist_ok=True)

        speakers: List[str] = []
        genders: Dict[str, Optional[str]] = {}
        for turn in dialog:
            spk = str(turn.get("speaker") or "1")
            if spk not in speakers:
                speakers.append(spk)
                genders[spk] = turn.get("gender")

        spk2voice = {spk: _pick_voice_for_gender(genders.get(spk)) for spk in speakers}

        tmp = final.parent / "chunks"
        tmp.mkdir(parents=True, exist_ok=True)

        full = AudioSegment.silent(duration=0)
        last = None

        for turn in dialog:
            txt = (turn.get("text") or "").strip()
            if not txt:
                continue

            spk = str(turn.get("speaker") or "1")
            lang = (turn.get("language") or default_language).lower()
            voice = spk2voice[spk]

            chunk = tmp / f"{uuid4().hex}.wav"
            _synthesize_segment(txt, lang, voice, chunk)
            seg = AudioSegment.from_wav(chunk)

            if last is not None:
                full += AudioSegment.silent(duration=150 if spk == last else 300)

            full += seg
            last = spk
            chunk.unlink(missing_ok=True)

        full.export(final, format="wav")
        return {"merged_audio_path": str(final.resolve()), "status": "generated"}
    except Exception as e:
        print(f"[ERROR][PID {mp.current_process().pid}] Exception during TTS: {e}")
        return {"merged_audio_path": str(final.resolve()), "status": "invalid"}


# ================= worker =================

def _process_one(args: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    item, default_language = args
    dialog = item.get("dialog")
    out = item.get("output_path")

    if not dialog or not isinstance(dialog, list) or not out or not isinstance(out, str):
        return {"merged_audio_path": None, "status": "invalid"}

    return text2speech_single(dialog, out, default_language=default_language)


# ================= CLI =================

def main():
    parser = argparse.ArgumentParser(
        description="Batch TTS from evaluation_with_audio_settings.json using XTTS v2"
    )
    parser.add_argument("--eval-json", required=True,
                        help="Path to evaluation_with_audio_settings.json")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local XTTS v2 model directory (overrides XTTS_MODEL_PATH env var)")
    parser.add_argument("--default-language", default="en")
    parser.add_argument("--target-per-worker", type=int, default=10,
                        help="Target samples per worker process (used to estimate pool size)")
    parser.add_argument("--max-workers", type=int, default=16)

    args = parser.parse_args()

    if args.model_path:
        os.environ["XTTS_MODEL_PATH"] = args.model_path

    with open(args.eval_json, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    items: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        transcript = record.get("_transcript")
        out_path = record.get("_tts_output_path")
        # TTS-only records may lack _tts_output_path; fall back to sample.input.audio_url
        if not out_path and not record.get("_add_environmental_noise_args"):
            out_path = ((record.get("sample") or {}).get("input") or {}).get("audio_url")
        if not transcript or not out_path:
            continue
        items.append({
            "record_idx": i,
            "dialog": transcript,
            "output_path": out_path,
        })

    if not items:
        print("[INFO] No records with _transcript + _tts_output_path found.")
    else:
        total = len(items)
        num_workers = min(
            max(1, math.ceil(total / args.target_per_worker)),
            args.max_workers,
            mp.cpu_count(),
        )
        print(f"[INFO] TTS: {total} items, {num_workers} workers")

        task_iter = [(item, args.default_language) for item in items]
        generated = exists = invalid = 0

        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_process_one, task_iter), total=total, desc="TTS"))

        for item, result in zip(items, results):
            status = (result or {}).get("status", "invalid")
            if status == "generated":
                generated += 1
            elif status == "exists":
                exists += 1
            else:
                invalid += 1

        print(f"[INFO] TTS done: generated={generated}, exists={exists}, invalid={invalid}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
