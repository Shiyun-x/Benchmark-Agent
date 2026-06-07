import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _extract_audio_tool_args(sample: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Extract text2speech and add_environmental_noise tool args from a sample."""
    messages = sample.get("_pure_tool_messages", {})
    tts_args: Dict = {}
    noise_args: Dict = {}
    if isinstance(messages, dict):
        tts_args = messages.get("text2speech_tool_args") or {}
        noise_args = messages.get("add_environmental_noise_tool_args") or {}
    elif isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if not tts_args:
                tts_args = msg.get("text2speech_tool_args") or {}
            if not noise_args:
                noise_args = msg.get("add_environmental_noise_tool_args") or {}
    return tts_args, noise_args


def _has_audio_args(verified_buffer: Dict[str, Any]) -> bool:
    """Return True if any sample in the buffer contains audio tool args."""
    for items in verified_buffer.values():
        if not isinstance(items, list):
            continue
        for item in items:
            sample = item.get("sample") or {}
            tts, noise = _extract_audio_tool_args(sample)
            if tts or noise:
                return True
    return False


def _find_audio_path_in_final(current: Dict[str, Any]) -> Optional[str]:
    """Scan current.input for a string value that looks like an audio file path."""
    try:
        inp = current.get("input") or {}
    except Exception:
        return None
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    for v in inp.values():
        if isinstance(v, str) and Path(v).suffix.lower() in audio_exts:
            return v
    return None


def _sort_key(path: Path) -> int:
    m = re.search(r"_(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def export_evaluation(cache_path: str) -> str:
    """
    Read all verified_transformed_buffer_st_*.json files from {cache_path}/verify_log/,
    auto-detect whether audio tool args are present, and write:
      - evaluation.json                      (no audio)
      - evaluation_with_audio_settings.json  (audio args present)

    Returns the path of the written file.
    """
    verify_dir = Path(cache_path) / "verify_log"
    files = sorted(verify_dir.glob("verified_transformed_buffer_st_*.json"), key=_sort_key)

    if not files:
        print(f"[export_results] No verified buffer files found in {verify_dir}")
        return ""

    # Load all verified buffers
    buffers: Dict[str, Dict] = {}
    for fp in files:
        key = "_".join(fp.stem.split("_")[-2:])  # e.g. st_01
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        buffers[key] = data.get("verified_buffer") or {}

    # Detect whether any sample has audio tool args
    has_audio = any(_has_audio_args(buf) for buf in buffers.values())

    evaluations: List[Dict] = []
    for subtask_id, verified_buffer in buffers.items():
        for dataset_id, items in verified_buffer.items():
            if not isinstance(items, list):
                continue
            for item in items:
                sample = item.get("sample") or {}
                original = sample.get("original") or {}
                current = sample.get("current") or {}

                record: Dict[str, Any] = {
                    "subtask_id": subtask_id,
                    "dataset_id": dataset_id,
                    "idx": item.get("idx", -1),
                    "id": original.get("id", ""),
                    "sample": current,
                }

                if has_audio:
                    tts_args, noise_args = _extract_audio_tool_args(sample)
                    if tts_args:
                        record["_transcript"] = tts_args.get("dialog", [])

                    noise_type = noise_args.get("noise_type", "")
                    if noise_type:
                        record["_add_environmental_noise_args"] = {
                            "noise_type": noise_type,
                            "intensity": noise_args.get("intensity", "medium"),
                        }
                        # audio_path in noise_args = the TTS output (input to add_noise)
                        tts_path = noise_args.get("audio_path")
                        if tts_path:
                            record["_tts_output_path"] = tts_path
                        # final audio path in the sample = noisy wav
                        final_audio = _find_audio_path_in_final(current)
                        if final_audio:
                            record["_final_audio_path"] = final_audio
                    else:
                        # TTS-only: final audio IS the TTS output
                        final_audio = _find_audio_path_in_final(current)
                        if final_audio:
                            record["_tts_output_path"] = final_audio

                evaluations.append(record)

    # Always write evaluation.json (clean version without audio args)
    eval_records = [
        {k: v for k, v in rec.items() if not k.startswith("_")}
        for rec in evaluations
    ]
    eval_path = os.path.join(cache_path, "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, ensure_ascii=False, indent=2)
    print(f"[export_results] Exported {len(eval_records)} samples -> {eval_path}")

    if has_audio:
        audio_path = os.path.join(cache_path, "evaluation_with_audio_settings.json")
        with open(audio_path, "w", encoding="utf-8") as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        print(f"[export_results] Exported {len(evaluations)} samples -> {audio_path}")
        return audio_path

    return eval_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export verified buffers to evaluation JSON(s)")
    parser.add_argument("--cache-path", required=True,
                        help="Path to cache/{topic_id}/ directory containing verify_log/")
    _args = parser.parse_args()
    export_evaluation(_args.cache_path)
