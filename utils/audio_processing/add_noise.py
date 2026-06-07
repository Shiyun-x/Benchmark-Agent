#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch-add environmental noise to speech using the TUT noise library.

Reads evaluation_with_audio_settings.json produced by export_results.py,
processes each record that has both _tts_output_path and _add_environmental_noise_args,
and writes the noisy wav to the path stored in _final_audio_path.
"""

from functools import lru_cache
from pydub import AudioSegment
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import json
import argparse
from tqdm import tqdm


# ============ Configuration ============

NOISE_ROOT = r"/data/noise_lib/TUT"

INTENSITY_TO_SNR: Dict[str, float] = {
    "low": 15.0,
    "medium": 10.0,
    "high": 5.0,
}


# ============ Noise index and mixing helpers ============

@lru_cache(maxsize=8)
def _load_tut_noise_index(noise_root: str) -> Dict[str, List[Path]]:
    """
    Load meta.txt from the TUT noise library and index entries by label:
        {"residential_area": [Path(...), ...], "city_center": [...], ...}
    """
    root = Path(noise_root)
    meta_path = root / "meta.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"TUT meta file not found: {meta_path}")

    index: Dict[str, List[Path]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_audio, label = parts[0], parts[1]
            index.setdefault(label, []).append(root / rel_audio)

    return index


def _tile_or_trim(noise: AudioSegment, target_ms: int) -> AudioSegment:
    if len(noise) == 0:
        return AudioSegment.silent(duration=target_ms)
    if len(noise) < target_ms:
        noise = noise * ((target_ms // len(noise)) + 1)
    return noise[:target_ms]


def _mix_with_snr(
    clean: AudioSegment,
    noise: AudioSegment,
    intensity: str,
) -> Tuple[AudioSegment, float]:
    snr_target = INTENSITY_TO_SNR.get(intensity, 10.0)
    clean_db = clean.dBFS if clean.dBFS is not None else -20.0
    noise_db = noise.dBFS if noise.dBFS is not None else -20.0
    target_noise_db = clean_db - snr_target
    gain = target_noise_db - noise_db
    noise_adj = noise.apply_gain(gain)
    mixed = clean.overlay(noise_adj)
    noise_adj_db = noise_adj.dBFS if noise_adj.dBFS is not None else target_noise_db
    actual_snr = clean_db - noise_adj_db
    return mixed, actual_snr


# ============ Core functions ============

def add_environmental_noise(
    audio_path: str,
    output_path: str,
    noise_type: str,
    intensity: str = "medium",
    noise_root: str = NOISE_ROOT,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    audio_path = Path(str(audio_path))
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    index = _load_tut_noise_index(noise_root)
    if not noise_type or noise_type not in index:
        noise_type = random.choice(list(index.keys()))

    noise_candidates = index[noise_type]
    if not noise_candidates:
        raise ValueError(f"No noise files for type '{noise_type}' in TUT index.")

    rnd = random.Random(seed)
    noise_path = rnd.choice(noise_candidates)

    clean = AudioSegment.from_file(audio_path)
    noise = AudioSegment.from_file(noise_path)
    noise = _tile_or_trim(noise, target_ms=len(clean))
    mixed, actual_snr = _mix_with_snr(clean, noise, intensity=intensity)

    output_path = Path(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".wav")

    mixed.export(output_path, format="wav")

    return {
        "noisy_audio_path": str(output_path.resolve()),
        "noise_type": noise_type,
        "intensity": intensity,
        "actual_snr": actual_snr,
    }


def _process_item(item: Dict[str, Any], noise_root: str) -> Dict[str, Any]:
    audio_path = item.get("audio_path")
    output_path = item.get("output_path")

    if not audio_path:
        return {"status": "skip_no_audio", "item": item}
    if not output_path:
        return {"status": "skip_no_output", "item": item}
    if not Path(str(audio_path)).exists():
        return {"status": "skip_not_found", "item": item}

    try:
        res = add_environmental_noise(
            audio_path=audio_path,
            output_path=output_path,
            noise_type=item.get("noise_type", ""),
            intensity=item.get("intensity", "medium"),
            noise_root=noise_root,
        )
        return {"status": "ok", "result": res, "item": item}
    except Exception as e:
        print(f"[ERROR] {audio_path}: {e}")
        return {"status": "error", "reason": str(e), "item": item}


# ============ CLI =================

def main():
    parser = argparse.ArgumentParser(
        description="Batch add TUT environmental noise from evaluation_with_audio_settings.json"
    )
    parser.add_argument("--eval-json", required=True,
                        help="Path to evaluation JSON with _tts_output_path + _add_environmental_noise_args")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads (<=1 for sequential)")
    parser.add_argument("--noise-root", type=str, default=NOISE_ROOT,
                        help="Root dir of TUT dataset (must contain meta.txt)")

    args = parser.parse_args()

    with open(args.eval_json, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    items_to_process: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        noise_args = record.get("_add_environmental_noise_args") or {}
        tts_path = record.get("_tts_output_path")
        final_path = record.get("_final_audio_path")
        # Fall back: noisy output path stored in sample.input.audio_url
        if not final_path:
            final_path = ((record.get("sample") or {}).get("input") or {}).get("audio_url")
        if not noise_args or not tts_path or not final_path:
            continue
        items_to_process.append({
            "record_idx": i,
            "audio_path": tts_path,
            "output_path": final_path,
            "noise_type": noise_args.get("noise_type", ""),
            "intensity": noise_args.get("intensity", "medium"),
        })

    if not items_to_process:
        print("[INFO] No records with _tts_output_path + _add_environmental_noise_args + _final_audio_path found.")
    else:
        success = skipped = errors = 0

        def _handle(res: Dict[str, Any]) -> None:
            nonlocal success, skipped, errors
            if res["status"] == "ok":
                success += 1
            elif res["status"].startswith("skip"):
                skipped += 1
            else:
                errors += 1

        if args.workers <= 1:
            for item in tqdm(items_to_process, desc="Adding noise", ncols=100):
                _handle(_process_item(item, args.noise_root))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_process_item, item, args.noise_root): item
                    for item in items_to_process
                }
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="Adding noise", ncols=100):
                    _handle(fut.result())

        print(f"\n===== SUMMARY =====\nSuccess: {success}\nSkipped: {skipped}\nErrors : {errors}")


if __name__ == "__main__":
    main()
