
from __future__ import annotations
from typing import Dict, Any, List, Optional,Tuple
from pathlib import Path
from uuid import uuid4
import random
import threading
import os
from pydub import AudioSegment
# from TTS.api import TTS
from functools import lru_cache

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = str(_PROJECT_ROOT / "workspace" / "audio")
NOISE_ROOT = os.getenv("NOISE_ROOT", str(_PROJECT_ROOT / "noise_library" / "TUT"))

# ================ XTTS Configuration ================

# _TTS_INIT_LOCK = threading.Lock()   
# _TTS_RUN_LOCK = threading.Lock()    
# _TTS_INSTANCE: Optional[TTS] = None
# _AVAILABLE_SPEAKERS: List[str] = []

# def _get_tts() -> TTS:
#     global _TTS_INSTANCE, _AVAILABLE_SPEAKERS
#     with _TTS_INIT_LOCK:
#         if _TTS_INSTANCE is None:
#             tts = TTS(
#                 model_name="tts_models/multilingual/multi-dataset/xtts_v2"
#             ).to("cuda")
#             _TTS_INSTANCE = tts

#             if hasattr(tts, "speakers") and isinstance(tts.speakers, (list, tuple)):
#                 _AVAILABLE_SPEAKERS = list(tts.speakers)
#             else:
#                 _AVAILABLE_SPEAKERS = []

#         return _TTS_INSTANCE

FEMALE_SPEAKER_NAMES = {
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
    "Henriette Usha",
    "Sofia Hellen",
    "Tammy Grit",
    "Tanja Adelina",
    "Vjollca Johnnie",
    "Nova Hogarth",
    "Maja Ruoho",
    "Uta Obando",
    "Lidiya Szekeres",
    "Szofi Granger",
    "Camilla Holmström",
    "Lilya Stainthorpe",
    "Zofija Kendrick",
    "Narelle Moon",
    "Barbora MacLean",
    "Alexandra Hisakawa",
    "Alma María",
    "Rosemary Okafor",
}

def merge_dialog(dialog_list):
    if not dialog_list:
        return []

    has_speaker = any("speaker" in d and d.get("speaker") for d in dialog_list)

    if not has_speaker:
        merged_text = " ".join(d.get("text", "") for d in dialog_list if d.get("text"))
        return [{"text": merged_text}]

    merged = []
    last = None

    for item in dialog_list:
        txt = item.get("text", "")
        spk = item.get("speaker")

        if not spk:
            spk = "__NO_SPEAKER__"

        if last is None:
            last = {"speaker": spk, "text": txt}
            continue

        if last["speaker"] == spk:
            last["text"] = last["text"] + " " + txt
        else:
            merged.append(last)
            last = {"speaker": spk, "text": txt}

    if last:
        merged.append(last)

    for m in merged:
        if m["speaker"] == "__NO_SPEAKER__":
            m.pop("speaker", None)

    return merged

# def _pick_voice_for_gender(gender: Optional[str]) -> str:

#     tts = _get_tts()
#     global _AVAILABLE_SPEAKERS
#     speakers = _AVAILABLE_SPEAKERS or getattr(tts, "speakers", None) or []

#     if not speakers:
#         return "default"

#     g = (gender or "auto").lower()

#     if g == "female":
#         pool = [s for s in speakers if s in FEMALE_SPEAKER_NAMES]
#     elif g == "male":
#         pool = [s for s in speakers if s not in FEMALE_SPEAKER_NAMES]
#     else:  # auto or other
#         pool = speakers

#     if not pool:
#         pool = speakers

#     return random.choice(pool)

# def _synthesize_segment(
#     text: str,
#     language: str,
#     speaker_name: str,
#     output_path: str,
# ) -> str:

#     text = (text or "").strip()
#     if not text:
#         raise ValueError("_synthesize_segment: text is empty")

#     tts = _get_tts()
#     out = Path(output_path)
#     out.parent.mkdir(parents=True, exist_ok=True)

#     with _TTS_RUN_LOCK:
#         tts.tts_to_file(
#             text=text,
#             speaker=speaker_name,
#             language=language,
#             file_path=str(out),
#         )

#     return str(out)



# ================ Add Environmental Noise Configuration ================
INTENSITY_TO_SNR = {
    "low": 15.0,
    "medium": 10.0,
    "high": 5.0,
}


@lru_cache(maxsize=8)
def _load_tut_noise_index(noise_root: str) -> Dict[str, List[Path]]:
    """
    Parse meta.txt from the TUT noise library and index entries by label:
        {
          "residential_area": [Path(...), Path(...), ...],
          "city_center": [...],
          ...
        }
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
            audio_path = root / rel_audio  # resolve to absolute path
            index.setdefault(label, []).append(audio_path)

    return index


def _tile_or_trim(noise: AudioSegment, target_ms: int) -> AudioSegment:
    if len(noise) == 0:
        return AudioSegment.silent(duration=target_ms)

    if len(noise) < target_ms:
        repeats = (target_ms // len(noise)) + 1
        noise = noise * repeats

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


# ================ Tools Interface ================
def text2speech(
    dialog: List[Dict[str, Any]],
    *args, **kwargs
) -> str:
    """
    transfer text to speech using XTTS model
    parameters
    ----------
        dialog: List of dict, each dict has keys:
            "speaker": str|int, speaker identifier
            "text": str, text content to synthesize
            "language": str, language code (e.g. 'en', 'zh'), optional
            "gender": str, 'male' | 'female' | 'auto', optional
    returns
        dict with keys:
            "merged_audio_path": path to merged audio file (absolute path)

    """
    output_root =  OUTPUT_DIR
    default_language = "en"
    if not dialog or not isinstance(dialog, list):
        raise ValueError("text2speech_tool: `dialog` must be a non-empty list")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    
    final_name = f"tts_{uuid4().hex}.wav"

    final_path = output_root_path / final_name

    # speakers_order: List[str] = []
    # speaker_first_gender: Dict[str, Optional[str]] = {}

    # for turn in dialog:
    #     raw_spk = turn.get("speaker")
    #     spk = str(raw_spk) if raw_spk is not None else "1"
    #     if spk not in speakers_order:
    #         speakers_order.append(spk)
    #         speaker_first_gender[spk] = turn.get("gender")  # may be None

    # speaker2voice: Dict[str, str] = {}
    # for spk in speakers_order:
    #     g = speaker_first_gender.get(spk)
    #     speaker2voice[spk] = _pick_voice_for_gender(g)

    # temp_dir = output_root_path / "chunks"
    # temp_dir.mkdir(parents=True, exist_ok=True)

    # full_audio = AudioSegment.silent(duration=0)
    # last_speaker: Optional[str] = None

    # for turn in dialog:
    #     raw_spk = turn.get("speaker")
    #     spk = str(raw_spk) if raw_spk is not None else "1"
    #     txt = (turn.get("text") or "").strip()
    #     if not txt:
    #         continue

    #     turn_lang = (turn.get("language") or default_language).lower()
    #     voice_name = speaker2voice.get(spk) or _pick_voice_for_gender(turn.get("gender"))

    #     chunk_name = f"chunk_{uuid4().hex}.wav"
    #     chunk_path = temp_dir / chunk_name

    #     _synthesize_segment(txt, turn_lang, voice_name, str(chunk_path))
    #     seg = AudioSegment.from_wav(chunk_path)

    #     if last_speaker is None:
    #         pass
    #     else:
    #         if spk == last_speaker:
    #             full_audio += AudioSegment.silent(duration=150)
    #         else:
    #             full_audio += AudioSegment.silent(duration=300)

    #     full_audio += seg
    #     try:
    #         chunk_path.unlink(missing_ok=True)
    #     except Exception:
    #         pass
    #     last_speaker = spk

    # full_audio.export(final_path, format="wav")

    return  {
        "merged_audio_path": str(final_path.resolve()),
    }


def speech2text(
    audio_path: str,
    *args, **kwargs
):
    """
    Transcribe an audio file to text (ASR).

    audio_path: path to the input audio file
    returns: transcribed text string
    """
    pass


def add_environmental_noise(
    audio_path: str,
    noise_type: str,
    intensity: str = "medium",
    noise_root: str = NOISE_ROOT,
    seed: Optional[int] = None,
    *args, **kwargs
) -> Dict[str, Any]:
    """
    add environmental_noise to audio file using TUT noise dataset

    parameters
    ----------
        audio_path: path to clean audio file
        noise_type: type of noise, as in TUT meta.txt labels
        intensity:  'low' | 'medium' | 'high', default 'medium'
        noise_root: root directory of TUT noise dataset (with audio/ and meta/meta.txt)
        seed:       random seed for selecting noise file (optional)
    returns
        dict with keys:
            "noisy_audio_path": path to noisy audio file (absolute path)
          
    """
    audio_path = Path(str(audio_path))  
    # if not audio_path.exists():
    #     raise FileNotFoundError(f"audio not found: {audio_path}")

    # index = _load_tut_noise_index(noise_root)
    # if noise_type not in index:
    #     noise_type = random.choice(list(index.keys()))

    # noise_candidates = index[noise_type]
    # if not noise_candidates:
    #     raise ValueError(f"No noise files for type '{noise_type}' in TUT index.")

    # rnd = random.Random(seed)
    # noise_path = rnd.choice(noise_candidates)

    # clean = AudioSegment.from_file(audio_path)
    # noise = AudioSegment.from_file(noise_path)

    # noise = _tile_or_trim(noise, target_ms=len(clean))
    # mixed, actual_snr = _mix_with_snr(clean, noise, intensity=intensity)

    noise_type = noise_type.replace("/", "_")
    out_root = audio_path.parent
    out_name = (
        f"{audio_path.stem}_noisy_{noise_type}_{intensity}_{uuid4().hex[:8]}.wav"
    )
    out_path = out_root / out_name

    # mixed.export(out_path, format="wav")

    return  {
        "noisy_audio_path": str(out_path.resolve())
    }

