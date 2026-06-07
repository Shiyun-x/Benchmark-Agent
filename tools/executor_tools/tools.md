# Executor Tools

Low-level transformation and conversion utilities used by the benchmark agent.

For full parameter specs, behavior descriptions, and I/O schemas of each tool, see [`utils/resources/tools.yaml`](../../../utils/resources/tools.yaml). That file also controls which tools are available to the agent for a given modality — adding a new entry there registers the tool in the pipeline.

To add your own tool: implement it in the appropriate `*_tools.py` file, then add a corresponding entry in `utils/resources/tools.yaml`.

---

## Image — `image_tools.py`

| Function | Description |
|---|---|
| `apply_image_degradation` | Apply one or more degradation ops to an image (entry point) |
| `degrade_spatial` | Spatial degradations: blur, downscale, rotation, etc. |
| `degrade_photometric` | Photometric degradations: brightness, contrast, color shift |
| `degrade_compression` | Compression artifacts (JPEG quality reduction) |
| `degrade_noise` | Add pixel-level noise |
| `degrade_occlusion` | Occlude regions of the image |
| `image2text` | Generate a text description of an image via VLM |
| `image2objects` | Detect objects and return bounding boxes with category labels |
| `ocr` | Extract text from an image via OCR |
| `mask_or_crop_to_bbox` | Crop or mask an image to a bounding box region |
| `bbox_to_point` | Convert a bounding box to a representative point |

---

## Audio — `audio_tools.py`

| Function | Description |
|---|---|
| `text2speech` | Synthesize speech from text (TTS) |
| `speech2text` | Transcribe audio to text (ASR) |
| `add_environmental_noise` | Mix environmental noise into an audio clip at a given SNR |
| `merge_dialog` | Concatenate a list of dialog audio segments into one clip |

### Deferred audio generation

During benchmark generation, **`text2speech` and `add_environmental_noise` do not synthesize audio**. Instead they record their call parameters (dialog script, noise type/intensity) into the sample's `_pure_tool_messages`. This keeps the pipeline fast and avoids requiring a GPU or noise library at generation time.

After the pipeline finishes, use the two batch scripts in `utils/audio_processing/` to perform the actual audio synthesis:

**Step 1 — TTS** (`utils/audio_processing/xtts_batch.py`):
- Model: [XTTS v2](https://huggingface.co/coqui/XTTS-v2) — multilingual, multi-speaker TTS via [Coqui TTS](https://github.com/idiap/coqui-ai-TTS). Pass `--model-path` to a local XTTS v2 checkpoint.
- Reads `_transcript` (dialog turns) and `_tts_output_path` from each record, synthesizes a merged wav and writes it to `_tts_output_path`.

**Step 2 — Noise** (`utils/audio_processing/add_noise.py`):
- Dataset: [TUT Urban Acoustic Scenes 2016](https://zenodo.org/record/165995) — labeled environmental noise clips (residential, city center, park, etc.). Pass `--noise-root` to the local dataset root (must contain `meta.txt`).
- Reads `_tts_output_path` (input wav) and `_add_environmental_noise_args` from each record, mixes at a target SNR, and writes the result to `_final_audio_path`.

```bash
# Step 1: TTS  (generates wav files to paths already recorded in the sample)
python utils/audio_processing/xtts_batch.py \
  --eval-json  cache/{topic_id}/evaluation_with_audio_settings.json \
  --model-path /path/to/xtts-v2

# Step 2: add noise (only for samples that have _add_environmental_noise_args)
python utils/audio_processing/add_noise.py \
  --eval-json  cache/{topic_id}/evaluation_with_audio_settings.json \
  --noise-root /path/to/TUT
```

Audio file paths (`_tts_output_path`, `_final_audio_path`) are determined at benchmark generation time by the stubs and recorded in the JSON — the batch scripts write to those exact paths.

---

## Adding a Custom Tool

**Step 1 — Implement the function** in the appropriate `*_tools.py` file (or a new file):

```python
def my_tool(arg1: str, arg2: int = 1) -> dict:
    # ... your logic ...
    return {"result_field": ...}
```

The function must return a plain `dict`. If it produces a file (audio, image), include the output path in the dict so the pipeline can record it.

**Step 2 — Register it in `utils/resources/tools.yaml`**:

Key fields to set: `name`, `modalities` (controls which task types the agent may select this tool for), `description`, `params`, and `typical_uses`. See the existing entries in the file for the full schema including `memory.retain` (which inputs/outputs to log in the sample record).

**Step 3 — Wire up the backend in `tools/executor_tools/run_pure_tools.py`**:

Add a `_build_spec_my_tool()` function that defines `param_schema` (the structured parameter spec the LLM planner sees), `return_schema`, and sets `backend=my_tool`. Then register it in `build_pure_tool_registry`:

```python
elif name == "my_tool":
    spec = _build_spec_my_tool(t)
```

---

## Note on partial implementations

This release includes implementations for the tools required by the example topics. 
Some tools (e.g. `speech2text`, `ocr`, `image2objects`) are provided as stubs and will be open-sourced in a future release.

---

## Web — `web_tools.py`

| Function | Description |
|---|---|
| `web_search` | Run a web search query and return summarized results |
