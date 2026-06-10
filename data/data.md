# Data

This directory contains a curated subset of datasets used for benchmark examples generation, along with their dataset cards.

Only a subset of samples is included here due to the licensing restrictions of the original datasets.

**To download the complete dataset**, visit the Hugging Face repository:

**[🤗 General-Level/General-Bench-Openset](https://huggingface.co/datasets/General-Level/General-Bench-Openset)**

We use the **`nlp`** subset and the **`image/comprehension`** subset from that repository.

---

## Directory Structure

```
data/
├── dataset_cards/          # One JSON card per dataset ({dataset_id}_card.json)
└── datasets/
    ├── nlp/                # Text-only datasets
    │   ├── Causal-Reasoning/
    │   │   └── annotation.json
    │   ├── Dialogue-Relation-Extraction/
    │   └── ...
    └── image/              # Image+text datasets
        ├── ArtImageVisualQuestionAnswering/
        │   ├── art_image_visual_question_answering.json
        │   └── images/
        ├── BirdDetection/
        │   ├── json/
        │   │   └── *.json
        │   └── images/
        └── ...
```

Place downloaded datasets following this layout. NLP datasets go under `datasets/nlp/<DatasetName>/`, image datasets go under `datasets/image/<DatasetName>/` with a sibling `images/` folder for the image files.

---

## Dataset Format

Every dataset is a JSON file with top-level metadata and a `data` array:

```json
{
  "task": "...",
  "type": "...",
  "modality": { "in": ["text", "image"], "out": ["text"] },
  "data": [ ... ]
}
```

Each entry in `data[]` has a fixed envelope of `id`, `input`, and `output`. The fields inside `input` and `output` vary by dataset. A sample may carry any combination of fields such as `question`, `options`, `context`, `image_file`, `prompt`, `answer`, `text`, and more:

```json
{
  "id": "sample_001",
  "input": {
    "question": "...",
    "options": ["...", "..."],
    "image_file": "foo.jpg"
  },
  "output": {
    "answer": "A"
  }
}
```

Key fields:
- `id`: unique sample identifier within the dataset
- `input`: all model inputs; field names and count vary by task and modality
- `output`: ground-truth response

---

## Adding Your Own Datasets

You can add custom datasets to `datasets/nlp/` or `datasets/image/` following the same format above. Then generate its card and register it in the dataset config.

### Generating Dataset Cards

Dataset cards are stored in `dataset_cards/` and named `{dataset_id}_card.json`. The ID is derived from a hash of the source file path.

**1. Edit the config** (`utils/build_dataset_cards/built_dataset_card.config.json`):

```json
{
  "build": {
    "root_dir": "data/datasets/image",
    "modality": "image",
    "out_dir": "data/dataset_cards"
  }
}
```

Set `root_dir` to the folder containing your new dataset(s), and `modality` to `"nlp"` or `"image"` accordingly.

**2. Run the script** from the project root:

```bash
python utils/build_dataset_cards/built_dataset_card.py
# or with a custom config:
python utils/build_dataset_cards/built_dataset_card.py --config path/to/config.json
```

The script scans `root_dir` for JSON files and generates a `{dataset_id}_card.json` for each dataset in `out_dir`.

**3. Register the new dataset** in `utils/resources/dataset_cards.yaml`:

```yaml
datasets:
  - <new_dataset_id>   # Your Dataset Name
  - ...                # existing entries
```

The `dataset_id` is printed by the script and also appears as the filename prefix of the generated card (e.g. `a1b2c3d4_card.json` → ID is `a1b2c3d4`).
