# Reproduction Guide

The public scripts support building training inputs from an authorized copy of the benchmark, training the dual-head policy, and scoring saved segmentations. The streaming inference runner and downstream evaluation pipeline are outside this release.

## Setup

From the repository root, install the dependencies.

```bash
python3 -m pip install -r requirements.txt
```

## Score saved segmentations

Supply a telemetry log and a prediction file for the same match.

```bash
python3 src/evaluate.py \
  --input /path/to/match.txt \
  --llm_result /path/to/predicted_segments.json \
  --output_dir output/evaluation
```

The JSON uses a `windows` array with `start` and `end` in seconds on the match timeline. The state values in this example are illustrative.

```json
{
  "windows": [
    {"start": 0.0, "end": 5.0, "state": "Looting"},
    {"start": 5.0, "end": 10.0, "state": "Combat"}
  ]
}
```

The evaluator uses separate time-bin construction paths for fixed windows and imported predictions. Compare saved predictions through the same input path when checking the effect of their boundaries.

## Build training inputs and train

Obtain an authorized copy of the telemetry logs, timestamped screenshots, reference-window and memory-card cache, and benchmark split. Set `TEST_MATCH_IDS_FILE` to a local text file with one held-out match ID per line. The constructor holds out those matches before splitting the remainder.

```bash
RUN_TAG=local_rebuild \
TLOG_BASE_DIR=/path/to/tlog_files \
FRAME_BASE_DIR=/path/to/frame_screenshots \
MEMORY_CACHE_PATH=/path/to/GT-Original.json \
TEST_MATCH_IDS_FILE=/path/to/held_out_match_ids.txt \
DEVICE=cuda:0 \
SKIP_DATASET=0 \
bash scripts/sft_train.sh
```

The launcher constructs samples, adds utility targets, builds the prepared cache, and trains the LoRA adapter plus the two task heads. The current-window text is built from telemetry and the reference window label. The memory-source option selects prior window cards. The launcher uses the defaults in [scripts/sft_train.sh](scripts/sft_train.sh). Check the paper for study settings before attempting to reproduce a reported run.

See the [checkpoint manifest](models/epistream-qwen25vl-3b-lora-0428/MANIFEST.md) for the released model artifacts and [the source map](src/README.md) for code entry points.
