# Reproducing Paper Results

This release is intentionally small. It keeps the code needed to rebuild and
evaluate the paper method, and removes one-off rollout, merge, batch, and
downstream exploration scripts.

Raw benchmark telemetry and screenshots are not included in this public folder.
Set the paths below to your local or approved benchmark copy.

## Environment

```bash
pip install -r requirements.txt
export TLOG_BASE_DIR=/path/to/tlog_files
export FRAME_BASE_DIR=/path/to/frame_screenshots
```

## Released Checkpoint

The 0428 paper checkpoint is included at:

```bash
models/epistream-qwen25vl-3b-lora-0428
```

It is a LoRA adapter plus the EpiStream regression and boundary heads. The base
model is `Qwen/Qwen2.5-VL-3B-Instruct`.

## Final Evaluation

Use `src/evaluate.py` as the final evaluation entry point. It compares a
predicted segmentation JSON against telemetry-derived paper metrics:

```bash
python src/evaluate.py \
  --input "$TLOG_BASE_DIR/<match_id>.txt" \
  --llm_result path/to/predicted_segments.json \
  --output_dir output/evaluation
```

The prediction JSON should contain a `windows` list with `start`/`end` or
`start_time`/`end_time` fields.

## Rebuilding the Training Pipeline

Use this only if you have the released benchmark splits, telemetry logs, and
frame screenshots.

```bash
RUN_TAG=paper \
TLOG_BASE_DIR=/path/to/tlog_files \
FRAME_BASE_DIR=/path/to/frame_screenshots \
DEVICE=cuda:0 \
EPOCHS=5 \
SKIP_DATASET=0 \
bash scripts/sft_train.sh
```

The script performs:

1. dataset construction
2. forward-looking utility target computation
3. advantage-cache preparation
4. LoRA plus EpiStream head training

## Source Map

See `src/README.md` for the canonical entry points. Files not listed as a main
entry point are helpers or analysis scripts.
