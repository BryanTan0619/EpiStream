# Model Release Notes

Yes, the trained model parameters can be released, but release only the minimal
artifacts required to reproduce the paper numbers.

## What To Release

For the main paper model, publish one selected `best_model/` directory, not every
training checkpoint. A complete EpiStream policy checkpoint should include:

- `adapter_model.safetensors`
- `adapter_config.json`
- `regression_head.pt`
- `boundary_head.pt` if the reported model uses the auxiliary boundary head
- tokenizer files saved with the checkpoint
- a small manifest recording the exact base model, threshold, code commit, and
  evaluation command

The selected paper checkpoint has been copied to:

- `models/epistream-qwen25vl-3b-lora-0428/`

Each selected checkpoint is roughly 31 MB including LoRA adapter, heads, and
tokenizer files.

## What Not To Release

Do not release:

- intermediate `checkpoint_step*` or `checkpoint_epoch*` directories
- raw `tlog` files, videos, screenshots, frame dumps, or per-match JSONL samples
- generated cache/output folders unless they are sanitized aggregate metrics
- private API tokens, `.env` files, internal absolute paths, or tensorboard logs
- full Qwen base-model weights unless the upstream license and hosting policy
  explicitly allow redistribution

## Required Sanitization

Before uploading a checkpoint, edit `adapter_config.json` so:

```json
"base_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct"
```

The released checkpoint in `models/epistream-qwen25vl-3b-lora-0428/` has already
been sanitized.

## Reproducibility Package

To let others reproduce the paper results without exposing private data, release:

- code in this `release/` folder
- one or more sanitized model checkpoints
- sanitized benchmark split metadata and labels, if allowed
- either public/synthetic sample data for smoke tests, or documented access
  instructions for the real benchmark
- aggregate result JSON/CSV/TEX tables used in the paper
- exact commands for training-free evaluation from the released checkpoint

If the real DAS-Utility benchmark cannot be public, the public repository should
state that the released checkpoint reproduces the paper numbers on the controlled
benchmark, while smoke-test data only verifies that the code path runs.
