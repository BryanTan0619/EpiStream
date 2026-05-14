# Source Map

This directory contains only the paper-method path.

## Main Pipeline

- `map.py`: telemetry parsing and state/evidence mapping.
- `segment_gt.py`: offline oracle episode construction.
- `compute_utility_forward_looking.py`: forward-looking utility target
  computation for training.
- `evaluate.py`: final evaluation entry point.
- `sft_train.py`: LoRA training plus EpiStream regression and boundary heads.

Training sample construction lives in `../dataset/dataset_construction.py`.
The released model checkpoint lives in
`../models/epistream-qwen25vl-3b-lora-0428/`.