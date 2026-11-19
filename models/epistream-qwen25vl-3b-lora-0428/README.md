---
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags:
- peft
- lora
- temporal-segmentation
---

# EpiStream 0428

This is the released EpiStream adapter and task-head checkpoint for utility-aware episode segmentation in dense gameplay streams. The model uses recent frames and a text prompt containing episode context to estimate commit advantage and predict high-advantage decision points.

- **Base model** — [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **Adapter** — PEFT LoRA on `q_proj` and `v_proj`, with rank 16, alpha 32, and dropout 0.05
- **Task heads** — scalar commit-advantage regression and auxiliary peak classification
- **Training inputs** — sampled frames, telemetry-derived current-window text conditioned on reference labels, and previous-episode memory cards

See the [manifest](MANIFEST.md) for the files and the [project README](../../README.md) for the paper and evaluation entry points.
