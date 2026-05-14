# EpiStream 0428 Best Checkpoint

Source checkpoint:

`output/vlm_method2_regression_adv_memory_schema_vlm_v1_0428/best_model`

Base model:

`Qwen/Qwen2.5-VL-3B-Instruct`

Included artifacts:

- `adapter_model.safetensors`: LoRA adapter weights
- `adapter_config.json`: sanitized PEFT adapter config
- `regression_head.pt`: commit-advantage regression head
- `boundary_head.pt`: auxiliary peak/boundary classification head
- tokenizer files saved with the checkpoint

This directory intentionally does not include raw tlog data, screenshots,
training JSONL files, intermediate checkpoints, or API credentials.
