# EpiStream 0428 Checkpoint Manifest

Base model — `Qwen/Qwen2.5-VL-3B-Instruct`

| File | Contents |
| --- | --- |
| `adapter_model.safetensors` | LoRA adapter weights |
| `adapter_config.json` | PEFT configuration and upstream model identifier |
| `regression_head.pt` | Commit-advantage regression head |
| `boundary_head.pt` | Auxiliary peak classification head |
| Tokenizer files and `chat_template.jinja` | Tokenizer configuration and chat template |

Use [the reproduction guide](../../REPRODUCE.md) to score saved segmentations.
