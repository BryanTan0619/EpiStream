# Model Release

The [EpiStream 0428 checkpoint](models/epistream-qwen25vl-3b-lora-0428/) contains a LoRA adapter, two task heads, and tokenizer files for `Qwen/Qwen2.5-VL-3B-Instruct`. The [checkpoint manifest](models/epistream-qwen25vl-3b-lora-0428/MANIFEST.md) lists the files.

To use the checkpoint, load the upstream base model and apply the PEFT adapter. Use both task heads for the trained dual-head output. The regression head estimates commit advantage and the auxiliary head produces a peak classification logit. See [src/sft_train.py](src/sft_train.py) for the head definitions and the saved-checkpoint loading code.

The [reproduction guide](REPRODUCE.md) describes the public training and offline scoring entry points.
