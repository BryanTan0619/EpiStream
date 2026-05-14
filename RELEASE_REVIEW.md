# Release Review

Status: release folder is reduced to the main paper-method path.

## Remaining Risks

1. No raw telemetry data should be committed. Keep `tlog`, frame screenshots,
   videos, JSONL training examples, model checkpoints, and generated caches out
   of git.

2. The released checkpoint has a sanitized `adapter_config.json`; keep future
   checkpoints on public base-model IDs, not internal filesystem paths.

## Fixed In This Release Folder

- Added `.gitignore` rules for credentials, raw telemetry, frames, videos,
  caches, outputs, checkpoints, and Python build artifacts.
- Fixed `scripts/sft_train.sh` to call the released
  `src/compute_utility_forward_looking.py` filename.
- Updated README privacy notes and corrected the citation key typo.
- Added `MODEL_RELEASE.md` with the minimal model artifact policy.
- Added `REPRODUCE.md` and `src/README.md` to make the public reproduction path
  explicit.
- Renamed the released checkpoint directory to
  `models/epistream-qwen25vl-3b-lora-0428/`.
- Removed one-off rollout, merge, batch, downstream-task, threshold-tuning, and
  debug scripts from `src/` and `scripts/`.
- Removed duplicate legacy model-code and internal cleanup scripts from the
  public package.

## Security Scan Summary

- No committed raw `*.tlog`, `*.jsonl`, checkpoint, or `.env` files were found
  inside `release`.
- No obvious literal API key pattern was found in `release`.
- API tokens should be supplied only through environment variables such as
  `OPENAI_API_KEY` or project-specific token variables, never hard-coded.

## Suggested Final Pass

Run these from the `release` directory before creating the GitHub repo:

```bash
rg -n --hidden -S "(api[_-]?key|secret|token|password|Bearer |sk-|AKIA|hf_)"
rg -n --hidden -S "(/private_mount|/internal_data|/absolute/local/path)"
find . -type f \( -name "*.tlog" -o -name "*.jsonl" -o -name "*.log" \)
find . -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" -o -name "*.ckpt" \)
python3 -m py_compile $(find src dataset models -name "*.py")
bash -n scripts/*.sh
```
