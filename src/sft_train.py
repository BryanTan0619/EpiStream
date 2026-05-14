
import os, sys, json, math, time, argparse, random
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

VL_MODEL_PATH = os.environ.get("VL_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")

UTILITY_CACHE_DIR = os.path.join(PROJECT_DIR, "cache", "utility")
DEFAULT_TRAIN_CACHE = os.path.join(UTILITY_CACHE_DIR, "train_with_utility.jsonl")
DEFAULT_VAL_CACHE = os.path.join(UTILITY_CACHE_DIR, "val_with_utility.jsonl")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "vlm_method2_regression")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache", "vlm_method2_regression")

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 2048
NUM_FRAMES_PER_SAMPLE = 8
SAVE_STEPS = 0
BOUNDARY_LOSS_WEIGHT = 0.5
AUX_ADVANTAGE_THRESHOLD = -0.05
PEAK_MSE_WEIGHT = 1.0
PEAK_MSE_ADVANTAGE_THRESHOLD = -0.05
USE_AUX_POS_WEIGHT = 0

DECISION_THRESHOLD = -0.55

SYSTEM_PROMPT = (
    "You are an online boundary decision agent for gameplay video segmentation. "
    "At each decision step, you observe recent memory, the current partial window, and a short video clip. "
    "Your objective is utility-aware segmentation: maximize within-window semantic consistency "
    "while maximizing between-window separability. "
    "Output <CLOSE> only when the current window is already semantically coherent and a boundary here would separate it "
    "from a different upcoming semantic phase. "
    "Output <CONTINUE> when the current window has not yet matured into a complete semantic unit, "
    "or when future content is likely to remain part of the same ongoing episode. "
    "Do not place boundaries based only on small visual changes; place them only when they improve semantic grouping. "
    "You must respond with exactly one token: <CLOSE> or <CONTINUE>."
)

def compute_advantage(sample: Dict[str, Any]) -> float:
    if "advantage" in sample:
        return float(sample["advantage"])
    return float(sample.get("u_close", 0.0) - sample.get("u_continue", 1.0))

def compute_aux_target(
    sample: Dict[str, Any],
    positive_threshold: float = AUX_ADVANTAGE_THRESHOLD,
) -> float:
    return 1.0 if compute_advantage(sample) >= positive_threshold else 0.0

def compute_regression_weight(
    sample: Dict[str, Any],
    peak_weight: float = PEAK_MSE_WEIGHT,
    peak_threshold: float = PEAK_MSE_ADVANTAGE_THRESHOLD,
) -> float:
    if peak_weight <= 1.0:
        return 1.0
    return float(peak_weight) if compute_advantage(sample) >= peak_threshold else 1.0

def prepare_training_data(jsonl_path: str, output_path: str, verbose: bool = True) -> str:
    if os.path.exists(output_path):
        if verbose:
            print(f"  Already prepared: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if verbose:
        print(f"  Loaded {len(samples)} samples from {jsonl_path}")

    episodes = defaultdict(list)
    for i, s in enumerate(samples):
        key = (s["match_id"], s["current_window_idx"])
        episodes[key].append(s)

    if verbose:
        print(f"  Found {len(episodes)} episodes")

    with open(output_path, 'w', encoding='utf-8') as f:
        for s in samples:
            s["advantage"] = compute_advantage(s)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if verbose:
        advantage_vals = [compute_advantage(s) for s in samples]
        boundaries = sum(1 for s in samples if s.get("is_boundary", False))
        print(f"  advantage stats: mean={np.mean(advantage_vals):.4f}, std={np.std(advantage_vals):.4f}, "
              f"min={np.min(advantage_vals):.4f}, max={np.max(advantage_vals):.4f}")
        print(f"  Boundary samples: {boundaries}/{len(samples)} "
              f"({100*boundaries/max(len(samples),1):.1f}%)")
        print(f"  Saved to: {output_path}")

    return output_path

class AdvantageRegressionDataset(Dataset):
    def __init__(self, jsonl_path, processor, tokenizer,
                 max_seq_length=MAX_SEQ_LENGTH, num_frames=NUM_FRAMES_PER_SAMPLE,
                 aux_advantage_threshold=AUX_ADVANTAGE_THRESHOLD,
                 peak_mse_weight=PEAK_MSE_WEIGHT,
                 peak_mse_advantage_threshold=PEAK_MSE_ADVANTAGE_THRESHOLD):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_frames = num_frames
        self.aux_advantage_threshold = aux_advantage_threshold
        self.peak_mse_weight = peak_mse_weight
        self.peak_mse_advantage_threshold = peak_mse_advantage_threshold

        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")

        advantage_vals = [compute_advantage(s) for s in self.samples]
        boundaries = sum(1 for s in self.samples if s.get("is_boundary", False))
        aux_positives = sum(
            1 for s in self.samples
            if compute_aux_target(s, self.aux_advantage_threshold) > 0.5
        )
        print(f"  advantage stats: mean={np.mean(advantage_vals):.4f}, std={np.std(advantage_vals):.4f}, "
              f"min={np.min(advantage_vals):.4f}, max={np.max(advantage_vals):.4f}")
        print(f"  GT-boundary samples: {boundaries}/{len(self.samples)}")
        print(f"  Aux positives (advantage >= {self.aux_advantage_threshold:.2f}): "
              f"{aux_positives}/{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, frame_paths):
        TARGET_SIZE = (224, 224)
        images = []
        for p in frame_paths[:self.num_frames]:
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize(TARGET_SIZE, Image.Resampling.BICUBIC)
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", TARGET_SIZE, (0, 0, 0)))
        while len(images) < self.num_frames:
            images.append(images[-1].copy() if images else Image.new("RGB", TARGET_SIZE))
        return images

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt_text = sample["prompt"]
        images = self._load_frames(sample.get("frame_paths", []))

        user_content = []
        for img in images:
            user_content.append({"type": "image", "image": img})

        text_prompt = prompt_text
        vs = text_prompt.find("<CURRENT_VIDEO>")
        ve = text_prompt.find("</CURRENT_VIDEO>")
        if vs >= 0 and ve >= 0:
            text_prompt = text_prompt[:vs].rstrip() + "\n\n" + text_prompt[ve + len("</CURRENT_VIDEO>"):].lstrip()
        user_content.append({"type": "text", "text": text_prompt})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=images,
            padding="max_length", max_length=self.max_seq_length,
            truncation=True, return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        advantage = compute_advantage(sample)
        aux_target = compute_aux_target(sample, self.aux_advantage_threshold)
        is_boundary = sample.get("is_boundary", False)
        regression_weight = compute_regression_weight(
            sample,
            peak_weight=self.peak_mse_weight,
            peak_threshold=self.peak_mse_advantage_threshold,
        )

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "advantage": torch.tensor(advantage, dtype=torch.float32),
            "aux_target": torch.tensor(aux_target, dtype=torch.float32),
            "is_boundary": torch.tensor(1.0 if is_boundary else 0.0, dtype=torch.float32),
            "regression_weight": torch.tensor(regression_weight, dtype=torch.float32),
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0) if pixel_values.dim() > 1 else pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw
        return result

class AdvantageRegressionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0)

        batch_size = hidden_states.size(0)
        last_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]

        return self.head(last_hidden).squeeze(-1)

class BoundaryClassificationHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0)
        batch_size = hidden_states.size(0)
        last_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]
        return self.head(last_hidden).squeeze(-1)

def collate_fn(batch):
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "advantage": torch.stack([b["advantage"] for b in batch]),
        "aux_target": torch.stack([b["aux_target"] for b in batch]),
        "is_boundary": torch.stack([b["is_boundary"] for b in batch]),
        "regression_weight": torch.stack([b["regression_weight"] for b in batch]),
    }
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return result

def setup_model_and_head(model_path=VL_MODEL_PATH, device="cuda:0",
                         lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                         lora_dropout=LORA_DROPOUT, lora_target_modules=None):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    if lora_target_modules is None:
        lora_target_modules = LORA_TARGET_MODULES

    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    print(f"Loading model from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}, targets={lora_target_modules})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_modules=lora_target_modules, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    hidden_size = model.config.hidden_size
    print(f"Creating regression + boundary heads (hidden_size={hidden_size})...")
    reg_head = AdvantageRegressionHead(hidden_size).to(device).to(torch.bfloat16)
    boundary_head = BoundaryClassificationHead(hidden_size).to(device).to(torch.bfloat16)

    return model, processor, tokenizer, reg_head, boundary_head

def train(model, reg_head, boundary_head, processor, tokenizer,
          train_jsonl, val_jsonl=None, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
          gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
          learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
          warmup_ratio=WARMUP_RATIO, max_seq_length=MAX_SEQ_LENGTH,
          device="cuda:0", output_dir=OUTPUT_DIR, save_steps=SAVE_STEPS,
          boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
          aux_advantage_threshold=AUX_ADVANTAGE_THRESHOLD,
          peak_mse_weight=PEAK_MSE_WEIGHT,
          peak_mse_advantage_threshold=PEAK_MSE_ADVANTAGE_THRESHOLD,
          use_aux_pos_weight=USE_AUX_POS_WEIGHT,
          tensorboard_log_dir=None,
          verbose=True):

    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating training dataset...")
    train_ds = AdvantageRegressionDataset(
        train_jsonl, processor, tokenizer, max_seq_length,
        aux_advantage_threshold=aux_advantage_threshold,
        peak_mse_weight=peak_mse_weight,
        peak_mse_advantage_threshold=peak_mse_advantage_threshold,
    )
    val_ds = None
    if val_jsonl and os.path.isfile(val_jsonl):
        print("Creating validation dataset...")
        val_ds = AdvantageRegressionDataset(
            val_jsonl, processor, tokenizer, max_seq_length,
            aux_advantage_threshold=aux_advantage_threshold,
            peak_mse_weight=peak_mse_weight,
            peak_mse_advantage_threshold=peak_mse_advantage_threshold,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(reg_head.parameters()) +
        list(boundary_head.parameters()),
        lr=learning_rate, weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    mse_loss_fn = nn.MSELoss(reduction="none")
    if use_aux_pos_weight:
        aux_targets = np.array(
            [compute_aux_target(s, aux_advantage_threshold) for s in train_ds.samples],
            dtype=np.float32,
        )
        pos_count = float(aux_targets.sum())
        neg_count = float(len(aux_targets) - pos_count)
        pos_weight_value = neg_count / max(pos_count, 1.0)
    else:
        pos_weight_value = 1.0
    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )

    print(f"\n{'='*70}\nAdvantage Regression Training")
    print(f"  Samples: {len(train_ds)}, Epochs: {num_epochs}, "
          f"Effective batch: {batch_size * gradient_accumulation_steps}")
    print(f"  Loss: weighted-MSE(advantage_hat, advantage_target) + "
          f"{boundary_loss_weight} * BCE(aux_logit, aux_target)")
    print(f"  Target range: approx [-1, 0] from normalized utility cache")
    print(f"  Aux positive threshold: advantage >= {aux_advantage_threshold}")
    print(f"  Peak-MSE weight: {peak_mse_weight} (active when advantage >= {peak_mse_advantage_threshold})")
    print(f"  Aux BCE pos_weight: {pos_weight_value:.4f}")
    print(f"  Decision rule: CUT if advantage_hat >= {DECISION_THRESHOLD}")
    if save_steps > 0:
        print(f"  Save checkpoint every {save_steps} optimizer steps")
    else:
        print(f"  Save checkpoint every epoch only")
    if tensorboard_log_dir:
        print(f"  TensorBoard log dir: {tensorboard_log_dir}")
    print(f"{'='*70}\n")

    writer = None
    if tensorboard_log_dir:
        if SummaryWriter is None:
            print("  Warning: TensorBoard writer unavailable in this environment; skipping TensorBoard logging.")
        else:
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_log_dir)

    model.train()
    reg_head.train()
    boundary_head.train()
    global_step = 0
    best_val_total_loss = float('inf')
    history = {
        "train_loss": [],
        "train_total_loss": [],
        "train_mse": [],
        "train_boundary_bce": [],
        "train_decision_acc": [],
        "train_boundary_acc": [],
        "val_loss": [],
        "val_total_loss": [],
        "val_mse": [],
        "val_boundary_bce": [],
        "val_decision_acc": [],
        "val_boundary_acc": [],
    }

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_bce = 0.0
        epoch_correct, epoch_total = 0, 0
        epoch_boundary_correct = 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{num_epochs}", ncols=130)
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            advantage_target = batch["advantage"].to(device)
            aux_target = batch["aux_target"].to(device)
            is_boundary = batch["is_boundary"].to(device)
            regression_weight = batch["regression_weight"].to(device)

            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask,
                            "output_hidden_states": True}
            if "pixel_values" in batch:
                model_kwargs["pixel_values"] = batch["pixel_values"].to(device)
            if "image_grid_thw" in batch:
                model_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)

            outputs = model(**model_kwargs)

            hidden_states = outputs.hidden_states[-1]

            advantage_pred = reg_head(hidden_states, attention_mask)
            boundary_logit = boundary_head(hidden_states, attention_mask)

            mse_per_sample = mse_loss_fn(advantage_pred.float(), advantage_target)
            mse_loss = (mse_per_sample * regression_weight).mean()
            boundary_loss = bce_loss_fn(boundary_logit.float(), aux_target)
            total_loss = mse_loss + boundary_loss_weight * boundary_loss
            (total_loss / gradient_accumulation_steps).backward()
            epoch_loss += total_loss.item()
            epoch_mse += mse_loss.item()
            epoch_bce += boundary_loss.item()

            with torch.no_grad():
                pred_cut = (advantage_pred >= DECISION_THRESHOLD).float()
                gt_cut = is_boundary
                correct = (pred_cut == gt_cut).sum().item()
                epoch_correct += correct
                boundary_pred = (torch.sigmoid(boundary_logit) >= 0.5).float()
                epoch_boundary_correct += (boundary_pred == aux_target).sum().item()
                epoch_total += input_ids.size(0)

            avg_loss = epoch_loss / (step + 1)
            avg_mse = epoch_mse / (step + 1)
            avg_bce = epoch_bce / (step + 1)
            acc = epoch_correct / max(epoch_total, 1)
            boundary_acc = epoch_boundary_correct / max(epoch_total, 1)
            pbar.set_postfix(
                total=f"{avg_loss:.4f}",
                mse=f"{avg_mse:.4f}",
                bce=f"{avg_bce:.4f}",
                dec_acc=f"{acc:.4f}",
                aux_acc=f"{boundary_acc:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(reg_head.parameters()) + list(boundary_head.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if writer is not None:
                    writer.add_scalar("train/step_total_loss", total_loss.item(), global_step)
                    writer.add_scalar("train/step_mse", mse_loss.item(), global_step)
                    writer.add_scalar("train/step_boundary_bce", boundary_loss.item(), global_step)
                    writer.add_scalar("train/step_decision_acc", acc, global_step)
                    writer.add_scalar("train/step_aux_acc", boundary_acc, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                if save_steps > 0 and global_step % save_steps == 0:
                    step_ckpt_path = os.path.join(output_dir, f"checkpoint_step{global_step}")
                    save_checkpoint(model, reg_head, boundary_head, tokenizer, step_ckpt_path)
                    step_meta = {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "train_total_loss_so_far": epoch_loss / (step + 1),
                        "train_mse_so_far": epoch_mse / (step + 1),
                        "train_boundary_bce_so_far": epoch_bce / (step + 1),
                    }
                    with open(os.path.join(step_ckpt_path, "step_meta.json"), 'w') as f:
                        json.dump(step_meta, f, indent=2)
                    if verbose:
                        print(f"\n  💾 Saved step checkpoint: {step_ckpt_path} "
                              f"(step={global_step}, total={epoch_loss/(step+1):.4f}, "
                              f"mse={epoch_mse/(step+1):.4f}, bce={epoch_bce/(step+1):.4f})")

        avg_epoch_total_loss = epoch_loss / max(len(train_loader), 1)
        avg_epoch_mse = epoch_mse / max(len(train_loader), 1)
        avg_epoch_bce = epoch_bce / max(len(train_loader), 1)
        epoch_acc = epoch_correct / max(epoch_total, 1)
        epoch_boundary_acc = epoch_boundary_correct / max(epoch_total, 1)
        history["train_loss"].append(avg_epoch_total_loss)
        history["train_total_loss"].append(avg_epoch_total_loss)
        history["train_mse"].append(avg_epoch_mse)
        history["train_boundary_bce"].append(avg_epoch_bce)
        history["train_decision_acc"].append(epoch_acc)
        history["train_boundary_acc"].append(epoch_boundary_acc)
        print(f"\n  Epoch {epoch+1}/{num_epochs}: total={avg_epoch_total_loss:.4f}, "
              f"mse={avg_epoch_mse:.4f}, boundary_bce={avg_epoch_bce:.4f}, "
              f"decision_acc={epoch_acc:.4f}, boundary_acc={epoch_boundary_acc:.4f}")
        if writer is not None:
            writer.add_scalar("epoch/train_total_loss", avg_epoch_total_loss, epoch + 1)
            writer.add_scalar("epoch/train_mse", avg_epoch_mse, epoch + 1)
            writer.add_scalar("epoch/train_boundary_bce", avg_epoch_bce, epoch + 1)
            writer.add_scalar("epoch/train_decision_acc", epoch_acc, epoch + 1)
            writer.add_scalar("epoch/train_aux_acc", epoch_boundary_acc, epoch + 1)

        if val_ds is not None:
            val_total_loss, val_mse, val_bce, val_acc, val_boundary_acc, val_metrics = evaluate_model(
                model,
                reg_head,
                boundary_head,
                val_ds,
                batch_size,
                device,
                boundary_loss_weight,
                pos_weight_value=pos_weight_value,
            )
            history["val_loss"].append(val_total_loss)
            history["val_total_loss"].append(val_total_loss)
            history["val_mse"].append(val_mse)
            history["val_boundary_bce"].append(val_bce)
            history["val_decision_acc"].append(val_acc)
            history["val_boundary_acc"].append(val_boundary_acc)
            print(f"  val_total={val_total_loss:.4f}, val_mse={val_mse:.4f}, "
                  f"val_boundary_bce={val_bce:.4f}, val_decision_acc={val_acc:.4f}, "
                  f"val_boundary_acc={val_boundary_acc:.4f}")
            print(f"  val_close_recall={val_metrics['close_recall']:.4f}, "
                  f"val_close_precision={val_metrics['close_precision']:.4f}, "
                  f"val_f1={val_metrics['f1']:.4f}")
            if writer is not None:
                writer.add_scalar("epoch/val_total_loss", val_total_loss, epoch + 1)
                writer.add_scalar("epoch/val_mse", val_mse, epoch + 1)
                writer.add_scalar("epoch/val_boundary_bce", val_bce, epoch + 1)
                writer.add_scalar("epoch/val_decision_acc", val_acc, epoch + 1)
                writer.add_scalar("epoch/val_aux_acc", val_boundary_acc, epoch + 1)
                writer.add_scalar("epoch/val_close_recall", val_metrics["close_recall"], epoch + 1)
                writer.add_scalar("epoch/val_close_precision", val_metrics["close_precision"], epoch + 1)
                writer.add_scalar("epoch/val_f1", val_metrics["f1"], epoch + 1)

            if val_total_loss < best_val_total_loss:
                best_val_total_loss = val_total_loss
                save_checkpoint(model, reg_head, boundary_head, tokenizer, os.path.join(output_dir, "best_model"))
                print(f"  Saved best model (val_total={val_total_loss:.4f}, "
                      f"val_mse={val_mse:.4f}, val_boundary_bce={val_bce:.4f})")

        save_checkpoint(model, reg_head, boundary_head, tokenizer,
                        os.path.join(output_dir, f"checkpoint_epoch{epoch+1}"))

    final_path = os.path.join(output_dir, "final_model")
    save_checkpoint(model, reg_head, boundary_head, tokenizer, final_path)
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    if writer is not None:
        writer.flush()
        writer.close()
    print(f"\nFinal model saved to {final_path}")
    return history

def save_checkpoint(model, reg_head, boundary_head, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(reg_head.state_dict(), os.path.join(path, "regression_head.pt"))
    torch.save(boundary_head.state_dict(), os.path.join(path, "boundary_head.pt"))

def evaluate_model(model, reg_head, boundary_head, dataset, batch_size=1, device="cuda:0",
                   boundary_loss_weight=BOUNDARY_LOSS_WEIGHT, pos_weight_value=1.0):
    model.eval()
    reg_head.eval()
    boundary_head.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate_fn)

    mse_loss_fn = nn.MSELoss(reduction="none")
    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    total_loss_sum = 0.0
    total_mse = 0.0
    total_bce = 0.0
    all_preds, all_targets, all_boundaries = [], [], []
    all_boundary_probs = []
    all_aux_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=100, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            advantage_target = batch["advantage"].to(device)
            aux_target = batch["aux_target"].to(device)
            is_boundary = batch["is_boundary"].to(device)
            regression_weight = batch["regression_weight"].to(device)

            mkw = {"input_ids": input_ids, "attention_mask": attention_mask,
                    "output_hidden_states": True}
            if "pixel_values" in batch:
                mkw["pixel_values"] = batch["pixel_values"].to(device)
            if "image_grid_thw" in batch:
                mkw["image_grid_thw"] = batch["image_grid_thw"].to(device)

            outputs = model(**mkw)
            hidden_states = outputs.hidden_states[-1]
            advantage_pred = reg_head(hidden_states, attention_mask)
            boundary_logit = boundary_head(hidden_states, attention_mask)

            mse = (mse_loss_fn(advantage_pred.float(), advantage_target) * regression_weight).mean().item()
            bce = bce_loss_fn(boundary_logit.float(), aux_target).item()
            total_loss_sum += mse + boundary_loss_weight * bce
            total_mse += mse
            total_bce += bce

            all_preds.extend(advantage_pred.cpu().tolist())
            all_targets.extend(advantage_target.cpu().tolist())
            all_boundaries.extend(is_boundary.cpu().tolist())
            all_aux_targets.extend(aux_target.cpu().tolist())
            all_boundary_probs.extend(torch.sigmoid(boundary_logit).cpu().tolist())

    model.train()
    reg_head.train()
    boundary_head.train()

    avg_total_loss = total_loss_sum / max(len(loader), 1)
    avg_mse = total_mse / max(len(loader), 1)
    avg_bce = total_bce / max(len(loader), 1)

    preds = np.array(all_preds)
    boundaries = np.array(all_boundaries)
    aux_targets = np.array(all_aux_targets)
    boundary_probs = np.array(all_boundary_probs)

    pred_cut = (preds >= DECISION_THRESHOLD).astype(float)
    gt_cut = boundaries

    total = len(pred_cut)
    correct = (pred_cut == gt_cut).sum()
    acc = correct / max(total, 1)

    tp = ((pred_cut == 1) & (gt_cut == 1)).sum()
    fp = ((pred_cut == 1) & (gt_cut == 0)).sum()
    fn = ((pred_cut == 0) & (gt_cut == 1)).sum()
    tn = ((pred_cut == 0) & (gt_cut == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    boundary_pred = (boundary_probs >= 0.5).astype(float)
    boundary_acc = float((boundary_pred == aux_targets).sum() / max(len(boundary_pred), 1))

    metrics = {
        "close_precision": float(precision),
        "close_recall": float(recall),
        "f1": float(f1),
        "continue_recall": float(tn / max(tn + fp, 1)),
        "boundary_acc": boundary_acc,
        "aux_positive_rate": float(aux_targets.mean()) if len(aux_targets) > 0 else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    return avg_total_loss, avg_mse, avg_bce, float(acc), boundary_acc, metrics

def run_test_evaluation(model, reg_head, boundary_head, processor, tokenizer,
                        test_jsonl, device="cuda:0", output_dir=OUTPUT_DIR):
    print(f"\n{'='*70}\nTest Set Evaluation (Advantage Regression)\n{'='*70}")

    test_ds = AdvantageRegressionDataset(
        test_jsonl, processor, tokenizer,
    )

    model.eval()
    reg_head.eval()
    boundary_head.eval()
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate_fn)

    results = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Testing", ncols=100)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            mkw = {"input_ids": input_ids, "attention_mask": attention_mask,
                    "output_hidden_states": True}
            if "pixel_values" in batch:
                mkw["pixel_values"] = batch["pixel_values"].to(device)
            if "image_grid_thw" in batch:
                mkw["image_grid_thw"] = batch["image_grid_thw"].to(device)

            outputs = model(**mkw)
            hidden_states = outputs.hidden_states[-1]
            advantage_pred = reg_head(hidden_states, attention_mask)
            boundary_logit = boundary_head(hidden_states, attention_mask)

            sample = test_ds.samples[i]
            pred_val = advantage_pred.item()
            pred_cut = pred_val >= DECISION_THRESHOLD
            boundary_prob = torch.sigmoid(boundary_logit).item()
            gt_boundary = sample.get("is_boundary", False)

            results.append({
                "match_id": sample["match_id"],
                "sample_time": sample["sample_time"],
                "window_idx": sample["current_window_idx"],
                "advantage_target": round(compute_advantage(sample), 4),
                "advantage_pred": round(pred_val, 4),
                "boundary_prob": round(boundary_prob, 4),
                "pred_decision": "CLOSE" if pred_cut else "CONTINUE",
                "gt_boundary": gt_boundary,
                "correct": bool(pred_cut == gt_boundary),
            })

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc = correct / max(total, 1)

    tp = sum(1 for r in results if r["pred_decision"] == "CLOSE" and r["gt_boundary"])
    fp = sum(1 for r in results if r["pred_decision"] == "CLOSE" and not r["gt_boundary"])
    fn = sum(1 for r in results if r["pred_decision"] == "CONTINUE" and r["gt_boundary"])
    tn = sum(1 for r in results if r["pred_decision"] == "CONTINUE" and not r["gt_boundary"])

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    mse = np.mean([(r["advantage_pred"] - r["advantage_target"])**2 for r in results])

    print(f"\n  Total: {total}, Decision Accuracy: {acc:.4f}")
    print(f"  Advantage MSE: {mse:.4f}")
    print(f"  CLOSE precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"  CONTINUE recall: {tn/max(tn+fp,1):.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "accuracy": round(acc, 4),
        "mse": round(float(mse), 4),
        "close_precision": round(precision, 4),
        "close_recall": round(recall, 4),
        "f1": round(f1, 4),
        "continue_recall": round(tn / max(tn + fp, 1), 4),
        "threshold": DECISION_THRESHOLD,
        "per_sample": results,
    }
    with open(os.path.join(output_dir, "test_results.json"), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {os.path.join(output_dir, 'test_results.json')}")
    return acc

def run_precompute(train_cache=DEFAULT_TRAIN_CACHE, val_cache=DEFAULT_VAL_CACHE):
    print("="*70 + "\nPhase 1: Preparing Data from Utility Cache\n" + "="*70)
    os.makedirs(CACHE_DIR, exist_ok=True)

    for name, cache_path in [("train", train_cache), ("val", val_cache)]:
        if cache_path and os.path.isfile(cache_path):
            print(f"\n  Processing {name}: {cache_path}")
            output_path = os.path.join(CACHE_DIR, f"{name}_prepared.jsonl")
            prepare_training_data(cache_path, output_path)
        else:
            print(f"\n  Warning: {name} cache not found: {cache_path}")
            print(f"  Please run compute_utility_forward_looking.py first.")

def run_train_pipeline(device="cuda:0", num_epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                       output_dir=OUTPUT_DIR, lora_r=LORA_R, save_steps=SAVE_STEPS,
                       boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
                       aux_advantage_threshold=AUX_ADVANTAGE_THRESHOLD,
                       peak_mse_weight=PEAK_MSE_WEIGHT,
                       peak_mse_advantage_threshold=PEAK_MSE_ADVANTAGE_THRESHOLD,
                       use_aux_pos_weight=USE_AUX_POS_WEIGHT,
                       tensorboard_log_dir=None):
    print("="*70 + "\nPhase 2: Advantage Regression Training\n" + "="*70)

    train_path = os.path.join(CACHE_DIR, "train_prepared.jsonl")
    val_path = os.path.join(CACHE_DIR, "val_prepared.jsonl")

    if not os.path.isfile(train_path):
        print(f"  Error: Training data not found: {train_path}")
        print(f"  Please run --mode precompute first.")
        return None, None, None, None

    model, processor, tokenizer, reg_head, boundary_head = setup_model_and_head(
        device=device, lora_r=lora_r)

    train(model, reg_head, boundary_head, processor, tokenizer, train_path,
          val_path if os.path.isfile(val_path) else None,
          num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
          device=device, output_dir=output_dir, save_steps=save_steps,
          boundary_loss_weight=boundary_loss_weight,
          aux_advantage_threshold=aux_advantage_threshold,
          peak_mse_weight=peak_mse_weight,
          peak_mse_advantage_threshold=peak_mse_advantage_threshold,
          use_aux_pos_weight=use_aux_pos_weight,
          tensorboard_log_dir=tensorboard_log_dir)

    return model, processor, tokenizer, reg_head, boundary_head

def run_eval_pipeline(device="cuda:0", model=None, processor=None,
                      tokenizer=None, reg_head=None, boundary_head=None,
                      output_dir=OUTPUT_DIR, lora_r=LORA_R):
    print("="*70 + "\nPhase 3: Test Evaluation\n" + "="*70)

    test_path = os.path.join(CACHE_DIR, "val_prepared.jsonl")
    if not os.path.isfile(test_path):
        print(f"  Test set not found: {test_path}")
        return

    if model is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        best_path = os.path.join(output_dir, "best_model")
        if not os.path.isdir(best_path):
            best_path = os.path.join(output_dir, "final_model")
        if not os.path.isdir(best_path):
            print(f"  No trained model found in {output_dir}")
            return

        processor = AutoProcessor.from_pretrained(best_path, trust_remote_code=True)
        tokenizer = processor.tokenizer

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VL_MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, best_path).to(device)

        hidden_size = model.config.hidden_size
        reg_head = AdvantageRegressionHead(hidden_size).to(device).to(torch.bfloat16)
        head_path = os.path.join(best_path, "regression_head.pt")
        if os.path.isfile(head_path):
            reg_head.load_state_dict(torch.load(head_path, map_location=device))
            print(f"  Loaded regression head from {head_path}")
        boundary_head = BoundaryClassificationHead(hidden_size).to(device).to(torch.bfloat16)
        boundary_head_path = os.path.join(best_path, "boundary_head.pt")
        if os.path.isfile(boundary_head_path):
            boundary_head.load_state_dict(torch.load(boundary_head_path, map_location=device))
            print(f"  Loaded boundary head from {boundary_head_path}")
        else:
            print(f"  Warning: boundary head not found at {boundary_head_path}; using randomly initialized head.")

    run_test_evaluation(model, reg_head, boundary_head, processor, tokenizer,
                        test_path, device, output_dir)

def run_full(device="cuda:0", num_epochs=NUM_EPOCHS,
             output_dir=OUTPUT_DIR, lora_r=LORA_R, save_steps=SAVE_STEPS,
             boundary_loss_weight=BOUNDARY_LOSS_WEIGHT,
             aux_advantage_threshold=AUX_ADVANTAGE_THRESHOLD,
             peak_mse_weight=PEAK_MSE_WEIGHT,
             peak_mse_advantage_threshold=PEAK_MSE_ADVANTAGE_THRESHOLD,
             use_aux_pos_weight=USE_AUX_POS_WEIGHT,
             tensorboard_log_dir=None):
    run_precompute()
    result = run_train_pipeline(
        device=device, num_epochs=num_epochs, output_dir=output_dir,
        lora_r=lora_r, save_steps=save_steps, boundary_loss_weight=boundary_loss_weight,
        aux_advantage_threshold=aux_advantage_threshold,
        peak_mse_weight=peak_mse_weight,
        peak_mse_advantage_threshold=peak_mse_advantage_threshold,
        use_aux_pos_weight=use_aux_pos_weight,
        tensorboard_log_dir=tensorboard_log_dir)
    if result[0] is not None:
        model, processor, tokenizer, reg_head, boundary_head = result
        run_eval_pipeline(device, model, processor, tokenizer, reg_head, boundary_head,
                          output_dir=output_dir, lora_r=lora_r)

def main():
    global DECISION_THRESHOLD, CACHE_DIR, BOUNDARY_LOSS_WEIGHT
    parser = argparse.ArgumentParser(
        description="VLM Decision Head Method 2: Advantage Regression"
    )
    parser.add_argument("--mode", choices=["full", "precompute", "train", "eval"],
                        default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--cache_dir", default=CACHE_DIR,
                        help="Directory for prepared train/val JSONL cache")
    parser.add_argument("--train_cache", default=DEFAULT_TRAIN_CACHE,
                        help="Path to train_with_utility.jsonl")
    parser.add_argument("--val_cache", default=DEFAULT_VAL_CACHE,
                        help="Path to val_with_utility.jsonl")
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD,
                        help="Decision threshold for CUT (default: -0.3)")
    parser.add_argument("--boundary_loss_weight", type=float, default=BOUNDARY_LOSS_WEIGHT,
                        help="Weight for auxiliary boundary BCE loss")
    parser.add_argument("--aux_advantage_threshold", type=float, default=AUX_ADVANTAGE_THRESHOLD,
                        help="Auxiliary positive label threshold on GT advantage")
    parser.add_argument("--peak_mse_weight", type=float, default=PEAK_MSE_WEIGHT,
                        help="Extra regression weight for samples near the GT utility peak")
    parser.add_argument("--peak_mse_advantage_threshold", type=float, default=PEAK_MSE_ADVANTAGE_THRESHOLD,
                        help="Apply peak_mse_weight when GT advantage >= this threshold")
    parser.add_argument("--use_aux_pos_weight", type=int, default=USE_AUX_POS_WEIGHT,
                        help="Use train-set class balancing for auxiliary BCE (0/1)")
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS,
                        help="Save checkpoint every N optimizer steps (0=epoch only)")
    parser.add_argument("--tensorboard_log_dir", default=None,
                        help="TensorBoard log directory (default: <output_dir>/tensorboard)")
    args = parser.parse_args()

    DECISION_THRESHOLD = args.threshold
    CACHE_DIR = args.cache_dir
    BOUNDARY_LOSS_WEIGHT = args.boundary_loss_weight

    if args.mode == "full":
        tb_log_dir = args.tensorboard_log_dir or os.path.join(args.output_dir, "tensorboard")
        run_full(args.device, args.epochs, output_dir=args.output_dir,
                 lora_r=args.lora_r, save_steps=args.save_steps,
                 boundary_loss_weight=args.boundary_loss_weight,
                 aux_advantage_threshold=args.aux_advantage_threshold,
                 peak_mse_weight=args.peak_mse_weight,
                 peak_mse_advantage_threshold=args.peak_mse_advantage_threshold,
                 use_aux_pos_weight=args.use_aux_pos_weight,
                 tensorboard_log_dir=tb_log_dir)
    elif args.mode == "precompute":
        run_precompute(args.train_cache, args.val_cache)
    elif args.mode == "train":
        tb_log_dir = args.tensorboard_log_dir or os.path.join(args.output_dir, "tensorboard")
        run_train_pipeline(args.device, args.epochs, args.batch_size, args.lr,
                           output_dir=args.output_dir, lora_r=args.lora_r,
                           save_steps=args.save_steps,
                           boundary_loss_weight=args.boundary_loss_weight,
                           aux_advantage_threshold=args.aux_advantage_threshold,
                           peak_mse_weight=args.peak_mse_weight,
                           peak_mse_advantage_threshold=args.peak_mse_advantage_threshold,
                           use_aux_pos_weight=args.use_aux_pos_weight,
                           tensorboard_log_dir=tb_log_dir)
    elif args.mode == "eval":
        run_eval_pipeline(args.device, output_dir=args.output_dir, lora_r=args.lora_r)

if __name__ == "__main__":
    main()
