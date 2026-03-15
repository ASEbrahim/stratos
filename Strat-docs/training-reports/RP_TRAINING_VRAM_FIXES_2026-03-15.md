# RP Training VRAM Fixes — Post-Mortem Report

**Date:** 2026-03-15
**Hardware:** AMD Radeon RX 7900 XTX (24GB VRAM), 30GB system RAM
**Model:** Qwen3.5-9B abliterated (18GB bf16) with DoRA (4.0M trainable params)
**Software:** transformers 5.3.0, trl, peft, PyTorch ROCm

---

## Timeline of Failures

### Crash 1 — OOM during training (pre-session, ~06:51)
- **What happened:** `train_rp.py` with `device_map="auto"` loaded the full 18GB model onto GPU, leaving no room for optimizer states + activations. System RAM also exhausted (8.7GB RSS).
- **Symptoms:** Linux OOM killer cascade — killed Brave, Chrome, then python3 twice. PC became unresponsive, corrupted 7 git objects.
- **Fix attempted:** Added `max_memory={0: "18GiB", "cpu": "20GiB"}` to cap GPU usage.
- **Result:** Git repo recovered from reflog. No data lost.

### Crash 2 — OOM at step 139/684 (first session run)
- **What happened:** VRAM started at 93% (22.3GB/24GB) with 18GiB cap. Variable-length sequences caused PyTorch's ROCm allocator to cache increasingly large blocks. A longer sequence pushed past 24GB.
- **Symptoms:** `amdgpu_cs_ioctl: Not enough memory for command submission` at 09:10.
- **Fix attempted:** Lowered GPU cap to 15GiB for more headroom.
- **Result:** Would have worked but was reverted for speed.

### Crash 3 — OOM at step 545/684 (second session run)
- **What happened:** Same VRAM creep issue with 18GiB cap. Made it further (80%) but eventually a long sequence spiked over.
- **Symptoms:** Same amdgpu errors at 14:42.
- **Fix attempted:** Add checkpoints every 200 steps to avoid losing progress.
- **Result:** Discovered the REAL problem — checkpoints themselves OOM.

### Crash 4 — OOM at step 200 checkpoint save
- **What happened:** HuggingFace Trainer's built-in checkpoint save triggers `caching_allocator_warmup()` which pre-allocates a 16.68GiB contiguous block as a "speed optimization." On a 24GB card already holding 18GB of model, this instantly OOMs.
- **Symptoms:** `torch.OutOfMemoryError: HIP out of memory. Tried to allocate 16.68 GiB.`
- **Fix attempted:** `save_only_model=True` — didn't help, warmup still triggers.
- **Result:** Same crash.

### Crash 5 — OOM at step 200 checkpoint save (save_only_model)
- **What happened:** `save_only_model=True` doesn't prevent `caching_allocator_warmup` from running during the save pipeline.
- **Fix attempted:** Custom `PEFTCheckpointCallback` that calls `model.save_pretrained()` directly with `save_strategy="no"` to bypass Trainer's save.
- **Result:** Same crash — because the warmup runs on MODEL LOAD too, not just save.

### Crash 6 — OOM on model load (zombie process)
- **What happened:** Previous crashed python3 process (PID 109127) was still holding 22GB VRAM as a zombie. New process couldn't load the model.
- **Fix:** `kill -9 109127` freed the VRAM.

---

## Root Cause

**`caching_allocator_warmup()`** in `transformers/modeling_utils.py:4746` (added in transformers 5.3.0).

This function pre-allocates a single contiguous block equal to the full model size (~16.68GiB for Qwen3.5-9B) as a malloc warmup to speed up subsequent weight loading. It's designed for multi-GPU H100 setups with abundant VRAM.

On a single 24GB card loading an 18GB model:
- Model weights: ~16.7 GiB
- Warmup allocation: ~16.7 GiB
- Total needed: ~33.4 GiB → instant OOM on 24GB card

The warmup runs during `from_pretrained()` AND during Trainer checkpoint saves (which internally call `from_pretrained()` for state verification).

### Why it wasn't caught earlier
- The scorer V2.2 training used an older transformers version (pre-5.3.0) that didn't have this warmup.
- The first few runs "worked" for training steps because the warmup OOM was sometimes caught internally and model loading fell back to the slow path. But checkpoint saves always crashed.

### Secondary issue: VRAM creep
Even without the warmup, VRAM crept from ~18GB to ~22GB over 500+ steps because PyTorch's ROCm allocator caches freed memory blocks by size. Variable-length RP conversations (3 turns to 1024 tokens) created diverse block sizes that were cached but never released.

`expandable_segments:True` is the standard fix for this, but it's **not supported on this ROCm version** (`UserWarning: expandable_segments not supported on this platform`).

---

## Final Fix

```python
# Monkey-patch out the warmup before any model loading
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **kw: None
```

Combined with:
1. `save_strategy="no"` — disable Trainer's built-in save entirely
2. Custom `PEFTCheckpointCallback` — saves only adapter weights (~16MB) via `model.save_pretrained()`
3. `torch.cuda.empty_cache()` before saves
4. `VRAMMonitorCallback` — logs VRAM every 50 steps
5. GPU cap at 20GiB — safe with warmup disabled
6. `--resume` flag for checkpoint recovery

### Verified
- Model loads at 16.7 GiB (not 22+ GiB)
- Checkpoint save: zero extra VRAM (delta: 0 MiB)
- 5-step dry run: clean start-to-finish, adapter saved successfully

---

## Lessons Learned

1. **Always test checkpoint saves before long runs.** A 2-step dry run doesn't hit step 200.
2. **Kill zombie GPU processes.** `rocm-smi --showpids` reveals VRAM-hogging zombies.
3. **Newer transformers versions add VRAM-hungry "optimizations."** The warmup is great for 8×H100 but lethal for consumer GPUs.
4. **Custom save callbacks > Trainer's built-in save** when VRAM is tight with PEFT models.
5. **VRAM monitoring should be built-in** — logging every 50 steps would have caught the creep hours earlier.

---

## Final Configuration

| Parameter | Value |
|-----------|-------|
| GPU cap | 20 GiB (of 24 GiB) |
| Model VRAM | 16.7 GiB |
| Available for activations | ~7.3 GiB |
| Checkpoint save cost | 0 MiB extra VRAM |
| Checkpoint interval | Every 200 steps |
| Checkpoints kept | Last 2 |
| VRAM monitoring | Every 50 steps |
| caching_allocator_warmup | Disabled (monkey-patched) |
| expandable_segments | Not used (unsupported on ROCm) |
