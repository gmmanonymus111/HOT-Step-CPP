"""Relative delta-norm analysis of a ComfyUI-format MM3 LoRA vs base weights.

Usage: python mm3-lora-relnorm.py <adapter.safetensors> [base_transformer_dir]
Computes ||(alpha/r) * B @ A||_F / ||W_base||_F per module, reports per-class
stats and one overall number. CPU-only, needs torch + safetensors.

ComfyUI key stems (SimpleTuner export) -> diffusers base weights:
  ...transformer.project_in                  -> proj_in.weight
  ...transformer.project_out                 -> proj_out.weight
  ...transformer.layers.N.self_attn.to_qkv   -> cat(attn.to_q/to_k/to_v .weight)
  ...transformer.layers.N.self_attn.to_out   -> attn.to_out.0.weight
  ...transformer.layers.N.ff.ff.0.proj       -> ff_in.weight
  ...transformer.layers.N.ff.ff.2            -> ff_out.weight
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

adapter_path = Path(sys.argv[1])
base_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/mnt/d/Ace-Step-Latest/mm3-weights/official/transformer")

# ---- load adapter ----
ad = {}
with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
    for k in f.keys():
        ad[k] = f.get_tensor(k)

ad = {k.replace("._orig_mod", ""): v for k, v in ad.items()}
stems = sorted({re.sub(r"\.(lora_A\.weight|lora_B\.weight|alpha)$", "", k) for k in ad})
print(f"adapter: {adapter_path.name}: {len(ad)} tensors, {len(stems)} modules")

# ---- map stems to base keys ----
def base_keys_for(stem: str):
    m = re.search(r"transformer\.layers\.(\d+)\.(.+)$", stem)
    if m:
        n, rest = m.group(1), m.group(2)
        blk = f"transformer_blocks.{n}"
        table = {
            "self_attn.to_qkv": [f"{blk}.attn.to_q.weight", f"{blk}.attn.to_k.weight", f"{blk}.attn.to_v.weight"],
            "self_attn.to_out": [f"{blk}.attn.to_out.0.weight"],
            "ff.ff.0.proj": [f"{blk}.ff_in.weight"],
            "ff.ff.2": [f"{blk}.ff_out.weight"],
        }
        return table.get(rest)
    if stem.endswith("transformer.project_in"):
        return ["proj_in.weight"]
    if stem.endswith("transformer.project_out"):
        return ["proj_out.weight"]
    return None

def class_of(stem: str) -> str:
    for tag in ("to_qkv", "to_out", "ff.ff.0.proj", "ff.ff.2", "project_in", "project_out"):
        if tag in stem:
            return {"ff.ff.0.proj": "ff_in", "ff.ff.2": "ff_out"}.get(tag, tag)
    return "other"

# ---- load base shards lazily ----
index = json.loads((base_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
weight_map = index["weight_map"]
shard_handles = {}
def base_tensor(key: str) -> torch.Tensor:
    shard = weight_map[key]
    if shard not in shard_handles:
        shard_handles[shard] = safe_open(str(base_dir / shard), framework="pt", device="cpu")
    return shard_handles[shard].get_tensor(key)

# ---- compute ----
rows = []
missing = []
for stem in stems:
    bk = base_keys_for(stem)
    if bk is None or any(k not in weight_map for k in bk):
        missing.append(stem)
        continue
    A = ad[f"{stem}.lora_A.weight"].float()
    B = ad[f"{stem}.lora_B.weight"].float()
    r = A.shape[0]
    alpha = float(ad.get(f"{stem}.alpha", torch.tensor(float(r))))
    delta = (alpha / r) * (B @ A)
    W = torch.cat([base_tensor(k).float() for k in bk], dim=0)
    if W.shape != delta.shape:
        missing.append(f"{stem} SHAPE {tuple(delta.shape)} vs {tuple(W.shape)}")
        continue
    rel = (delta.norm() / W.norm()).item()
    rows.append((stem, class_of(stem), rel))

if missing:
    print(f"UNMAPPED ({len(missing)}):")
    for m in missing[:10]:
        print("  ", m)

by_class = defaultdict(list)
for _, c, rel in rows:
    by_class[c].append(rel)

print(f"\n{'class':<14}{'n':>5}{'mean%':>10}{'median%':>10}{'max%':>10}")
for c in sorted(by_class):
    v = sorted(by_class[c])
    print(f"{c:<14}{len(v):>5}{100*sum(v)/len(v):>10.3f}{100*v[len(v)//2]:>10.3f}{100*v[-1]:>10.3f}")

allv = [rel for _, _, rel in rows]
print(f"\nOVERALL mean relative delta: {100*sum(allv)/len(allv):.3f}%  (n={len(allv)} modules)")
worst = sorted(rows, key=lambda t: -t[2])[:8]
print("\nworst modules:")
for stem, c, rel in worst:
    print(f"  {100*rel:7.3f}%  {stem}")
