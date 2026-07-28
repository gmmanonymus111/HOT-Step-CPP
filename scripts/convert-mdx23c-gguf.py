#!/usr/bin/env python
"""convert-mdx23c-gguf.py — MDX23C (TFC-TDF v3) .ckpt → GGUF for HOT-Step.

A pure state-dict repack, like convert-bs-roformer-gguf.py: no tracing, peak
memory ~= the checkpoint (~440 MB for DrumSep).

Consumed by engine/src/mdx23c-ggml.h.

ARCHITECTURE (models_without_stft/mdx23c_tfc_tdf_v3_no_stft.py)
---------------------------------------------------------------
A 5-scale conv U-Net over the complex spectrogram, with subband folding:

    x = cac2cws(x)                    fold num_subbands into channels
    mix = x
    first_conv_out = x = first_conv(x)        Conv2d(dim_c -> c, 1x1)
    x = x.transpose(-1, -2)                   [b, c, f, t] -> [b, c, t, f]
    for s in scales:  x = tfc_tdf(x); skip.append(x); x = downscale(x)
    x = bottleneck(x)
    for s in scales:  x = upscale(x); x = cat(x, skip.pop()); x = tfc_tdf(x)
    x = x.transpose(-1, -2)
    x = x * first_conv_out                    artifact reduction
    x = final_conv(cat(mix, x))
    x = cws2cac(x)                    unfold subbands

TFC_TDF sub-block (residual):
    s = shortcut(x)                           Conv2d 1x1
    x = tfc1(x)                               norm -> act -> Conv2d 3x3
    x = x + tdf(x)                            norm->act->Linear->norm->act->Linear
    x = tfc2(x)                               norm -> act -> Conv2d 3x3
    x = x + s

`norm` is InstanceNorm2d(affine=True) — normalise each (sample, channel) over
its spatial extent, i.e. GroupNorm with n_groups == n_channels, then an affine
weight/bias per channel. `act` is GELU.

TENSOR NAMING / LAYOUT
----------------------
gguf reverses numpy dims on write, so a torch Conv2d weight (OC, IC, KH, KW)
lands as ne = [KW, KH, IC, OC] — exactly ggml_conv_2d's kernel layout. A torch
ConvTranspose2d weight is (IC, OC, KH, KW) and lands as [KW, KH, OC, IC],
which is ggml_conv_transpose_2d_p0's layout. Both pass through unchanged.

    first_conv                     [1,1,dim_c,c]
    enc.{s}.blk.{b}.*              per TFC_TDF sub-block (see below)
    enc.{s}.down_norm_w / _b       [c]
    enc.{s}.down_conv              [sw,sh,c,c+g]
    bot.blk.{b}.*
    dec.{s}.up_norm_w / _b         [c]
    dec.{s}.up_conv                [sw,sh,c-g,c]   (transposed)
    dec.{s}.blk.{b}.*
    final1                         [1,1,c+dim_c,c]
    final2                         [1,1,c,n_inst*dim_c]

  per TFC_TDF sub-block:
    tfc1_norm_w/_b, tfc1_conv      [3,3,in_c,c]
    tdf_n1_w/_b, tdf_l1            [f, f//bn]
    tdf_n2_w/_b, tdf_l2            [f//bn, f]
    tfc2_norm_w/_b, tfc2_conv      [3,3,c,c]
    shortcut                       [1,1,in_c,c]

USAGE
-----
    $py = "d:\\Ace-Step-Latest\\hot-step-9000\\.venv\\Scripts\\python.exe"
    & $py scripts\\convert-mdx23c-gguf.py `
        --config <SuperSep>\\models\\config_drumsep_mdx23c.yaml `
        --ckpt   <SuperSep>\\models\\MDX23C-DrumSep-aufr33-jarredou.ckpt `
        --output models\\supersep\\mdx23c_drumsep-F32.gguf
"""

import argparse
import os

import torch
import yaml

import gguf

ARCH = "mdx23c"


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def load_state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if sd and all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    return sd


class Repacker:
    def __init__(self, sd, writer):
        self.sd, self.writer = sd, writer
        self.used, self.written = set(), 0

    def put(self, dst, src, expect_dims=None):
        if src not in self.sd:
            raise SystemExit(f"missing tensor in checkpoint: {src}")
        t = self.sd[src]
        if expect_dims is not None and t.dim() != expect_dims:
            raise SystemExit(f"{src}: expected {expect_dims}-D, got {tuple(t.shape)}")
        self.writer.add_tensor(dst, t.detach().to(torch.float32).contiguous().numpy())
        self.used.add(src)
        self.written += 1

    def report(self):
        leftover = sorted(set(self.sd) - self.used)
        if leftover:
            raise SystemExit(
                f"{len(leftover)} checkpoint tensors were not written, e.g. "
                f"{leftover[:6]}\nRefusing to emit a partial model.")
        print(f"[ok] wrote {self.written} tensors, none left over")


def put_tfc_tdf(r, dst, src, n_blocks):
    """One TFC_TDF stack: n_blocks residual sub-blocks.

    Sequential indices in the checkpoint:
      tfc1 / tfc2 : 0=norm, 1=act, 2=Conv2d
      tdf         : 0=norm, 1=act, 2=Linear, 3=norm, 4=act, 5=Linear
    """
    for b in range(n_blocks):
        s = f"{src}.blocks.{b}"
        d = f"{dst}.blk.{b}"
        r.put(f"{d}.tfc1_norm_w", f"{s}.tfc1.0.weight", 1)
        r.put(f"{d}.tfc1_norm_b", f"{s}.tfc1.0.bias", 1)
        r.put(f"{d}.tfc1_conv",   f"{s}.tfc1.2.weight", 4)

        r.put(f"{d}.tdf_n1_w",    f"{s}.tdf.0.weight", 1)
        r.put(f"{d}.tdf_n1_b",    f"{s}.tdf.0.bias", 1)
        r.put(f"{d}.tdf_l1",      f"{s}.tdf.2.weight", 2)
        r.put(f"{d}.tdf_n2_w",    f"{s}.tdf.3.weight", 1)
        r.put(f"{d}.tdf_n2_b",    f"{s}.tdf.3.bias", 1)
        r.put(f"{d}.tdf_l2",      f"{s}.tdf.5.weight", 2)

        r.put(f"{d}.tfc2_norm_w", f"{s}.tfc2.0.weight", 1)
        r.put(f"{d}.tfc2_norm_b", f"{s}.tfc2.0.bias", 1)
        r.put(f"{d}.tfc2_conv",   f"{s}.tfc2.2.weight", 4)

        r.put(f"{d}.shortcut",    f"{s}.shortcut.weight", 4)


def convert(cfg, sd, out_path):
    mc, ac = cfg["model"], cfg["audio"]
    n_scales   = mc["num_scales"]
    n_blocks   = mc["num_blocks_per_scale"]
    c0         = mc["num_channels"]
    growth     = mc["growth"]
    bn         = mc["bottleneck_factor"]
    n_subbands = mc["num_subbands"]
    scale      = list(mc["scale"])
    norm_type  = mc.get("norm", "InstanceNorm")
    act_type   = mc.get("act", "gelu")

    n_audio_ch = ac["num_channels"]
    dim_c      = n_subbands * n_audio_ch * 2
    dim_f      = ac["dim_f"]
    n_fft      = ac["n_fft"]
    hop        = ac["hop_length"]
    chunk      = ac["chunk_size"]

    instruments = cfg.get("training", {}).get("instruments", [])
    target = cfg.get("training", {}).get("target_instrument", None)
    n_inst = 1 if target else len(instruments)

    if norm_type != "InstanceNorm":
        raise SystemExit(f"only InstanceNorm is supported, got {norm_type}")
    if act_type != "gelu":
        raise SystemExit(f"only gelu is supported, got {act_type}")

    print(f"[info] scales={n_scales} blocks/scale={n_blocks} c={c0} growth={growth}")
    print(f"[info] subbands={n_subbands} dim_c={dim_c} dim_f={dim_f} bn={bn}")
    print(f"[info] instruments={n_inst} {instruments}")
    print(f"[info] n_fft={n_fft} hop={hop} chunk={chunk} scale={scale}")

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_uint32("mdx23c.num_scales", n_scales)
    w.add_uint32("mdx23c.blocks_per_scale", n_blocks)
    w.add_uint32("mdx23c.num_channels", c0)
    w.add_uint32("mdx23c.growth", growth)
    w.add_uint32("mdx23c.bottleneck_factor", bn)
    w.add_uint32("mdx23c.num_subbands", n_subbands)
    w.add_uint32("mdx23c.dim_c", dim_c)
    w.add_uint32("mdx23c.dim_f", dim_f)
    w.add_uint32("mdx23c.n_fft", n_fft)
    w.add_uint32("mdx23c.hop_length", hop)
    w.add_uint32("mdx23c.chunk_size", chunk)
    w.add_uint32("mdx23c.n_instruments", n_inst)
    w.add_uint32("mdx23c.n_audio_channels", n_audio_ch)
    w.add_array("mdx23c.scale", scale)
    w.add_string("mdx23c.instruments", ",".join(instruments))

    r = Repacker(sd, w)
    r.put("first_conv", "first_conv.weight", 4)

    c = c0
    for s in range(n_scales):
        put_tfc_tdf(r, f"enc.{s}", f"encoder_blocks.{s}.tfc_tdf", n_blocks)
        r.put(f"enc.{s}.down_norm_w", f"encoder_blocks.{s}.downscale.conv.0.weight", 1)
        r.put(f"enc.{s}.down_norm_b", f"encoder_blocks.{s}.downscale.conv.0.bias", 1)
        r.put(f"enc.{s}.down_conv",   f"encoder_blocks.{s}.downscale.conv.2.weight", 4)
        c += growth

    put_tfc_tdf(r, "bot", "bottleneck_block", n_blocks)

    for s in range(n_scales):
        r.put(f"dec.{s}.up_norm_w", f"decoder_blocks.{s}.upscale.conv.0.weight", 1)
        r.put(f"dec.{s}.up_norm_b", f"decoder_blocks.{s}.upscale.conv.0.bias", 1)
        r.put(f"dec.{s}.up_conv",   f"decoder_blocks.{s}.upscale.conv.2.weight", 4)
        put_tfc_tdf(r, f"dec.{s}", f"decoder_blocks.{s}.tfc_tdf", n_blocks)
        c -= growth

    r.put("final1", "final_conv.0.weight", 4)
    r.put("final2", "final_conv.2.weight", 4)

    r.report()

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[ok] {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = read_config(args.config)
    sd = load_state_dict(args.ckpt)
    print(f"[info] checkpoint: {len(sd)} tensors, "
          f"{sum(v.numel() for v in sd.values()):,} params")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    convert(cfg, sd, args.output)


if __name__ == "__main__":
    main()
