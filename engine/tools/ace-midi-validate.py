# ace-midi-validate.py — reference dumps for the ace-midi GGML port (Phase 1/2)
#
# Runs inside the kept MuScriptor oracle venv (CPU torch, fp32 — deterministic):
#   server\data\muscriptor\venv\Scripts\python.exe engine\tools\ace-midi-validate.py \
#       --weights server\data\models\muscriptor\small\model.safetensors \
#       --out server\data\muscriptor\validation
#
# Dumps (little-endian f32 raw + manifest.json):
#   wav.bin        [80000]        deterministic synthetic 5 s chunk (sines + click train)
#   prefix.bin     [T_prefix,dim] conditioning prefix embeddings in final sequence order
#                                 (exactly what LMModel.forward prepends before BOS)
#   logits_bos.bin [card]         forward([[BOS]], conditions, first_step=True) last-step logits
#   tokens_ref.json                greedy token stream for the chunk (max 256, EOS-stripped)
#
# See docs/plans/muscriptor-cpp-port.md §6 (validation plan).

import argparse
import json
import math
from pathlib import Path

import torch

from muscriptor.transcription_model import TranscriptionModel
from muscriptor.modules.streaming import init_states


def synth_wav(n: int = 80_000, sr: int = 16_000) -> torch.Tensor:
    """Deterministic, spectrally busy test signal: three sines + a click train."""
    t = torch.arange(n, dtype=torch.float32) / sr
    wav = (
        0.40 * torch.sin(2 * math.pi * 220.0 * t)
        + 0.25 * torch.sin(2 * math.pi * 554.37 * t)   # C#5
        + 0.15 * torch.sin(2 * math.pi * 1318.5 * t)   # E6
    )
    # 4 Hz click train for onset structure
    clicks = ((torch.arange(n) % (sr // 4)) < 32).float() * 0.5
    return (wav + clicks).clamp(-1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tm = TranscriptionModel.load_model(weights_path=Path(args.weights), device="cpu")
    model = tm._model
    model.eval()

    wav = synth_wav()
    wav.numpy().tofile(out / "wav.bin")

    conds = tm._build_conditions(wav.unsqueeze(0).to(tm._device))
    prepared = model.condition_provider.tokenize(conds)
    cond_tensors = model.condition_provider(prepared)

    # Replicate LMModel.forward's prepend loop to get the prefix in final order
    dim = model.dim
    prefix = torch.zeros(1, 0, dim)
    for cond, _mask in cond_tensors.values():
        prefix = torch.cat([cond, prefix], dim=1)
    prefix[0].detach().float().numpy().tofile(out / "prefix.bin")

    # Single forward: [BOS] with conditions prepended (prefill parity target)
    with torch.no_grad():
        seq = torch.tensor([[model.initial_token_id]], dtype=torch.long)
        state = init_states(model, batch_size=1, sequence_length=prefix.shape[1] + 16)
        logits = model(seq, cond_tensors, first_step=True, model_state=state)
    logits[0, -1].detach().float().numpy().tofile(out / "logits_bos.bin")

    # Greedy token stream for the chunk (Phase 2 target)
    tokens: list[int] = []
    eos = tm._tokenizer.eos_id
    with torch.no_grad():
        for step in model.generate(
            conditions=conds,
            max_gen_len=args.max_tokens,
            use_sampling=False,
            early_stop_on_token=eos,
        ):
            tok = int(step[0].item())
            if tok == eos:
                break
            tokens.append(tok)
    (out / "tokens_ref.json").write_text(json.dumps(tokens))

    top = torch.topk(logits[0, -1], 5)
    manifest = {
        "dim": dim,
        "card": int(model.card),
        "bos_id": int(model.initial_token_id),
        "eos_id": int(eos),
        "prefix_len": int(prefix.shape[1]),
        "wav_samples": int(wav.shape[0]),
        "greedy_tokens": len(tokens),
        "logits_top5_ids": top.indices.tolist(),
        "logits_top5_vals": [round(v, 6) for v in top.values.tolist()],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
