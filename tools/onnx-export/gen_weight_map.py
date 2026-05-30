"""
Generate weight_names.json sidecar for TRT adapter refit.

Maps ONNX val_N initializer names to human-readable parameter names.
Uses the deterministic linear_N numbering from dynamo decomposition.

Can be run standalone (no model loading needed, just the ONNX file).
"""
import onnx
import json
import sys
import os

def build_weight_map(onnx_path):
    m = onnx.load(onnx_path, load_external_data=False)
    
    # Build val_N -> MatMul node name mapping
    val_to_node = {}
    for node in m.graph.node:
        if node.op_type == 'MatMul':
            for inp in node.input:
                if inp.startswith('val_'):
                    val_to_node[inp] = node.name
    
    # Get all val_ weights sorted by N
    val_inits = sorted(
        [(i.name, list(i.dims)) for i in m.graph.initializer if i.name.startswith('val_')],
        key=lambda x: int(x[0].split('_')[1])
    )
    
    # Filter to only MatMul weights (skip small constants)
    matmul_weights = []
    for vname, vshape in val_inits:
        if vname in val_to_node:
            numel = 1
            for d in vshape:
                numel *= d
            if numel > 1000:
                matmul_weights.append((vname, vshape, val_to_node[vname]))
    
    # The first two MatMul nodes are proj_in and condition_embedder
    # (before the per-layer linears start)
    # node_MatMul_66 = proj_in.1.weight (Conv1d -> PatchEmbedLinear)
    # node_MatMul_68 = condition_embedder.weight
    
    # After that, the per-layer linears follow a repeating pattern.
    # Each layer has 11 linear projections in this order:
    LAYER_PATTERN = [
        # (param_suffix, expected_shapes)
        ("self_attn.q_proj.weight",   None),
        ("self_attn.k_proj.weight",   None),
        ("self_attn.v_proj.weight",   None),
        ("self_attn.o_proj.weight",   None),
        ("cross_attn.q_proj.weight",  None),
        ("cross_attn.k_proj.weight",  None),
        ("cross_attn.v_proj.weight",  None),
        ("cross_attn.o_proj.weight",  None),
        ("mlp.gate_proj.weight",      None),
        ("mlp.up_proj.weight",        None),
        ("mlp.down_proj.weight",      None),
    ]
    
    # After the last layer, there should be a proj_out linear
    
    rename_map = {}  # val_N -> param_name
    
    # Map the first two special cases
    if len(matmul_weights) >= 2:
        # proj_in
        rename_map[matmul_weights[0][0]] = "dit.proj_in.1.linear.weight"
        # condition_embedder  
        rename_map[matmul_weights[1][0]] = "dit.condition_embedder.weight"
    
    # Map per-layer linears
    layer_start = 2  # skip proj_in + condition_embedder
    linears_per_layer = len(LAYER_PATTERN)
    remaining = matmul_weights[layer_start:]
    
    # Detect number of layers from count
    # Last entry might be proj_out
    num_layers = len(remaining) // linears_per_layer
    leftover = len(remaining) % linears_per_layer
    
    print(f"Total MatMul weights: {len(matmul_weights)}")
    print(f"Layer weights: {len(remaining)} ({num_layers} layers * {linears_per_layer} + {leftover} extra)")
    
    for layer_idx in range(num_layers):
        for proj_idx, (suffix, _) in enumerate(LAYER_PATTERN):
            w_idx = layer_start + layer_idx * linears_per_layer + proj_idx
            if w_idx < len(matmul_weights):
                vname = matmul_weights[w_idx][0]
                param_name = f"dit.layers.{layer_idx}.{suffix}"
                rename_map[vname] = param_name
    
    # Map leftover (proj_out)
    if leftover > 0:
        proj_out_idx = layer_start + num_layers * linears_per_layer
        if proj_out_idx < len(matmul_weights):
            rename_map[matmul_weights[proj_out_idx][0]] = "dit.proj_out.1.inner.linear.weight"
    
    # Build both directions
    forward_map = rename_map  # val_N -> param_name
    reverse_map = {v: k for k, v in rename_map.items()}  # param_name -> val_N
    
    return {
        "val_to_param": forward_map,
        "param_to_val": reverse_map,
    }


if __name__ == "__main__":
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else r'D:\Ace-Step-Latest\hot-step-cpp\models\onnx\dit_acestep-v15-merge-sft-turbo-xl-ta-0.7.onnx'
    
    mapping = build_weight_map(onnx_path)
    
    # Print summary
    print(f"\nMapped {len(mapping['val_to_param'])} weights")
    print("\nFirst 15 mappings:")
    for val_name, param_name in sorted(mapping['val_to_param'].items(), key=lambda x: int(x[0].split('_')[1]))[:15]:
        print(f"  {val_name} -> {param_name}")
    
    # Save
    out_path = onnx_path + ".weight_names.json"
    with open(out_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\nSaved to {out_path}")
