"""
STAGE 4: Causal Indirect Effect (CIE) per attention head.

Replicates ARENA exercise: for each (layer, head), measure the effect of
patching that head's z output from a clean ICL prompt into a corrupted
(shuffled-output) ICL prompt.

CIE(layer, head) = mean log-prob recovery on correct token after patching.

Output: results/04_cie_heatmaps.json (per atom: layer x head matrix)
"""
import json
import numpy as np
import torch
from utils_pipeline import (load_model, get_layer, unwrap_saved, make_icl_prompt,
                            DEVICE, DTYPE, N_ICL, RESULTS_DIR)
from tasks import TASKS, get_extraction_pool


N_PROMPTS_CIE = 20    # number of clean/corrupted pairs per CIE estimate
LAYER_STRIDE = 2      # sweep every 2 layers


def make_corrupted_pairs(rng, extraction):
    """Shuffle outputs across pairs to create corrupted ICL."""
    inputs = [x for x, _ in extraction]
    outputs = [y for _, y in extraction]
    shuffled = rng.permutation(outputs).tolist()
    return list(zip(inputs, shuffled))


@torch.no_grad()
def compute_cie_for_atom(model, info, task_name, pairs, seed=0):
    """For each (layer, head), compute mean CIE."""
    rng = np.random.default_rng(seed)
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]
    d_head = info["d_head"]
    tok = model.tokenizer

    extraction = get_extraction_pool(pairs)

    # Build clean and corrupted prompts (matched pairs)
    clean_prompts, corr_prompts, gold_token_ids = [], [], []
    for _ in range(N_PROMPTS_CIE):
        idx = rng.permutation(len(extraction))
        icl_clean = [extraction[i] for i in idx[:N_ICL]]
        query, gold = extraction[idx[N_ICL]]
        # Corrupted = same ICL examples but shuffled outputs
        icl_corr = make_corrupted_pairs(rng, icl_clean)
        clean_prompts.append(make_icl_prompt(icl_clean, query))
        corr_prompts.append(make_icl_prompt(icl_corr, query))
        # Gold token = first token of " <gold>"
        gold_ids = tok(" " + gold, add_special_tokens=False)["input_ids"]
        gold_token_ids.append(gold_ids[0])

    cie_matrix = np.zeros((n_layers, n_heads))

    layers_to_test = list(range(0, n_layers, LAYER_STRIDE))

    for layer_idx in layers_to_test:
        # Step 1: cache z (input to o_proj) from clean run, last token
        clean_zs = []
        for prompt in clean_prompts:
            with model.trace(prompt):
                # In Gemma-2, the input to o_proj is the head-mixed z.
                # We get z by accessing the input to the o_proj module.
                z = get_layer(model, layer_idx).self_attn.o_proj.input.save()
            z_t = unwrap_saved(z)
            # z shape: [batch, seq, d_model] -- last token, then split into heads
            clean_zs.append(z_t[:, -1, :].float().cpu())

        # Step 2: corrupted baseline (no patch)
        corr_logprobs = []
        for prompt, gold_id in zip(corr_prompts, gold_token_ids):
            with model.trace(prompt):
                logits = model.lm_head.output[:, -1, :].save()
            lp = torch.log_softmax(unwrap_saved(logits).float(), dim=-1)
            corr_logprobs.append(lp[0, gold_id].item())
        corr_baseline = np.mean(corr_logprobs)

        # Step 3: patch each head individually
        for head_idx in range(n_heads):
            patched_logprobs = []
            for prompt, clean_z, gold_id in zip(corr_prompts, clean_zs, gold_token_ids):
                with model.trace(prompt):
                    z_target = get_layer(model, layer_idx).self_attn.o_proj.input
                    # Reshape last-token z to [n_heads, d_head]
                    last_tok_z = z_target[:, -1, :]
                    # Replace just this head's slice
                    head_start = head_idx * d_head
                    head_end = (head_idx + 1) * d_head
                    last_tok_z[:, head_start:head_end] = clean_z[:, head_start:head_end].to(DEVICE).to(DTYPE)
                    logits = model.lm_head.output[:, -1, :].save()
                lp = torch.log_softmax(unwrap_saved(logits).float(), dim=-1)
                patched_logprobs.append(lp[0, gold_id].item())
            cie = np.mean(patched_logprobs) - corr_baseline
            cie_matrix[layer_idx, head_idx] = cie

        print(f"  layer {layer_idx:2d}: max CIE = {cie_matrix[layer_idx].max():+.3f}, "
              f"argmax head = {cie_matrix[layer_idx].argmax()}")

    return cie_matrix


def main():
    model, info = load_model()
    out = {}

    for task_name, pairs in TASKS.items():
        print(f"\n=== {task_name} ===")
        cie_matrix = compute_cie_for_atom(model, info, task_name, pairs)
        # Save matrix
        out[task_name] = {
            "cie_matrix": cie_matrix.tolist(),
            "n_layers": info["n_layers"],
            "n_heads": info["n_heads"],
        }
        # Top-K heads
        flat = cie_matrix.flatten()
        top_k_idx = np.argsort(flat)[-20:][::-1]
        top_k = [(int(i // info["n_heads"]), int(i % info["n_heads"]),
                  float(flat[i])) for i in top_k_idx]
        out[task_name]["top_20_heads"] = top_k
        print(f"  Top 5 heads: {top_k[:5]}")

    out_path = RESULTS_DIR / "04_cie_heatmaps.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
