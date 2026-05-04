"""
STAGE 5: Build CIE-based FV from top-K heads, validate zero-shot.

Per atom: take top-K heads from CIE matrix, sum their attention output
contributions over clean ICL prompts -> CIE-based FV.

Then patch into zero-shot prompts at the head's layer (or top head's layer).

Compares: residual-stream FV vs CIE-based FV vs random.
Output: results/05_cie_fv_validation.json
"""
import json
import numpy as np
import torch
from utils_pipeline import (load_model, get_layer, unwrap_saved, make_icl_prompt,
                            make_zero_shot, generate_with_patch, generate_no_patch,
                            extract_fv_residual, make_atom_classifier, summarize,
                            DEVICE, DTYPE, N_ICL, N_FV_PROMPTS, SEEDS, RESULTS_DIR)
from tasks import TASKS, get_extraction_pool, get_eval_pool


TOP_K = 10
N_FV_PROMPTS_CIE = 50


@torch.no_grad()
def build_cie_fv(model, info, pairs, top_k_heads, seed=0):
    """Sum attention outputs of top-K heads, averaged over clean ICL prompts.
    
    Method (ARENA-style without weight access): for each prompt, run forward with
    all heads ablated EXCEPT the target head, save o_proj output, then sum.
    But that's K forward passes per prompt -> expensive.
    
    Alternative: extract z directly per head, project via approximation.
    Simpler: use the residual-stream contribution at the layer of each head,
    isolated by zero-ing other heads' z slices.
    """
    rng = np.random.default_rng(seed)
    extraction = get_extraction_pool(pairs)
    d_head = info["d_head"]
    n_heads = info["n_heads"]

    # Group heads by layer
    by_layer = {}
    for layer, head, score in top_k_heads:
        by_layer.setdefault(layer, []).append(head)

    # Accumulate per-head attention output (in d_model space)
    head_outputs = {(l, h): [] for l, h, _ in top_k_heads}

    for _ in range(N_FV_PROMPTS_CIE):
        idx = rng.permutation(len(extraction))[:N_ICL+1]
        icl = [extraction[i] for i in idx[:N_ICL]]
        query = extraction[idx[N_ICL]][0]
        prompt = make_icl_prompt(icl, query)

        for layer, heads_in_layer in by_layer.items():
            # For each head in this layer, ablate all OTHER heads in z, get o_proj output
            for head in heads_in_layer:
                with model.trace(prompt):
                    z = get_layer(model, layer).self_attn.o_proj.input
                    # z is [batch, seq, d_model] = [batch, seq, n_heads*d_head]
                    # Zero out all heads except `head`
                    z_clone = z.clone()
                    for h in range(n_heads):
                        if h != head:
                            z_clone[:, :, h*d_head:(h+1)*d_head] = 0
                    # Forward through o_proj manually: trace its output after substitution
                    # Trick: replace input, capture output
                    z[:, :, :] = z_clone
                    head_attn_out = get_layer(model, layer).self_attn.o_proj.output[:, -1, :].save()
                head_outputs[(layer, head)].append(unwrap_saved(head_attn_out).float().cpu().squeeze())

    # Average per head, sum across heads
    head_means = {k: torch.stack(v).mean(dim=0) for k, v in head_outputs.items()}
    fv = sum(head_means.values())
    return fv, head_means


@torch.no_grad()
def evaluate_fv(model, fv, atom_pairs, layer_idx):
    """Patch fv at zero-shot last token of layer_idx, score."""
    eval_pool = get_eval_pool(atom_pairs)
    eval_qs = [q for q, _ in eval_pool]
    prompts = [make_zero_shot(q) for q in eval_qs]
    completions = generate_with_patch(model, prompts, fv, layer_idx)
    clf = make_atom_classifier(eval_pool)
    counter = summarize(completions, eval_qs, clf)
    n = sum(counter.values())
    return {
        "acc": counter.get("CORRECT_ATOM", 0) / n if n > 0 else 0.0,
        "dist": dict(counter),
        "samples": list(zip(eval_qs[:5], completions[:5])),
    }


def main():
    model, info = load_model()
    # Load CIE results
    with open(RESULTS_DIR / "04_cie_heatmaps.json") as f:
        cie_data = json.load(f)
    # Load layer sweep best layer
    with open(RESULTS_DIR / "03_layer_sweep.json") as f:
        sweep = json.load(f)

    out = {}
    for task_name, pairs in TASKS.items():
        print(f"\n=== {task_name} ===")
        top_heads = cie_data[task_name]["top_20_heads"][:TOP_K]
        # Patch layer = layer of top head (most causal)
        patch_layer = top_heads[0][0]
        print(f"  Top-{TOP_K} heads: {top_heads[:5]}...")
        print(f"  Patching at layer = {patch_layer} (top head's layer)")

        # CIE-based FV
        fv_cie, _ = build_cie_fv(model, info, pairs, top_heads)
        cie_result = evaluate_fv(model, fv_cie, pairs, patch_layer)
        print(f"  CIE-FV:      acc={cie_result['acc']:.2%}  ||fv||={fv_cie.norm():.1f}")

        # Residual-stream FV at the same layer (for comparison)
        fv_resid = extract_fv_residual(
            model, get_extraction_pool(pairs), patch_layer,
            n_prompts=N_FV_PROMPTS, n_icl=N_ICL, seed=0
        )
        resid_result = evaluate_fv(model, fv_resid, pairs, patch_layer)
        print(f"  Resid-FV:    acc={resid_result['acc']:.2%}  ||fv||={fv_resid.norm():.1f}")

        # Random baseline (matched norm of CIE-FV)
        norm = fv_cie.norm()
        v_rand = torch.randn_like(fv_cie) * norm / np.sqrt(fv_cie.numel())
        rand_result = evaluate_fv(model, v_rand, pairs, patch_layer)
        print(f"  Random:      acc={rand_result['acc']:.2%}")

        out[task_name] = {
            "patch_layer": patch_layer,
            "top_k_heads": top_heads,
            "cie_fv_norm": float(fv_cie.norm()),
            "cie_fv": fv_cie.tolist(),
            "results": {
                "cie_fv": cie_result,
                "residual_fv": resid_result,
                "random": rand_result,
            },
        }

    # Save (without storing full FV tensors in JSON to keep size down)
    out_lite = {k: {kk: vv for kk, vv in v.items() if kk != "cie_fv"} for k, v in out.items()}
    with open(RESULTS_DIR / "05_cie_fv_validation.json", "w") as f:
        json.dump(out_lite, f, indent=2, default=str)
    # Save FVs separately as torch tensors
    torch.save({k: torch.tensor(v["cie_fv"]) for k, v in out.items()},
               RESULTS_DIR / "05_cie_fvs.pt")
    print(f"\nSaved to results/05_cie_fv_validation.json + 05_cie_fvs.pt")


if __name__ == "__main__":
    main()
