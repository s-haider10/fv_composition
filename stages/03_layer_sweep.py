"""
STAGE 3: Per-task residual-stream FV layer sweep.

For each atom, extract FV at every layer, patch into zero-shot, measure recovery.
Find best layer per atom. This addresses "Layer 15 chosen on country_capital alone."

Output: results/03_layer_sweep.json + heatmap data
"""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import numpy as np
import torch
from utils_pipeline import (load_model, extract_fv_residual, generate_with_patch,
                            make_zero_shot, make_atom_classifier, summarize,
                            N_FV_PROMPTS, N_ICL, SEEDS, RESULTS_DIR)
from tasks import TASKS, get_extraction_pool, get_eval_pool


# Sweep every 2 layers to save time (Gemma-2-2b has 26 layers)
LAYER_STRIDE = 2


def sweep_atom(model, info, task_name, pairs, seed):
    extraction = get_extraction_pool(pairs)
    eval_pool = get_eval_pool(pairs)
    eval_queries = [q for q, _ in eval_pool]
    zero_shot_prompts = [make_zero_shot(q) for q in eval_queries]
    clf = make_atom_classifier(eval_pool)

    results = {}
    layers = list(range(0, info["n_layers"], LAYER_STRIDE))
    for layer in layers:
        fv = extract_fv_residual(model, extraction, layer,
                                 n_prompts=N_FV_PROMPTS, n_icl=N_ICL, seed=seed)
        completions = generate_with_patch(model, zero_shot_prompts, fv, layer)
        counter = summarize(completions, eval_queries, clf)
        n = sum(counter.values())
        acc = counter.get("CORRECT_ATOM", 0) / n if n > 0 else 0.0
        results[layer] = {"acc": acc, "n": n, "fv_norm": float(fv.norm())}
        print(f"    layer {layer:2d}: acc={acc:.2%}  ||fv||={fv.norm():.1f}")
    return results


def main():
    model, info = load_model()
    all_results = {}

    for task_name, pairs in TASKS.items():
        print(f"\n=== {task_name} ===")
        per_seed = {}
        for seed in SEEDS[:1]:   # 1 seed for sweep (cost), more seeds at chosen layer
            print(f"  --- seed {seed} ---")
            per_seed[seed] = sweep_atom(model, info, task_name, pairs, seed)

        # Find best layer (averaged over seeds)
        layer_accs = {}
        for seed, layer_dict in per_seed.items():
            for layer, r in layer_dict.items():
                layer_accs.setdefault(layer, []).append(r["acc"])
        mean_layer_accs = {l: float(np.mean(v)) for l, v in layer_accs.items()}
        best_layer = max(mean_layer_accs, key=mean_layer_accs.get)
        best_acc = mean_layer_accs[best_layer]

        all_results[task_name] = {
            "best_layer": best_layer,
            "best_acc": best_acc,
            "per_layer": mean_layer_accs,
            "per_seed_raw": per_seed,
        }
        print(f"  >>> BEST: layer {best_layer}, acc={best_acc:.2%}")

    out_path = RESULTS_DIR / "03_layer_sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
