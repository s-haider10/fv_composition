"""
STAGE 2: Atom baseline ICL accuracy.

For each atomic task, measure ICL accuracy on the held-out eval pool.
Uses extraction pool for ICL examples; eval pool for queries.
Multiple seeds for variance estimate.

Output: results/02_atom_accuracy.json
"""
import json
import numpy as np
from utils_pipeline import (load_model, make_icl_prompt, generate_no_patch,
                            make_atom_classifier, summarize, fmt_dist,
                            N_ICL, SEEDS, RESULTS_DIR)
from tasks import TASKS, get_extraction_pool, get_eval_pool


def evaluate_atom(model, task_name, pairs, seed):
    rng = np.random.default_rng(seed)
    extraction = get_extraction_pool(pairs)
    eval_pool = get_eval_pool(pairs)

    # ICL prompts: sample N_ICL from extraction pool, query from eval pool
    prompts = []
    for q, _ in eval_pool:
        idx = rng.permutation(len(extraction))[:N_ICL]
        icl = [extraction[i] for i in idx]
        prompts.append(make_icl_prompt(icl, q))

    completions = generate_no_patch(model, prompts)
    clf = make_atom_classifier(eval_pool)
    counter = summarize(completions, [q for q, _ in eval_pool], clf)
    return counter, completions


def main():
    model, info = load_model()
    results = {}

    for task_name, pairs in TASKS.items():
        print(f"\n--- {task_name} ---")
        accs = []
        seed_results = []
        for seed in SEEDS:
            counter, completions = evaluate_atom(model, task_name, pairs, seed)
            n = sum(counter.values())
            correct = counter.get("CORRECT_ATOM", 0)
            acc = correct / n if n > 0 else 0.0
            accs.append(acc)
            seed_results.append({
                "seed": seed,
                "acc": acc,
                "dist": dict(counter),
                "sample_completions": completions[:3],
            })
            print(f"  seed {seed}: acc={acc:.2%}  ({fmt_dist(counter, n)})")
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        print(f"  MEAN acc = {mean_acc:.2%} ± {std_acc:.2%}")
        results[task_name] = {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "per_seed": seed_results,
        }

    out_path = RESULTS_DIR / "02_atom_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
