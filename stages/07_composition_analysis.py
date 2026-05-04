"""
STAGE 7: Composition analysis with CIE-based FVs.

Tests three hypotheses for each composition (h = f∘g):
  H1 (Additivity): v_h ≈ v_f + v_g  -- patch v_f+v_g, measure CORRECT_COMP
  H2 (Dominance):  v_h ≈ v_dominant -- check if v_h patch matches dominant atom output
  H3 (Novel):      v_h is a separate vector  -- direct v_h patch produces CORRECT_COMP

Uses CIE-FVs from stage 5, patches at each composition's best layer.

Output: results/07_composition_analysis.json
"""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import numpy as np
import torch
from utils_pipeline import (load_model, generate_with_patch, generate_no_patch,
                            make_zero_shot, make_composition_classifier,
                            summarize, fmt_dist, DEVICE, DTYPE, RESULTS_DIR)
from tasks import TASKS, COMPOSITIONS, get_eval_pool


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main():
    model, info = load_model()
    cie_fvs = torch.load(RESULTS_DIR / "05_cie_fvs.pt")
    with open(RESULTS_DIR / "05_cie_fv_validation.json") as f:
        cie_meta = json.load(f)

    out = {}
    for cname, comp in COMPOSITIONS.items():
        if cname not in cie_fvs or comp["f"] not in cie_fvs or comp["g"] not in cie_fvs:
            print(f"  Skipping {cname} (missing FVs)")
            continue
        print(f"\n=== {cname}: h = {comp['f']}({comp['g']}(x)) ===")

        v_h = cie_fvs[cname]
        v_f = cie_fvs[comp["f"]]
        v_g = cie_fvs[comp["g"]]
        v_sum = v_f + v_g
        v_rand = torch.randn_like(v_h) * v_h.norm() / np.sqrt(v_h.numel())

        layer = cie_meta[cname]["patch_layer"]
        eval_pool = get_eval_pool(comp["pairs"])
        eval_qs = [q for q, _ in eval_pool]
        prompts = [make_zero_shot(q) for q in eval_qs]
        clf = make_composition_classifier(comp, TASKS)

        # Geometric measures
        # Format-corrected: subtract baseline cos against unrelated atoms
        unrelated_atoms = [a for a in TASKS if a != comp["f"] and a != comp["g"]]
        unrelated_cos = [cos(v_h, cie_fvs[a]) for a in unrelated_atoms if a in cie_fvs]
        baseline_cos = float(np.mean(unrelated_cos)) if unrelated_cos else 0.0

        cos_h_f = cos(v_h, v_f) - baseline_cos
        cos_h_g = cos(v_h, v_g) - baseline_cos
        cos_h_sum = cos(v_h, v_sum) - baseline_cos
        cos_f_g = cos(v_f, v_g) - baseline_cos
        print(f"  cos_corrected(v_h, v_f)     = {cos_h_f:+.3f}")
        print(f"  cos_corrected(v_h, v_g)     = {cos_h_g:+.3f}")
        print(f"  cos_corrected(v_h, v_f+v_g) = {cos_h_sum:+.3f}")
        print(f"  cos_corrected(v_f, v_g)     = {cos_f_g:+.3f}")

        # Causal interventions
        baseline_compl = generate_no_patch(model, prompts)
        h_compl = generate_with_patch(model, prompts, v_h, layer)
        f_compl = generate_with_patch(model, prompts, v_f, layer)
        g_compl = generate_with_patch(model, prompts, v_g, layer)
        sum_compl = generate_with_patch(model, prompts, v_sum, layer)
        rand_compl = generate_with_patch(model, prompts, v_rand, layer)

        results = {}
        for label, completions in [("baseline", baseline_compl),
                                   ("v_h", h_compl),
                                   ("v_f", f_compl),
                                   ("v_g", g_compl),
                                   ("v_f+v_g", sum_compl),
                                   ("v_random", rand_compl)]:
            counter = summarize(completions, eval_qs, clf)
            n = sum(counter.values())
            results[label] = {
                "dist": dict(counter),
                "n": n,
                "correct_comp_rate": counter.get("CORRECT_COMP", 0) / n if n > 0 else 0,
                "f_only_rate": counter.get("F_ONLY", 0) / n if n > 0 else 0,
                "g_only_rate": counter.get("G_ONLY", 0) / n if n > 0 else 0,
                "samples": list(zip(eval_qs[:3], completions[:3])),
            }
            print(f"  {label:<10s}: {fmt_dist(counter, n)}")

        out[cname] = {
            "layer": layer,
            "f": comp["f"], "g": comp["g"],
            "geometry": {
                "cos_h_f_corrected": cos_h_f,
                "cos_h_g_corrected": cos_h_g,
                "cos_h_sum_corrected": cos_h_sum,
                "cos_f_g_corrected": cos_f_g,
                "baseline_cos": baseline_cos,
            },
            "causal": results,
        }

        # Hypothesis check
        comp_rate_h = results["v_h"]["correct_comp_rate"]
        comp_rate_sum = results["v_f+v_g"]["correct_comp_rate"]
        baseline_comp_rate = results["baseline"]["correct_comp_rate"]
        print(f"  >> v_h: {comp_rate_h:.0%} CORRECT_COMP  (baseline {baseline_comp_rate:.0%})")
        print(f"  >> v_f+v_g: {comp_rate_sum:.0%} CORRECT_COMP")

    with open(RESULTS_DIR / "07_composition_analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to results/07_composition_analysis.json")


if __name__ == "__main__":
    main()
