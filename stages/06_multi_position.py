"""
STAGE 6: Multi-position patching for compositions.

Tests whether composition signal lives at positions other than last token.
Patches FV at every position of zero-shot prompt and reports best position.

Output: results/06_multi_position.json
"""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import torch
from utils_pipeline import (load_model, get_layer, unwrap_saved, make_zero_shot,
                            make_composition_classifier, summarize,
                            DEVICE, DTYPE, RESULTS_DIR)
from tasks import TASKS, COMPOSITIONS, get_eval_pool


@torch.no_grad()
def patch_at_position(model, prompts, fv, layer_idx, position):
    """Patch fv at given seq position. position can be -1, -2, etc."""
    completions = []
    for p in prompts:
        with model.generate(p, max_new_tokens=5, do_sample=False):
            out = get_layer(model, layer_idx).output
            hidden = out[0] if isinstance(out, tuple) else out
            hidden[:, position, :] = fv.to(DEVICE).to(DTYPE)
            gen = model.generator.output.save()
        g = unwrap_saved(gen)
        text = model.tokenizer.decode(g[0], skip_special_tokens=True)
        completions.append(text[len(p):].strip().split("\n")[0].strip())
    return completions


def main():
    model, info = load_model()
    # Load CIE FVs
    cie_fvs = torch.load(RESULTS_DIR / "05_cie_fvs.pt")
    with open(RESULTS_DIR / "05_cie_fv_validation.json") as f:
        cie_meta = json.load(f)

    out = {}
    for cname, comp in COMPOSITIONS.items():
        if cname not in cie_fvs:
            continue
        print(f"\n=== {cname} ===")
        eval_pool = get_eval_pool(comp["pairs"])
        eval_qs = [q for q, _ in eval_pool]
        prompts = [make_zero_shot(q) for q in eval_qs]

        fv_h = cie_fvs[cname]
        layer = cie_meta[cname]["patch_layer"]
        clf = make_composition_classifier(comp, TASKS)

        # Try positions -1, -2, -3 (last token, query token, "->" token)
        position_results = {}
        for pos in [-1, -2, -3]:
            try:
                completions = patch_at_position(model, prompts, fv_h, layer, pos)
                counter = summarize(completions, eval_qs, clf)
                n = sum(counter.values())
                position_results[pos] = {
                    "dist": dict(counter),
                    "n": n,
                    "correct_comp": counter.get("CORRECT_COMP", 0) / n if n > 0 else 0,
                }
                print(f"  pos {pos:>2d}: {dict(counter)}")
            except Exception as e:
                print(f"  pos {pos}: FAILED ({e})")
                position_results[pos] = {"error": str(e)}

        out[cname] = {"layer": layer, "positions": position_results}

    with open(RESULTS_DIR / "06_multi_position.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to results/06_multi_position.json")


if __name__ == "__main__":
    main()
