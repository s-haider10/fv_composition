# Function Vector Composition: Full Pipeline

End-to-end mech interp pipeline addressing all methodological gaps.

## Pipeline stages

| Stage | Script | What it does | Addresses |
|---|---|---|---|
| 01 | `01_tokenization_audit.py` | Audit every (x,y) for token splits | Tokenization not audited |
| 02 | `02_atom_accuracy.py` | Atom ICL accuracy w/ disjoint pools, 3 seeds | Single ICL pool, no seeds |
| 03 | `03_layer_sweep.py` | Residual FV at every 2nd layer per task | Single layer for all tasks |
| 04 | `04_cie_heatmaps.py` | CIE per (layer, head) per atom | No CIE ranking |
| 05 | `05_build_cie_fv.py` | Build top-K-head CIE FV, validate vs residual & random | Residual not heads |
| 06 | `06_multi_position.py` | Patch at -1, -2, -3 positions | Last token only |
| 07 | `07_composition_analysis.py` | v_h vs v_f+v_g vs v_dominant causal | Cosine-only |
| 08 | `08_make_plots.py` | All publication figures | — |

## Run

```bash
cd /home/claude/full_pipeline
python run_all.py
```

Estimated runtime on 1× A6000: 2–4 hours total.

To skip stages: `python run_all.py --skip 01 02`
To run one stage: `python run_all.py --only 04`

## Outputs

- `results/01_tokenization_audit.txt|.json`
- `results/02_atom_accuracy.json`
- `results/03_layer_sweep.json`
- `results/04_cie_heatmaps.json`
- `results/05_cie_fv_validation.json` + `05_cie_fvs.pt`
- `results/06_multi_position.json`
- `results/07_composition_analysis.json`
- `results/plots/*.pdf` and `*.png`

## Key methodology choices

- **Disjoint pools**: first 20 pairs per atom = extraction; last 10 = eval. Eval queries never appear in FV extraction.
- **CIE corrupted = shuffled outputs**: clean ICL has matched pairs, corrupted has same inputs with permuted outputs. Standard ARENA approach.
- **3 seeds** for atom accuracy (stage 02). Layer sweep uses 1 seed for cost; could extend.
- **Top-K = 10 heads** for CIE-FV (Function Vectors paper uses ~10).
- **Patch layer for CIE-FV** = layer of top-1 CIE head.
- **Format-corrected cosine** = raw cos minus mean cos with unrelated atoms.

## Decision tree for paper

After running:

1. **If CIE-FV recovers atoms ≥70% across all atoms** → methodology validated. Composition analysis (stage 07) is the headline.
2. **If CIE-FV recovers some atoms but not arithmetic** → "compositional structure is type-specific" paper.
3. **If CIE-FV doesn't recover anything beyond residual** → write up methodological negative result with clean CIE plots.

All three are publishable.
