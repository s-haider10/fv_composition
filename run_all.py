"""
Master pipeline runner.
Runs stages 01-08 in order.

Usage: python run_all.py [--skip-stage N]
"""
import subprocess
import sys
import time
import argparse


STAGES = [
    ("01", "01_tokenization_audit.py", "Tokenization audit"),
    ("02", "02_atom_accuracy.py",      "Atom ICL accuracy"),
    ("03", "03_layer_sweep.py",        "Per-task layer sweep"),
    ("04", "04_cie_heatmaps.py",       "CIE per attention head"),
    ("05", "05_build_cie_fv.py",       "Build CIE-FV + validate"),
    ("06", "06_multi_position.py",     "Multi-position patching"),
    ("07", "07_composition_analysis.py","Composition analysis"),
    ("08", "08_make_plots.py",         "Generate plots"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], help="Stage IDs to skip (e.g. 01 02)")
    parser.add_argument("--only", nargs="*", default=None, help="Only run these stages")
    args = parser.parse_args()

    for stage_id, script, desc in STAGES:
        if args.only and stage_id not in args.only:
            continue
        if stage_id in args.skip:
            print(f"\n[SKIP] Stage {stage_id}: {desc}")
            continue

        print("\n" + "=" * 70)
        print(f"STAGE {stage_id}: {desc}")
        print("=" * 70)
        t0 = time.time()
        result = subprocess.run(["python", script], capture_output=False)
        elapsed = time.time() - t0
        print(f"\n[Stage {stage_id} done in {elapsed:.1f}s, exit={result.returncode}]")
        if result.returncode != 0:
            print(f"STAGE {stage_id} FAILED. Stopping.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Results in ./results/")


if __name__ == "__main__":
    main()
