"""
STAGE 1: Tokenization audit.

Verify every (input, output) pair tokenizes cleanly.
Flag multi-token answers and unexpected splits.
Output: results/01_tokenization_audit.txt + .json
"""
import json
from utils_pipeline import load_model, audit_tokenization, RESULTS_DIR
from tasks import TASKS, COMPOSITIONS


def main():
    model, info = load_model()
    log_path = RESULTS_DIR / "01_tokenization_audit.txt"
    json_path = RESULTS_DIR / "01_tokenization_audit.json"

    # Combine atoms and compositions for audit
    all_tasks = dict(TASKS)
    for cname, comp in COMPOSITIONS.items():
        all_tasks[f"COMP::{cname}"] = comp["pairs"]

    audit, issues = audit_tokenization(model, all_tasks, log_path)

    print(f"\nAudit written to {log_path}")
    print(f"\nMulti-token answer issues: {len(issues)}")
    if issues:
        print("Top issues (task, x, y, n_tokens):")
        for i in issues[:20]:
            print(f"  {i}")

    # Summary stats
    summary = {}
    for task, entries in audit.items():
        n_total = len(entries)
        n_multi = sum(1 for e in entries if e["multi_token_y"])
        summary[task] = {"total": n_total, "multi_token": n_multi}

    with open(json_path, "w") as f:
        json.dump({"summary": summary, "n_issues": len(issues)}, f, indent=2)

    print("\nPer-task summary:")
    for t, s in summary.items():
        flag = " *MULTI*" if s["multi_token"] > 0 else ""
        print(f"  {t}: {s['multi_token']}/{s['total']} multi-token{flag}")


if __name__ == "__main__":
    main()
