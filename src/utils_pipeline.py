"""
Shared utilities for the full pipeline.
"""
import torch
import numpy as np
from nnsight import LanguageModel
from collections import Counter
from pathlib import Path
import json

# ============= CONFIG =============
MODEL_NAME = "google/gemma-2-2b"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Layer/head structure (Gemma-2-2b: 26 layers, 8 KV heads, but actually 8 attn heads of d_head=256, d_model=2304)
# We'll dynamically read these.
SEEDS = [0, 1, 2]
N_ICL = 10
N_FV_PROMPTS = 50
N_EVAL = 30

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============= MODEL ACCESS =============
def load_model():
    print(f"Loading {MODEL_NAME}...")
    model = LanguageModel(MODEL_NAME, device_map=DEVICE, dtype=DTYPE)
    cfg = model.config
    info = {
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "d_model": cfg.hidden_size,
        "d_head": cfg.hidden_size // cfg.num_attention_heads,
    }
    print(f"  n_layers={info['n_layers']}, n_heads={info['n_heads']}, "
          f"d_model={info['d_model']}, d_head={info['d_head']}")
    return model, info


def get_layer(model, i):
    return model.model.layers[i]


def unwrap_saved(x):
    """Handle both old (.value) and new (direct) nnsight save APIs."""
    return x if isinstance(x, torch.Tensor) else x.value


# ============= PROMPTS =============
def make_icl_prompt(pairs, query):
    return "\n".join([f"{x} -> {y}" for x, y in pairs]) + f"\n{query} ->"


def make_zero_shot(query):
    return f"{query} ->"


# ============= TOKENIZATION AUDIT =============
def audit_tokenization(model, tasks, log_path):
    """Print and save tokenization details for every task pair."""
    tok = model.tokenizer
    audit = {}
    issues = []
    with open(log_path, "w") as f:
        for task_name, pairs in tasks.items():
            f.write(f"\n=== {task_name} ===\n")
            audit[task_name] = []
            for x, y in pairs:
                # Tokenize the answer ' y' with leading space (how it appears after '->')
                y_tokens = tok(" " + y, add_special_tokens=False)["input_ids"]
                x_tokens = tok(x, add_special_tokens=False)["input_ids"]
                # Test in actual prompt context
                test_prompt = f"a -> b\n{x} ->"
                full_tokens = tok(test_prompt, add_special_tokens=False)["input_ids"]
                full_strs = [tok.decode([t]) for t in full_tokens]

                first_y_tok = tok.decode(y_tokens[:1])
                multi_token_y = len(y_tokens) > 1
                if multi_token_y:
                    issues.append((task_name, x, y, len(y_tokens)))
                f.write(f"  {x} -> {y}  [x_tok={x_tokens}, y_tok={y_tokens}, "
                        f"first_y='{first_y_tok}', multitok={multi_token_y}]\n")
                audit[task_name].append({
                    "x": x, "y": y, "y_tokens": y_tokens,
                    "first_y_token_str": first_y_tok,
                    "multi_token_y": multi_token_y,
                })

        f.write(f"\n=== ISSUES (multi-token answers) ===\n")
        for issue in issues:
            f.write(f"  {issue}\n")

    return audit, issues


# ============= FV EXTRACTION (residual stream version) =============
@torch.no_grad()
def extract_fv_residual(model, task_pairs, layer_idx, n_prompts=N_FV_PROMPTS,
                        n_icl=N_ICL, seed=0):
    """Extract FV from last-token residual at layer_idx, averaged over n permutations."""
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(n_prompts):
        idx = rng.permutation(len(task_pairs))
        icl = [task_pairs[i] for i in idx[:n_icl]]
        query = task_pairs[idx[n_icl]][0]
        prompt = make_icl_prompt(icl, query)
        with model.trace(prompt):
            out = get_layer(model, layer_idx).output
            hidden = out[0] if isinstance(out, tuple) else out
            resid = hidden[:, -1, :].save()
        t = unwrap_saved(resid)
        vectors.append(t.float().cpu().squeeze())
    return torch.stack(vectors).mean(dim=0)


# ============= GENERATION =============
@torch.no_grad()
def generate_no_patch(model, prompts, max_tokens=5):
    completions = []
    for p in prompts:
        with model.generate(p, max_new_tokens=max_tokens, do_sample=False):
            gen = model.generator.output.save()
        g = unwrap_saved(gen)
        text = model.tokenizer.decode(g[0], skip_special_tokens=True)
        completions.append(text[len(p):].strip().split("\n")[0].strip())
    return completions


@torch.no_grad()
def generate_with_patch(model, prompts, fv, layer_idx, max_tokens=5):
    """Patch fv at last token of layer_idx during generation."""
    completions = []
    for p in prompts:
        with model.generate(p, max_new_tokens=max_tokens, do_sample=False):
            out = get_layer(model, layer_idx).output
            hidden = out[0] if isinstance(out, tuple) else out
            hidden[:, -1, :] = fv.to(DEVICE).to(DTYPE)
            gen = model.generator.output.save()
        g = unwrap_saved(gen)
        text = model.tokenizer.decode(g[0], skip_special_tokens=True)
        completions.append(text[len(p):].strip().split("\n")[0].strip())
    return completions


# ============= CLASSIFIERS =============
def classify_match(output, gold):
    """Does output start with gold (case-insensitive)?"""
    return output.lower().strip().startswith(gold.lower().strip())


def make_atom_classifier(pairs):
    """For atom: query -> match against gold."""
    expected = {x: y for x, y in pairs}
    def clf(query, output):
        if query not in expected:
            return "NO_GOLD"
        return "CORRECT_ATOM" if classify_match(output, expected[query]) else "WRONG"
    return clf


def make_composition_classifier(comp, atom_tasks):
    """Categorize composition output."""
    f_op, g_op = comp["f"], comp["g"]
    comp_map = {x: y for x, y in comp["pairs"]}

    def clf(query, output):
        if query in comp_map and classify_match(output, comp_map[query]):
            return "CORRECT_COMP"
        # G_ONLY: matches g(query)
        if g_op in atom_tasks:
            g_map = {x: y for x, y in atom_tasks[g_op]}
            if query in g_map and classify_match(output, g_map[query]):
                return "G_ONLY"
        # F_ONLY: matches f(query) directly (works for uppercase, first_letter)
        if f_op == "uppercase" and classify_match(output, query.upper()):
            return "F_ONLY"
        if f_op == "first_letter" and classify_match(output, query[0]):
            return "F_ONLY"
        # Arithmetic
        try:
            x = int(query)
            out = int(output.split()[0])
            ops = {"successor": x+1, "double": x*2, "triple": x*3}
            if f_op in ops and out == ops[f_op]:
                return "F_ONLY"
            if g_op in ops and out == ops[g_op]:
                return "G_ONLY"
            ops_fn = {"successor": lambda v: v+1, "double": lambda v: v*2, "triple": lambda v: v*3}
            if f_op in ops_fn and g_op in ops_fn:
                if out == ops_fn[f_op](ops_fn[g_op](x)): return "CORRECT_COMP"
                if out == ops_fn[g_op](ops_fn[f_op](x)): return "WRONG_ORDER"
        except (ValueError, IndexError):
            pass
        return "OTHER"
    return clf


def summarize(completions, queries, classifier):
    counter = Counter()
    for q, c in zip(queries, completions):
        counter[classifier(q, c)] += 1
    return counter


def fmt_dist(counter, n):
    return " ".join(f"{cat}={count}/{n}" for cat, count in counter.most_common())
