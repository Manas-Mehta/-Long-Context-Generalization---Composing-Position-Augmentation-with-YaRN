"""
Dump the actual top-K head identities per (condition, bin) so we can see
*which* heads overlap vs differ — not just counts.

Output:
  analysis/head_identities.md   human-readable per-bin tables
  analysis/head_identities.csv  machine-readable
  analysis/figures/layer_histogram.png  layer distribution of top-16
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter


BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
CONDITIONS = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]
SHORT = {"lora_base": "lora", "y2_base": "y2", "y2_rpe_cur_L16k": "y2_rpe"}

PUBLISHED_TOP16 = [
    "16-19", "16-2", "16-20", "16-14", "16-0", "15-24", "16-18", "17-18",
    "16-1", "19-18", "19-19", "18-16", "16-17", "19-25", "19-17", "20-21",
]

ROOT = Path("composable_cot/retrieval_head_analysis")
OUT = ROOT / "analysis"


def load_topk(cond, b, k):
    p = ROOT / "results" / cond / f"{b}_head_scores.json"
    pairs = json.loads(p.read_text())
    return [h for h, _ in pairs[:k]]


def head_layer(h):
    return int(h.split("-")[0])


def main():
    K = 16

    # CSV: row per (cond, bin, rank, head_id, layer)
    csv = OUT / "head_identities.csv"
    with csv.open("w") as f:
        f.write("condition,bin,rank,head_id,layer\n")
        for cond in CONDITIONS:
            for b in BINS:
                for r, h in enumerate(load_topk(cond, b, K), 1):
                    f.write(f"{cond},{b},{r},{h},{head_layer(h)}\n")
        for r, h in enumerate(PUBLISHED_TOP16, 1):
            f.write(f"published_zeroshot,(NQ),{r},{h},{head_layer(h)}\n")
    print(f"Wrote {csv}")

    # Markdown — focus on a few key bins (8K, 32K, 128K)
    md = OUT / "head_identities.md"
    lines = []
    lines.append("# Top-16 head identities per (condition, bin)\n")
    lines.append(f"Reference: published zero-shot Qwen top-16 (BEIR-NQ, paper-derived):")
    lines.append(f"`{', '.join(PUBLISHED_TOP16)}`\n")
    lines.append("---\n")

    for b in BINS:
        cond_tops = {c: load_topk(c, b, K) for c in CONDITIONS}
        # union and intersection across our 3 trained conditions
        union3 = set().union(*[set(v) for v in cond_tops.values()])
        inter3 = set(cond_tops[CONDITIONS[0]])
        for c in CONDITIONS[1:]:
            inter3 &= set(cond_tops[c])

        lines.append(f"## Bin {b}\n")
        lines.append(f"**Heads in ALL 3 trained conditions' top-16** ({len(inter3)} heads):")
        lines.append(f"`{', '.join(sorted(inter3))}`\n")
        lines.append(f"**Union of top-16 across all 3** ({len(union3)} heads): "
                     f"`{', '.join(sorted(union3))}`\n")

        # Per-condition: which are shared with published, which are unique to this condition
        for c in CONDITIONS:
            top = cond_tops[c]
            shared_with_pub = [h for h in top if h in PUBLISHED_TOP16]
            others = [h for h in top if h not in inter3]  # heads not shared with all 3
            lines.append(f"**{SHORT[c]}** top-16: `{', '.join(top)}`")
            lines.append(f"  - shared with published top-16 ({len(shared_with_pub)}): "
                         f"`{', '.join(shared_with_pub) or '—'}`")
            unique_to_c = [h for h in top if h not in
                           set().union(*[set(v) for k, v in cond_tops.items() if k != c])]
            lines.append(f"  - **unique to {SHORT[c]}** (not in other 2 trained tops): "
                         f"`{', '.join(unique_to_c) or '—'}`\n")

        # Pairwise diffs
        for ca in CONDITIONS:
            for cb in CONDITIONS:
                if ca >= cb:
                    continue
                only_a = sorted(set(cond_tops[ca]) - set(cond_tops[cb]))
                only_b = sorted(set(cond_tops[cb]) - set(cond_tops[ca]))
                lines.append(f"  - {SHORT[ca]} only (not in {SHORT[cb]}): "
                             f"`{', '.join(only_a) or '—'}`")
                lines.append(f"  - {SHORT[cb]} only (not in {SHORT[ca]}): "
                             f"`{', '.join(only_b) or '—'}`\n")
        lines.append("")

    md.write_text("\n".join(lines))
    print(f"Wrote {md}")

    # Layer histogram of top-16, per condition, at 8K / 32K / 128K
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True)
    for col, b in enumerate(["8k", "32k", "128k"]):
        for row, c in enumerate(CONDITIONS):
            ax = axes[row, col]
            top = load_topk(c, b, K)
            layers = [head_layer(h) for h in top]
            cnt = Counter(layers)
            xs = sorted(cnt.keys())
            ys = [cnt[x] for x in xs]
            ax.bar(xs, ys, color="C0")
            # overlay published in faint
            pub_layers = Counter(head_layer(h) for h in PUBLISHED_TOP16)
            ax.bar(sorted(pub_layers.keys()),
                    [pub_layers[x] for x in sorted(pub_layers.keys())],
                    alpha=0.25, color="red", label="published")
            ax.set_title(f"{SHORT[c]} @ {b}")
            ax.set_xlabel("layer")
            ax.set_xticks(range(0, 28, 4))
            ax.set_xlim(-0.5, 27.5)
            if col == 0:
                ax.set_ylabel("# heads in top-16")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Layer distribution of top-16 heads (red overlay = published zero-shot)")
    fig.tight_layout()
    p = OUT / "figures" / "layer_histogram.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
