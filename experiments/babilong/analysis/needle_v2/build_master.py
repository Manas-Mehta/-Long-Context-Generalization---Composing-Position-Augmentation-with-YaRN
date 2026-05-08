"""
Build master long-format CSV for v2 needle-position eval.

Joins per-sample predictions (3 models x 3 zones x 8 bins) with eval-data metadata
(tier, target_entries, fact_positions_rel, original_idx, etc.) by file order.

Output: analysis/needle_v2/master.csv
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/reach/CodingRepositories/02 TAUR Labs/RPE/experiments/babilong")
DATA_DIR = ROOT / "data" / "eval_needle_v2"
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "analysis" / "needle_v2"

MODELS = {
    "lora_base":         "lora_base_needle_v2",
    "y2_base":           "y2_base_needle_v2",
    "y2_rpe_cur_L16k":   "y2_rpe_cur_L16k_needle_v2",
}
ZONES = ["beg", "mid", "end"]
BINS = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


def normalize(s):
    if s is None:
        return ""
    return str(s).strip().lower().rstrip(".,!?\"'")


def load_cell_meta(zone, bin_):
    path = DATA_DIR / f"{zone}_{bin_}.json"
    data = json.load(open(path))
    rows = []
    for i, s in enumerate(data):
        fp = s.get("fact_positions_rel") or []
        rows.append({
            "zone": zone,
            "bin": bin_,
            "row_in_cell": i,
            "original_idx": s.get("original_idx"),
            "tier": s.get("tier"),
            "object": s.get("object"),
            "answer": s.get("answer"),
            "question": s.get("question"),
            "target_entries": s.get("target_entries"),
            "n_facts": len(fp),
            "min_fact_pos": min(fp) if fp else None,
            "max_fact_pos": max(fp) if fp else None,
            "mean_fact_pos": sum(fp) / len(fp) if fp else None,
            "first_fact_pos": fp[0] if fp else None,
            "last_fact_pos": fp[-1] if fp else None,
            "zone_pct": s.get("zone_pct"),
            "token_count": s.get("token_count"),
        })
    return rows


def main():
    meta_rows = []
    for zone in ZONES:
        for b in BINS:
            meta_rows.extend(load_cell_meta(zone, b))
    meta = pd.DataFrame(meta_rows)
    meta["bin_tokens"] = meta["bin"].map({
        "1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192,
        "16k": 16384, "32k": 32768, "64k": 65536, "128k": 130700,
    })

    print(f"Meta rows: {len(meta)}")
    print(meta.groupby(["zone", "bin"]).size().unstack().to_string())

    long_rows = []
    for model_name, dir_prefix in MODELS.items():
        for zone in ZONES:
            for b in BINS:
                pred_path = RESULTS_DIR / f"{dir_prefix}_{zone}" / f"predictions_{b}.json"
                preds = json.load(open(pred_path))
                cell_meta = meta[(meta.zone == zone) & (meta.bin == b)].sort_values("row_in_cell").reset_index(drop=True)
                assert len(preds) == len(cell_meta), f"len mismatch {model_name}/{zone}/{b}: preds={len(preds)} meta={len(cell_meta)}"
                for i, p in enumerate(preds):
                    row = cell_meta.iloc[i].to_dict()
                    row["model"] = model_name
                    row["prediction"] = p["prediction"]
                    row["correct"] = bool(p["correct"])
                    row["pred_norm"] = normalize(p["prediction"])
                    row["target_norm"] = normalize(p["target"])
                    row["n_tokens_pred"] = p.get("n_tokens")
                    long_rows.append(row)

    long = pd.DataFrame(long_rows)
    print(f"\nLong rows: {len(long)}")
    print(f"Models: {long['model'].unique().tolist()}")
    print(f"Tiers: {long['tier'].value_counts().to_dict()}")

    out = OUT_DIR / "master.csv"
    long.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # Also save meta separately
    meta.to_csv(OUT_DIR / "sample_meta.csv", index=False)
    print(f"Wrote {OUT_DIR / 'sample_meta.csv'}")


if __name__ == "__main__":
    main()
