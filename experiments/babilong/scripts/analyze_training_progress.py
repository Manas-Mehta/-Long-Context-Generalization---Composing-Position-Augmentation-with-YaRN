#!/usr/bin/env python
"""Parse SLURM training logs to estimate epoch-1 completion time and assess training health.

Usage:
    python experiments/babilong/scripts/analyze_training_progress.py \
        --log-dir experiments/babilong/slurm_logs/

Reads all train_*.out files, parses step/loss/epoch/elapsed, and prints:
  1. Time to epoch 1 (step 5000) per condition
  2. Training health assessment (loss curve, grad norms, warnings)
  3. Mid-training accuracy if available in the log
"""

import argparse
import glob
import os
import re
from datetime import datetime, timedelta


# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------

def parse_log(path: str) -> dict:
    """Parse one SLURM .out file. Returns dict with all extracted info."""
    condition = "unknown"
    steps     = []  # list of {step, epoch, loss, grad_norm, elapsed_s, lr}
    start_ts  = None
    mid_evals = []  # list of {step, bin, accuracy}
    yarn_ok   = None
    oom       = False

    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        # Condition name
        m = re.search(r"Condition:\s+(\S+)", line)
        if m:
            condition = m.group(1)

        # Training start timestamp
        m = re.search(r"Start:\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m:
            start_ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

        # Step line: "  Step   500/15000 | Ep  0.10 | Loss 0.4321 | LR 1.23e-04 | GPU 33.8GB | ETA 35:12:00"
        m = re.match(
            r"Step\s+(\d+)/(\d+)\s+\|\s+Ep\s+([\d.]+)\s+\|\s+Loss\s+([\d.]+)\s+\|\s+LR\s+([\d.eE+-]+)\s+\|\s+GPU\s+\S+\s+\|\s+ETA\s+(\S+)",
            line,
        )
        if m:
            step      = int(m.group(1))
            max_steps = int(m.group(2))
            epoch     = float(m.group(3))
            loss      = float(m.group(4))
            lr        = float(m.group(5))
            steps.append({
                "step": step, "max_steps": max_steps,
                "epoch": epoch, "loss": loss, "lr": lr,
            })

        # Grad norm from HF JSON line: {'loss': ..., 'grad_norm': ..., ...}
        m = re.search(r"'grad_norm':\s*([\d.]+)", line)
        if m and steps:
            steps[-1]["grad_norm"] = float(m.group(1))

        # elapsed_s — not directly in our log format, derive from step timing below

        # Mid-training eval: "MidTrainingEval [step=500] 0k: 0.8450"
        m = re.search(r"MidTrainingEval.*step[=\s]*(\d+).*?(\w+k):\s*([\d.]+)", line)
        if m:
            mid_evals.append({
                "step": int(m.group(1)),
                "bin":  m.group(2),
                "acc":  float(m.group(3)),
            })

        # YaRN verification
        if "YaRN NOT APPLIED" in line:
            yarn_ok = False
        if "inv_freq dims differ" in line and yarn_ok is None:
            m = re.search(r"(\d+)/(\d+) inv_freq dims differ", line)
            if m and int(m.group(1)) > 0:
                yarn_ok = True

        # OOM
        if "OutOfMemoryError" in line or "CUDA out of memory" in line:
            oom = True

    return {
        "path":      path,
        "condition": condition,
        "steps":     steps,
        "start_ts":  start_ts,
        "mid_evals": mid_evals,
        "yarn_ok":   yarn_ok,
        "oom":       oom,
    }


# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

def estimate_epoch1_eta(data: dict) -> str:
    steps = data["steps"]
    if not steps:
        return "no data"

    last = steps[-1]
    current_step = last["step"]
    max_steps    = last.get("max_steps", 15000)
    epoch1_step  = max_steps // 3  # 5000 for 15000-step run

    if current_step >= epoch1_step:
        return "DONE (epoch 1 complete)"

    # Estimate seconds/step from last 100 steps
    recent = steps[-100:] if len(steps) >= 100 else steps
    if len(recent) < 2:
        return "insufficient data"

    # Each step entry has a step number; use step count vs wall time isn't stored
    # Instead derive from the ETA field that was printed — or use step density
    # We'll use: total elapsed / steps_done * steps_remaining
    # elapsed_s isn't stored directly; approximate from start_ts if available
    start_ts = data["start_ts"]
    if start_ts:
        from datetime import datetime as dt
        now_approx = start_ts  # we don't know "now" exactly from the log
        # Use step count ratio instead
        steps_done    = current_step
        steps_to_ep1  = epoch1_step - current_step
        # Estimate from the last printed ETA
        # Find last ETA in log
        eta_str = _find_last_eta(data["path"])
        if eta_str:
            return f"~{_eta_to_ep1(eta_str, current_step, epoch1_step, max_steps)}"

    steps_done   = current_step
    steps_to_ep1 = epoch1_step - current_step
    # Rate: assume we can get steps/elapsed from step numbers alone if they're dense
    return f"{steps_to_ep1} steps remaining to epoch 1"


def _find_last_eta(path: str) -> str | None:
    """Scan file backwards for the last ETA value."""
    with open(path, "rb") as f:
        # Read last 50KB
        try:
            f.seek(-50000, 2)
        except OSError:
            f.seek(0)
        tail = f.read().decode("utf-8", errors="ignore")

    matches = re.findall(
        r"Step\s+(\d+)/(\d+).*?ETA\s+(\d+:\d+:\d+)",
        tail,
    )
    if not matches:
        return None
    last = matches[-1]
    return last  # (step, max_steps, eta_str)


def _eta_to_ep1(eta_tuple, current_step, epoch1_step, max_steps):
    """Given ETA to end of training, compute ETA to epoch 1."""
    step_str, max_str, hms = eta_tuple
    step     = int(step_str)
    max_s    = int(max_str)

    # Parse HH:MM:SS
    parts = hms.split(":")
    total_remaining_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    steps_remaining_total = max_s - step
    if steps_remaining_total <= 0:
        return "done"

    sps = total_remaining_s / steps_remaining_total  # seconds per step
    steps_to_ep1 = epoch1_step - step

    if steps_to_ep1 <= 0:
        return "epoch 1 already passed"

    seconds_to_ep1 = steps_to_ep1 * sps
    td = timedelta(seconds=int(seconds_to_ep1))
    hours   = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"


def assess_health(data: dict) -> list[str]:
    """Return list of health observations."""
    steps  = data["steps"]
    notes  = []

    if not steps:
        notes.append("WARNING: No step data parsed — log may be too early or format mismatch")
        return notes

    if data["oom"]:
        notes.append("ERROR: OOM detected in log")

    last = steps[-1]
    current_step = last["step"]
    notes.append(f"Current step: {current_step}/15000 (epoch {last['epoch']:.2f})")

    # Loss trend — compare first 10 steps vs last 50
    if len(steps) >= 60:
        early_loss = sum(s["loss"] for s in steps[:10]) / 10
        recent_loss = sum(s["loss"] for s in steps[-50:]) / 50
        drop_pct = (early_loss - recent_loss) / early_loss * 100
        notes.append(f"Loss: {early_loss:.3f} (start) → {recent_loss:.3f} (recent) — {drop_pct:.0f}% drop")
        if recent_loss > 1.5:
            notes.append("WARNING: Loss still high (>1.5) — model may not be learning")
        elif recent_loss < 0.05 and current_step < 3000:
            notes.append("WARNING: Loss very low early — possible memorization")
        else:
            notes.append("OK: Loss is decreasing normally")

    # Grad norm — look for spikes
    grad_norms = [s["grad_norm"] for s in steps if "grad_norm" in s]
    if grad_norms:
        max_gn   = max(grad_norms)
        recent_gn = sum(grad_norms[-20:]) / min(20, len(grad_norms))
        notes.append(f"Grad norm: max={max_gn:.1f}, recent avg={recent_gn:.1f}")
        if max_gn > 100:
            notes.append(f"WARNING: Grad norm spike detected (max={max_gn:.1f}) — check for instability")
        else:
            notes.append("OK: Grad norms look stable")

    # YaRN check
    if data["yarn_ok"] is False:
        notes.append("ERROR: YaRN NOT APPLIED — this condition's results will be wrong")
    elif data["yarn_ok"] is True:
        notes.append("OK: YaRN applied successfully")

    # Loss at key milestones
    milestones = [500, 1000, 2000, 3000, 5000]
    milestone_losses = {}
    for s in steps:
        for m in milestones:
            if s["step"] == m:
                milestone_losses[m] = s["loss"]
    if milestone_losses:
        ml_str = "  |  ".join(f"step {k}: {v:.3f}" for k, v in sorted(milestone_losses.items()))
        notes.append(f"Loss at milestones: {ml_str}")

    return notes


def print_mid_evals(mid_evals: list):
    if not mid_evals:
        # Try to parse from log differently — print nothing, caller handles
        return

    # Group by bin
    by_bin = {}
    for e in mid_evals:
        by_bin.setdefault(e["bin"], []).append((e["step"], e["acc"]))

    bins = sorted(by_bin.keys())
    if not bins:
        return

    print(f"  Mid-training accuracy:")
    header = f"    {'Step':>6}  " + "  ".join(f"{b:>6}" for b in bins)
    print(header)

    # Collect all steps
    all_steps = sorted(set(e["step"] for e in mid_evals))
    for step in all_steps:
        row = f"    {step:>6}  "
        for b in bins:
            val = next((acc for s, acc in by_bin[b] if s == step), None)
            row += f"  {val:>6.3f}" if val is not None else f"  {'N/A':>6}"
        print(row)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="experiments/babilong/slurm_logs/")
    args = p.parse_args()

    pattern = os.path.join(args.log_dir, "train_*.out")
    log_files = sorted(glob.glob(pattern))

    if not log_files:
        print(f"No log files found matching: {pattern}")
        return

    print("=" * 70)
    print("BABILong TRAINING PROGRESS ANALYSIS")
    print(f"Log dir: {args.log_dir}")
    print("=" * 70)

    for path in log_files:
        fname = os.path.basename(path)
        print(f"\n{'─'*70}")
        print(f"FILE: {fname}")
        print(f"{'─'*70}")

        data = parse_log(path)
        print(f"Condition: {data['condition']}")

        # ETA to epoch 1
        eta = estimate_epoch1_eta(data)
        print(f"Time to epoch 1: {eta}")

        # Health
        print("\nHealth assessment:")
        for note in assess_health(data):
            prefix = "  ⚠ " if "WARNING" in note or "ERROR" in note else "  ✓ " if "OK" in note else "    "
            print(f"{prefix}{note}")

        # Mid-training eval
        if data["mid_evals"]:
            print()
            print_mid_evals(data["mid_evals"])
        else:
            print("\n  Mid-training eval: not yet logged (first eval at step 500)")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
