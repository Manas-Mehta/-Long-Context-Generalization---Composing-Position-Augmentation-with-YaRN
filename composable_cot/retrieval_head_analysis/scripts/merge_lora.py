"""
Phase 2 — merge a PEFT LoRA adapter into the base Qwen2.5-7B-Instruct model and
save a standalone checkpoint that QRHead's `from_pretrained` can load directly.

For YaRN-using conditions (y2_base, y2_rpe_cur_L16k), bakes
`config.rope_scaling = {"type": "yarn", "factor": 4.0}` into the saved config,
so subsequent loads automatically apply YaRN. Equivalent to passing
`--enable-yarn --yarn-factor 4.0` to eval_babilong.py.

Verifies (--verify-yarn) that the saved config round-trips correctly by
reloading the bare config from disk and checking rope_scaling is what we set.
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="base model path or HF id (e.g. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--adapter", required=True,
                    help="PEFT adapter checkpoint dir (contains adapter_model.safetensors)")
    ap.add_argument("--output", required=True,
                    help="output dir for the merged checkpoint")
    ap.add_argument("--yarn-factor", type=float, default=None,
                    help="if set, bake YaRN rope_scaling with this factor into config")
    ap.add_argument("--verify-yarn", action="store_true",
                    help="reload the saved config and check yarn rope_scaling stuck")
    args = ap.parse_args()

    print("=" * 70)
    print("merge_lora.py")
    print(f"  base:    {args.base}")
    print(f"  adapter: {args.adapter}")
    print(f"  output:  {args.output}")
    if args.yarn_factor is not None:
        print(f"  yarn:    factor={args.yarn_factor}")
    print("=" * 70, flush=True)

    print("Loading base model in fp16 on CPU (this takes ~60s)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    print("  base model loaded", flush=True)

    print(f"Loading PEFT adapter from {args.adapter}...", flush=True)
    peft_model = PeftModel.from_pretrained(base, args.adapter)
    print("  adapter loaded", flush=True)

    print("Merging LoRA into base weights...", flush=True)
    merged = peft_model.merge_and_unload()
    print("  merged", flush=True)

    if args.yarn_factor is not None:
        print(f"Baking yarn rope_scaling into config: factor={args.yarn_factor}",
              flush=True)
        merged.config.rope_scaling = {"type": "yarn", "factor": args.yarn_factor}

    print(f"Saving merged checkpoint to {args.output}...", flush=True)
    merged.save_pretrained(args.output)

    print("Saving tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.output)

    print("Saved.", flush=True)

    # Free RAM before verify (the merged model is ~14 GB).
    del merged, peft_model, base
    import gc
    gc.collect()

    if args.verify_yarn and args.yarn_factor is not None:
        print("Verifying yarn config round-trips...", flush=True)
        cfg = AutoConfig.from_pretrained(args.output)
        rs = getattr(cfg, "rope_scaling", None)
        print(f"  config.rope_scaling = {rs}", flush=True)
        if rs is None:
            print("  ❌ rope_scaling missing from saved config", flush=True)
            sys.exit(1)
        if rs.get("type") != "yarn":
            print(f"  ❌ rope_scaling.type = {rs.get('type')}, expected 'yarn'",
                  flush=True)
            sys.exit(1)
        if abs(rs.get("factor", 0) - args.yarn_factor) > 1e-6:
            print(f"  ❌ rope_scaling.factor = {rs.get('factor')}, "
                  f"expected {args.yarn_factor}", flush=True)
            sys.exit(1)
        print("  ✅ YaRN config verified", flush=True)


if __name__ == "__main__":
    main()
