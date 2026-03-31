"""
TD4 — Full Pipeline Orchestrator
=================================
Runs all 4 steps in sequence:
  Step 1 — Build private KB from TD1 outputs
  Step 2 — Align entities & predicates with Wikidata
  Step 3 — Expand KB via Wikidata SPARQL  (target: 50k-200k triples)
  Step 4 — Generate statistics + report

Usage:
  python pipeline.py              # full pipeline
  python pipeline.py --from 2     # restart from step 2 (1/2/3/4)
  python pipeline.py --only 1     # run only step 1
  python pipeline.py --fast-expand  # step 3 in fast mode (skip entity-level)
  python pipeline.py --skip-align   # skip step 2 (use existing alignment.ttl)
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def run_step(n: int, module_name: str, fn_name: str = "main", **kwargs):
    print(f"\n{'='*60}")
    print(f" STEP {n}: {module_name}")
    print(f"{'='*60}")
    t0 = time.time()

    import importlib
    mod = importlib.import_module(module_name.replace(".py", ""))
    fn  = getattr(mod, fn_name)
    fn(**kwargs)

    elapsed = time.time() - t0
    print(f"\n  Step {n} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="TD4 Full Pipeline")
    parser.add_argument("--from",        dest="from_step", type=int, default=1,
                        help="Start from step N (1-4)")
    parser.add_argument("--only",        dest="only_step", type=int, default=None,
                        help="Run only step N")
    parser.add_argument("--fast-expand", action="store_true",
                        help="Step 3: skip entity-level expansion")
    parser.add_argument("--skip-align",  action="store_true",
                        help="Skip step 2 (use existing alignment.ttl)")
    parser.add_argument("--limit",       type=int, default=180_000,
                        help="Max triples for expansion (default 180000)")
    args = parser.parse_args()

    start = args.only_step or args.from_step
    end   = (args.only_step or 4) + 1

    t_pipeline = time.time()

    for step in range(start, end):
        if step == 1:
            run_step(1, "step1_build_kb")
        elif step == 2:
            if args.skip_align:
                print("\n[Step 2 skipped — using existing alignment.ttl]")
            else:
                run_step(2, "step2_align")
        elif step == 3:
            run_step(3, "step3_expand", fast=args.fast_expand, limit=args.limit)
        elif step == 4:
            run_step(4, "step4_report")

    total = time.time() - t_pipeline
    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {total/60:.1f} min")
    print(f"  Outputs in: {HERE / 'data'}")
    print(f"{'='*60}\n")

    # Quick summary
    data = HERE / "data"
    for fname, label in [
        ("private_kb.nt",  "Private KB"),
        ("alignment.ttl",  "Alignment"),
        ("expanded_kb.nt", "Expanded KB"),
        ("TD4_REPORT.md",  "Report"),
    ]:
        p = data / fname
        if p.exists():
            print(f"  ✓ {label:15} {p.name}  ({p.stat().st_size:,} bytes)")
        else:
            print(f"  ✗ {label:15} {fname}  (not generated)")


if __name__ == "__main__":
    main()
