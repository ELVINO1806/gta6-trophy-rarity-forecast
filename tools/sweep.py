"""Parameter sweep for the entry model on simulated worlds (dev only).

Generates each world once, then re-scores the model under patched constants
(SIGMA_SCALE, W_TREND, MOVE_SHRINK) so calibration choices are compared on
identical data. Use one seed set to tune and a disjoint one to confirm.

Usage: python3 tools/sweep.py --seeds 1000-1011
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate  # noqa: E402
from score import persistence_forecast, pinball_logit_loss  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000-1011")
    args = ap.parse_args()
    seeds = simulate.parse_seeds(args.seeds) if hasattr(simulate, "parse_seeds") else None
    if seeds is None:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))

    entry = simulate.load_entry()
    worlds = []  # (regime, case, truth)
    for regime in ("growth", "dilution"):
        for seed in seeds:
            rng = random.Random(seed * 7919 + (0 if regime == "growth" else 1))
            case, truth = simulate.make_world(rng, regime)
            worlds.append((regime, case, truth))

    pers = {}
    for regime, case, truth in worlds:
        pers.setdefault(regime, []).append(
            pinball_logit_loss(persistence_forecast(case), truth)
        )
    for regime in ("growth", "dilution"):
        vals = pers[regime]
        print(f"persistence {regime:9s} mean={sum(vals) / len(vals):.4f}")

    combos = []
    for sigma_scale in (0.5, 0.65, 0.8, 1.0):
        for w_trend in (0.5, 0.65, 0.8):
            for move_shrink in (0.45, 0.6, 0.75):
                combos.append((sigma_scale, w_trend, move_shrink))
    results = []
    for sigma_scale, w_trend, move_shrink in combos:
        entry.SIGMA_SCALE = sigma_scale
        entry.W_TREND = w_trend
        entry.W_DECOMP = 1.0 - w_trend
        entry.MOVE_SHRINK = move_shrink
        per = {"growth": [], "dilution": []}
        for regime, case, truth in worlds:
            per[regime].append(pinball_logit_loss(entry.forecast_case(case), truth))
        g = sum(per["growth"]) / len(per["growth"])
        d = sum(per["dilution"]) / len(per["dilution"])
        results.append((g + d, g, d, sigma_scale, w_trend, move_shrink))
    results.sort()
    print(f"{'total':>7} {'growth':>7} {'dilution':>8}  sigma  w_trend  shrink")
    for total, g, d, ss, wt, ms in results[:10]:
        print(f"{total:7.4f} {g:7.4f} {d:8.4f}  {ss:5.2f}  {wt:6.2f}  {ms:5.2f}")
    # restore shipped constants
    entry.SIGMA_SCALE, entry.W_TREND, entry.W_DECOMP, entry.MOVE_SHRINK = 0.65, 0.65, 0.35, 0.6


if __name__ == "__main__":
    main()
