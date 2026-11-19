"""World simulator + backtest for gta6-trophy-rarity-forecast-01 (dev only).

Generates plausible Exophase worlds and scores the entry's model against the house
persistence baseline with tools/score.py's exact pinball-on-logit loss.

World model:
  * The tracked pool grows logistically from 638 players toward a ceiling K
    (growth regime: K in 30k..60k; dilution regime: K in 80k..140k).
  * Players arrive in hourly cohorts. A cohort of age x has earned trophy i with
    probability cap_i * (1 - exp(-x / tau_i)); caps/taus are tier-based with per-trophy
    jitter. In the dilution regime cohorts earn more slowly (taus x 2.2, caps x 0.85):
    the flood of new players drags every rate down even as earners accumulate.
  * Exophase republishes in jumps: recorded reads freeze between refreshes that land
    every 5..11 hours. Reads are taken every 30 minutes; each read shows the values of
    the last refresh at or before it, rates rounded to 2 decimals like the real page.
  * History is cut at 48 h; the horizon is 6,510 minutes; the answer is the first
    recorded read inside the target hour.

Usage: python3 tools/simulate.py [--seeds 1000-1011] [--regimes growth,dilution]
Prints per-regime mean losses for the model and for persistence.
"""

import argparse
import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import persistence_forecast, pinball_logit_loss, validate_output  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
CUT_H = 48.0
HORIZON_MIN = 6510
TARGET_H = CUT_H + HORIZON_MIN / 60.0
END_H = TARGET_H + 2.0

TIER_CAPS = {  # (low, high) eventual earn fraction for a cohort
    "platinum": (0.03, 0.06),
    "gold": (0.12, 0.30),
    "silver": (0.25, 0.50),
    "bronze": (0.40, 0.85),
}
TIER_TAUS = {  # (low, high) earning timescale in hours
    "platinum": (90.0, 160.0),
    "gold": (50.0, 120.0),
    "silver": (25.0, 80.0),
    "bronze": (4.0, 60.0),
}
TIER_MIX = ["platinum"] + ["gold"] * 2 + ["silver"] * 6 + ["bronze"] * 46


def load_entry():
    spec = importlib.util.spec_from_file_location("entry_main", REPO / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iso(hours: float) -> str:
    return (BASE + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_world(rng: random.Random, regime: str):
    if regime == "growth":
        ceiling = rng.uniform(30_000, 60_000)
        tau_mult, cap_mult = 1.0, 1.0
    else:  # dilution
        ceiling = rng.uniform(80_000, 140_000)
        tau_mult, cap_mult = 2.2, 0.85
    growth_r = rng.uniform(0.030, 0.050)

    def pool(t: float) -> float:
        return ceiling / (1.0 + (ceiling / 638.0 - 1.0) * math.exp(-growth_r * t))

    trophies = []
    caps, taus = [], []
    for idx, tier in enumerate(TIER_MIX):
        trophies.append(
            {
                "id": idx + 1,
                "name": f"Trophy {idx + 1}",
                "description": f"Simulated {tier} trophy.",
                "tier": tier,
                "points": {"platinum": 300, "gold": 90, "silver": 30, "bronze": 15}[tier],
                "secret": tier in ("gold", "bronze") and idx % 3 != 0,
            }
        )
        lo, hi = TIER_CAPS[tier]
        caps.append(min(0.98, rng.uniform(lo, hi) * cap_mult))
        tlo, thi = TIER_TAUS[tier]
        taus.append(rng.uniform(tlo, thi) * tau_mult)

    # Hourly cohorts: arrivals during [h, h+1).
    cohort_size = [pool(h + 1.0) - pool(h) for h in range(int(END_H) + 1)]

    def true_state(t: float):
        """(players, [earn rates in %]) in the live world at time t."""
        players = pool(t)
        earners = [0.0] * len(trophies)
        for a, size in enumerate(cohort_size):
            if a > t:
                break
            age = t - a
            for i in range(len(trophies)):
                earners[i] += size * caps[i] * (1.0 - math.exp(-age / taus[i]))
        return players, [100.0 * e / players for e in earners], earners

    # Refresh schedule.
    refreshes = [0.0]
    while refreshes[-1] < END_H:
        refreshes.append(refreshes[-1] + rng.uniform(5.0, 11.0))
    frozen = {rt: true_state(rt) for rt in refreshes}

    def recorded(t: float):
        rt = max(r for r in refreshes if r <= t)
        players, rates, earners = frozen[rt]
        return players, rates, earners

    history = []
    t = 0.0
    while t <= CUT_H + 1e-9:
        players, rates, earners = recorded(t)
        history.append(
            {
                "at": iso(t),
                "players_tracked": int(round(players)),
                "trophies_earned": int(round(sum(earners))),
                "rates": [round(r, 2) for r in rates],
            }
        )
        t += 0.5

    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "sim",
        "at": iso(CUT_H),
        "horizon_minutes": HORIZON_MIN,
        "target": iso(TARGET_H),
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5"},
        "trophies": trophies,
        "history": history,
        "notes": "simulated",
    }
    _players, ans_rates, _e = recorded(TARGET_H)
    truth = {trophies[i]["id"]: ans_rates[i] for i in range(len(trophies))}
    return case, truth


def run(seeds, regimes):
    entry = load_entry()
    for regime in regimes:
        model_losses, pers_losses = [], []
        for seed in seeds:
            rng = random.Random(seed * 7919 + (0 if regime == "growth" else 1))
            case, truth = make_world(rng, regime)
            forecast = entry.forecast_case(case)
            raw = (json.dumps(forecast) + "\n").encode()
            problems = validate_output(raw, case)
            if problems:
                print(f"  !! {regime} seed {seed}: invalid model output: {problems[:2]}")
                continue
            model_losses.append(pinball_logit_loss(forecast, truth))
            pers_losses.append(pinball_logit_loss(persistence_forecast(case), truth))
        if model_losses:
            mm = sum(model_losses) / len(model_losses)
            pm = sum(pers_losses) / len(pers_losses)
            print(
                f"{regime:9s} seeds={len(model_losses):2d}  model={mm:.4f}  "
                f"persistence={pm:.4f}  (model {'BEATS' if mm < pm else 'LOSES TO'} persistence)"
            )


def parse_seeds(text: str):
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",") if s]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000-1011")
    ap.add_argument("--regimes", default="growth,dilution")
    args = ap.parse_args()
    run(parse_seeds(args.seeds), [r.strip() for r in args.regimes.split(",") if r.strip()])
