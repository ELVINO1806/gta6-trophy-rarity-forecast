"""Local scoring rig for gta6-trophy-rarity-forecast-01 (dev only; not run by the grader).

Three pieces, all written from the competition's published rules:

  * pinball_logit_loss(forecast, truth, levels) — the contest's exact score:
    mean over trophies x levels of 2*max(tau*d, (tau-1)*d) with
    d = logit(y) - logit(q), logit(x) = ln(p/(1-p)), p = x/100 clamped to 0.01..99.99.
  * validate_output(path_or_bytes, case) — the full output contract: one JSON object with
    exactly the key "trophies"; exactly one row per case trophy id; each row exactly
    {"id", "q"}; q = 5 finite numbers in [0, 100], non-decreasing; file <= 65,536 bytes.
  * persistence_forecast(case) — an independent reimplementation of the house persistence
    baseline (median = last read's rate, logit spread 0.5*sqrt(horizon days)), used to
    cross-check the starter and as the benchmark the real model must beat.

Usage:
    python3 tools/score.py check <case.in> <out.out>          # validate + (needs truth) -
    python3 tools/score.py score <case.in> <out.out> <truth.json>
        truth.json: {"rates": [...]} in the case's trophy order, or {"<id>": rate, ...}
"""

import json
import math
import sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
MAX_BYTES = 65536


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit_pct(x: float) -> float:
    p = clamp_pct(x)
    return math.log(p / (100.0 - p))


def logistic(logit_value: float) -> float:
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4_half_up(x: float):
    """Round to 4 decimals, half up (JavaScript Math.round semantics); whole -> int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def pinball_logit_loss(forecast: dict, truth_by_id: dict, levels=LEVELS) -> float:
    """Mean pinball loss on the logit scale over every trophy x level."""
    total = 0.0
    count = 0
    for row in forecast["trophies"]:
        y = truth_by_id[row["id"]]
        ly = logit_pct(y)
        for tau, q in zip(levels, row["q"]):
            d = ly - logit_pct(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    if count == 0:
        raise ValueError("no forecast rows to score")
    return total / count


def validate_output(raw: bytes, case: dict) -> list:
    """Return a list of contract violations (empty list = valid)."""
    problems = []
    if len(raw) > MAX_BYTES:
        problems.append(f"file is {len(raw)} bytes > {MAX_BYTES}")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - report any parse failure as a violation
        return problems + [f"not UTF-8 JSON: {exc}"]
    if not isinstance(doc, dict):
        return problems + ["top level is not an object"]
    if set(doc.keys()) != {"trophies"}:
        problems.append(f"top-level keys are {sorted(doc.keys())}, not exactly ['trophies']")
    rows = doc.get("trophies")
    if not isinstance(rows, list):
        return problems + ["'trophies' is not a list"]
    want_ids = [t["id"] for t in case["trophies"]]
    got_ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row.keys()) != {"id", "q"}:
            problems.append(f"row keys are not exactly ['id', 'q']: {row!r:.80}")
            continue
        got_ids.append(row["id"])
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            problems.append(f"id {row['id']}: q is not a list of 5")
            continue
        for v in q:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append(f"id {row['id']}: q value {v!r} is not a finite number")
            elif v < 0 or v > 100:
                problems.append(f"id {row['id']}: q value {v} outside [0, 100]")
        nums = [v for v in q if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) == 5 and any(nums[i] > nums[i + 1] for i in range(4)):
            problems.append(f"id {row['id']}: q is not non-decreasing: {q}")
    if sorted(got_ids) != sorted(want_ids):
        problems.append(f"row ids {sorted(got_ids)} != case trophy ids {sorted(want_ids)}")
    if len(got_ids) != len(set(got_ids)):
        problems.append("duplicate trophy ids in output")
    return problems


def persistence_forecast(case: dict, sigma_per_sqrt_day: float = 0.5) -> dict:
    """House persistence baseline, reimplemented from the rules text.

    Median = the trophy's rate in the last read (null -> 0 before the logit clamp);
    quantiles spread on the logit scale by z * sigma * sqrt(horizon in days).
    """
    if not case["history"]:
        raise ValueError("persistence needs at least one history read")
    last = case["history"][-1]
    spread = sigma_per_sqrt_day * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit_pct(rate if rate is not None else 0.0)
        rows.append(
            {"id": trophy["id"], "q": [round4_half_up(logistic(lg + z * spread)) for z in Z]}
        )
    return {"trophies": rows}


def _load_truth(path: str, case: dict) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if "rates" in doc:
        return {t["id"]: doc["rates"][i] for i, t in enumerate(case["trophies"])}
    return {int(k): v for k, v in doc.items()}


def main(argv) -> int:
    if len(argv) >= 2 and argv[1] == "score" and len(argv) == 5:
        case = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        raw = Path(argv[3]).read_bytes()
        problems = validate_output(raw, case)
        if problems:
            print("INVALID:", *problems, sep="\n  ")
            return 1
        forecast = json.loads(raw.decode("utf-8"))
        truth = _load_truth(argv[4], case)
        print(f"loss = {pinball_logit_loss(forecast, truth):.6f}")
        return 0
    if len(argv) >= 2 and argv[1] == "check" and len(argv) == 4:
        case = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        problems = validate_output(Path(argv[3]).read_bytes(), case)
        if problems:
            print("INVALID:", *problems, sep="\n  ")
            return 1
        print("valid")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
