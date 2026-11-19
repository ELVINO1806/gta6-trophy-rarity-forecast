"""Edge-case suite for the entry (dev only).

Every malformed or extreme case from the model's defensive list must still produce a
contract-valid forecast, and the big case (200 trophies x 300 reads) must run through the
real program twice with byte-identical output, well under the 8 s per-case limit.

Run: python3 tools/edge_cases.py
"""

import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import validate_output  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def load_entry():
    spec = importlib.util.spec_from_file_location("entry_main", REPO / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trophies(n, tiers=("bronze", "silver", "gold", "platinum")):
    return [
        {
            "id": i + 1,
            "name": f"T{i + 1}",
            "description": "edge",
            "tier": tiers[i % len(tiers)],
            "points": 15,
            "secret": i % 2 == 0,
        }
        for i in range(n)
    ]


def read(at, players, rates):
    r = {"at": at, "rates": rates}
    if players is not ...:
        r["players_tracked"] = players
    return r


BASE_CASES = {}


def build_cases():
    t55 = trophies(55)
    flat = [1.1] * 55
    cases = {}

    cases["empty history"] = {"trophies": t55, "history": [], "horizon_minutes": 6510}
    cases["single read"] = {
        "trophies": t55,
        "history": [read("2026-11-19T00:00:00Z", 638, flat)],
        "horizon_minutes": 6510,
    }
    cases["all-null rates"] = {
        "trophies": t55,
        "history": [read("2026-11-19T00:00:00Z", 638, [None] * 55),
                    read("2026-11-19T01:00:00Z", 900, [None] * 55)],
        "horizon_minutes": 6510,
    }
    cases["rates at 0 and 100"] = {
        "trophies": t55,
        "history": [
            read("2026-11-19T00:00:00Z", 638, [0, 100] + [50.0] * 53),
            read("2026-11-19T06:00:00Z", 5000, [0.0, 100.0] + [49.5] * 53),
            read("2026-11-19T12:00:00Z", 9000, [0.01, 100.0] + [49.0] * 53),
        ],
        "horizon_minutes": 6510,
    }
    cases["no players_tracked"] = {
        "trophies": t55,
        "history": [
            read("2026-11-19T00:00:00Z", ..., [1.0 + 0.1 * j for j in range(55)]),
            read("2026-11-19T05:00:00Z", ..., [1.2 + 0.1 * j for j in range(55)]),
            read("2026-11-19T10:00:00Z", ..., [1.4 + 0.1 * j for j in range(55)]),
        ],
        "horizon_minutes": 6510,
    }
    cases["players null"] = {
        "trophies": t55,
        "history": [
            read("2026-11-19T00:00:00Z", None, flat),
            read("2026-11-19T05:00:00Z", None, [1.3] * 55),
            read("2026-11-19T10:00:00Z", None, [1.5] * 55),
        ],
        "horizon_minutes": 6510,
    }
    cases["no horizon, no target"] = {
        "trophies": t55,
        "history": [
            read("2026-11-19T00:00:00Z", 638, flat),
            read("2026-11-19T05:00:00Z", 3000, [1.4] * 55),
            read("2026-11-19T10:00:00Z", 8000, [1.7] * 55),
        ],
    }
    cases["target only, no horizon"] = {
        "trophies": t55,
        "at": "2026-11-21T00:00:00Z",
        "target": "2026-11-26T00:00:00Z",
        "history": [
            read("2026-11-20T00:00:00Z", 638, flat),
            read("2026-11-20T12:00:00Z", 3000, [1.4] * 55),
            read("2026-11-21T00:00:00Z", 8000, [1.7] * 55),
        ],
    }
    cases["one trophy"] = {
        "trophies": trophies(1),
        "history": [read("2026-11-19T00:00:00Z", 638, [2.5]),
                    read("2026-11-19T03:00:00Z", 2000, [2.1]),
                    read("2026-11-19T06:00:00Z", 6000, [1.8])],
        "horizon_minutes": 6510,
    }
    cases["zero trophies"] = {"trophies": [], "history": [], "horizon_minutes": 6510}
    cases["short rates arrays"] = {
        "trophies": t55,
        "history": [
            read("2026-11-19T00:00:00Z", 638, [1.0, 2.0]),
            read("2026-11-19T05:00:00Z", 3000, []),
            read("2026-11-19T10:00:00Z", 8000, flat),
        ],
        "horizon_minutes": 6510,
    }
    cases["reads missing rates key"] = {
        "trophies": t55,
        "history": [
            {"at": "2026-11-19T00:00:00Z", "players_tracked": 638},
            read("2026-11-19T05:00:00Z", 3000, [1.4] * 55),
            read("2026-11-19T10:00:00Z", 8000, [1.7] * 55),
        ],
        "horizon_minutes": 6510,
    }
    cases["unparseable timestamps"] = {
        "trophies": t55,
        "history": [
            read("not-a-time", 638, flat),
            read("", 3000, [1.4] * 55),
            read("2026-13-45T99:99:99Z", 8000, [1.7] * 55),
        ],
        "horizon_minutes": 6510,
    }
    cases["junk entries in history"] = {
        "trophies": t55,
        "history": [
            None,
            "garbage",
            read("2026-11-19T00:00:00Z", 638, flat),
            read("2026-11-19T05:00:00Z", 3000, [1.4] * 55),
            read("2026-11-19T10:00:00Z", 8000, [1.7] * 55),
        ],
        "horizon_minutes": 6510,
    }
    cases["declining rates, shrinking players"] = {
        "trophies": t55,
        "history": [
            read("2026-11-19T00:00:00Z", 9000, [5.0] * 55),
            read("2026-11-19T05:00:00Z", 4000, [4.0] * 55),
            read("2026-11-19T10:00:00Z", 1000, [3.0] * 55),
        ],
        "horizon_minutes": 6510,
    }
    return cases


def big_case():
    rng = random.Random(42)
    t200 = trophies(200)
    history = []
    players = 638.0
    base_rates = [rng.uniform(0.05, 60.0) for _ in range(200)]
    for j in range(300):
        players *= 1.01
        rates = []
        for i in range(200):
            drift = 1.0 + 0.002 * j * (0.5 + (i % 7) / 7.0)
            v = base_rates[i] * drift
            rates.append(None if (i + j) % 37 == 0 else round(min(99.9, v), 2))
        history.append(
            {
                "at": f"2026-11-{(19 + (j * 30) // 1440):02d}T{(j * 30 // 60) % 24:02d}:{(j * 30) % 60:02d}:00Z",
                "players_tracked": int(players),
                "trophies_earned": 1000 + j,
                "rates": rates,
            }
        )
    return {"trophies": t200, "history": history, "horizon_minutes": 6510}


def main() -> int:
    entry = load_entry()
    failures = 0
    for name, case in build_cases().items():
        try:
            forecast = entry.forecast_case(case)
            raw = (json.dumps(forecast) + "\n").encode()
            problems = validate_output(raw, case)
        except Exception as exc:  # noqa: BLE001
            problems = [f"exception: {exc!r}"]
        status = "ok" if not problems else f"FAIL {problems[:2]}"
        if problems:
            failures += 1
        print(f"{name:38s} {status}")

    # Big case through the real program, twice: validity, byte-identity, runtime.
    case = big_case()
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        in_dir.mkdir()
        (in_dir / "big.in").write_text(json.dumps(case), encoding="utf-8")
        outputs = []
        for run in (1, 2):
            out_dir = Path(tmp) / f"out{run}"
            env = dict(os.environ, MAKEFUN_INPUT_DIR=str(in_dir), MAKEFUN_OUTPUT_DIR=str(out_dir))
            start = time.perf_counter()
            subprocess.run(
                [sys.executable, str(REPO / "main.py")], env=env, check=True,
                capture_output=True, text=True,
            )
            elapsed = time.perf_counter() - start
            outputs.append((out_dir / "big.out").read_bytes())
            print(f"big case run {run}: {elapsed:.2f}s (limit 8s)")
            if elapsed >= 8.0:
                failures += 1
        same = outputs[0] == outputs[1]
        problems = validate_output(outputs[0], case)
        print(f"big case (200 trophies x 300 reads): byte-identical={same}, problems={problems}")
        if not same or problems:
            failures += 1

    print("ALL OK" if failures == 0 else f"{failures} FAILURES")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
