"""Verify tools/score.py's persistence reimplementation against the kit starter.

  1. Run tools/starter_main.py (a pristine copy of the starter) on p00 and p01 and
     compare every output row against score.persistence_forecast — must match exactly.
  2. Check the documented worked example: a trophy last at 1.10% with a 6,510-minute
     horizon gives [0.1931, 0.5403, 1.1, 2.2264, 6.0088].
  3. Validate the starter outputs against the output contract.

Run: python3 tools/verify_persistence.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import persistence_forecast, validate_output  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
KIT_INPUT = (
    REPO.parent
    / "gta6-trophy-rarity-forecast-01"
    / "kit"
    / "gta6-trophy-rarity-forecast-01-kit"
    / "input"
)


def run_starter(case_path: Path, out_dir: Path) -> dict:
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(case_path.parent), MAKEFUN_OUTPUT_DIR=str(out_dir))
    # The starter reads every *.in in the dir; isolate the one case via a temp input dir.
    with tempfile.TemporaryDirectory() as tmp:
        single = Path(tmp) / case_path.name
        single.write_bytes(case_path.read_bytes())
        env["MAKEFUN_INPUT_DIR"] = str(single.parent)
        subprocess.run(
            [sys.executable, str(REPO / "tools" / "starter_main.py")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads((out_dir / (case_path.stem + ".out")).read_text(encoding="utf-8"))


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for name in ("p00.in", "p01.in"):
            case = json.loads((KIT_INPUT / name).read_text(encoding="utf-8"))
            starter_out = run_starter(KIT_INPUT / name, out_dir)
            mine = persistence_forecast(case)
            problems = validate_output(
                (out_dir / (Path(name).stem + ".out")).read_bytes(), case
            )
            same = starter_out == mine
            print(f"{name}: starter == reimplementation -> {same}; contract problems: {problems}")
            if not same:
                ok = False
                for a, b in zip(starter_out["trophies"], mine["trophies"]):
                    if a != b:
                        print(f"  first mismatch: starter {a} vs mine {b}")
                        break
            if problems:
                ok = False

    # Documented example: last rate 1.10%, horizon 6,510 minutes.
    case = {
        "trophies": [{"id": 4}],
        "history": [{"rates": [1.10]}],
        "horizon_minutes": 6510,
    }
    got = persistence_forecast(case)["trophies"][0]["q"]
    want = [0.1931, 0.5403, 1.1, 2.2264, 6.0088]
    print(f"documented example: got {got}, want {want}, match -> {got == want}")
    if got != want:
        ok = False

    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
