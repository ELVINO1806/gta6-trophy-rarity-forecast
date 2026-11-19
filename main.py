"""gta6-trophy-rarity-forecast-01 entry: a damped, shrunk trend forecaster (Python 3.13, stdlib only).

For each trophy the program anchors at the rate in the last read that lists it, then moves that
anchor on the logit scale by a blend of two candidate moves, heavily damped:

  * Move A (trend): a robust (Huber IRLS) linear fit of logit(rate) against ln(1 + hours since
    the first read). The slope is shrunk toward the mean slope of the trophy's tier x secret
    group, with shrinkage weight n/(n+K) so trophies with few reads borrow the group's trend.
    The move is the shrunk slope times the ln(1 + t) distance from the anchor read to the target.
  * Move B (decomposition): earners = rate/100 * players_tracked. Log-linear trends of earners
    and of players are extrapolated to the target; players growth is capped, and earners growth
    is capped at 6x the players growth factor, so the implied rate cannot run away on a
    denominator story alone. The move is logit(implied rate) - logit(anchor rate).

The blended move (0.8 A + 0.2 B) is damped by W/(W+H) (W = the trophy's observed window in
hours, H = the horizon in hours), scaled by MOVE_SHRINK, and capped at
min(5, 0.75 * observed logit range + 1.25) in absolute value. Damping is the point: at the final
horizon the forecast distance is several times the observed window, and an undamped trend
extrapolation loses to plain persistence.

The spread sigma combines, in quadrature, a random-walk term 0.5*sqrt(horizon days), the trend
fit's residual sigma, a model-uncertainty term that grows with the size of the move, and a
refresh-jump allowance (Exophase republishes in jumps, so a rate can step between reads); the
total is scaled by SIGMA_SCALE. Quantiles are logistic(anchor_logit + move + z_tau * sigma).

Trophies with fewer than 3 usable reads fall back to widened persistence (anchor median, a
spread 1.5x the house one). Every step is defensive: empty history, a single read, all-null
rates, rates at 0 or 100, missing players/horizon/target, or an unexpected trophy count all
still produce a valid forecast file. The program is fully deterministic: no randomness, no
wall-clock reads, fixed iteration orders — two runs write identical bytes.

The grader runs `python3 -P -s /bot/main.py`; MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in
for /input and /output outside the container. Each <id>.in in the input directory produces
<id>.out: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

W_TREND = 0.8           # blend weight of the trend move
W_DECOMP = 0.2          # blend weight of the earners/players decomposition move
MOVE_SHRINK = 0.75      # global damping on the blended move
SIGMA_SCALE = 0.5       # global scale on the combined spread
SHRINK_K = 6.0          # slope shrinkage: weight n/(n+K) on the trophy's own slope
SPARSE_SPREAD_MULT = 1.5  # widened-persistence spread multiplier for < 3 usable reads
EARNERS_GROWTH_CAP = 6.0  # earners growth factor <= this x players growth factor
PLAYERS_GROWTH_CAP = 250.0  # players growth factor cap over the horizon


# ---------- small numeric helpers ----------

def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit_pct(x: float) -> float:
    p = clamp_pct(x)
    return math.log(p / (100.0 - p))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as int (JSON writes 100, not 100.0)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    if value == 0:
        value = 0.0  # avoid -0.0
    return int(value) if float(value).is_integer() else value


def parse_time(value):
    """Parse an ISO-8601 timestamp to epoch seconds; None when unparseable."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def robust_line(xs, ys):
    """Robust linear fit y ~ a + b*x via least squares + two Huber IRLS passes.

    Returns (slope, intercept, residual_sigma). Caller guarantees len(xs) >= 2 and
    not all xs equal.
    """
    n = len(xs)
    weights = [1.0] * n
    slope = intercept = 0.0
    resid_sigma = 0.0
    for _ in range(3):
        sw = sum(weights)
        mx = sum(w * x for w, x in zip(weights, xs)) / sw
        my = sum(w * y for w, y in zip(weights, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(weights, xs))
        if sxx <= 0:
            slope, intercept = 0.0, my
            break
        sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(weights, xs, ys))
        slope = sxy / sxx
        intercept = my - slope * mx
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        resid_sigma = math.sqrt(sum(r * r for r in residuals) / n)
        scale = 1.345 * resid_sigma
        if scale <= 1e-12:
            break
        weights = [1.0 if abs(r) <= scale else scale / abs(r) for r in residuals]
    return slope, intercept, resid_sigma


def plain_line(xs, ys):
    """Ordinary least squares y ~ a + b*x. Returns (slope, intercept); slope 0 if degenerate."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


# ---------- per-case preparation ----------

class CaseData:
    """Parsed, defensive view of one case file."""

    def __init__(self, case: dict):
        self.trophies = case.get("trophies") or []
        history = case.get("history") or []
        self.reads = [r for r in history if isinstance(r, dict)]

        # Read times in hours since the first read; fall back to 30-minute steps.
        times = [parse_time(r.get("at")) for r in self.reads]
        if self.reads and all(t is not None for t in times):
            base = times[0]
            self.t_hours = [(t - base) / 3600.0 for t in times]
        else:
            self.t_hours = [0.5 * j for j in range(len(self.reads))]

        # Horizon in hours: prefer horizon_minutes, then target - at, then target - last read.
        horizon = None
        hm = case.get("horizon_minutes")
        if isinstance(hm, (int, float)) and hm > 0:
            horizon = float(hm) / 60.0
        if horizon is None:
            target_t = parse_time(case.get("target"))
            at_t = parse_time(case.get("at"))
            if target_t is not None and at_t is not None and target_t > at_t:
                horizon = (target_t - at_t) / 3600.0
            elif target_t is not None and times and times[-1] is not None:
                horizon = max(0.0, (target_t - times[-1]) / 3600.0)
        self.horizon_hours = horizon if horizon is not None else 24.0
        self.horizon_days = self.horizon_hours / 24.0

        self.players = []
        for r in self.reads:
            p = r.get("players_tracked")
            self.players.append(float(p) if isinstance(p, (int, float)) and p > 0 else None)

    def usable(self, idx: int):
        """(t_hours, rate) pairs for trophy idx over reads that list a rate for it."""
        out = []
        for j, r in enumerate(self.reads):
            rates = r.get("rates")
            if isinstance(rates, list) and idx < len(rates):
                v = rates[idx]
                if isinstance(v, (int, float)) and math.isfinite(v):
                    out.append((self.t_hours[j], float(v)))
        return out


# ---------- the model ----------

def forecast_case(case: dict) -> dict:
    data = CaseData(case)
    n = len(data.trophies)
    horizon_h = data.horizon_hours
    horizon_d = data.horizon_days

    # Pass 1: anchors, usable reads, trend fits.
    anchors = []        # (t_anchor, rate_anchor) or None
    spans = []          # observed window hours per trophy (usable reads)
    logit_ranges = []
    fits = []           # (slope, resid_sigma) or None
    for i in range(n):
        usable = data.usable(i)
        if not usable:
            anchors.append(None)
            spans.append(0.0)
            logit_ranges.append(0.0)
            fits.append(None)
            continue
        anchors.append(usable[-1])
        spans.append(usable[-1][0] - usable[0][0])
        ys = [logit_pct(rate) for _, rate in usable]
        logit_ranges.append(max(ys) - min(ys))
        if len(usable) >= 3 and usable[-1][0] > usable[0][0]:
            xs = [math.log1p(t) for t, _ in usable]
            if max(xs) - min(xs) > 1e-9:
                slope, _intercept, resid = robust_line(xs, ys)
                fits.append((slope, resid))
            else:
                fits.append(None)
        else:
            fits.append(None)

    # Group mean slopes by (tier, secret), weighted by usable-read count; global mean too.
    group_sum = {}
    group_w = {}
    glob_sum = 0.0
    glob_w = 0.0
    for i in range(n):
        if fits[i] is None:
            continue
        trophy = data.trophies[i] if isinstance(data.trophies[i], dict) else {}
        key = (trophy.get("tier"), bool(trophy.get("secret")))
        w = float(len(data.usable(i)))
        group_sum[key] = group_sum.get(key, 0.0) + fits[i][0] * w
        group_w[key] = group_w.get(key, 0.0) + w
        glob_sum += fits[i][0] * w
        glob_w += w
    group_mean = {k: group_sum[k] / group_w[k] for k in group_sum if group_w[k] > 0}
    global_mean = glob_sum / glob_w if glob_w > 0 else 0.0

    # Cross-trophy fallback anchor rate for trophies with no usable read at all.
    anchor_rates = sorted(a[1] for a in anchors if a is not None)
    if anchor_rates:
        m = len(anchor_rates)
        fallback_rate = anchor_rates[m // 2] if m % 2 == 1 else 0.5 * (
            anchor_rates[m // 2 - 1] + anchor_rates[m // 2]
        )
    else:
        fallback_rate = 1.0

    rw = 0.5 * math.sqrt(horizon_d)          # random-walk spread term
    jump = 0.30 * math.sqrt(horizon_d)       # refresh-jump allowance

    rows = []
    for i in range(n):
        trophy = data.trophies[i] if isinstance(data.trophies[i], dict) else {}
        tid = trophy.get("id", i + 1)
        usable = data.usable(i)
        anchor = anchors[i]

        if anchor is None:
            median_logit = logit_pct(fallback_rate)
            sigma = SIGMA_SCALE * (rw + 2.0)
            rows.append(_row(tid, median_logit, sigma))
            continue

        t_a, rate_a = anchor
        anchor_logit = logit_pct(rate_a)

        if len(usable) < 3:
            # Widened persistence: house spread, widened, no move.
            sigma = SPARSE_SPREAD_MULT * rw + 0.25
            rows.append(_row(tid, anchor_logit, sigma))
            continue

        # --- Move A: shrunk trend ---
        move_a = None
        resid_sigma = 0.0
        if fits[i] is not None:
            slope, resid_sigma = fits[i]
            key = (trophy.get("tier"), bool(trophy.get("secret")))
            g = group_mean.get(key, global_mean)
            cnt = float(len(usable))
            slope_used = (cnt * slope + SHRINK_K * g) / (cnt + SHRINK_K)
            dx = math.log1p(t_a + horizon_h) - math.log1p(t_a)
            move_a = slope_used * dx

        # --- Move B: earners / players decomposition ---
        move_b = None
        pairs = []
        for j in range(len(data.reads)):
            rate_j = _rate_at(data, j, i)
            players_j = data.players[j]
            if rate_j is not None and rate_j > 0 and players_j is not None:
                pairs.append((data.t_hours[j], rate_j, players_j))
        if len(pairs) >= 2 and pairs[-1][0] > pairs[0][0]:
            ts = [p_[0] for p_ in pairs]
            earners = [p_[1] / 100.0 * p_[2] for p_ in pairs]
            players = [p_[2] for p_ in pairs]
            if earners[-1] > 0 and players[-1] > 0:
                ge, _ = plain_line(ts, [math.log(e) for e in earners])
                gp, _ = plain_line(ts, [math.log(p) for p in players])
                span_b = (t_a + horizon_h) - pairs[-1][0]  # last paired read -> target
                players_factor = math.exp(max(-30.0, min(30.0, gp * span_b)))
                players_factor = min(players_factor, PLAYERS_GROWTH_CAP)
                players_target = players[-1] * players_factor
                earners_factor = math.exp(max(-30.0, min(30.0, ge * span_b)))
                earners_factor = min(earners_factor, EARNERS_GROWTH_CAP * players_factor)
                earners_target = earners[-1] * earners_factor
                if players_target > 0:
                    rate_b = 100.0 * earners_target / players_target
                    move_b = logit_pct(rate_b) - anchor_logit

        # --- Blend, damp, shrink, cap ---
        if move_a is not None and move_b is not None:
            move = W_TREND * move_a + W_DECOMP * move_b
        elif move_a is not None:
            move = move_a
        elif move_b is not None:
            move = move_b
        else:
            move = 0.0
        w_span = spans[i]
        damp = w_span / (w_span + horizon_h) if (w_span + horizon_h) > 0 else 0.0
        move *= damp * MOVE_SHRINK
        cap = min(5.0, 0.75 * logit_ranges[i] + 1.25)
        move = max(-cap, min(cap, move))
        median_logit = anchor_logit + move

        # --- Spread ---
        resid = min(2.0, max(0.05, resid_sigma)) if fits[i] is not None else 0.35
        unc = 0.30 + 0.45 * abs(move)
        sigma = SIGMA_SCALE * math.sqrt(rw * rw + resid * resid + unc * unc + jump * jump)
        rows.append(_row(tid, median_logit, sigma))

    return {"trophies": rows}


def _rate_at(data: CaseData, j: int, idx: int):
    rates = data.reads[j].get("rates")
    if isinstance(rates, list) and idx < len(rates):
        v = rates[idx]
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _row(tid, median_logit: float, sigma: float) -> dict:
    qs = []
    prev = 0.01
    for z in Z:
        q = clamp_pct(logistic(median_logit + z * sigma))
        q = max(q, prev)  # keep the row non-decreasing even before rounding
        prev = q
        qs.append(round4(q))
    # Rounding can in principle break monotonicity at the 4th decimal; repair by max-so-far.
    fixed = []
    running = None
    for v in qs:
        running = v if running is None else max(running, v)
        fixed.append(running)
    return {"id": tid, "q": fixed}


def fallback_forecast(case: dict) -> dict:
    """Last-resort forecast if the model path ever fails on a case: persistence, wide spread."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    last = history[-1] if history and isinstance(history[-1], dict) else {}
    rates = last.get("rates") if isinstance(last.get("rates"), list) else []
    hm = case.get("horizon_minutes")
    days = (float(hm) / 1440.0) if isinstance(hm, (int, float)) and hm > 0 else 1.0
    spread = 1.0 * math.sqrt(days)
    rows = []
    for i, trophy in enumerate(trophies):
        tid = trophy.get("id", i + 1) if isinstance(trophy, dict) else i + 1
        rate = rates[i] if i < len(rates) and isinstance(rates[i], (int, float)) else 1.0
        rows.append(_row(tid, logit_pct(rate if rate is not None else 1.0), spread))
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.is_dir():
        return
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_forecast(case)
            except Exception:
                continue  # nothing sane can be written for an unreadable case file
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
