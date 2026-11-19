#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a future target.

Design (state-based, count-space):
  * The house reads a page that only changes at Exophase refreshes, so the
    reads are first collapsed into distinct page states; duplicate reads
    carry no information and must not be fitted as if they did.
  * Earners are integers in disguise (rate = round2(100 * E / P)), so the
    counts E_i and the pool P are reconstructed and forecast separately,
    with saturating growth, then blended with a direct logit-rate trend.
  * The spread comes from refresh arithmetic (per-refresh step variance
    times the expected number of refreshes in the horizon) plus a
    count-aware floor, and the damping / spread calibration is chosen by
    an in-case backtest over state-to-state transitions when the case
    holds enough states -- no hand-tuned constants carried between cases.
  * A case whose history shows no refresh at all (one or two states) is
    forecast with a launch-week refresh prior instead: a stale page is
    expected to republish inside the horizon, diluting every rate.

Stdlib only, deterministic, one output row per trophy.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z_LEVELS = (-1.645, -0.674, 0.0, 0.674, 1.645)
SHRINK_K = 4.0
SHARE_CAP = {"platinum": 0.25, "gold": 0.60, "silver": 0.85, "bronze": 0.97}
EARN_SHARE = {"platinum": 0.05, "gold": 0.07, "silver": 0.09, "bronze": 0.11}
PRIOR_SIGMA_STEP = 0.45
PRIOR_CADENCE_H = 3.0
PRIOR_SIGMA_PARAM = 0.30
LAUNCH_JUMP = 6.0
DAMP_GRID = (0.35, 0.5, 0.7, 0.85, 1.0)
CAL_GRID = (0.8, 1.0, 1.25, 1.6)
DEFAULT_DAMP = 0.7


def parse_time(text):
    """ISO-8601 text -> epoch seconds (float), or None."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def clamp_rate(rate):
    return min(99.99, max(0.01, rate))


def logit_of_rate(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def rate_of_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (values here are >= 0)."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b) or None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def huber_fit(xs, ys, rounds=4):
    """Huber IRLS line fit; falls back to OLS when anything degenerates."""
    fit = ols(xs, ys)
    if fit is None:
        return None
    a, b = fit
    n = len(xs)
    for _ in range(rounds):
        res = [y - (a + b * x) for x, y in zip(xs, ys)]
        scale = sorted(abs(r) for r in res)[n // 2] * 1.4826
        if scale <= 1e-9:
            break
        cutoff = 1.345 * scale
        ws = [1.0 if abs(r) <= cutoff else cutoff / abs(r) for r in res]
        sw = sum(ws)
        if sw <= 1e-12:
            break
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        if sxx <= 1e-12:
            a, b = my, 0.0
            break
        sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
        b = sxy / sxx
        a = my - b * mx
    return (a, b)


def group_key(trophy):
    return (trophy.get("tier"), bool(trophy.get("secret")))


def shrink_slopes(raw, keys):
    """Shrink per-trophy slopes toward their (tier x secret) group mean.

    raw: list of (slope or None, n_points). Returns a list of slopes.
    """
    sums = {}
    counts = {}
    for (slope, _n), key in zip(raw, keys):
        if slope is not None:
            sums[key] = sums.get(key, 0.0) + slope
            counts[key] = counts.get(key, 0) + 1
    means = {k: sums[k] / counts[k] for k in sums}
    all_slopes = [s for (s, _n) in raw if s is not None]
    global_mean = (sum(all_slopes) / len(all_slopes)) if all_slopes else 0.0
    out = []
    for (slope, n), key in zip(raw, keys):
        mean = means.get(key, global_mean)
        if slope is None:
            out.append(mean)
        else:
            out.append((n * slope + SHRINK_K * mean) / (n + SHRINK_K))
    return out


class Case(object):
    """A parsed case with its history collapsed into page states."""

    def __init__(self, data):
        self.data = data
        self.trophies = list(data.get("trophies") or [])
        self.n = len(self.trophies)
        self.at = parse_time(data.get("at"))
        self.target = parse_time(data.get("target"))
        horizon_min = data.get("horizon_minutes")
        if self.at is not None and self.target is not None:
            self.horizon_h = (self.target - self.at) / 3600.0
        elif isinstance(horizon_min, (int, float)) and horizon_min > 0:
            self.horizon_h = float(horizon_min) / 60.0
        else:
            self.horizon_h = 24.0
        if self.horizon_h <= 0:
            self.horizon_h = 24.0
        launch = None
        game = data.get("game") or {}
        rel = game.get("release_date")
        if isinstance(rel, str):
            launch = parse_time(rel.strip() + "T00:00:00Z")
        self.launch = launch
        self.states = self._collapse()

    def hours(self, epoch):
        """Hours since launch for an epoch time (launch falls back to the
        first read when the release date is unusable)."""
        base = self.launch
        if base is None:
            base = self._first_read_time()
        if base is None or epoch is None:
            return 0.0
        return (epoch - base) / 3600.0

    def _first_read_time(self):
        for read in self.data.get("history") or []:
            t = parse_time(read.get("at"))
            if t is not None:
                return t
        return None

    def _collapse(self):
        """Walk the reads; a new state starts when (players, rates) change.

        Each state: time (epoch of its first read), players, rates tuple,
        earners list (reconstructed, non-decreasing per trophy), and for
        each trophy the index of its last state with a real rate is kept
        derivable from the earners/rates history.
        """
        states = []
        prev_key = None
        for read in self.data.get("history") or []:
            players = read.get("players_tracked")
            if not isinstance(players, (int, float)) or players <= 0:
                continue
            rates = read.get("rates")
            if not isinstance(rates, list) or len(rates) != self.n:
                continue
            t = parse_time(read.get("at"))
            if t is None:
                continue
            key = (players, tuple(rates))
            if key == prev_key:
                continue
            prev_key = key
            states.append({
                "time": t,
                "players": float(players),
                "rates": list(rates),
            })
        # Reconstruct earner counts per state, absorbing rounding by
        # holding each trophy's count non-decreasing across states.
        running = [None] * self.n
        for st in states:
            earners = []
            for i in range(self.n):
                r = st["rates"][i]
                e = None
                if isinstance(r, (int, float)):
                    e = int(math.floor(r / 100.0 * st["players"] + 0.5))
                    if running[i] is not None and e < running[i]:
                        e = running[i]
                    running[i] = e
                earners.append(e)
            st["earners"] = earners
        return states

    def anchor(self, states, i):
        """(state_index, earners, rate) of trophy i's last real observation
        inside the given state list, or None."""
        for j in range(len(states) - 1, -1, -1):
            r = states[j]["rates"][i]
            if isinstance(r, (int, float)):
                return (j, states[j]["earners"][i], float(r))
        return None


def players_forecast(case, states, t_tgt):
    """Saturating pool forecast: OLS of log P on log(1 + t), hard caps."""
    p_last = states[-1]["players"]
    if len(states) < 3:
        return p_last * flat_players_growth(case, states)
    xs = [math.log1p(max(case.hours(st["time"]), 0.0)) for st in states]
    ys = [math.log(st["players"]) for st in states]
    fit = ols(xs, ys)
    if fit is None:
        return p_last
    a, b = fit
    u_tgt = math.log1p(max(t_tgt, 0.0))
    p_hat = math.exp(a + b * u_tgt)
    growth = math.exp(b * (u_tgt - xs[-1]))
    cap = p_last * min(4.0, growth + 1.0)
    return min(max(p_hat, p_last * 0.9), cap)


def flat_players_growth(case, states):
    """Expected pool growth over the horizon when no refresh is visible:
    a launch-day snapshot is stale by construction and republishes big."""
    last = states[-1]
    age_h = 0.0
    if case.at is not None:
        age_h = max(0.0, (case.at - last["time"]) / 3600.0)
    launch_day = False
    if case.launch is not None:
        launch_day = last["time"] < case.launch + 36.0 * 3600.0
    h = case.horizon_h
    if launch_day and age_h >= 1.0:
        # The pool kept growing while the page sat stale, so the jump the
        # refresh will publish has been accumulating since the last state:
        # the saturation clock runs on staleness + horizon, not horizon
        # alone. A refresh publishes the pool as it stands, in full, so a
        # page already hours old on launch day republishes nearly the
        # whole accumulated jump inside even a short horizon.
        eff = age_h + h
        return 1.0 + (LAUNCH_JUMP - 1.0) * (1.0 - math.exp(-eff / 3.0))
    return 1.0 + 0.5 * (1.0 - math.exp(-h / 24.0))


def earners_slopes(case, states):
    """Per-trophy shrunk slopes of log earners on log(1 + t)."""
    raw = []
    for i in range(case.n):
        xs, ys = [], []
        for st in states:
            e = st["earners"][i]
            if e is not None and e > 0:
                xs.append(math.log1p(max(case.hours(st["time"]), 0.0)))
                ys.append(math.log(e))
        fit = ols(xs, ys) if len(xs) >= 2 else None
        raw.append((fit[1] if fit else None, len(xs)))
    keys = [group_key(t) for t in case.trophies]
    return shrink_slopes(raw, keys)


def trend_slopes(case, states):
    """Per-trophy shrunk Huber slopes of logit(rate) on log(1 + t)."""
    raw = []
    for i in range(case.n):
        xs, ys = [], []
        for st in states:
            r = st["rates"][i]
            if isinstance(r, (int, float)) and 0.0 < r < 100.0:
                xs.append(math.log1p(max(case.hours(st["time"]), 0.0)))
                ys.append(logit_of_rate(r))
        fit = huber_fit(xs, ys) if len(xs) >= 2 else None
        raw.append((fit[1] if fit else None, len(xs)))
    keys = [group_key(t) for t in case.trophies]
    return shrink_slopes(raw, keys)


def step_sigma(case, states):
    """RMS per-refresh logit step of rates across trophies and states."""
    if len(states) < 3:
        return PRIOR_SIGMA_STEP
    acc = 0.0
    cnt = 0
    for a, b in zip(states, states[1:]):
        for i in range(case.n):
            ra, rb = a["rates"][i], b["rates"][i]
            if (isinstance(ra, (int, float)) and isinstance(rb, (int, float))
                    and 0.05 < ra < 99.95 and 0.05 < rb < 99.95):
                d = logit_of_rate(rb) - logit_of_rate(ra)
                acc += d * d
                cnt += 1
    if cnt == 0:
        return PRIOR_SIGMA_STEP
    return math.sqrt(acc / cnt)


def mean_dwell_h(case, states):
    if len(states) < 2:
        return PRIOR_CADENCE_H
    span = (states[-1]["time"] - states[0]["time"]) / 3600.0
    if span <= 0:
        return PRIOR_CADENCE_H
    return span / (len(states) - 1)


def forecast_states(case, states, t_tgt, damp, sigma_param, cal):
    """Median logit + sigma per trophy for a state list and target hour.

    Returns a list of (median_logit, sigma) in trophy order.
    """
    n = case.n
    last_idx = len(states) - 1
    t_last = case.hours(states[-1]["time"])
    p_hat = players_forecast(case, states, t_tgt)
    e_slopes = earners_slopes(case, states)
    b_slopes = trend_slopes(case, states)
    s_step = step_sigma(case, states)
    if len(states) >= 3:
        n_ref = case.horizon_h / mean_dwell_h(case, states)
    else:
        n_ref = case.horizon_h / PRIOR_CADENCE_H
    n_ref = min(40.0, max(0.5, n_ref))
    u_tgt = math.log1p(max(t_tgt, 0.0))
    out = []
    for i in range(n):
        trophy = case.trophies[i]
        tier = trophy.get("tier")
        anch = case.anchor(states, i)
        if anch is None:
            out.append((logit_of_rate(1.0), 1.2))
            continue
        j, e_last, rate_last = anch
        t_anchor = case.hours(states[j]["time"])
        u_anchor = math.log1p(max(t_anchor, 0.0))
        stale = 1.5 if j < last_idx else 1.0
        # The all-earned trophy is one-sided: everyone tracked has it, so
        # its risk is a small slip, never a two-sided logit swing.
        if rate_last >= 99.99 and e_last is not None \
                and e_last >= states[j]["players"]:
            out.append((logit_of_rate(99.99), 0.02))
            continue
        # Forecast A: count space.
        if e_last is not None and e_last > 0:
            e_hat = e_last * math.exp(e_slopes[i] * (u_tgt - u_anchor))
        elif e_last is not None:
            e_hat = 0.0
        else:
            e_hat = rate_last / 100.0 * states[j]["players"]
        cap_share = SHARE_CAP.get(tier, 0.9)
        e_hat = min(e_hat, p_hat * cap_share, p_hat)
        if e_last is not None:
            e_hat = max(e_hat, e_last * 0.95)
        rate_a = 100.0 * e_hat / p_hat if p_hat > 0 else rate_last
        # Forecast B: direct logit-rate trend, backtest-damped.
        move = b_slopes[i] * (u_tgt - u_anchor) * damp
        move = min(3.0, max(-3.0, move))
        logit_b = logit_of_rate(rate_last) + move
        median_logit = 0.5 * logit_of_rate(max(rate_a, 0.01)) + 0.5 * logit_b
        # Sanity rail in rate space.
        if 0.0 < rate_last < 99.99:
            lo = 0.5 * min(rate_last, rate_a)
            hi = 2.0 * max(rate_last, rate_a)
            med_rate = rate_of_logit(median_logit)
            med_rate = min(max(med_rate, lo), hi)
            median_logit = logit_of_rate(med_rate)
        # Spread: refresh steps + parameter error + small-count term.
        e_for_count = e_last if e_last is not None else 1
        s_count = 1.0 / math.sqrt(max(e_for_count, 1))
        var = s_step * s_step * n_ref + sigma_param ** 2 + s_count ** 2
        sigma = math.sqrt(var) * cal
        sigma = min(2.20, max(0.15, sigma)) * stale
        if e_last is not None and e_last == 0:
            sigma = max(sigma, 0.8)
        out.append((median_logit, sigma))
    return out


def backtest_calibrate(case, states):
    """Pick (damp, cal, sigma_param) by replaying state transitions.

    Fits on states up to a cut, forecasts a state a few steps later, and
    scores the median error and interval coverage. Deterministic grids.
    """
    s = len(states)
    cuts = sorted(set([int(math.ceil(s / 2.0)), int(math.ceil(3.0 * s / 4.0))]))
    jobs = []
    for k in cuts:
        for m in (2, 4):
            if 0 < k < s and k + m < s:
                jobs.append((k, m))
    if not jobs:
        return (DEFAULT_DAMP, 1.0, PRIOR_SIGMA_PARAM)
    # Median errors per damp value.
    errors = {d: [] for d in DAMP_GRID}
    for k, m in jobs:
        sub = states[:k + 1]
        truth_state = states[k + m]
        t_tgt = case.hours(truth_state["time"])
        for d in DAMP_GRID:
            preds = forecast_states(case, sub, t_tgt, d, PRIOR_SIGMA_PARAM, 1.0)
            for i in range(case.n):
                r = truth_state["rates"][i]
                if isinstance(r, (int, float)) and 0.0 < r < 100.0:
                    errors[d].append(abs(preds[i][0] - logit_of_rate(r)))
    best_damp = DEFAULT_DAMP
    best_err = None
    for d in DAMP_GRID:
        errs = errors[d]
        if not errs:
            continue
        mean_err = sum(errs) / len(errs)
        if best_err is None or mean_err < best_err:
            best_err = mean_err
            best_damp = d
    # Coverage per cal value at the chosen damp; sigma_param from RMS error.
    sq = 0.0
    cnt = 0
    cover = {c: [0, 0] for c in CAL_GRID}
    for k, m in jobs:
        sub = states[:k + 1]
        truth_state = states[k + m]
        t_tgt = case.hours(truth_state["time"])
        preds = forecast_states(case, sub, t_tgt, best_damp,
                                PRIOR_SIGMA_PARAM, 1.0)
        for i in range(case.n):
            r = truth_state["rates"][i]
            if isinstance(r, (int, float)) and 0.0 < r < 100.0:
                med, sig = preds[i]
                err = med - logit_of_rate(r)
                sq += err * err
                cnt += 1
                for c in CAL_GRID:
                    lo = med - 1.645 * sig * c
                    hi = med + 1.645 * sig * c
                    cover[c][1] += 1
                    if lo <= logit_of_rate(r) <= hi:
                        cover[c][0] += 1
    sigma_param = PRIOR_SIGMA_PARAM
    if cnt:
        sigma_param = min(0.80, max(0.10, math.sqrt(sq / cnt)))
    best_cal = 1.0
    best_gap = None
    for c in CAL_GRID:
        inside, total = cover[c]
        if not total:
            continue
        gap = abs(inside / total - 0.90)
        if best_gap is None or gap < best_gap or (gap == best_gap and c < best_cal):
            best_gap = gap
            best_cal = c
    return (best_damp, best_cal, sigma_param)


def flat_branch(case, states):
    """Forecast when the history shows at most one page change: the page
    is stale, a refresh is expected inside the horizon, and the median
    must dilute instead of persisting."""
    last = states[-1]
    p_last = last["players"]
    g_p = flat_players_growth(case, states)
    p_hat = p_last * g_p
    s_step = PRIOR_SIGMA_STEP
    n_ref = min(40.0, max(0.5, case.horizon_h / PRIOR_CADENCE_H))
    out = []
    for i in range(case.n):
        trophy = case.trophies[i]
        tier = trophy.get("tier")
        anch = case.anchor(states, i)
        if anch is None:
            out.append((logit_of_rate(1.0), 1.2))
            continue
        _j, e_last, rate_last = anch
        if rate_last >= 99.99 and e_last is not None and e_last >= p_last:
            out.append((logit_of_rate(99.99), 0.02))
            continue
        if e_last is None or e_last == 0:
            out.append((zero_earner_median(case, tier), 0.9))
            continue
        g_e = 1.0 + (g_p - 1.0) * EARN_SHARE.get(tier, 0.09)
        med_rate = 100.0 * (e_last * g_e) / p_hat
        frozen = 100.0 * e_last / p_hat
        med_rate = max(med_rate, frozen * 0.8)
        med_rate = min(med_rate, rate_last * 1.05)
        s_count = 1.0 / math.sqrt(max(e_last, 1))
        var = s_step * s_step * n_ref + PRIOR_SIGMA_PARAM ** 2 + s_count ** 2
        sigma = max(0.30, min(2.20, math.sqrt(var)))
        out.append((logit_of_rate(max(med_rate, 0.01)), sigma))
    return out


def zero_earner_median(case, tier):
    """Median logit for a trophy nobody tracked has earned yet: half the
    smallest positive rate among its tier peers, else a 0.05% floor."""
    best = None
    last = case.states[-1] if case.states else None
    if last is not None:
        for i, t in enumerate(case.trophies):
            if t.get("tier") == tier:
                r = last["rates"][i]
                if isinstance(r, (int, float)) and r > 0:
                    if best is None or r < best:
                        best = r
    if best is None:
        return logit_of_rate(0.05)
    return logit_of_rate(max(best * 0.5, 0.01))


def fallback_rows(case):
    """Widened persistence, used whenever anything about a case is off."""
    days = max(case.horizon_h / 24.0, 0.01)
    sigma = max(1.0 * math.sqrt(days), 0.35)
    rows = []
    last = case.states[-1] if case.states else None
    for i in range(case.n):
        rate = None
        if last is not None:
            r = last["rates"][i]
            if isinstance(r, (int, float)):
                rate = float(r)
        if rate is None or rate <= 0:
            rows.append((logit_of_rate(0.05), max(sigma, 0.8)))
        elif rate >= 99.99:
            rows.append((logit_of_rate(99.99), 0.02))
        else:
            rows.append((logit_of_rate(rate), sigma))
    return rows


def forecast_case(case):
    """(median_logit, sigma) per trophy for one case."""
    states = case.states
    if not states or case.n == 0:
        return fallback_rows(case)
    if case.target is not None:
        t_tgt = case.hours(case.target)
    else:
        t_tgt = case.hours(states[-1]["time"]) + case.horizon_h
    if len(states) <= 2:
        return flat_branch(case, states)
    damp, cal, sigma_param = DEFAULT_DAMP, 1.0, PRIOR_SIGMA_PARAM
    if len(states) >= 8:
        damp, cal, sigma_param = backtest_calibrate(case, states)
    preds = forecast_states(case, states, t_tgt, damp, sigma_param, cal)
    # Zero-earner trophies get their peer-based median and a wide spread.
    fixed = []
    for i in range(case.n):
        anch = case.anchor(states, i)
        if anch is not None and anch[1] is not None and anch[1] == 0:
            tier = case.trophies[i].get("tier")
            fixed.append((zero_earner_median(case, tier),
                          max(preds[i][1], 0.8)))
        else:
            fixed.append(preds[i])
    return fixed


def quantiles(median_logit, sigma):
    qs = []
    for z in Z_LEVELS:
        q = rate_of_logit(median_logit + z * sigma)
        q = min(100.0, max(0.0, q))
        qs.append(round4(q))
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def run_case(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    case = Case(data)
    try:
        preds = forecast_case(case)
    except Exception:
        preds = fallback_rows(case)
    rows = []
    for trophy, (med, sig) in zip(case.trophies, preds):
        rows.append({"id": trophy.get("id"), "q": quantiles(med, sig)})
    return data.get("id"), {"trophies": rows}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id, payload = run_case(os.path.join(in_dir, name))
        stem = case_id if isinstance(case_id, str) else name[:-3]
        out_path = os.path.join(out_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
