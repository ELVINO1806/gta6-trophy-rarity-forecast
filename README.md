# GTA VI Trophy Rarity Forecast

[![Competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ELVINO1806/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ELVINO1806/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for each of GTA VI's 55 PS5 trophies, the percent of Exophase-tracked players who
will have earned it one week after launch (2026-11-26T00:00:00Z), as five quantiles
(5/25/50/75/95%), scored by pinball loss on the logit scale.

## The model

`main.py` (Python 3.13, standard library only) reads each case file in the input directory
and writes one forecast per case. For every trophy it:

1. **Anchors** at the rate in the last read that lists the trophy.
2. **Trend move** — a robust (Huber IRLS) linear fit of logit(rate) against
   ln(1 + hours since the first read), with the slope shrunk toward the mean slope of the
   trophy's tier × secret group so sparse trophies borrow the group's trend.
3. **Decomposition move** — earners ≈ rate × players_tracked are extrapolated separately
   from the player pool (players growth capped; earners growth capped at 6× players
   growth), giving the rate implied by "earners keep arriving while the pool floods in".
4. **Blends** the two moves (0.8 / 0.2), then **damps** by W/(W+H) (observed window vs
   horizon), scales by 0.75, and caps the move at min(5, 0.75 × observed logit range + 1.25).
   Damping is the point: the forecast distance is several times the observed window, and an
   undamped extrapolation loses to plain persistence.
5. **Spread** — a sigma combining a random-walk term, the trend fit's residual sigma, a
   model-uncertainty term that grows with the move, and a refresh-jump allowance (Exophase
   republishes in jumps), scaled by 0.5. Quantiles are logistic(anchor + move + z·sigma).

Trophies with fewer than 3 usable reads fall back to widened persistence. The program is
fully deterministic and defensive: empty histories, all-null rates, missing player counts or
horizons, and unexpected trophy counts all still yield a valid forecast file.

## Local rig (`tools/`)

- `tools/score.py` — the contest's exact pinball-on-logit scorer, a full output-contract
  validator, and an independent reimplementation of the house persistence baseline.
- `tools/verify_persistence.py` — proves the reimplementation reproduces the starter's
  outputs on the practice cases row-for-row, plus the rules' documented worked example.
- `tools/simulate.py` — generates plausible worlds (logistic pool growth, hourly cohorts,
  Exophase-style refresh freezes) in growth and dilution regimes and backtests the model
  against persistence at the real geometry (48 h history cut, 6,510-minute horizon).
- `tools/sweep.py` — parameter calibration on identical worlds; the shipped constants were
  confirmed on a held-out seed set.
- `tools/edge_cases.py` — malformed/extreme case suite, including 200 trophies × 300 reads
  through the real program (byte-identical across runs, well under the 8 s limit).

Backtest on simulated worlds (mean pinball-on-logit loss, 12 seeds per regime, held-out set):
growth — model 0.287 vs persistence 0.670; dilution — model 0.270 vs persistence 0.586.
