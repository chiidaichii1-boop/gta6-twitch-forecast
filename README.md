# GTA VI launch-day Twitch viewers — hour-by-hour forecast

Entry for make.fun bounty **gta6-twitch-ccu-forecast-01**: forecast how many
people watch Grand Theft Auto VI live on Twitch in each hour of its release
day (2026-11-19 UTC), as five quantiles an hour (5%, 25%, 50%, 75%, 95%),
from the first hour's print. Scored by mean pinball loss on the log scale
against the actual hourly viewer counts; a prize needs a loss strictly below
the best house entry.

## The model

`main.py` (Python 3.13, standard library only) learns the forecast at run
time from the case's own `comparables` — past big launches' full hourly
viewer curves. Everything is in natural-log space, as ratios to the case's
hour-1 print `p1`.

1. **Shape from comparables.** For each comparable launch, take its
   release-day hourly viewers and form the log-ratio
   `r(h) = ln(viewers(h) / viewers(1))` for hours 2..24.
2. **Per-hour regression.** For each hour, fit an OLS regression of `r(h)` on
   two features of the launch: `ln(p1)` (its hour-1 print) and `ln(channels)`
   (its hour-1 live channels). The fitted line predicts the case's
   log-ratio; the spread is the regression's residual standard deviation.
3. **Quantiles.** `q_tau(h) = p1 * exp(pred_h + z_tau * sigma_h)`, rounded to
   2 decimals, with `z` the standard normal quantiles at the scored levels.

Why this beats the persistence baseline (every hour stays at `p1`, spread
widening like a random walk):

- The comparables' shape captures the real launch-day dynamics persistence
  misses: a dip in hours 2–4 (only the earliest time zones are awake), a
  jump at hour 5 (US East unlocks at 05:00Z), and a steady ramp to an
  evening peak in hours 17–22 as the US comes online.
- The regression captures the strong pattern that bigger launches have
  flatter curves: the largest comparables peak at only ~1.0–1.3x their
  hour-1 print, while small launches peak at 3–11x. A launch's hour-1 print
  (and how concentrated its channels are) already says a lot about how much
  headroom its curve has.
- The spread comes from the actual dispersion of launch curves around the
  regression (residual SD ~0.2 in the small hours, ~0.5 near the peak), not
  from a fixed 0.15 random walk.

## Guardrails (for the final case — GTA VI is a bigger launch than any comparable)

- Only comparables with an hour-1 print of at least 10,000 viewers are
  used. Smaller categories have peak ratios in the hundreds and would drown
  the regression.
- The Grand Theft Auto VI category's own pre-release history, shipped as a
  comparable, is not a launch day and is excluded.
- The target's `ln(p1)` and `ln(channels)` are clamped to the comparables'
  observed range plus 0.5 log units, so an out-of-range launch cannot
  extrapolate the regression to an absurd ratio.
- Fallbacks, in order: two-feature OLS; one-feature OLS on `ln(p1)` when
  hour-1 channels are missing; the median log-ratio with an IQR-based
  spread when too few comparables have that hour; the persistence baseline
  when the case carries no usable comparable at all.

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]},
...]}` — exactly 23 rows, hours 2..24, each `q` five finite numbers > 0
and <= 1e9, non-decreasing. Deterministic (same bytes on two runs), no
network, standard library only. Runs in ~0.05 s a case, far under the
8-second cap.

## Practice-case results (pinball loss, lower is better)

Scored locally against the practice cases' true SullyGnome hourly answers
(the kit ships no reference answers, so the four launches' release-day
"Average viewers" series were re-read from SullyGnome's documented chart
API and verified to match each case's hour-1 print exactly).

| case | launch | this entry | persistence baseline |
| --- | --- | --- | --- |
| p00 | Marvel's Wolverine | 0.3084 | 0.7437 |
| p01 | Red Dead Redemption 2 | 0.2075 | 0.2868 |
| p02 | God of War Ragnarök | 0.1412 | 0.2146 |
| p03 | Marvel's Spider-Man 2 | 0.1920 | 0.3033 |
| **mean** | | **0.2123** | **0.3871** |

45% lower mean loss than the baseline, and lower on every case. Typical
miss 24–71% (baseline 42–189%); 90% coverage 0.78–1.00 (honest spread
covers about 0.9).

The design was chosen by leave-one-out cross-validation over the 28 big
comparables themselves (each launch held out and forecast from the other
27): the two-feature OLS at slope 1.0 scored 0.261 there, against 0.351
for persistence, and beat every tuned variant (a steeper slope, an
empirical spread, a third feature) on that held-out set — so no constant
was tuned to the four practice cases.

## Files

- `main.py` — the program (the entrypoint).
- `README.md` — this file.
- `LICENSE` — MIT (`SPDX-License-Identifier: MIT`).
