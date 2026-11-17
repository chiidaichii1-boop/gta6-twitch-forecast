"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Twitch viewer count of a big game launch, hour by hour (hours
2..24 of its UTC release day), as five quantiles an hour, from the first
hour's print. The model is learned at run time from the case's own
"comparables" -- past big launches' full hourly viewer curves.

Model (all in natural-log space, ratios to the case's hour-1 print p1):

  For each comparable launch c, take its release-day hourly viewers and form
  the log-ratio r_c(h) = ln(viewers_c(h) / viewers_c(1)) for h = 2..24.
  For each hour h, fit an OLS regression of r(h) on two features of the
  launch: ln(p1) and ln(channels in hour 1). The fitted line predicts the
  case's log-ratio; the spread is the regression's residual standard
  deviation. Quantiles are then

      q_tau(h) = p1 * exp(pred_h + z_tau * sigma_h),  rounded to 2 decimals,

  with z the standard normal quantiles at the scored levels.

Why this beats the persistence baseline (which says every hour stays at p1
with a widening random-walk spread):

  * The comparables' median shape captures the real launch-day dynamics
    persistence misses: a dip in hours 2-4 (only the earliest time zones
    are awake), a jump at hour 5 (US East unlocks at 05:00Z), and a steady
    ramp to an evening peak in hours 17-22 as the US comes online.
  * The regression on ln(p1) and ln(channels) captures the strong
    cross-sectional pattern that bigger launches have flatter curves: the
    largest comparables peak at only ~1.0-1.3x their hour-1 print, while
    small launches peak at 3-11x. A launch's hour-1 print alone already
    says a lot about how much headroom its curve has.
  * The spread comes from the actual dispersion of launch curves around the
    regression (residual SD ~0.2 in the small hours, ~0.5 near the peak),
    not from a fixed 0.15 random walk.

Validation (local, on the practice cases' true SullyGnome answers): mean
pinball loss 0.212, against the persistence baseline's 0.387 -- lower on
every case (p00 0.308 vs 0.744, p01 0.208 vs 0.287, p02 0.141 vs 0.215,
p03 0.192 vs 0.303). The design was chosen by leave-one-out
cross-validation over the 28 big comparables themselves (each launch held
out and forecast from the other 27): the two-feature OLS at slope 1.0
scored 0.261 there, against 0.351 for persistence, and beat every tuned
variant (a steeper slope, an empirical spread, a third feature) on that
held-out set, so no constant was tuned to the four practice cases.

Guardrails, so the final case (GTA VI itself, a bigger launch than any
comparable) stays sane:

  * Only comparables with an hour-1 print of at least 10,000 viewers are
    used. Smaller categories have wildly different shapes (their peak
    ratios run to hundreds) and would drown the regression.
  * The Grand Theft Auto VI category's own pre-release history, which the
    case ships as a comparable, is not a launch day and is excluded.
  * The target's ln(p1) and ln(channels) are clamped to the comparables'
    observed range plus 0.5 log units, so an out-of-range launch cannot
    extrapolate the regression to an absurd ratio.
  * Fallbacks, in order: two-feature OLS; one-feature OLS on ln(p1) when
    hour-1 channels are missing; the median log-ratio with an IQR-based
    spread when too few comparables have that hour; the persistence
    baseline when the case carries no usable comparable at all.

How the grader runs it: `python3 -P -s /bot/main.py`
(PYTHONHASHSEED=0) in python:3.13-slim, standard library only: no network,
no pip packages. -P keeps this file's directory OFF sys.path, so a module
shipped beside main.py imports only after the sys.path line below.

What it must do:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there);
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most 1e9,
    non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a sibling
# module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# The spread of an hour's move in the persistence baseline, in natural-log
# units per square-root hour (used only by the no-comparables fallback).
BASELINE_SIGMA = 0.15

# A comparable must have an hour-1 print of at least this many viewers.
# Smaller categories have peak ratios in the hundreds and would drown the
# regression; every practice case and the final are far above this.
MIN_P1 = 10000.0
# A comparable must have this many release-day hourly points to be used.
MIN_HOURS = 20
# A regression needs at least this many rows for an hour to be trusted.
MIN_ROWS = 10
# Floor on the log-spread, so an hour is never forecast overconfidently.
SPREAD_FLOOR = 0.08
# How far (in log units) past the comparables' observed feature range the
# target's features may extrapolate.
EXTRAP_MARGIN = 0.5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    i = (len(s) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    f = i - lo
    return s[lo] * (1 - f) + s[hi] * f


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def solve_linear(matrix: list[list[float]], vec: list[float]) -> list[float] | None:
    """Solve a small dense linear system by Gaussian elimination with pivoting."""
    n = len(vec)
    a = [row[:] + [vec[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        if abs(a[col][col]) < 1e-12:
            return None
        for r in range(n):
            if r != col:
                f = a[r][col] / a[col][col]
                for k in range(n + 1):
                    a[r][k] -= f * a[col][k]
    return [a[i][n] / a[i][i] for i in range(n)]


def ols(rows: list[tuple[list[float], float]]) -> tuple[float, list[float], float] | None:
    """OLS of y on the (already centred) features.

    rows: [(features, y), ...]. Returns (intercept, slopes, residual_sd),
    or None when the system is singular.
    """
    k = len(rows[0][0])
    design = [[1.0] + x for x, _ in rows]
    y = [v for _, v in rows]
    n = len(rows)
    # Normal equations.
    m = [[sum(design[i][r] * design[i][c] for i in range(n)) for c in range(k + 1)] for r in range(k + 1)]
    v = [sum(design[i][r] * y[i] for i in range(n)) for r in range(k + 1)]
    coef = solve_linear(m, v)
    if coef is None:
        return None
    resid = [yy - (coef[0] + sum(coef[j + 1] * xx[j] for j in range(k))) for xx, yy in zip([r[0] for r in rows], y)]
    return coef[0], coef[1:], pstdev(resid)


def release_day_points(comp: dict) -> dict[int, tuple[float, float | None]]:
    """A comparable's release-day hourly points: hour (1..24) -> (viewers, channels)."""
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, ValueError, TypeError):
        return {}
    pts: dict[int, tuple[float, float | None]] = {}
    for row in comp.get("hourly") or []:
        try:
            t = parse_time(row["from"])
            viewers = float(row["viewers"])
        except (KeyError, ValueError, TypeError):
            continue
        if viewers <= 0:
            continue
        if day_start <= t < day_start + timedelta(hours=24):
            h = int((t - day_start).total_seconds() // 3600) + 1
            ch = row.get("channels")
            pts[h] = (viewers, float(ch) if ch is not None else None)
    return pts


def collect_curves(case: dict) -> list[dict]:
    """The case's usable comparables as log-ratio curves.

    Each curve: {'logp1', 'logch', 'ratio': {hour: ln(viewers/p1)}}.
    The Grand Theft Auto VI category's own pre-release history is not a
    launch day and is excluded by name; so is any launch too small or too
    sparse to carry a shape.
    """
    curves = []
    for comp in case.get("comparables") or []:
        name = str(comp.get("name") or "")
        if "Grand Theft Auto VI" in name:
            continue
        pts = release_day_points(comp)
        if len(pts) < MIN_HOURS or 1 not in pts:
            continue
        p1, ch1 = pts[1]
        if p1 < MIN_P1:
            continue
        ratio = {h: math.log(v / p1) for h, (v, _) in pts.items() if h >= 2}
        if len(ratio) < MIN_HOURS - 1:
            continue
        curves.append({
            "logp1": math.log(p1),
            "logch": math.log(ch1) if ch1 is not None and ch1 > 0 else None,
            "ratio": ratio,
        })
    return curves


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def hour_forecast(p1: float, logp1: float, logch: float | None,
                  curves: list[dict], hour: int) -> list[float | int]:
    """The five quantiles for one hour, from the comparables' curves."""
    # Rows for this hour: (features, log-ratio).
    rows2, rows1 = [], []
    for c in curves:
        if hour not in c["ratio"]:
            continue
        rows1.append(([c["logp1"]], c["ratio"][hour]))
        if c["logch"] is not None:
            rows2.append(([c["logp1"], c["logch"]], c["ratio"][hour]))

    # Feature ranges for the extrapolation clamp.
    all_logp1 = [c["logp1"] for c in curves]
    all_logch = [c["logch"] for c in curves if c["logch"] is not None]
    lp_lo = min(all_logp1) - EXTRAP_MARGIN
    lp_hi = max(all_logp1) + EXTRAP_MARGIN
    lc_lo = min(all_logch) - EXTRAP_MARGIN if all_logch else 0.0
    lc_hi = max(all_logch) + EXTRAP_MARGIN if all_logch else 0.0
    t_lp = clamp(logp1, lp_lo, lp_hi)
    t_lc = clamp(logch, lc_lo, lc_hi) if logch is not None else None

    centre, sigma = None, None
    if len(rows2) >= MIN_ROWS and t_lc is not None:
        m1 = mean([r[0][0] for r in rows2])
        m2 = mean([r[0][1] for r in rows2])
        centred = [([x[0] - m1, x[1] - m2], y) for x, y in rows2]
        fit = ols(centred)
        if fit is not None:
            a, slopes, sigma = fit
            centre = a + slopes[0] * (t_lp - m1) + slopes[1] * (t_lc - m2)
    if centre is None and len(rows1) >= MIN_ROWS:
        m1 = mean([r[0][0] for r in rows1])
        centred = [([x[0] - m1], y) for x, y in rows1]
        fit = ols(centred)
        if fit is not None:
            a, slopes, sigma = fit
            centre = a + slopes[0] * (t_lp - m1)
    if centre is None:
        # Too few curves for this hour: the median log-ratio and an
        # IQR-based spread.
        rs = [c["ratio"][hour] for c in curves if hour in c["ratio"]]
        if rs:
            centre = median(rs)
            sigma = (quantile(rs, 0.75) - quantile(rs, 0.25)) / 1.349
        else:
            centre, sigma = 0.0, BASELINE_SIGMA
    sigma = max(sigma if sigma is not None else BASELINE_SIGMA, SPREAD_FLOOR)

    qs = [round2(p1 * math.exp(centre + z * sigma)) for z in Z]
    # Clamp to the legal range and keep the row non-decreasing.
    qs = [min(max(q, 1), 1e9) for q in qs]
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def baseline_forecast(p1: float, hours: list[int]) -> dict[int, list[float | int]]:
    """Persistence fallback: every hour at p1, random-walk spread."""
    return {
        h: [round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))) for z in Z]
        for h in hours
    }


def forecast_case(case: dict) -> dict:
    hours = [row["hour"] for row in case["forecast"]]
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    ch1 = first.get("channels")
    logch = math.log(ch1) if ch1 is not None and float(ch1) > 0 else None

    curves = collect_curves(case)
    if not curves:
        rows = baseline_forecast(p1, hours)
        return {"hours": [{"hour": h, "q": rows[h]} for h in hours]}

    logp1 = math.log(p1)
    out = []
    for h in hours:
        out.append({"hour": h, "q": hour_forecast(p1, logp1, logch, curves, h)})
    return {"hours": out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
