"""Week-one trophy rarity forecaster for Grand Theft Auto VI (PS5).

Entry for gta6-trophy-rarity-forecast-01. For each case the program reads
(schema makefun/gta6-trophies-case/v1) it writes one forecast file: five
quantiles (5/25/50/75/95%) of every trophy's earn rate on Exophase at the
case target, scored by pinball loss on the logit scale.

How the forecast is built -- the full reasoning is in README.md:

* Exophase refreshes its counts in jumps, hours apart, so the raw history
  is a staircase. The program first squeezes it down to the distinct
  reads (the refresh points) and works in log-time,
  x = ln(1 + hours since the first read). Fitting the raw 10-minute
  reads would let the long flat stretches dominate the fit.
* Only the most recent 36 hours of distinct reads are fitted. A trophy's
  earn rate is the share of tracked players who have reached some point
  in the game, and that share grows along an S-curve: launch weekend's
  slope is far steeper than day four's, so a slope fitted on the whole
  history overshoots a 4.5-day extrapolation. The last day and a half
  measures the growth rate that is actually current.
* For each trophy four estimates are combined in logit space:
  a straight trend of logit(rate) on x; a saturating curve of
  logit(rate / cap) on x that bends toward a per-trophy plausibility
  cap instead of running away log-linearly; a ratio estimate that fits
  log(players tracked) and log(earned counts) separately and divides
  them at the target -- the only view that can say rates FALL when the
  tracked base grows faster than players earn trophies, which is
  exactly what Exophase's first refresh did on launch night
  (638 -> 3,868 players tracked; every rate but the 100% one fell);
  and plain persistence of the last read. Fitted slopes are shrunk
  toward persistence when a trophy has few distinct reads.
* When the history has not moved at all (one distinct read, as in the
  practice cases) no slope exists to fit, so the forecast expects the
  next refresh to dilute the rates the way the launch-night refresh
  did: median = last rate x 0.205, with a 0.5 logit spread. A trophy
  already at 100% stays put.
* The spread per trophy combines the trend fit's residual (with its
  prediction-interval widening), a random-walk term growing with the
  horizon, the size of the downward refresh jumps the case's own
  history shows, and how far the four views disagree; clamped to
  [0.35, 1.6].

Standard library only, no network, no randomness: the same case in
always produces the same bytes out. Well inside the 8 s cap.
"""

import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels above.
NORMAL_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Fitting window: only distinct reads from the last day and a half count.
# With fewer than three distinct reads inside it, the whole history is used.
WINDOW_HOURS = 36.0

# Frozen-history rule, calibrated on the launch-night refresh (638 -> 3,868
# players tracked; every rate but the 100% one fell, median ratio 0.205).
DILUTION = 0.205
DILUTION_SIGMA = 0.5

# Spread constants, in logit units.
WALK_SIGMA_PER_ROOT_DAY = 0.30   # random-walk sigma per sqrt(day) of horizon
SIGMA_LOW = 0.35
SIGMA_HIGH = 1.6
NO_DATA_MEDIAN = 1.0             # only when a case carries no usable history
NO_DATA_SIGMA = 1.5


def bound(x):
    """Keep a percent inside (0, 100) so its logit stays finite."""
    return min(99.99, max(0.01, x))


def to_logit(pct):
    p = bound(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round_4dp(x):
    """Round to 4 decimals, halves up, as the house baseline does."""
    scaled = x * 10000.0
    n = math.floor(scaled)
    if scaled - n >= 0.5:
        n += 1
    out = n / 10000.0
    return int(out) if out == int(out) else out


def read_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def fit_line(xs, ys):
    """Ordinary least squares for y ~ a + b*x.

    Returns a dict with intercept, slope, count, residual sd, mean x and
    sum of squared x deviations, or None when the fit is degenerate.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        resid = math.sqrt(rss / (n - 2))
    else:
        resid = 0.30
    return {"a": intercept, "b": slope, "n": n, "resid": resid,
            "xbar": mean_x, "sxx": sxx}


def median_of(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# Plausibility caps (percent) for a trophy's rate a week after launch,
# keyed on its description. Each entry is (match kind, needle, cap):
# "start" matches a description prefix, "in" a substring anywhere.
# Deliberately generous: a cap only bites when an extrapolation runs
# away; the data does the work wherever the history is rich. The first
# match wins, so the specific rows come before the generic ones.
CAP_RULES = [
    ("in", "100% Completion", 4.0),
    ("start", "Complete Chapter 1", 75.0),
    ("start", "Complete Chapter 2", 65.0),
    ("start", "Complete Chapter 3", 55.0),
    ("start", "Complete Chapter 4", 50.0),
    ("start", "Complete the final mission", 30.0),
    ("in", "250 Stunts", 8.0),
    ("in", "Shooting Range", 8.0),
    ("in", "Drone Race", 10.0),
    ("in", "20 different races", 15.0),
    ("in", "Card sets", 6.0),
    ("in", "100 locations", 6.0),
    ("in", "50 Challenges", 15.0),
    ("in", "100 vehicles", 20.0),
    ("in", "5 Bank Vaults", 10.0),
    ("in", "Explore all of Leonida", 20.0),
    ("in", "Meditate at every spot", 12.0),
    ("in", "Hunt a unique animal", 20.0),
    ("in", "Beat Cal", 35.0),
    ("in", "arcade game", 40.0),
    ("in", "each type of activity", 30.0),
    ("in", "8 mission strands", 12.0),
    ("start", "Complete ", 35.0),
]


def plausibility_cap(trophy):
    """Generous day-7 ceiling for one trophy, from tier and description."""
    if trophy["tier"] == "platinum":
        return 4.0
    desc = trophy["description"]
    for kind, needle, cap in CAP_RULES:
        if kind == "start" and desc.startswith(needle):
            return cap
        if kind == "in" and needle in desc:
            return cap
    return 45.0


def squeeze_history(history):
    """Collapse the history to its distinct reads (refresh points).

    A read is new when its players-tracked count or any rate differs
    from the previous distinct read. Returns dicts with the read time,
    players tracked and rates, oldest first.
    """
    distinct = []
    previous = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == previous:
            continue
        previous = key
        try:
            when = read_time(read.get("slot") or read["at"])
        except Exception:
            continue
        distinct.append({"when": when,
                         "players": read.get("players_tracked"),
                         "rates": rates})
    return distinct


def typical_drop(reads, trophy_count):
    """Typical size, in logit units, of a downward refresh jump.

    For each pair of consecutive distinct reads, take the median across
    trophies of the positive logit fall; then the median of those across
    pairs. 0.10 when the history shows no fall at all.
    """
    per_pair = []
    for before, after in zip(reads, reads[1:]):
        falls = []
        for i in range(trophy_count):
            old = before["rates"][i] if i < len(before["rates"]) else None
            new = after["rates"][i] if i < len(after["rates"]) else None
            if old is None or new is None:
                continue
            gap = to_logit(old) - to_logit(new)
            if gap > 0:
                falls.append(gap)
        if falls:
            per_pair.append(median_of(falls))
    if not per_pair:
        return 0.10
    return min(0.60, max(0.10, median_of(per_pair)))


def predict_one(idx, trophy, reads, x_target, horizon_days, players_line,
                drop_size):
    """Return (median percent, logit sigma) for one trophy."""
    points = []
    last_rate = None
    for read in reads:
        rates = read["rates"]
        rate = rates[idx] if idx < len(rates) else None
        if rate is not None:
            last_rate = rate
        if rate is None or not read["players"]:
            continue
        points.append((read["x"], rate, read["players"]))
    if last_rate is None:
        return NO_DATA_MEDIAN, NO_DATA_SIGMA

    n = len(points)
    stay = to_logit(last_rate)

    # Frozen history: one distinct read carries no trend at all. Expect
    # the next refresh to dilute the rates as the launch-night one did.
    if n <= 1:
        if last_rate >= 99.5:
            return last_rate, 0.35
        return max(0.01, last_rate * DILUTION), DILUTION_SIGMA

    # Trust in a fitted slope grows with the number of distinct reads:
    # full weight from four reads on.
    shrink = min(1.0, (n - 1) / 3.0)

    xs = [p[0] for p in points]
    fit_var = 0.09

    # View 1 -- trend: logit(rate) against log-time.
    trend = None
    line = fit_line(xs, [to_logit(p[1]) for p in points])
    if line is not None:
        guessed = line["a"] + line["b"] * x_target
        trend = stay + shrink * (guessed - stay)
        widen = 1.0 + 1.0 / line["n"] + (x_target - line["xbar"]) ** 2 / line["sxx"]
        fit_var = (line["resid"] ** 2) * widen

    # The cap for this trophy: its class cap, but never below the highest
    # rate it has actually reached -- a cap below the observed rate is
    # incoherent, and it would clip a near-universal trophy down.
    cap = max(plausibility_cap(trophy), max(p[1] for p in points))

    # View 2 -- saturating: logit(rate / cap) against log-time, scaled
    # back by the cap. Far below the cap this is the trend again; near
    # the cap it bends down toward it instead of running away.
    saturating = None
    capped = []
    for p in points:
        frac = min(0.999, max(0.001, p[1] / cap))
        capped.append(math.log(frac / (1.0 - frac)))
    sat_line = fit_line(xs, capped)
    if sat_line is not None:
        sat_rate = cap * from_logit(sat_line["a"] + sat_line["b"] * x_target) / 100.0
        saturating = stay + shrink * (to_logit(sat_rate) - stay)

    # View 3 -- ratio: log(players) and log(earned) fitted separately,
    # divided at the target. Earned counts come from rate x players.
    ratio = None
    if players_line is not None:
        earned_line = fit_line(
            xs, [math.log(p[1] * p[2] / 100.0 + 0.5) for p in points])
        if earned_line is not None:
            players_at = math.exp(players_line["a"] + players_line["b"] * x_target)
            earned_at = math.exp(earned_line["a"] + earned_line["b"] * x_target) - 0.5
            if players_at > 0 and earned_at >= 0:
                ratio_rate = 100.0 * earned_at / players_at
                # The ratio view respects the same plausibility cap as the
                # median does: a wild players/earned fit must not drag
                # the blend to an absurd rate on its own.
                ratio_rate = min(ratio_rate, cap)
                ratio = stay + shrink * (to_logit(ratio_rate) - stay)

    # Blend the views in logit space, persistence always in the mix.
    weighted = []
    if trend is not None:
        weighted.append((0.30, trend))
    if saturating is not None:
        weighted.append((0.30, saturating))
    if ratio is not None:
        weighted.append((0.25, ratio))
    weighted.append((0.15, stay))
    total_w = sum(w for w, _ in weighted)
    blended = sum(w * v for w, v in weighted) / total_w

    median_pct = from_logit(blended)
    if median_pct > cap:
        median_pct = cap
        blended = to_logit(median_pct)

    # Spread: fit residual, random walk over the horizon, the refresh
    # drops this case actually shows, and disagreement between views.
    walk_var = (WALK_SIGMA_PER_ROOT_DAY * math.sqrt(max(horizon_days, 0.0))) ** 2
    views = [v for _, v in weighted]
    if len(views) > 1:
        centre = sum(views) / len(views)
        disagree_var = sum((v - centre) ** 2 for v in views) / len(views)
    else:
        disagree_var = 0.0
    sigma = math.sqrt(fit_var + walk_var + drop_size ** 2 + disagree_var)
    sigma = min(SIGMA_HIGH, max(SIGMA_LOW, sigma))
    return median_pct, sigma


def run_case(case, label):
    if case.get("levels") != QUANTILE_LEVELS:
        raise ValueError(f"{label} asks for levels {case.get('levels')}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{label} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    reads = squeeze_history(history)
    if not reads:
        raise ValueError(f"{label} has no readable history")
    start = reads[0]["when"]
    target = read_time(case["target"])

    def log_time(when):
        hours = max((when - start).total_seconds(), 0.0) / 3600.0
        return math.log(1.0 + hours)

    x_target = log_time(target)
    for read in reads:
        read["x"] = log_time(read["when"])

    # Fitting window: the last WINDOW_HOURS of distinct reads, or the
    # whole history when that holds fewer than three.
    latest = reads[-1]["when"]
    window = [r for r in reads
              if (latest - r["when"]).total_seconds() <= WINDOW_HOURS * 3600.0]
    if len(window) < 3:
        window = reads

    # One players-tracked line for the whole case, on the same window.
    players_line = None
    px = [r["x"] for r in window if r["players"]]
    py = [math.log(r["players"]) for r in window if r["players"]]
    if len(px) >= 2:
        players_line = fit_line(px, py)

    drop_size = typical_drop(reads, len(trophies))

    rows = []
    for i, trophy in enumerate(trophies):
        median_pct, sigma = predict_one(
            i, trophy, window, x_target, horizon_days, players_line, drop_size)
        centre = to_logit(median_pct)
        qs = [round_4dp(bound(from_logit(centre + z * sigma))) for z in NORMAL_Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def empty_forecast(case):
    """Last-resort forecast when a case cannot be fitted at all."""
    rows = []
    for trophy in case["trophies"]:
        centre = to_logit(NO_DATA_MEDIAN)
        qs = [round_4dp(bound(from_logit(centre + z * NO_DATA_SIGMA)))
              for z in NORMAL_Z]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = run_case(case, path.name)
        except Exception:
            traceback.print_exc()
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = empty_forecast(case)
            except Exception:
                traceback.print_exc()
                continue
        out_path = OUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
