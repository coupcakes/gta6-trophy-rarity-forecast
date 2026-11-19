[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/coupcakes/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/coupcakes/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast — week one

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the
percent of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z — one week after launch — from the house's recording
of Exophase's trophy page up to the case checkpoint. A single program,
`main.py` (Python 3.13, standard library only, no network), reads each
case in the input directory and writes one forecast file per case, with
five quantiles (5/25/50/75/95%) per trophy.

Run it the way the kit runs the starter:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## The data is a staircase, not a slope

Exophase does not update continuously. It refreshes a game's counts on
its own schedule, hours apart, so the history is a step function: long
frozen stretches, then a jump. The first jump on launch night tells the
whole story of the practice cases — 638 players tracked became 3,868,
and every earn rate but the one at 100% fell (median ratio 0.205),
because the newly tracked players had earned far fewer trophies than
the launch-night hardcore sample. A forecaster that treats the last
read as the truth and adds a symmetric random-walk spread (the house
baseline) scores a pinball loss of about 1.51 on the practice for
exactly that reason: the refresh pushed every rate outside its band.

So the program is built around three properties of the data:

1. **A rate is a ratio.** `rate = earned / players tracked`, and the two
   sides move for different reasons: the tracked base surges as more
   players link their profiles (diluting every rate), while earned
   counts grow as players actually play. The program fits both sides
   separately — log(players) and log(earned) against log-time — and
   divides them at the target. When the base is surging faster than
   players earn, this view says rates fall, which a direct trend on
   the rate cannot say honestly.
2. **Nothing happens between refreshes.** Consecutive identical reads
   carry no information, so the program squeezes the history down to
   its distinct reads (the refresh points) and fits on those, in
   log-time `x = ln(1 + hours since the first read)`. Fitting the raw
   10-minute reads would let the frozen stretches dominate the fit.
   The fits use only the last 36 hours of distinct reads: a trophy's
   earn rate is the share of players who have reached a point in the
   game, and that share grows along an S-curve, so launch weekend's
   growth is far steeper than day four's, and a slope fitted on the
   whole history overshoots a 4.5-day extrapolation badly. The last
   day and a half measures the growth rate that is actually current.
3. **A trophy's curve bends, and its description says where.** A
   log-linear trend fitted on two days of launch data will happily send
   Chapter 1 to 95% or the platinum to 20% by day seven. The program
   caps each trophy's median at a generous day-7 plausibility ceiling
   read from its description and tier (story chapters 50–75%, the final
   mission 30%, the platinum and 100% completion 4%, long grinds
   6–15%, ordinary trophies 35–45%). The ceilings sit above any rate
   the trophy could plausibly reach in a week, so they only bite when
   an extrapolation runs away.

## The forecast

Per trophy, four estimates are blended in logit space:

* **Trend** (weight 0.30): least squares of `logit(rate)` on log-time
  over the fitting window (the last 36 hours of distinct reads) — the
  house curve baseline's shape, but over a longer window than its
  24 hours.
* **Saturating** (weight 0.30): least squares of `logit(rate / cap)` on
  log-time, scaled back by the cap. Far below its cap a trophy behaves
  like the trend; near it, the curve bends down toward the cap instead
  of running away log-linearly. This is the view that stops Chapter 1
  from being forecast at 95% by day seven: a trophy's earn rate is the
  share of players who have reached a point in the game, and that share
  follows an S-curve, not a power law.
* **Ratio** (weight 0.25): the earned/players ratio described above,
  extrapolated to the target.
* **Persistence** (weight 0.15): the trophy's rate in the last read.

When a trophy has few distinct reads its fitted slopes are shrunk
toward persistence (full weight from four distinct reads). When the
history shows no movement at all — a single distinct read, as in the
practice cases — no trend can be fitted, so the program expects the
next refresh to dilute the rates the way the launch-night refresh did:
median = last rate × 0.205 (the median ratio of that refresh), with a
0.5 logit spread. (A trophy already at 100% stays there.)

## The spread

Each trophy gets its own logit sigma, combining four terms:

* the trend fit's residual, with its prediction-interval widening (how
  far the target sits past the data);
* a random-walk term, `0.30 × sqrt(horizon in days)`;
* the size of the downward refresh jumps the case's own history shows
  (the median positive logit fall between consecutive distinct reads);
* how much the views disagree with each other.

Sigma is clamped to [0.35, 1.6] (0.5 under the frozen-history rule). The
five quantiles are `inv_logit(median + z_tau × sigma)` with the standard
normal quantiles, rounded to 4 decimals halves-up.

## Honest limits

* The final's answer is the first valid read in the hour from the
  target, and it shows the rates as of Exophase's last refresh before
  it — which may be hours earlier. The spread carries that uncertainty;
  the median cannot know it.
* The class caps are judgement calls, stated in `CAP_RULES` in
  `main.py`. They are deliberately generous, so a wrong cap costs
  little; the data does the work wherever the history is rich.
* No number in this entry comes from a source other than the case file
  itself. The launch-night refresh figures quoted above (638 → 3,868
  players, median rate ratio 0.205) are the kit's own practice data
  and are used only to calibrate the frozen-history rule.
* Everything is deterministic: same case in, same bytes out.

## Files

* `main.py` — the forecaster (the entrypoint).
* `README.md` — this file.
* `LICENSE` — MIT.
