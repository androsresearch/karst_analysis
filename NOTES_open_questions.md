# Open questions and known issues

This file tracks methodological and code issues that have been identified
but not yet resolved. They are not blockers for the current pipeline, but
should be revisited (some before thesis submission, some after).

---

## 1. Spatial balance of the input grid to the breakpoint detector

**Identified:** 2026-05-07 (v15 development), well LRS70D, campaign 2022_02.

**Symptom.** When extending the BIC sweep from N=10 to N=15, breakpoints
at high N do not visibly improve the segmentation — they cluster on flat
saltwater regions instead of refining the freshwater→saltwater transition.

**Root cause.** The pipeline resamples the SEC profile on a uniform Δz
grid (`resample_pchip`, `dz_method="percentile95"`). For LRS70D this
gives ~2049 points from ~14624 raw, with dz ≈ 14 mm. Diagnostic figures
in `results/figures/diagnostics/LRS70D_diag*.png` show the resulting grid
puts ~55 % of points on flat saltwater (depth > 13 m), while the
freshwater→saltwater transition (4–9 m) holds the actual physical signal
but only ~17 % of the points.

`piecewise_regression` minimises RSS, which is a sum over points. Many
points in flat regions ⇒ the optimiser is biased toward placing
breakpoints there.

**Status.** Mariana investigates non-uniform resampling alternatives in
a separate chat. Diagnostic script preserved at
`scripts/diagnostics/quick_diagnose_lrs70d.py`. Open until resolved.

**Defendable framing for the thesis (in case it isn't fixed by submission):**
*"Trial-to-trial instability at high N traces to the spatial imbalance of
the uniform-z input grid combined with the bimodal log10(SEC)
distribution of coastal karst profiles. A non-uniform resampling
strategy (e.g. arc-length parameterisation) is left as future work."*

---

## 2. `extract_segments` reports a constant slope across all segments

**Identified:** 2026-05-07 during Task 2 (slopes module) review.

**Where.** `src/karst_analysis/sec/breakpoints/segments.py`, function
`extract_segments`. The `slope` field of each output segment is set to
`alpha` (the linear coefficient of the Muggeo model, read once outside
the loop). All segments end up with the same value.

**Correct behaviour.** In a Muggeo piecewise model with N breakpoints,
the slope of segment k is `alpha + sum(beta_1..beta_k)`. The current
code returns only `alpha` for every segment, which is the slope of the
first segment.

**Impact assessment.**
- The `slopes.py` module (Task 2) does NOT use `extract_segments` — it
  computes chord slopes directly from breakpoint coordinates. Unaffected.
- No production figure or downstream module currently consumes
  `segments[i]["slope"]`. The bug is dormant.

**Fix sketch.** In the loop over segments, accumulate `alpha + beta_1
+ ... + beta_k` from `raw_params` and assign that as `slope` for
segment k. ~10 lines of code change. Add a regression test on a
two-breakpoint synthetic profile with known slopes.

**Status.** Deferred to v15.1 or post-thesis. Documented here so it
isn't forgotten.

---

## 3. `slopes_batch.py` does not register runs in `runs.csv`

**Identified:** 2026-05-07 during v15 packaging.

**Symptom.** `breakpoints_batch.py` and other batch scripts append a row
to `results/runs.csv` for every run via the `run_ledger` context
manager. `slopes_batch.py` does not.

**Why it was deferred.** The schema for slope runs (which columns?
which fields are mandatory?) wasn't settled at the time the batch was
written. Skipping the ledger was the safer default than committing to
a wrong schema.

**Fix sketch.** Decide schema (suggested: well, date, method, trial, n,
threshold, n_pairs, top_mz_depth, bot_mz_depth, csv_out, fig_dir),
wrap `_process_job` in `with run_ledger(...) as run_id:` block, write
the row at the end with the run_id and elapsed time.

**Status.** Deferred to v15.1. Marked with a `TODO (v15.1)` comment in
the code at the appropriate spot in `_process_job`.

---

## 4. `piecewise_regression` is not seedable → trial-to-trial instability

**Identified:** earlier in v15 development, related to issue 1.
**Updated:** 2026-05-08 (v16 development), partial workaround implemented.

**Symptom.** The same well, same method, same N produces different
breakpoint positions across trials. SEC works around this by running
3 trials per JSON and inspecting them visually.

**Original framing (still valid for SEC).** `piecewise_regression`
does not expose a `seed` parameter; under the hood it uses
`np.random.uniform(...)` in `_generate_breakpoints`. Patching upstream
is out of scope for this thesis.

**Update v16.** Empirical testing showed that seeding the **global**
numpy RNG immediately before a `Fit(...)` call IS deterministic,
because under the hood the package uses `np.random.uniform(...)` and
`np.random.choice(...)` against the global state. Verified on
viz_sharp T16 x=160, log10 scale, N=8:

    np.random.seed(42)              # set global state
    fit = pw.Fit(x, y, n_breakpoints=8, ...)
    # → same breakpoints, bit-for-bit, on every re-run

Caveat: any intervening `np.random.*` call between the seed and the
Fit consumes randomness and breaks the determinism.

**v16 implementation (ERT only).** A seed-discovery wrapper iterates
seeds starting at 0 until `pw.Fit` converges, recording the
successful seed alongside the breakpoints in
`ErtBreakpointFit.seed_used`. Reproducing a published figure means
re-running with the logged seed. This is bookkeeping, not an upstream
patch. Implemented in
`src/karst_analysis/ert/breakpoints.py::detect_breakpoints_with_seed_discovery`.

**Defendable framing for the thesis (revised):**
*"Reproducibility for the ERT pipeline is achieved by capturing the
numpy global seed used for each successful trial. Re-running with the
same seed reproduces breakpoint positions to machine precision.
Independent reproducibility across numpy versions remains future
work. The SEC pipeline currently uses the original three-trial
protocol; retrofitting seed-capture to SEC is post-thesis bookkeeping."*

**Decision needed.** Whether to retrofit this into
`breakpoints_batch.py` and the `runs.csv` schema (one column
`np_seed`) before thesis submission, or defer post-thesis.
Recommendation: defer, document here.

---

## 5. Convergence is seed-dependent at high N

**Identified:** 2026-05-08 during ERT 1D exploration on
viz_sharp T16 x=160 (220 points, max_breakpoints=15, n_trials=1).

**Symptom.** Two consecutive runs with the same data, same code, same
N gave **different convergence outcomes** for high N:

| Run | linear N=14 | linear N=15 | log10 all N |
|---|---|---|---|
| 1 | failed | failed | converged |
| 2 | converged | converged | converged |

A controlled seed sweep confirmed that for linear N=15, only **5 of
10 random seeds** (seeds 0, 3, 5, 6, 9 out of 0..9) led to
convergence; the other half failed silently
(`fit.best_muggeo` was `None`). For log10 N=15 on the same data,
**8 of 10 seeds** converged (seeds 0 and 7 failed).

**Implication.** Reporting "the detector did/did not converge for
well W at N=N" without recording the seed is methodologically weak —
the result depends on a random number, not on the data. This is the
same underlying issue as #4, but with a more pointed consequence:
at high N, randomness controls **whether you get an answer at all**,
not just which answer you get.

**Mitigation in v16+ (ERT).** The seed-discovery wrapper iterates
until convergence and records the successful seed (default
`max_seed_attempts=20`). The set of failed seeds before success is
also captured in `ErtBreakpointFit.seeds_tried` for auditing.

**Mitigation in production SEC (current).** The 3-trial protocol
masks this partially: if any of 3 trials converges, you get a
result. But you cannot tell from the JSON whether trial_2 failed
because the model is overspecified or because the seed was unlucky.

**Status.** Documented. ERT mitigation in place. SEC retrofit is
post-thesis.

---

## 5b. Multi-seed BIC comparison (extension of #4 and #5)

**Identified:** 2026-05-08 during ERT module design discussion.

**Idea.** The seed-discovery pattern in updated #4 stops at the
first seed that produces a converged fit. Mariana noted this is
the **minimum** ambition. A more rigorous variant:

    For a given (well, scale, N), iterate seeds 0, 1, 2, ... and keep
    going until THREE distinct seeds have produced converged fits.
    Compare the BIC of the three converged fits. Keep the seed
    associated with the lowest BIC; record all three (seed, bic) pairs
    in the JSON for transparency.

**Why this matters.** A converged fit is not necessarily the best
local optimum that `piecewise_regression` can reach. Different seeds
land on different starting points, which Muggeo's algorithm refines
to different local optima. With a single converged seed you get
"a" local optimum; with N converged seeds you can pick the best-BIC
one and report variance across seeds.

**Cost.** Roughly 3× the fit time per (well, scale, N), plus the
bookkeeping of N seed/BIC tuples in the JSON.

**For the thesis text:**
*"Reproducibility achieved by capturing the seed of a converged fit
(see entry #4). The current implementation uses the first converged
seed. A future extension will compare three converged seeds by BIC
and keep the best, providing both reproducibility AND robustness
against seed-induced local-optimum bias."*

**Status.** Implementation deferred (post-thesis). Today's ERT
module implements the simple version: first converged seed wins.
TODO comment in `src/karst_analysis/ert/breakpoints.py` points
back to this entry.

---

## 6. `elbow_max_distance` is sensitive to BIC magnitude rescaling

**Identified:** 2026-05-08, ERT 1D exploration.

**Symptom.** On viz_sharp T16 x=160:

- Linear scale (resist): BIC ranges ~+900 → −600. Elbow detected
  at N=5. Plausible.
- Log10 scale (resistlog10): BIC ranges ~−1100 → −2650. Elbow
  detected at N=8. The N=8 fit places two pairs of breakpoints
  within 2 m of each other (18.50/20.54 and 23.45/27.01),
  suggesting overfit.

The BIC curves have **similar shape** (rapid drop to N≈5–7, then
flat-ish plateau with small ripples). The elbow position differs
mainly because the algorithm's perpendicular-distance-to-chord
metric scales with the absolute BIC range, not its second
derivative.

**Why this matters for ERT.** SEC always uses log10(SEC) (locked
methodological decision, BRIEF §4). If we apply the same convention
to ERT (use log10 resistivity), we inherit the bias toward higher
elbow N when the post-N=K plateau is long. For ERT 1D profiles
with 200-ish points and very smooth inverted resistivity, plateaus
are longer than for SEC.

**Possible alternative metrics** (not implemented, just listed):

1. Use the **discrete second derivative** of BIC: argmax of
   `|BIC[N+1] − 2·BIC[N] + BIC[N−1]|` as the elbow.
2. Use the **fractional BIC drop**: stop at the first N where
   `(BIC[N−1] − BIC[N]) / (BIC[0] − BIC[N_min]) < epsilon`.
3. Keep `elbow_max_distance` but normalise BIC to `[0, 1]` before
   computing the elbow.

**Status.** Documented as a methodological caveat for the ERT
chapter. A switch to a different elbow rule for ERT (vs SEC) is a
thesis-level decision; defer to Mariana.

For the thesis text:
*"Where SEC and ERT elbow Ns differ, this reflects in part the
elbow-detection metric's sensitivity to the BIC dynamic range,
which differs by ~10× between the two techniques. Reported
breakpoints should be interpreted as transitions detected by the
model, not as strict count optima."*

---

## 7. Mixing-zone tie detection uses exact equality

**Identified:** 2026-05-08 while writing tests for the new ERT module.

**Symptom.** `_mark_mixing_zone` (SEC) and `select_ert_mixing_zone`
(ERT) both detect ties between BOT-MZ-eligible breakpoints with:

    np.isclose(m1, m2, rtol=0.0, atol=0.0)

which collapses to exact bit-for-bit equality. In practice, even
perfectly-symmetric synthetic profiles yield turning angles that
match to ~15 decimal places but differ in the last bit, so the tie
warning **never fires on real data**.

**Implication.** If two BPs do have near-identical curvature, the
user gets no warning. The selection still runs (first-occurrence
wins via the stable sort), but silently — the methodological
ambiguity is hidden.

**Fix (post-thesis).** Change to `np.isclose(m1, m2, rtol=1e-12)`
or similar small tolerance, in BOTH:
  - `src/karst_analysis/sec/slopes.py::_mark_mixing_zone` (line ~346)
  - `src/karst_analysis/ert/mixing_zone.py::select_ert_mixing_zone`

**Status.** Test in
`tests/test_ert_mixing_zone.py::test_tie_warning` is currently
marked `xfail(strict=True)` to track the issue; it will start
passing automatically when the tolerance is loosened, and pytest
will flag it for un-marking.
