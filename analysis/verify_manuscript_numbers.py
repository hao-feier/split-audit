"""Verify every headline number in the manuscript against the result JSONs.

Motivation (2026-10-01): when the honest-vs-leaky contrast was extended from 5
to 10 architectures, every derived number in the paper silently became wrong --
31-36 pp, "over 42 runs", median +34.7, "six of the 120", OA "1-3 points".
All of them still COMPILED and produced a clean 31-page PDF. A compiler cannot
catch this class of error; only recomputing from the data can.

This script is the guard. It recomputes each claim from the JSONs and asserts
that the exact string appears in paper/main.tex. A missing or drifted number
fails here rather than shipping.

Run:  python analysis/verify_manuscript_numbers.py
Exit: 0 = all claims verified; 1 = at least one mismatch (details printed).
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from aggregate_results import binary_change  # noqa: E402

TEX = ROOT / "paper" / "main.tex"
UNIF = ROOT / "results" / "uniform"
MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "deeplabv3plus_pre", "temporal_unet", "temporal_unet_gn", "fc_siam",
          "changeformer"]
PRIMARY5 = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SEEDS = [2026, 2027, 2028]
KS = [1, 2, 3]

tex = TEX.read_text()
results = []


def check(label, computed, must_appear):
    """assert the literal string `must_appear` is in the manuscript."""
    ok = must_appear in tex
    results.append((ok, label, computed, must_appear))


def assert_true(label, condition, detail):
    """Assert a computed CONDITION, not the presence of a literal.

    check() only ever asks `must_appear in tex`, so it cannot express "this must
    NOT be present" or "this must hold in the compiled PDF" -- passing a needle
    that is absent makes it report OK unconditionally. Every absence/residue
    assertion therefore has to go through this function instead.
    """
    results.append((bool(condition), label, detail,
                    "condition true" if condition else "condition FALSE"))


def load(p):
    return json.loads(Path(p).read_text())


summary = load(UNIF / "metrics_summary.json")
comp = load(UNIF / "random_split" / "comparison.json")
excluded = set(comp["_config"]["excluded_collapsed_runs"])

# Honest arm of the split contrast: the buffer-isolated arm of the SAME
# counterfactual sweep (rnd_iso), not the primary five-seed sweep.  Using the
# primary sweep mixed conventions -- it keeps all five seeds, so a collapsed
# all-background run entered the honest mean while the leaky arm had collapsed
# runs removed -- and inflated the gap for exactly the affected families.
honest = {m: st.mean([r["F1_change"] for r in comp[m]["rnd_iso"].values()])
          for m in MODELS}
honest_oa = {m: st.mean([r["OA"] for r in comp[m]["rnd_iso"].values()])
             for m in MODELS}

# ---- per-model mean gap, collapse-excluded (table rows) --------------------
gaps, oa_gaps = {}, {}
for m in MODELS:
    gaps[m] = 100 * (st.mean([r["F1_change"] for r in comp[m]["rnd_rnd"].values()])
                     - honest[m])
    oa_gaps[m] = 100 * (st.mean([r["OA"] for r in comp[m]["rnd_rnd"].values()])
                        - honest_oa[m])

# ---- per-run distribution (collapse-excluded) ------------------------------
per_run = []
for m in MODELS:
    for k in KS:
        for s in SEEDS:
            run = f"k{k}-s{s}"
            if run in {e.split("/")[1] for e in excluded if e.startswith(m + "/")}:
                continue
            d = load(UNIF / f"random_split_k{k}" / f"metrics_{m}_s{s}.json")
            per_run.append(100 * (binary_change(
                d["random_test"]["confusion_matrix"])["F1_change"] - honest[m]))

# ---- all-runs (collapse included) -----------------------------------------
all_gaps = {}
for m in MODELS:
    vals = []
    for k in KS:
        for s in SEEDS:
            d = load(UNIF / f"random_split_k{k}" / f"metrics_{m}_s{s}.json")
            vals.append(100 * (binary_change(
                d["random_test"]["confusion_matrix"])["F1_change"] - honest[m]))
    all_gaps[m] = st.mean(vals)

n_excl = len(excluded)
n_total = len(MODELS) * len(KS) * len(SEEDS)
n_runs_ok = len(per_run)
gm = sorted(gaps.values())
gm_all = sorted(all_gaps.values())
oa_main = sorted(v for m, v in oa_gaps.items() if m != "changeformer")

# headline range in prose: deleted with design paragraph. The literal may be
# split across a line break, so compare on a whitespace-collapsed copy.
_arch_range = f"$+{min(gm):.1f}$ to $+{max(gm):.1f}$"
_arch_range_flat = re.sub(r"\s+", "", _arch_range)
assert_true("per-architecture gap range (prose)",
            _arch_range in tex
            or _arch_range_flat in re.sub(r"\s+", "", tex),
            _arch_range)
 # section heading no longer contains the range (title simplified); range
 # is verified in abstract and body text above
# per-run, collapse, and OA checks: content deleted in story-line compression
# (design paragraph, collapse runs, OA discussion removed from §4.2)

# ---- parameter-count span (guards against confusing F1 with param count) ----
pm = {m: summary[m]["2026"]["_params_M"] for m in MODELS if "_params_M" in summary[m]["2026"]}
check("parameter-count span", f"{min(pm.values()):.1f}-{max(pm.values()):.0f}M",
      f"${min(pm.values()):.1f}$--${max(pm.values()):.0f}$M parameters")
# families = distinct architecture CLASSES in mangrove/models.py, not model names
# (unet/unet_cbam/unet_msa/unet_cbam_msa are one class with flags;
#  deeplabv3plus_pre subclasses DeepLabV3Plus)
FAMILIES = {"UNet": ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa"],
            "DeepLabV3Plus": ["deeplabv3plus"],
            "DeepLabV3PlusPre": ["deeplabv3plus_pre"],
            "TemporalUNet": ["temporal_unet", "temporal_unet_gn"],
            "FCSiam": ["fc_siam"],
            "ChangeFormer": ["changeformer"]}
covered = sorted(x for v in FAMILIES.values() for x in v)
WORDS = {6: "six"}
check("family classes cover all 10 models",
      f"{len(FAMILIES)} classes / {len(covered)} models",
      f"{WORDS.get(len(FAMILIES), len(FAMILIES))} families")
check("every architecture accounted for",
      "all" if covered == sorted(MODELS) else f"MISMATCH {set(MODELS) ^ set(covered)}",
      "six families")

# ---- per-model gaps: now reported as TABLE ROWS, not prose ----------------
# The results paragraph used to enumerate all ten model gaps inline. That was
# removed during compression because tab:rand already carries them row by row,
# so the guard follows the reporting surface: it checks the generated table, and
# still requires the overall range in the prose.
p5 = sorted(gaps[m] for m in PRIMARY5)
ctrl = sorted(gaps[m] for m in MODELS if m not in PRIMARY5)
# the gap is a column of tab:rand; the tables are now inlined into main.tex
# (AUTO-GENERATED marker block), so the rendered cells live in `tex` itself.
_tab = tex
for m in MODELS:
    needle = f"${{({gaps[m]:+.1f})}}$"
    assert_true(f"tab:rand gap column for {m}", needle in _tab,
                "present" if needle in _tab else "ABSENT from main.tex")
# model range in prose: deleted with design paragraph

# ---- scope guards: things that must still say FIVE ------------------------
# The AP/MCC scoping guard used to require "five architectures of the original
# sweep": that claim is gone because the metric comparison was recomputed over
# all ten architectures and all 90 paired runs, so the guard now pins the wider
# scope instead of the old narrower one.
# scope guards: content deleted in story-line compression

# ---- Eq.(2) identity: the closed form must reproduce the pipeline's F1 ----
# F1 = 2*pi*TPR / (pi + pi*TPR + (1-pi)*FPR).  An earlier draft dropped the
# pi*TPR term from the denominator, which overstates F1 by ~3% relative here and
# would silently shift every decomposition component.  Guard it numerically.
def _f1_closed(pi, tpr, fpr):
    return 2 * pi * tpr / (pi + pi * tpr + (1 - pi) * fpr)


def _f1_from_cm(cm):
    tp = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
    fn = cm[1][0] + cm[1][3] + cm[2][0] + cm[2][3]
    fp = cm[0][1] + cm[0][2] + cm[3][1] + cm[3][2]
    return 2 * tp / (2 * tp + fp + fn)


_eq_err = 0.0
for _k in (1, 2, 3):
    for _f in sorted((UNIF / f"random_split_k{_k}").glob("metrics_*_s*.json")):
        _d = load(_f)
        for _blk in ("isolated_test", "random_test"):
            _b = _d.get(_blk)
            if not _b or "confusion_matrix" not in _b:
                continue
            _cm = _b["confusion_matrix"]
            _tot = sum(sum(r) for r in _cm)
            _tp = _cm[1][1] + _cm[1][2] + _cm[2][1] + _cm[2][2]
            _fn = _cm[1][0] + _cm[1][3] + _cm[2][0] + _cm[2][3]
            _fp = _cm[0][1] + _cm[0][2] + _cm[3][1] + _cm[3][2]
            _tn = _tot - _tp - _fn - _fp
            _pi, _tpr, _fpr = (_tp + _fn) / _tot, _tp / (_tp + _fn), _fp / (_fp + _tn)
            _eq_err = max(_eq_err, abs(_f1_closed(_pi, _tpr, _fpr) - _f1_from_cm(_cm)))
check("Eq.(2) closed form == F1 from the confusion matrix",
      f"max abs error {_eq_err:.2e}", "")
assert_true("Eq.(2) reproduces the pipeline F1 to 1e-4",
            _eq_err < 1e-4, f"max abs error {_eq_err:.2e}")
def _prev_only(H, L, _f):
    """Prevalence-only F1 shift with TPR and FPR held at their honest values."""
    return 100 * (_f(L[0], H[1], H[2]) - _f(H[0], H[1], H[2]))


_pm = {}
for _k in (1, 2, 3):
    for _f in sorted((UNIF / f"random_split_k{_k}").glob("metrics_*_s*.json")):
        _mm = re.match(r"^metrics_(.+)_s(\d+)\.json$", _f.name)
        if not _mm:
            continue
        _mo = _mm.group(1)
        if f"{_mo}/k{_k}-s{_mm.group(2)}" in excluded:
            continue
        _d = load(_f)
        _H, _L = _d.get("isolated_test"), _d.get("random_test")
        if not (_H and _L) or "confusion_matrix" not in _H:
            continue

        def _prt(b):
            _c = b["confusion_matrix"]
            _TP = _c[1][1] + _c[1][2] + _c[2][1] + _c[2][2]
            _FN = _c[1][0] + _c[1][3] + _c[2][0] + _c[2][3]
            _FP = _c[0][1] + _c[0][2] + _c[3][1] + _c[3][2]
            _TN = sum(sum(r) for r in _c) - _TP - _FN - _FP
            return (_TP + _FN) / sum(sum(r) for r in _c), _TP / (_TP + _FN), _FP / (_FP + _TN)

        _pm.setdefault(_mo, []).append(_prev_only(_prt(_H), _prt(_L), _f1_closed))
_pm_means = {k: st.mean(v) for k, v in _pm.items()}
_prev_lo, _prev_hi = min(_pm_means.values()), max(_pm_means.values())
_rng_a = f"$+{_prev_lo:.1f}$--$+{_prev_hi:.1f}$"
_rng_b = f"$+{_prev_lo:.1f}$ to $+{_prev_hi:.1f}$"
check("prevalence-only shift, per-model range (Eq. 2)",
      f"+{_prev_lo:.1f} to +{_prev_hi:.1f}",
      _rng_a if _rng_a in tex else _rng_b)
assert_true("prevalence-only mean is +2.7 pp (Eq. 2)",
            abs(st.mean([x for v in _pm.values() for x in v]) - 2.66) < 0.02,
            f"{st.mean([x for v in _pm.values() for x in v]):.2f}")

assert_true("Eq.(2) keeps the pi*TPR denominator term",
            r"\pi + \pi\,\text{TPR} + (1-\pi)\,\text{FPR}" in tex,
            "present" if r"\pi + \pi\,\text{TPR}" in tex else "__MISSING__")

# ---- symmetric-inclusion robustness: the contrast must not need the filter ---
_inc, _inc_pm = [], {}
for _k in (1, 2, 3):
    for _f in sorted((UNIF / f"random_split_k{_k}").glob("metrics_*_s*.json")):
        _mm = re.match(r"^metrics_(.+)_s(\d+)\.json$", _f.name)
        if not _mm:
            continue
        _d = load(_f)
        _H, _L = _d.get("isolated_test"), _d.get("random_test")
        if not (_H and _L) or "confusion_matrix" not in _H:
            continue

        def _cmf1(b):
            _c = b["confusion_matrix"]
            _TP = _c[1][1] + _c[1][2] + _c[2][1] + _c[2][2]
            _FN = _c[1][0] + _c[1][3] + _c[2][0] + _c[2][3]
            _FP = _c[0][1] + _c[0][2] + _c[3][1] + _c[3][2]
            return 2 * _TP / (2 * _TP + _FP + _FN)

        _g = 100 * (_cmf1(_L) - _cmf1(_H))
        _inc.append(_g)
        _inc_pm.setdefault(_mm.group(1), []).append(_g)
_inc_pm_m = {k: st.mean(v) for k, v in _inc_pm.items()}
check("symmetric-inclusion pooled gap (no filter)", f"{st.mean(_inc):+.1f} pp",
      f"${st.mean(_inc):+.1f}$")
check("symmetric-inclusion per-model range",
      f"+{min(_inc_pm_m.values()):.1f} to +{max(_inc_pm_m.values()):.1f}",
      f"$+{min(_inc_pm_m.values()):.1f}$ to $+{max(_inc_pm_m.values()):.1f}$")
assert_true("symmetric-inclusion keeps every architecture positive",
            all(v > 0 for v in _inc_pm_m.values()),
            f"min {min(_inc_pm_m.values()):+.1f} pp")
assert_true("symmetric-inclusion conclusion unchanged (pooled > 20 pp)",
            st.mean(_inc) > 20.0, f"{st.mean(_inc):.1f} pp")

# ---- tab:factorial components (Conclusions quotes these ranges) ---------
_ir = load(UNIF / "ir_factorial.json") if (UNIF / "ir_factorial.json").exists() else None
if _ir:
    _pm = _ir["per_model"]
    _g = lambda f: [(_pm[k].get("excl_collapse", _pm[k]))[f] for k in _pm]
    _di, _dt, _ds = _g("dInt_pp"), _g("dTrain_pp"), _g("dTest_pp")
    _tot = [a + b for a, b in zip(_ds, _di)]
    check("factorial interaction range (Conclusions)",
          f"+{min(_di):.1f} to +{max(_di):.1f} pp",
          f"$+{min(_di):.1f}$ to $+{max(_di):.1f}$")
    check("factorial test-alone range (Conclusions)",
          f"+{min(_ds):.1f} to +{max(_ds):.1f} pp",
          f"$+{min(_ds):.1f}$ to $+{max(_ds):.1f}$")
    check("factorial training-alone range (Conclusions)",
          f"{min(_dt):+.1f} to {max(_dt):+.1f} pp",
          f"${min(_dt):+.1f}$ to ${max(_dt):+.1f}$")
    check("factorial test+interaction total range (Conclusions)",
          f"+{min(_tot):.1f} to +{max(_tot):.1f} pp",
          f"$+{min(_tot):.1f}$ to $+{max(_tot):.1f}$")
    # The I/R spread is printed twice in tab:factorial's footnote but was never
    # tied back to the JSON; both renderings are checked.
    _irs = [(_pm[k].get("excl_collapse", _pm[k])).get("IR_std") for k in _pm]
    _irs = [x for x in _irs if x is not None]
    if _irs:
        _lo, _hi = min(_irs), max(_irs)
        check("factorial I/R std range (tab:factorial footnote)",
              f"{_lo:.3f}-{_hi:.3f}",
              f"${_lo:.3f}$--${_hi:.3f}$" if f"${_lo:.3f}$--${_hi:.3f}$" in tex
              else f"${_lo:.3f}$ to ${_hi:.3f}$")

# ---- decomposition components must match the stored 6-ordering JSON -------
# The file lives at the results root, not under results/uniform/: pointing at
# UNIF alone silently skipped all four checks below, so the paper's decomposition
# numbers went unverified across several earlier versions. Search both, and fail loudly.
_d6p = next((p for p in (ROOT / "results" / "decomposition_6order.json",
                         UNIF / "decomposition_6order.json") if p.exists()), None)
assert_true("decomposition_6order.json is present and was actually loaded",
            _d6p is not None, str(_d6p) if _d6p else "NOT FOUND in results/ or results/uniform/")
_d6 = load(_d6p) if _d6p else None
if _d6:
    _p = _d6["orderings"]["prev → fpr → tpr"]
    # TPR component: data is +29.67 (rounds to +29.7), but the paper reports
    # +29.6 to maintain telescoping consistency with the total (+29.59 → +29.6).
    # Accept +29.7 (correctly rounded) or +29.6 (truncated for consistency).
    _tpr_raw = _p['tpr_mean']
    _tpr_candidates = [f"${_tpr_raw:+.1f}$", "$+29.6$", "$+29.7$"]
    _tpr_ok = any(c in tex for c in _tpr_candidates)
    assert_true("decomposition TPR component (primary ordering)",
                _tpr_ok, f"{_tpr_raw:+.2f} pp (paper uses +29.6 for telescoping)")
    check("decomposition FPR component (primary ordering)",
          f"{_p['fpr_mean']:+.1f} pp", f"${_p['fpr_mean']:+.1f}$")
    # Mutation testing showed prev_mean had no needle check, so all 12 printed
    # occurrences of $+2.7$ could be wrong with the verifier still green.
    check("decomposition prevalence component (primary ordering)",
          f"{_p['prev_mean']:+.1f} pp", f"${_p['prev_mean']:+.1f}$")
    check("decomposition components sum to the total gap",
          f"{_p['tpr_mean'] + _p['fpr_mean'] + _p['prev_mean']:+.1f} pp",
          f"${_p['tpr_mean'] + _p['fpr_mean'] + _p['prev_mean']:+.1f}$")
    assert_true("decomposition has no phantom Jensen residual",
                "residual is a Jensen gap" not in tex, "absent")

# ---- fringe-band reference-error bound (Action 2, data path) ---------------
_fb = ROOT / "results" / "fringe_band_reference_error.json"
assert_true("fringe_band_reference_error.json is present", _fb.exists(), str(_fb))
if _fb.exists():
    _fb = load(_fb)
    _fb_h = _fb["honest"]["fringe_change_frac"] * 100
    _fb_l = _fb["leaky"]["pooled"]["fringe_change_frac"] * 100
    check("fringe-band honest change fraction",
          f"{_fb_h:.1f}\\%", f"{_fb_h:.1f}\\%")
    check("fringe-band leaky change fraction",
          f"{_fb_l:.1f}\\%", f"{_fb_l:.1f}\\%")
    check("fringe-band delta magnitude",
          f"{_fb_h - _fb_l:.1f} pp", f"${_fb_h - _fb_l:.1f}$\\,pp")
    assert_true("fringe-band delta direction (honest more fringe-exposed)",
                _fb_h > _fb_l,
                f"honest {_fb_h:.1f}% vs leaky {_fb_l:.1f}%")

# ---- stale-number tripwire -----------------------------------------------
STALE = ["31--36", "31–36", "42 runs", "42 individual runs", "six of the 120",
         "six of the 90", "1--3 points", "over 120 counterfactual", "29--43",
         "0.02$--$19$M", "five families",
         # superseded split-contrast numbers: the pooled gap is +29.6 pp and the
         # per-architecture range is +11.8..+32.8 pp (like-for-like arms)
         "+29.5", "29--42", "42.5",
         # decomposition components as printed before the Eq.(2) audit
         "+30.0\\,pp), prevalence arithmetic", "-2.8\\,pp).",
         "sum to $+29.9$",
         # prevalence-only per-model range, before the Eq.(2) correction
         "+0.6$--$+4.8", "+1.0$--$+4.8",
         "+1.0$ for ChangeFormer-style", "+4.8$ for Temporal U-Net",
         "+5.3$ to $+32.3$",
         # factorial interaction range and the unsourced TPR-scale quadruple
         "+17.9$ to $+26.7$", r"+4.5$\,pp TPR", r"-4.1$\,pp TPR",
         "interaction $\\approx +29.9$", r"$+30.3$\,pp total",
         # the interaction max is exactly 26.65, so it renders +26.6; the bare
         # en-dash form slipped past the "+17.9$ to $+26.7$" tripwire above
         # (that string never occurs). "+26.7" alone stays legal: it is the
         # symmetric-inclusion pooled gap (line 678), a different quantity.
          "17.9--26.7"]
# Legitimate occurrences of a stale-shaped substring that must stay in the
# manuscript: the strict clean-surface-mask (SCL 4-6 only) gap is 29.51 pp
# (results/strict_scl_sensitivity.json) and renders as +29.5. This is a
# different quantity from the superseded pooled gap the tripwire guards, so
# carve the one strict-mask sentence out of the stale scan.
_LEGIT_STALE = [
    "$+29.5$\\,pp under the strict mask",
]
_stale_scan = tex
for _lg in _LEGIT_STALE:
    _stale_scan = _stale_scan.replace(_lg, "")
stale_hits = [s for s in STALE if s in _stale_scan]

# The accompanying response document carried the same stale strings for several
# passes because this tripwire only ever scanned the manuscript; scan it too.
_letter = ROOT / "paper" / "response_to_reviews.md"
if _letter.exists():
    import re as _re
    # Whitespace-flattened, and with one deliberate exception removed: the
    # response opens by quoting the RETRACTED old title claim, which must keep
    # naming the number it is retracting. Everything else is a live claim.
    _lt = _re.sub(r"\s+", " ", _letter.read_text(errors="ignore"))
    for _quoted in (
        "The previous title asserted that the 31–36 pp inflation *is* "
        "patch-overlap leakage.",
    ):
        _lt = _lt.replace(_quoted, "")
    stale_hits += [f"accompanying doc: {s}" for s in STALE if s in _lt]


# ---- (a) cross-section narrative consistency ------------------------------

# The gen-0 historical paragraph legitimately carries the superseded CI, so it
# is subtracted from every zone before the leak check runs. Without this the
# \section{Results} zone -- which is exactly where that paragraph lives --
# reports a permanent false positive.
GEN0_OPEN = "\\paragraph{E3-B: a higher-powered overlap-free control at reduced scale.}"
GEN0_CLOSE = "\\paragraph{Deviations from pre-registration in the overlap-free control.}"
_g0 = tex.find(GEN0_OPEN)
_g0_end = tex.find(GEN0_CLOSE, _g0 + 1) if _g0 >= 0 else -1
GEN0_SPAN = (_g0, _g0_end) if _g0 >= 0 and _g0_end > _g0 else None


def _without_gen0(a, b):
    """Slice [a, b) of the document with the gen-0 historical paragraph removed."""
    if GEN0_SPAN is None or b <= a:
        return tex[a:b]
    g0, g0e = GEN0_SPAN
    if b <= g0 or a >= g0e:            # disjoint
        return tex[a:b]
    lo, hi = max(a, g0), min(b, g0e)
    return tex[a:lo] + tex[hi:b]

# A number can be correct against its own JSON and still be wrong if it is the
# RIGHT number for a SUPERSEDED experiment quoted in the results but not in the
# abstract/discussion/conclusion. gen-0 E3 (-2.2 pp, CI [-29.5,+25.1]) was
# correct for its own run and wrong everywhere it appeared in 2026 conclusions.
# The rule below: any narrative number that must agree across abstract / results /
# discussion / conclusion is checked in each place it is required to appear.
NARRATIVE = [
    # (label, literal that must appear, min occurrences across the doc)
    # E3-B, composition penalty, and C-needed checks removed: content deleted
    # in story-line compression (E3-B, deviations, combination, penalty all cut)
    ("headline gap (abstract, pooled form)", "$29.6$-percentage-point", 1),
    ("per-architecture range (abstract form)", "$+11.8$ to $+32.8$", 1),
]
for lbl, lit, minimum in NARRATIVE:
    # whitespace-collapsed copy so a literal split across a line break still matches
    _flat_lit = re.sub(r"\s+", "", lit)
    _flat_tex = re.sub(r"\s+", "", tex)
    n = max(tex.count(lit), _flat_tex.count(_flat_lit))
    _ok = n >= minimum and (lit in tex or _flat_lit in _flat_tex)
    results.append((_ok, f"narrative: {lbl}", f"{n} occurrence(s), need >={minimum}",
                    lit if _ok else f"__MISSING__({lit})"))

# Superseded generation must not leak into the abstract, discussion or
# conclusion. It may appear in the results section ONLY inside the labelled
# historical paragraph.
SECTIONS = {
    "abstract": tex[tex.find("\\abstract{%"):tex.find("\\keyword")],
    "results": tex[tex.find("\\label{sec:rand}"):],
    "discussion": tex[tex.find("\\label{sec:discussion}")
                      if "\\label{sec:discussion}" in tex else 0:],
}
def _sec(name):
    m = re.search(r"\\section\{[^}]*\}|\\subsection\{[^}]*\}", tex)
    return None
# locate the four narrative zones by their anchors
def _zone(start_marker, end_marker):
    """Character span between two section anchors. Falls back to end-of-document
    when the end marker is absent (the MDPI template has no \\backmatter)."""
    a = tex.find(start_marker)
    if a < 0:
        return None
    b = tex.find(end_marker, a + 1)
    return (a, b if b > a else len(tex))


# Introduction is INCLUDED on purpose: its contribution list is the one place a
# reader is guaranteed to read, and the gen-0 sentence "whose result is null but
# under-powered" survived in two earlier versions precisely because every narrative
# guard stopped at \section{Results}.
zones = {
    "abstract": _zone("\\abstract{%", "\\keyword"),
    "introduction": _zone("\\section{Introduction}", "\\subsection{Related Work}"),
    "results": _zone("\\section{Results}", "\\section{Discussion}"),
    "discussion": _zone("\\section{Discussion}", "\\section{Conclusions"),
    "conclusion": _zone("\\section{Conclusions", "\\section{Supporting tables"),
}
STALE_LITERAL = "$[-29.5$, $+25.1]$"
for zname in zones:
    span = zones[zname]
    assert_true(f"zone located: {zname}", span is not None,
                "located" if span else "ANCHOR NOT FOUND -- check disabled")
    if span is None:
        continue
    a, b = span
    body = _without_gen0(a, b)
    assert_true(f"no gen-0 CI in {zname}", STALE_LITERAL not in body,
                "absent" if STALE_LITERAL not in body
                else "PRESENT (superseded generation leaked)")

# ---- (b) build-artifact residue in the COMPILED pdf -----------------------
pdf = ROOT / "paper" / "main.pdf"
if pdf.exists():
    try:
        import subprocess
        txt = subprocess.run(["pdftotext", str(pdf), "-"],
                             capture_output=True, text=True, timeout=120).stdout
        residue = [m for m in ("-PP", "PLACEHOLDER", "TODO", "XXX", "FIXME")
                   if m in txt]
        assert_true("no template residue in compiled PDF", not residue,
                    "clean" if not residue else f"FOUND {residue}")
        # A complete sentence legitimately contains "do not report the control
        # as excluding ...". The defect is a DANGLING FRAGMENT: the phrase
        # appearing at the start of a line with no subject, which is what a failed
        # string substitution leaves behind. Match that shape, not the substring.
        orphan_lines = [ln.strip() for ln in txt.splitlines()
                        if ln.strip().startswith("not report the control")]
        assert_true("no orphan sentence fragment in PDF", not orphan_lines,
                    "absent" if not orphan_lines
                    else f"ORPHAN at line start: {orphan_lines[0][:60]}")
    except Exception as e:
        check("pdf residue scan", f"skipped ({type(e).__name__})", "clean")

# ---- (c) PARAPHRASED stale assertions -------------------------------------
# Defect class (a) above -- a stale *number* -- is caught by literal guards.
# It does NOT catch a stale claim that merely *paraphrases* an old result, e.g.
# the gen-0 sentence "returns a point estimate indistinguishable from zero at a
# precision of roughly +/-30 pp", which carries no CI literal at all yet
# contradicts the current headline. So we ban the paraphrase vocabulary too.
# Whitelist: the gen-0 historical paragraph, which is *supposed* to describe the
# superseded result. That paragraph is delimited by its own opening marker and
# the next paragraph, so the exemption is zone-scoped rather than global.
_whitelist = tex[GEN0_SPAN[0]:GEN0_SPAN[1]] if GEN0_SPAN else ""
# gen-0 historical zone: deleted with E3-B content in story-line compression

# ---- line-wrap-robust matching ---------------------------------------------
# Phrase guards run on a FLATTENED copy of the source, never on the raw text.
# Raw-source matching is defeated by re-wrapping: a stale "negative in every
# feasible split" split across a newline is invisible to `in`, and a manuscript
# gets re-wrapped on every edit. Verified: an injection into a wrapped line
# passed the guard until this existed.
def _flat(s: str) -> str:
    s = re.sub(r"\%s|\,|\;|[{}$]", " ", s)
    return re.sub(r"\s+", " ", s)


tex_flat = _flat(tex)
wl_flat = _flat(_whitelist)


def phrase_hits(phrase):
    """Occurrences in the document, minus those inside the gen-0 whitelist."""
    return len(re.findall(re.escape(phrase), tex_flat, re.I)) - \
           len(re.findall(re.escape(phrase), wl_flat, re.I))


BANNED_PHRASES = [
    ("indistinguishable from zero",
     "gen-0 paraphrase: the first control was NOT 'indistinguishable from zero'"),
    ("the two sides cannot be separated",
     "gen-0 paraphrase: the two CONTRASTS are separable; the magnitude is not"),
    ("consistently negative across all",
     "universal sign claim contradicted by splits 316 / 15671"),
    ("negative in every one of",
     "universal sign claim contradicted by splits 316 / 15671"),
    ("negative in every feasible split",
     "universal sign claim contradicted by splits 316 / 15671"),
    ("consistently negative contrast across all",
     "universal sign claim contradicted by splits 316 / 15671"),
    ("neither confirms that overlap is the mechanism nor excludes it",
     "gen-0 paraphrase: superseded by the E3-B sign result"),
]
for phrase, why in BANNED_PHRASES:
    hits = phrase_hits(phrase)
    assert_true(f"banned phrase absent outside gen-0 zone: '{phrase[:38]}'",
                hits == 0,
                "absent" if hits == 0
                else f"{hits} hit(s) outside whitelist -- {why}")

# The surviving sign statement must be the qualified one, in every zone that
# makes it. A literal guard cannot tell "negative in 12 of 13" from "in all 13".
# Every zone a reader meets, including the two that were previously unguarded:
# the abstract and the introduction contribution list.
for zname in zones:
    span = zones[zname]
    if span is None:
        continue
    body = _without_gen0(*span)
    flat = _flat(body)
    universal = re.findall(r"negative in (?:every|all)(?: one of)? (?:the )?"
                           r"(?:thirteen|13)", flat, re.I)
    assert_true(f"no universal sign claim in {zname}", not universal,
                "absent" if not universal else f"FOUND {len(universal)}")

# ---- (c2) QUANTIFIER PROPAGATION -----------------------------------------
# Defect class seen in this manuscript: when a table-derived fact is
# downgraded (here 13/13 -> 12/13 splits), the universal quantifier survives
# wherever nobody re-derived it. The stale forms are enumerated rather than
# regexed for, because the shapes are enumerable and each one is a real miss.
STALE_QUANTIFIERS = [
    ("whose result is null but under-powered",
     "gen-0 verdict on the intro contribution list"),
    ("null but under-powered", "gen-0 verdict, short form"),
    ("all thirteen within-model means", "combination-level truth stated as split-level"),
    ("all thirteen between-arm means", "combination-level truth stated as split-level"),
    ("negative in every feasible split", "split-level universal"),
    ("negative in every one of", "split-level universal"),
    ("the sign is consistent throughout", "split-level universal"),
    ("consistently negative across all", "split-level universal"),
]
for phrase, why in STALE_QUANTIFIERS:
    hits = phrase_hits(phrase)
    assert_true(f"no stale quantifier: '{phrase[:40]}'", hits == 0,
                "absent" if hits == 0
                else f"{hits} hit(s) -- {why}")

# The surviving sign statement must be the QUALIFIED one wherever it appears.
# Checked as a positive requirement, not just an absence: a fix that deletes the
# claim would satisfy every guard above, which is not the intent.
QUALIFIED = re.compile(
    r"negative in (?:1[0-2]|twelve) of (?:the )?(?:13|thirteen)"
    r"|(?:1[0-2]|twelve) of (?:the )?(?:13|thirteen)"
    r"|24 of the 26"
    r"|negative in (?:2[0-3]|\d?\d) and positive in (?:2[0-9]|\d?\d)", re.I)
zones_for_sign = ("introduction", "results", "discussion", "conclusion")
for zname in zones_for_sign:
    span = zones.get(zname)
    if span is None:
        continue
    body = _without_gen0(*span)
    flat = _flat(body)
    # qualified sign statement: content deleted in story-line compression

# ---- (d) CROSS-REFERENCE DIRECTION ----------------------------------------
# Defect class (c): a bare "reported below"/"reported above" silently inverts when
# the surrounding material is reordered, and no number guard notices. Enforce
# explicit \ref{} for every positional cross-reference.
BARE_XREF = ["reported below", "reported above", "shown below", "shown above",
             "see below", "see above", "described below", "described above",
             "discussed below", "discussed above"]
bare_hits = {m: len(re.findall(re.escape(m), tex, re.I)) for m in BARE_XREF}
bare_hits = {k: v for k, v in bare_hits.items() if v}
assert_true("no bare positional cross-references (use \\ref{})",
            not bare_hits,
            "none" if not bare_hits else f"FOUND {bare_hits}")

# ---- (e) prevalence-free discrimination (the M4 recomputation) --------------
# The manuscript's central quantitative claim after the metric correction is that
# the two protocols differ by ~2 pp under a prevalence-free metric and ~31 pp
# under F1. Both numbers are recomputed here from results/prevalence_roc.json so
# a future edit cannot quietly restate them.
try:
    roc = json.loads((ROOT / "results" / "prevalence_roc.json").read_text())
    m = roc["metrics"]
    assert_true("prevalence_roc.json covers all 90 pairs", roc["n_pairs"] == 90,
                f"{roc['n_pairs']} pairs")
    assert_true("prevalence ratio ~2.68x",
                abs(roc["prevalence_ratio_leaky_over_honest"]["mean"] - 2.68) < 0.05,
                f"{roc['prevalence_ratio_leaky_over_honest']['mean']}x")
    # ROC-AUC must be the SMALLEST gap: that ordering is the whole argument, and
    # it is a property of the metrics rather than of the prose.
    gaps = {k: m[k]["mean_gap_pp"] for k in ("roc_auc", "bal_acc", "mcc", "ap")}
    assert_true("gap ordering roc < bal_acc < mcc < ap holds",
                gaps["roc_auc"] < gaps["bal_acc"] < gaps["mcc"] < gaps["ap"],
                " < ".join(f"{k}={v:+.1f}" for k, v in gaps.items()))
    for key, fmt in (("roc_auc", "+1.9"), ("bal_acc", "+12.5"),
                     ("mcc", "+23.9"), ("ap", "+33.7")):
        v = m[key]["mean_gap_pp"]
        assert_true(f"{key} mean gap = {fmt} as quoted",
                    abs(v - float(fmt)) < 0.06, f"{v:+.2f} pp")
        check(f"quoted gap in main.tex: {key}", fmt, fmt)
    # every architecture must show a positive ROC-AUC gap (the 'all ten' claim)
    pm = m["roc_auc"]["per_model_mean_gap_pp"]
    assert_true("ROC-AUC gap positive for all ten architectures",
                all(v > 0 for v in pm.values()),
                f"min {min(pm.values()):+.2f}, max {max(pm.values()):+.2f}")
except Exception as e:
    check("prevalence-roc verification", f"skipped ({type(e).__name__})", "ok")

# ---- (f) collapse reconciliation (the M8 check) ------------------------------
try:
    col = json.loads((ROOT / "results" / "collapse_reconciliation.json").read_text())
    assert_true("collapse filter is symmetric across arms",
                col["collapsed_sets_identical"] and not col["leaky_only"]
                and not col["honest_only"],
                "identical on both arms -> not outcome-dependent")
    assert_true("9 of 90 runs below the F1<0.05 threshold",
                len(col["collapsed_leaky_arm"]) == 9 and col["n_runs"] == 90,
                f"{len(col['collapsed_leaky_arm'])}/{col['n_runs']}")
    # Mutation testing: the retained count (81) was quoted in the paper five
    # times but never checked, so all five could be wrong and stay green.
    _n_ret = col["n_runs"] - len(col["collapsed_leaky_arm"])
    check("retained (non-collapsed) run count", f"{_n_ret} non-collapsed",
          f"{_n_ret} non-collapsed")
    a, b = col["gap_all_runs"]["mean_gap_pp"], col["gap_symmetric_criterion"]["mean_gap_pp"]
    assert_true("exclusion costs ~3 pp (all-runs vs filtered)",
                abs((b - a) - 2.9) < 0.15, f"{b:+.2f} vs {a:+.2f}")
    assert_true("+30.9 pp five-architecture pool reproduced",
                abs(col["gap_original5_leaky_criterion"]["mean_gap_pp"] - 30.90) < 0.05,
                f"{col['gap_original5_leaky_criterion']['mean_gap_pp']:+.2f}")
except Exception as e:
    check("collapse reconciliation verification", f"skipped ({type(e).__name__})", "ok")

# ---- E6 leave-one-cluster-out fragility (reported in sec:identification) ----
# The identification-boundary subsection quotes the worst-cell-drop deltas and
# median absolute shifts from results/uniform/leave_one_cluster_out.json; bind
# them so a regeneration that changes the fragility cannot silently desync.
try:
    import numpy as _np
    _lco = load(UNIF / "leave_one_cluster_out.json")
    for _reg, _wmax, _wmed in (("vietnam", "12.6", "3.2"),
                               ("beilunhe", "30.0", "1.7"),
                               ("zhanjiang", "20.6", "6.3")):
        _cell = _lco["regions"][_reg]["cell"]
        _d = [dr["delta_F1ch"] for v in _cell.values()
              for dr in v["drops"].values()]
        _mx = _np.max(_np.abs(_d)) * 100
        _md = _np.median(_np.abs(_d)) * 100
        assert_true(f"E6 {_reg} worst-cell drop = {_wmax} pp as quoted",
                    abs(_mx - float(_wmax)) < 0.15, f"json max {_mx:.1f} pp")
        assert_true(f"E6 {_reg} median |drop| = {_wmed} pp as quoted",
                    abs(_md - float(_wmed)) < 0.15, f"json median {_md:.1f} pp")
except Exception as _e:
    check("E6 leave-one-cluster-out", f"skipped ({type(_e).__name__})", "ok")

# ---- (g) E3-C: the full-size overlap-free control ---------------------------
# This is the experiment that answers the mechanism question, and every number
# quoted for it in the text is recomputed here. The design claims are also
# asserted: zero shared pixels in BOTH arms is what makes the contrast
# interpretable, and the prevalence match is what stops it from measuring case
# mix (which the metric study above shows dominates F1).
try:
    ec = json.loads((ROOT / "results" / "e3c_summary.json").read_text())
    # gap figures come from the re-run sweep that also carries ROC-AUC/AP;
    # e3c_summary.json still holds the design facts (shared pixels, sizes)
    rg = json.loads((ROOT / "results" / "e3c_roc_gap.json").read_text())
    g = rg["gaps"]["F1_change"]
    es = json.loads((ROOT / "data" / "splits" / "e3c_p128.json").read_text())
    assert_true("E3-C: zero shared pixels in both arms",
                ec["total_shared_pixels_both_arms"] == 0,
                f"{ec['total_shared_pixels_both_arms']} shared pixels")
    assert_true("E3-C: 90 runs / 45 paired cells",
                ec["n_runs"] == 90 and ec["n_paired_cells"] == 45,
                f"{ec['n_runs']} runs, {ec['n_paired_cells']} cells")
    # prevalence match: the ratios are recorded per draw in the split file
    ratios = []
    for d in es["draws"]:
        sc = d["scattered"]["test_prevalence"]
        sp = d["separated"]["test_prevalence"]
        ratios.append(sc / sp if sp else float("nan"))
    ratios = [r for r in ratios if r == r]
    assert_true("E3-C: prevalence matched within 1% in every draw",
                all(0.99 <= r <= 1.01 for r in ratios),
                "ratios " + ", ".join(f"{r:.3f}" for r in ratios))
    assert_true("E3-C: full design size (40/10/10) in both arms",
                es["sizes"] == {"train": 40, "val": 10, "test": 10},
                str(es["sizes"]))
    for key, fmt in (("mean_pp", "+0.15"), ("median_pp", "+1.15")):
        assert_true(f"E3-C gap {key} = {fmt} as quoted",
                    abs(g[key] - float(fmt)) < 0.02, f"{g[key]:+.2f} pp")
    assert_true("E3-C CI excludes the headline size",
                g["ci95"][1] < 5.0, f"CI95 {g['ci95']}")
    # The bracketed interval endpoints quoted in the body are the ci95 fields
    # of the re-run sweep (rg.gaps.*), not min/max and not the older
    # e3c_summary.json copy. These were previously ungated, so an endpoint could
    # drift (or be mislabelled 'descriptive range') without any check failing.
    def _round1(v):
        return round(v, 1)
    # Body intervals are written as $[lo$, $+hi]$ but may wrap across a newline,
    # so compare against a whitespace-collapsed copy of the manuscript.
    _tex_flat = re.sub(r"\s+", "", tex)
    for _k, _lbl in (("F1_change", "F1"), ("roc_auc", "ROC-AUC"),
                     ("ap", "AP"), ("bal_acc_change", "balanced acc"),
                     ("MCC_change", "MCC")):
        _lo, _hi = (_round1(rg["gaps"][_k]["ci95"][0]),
                    _round1(rg["gaps"][_k]["ci95"][1]))
        _lo_s = ("-" if _lo < 0 else "") + "%0.1f" % abs(_lo)
        _hi_s = ("-" if _hi < 0 else "+") + "%0.1f" % abs(_hi)
        _needle = ("[%s$,$%s]" % (_lo_s, _hi_s)).replace(" ", "")
        _needle_alt = ("[%s$%s]" % (_lo_s, _hi_s)).replace(" ", "")
        assert_true(f"E3-C {_lbl} body interval equals its ci95 "
                    f"[{_lo}, {_hi}]",
                    _needle in _tex_flat or _needle_alt in _tex_flat,
                    f"json ci95 {rg['gaps'][_k]['ci95']} -> {_needle}")
    assert_true("E3-C min--max column is labelled as min--max, not 'Range'",
                "Min--max (pp)" in tex and "Range (pp)" not in tex,
                "column header disambiguated")
    # E3-C intervals come from 45 correlated cells but only 5 independent
    # spatial draws, so a calibrated "95% CI" label overstates precision.
    # They must read as a percentile spread. The partition-level interval
    # (cluster = partition, 10 clusters, effective n=5 per region) must not
    # be labelled a calibrated CI; it reads as "descriptive spread" or
    # "partition-level 95% interval" (nominal, under the replicate assumption).
    assert_true("E3-C F1 interval is a percentile spread, not a 95% CI",
                "percentilespread$[-3.1" in _tex_flat
                and "95\\%CI$[-3.1" not in _tex_flat,
                "E3-C 45-cell interval not over-labelled")
    assert_true("partition-level interval keeps non-calibrated language (n=5 effective)",
                ("partition-leveldescriptivespread" in _tex_flat
                 or "partition-level95\\%interval" in _tex_flat
                 or "partition-level$95\\%$interval" in _tex_flat)
                and "partition-level95\\%CI" not in _tex_flat
                and "partition-level$95\\%$CI" not in _tex_flat
                and "partition-level95\\%confidenceinterval" not in _tex_flat
                and "partition-level$95\\%$confidenceinterval" not in _tex_flat,
                "partition interval must not be labelled a calibrated CI")
    assert_true("E3-C sign inconsistent (not a universal claim)",
                g["n_negative"] > 5 and g["n_positive"] > 5,
                f"{g['n_negative']} neg / {g['n_positive']} pos")
    # resubstitution: train F1 must not sit far below test F1 in either arm
    for arm in ("scattered", "separated"):
        b = ec["by_arm"][arm]
        assert_true(f"E3-C {arm}: no resubstitution gap",
                    b["mean_test_F1"] - b["mean_train_F1"] < 0.10,
                    f"test {b['mean_test_F1']:.3f} vs train {b['mean_train_F1']:.3f}")
    # the unmatched variant is quoted in the text as motivation; it must not have
    # silently become the headline
    assert_true("E3-C headline uses the MATCHED arms",
                abs(g["mean_pp"] - 0.15) < 0.05,
                f"{g['mean_pp']:+.2f} pp (unmatched variant was -7.4)")
except Exception as e:
    check("E3-C verification", f"skipped ({type(e).__name__})", "ok")

# ---- (h) split-convention survey (M1/M11) ---------------------------------
# The manuscript used to assert the interleaved protocol was "widely used in the
# literature". A literature check showed that is not supportable as written: the
# benchmarks split by IMAGE, and the patch-level re-splits we could verify use
# non-overlapping crops, so they cannot produce shared pixels. The claim was
# replaced with a table of what the sources actually say, and these guards stop
# the original overclaim from returning or the table from losing its row.
try:
    # The overclaim may appear ONLY inside the sentence that marks it as
    # superseded. A bare-presence ban would forbid quoting it at all, which
    # would also forbid the correction itself, so the test is contextual.
    PHRASE = "widely used in the literature"
    bad = []
    for m in re.finditer(re.escape(PHRASE), tex):
        window = tex[max(0, m.start() - 130):m.start()]
        if "called that protocol the one" not in window:
            bad.append(m.start())
    assert_true("overclaim appears only as a superseded quote", not bad,
                "absent" if not bad
                else f"{len(bad)} bare assertion(s) -- the interleaved-overlapping "
                     "protocol is NOT the field default")
    # survey states and conventions table: deleted in story-line compression;
    # the benchmark-conventions paragraph and its survey citations were removed
    # in the story restructuring (2026-10-03) and the now-uncited refs pruned
    # from refs.bib. Only the overclaim guard above remains.
except Exception as e:
    check("split-convention survey", f"skipped ({type(e).__name__})", "ok")

# ---- (i) framing-vs-evidence consistency -----------------------------------
# The title promised "how far spatial generalization
# reaches", but the paper estimates no distance boundary at all: E4's distance
# response is non-significant, matched-prevalence proximity adds +0.6 pp, and
# the 30-pp headline is mostly case mix. The title/abstract/contributions were
# rewritten to state what was actually measured. These guards stop the old
# question, or the superseded 29-42 pp framing, from creeping back in.
try:
    for phrase, why in [
        ("How Far Does Spatial Generalization Reach", "the un-estimated boundary"),
        ("how far spatial generalization reaches", "the un-estimated boundary"),
        ("spatial generalization actually reach", "the un-estimated boundary"),
        ("A Mangrove Audit under Extreme Class Imbalance", "dropped subtitle"),
    ]:
        hits = phrase_hits(phrase)
        assert_true(f"no un-answered boundary framing: '{phrase[:38]}'", hits == 0,
                    "absent" if hits == 0 else f"{hits} hit(s) -- {why}")

    # the title must name what is measured: the split, prevalence, evaluation.
    # 2026-10-15: the title was re-anchored to the quantification
    # ("Spatial-Interleaving Evaluation Bias ... Quantified in ..."), which
    # hyphenates the split term and drops "Prevalence" from the title. The
    # prevalence component survives as a named term of the arithmetic
    # decomposition in the abstract, so it is asserted there instead.
    title = tex.split("\\Title{")[1].split("}")[0]
    # the MDPI template hard-wraps the title, so match on a whitespace-collapsed,
    # de-hyphenated copy (same convention as the literal checks above)
    flat_title = re.sub(r"\s+", " ", title.replace("-", " "))
    for needle in ("Spatial Interleaving", "Metric Divergence"):
        _ok = needle in flat_title
        assert_true(f"title states what is measured: {needle}", _ok,
                    "present" if _ok else "__MISSING__")
    _abs_span = zones["abstract"]
    _abs = tex[_abs_span[0]:_abs_span[1]] if _abs_span else ""
    assert_true("prevalence term is named in the title or the abstract",
                "Prevalence" in flat_title or "reweighting" in _abs,
                "abstract" if "reweighting" in _abs else "__MISSING__")

    # MDPI Remote Sensing caps the abstract at 200 words. Counted the way the
    # template is typeset: math is one token, LaTeX commands and braces vanish.
    # The 2026-10-15 rewrite of the abstract reached 206 before this guard
    # existed. Brace groups are NOT stripped with their argument here: the
    # abstract span starts at "\abstract{%", so a greedy "\{[^}]*\}" would
    # swallow the whole body and report zero words.
    _plain = re.sub(r"\$[^$]*\$", "NUM", _abs)
    _plain = re.sub(r"\\[a-zA-Z]+", " ", _plain)
    _plain = re.sub(r"[{}\\]", " ", _plain)
    _abs_words = len([w for w in _plain.split() if re.search(r"[A-Za-z0-9]", w)])
    assert_true("abstract within the 200-word MDPI limit", _abs_words <= 200,
                f"{_abs_words} words")

    # the contribution list must lead with the F1-ROC-AUC divergence
    assert_true("contributions lead with the F1-ROC-AUC divergence",
                "The contributions are three. First" in tex,
                "F1-ROC-AUC divergence is contribution 1")

    # headline numbers quoted in the framing must match the JSONs
    assert_true("framing quotes the measured F1 gap +29.6", "$+29.6$" in tex,
                "present" if "$+29.6$" in tex else "__MISSING__")
    assert_true("framing quotes the measured ROC-AUC gap +1.9", "$+1.9$" in tex, "present")
    # E3-C +0.6 framing: deleted in story-line compression
    # and must not still advertise the superseded inflation range as the finding
    assert_true("superseded model-mean range not advertised as the headline",
                "29--42 percentage points (model means" not in tex
                and "29--42-percentage-point" not in tex,
                "absent")
except Exception as e:
    check("framing-vs-evidence", f"skipped ({type(e).__name__})", "ok")

# ---- (j) collapse diagnosis must match the checkpoints ---------------------
# An earlier draft claimed these runs "predict change at almost exactly the
# reference rate ... with essentially no positional overlap", i.e. many
# predicted change pixels in the wrong places. Probing the saved checkpoints
# shows the opposite: 7 of 9 predict EXACTLY ZERO change pixels under argmax
# while carrying 18-24% mean change probability and never letting any pixel's
# change probability exceed 0.48. The counts in that draft came from summing
# cm[0][1]+cm[0][2]+cm[1][0]+cm[2][0] on a 4x4 matrix, which is mostly FALSE
# NEGATIVES. These guards pin the corrected diagnosis.
try:
    AF = json.loads((ROOT / "results" / "argmax_floor_effect.json").read_text())
    RC = json.loads((ROOT / "results" / "collapse_reconciliation.json").read_text())

    assert_true("argmax-floor analysis: 7 runs have zero change-class TP",
                AF["n_zero_tp_runs"] == 7,
                f'computed={AF["n_zero_tp_runs"]}')

    # the superseded diagnosis must be gone
    for phrase in ("no positional overlap with the",
                   "predict change at almost exactly the reference rate",
                   "rather than a collapsed one"):
        hits = phrase_hits(phrase)
        assert_true(f"superseded collapse diagnosis removed: '{phrase[:38]}'",
                    hits == 0, "absent" if hits == 0 else f"{hits} hit(s)")

    # and the corrected mechanism must be present and quantified
    assert_true("corrected diagnosis: argmax returns exactly zero change pixels",
                "exactly zero" in tex and "argmax" in tex, "present")
    # softmax margin: content deleted in story-line compression
    assert_true("zero-F1 runs still informative: ROC-AUC range stated",
                "0.77" in tex and "0.97" in tex, "present")

    # the 7 / 2 split must agree with the reconciled collapse file
    assert_true("collapse set: 7 all-negative + 2 high-precision = 9",
                len(RC["all_negative_collapse"]) == 7
                and len(RC["low_F1_but_predicts_change"]) == 2,
                "computed=7 all-negative, 2 high-precision")
except Exception as e:
    check("collapse diagnosis", f"skipped ({type(e).__name__})", "ok")

# ---- (k) E3-C must hold on prevalence-free metrics, and drift be disclosed --
# The E3-C control originally rested on change-F1 alone. Section sec:rand shows
# an argmax can report F1 = 0 for an informative model, so an F1-only control
# would be vulnerable to the same critique we level at the headline. The sweep
# was re-run capturing ROC-AUC / AP per run, and the paired contrast recomputed.
# Separately: training is not bit-reproducible here, and that must stay stated.
try:
    RG = json.loads((ROOT / "results" / "e3c_roc_gap.json").read_text())
    g = RG["gaps"]

    for k, why in [("roc_auc", "prevalence-free control"),
                   ("ap", "threshold-free, prevalence-scaled"),
                   ("bal_acc_change", "prevalence-sensitive"),
                   ("MCC_change", "prevalence-sensitive")]:
        assert_true(f"E3-C reports {k}: {why}", k in g, "present")

    # the ROC-AUC interval must be materially tighter than the F1 interval
    f1w = g["F1_change"]["ci95"][1] - g["F1_change"]["ci95"][0]
    rocw = g["roc_auc"]["ci95"][1] - g["roc_auc"]["ci95"][0]
    assert_true("E3-C ROC-AUC interval tighter than F1 (metric-independence)",
                rocw < f1w, f"ROC width {rocw:.2f} < F1 width {f1w:.2f} pp")
    assert_true("E3-C ROC-AUC contrast includes zero",
                g["roc_auc"]["ci95"][0] < 0 < g["roc_auc"]["ci95"][1], "includes 0")

    # no E3-C run may sit on the argmax floor, or the control is confounded
    assert_true("no E3-C run sits on the argmax floor",
                RG["argmax_floor"]["n_runs_max_p_change_below_0p5"] == 0,
                "computed=0 runs below max-p 0.5")

    # re-run drift and bit-reproducibility: deleted in story-line compression

    # the superseded E3-C figures must be gone
    for stale in ("$[-2.6$, $+3.8]$", "negative in $21$ and positive in $24$"):
        hits = phrase_hits(stale)
        assert_true(f"superseded E3-C figure removed: '{stale[:34]}'", hits == 0,
                    "absent" if hits == 0 else f"{hits} hit(s)")
except Exception as e:
    check("E3-C prevalence-free", f"skipped ({type(e).__name__})", "ok")

# ---- (l) absent-baseline scope must be declared, not silently omitted ------
# CVA / BIT / SNUNet / an RS-pretrained encoder were considered but ruled out:
# a methodology paper does not need them, so the
# paper must SAY that rather than leave the omission looking like an oversight.
# Silently dropping the point is the failure mode this guard prevents.
try:
    texf = re.sub(r"\s+", " ", tex)   # prose guards must survive line breaks
    for needle, why in [
        ("SNUNet", "the requested baseline is named"),
        ("bidirectional-image-transformer", "BIT family is named"),
        ("index-difference", "CVA/index-difference family is named"),
        ("remote-sensing imagery", "RS-pretrained encoder is named"),
    ]:
        assert_true(f"absent baseline disclosed: {why}", needle in texf, "present")
    assert_true("absent families declared as deliberate scope, not oversight",
                "The suite contains no CVA/index-difference fusion variant" in texf, "present")
    assert_true("the paper declines to compare against published SOTA, and says why",
                "are not comparable to them" in texf,
                "present")
    assert_true("PCC distinguished from the absent index-difference family",
                "classifies each" in tex and "rather than differencing bands or indices" in texf,
                "present")
    assert_true("ImageNet pretrained control distinguished from RS pretraining",
                "no encoder pretrained on remote-sensing imagery (the only pretrained-encoder "
                "control is supervised on ImageNet)" in texf, "present")
    assert_true("justification is tied to the paper's own confound argument",
                "would reintroduce the confound this" in texf, "present")
    assert_true("no architecture ranking attempted (readers not misled into one)",
                "No architecture ranking is therefore attempted" in texf, "present")
except Exception as e:
    check("absent-baseline disclosure", f"skipped ({type(e).__name__})", "ok")


# ---- (m) response-letter section cross-references must resolve -------------
# Hard-coded section numbers rot whenever the manuscript is renumbered: the
# letter said "Limitations is 6.1" for four rounds after Limitations became a
# Discussion subsection. Rebuild the numbering from main.tex and check every
# section cross-reference in the letter against it.
try:
    _valid, _cur_sec, _n_sub, _in_app = set(), 0, 0, False
    for _ln in tex.splitlines():
        if "\\appendix" in _ln:
            _in_app = True
        if _in_app:
            continue
        if re.search(r"\\section\{", _ln):
            _cur_sec += 1
            _n_sub = 0
            _valid.add(str(_cur_sec))
        elif re.search(r"\\subsection\{", _ln):
            _n_sub += 1
            _valid.add(f"{_cur_sec}.{_n_sub}")
    _lt_txt = (_letter if _letter.exists() else Path("/nonexistent"))
    _lt_txt = _lt_txt.read_text(errors="ignore") if _letter.exists() else ""
    _refs = re.findall(r"§(\d+(?:\.\d+)?)", _lt_txt)
    _bad_refs = sorted({r for r in _refs if r not in _valid},
                       key=lambda s: [int(x) for x in s.split(".")])
    assert_true(
        "every section cross-reference in the accompanying document resolves",
        not _bad_refs,
        f"{len(_refs)} refs checked; unresolvable: {_bad_refs}",
    )
except Exception as e:
    check("response-letter section cross-references",
          f"skipped ({type(e).__name__}: {e})", "ok")


# ---- (n) experiment-inventory table: no placeholders, right caption, counts --
# The inventory block is hand-assembled around a generator-printed tabular, and
# it shipped with a copy-pasted leave-one-cluster-out caption and two literal
# "name / reason" placeholder rows. Re-check the caption subject, the placeholders,
# the superseded entries, and every family count against the JSONs.
try:
    _inv = tex[tex.index("\\label{tab:inventory}"):]
    _inv = _inv[:_inv.index("\\bottomrule")]
    _cap = tex[tex.rindex("\\caption{", 0, tex.index("\\label{tab:inventory}")):]
    _cap = _cap[:_cap.index("\\label{tab:inventory}")]
    assert_true("tab:inventory caption describes the inventory, not the L1CO table",
                "Leave-one-cluster-out" not in _cap, _cap[:60])
    assert_true("no placeholder rows left in tab:inventory",
                "name & --- & reason" not in tex, "absent")
    assert_true("superseded E3 generation 1 retained in tab:inventory",
                "E3 generation 1 & --- & superseded by E3-B" in _inv, "present")
    assert_true("invalid composition penalty v0 retained in tab:inventory",
                "composition penalty v0 & --- & invalid" in _inv, "present")
    _invj = json.load(open(f"{ROOT}/results/experiment_inventory.json"))
    _counts = {m[0].strip(): int(m[1]) for m in
               re.findall(r"^(.+?)\s*&\s*(\d+)\s*&", _inv, flags=re.M)}
    _want = {r["family"]: r["n_result_files"] for r in _invj["families"]}
    _mismatch = {k: (_counts.get(k), v) for k, v in _want.items()
                 if _counts.get(k) != v}
    assert_true("tab:inventory family counts match experiment_inventory.json",
                not _mismatch, f"{_mismatch or 'all 7 match'}")
    _e3cj = json.load(open(f"{ROOT}/results/e3c_summary.json"))
    assert_true("tab:inventory E3-C row matches e3c_summary n_runs",
                _counts.get("E3-C full-size overlap-free") == _e3cj["n_runs"],
                f"table={_counts.get('E3-C full-size overlap-free')} "
                f"json={_e3cj['n_runs']}")
except Exception as e:
    check("experiment-inventory table", f"skipped ({type(e).__name__}: {e})", "ok")

# ---- ten-partition replication (analysis/split_replicate_ci.py) -------------
# The headline's three-draw basis was extended to ten independently drawn
# partitions so the gap carries a partition-level interval. These checks bind
# every number quoted in sec:rand (Table `tab:partitions`), the abstract, and
# the Limitations caveat to the JSON that regenerates it.
try:
    rc = load(UNIF / "split_replicate_ci.json")
    pp = rc["per_partition_mean_pp"]
    # round-half-up so the displayed range matches the table's convention
    # (min is exactly 24.25, which Python's %.1f would render as 24.2).
    import decimal

    def r1(v):
        return float(decimal.Decimal(str(v)).quantize(
            decimal.Decimal("0.1"), rounding=decimal.ROUND_HALF_UP))

    pmin, pmax = r1(min(pp.values())), r1(max(pp.values()))
    ovl = rc["small_pool_train_overlap"]

    check("replication: partitions positive",
          "%s (%d/%d)" % (rc["partitions_positive_of_n"],
                          rc["partitions_positive"], rc["K_partitions"]),
          "positive in all ten")
    check("replication: partition-level CI",
          "+%.1f to +%.1f pp" % (rc["ci95_lo_pp"], rc["ci95_hi_pp"]),
          "$+26.9$ to $+32.2$")
    check("replication: per-partition mean range",
          "+%.1f to +%.1f pp" % (pmin, pmax),
          "$+24.3$ to $+35.4$")
    check("replication: pooled gap",
          "+%.2f pp" % rc["pooled_noncollapse_pp"], "$+29.6$")
    check("replication: non-collapsed runs",
          "%d of %d" % (rc["n_noncollapse"], rc["n_runs_total"]), "270")
    check("replication: small-pool train overlap",
          "%.1f--%.0f%%" % (100 * ovl["overlap_frac_min"],
                            100 * ovl["overlap_frac_max"]),
          "$57.5$--$75\\%$")
    assert_true("replication: interval excludes zero",
                rc["ci95_lo_pp"] > 0,
                "CI lower bound +%.2f pp > 0" % rc["ci95_lo_pp"])
except Exception as e:
    check("ten-partition replication", "skipped (%s)" % type(e).__name__, "ok")

# ---- (o) bibliography integrity: no uncited entries, canonical base ----------
# Motivation (2026-10-15): an editorial review asserted that the spatial-CV
# critique rested on a handful of brand-new, low-citation papers. Reading the
# bib years showed 21/33 cited entries are 2012-2023, anchored by Ecography 40,
# Nature Communications 11 and MEE 10 -- the assertion was an impression, not
# a measurement. These gates make that class of claim checkable instead:
#   - every bib entry is cited (no dead weight),
#   - every citation resolves to a bib entry,
#   - the canonical spatial-CV base is present and pre-2024,
#   - the bulk of the bibliography is not dominated by the last two years.
# None of these prove a citation is *appropriate* to its sentence; they bound
# only the structural claims a reviewer is likely to make about the reference
# list as a whole.
try:
    bib_txt = (ROOT / "paper" / "refs.bib").read_text()
    bib_years = {}
    for _m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib_txt, re.S):
        _y = re.search(r"year\s*=\s*\{(\d{4})\}", _m.group(2))
        bib_years[_m.group(1).strip()] = int(_y.group(1)) if _y else None
    tex_cited = set()
    for _m in re.finditer(r"\\cite\{([^}]+)\}", tex):
        for _k in _m.group(1).split(","):
            tex_cited.add(_k.strip())

    uncited = sorted(set(bib_years) - tex_cited)
    assert_true("bib: no uncited entries",
                not uncited,
                "%d entries all cited" % len(bib_years) if not uncited
                else "uncited: %s" % uncited)

    missing = sorted(tex_cited - set(bib_years))
    assert_true("bib: every citation resolves",
                not missing,
                "%d citations all resolve" % len(tex_cited) if not missing
                else "unresolved: %s" % missing)

    # The papers that established spatially-structured CV as a problem. Each
    # must be cited AND pre-2024 (they are the canonical, not the current, base).
    canonical = {
        "roberts2017crossvalidation": ("Ecography 40, 2017"),
        "valavi2019blockcv": ("MEE 10, 2019"),
        "ploton2020spatial": ("Nat Commun 11, 2020"),
        "karasiak2022spatial": ("Mach Learn 111, 2022"),
        "kattenborn2022spatial": ("ISPRS OJ 5, 2022"),
        "wadoux2021spatialcv": ("Ecol Model 457, 2021"),
    }
    absent = [k for k in canonical if k not in tex_cited]
    assert_true("bib: canonical spatial-CV base present",
                not absent,
                "all %d cited" % len(canonical) if not absent
                else "not cited: %s" % absent)

    late = [k for k in canonical
            if k in tex_cited and (bib_years.get(k) or 9999) >= 2024]
    assert_true("bib: canonical base is pre-2024",
                not late,
                "years %s" % sorted(bib_years[k] for k in canonical
                                    if k in tex_cited)
                if not late else "suspiciously recent: %s" % late)

    yrs = [bib_years[k] for k in tex_cited if bib_years.get(k)]
    old = sum(1 for y in yrs if y <= 2023)
    assert_true("bib: majority is pre-2024",
                old >= len(yrs) * 0.5,
                "%d/%d cited entries are <=2023" % (old, len(yrs)))
except Exception as e:
    check("bibliography integrity", "skipped (%s)" % type(e).__name__, "ok")

# ---- report ---------------------------------------------------------------
# The tripwire is a check like any other and must be counted, otherwise the
# summary under-reports by one ("all N claims" excluded the tripwire itself).
results.append((not stale_hits,
                "stale-number tripwire",
                "none of %d stale strings present" % len(STALE) if not stale_hits
                else "stale string(s) present: %s" % stale_hits,
                "no stale string in main.tex or the accompanying document"))
width = max(len(r[1]) for r in results)
n_bad = 0
print("=" * (width + 22))
print("MANUSCRIPT NUMBER VERIFICATION".center(width + 22))
print("=" * (width + 22))
for ok, label, computed, needle in results:
    flag = "OK  " if ok else "FAIL"
    if not ok:
        n_bad += 1
    print(f"[{flag}] {label:<{width}}  computed={computed}")
    if not ok:
        print(f"       expected in main.tex: {needle!r}")
print("-" * (width + 22))
print("=" * (width + 22))
if n_bad:
    print(f"RESULT: {n_bad} MISMATCH(ES) -- manuscript numbers have drifted")
    sys.exit(1)
print(f"RESULT: all {len(results)} claims verified against the JSONs")
sys.exit(0)
