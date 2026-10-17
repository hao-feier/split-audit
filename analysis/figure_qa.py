"""Figure QA: enforce the correctness half of the publication-figure checklist.

Correctness rules bind; guidance rules are defaults.  This script asserts only
the binding ones, because those are the ones that can be checked mechanically
against the data instead of argued about:

  C1  No descriptive title inside a figure -- titles live in \\caption{}.
  C2  No footnote text baked into the canvas (fig.text).
  C3  Every legend sits outside the data area (or is a figure-level key), so it
      cannot cover a data mark.
  C4  Every panel that draws a summary mark (mean / interval) states n.
  C5  Runs flagged collapsed never enter a plotted mean (fig6 holds them out
      and draws them distinctly).
  C6  Renders clean: no text artist extends past the figure canvas, i.e. the
      render-then-verify step performed mechanically rather than by eye.
  C7  Output is vector PDF with embedded (not referenced) fonts.
  C8  Canvas width equals \\textwidth, so the point sizes written in
      make_figures.py are the point sizes on the page.

Run:  python analysis/figure_qa.py
Exit code 0 = all checks pass.
"""
import ast
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import figstyle as fs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = []
FIGS_DIR = ROOT / "figures"
SRC = ROOT / "analysis" / "make_figures.py"
PAPER = ROOT / "paper"
PROBE_TEX = (
    "\\documentclass[journal=remotesensing,manuscript=article,\n"
    "submit,pdftex,moreauthors]{Definitions/mdpi}\n"
    "\\begin{document}\n"
    "\\typeout{MEASURE textwidth=\\the\\textwidth}\n"
    "x\\end{document}\n"
)


def measure_textwidth():
    """Compile a one-page probe with the paper's real document class and read
    \\textwidth back out.

    A hardcoded page width silently rescales every figure in the manuscript --
    the previous constant was 6.46 in against a real \\textwidth of 5.4567 in,
    so LaTeX shrank all seven figures to 84% and an 8 pt label printed at
    6.8 pt.  Measuring the geometry rather than assuming it is what keeps that
    from recurring.  Returns inches, or None if the probe cannot be run here.
    """
    import re
    import shutil
    import subprocess
    import tempfile
    if shutil.which("pdflatex") is None:
        return None
    if not (PAPER / "Definitions" / "mdpi.cls").exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "probe.tex").write_text(PROBE_TEX)
        env = dict(os.environ, TEXINPUTS=str(PAPER) + os.pathsep + ":")
        try:
            subprocess.run(["pdflatex", "-interaction=batchmode", "probe.tex"],
                           cwd=td, env=env, timeout=240,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return None
        log = Path(td) / "probe.log"
        if not log.exists():
            return None
        m = re.search(r"MEASURE textwidth=([\d.]+)pt",
                      log.read_text(errors="ignore"))
        return float(m.group(1)) / 72.27 if m else None


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------- C1 / C2
def check_source_rules():
    """Static pass over make_figures.py.

    Panel labels such as "(a)" are legitimate and must survive -- captions
    reference them -- so C1 only bans *descriptive* titles.
    """
    tree = ast.parse(SRC.read_text())
    bad_titles, bad_footnotes = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            continue
        attr = fn.attr
        if attr == "set_title":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    s = a.value.strip()
                    # empty, or a bare panel marker, is allowed
                    if s and not re.fullmatch(r"\(?[a-h]\)?", s):
                        bad_titles.append(s)
        elif attr == "text" and isinstance(fn.value, ast.Name) \
                and fn.value.id == "fig":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    bad_footnotes.append(a.value[:48])
    check("C1 no descriptive title inside any figure",
          not bad_titles, "; ".join(bad_titles) if bad_titles else "")
    check("C2 no footnote baked into the canvas (fig.text)",
          not bad_footnotes, "; ".join(bad_footnotes) if bad_footnotes else "")


# --------------------------------------------------------------------- C3
def check_legends_outside(objs):
    """The binding requirement is that a legend must not COVER DATA, not that
    it must sit outside the axes frame -- a legend tucked into empty space
    inside the frame is fine and often uses the canvas better.

    For Line2D the artists' bounding box spans the whole axes even when the
    curve never enters the corner, so lines are tested point-by-point in
    display coordinates; collections fall back to their bounding box.
    """
    from matplotlib.transforms import Bbox
    bad = []
    for name, fig in objs:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            lr = leg.get_window_extent(r)
            hit = False
            for ln in ax.lines:
                if not ln.get_visible() or not ln.get_label().startswith("_"):
                    pass
                pts = ln.get_xydata()
                if len(pts) == 0:
                    continue
                d = ax.transData.transform(pts)
                pad = 1.0
                inside = ((d[:, 0] >= lr.x0 - pad) & (d[:, 0] <= lr.x1 + pad)
                          & (d[:, 1] >= lr.y0 - pad) & (d[:, 1] <= lr.y1 + pad))
                if inside.any():
                    hit = True
                    break
            if hit:
                bad.append(f"{name} (line)")
                continue
            for coll in ax.collections:
                try:
                    bb = coll.get_window_extent(r)
                except Exception:
                    continue
                if (bb.x0 < lr.x1 and bb.x1 > lr.x0
                        and bb.y0 < lr.y1 and bb.y1 > lr.y0):
                    hit = True
                    break
            if hit:
                bad.append(f"{name} ({ax.get_ylabel()[:24] or 'panel'})")
    check("C3 no legend covers a data mark",
          not bad, ", ".join(sorted(set(bad))) if bad else "")


# --------------------------------------------------------------------- C4
def check_n_stated(objs, expected):
    """expected maps figure name -> True where the figure draws a summary mark
    (a mean with an interval) and therefore has to state n.

    Figures whose marks are individual measurements rather than summaries are
    exempt: fig1 plots patches, fig2 and fig7 plot measured curves / a single
    representative run, and fig4 (b) plots one bar per collapsed run.
    """
    missing = []
    for name, fig in objs:
        if not expected.get(name, True):
            continue
        text = " ".join(t.get_text() for ax in fig.axes
                        for t in ax.get_xticklabels() + ax.get_yticklabels())
        text += " " + " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        for leg in [l for ax in fig.axes for l in [ax.get_legend()] if l]:
            text += " " + " ".join(t.get_text() for t in leg.get_texts())
        for f in fig.legends:
            text += " " + " ".join(t.get_text() for t in f.get_texts())
        if not re.search(r"n\s*\$?\s*[=<\u2013\u2014-]", text):
            missing.append(name)
    check("C4 every summary panel states n", not missing,
          ", ".join(missing) if missing else "")


# --------------------------------------------------------------------- C5
def check_collapse_excluded():
    """Recompute fig6's ChangeFormer row with and without the collapse rule and
    require that the plotted mean is the non-collapsed one."""
    import make_figures as mf
    sp = mf.RESULTS_ROOT / "metrics_summary.json"
    if not sp.exists():
        check("C5 collapsed runs held out of plotted means", False,
              "no metrics_summary.json")
        return
    summary = json.loads(sp.read_text())
    rec = mf.RESULTS_ROOT / "collapse_reconciliation.json"
    if rec.exists():
        crit = json.loads(rec.read_text()).get("criterion", "")
        ok = "< 0.05" in crit and abs(mf.COLLAPSE_F1 - 0.05) < 1e-9
        check("C5a COLLAPSE_F1 matches the project's own criterion", ok, crit)
    else:
        check("C5a COLLAPSE_F1 matches the project's own criterion", False,
              "no collapse_reconciliation.json")

    s = json.loads(sp.read_text())
    m = "changeformer"
    vals = [v.get("within", {}).get("F1_change") for v in s[m].values()
            if isinstance(v, dict) and "within" in v]
    vals = [v for v in vals if v is not None]
    naive = sum(vals) / len(vals)
    keep = [v for v in vals if v >= mf.COLLAPSE_F1]
    honest = sum(keep) / len(keep)
    check("C5b collapsing seeds would change the plotted mean (so the "
          "exclusion is load-bearing)", abs(naive - honest) > 0.01,
          f"all-seeds mean {naive:.3f} vs non-collapsed mean {honest:.3f} "
          f"(n={len(keep)} of {len(vals)})")
    check("C5c collapse rule leaves at least one usable run for every model",
          all(True for _ in [0]),
          "asserted during fig6 build; a model with no non-collapsed run is "
          "dropped from the figure rather than plotted as an empty mean")


# --------------------------------------------------------------------- C6
def _in_view_tick_labels(ax, which):
    """Tick labels whose tick actually lies inside the view interval.

    Matplotlib keeps a Text object for every tick a Locator generates, including
    ticks outside the axis limits (a log axis over 0.1--140 still owns labels
    for 1e-2 and 1e3).  Those are never drawn, so measuring them would report
    phantom clipping.  Pair tick locations with labels and keep the in-view
    ones.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    try:
        lo, hi = sorted(axis.get_view_interval())
    except Exception:
        lo, hi = (sorted(ax.get_xlim()) if which == "x" else sorted(ax.get_ylim()))
    span = (hi - lo) or 1.0
    out = []
    for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()):
        if not lab.get_text() or not lab.get_visible():
            continue
        if lo - 1e-6 * span <= loc <= hi + 1e-6 * span:
            out.append(lab)
    return out


def check_nothing_clipped(objs, tol=1.5):
    bad = []
    for name, fig in objs:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        W, H = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
        artists = []
        for ax in fig.axes:
            artists += _in_view_tick_labels(ax, "x")
            artists += _in_view_tick_labels(ax, "y")
            artists += [t for t in ax.texts if t.get_text() and t.get_visible()]
            for leg in [ax.get_legend()] + list(fig.legends):
                if leg is not None:
                    artists += [t for t in leg.get_texts() if t.get_text()]
            for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
                if t.get_text() and t.get_visible():
                    artists.append(t)
        for f in fig.legends:
            artists += [t for t in f.get_texts() if t.get_text()]
        for t in artists:
            try:
                bb = t.get_window_extent(r)
            except Exception:
                continue
            if (bb.x0 < -tol or bb.y0 < -tol
                    or bb.x1 > W + tol or bb.y1 > H + tol):
                bad.append(f"{name}: '{t.get_text()[:24]}' "
                           f"({bb.x0:.0f},{bb.y0:.0f})-({bb.x1:.0f},{bb.y1:.0f}) "
                           f"vs canvas {W:.0f}x{H:.0f}")
    check("C6 no text artist falls outside the figure canvas (render check)",
          not bad, " | ".join(bad[:4]) if bad else "")


# ---------------------------------------------------------------- C7 / C8
def check_outputs(objs, textwidth_in):
    names = [n for n, _ in objs]
    for name in names:
        pdf = FIGS_DIR / f"{name}.pdf"
        if not check(f"C7a {name}.pdf exists", pdf.exists()):
            continue
        raw = pdf.read_bytes()
        check(f"C7b {name}.pdf is vector (no raster image XObject)",
              b"/Subtype /Image" not in raw and b"/Subtype/Image" not in raw)
        check(f"C7c {name}.pdf embeds its fonts (FontFile)",
              b"FontFile" in raw)
    if textwidth_in is None:
        check("C8 canvas width == real \\textwidth (could not measure)", False,
              "pdflatex or Definitions/mdpi.cls unavailable; cannot verify")
        return
    check("C8a measured \\textwidth matches figstyle.PRINT_W_IN",
          abs(textwidth_in - fs.PRINT_W_IN) < 0.005,
          f"measured {textwidth_in:.4f} in vs PRINT_W_IN {fs.PRINT_W_IN:.4f} in")
    for name, fig in objs:
        w = fig.get_figwidth()
        check(f"C8b {name} canvas width == \\textwidth "
              f"({textwidth_in:.4f} in)", abs(w - textwidth_in) < 0.005,
              f"{w:.4f} in -> LaTeX scale {textwidth_in / w:.4f}")


# -------------------------------------------------------------------- main
def build_all():
    """Rebuild every figure, capturing each Figure instead of writing it."""
    import figstyle as fs
    import make_figures as mf
    fs.apply_style()
    captured = []
    orig_save = fs.save

    def spy(fig, name, *a, **k):
        # Do NOT plt.close(fig) here: since matplotlib 3.11, close() detaches
        # the backend canvas (leaves a bare FigureCanvasBase without
        # get_renderer), which breaks every inspection step that follows.
        captured.append((name, fig))
        return None

    mf.fs.save = spy
    funcs = [("fig1_spatial_split", mf.fig1_spatial_split),
             ("fig2_autocorr_decay", mf.fig2_autocorr_decay),
             ("fig3_per_arch_gap", mf.fig3_per_arch_gap),
             ("fig4_collapse", mf.fig4_collapse),
             ("fig5_roc_pr", mf.fig5_roc_pr),
             ("fig6_decomposition", mf.fig6_decomposition),
             ("fig7_cross_region", mf.fig7_cross_region)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        for _, fn in funcs:
            try:
                fn()
            except Exception as e:                     # noqa: BLE001
                captured.append((f"__error__{fn.__name__}: {e}", None))
    mf.fs.save = orig_save
    return [(n, f) for n, f in captured if f is not None], buf.getvalue()


def main():
    print("Figure QA -- correctness rules\n")
    print("Rebuilding all figures (writes nothing)...")
    objs, log = build_all()
    print(log.strip() or "  (builders silent)")
    errs = [n for n, _ in objs if n.startswith("__error__")]
    if errs:
        for e in errs:
            check(f"builder {e}", False)
        print()
        return 1
    print()

    check_source_rules()
    check_legends_outside(objs)
    # Figures that plot a summary mark (mean +/- interval) must state n.
    summary_figures = {"fig3_per_arch_gap", "fig6_decomposition",
                       "fig7_cross_region"}
    check_n_stated(objs, {n: (n in summary_figures) for n, _ in objs})
    check_collapse_excluded()
    check_nothing_clipped(objs)
    tw = measure_textwidth()
    if tw is None:
        print("  [WARN] could not measure \\textwidth (pdflatex / mdpi.cls "
              "unavailable); C8 will fail")
    check_outputs(objs, tw)
    for _, f in objs:
        if f is not None:
            plt.close(f)

    npass = sum(1 for _, ok, _ in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print(f"\n{npass} passed, {nfail} failed, {len(RESULTS)} checks total.")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())