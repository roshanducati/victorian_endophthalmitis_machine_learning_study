"""
journal_style.py
================
Figure styling shared by the figure scripts (07_tables_and_figures.py,
08_figures_test_set.py and participant_flow.py), so that artwork settings
cannot drift between figures:

  * 8-point Helvetica throughout (Arial, which is metrically compatible, where
    Helvetica is absent); enforce_fontsizes() is the backstop.
  * Decimal points, never commas, in tick labels; no offset notation.
  * Bold part labels (A, B, ...) on multi-panel figures (label_panels()).
  * Figures at most 7.0 in (178 mm) wide; the export resolution is set by the
    save() helper in 07_tables_and_figures.py (TIFF at 1200 dpi).

Usage
-----
    import journal_style as js
    js.apply()                       # once, when the calling script is imported

Run this file directly to print the font family that matplotlib resolves.
"""

from __future__ import annotations

import matplotlib as mpl

# ── Artwork constants ─────────────────────────────────────────────────────────
DPI = 600                 # >= 500 dpi for combination art; 600 is the safe common setting
MAX_WIDTH_IN = 7.0        # journal hard cap on figure width
BASE_FONT_PT = 8.0        # journal-specified type size
SINGLE_COL_IN = 3.4       # single-column target width
FULL_WIDTH_IN = 7.0       # full-width / multi-panel target width

# Helvetica first; Arial is the metrically compatible substitute on Windows.
FONT_STACK = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]

RCPARAMS = {
    "font.family":      "sans-serif",
    "font.sans-serif":  FONT_STACK,
    "font.size":        BASE_FONT_PT,
    "axes.titlesize":   BASE_FONT_PT,
    "axes.labelsize":   BASE_FONT_PT,
    "xtick.labelsize":  BASE_FONT_PT,
    "ytick.labelsize":  BASE_FONT_PT,
    "legend.fontsize":  BASE_FONT_PT,
    "legend.title_fontsize": BASE_FONT_PT,
    "figure.titlesize": BASE_FONT_PT,
    "figure.labelsize": BASE_FONT_PT,
    "figure.dpi":       DPI,
    "savefig.dpi":      DPI,
    "axes.formatter.use_locale": False,   # decimal points, never commas
    "axes.formatter.useoffset":  False,   # no "+1e3" offset text
    "axes.unicode_minus": True,
    # Line/tick weights that stay visible at 600 dpi without going heavy.
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.linewidth":   0.5,
    "lines.linewidth":  1.2,
    "legend.frameon":   True,
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35,
    # Keep text as text in the vector drafts.
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
}


_applied = False


def apply() -> None:
    """Install the journal rcParams.  Safe to call more than once."""
    global _applied
    mpl.rcParams.update(RCPARAMS)
    _applied = True


def resolved_font() -> str:
    """Return the font family matplotlib will actually use, for reporting."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_STACK:
        if name in installed:
            return name
    return "DejaVu Sans"


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.06) -> None:
    """Typed part label (A, B, ...) in the corner of a panel.

    Bold, same 8-pt family as the rest of the figure, positioned in axes
    coordinates so it sits just outside the top-left corner of the panel.
    """
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=BASE_FONT_PT, fontweight="bold",
            va="bottom", ha="left", zorder=10)


def label_panels(axes, letters="ABCDEFGH", **kwargs) -> None:
    """Apply part labels A, B, ... across a sequence of axes."""
    for ax, letter in zip(list(axes), letters):
        panel_label(ax, letter, **kwargs)


def enforce_fontsizes(fig, size: float = BASE_FONT_PT, exact: bool = True) -> None:
    """Force every text artist in the figure to the journal type size.

    The rcParams already put all matplotlib-drawn text at 8 pt, but third-party
    plotters (shap, seaborn) hard-code their own sizes.  This is the backstop
    that guarantees the 8-pt specification holds across the whole figure: with
    exact=True nothing is left larger or smaller than 8 pt; with exact=False
    only undersized text is raised.
    """
    def fix(txt):
        try:
            cur = txt.get_fontsize()
        except (TypeError, ValueError, AttributeError):
            return
        if exact:
            if cur != size:
                txt.set_fontsize(size)
        elif cur < size:
            txt.set_fontsize(size)

    for txt in fig.findobj(mpl.text.Text):
        fix(txt)
    for ax in fig.get_axes():
        for lbl in (ax.get_xticklabels() + ax.get_yticklabels()):
            fix(lbl)
        ax.tick_params(axis="both", labelsize=size)
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                fix(txt)
            if leg.get_title() is not None:
                fix(leg.get_title())


if __name__ == "__main__":
    apply()
    print(f"Resolved font family: {resolved_font()}  (stack: {FONT_STACK})")
