"""
Import-only, by optimisation_model. Summary cover sheet + figures for the optimisation results workbook.

Workbook sheets, in the order they are written (chart tabs first, then the data behind them).

    Cover                 key inputs & assumptions (economics, price bands, capex, grid limits, run metadata)
    Status                solver status by district x heating (one grid, a column per heating system),
                          with each cell's per-building grid-headroom margin as cell text; feasible
                          cells shaded by NPV savings (green above break-even, amber below)
    NPV                   top-9 savings, best savings by district, and the across-scenario robustness
                          spread (stochastic rounds only)
    Carbon                the same views for carbon savings, plus the district x heating MAC grid
    Pareto                cost/carbon scatter, best-value Pareto heating by district, and the
                          featured-cell frontier
    Dispatch              featured-cell knee dispatch + the underlying demand profile (optional)
    NPV Data              the full ranked cost sweep
    Carbon Data           the full ranked emissions sweep
    Pareto Data           the pooled non-dominated set
    Pareto Front Data     the featured cell's epsilon-constraint frontier (optional)
    Self-supply Data      NPV-optimal design per district x class, with self-supply rates
    Roof Util Data        PV roof-area utilisation factors
    Grid Sensitivity Data the grid-headroom bisection, when one was run or carried over (optional)
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless: no GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill

import model_params as mp
import demand_profile_model as dm
import optimisation_engine as oe
import optimisation_plots as oplt
from districts import DISTRICT_STATIONS

# Display orders + short labels.
DIST_ORDER  = list(DISTRICT_STATIONS.keys())
ACT_SHORT   = {"Health: Hospital": "Hospital"}
ACT_ORDER   = [ACT_SHORT.get(a, a) for a in mp.ROOF_PROPERTIES]
HEAT_ORDER  = ["Gas Boiler", "ASHP", "GSHP (vertical)", "GSHP (horizontal)"]
HEAT_SHORT  = {"Gas Boiler": "Gas", "ASHP": "ASHP",
               "GSHP (vertical)": "GSHP-V", "GSHP (horizontal)": "GSHP-H"}
HEAT_COLORS = {"Gas Boiler": "#9e9e9e", "ASHP": "#1f77b4",
               "GSHP (vertical)": "#2ca02c", "GSHP (horizontal)": "#17becf"}
ACT_COLORS  = {"Hospital": "#c44e52"}

# Bars on the "Top N scenarios" charts: the best design in each district, so no district appears twice.
# 9 = every district. Filenames stay "*_top10.png".
TOP_N = 9

# Make every figure filename carry the round as a prefix ("deterministic_npv_top10.png", "stochastic_npv_top10.png").
_CHART_PREFIX = ""

_HDR_FILL = PatternFill("solid", fgColor="DDE7F0")
_TITLE_FONT, _SECT_FONT, _BOLD = Font(bold=True, size=14), Font(bold=True, size=11), Font(bold=True)


def _act_area_map() -> dict:
    # {act_short: BEES median premises floor area m²}. 
    # Append floor area to activity-class labels on categorical axes.
    areas = getattr(dm, "bees_floor_areas", None) or {}
    return {ACT_SHORT.get(a, a): v for a, v in areas.items()}


def _act_label(act_short: str, area_map: dict) -> str:
    # Append the building floor area to an activity-class label: "Hospital" -> "Hospital (4,150 m²)".
    a = area_map.get(act_short)
    return f"{act_short} ({a:,.0f} m²)" if a else act_short


def _gbp(x: float) -> str:
    # Compact GBP label.
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    a, sign = abs(x), "−" if x < 0 else ""                 # sign before the £: −£500k, not £-500k
    if a >= 1e6: return f"{sign}£{a/1e6:.2f}m"
    if a >= 1e3: return f"{sign}£{a/1e3:.0f}k"
    return f"{sign}£{a:.0f}"


def _gbp_axis(vmax: float):
    # Tick formatter with one unit for the whole axis (−£0.5m, −£1m, not −£500k, −£1.00m).
    def fmt(v, _):
        if abs(v) < 1e-9:
            return "£0"
        sign = "−" if v < 0 else ""
        if vmax >= 1e6:
            return f"{sign}£{abs(v)/1e6:.2f}".rstrip("0").rstrip(".") + "m"
        return f"{sign}£{abs(v)/1e3:,.0f}k"
    return FuncFormatter(fmt)


def _pct(x: float) -> str:
    return f"{x*100:+.1f}%"


def _tco2e(x: float) -> str:
    # Compact tCO2e label.
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x/1e3:.1f}k t" if abs(x) >= 1e3 else f"{x:.0f} t"


def _mac(x: float) -> str:
    # Signed £/tCO₂e label. Negative: abatement that pays for itself; Positive: abatement that costs money.
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{'−' if x < 0 else ''}£{abs(x):,.0f}/t"


def _dual_text(value_fmt, v, cost=None, carbon=None) -> str:
    # Data-callout text for one bar/cell: "£X, Y t".
    if cost is not None and carbon is not None:
        return f"{_gbp(cost)}, {_tco2e(carbon)}"
    return value_fmt(v)


# Compact design summary (PV size + capacity factor + battery / thermal store). Defined once in
# optimisation_plots so the workbook labels and the standalone charts cannot drift apart.
def _spec_line(r) -> str:
    return oplt.spec_line(r)


def _spec_line_wrapped(r) -> str:
    return oplt.spec_line(r, wrapped=True)


# FIGURES 
def _ranked_barh(top: pd.DataFrame, texts: list, charts_dir: str, *, value_col: str, value_fmt,
                 value_label: str, title: str, fname: str) -> str:
    # Shared layout of the ranked-bar charts. top is best-first, one row per bar; texts are the bar callouts.
    top, texts = top.iloc[::-1], texts[::-1]        # reverse: best at top of barh
    AM = _act_area_map()
    labels = [f"{r.district} · {_act_label(r.act_short, AM)}\n{_spec_line(r)}" for r in top.itertuples()]
    vals   = top[value_col].to_numpy()
    colors = [HEAT_COLORS[h] for h in top["heating"]]

    fig, ax = plt.subplots(figsize=(10, 6.8))
    y = np.arange(len(top))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8, linespacing=1.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(value_label)
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: value_fmt(v)))
    pad = 0.01 * (np.nanmax(vals) - min(0, np.nanmin(vals)) + 1)
    for i, (v, txt) in enumerate(zip(vals, texts)):
        ax.text(v + (pad if v >= 0 else -pad), i, txt,
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    # Legend sits below the x-axis label, outside the plot, so it never covers a bar's value label.
    ax.legend(handles=[Patch(color=HEAT_COLORS[h], label=HEAT_SHORT[h]) for h in HEAT_ORDER],
              title="Heating", fontsize=8, title_fontsize=8, ncol=len(HEAT_ORDER),
              loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)
    fig.tight_layout()
    return _save(fig, charts_dir, fname)


def _fig_top10(dfo: pd.DataFrame, charts_dir: str, *, value_col: str, value_fmt,
               value_label: str, title: str, fname: str,
               cost_col: str = None, carbon_col: str = None) -> str:
    top = (dfo.sort_values(value_col, ascending=False)
             .drop_duplicates("district").head(TOP_N))      # best design per district
    texts = [_dual_text(value_fmt, getattr(r, value_col),
                        getattr(r, cost_col) if cost_col else None,
                        getattr(r, carbon_col) if carbon_col else None) for r in top.itertuples()]
    return _ranked_barh(top, texts, charts_dir, value_col=value_col, value_fmt=value_fmt,
                        value_label=value_label, title=title, fname=fname)


def _fig_marginal(dfo: pd.DataFrame, charts_dir: str, *, value_col: str, value_fmt,
                  value_label: str, fname: str,
                  cost_col: str = None, carbon_col: str = None) -> str:
    # Headline ranking = best achievable value per district (across heating systems).
    # Districts with no optimal design (every cell infeasible) are dropped, not drawn empty.
    idx  = dfo.groupby("district")[value_col].idxmax()
    best = (dfo.loc[idx.values].set_index("district").reindex(DIST_ORDER)
               .dropna(subset=[value_col]).sort_values(value_col))
    AM = _act_area_map()
    one_act = dfo["act_short"].nunique() == 1

    fig, ax = plt.subplots(figsize=(10, 6.5))
    vals = best[value_col].to_numpy()
    y = np.arange(len(best))
    ax.barh(y, vals, color="#55a868")
    ticklabels = []
    for name, r in zip(best.index.astype(str), best.itertuples()):
        what = HEAT_SHORT[r.heating] if one_act else f"{_act_label(r.act_short, AM)} · {HEAT_SHORT[r.heating]}"
        ticklabels.append(f"{name}\n{what}\n{_spec_line(r)}")
    ax.set_yticks(y, ticklabels, fontsize=7.5, linespacing=1.5)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(value_label + " (best across heating systems)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: value_fmt(v)))
    cost_vals   = best[cost_col].to_numpy()   if cost_col   else None
    carbon_vals = best[carbon_col].to_numpy() if carbon_col else None
    for i, v in enumerate(vals):
        txt = _dual_text(value_fmt, v, cost_vals[i] if cost_vals is not None else None,
                          carbon_vals[i] if carbon_vals is not None else None)
        ax.text(v, i, "  " + txt, va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    top = dfo.loc[dfo[value_col].idxmax()]
    top_txt = _dual_text(value_fmt, top[value_col],
                         top[cost_col] if cost_col else None,
                         top[carbon_col] if carbon_col else None)
    act_txt = f" — {_act_label(top.act_short, AM)}" if one_act else ""
    ax.set_title(f"{value_label} by district{act_txt}\n"
                 f"Best single scenario: {top.district} · {HEAT_SHORT[top.heating]}  ({top_txt})",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    return _save(fig, charts_dir, fname)


# Solver status category colours.
_STATUS_COLORS = {
    "Optimal":                        "#4caf50",
    "No incumbent":                   "#9e9e9e",
    "Infeasible":                     "#e53935",
}
_STATUS_ORDER = list(_STATUS_COLORS)

# Hue says whether the project is worth doing (green = positive NPV, amber = negative NPV, red = not feasible);
# Depth says by how much.
_NPV_DIVERGING = LinearSegmentedColormap.from_list(
    "feasible_npv", [(0.00, "#7f3f00"), (0.25, "#e08214"), (0.50, "#ffffff"),
                     (0.75, "#41ab5d"), (1.00, "#00441b")])
# Flat fallbacks for a figure whose feasible cells carry no usable NPV range 
# (single value, or a results set without the column).
_NPV_FLAT_POS = _STATUS_COLORS["Optimal"]
_NPV_FLAT_NEG = "#e08214"


def _npv_norm(vmin, vmax):
    # Diverging norm centred on zero NPV.
    if vmin is None or vmax is None or not np.isfinite([vmin, vmax]).all():
        return None
    lo, hi = min(float(vmin), 0.0), max(float(vmax), 0.0)
    if lo == hi:
        return None
    span = hi - lo
    lo = min(lo, -0.1 * span)
    hi = max(hi, 0.1 * span)
    return TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi)


def _npv_shade(v, norm) -> tuple:
    if v is None or not np.isfinite(v):
        return to_rgba(_NPV_FLAT_POS)
    if norm is None:
        return to_rgba(_NPV_FLAT_NEG if v < 0 else _NPV_FLAT_POS)
    return _NPV_DIVERGING(float(np.clip(norm(v), 0.0, 1.0)))


def _text_color(rgba: tuple) -> str:
    # Black on pale cells, white on dark cells.
    r, g, b = rgba[:3]
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"


def _edge_margin_label(r) -> str:
    # One cell's grid-connection headroom margin as text.
    # positive = demand-growth margin today's DNO ceiling can still absorb (cell Optimal today);
    # negative = how much bigger the ceiling would need to be to reach feasibility (cell infeasible).
    # Cells whose bisection ran out of search range never measured an edge labelled as a "non-result".
    if not bool(r["search_bounded"]):
        return "unbounded"
    v = r["edge_margin_pct"]
    if pd.isna(v):
        return "N/A"
    mult = 1.0 + abs(v) / 100.0
    mult_txt = f"{mult:.1f}x" if mult < 10.0 else f"{mult:.0f}x"
    kw = r["edge_margin_kw"]
    kw_txt = f"{kw:+,.0f} kW" if pd.notna(kw) else "? kW"
    # One line, multiplier first with absolute headroom in brackets — e.g. "41x (+11,084 kW)".
    return f"{mult_txt} ({kw_txt})"


def _grid_rows(d: pd.DataFrame) -> tuple:
    # Rows of the single-panel district x heating grids: one per district when the results hold one
    # activity class (its label is returned for the title), else one per district x class.
    present = set(zip(d["district"], d["act_short"]))
    acts = [a for a in ACT_ORDER if a in set(d["act_short"])]
    keys = [(dist, a) for dist in DIST_ORDER for a in acts if (dist, a) in present]
    AM = _act_area_map()
    if len(acts) == 1:
        return keys, [dist for dist, _ in keys], _act_label(acts[0], AM)
    return keys, [f"{dist} · {_act_label(a, AM)}" for dist, a in keys], None


def _grid_frame(ax, n_rows: int, n_cols: int, col_labels: list, row_labels: list) -> None:
    # Shared styling for the district x heating grids: heating headers on top, white rules between cells.
    ax.set_xticks(range(n_cols), col_labels, fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(n_rows), row_labels, fontsize=10.5)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def _fig_status_grid(df_raw: pd.DataFrame, charts_dir: str, *, title: str, fname: str,
                     grid_sensitivity: pd.DataFrame = None, npv_col: str = "npv_savings_GBP") -> str:
    # One district x heating grid (a column per heating system). Cell text carries each cell's NPV savings
    # and grid-headroom margin. Heat-pump heatings only carry a margin: Gas Boiler draws no heat-pump electricity.
    d = df_raw.copy()
    d["status"] = d["status"].where(~d["status"].astype(str).str.startswith("Infeasible"),
                                    "Infeasible")
    d["act_short"] = d["activity"].map(lambda a: ACT_SHORT.get(a, a))
    other = [s for s in d["status"].unique() if s not in _STATUS_COLORS]
    status_colors = dict(_STATUS_COLORS)
    status_order = list(_STATUS_ORDER)
    for s in other:
        status_colors[s] = "#6a1b9a"
        status_order.append(s)
    row_keys, row_labels, act_title = _grid_rows(d)
    heats = [h for h in HEAT_ORDER if h in set(d["heating"])]

    # NPV shading range is taken across every feasible cell in the figure (all heating columns).
    has_npv = npv_col in d.columns
    npv_vals = pd.to_numeric(d.loc[d["status"] == "Optimal", npv_col], errors="coerce").dropna() \
        if has_npv else pd.Series(dtype=float)
    vmin = float(npv_vals.min()) if not npv_vals.empty else None
    vmax = float(npv_vals.max()) if not npv_vals.empty else None
    npv_norm = _npv_norm(vmin, vmax)

    margins = {}
    if grid_sensitivity is not None and not grid_sensitivity.empty:
        g = grid_sensitivity.copy()
        g["act_short"] = g["activity"].map(lambda a: ACT_SHORT.get(a, a))
        for _, r in g.iterrows():
            margins[(r["heating"], r["district"], r["act_short"])] = _edge_margin_label(r)

    rows = pd.MultiIndex.from_tuples(row_keys, names=["district", "act_short"])
    piv = (d.pivot_table(index=["district", "act_short"], columns="heating", values="status",
                         aggfunc="first")
            .reindex(index=rows, columns=heats))
    npv_piv = (d.pivot_table(index=["district", "act_short"], columns="heating", values=npv_col,
                             aggfunc="max")
                .reindex(index=rows, columns=heats)) if has_npv else None
    n_rows, n_cols = len(row_keys), len(heats)
    rgba = np.ones((n_rows, n_cols, 4))                  # white = cell absent from results
    for i in range(n_rows):
        for j in range(n_cols):
            s = piv.iat[i, j]
            if not isinstance(s, str):
                continue
            rgba[i, j] = (_npv_shade(npv_piv.iat[i, j] if npv_piv is not None else None, npv_norm)
                          if s == "Optimal" else to_rgba(status_colors[s]))

    fig, ax = plt.subplots(figsize=(2.9 * n_cols + 3.2, 0.72 * n_rows + 2.6))
    ax.imshow(rgba, aspect="auto")
    col_labels = [f"{h}\n({int((d.loc[d['heating'] == h, 'status'] == 'Infeasible').sum())} infeasible)"
                  for h in heats]
    _grid_frame(ax, n_rows, n_cols, col_labels, row_labels)
    for i, (dist, act) in enumerate(row_keys):
        for j, h in enumerate(heats):
            s = piv.iat[i, j]
            if not isinstance(s, str):
                continue
            npv_v = npv_piv.iat[i, j] if npv_piv is not None else None
            head = [_gbp(npv_v)] if (s == "Optimal" and npv_v is not None and np.isfinite(npv_v)) else \
                   ([] if s == "Optimal" else [s])
            # Margin only on cells that solved.
            show_margin = s == "Optimal" and (h, dist, act) in margins
            parts = head + ([margins[(h, dist, act)]] if show_margin else [])
            if parts:
                ax.text(j, i, "\n".join(parts), ha="center", va="center", fontsize=10.5,
                        color=_text_color(rgba[i, j]), linespacing=1.35)
    if act_title:
        title += f" — {act_title}"
    shaded = npv_norm is not None
    notes = []
    if shaded:
        notes.append("Cell text (Line 1): NPV savings vs gas boiler BAU")
    # Fixed-sizing bisection (baseline PV, no battery, baseline thermal store),
    # because a free re-solve lets the optimiser buy storage to reduce the peak.
    if margins:
        notes.append("Cell text (Line 2): demand growth each building can add before the DNO "
                     "import ceiling binds")
    if notes:
        title += "\n" + "\n".join(notes)
    fig.suptitle(title, fontweight="bold", fontsize=13)
    handles = [Patch(color=status_colors[s], label=s) for s in status_order
               if not (shaded and s == "Optimal")]
    fig.legend(handles=handles, title="Solver status", fontsize=9, title_fontsize=9,
               loc="upper left", bbox_to_anchor=(1.0, 0.9))
    fig.tight_layout(rect=(0, 0, 0.98, 0.92))    # before add_axes: tight_layout ignores manual axes
    if shaded:
        # White point sits exactly on break-even.
        sm = ScalarMappable(norm=npv_norm, cmap=_NPV_DIVERGING)
        cax = fig.add_axes((1.01, 0.15, 0.012, 0.5))
        cb  = fig.colorbar(sm, cax=cax, format=FuncFormatter(lambda v, _: _gbp(v)))
        cb.set_label("Optimal (feasible) — NPV savings vs gas-boiler BAU"
                     "\n(amber = negative, white = break-even)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    return _save(fig, charts_dir, fname)


_STATUS_TITLE = "Solver status by district × heating"


def _status_figs(dfs: dict, charts_dir: str, grid_sensitivity) -> list:
    # Feasibility does not depend on objective; cost and emissions status grids typically identical.
    key  = ["district", "activity", "heating"]
    npv, carb = dfs["NPV"], dfs["Carbon"]
    merged = npv[key + ["status"]].merge(carb[key + ["status"]], on=key, suffixes=("_n", "_c"))
    same = len(merged) and (merged["status_n"] == merged["status_c"]).all()

    if same:
        figs = [_fig_status_grid(npv, charts_dir, title=_STATUS_TITLE, fname="status_grid.png",
                                 grid_sensitivity=grid_sensitivity)]
    else:
        figs = [_fig_status_grid(npv, charts_dir, title=_STATUS_TITLE + " (cost objective)",
                                 fname="status_grid_cost.png", grid_sensitivity=grid_sensitivity),
                _fig_status_grid(carb, charts_dir, title=_STATUS_TITLE + " (emissions objective)",
                                 fname="status_grid_carbon.png", grid_sensitivity=grid_sensitivity)]
    return figs


_MAC_TITLE = ("Marginal Abatement Cost by district × heating{act}\n"
              "(against the cost-optimal gas boiler + PV design at the same site)\n"
              "Cell text (Line 1): marginal abatement cost   ·   (Line 2): change in annual "
              "primary energy vs that same design")


def _model_carbon_value(d: pd.DataFrame) -> float:
    # The model's own discounted average carbon value (£/tCO2e).
    need = {"carbon_value_npv_GBP", "lifetime_emissions_tco2e"}
    if not need <= set(d.columns):
        return float("nan")
    num = pd.to_numeric(d["carbon_value_npv_GBP"], errors="coerce")
    den = pd.to_numeric(d["lifetime_emissions_tco2e"], errors="coerce")
    return float(np.nanmedian((num / den.where(den > 0)).to_numpy(dtype=float)))


def _primary_energy_label(pct, kwh) -> str:
    # One cell's change in annual primary energy against the gas+PV counterfactual.
    if pct is None or kwh is None or not np.isfinite([pct, kwh]).all():
        return ""
    mwh = kwh / 1000.0
    mwh_txt = f"{mwh:+,.0f} MWh/yr" if abs(mwh) >= 10 else f"{mwh:+,.1f} MWh/yr"
    return f"{pct:+.0f}% primary\n({mwh_txt})"


def _fig_mac_grid(dfo: pd.DataFrame, charts_dir: str, *, fname: str = "mac_grid.png") -> str:
    # Built from the cost sweep only. MAC divides a design's extra cost over the counterfactual by the
    # tonnes it abates, so it is only a cost abatement when the design was chosen under cost pressure.
    # One district x heating grid; only the heat pump designs are plotted.
    col = "mac_GBP_per_tco2e"
    if col not in dfo.columns:
        return None
    d = dfo.copy()
    d[col] = pd.to_numeric(d[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    vals = d[col].dropna()
    if vals.empty:
        return None

    cval   = _model_carbon_value(d)
    centre = cval if np.isfinite(cval) else float(vals.median())
    lo = 0.0 if float(vals.min()) < 0.0 else float(np.nanpercentile(vals, 5))
    lo = min(lo, centre - 1.0)
    # Percentile bound: one runaway cell would otherwise flatten every other colour.
    hi = max(float(np.nanpercentile(vals, 95)), centre + 1.0)
    norm = TwoSlopeNorm(vmin=lo, vcenter=centre, vmax=hi)
    cmap = plt.get_cmap("RdYlGn_r")

    heats = [h for h in HEAT_ORDER if h != "Gas Boiler"]      # gas is the counterfactual
    # Only what actually solved gets drawn.
    solved = d[d[col].notna() & d["heating"].isin(heats)]
    if solved.empty:
        return None
    heats = [h for h in heats if h in set(solved["heating"])]
    row_keys, row_labels, act_title = _grid_rows(solved)
    rows = pd.MultiIndex.from_tuples(row_keys, names=["district", "act_short"])
    pe_cols = [c for c in ("primary_energy_change_pct", "primary_energy_change_kwh")
               if c in d.columns]
    n_rows, n_cols = len(row_keys), len(heats)
    data = np.full((n_rows, n_cols), np.nan)
    pe = {c: np.full((n_rows, n_cols), np.nan) for c in pe_cols}
    for j, h in enumerate(heats):
        # One row per cell: the cost-optimal (lowest-MAC) design.
        sub = (d[d["heating"] == h]
               .sort_values(col, na_position="last")
               .drop_duplicates(subset=["district", "act_short"], keep="first")
               .set_index(["district", "act_short"]).reindex(rows))
        data[:, j] = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        for c in pe_cols:
            pe[c][:, j] = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
    rgba = np.ones((n_rows, n_cols, 4))
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isfinite(v):
                rgba[i, j] = cmap(float(np.clip(norm(v), 0.0, 1.0)))

    fig, ax = plt.subplots(figsize=(3.4 * n_cols + 3.6, 0.8 * n_rows + 2.8))
    ax.imshow(rgba, aspect="auto")
    col_labels = []
    for j, h in enumerate(heats):
        n_cells = int(np.isfinite(data[:, j]).sum())
        if np.isfinite(cval):
            n_below = int((data[:, j] < cval).sum())
            col_labels.append(f"{h}\n({n_below}/{n_cells} below the carbon value)")
        else:
            col_labels.append(h)
    _grid_frame(ax, n_rows, n_cols, col_labels, row_labels)
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            pe_txt = _primary_energy_label(
                pe["primary_energy_change_pct"][i, j] if "primary_energy_change_pct" in pe else None,
                pe["primary_energy_change_kwh"][i, j] if "primary_energy_change_kwh" in pe else None)
            tcol = _text_color(rgba[i, j])
            if pe_txt:
                ax.text(j, i - 0.13, f"£{v:,.0f}/t", ha="center", va="bottom", fontsize=12,
                        fontweight="bold", color=tcol)
                ax.text(j, i - 0.07, pe_txt, ha="center", va="top", fontsize=9.5,
                        color=tcol, linespacing=1.3)
            else:
                ax.text(j, i, f"£{v:,.0f}/t", ha="center", va="center", fontsize=12,
                        fontweight="bold", color=tcol)

    fig.suptitle(_MAC_TITLE.format(act=f" — {act_title}" if act_title else ""),
                 fontweight="bold", fontsize=12, y=0.985, va="top")
    # Headroom for the three-line title above the heating headers.
    fig.tight_layout(rect=(0, 0, 0.98, 0.95))
    cax = fig.add_axes((1.005, 0.2, 0.015, 0.5))
    cb  = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       format=FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    cb.set_label("Marginal abatement cost (£/tCO2e) — scale floors at £0", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if np.isfinite(cval) and lo <= cval <= hi:
        cb.ax.axhline(cval, color="black", linewidth=1.2)
    return _save(fig, charts_dir, fname)


def _fig_robustness(dfo: pd.DataFrame, charts_dir: str) -> str:
    # Robustness of top NPV designs under the import-price scenario set.
    need = {"npv_savings_min_GBP", "npv_savings_max_GBP"}
    if not need.issubset(dfo.columns):
        return None                                    
    top = (dfo.sort_values("npv_savings_GBP", ascending=False)
              .drop_duplicates("district").head(TOP_N).iloc[::-1])   # best design per district
    AM = _act_area_map()
    labels = [f"{r.district} · {_act_label(r.act_short, AM)}" for r in top.itertuples()]
    exp = top["npv_savings_GBP"].to_numpy()
    lo  = top["npv_savings_min_GBP"].to_numpy()
    hi  = top["npv_savings_max_GBP"].to_numpy()
    colors = [HEAT_COLORS[h] for h in top["heating"]]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(top))
    ax.barh(y, exp, color=colors, zorder=2)
    ax.errorbar(exp, y, xerr=[exp - lo, hi - exp], fmt="none", ecolor="0.25",
                elinewidth=1.1, capsize=3, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("15-year NPV savings vs BAU — expected (bar), worst..best case (whisker)")
    ax.set_title("Robustness of top designs under energy-price uncertainty", fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _gbp(v)))
    ax.legend(handles=[Patch(color=HEAT_COLORS[h], label=HEAT_SHORT[h]) for h in HEAT_ORDER],
              title="Heating", fontsize=8, title_fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, charts_dir, "npv_robustness.png")


# SELF-SUPPLY + ROOF-UTILISATION DATA SHEETS
def _roof_utilisation_rows() -> list:
    # Effective usable fraction of the roof footprint available for PV, split by roof type.
    areas = getattr(dm, "bees_floor_areas", None) or {}
    rows = []
    for a in mp.ROOF_PROPERTIES:
        flat_eff, pitched_eff = oe.roof_type_usable_fracs(a)
        rows.append({"activity": ACT_SHORT.get(a, a), "activity_full": a,
                     "floor_area_m2": areas.get(a),
                     "utilisation_rate": mp.ROOF_PROPERTIES[a]["pv_usable_frac"],
                     "inter_row_spacing_factor": mp.ROOF_PROPERTIES[a]["pv_inter_row_frac"],
                     "effective_utilisation_flat": flat_eff,
                     "effective_utilisation_pitched": pitched_eff})
    return rows


def _roof_utilisation_table() -> pd.DataFrame:
    df = pd.DataFrame(_roof_utilisation_rows())
    for c in ("utilisation_rate", "inter_row_spacing_factor",
              "effective_utilisation_flat", "effective_utilisation_pitched"):
        df[c] = df[c].round(4)
    return df.drop(columns=["activity_full"])


def _rq1_table(npv: pd.DataFrame) -> pd.DataFrame:
    # NPV-optimal (recommended) design per district × class, with sizing and self-supply rates. 
    cols = ["district", "act_short", "heating", "pv_kwp", "e_batt_kwh", "annual_demand_kwh",
            "annual_pv_gen_kwh", "annual_export_kwh", "self_consumption_rate",
            "self_sufficiency_rate", "npv_savings_GBP"]
    have = [c for c in cols if c in npv.columns]
    idx  = npv.groupby(["district", "act_short"])["npv_savings_GBP"].idxmax()
    t = npv.loc[idx, have].sort_values(["act_short", "district"]).reset_index(drop=True)
    return t.rename(columns={"act_short": "activity"})


# PARETO FIGURES

def _fig_pareto_scatter(dfp: pd.DataFrame, charts_dir: str) -> str:
    # Every candidate design in (lifetime emissions, NPV savings) space, coloured by heating,
    # sized by total capex (PV/battery/thermal-store + heat-plant). Non-dominated designs are
    # ring-marked, annotated with a compact spec line, and joined by a step line (the frontier).
    total_capex = (dfp["capex_GBP"].fillna(0.0) + dfp["heat_capex_GBP"].fillna(0.0))
    cmin, cmax = total_capex.min(), total_capex.max()
    def _bubble_size(cap: pd.Series) -> pd.Series:
        if cmax <= cmin:
            return pd.Series(60.0, index=cap.index)
        return 15.0 + 185.0 * (cap - cmin) / (cmax - cmin)

    fig, ax = plt.subplots(figsize=(12.5, 7))
    heat_present = [h for h in HEAT_ORDER if not dfp[dfp["heating"] == h].empty]
    for h in heat_present:
        sub = dfp[dfp["heating"] == h]
        cap = sub["capex_GBP"].fillna(0.0) + sub["heat_capex_GBP"].fillna(0.0)
        ax.scatter(sub["lifetime_emissions_tco2e"], sub["npv_savings_GBP"],
                   s=_bubble_size(cap), c=HEAT_COLORS[h], alpha=0.55, zorder=2)
    pf = dfp[dfp["pareto_optimal"]].sort_values("lifetime_emissions_tco2e")
    ax.scatter(pf["lifetime_emissions_tco2e"], pf["npv_savings_GBP"],
               facecolors="none", edgecolors="k", s=90, linewidths=1.2, zorder=3)
    ax.axhline(0, color="k", lw=0.6)                       # BAU NPV reference
    ax.set_xlabel("Lifetime emissions (tCO₂e, lower = better)")
    ax.set_ylabel("15-year NPV savings vs BAU")
    ax.set_title("Cost-carbon performance across all candidate designs", fontweight="bold")
    ax.yaxis.set_major_formatter(_gbp_axis(dfp["npv_savings_GBP"].abs().max()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))   # unit is in the label
    # Both legends sit to the right of the axes. 
    heat_handles = [ax.scatter([], [], s=70.0, c=HEAT_COLORS[h], alpha=0.55, label=HEAT_SHORT[h])
                    for h in heat_present]
    heat_handles.append(ax.scatter([], [], facecolors="none", edgecolors="k", s=70.0,
                                   linewidths=1.2, label="Pareto-optimal"))
    heat_leg = ax.legend(handles=heat_handles, title="Heating", fontsize=8, title_fontsize=8,
                         loc="upper left", bbox_to_anchor=(1.02, 1.0))
    heat_leg._legend_box.align = "left"
    ax.add_artist(heat_leg)                                # keep it when the second legend is added
    ref = pd.Series([100e3, 750e3, 3.4e6])
    cap_handles = [ax.scatter([], [], s=float(_bubble_size(ref).iloc[i]), c="0.6", alpha=0.55,
                              label=_gbp(float(ref.iloc[i]))) for i in range(len(ref))]
    # Anchored in axes coords like the heating legend, so both boxes share a left edge.
    cap_leg = ax.legend(handles=cap_handles, title="Total capex (bubble area)", fontsize=8,
                        title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 0.74),
                        labelspacing=1.1, borderpad=0.9)
    cap_leg._legend_box.align = "left"
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    return _save(fig, charts_dir, "pareto_scatter.png")


def _best_mac_idx(g: pd.DataFrame):
    # Lowest £/tCO2e among the group's designs that electrify heat; a gas-only group falls back to gas.
    e = g[(g["heating"] != "Gas Boiler") & g["mac_GBP_per_tco2e"].notna()]
    if e.empty:
        e = g[g["mac_GBP_per_tco2e"].notna()]
    return None if e.empty else e["mac_GBP_per_tco2e"].idxmin()


def _fig_pareto_heating(dfp: pd.DataFrame, charts_dir: str) -> str:
    # Best-value non-dominated design per district on a MAC basis, ranked lowest £/tCO2e first.
    pf   = dfp[dfp["pareto_optimal"].astype(bool)]
    idx  = [i for _, g in pf.groupby("district") if (i := _best_mac_idx(g)) is not None]
    if not idx:
        return None
    best = pf.loc[idx].sort_values("mac_GBP_per_tco2e").head(TOP_N)
    t_col = ("emissions_saving_vs_gaspv_tco2e" if "emissions_saving_vs_gaspv_tco2e" in best.columns
             else "emissions_saving_tco2e")
    texts = [f"{_mac(m)}, {_tco2e(t)}" for m, t in zip(best["mac_GBP_per_tco2e"], best[t_col])]
    return _ranked_barh(best, texts, charts_dir, value_col="mac_GBP_per_tco2e", value_fmt=_mac,
                        value_label="Marginal abatement cost vs gas boiler + PV (£/tCO₂e, lower = better)",
                        title="Best-value Pareto heating by district\n"
                              "(lowest £/tCO₂e among designs that electrify heat)",
                        fname="pareto_heating.png")


def _save(fig, charts_dir: str, name: str) -> str:
    path = os.path.join(charts_dir, _CHART_PREFIX + name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _stack_images(ws, pngs: list, start_row: int = 2, col: str = "B", pad_rows: int = 3) -> None:
    # Stack chart PNGs down one sheet.
    r = start_row
    for png in pngs:
        img = XLImage(png)
        ws.add_image(img, f"{col}{r}")
        r += int(img.height / 20) + pad_rows


# COVER SHEET

def _build_cover(book, run_meta: dict) -> None:
    ws = book.create_sheet("Cover", 0)
    for col, w in zip("ABCD", (34, 20, 16, 16)):
        ws.column_dimensions[col].width = w
    r = [1]
    def _row(): v = r[0]; r[0] += 1; return v
    def title(t):
        c = ws.cell(_row(), 1, t); c.font = _TITLE_FONT
    def section(t):
        v = _row(); c = ws.cell(v, 1, t); c.font = _SECT_FONT
        for col in range(1, 5): ws.cell(v, col).fill = _HDR_FILL
    def kv(k, val):
        v = _row(); ws.cell(v, 1, k); ws.cell(v, 2, val)
    def blank(): _row()
    def table(headers, rows):
        v = _row()
        for j, h in enumerate(headers, 1): c = ws.cell(v, j, h); c.font = _BOLD
        for row in rows:
            v = _row()
            for j, val in enumerate(row, 1): ws.cell(v, j, val)

    tc = mp.TECH_COSTS
    title("Building PV · Battery · Heat — Optimisation Results")
    kv("Run", run_meta.get("timestamp", ""))
    kv("Scenarios", f"{run_meta.get('n_feasible','?')} / {run_meta.get('n_scenarios','?')} feasible")
    kv("Solver", f"{run_meta.get('solver','HiGHS')}")
    blank()

    section("Economics (real / constant-2025 prices)")
    kv("Discount rate (WACC, real)", _pct(tc["discount_rate"]))
    kv("Horizon", f"{run_meta.get('horizon_years','?')} years")
    kv("General inflation (real O&M)", _pct(tc["general_inflation"]))
    kv("Electricity price growth (real)", _pct(tc["elec_price_growth"]) + "/yr")
    kv("Gas price growth (real)", _pct(tc["gas_price_growth"]) + "/yr")
    kv("Export price (SEG)", f"£{tc['elec_export_price']:.2f}/kWh")
    blank()

    section("Active scenarios")
    kv("PV capex", f"{mp.PV_CAPEX_ACTIVE} (£{mp.PV_CAPEX_SCENARIOS[mp.PV_CAPEX_ACTIVE]:.0f}/kWp)")
    kv("Export price", f"{mp.ELEC_EXPORT_ACTIVE} (£{mp.ELEC_EXPORT_SCENARIOS[mp.ELEC_EXPORT_ACTIVE]:.2f}/kWh)")
    kv("Energy price growth", mp.ENERGY_GROWTH_ACTIVE)
    blank()

    price_scen = run_meta.get("price_scenarios")
    if price_scen:
        section("Uncertainty — electricity + gas import-price scenarios (TSSP stage-2)")
        kv("Method", "2-D Latin Hypercube → k-medoid reduction (uncertainty.py)")
        kv("Uncertain input", "electricity + gas import price (level × escalation), paired "
                              "percentiles; export central")
        for col in "EF":                             # the gas columns sit beyond the sheet's A-D widths
            ws.column_dimensions[col].width = 16
        table(["Scenario", "Weight", "Elec yr-0 level", "Elec escalation/yr",
               "Gas yr-0 level", "Gas escalation/yr"],
              [[sid, f"{w:.3f}", f"×{lvl:.3f}", _pct(g), f"×{gl:.3f}", _pct(gg)]
               for (sid, w, lvl, g, gl, gg) in price_scen])
        # The level bands are right-skewed. Most of the stochastic round's higher cost is thus the level shift,
        # not a cost of ignoring uncertainty.
        _wsum = sum(s[1] for s in price_scen)
        if _wsum > 0:
            _mean_lvl = sum(s[1] * s[2] for s in price_scen) / _wsum
            _mean_gas = sum(s[1] * s[4] for s in price_scen) / _wsum
            kv("Weighted mean elec level", f"×{_mean_lvl:.3f}  (deterministic round = ×1.000)")
            kv("Weighted mean gas level", f"×{_mean_gas:.3f}")
        blank()

    section("Capex (headline)")
    kv("PV", f"£{tc['pv']['capex_per_kwp']:.0f}/kWp")
    kv("Battery", f"£{tc['battery']['energy_capex_per_kwh']:.0f}/kWh + £{tc['battery']['power_capex_per_kw']:.0f}/kW")
    kv("Thermal store", f"£{mp.THERMAL_STORE['energy_capex_per_kwh']:.0f}/kWh")
    kv("Heat plant (£/kW_th)", " · ".join(
        f"{HEAT_SHORT[h]} £{mp.HEAT_COSTS[h]['capex_per_kwth']:.0f}" for h in HEAT_ORDER))
    blank()

    section("Electricity import price bands (DESNZ non-domestic, incl CCL)")
    table(["Band", "MWh/yr", "£/kWh"],
          [[b["name"], _band_range(b), round(b["price"], 4)] for b in mp.ELEC_IMPORT_BANDS])
    blank()
    section("Gas import price bands (DESNZ non-domestic, incl CCL)")
    table(["Band", "MWh/yr", "£/kWh"],
          [[b["name"], _band_range(b), round(b["price"], 4)] for b in mp.GAS_IMPORT_BANDS])
    blank()

    section("Grid connection limits (DNO network headroom, present-day)")
    table(["District", "Import kW", "Export kW"],
          [[d, mp.GRID_LIMITS[d].get("import_kw"), mp.GRID_LIMITS[d].get("export_kw")]
           for d in DIST_ORDER if d in mp.GRID_LIMITS])


def _band_range(b: dict) -> str:
    lo, hi = b["lo"], b["hi"]
    return f"{lo:g}+" if hi == float("inf") else f"{lo:g}–{hi:g}"


# ENTRY POINT
def write_report(dfs: dict, out_path: str, *,
                 run_meta: dict, charts_dir: str, dispatch_png: str = None,
                 demand_png: str = None, front_png: str = None,
                 grid_sensitivity: pd.DataFrame = None, chart_prefix: str = "") -> list:
    # Build the three-objective workbook + standalone PNGs.
    # dispatch_png (optional):      pre-rendered stage-2 dispatch of the featured-cell knee design ("Dispatch" tab).
    # demand_png (optional):        underlying heat vs non-heat electricity demand for the same featured cell/months ("Dispatch" tab).
    # front_png (optional):         the featured-cell continuous cost/carbon frontier + MAC chart ("Pareto" tab).
    # grid_sensitivity (optional):  the district x activity x heat-pump-heating grid-headroom bisection ("Status" tab).
    # chart_prefix (optional):      "deterministic_" / "stochastic_" — keeps both rounds' PNGs in one charts/ folder.
    os.makedirs(charts_dir, exist_ok=True)
    global _CHART_PREFIX
    _CHART_PREFIX = chart_prefix
    try:
        return _write_report(dfs, out_path, run_meta=run_meta, charts_dir=charts_dir,
                             dispatch_png=dispatch_png, demand_png=demand_png,
                             front_png=front_png, grid_sensitivity=grid_sensitivity)
    finally:
        _CHART_PREFIX = ""


def _write_report(dfs: dict, out_path: str, *,
                  run_meta: dict, charts_dir: str, dispatch_png: str = None,
                  demand_png: str = None, front_png: str = None,
                  grid_sensitivity: pd.DataFrame = None) -> list:

    def _prep(df, value_col):
        d = df[df["status"] == "Optimal"].dropna(subset=[value_col]).copy()
        if d.empty:
            raise ValueError(f"No optimal scenarios with {value_col} to report.")
        d["act_short"] = d["activity"].map(lambda a: ACT_SHORT.get(a, a))
        return d

    # Rebase every marginal abatement cost onto the cost-optimal gas+PV design.
    _ref = oe.gaspv_reference(dfs["NPV"])
    if not _ref.empty:
        for _k in ("NPV", "Carbon", "Pareto", "Pareto front"):
            if _k in dfs and dfs[_k] is not None and not dfs[_k].empty:
                dfs[_k] = oe.apply_gaspv_mac(dfs[_k], _ref)

    npv    = _prep(dfs["NPV"],    "npv_savings_GBP")
    carbon = _prep(dfs["Carbon"], "emissions_saving_tco2e")
    pareto = _prep(dfs["Pareto"], "npv_savings_GBP")     

    chart_sets = {                              
        "Status": _status_figs(dfs, charts_dir, grid_sensitivity),
        "NPV": [
            _fig_top10(npv, charts_dir, value_col="npv_savings_GBP", value_fmt=_gbp,
                       value_label="15-year NPV savings vs gas-boiler BAU",
                       title=f"Top {TOP_N} scenarios by NPV savings", fname="npv_top10.png",
                       cost_col="npv_savings_GBP", carbon_col="emissions_saving_tco2e"),
            _fig_marginal(npv, charts_dir, value_col="npv_savings_GBP", value_fmt=_gbp,
                          value_label="NPV savings", fname="npv_marginal.png",
                          cost_col="npv_savings_GBP", carbon_col="emissions_saving_tco2e"),
            _fig_robustness(npv, charts_dir),     # None on a deterministic run
        ],
        "Carbon": [
            _fig_top10(carbon, charts_dir, value_col="emissions_saving_tco2e", value_fmt=_tco2e,
                       value_label="Lifetime carbon savings vs gas-boiler BAU",
                       title=f"Top {TOP_N} scenarios by carbon savings", fname="carbon_top10.png",
                       cost_col="npv_savings_GBP", carbon_col="emissions_saving_tco2e"),
            _fig_marginal(carbon, charts_dir, value_col="emissions_saving_tco2e", value_fmt=_tco2e,
                          value_label="Carbon savings", fname="carbon_marginal.png",
                          cost_col="npv_savings_GBP", carbon_col="emissions_saving_tco2e"),
            _fig_mac_grid(npv, charts_dir),
        ],
        "Pareto": [
            _fig_pareto_scatter(pareto, charts_dir),
            _fig_pareto_heating(pareto, charts_dir),
        ],
    }
    if front_png and os.path.exists(front_png):
        chart_sets["Pareto"].append(front_png)             # featured-cell continuous frontier + MAC
    if dispatch_png and os.path.exists(dispatch_png):
        chart_sets["Dispatch"] = [dispatch_png]            # stage-2 dispatch of the featured-cell knee
    if demand_png and os.path.exists(demand_png):
        chart_sets.setdefault("Dispatch", []).append(demand_png)   # underlying heat vs non-heat demand

    chart_sets = {tab: [p for p in pngs if p] for tab, pngs in chart_sets.items()}   # drop skipped figs
    chart_sets = {tab: pngs for tab, pngs in chart_sets.items() if pngs}   # drop tabs left with nothing

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        dfs["NPV"].to_excel(xw,    sheet_name="NPV Data",    index=False)
        # On the emissions sweep the £/tCO2e figure is an upper bound on what the abatement costs, not the cheapest way to buy it. 
        dfs["Carbon"].rename(
            columns={"mac_GBP_per_tco2e": "cost_of_emissions_optimal_design_GBP_per_tco2e"}
        ).to_excel(xw, sheet_name="Carbon Data", index=False)
        dfs["Pareto"].to_excel(xw, sheet_name="Pareto Data", index=False)
        if "Pareto front" in dfs and not dfs["Pareto front"].empty:
            dfs["Pareto front"].to_excel(xw, sheet_name="Pareto Front Data", index=False)
        _rq1_table(npv).to_excel(xw, sheet_name="Self-supply Data", index=False)
        _roof_utilisation_table().to_excel(xw, sheet_name="Roof Util Data", index=False)
        if grid_sensitivity is not None and not grid_sensitivity.empty:
            gs = grid_sensitivity.copy()
            gs["edge_margin_label"] = [_edge_margin_label(r) for _, r in gs.iterrows()]
            gs.to_excel(xw, sheet_name="Grid Sensitivity Data", index=False)
        book = xw.book
        _build_cover(book, run_meta)
        for tab, pngs in chart_sets.items():
            _stack_images(book.create_sheet(tab), pngs)
        order = ["Cover", "Status", "NPV", "Carbon", "Pareto", "Dispatch",
                 "NPV Data", "Carbon Data", "Pareto Data", "Pareto Front Data",
                 "Self-supply Data", "Roof Util Data", "Grid Sensitivity Data"]
        book._sheets.sort(key=lambda s: order.index(s.title) if s.title in order else 99)

    return [p for pngs in chart_sets.values() for p in pngs]
