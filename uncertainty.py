"""
Import-only. Electricity and gas import-price uncertainty for the TSSP.

Generates a weighted set of representative 15-year IMPORT-price scenarios that drive the
TSSP second stage (recourse dispatch). Each scenario carries an electricity path AND a gas path.
Uncertainty is sampled over two facets, for each fuel:
  1. price level multiplier — IMPORT_PRICE_SCENARIOS (Low/Central/High), relative to Central = 1.0
  2. annual escalation rate — ENERGY_GROWTH_SCENARIOS (Low/Central/High)
Each draw maps to a 15-year multiplier path per fuel
    path[y] = level * (1 + growth) ** y
applied on top of that fuel's year-0 size-band CENTRAL import price the deterministic model already selects.

Electricity and gas are paired (perfectly rank-correlated): one LHS percentile sets both fuels' level,
and a second sets both fuels' escalation, each read through that fuel's own triangular band. A
low-electricity draw is therefore always a low-gas draw, as in the paired Low/Central/High bands of
model_parameters.xlsx. This is the upper bound on co-movement: the elec/gas price ratio varies less
than it would under partial correlation. Level and escalation remain independent of each other.

Method (literature: Latin Hypercube Sampling -> scenario reduction by clustering):
  - 2-D LHS (scipy.stats.qmc.LatinHypercube) using triangular marginals (each band's min/mode/max).
  - k-medoid reduction of the joined [electricity, gas] paths to k weighted representatives.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import qmc, triang

from model_params import IMPORT_PRICE_SCENARIOS, ENERGY_GROWTH_SCENARIOS
from optimisation_config import HORIZON_YEARS   # single source of truth for the horizon

# Default sampling settings
DEFAULT_N_SAMPLES = 1000
DEFAULT_N_REDUCED = 3      # reduced scenario count (k)
DEFAULT_SEED      = 0


# 1 - DEFINITION 
def _triangular_params(low: float, mode: float, high: float) -> dict:
    span = high - low
    if span <= 0:
        raise ValueError(f"degenerate triangular band: low={low}, mode={mode}, high={high}")
    return {"c": (mode - low) / span, "loc": low, "scale": span}

def level_band(fuel: str = "elec") -> tuple:
    # (low, central, high) import-price level multipliers for fuel "elec" or "gas", relative to Central.
    s = IMPORT_PRICE_SCENARIOS
    return tuple(s[k][f"{fuel}_import_multiplier"] for k in ("Low", "Central", "High"))

def growth_band(fuel: str = "elec") -> tuple:
    # (low, central, high) annual real escalation rates of the import price for fuel "elec" or "gas".
    g = ENERGY_GROWTH_SCENARIOS
    return tuple(g[k][f"{fuel}_price_growth"] for k in ("Low", "Central", "High"))


def price_multiplier_path(level: float, growth: float, horizon: int = HORIZON_YEARS) -> np.ndarray:
    # 15-year import-price multiplier (relative to the year-0 Central band price); y=0 -> level.
    y = np.arange(horizon)
    return level * (1.0 + growth) ** y


# 2 - SAMPLING of (level, growth) per fuel
def lhs_samples(n: int = DEFAULT_N_SAMPLES, seed: int = DEFAULT_SEED) -> np.ndarray:
    # n x 4 array of (elec level, elec growth, gas level, gas growth), stratified by LHS.
    # Paired fuels: the same unit draw is read through each fuel's own band (column 0 -> levels, 1 -> growths).
    unit_draws = qmc.LatinHypercube(d=2, seed=seed).random(n)
    cols = []
    for fuel in ("elec", "gas"):
        cols.append(triang.ppf(unit_draws[:, 0], **_triangular_params(*level_band(fuel))))
        cols.append(triang.ppf(unit_draws[:, 1], **_triangular_params(*growth_band(fuel))))
    return np.column_stack(cols)

def _paths(draws: np.ndarray, horizon: int) -> tuple:
    # (elec paths, gas paths), each N x horizon, from lhs_samples() draws.
    elec = np.array([price_multiplier_path(l, g, horizon) for l, g in draws[:, 0:2]])
    gas  = np.array([price_multiplier_path(l, g, horizon) for l, g in draws[:, 2:4]])
    return elec, gas


# 3 - SCENARIO REDUCTION
def reduce_scenarios(paths: np.ndarray, k: int, seed: int = DEFAULT_SEED) -> tuple:
    # Cluster the N x horizon path matrix into <= k groups, then pick each cluster's sample nearest its centroid.
    # Returns (medoid_row_indices, weights) with weights = cluster share (summing to 1).
    from scipy.cluster.vq import kmeans2

    n = paths.shape[0]
    if k >= n:
        return np.arange(n), np.full(n, 1.0 / n)

    centroids, labels = kmeans2(paths, k, seed=seed, minit="++", missing="warn")
    medoid_idx, weights = [], []
    for c in range(len(centroids)):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue                                        # empty cluster — drop, renormalise below
        d = np.linalg.norm(paths[members] - centroids[c], axis=1)
        medoid_idx.append(int(members[np.argmin(d)]))
        weights.append(members.size / n)
    weights = np.asarray(weights)
    return np.asarray(medoid_idx), weights / weights.sum()


# 4 - PUBLIC INTERFACE
@dataclass(frozen=True)
class Scenario:
    # One representative (electricity, gas) import-price realisation for the TSSP second stage.
    # level / growth / path are the electricity facets; gas_level / gas_growth / gas_path the gas ones.
    id: str
    weight: float                 # probability mass (reduced scenarios sum to 1)
    level: float                  # electricity year-0 multiplier relative to Central
    growth: float                 # electricity annual real escalation rate
    path: np.ndarray = field(repr=False)       # electricity multiplier path, path[y] = level*(1+growth)**y
    gas_level: float              # gas year-0 multiplier relative to Central
    gas_growth: float             # gas annual real escalation rate
    gas_path: np.ndarray = field(repr=False)   # gas multiplier path, same form

    def import_price(self, y: int, base_central: float) -> float:
        # Scenario electricity import price in year y, GBP/kWh. base_central = year-0 central size-band price.
        return base_central * float(self.path[y])

    def gas_price(self, y: int, base_central: float) -> float:
        # Scenario gas import price in year y, GBP/kWh. base_central = year-0 central gas size-band price.
        return base_central * float(self.gas_path[y])


def generate_price_scenarios(n_samples: int = DEFAULT_N_SAMPLES,
                             n_reduced: int = DEFAULT_N_REDUCED,
                             horizon: int = HORIZON_YEARS,
                             seed: int = DEFAULT_SEED) -> list:
    # Full pipeline: LHS-sample (level, growth) per fuel -> build paths -> reduce to weighted scenarios.
    draws = lhs_samples(n_samples, seed=seed)
    elec, gas = _paths(draws, horizon)
    # Both paths are multipliers on their own Central year-0 price, so they share a scale and are clustered jointly.
    idx, weights = reduce_scenarios(np.hstack([elec, gas]), n_reduced, seed=seed)

    order = np.argsort([elec[i][-1] for i in idx])           # order low -> high final-year electricity price
    scenarios = []
    for rank, j in enumerate(order):
        i = int(idx[j])
        scenarios.append(Scenario(id=f"S{rank+1}", weight=float(weights[j]),
                                   level=float(draws[i, 0]), growth=float(draws[i, 1]),
                                   path=elec[i].copy(),
                                   gas_level=float(draws[i, 2]), gas_growth=float(draws[i, 3]),
                                   gas_path=gas[i].copy()))
    return scenarios


def scenario_table(scenarios: list) -> pd.DataFrame:
    # Tidy summary table of the reduced scenario set (for the report / assumptions).
    return pd.DataFrame([{
        "scenario":        s.id,
        "weight":          round(s.weight, 4),
        "elec_level_mult": round(s.level, 4),
        "elec_growth":     round(s.growth, 5),
        "elec_yr14_mult":  round(float(s.path[-1]), 4),
        "gas_level_mult":  round(s.gas_level, 4),
        "gas_growth":      round(s.gas_growth, 5),
        "gas_yr14_mult":   round(float(s.gas_path[-1]), 4),
    } for s in scenarios])


def validate(scenarios: list, n_samples: int = DEFAULT_N_SAMPLES,
             horizon: int = HORIZON_YEARS, seed: int = DEFAULT_SEED) -> dict:
    # Sanity checks: weights sum to 1, and each fuel's weighted-mean reduced path tracks its full-sample mean.
    elec, gas = _paths(lhs_samples(n_samples, seed=seed), horizon)
    w = [s.weight for s in scenarios]
    out = {"weight_sum": float(sum(w)), "n_scenarios": len(scenarios)}
    for fuel, full_paths, attr in (("elec", elec, "path"), ("gas", gas, "gas_path")):
        full = full_paths.mean(axis=0)
        red  = np.average(np.array([getattr(s, attr) for s in scenarios]), axis=0, weights=w)
        out.update({f"{fuel}_full_mean_yr0":    float(full[0]),
                    f"{fuel}_reduced_mean_yr0": float(red[0]),
                    f"{fuel}_full_mean_yr14":   float(full[-1]),
                    f"{fuel}_reduced_mean_yr14": float(red[-1]),
                    f"{fuel}_max_abs_path_err": float(np.max(np.abs(full - red)))})
    return out


# 5 - DIAGNOSTICS 
def _fan_chart(out_path: str, scenarios: list, n: int = DEFAULT_N_SAMPLES,
               horizon: int = HORIZON_YEARS, seed: int = DEFAULT_SEED) -> None:
    import matplotlib.pyplot as plt
    elec, gas = _paths(lhs_samples(n, seed=seed), horizon)
    yrs = np.arange(horizon)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, paths, attr, fuel in ((axes[0], elec, "path", "Electricity"), (axes[1], gas, "gas_path", "Gas")):
        for p in (5, 25, 50, 75, 95):
            ax.plot(yrs, np.percentile(paths, p, axis=0), color="0.6", lw=0.8)
        ax.fill_between(yrs, np.percentile(paths, 5, axis=0), np.percentile(paths, 95, axis=0),
                        color="0.85", label="LHS 5-95th pct")
        for s in scenarios:
            ax.plot(yrs, getattr(s, attr), lw=2, label=f"{s.id} (w={s.weight:.2f})")
        ax.set_xlabel("horizon year")
        ax.set_title(f"{fuel} import-price scenarios (LHS envelope + reduced set)")
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("import-price multiplier vs Central yr-0")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def main():
    import os
    out_dir = os.path.join("outputs", "uncertainty")
    os.makedirs(out_dir, exist_ok=True)
    scen = generate_price_scenarios()
    tbl  = scenario_table(scen)
    print("Reduced electricity + gas import-price scenarios:")
    print(tbl.to_string(index=False))
    print("\nValidation:", validate(scen))
    tbl.to_csv(os.path.join(out_dir, "price_scenarios.csv"), index=False)
    _fan_chart(os.path.join(out_dir, "price_scenario_fan.png"), scen)
    print(f"\nWrote scenario table + figure to {out_dir}")


if __name__ == "__main__":
    main()
