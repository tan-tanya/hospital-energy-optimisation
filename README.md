# energy-optimisation

Optimal sizing and dispatch of on-site electricity and heating systems for UK hospitals, across nine UK districts.

The model builds half-hourly building demand profiles from published benchmarks, then solves a linear program (LP) that sizes rooftop PV, battery storage, thermal storage and a heating system over a 15-year horizon, ranking each configuration against a gas-boiler business-as-usual baseline on both net present value and cumulative carbon.

---

## Methodology

**1. Demand modelling.** Half-hourly hospital electricity and heat profiles per district, assembled from CIBSE annual benchmarks, NCM occupancy schedules, TM46 baseload/HDD splits, BEES floor areas and always-on base-load fractions, a UKPN EV-charging layer, Met Office degree-days and sunshine hours, and ERA5 hourly temperatures. Over the horizon, non-heat electricity grows per the DESNZ Reference projection and heat demand falls with UKCP18 warming.

**2. Optimisation.** For each (district, heating) cell, the LP sizes PV, battery, thermal store and heating capacity to minimise discounted total cost, subject to energy balance, roof area, grid import/export limits and land availability.

**3. Uncertainty.** A two-stage stochastic program (TSSP) over reduced import-price scenarios for electricity and gas, run alongside a deterministic central-price round for comparison. Each fuel's price level and annual escalation are drawn by Latin hypercube sampling (1,000 draws) from triangular Low/Central/High bands, then reduced by k-means clustering to 3 weighted scenarios, keeping the draw nearest each centroid. The two fuels are paired: each scenario sits at the same percentile of both fuels' price bands.

**4. Downstream analysis.** Carbon objective and epsilon-constraint Pareto fronts, grid-headroom sensitivity, back-calculated policy levers at the best-NPV district (battery and heat-pump capital rebates, a heat-pump electricity price cut, a raised export price, and a battery-grant sweep), and two technology sensitivities (a cheaper low-efficiency PV panel, a lower-COP air-source heat pump).

Scope: 9 districts × 4 heating systems, for a single activity class (Health: Hospital).

| Districts | Heating systems |
|---|---|
| East Anglia | Gas Boiler |
| England E and NE | ASHP |
| England NW and N Wales | GSHP (vertical) |
| England SE and Central S | GSHP (horizontal) |
| England SW and S Wales | |
| Midlands | |
| Scotland E | |
| Scotland N | |
| Scotland W | |

---

## Installation

Requires Python ≥ 3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra optim
```

The `optim` extra adds the solver stack (PuLP + HiGHS). A plain `uv sync` is enough for the demand-profile model alone.

---

## Running

### Full pipeline

```bash
python optimisation_model.py
```

Writes everything to a timestamped `outputs/Optimisation (YYYYMMDD, HHMM)/` directory: one results workbook per cost round (deterministic and stochastic), standalone chart PNGs, the demand chart set, policy recommendations, and both sensitivity workbooks.

**The entire run takes around 26 hours** (the last full run took 26 h 18 m). Skipping the stochastic round shortens it considerably:

```bash
python optimisation_model.py --skip-stochastic
```

Useful flags:

| Flag | Effect |
|---|---|
| `--skip-stochastic` | Deterministic central-price round only |
| `--skip-deterministic` | Stochastic round only — the second half of a split run |
| `--skip-policy` | Skip the capital-rebate back-calculation |
| `--skip-grid-sensitivity` | Skip the grid-headroom bisection |
| `--grid-sensitivity-from RUN_DIR` | Reuse a completed run's grid-headroom results instead of re-solving |
| `--skip-ashp-sensitivity` | Skip the Aerona3 heat-pump sensitivity |
| `--skip-pv-sensitivity` | Skip the Panel-1 PV sensitivity |
| `--jobs N` | Parallel worker processes (default: a quarter of the logical cores) |
| `--time-limit S` | Per-solve time limit, seconds |

The three stages that re-solve off the deterministic round — both technology sensitivities and the
grid bisection — are skipped automatically under `--skip-deterministic`, since that round is not in
memory.

### Demand profiles alone

```bash
python demand_profile_model.py
```

Writes the full demand chart set (every district) — to `outputs/Demand ({timestamp})/demand/`. 
Also importable; this is how the optimiser consumes demand.

---

## Module map

| Module | Role |
|---|---|
| `optimisation_model.py` | Entry point |
| `optimisation_engine.py` | LP construction, solve, sweep and pool mechanics |
| `optimisation_config.py` | Solver and horizon constants |
| `optimisation_report.py` | Results workbook assembly |
| `optimisation_plots.py` | Chart generation |
| `demand_profile_model.py` | Half-hourly hospital demand profiles per district |
| `demand_report.py` | Demand chart rendering |
| `model_params.py` | Technology costs and parameters, read from `data/model_parameters.xlsx` |
| `districts.py` | District table (ICAO stations, UKCP regions, coordinates) |
| `seasons.py` | Calendar / season conventions shared by the demand, optimisation and `api_` modules |
| `datasets.py` | Input file loading |
| `pricing.py` | Wholesale + DUoS + CCL import price build-up |
| `projections.py` | Climate and electricity demand projections |
| `growth.py` | Demand growth and COP improvement over the horizon |
| `uncertainty.py` | Price-scenario generation and reduction |
| `grid_sensitivity.py` | Grid headroom / demand-margin bisection |
| `policy_recommendations.py` | Back-calculated regional capital rebates |
| `pv_panel1_sensitivity.py` | Alternative PV panel sensitivity |
| `ashp_grant_aerona3_sensitivity.py` | Alternative ASHP sensitivity |
| `api_osm_storeys.py` | OpenStreetMap hospital storey survey |
| `api_temperature_profiles.py` | ERA5 hourly temperature via Open-Meteo |
| `api_wholesale_prices.py` | Half-hourly GB wholesale electricity prices (Elexon MID) |

Modules prefixed `api_` pull live external data and write their output into `data/`. 
They are imported at runtime by other modules but only need re-running when refreshing the underlying data.

### Standalone scripts

| Script | Role |
|---|---|
| `scripts/dec_validation.py` | Compares modelled year-0 hospital energy intensities against metered Display Energy Certificates (DEC) and their TM46 benchmarks. Needs the DEC bulk CSV from the Energy Performance of Buildings Register |
| `scripts/headroom_medians.py` | Recomputes the per-district grid import/export ceilings in `data/model_parameters.xlsx` from the raw DNO extracts in `data/headroom/` (median of non-zero headroom) |

Neither is imported by the model.

---

## Data

| Directory | Contents | Source |
|---|---|---|
| `data/hdd/` | Daily heating degree-days, base 15.5 °C | Met Office |
| `data/sunlighthours/` | Monthly sunshine hours | Met Office |
| `data/climateprojections/` | UKCP18 RCP8.5 temperature anomalies | Met Office / CEDA |
| `data/headroom/` | Substation headroom and network development data | UKPN, NPG, ENWL, SP, NGED, SSEN |
| `data/duoscharges/` | 2025 DUoS Schedules of Charges | All UK DNOs |
| `data/inputs.xlsx` | CIBSE, NCM, TM46, BEES benchmarks | Various |
| `data/model_parameters.xlsx` | Technology costs and model parameters | Various |
| `data/api_*.xlsx` | Outputs of the `api_` modules (OSM storeys, ERA5 temperatures, wholesale prices) | OpenStreetMap, Open-Meteo, Elexon |
| `data/*_projection_output.csv` | Processed electricity-demand and climate (HDD) projections, written by `projections.py` | DESNZ, UKCP18 |
| `data/00 readme.xlsx` | Directory tab indexing every source | — |
| `cache/` | OpenStreetMap survey checkpoints | OpenStreetMap |

`cache/` pins the hospital building-stock survey to the data the model was run against.
