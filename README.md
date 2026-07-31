# Metro Passenger Flow Forecasting

Short-term (15-minute) metro passenger inflow forecasting for station-level
crowd management. AI/ML Fundamentals Capstone.

Given a station and its recent flow history, the model predicts how many
passengers will enter that station in the next 15-minute interval — giving
operators lead time to act before congestion forms.

## Results
| Model | Validation MAE | Test MAE |
|---|---|---|
| Historical Average (baseline) | 29.34 | 37.55 |
| Ridge regression | 29.29 | 39.29 |
| **XGBoost (selected)** | **22.42** | **25.75** |

XGBoost beats the baseline by ~24% on validation MAE.

## Data
HZMetro (Hangzhou Metro), Zenodo DOI 10.5281/zenodo.3145404, CC-BY 4.0.
Raw AFC tap records, 80 stations, 1–25 January 2019.

**Raw data is not included in this repo** (1.3 GB). To reproduce from scratch:
1. Download from https://doi.org/10.5281/zenodo.3145404
2. Place the `record_2019-01-*.csv` files in `data/raw/`
3. Run `notebooks/01_aggregate.ipynb`

The aggregated/processed files needed for modeling are included in
`data/processed/`, so you can skip straight to modeling if preferred.

## How to run
```bash
python3 -m venv metro
source metro/bin/activate
pip install -r requirements.txt
```
Then run the notebooks in order:
1. `01_aggregate.ipynb` — raw AFC → 15-min inflow/outflow per station
2. `02_eda.ipynb` — EDA, leakage-safe split, feature engineering
3. `03_models.ipynb` — baseline, Ridge, XGBoost, MLflow tracking, saved artifact
4. `04_error_analysis.ipynb` — error by peak/off-peak and station busyness

## Method
- **Task:** supervised time-series regression (next-interval inflow)
- **Split:** chronological — train Jan 1–18, val Jan 19–20, test Jan 21–25 (no shuffling)
- **Features:** calendar (hour, weekday, weekend) + lags (1–4 intervals) + rolling mean
- **Leakage control:** scalers fit on train only; lags computed within station using only past intervals; test touched once
- **Tracking:** MLflow (3 logged runs)

## Repository structure
