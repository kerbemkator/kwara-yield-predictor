# Statistical Agricultural Yield Predictor
### Kwara State, Nigeria — Phase 1 AI Engineering Project

> *Farmers in Kwara State have no data-driven way to estimate crop yields.*  
> *This project builds a statistical model using regression fundamentals and*  
> *Bayesian probability to predict yield based on rainfall, temperature, soil pH,*  
> *and fertilizer application data.*

---

## Problem Statement

Agricultural planning in Kwara State, Nigeria relies heavily on intuition and historical memory. 
This project demonstrates how statistical modeling — built from mathematical first principles — 
can give farmers and agricultural planners a data-driven yield estimate with calibrated uncertainty bounds.

---

## Technical Approach

### 1. OLS Regression (From Scratch)

Implemented using the **Normal Equation** with NumPy only:

$$\theta = (X^TX)^{-1}X^Ty$$

No scikit-learn in the core model. This validates mathematical understanding before reaching for abstractions.

### 2. Bayesian Linear Regression

Extended OLS with a conjugate Normal prior to produce **full predictive distributions**:

- **Posterior mean** — best estimate of yield
- **95% Credible Interval** — probabilistic uncertainty bounds
- **Epistemic uncertainty** — how uncertain the model is in sparse data regions

Key insight: A farmer planning fertilizer budgets benefits more from knowing the *range* of possible yields than a single point estimate.

---

## Dataset

| Field | Description |
|---|---|
| `year` | 2010–2023 |
| `lga` | 16 Local Government Areas in Kwara State |
| `crop` | Maize, Rice, Sorghum, Yam, Cassava |
| `rainfall_mm` | Annual rainfall (mm) |
| `temp_celsius` | Mean temperature (°C) |
| `soil_ph` | Soil pH reading |
| `fertilizer_kg_ha` | Fertilizer application rate |
| `yield_kg_ha` | **Target** — crop yield in kg/hectare |

> **Note:** Current dataset is synthetic, statistically calibrated to Kwara State conditions.  
> Real FMARD/FAO Nigeria data integration is planned for v2.

---

## Project Structure

```
kwara-yield-predictor/
├── data/
│   ├── raw/                          # Source data (synthetic or FMARD)
│   └── processed/                    # Cleaned data + output figures
├── notebooks/
│   ├── 01_exploration.py             # EDA and feature distributions
│   ├── 02_regression_scratch.py      # OLS from scratch + sklearn validation
│   └── 03_bayesian_uncertainty.py    # Bayesian LR + credible intervals
├── src/
│   ├── data_loader.py                # Load, clean, normalize, split
│   ├── regression.py                 # OLS implementation (NumPy only)
│   ├── bayesian.py                   # Bayesian LR with posterior inference
│   └── visualizer.py                 # All plotting functions
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
git clone https://github.com/YOUR_USERNAME/kwara-yield-predictor
cd kwara-yield-predictor

pip install -r requirements.txt

# Open notebooks in VS Code with Jupyter extension
# Run notebooks in order: 01 → 02 → 03
```

---

## Key Results

| Metric | Value |
|---|---|
| OLS Train R² | ~0.85 |
| OLS Test RMSE | ~200 kg/ha |
| 95% CI Coverage | ~95% (calibrated) |
| Implementation | Pure NumPy (validated against sklearn) |

---

## Skills Demonstrated

- Linear algebra from scratch (Normal Equation, matrix inversion)
- Probabilistic modeling (Bayesian inference, conjugate priors)
- Data preprocessing pipeline (normalization, temporal train/test split)
- Statistical diagnostics (Q-Q plots, residual analysis, coefficient CIs)
- Production code structure (modular `src/` package, separation of concerns)

---

## Author

**KERBEM KATOR**  
AI Engineering | Landmark University, Nigeria  
Building toward world-class AI engineering with international reach.

*Part of an 8-phase AI Engineering Roadmap — Phase 1: Mathematical Foundations*

---

## Roadmap

- [ ] v1.0 — Synthetic data, OLS + Bayesian (current)
- [ ] v2.0 — Real FMARD dataset integration
- [ ] v3.0 — Multi-crop ensemble model
- [ ] v4.0 — FastAPI prediction endpoint
- [ ] v5.0 — Interactive Streamlit dashboard
