# %% [markdown]
# # Notebook 03 — Bayesian Uncertainty Modeling
# **Project:** Statistical Agricultural Yield Predictor (Kwara State, Nigeria)
# **Author:** KERBEM KATOR | Landmark University
#
# We extend OLS with Bayesian Linear Regression.
# The key insight: OLS gives a single number ("yield will be X kg/ha").
# Bayes gives a distribution ("yield will be X ± σ with 95% probability").
# For agricultural planning, knowing the uncertainty is as valuable as the estimate.

# %%
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_raw, clean, prepare_arrays, train_test_split_temporal, FEATURE_COLS
from src.regression import OLSRegression
from src.bayesian import BayesianLinearRegression
from src.visualizer import plot_bayesian_uncertainty

# %% [markdown]
# ## 1. Prepare Data (Same Split as Notebook 02)

# %%
df = clean(load_raw('../data/raw/kwara_crop_yield_synthetic.csv'))
df_maize = df[df['crop'] == 'Maize'].copy()

train, test = train_test_split_temporal(df_maize, test_year_cutoff=2021)

train_data = prepare_arrays(train, normalize_features=True)
test_data = prepare_arrays(test, normalize_features=False)

X_train = train_data['X_norm']
y_train = train_data['y']
X_test = (test_data['X'] - train_data['norm_mean']) / train_data['norm_std']
y_test = test_data['y']

# %% [markdown]
# ## 2. Fit Bayesian Model

# %%
bayes = BayesianLinearRegression(alpha=1.0, beta=25.0)
bayes.fit(X_train, y_train)

mean_train, std_train = bayes.predict(X_train)

print(f"Posterior mean weights: {bayes.m_N}")
print(f"\nMean prediction uncertainty (std): {std_train.mean():.2f} kg/ha")
print(f"Min uncertainty: {std_train.min():.2f} | Max: {std_train.max():.2f}")

# %% [markdown]
# ## 3. Compare OLS vs Bayesian Point Estimates

# %%
ols = OLSRegression().fit(X_train, y_train)
ols_pred = ols.predict(X_test)
bayes_mean, bayes_std = bayes.predict(X_test)

ols_rmse = np.sqrt(np.mean((y_test - ols_pred)**2))
bayes_rmse = np.sqrt(np.mean((y_test - bayes_mean)**2))

print(f"OLS  Test RMSE:   {ols_rmse:.2f} kg/ha")
print(f"Bayes Test RMSE:  {bayes_rmse:.2f} kg/ha")
print(f"\n(Bayesian acts as regularized OLS — slight RMSE difference is expected)")

# %% [markdown]
# ## 4. Credible Intervals

# %%
lower_95, upper_95 = bayes.credible_interval(X_test, credibility=0.95)
lower_80, upper_80 = bayes.credible_interval(X_test, credibility=0.80)

# Coverage check — what % of true values fall inside the interval?
coverage_95 = np.mean((y_test >= lower_95) & (y_test <= upper_95))
coverage_80 = np.mean((y_test >= lower_80) & (y_test <= upper_80))

print(f"95% Credible Interval Coverage: {coverage_95*100:.1f}% (target: ~95%)")
print(f"80% Credible Interval Coverage: {coverage_80*100:.1f}% (target: ~80%)")

# %% [markdown]
# ## 5. Posterior Predictive Plot (Single Feature)

# %%
X_rain = train['rainfall_mm'].values.reshape(-1, 1)
X_rain_norm = (X_rain - X_rain.mean()) / X_rain.std()
y = train['yield_kg_ha'].values

bayes_1d = BayesianLinearRegression(alpha=1.0, beta=25.0).fit(X_rain_norm, y)

b_mean, b_std = bayes_1d.predict(X_rain_norm)
b_lower, b_upper = bayes_1d.credible_interval(X_rain_norm, credibility=0.95)

# Draw posterior samples to visualize epistemic uncertainty
weight_samples = bayes_1d.weight_posterior_samples(n_samples=200)
sample_preds = []
X_b = np.c_[np.ones(len(X_rain_norm)), X_rain_norm]
for w in weight_samples:
    sample_preds.append(X_b @ w)

fig = plot_bayesian_uncertainty(
    feature_vals=train['rainfall_mm'].values,
    y_true=y,
    bayes_mean=b_mean,
    credible_lower=b_lower,
    credible_upper=b_upper,
    posterior_samples=np.array(sample_preds),
    feature_name='Rainfall (mm)',
    save_path='../data/processed/fig_bayesian_uncertainty.png'
)
plt.show()

# %% [markdown]
# ## 6. Practical Prediction Example

# %%
print("=" * 60)
print("YIELD PREDICTION EXAMPLE — Kwara State Farmer Advisory")
print("=" * 60)

# Scenario: A farmer in Ilorin with specific conditions
scenario = np.array([[1150.0, 28.5, 6.2, 50.0]])  # [rainfall, temp, pH, fertilizer]
scenario_norm = (scenario - train_data['norm_mean']) / train_data['norm_std']

pred_mean, pred_std = bayes.predict(scenario_norm)
ci_lo, ci_hi = bayes.credible_interval(scenario_norm, credibility=0.95)

print(f"\nInput Conditions:")
for feat, val in zip(FEATURE_COLS, scenario[0]):
    print(f"  {feat:<22}: {val}")

print(f"\nPredicted Maize Yield:")
print(f"  Point Estimate     : {pred_mean[0]:.0f} kg/ha")
print(f"  95% Credible Range : {ci_lo[0]:.0f} – {ci_hi[0]:.0f} kg/ha")
print(f"  Uncertainty (±1σ)  : ±{pred_std[0]:.0f} kg/ha")

# %% [markdown]
# **Key Takeaways:**
# - Bayesian LR produces calibrated uncertainty — coverage matches stated probability
# - The credible interval is interpretable: "95% chance true yield falls in this range"
# - Posterior samples show HOW the model's uncertainty varies across the input space
# - High-uncertainty predictions (sparse data regions) automatically get wider intervals
# - This is directly actionable for farmers: plan for the lower credible bound, not the mean
