"""
visualizer.py
All visualization logic for the Kwara Yield Predictor.
Produces publication-quality plots using Matplotlib and Seaborn.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path


# ── Style Configuration ────────────────────────────────────────────────────────
PALETTE = {
    'primary': '#E6A817',    # Gold
    'secondary': '#2E86AB',  # Blue
    'accent': '#A23B72',     # Purple
    'success': '#2D6A4F',    # Green
    'danger': '#C1121F',     # Red
    'bg': '#0D1117',
    'surface': '#161B22',
    'text': '#E6EDF3',
    'muted': '#8B949E',
}

def _apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor': PALETTE['surface'],
        'axes.edgecolor': PALETTE['muted'],
        'axes.labelcolor': PALETTE['text'],
        'xtick.color': PALETTE['muted'],
        'ytick.color': PALETTE['muted'],
        'text.color': PALETTE['text'],
        'grid.color': '#21262D',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'font.family': 'monospace',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })


def plot_feature_distributions(df: pd.DataFrame, save_path: str = None):
    """Plot distributions of all key features split by crop type."""
    _apply_dark_style()

    features = ['rainfall_mm', 'temp_celsius', 'soil_ph', 'fertilizer_kg_ha', 'yield_kg_ha']
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Feature Distributions — Kwara State Crop Dataset', 
                 color=PALETTE['primary'], fontsize=15, y=1.02)

    crops = df['crop'].unique()
    colors = [PALETTE['primary'], PALETTE['secondary'], PALETTE['accent'],
              PALETTE['success'], PALETTE['danger']]

    for ax, feat in zip(axes, features):
        for crop, color in zip(crops, colors):
            subset = df[df['crop'] == crop][feat]
            ax.hist(subset, bins=20, alpha=0.5, color=color, label=crop, edgecolor='none')
        ax.set_title(feat.replace('_', ' ').title())
        ax.grid(True)
        ax.set_xlabel('')

    axes[0].legend(fontsize=8, labelcolor=PALETTE['text'],
                   facecolor=PALETTE['surface'], edgecolor=PALETTE['muted'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=PALETTE['bg'])
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """Correlation matrix of numeric features."""
    _apply_dark_style()

    numeric_cols = ['rainfall_mm', 'temp_celsius', 'soil_ph', 'fertilizer_kg_ha', 'yield_kg_ha']
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Feature Correlation Matrix', color=PALETTE['primary'], fontsize=14)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap='YlOrBr', vmin=-1, vmax=1,
        annot=True, fmt='.2f', linewidths=0.5,
        linecolor=PALETTE['bg'],
        cbar_kws={'shrink': 0.8}
    )
    ax.set_facecolor(PALETTE['surface'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=PALETTE['bg'])
    return fig


def plot_ols_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                          residuals: np.ndarray, save_path: str = None):
    """
    4-panel OLS diagnostic plot:
      1. Actual vs Predicted
      2. Residuals vs Fitted
      3. Residual histogram
      4. Q-Q plot (normality check)
    """
    _apply_dark_style()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('OLS Regression Diagnostics — Yield Predictor',
                 color=PALETTE['primary'], fontsize=15)
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # 1. Actual vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.5, color=PALETTE['secondary'], s=25, edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, '--', color=PALETTE['primary'], lw=1.5, label='Perfect fit')
    ax1.set_xlabel('Actual Yield (kg/ha)')
    ax1.set_ylabel('Predicted Yield (kg/ha)')
    ax1.set_title('Actual vs Predicted')
    ax1.legend()
    ax1.grid(True)

    # 2. Residuals vs Fitted
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, residuals, alpha=0.5, color=PALETTE['accent'], s=25, edgecolors='none')
    ax2.axhline(0, color=PALETTE['primary'], lw=1.5, linestyle='--')
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Fitted')
    ax2.grid(True)

    # 3. Residual histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=30, color=PALETTE['success'], edgecolor='none', alpha=0.8)
    ax3.axvline(0, color=PALETTE['primary'], lw=1.5, linestyle='--')
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Distribution')
    ax3.grid(True)

    # 4. Q-Q plot
    from scipy import stats
    ax4 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist='norm')
    ax4.scatter(osm, osr, alpha=0.5, color=PALETTE['danger'], s=25, edgecolors='none')
    ax4.plot(osm, slope * np.array(osm) + intercept,
             '--', color=PALETTE['primary'], lw=1.5)
    ax4.set_xlabel('Theoretical Quantiles')
    ax4.set_ylabel('Sample Quantiles')
    ax4.set_title('Q-Q Plot (Normality Check)')
    ax4.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=PALETTE['bg'])
    return fig


def plot_confidence_intervals(
    feature_vals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    feature_name: str = 'Rainfall (mm)',
    save_path: str = None
):
    """Plot OLS predictions with 95% confidence interval band."""
    _apply_dark_style()

    sort_idx = np.argsort(feature_vals)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(feature_vals, y_true, alpha=0.4, color=PALETTE['muted'],
               s=20, label='Observed', edgecolors='none')
    ax.plot(feature_vals[sort_idx], y_pred[sort_idx],
            color=PALETTE['primary'], lw=2, label='OLS Prediction')
    ax.fill_between(
        feature_vals[sort_idx],
        ci_lower[sort_idx], ci_upper[sort_idx],
        alpha=0.2, color=PALETTE['secondary'], label='95% CI'
    )

    ax.set_xlabel(feature_name)
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title(f'Yield vs {feature_name} — OLS with 95% Confidence Interval',
                 color=PALETTE['primary'])
    ax.legend(facecolor=PALETTE['surface'], edgecolor=PALETTE['muted'])
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=PALETTE['bg'])
    return fig


def plot_bayesian_uncertainty(
    feature_vals: np.ndarray,
    y_true: np.ndarray,
    bayes_mean: np.ndarray,
    credible_lower: np.ndarray,
    credible_upper: np.ndarray,
    posterior_samples: np.ndarray = None,
    feature_name: str = 'Rainfall (mm)',
    save_path: str = None
):
    """
    Bayesian predictive plot showing:
    - Posterior predictive mean
    - 95% credible interval
    - Optional: sampled posterior functions (shows epistemic uncertainty)
    """
    _apply_dark_style()

    sort_idx = np.argsort(feature_vals)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Posterior samples (epistemic uncertainty visualization)
    if posterior_samples is not None:
        for sample in posterior_samples[:50]:
            ax.plot(feature_vals[sort_idx], sample[sort_idx],
                    alpha=0.04, color=PALETTE['accent'], lw=1)

    ax.scatter(feature_vals, y_true, alpha=0.4, color=PALETTE['muted'],
               s=20, label='Observed', edgecolors='none', zorder=3)
    ax.fill_between(
        feature_vals[sort_idx],
        credible_lower[sort_idx], credible_upper[sort_idx],
        alpha=0.25, color=PALETTE['secondary'], label='95% Credible Interval'
    )
    ax.plot(feature_vals[sort_idx], bayes_mean[sort_idx],
            color=PALETTE['primary'], lw=2.5, label='Posterior Mean', zorder=4)

    ax.set_xlabel(feature_name)
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title(f'Bayesian Yield Prediction — 95% Credible Interval',
                 color=PALETTE['primary'])
    ax.legend(facecolor=PALETTE['surface'], edgecolor=PALETTE['muted'])
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=PALETTE['bg'])
    return fig


def plot_coefficient_comparison(ols_ci: dict, feature_names: list, save_path: str = None):
    """
    Plot OLS coefficient estimates with confidence intervals.
    Useful for interpreting which features drive yield most strongly.
    """
    _apply_dark_style()

    names = ['Intercept'] + feature_names
    coefs = ols_ci['coefficients']
    lower = ols_ci['lower']
    upper = ols_ci['upper']
    errors = np.array([coefs - lower, upper - coefs])

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE['primary'] if c > 0 else PALETTE['danger'] for c in coefs]

    ax.barh(names, coefs, xerr=errors, color=colors, alpha=0.8,
            error_kw={'ecolor': PALETTE['muted'], 'capsize': 4})
    ax.axvline(0, color=PALETTE['muted'], lw=1, linestyle='--')
    ax.set_xlabel('Coefficient Value')
    ax.set_title('OLS Coefficient Estimates with 95% CI', color=PALETTE['primary'])
    ax.grid(True, axis='x')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=PALETTE['bg'])
    return fig
