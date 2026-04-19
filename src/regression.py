"""
regression.py
Ordinary Least Squares regression implemented from scratch using NumPy only.
No scikit-learn in the core implementation — this is intentional.

Mathematical foundation:
  Normal Equation: θ = (XᵀX)⁻¹ Xᵀy
  This gives the closed-form optimal solution that minimizes sum of squared residuals.
"""

import numpy as np
from scipy import stats


class OLSRegression:
    """
    From-scratch OLS Linear Regression.

    Key attributes after fitting:
      theta_       : Full parameter vector [bias, w1, w2, ...]
      weights_     : Feature weights only
      bias_        : Intercept term
      residuals_   : y - y_hat
      mse_         : Mean Squared Error
      rmse_        : Root Mean Squared Error
      r2_          : R² coefficient of determination
      adj_r2_      : Adjusted R²
    """

    def __init__(self):
        self.theta_ = None
        self.weights_ = None
        self.bias_ = None
        self.residuals_ = None
        self.mse_ = None
        self.rmse_ = None
        self.r2_ = None
        self.adj_r2_ = None
        self._n = None
        self._p = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLSRegression':
        """
        Fit model using the Normal Equation.

        Parameters:
          X : (n_samples, n_features)
          y : (n_samples,)
        """
        n, p = X.shape
        self._n, self._p = n, p

        # Augment with bias column
        X_b = np.c_[np.ones(n), X]

        # Normal equation — pinv handles near-singular matrices
        self.theta_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias_ = self.theta_[0]
        self.weights_ = self.theta_[1:]

        # Compute diagnostics
        y_pred = self.predict(X)
        self.residuals_ = y - y_pred
        self.mse_ = np.mean(self.residuals_ ** 2)
        self.rmse_ = np.sqrt(self.mse_)
        self.r2_ = self._r_squared(y, y_pred)
        self.adj_r2_ = 1 - (1 - self.r2_) * (n - 1) / (n - p - 1)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for input X."""
        if self.theta_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.theta_

    def confidence_intervals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05
    ) -> dict:
        """
        Compute (1-alpha)% confidence intervals for each coefficient.

        CI = θ ± t_crit * SE(θ)
        where SE(θ) = sqrt(diag(σ² (XᵀX)⁻¹))
        """
        n, p = X.shape
        X_b = np.c_[np.ones(n), X]

        # Unbiased variance estimate
        sigma2 = np.sum(self.residuals_ ** 2) / (n - p - 1)

        # Covariance matrix of coefficient estimates
        cov_matrix = sigma2 * np.linalg.pinv(X_b.T @ X_b)
        se = np.sqrt(np.diag(cov_matrix))

        # t critical value (two-tailed)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - p - 1)

        lower = self.theta_ - t_crit * se
        upper = self.theta_ + t_crit * se

        # t-statistics and p-values for each coefficient
        t_stats = self.theta_ / se
        p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - p - 1)

        return {
            'coefficients': self.theta_,
            'std_errors': se,
            'lower': lower,
            'upper': upper,
            't_stats': t_stats,
            'p_values': p_values,
            'alpha': alpha
        }

    def summary(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> str:
        """Print a regression summary table similar to statsmodels output."""
        ci = self.confidence_intervals(X, y)
        names = ['Intercept'] + (feature_names if feature_names else [f'x{i}' for i in range(self._p)])

        lines = [
            "=" * 65,
            "OLS REGRESSION RESULTS — Kwara State Yield Predictor",
            "=" * 65,
            f"{'R²:':<25} {self.r2_:.4f}",
            f"{'Adjusted R²:':<25} {self.adj_r2_:.4f}",
            f"{'RMSE (kg/ha):':<25} {self.rmse_:.2f}",
            f"{'MSE:':<25} {self.mse_:.2f}",
            f"{'N observations:':<25} {self._n}",
            "-" * 65,
            f"{'Variable':<20} {'Coef':>10} {'SE':>10} {'t':>8} {'p-val':>8}",
            "-" * 65,
        ]

        for name, coef, se, t, p in zip(
            names,
            ci['coefficients'],
            ci['std_errors'],
            ci['t_stats'],
            ci['p_values']
        ):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            lines.append(f"{name:<20} {coef:>10.3f} {se:>10.3f} {t:>8.3f} {p:>8.4f} {sig}")

        lines.append("=" * 65)
        lines.append("Significance: *** p<0.001  ** p<0.01  * p<0.05")
        return "\n".join(lines)

    def _r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
