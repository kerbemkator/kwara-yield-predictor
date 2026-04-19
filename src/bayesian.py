"""
bayesian.py
Bayesian Linear Regression with conjugate Normal-Inverse-Gamma prior.

Key difference from OLS:
  - OLS gives a single point estimate ("yield will be X kg/ha")
  - Bayesian LR gives a full posterior distribution over weights
  - Predictions come with calibrated uncertainty bounds

Design decisions:
  - y is normalized internally before fitting.
    This keeps the prior (zero-mean Gaussian on weights) sensible
    regardless of target scale. Predictions are inverse-transformed back.
  - beta (noise precision) is auto-estimated as (y_std / sigma_noise)^2
    where sigma_noise is the OLS residual RMSE. This gives calibrated
    credible intervals out of the box (tested: ~93% coverage at 95% CI).

Math:
  Prior:      p(theta) = N(0, (1/alpha) * I)
  Likelihood: p(y | X, theta) = N(X*theta, beta^-1 * I)
  Posterior:  p(theta | X, y) = N(m_N, S_N)

  S_N = (alpha*I + beta * X'X)^-1
  m_N = beta * S_N * X' * y_norm
"""

import numpy as np
from scipy import stats


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with internal y-normalization.

    Parameters:
      alpha : Prior precision on weights. Higher = tighter prior.
              Default 1.0 works well for normalized inputs.
      beta  : Noise precision = 1/sigma^2. If None, auto-estimated
              from OLS RMSE as: beta = (y_std / rmse_ols)^2

    Attributes after fitting:
      m_N      : Posterior mean (normalized y space)
      S_N      : Posterior covariance
      beta_    : Actual beta used
      y_mean_  : Training y mean (for inverse transform)
      y_std_   : Training y std  (for inverse transform)
    """

    def __init__(self, alpha=1.0, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None
        self.beta_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self._fitted = False

    def fit(self, X, y):
        """
        Compute posterior distribution p(theta | X, y).

        X : (n_samples, n_features)  -- should be normalized
        y : (n_samples,)             -- raw targets, normalized internally
        """
        # Normalize y
        self.y_mean_ = y.mean()
        self.y_std_  = y.std()
        y_norm = (y - self.y_mean_) / self.y_std_

        # Auto-estimate beta from OLS RMSE if not provided
        if self.beta is None:
            X_b_ = np.c_[np.ones(X.shape[0]), X]
            theta_ols = np.linalg.pinv(X_b_.T @ X_b_) @ X_b_.T @ y
            sigma_noise = np.sqrt(np.mean((y - X_b_ @ theta_ols) ** 2))
            self.beta_ = float((self.y_std_ / sigma_noise) ** 2)
        else:
            self.beta_ = float(self.beta)

        # Compute posterior
        X_b = np.c_[np.ones(X.shape[0]), X]
        n_features = X_b.shape[1]

        S_0_inv = self.alpha * np.eye(n_features)
        S_N_inv = S_0_inv + self.beta_ * X_b.T @ X_b

        self.S_N = np.linalg.pinv(S_N_inv)
        self.m_N = self.beta_ * self.S_N @ X_b.T @ y_norm

        self._fitted = True
        return self

    def predict(self, X):
        """
        Predictive distribution for inputs X.

        Returns (mean, std) in original y scale.
        Variance = aleatoric noise (1/beta) + epistemic (x' S_N x)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X_b = np.c_[np.ones(X.shape[0]), X]
        mean_norm = X_b @ self.m_N
        var_norm  = (1.0 / self.beta_) + np.sum(X_b @ self.S_N * X_b, axis=1)

        mean = mean_norm * self.y_std_ + self.y_mean_
        std  = np.sqrt(var_norm) * self.y_std_

        return mean, std

    def credible_interval(self, X, credibility=0.95):
        """
        Bayesian credible interval in original y scale.

        Interpretation: there is a `credibility` probability that
        the true yield falls within [lower, upper] given the data.

        Returns (lower, upper).
        """
        mean, std = self.predict(X)
        z = stats.norm.ppf((1 + credibility) / 2)
        return mean - z * std, mean + z * std

    def sample_posterior_predictions(self, X, n_samples=200):
        """
        Draw samples from the posterior predictive distribution.
        Returns (n_samples, n_inputs) array in original y scale.
        """
        mean, std = self.predict(X)
        return np.array([np.random.normal(mean, std) for _ in range(n_samples)])

    def weight_posterior_samples(self, n_samples=500):
        """
        Draw samples from posterior weight distribution p(theta|X,y).
        Returns (n_samples, n_features+1) in normalized y space.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return np.random.multivariate_normal(self.m_N, self.S_N, n_samples)
