"""Step 4b: the forecasting models.

All models predict the same thing: a target month's value divided by the series' scale
(its average month over the 12 months up to the forecast date). A prediction of 1.0 means
"the same as an average recent month", and 0.8 means "20% lower". Every model has the same
``fit(rows)`` / ``predict(rows)`` interface, so the backtest can treat them alike.

The four forecasts, and why each is here
----------------------------------------
* **SimpleAverage**: always 1.0, i.e. "next month will be like the average month of the last 12".
  The baseline from step 3. Any model worth using has to beat it.
* **SameMonth**: the same month a year earlier. The other obvious baseline; step 3 showed it is
  worse than the average because imports have little seasonality.
* **LinearModel**: a weighted ridge regression written from scratch with NumPy.
  Pros: transparent (every coefficient can be read and checked), fast, and hard to overfit with
  the ridge penalty; writing it by hand also shows exactly what is being optimised. Cons: it can
  only add up effects in straight lines, so it can't learn interactions (for example, "growth
  matters more for small series") unless they are built in as features.
* **BoostedTrees**: gradient-boosted decision trees (scikit-learn's HistGradientBoosting, the
  same method as LightGBM). Pros: learns interactions and non-linear effects on its own, and
  is a strong general-purpose method for tabular data. Cons: harder to explain, more settings to
  tune, and with only a few years of history it can fit noise. It is used as a second opinion.

Training rows are weighted by the series' scale, so getting a big series right matters more
than a tiny one. That matches how the models are judged: on errors in Bahraini dinars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import CATEGORICAL, FEATURES, NUMERIC

# A month more than 10x a series' recent average is treated as 10x when training. Such months are
# one-off deliveries (an aircraft engine, say); without the cap a handful of them would pull the
# squared-error fit far away from every ordinary month.
TARGET_CAP = 10.0
# Features expressed as a multiple of the scale. Missing values (for example, no history two years
# back) are filled with 1.0, "a normal month", and flagged with a separate 0/1 column.
RATIO_FEATURES = ["last_1", "mean_3", "mean_6", "prev_12", "growth", "same_month_1y", "same_month_2y", "volatility"]


def clipped_target(rows: pd.DataFrame) -> np.ndarray:
    """The training target (value / scale), limited to 0..TARGET_CAP."""
    return np.clip(rows["target"].to_numpy(dtype=float), 0, TARGET_CAP)


class SimpleAverage:
    """Forecast = the average month of the last 12 months (a prediction of 1.0 on the scaled target)."""
    name = "Last year's average month"

    def fit(self, rows: pd.DataFrame) -> "SimpleAverage":
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.ones(len(rows))


class SameMonth:
    """Forecast = the same calendar month a year earlier (1.0, the average, if that month is unknown)."""
    name = "Same month last year"

    def fit(self, rows: pd.DataFrame) -> "SameMonth":
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["same_month_1y"].fillna(1.0).to_numpy(dtype=float)


class LinearModel:
    """Weighted ridge regression: minimise  sum_i w_i (y_i - x_i.b)^2 + ridge * |b|^2  (intercept not penalised).

    How the inputs are prepared (``_raw_matrix`` and ``_design``):
      * missing values are filled with a neutral value (1.0 for ratios, 0 otherwise) plus a
        0/1 "was missing" column, so the model can learn that missing history means something;
      * ratio features are capped at TARGET_CAP, like the target;
      * section and region become 0/1 columns (one level of each is left out as the reference,
        otherwise the columns would add up to the intercept and the problem would have no
        unique solution);
      * every column is standardised to weighted mean 0 and standard deviation 1. This puts
        all features on the same footing for the ridge penalty, and makes gradient descent
        converge much faster.

    The ridge penalty shrinks coefficients towards zero. With many correlated features (the
    3-, 6- and 12-month averages move together) plain least squares can give large, unstable
    coefficients that cancel each other out; the penalty trades a little bias for much more
    stable forecasts. Its strength is chosen in the backtest on the tuning year.
    """
    name = "Linear model (from scratch)"

    def __init__(self, ridge: float = 1.0, method: str = "normal", n_iter: int = 300):
        self.ridge, self.method, self.n_iter = ridge, method, n_iter
        self.loss_history: list[float] = []

    # -- design matrix --------------------------------------------------------
    def _raw_matrix(self, rows: pd.DataFrame) -> np.ndarray:
        """Feature columns before standardising (missing-value flags, capped ratios, 0/1 columns)."""
        cols = []
        for name in NUMERIC:
            x = rows[name].to_numpy(dtype=float)
            if name in RATIO_FEATURES:
                cols.append(np.isnan(x).astype(float))
                x = np.clip(np.nan_to_num(x, nan=1.0), 0, TARGET_CAP)
            else:
                x = np.nan_to_num(x, nan=0.0)
            cols.append(x)
        for name, levels in self.levels_.items():
            values = rows[name].to_numpy()
            cols.extend((values == level).astype(float) for level in levels[1:])   # first level is the reference
        return np.column_stack(cols)

    def _design(self, rows: pd.DataFrame) -> np.ndarray:
        """Standardised feature matrix with a column of ones (the intercept) in front."""
        X = (self._raw_matrix(rows)[:, self.keep_] - self.mean_) / self.std_
        return np.column_stack([np.ones(len(X)), X])

    # -- fitting ----------------------------------------------------------------
    def fit(self, rows: pd.DataFrame) -> "LinearModel":
        """Learn the standardisation and the coefficients from the training rows."""
        self.levels_ = {c: np.sort(rows[c].unique()) for c in ("section_id", "region_id")}
        raw = self._raw_matrix(rows)
        # Weights: each row counts in proportion to its series' size, normalised to sum to 1.
        w = rows["scale"].to_numpy(dtype=float)
        w = w / w.sum()
        mean = w @ raw
        std = np.sqrt(w @ (raw - mean) ** 2)
        self.keep_ = std > 1e-9 * np.maximum(1.0, np.abs(mean))     # columns that are constant in training carry no information
        self.mean_, self.std_ = mean[self.keep_], std[self.keep_]
        X = self._design(rows)
        y = clipped_target(rows)
        penalty = np.full(X.shape[1], self.ridge)
        penalty[0] = 0.0                     # the intercept sets the overall level and is not shrunk
        if self.method == "normal":
            self.coef_ = self.solve_normal_equations(X, y, w, penalty)
        else:
            self.coef_ = self.gradient_descent(X, y, w, penalty)
        return self

    @staticmethod
    def solve_normal_equations(X, y, w, penalty):
        """b = (X'WX + P)^-1 X'Wy, solved without forming the inverse.

        Setting the gradient of the loss to zero gives the linear system (X'WX + P) b = X'Wy.
        np.linalg.solve factorises the matrix, which is faster and more accurate than inverting it.
        """
        XtW = X.T * w
        return np.linalg.solve(XtW @ X + np.diag(penalty), XtW @ y)

    def gradient_descent(self, X, y, w, penalty):
        """Repeatedly step downhill: b <- b - rate * gradient, where gradient = 2 X'W(Xb - y) + 2 P b.

        The step size is 1 / L, where L (the largest eigenvalue of the Hessian 2(X'WX + P)) is how
        sharply the loss can curve; any bigger step risks overshooting the minimum.

        For a convex quadratic loss like this one, gradient descent with this step size is
        guaranteed to converge to the same answer as the normal equations. It is included to show
        that, and because it scales to problems too large to solve directly. The loss is recorded
        at every step so the convergence can be plotted (step 4, figure 4).
        """
        hessian = 2 * ((X.T * w) @ X + np.diag(penalty))
        self.learning_rate = 1.0 / np.linalg.eigvalsh(hessian).max()
        b = np.zeros(X.shape[1])
        self.loss_history = []
        for _ in range(self.n_iter):
            residual = X @ b - y
            self.loss_history.append(float(w @ residual ** 2 + penalty @ b ** 2))
            gradient = 2 * (X.T @ (w * residual)) + 2 * penalty * b
            b -= self.learning_rate * gradient
        return b

    @staticmethod
    def loss(X, y, w, penalty, b) -> float:
        """The ridge loss for coefficients b (used by the tests to compare the two solvers)."""
        residual = X @ b - y
        return float(w @ residual ** 2 + penalty @ b ** 2)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Scaled forecasts; negative predictions are set to 0, since imports can't be negative."""
        return np.clip(self._design(rows) @ self.coef_, 0, None)

    def feature_names(self) -> list[str]:
        """Name of each coefficient, in order (intercept first)."""
        names = []
        for name in NUMERIC:
            if name in RATIO_FEATURES:
                names.append(f"{name} (missing)")
            names.append(name)
        for col, levels in self.levels_.items():
            names += [f"{col}={level}" for level in levels[1:]]
        return ["intercept"] + [n for n, keep in zip(names, self.keep_) if keep]


class BoostedTrees:
    """Gradient-boosted trees, weighted by scale**weight_power.

    Boosting builds many small decision trees one after another; each new tree is fitted to what
    the trees so far got wrong, and its predictions are added in with a small learning rate.

    Choices (tuned on 2024 in the backtest):
      * The loss must aim at the *average* (squared error or Poisson), not the median (absolute
        error): forecasts are added up across hundreds of series, and medians of lopsided series
        add up to too little (step 4 measured a 5-9% undershoot with absolute error).
      * Small trees (4 leaves) with at least 1,000 rows per leaf, and a low learning rate, keep the
        model from memorising individual months of individual series.
      * Rows are weighted by the square root of the scale: big series still matter more, but not
        so much that the model ignores everything else.
    """
    name = "Gradient-boosted trees"

    def __init__(self, loss: str = "squared_error", learning_rate: float = 0.03, max_iter: int = 100,
                 max_leaf_nodes: int = 4, min_samples_leaf: int = 1000, l2_regularization: float = 0.0,
                 weight_power: float = 0.5, random_state: int = 0):
        self.loss = loss
        self.params = dict(learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization)
        self.weight_power = weight_power
        self.random_state = random_state

    def fit(self, rows: pd.DataFrame) -> "BoostedTrees":
        # Features with no information in this training table (for example "same month two years ago"
        # when only one year of history is used) are left out.
        self.features_ = [f for f in FEATURES if rows[f].nunique(dropna=True) > 1]
        self.model_ = HistGradientBoostingRegressor(
            loss=self.loss, categorical_features=[f in CATEGORICAL for f in self.features_],
            early_stopping=False, random_state=self.random_state, **self.params)
        weight = rows["scale"].to_numpy(dtype=float) ** self.weight_power
        self.model_.fit(rows[self.features_].to_numpy(dtype=float), clipped_target(rows), sample_weight=weight)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model_.predict(rows[self.features_].to_numpy(dtype=float)), 0, None)
