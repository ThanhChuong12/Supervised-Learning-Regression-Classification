import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from scipy.stats import ttest_rel, wilcoxon as scipy_wilcoxon


# =============================================================================
# BASIS EXPANSION MODULE
# Groups all feature engineering / design matrix construction utilities.
# =============================================================================
class BasisExpansion:
    """
    Utilities for constructing design matrices via basis expansion.
    Supports Polynomial, RBF, Sigmoid, and Spline feature transformers.
    """

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_stable(z: np.ndarray) -> np.ndarray:
        return BasisExpansion.sigmoid(z)

    @staticmethod
    def make_sigmoid_basis(
        X_s: np.ndarray,
        centers: np.ndarray,
        slope: float,
        include_linear: bool = True,
        include_bias: bool = True,
    ) -> np.ndarray:
        N, D = X_s.shape
        M = centers.shape[1]

        parts = []
        if include_bias:
            parts.append(np.ones((N, 1), dtype=float))

        if include_linear:
            parts.append(X_s.astype(float))

        sig_parts = []
        for d in range(D):
            z = slope * (X_s[:, [d]] - centers[d][None, :])
            sig_parts.append(BasisExpansion.sigmoid(z))
        parts.append(np.hstack(sig_parts))

        return np.hstack(parts)

    @staticmethod
    def poly_features(X_s: np.ndarray, degree: int) -> np.ndarray:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        feats = [X_s.astype(float)]
        for p in range(2, degree + 1):
            feats.append(X_s.astype(float) ** p)
        return np.hstack(feats)

    @staticmethod
    def rbf_features(X_s: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
        x2 = np.sum(X_s * X_s, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T
        d2 = x2 + c2 - 2.0 * (X_s @ centers.T)
        return np.exp(-gamma * d2)

    @staticmethod
    def spline_features(
        X_s: np.ndarray,
        n_knots: int,
        degree: int = 3,
        *,
        transformer: dict | None = None,
        fit: bool = False,
    ) -> tuple[np.ndarray, dict]:
        N, D = X_s.shape
        if transformer is None or fit:
            # Generate interior knots based on quantiles
            knots = []
            for d in range(D):
                q = np.linspace(0, 1, n_knots + 2)[1:-1]
                k_d = np.quantile(X_s[:, d], q)
                knots.append(k_d)
            transformer = {'knots': knots, 'degree': degree}

        knots = transformer['knots']
        degree_actual = transformer['degree']

        parts = []
        for d in range(D):
            x = X_s[:, d:d+1]  # (N, 1)
            for p in range(2, degree_actual + 1):
                parts.append(x ** p)

            k_d = knots[d]
            for k in k_d:
                trunc = np.maximum(0, x - k) ** degree_actual
                parts.append(trunc)

        Z = np.hstack(parts) if parts else np.zeros((N, 0))
        return Z.astype(float), transformer

    @staticmethod
    def sigmoid_features(X_s: np.ndarray, centers_per_feature: np.ndarray, slope: float) -> np.ndarray:
        N, D = X_s.shape
        M = centers_per_feature.shape[1]
        out = []
        for d in range(D):
            z = slope * (X_s[:, [d]] - centers_per_feature[d][None, :])
            out.append(BasisExpansion.sigmoid_stable(z))
        return np.hstack(out)

    @staticmethod
    def make_design_matrix(
        X_s: np.ndarray,
        basis: dict,
        interactions: np.ndarray | None = None,
        add_linear: bool = True,
    ) -> np.ndarray:
        parts = [np.ones((X_s.shape[0], 1), dtype=float)]
        if add_linear:
            parts.append(X_s.astype(float))

        if basis.get('poly_degree') is not None:
            d = int(basis['poly_degree'])
            if d >= 2:
                parts.append(BasisExpansion.poly_features(X_s, degree=d)[:, X_s.shape[1]:])

        if basis.get('rbf') is not None:
            r = basis['rbf']
            parts.append(BasisExpansion.rbf_features(X_s, centers=r['centers'], gamma=float(r['gamma'])))

        if basis.get('sigmoid') is not None:
            s = basis['sigmoid']
            parts.append(BasisExpansion.sigmoid_features(X_s, centers_per_feature=s['centers'], slope=float(s['slope'])))

        if basis.get('spline') is not None:
            sp = basis['spline']
            n_knots = int(sp['n_knots'])
            degree = int(sp.get('degree', 3))
            transformer = sp.get('transformer', None)
            Z, fitted = BasisExpansion.spline_features(
                X_s,
                n_knots=n_knots,
                degree=degree,
                transformer=transformer,
                fit=(transformer is None),
            )
            sp['transformer'] = fitted
            parts.append(Z)

        if interactions is not None:
            parts.append(interactions)

        return np.hstack(parts)

    @staticmethod
    def compute_rbf_gamma(centers: np.ndarray) -> float:
        d2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        triu = np.triu_indices(d2.shape[0], k=1)
        med = np.median(d2[triu]) if np.any(d2[triu] > 0) else 1.0
        return 1.0 / (2.0 * max(med, 1e-9))

    @staticmethod
    def build_4basis_configs(
        X_train_s: np.ndarray,
        poly_degree: int = 3,
        rbf_K: int = 50,
        sig_M: int = 5,
        spline_knots: int = 5,
        rng_seed: int = 0,
    ) -> dict:
        D = X_train_s.shape[1]
        poly_config = {'poly_degree': poly_degree, 'rbf': None, 'sigmoid': None, 'spline': None}

        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(X_train_s.shape[0], size=min(rbf_K, X_train_s.shape[0]), replace=False)
        rbf_centers = X_train_s[idx]
        rbf_gamma = BasisExpansion.compute_rbf_gamma(rbf_centers)
        rbf_config = {
            'poly_degree': None,
            'rbf': {'centers': rbf_centers, 'gamma': rbf_gamma},
            'sigmoid': None,
            'spline': None,
        }

        qs = np.linspace(0.1, 0.9, sig_M)
        centers_sig = np.vstack([np.quantile(X_train_s[:, d], qs) for d in range(D)])
        sigmoid_config = {
            'poly_degree': None,
            'rbf': None,
            'sigmoid': {'centers': centers_sig, 'slope': 2.0},
            'spline': None,
        }

        spline_config = {
            'poly_degree': None,
            'rbf': None,
            'sigmoid': None,
            'spline': {'n_knots': spline_knots, 'degree': 3},
        }

        return {
            f'Polynomial (d={poly_degree})': poly_config,
            f'RBF (K={rbf_K})': rbf_config,
            f'Sigmoid (M={sig_M})': sigmoid_config,
            f'Spline (knots={spline_knots})': spline_config,
        }


# =============================================================================
# LINEAR REGRESSION MODULE
# Groups OLS, Ridge, Lasso, Elastic Net, WLS, Mini-batch GD, and helpers.
# =============================================================================
class LinearRegression:
    """
    Linear Regression implementations including:
    - OLS (closed-form Normal Equations)
    - Ridge (L2 regularization)
    - Lasso (Coordinate Descent)
    - Elastic Net (Coordinate Descent)
    - WLS (Weighted Least Squares)
    - Mini-batch Gradient Descent with learning rate schedules
    """

    # ------------------------------------------------------------------
    # Learning rate schedules (static helpers)
    # ------------------------------------------------------------------
    @staticmethod
    def step_decay_schedule(epoch: int, initial_lr: float, drop_rate: float = 0.5,
                            epochs_drop: int = 10) -> float:
        return initial_lr * (drop_rate ** np.floor(epoch / epochs_drop))

    @staticmethod
    def cosine_annealing_schedule(epoch: int, initial_lr: float, T_max: int) -> float:
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / T_max))

    # ------------------------------------------------------------------
    # Core fitting methods
    # ------------------------------------------------------------------
    @staticmethod
    def fit_ols(Phi: np.ndarray, y: np.ndarray, bias_is_first: bool = True) -> np.ndarray:
        # Normal Equations: w = (Phi^T Phi)^(-1) Phi^T y
        PhiT_Phi = Phi.T @ Phi
        PhiT_y = Phi.T @ y

        # Solve linear system; fall back to least-squares if matrix is singular
        try:
            w = np.linalg.solve(PhiT_Phi, PhiT_y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)

        return w

    @staticmethod
    def fit_ridge(Phi: np.ndarray, y: np.ndarray, lam: float, bias_is_first: bool = True) -> np.ndarray:
        # P is the number of features in the input matrix
        P = Phi.shape[1]

        # compute the core part of the normal equation
        A = Phi.T @ Phi

        # create a diagonal matrix containing the lambda penalty coefficients
        reg = lam * np.eye(P)

        # do not apply L2 penalty to the intercept term (bias) in the first position
        if bias_is_first:
            reg[0, 0] = 0.0

        # find the weight vector w by solving the linear system
        return np.linalg.solve(A + reg, Phi.T @ y)

    @staticmethod
    def fit_ridge_closed_form(Phi: np.ndarray, y: np.ndarray, lam: float, bias_is_first: bool = True) -> np.ndarray:
        return LinearRegression.fit_ridge(Phi, y, lam, bias_is_first=bias_is_first)

    @staticmethod
    def soft_threshold(rho: float, lam: float) -> float:
        # force the weights of unimportant features to zero
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0.0

    @staticmethod
    def fit_lasso_cd(Phi: np.ndarray, y: np.ndarray, lam: float, num_iters: int = 1000, tol: float = 1e-4, bias_is_first: bool = True, w_init: np.ndarray = None) -> np.ndarray:
        n_samples, n_features = Phi.shape

        # Use w_init if provided, otherwise initialize with zeros
        w = np.zeros(n_features) if w_init is None else w_init.copy()

        # z is the sum of squares of the elements in each column of matrix Phi
        z = np.sum(Phi**2, axis=0)

        for _ in range(num_iters):
            w_old = w.copy()

            for j in range(n_features):
                if z[j] == 0:
                    continue

                # compute the current prediction error
                y_pred = Phi @ w

                # rho_j is the correlation between feature j and the residual (temporarily excluding w_j)
                rho_j = Phi[:, j].T @ (y - y_pred) + w[j] * z[j]

                # the first column (bias) is not penalized, other columns go through soft-thresholding
                if bias_is_first and j == 0:
                    w[j] = rho_j / z[j]
                else:
                    w[j] = LinearRegression.soft_threshold(rho_j, lam) / z[j]

            # stop the algorithm if the change in w is very small (converged)
            if np.max(np.abs(w - w_old)) < tol:
                break

        return w

    @staticmethod
    def fit_elastic_net_cd(Phi: np.ndarray, y: np.ndarray, lam1: float, lam2: float, num_iters: int = 1000, tol: float = 1e-4, bias_is_first: bool = True) -> np.ndarray:
        # implement Elastic Net using Coordinate Descent
        n_samples, n_features = Phi.shape
        w = np.zeros(n_features)

        # z is the sum of squares of the elements in each column
        z = np.sum(Phi**2, axis=0)

        for _ in range(num_iters):
            w_old = w.copy()

            for j in range(n_features):
                if z[j] == 0: continue

                # compute the current prediction error
                y_pred = Phi @ w

                # rho_j is the correlation between feature j and the residual error
                rho_j = Phi[:, j].T @ (y - y_pred) + w[j] * z[j]

                if bias_is_first and j == 0:
                    # the intercept term is not penalized
                    w[j] = rho_j / z[j]
                else:
                    # numerator applies L1 penalty, denominator adds L2 penalty
                    w[j] = LinearRegression.soft_threshold(rho_j, lam1) / (z[j] + lam2)

            # stop the algorithm if the weights change very little
            if np.max(np.abs(w - w_old)) < tol: break

        return w

    @staticmethod
    def estimate_weights_from_residuals(residuals: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        # Weights are inversely proportional to squared residuals
        weights = 1.0 / (residuals ** 2 + epsilon)

        # Normalize weights to have mean = 1
        weights = weights / np.mean(weights)

        return weights

    @staticmethod
    def fit_wls(Phi: np.ndarray, y: np.ndarray, weights: np.ndarray,
                bias_is_first: bool = True) -> np.ndarray:
        # Tối ưu RAM: Dùng broadcasting thay vì ma trận chéo N x N
        sqrt_w = np.sqrt(weights)

        # Transform features and targets
        Phi_weighted = Phi * sqrt_w[:, np.newaxis]
        y_weighted = y * sqrt_w

        # Solve weighted normal equations: w = (Phi_w^T Phi_w)^(-1) Phi_w^T y_w
        w = LinearRegression.fit_ols(Phi_weighted, y_weighted, bias_is_first=bias_is_first)

        return w

    @staticmethod
    def fit_ols_minibatch_gd(
        Phi: np.ndarray, y: np.ndarray,
        lr_schedule: str = 'step_decay',
        initial_lr: float = 0.01,
        batch_size: int = 32,
        num_epochs: int = 100,
        bias_is_first: bool = True,
        drop_rate: float = 0.5,
        epochs_drop: int = 10
    ) -> tuple:
        N, D = Phi.shape
        w = np.zeros(D)
        loss_history = []

        for epoch in range(num_epochs):
            # Get learning rate for current epoch
            if lr_schedule == 'step_decay':
                lr = LinearRegression.step_decay_schedule(epoch, initial_lr, drop_rate, epochs_drop)
            elif lr_schedule == 'cosine_annealing':
                lr = LinearRegression.cosine_annealing_schedule(epoch, initial_lr, num_epochs)
            else:
                lr = initial_lr

            # Shuffle data
            indices = np.random.permutation(N)
            Phi_shuffled = Phi[indices]
            y_shuffled = y[indices]

            # Mini-batch gradient descent
            for i in range(0, N, batch_size):
                batch_end = min(i + batch_size, N)
                Phi_batch = Phi_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]

                # Compute gradient: grad = -2/n * Phi^T (y - Phi w)
                predictions = Phi_batch @ w
                errors = y_batch - predictions
                gradient = -2.0 / len(y_batch) * (Phi_batch.T @ errors)

                # Update weights
                w = w - lr * gradient

            # Compute epoch loss
            y_pred = Phi @ w
            epoch_loss = np.mean((y - y_pred) ** 2)
            loss_history.append(epoch_loss)

        return w, loss_history

    # ------------------------------------------------------------------
    # Prediction and metrics
    # ------------------------------------------------------------------
    @staticmethod
    def predict(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
        return Phi @ w

    @staticmethod
    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        mse_val = np.mean((y_true - y_pred) ** 2)
        rmse = float(np.sqrt(mse_val))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return {'MSE': float(mse_val), 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        return float(np.mean((y_true - y_pred) ** 2))

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_true - y_pred

    @staticmethod
    def breusch_pagan_test(Phi: np.ndarray, residuals: np.ndarray) -> float:
        from scipy import stats

        N = len(residuals)

        # Step 1: Compute squared residuals
        residuals_squared = residuals ** 2

        # Step 2: Regress squared residuals on original features
        # Normalize squared residuals by their mean
        sigma_squared = np.mean(residuals_squared)
        normalized_residuals_sq = residuals_squared / sigma_squared

        # Fit auxiliary regression
        w_aux = LinearRegression.fit_ols(Phi, normalized_residuals_sq, bias_is_first=True)
        fitted_values = Phi @ w_aux

        # Step 3: Compute test statistic
        # LM = N * R^2 from auxiliary regression
        ss_total = np.sum((normalized_residuals_sq - np.mean(normalized_residuals_sq)) ** 2)
        ss_residual = np.sum((normalized_residuals_sq - fitted_values) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

        lm_statistic = N * r_squared

        # Step 4: Compare to chi-square distribution
        # Degrees of freedom = number of features (excluding bias)
        df = Phi.shape[1] - 1
        p_value = 1 - stats.chi2.cdf(lm_statistic, df)

        return p_value

    # ------------------------------------------------------------------
    # Cross-validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def time_series_cv_indices(n_samples: int, k: int = 10, random_seed: int = 42):
        # Split the data into (k + 1) equal parts
        chunk_size = n_samples // (k + 1)

        folds = []

        for i in range(1, k + 1):
            # Training set: from the beginning up to the current chunk
            train_end = i * chunk_size
            train_idx = np.arange(0, train_end)

            # Validation set
            val_end = (i + 1) * chunk_size if i < k else n_samples
            val_idx = np.arange(train_end, val_end)

            folds.append((train_idx, val_idx))

        return folds

    @staticmethod
    def kfold_cv_lasso(Phi: np.ndarray, y: np.ndarray, lambdas: list, k: int = 10):
        # sort lambdas in descending order to make warm start more effective
        lambdas = sorted(lambdas, reverse=True)

        # call the function to split indices into folds
        folds = LinearRegression.time_series_cv_indices(len(Phi), k=k)

        cv_errors = []

        # create a list to store the initial weights w for each fold.
        # since there are k folds, we need k different starting points. initially all are None (will initialize w=0)
        w_inits = [None] * k

        for lam in lambdas:
            fold_errors = []

            for i, (train_idx, val_idx) in enumerate(folds):
                Phi_tr, y_tr = Phi[train_idx], y[train_idx]
                Phi_va, y_va = Phi[val_idx], y[val_idx]

                # fit the model with Warm Start: pass w_inits[i] from the previous lambda loop
                w = LinearRegression.fit_lasso_cd(Phi_tr, y_tr, lam, w_init=w_inits[i])

                # update the starting point for fold i to use for the next lambda value
                w_inits[i] = w.copy()

                # compute the error on the validation set
                y_pred = Phi_va @ w
                mse_val = np.mean((y_va - y_pred)**2)
                fold_errors.append(mse_val)

            # store the average error across k folds for the current lambda
            cv_errors.append(np.mean(fold_errors))

        # find the lambda with the smallest average validation error
        best_idx = np.argmin(cv_errors)
        best_lam = lambdas[best_idx]

        return best_lam, cv_errors, lambdas

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def add_bias(X_like: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X_like.shape[0], 1), dtype=float), X_like])

    @staticmethod
    def evaluate_and_print(name, y_true, y_pred):
        # Calculate metrics
        eval_metrics = LinearRegression.metrics(y_true, y_pred)
        test_mse = LinearRegression.mse(y_true, y_pred)

        # Print formatted row
        print(f"{name:<35} | {test_mse:<10.4f} | {eval_metrics['RMSE']:<10.4f} | {eval_metrics['MAE']:<10.4f} | {eval_metrics['R2']:<10.4f}")

    @staticmethod
    def evaluate_configs(
        all_configs: dict,
        X_train_s: np.ndarray,
        y_train: np.ndarray,
        X_val_s: np.ndarray,
        y_val: np.ndarray,
        X_test_s: np.ndarray,
        y_test: np.ndarray,
        lam: float = 1.0,
    ) -> dict:
        results = {}
        for model_name, cfg in all_configs.items():
            Phi_tr = BasisExpansion.make_design_matrix(X_train_s, basis=cfg, add_linear=True)
            Phi_va = BasisExpansion.make_design_matrix(X_val_s, basis=cfg, add_linear=True)
            Phi_te = BasisExpansion.make_design_matrix(X_test_s, basis=cfg, add_linear=True)

            w = LinearRegression.fit_ridge(Phi_tr, y_train, lam=lam)
            pred_tr = LinearRegression.predict(Phi_tr, w)
            pred_va = LinearRegression.predict(Phi_va, w)
            pred_te = LinearRegression.predict(Phi_te, w)

            results[model_name] = {
                'train': LinearRegression.metrics(y_train, pred_tr),
                'val': LinearRegression.metrics(y_val, pred_va),
                'test': LinearRegression.metrics(y_test, pred_te),
            }

        return results


# =============================================================================
# FEATURE SELECTOR MODULE
# Groups greedy feature selection algorithms.
# =============================================================================
class FeatureSelector:
    """
    Greedy feature selection strategies:
    - Forward Selection
    - Backward Elimination
    - Feature group construction for domain-specific datasets
    """

    @staticmethod
    def select_feature_groups(names: list[str]) -> dict[str, list[int]]:
        idx = {n: i for i, n in enumerate(names)}

        def pick(prefix: str) -> list[int]:
            return [i for n, i in idx.items() if n.startswith(prefix)]

        groups: dict[str, list[int]] = {}
        groups['lights'] = [idx['lights']] if 'lights' in idx else []
        groups['temp_indoor'] = pick('T')
        if 'T_out' in idx and idx['T_out'] in groups['temp_indoor']:
            groups['temp_indoor'].remove(idx['T_out'])
            groups['temp_outdoor'] = [idx['T_out']]
        else:
            groups['temp_outdoor'] = []

        groups['humidity'] = pick('RH_')
        if 'RH_out' in idx:
            groups['humidity_outdoor'] = [idx['RH_out']]
        else:
            groups['humidity_outdoor'] = []

        for k in ['Press_mm_hg', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']:
            groups[k] = [idx[k]] if k in idx else []

        return {g: cols for g, cols in groups.items() if len(cols) > 0}

    @staticmethod
    def interaction_terms(X_s: np.ndarray, cols: list[int]) -> np.ndarray:
        cols = list(cols)
        k = len(cols)
        if k < 2:
            return np.zeros((X_s.shape[0], 0), dtype=float)
        inter = []
        for a in range(k):
            for b in range(a + 1, k):
                inter.append((X_s[:, cols[a]] * X_s[:, cols[b]])[:, None])
        return np.hstack(inter)

    @staticmethod
    def forward_selection(Phi_train, y_train, Phi_val, y_val, k_features, lam=0.1):
        selected = [0]  # always keep the bias column (column 0)
        remaining = list(range(1, Phi_train.shape[1]))

        for _ in range(k_features):
            best_feature = None
            best_error = float("inf")

            for f in remaining:
                trial = selected + [f]
                # Fit the model on the training set
                w = LinearRegression.fit_ridge(Phi_train[:, trial], y_train, lam)
                # Predict and compute the error on the validation set
                y_pred_val = Phi_val[:, trial] @ w
                error = np.mean((y_val - y_pred_val)**2)

                if error < best_error:
                    best_error = error
                    best_feature = f

            selected.append(best_feature)
            remaining.remove(best_feature)

        return selected

    @staticmethod
    def backward_elimination(Phi_train, y_train, Phi_val, y_val, target_features, lam=0.1):
        features = list(range(Phi_train.shape[1]))

        while len(features) > target_features:
            best_error = float("inf")
            worst_feature = None

            for f in features:
                if f == 0: continue

                trial = [x for x in features if x != f]
                w = LinearRegression.fit_ridge(Phi_train[:, trial], y_train, lam)
                y_pred_val = Phi_val[:, trial] @ w
                error = np.mean((y_val - y_pred_val)**2)

                if error < best_error:
                    best_error = error
                    worst_feature = f

            features.remove(worst_feature)

        return features


# =============================================================================
# ROBUST REGRESSION MODULE
# IRLS with Huber Loss for outlier-robust regression.
# =============================================================================
class RobustRegression:
    """
    Robust Regression via Iteratively Reweighted Least Squares (IRLS)
    using the Huber Loss function.
    """

    @staticmethod
    def huber_loss(residuals, delta):
        abs_r = np.abs(residuals)
        loss = np.where(
            abs_r <= delta,
            0.5 * residuals**2,
            delta * abs_r - 0.5 * delta**2
        )
        return np.mean(loss)

    @staticmethod
    def huber_weights(residuals, delta):
        """Tính trọng số IRLS từ Huber Loss."""
        abs_r = np.abs(residuals)
        weights = np.where(
            abs_r <= delta,
            1.0,
            delta / (abs_r + 1e-8)
        )
        return weights

    @staticmethod
    def fit_irls_huber(Phi, y, delta=1.345, max_iter=50, tol=1e-6, lam=0.0):
        N, D = Phi.shape

        # Bước 0: Khởi tạo bằng OLS (hoặc Ridge nếu lam > 0)
        if lam > 0:
            w = LinearRegression.fit_ridge(Phi, y, lam, bias_is_first=True)
        else:
            w = LinearRegression.fit_ols(Phi, y, bias_is_first=True)

        loss_history = []

        for iteration in range(max_iter):
            # 1. Tính phần dư (residuals)
            y_pred = Phi @ w
            residuals = y - y_pred

            # 2. Tính Huber Loss hiện tại
            current_loss = RobustRegression.huber_loss(residuals, delta)
            loss_history.append(current_loss)

            # 3. Tính trọng số IRLS từ Huber
            sample_weights = RobustRegression.huber_weights(residuals, delta)

            # 4. Giải bài toán WLS bằng Broadcasting để ngăn tràn RAM (MemoryError)
            sqrt_w = np.sqrt(sample_weights)
            Phi_w = Phi * sqrt_w[:, np.newaxis]
            y_w = y * sqrt_w

            # Giải bằng Ridge (hoặc OLS nếu lam=0)
            A = Phi_w.T @ Phi_w
            reg = lam * np.eye(D)
            reg[0, 0] = 0.0  # Không phạt bias
            w_new = np.linalg.solve(A + reg, Phi_w.T @ y_w)

            # 5. Kiểm tra hội tụ
            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                y_pred = Phi @ w
                residuals = y - y_pred
                loss_history.append(RobustRegression.huber_loss(residuals, delta))
                print(f"  IRLS hội tụ sau {iteration + 1} vòng lặp.")
                break

            w = w_new
        else:
            print(f"  IRLS chưa hội tụ sau {max_iter} vòng lặp.")

        return w, loss_history

    @staticmethod
    def inject_outliers(y, fraction=0.05, multiplier=10, seed=42):
        rng = np.random.default_rng(seed)
        y_corrupted = y.copy().astype(float)
        n = len(y)
        n_outliers = int(n * fraction)

        outlier_indices = rng.choice(n, size=n_outliers, replace=False)
        outlier_mask = np.zeros(n, dtype=bool)
        outlier_mask[outlier_indices] = True

        y_mean = np.mean(y)
        y_std = np.std(y)

        outlier_values = y_mean + multiplier * y_std * rng.choice([-1, 1], size=n_outliers)
        y_corrupted[outlier_indices] = outlier_values

        return y_corrupted, outlier_mask


# =============================================================================
# KERNEL REGRESSION MODULE
# Kernel Ridge Regression and Gaussian Process Regression.
# =============================================================================
class KernelRegression:
    """
    Kernel-based regression methods:
    - Kernel Ridge Regression (RBF and Polynomial kernels)
    - Gaussian Process Regression with gradient-based hyperparameter optimization
    """

    # ------------------------------------------------------------------
    # Kernel functions
    # ------------------------------------------------------------------
    @staticmethod
    def rbf_kernel_matrix(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        # Calculate the rbf (gaussian) kernel matrix between two sets of data points.
        dists = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-gamma * dists)

    @staticmethod
    def poly_kernel_matrix(X1: np.ndarray, X2: np.ndarray, degree: int = 3, coef0: float = 1.0) -> np.ndarray:
        # Calculate the polynomial kernel matrix.
        return (X1 @ X2.T + coef0) ** degree

    # ------------------------------------------------------------------
    # Kernel Ridge Regression
    # ------------------------------------------------------------------
    @staticmethod
    def fit_kernel_ridge(K_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
        # Train the kernel ridge regression model and return the dual variables (alpha).
        n = K_train.shape[0]
        # Solve the linear system (K + lambda * I) * alpha = y to find alpha.
        alpha = np.linalg.solve(K_train + lam * np.eye(n), y_train)
        return alpha

    @staticmethod
    def predict_kernel_ridge(K_test: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        # Predict the target values using the learned alpha weights and the test kernel matrix.
        return K_test @ alpha

    # ------------------------------------------------------------------
    # Gaussian Process Regression
    # ------------------------------------------------------------------
    @staticmethod
    def gp_lml_and_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple:
        # Calculate the log-marginal-likelihood and its gradients for gpr using an rbf kernel.
        # The theta array contains [log(sigma_f), log(l), log(sigma_n)].
        sigma_f, l, sigma_n = np.exp(theta)
        n = X.shape[0]

        # Compute the rbf kernel matrix for the training data.
        dists = cdist(X, X, metric='sqeuclidean')
        K = (sigma_f ** 2) * np.exp(-0.5 * dists / (l ** 2))

        # Add noise variance and a small jitter term to the diagonal for numerical stability.
        Ky = K + (sigma_n ** 2) * np.eye(n) + 1e-8 * np.eye(n)

        try:
            # Use cholesky decomposition to stably invert the covariance matrix.
            L = np.linalg.cholesky(Ky)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            invKy = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
        except np.linalg.LinAlgError:
            # Return negative infinity if the matrix is not positive definite.
            return -np.inf, np.zeros(3)

        # Compute the scalar log-marginal likelihood.
        lml = -0.5 * y.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)

        # Compute the matrix W used in gradient calculations.
        W = np.outer(alpha, alpha) - invKy

        # Calculate the gradient with respect to log(sigma_f).
        dK_dlsf = 2 * K
        grad_sf = 0.5 * np.trace(W @ dK_dlsf)

        # Calculate the gradient with respect to log(l).
        dK_dll = K * (dists / (l ** 2))
        grad_l = 0.5 * np.trace(W @ dK_dll)

        # Calculate the gradient with respect to log(sigma_n).
        dK_dlsn = 2 * (sigma_n ** 2) * np.eye(n)
        grad_sn = 0.5 * np.trace(W @ dK_dlsn)

        return lml, np.array([grad_sf, grad_l, grad_sn])

    @staticmethod
    def optimize_gp_hyperparameters(X: np.ndarray, y: np.ndarray, init_theta: np.ndarray = None, lr: float = 0.01, max_iters: int = 100) -> tuple:
        # Optimize the kernel hyperparameters using the gradient ascent algorithm.
        if init_theta is None:
            # Initialize log hyperparameters to zero if not provided.
            theta = np.array([0.0, 0.0, 0.0])
        else:
            theta = np.array(init_theta)

        lml_history = []

        # Iteratively update the hyperparameters to maximize the log-marginal likelihood.
        for i in range(max_iters):
            iter_start = time.time()
            lml, grad = KernelRegression.gp_lml_and_grad(theta, X, y)
            if lml == -np.inf:
                break

            # Apply the gradient ascent update rule.
            theta = theta + lr * grad
            lml_history.append(lml)
            iter_time = time.time() - iter_start
            print(f"Iteration {i+1}/{max_iters} - LML: {lml:.4f} - Thời gian: {iter_time:.4f}s")

        return theta, lml_history

    @staticmethod
    def predict_gp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, theta: np.ndarray) -> tuple:
        # Predict the posterior predictive mean and variance for new test points.
        sigma_f, l, sigma_n = np.exp(theta)
        n_train = X_train.shape[0]

        # Compute the kernel matrices required for prediction.
        dists_train = cdist(X_train, X_train, 'sqeuclidean')
        K = (sigma_f ** 2) * np.exp(-0.5 * dists_train / (l ** 2))
        Ky = K + (sigma_n ** 2) * np.eye(n_train) + 1e-8 * np.eye(n_train)

        dists_cross = cdist(X_train, X_test, 'sqeuclidean')
        K_cross = (sigma_f ** 2) * np.exp(-0.5 * dists_cross / (l ** 2))

        dists_test = cdist(X_test, X_test, 'sqeuclidean')
        K_test = (sigma_f ** 2) * np.exp(-0.5 * dists_test / (l ** 2))

        # Use cholesky decomposition for stable computation.
        L = np.linalg.cholesky(Ky)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # Calculate the posterior mean.
        mu = K_cross.T @ alpha

        # Calculate the posterior variance and ensure it is non-negative.
        v = np.linalg.solve(L, K_cross)
        var = np.diag(K_test) - np.sum(v ** 2, axis=0)

        return mu, np.maximum(var, 0)


# =============================================================================
# BAYESIAN LINEAR REGRESSION MODULE
# Posterior inference, predictive distribution, Evidence Maximization, CV tuning.
# =============================================================================
class BayesianLinearRegression:
    """
    Bayesian Linear Regression with:
    1. RBF Gaussian basis function construction
    2. Posterior inference (closed-form)
    3. Predictive distribution with epistemic + aleatoric uncertainty
    4. Evidence Maximization (Empirical Bayes) for alpha, beta
    5. Cross-validation-based hyperparameter tuning
    """

    @staticmethod
    def gaussian_rbf(X: np.ndarray, centers: np.ndarray, s: float) -> np.ndarray:
        X = np.atleast_2d(X).reshape(-1, 1)          # (N, 1)
        centers = np.atleast_1d(centers).ravel()      # (M,)

        # Broadcast difference: (N, 1) - (1, M) = (N, M)
        diff = X - centers[np.newaxis, :]             # (N, M)
        return np.exp(-(diff ** 2) / (2.0 * s ** 2))

    @staticmethod
    def compute_posterior(
        Phi_train: np.ndarray,
        t_train: np.ndarray,
        alpha: float,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_train = np.atleast_1d(t_train).ravel()   # ensure shape (N,)
        M = Phi_train.shape[1]

        # Compute the inverse of S_N:  S_N^{-1} = alpha*I + beta * Phi^T @ Phi
        S_N_inv = alpha * np.eye(M) + beta * (Phi_train.T @ Phi_train)

        # Solve S_N^{-1} @ S_N = I  =>  S_N = solve(S_N_inv, I)
        S_N = np.linalg.solve(S_N_inv, np.eye(M))

        # Posterior mean:  m_N = beta * S_N @ Phi^T @ t
        m_N = beta * S_N @ Phi_train.T @ t_train

        return m_N, S_N

    @staticmethod
    def compute_predictive_distribution(
        Phi_test: np.ndarray,
        m_N: np.ndarray,
        S_N: np.ndarray,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Predictive mean: f* = Phi_test @ m_N
        f_star = Phi_test @ m_N                              # (N_test,)

        # Epistemic variance (model uncertainty): diag(Phi_test @ S_N @ Phi_test^T)
        # Efficient computation: row-wise dot of (Phi_test @ S_N) and Phi_test
        epistemic_var = np.sum((Phi_test @ S_N) * Phi_test, axis=1)   # (N_test,)

        # Aleatoric variance (irreducible noise): 1 / beta
        aleatoric_var = 1.0 / beta

        # Total predictive variance and standard deviation
        sigma_N_sq = aleatoric_var + epistemic_var           # (N_test,)
        sigma_N = np.sqrt(np.maximum(sigma_N_sq, 0.0))      # guard against tiny negatives

        return f_star, sigma_N

    @staticmethod
    def evidence_maximization(
        Phi_train,
        t_train,
        alpha_init=1.0,
        beta_init=1.0,
        max_iter=100,
        tol=1e-6
    ):
        import numpy as np

        N, M = Phi_train.shape
        alpha = alpha_init
        beta = beta_init

        # Ensure t_train is 1D array
        t_train = np.atleast_1d(t_train).ravel()

        # Precompute Phi^T @ Phi for efficiency
        PhiT_Phi = Phi_train.T @ Phi_train
        PhiT_t = Phi_train.T @ t_train

        # Store history for analysis
        history = []

        for iteration in range(max_iter):
            alpha_old = alpha
            beta_old = beta

            # E-step: Compute posterior with current alpha, beta
            # S_N^{-1} = alpha*I + beta*Phi^T*Phi
            S_N_inv = alpha * np.eye(M) + beta * PhiT_Phi

            try:
                S_N = np.linalg.inv(S_N_inv)
            except np.linalg.LinAlgError:
                S_N = np.linalg.pinv(S_N_inv)

            # m_N = beta * S_N * Phi^T * t
            m_N = beta * S_N @ PhiT_t

            # Compute residuals and norms BEFORE updating
            residuals = t_train - Phi_train @ m_N
            residual_sum_sq = np.sum(residuals ** 2)
            m_N_norm_sq = np.dot(m_N, m_N)

            # Compute log marginal likelihood (evidence) BEFORE updating alpha and beta
            # This way we track the evidence at the current parameters
            try:
                # Compute log|S_N|
                try:
                    L = np.linalg.cholesky(S_N)
                    log_det_S_N = 2.0 * np.sum(np.log(np.diag(L)))
                except np.linalg.LinAlgError:
                    eigvals_S_N = np.linalg.eigvalsh(S_N)
                    log_det_S_N = np.sum(np.log(np.maximum(eigvals_S_N, 1e-10)))

                # log|S_N^{-1}| = -log|S_N|
                log_det_S_N_inv = -log_det_S_N

                # log|C| = -N*log(beta) + log|S_N^{-1}| - M*log(alpha)
                log_det_C = -N * np.log(beta) + log_det_S_N_inv - M * np.log(alpha)

                # t^T*C^{-1}*t = beta*||t||^2 - beta^2*t^T*Phi*S_N*Phi^T*t
                t_norm_sq = np.sum(t_train ** 2)
                quad_form = beta * t_norm_sq - (beta**2) * (PhiT_t.T @ S_N @ PhiT_t)

                # Final evidence
                log_evidence = -(N / 2.0) * np.log(2.0 * np.pi) - (1.0 / 2.0) * log_det_C - (1.0 / 2.0) * quad_form

            except (np.linalg.LinAlgError, ValueError):
                log_evidence = np.nan

            # M-step: Compute eigenvalues for gamma calculation
            # gamma = sum_i (lambda_i / (alpha + lambda_i))
            # where lambda_i are eigenvalues of beta * Phi^T * Phi
            eigvals = np.linalg.eigvalsh(beta * PhiT_Phi)
            gamma = np.sum(eigvals / (alpha + eigvals))

            # Ensure gamma is in valid range [0, M]
            gamma = np.clip(gamma, 1e-10, M)

            # Update alpha using re-estimation equation (Bishop 3.92)
            # alpha_new = gamma / (m_N^T * m_N)
            if m_N_norm_sq > 1e-10:
                alpha_new = gamma / m_N_norm_sq
            else:
                alpha_new = alpha_old

            # Update beta using re-estimation equation (Bishop 3.95)
            # beta_new = (N - gamma) / ||t - Phi*m_N||^2
            if residual_sum_sq > 1e-10:
                beta_new = (N - gamma) / residual_sum_sq
            else:
                beta_new = beta_old

            # NO DAMPING - use direct update for faster convergence
            alpha = alpha_new
            beta = beta_new

            # Prevent numerical issues
            alpha = np.clip(alpha, 1e-10, 1e10)
            beta = np.clip(beta, 1e-10, 1e10)

            # Compute changes for convergence check
            delta_alpha = abs(alpha - alpha_old) / (abs(alpha_old) + 1e-10)
            delta_beta = abs(beta - beta_old) / (abs(beta_old) + 1e-10)

            # Store history
            history.append((iteration + 1, alpha, beta, gamma, log_evidence, delta_alpha, delta_beta))

            # Print progress
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"Iter {iteration+1:3d}: alpha={alpha:8.4f}, beta={beta:8.6f}, gamma={gamma:5.2f}, Evidence={log_evidence:12.2f}")

            # Check convergence
            if delta_alpha < tol and delta_beta < tol and iteration > 10:
                print(f"Converged at iteration {iteration+1}")
                break

        return alpha, beta, m_N, S_N, history

    @staticmethod
    def cv_bayesian_hyperparams(
        Phi_train: np.ndarray,
        t_train: np.ndarray,
        alpha_grid: np.ndarray,
        beta_grid: np.ndarray,
        k_folds: int = 5
    ) -> tuple[float, float, list]:
        # Create time-series CV folds
        folds = LinearRegression.time_series_cv_indices(len(t_train), k=k_folds)

        best_alpha = None
        best_beta = None
        best_cv_score = float('inf')
        cv_results = []

        # Grid search
        for alpha in alpha_grid:
            for beta in beta_grid:
                fold_scores = []

                for train_idx, val_idx in folds:
                    Phi_fold_train = Phi_train[train_idx]
                    t_fold_train = t_train[train_idx]
                    Phi_fold_val = Phi_train[val_idx]
                    t_fold_val = t_train[val_idx]

                    # Compute posterior with current alpha, beta
                    m_N_fold, S_N_fold = BayesianLinearRegression.compute_posterior(
                        Phi_fold_train, t_fold_train, alpha, beta
                    )

                    # Predict on validation fold
                    pred_fold = Phi_fold_val @ m_N_fold
                    mse_fold = np.mean((t_fold_val - pred_fold) ** 2)
                    fold_scores.append(mse_fold)

                # Average CV score
                avg_cv_score = np.mean(fold_scores)
                cv_results.append((alpha, beta, avg_cv_score))

                if avg_cv_score < best_cv_score:
                    best_cv_score = avg_cv_score
                    best_alpha = alpha
                    best_beta = beta

        return best_alpha, best_beta, cv_results


# =============================================================================
# MODEL EVALUATOR MODULE
# Learning curves, residual analysis, bias-variance, complexity, statistical tests.
# =============================================================================
class BaseEvaluator:
    """
    Base class providing shared metrics and CV utilities for model evaluation.
    """

    @staticmethod
    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return LinearRegression.metrics(y_true, y_pred)

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return LinearRegression.mse(y_true, y_pred)

    @staticmethod
    def build_model_comparison_table(model_results: dict):
        """
        Print a formatted comparison table.

        Parameters:
            model_results: dict of {model_name: {'MSE':..., 'RMSE':..., 'MAE':..., 'R2':...}}
        """
        header = f"{'Model':<35} {'MSE':>12} {'RMSE':>12} {'MAE':>12} {'R2':>12}"
        sep = "=" * len(header)
        print(sep)
        print(header)
        print(sep)
        for name, m in model_results.items():
            print(f"{name:<35} {m['MSE']:>12.4f} {m['RMSE']:>12.4f} "
                  f"{m['MAE']:>12.4f} {m['R2']:>12.4f}")
        print(sep)

    @staticmethod
    def compute_model_summary_from_cv(cv_results: dict) -> dict:
        summary = {}
        for name, folds in cv_results.items():
            mse_vals  = np.array([f['MSE']  for f in folds])
            rmse_vals = np.array([f['RMSE'] for f in folds])
            mae_vals  = np.array([f['MAE']  for f in folds])
            r2_vals   = np.array([f['R2']   for f in folds])
            summary[name] = {
                'mean_mse':  float(np.mean(mse_vals)),
                'std_mse':   float(np.std(mse_vals)),
                'mean_rmse': float(np.mean(rmse_vals)),
                'std_rmse':  float(np.std(rmse_vals)),
                'mean_mae':  float(np.mean(mae_vals)),
                'std_mae':   float(np.std(mae_vals)),
                'mean_r2':   float(np.mean(r2_vals)),
                'std_r2':    float(np.std(r2_vals)),
            }
        return summary

    @staticmethod
    def time_method_comparison(methods: dict, Phi: np.ndarray, y: np.ndarray,
                               n_repeats: int = 5) -> dict:
        results = {}
        for name, fn in methods.items():
            times = []
            for _ in range(n_repeats):
                t0 = time.time()
                fn(Phi, y)
                times.append(time.time() - t0)
            results[name] = {
                'mean_s': float(np.mean(times)),
                'std_s':  float(np.std(times)),
            }
        return results


class ModelEvaluator(BaseEvaluator):
    """
    Comprehensive model evaluation utilities including:
    - Learning curves (single and multi-model)
    - K-Fold Cross-Validation (Time-Series aware)
    - Statistical hypothesis tests (t-test / Wilcoxon)
    - Residual analysis and diagnostics
    - Bias-Variance decomposition via Bootstrap
    - Computational complexity / timing analysis
    - Model comparison and ranking
    """

    # ------------------------------------------------------------------
    # 12.1 Learning Curve Analysis
    # ------------------------------------------------------------------
    @staticmethod
    def compute_learning_curve(Phi_train, y_train, Phi_val, y_val,
                               fit_fn, n_points=10):
        """
        Compute learning curves by training on increasing subsets of training data.

        Parameters:
            Phi_train, y_train: training data
            Phi_val, y_val: validation data
            fit_fn: callable(Phi, y) -> w  (a function that fits and returns weights)
            n_points: number of data points on the curve

        Returns:
            train_sizes, train_losses, val_losses  (all numpy arrays)
        """
        n, d = Phi_train.shape
        min_size = max(d + 10, n // n_points)  # must exceed #features to avoid singular matrix
        sizes = np.linspace(min_size, n, n_points, dtype=int)
        train_losses = []
        val_losses = []

        for size in sizes:
            Phi_sub = Phi_train[:size]
            y_sub = y_train[:size]
            w = fit_fn(Phi_sub, y_sub)
            train_pred = Phi_sub @ w
            val_pred = Phi_val @ w
            train_losses.append(float(np.mean((y_sub - train_pred) ** 2)))
            val_losses.append(float(np.mean((y_val - val_pred) ** 2)))

        return np.array(sizes), np.array(train_losses), np.array(val_losses)

    @staticmethod
    def plot_learning_curves(train_sizes, train_losses, val_losses,
                             title='Learning Curves'):
        """Plot train loss and validation loss vs. number of training samples."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_sizes, train_losses, 'o-', label='Train MSE', linewidth=2)
        plt.plot(train_sizes, val_losses, 's-', label='Validation MSE', linewidth=2)
        plt.xlabel('Number of Training Samples', fontsize=12)
        plt.ylabel('MSE', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred, title='Residual Plot'):
        """Scatter plot of predicted vs residuals to check randomness of errors."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 5))
        plt.scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals (y_true - y_pred)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_predicted_vs_actual(y_true, y_pred, title='Predicted vs Actual'):
        """Scatter plot of actual vs predicted with y=x reference line."""
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, 'r--', linewidth=1.5, label='y = x (ideal)')
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_learning_curves_multiple_models(
        Phi_train: np.ndarray,
        y_train: np.ndarray,
        Phi_val: np.ndarray,
        y_val: np.ndarray,
        model_configs: dict,
        n_points: int = 10
    ) -> dict:
        n, d = Phi_train.shape
        min_size = max(d + 10, n // n_points)
        sizes = np.linspace(min_size, n, n_points, dtype=int)

        results = {}

        for model_name, fit_fn in model_configs.items():
            train_losses = []
            val_losses = []

            for size in sizes:
                Phi_sub = Phi_train[:size]
                y_sub = y_train[:size]

                try:
                    w = fit_fn(Phi_sub, y_sub)
                    train_pred = Phi_sub @ w
                    val_pred = Phi_val @ w

                    train_losses.append(float(np.mean((y_sub - train_pred) ** 2)))
                    val_losses.append(float(np.mean((y_val - val_pred) ** 2)))
                except Exception as e:
                    print(f"Warning: {model_name} failed at size {size}: {e}")
                    train_losses.append(np.nan)
                    val_losses.append(np.nan)

            results[model_name] = {
                'sizes': sizes,
                'train_mse': np.array(train_losses),
                'val_mse': np.array(val_losses)
            }

        return results

    @staticmethod
    def analyze_learning_curve_convergence(sizes: np.ndarray, val_mse: np.ndarray,
                                           threshold: float = 0.01) -> dict:
        # Check if curve is converging (slope flattening)
        if len(val_mse) < 3:
            return {'converged': False, 'message': 'Not enough data points'}

        # Compute relative changes in last 3 points
        recent_changes = np.abs(np.diff(val_mse[-3:])) / (val_mse[-3:-1] + 1e-10)

        converged = np.all(recent_changes < threshold)

        # Estimate if more data would help (check slope)
        if len(val_mse) >= 2:
            final_slope = (val_mse[-1] - val_mse[-2]) / (sizes[-1] - sizes[-2])
            improvement_potential = abs(final_slope) > 1e-6
        else:
            improvement_potential = True

        return {
            'converged': converged,
            'final_val_mse': float(val_mse[-1]),
            'recent_changes': recent_changes.tolist(),
            'improvement_potential': improvement_potential,
            'message': 'Converged' if converged else 'Still improving' if improvement_potential else 'Plateaued'
        }

    # ------------------------------------------------------------------
    # CV and statistical tests
    # ------------------------------------------------------------------
    @staticmethod
    def kfold_cross_validation_ts(Phi, y, fit_fn, k=10):
        """
        Time-Series K-Fold Cross-Validation (Expanding Window).

        Parameters:
            Phi: design matrix (n_samples, n_features)
            y: target vector
            fit_fn: callable(Phi, y) -> w
            k: number of folds

        Returns:
            fold_metrics: list of dicts, each with MSE, RMSE, MAE, R2
        """
        folds = LinearRegression.time_series_cv_indices(len(y), k=k)
        fold_metrics_list = []

        for train_idx, val_idx in folds:
            Phi_tr, y_tr = Phi[train_idx], y[train_idx]
            Phi_va, y_va = Phi[val_idx], y[val_idx]

            w = fit_fn(Phi_tr, y_tr)
            y_pred = Phi_va @ w
            m = LinearRegression.metrics(y_va, y_pred)
            fold_metrics_list.append(m)

        return fold_metrics_list

    @staticmethod
    def statistical_test_models(scores_a, scores_b, metric_name='MSE',
                                test_type='wilcoxon'):
        """
        Perform paired statistical test on per-fold scores of two models.

        Parameters:
            scores_a: list/array of per-fold metric values for model A
            scores_b: list/array of per-fold metric values for model B
            metric_name: name of the metric (for display)
            test_type: 'ttest' for paired t-test, 'wilcoxon' for Wilcoxon signed-rank

        Returns:
            stat, p_value, is_significant (at alpha=0.05)
        """
        a = np.array(scores_a)
        b = np.array(scores_b)

        if test_type == 'ttest':
            stat, p_value = ttest_rel(a, b)
            test_name = 'Paired t-test'
        else:
            stat, p_value = scipy_wilcoxon(a, b)
            test_name = 'Wilcoxon signed-rank test'

        is_significant = p_value < 0.05
        return {
            'test': test_name,
            'metric': metric_name,
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': is_significant,
        }

    # ------------------------------------------------------------------
    # 12.2 Residual Analysis
    # ------------------------------------------------------------------
    @staticmethod
    def compute_residual_statistics(residuals: np.ndarray) -> dict:
        from scipy import stats as scipy_stats

        residuals = residuals.ravel()

        # Basic statistics
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        median_res = float(np.median(residuals))

        # Normality test (Shapiro-Wilk)
        # Only use subset if too large (Shapiro-Wilk has sample size limit)
        if len(residuals) > 5000:
            sample_idx = np.random.choice(len(residuals), 5000, replace=False)
            shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals[sample_idx])
        else:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)

        # Skewness and Kurtosis
        skewness = float(scipy_stats.skew(residuals))
        kurtosis = float(scipy_stats.kurtosis(residuals))

        # Quantiles for QQ-plot
        quantiles = np.percentile(residuals, [25, 50, 75])

        return {
            'mean': mean_res,
            'std': std_res,
            'median': median_res,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_statistic': float(shapiro_stat),
            'shapiro_pvalue': float(shapiro_p),
            'is_normal': shapiro_p > 0.05,
            'q25': float(quantiles[0]),
            'q50': float(quantiles[1]),
            'q75': float(quantiles[2])
        }

    @staticmethod
    def durbin_watson_test(residuals: np.ndarray) -> float:
        residuals = residuals.ravel()
        diff_residuals = np.diff(residuals)

        numerator = np.sum(diff_residuals ** 2)
        denominator = np.sum(residuals ** 2)

        dw_stat = numerator / (denominator + 1e-10)

        return float(dw_stat)

    @staticmethod
    def analyze_residual_patterns(y_pred: np.ndarray, residuals: np.ndarray,
                                  n_bins: int = 10) -> dict:
        y_pred = y_pred.ravel()
        residuals = residuals.ravel()

        # Divide predictions into bins
        bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute statistics per bin
        bin_stats = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_residuals = residuals[mask]
                bin_stats.append({
                    'bin_center': float((bins[i] + bins[i+1]) / 2),
                    'count': int(np.sum(mask)),
                    'mean_residual': float(np.mean(bin_residuals)),
                    'std_residual': float(np.std(bin_residuals)),
                    'abs_mean_residual': float(np.mean(np.abs(bin_residuals)))
                })

        # Check for heteroscedasticity pattern (funnel shape)
        stds = [b['std_residual'] for b in bin_stats]
        std_trend = np.polyfit(range(len(stds)), stds, 1)[0]  # slope of std vs bin

        has_funnel = abs(std_trend) > 0.1 * np.mean(stds)

        return {
            'bin_statistics': bin_stats,
            'std_trend_slope': float(std_trend),
            'has_heteroscedasticity_pattern': has_funnel
        }

    # ------------------------------------------------------------------
    # 12.3 Predicted vs Actual - Error Pattern Analysis
    # ------------------------------------------------------------------
    @staticmethod
    def analyze_prediction_errors_by_range(y_true: np.ndarray, y_pred: np.ndarray,
                                           n_quantiles: int = 4) -> dict:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        # Divide into quantiles based on true values
        quantile_edges = np.percentile(y_true, np.linspace(0, 100, n_quantiles + 1))

        results = []
        for i in range(n_quantiles):
            lower = quantile_edges[i]
            upper = quantile_edges[i + 1]

            # Include upper bound in last quantile
            if i == n_quantiles - 1:
                mask = (y_true >= lower) & (y_true <= upper)
            else:
                mask = (y_true >= lower) & (y_true < upper)

            if np.sum(mask) > 0:
                y_true_q = y_true[mask]
                y_pred_q = y_pred[mask]

                errors = y_true_q - y_pred_q
                abs_errors = np.abs(errors)

                results.append({
                    'quantile': i + 1,
                    'range': f'[{lower:.1f}, {upper:.1f}]',
                    'n_samples': int(np.sum(mask)),
                    'mean_true': float(np.mean(y_true_q)),
                    'mse': float(np.mean(errors ** 2)),
                    'rmse': float(np.sqrt(np.mean(errors ** 2))),
                    'mae': float(np.mean(abs_errors)),
                    'median_ae': float(np.median(abs_errors)),
                    'max_error': float(np.max(abs_errors))
                })

        return {'quantile_analysis': results}

    @staticmethod
    def identify_worst_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                                   top_k: int = 10) -> dict:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        abs_errors = np.abs(y_true - y_pred)
        worst_indices = np.argsort(abs_errors)[-top_k:][::-1]

        worst_predictions = []
        for idx in worst_indices:
            worst_predictions.append({
                'index': int(idx),
                'true_value': float(y_true[idx]),
                'predicted_value': float(y_pred[idx]),
                'error': float(y_true[idx] - y_pred[idx]),
                'abs_error': float(abs_errors[idx]),
                'relative_error': float(abs_errors[idx] / (abs(y_true[idx]) + 1e-10))
            })

        return {
            'worst_predictions': worst_predictions,
            'mean_worst_error': float(np.mean([p['abs_error'] for p in worst_predictions]))
        }

    # ------------------------------------------------------------------
    # 12.4 Computational Complexity Analysis
    # ------------------------------------------------------------------
    @staticmethod
    def measure_training_time_vs_samples(Phi_train: np.ndarray, y_train: np.ndarray,
                                         fit_fn, sample_sizes: list = None,
                                         n_repeats: int = 3) -> dict:
        n_total = Phi_train.shape[0]

        if sample_sizes is None:
            # Logarithmic scale from 100 to n_total
            sample_sizes = np.unique(np.logspace(
                np.log10(100),
                np.log10(n_total),
                10,
                dtype=int
            ))
            sample_sizes = [s for s in sample_sizes if s <= n_total]

        results = []

        for size in sample_sizes:
            times = []
            for _ in range(n_repeats):
                Phi_sub = Phi_train[:size]
                y_sub = y_train[:size]

                start = time.time()
                try:
                    _ = fit_fn(Phi_sub, y_sub)
                    elapsed = time.time() - start
                    times.append(elapsed)
                except Exception as e:
                    print(f"Warning: Failed at size {size}: {e}")
                    times.append(np.nan)

            results.append({
                'n_samples': int(size),
                'mean_time': float(np.nanmean(times)),
                'std_time': float(np.nanstd(times)),
                'min_time': float(np.nanmin(times)),
                'max_time': float(np.nanmax(times))
            })

        return {'timing_results': results}

    @staticmethod
    def measure_training_time_vs_features(X_train: np.ndarray, y_train: np.ndarray,
                                          fit_fn, feature_counts: list = None,
                                          n_repeats: int = 3) -> dict:
        n_total_features = X_train.shape[1]

        if feature_counts is None:
            feature_counts = np.unique(np.linspace(5, n_total_features, 8, dtype=int))
            feature_counts = [f for f in feature_counts if f <= n_total_features]

        results = []

        for n_features in feature_counts:
            times = []
            for _ in range(n_repeats):
                # Select first n_features
                X_sub = X_train[:, :n_features]
                Phi_sub = LinearRegression.add_bias(X_sub)

                start = time.time()
                try:
                    _ = fit_fn(Phi_sub, y_train)
                    elapsed = time.time() - start
                    times.append(elapsed)
                except Exception as e:
                    print(f"Warning: Failed at {n_features} features: {e}")
                    times.append(np.nan)

            results.append({
                'n_features': int(n_features),
                'mean_time': float(np.nanmean(times)),
                'std_time': float(np.nanstd(times)),
                'min_time': float(np.nanmin(times)),
                'max_time': float(np.nanmax(times))
            })

        return {'timing_results': results}

    @staticmethod
    def estimate_memory_usage(Phi: np.ndarray, method: str = 'normal_equations') -> dict:
        N, M = Phi.shape
        bytes_per_float = 8  # float64

        # Common: store Phi and y
        base_memory = (N * M + N) * bytes_per_float

        if method == 'normal_equations':
            # Need: Phi^T @ Phi (M x M), Phi^T @ y (M), w (M)
            additional = (M * M + M + M) * bytes_per_float

        elif method == 'gradient_descent':
            # Need: w (M), gradient (M), predictions (N)
            additional = (M + M + N) * bytes_per_float

        elif method == 'kernel_ridge':
            # Need: K (N x N), alpha (N)
            additional = (N * N + N) * bytes_per_float
        else:
            additional = 0

        total_bytes = base_memory + additional
        total_mb = total_bytes / (1024 ** 2)

        return {
            'method': method,
            'n_samples': N,
            'n_features': M,
            'base_memory_mb': float(base_memory / (1024 ** 2)),
            'additional_memory_mb': float(additional / (1024 ** 2)),
            'total_memory_mb': float(total_mb)
        }

    @staticmethod
    def extrapolate_scalability(timing_results: list, target_sizes: list,
                                complexity_order: float = 3.0) -> dict:
        # Fit power law: time = a * N^complexity_order
        sizes = np.array([r['n_samples'] for r in timing_results])
        times = np.array([r['mean_time'] for r in timing_results])

        # Use log-log linear regression
        log_sizes = np.log(sizes)
        log_times = np.log(times + 1e-10)

        # Fit: log(time) = log(a) + complexity_order * log(N)
        # But we fix complexity_order, so just find a
        a = np.exp(np.mean(log_times - complexity_order * log_sizes))

        predictions = []
        for target_size in target_sizes:
            predicted_time = a * (target_size ** complexity_order)
            predictions.append({
                'n_samples': int(target_size),
                'predicted_time_seconds': float(predicted_time),
                'predicted_time_minutes': float(predicted_time / 60),
                'predicted_time_hours': float(predicted_time / 3600)
            })

        return {
            'complexity_order': complexity_order,
            'coefficient_a': float(a),
            'predictions': predictions
        }

    # ------------------------------------------------------------------
    # 12.5 Bias-Variance Decomposition (Bootstrap)
    # ------------------------------------------------------------------
    @staticmethod
    def bias_variance_decomposition(Phi_train, y_train, Phi_test, y_test,
                                    lambdas, n_bootstrap=200, seed=42):
        rng = np.random.default_rng(seed)
        N_train = Phi_train.shape[0]
        N_test = Phi_test.shape[0]
        n_lambdas = len(lambdas)

        bias_squared_list = []
        variance_list = []
        mse_list = []

        total_start = time.time()

        for lam_idx, lam in enumerate(lambdas):
            all_predictions = np.zeros((n_bootstrap, N_test))

            for b in range(n_bootstrap):
                bootstrap_indices = rng.choice(N_train, size=N_train, replace=True)
                Phi_b = Phi_train[bootstrap_indices]
                y_b = y_train[bootstrap_indices]

                w_b = LinearRegression.fit_ridge(Phi_b, y_b, lam=lam, bias_is_first=True)
                all_predictions[b, :] = Phi_test @ w_b

            mean_predictions = np.mean(all_predictions, axis=0)

            bias_sq = np.mean((mean_predictions - y_test) ** 2)
            variance = np.mean(np.var(all_predictions, axis=0))
            mse_total = np.mean((all_predictions - y_test[np.newaxis, :]) ** 2)

            bias_squared_list.append(bias_sq)
            variance_list.append(variance)
            mse_list.append(mse_total)

            print(f"  lambda = {lam:>10.4f} (log10={np.log10(lam):>6.2f}): "
                  f"Bias^2 = {bias_sq:>10.4f}, Var = {variance:>10.4f}, MSE = {mse_total:>10.4f}",
                  end='\r')

        elapsed = time.time() - total_start
        print(f"\n\nBootstrapping done! Time: {elapsed:.1f}s ({n_bootstrap} x {n_lambdas} lambdas)")

        return bias_squared_list, variance_list, mse_list

    @staticmethod
    def bias_variance_decomposition_single_lambda(
        Phi_train: np.ndarray,
        y_train: np.ndarray,
        Phi_test: np.ndarray,
        y_test: np.ndarray,
        fit_fn,
        n_bootstrap: int = 100,
        seed: int = 42
    ) -> dict:
        rng = np.random.default_rng(seed)
        N_train = Phi_train.shape[0]
        N_test = Phi_test.shape[0]

        all_predictions = np.zeros((n_bootstrap, N_test))

        for b in range(n_bootstrap):
            bootstrap_indices = rng.choice(N_train, size=N_train, replace=True)
            Phi_b = Phi_train[bootstrap_indices]
            y_b = y_train[bootstrap_indices]

            try:
                w_b = fit_fn(Phi_b, y_b)
                all_predictions[b, :] = Phi_test @ w_b
            except Exception as e:
                print(f"Warning: Bootstrap {b} failed: {e}")
                all_predictions[b, :] = np.nan

        # Remove failed bootstraps
        valid_mask = ~np.isnan(all_predictions).any(axis=1)
        all_predictions = all_predictions[valid_mask]

        if len(all_predictions) == 0:
            return {
                'bias_squared': np.nan,
                'variance': np.nan,
                'mse': np.nan,
                'irreducible_error': np.nan
            }

        # Compute mean prediction across bootstraps
        mean_predictions = np.mean(all_predictions, axis=0)

        # Bias^2: squared difference between mean prediction and true values
        bias_sq = np.mean((mean_predictions - y_test) ** 2)

        # Variance: variance of predictions across bootstraps
        variance = np.mean(np.var(all_predictions, axis=0))

        # Total MSE
        mse_total = np.mean((all_predictions - y_test[np.newaxis, :]) ** 2)

        # Irreducible error (approximation)
        irreducible = mse_total - bias_sq - variance

        return {
            'bias_squared': float(bias_sq),
            'variance': float(variance),
            'mse': float(mse_total),
            'irreducible_error': float(max(0, irreducible)),  # ensure non-negative
            'n_valid_bootstraps': int(len(all_predictions))
        }

    @staticmethod
    def bias_variance_tradeoff_analysis(
        Phi_train: np.ndarray,
        y_train: np.ndarray,
        Phi_test: np.ndarray,
        y_test: np.ndarray,
        model_configs: dict,
        n_bootstrap: int = 100,
        seed: int = 42
    ) -> dict:
        results = {}

        for model_name, fit_fn in model_configs.items():
            print(f"Computing bias-variance for {model_name}...")
            decomp = ModelEvaluator.bias_variance_decomposition_single_lambda(
                Phi_train, y_train, Phi_test, y_test,
                fit_fn, n_bootstrap, seed
            )
            results[model_name] = decomp

        return results

    # ------------------------------------------------------------------
    # 12.6 Model Comparison & Ranking
    # ------------------------------------------------------------------
    @staticmethod
    def create_comprehensive_model_comparison(
        model_metrics: dict,
        timing_info: dict = None,
        bias_variance_info: dict = None,
        complexity_info: dict = None
    ) -> dict:
        comparison = {}

        for model_name in model_metrics.keys():
            entry = {
                'model': model_name,
                **model_metrics[model_name]
            }

            if timing_info and model_name in timing_info:
                entry['training_time_s'] = timing_info[model_name]

            if bias_variance_info and model_name in bias_variance_info:
                entry['bias_squared'] = bias_variance_info[model_name]['bias_squared']
                entry['variance'] = bias_variance_info[model_name]['variance']

            if complexity_info and model_name in complexity_info:
                entry['complexity'] = complexity_info[model_name]

            comparison[model_name] = entry

        return comparison

    @staticmethod
    def rank_models_multi_criteria(comparison: dict,
                                   criteria_weights: dict = None) -> list:
        if criteria_weights is None:
            criteria_weights = {'RMSE': 1.0}

        # Normalize criteria to [0, 1] range
        all_values = {criterion: [] for criterion in criteria_weights.keys()}

        for model_name, entry in comparison.items():
            for criterion in criteria_weights.keys():
                if criterion in entry and not np.isnan(entry[criterion]):
                    all_values[criterion].append(entry[criterion])

        # Compute min/max for normalization
        ranges = {}
        for criterion, values in all_values.items():
            if len(values) > 0:
                ranges[criterion] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }

        # Compute weighted scores
        scores = []
        for model_name, entry in comparison.items():
            score = 0.0
            total_weight = 0.0

            for criterion, weight in criteria_weights.items():
                if criterion in entry and criterion in ranges:
                    value = entry[criterion]
                    if not np.isnan(value):
                        # Normalize to [0, 1]
                        r = ranges[criterion]
                        if r['range'] > 0:
                            normalized = (value - r['min']) / r['range']
                        else:
                            normalized = 0.0

                        score += weight * normalized
                        total_weight += weight

            if total_weight > 0:
                score /= total_weight

            scores.append((model_name, float(score)))

        # Sort by score (lower is better)
        ranked = sorted(scores, key=lambda x: x[1])

        return ranked

    @staticmethod
    def generate_model_selection_recommendation(ranked_models: list,
                                                comparison: dict,
                                                dataset_size: int) -> dict:
        best_model = ranked_models[0][0]
        best_entry = comparison[best_model]

        # Determine dataset size category
        if dataset_size < 5000:
            size_category = 'small'
        elif dataset_size < 50000:
            size_category = 'medium'
        else:
            size_category = 'large'

        # Generate recommendation text
        recommendation = {
            'best_overall_model': best_model,
            'best_model_metrics': best_entry,
            'dataset_size_category': size_category,
            'top_3_models': [name for name, _ in ranked_models[:3]],
            'reasoning': []
        }

        # Add reasoning
        if 'RMSE' in best_entry:
            recommendation['reasoning'].append(
                f"Best RMSE: {best_entry['RMSE']:.4f}"
            )

        if 'training_time_s' in best_entry:
            recommendation['reasoning'].append(
                f"Training time: {best_entry['training_time_s']:.2f}s"
            )

        if 'R2' in best_entry:
            if best_entry['R2'] < 0.3:
                recommendation['reasoning'].append(
                    "Warning: Low R2 suggests model may be too simple for this data"
                )

        return recommendation

# =============================================================================
# SENSITIVITY ANALYZER MODULE
# Repeated-run sensitivity analysis across different train/test split ratios.
# =============================================================================
class SensitivityAnalyzer:
    """
    Sensitivity Analysis: evaluates how model performance varies as the
    train/test split ratio changes.

    Workflow
    --------
    1. run_experiment()  — loop over split ratios × repeated random seeds
    2. compute_summary() — median / std per (model, split ratio)
    3. plot_boxplots()   — grouped boxplots for R² and RMSE
    4. print_findings()  — automated English-language interpretation
    """

    # Default model names used throughout
    DEFAULT_MODELS = ['OLS', 'Ridge', 'Lasso', 'Elastic Net', 'WLS']

    # ------------------------------------------------------------------ #
    #  1. Experiment runner                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def run_experiment(
        X: np.ndarray,
        y: np.ndarray,
        test_sizes: list = None,
        n_repeats: int = 20,
        lam_ridge: float = 1.0,
        lam_lasso: float = 0.1,
        lam_en_l1: float = 0.1,
        lam_en_l2: float = 1.0,
        lasso_iters: int = 500,
        seed_base: int = 0,
    ) -> 'pd.DataFrame':
        """
        Run the full sensitivity experiment.

        Parameters
        ----------
        X          : feature matrix  (N, D), unscaled
        y          : target vector   (N,)
        test_sizes : list of test fractions, e.g. [0.4, 0.3, 0.2]
        n_repeats  : number of random seeds per split ratio
        lam_ridge  : Ridge L2 penalty
        lam_lasso  : Lasso L1 penalty
        lam_en_l1  : Elastic Net L1 penalty
        lam_en_l2  : Elastic Net L2 penalty
        lasso_iters: max coordinate-descent iterations for Lasso / EN
        seed_base  : first random seed; seeds = seed_base, seed_base+1, ...

        Returns
        -------
        pd.DataFrame with columns:
            split_label, test_size, train_pct, run, model,
            R2, RMSE, MAE, MSE
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        if test_sizes is None:
            test_sizes = [0.4, 0.3, 0.2]

        records = []

        print("  13.1  Sensitivity Analysis — Train/Test Split Ratio")
        print(f"  Split ratios  : {[f'{int((1-ts)*100)}/{int(ts*100)}' for ts in test_sizes]}")
        print(f"  Repeats/ratio : {n_repeats}")
        print(f"  Total fits    : {len(test_sizes) * n_repeats * len(SensitivityAnalyzer.DEFAULT_MODELS)}")
        print(f"  lam_ridge={lam_ridge}  lam_lasso={lam_lasso}  "
              f"lam_en=({lam_en_l1},{lam_en_l2})")
        print()

        for test_size in test_sizes:
            train_pct = int(round((1 - test_size) * 100))
            label = f"{train_pct}% Train\n({int(test_size*100)}% Test)"

            for run_idx in range(n_repeats):
                seed = seed_base + run_idx

                # ── Split ──────────────────────────────────────────
                X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=seed
                )

                # ── Scale (fit on train only — no leakage) ─────────
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_raw)
                X_te_s = scaler.transform(X_te_raw)

                # ── Design matrices ────────────────────────────────
                Phi_tr = LinearRegression.add_bias(X_tr_s)
                Phi_te = LinearRegression.add_bias(X_te_s)

                # ── WLS weights from initial OLS residuals ─────────
                w_init     = LinearRegression.fit_ols(Phi_tr, y_tr)
                res_init   = LinearRegression.compute_residuals(y_tr, Phi_tr @ w_init)
                wls_w      = LinearRegression.estimate_weights_from_residuals(res_init)

                # ── Fit all models ─────────────────────────────────
                model_weights = {
                    'OLS': LinearRegression.fit_ols(Phi_tr, y_tr),
                    'Ridge': LinearRegression.fit_ridge(Phi_tr, y_tr, lam=lam_ridge),
                    'Lasso': LinearRegression.fit_lasso_cd(
                        Phi_tr, y_tr, lam=lam_lasso, num_iters=lasso_iters
                    ),
                    'Elastic Net': LinearRegression.fit_elastic_net_cd(
                        Phi_tr, y_tr, lam1=lam_en_l1, lam2=lam_en_l2,
                        num_iters=lasso_iters
                    ),
                    'WLS': LinearRegression.fit_wls(Phi_tr, y_tr, weights=wls_w),
                }

                # ── Record test-set metrics ────────────────────────
                for model_name, w in model_weights.items():
                    y_pred = LinearRegression.predict(Phi_te, w)
                    m = LinearRegression.metrics(y_te, y_pred)
                    records.append({
                        'split_label': label,
                        'test_size':   test_size,
                        'train_pct':   train_pct,
                        'run':         run_idx,
                        'model':       model_name,
                        'R2':          m['R2'],
                        'RMSE':        m['RMSE'],
                        'MAE':         m['MAE'],
                        'MSE':         m['MSE'],
                    })

            print(f"  [Done]  test_size={test_size:.1f}  "
                  f"({train_pct}% train)  — {n_repeats} runs completed.")

        df_result = pd.DataFrame(records)
        print(f"\n  Total records collected: {len(df_result)}\n")
        return df_result

    # ------------------------------------------------------------------ #
    #  2. Summary & stability table                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_summary(df_sens: 'pd.DataFrame') -> tuple:
        """
        Compute median/std per (model, train_pct) and a stability ranking.

        Returns
        -------
        summary   : DataFrame — median & std of R²/RMSE per group
        stability : DataFrame — models ranked by average std(R²)
        """
        import pandas as pd

        summary = (
            df_sens
            .groupby(['model', 'train_pct'])
            .agg(
                median_R2   = ('R2',   'median'),
                std_R2      = ('R2',   'std'),
                median_RMSE = ('RMSE', 'median'),
                std_RMSE    = ('RMSE', 'std'),
            )
            .reset_index()
            .sort_values(['model', 'train_pct'])
        )

        stability = (
            summary
            .groupby('model')['std_R2']
            .mean()
            .reset_index()
            .rename(columns={'std_R2': 'avg_std_R2'})
            .sort_values('avg_std_R2')
            .reset_index(drop=True)
        )
        stability['rank'] = range(1, len(stability) + 1)

        print("  Summary: Median R² and RMSE across repeated runs")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n  Stability Ranking  (lower avg std(R²) → more stable)")
        for _, row in stability.iterrows():
            tag = ("[ Most stable ]" if row['rank'] == 1
                   else "[ Least stable ]" if row['rank'] == len(stability)
                   else "")
            print(f"  #{int(row['rank'])}  {row['model']:<15}  "
                  f"avg std(R²) = {row['avg_std_R2']:.6f}  {tag}")

        return summary, stability

    # ------------------------------------------------------------------ #
    #  3. Automated findings                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def print_findings(
        df_sens: 'pd.DataFrame',
        stability: 'pd.DataFrame',
    ) -> None:
        """
        Print an automated English-language interpretation of the results.

        Parameters
        ----------
        df_sens   : output of run_experiment()
        stability : output of compute_summary()[1]
        """
        print("  13.1  Sensitivity Analysis — Key Findings")

        train_pcts = sorted(df_sens['train_pct'].unique())
        model_names = SensitivityAnalyzer.DEFAULT_MODELS

        # ── Best model at the smallest training set ────────────────
        smallest_pct = min(train_pcts)
        df_hard = df_sens[df_sens['train_pct'] == smallest_pct]
        best_r2_series = (
            df_hard.groupby('model')['R2']
            .median()
            .sort_values(ascending=False)
        )
        print(f"\n  Best median R² at {smallest_pct}% training data (hardest split):")
        for model, r2 in best_r2_series.items():
            print(f"    {model:<15}:  R² = {r2:.4f}")

        # ── Stability ranking ──────────────────────────────────────
        most_stable  = stability.iloc[0]['model']
        least_stable = stability.iloc[-1]['model']
        print(f"\n  Most stable model  : {most_stable} "
              f"  (avg σ(R²) = {stability.iloc[0]['avg_std_R2']:.5f})")
        print(f"  Least stable model : {least_stable} "
              f"  (avg σ(R²) = {stability.iloc[-1]['avg_std_R2']:.5f})")

        # ── R² drop: largest → smallest training set ──────────────
        largest_pct = max(train_pcts)
        print(f"\n  Median R² drop from {largest_pct}% → {smallest_pct}% train:")
        for model_name in model_names:
            r2_max = df_sens[
                (df_sens['model'] == model_name) &
                (df_sens['train_pct'] == largest_pct)
            ]['R2'].median()
            r2_min = df_sens[
                (df_sens['model'] == model_name) &
                (df_sens['train_pct'] == smallest_pct)
            ]['R2'].median()
            drop = r2_max - r2_min
            flag = ("robust"   if drop < 0.02 else
                    "moderate" if drop < 0.05 else "sensitive")
            print(f"    {model_name:<15}:  ΔR² = {drop:+.4f}  [{flag}]")

        # ── Summary ───────────────────────────────────────────────
        print("\n  Conclusion")
        print(f"""
  - '{most_stable}' is the most stable model: its R² variance is the
    lowest across all split ratios, meaning it is the least sensitive
    to the amount of training data available.

  - '{least_stable}' shows the highest variance in R², making it the
    most sensitive model to changes in the train/test split.

  - All models experience an R² decrease as the training set shrinks,
    but the magnitude of this decrease differs substantially.

  - Recommendation: prefer models with narrow boxplots and a stable
    median R², especially when the available dataset is limited.
""")


# =============================================================================
# NOISE INJECTION ANALYZER MODULE
# Robustness test: train on clean data, evaluate on Gaussian-noised test set.
# =============================================================================
class NoiseInjectionAnalyzer:
    DEFAULT_MODELS = ['OLS', 'Ridge', 'Lasso', 'Elastic Net', 'WLS']

    @staticmethod
    def add_gaussian_noise(X, sigma, seed=42):
        """Add N(0, sigma^2) noise to every element of X."""
        rng = np.random.default_rng(seed)
        return X.astype(float) + rng.normal(loc=0.0, scale=sigma, size=X.shape)

    @staticmethod
    def run_experiment(
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        sigma_levels=None,
        lam_ridge=1.0,
        lam_lasso=0.1,
        lam_en_l1=0.1,
        lam_en_l2=1.0,
        lasso_iters=500,
        noise_seed=42,
    ):
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        if sigma_levels is None:
            sigma_levels = [0.0, 0.1, 0.5, 1.0]

        # Standardise on train, apply to test (no leakage)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train_raw.astype(float))
        X_te_s = scaler.transform(X_test_raw.astype(float))

        # Build clean training design matrix
        Phi_tr = LinearRegression.add_bias(X_tr_s)

        # WLS weights from initial OLS residuals
        w_init   = LinearRegression.fit_ols(Phi_tr, y_train)
        res_init = LinearRegression.compute_residuals(y_train, Phi_tr @ w_init)
        wls_w    = LinearRegression.estimate_weights_from_residuals(res_init)

        # Fit all models on CLEAN training data
        model_weights = {
            'OLS':         LinearRegression.fit_ols(Phi_tr, y_train),
            'Ridge':       LinearRegression.fit_ridge(Phi_tr, y_train, lam=lam_ridge),
            'Lasso':       LinearRegression.fit_lasso_cd(
                               Phi_tr, y_train, lam=lam_lasso,
                               num_iters=lasso_iters),
            'Elastic Net': LinearRegression.fit_elastic_net_cd(
                               Phi_tr, y_train, lam1=lam_en_l1,
                               lam2=lam_en_l2, num_iters=lasso_iters),
            'WLS':         LinearRegression.fit_wls(Phi_tr, y_train,
                               weights=wls_w),
        }

        print("  13.2  Noise Injection Analysis -- Robustness Test")
        print(f"  Training set : {X_tr_s.shape[0]} samples (clean, no noise)")
        print(f"  Test set     : {X_te_s.shape[0]} samples (noise added per level)")
        print(f"  Sigma levels : {sigma_levels}")
        print(f"  lam_ridge={lam_ridge}  lam_lasso={lam_lasso}  "
              f"lam_en=({lam_en_l1},{lam_en_l2})")
        print()

        records  = []
        baseline = {}  # sigma=0 metrics for delta computation

        for sigma in sigma_levels:
            # Add noise to standardised test features
            X_te_noisy   = NoiseInjectionAnalyzer.add_gaussian_noise(
                X_te_s, sigma=sigma, seed=noise_seed
            )
            Phi_te_noisy = LinearRegression.add_bias(X_te_noisy)

            for model_name, w in model_weights.items():
                y_pred = LinearRegression.predict(Phi_te_noisy, w)
                m      = LinearRegression.metrics(y_test, y_pred)

                if sigma == min(sigma_levels):
                    baseline[model_name] = m.copy()

                base_r2   = baseline.get(model_name, {}).get('R2',   m['R2'])
                base_rmse = baseline.get(model_name, {}).get('RMSE', m['RMSE'])

                records.append({
                    'sigma':      sigma,
                    'model':      model_name,
                    'R2':         m['R2'],
                    'RMSE':       m['RMSE'],
                    'MAE':        m['MAE'],
                    'MSE':        m['MSE'],
                    'delta_R2':   m['R2']   - base_r2,
                    'delta_RMSE': m['RMSE'] - base_rmse,
                })

            print(f"  [Done]  sigma = {sigma:.2f}  -- all models evaluated.")

        df_result = pd.DataFrame(records)
        print(f"\n  Total records: {len(df_result)}\n")
        return df_result

    @staticmethod
    def compute_summary(df_noise):
        """Print pivot tables and robustness ranking. Returns (pivot_r2, pivot_rmse, robustness_df)."""
        import pandas as pd

        pivot_r2   = df_noise.pivot(index='sigma', columns='model', values='R2').round(4)
        pivot_rmse = df_noise.pivot(index='sigma', columns='model', values='RMSE').round(4)

        print("  R2 Score per Noise Level (higher is better)")
        print(pivot_r2.to_string())

        print()
        print("  RMSE per Noise Level (lower is better)")
        print(pivot_rmse.to_string())

        sigma_min = df_noise['sigma'].min()
        sigma_max = df_noise['sigma'].max()

        robustness = []
        for model in NoiseInjectionAnalyzer.DEFAULT_MODELS:
            sub      = df_noise[df_noise['model'] == model]
            r2_base  = sub[sub['sigma'] == sigma_min]['R2'].values[0]
            r2_worst = sub[sub['sigma'] == sigma_max]['R2'].values[0]
            robustness.append({'model': model, 'R2_drop': r2_base - r2_worst})

        robustness_df = (
            pd.DataFrame(robustness)
            .sort_values('R2_drop')
            .reset_index(drop=True)
        )
        robustness_df['rank'] = range(1, len(robustness_df) + 1)

        print()
        print("  Robustness Ranking  (smallest R2 drop = most robust)")
        for _, row in robustness_df.iterrows():
            tag = ("[ Most robust ]"  if row['rank'] == 1
                   else "[ Least robust ]" if row['rank'] == len(robustness_df)
                   else "")
            print(f"  #{int(row['rank'])}  {row['model']:<15}  "
                  f"R2 drop = {row['R2_drop']:.4f}  {tag}")

        return pivot_r2, pivot_rmse, robustness_df

    @staticmethod
    def plot_degradation(df_noise, model_names=None):
        import matplotlib.pyplot as plt

        if model_names is None:
            model_names = NoiseInjectionAnalyzer.DEFAULT_MODELS

        PALETTE = {
            'OLS':         '#4C72B0',
            'Ridge':       '#DD8452',
            'Lasso':       '#55A868',
            'Elastic Net': '#C44E52',
            'WLS':         '#8172B2',
        }
        MARKERS = ['o', 's', '^', 'D', 'v']

        sigma_vals = sorted(df_noise['sigma'].unique())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            'Noise Injection Robustness Test\n'
            'Train on clean data -- Test on Gaussian-noised features',
            fontsize=15, fontweight='bold'
        )

        for ax_idx, (metric, ylabel) in enumerate([
            ('R2',   'R2 Score  (higher is better)'),
            ('RMSE', 'RMSE      (lower is better)'),
        ]):
            ax = axes[ax_idx]
            for m_idx, model_name in enumerate(model_names):
                sub   = df_noise[df_noise['model'] == model_name].sort_values('sigma')
                vals  = sub[metric].values
                color  = PALETTE.get(model_name, '#888888')
                marker = MARKERS[m_idx % len(MARKERS)]
                ax.plot(sigma_vals, vals, marker=marker, linewidth=2.2,
                        markersize=8, color=color, label=model_name)

            ax.set_xlabel('Noise Level  (sigma)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{metric} vs Noise Level', fontsize=13,
                         fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xticks(sigma_vals)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_findings(df_noise, robustness_df):
        """Print automated English-language interpretation of noise injection results."""
        sigma_vals   = sorted(df_noise['sigma'].unique())
        sigma_min    = min(sigma_vals)
        sigma_max    = max(sigma_vals)
        most_robust  = robustness_df.iloc[0]['model']
        least_robust = robustness_df.iloc[-1]['model']
        model_names  = NoiseInjectionAnalyzer.DEFAULT_MODELS

        print("  13.2  Noise Injection -- Key Findings")

        df_base = df_noise[df_noise['sigma'] == sigma_min]
        df_max  = df_noise[df_noise['sigma'] == sigma_max]

        print(f"\n  Baseline R2 (sigma={sigma_min}, no noise):")
        for model_name in model_names:
            r2 = df_base[df_base['model'] == model_name]['R2'].values[0]
            print(f"    {model_name:<15}:  R2 = {r2:.4f}")

        print(f"\n  R2 at sigma={sigma_max:.1f} (highest noise):")
        for model_name in model_names:
            r2_now  = df_max[df_max['model'] == model_name]['R2'].values[0]
            r2_base = df_base[df_base['model'] == model_name]['R2'].values[0]
            drop    = r2_base - r2_now
            flag    = ("robust"   if drop < 0.02 else
                       "moderate" if drop < 0.05 else "sensitive")
            print(f"    {model_name:<15}:  R2 = {r2_now:.4f}  "
                  f"(drop = {drop:+.4f})  [{flag}]")

        print("\n  Conclusion")
        print(f"""
  - '{most_robust}' is the most robust model: its R2 decreases the
    least as Gaussian noise is added to the test features.

  - '{least_robust}' is the most sensitive model: its performance
    degrades the most, suggesting its weight vector ||w|| is large,
    which amplifies the effect of input perturbations.

  - Regularised models (Ridge / Elastic Net) tend to be more robust
    because smaller weights reduce noise amplification:
    output_perturbation ~ ||w|| * sigma.

  - Recommendation: when deploying in noisy sensor environments,
    prefer regularised models over plain OLS.
""")

