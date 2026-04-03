import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    return sigmoid(z)


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
        sig_parts.append(sigmoid(z))
    parts.append(np.hstack(sig_parts))

    return np.hstack(parts)


def poly_features(X_s: np.ndarray, degree: int) -> np.ndarray:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    feats = [X_s.astype(float)]
    for p in range(2, degree + 1):
        feats.append(X_s.astype(float) ** p)
    return np.hstack(feats)


def rbf_features(X_s: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    x2 = np.sum(X_s * X_s, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X_s @ centers.T)
    return np.exp(-gamma * d2)


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
        x = X_s[:, d:d+1] # (N, 1)
        for p in range(2, degree_actual + 1):
            parts.append(x ** p)
        
        k_d = knots[d]
        for k in k_d:
            trunc = np.maximum(0, x - k) ** degree_actual
            parts.append(trunc)
            
    Z = np.hstack(parts) if parts else np.zeros((N, 0))
    return Z.astype(float), transformer


def sigmoid_features(X_s: np.ndarray, centers_per_feature: np.ndarray, slope: float) -> np.ndarray:
    N, D = X_s.shape
    M = centers_per_feature.shape[1]
    out = []
    for d in range(D):
        z = slope * (X_s[:, [d]] - centers_per_feature[d][None, :])
        out.append(sigmoid_stable(z))
    return np.hstack(out)


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
            parts.append(poly_features(X_s, degree=d)[:, X_s.shape[1]:])

    if basis.get('rbf') is not None:
        r = basis['rbf']
        parts.append(rbf_features(X_s, centers=r['centers'], gamma=float(r['gamma'])))

    if basis.get('sigmoid') is not None:
        s = basis['sigmoid']
        parts.append(sigmoid_features(X_s, centers_per_feature=s['centers'], slope=float(s['slope'])))

    if basis.get('spline') is not None:
        sp = basis['spline']
        n_knots = int(sp['n_knots'])
        degree = int(sp.get('degree', 3))
        transformer = sp.get('transformer', None)
        Z, fitted = spline_features(
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




def compute_rbf_gamma(centers: np.ndarray) -> float:
    d2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    triu = np.triu_indices(d2.shape[0], k=1)
    med = np.median(d2[triu]) if np.any(d2[triu] > 0) else 1.0
    return 1.0 / (2.0 * max(med, 1e-9))


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
    rbf_gamma = compute_rbf_gamma(rbf_centers)
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
        Phi_tr = make_design_matrix(X_train_s, basis=cfg, add_linear=True)
        Phi_va = make_design_matrix(X_val_s, basis=cfg, add_linear=True)
        Phi_te = make_design_matrix(X_test_s, basis=cfg, add_linear=True)

        w = fit_ridge(Phi_tr, y_train, lam=lam)
        pred_tr = predict(Phi_tr, w)
        pred_va = predict(Phi_va, w)
        pred_te = predict(Phi_te, w)

        results[model_name] = {
            'train': metrics(y_train, pred_tr),
            'val': metrics(y_val, pred_va),
            'test': metrics(y_test, pred_te),
        }

    return results

def fit_ridge_closed_form(Phi: np.ndarray, y: np.ndarray, lam: float, bias_is_first: bool = True) -> np.ndarray:
    return fit_ridge(Phi, y, lam, bias_is_first=bias_is_first)


def predict(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    return Phi @ w


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'MSE': float(mse), 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    return float(np.mean((y_true - y_pred) ** 2))

def add_bias(X_like: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X_like.shape[0], 1), dtype=float), X_like])

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


# LINEAR REGRESSION IMPLEMENTATIONS (OLS, Mini-batch GD, WLS)
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


def step_decay_schedule(epoch: int, initial_lr: float, drop_rate: float = 0.5, 
                        epochs_drop: int = 10) -> float:
    return initial_lr * (drop_rate ** np.floor(epoch / epochs_drop))


def cosine_annealing_schedule(epoch: int, initial_lr: float, T_max: int) -> float:
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / T_max))


def fit_ols_minibatch_gd(Phi: np.ndarray, y: np.ndarray, 
                         lr_schedule: str = 'step_decay',
                         initial_lr: float = 0.01,
                         batch_size: int = 32,
                         num_epochs: int = 100,
                         bias_is_first: bool = True,
                         drop_rate: float = 0.5,
                         epochs_drop: int = 10) -> tuple:
    N, D = Phi.shape
    w = np.zeros(D)
    loss_history = []
    
    for epoch in range(num_epochs):
        # Get learning rate for current epoch
        if lr_schedule == 'step_decay':
            lr = step_decay_schedule(epoch, initial_lr, drop_rate, epochs_drop)
        elif lr_schedule == 'cosine_annealing':
            lr = cosine_annealing_schedule(epoch, initial_lr, num_epochs)
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


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_true - y_pred


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
    w_aux = fit_ols(Phi, normalized_residuals_sq, bias_is_first=True)
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


def estimate_weights_from_residuals(residuals: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    # Weights are inversely proportional to squared residuals
    weights = 1.0 / (residuals ** 2 + epsilon)
    
    # Normalize weights to have mean = 1
    weights = weights / np.mean(weights)
    
    return weights


def fit_wls(Phi: np.ndarray, y: np.ndarray, weights: np.ndarray, 
            bias_is_first: bool = True) -> np.ndarray:
    # Create diagonal weight matrix
    W = np.diag(np.sqrt(weights))
    
    # Transform features and targets: Phi_w = W @ Phi, y_w = W @ y
    Phi_weighted = W @ Phi
    y_weighted = W @ y
    
    # Solve weighted normal equations: w = (Phi_w^T Phi_w)^(-1) Phi_w^T y_w
    w = fit_ols(Phi_weighted, y_weighted, bias_is_first=bias_is_first)
    
    return w

def soft_threshold(rho: float, lam: float) -> float:
    # force the weights of unimportant features to zero
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

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
                w[j] = soft_threshold(rho_j, lam) / z[j]
                
        # stop the algorithm if the change in w is very small (converged)
        if np.max(np.abs(w - w_old)) < tol:
            break
            
    return w

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

def kfold_cv_lasso(Phi: np.ndarray, y: np.ndarray, lambdas: list, k: int = 10):
    # sort lambdas in descending order to make warm start more effective
    lambdas = sorted(lambdas, reverse=True)
    
    # call the function to split indices into folds
    folds = time_series_cv_indices(len(Phi), k=k)
    
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
            w = fit_lasso_cd(Phi_tr, y_tr, lam, w_init=w_inits[i])
            
            # update the starting point for fold i to use for the next lambda value
            w_inits[i] = w.copy()
            
            # compute the error on the validation set
            y_pred = Phi_va @ w
            mse = np.mean((y_va - y_pred)**2)
            fold_errors.append(mse)
            
        # store the average error across k folds for the current lambda
        cv_errors.append(np.mean(fold_errors))
        
    # find the lambda with the smallest average validation error
    best_idx = np.argmin(cv_errors)
    best_lam = lambdas[best_idx]
    
    return best_lam, cv_errors, lambdas


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
                w[j] = soft_threshold(rho_j, lam1) / (z[j] + lam2)
                
        # stop the algorithm if the weights change very little
        if np.max(np.abs(w - w_old)) < tol: break
        
    return w

def forward_selection(Phi_train, y_train, Phi_val, y_val, k_features, lam=0.1):
    selected = [0] # always keep the bias column (column 0)
    remaining = list(range(1, Phi_train.shape[1]))
    
    for _ in range(k_features):
        best_feature = None
        best_error = float("inf")
        
        for f in remaining:
            trial = selected + [f]
            # Fit the model on the training set
            w = fit_ridge(Phi_train[:, trial], y_train, lam)
            # Predict and compute the error on the validation set
            y_pred_val = Phi_val[:, trial] @ w
            error = np.mean((y_val - y_pred_val)**2)
            
            if error < best_error:
                best_error = error
                best_feature = f
                
        selected.append(best_feature)
        remaining.remove(best_feature)
        
    return selected

def backward_elimination(Phi_train, y_train, Phi_val, y_val, target_features, lam=0.1):
    features = list(range(Phi_train.shape[1]))
    
    while len(features) > target_features:
        best_error = float("inf")
        worst_feature = None
        
        for f in features:
            if f == 0: continue 
                
            trial = [x for x in features if x != f]
            w = fit_ridge(Phi_train[:, trial], y_train, lam)
            y_pred_val = Phi_val[:, trial] @ w
            error = np.mean((y_val - y_pred_val)**2)
            
            if error < best_error:
                best_error = error
                worst_feature = f
                
        features.remove(worst_feature)
        
    return features

# KERNEL RIDGE REGRESSION 
def rbf_kernel_matrix(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # Calculate the rbf (gaussian) kernel matrix between two sets of data points.
    dists = cdist(X1, X2, metric='sqeuclidean')
    return np.exp(-gamma * dists)

def poly_kernel_matrix(X1: np.ndarray, X2: np.ndarray, degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    # Calculate the polynomial kernel matrix.
    return (X1 @ X2.T + coef0) ** degree

def fit_kernel_ridge(K_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
    # Train the kernel ridge regression model and return the dual variables (alpha).
    n = K_train.shape[0]
    # Solve the linear system (K + lambda * I) * alpha = y to find alpha.
    alpha = np.linalg.solve(K_train + lam * np.eye(n), y_train)
    return alpha

def predict_kernel_ridge(K_test: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # Predict the target values using the learned alpha weights and the test kernel matrix.
    return K_test @ alpha

# GAUSSIAN PROCESS REGRESSION 
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
        lml, grad = gp_lml_and_grad(theta, X, y)
        if lml == -np.inf:
            break
        
        # Apply the gradient ascent update rule.
        theta = theta + lr * grad
        lml_history.append(lml)
        iter_time = time.time() - iter_start
        print(f"Iteration {i+1}/{max_iters} - LML: {lml:.4f} - Thời gian: {iter_time:.4f}s")
        
    return theta, lml_history

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

def evaluate_and_print(name, y_true, y_pred):
    # Calculate metrics 
    eval_metrics = metrics(y_true, y_pred) 
    test_mse = mse(y_true, y_pred)
    
    # Print formatted row
    print(f"{name:<35} | {test_mse:<10.4f} | {eval_metrics['RMSE']:<10.4f} | {eval_metrics['MAE']:<10.4f} | {eval_metrics['R2']:<10.4f}")

# ROBUST REGRESSION — IRLS with Huber Loss
def huber_loss(residuals, delta):
    abs_r = np.abs(residuals)
    loss = np.where(
        abs_r <= delta,
        0.5 * residuals**2,
        delta * abs_r - 0.5 * delta**2
    )
    return np.mean(loss)


def huber_weights(residuals, delta):
    """Tính trọng số IRLS từ Huber Loss."""
    abs_r = np.abs(residuals)
    weights = np.where(
        abs_r <= delta,
        1.0,
        delta / (abs_r + 1e-8)
    )
    return weights


def fit_irls_huber(Phi, y, delta=1.345, max_iter=50, tol=1e-6, lam=0.0):
    N, D = Phi.shape
    
    # Bước 0: Khởi tạo bằng OLS (hoặc Ridge nếu lam > 0)
    if lam > 0:
        w = fit_ridge(Phi, y, lam, bias_is_first=True)
    else:
        w = fit_ols(Phi, y, bias_is_first=True)
    
    loss_history = []
    
    for iteration in range(max_iter):
        # 1. Tính phần dư (residuals)
        y_pred = Phi @ w
        residuals = y - y_pred
        
        # 2. Tính Huber Loss hiện tại
        current_loss = huber_loss(residuals, delta)
        loss_history.append(current_loss)
        
        # 3. Tính trọng số IRLS từ Huber
        sample_weights = huber_weights(residuals, delta)
        
        # 4. Giải bài toán WLS (Weighted Least Squares)
        W_diag = np.diag(np.sqrt(sample_weights))
        Phi_w = W_diag @ Phi
        y_w = W_diag @ y
        
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
            loss_history.append(huber_loss(residuals, delta))
            print(f"  IRLS hội tụ sau {iteration + 1} vòng lặp.")
            break
        
        w = w_new
    else:
        print(f"  IRLS chưa hội tụ sau {max_iter} vòng lặp.")
    
    return w, loss_history

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


# BIAS-VARIANCE DECOMPOSITION — via Bootstrapping
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
            
            w_b = fit_ridge(Phi_b, y_b, lam=lam, bias_is_first=True)
            all_predictions[b, :] = Phi_test @ w_b
        
        mean_predictions = np.mean(all_predictions, axis=0)
        
        bias_sq = np.mean((mean_predictions - y_test) ** 2)
        variance = np.mean(np.var(all_predictions, axis=0))
        mse_total = np.mean((all_predictions - y_test[np.newaxis, :]) ** 2)
        
        bias_squared_list.append(bias_sq)
        variance_list.append(variance)
        mse_list.append(mse_total)
        
        print(f"  λ = {lam:>10.4f} (log₁₀={np.log10(lam):>6.2f}): "
              f"Bias² = {bias_sq:>10.4f}, Var = {variance:>10.4f}, MSE = {mse_total:>10.4f}",
              end='\r')
    
    elapsed = time.time() - total_start
    print(f"\n\nBootstrapping hoàn tất! Thời gian: {elapsed:.1f}s ({n_bootstrap} lần × {n_lambdas} λ)")
    
    return bias_squared_list, variance_list, mse_list


# =====================================================================
# Model Evaluation Utilities
# =====================================================================

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


def build_model_comparison_table(model_results: dict):
    """
    Print a formatted comparison table.
    
    Parameters:
        model_results: dict of {model_name: {'MSE':..., 'RMSE':..., 'MAE':..., 'R2':...}}
    """
    header = f"{'Model':<35} {'MSE':>12} {'RMSE':>12} {'MAE':>12} {'R²':>12}"
    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, m in model_results.items():
        print(f"{name:<35} {m['MSE']:>12.4f} {m['RMSE']:>12.4f} "
              f"{m['MAE']:>12.4f} {m['R2']:>12.4f}")
    print(sep)


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
    folds = time_series_cv_indices(len(y), k=k)
    fold_metrics_list = []
    
    for train_idx, val_idx in folds:
        Phi_tr, y_tr = Phi[train_idx], y[train_idx]
        Phi_va, y_va = Phi[val_idx], y[val_idx]
        
        w = fit_fn(Phi_tr, y_tr)
        y_pred = Phi_va @ w
        m = metrics(y_va, y_pred)
        fold_metrics_list.append(m)
    
    return fold_metrics_list


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
    from scipy.stats import ttest_rel, wilcoxon as scipy_wilcoxon
    
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

# BAYESIAN LINEAR REGRESSION
def gaussian_rbf(X: np.ndarray, centers: np.ndarray, s: float) -> np.ndarray:
    X = np.atleast_2d(X).reshape(-1, 1)          # (N, 1)
    centers = np.atleast_1d(centers).ravel()      # (M,)

    # Broadcast difference: (N, 1) - (1, M) = (N, M)
    diff = X - centers[np.newaxis, :]             # (N, M)
    return np.exp(-(diff ** 2) / (2.0 * s ** 2))


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
