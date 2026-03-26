
import numpy as np


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
    """Cubic spline basis (truncated power basis) implemented from scratch."""
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
def fit_ridge(Phi: np.ndarray, y: np.ndarray, lam: float, bias_is_first: bool = True) -> np.ndarray:
    P = Phi.shape[1]
    A = Phi.T @ Phi
    reg = lam * np.eye(P)
    if bias_is_first:
        reg[0, 0] = 0.0
    return np.linalg.solve(A + reg, Phi.T @ y)


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
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

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
