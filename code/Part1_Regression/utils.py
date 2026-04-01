
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def soft_threshold(rho: float, lam: float) -> float:
    # ép trọng số của các đặc trưng không quan trọng về 0
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def fit_lasso_cd(Phi: np.ndarray, y: np.ndarray, lam: float, num_iters: int = 1000, tol: float = 1e-4, bias_is_first: bool = True, w_init: np.ndarray = None) -> np.ndarray:
    n_samples, n_features = Phi.shape
    
    # Dùng w_init nếu có, ngược lại tạo mảng 0
    w = np.zeros(n_features) if w_init is None else w_init.copy()
    
    # z là tổng bình phương các phần tử trên từng cột của ma trận Phi
    z = np.sum(Phi**2, axis=0)
    
    for _ in range(num_iters):
        w_old = w.copy()
        
        for j in range(n_features):
            if z[j] == 0:
                continue
            
            # tính sai số dự đoán hiện tại
            y_pred = Phi @ w
            
            # rho_j là độ tương quan giữa đặc trưng j và sai số (khi tạm bỏ w_j)
            rho_j = Phi[:, j].T @ (y - y_pred) + w[j] * z[j]
            
            # cột đầu tiên (bias) không bị phạt, các cột khác sẽ đi qua bộ lọc soft threshold
            if bias_is_first and j == 0:
                w[j] = rho_j / z[j]  
            else:
                w[j] = soft_threshold(rho_j, lam) / z[j]
                
        # dừng thuật toán nếu mức độ thay đổi của w quá nhỏ (đã hội tụ)
        if np.max(np.abs(w - w_old)) < tol:
            break
            
    return w

def fit_ridge(Phi: np.ndarray, y: np.ndarray, lam: float, bias_is_first: bool = True) -> np.ndarray:
    # p là số lượng đặc trưng của ma trận đầu vào
    P = Phi.shape[1]
    
    # tính phần lõi của phương trình chuẩn (normal equation)
    A = Phi.T @ Phi
    
    # tạo ma trận đường chéo chứa các hệ số phạt lambda
    reg = lam * np.eye(P)
    
    # không áp dụng hình phạt l2 lên hệ số tự do (bias) ở vị trí đầu tiên
    if bias_is_first:
        reg[0, 0] = 0.0
        
    # tìm trọng số w bằng cách giải hệ phương trình tuyến tính
    return np.linalg.solve(A + reg, Phi.T @ y)

def time_series_cv_indices(n_samples: int, k: int = 10, random_seed: int = 42):
    # Chia dữ liệu làm (k + 1) phần bằng nhau
    chunk_size = n_samples // (k + 1)
    
    folds = []
    
    for i in range(1, k + 1):
        # Tập Train: Từ đầu cho đến chunk hiện tại 
        train_end = i * chunk_size
        train_idx = np.arange(0, train_end)
        
        # Tập Validation
        val_end = (i + 1) * chunk_size if i < k else n_samples
        val_idx = np.arange(train_end, val_end)
        
        folds.append((train_idx, val_idx))
        
    return folds

def kfold_cv_lasso(Phi: np.ndarray, y: np.ndarray, lambdas: list, k: int = 10):
    # sắp xếp lambda giảm dần để warm start hiệu quả
    lambdas = sorted(lambdas, reverse=True)
    
    # gọi hàm chia index
    folds = kfold_indices(len(Phi), k=k)
    
    cv_errors = []
    
    # tạo danh sách lưu trọng số w khởi tạo cho từng fold.
    # vì có k fold, ta cần k vạch xuất phát khác nhau. ban đầu tất cả là None (sẽ khởi tạo w=0)
    w_inits = [None] * k
    
    for lam in lambdas:
        fold_errors = []
        
        for i, (train_idx, val_idx) in enumerate(folds):
            Phi_tr, y_tr = Phi[train_idx], y[train_idx]
            Phi_va, y_va = Phi[val_idx], y[val_idx]
            
            # fit mô hình với Warm Start: truyền w_inits[i] của vòng lambda trước vào
            w = fit_lasso_cd(Phi_tr, y_tr, lam, w_init=w_inits[i])
            
            # cập nhật lại vạch xuất phát cho fold thứ i để dùng cho mức lambda tiếp theo
            w_inits[i] = w.copy()
            
            # tính lỗi trên tập Validation
            y_pred = Phi_va @ w
            mse = np.mean((y_va - y_pred)**2)
            fold_errors.append(mse)
            
        # lưu lỗi trung bình của k folds tại mức lambda hiện tại
        cv_errors.append(np.mean(fold_errors))
        
    # tìm lambda có lỗi validation trung bình nhỏ nhất
    best_idx = np.argmin(cv_errors)
    best_lam = lambdas[best_idx]
    
    return best_lam, cv_errors, lambdas


def fit_elastic_net_cd(Phi: np.ndarray, y: np.ndarray, lam1: float, lam2: float, num_iters: int = 1000, tol: float = 1e-4, bias_is_first: bool = True) -> np.ndarray:
    # cài đặt elastic net bằng coordinate descent
    n_samples, n_features = Phi.shape
    w = np.zeros(n_features)
    
    # z là tổng bình phương các phần tử trên từng cột
    z = np.sum(Phi**2, axis=0)
    
    for _ in range(num_iters):
        w_old = w.copy()
        
        for j in range(n_features):
            if z[j] == 0: continue
            
            # tính sai số dự đoán hiện tại
            y_pred = Phi @ w
            
            # rho_j là độ tương quan giữa đặc trưng j và phần sai số
            rho_j = Phi[:, j].T @ (y - y_pred) + w[j] * z[j]
            
            if bias_is_first and j == 0:
                # hệ số tự do không bị phạt
                w[j] = rho_j / z[j]
            else:
                # tử số là l1, mẫu số cộng thêm l2 
                w[j] = soft_threshold(rho_j, lam1) / (z[j] + lam2)
                
        # dừng thuật toán nếu trọng số gần như không đổi
        if np.max(np.abs(w - w_old)) < tol: break
        
    return w

def forward_selection(Phi_train, y_train, Phi_val, y_val, k_features, lam=0.1):
    selected = [0] # luôn giữ lại cột bias (cột 0)
    remaining = list(range(1, Phi_train.shape[1]))
    
    for _ in range(k_features):
        best_feature = None
        best_error = float("inf")
        
        for f in remaining:
            trial = selected + [f]
            # Fit mô hình trên tập train
            w = fit_ridge(Phi_train[:, trial], y_train, lam)
            # Dự đoán và tính lỗi trên tập validation
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


# LINEAR REGRESSION IMPLEMENTATIONS (OLS, Mini-batch GD, WLS)

def fit_ols(Phi: np.ndarray, y: np.ndarray, bias_is_first: bool = True) -> np.ndarray:
    # Normal Equations: w = (Phi^T Phi)^(-1) Phi^T y
    PhiT_Phi = Phi.T @ Phi
    PhiT_y = Phi.T @ y
    
    # Solve linear system instead of computing inverse for numerical stability
    w = np.linalg.solve(PhiT_Phi, PhiT_y)
    
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


# ============================================================
# ROBUST REGRESSION — IRLS with Huber Loss
# ============================================================

def huber_loss(residuals, delta):
    """Tính Huber Loss cho từng phần tử."""
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
    """
    Hồi quy bền vững bằng IRLS với Huber Loss.
    
    Parameters:
        Phi      : Ma trận đặc trưng (N x D)
        y        : Vector mục tiêu (N,)
        delta    : Ngưỡng Huber — điểm chuyển từ L2 sang L1
        max_iter : Số vòng lặp tối đa
        tol      : Ngưỡng hội tụ (thay đổi trọng số w giữa 2 vòng lặp)
        lam      : Hệ số regularization (Ridge) — mặc định 0 (không regularize)
    
    Returns:
        w            : Trọng số tối ưu
        loss_history : Lịch sử Huber Loss qua từng vòng lặp
    """
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


# ============================================================
# OUTLIER INJECTION — For Sensitivity Analysis
# ============================================================

def inject_outliers(y, fraction=0.05, multiplier=10, seed=42):
    """
    Chèn outliers nhân tạo vào vector y.
    
    Parameters:
        y          : Vector mục tiêu gốc
        fraction   : Tỉ lệ mẫu bị biến thành outlier (0.05 = 5%)
        multiplier : Hệ số nhân để tạo outlier (y_outlier = y_mean + multiplier * y_std)
        seed       : Random seed
    
    Returns:
        y_corrupted    : Vector y đã bị chèn outlier
        outlier_mask   : Boolean array đánh dấu vị trí outlier
    """
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


# ============================================================
# BIAS-VARIANCE DECOMPOSITION — via Bootstrapping
# ============================================================

def bias_variance_decomposition(Phi_train, y_train, Phi_test, y_test,
                                 lambdas, n_bootstrap=200, seed=42):
    """
    Phân tích Bias-Variance Tradeoff bằng Bootstrapping.
    
    Parameters:
        Phi_train   : Ma trận đặc trưng tập train (N_train x D)
        y_train     : Vector mục tiêu tập train (N_train,)
        Phi_test    : Ma trận đặc trưng tập test (N_test x D) 
        y_test      : Vector mục tiêu tập test (N_test,)
        lambdas     : Danh sách các giá trị lambda
        n_bootstrap : Số lần lặp bootstrap
        seed        : Random seed
    
    Returns:
        bias_squared_list : Bias² cho mỗi lambda
        variance_list     : Variance cho mỗi lambda
        mse_list          : MSE tổng cho mỗi lambda
    """
    import time
    
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
