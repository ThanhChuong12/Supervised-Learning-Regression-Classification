import numpy as np
import time
import pandas as pd
from typing import List, Tuple, Optional, Type, Dict
from scipy.stats import norm

class Perceptron:
    """
    Original Perceptron algorithm implemented from scratch.
    Stores error history to observe convergence.
    """
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.errors_history = [] # Store number of misclassified samples per epoch

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_history = []
        
        # Standard Perceptron works best with labels {-1, 1}
        # Convert labels {0, 1} to {-1, 1} if needed
        y_ = np.where(y <= 0, -1, 1)

        for epoch in range(self.max_iter):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Update if prediction is wrong (y_true * y_pred <= 0)
                if y_[idx] * linear_output <= 0:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]
                    errors += 1
            
            self.errors_history.append(errors)
            # Early stopping: stop if no misclassified samples (linearly separable data)
            if errors == 0:
                break
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        # Return labels {0, 1} to match original dataset
        return np.where(linear_output >= 0, 1, 0)

class LogisticRegression:
    """
    Logistic Regression algorithm with:
    1. Regularization (L1, L2).
    2. Class-weighted loss to handle imbalanced data.
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, penalty=None, lambda_reg=0.1, class_weight=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = penalty           # 'l1', 'l2', or None
        self.lambda_reg = lambda_reg     # Regularization coefficient lambda
        self.class_weight = class_weight # 'balanced' or None
        
        self.weights = None
        self.bias = None
        self.loss_history = [] # Store cost function values for plotting

    def _sigmoid(self, z):
        # np.clip to avoid overflow when z is too large or too small
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _compute_class_weights(self, y):
        n_samples = len(y)
        classes = np.unique(y)
        weights = np.ones(n_samples)
        
        if self.class_weight == 'balanced':
            # Formula: c_k proportional to N / N_k
            for c in classes:
                n_classes = len(classes)
                n_c = np.sum(y == c)
                # Compute weight: N / (K * N_c)
                w_c = n_samples / (n_classes * n_c) 
                weights[y == c] = w_c
        return weights

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # Compute sample weights (handle class imbalance)
        sample_weights = self._compute_class_weights(y)

        for i in range(self.max_iter):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Compute Loss (weighted cross-entropy)
            epsilon = 1e-15 # Avoid log(0)
            loss_n = -sample_weights * (y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
            loss = np.mean(loss_n)

            # Add regularization term to Loss
            if self.penalty == 'l2':
                loss += (self.lambda_reg / 2) * np.sum(self.weights ** 2)
            elif self.penalty == 'l1':
                loss += self.lambda_reg * np.sum(np.abs(self.weights))
            
            self.loss_history.append(loss)

            # Compute Gradient with sample weights
            # Derivative: dw = (1/N) * X^T * (sample_weights * (y_pred - y))
            error_weighted = sample_weights * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, error_weighted)
            db = (1 / n_samples) * np.sum(error_weighted)

            # Add gradient of regularization term
            if self.penalty == 'l2':
                dw += self.lambda_reg * self.weights
            elif self.penalty == 'l1':
                dw += self.lambda_reg * np.sign(self.weights)

            # Update weights (Gradient Descent)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_pred_prob = self.predict_proba(X)
        return np.where(y_pred_prob >= threshold, 1, 0)
    
# BINARY LOGISTIC REGRESSION
class BinaryLogisticRegression:
    """
    Binary Logistic Regression with:
    - Gradient Descent (GD)
    - Newton-Raphson (via Hessian-Free Conjugate Gradient)
    """

    def __init__(
        self,
        method: str = 'gd',
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-5,
        fit_intercept: bool = True,
        verbose: bool = False
    ):
        self.method = method
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        self.w: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.time_history: List[float] = []

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid (branch-wise)."""
        return np.piecewise(
            z,
            [z > 0],
            [lambda x: 1 / (1 + np.exp(-x)),
             lambda x: np.exp(x) / (1 + np.exp(x))]
        )

    @staticmethod
    def _compute_loss(y: np.ndarray, z: np.ndarray) -> float:
        """
        Stable binary cross-entropy using log-sum-exp trick.
        """
        loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
        return np.mean(loss)

    def _hvp(
        self,
        X: np.ndarray,
        R_diag: np.ndarray,
        v: np.ndarray,
        reg: float = 1e-5
    ) -> np.ndarray:
        """
        Hessian-vector product (H * v).
        """
        N = X.shape[0]
        return (X.T @ (R_diag * (X @ v))) / N + reg * v

    def _conjugate_gradient(
        self,
        X: np.ndarray,
        R_diag: np.ndarray,
        b: np.ndarray,
        tol: float = 1e-5,
        max_iter: int = 50
    ) -> np.ndarray:
        """
        Solve Hx = b using Conjugate Gradient without explicit Hessian.
        """
        x = np.zeros_like(b)
        r = b - self._hvp(X, R_diag, x)
        p = r.copy()
        rsold = r.T @ r
        for _ in range(max_iter):
            Ap = self._hvp(X, R_diag, p)
            alpha = rsold / (p.T @ Ap + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.T @ r
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._add_intercept(X)
        N, d = X.shape
        self.w = np.zeros(d)
        self.loss_history = []
        self.time_history = []
        start_time = time.time()
        for i in range(self.max_iter):
            z = X @ self.w
            y_pred = self._sigmoid(z)
            loss = self._compute_loss(y, z)
            self.loss_history.append(loss)
            self.time_history.append(time.time() - start_time)

            # Convergence check
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"[{self.method.upper()}] Converged at iteration {i}")
                break
            gradient = (X.T @ (y_pred - y)) / N

            if self.method == 'gd':
                self.w -= self.lr * gradient
            elif self.method == 'newton':
                R_diag = y_pred * (1 - y_pred)
                delta = self._conjugate_gradient(X, R_diag, gradient)
                self.w -= delta
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(self._add_intercept(X) @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

# SOFTMAX REGRESSION (MULTICLASS)
class SoftmaxRegression:
    """
    Multiclass Logistic Regression using Softmax with Log-Sum-Exp stability.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-5,
        fit_intercept: bool = True,
        verbose: bool = False
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.W: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.time_history: List[float] = []

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X

    @staticmethod
    def _logsumexp(Z: np.ndarray) -> np.ndarray:
        """Stable log-sum-exp computation."""
        M = np.max(Z, axis=1, keepdims=True)
        return M + np.log(np.sum(np.exp(Z - M), axis=1, keepdims=True))

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        return np.exp(Z - self._logsumexp(Z))

    def _compute_loss(self, y_onehot: np.ndarray, Z: np.ndarray) -> float:
        lse = self._logsumexp(Z)
        true_scores = np.sum(y_onehot * Z, axis=1, keepdims=True)
        return np.mean(lse - true_scores)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        X = self._add_intercept(X)
        N, d = X.shape
        self.W = np.zeros((d, K))
        y_onehot = np.eye(K)[y]
        start_time = time.time()

        for i in range(self.max_iter):
            Z = X @ self.W
            loss = self._compute_loss(y_onehot, Z)
            self.loss_history.append(loss)
            self.time_history.append(time.time() - start_time)
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"[SOFTMAX] Converged at iteration {i}")
                break
            probs = self._softmax(Z)
            gradient = (X.T @ (probs - y_onehot)) / N
            self.W -= self.lr * gradient
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(self._add_intercept(X) @ self.W)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

# ONE-VS-REST (OvR) CLASSIFIER
class OneVsRestClassifier:
    """
    One-vs-Rest (OvR) strategy for multiclass classification.
    This meta-estimator converts any binary classifier into a multiclass classifier
    by training one classifier per class.
    Each classifier learns: "class c vs all other classes".
    """

    def __init__(self, estimator_cls: Type, **kwargs):
        self.estimator_cls = estimator_cls
        self.kwargs = kwargs
        self.estimators_: List = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train one binary classifier per class.
        """
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for c in self.classes_:
            # Convert to binary problem: class c = 1, others = 0
            y_binary = (y == c).astype(int)
            estimator = self.estimator_cls(**self.kwargs)
            estimator.fit(X, y_binary)
            self.estimators_.append(estimator)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        """
        # Collect probability of positive class from each estimator
        probas = np.column_stack([
            est.predict_proba(X) for est in self.estimators_
        ])

        # Normalize probabilities to ensure they sum to 1
        probas_sum = np.sum(probas, axis=1, keepdims=True)
        probas_sum = np.where(probas_sum == 0, 1, probas_sum)  # avoid division by zero

        return probas / probas_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


# ONE-VS-ONE (OvO) CLASSIFIER
class OneVsOneClassifier:
    """
    One-vs-One (OvO) strategy for multiclass classification.
    This meta-estimator trains one classifier per pair of classes.
    Final prediction is based on majority voting.
    """

    def __init__(self, estimator_cls: Type, **kwargs):
        self.estimator_cls = estimator_cls
        self.kwargs = kwargs
        self.estimators_: List = []
        self.class_pairs_: List[Tuple[int, int]] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train one classifier for each pair of classes.
        """
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.class_pairs_ = []
        n_classes = len(self.classes_)

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                c1, c2 = self.classes_[i], self.classes_[j]
                # Filter dataset for the current pair
                mask = (y == c1) | (y == c2)
                X_pair = X[mask]
                y_pair = y[mask]
                # Binary labels: c1 -> 1, c2 -> 0
                y_binary = (y_pair == c1).astype(int)
                estimator = self.estimator_cls(**self.kwargs)
                estimator.fit(X_pair, y_binary)
                self.estimators_.append(estimator)
                self.class_pairs_.append((c1, c2))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        votes = np.zeros((n_samples, n_classes))
        class_to_index = {c: idx for idx, c in enumerate(self.classes_)}
        for estimator, (c1, c2) in zip(self.estimators_, self.class_pairs_):
            predictions = estimator.predict(X)  # 1 => c1, 0 => c2
            idx_c1 = class_to_index[c1]
            idx_c2 = class_to_index[c2]
            # Accumulate votes
            votes[np.arange(n_samples), idx_c1] += predictions
            votes[np.arange(n_samples), idx_c2] += (1 - predictions)
        return self.classes_[np.argmax(votes, axis=1)]    
    
# =============================================================================
# DISCRIMINANT ANALYSIS MODULE
# =============================================================================
class BaseDiscriminantAnalysis:
    """
    Base class providing shared utilities for discriminant analysis models.
    """

    def fisher_ratio_per_feature(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Compute Fisher ratio for all features using vectorization.

        This implementation reduces computational complexity from O(D * N)
        to approximately O(N) by leveraging NumPy matrix operations.
        """
        n_features = X.shape[1]
        features = np.arange(n_features)
        overall_mean = np.mean(X, axis=0)
        between_var = np.zeros(n_features)
        within_var = np.zeros(n_features)
        classes = np.unique(y)

        for cls in classes:
            Xk = X[y == cls]
            Nk = Xk.shape[0]
            mean_k = np.mean(Xk, axis=0)

            # Between-class variance
            between_var += Nk * (mean_k - overall_mean) ** 2

            # Within-class variance
            within_var += np.sum((Xk - mean_k) ** 2, axis=0)

        # Safe division to avoid division-by-zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(within_var == 0, np.inf, between_var / within_var)
        df = pd.DataFrame({
            "feature": features,
            "fisher_ratio": ratios
        })
        df["rank"] = df["fisher_ratio"].rank(ascending=False).astype(int)
        return df.sort_values("fisher_ratio", ascending=False).reset_index(drop=True)


# LINEAR DISCRIMINANT ANALYSIS (LDA)
class LinearDiscriminantAnalysis(BaseDiscriminantAnalysis):
    """
    Linear Discriminant Analysis (LDA).
    """

    def __init__(self, reg: float = 1e-6):
        self.reg = reg
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.scalings_: Optional[np.ndarray] = None  # Projection directions

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        N, D = X.shape
        self.means_ = np.zeros((K, D))
        self.priors_ = np.zeros(K)

        # Compute pooled covariance matrix
        scatter_within = np.zeros((D, D))
        for i, cls in enumerate(self.classes_):
            Xk = X[y == cls]
            Nk = Xk.shape[0]
            self.means_[i] = np.mean(Xk, axis=0)
            self.priors_[i] = Nk / N
            # Convert covariance to scatter matrix
            scatter_within += (Nk - 1) * np.cov(Xk, rowvar=False)
        self.cov_ = scatter_within / (N - K) + self.reg * np.eye(D)

        # Fisher projection: solve generalized eigenvalue problem
        S_W = scatter_within
        S_B = np.zeros((D, D))
        global_mean = np.mean(X, axis=0)
        for i in range(K):
            diff = (self.means_[i] - global_mean).reshape(-1, 1)
            S_B += (N * self.priors_[i]) * (diff @ diff.T)

        # Solve S_W^{-1} S_B
        inv_SW = np.linalg.pinv(S_W)
        M = inv_SW @ S_B
        eigvals, eigvecs = np.linalg.eig(M)

        # Sort eigenvectors by descending eigenvalues
        idx = np.argsort(eigvals.real)[::-1]
        self.scalings_ = eigvecs[:, idx].real

        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute linear discriminant scores.
        """
        inv_cov = np.linalg.pinv(self.cov_)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for k in range(n_classes):
            mu = self.means_[k]
            linear_term = X @ inv_cov @ mu
            constant_term = -0.5 * (mu.T @ inv_cov @ mu) + np.log(self.priors_[k])
            scores[:, k] = linear_term + constant_term
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._decision_function(X)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._decision_function(X), axis=1)]

    def transform(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        max_components = min(self.scalings_.shape[1], n_components)
        return X @ self.scalings_[:, :max_components]


# QUADRATIC DISCRIMINANT ANALYSIS (QDA)
class QuadraticDiscriminantAnalysis(BaseDiscriminantAnalysis):
    """
    Quadratic Discriminant Analysis (QDA).
    """

    def __init__(self, reg: float = 1e-6):
        self.reg = reg
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[List[np.ndarray]] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        N, D = X.shape
        self.means_ = np.zeros((K, D))
        self.priors_ = np.zeros(K)
        self.covariances_ = []

        for i, cls in enumerate(self.classes_):
            Xk = X[y == cls]
            self.means_[i] = np.mean(Xk, axis=0)
            self.priors_[i] = Xk.shape[0] / N
            # Class-specific covariance
            if Xk.shape[0] > 1:
                cov_k = np.cov(Xk, rowvar=False)
            else:
                cov_k = np.zeros((D, D))
            cov_k += self.reg * np.eye(D)
            self.covariances_.append(cov_k)
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute quadratic discriminant scores.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for k in range(n_classes):
            mu = self.means_[k]
            cov = self.covariances_[k]
            inv_cov = np.linalg.pinv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            logdet = sign * logdet
            diff = X - mu

            # Mahalanobis distance
            quadratic_term = np.sum((diff @ inv_cov) * diff, axis=1)
            scores[:, k] = -0.5 * (logdet + quadratic_term) + np.log(self.priors_[k])
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._decision_function(X)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._decision_function(X), axis=1)]

# =====================================================================
# ADVANCED & BONUS MODELS: KERNEL LOGISTIC REGRESSION, GNB, AND LDA
# =====================================================================

# Binary Probit Regression
class ProbitRegression:
    """
    Binary Probit Regression optimized via fully vectorized Gradient Descent.

    Model assumption:
        P(y = 1 | x) = Φ(w^T x)

    where Φ is the CDF of the standard normal distribution.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        verbose: bool = False,
        eps: float = 1e-8,
    ) -> None:
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.eps = eps

        self.w: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.time_history: List[float] = []

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Add intercept (bias term) to input matrix.
        """
        if self.fit_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X

    @staticmethod
    def _cdf(z: np.ndarray) -> np.ndarray:
        """Standard Normal CDF Φ(z)."""
        return norm.cdf(z)

    @staticmethod
    def _pdf(z: np.ndarray) -> np.ndarray:
        """Standard Normal PDF φ(z)."""
        return norm.pdf(z)

    def _compute_loss(self, y: np.ndarray, z: np.ndarray) -> float:
        """
        Compute negative log-likelihood (binary cross-entropy form).
        """
        p = np.clip(self._cdf(z), self.eps, 1.0 - self.eps)
        loss = -np.mean(
            y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
        )
        return float(loss)

    def _gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of negative log-likelihood (vectorized).
        """
        n_samples = X.shape[0]

        phi_z = self._pdf(z)
        Phi_z = np.clip(self._cdf(z), self.eps, 1.0 - self.eps)

        # Derivative of loss w.r.t z
        factor = (
            -(y * phi_z / Phi_z)
            + ((1.0 - y) * phi_z / (1.0 - Phi_z))
        )
        grad = (X.T @ factor) / n_samples
        return grad
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbitRegression":
        """
        Train the Probit Regression model using Gradient Descent.
        """
        X_aug = self._add_intercept(X)
        n_samples, n_features = X_aug.shape

        # Initialize weights
        self.w = np.zeros(n_features)
        self.loss_history = []
        self.time_history = []

        start_time = time.time()

        for i in range(self.max_iter):
            z = X_aug @ self.w
            loss = self._compute_loss(y, z)

            self.loss_history.append(loss)
            self.time_history.append(time.time() - start_time)

            # Convergence check
            if i > 0:
                if abs(self.loss_history[-2] - loss) < self.tol:
                    if self.verbose:
                        print(f"[Probit] Converged at iteration {i}")
                    break

            # Gradient update
            grad = self._gradient(X_aug, y, z)
            self.w -= self.lr * grad

        if self.verbose and len(self.loss_history) == self.max_iter:
            print(f"[Probit] Stopped at max_iter={self.max_iter} (no full convergence)")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability P(y = 1 | X).
        """
        if self.w is None:
            raise ValueError("Model must be fitted before calling predict_proba().")

        X_aug = self._add_intercept(X)
        z = X_aug @ self.w
        return self._cdf(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

"""
Noise Robustness Evaluation Module

This module provides utilities to:
- Inject label noise into training data
- Evaluate model robustness under different noise levels
"""
def inject_label_noise(
    y: np.ndarray,
    noise_rate: float,
    random_state: int = 42
) -> np.ndarray:
    """
    Inject symmetric label noise by flipping a proportion of labels.
    """
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError("noise_rate must be between 0 and 1.")
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_samples = len(y_noisy)
    n_flips = int(noise_rate * n_samples)

    if n_flips == 0:
        return y_noisy

    flip_indices = rng.choice(n_samples, n_flips, replace=False)
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

    return y_noisy

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    """
    return np.mean(y_true == y_pred)

def evaluate_noise_robustness(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_rates: List[float],
    logistic_model_cls,
    probit_model_cls,
    logistic_params: Dict = None,
    probit_params: Dict = None,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Evaluate model performance under increasing label noise.
    """

    logistic_params = logistic_params or {}
    probit_params = probit_params or {}
    acc_logit = []
    acc_probit = []

    if verbose:
        print("Starting noise robustness evaluation...\n")

    for noise in noise_rates:
        if verbose:
            print(f"[INFO] Noise level: {int(noise * 100):02d}%")

        # Inject noise
        y_train_noisy = inject_label_noise(
            y_train,
            noise_rate=noise,
            random_state=random_state
        )

        # Train Logistic Regression
        logit_model = logistic_model_cls(**logistic_params)
        logit_model.fit(X_train, y_train_noisy)
        y_pred_logit = logit_model.predict(X_test)
        acc_logit.append(compute_accuracy(y_test, y_pred_logit))

        # Train Probit Regression
        probit_model = probit_model_cls(**probit_params)
        probit_model.fit(X_train, y_train_noisy)
        y_pred_probit = probit_model.predict(X_test)
        acc_probit.append(compute_accuracy(y_test, y_pred_probit))

    if verbose:
        print("\nEvaluation completed.")

    return acc_logit, acc_probit


def rbf_kernel(X1, X2, gamma=1.0):
    """
    Computes the Radial Basis Function (RBF) kernel matrix between two datasets.
    K(x, x') = exp(-gamma * ||x - x'||^2)
    """
    # Calculate squared Euclidean distance using vectorized operations
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

class KernelLogisticRegression:
    """
    Kernel Logistic Regression using the Dual Formulation.
    Maps data into a higher-dimensional space using the Kernel Trick 
    to solve non-linearly separable problems (e.g., XOR patterns).
    """
    def __init__(self, gamma=1.0, lambda_reg=0.01, learning_rate=0.1, max_iter=1000, tol=1e-5):
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.X_train = None

    def _sigmoid(self, z):
        # Clip z to prevent overflow in exp
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        
        # Precompute the Kernel matrix for the training data
        K = rbf_kernel(X, X, self.gamma)
        
        # Initialize the dual coefficients (alpha)
        self.alpha = np.zeros(n_samples)

        for epoch in range(self.max_iter):
            # Linear output in the feature space: z = K * alpha
            z = K @ self.alpha
            y_pred = self._sigmoid(z)

            # Gradient of the log-loss with respect to alpha
            # (Simplified by dropping K since we optimize alpha directly)
            grad_alpha = (y_pred - y) + self.lambda_reg * self.alpha

            # Update alpha using Gradient Descent
            self.alpha -= self.learning_rate * grad_alpha

            # Early stopping condition
            if np.linalg.norm(grad_alpha) < self.tol:
                break

    def predict_proba(self, X):
        # Compute Kernel matrix between test data and training data
        K_test = rbf_kernel(X, self.X_train, self.gamma)
        z = K_test @ self.alpha
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.
    Assumes that continuous features follow a Gaussian distribution 
    and are mutually independent (diagonal covariance matrix).
    """
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = X.shape[0]

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            # Add a small epsilon to variance to prevent division by zero
            self.vars[c] = np.var(X_c, axis=0) + 1e-9

    def _calculate_log_likelihood(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        # Use log-likelihood to prevent underflow issues with tiny probabilities
        log_numerator = - ((x - mean)**2) / (2 * var)
        log_denominator = 0.5 * np.log(2 * np.pi * var)
        return log_numerator - log_denominator

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                # Due to the independence assumption, the joint log-likelihood 
                # is the sum of individual feature log-likelihoods
                likelihood = np.sum(self._calculate_log_likelihood(c, x))
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)


# class LinearDiscriminantAnalysis:
#     """
#     Linear Discriminant Analysis (LDA) classifier.
#     A generative model that assumes all classes share the same covariance matrix.
#     """
#     def __init__(self):
#         self.classes = None
#         self.priors = {}
#         self.means = {}
#         self.covariance = None
#         self.inv_covariance = None

#     def fit(self, X, y):
#         self.classes = np.unique(y)
#         n_samples, n_features = X.shape
        
#         self.covariance = np.zeros((n_features, n_features))
        
#         for c in self.classes:
#             X_c = X[y == c]
#             self.priors[c] = X_c.shape[0] / n_samples
#             self.means[c] = np.mean(X_c, axis=0)
            
#             # Compute within-class scatter matrix
#             centered_X_c = X_c - self.means[c]
#             self.covariance += centered_X_c.T @ centered_X_c
            
#         # Share the covariance matrix across all classes
#         self.covariance /= n_samples
#         # Add a small epsilon to the diagonal to ensure invertibility
#         self.covariance += np.eye(n_features) * 1e-9
#         self.inv_covariance = np.linalg.inv(self.covariance)

#     def predict(self, X):
#         y_pred = []
#         for x in X:
#             discriminants = []
#             for c in self.classes:
#                 mean = self.means[c]
#                 prior = self.priors[c]
#                 # Linear discriminant function (log-posterior odds)
#                 d_k = x.T @ self.inv_covariance @ mean - 0.5 * mean.T @ self.inv_covariance @ mean + np.log(prior)
#                 discriminants.append(d_k)
#             y_pred.append(self.classes[np.argmax(discriminants)])
#         return np.array(y_pred)

