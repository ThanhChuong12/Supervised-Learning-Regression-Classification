import numpy as np
import time
from typing import Optional, List

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
    


# Advanced Logistic Regression with support for both Gradient Descent and Newton-Raphson optimization methods.
class BinaryLogisticRegression:
    """
    Binary Logistic Regression classifier from scratch.
    Supports Gradient Descent ('gd') and Newton-Raphson ('newton') optimization.
    """
    def __init__(self, method: str = 'gd', learning_rate: float = 0.01, 
                 max_iter: int = 1000, tol: float = 1e-5, 
                 fit_intercept: bool = True, verbose: bool = False):
        if method not in ['gd', 'newton']:
            raise ValueError("Method must be either 'gd' or 'newton'.")
        
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
        """Appends a column of ones to X for the bias term."""
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate((intercept, X), axis=1)
        return X

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the binary cross-entropy loss."""
        eps = 1e-15 # Prevent log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the model to the training data."""
        X_processed = self._add_intercept(X)
        N, d = X_processed.shape
        self.w = np.zeros(d)
        
        start_time = time.time()
        self.loss_history = []
        self.time_history = []

        for i in range(self.max_iter):
            # Forward pass
            z = X_processed @ self.w
            y_pred = self._sigmoid(z)
            
            # Record metrics
            current_loss = self._compute_loss(y, y_pred)
            self.loss_history.append(current_loss)
            self.time_history.append(time.time() - start_time)

            # Check convergence
            if i > 0 and abs(self.loss_history[-2] - current_loss) < self.tol:
                if self.verbose:
                    print(f"[{self.method.upper()}] Converged at epoch {i}")
                break

            # Backward pass & Update
            gradient = (X_processed.T @ (y_pred - y)) / N
            
            if self.method == 'gd':
                self.w -= self.lr * gradient
                
            elif self.method == 'newton':
                # Optimized Hessian computation using broadcasting to avoid NxN matrix
                R_diag = y_pred * (1 - y_pred)
                Hessian = (X_processed.T * R_diag) @ X_processed / N
                
                # Add L2 regularization jitter to Hessian for numerical stability (non-singular)
                Hessian += np.eye(d) * 1e-5 
                
                try:
                    delta = np.linalg.solve(Hessian, gradient)
                except np.linalg.LinAlgError:
                    delta = np.linalg.pinv(Hessian) @ gradient
                    
                self.w -= delta # Newton step size is exactly 1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_processed = self._add_intercept(X)
        return self._sigmoid(X_processed @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegression:
    """
    Multinomial Logistic Regression (Softmax) from scratch using Gradient Descent.
    """
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 tol: float = 1e-5, fit_intercept: bool = True, verbose: bool = False):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
        self.W: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.time_history: List[float] = []
        self.classes_: Optional[np.ndarray] = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate((intercept, X), axis=1)
        return X

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax transformation with max-subtraction for numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        
        X_processed = self._add_intercept(X)
        N, d = X_processed.shape
        self.W = np.zeros((d, K))
        
        # One-hot encoding y
        y_onehot = np.eye(K)[y]
        
        start_time = time.time()
        self.loss_history = []
        self.time_history = []

        for i in range(self.max_iter):
            # Forward pass
            scores = X_processed @ self.W
            probs = self._softmax(scores)
            
            # Compute categorical cross-entropy loss
            eps = 1e-15
            current_loss = -np.mean(np.sum(y_onehot * np.log(np.clip(probs, eps, 1.)), axis=1))
            self.loss_history.append(current_loss)
            self.time_history.append(time.time() - start_time)

            if i > 0 and abs(self.loss_history[-2] - current_loss) < self.tol:
                if self.verbose:
                    print(f"[SOFTMAX] Converged at epoch {i}")
                break
                
            # Compute gradient and update
            gradient = (X_processed.T @ (probs - y_onehot)) / N
            self.W -= self.lr * gradient

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_processed = self._add_intercept(X)
        return self._softmax(X_processed @ self.W)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class OneVsRestClassifier:
    """Meta-estimator to transform a binary classifier into a multiclass one using OvR strategy."""
    def __init__(self, estimator_cls, **kwargs):
        self.estimator_cls = estimator_cls
        self.kwargs = kwargs
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.estimators_ = []
        
        for c in self.classes_:
            y_binary = (y == c).astype(int)
            estimator = self.estimator_cls(**self.kwargs)
            estimator.fit(X, y_binary)
            self.estimators_.append(estimator)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns normalized probabilities for each class."""
        # Get probability of positive class (1) for each estimator
        probas = np.array([est.predict_proba(X) for est in self.estimators_]).T
        # Normalize so they sum to 1
        return probas / np.sum(probas, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class OneVsOneClassifier:
    """Meta-estimator to transform a binary classifier into a multiclass one using OvO strategy."""
    def __init__(self, estimator_cls, **kwargs):
        self.estimator_cls = estimator_cls
        self.kwargs = kwargs
        self.estimators_ = []
        self.class_pairs_ = []
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.class_pairs_ = []
        
        n_classes = len(self.classes_)
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                c1, c2 = self.classes_[i], self.classes_[j]
                
                # Filter data for the two classes
                mask = (y == c1) | (y == c2)
                X_pair, y_pair = X[mask], y[mask]
                
                # Binarize labels: c1 -> 1, c2 -> 0
                y_binary = (y_pair == c1).astype(int)
                
                estimator = self.estimator_cls(**self.kwargs)
                estimator.fit(X_pair, y_binary)
                
                self.estimators_.append(estimator)
                self.class_pairs_.append((c1, c2))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        votes = np.zeros((N, len(self.classes_)))
        
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}
        
        for est, (c1, c2) in zip(self.estimators_, self.class_pairs_):
            pred_is_c1 = est.predict(X) # 1 if c1, 0 if c2
            
            # Add votes
            votes[np.arange(N), class_to_idx[c1]] += pred_is_c1
            votes[np.arange(N), class_to_idx[c2]] += (1 - pred_is_c1)
            
        return self.classes_[np.argmax(votes, axis=1)]