import numpy as np

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