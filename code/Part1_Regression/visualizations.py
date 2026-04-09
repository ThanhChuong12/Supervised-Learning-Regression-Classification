import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_lml_history(lml_history):
    """
    Parameters
    ----------
    lml_history : list[float]
        List of LML values recorded after each iteration.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(lml_history, color='red', linewidth=2, marker='o', markersize=4)
    plt.title("Log-Marginal-Likelihood Optimization (Gradient Ascent)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Log-Marginal-Likelihood", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_gp_posterior_predictive(X_te, y_test, mu_gp, std_gp, feature_names):
    """
    For each feature, three elements are drawn:
      - Actual data points (black dots)
      - GP predictive mean (blue line)
      - 95% confidence band: mean ± 2·std (shaded blue area)

    Parameters
    ----------
    X_te : np.ndarray, shape (N_test, n_features)
        Test feature matrix (without bias column).
    y_test : array-like, shape (N_test,)
        Ground-truth target values for the test set.
    mu_gp : np.ndarray, shape (N_test,)
        Predictive mean scaled back to the original target range.
    std_gp : np.ndarray, shape (N_test,)
        Predictive std scaled back to the original target range.
    feature_names : array-like of str
        Feature names in the same column order as X_te.
    """
    n_features = X_te.shape[1]
    cols = 3
    rows = math.ceil(n_features / cols)

    # Normalise y_test to a numpy array (supports both pandas Series and ndarray)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

    plt.figure(figsize=(20, rows * 5))

    for i in range(n_features):
        feature_name = feature_names[i]

        x_axis_values = X_te[:, i]
        sort_idx = np.argsort(x_axis_values)

        x_plot    = x_axis_values[sort_idx]
        y_plot    = y_test_np[sort_idx]
        mu_plot   = mu_gp[sort_idx]
        std_plot  = std_gp[sort_idx]

        plt.subplot(rows, cols, i + 1)

        # Actual data points
        plt.scatter(x_plot, y_plot,
                    color='black', alpha=0.3, s=5,
                    label='Actual Data (Black Dots)')

        # Predictive mean line
        plt.plot(x_plot, mu_plot,
                 color='blue', linewidth=1.5,
                 label='GP Mean (Blue Line)')

        # 95% confidence band
        plt.fill_between(x_plot,
                         mu_plot - 2 * std_plot,
                         mu_plot + 2 * std_plot,
                         color='blue', alpha=0.15,
                         label='95% Uncertainty (Shaded Area)')

        plt.title(f"Feature: {feature_name}", fontsize=10, fontweight='bold')
        plt.xlabel("Standardized Value", fontsize=9)
        plt.ylabel("Appliances (Wh)", fontsize=9)
        plt.legend(fontsize=7, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_regularization_path(lam_grid, ridge_paths, lasso_paths,
                              best_lam_ridge, best_lam_lasso):
    """
    Each path shows how every feature weight (excluding bias) changes as the
    regularization strength log10(λ) varies from strong (left) to weak (right).
    A vertical dashed line marks the CV-optimal λ for each model.
 
    Parameters
    ----------
    lam_grid : np.ndarray, shape (n_lambdas,)
        Array of λ values used during the grid search (descending order).
    ridge_paths : np.ndarray, shape (n_lambdas, n_features + 1)
        Ridge weight vectors for every λ. Column 0 is the bias (skipped).
    lasso_paths : np.ndarray, shape (n_lambdas, n_features + 1)
        Lasso weight vectors for every λ. Column 0 is the bias (skipped).
    best_lam_ridge : float
        CV-optimal λ for Ridge Regression.
    best_lam_lasso : float
        CV-optimal λ for Lasso Regression.
    """
    log_lam = np.log10(lam_grid)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
    # Ridge path — feature weights only (skip bias at index 0)
    axes[0].plot(log_lam, ridge_paths[:, 1:])
    axes[0].axvline(np.log10(best_lam_ridge), color='black', linestyle='--',
                    linewidth=2,
                    label=f'Best log(λ) = {np.log10(best_lam_ridge):.2f}')
    axes[0].invert_xaxis()
    axes[0].set_title('Ridge Regularization Path (L2 Shrinkage)', fontsize=14)
    axes[0].set_xlabel('log10(λ)', fontsize=12)
    axes[0].set_ylabel('Feature weight w', fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.7)
    axes[0].legend()
 
    # Lasso path — feature weights only (skip bias at index 0)
    axes[1].plot(log_lam, lasso_paths[:, 1:])
    axes[1].axvline(np.log10(best_lam_lasso), color='red', linestyle='--',
                    linewidth=2,
                    label=f'Best log(λ) = {np.log10(best_lam_lasso):.2f}')
    axes[1].invert_xaxis()
    axes[1].set_title('Lasso Path (L1 Feature Selection)', fontsize=14)
    axes[1].set_xlabel('log10(λ)', fontsize=12)
    axes[1].set_ylabel('Feature weight w', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    axes[1].legend()
 
    plt.tight_layout()
    plt.show()
 
 
def plot_elastic_net_contour(l1_vals, l2_vals, mse_matrix, best_l1, best_l2):
    """
    The heatmap shows validation MSE across the 2-D grid of (log10 L1, log10 L2)
    penalty values. The optimal combination is highlighted with a marker.
 
    Parameters
    ----------
    l1_vals : np.ndarray, shape (n_l1,)
        Grid of L1 (Lasso) penalty values.
    l2_vals : np.ndarray, shape (n_l2,)
        Grid of L2 (Ridge) penalty values.
    mse_matrix : np.ndarray, shape (n_l1, n_l2)
        Average cross-validation MSE for each (l1, l2) combination.
    best_l1 : float
        L1 value that achieved the lowest validation MSE.
    best_l2 : float
        L2 value that achieved the lowest validation MSE.
    """
    plt.figure(figsize=(11, 6))
 
    X, Y = np.meshgrid(np.log10(l2_vals), np.log10(l1_vals))
    cp = plt.contourf(X, Y, mse_matrix, levels=20, cmap='viridis_r')
    plt.colorbar(cp, label='Validation MSE')
 
    plt.plot(np.log10(best_l2), np.log10(best_l1),
             marker='o', color='red',
             markeredgecolor='white', markersize=12, markeredgewidth=2,
             label='Optimal point (Min MSE)')
 
    plt.title('Elastic Net Optimal Region Analysis (L1 vs L2)', fontsize=14)
    plt.xlabel('log10(Lambda 2 — Ridge Penalty)', fontsize=12)
    plt.ylabel('log10(Lambda 1 — Lasso Penalty)', fontsize=12)
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_curves(losses_step, losses_cosine):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses_step, label='Step Decay', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss (MSE)', fontsize=12)
    plt.title('Convergence: Step Decay Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.subplot(1, 2, 2)
    plt.plot(losses_cosine, label='Cosine Annealing', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss (MSE)', fontsize=12)
    plt.title('Convergence: Cosine Annealing Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

def plot_residual_and_qq(y_train_pred_ols, residuals_train, y_test_pred_ols, residuals_test):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Residual Plot - Train
    axes[0, 0].scatter(y_train_pred_ols, residuals_train, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
    axes[0, 0].set_ylabel('Residuals', fontsize=12)
    axes[0, 0].set_title('Residual Plot (Train Set)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Residual Plot - Test
    axes[0, 1].scatter(y_test_pred_ols, residuals_test, alpha=0.5, s=10, color='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Values', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residual Plot (Test Set)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # QQ-Plot - Train
    stats.probplot(residuals_train, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('QQ-Plot (Train Set)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # QQ-Plot - Test
    stats.probplot(residuals_test, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('QQ-Plot (Test Set)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test, pred_test):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, pred_test, s=10, alpha=0.35)
    mn = min(y_test.min(), pred_test.min())
    mx = max(y_test.max(), pred_test.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.xlabel('Actual Appliances')
    plt.ylabel('Predicted Appliances')
    plt.title('Test: Actual vs Predicted (Sigmoid basis)')
    plt.tight_layout()
    plt.show()

def plot_time_order_predictions(y_test, pred_test):
    plt.figure(figsize=(10, 4))
    idx = np.arange(len(y_test))
    cut = min(500, len(y_test))
    plt.plot(idx[:cut], y_test[:cut], label='Actual', linewidth=1)
    plt.plot(idx[:cut], pred_test[:cut], label='Predicted', linewidth=1)
    plt.title('Test (first window): Actual vs Predicted over time')
    plt.xlabel('Test index (time order)')
    plt.ylabel('Appliances')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_basis_validation_curves(poly_degrees, poly_val_mse, rbf_Ks, rbf_val_mse, sig_Ms, sig_val_mse, spline_knots, spline_val_mse):
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt.plot(poly_degrees, poly_val_mse, marker='o')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Validation MSE')
    plt.title('Validation curve: Polynomial')

    plt.subplot(1, 4, 2)
    plt.plot(rbf_Ks, rbf_val_mse, marker='o')
    plt.xlabel('Number of RBF centers (K)')
    plt.ylabel('Validation MSE')
    plt.title('Validation curve: Gaussian RBF')

    plt.subplot(1, 4, 3)
    plt.plot(sig_Ms, sig_val_mse, marker='o')
    plt.xlabel('Sigmoid bases per feature (M)')
    plt.ylabel('Validation MSE')
    plt.title('Validation curve: Sigmoid')

    plt.subplot(1, 4, 4)
    plt.plot(spline_knots, spline_val_mse, marker='o')
    plt.xlabel('Number of knots (n_knots)')
    plt.ylabel('Validation MSE')
    plt.title('Validation curve: Cubic Splines')

    plt.tight_layout()
    plt.show()


# Section 12.3 — Visualization: Learning Curves, Residual Plots, QQ-Plots 
def plot_section12_learning_curves(train_sizes, train_mse, val_mse):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train_sizes, train_mse, 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Train MSE')
    ax.plot(train_sizes, val_mse,  's-', color='tomato',
            linewidth=2, markersize=6, label='Validation MSE')

    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Learning Curves — OLS (Normal Equations)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_section12_residuals(y_pred_train, residuals_train,
                              y_pred_test,  residuals_test):
    def _add_bin_mean(ax, y_pred_flat, res_flat, n_bins=10):
        bin_edges = np.percentile(y_pred_flat, np.linspace(0, 100, n_bins + 1))
        centers, means = [], []
        for i in range(n_bins):
            mask = (y_pred_flat >= bin_edges[i]) & (y_pred_flat < bin_edges[i + 1])
            if mask.sum() > 0:
                centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                means.append(res_flat[mask].mean())
        ax.plot(centers, means, 'o--', color='orange',
                linewidth=2, markersize=7, label='Bin mean')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train set
    res_tr = residuals_train.ravel()
    pred_tr = y_pred_train.ravel()
    axes[0].scatter(pred_tr, res_tr, alpha=0.15, s=5, color='seagreen')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    _add_bin_mean(axes[0], pred_tr, res_tr)
    axes[0].set_xlabel('Predicted Values (ŷ)', fontsize=12)
    axes[0].set_ylabel('Residuals (y − ŷ)', fontsize=12)
    axes[0].set_title('Residual Plot — Train Set', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Test set
    res_te = residuals_test.ravel()
    pred_te = y_pred_test.ravel()
    axes[1].scatter(pred_te, res_te, alpha=0.25, s=8, color='steelblue')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    _add_bin_mean(axes[1], pred_te, res_te)
    axes[1].set_xlabel('Predicted Values (ŷ)', fontsize=12)
    axes[1].set_ylabel('Residuals (y − ŷ)', fontsize=12)
    axes[1].set_title('Residual Plot — Test Set', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Residuals vs. Predicted — OLS (Normal Equations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_section12_predicted_vs_actual(y_train, y_pred_train,
                                        y_test,  y_pred_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_true, y_pred_plot, label, color in [
        (axes[0], y_train.ravel(), y_pred_train.ravel(), 'Train Set', 'seagreen'),
        (axes[1], y_test.ravel(),  y_pred_test.ravel(),  'Test Set',  'steelblue'),
    ]:
        ax.scatter(y_true, y_pred_plot, alpha=0.25, s=8, color=color)
        lo = min(y_true.min(), y_pred_plot.min()) - 10
        hi = max(y_true.max(), y_pred_plot.max()) + 10
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = ŷ  (perfect)')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Actual Values (y)', fontsize=12)
        ax.set_ylabel('Predicted Values (ŷ)', fontsize=12)
        ax.set_title(f'Predicted vs. Actual — {label}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Predicted vs. Actual — OLS (Normal Equations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_section12_qq_histogram(residuals_train, stats_train,
                                 residuals_test,  stats_test):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pairs = [
        (residuals_train.ravel(), stats_train, 'Train Set', 'seagreen', 0),
        (residuals_test.ravel(),  stats_test,  'Test Set',  'steelblue', 1),
    ]

    for res_flat, stat, label, color, row in pairs:
        # QQ-Plot
        ax_qq = axes[row][0]
        (osm, osr), (slope, intercept, _) = stats.probplot(res_flat, dist='norm')
        ax_qq.scatter(osm, osr, alpha=0.3, s=5, color=color)
        xx = np.array([osm[0], osm[-1]])
        ax_qq.plot(xx, slope * xx + intercept, 'r-', linewidth=2, label='Normal line')
        ax_qq.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax_qq.set_ylabel('Sample Quantiles', fontsize=11)
        ax_qq.set_title(f'QQ-Plot of Residuals — {label}',
                        fontsize=12, fontweight='bold')
        ax_qq.legend(fontsize=10)
        ax_qq.grid(True, alpha=0.3)
        sw_p = stat['shapiro_pvalue']
        verdict = '✓ Normal (p>0.05)' if stat['is_normal'] else '✗ Non-normal (p≤0.05)'
        ax_qq.text(0.03, 0.95,
                   f"Shapiro-Wilk p = {sw_p:.4f}\n{verdict}",
                   transform=ax_qq.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Histogram
        ax_hist = axes[row][1]
        ax_hist.hist(res_flat, bins=60, color=color, alpha=0.7,
                     edgecolor='white', linewidth=0.4, density=True)
        x_norm = np.linspace(res_flat.min(), res_flat.max(), 300)
        from scipy.stats import norm as _norm
        y_norm = _norm.pdf(x_norm, stat['mean'], stat['std'])
        ax_hist.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal PDF')
        ax_hist.axvline(0, color='navy', linestyle='--', linewidth=1.2,
                        label='Zero line')
        ax_hist.set_xlabel('Residual Value', fontsize=11)
        ax_hist.set_ylabel('Density', fontsize=11)
        ax_hist.set_title(f'Residual Distribution — {label}',
                          fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=10)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.text(0.97, 0.95,
                     f"Skew = {stat['skewness']:.3f}\nKurt = {stat['kurtosis']:.3f}",
                     transform=ax_hist.transAxes, ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Residual Normality Diagnostics — OLS (Normal Equations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_section12_model_comparison(model_summary):
    names     = list(model_summary.keys())
    rmses     = [model_summary[n]['mean_rmse'] for n in names]
    rmse_stds = [model_summary[n]['std_rmse']  for n in names]
    r2s       = [model_summary[n]['mean_r2']   for n in names]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    bars = axes[0].bar(x, rmses, yerr=rmse_stds, capsize=4,
                       color='steelblue', alpha=0.8, edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=35, ha='right', fontsize=10)
    axes[0].set_ylabel('Mean CV RMSE', fontsize=12)
    axes[0].set_title('Model Comparison — Mean CV RMSE (± std)',
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    bars[int(np.argmin(rmses))].set_color('tomato')

    bars2 = axes[1].bar(x, r2s, color='seagreen', alpha=0.8, edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=35, ha='right', fontsize=10)
    axes[1].set_ylabel('Mean CV R²', fontsize=12)
    axes[1].set_title('Model Comparison — Mean CV R²',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)
    bars2[int(np.argmax(r2s))].set_color('gold')

    plt.tight_layout()
    plt.show()


def plot_section12_bias_variance(lambdas, bias_sq_list, variance_list, mse_list,
                                  best_lam=None):
    log_lam = np.log10(lambdas)

    plt.figure(figsize=(10, 5))
    plt.plot(log_lam, bias_sq_list,  'o-', color='tomato',    linewidth=2, label='Bias²')
    plt.plot(log_lam, variance_list, 's-', color='steelblue', linewidth=2, label='Variance')
    plt.plot(log_lam, mse_list,      '^-', color='slategray', linewidth=2, label='MSE (total)')

    if best_lam is not None:
        plt.axvline(np.log10(best_lam), color='black', linestyle='--',
                    linewidth=1.5, label=f'Best λ  (log₁₀ = {np.log10(best_lam):.2f})')

    plt.xlabel('log₁₀(λ)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Bias–Variance Tradeoff vs. Regularization Strength (Ridge)',
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_section12_runtime(timing_results):
    names = list(timing_results.keys())
    means = [timing_results[n]['mean_s'] for n in names]
    stds  = [timing_results[n]['std_s']  for n in names]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.8)))

    bars = ax.barh(y, means, xerr=stds, capsize=4,
                   color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Mean Training Time (seconds)', fontsize=12)
    ax.set_title('Computational Cost Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    max_val = max(means) if means else 1.0
    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}s', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
