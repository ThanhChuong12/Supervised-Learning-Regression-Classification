"""
Script to append Model Evaluation section cells to 02_modeling.ipynb
"""
import json, os

notebook_path = os.path.join(os.path.dirname(__file__), '02_modeling.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

# ---------- Cell 1: Section Header ----------
new_cells.append({
    "cell_type": "markdown",
    "id": "eval_header_01",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "# **MODEL EVALUATION**\n",
        "\n",
        "This section provides a comprehensive evaluation of all regression models trained in this notebook.\n",
        "\n",
        "**Evaluation Metrics:**\n",
        "\n",
        "| Metric | Formula | Description |\n",
        "|--------|---------|-------------|\n",
        "| **MSE** | $\\frac{1}{N}\\sum(t_n - y_n)^2$ | Mean Squared Error |\n",
        "| **RMSE** | $\\sqrt{\\text{MSE}}$ | Root Mean Squared Error |\n",
        "| **MAE** | $\\frac{1}{N}\\sum|t_n - y_n|$ | Mean Absolute Error |\n",
        "| **R\u00b2** | $1 - \\frac{SS_{res}}{SS_{tot}}$ | Coefficient of Determination |\n",
        "\n",
        "**Evaluation Tasks:**\n",
        "1. Learning Curves (Train Loss vs Validation Loss by number of training samples)\n",
        "2. Residual Analysis (checking randomness of errors)\n",
        "3. Predicted vs Actual Comparison\n",
        "4. Unified Model Comparison Table\n",
        "5. K-Fold Cross-Validation (k=10, Time-Series Expanding Window)\n",
        "6. Statistical Significance Tests (Paired t-test / Wilcoxon signed-rank test)"
    ]
})

# ---------- Cell 2: Imports ----------
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "eval_imports_02",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Import evaluation utility functions from utils\n",
        "from utils import (\n",
        "    compute_learning_curve, plot_learning_curves,\n",
        "    plot_residuals, plot_predicted_vs_actual,\n",
        "    build_model_comparison_table,\n",
        "    kfold_cross_validation_ts, statistical_test_models\n",
        ")"
    ]
})

# ---------- Cell 3: Learning Curves header ----------
new_cells.append({
    "cell_type": "markdown",
    "id": "eval_lc_header_03",
    "metadata": {},
    "source": [
        "## **11.1 Learning Curves**\n",
        "\n",
        "Learning curves show how **train loss** and **validation loss** evolve as the number of training samples increases. This helps diagnose:\n",
        "- **High Bias (underfitting):** Both curves converge to a high error.\n",
        "- **High Variance (overfitting):** Large gap between train and validation error."
    ]
})

# ---------- Cell 4: Learning Curves code ----------
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "eval_lc_code_04",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Learning Curves for OLS (on full 40 features) ---\n",
        "fit_ols_fn = lambda Phi, y: fit_ols(Phi, y, bias_is_first=True)\n",
        "sizes_ols, train_loss_ols, val_loss_ols = compute_learning_curve(\n",
        "    Phi_train, y_train, Phi_val, y_val, fit_fn=fit_ols_fn, n_points=15\n",
        ")\n",
        "plot_learning_curves(sizes_ols, train_loss_ols, val_loss_ols,\n",
        "                     title='Learning Curves \u2014 OLS (Normal Equations)')\n",
        "\n",
        "# --- Learning Curves for Ridge (on selected 24 features) ---\n",
        "fit_ridge_fn = lambda Phi, y: fit_ridge(Phi, y, lam=best_lam_ridge, bias_is_first=True)\n",
        "sizes_r, train_loss_r, val_loss_r = compute_learning_curve(\n",
        "    Phi_train_sel, y_train, Phi_val[:, selected_indices], y_val,\n",
        "    fit_fn=fit_ridge_fn, n_points=15\n",
        ")\n",
        "plot_learning_curves(sizes_r, train_loss_r, val_loss_r,\n",
        "                     title='Learning Curves \u2014 Ridge Regression (24 features)')"
    ]
})

# ---------- Cell 5: Residual Analysis header ----------
new_cells.append({
    "cell_type": "markdown",
    "id": "eval_res_header_05",
    "metadata": {},
    "source": [
        "## **11.2 Residual Analysis**\n",
        "\n",
        "A residual plot shows **Predicted Values** on the x-axis and **Residuals (y_true \u2212 y_pred)** on the y-axis.\n",
        "\n",
        "**What to look for:**\n",
        "- Residuals should be **randomly scattered** around the horizontal line y = 0.\n",
        "- Systematic patterns (funnel shape, curves) indicate violations of model assumptions."
    ]
})

# ---------- Cell 6: Residual Analysis code ----------
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "eval_res_code_06",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Residual plots for key models on the TEST set\n",
        "models_for_residuals = {\n",
        "    'OLS (Normal Eq.)': (Phi_test, w_ols),\n",
        "    'Ridge (24 feat.)': (Phi_test_sel, w_ridge_final),\n",
        "    'Lasso (24 feat.)': (Phi_test_sel, w_lasso_final),\n",
        "}\n",
        "\n",
        "for name, (Phi, w) in models_for_residuals.items():\n",
        "    y_pred = predict(Phi, w)\n",
        "    plot_residuals(y_test, y_pred, title=f'Residuals \u2014 {name}')"
    ]
})

# ---------- Cell 7: Predicted vs Actual header ----------
new_cells.append({
    "cell_type": "markdown",
    "id": "eval_pva_header_07",
    "metadata": {},
    "source": [
        "## **11.3 Predicted vs. Actual**\n",
        "\n",
        "This plot compares actual target values (x-axis) to predicted values (y-axis).  \n",
        "A perfect model would place all points exactly on the diagonal **y = x** line."
    ]
})

# ---------- Cell 8: Predicted vs Actual code ----------
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "eval_pva_code_08",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Predicted vs Actual for key models on the TEST set\n",
        "for name, (Phi, w) in models_for_residuals.items():\n",
        "    y_pred = predict(Phi, w)\n",
        "    plot_predicted_vs_actual(y_test, y_pred, title=f'Predicted vs Actual \u2014 {name}')"
    ]
})

# ---------- Cell 9: Model Comparison Table header ----------
new_cells.append({
    "cell_type": "markdown",
    "id": "eval_table_header_09",
    "metadata": {},
    "source": [
        "## **11.4 Unified Model Comparison Table**\n",
        "\n",
        "Compare all models in a single summary table using **MSE, RMSE, MAE, R\u00b2** on the **test set**."
    ]
})

# ---------- Cell 10: Model Comparison Table code ----------
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "eval_table_code_10",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Collect test-set metrics for all models\n",
        "all_model_results = {}\n",
        "\n",
        "# 1. OLS (40 features)\n",
        "all_model_results['OLS (Normal Eq.)'] = metrics(y_test, predict(Phi_test, w_ols))\n",
        "\n",
        "# 2. Ridge (24 selected features)\n",
        "all_model_results['Ridge (24 feat.)'] = metrics(y_test, predict(Phi_test_sel, w_ridge_final))\n",
        "\n",
        "# 3. Lasso (24 selected features)\n",
        "all_model_results['Lasso (24 feat.)'] = metrics(y_test, predict(Phi_test_sel, w_lasso_final))\n",
        "\n",
        "# 4. Elastic Net (24 selected features)\n",
        "all_model_results['Elastic Net (24 feat.)'] = metrics(y_test, predict(Phi_test_sel, w_enet_final))\n",
        "\n",
        "# 5. Mini-batch GD - Step Decay (40 features)\n",
        "all_model_results['Mini-batch GD (Step Decay)'] = metrics(y_test, predict(Phi_test, w_gd_step))\n",
        "\n",
        "# 6. Mini-batch GD - Cosine Annealing (40 features)\n",
        "all_model_results['Mini-batch GD (Cosine)'] = metrics(y_test, predict(Phi_test, w_gd_cosine))\n",
        "\n",
        "# 7. WLS (40 features) \u2014 if trained in Gauss-Markov / WLS section\n",
        "if 'w_wls' in dir():\n",
        "    all_model_results['WLS'] = metrics(y_test, predict(Phi_test, w_wls))\n",
        "\n",
        "print('\\nModel Comparison on Test Set:')\n",
        "build_model_comparison_table(all_model_results)"
    ]
})

# ---------- Cell 11: K-Fold CV header ----------
new_cells.append({
    "cell_type": "markdown",
    "id": "eval_cv_header_11",
    "metadata": {},
    "source": [
        "## **11.5 K-Fold Cross-Validation (k = 10)**\n",
        "\n",
        "To obtain a more robust estimate of model performance, we perform **10-fold Time-Series Cross-Validation** (Expanding Window) and report **mean \u00b1 std** for each metric.\n",
        "\n",
        "This method respects the temporal ordering of the data \u2014 the training window expands forward in time with each fold, and the validation fold always follows immediately after."
    ]
})

# ---------- Cell 12: K-Fold CV code ----------
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "eval_cv_code_12",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Define fit functions for each model\n",
        "cv_models = {\n",
        "    'OLS': lambda Phi, y: fit_ols(Phi, y, bias_is_first=True),\n",
        "    'Ridge': lambda Phi, y: fit_ridge(Phi, y, lam=best_lam_ridge, bias_is_first=True),\n",
        "    'Lasso': lambda Phi, y: fit_lasso_cd(Phi, y, lam=best_lam_lasso, num_iters=1000, bias_is_first=True),\n",
        "    'Elastic Net': lambda Phi, y: fit_elastic_net_cd(Phi, y, lam1=best_l1, lam2=best_l2, num_iters=1000, bias_is_first=True),\n",
        "}\n",
        "\n",
        "cv_results = {}  # {model_name: list of fold metric dicts}\n",
        "k_folds_eval = 10\n",
        "\n",
        "print(f'Running {k_folds_eval}-Fold Time-Series Cross-Validation ...\\n')\n",
        "\n",
        "for model_name, fit_fn in cv_models.items():\n",
        "    print(f'  Evaluating: {model_name} ...', end='')\n",
        "    fold_metrics = kfold_cross_validation_ts(Phi_train, y_train, fit_fn=fit_fn, k=k_folds_eval)\n",
        "    cv_results[model_name] = fold_metrics\n",
        "    print(' Done.')\n",
        "\n",
        "print()\n",
        "\n",
        "# Report mean \u00b1 std\n",
        "header = f\"{'Model':<20} {'MSE':>18} {'RMSE':>18} {'MAE':>18} {'R\u00b2':>18}\"\n",
        "sep = '=' * len(header)\n",
        "print(sep)\n",
        "print(header)\n",
        "print(sep)\n",
        "for model_name, folds in cv_results.items():\n",
        "    mse_vals = np.array([f['MSE'] for f in folds])\n",
        "    rmse_vals = np.array([f['RMSE'] for f in folds])\n",
        "    mae_vals = np.array([f['MAE'] for f in folds])\n",
        "    r2_vals = np.array([f['R2'] for f in folds])\n",
        "    print(f\"{model_name:<20} \"\n",
        "          f\"{mse_vals.mean():>8.2f}\u00b1{mse_vals.std():>6.2f} \"\n",
        "          f\"{rmse_vals.mean():>8.2f}\u00b1{rmse_vals.std():>6.2f} \"\n",
        "          f\"{mae_vals.mean():>8.2f}\u00b1{mae_vals.std():>6.2f} \"\n",
        "          f\"{r2_vals.mean():>8.4f}\u00b1{r2_vals.std():>6.4f}\")\n",
        "print(sep)"
    ]
})

# ---------- Cell 13: Statistical Tests header ----------
new_cells.append({
    "cell_type": "markdown",
    "id": "eval_stat_header_13",
    "metadata": {},
    "source": [
        "## **11.6 Statistical Significance Tests**\n",
        "\n",
        "To determine whether the performance differences between models are **statistically significant**, we apply:\n",
        "- **Paired t-test:** Assumes normally distributed differences.\n",
        "- **Wilcoxon signed-rank test:** Non-parametric alternative, does not require normality.\n",
        "\n",
        "We compare each model pair using the **per-fold MSE** scores from cross-validation. A p-value < 0.05 indicates a statistically significant difference."
    ]
})

# ---------- Cell 14: Statistical Tests code ----------
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "eval_stat_code_14",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Extract per-fold MSE for each model\n",
        "cv_mse_scores = {}\n",
        "for name, folds in cv_results.items():\n",
        "    cv_mse_scores[name] = [f['MSE'] for f in folds]\n",
        "\n",
        "model_names = list(cv_mse_scores.keys())\n",
        "\n",
        "print('Statistical Significance Tests (Paired t-test & Wilcoxon signed-rank)')\n",
        "print('Metric: MSE  |  Significance level: \u03b1 = 0.05')\n",
        "print('=' * 95)\n",
        "print(f'{\"Model A\":<20} {\"Model B\":<20} {\"Test\":<28} {\"p-value\":>10} {\"Significant?\":>14}')\n",
        "print('=' * 95)\n",
        "\n",
        "for i in range(len(model_names)):\n",
        "    for j in range(i + 1, len(model_names)):\n",
        "        name_a, name_b = model_names[i], model_names[j]\n",
        "        scores_a = cv_mse_scores[name_a]\n",
        "        scores_b = cv_mse_scores[name_b]\n",
        "\n",
        "        # Paired t-test\n",
        "        res_t = statistical_test_models(scores_a, scores_b, metric_name='MSE', test_type='ttest')\n",
        "        sig_t = 'YES' if res_t['significant'] else 'NO'\n",
        "        print(f\"{name_a:<20} {name_b:<20} {res_t['test']:<28} {res_t['p_value']:>10.6f} {sig_t:>14}\")\n",
        "\n",
        "        # Wilcoxon signed-rank test\n",
        "        try:\n",
        "            res_w = statistical_test_models(scores_a, scores_b, metric_name='MSE', test_type='wilcoxon')\n",
        "            sig_w = 'YES' if res_w['significant'] else 'NO'\n",
        "            print(f\"{'':20} {'':20} {res_w['test']:<28} {res_w['p_value']:>10.6f} {sig_w:>14}\")\n",
        "        except Exception as e:\n",
        "            print(f\"{'':20} {'':20} {'Wilcoxon: skipped':<28} {'N/A':>10} {'N/A':>14}\")\n",
        "        print('-' * 95)\n",
        "\n",
        "print()"
    ]
})

# ---------- Cell 15: Summary header ----------
new_cells.append({
    "cell_type": "markdown",
    "id": "eval_summary_15",
    "metadata": {},
    "source": [
        "## **11.7 Summary**\n",
        "\n",
        "**Key Findings:**\n",
        "\n",
        "1. **Learning Curves** reveal whether models suffer from high bias (underfitting) or high variance (overfitting). If both train and validation errors converge to a high value, the model is too simple.\n",
        "\n",
        "2. **Residual Plots** help verify the assumption that errors are random and unstructured. Patterns (e.g., funnel shapes) indicate heteroscedasticity, which was confirmed by the Breusch-Pagan test in Section 9.\n",
        "\n",
        "3. **Predicted vs. Actual** plots visually show how well predictions match reality. Points clustering near the diagonal line indicate good fit.\n",
        "\n",
        "4. **Model Comparison Table** provides side-by-side performance metrics (MSE, RMSE, MAE, R\u00b2) for all models on the test set.\n",
        "\n",
        "5. **K-Fold Cross-Validation (k=10)** gives robust performance estimates with uncertainty bounds (mean \u00b1 std), reducing the effect of any single train/test split.\n",
        "\n",
        "6. **Statistical Tests** (paired t-test & Wilcoxon) determine whether observed performance differences are statistically significant, not just due to random chance."
    ]
})

# Append all new cells
nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Successfully added {len(new_cells)} cells to {notebook_path}")
