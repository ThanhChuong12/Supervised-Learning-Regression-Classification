import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from typing import Tuple, Optional, List
from sklearn.decomposition import PCA
from itertools import cycle

from typing import Tuple, List, Optional
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize


def plot_convergence_comparison(
    gd_model, 
    newton_model, 
    figsize: tuple = (16, 6), 
    dpi: int = 120
) -> None:
    """
    Visualize and compare the convergence speeds between Gradient Descent and Newton-Raphson.
    """
    # VISUALIZATION CONFIGURATION
    plt.style.use('seaborn-v0_8-whitegrid')
    COLOR_GD = '#0072B2'       # Blue
    COLOR_NEWTON = '#D55E00'   # Dark orange
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # PLOT 1: LOSS VS. EPOCHS (WITH ZOOM-IN INSET)
    ax1.plot(gd_model.loss_history, color=COLOR_GD, linewidth=2.5)
    ax1.plot(newton_model.loss_history, color=COLOR_NEWTON, linewidth=2.5, marker='o', markersize=6)

    # Direct Labeling
    ax1.text(len(gd_model.loss_history), gd_model.loss_history[-1], ' Gradient Descent', 
             color=COLOR_GD, fontsize=12, fontweight='bold', va='center')
    ax1.text(len(newton_model.loss_history), newton_model.loss_history[-1], ' Newton-Raphson', 
             color=COLOR_NEWTON, fontsize=12, fontweight='bold', va='center')

    # Inset plot: zoom into early training phase
    axins = ax1.inset_axes([0.45, 0.35, 0.45, 0.5]) 
    axins.plot(gd_model.loss_history, color=COLOR_GD, linewidth=2)
    axins.plot(newton_model.loss_history, color=COLOR_NEWTON, linewidth=2, marker='o', markersize=6)

    # Focus on first 20 epochs
    axins.set_xlim(-0.5, 20)
    axins.set_ylim(bottom=-0.02, top=max(gd_model.loss_history[0], newton_model.loss_history[0]))
    axins.set_title("Zoom: First 20 Epochs", fontsize=10, color='dimgray')
    axins.tick_params(axis='both', labelsize=9)

    ax1.indicate_inset_zoom(axins, edgecolor="black", alpha=0.3)
    ax1.set_title("Convergence by Iterations\n(Loss vs. Epochs)", fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)

    # PLOT 2: LOSS VS. WALL-CLOCK TIME
    ax2.plot(gd_model.time_history, gd_model.loss_history, color=COLOR_GD, linewidth=2.5)
    ax2.plot(newton_model.time_history, newton_model.loss_history, color=COLOR_NEWTON, linewidth=2.5, marker='o', markersize=6)

    # Direct Labeling
    ax2.text(gd_model.time_history[-1], gd_model.loss_history[-1], ' Gradient Descent', 
             color=COLOR_GD, fontsize=12, fontweight='bold', va='center')
    ax2.text(newton_model.time_history[-1], newton_model.loss_history[-1], ' Newton-Raphson', 
             color=COLOR_NEWTON, fontsize=12, fontweight='bold', va='center')

    ax2.set_title("Convergence by Wall-Clock Time\n(Loss vs. Time)", fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)

    for ax in [ax1, ax2]:
        ax.margins(x=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('dimgray')
        ax.spines['left'].set_color('dimgray')
        
    plt.tight_layout()
    plt.show()

def plot_fisher_ratio_ranking(
    fisher_df: pd.DataFrame,
    top_k: int = 5,
    title: str = "Feature Importance Ranking (Fisher Ratio)",
    figsize: tuple = (12, 8),
    dpi: int = 120,
) -> None:
    """
    Plot feature importance ranking based on Fisher Ratio.
    """
    required_columns = {"fisher_ratio", "feature_name"}
    if not required_columns.issubset(fisher_df.columns):
        raise ValueError(
            f"Input DataFrame must contain columns: {required_columns}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    plot_df = (
        fisher_df
        .sort_values("fisher_ratio", ascending=False)
        .reset_index(drop=True)
        .copy()
    )

    highlight_label = f"Top {top_k} Features"
    plot_df["highlight"] = "Normal"
    plot_df.loc[: top_k - 1, "highlight"] = highlight_label

    # Reverse order for horizontal plotting (largest on top)
    plot_df = plot_df.iloc[::-1]

    # Plot configuration
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.barplot(
        data=plot_df,
        x="fisher_ratio",
        y="feature_name",
        hue="highlight",          # Required for palette
        dodge=False,              # Single bar per feature
        width=0.7,
        palette={
            "Normal": "#B0BEC5",          # neutral gray
            highlight_label: "#D32F2F",   # highlight color
        },
        edgecolor="none"
    )

    # xis and legend configuration
    ax.set_xlim(left=0)
    ax.legend(
        title="Feature Type",
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=10
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=16,
        color="#263238"
    )
    ax.set_xlabel(
        "Fisher Ratio $J(w)$",
        fontsize=12,
        fontweight="bold",
        labelpad=10
    )
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=11)

    # Grid and styling
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    plt.tight_layout()
    plt.show()

# LDA vs QDA decision boundaries in Fisher space
def plot_lda_qda_decision_analysis(
    lda_model,
    qda_model,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "LDA vs QDA Decision Boundary Analysis (Confidence & Misclassifications)",
    grid_resolution: int = 400,
    figsize: Tuple[int, int] = (18, 8),
    dpi: int = 120,
) -> None:
    """
    Visualize and compare LDA and QDA decision boundaries in 2D Fisher space.

    This function:
    - Projects data into 2D using LDA (Fisher Discriminant space)
    - Trains auxiliary LDA/QDA models in projected space
    - Plots:
        + Decision regions
        + Decision boundaries
        + Confidence contours
        + Correct vs incorrect classifications
    """

    X_proj = lda_model.transform(X, n_components=2)
    # Train auxiliary models in projected space
    lda_2d = type(lda_model)(reg=lda_model.reg).fit(X_proj, y)
    qda_2d = type(qda_model)(reg=qda_model.reg).fit(X_proj, y)
    models = [
        ("LDA (Linear Decision Boundary)", lda_2d),
        ("QDA (Nonlinear Decision Boundary)", qda_2d),
    ]

    x_min, x_max = X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1
    y_min, y_max = X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Define color palettes
    n_classes = len(lda_model.classes_)
    base_colors = ["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFFFAA"]
    point_colors = ["red", "green", "blue", "orange"]
    cmap_bg = mcolors.ListedColormap(base_colors[:n_classes])
    cmap_points = mcolors.ListedColormap(point_colors[:n_classes])

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    for ax, (model_name, model) in zip(axes, models):
        # Predictions on grid
        preds = model.predict(grid).reshape(xx.shape)
        probas = model.predict_proba(grid)
        confidence = np.max(probas, axis=1).reshape(xx.shape)

        # Decision regions and boundaries
        ax.contourf(xx, yy, preds, alpha=0.2, cmap=cmap_bg)
        ax.contour(xx, yy, preds, colors="black", linewidths=1.5)
        # Confidence contours
        contour_lines = ax.contour(
            xx,
            yy,
            confidence,
            levels=[0.55, 0.75, 0.95],
            linestyles=[":", "--", "-."],
            colors="dimgray",
            alpha=0.6,
        )
        ax.clabel(contour_lines, inline=True, fontsize=9, fmt="%.2f")

        # Classification correctness
        y_pred = model.predict(X_proj)
        correct_mask = y == y_pred
        incorrect_mask = ~correct_mask
        # Correct predictions
        ax.scatter(
            X_proj[correct_mask, 0],
            X_proj[correct_mask, 1],
            c=y[correct_mask],
            cmap=cmap_points,
            edgecolors="k",
            s=35,
            alpha=0.6,
        )
        # Misclassified samples
        if np.any(incorrect_mask):
            ax.scatter(
                X_proj[incorrect_mask, 0],
                X_proj[incorrect_mask, 1],
                c=y[incorrect_mask],
                cmap=cmap_points,
                edgecolors="red",
                linewidths=1.5,
                s=100,
                marker="X",
                alpha=1.0,
            )

        legend_handles = []
        # Class colors
        for i, cls in enumerate(lda_model.classes_):
            legend_handles.append(
                mpatches.Patch(color=point_colors[i], label=f"Class {cls}")
            )
        # Separator
        legend_handles.append(mpatches.Patch(color="none", label="---"))

        # Marker meanings
        legend_handles.append(
            mlines.Line2D(
                [], [],
                color="gray",
                marker="o",
                linestyle="None",
                markersize=8,
                label="Correct Classification",
            )
        )
        if np.any(incorrect_mask):
            legend_handles.append(
                mlines.Line2D(
                    [], [],
                    color="red",
                    marker="X",
                    linestyle="None",
                    markersize=10,
                    label="Misclassification",
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            framealpha=0.9,
            fontsize=10,
        )

        # Labels and styling
        ax.set_title(model_name, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Linear Discriminant 1 (LD1)", fontsize=12, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Linear Discriminant 2 (LD2)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_perceptron_ovo_convergence_curves(perceptron_ovo):
    # Plot the convergence curve for each binary classifier using subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Loop through each model to plot their specific error history on a separate subplot
    for idx, estimator in enumerate(perceptron_ovo.estimators_):
        errors = estimator.errors_history
        c1, c2 = perceptron_ovo.class_pairs_[idx]
        
        axes[idx].plot(range(1, len(errors) + 1), errors, color='crimson', linewidth=1.5, alpha=0.9)
        axes[idx].set_title(f'Class {c1} vs Class {c2}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epochs', fontsize=10)
        axes[idx].set_ylabel('Misclassified Samples', fontsize=10)
        axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.suptitle('Perceptron Convergence Curves (OvO Strategy - 6 Pairs)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_multiclass_logistic_regression_ovr_diagnostics(best_lr_model, y_test, y_pred_test_lr):
    # Visualizations: Loss Curve and Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the Loss Curve for each of the 4 internal models
    for idx, estimator in enumerate(best_lr_model.estimators_):
        axes[0].plot(range(1, len(estimator.loss_history) + 1), estimator.loss_history, 
                     linewidth=2, label=f'Class {best_lr_model.classes_[idx]} vs Rest')

    axes[0].set_title('Logistic Regression: Loss Curve (OvR)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Weighted Cross-Entropy Loss', fontsize=12)
    axes[0].legend()

    # Plot the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test_lr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
                xticklabels=best_lr_model.classes_, 
                yticklabels=best_lr_model.classes_,
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title('Confusion Matrix on Test Set', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_models_confusion_matrices_grid(models_dict, X_test, y_test, classes):
    """
    Visualizes the Confusion Matrix for each evaluated model using Seaborn heatmaps.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=classes, yticklabels=classes,
                    annot_kws={"size": 13, "weight": "bold"})
        
        axes[idx].set_title(name, fontsize=13, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
        axes[idx].set_ylabel('True Label', fontsize=11)
        
    plt.tight_layout()
    plt.show()


def plot_logistic_regression_ovr_loss_convergence_curve(ovr):
    plt.figure(figsize=(10, 6))

    # Extract the estimators from the Baseline OvR model (custom implementation)
    # We use the baseline model here as it purely shows the optimization trajectory without penalties
    for idx, estimator in enumerate(ovr.estimators_):
        # Retrieve the loss history recorded during Gradient Descent / IRLS optimization
        if hasattr(estimator, 'loss_history') and len(estimator.loss_history) > 0:
            plt.plot(range(1, len(estimator.loss_history) + 1), estimator.loss_history, 
                     linewidth=2, label=f'Class {idx} vs Rest')

    plt.title('Loss Convergence Curve - Binary Logistic Regression (OvR Base)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Optimization Epochs / Iterations', fontsize=12)
    plt.ylabel('Log-Loss (Cross-Entropy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_multiclass_roc_pr_curves(models_dict, X_test, y_test, n_classes, class_labels):
    """
    Plots Multi-class ROC and Precision-Recall curves using the One-vs-Rest strategy.
    Only evaluates models that support predict_proba().
    """
    # Binarize the output labels for OvR evaluation
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue'])
    
    # Filter out models that do not support probability estimation (e.g., Perceptron)
    prob_models = {name: model for name, model in models_dict.items() if hasattr(model, 'predict_proba')}
    n_prob_models = len(prob_models)
    
    if n_prob_models == 0:
        print("No models support predict_proba. Cannot plot ROC/PR curves.")
        return

    fig, axes = plt.subplots(n_prob_models, 2, figsize=(16, 6 * n_prob_models))
    # Handle single model case
    if n_prob_models == 1:
        axes = np.array([axes])

    for row_idx, (name, model) in enumerate(prob_models.items()):
        # Predict probabilities
        y_score = model.predict_proba(X_test)
        
        # --- 1. Compute and Plot ROC Curve ---
        ax_roc = axes[row_idx, 0]
        # Compute Macro-average ROC curve and ROC area
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot macro average
        ax_roc.plot(fpr["macro"], tpr["macro"],
                 label=f'Macro-average ROC (area = {roc_auc["macro"]:.4f})',
                 color='navy', linestyle=':', linewidth=4)
        
        # Plot individual class ROC
        for i, color in zip(range(n_classes), colors):
            ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:.4f})')

        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title(f'ROC Curve: {name}', fontsize=14, fontweight='bold')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.3)

        # --- 2. Compute and Plot Precision-Recall Curve ---
        ax_pr = axes[row_idx, 1]
        precision, recall, average_precision = dict(), dict(), dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
            
        # Compute macro-average PR curve
        # (Macro-averaging PR curves is less standardized, so we report Micro-average for overall)
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")
        
        ax_pr.plot(recall["micro"], precision["micro"],
                 label=f'Micro-average PR (area = {average_precision["micro"]:.4f})',
                 color='navy', linestyle=':', linewidth=4)

        for i, color in zip(range(n_classes), colors):
            ax_pr.plot(recall[i], precision[i], color=color, lw=2,
                     label=f'PR curve of class {class_labels[i]} (area = {average_precision[i]:.4f})')

        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall', fontsize=12)
        ax_pr.set_ylabel('Precision', fontsize=12)
        ax_pr.set_title(f'Precision-Recall Curve: {name}', fontsize=14, fontweight='bold')
        ax_pr.legend(loc="lower left")
        ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    

# =====================================================================
# ADVANCED & BONUS MODELS
# =====================================================================

# Probability distribution comparison: Logistic vs Probit
def plot_probability_density_comparison(
    proba_logit: np.ndarray,
    proba_probit: np.ndarray,
    title: str = "Predicted Probability Distribution: Logistic vs Probit",
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 120,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot KDE (Kernel Density Estimation) of predicted probabilities
    for Logistic and Probit regression models.
    """

    proba_logit = np.asarray(proba_logit).ravel()
    proba_probit = np.asarray(proba_probit).ravel()
    if proba_logit.shape != proba_probit.shape:
        raise ValueError("Input arrays must have the same shape.")

    plt.figure(figsize=figsize, dpi=dpi)

    # KDE plots
    sns.kdeplot(
        proba_logit,
        fill=True,
        linewidth=2,
        label="Logistic Regression (Sigmoid)",
        color="#1f77b4",
        alpha=0.5,
    )

    sns.kdeplot(
        proba_probit,
        fill=True,
        linewidth=2,
        label="Probit Regression (Normal CDF)",
        color="#d62728",
        alpha=0.5,
    )

    plt.title(title, fontsize=15, fontweight="bold", pad=15)
    plt.xlabel(
        r"Predicted Probability $P(Y=1 \mid \mathbf{x})$",
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel(
        "Density",
        fontsize=12,
        fontweight="bold",
    )

    # Probability range
    plt.xlim(-0.05, 1.05)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc="upper center", fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

def compare_model_probability_distributions(
    logit_model,
    probit_model,
    X: np.ndarray,
    **kwargs,
) -> None:
    """
    Convenience wrapper to directly compare probability distributions
    from trained Logistic and Probit models.
    """
    proba_logit = logit_model.predict_proba(X)
    proba_probit = probit_model.predict_proba(X)

    plot_probability_density_comparison(
        proba_logit,
        proba_probit,
        **kwargs,
    )

# Logistic vs Probit Decision Boundary Comparison in PCA Space
def plot_logit_vs_probit_pca_boundary(
    X: np.ndarray,
    y: np.ndarray,
    logit_model_cls,
    probit_model_cls,
    title: str = "Decision Boundary Analysis: Logistic vs Probit (PCA Projection)",
    grid_resolution: int = 400,
    figsize: Tuple[int, int] = (14, 9),
    dpi: int = 120,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize and compare Logistic vs Probit decision boundaries
    in a 2D PCA-projected feature space.

    This function:
    - Reduces data to 2D using PCA
    - Trains Logistic and Probit models in PCA space
    - Plots:
        + Data distribution (faded background)
        + Decision boundaries (P = 0.5)
        + Confidence margins (P = 0.1 and P = 0.9)
    """

    # PCA projection (2D)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    logit_2d = logit_model_cls().fit(X_pca, y)
    probit_2d = probit_model_cls().fit(X_pca, y)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probas_logit = logit_2d.predict_proba(grid).reshape(xx.shape)
    probas_probit = probit_2d.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=figsize, dpi=dpi)
    cmap_points = mcolors.ListedColormap(["#00897b", "#fb8c00"])

    # Background scatter (faded)
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap=cmap_points,
        edgecolors="k",
        s=35,
        alpha=0.25,
    )

    # Logistic decision boundary
    plt.contour(
        xx,
        yy,
        probas_logit,
        levels=[0.5],
        colors="blue",
        linewidths=3.0,
        linestyles="-",
    )
    contour_logit = plt.contour(
        xx,
        yy,
        probas_logit,
        levels=[0.1, 0.9],
        colors="blue",
        linewidths=1.2,
        linestyles=":",
        alpha=0.7,
    )
    plt.clabel(contour_logit, inline=True, fontsize=9, fmt="P=%.1f")

    # Probit decision boundary
    plt.contour(
        xx,
        yy,
        probas_probit,
        levels=[0.5],
        colors="red",
        linewidths=3.0,
        linestyles="--",
    )

    contour_probit = plt.contour(
        xx,
        yy,
        probas_probit,
        levels=[0.1, 0.9],
        colors="red",
        linewidths=1.2,
        linestyles=":",
        alpha=0.7,
    )
    plt.clabel(contour_probit, inline=True, fontsize=9, fmt="P=%.1f")

    # Labels & styling
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Principal Component 1 (PC1)", fontsize=12, fontweight="bold")
    plt.ylabel("Principal Component 2 (PC2)", fontsize=12, fontweight="bold")
    legend_elements = [
        mlines.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="#00897b",
            markersize=10,
            alpha=0.5,
            label="Class 0",
        ),
        mlines.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="#fb8c00",
            markersize=10,
            alpha=0.5,
            label="Class 1",
        ),
        mlines.Line2D([0], [0], color="none", label="--- Decision Boundary (P=0.5) ---"),
        mlines.Line2D([0], [0], color="blue", lw=3, linestyle="-", label="Logistic"),
        mlines.Line2D([0], [0], color="red", lw=3, linestyle="--", label="Probit"),
        mlines.Line2D([0], [0], color="none", label="--- Confidence Margins ---"),
        mlines.Line2D([0], [0], color="blue", lw=1.2, linestyle=":", label="Logistic (0.1 / 0.9)"),
        mlines.Line2D([0], [0], color="red", lw=1.2, linestyle=":", label="Probit (0.1 / 0.9)"),
    ]

    plt.legend(
        handles=legend_elements,
        loc="best",
        framealpha=0.95,
        fontsize=10,
        edgecolor="gray",
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

def plot_noise_robustness(
    noise_rates: List[float],
    acc_logit: List[float],
    acc_probit: List[float],
    title: str = "Performance Degradation Under Label Noise"
) -> None:
    """
    Plot performance degradation curves for models.
    """

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(
        noise_rates,
        acc_logit,
        marker='o',
        markersize=7,
        linewidth=2.2,
        label='Logistic Regression',
    )
    plt.plot(
        noise_rates,
        acc_probit,
        marker='s',
        markersize=7,
        linewidth=2.2,
        linestyle='--',
        label='Probit Regression',
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Noise Rate (Proportion of Flipped Labels)", fontsize=11)
    plt.ylabel("Test Accuracy", fontsize=11)
    plt.xticks(noise_rates, [f"{int(n * 100)}%" for n in noise_rates])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(framealpha=0.9)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_covariance_heatmap(
    sigma: np.ndarray,
    feature_names: List[str],
    subset_size: int = 15,
    title: str = "Posterior Covariance Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "coolwarm"
) -> None:
    """
    Plot a heatmap of the posterior covariance matrix (subset).
    """

    if sigma is None:
        raise ValueError("Sigma (covariance matrix) must not be None.")

    if subset_size > sigma.shape[0]:
        raise ValueError("subset_size cannot exceed matrix dimensions.")

    # Extract submatrix
    sigma_subset = sigma[:subset_size, :subset_size]

    # Construct labels (bias + feature names)
    labels = ["Bias (w0)"] + feature_names[: subset_size - 1]

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        sigma_subset,
        annot=False,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Variance / Covariance"},
    )

    # Align ticks to cell centers
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)

    # Format labels
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)

    plt.title(title, fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.show()

def plot_bayesian_decision_boundary_with_uncertainty(
    X_2d: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    mu_map: np.ndarray,
    sigma_map: np.ndarray,
    mag_factor: float = 2.0,
) -> None:
    """
    Visualize Bayesian Logistic Regression decision boundary with uncertainty bands.
        - X_2d: 2D PCA-projected data points
        - y: class labels
    """

    plt.figure(figsize=(12, 8), dpi=120)

    # Scatter plot of data points
    plt.scatter(
        X_2d[y == 0, 0], X_2d[y == 0, 1], c='#00897b',
        edgecolor='k', s=40, alpha=0.5, label='Empty Room (0)'
    )
    plt.scatter(
        X_2d[y == 1, 0], X_2d[y == 1, 1], c='#fb8c00',
        edgecolor='k', s=40, alpha=0.5, label='Occupied (1)'
    )

    # Decision boundary (mu = 0)
    plt.contour(
        xx, yy, mu_map,
        levels=[0],
        linewidths=3.0,
        linestyles='solid'
    )

    # Uncertainty bands
    plt.contour(
        xx, yy, mu_map + mag_factor * sigma_map,
        levels=[0],
        colors='red',
        linewidths=2.0,
        linestyles='dashed'
    )

    plt.contour(
        xx, yy, mu_map - mag_factor * sigma_map,
        levels=[0],
        colors='blue',
        linewidths=2.0,
        linestyles='dashed'
    )

    # Custom legend
    legend_elements = [
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#00897b',
                      markersize=8, alpha=0.5, label='Empty Room (0)'),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#fb8c00',
                      markersize=8, alpha=0.5, label='Occupied (1)'),
        mlines.Line2D([], [], color='black', lw=3.0,
                      label='Decision Boundary ($\\mu_a = 0$)'),
        mlines.Line2D([], [], color='red', lw=2.0, linestyle='--',
                  label=f'Upper Boundary ($\\mu_a + {mag_factor}\\sigma_a = 0$)'),
        mlines.Line2D([], [], color='blue', lw=2.0, linestyle='--',
                  label=f'Lower Boundary ($\\mu_a - {mag_factor}\\sigma_a = 0$)')
    ]

    plt.legend(handles=legend_elements, loc='upper left', framealpha=0.95)
    plt.xlabel('Principal Component 1 (PC1)', fontweight='bold')
    plt.ylabel('Principal Component 2 (PC2)', fontweight='bold')
    plt.title(
        f'Bayesian Logistic Regression\nDecision Boundary with Uncertainty ({mag_factor}x)',
        fontsize=14,
        fontweight='bold'
    )
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary_comparison(X, y, model_linear, model_kernel, title_lin, title_ker):
    """
    Visualizes and compares the decision boundaries between the linear model and the kernel model side-by-side.
    """
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Standard Linear Logistic Regression ---
    # Call predict_proba directly; the custom method handles adding the intercept internally.
    # It returns a 1D array of probabilities for Class 1.
    # FIX: Add bias column to match the training data shape (X_xor_bias)
    grid_points_bias = np.c_[np.ones((grid_points.shape[0], 1)), grid_points]
    Z_lin = model_linear.predict_proba(grid_points_bias).reshape(xx.shape)
    
    contour1 = axes[0].contourf(xx, yy, Z_lin, levels=20, cmap='RdBu', alpha=0.8)
    axes[0].contour(xx, yy, Z_lin, levels=[0.5], colors='black', linewidths=2, linestyles='dashed')
    
    # Scatter data points
    axes[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='crimson', edgecolors='k', label='Class 0')
    axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='royalblue', edgecolors='k', label='Class 1')
    axes[0].set_title(title_lin, fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Feature 1 ($X_1$)')
    axes[0].set_ylabel('Feature 2 ($X_2$)')
    axes[0].legend(loc='best')

    # --- Plot 2: Kernel Logistic Regression (RBF) ---
    Z_ker = model_kernel.predict_proba(grid_points).reshape(xx.shape)
    
    contour2 = axes[1].contourf(xx, yy, Z_ker, levels=20, cmap='RdBu', alpha=0.8)
    axes[1].contour(xx, yy, Z_ker, levels=[0.5], colors='black', linewidths=2, linestyles='dashed')
    
    # Scatter data points
    axes[1].scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='crimson', edgecolors='k', label='Class 0')
    axes[1].scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='royalblue', edgecolors='k', label='Class 1')
    axes[1].set_title(title_ker, fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Feature 1 ($X_1$)')
    axes[1].legend(loc='best')

    # Add a shared colorbar for both subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour2, cax=cbar_ax, label='Predicted Probability (Class 1)')

    plt.subplots_adjust(right=0.9)
    plt.show()

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    model_name: str = "Model",
    title: str = "Reliability Diagram",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 120
) -> None:
    """
    Plot the Reliability Diagram along with a histogram of predicted probabilities
    to analyze the calibration of output probabilities.
    
    Parameters:
    -----------
    y_true : 1D array-like
        True binary labels.
    y_prob : 1D array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins for calibration curve.
    model_name : str
        Name of the model for the legend.
    title : str
        Title of the plot.
    figsize : tuple
        Size of the figure.
    dpi : int
        DPI of the figure.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(4, 1, hspace=0.4)
    
    # Top plot: Calibration Curve
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label=model_name, color="#D55E00", linewidth=2.5, markersize=8)
    
    ax1.set_ylabel("Fraction of positives", fontsize=12, fontweight="bold")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax1.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.spines["bottom"].set_color("dimgray")
    ax1.spines["left"].set_color("dimgray")
    
    # Bottom plot: Histogram of predicted probabilities
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.hist(y_prob, range=(0, 1), bins=n_bins, label=model_name,
             histtype="bar", lw=2, color="#0072B2", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Mean predicted probability", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax2.set_xlim([-0.05, 1.05])
    ax2.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["bottom"].set_color("dimgray")
    ax2.spines["left"].set_color("dimgray")

    plt.tight_layout()
    plt.show()


def plot_structural_risk_minimization_srm(
    model_complexity: np.ndarray,
    empirical_risk: np.ndarray,
    vc_confidence: np.ndarray,
    true_risk: np.ndarray,
    opt_complexity: float,
    opt_risk: float,
) -> None:
    plt.figure(figsize=(10, 6))

    # Plot the three fundamental curves
    plt.plot(model_complexity, empirical_risk, 'b-', linewidth=2.5, label='Empirical Risk (Train Error)')
    plt.plot(model_complexity, vc_confidence, 'g--', linewidth=2, label='VC Confidence (Penalty)')
    plt.plot(model_complexity, true_risk, 'r-', linewidth=3, label='True Risk (Test Error)')

    # Mark the optimal SRM sweet spot
    plt.axvline(x=opt_complexity, color='black', linestyle=':', linewidth=1.5)
    plt.scatter(opt_complexity, opt_risk, color='black', s=100, zorder=5)
    plt.annotate('Optimal Capacity\n(SRM Sweet Spot)',
                 xy=(opt_complexity, opt_risk),
                 xytext=(opt_complexity + 0.5, opt_risk + 0.25),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold')

    plt.title('Structural Risk Minimization (SRM)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Model Complexity (VC Dimension)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.xlim(1, 10)
    plt.ylim(0, 1.2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_calibration_overview(
    prob_models: dict,
    X_test_scaled: np.ndarray,
    y_test_bin: np.ndarray,
    class_names: list,
    color_palette: list = None,
    figsize: tuple = (16, 12),
    dpi: int = 110,
) -> None:
    """
    Overview grid: overlay all models on the same axes for each class.
    One subplot per class, all models shown together for direct comparison.
    """
    if color_palette is None:
        color_palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']
 
    from sklearn.calibration import calibration_curve
 
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
 
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        x = np.linspace(0, 1, 200)
 
        # Reference diagonal (perfectly calibrated)
        ax.plot([0, 1], [0, 1], 'k--', lw=2.0, label='Perfectly calibrated', zorder=0)
 
        # Shade under-confident and over-confident regions
        ax.fill_between(x, x, 1, alpha=0.06, color='red',
                        label='Under-confident region', zorder=0)
        ax.fill_between(x, 0, x, alpha=0.06, color='blue',
                        label='Over-confident region', zorder=0)
 
        # Plot calibration curve for each model
        for color, (model_name, model) in zip(color_palette, prob_models.items()):
            y_proba      = model.predict_proba(X_test_scaled)
            y_prob_class = y_proba[:, class_idx]
            y_true_class = y_test_bin[:, class_idx]
 
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_true_class, y_prob_class, n_bins=10, strategy='uniform'
                )
                ax.plot(mean_pred, frac_pos, 's-',
                        color=color, linewidth=2.5, markersize=7,
                        label=model_name, alpha=0.9)
            except Exception as e:
                print(f"  [Skip] {model_name} — Class {class_idx}: {e}")
 
        ax.set_title(class_names[class_idx], fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.12])
        ax.grid(True, linestyle='--', alpha=0.35)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
 
    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12, framealpha=0.95)
 
    fig.suptitle(
        'Reliability Diagrams — Calibration Comparison across Models (OvR Strategy)',
        fontsize=18, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
 
 
def plot_calibration_per_model(
    prob_models: dict,
    X_test_scaled: np.ndarray,
    y_test_bin: np.ndarray,
    class_names: list,
    n_cols: int = 3,
    figsize_per_subplot: tuple = (8, 7),
    dpi: int = 110,
) -> None:
    """
    Per-model reliability diagrams: for each class, one subplot per model
    showing the calibration curve with ECE score and confidence regions.
    """
    from sklearn.calibration import calibration_curve
 
    for focus_class, focus_name in enumerate(class_names):
        print(f"Reliability Diagrams — {focus_name} (OvR)")
 
        n_models = len(prob_models)
        n_rows = int(np.ceil(n_models / n_cols))
        fw = figsize_per_subplot[0] * n_cols
        fh = figsize_per_subplot[1] * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw, fh), dpi=dpi)
        axes = axes.flatten()
 
        for idx, (model_name, model) in enumerate(prob_models.items()):
            ax = axes[idx]
            y_proba  = model.predict_proba(X_test_scaled)
            y_prob_c = y_proba[:, focus_class]
            y_true_c = y_test_bin[:, focus_class]
 
            frac_pos, mean_pred = calibration_curve(
                y_true_c, y_prob_c, n_bins=10, strategy='uniform'
            )
 
            # Reference diagonal
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly calibrated')
 
            # Model calibration curve
            ax.plot(mean_pred, frac_pos, 's-',
                    color='#D55E00', linewidth=2.2, markersize=8, label=model_name)
 
            # Red region (above diagonal): model is over-confident
            ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.06, color='red',
                            label='Over-confident region')
 
            # Blue region (below diagonal): model is under-confident
            ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.06, color='blue',
                            label='Under-confident region')
 
            # ECE = mean absolute deviation between predicted probability and true fraction
            # Lower ECE → better calibrated
            ece = np.mean(np.abs(frac_pos - mean_pred))
            ax.text(0.03, 0.93, f'ECE = {ece:.4f}', transform=ax.transAxes,
                    fontsize=14, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
 
            ax.set_title(model_name, fontsize=15, fontweight='bold', pad=10)
            ax.set_xlabel('Mean Predicted Probability', fontsize=13)
            ax.set_ylabel('Fraction of Positives', fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.12])
            ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.35)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
 
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
 
        fig.suptitle(f'Per-model Reliability Diagram — {focus_name} vs Rest',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
 
 
def plot_mcnemar_heatmaps(
    all_comparisons: list,
    alpha: float = 0.05,
    n_cols: int = 3,
    figsize: tuple = (18, 10),
    dpi: int = 100,
) -> None:
    """
    Visualize McNemar's test results as contingency heatmaps.
    Green = statistically significant, Red = not significant.
    """
    fig, axes = plt.subplots(2, n_cols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
 
    for idx, row in enumerate(all_comparisons):
        if idx >= len(axes):
            break
        ax = axes[idx]
 
        # Build 2×2 contingency table
        table = np.array([[row['n11'], row['n10']],
                          [row['n01'], row['n00']]])
        labels = np.array([
            [f"Both correct\n{row['n11']}",      f"A correct & B incorrect\n{row['n10']}"],
            [f"A incorrect & B correct\n{row['n01']}", f"Both wrong\n{row['n00']}"]
        ])
 
        # Green for significant results, red for non-significant
        cmap = 'Greens' if row['significant'] else 'Reds'
        sns.heatmap(table, annot=labels, fmt='', cmap=cmap, ax=ax,
                    linewidths=2, linecolor='white', cbar=False,
                    annot_kws={'size': 13, 'weight': 'bold'})
 
        # Shorten model names for the title
        short_a = row['Model A'].replace('Logistic Regression', 'LR')
        short_b = (row['Model B']
                   .replace('Logistic Regression', 'LR')
                   .replace('Linear Discriminant Analysis', 'LDA')
                   .replace('Quadratic Discriminant Analysis', 'QDA')
                   .replace('Multiclass Perceptron', 'Perceptron'))
 
        title_color = '#1a5c1a' if row['significant'] else '#8b0000'
        ax.set_title(
            f"{short_a}\nvs\n{short_b}\n"
            f"χ²={row['chi2']:.3f}  p={row['p_value']:.4f}  →  {row['conclusion']}",
            fontsize=10, fontweight='bold', color=title_color, pad=8
        )
        ax.set_xticklabels(['Model B correct', 'Model B wrong'], fontsize=10)
        ax.set_yticklabels(['Model A correct', 'Model A wrong'], fontsize=10, rotation=0)
 
    # Hide unused axes
    for idx in range(len(all_comparisons), len(axes)):
        axes[idx].set_visible(False)
 
    fig.suptitle(
        "McNemar's Test — Contingency Tables for All Comparisons (α = 0.05)\n"
        "Green = Statistically Significant | Red = Not Significant",
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.show()
