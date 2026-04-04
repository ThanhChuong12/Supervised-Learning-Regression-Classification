import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from typing import Tuple

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
