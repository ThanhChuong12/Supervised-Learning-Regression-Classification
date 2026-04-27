# Supervised Learning: Regression & Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

> **Assignment Report — Introduction to Machine Learning (Nhập môn học máy)**
>
> *Faculty of Information Technology, VNU-HCM University of Science*

---

## Table of Contents
- [1. About The Project](#1-about-the-project)
- [2. Datasets](#2-datasets)
  - [Regression Dataset: Appliances Energy Prediction](#regression-dataset-appliances-energy-prediction)
  - [Classification Dataset: Room Occupancy Estimation](#classification-dataset-room-occupancy-estimation)
- [3. Part 1 — Regression](#3-part-1--regression)
  - [Regression Models Implemented](#regression-models-implemented)
  - [Regression Key Techniques](#regression-key-techniques)
- [4. Part 2 — Classification](#4-part-2--classification)
  - [Classification Models Implemented](#classification-models-implemented)
  - [Classification Key Techniques](#classification-key-techniques)
- [5. Repository Structure](#5-repository-structure)
- [6. Getting Started](#6-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [7. Contributors](#7-contributors)
- [8. License \& Acknowledgments](#8-license--acknowledgments)
  - [Academic Acknowledgments](#academic-acknowledgments)
  - [Data Attribution](#data-attribution)
  - [License](#license)


---

## 1. About The Project

This project is the **Assignment Report No. 1** for the *Introduction to Machine Learning* course. It provides a comprehensive, **from-scratch** implementation of fundamental supervised learning algorithms — covering both regression and classification — applied to real-world IoT sensor datasets.

The project is organized into two interconnected parts:

| **Part 1: Regression** | **Part 2: Classification** |
| :--- | :--- |
| Predict appliance energy consumption (Wh) from indoor sensor readings and outdoor weather data. | Estimate the number of occupants in a room (0–3 people) from multi-modal IoT sensor streams. |
| Implements OLS, Ridge, Lasso, Elastic Net, WLS, and Kernel methods from scratch. | Implements Perceptron, Logistic/Probit Regression, LDA, QDA, Naive Bayes, and Bayesian models from scratch. |
| Explores basis function expansion, regularization, and Bayesian linear regression. | Explores multiclass strategies (OvR, OvO), discriminant analysis, and probabilistic calibration. |

> **Core Philosophy:** Every algorithm in this project is implemented from scratch using **NumPy only** — no black-box sklearn estimators for the core models — to build a deep, principled understanding of machine learning fundamentals.

---

## 2. Datasets

### Regression Dataset: Appliances Energy Prediction

| Info | Details |
| :--- | :--- |
| **Primary Source** | [Appliances Energy Prediction (Kaggle)](https://www.kaggle.com/datasets/sohommajumder21/appliances-energy-prediction-data-set) |
| **Original Custodian** | [UCI ML Repository (ID: 374)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) |
| **Temporal Coverage** | ~4.5 months, sampled every **10 minutes** |
| **Dataset Size** | **19,735** records × **29** attributes |
| **Target Variable** | `Appliances` — appliance energy consumption (Wh) |

The dataset records energy use of home appliances alongside temperature and humidity readings from a ZigBee wireless sensor network inside a low-energy house in Mons, Belgium, merged with outdoor weather data from the nearest airport weather station. Two random noise variables (`rv1`, `rv2`) are included to test feature selection quality.

**Key Feature Groups:**

- **Indoor Conditions:** Temperature (`T1`–`T9`) & Relative Humidity (`RH_1`–`RH_9`) across 9 zones (kitchen, living room, laundry, office, bathroom, ironing room, teen room, parents' room, etc.)
- **Outdoor / Weather:** `T_out`, `T6`, `RH_out`, `RH_6`, `Press_mm_hg`, `Windspeed`, `Visibility`, `Tdewpoint`
- **Other Energy:** `lights` (lighting energy in Wh)
- **Random Noise:** `rv1`, `rv2` (used for feature selection validation)

---

### Classification Dataset: Room Occupancy Estimation

| Info | Details |
| :--- | :--- |
| **Primary Source** | [Room Occupancy Estimation (Kaggle)](https://www.kaggle.com/datasets/ruchikakumbhar/room-occupancy-estimation/data) |
| **Original Custodian** | [UCI ML Repository (ID: 864)](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation) |
| **Temporal Coverage** | 4 continuous days (starting 22/12/2017) |
| **Environment** | Standard lab room, 6m × 4.6m, equipped with 7 sensor nodes |
| **Sampling Rate** | Every **30 seconds** |
| **Dataset Size** | **10,129** records × **19** attributes |
| **Target Variable** | `Room_Occupancy_Count` — number of people present (0, 1, 2, or 3) |

The dataset captures multi-modal environmental signals from IoT sensors for **non-intrusive occupancy estimation** — no cameras, no wearables.

**Key Feature Groups:**

- **Thermodynamic:** `S1_Temp`, `S2_Temp`, `S3_Temp`, `S4_Temp`
- **Illuminance:** `S1_Light`, `S2_Light`, `S3_Light`, `S4_Light`
- **Acoustic:** `S1_Sound`, `S2_Sound`, `S3_Sound`, `S4_Sound`
- **Air Quality:** `S5_CO2`, `S5_CO2_Slope` (rate of CO₂ change)
- **Motion (PIR):** `S6_PIR`, `S7_PIR`

---

## 3. Part 1 — Regression

### Regression Models Implemented

All regression models are implemented from scratch in `code/Part1_Regression/models.py`.

| Model | Description |
| :--- | :--- |
| **OLS (Ordinary Least Squares)** | Closed-form normal equations: $\mathbf{w} = (\Phi^T\Phi)^{-1}\Phi^T\mathbf{y}$ |
| **Mini-Batch Gradient Descent** | Iterative OLS optimization with Step Decay & Cosine Annealing learning rate schedules |
| **WLS (Weighted Least Squares)** | Observation-weighted regression for heteroscedastic noise; weights estimated from OLS residuals |
| **Ridge Regression** | L2-penalized closed-form solution; bias term excluded from regularization |
| **Lasso Regression** | L1-penalized via Coordinate Descent with warm-start for efficient $\lambda$ path search |
| **Elastic Net** | Combined L1+L2 penalty via Coordinate Descent |
| **Kernel Ridge Regression (KRR)** | Dual formulation with RBF and Polynomial kernels; solves $(K + \lambda I)\alpha = \mathbf{y}$ |
| **Gaussian Process Regression (GPR)** | Full Bayesian non-parametric model; hyperparameters optimized via gradient ascent on log-marginal-likelihood (LML) |
| **Bayesian Linear Regression** | Analytically computes posterior $p(\mathbf{w} \mid \mathbf{t})$ and predictive distribution $\bar{f}^* \pm 2\sigma_N$ using Gaussian RBF basis functions |
| **Robust Regression (IRLS + Huber)** | Iteratively Reweighted Least Squares with Huber loss for outlier robustness |

### Regression Key Techniques

- **Basis Function Expansion:** Four families implemented — Polynomial, RBF (Radial Basis Function), Sigmoid, and Natural Cubic Spline — all constructable via a unified `make_design_matrix()` API.

- **Feature Engineering & Selection:**
  - Interaction terms between feature groups (temperature × humidity, etc.)
  - Forward Selection and Backward Elimination using ridge-penalized validation loss
  - Feature group identification by name prefix (`select_feature_groups()`)

- **Regularization & Hyperparameter Tuning:**
  - Time-Series K-Fold Cross-Validation (expanding window) to respect temporal ordering
  - Warm-start grid search for Lasso $\lambda$
  - Evidence Maximization (Empirical Bayes) for Bayesian hyperparameters $\alpha$ and $\beta$

- **Heteroscedasticity Testing:** Breusch-Pagan test implemented from first principles.

- **Model Diagnostics:**
  - Bias–Variance decomposition via Bootstrap (200 resamples)
  - Residual plots, Predicted vs. Actual plots, Learning curves
  - Wilcoxon signed-rank test and paired t-test for statistical model comparison

- **Evaluation Metrics:** MSE, RMSE, MAE, R²

---

## 4. Part 2 — Classification

### Classification Models Implemented

All classification models are implemented from scratch in `code/Part2_Classification/models.py`.

| Model | Description |
| :--- | :--- |
| **Perceptron** | Original Rosenblatt Perceptron with error history tracking and early stopping |
| **Logistic Regression** | Gradient descent with L1/L2 regularization and class-balanced sample weighting |
| **Binary Logistic Regression** | Supports both GD and Newton-Raphson (Hessian-free Conjugate Gradient) optimization |
| **Softmax Regression** | Multiclass logistic regression with numerically stable Log-Sum-Exp trick |
| **One-vs-Rest (OvR)** | Meta-estimator wrapping any binary classifier for multiclass; probability normalization |
| **One-vs-One (OvO)** | Meta-estimator with majority voting over all pairwise classifiers |
| **Linear Discriminant Analysis (LDA)** | Pooled covariance, Fisher projection via generalized eigenvalue problem, `transform()` for dimensionality reduction |
| **Quadratic Discriminant Analysis (QDA)** | Class-specific covariance matrices, Mahalanobis distance scoring |
| **Probit Regression** | Vectorized gradient descent using Standard Normal CDF/PDF instead of sigmoid |
| **Bayesian Logistic Regression** | MAP estimation + Laplace approximation of posterior; predictive uncertainty via probit approximation |
| **Kernel Logistic Regression** | Dual formulation with RBF kernel trick for non-linearly separable problems (e.g., XOR) |
| **Gaussian Naive Bayes** | Class-conditional Gaussian assumption with log-likelihood for numerical stability |

### Classification Key Techniques

- **Fisher Ratio Feature Ranking:** Vectorized computation of between-class vs. within-class variance ratio for feature importance, available in `BaseDiscriminantAnalysis`.

- **Multiclass Strategies:** Full OvR and OvO implementations support any custom binary estimator instance, with proper probability normalization.

- **Noise Robustness Evaluation:** `inject_label_noise()` flips a controlled proportion of training labels to benchmark model robustness under label corruption.

- **Bayesian Uncertainty Quantification:** `BayesianLogisticRegression` computes the Laplace-approximated posterior covariance $\Sigma = A^{-1}$, enabling predictive standard deviation estimates $\sigma_a$ per data point.

- **Visualization Module (`visualizations.py`):**
  - Convergence comparison: GD vs. Newton-Raphson (loss vs. epochs + wall-clock time)
  - LDA vs. QDA decision boundaries in 2D Fisher discriminant space with confidence contours
  - Logistic vs. Probit KDE probability density comparison
  - Decision boundaries in PCA space with confidence margins (P = 0.1 / 0.5 / 0.9)
  - Bayesian decision boundary with uncertainty bands ($\mu_a \pm k \cdot \sigma_a$)
  - Reliability diagrams (calibration curves)
  - Noise robustness degradation curves

- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix.

---

## 5. Repository Structure

The project follows a **modular architecture** cleanly separating Data, Source Code, Notebooks, and Reports.

```text
Supervised-Learning-Regression-Classification/
│
├── code/
│   ├── Part1_Regression/
│   │   ├── models.py                       # All regression algorithms & utilities
│   │   ├── visualizations.py               # Rich visualization utilities
│   │   ├── 01_eda_and_preprocessing.ipynb  # EDA, feature engineering & data pipeline
│   │   ├── 02_modeling.ipynb               # Core model training & evaluation
│   │   └── 03_advanced_bonus_experiments.ipynb  # Kernel Ridge, GPR, Bayesian LR, Bias-Variance
│   │
│   └── Part2_Classification/
│       ├── models.py                       # All classification algorithms
│       ├── visualizations.py               # Rich visualization utilities
│       ├── 01_eda_and_preprocessing.ipynb  # EDA, feature analysis & preprocessing pipeline
│       ├── 02_modeling.ipynb               # Core classifiers: Perceptron, LogReg, LDA, QDA, NB
│       └── 03_advanced_bonus_experiments.ipynb  # Bayesian LogReg, Kernel LR, noise robustness
│
├── data/
│   ├── raw/
│   │   ├── Energy_Use.csv                  # Raw appliance energy data
│   │   └── Room_Occupancy.csv              # Raw room occupancy data
│   ├── processed/
│   │   ├── Energy_Use_train.csv            # Regression training split
│   │   ├── Energy_Use_val.csv              # Regression validation split
│   │   ├── Energy_Use_test.csv             # Regression test split
│   │   ├── Room_Occupancy_train.csv        # Classification training split
│   │   ├── Room_Occupancy_val.csv          # Classification validation split
│   │   └── Room_Occupancy_test.csv         # Classification test split
│   └── README.md                           # Data catalog & provenance documentation
│
├── logs/
│   ├── logs_classification.json            # Execution logs and results for classification tests
│   └── logs_regression.json                # Execution logs and results for regression tests
│
├── report/
│   ├── chapters/                           # Chapter files
│   │   ├── 01_tong_quan.tex                # Chapter 1: Overview
│   │   ├── 02_hoi_quy.tex                  # Chapter 2: Regression
│   │   ├── 03_phan_lop.tex                 # Chapter 3: Classification
│   │   ├── 04_so_sanh.tex                  # Chapter 4: Comparison
│   │   └── 05_tong_ket.tex                 # Chapter 5: Conclusion
│   ├── refs/                               
│   │   └── example.bib                     # BibTeX references
│   ├── codespace.sty                       # LaTeX styling packages
│   ├── hcmus-report.cls                    # HCMUS template class file
│   └── report.tex                          # LaTeX main source
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt                        # Project dependencies
```

---

## 6. Getting Started

### Prerequisites

- **Python**: 3.9 or later
- **Package Manager**: `pip` or `conda`
- **Git**: to clone the repository

### Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/ThanhChuong12/Supervised-Learning-Regression-Classification.git
cd Supervised-Learning-Regression-Classification
```

**Step 2: Create a virtual environment**

*Option A — using `venv` (recommended for VS Code):*
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

*Option B — using `conda` (recommended for Jupyter Lab):*
```bash
conda create --name ml-env python=3.9
conda activate ml-env
```

**Step 3: Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Set up the data**

The raw datasets must be downloaded from Kaggle and placed in `data/raw/`. You can use the Kaggle CLI:

```bash
# Regression dataset
kaggle datasets download -d sohommajumder21/appliances-energy-prediction-data-set -p ./data/raw/ --unzip

# Classification dataset
kaggle datasets download -d ruchikakumbhar/room-occupancy-estimation -p ./data/raw/ --unzip
```

Rename the CSV files to `Energy_Use.csv` and `Room_Occupancy.csv` respectively as expected by the preprocessing notebooks.

> **Note:** Ensure your Kaggle API token (`kaggle.json`) is placed in `~/.kaggle/` (Linux/Mac) or `C:\Users\<Username>\.kaggle\` (Windows).

---

### Usage

The notebooks are designed to be executed **in sequential order** within each part to maintain the data pipeline:

**Step 1: Launch Jupyter**
```bash
jupyter notebook
```
*(Or open the project folder in VS Code and select the virtual environment as the kernel.)*

**Step 2: Execute notebooks in order**

*For Part 1 — Regression:*
1. `code/Part1_Regression/01_eda_and_preprocessing.ipynb` — EDA & feature engineering → produces processed CSVs
2. `code/Part1_Regression/02_modeling.ipynb` — Train & evaluate OLS, Ridge, Lasso, Elastic Net, WLS
3. `code/Part1_Regression/03_advanced_bonus_experiments.ipynb` — Kernel Ridge, GPR, Bayesian LR, Bias-Variance

*For Part 2 — Classification:*
1. `code/Part2_Classification/01_eda_and_preprocessing.ipynb` — EDA, Fisher ratio analysis & preprocessing
2. `code/Part2_Classification/02_modeling.ipynb` — Train & evaluate Perceptron, Logistic/Probit, LDA, QDA, GNB
3. `code/Part2_Classification/03_advanced_bonus_experiments.ipynb` — Bayesian LogReg, Kernel LR, OvR/OvO, noise robustness

> **Important:** `01_eda_and_preprocessing.ipynb` **must be run first** in each part, as it generates the train/val/test splits used by all subsequent notebooks.

---

## 7. Contributors

This project was developed by a team of 5 students from the *Faculty of Information Technology, VNU-HCM University of Science*.

| Contributor | Student ID | Role | Main Responsibilities (Algorithms & Analysis) | Contribution |
| :--- | :---: | :--- | :--- | :---: |
| **Lê Hà Thanh Chương** | `23120195` | **Project Lead** | Project management; Classification EDA; LogReg (GD, IRLS, Multiclass), LDA/QDA; Probit Model, Laplace Approximation; VC Dimension analysis. | 100% |
| **Trà Văn Sỹ** | `23120197` | **ML Engineer** | Perceptron, Regularized LogReg (L1/L2); Kernel LogReg, Gaussian Naive Bayes; Empirical VC Dimension & K-fold CV metrics visualization (ROC/AUC). | 100% |
| **Huỳnh Đức Thịnh** | `23120199` | **Data Scientist** | Regression preprocessing & EDA; Linear Reg (OLS, Mini-batch GD); Full Bayesian Reg, Evidence Maximization; Learning curves & sensitivity analysis. | 100% |
| **Bùi Trung Hiếu** | `23120257` | **ML Researcher** | Imputation & scaling; Ridge, Lasso, Feature Selection; Kernel Ridge Reg, GPR; Statistical tests (t-test, Wilcoxon, McNemar); Report aggregation. | 100% |
| **Lê Công Phúc** | `23120330` | **ML Analyst** | Non-linear regression, Validation Curves, Ablation Study; Robust Regression; Bias-Variance Tradeoff, K-fold CV evaluation, decision boundaries. | 100% |

---

## 8. License & Acknowledgments

### Academic Acknowledgments

This project is the **Assignment Report No. 1** for the *Introduction to Machine Learning* course at *VNU-HCM University of Science*.

The team sincerely thanks the course instructor who provided theoretical foundations and guidance throughout the project:

- **Instructor:** MSc. Lê Nhựt Nam

### Data Attribution

- **Regression Dataset:** Extracted from the study *"Data driven prediction models of energy use of appliances in a low-energy house"*, published in **Energy and Buildings** (Vol. 140, April 2017) by researchers at the University of Mons (UMONS), Belgium. Hosted on [UCI ML Repository (ID: 374)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction). Licensed under **CC BY 4.0**.

- **Classification Dataset:** Extracted from the study *"Machine Learning-Based Occupancy Estimation Using Multivariate Sensor Nodes"*, published at **IEEE Globecom Workshops 2018**. Hosted on [UCI ML Repository (ID: 864)](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation). Licensed under **CC BY 4.0**.

### License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

The source code of this project is distributed under the **MIT License**.
You are free to use, copy, modify, merge, publish, and distribute this code, provided that the original copyright notice is retained. See the `LICENSE` file for full details.

<br>
<p align="center">
  <i>Built with ❤️ by the ML Team | University of Science, VNU-HCM | 2026</i>
</p>
