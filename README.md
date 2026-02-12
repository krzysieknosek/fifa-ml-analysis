# ⚽ FIFA 22 Machine Learning Analysis & Prediction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📖 Project Overview
This project is a comprehensive Machine Learning study based on the **FIFA 22 dataset** containing over **19,000 players** and 100+ attributes.

The goal was not only to use existing libraries but also to **implement core ML algorithms from scratch** (Linear Regression via Gradient Descent & Closed-Form) to understand the mathematics behind them. The project evolves from basic Exploratory Data Analysis (EDA) to Deep Learning models using **PyTorch**.

## 🚀 Key Features (Project Stages)

### Part I: Exploratory Data Analysis (EDA)
Analysis of physical and technical attributes to understand player distribution.
- **Statistical Analysis:** Calculated stats for numerical/categorical features for 19k+ records.
- **Visualizations:**
  - Correlation Heatmaps (Height/Weight vs. Pace).
  - Boxplots & Violin Plots (Overall rating & Weight distribution).
  - Histograms (Age distribution, Shooting skills).
- **Key Insight:** A strong correlation exists where market value skyrockets for players with an "Overall" rating above 75.

### Part II: Model Implementation & Training
Building predictive models for player ratings (`overall`) and classifying positions (`club_position`).
- **Scikit-learn Models:** Comparison of Linear Regression, Decision Tree, and SVR.
- **Manual Implementation:**
  - **Gradient Descent:** Custom implementation of the optimization loop.
  - **Closed-Form Solution:** Analytical solution using Linear Algebra (Normal Equation).
- **Deep Learning:** Implementation of a Neural Network (MLP) in **PyTorch** for multi-class classification, comparing CPU vs. GPU training times.

### Part III: Optimization & Tuning
Refining models to improve generalization and handle data issues.
- **Ablation Study:** Analyzing the impact of Cross-Validation and Regularization (L1 Lasso & L2 Ridge).
- **Imbalanced Data:** Handling class imbalance using **Oversampling** and **Undersampling** techniques (Oversampling improved F1-score significantly).
- **Hyperparameter Tuning:** Using `GridSearchCV` to optimize Decision Tree parameters (`max_depth`, `min_samples_split`).

---

## 📂 Project Structure

| File Name | Description |
| :--- | :--- |
| **EDA & Scikit-learn Models** | |
| `sklearn_model_comparison.py` | Pipeline setup and comparison of LR, Decision Tree, and SVR. |
| **Manual Implementations** | |
| `linear_regression_gradient_descent.py` | Custom Gradient Descent implementation from scratch. |
| `linear_regression_closed_form.py` | Analytical solution using Linear Algebra (Normal Equation). |
| `linear_regression_convergence_monitor.py` | Tracking MSE loss per epoch to visualize convergence. |
| `polynomial_regression_convergence.py` | Extending the manual model with Polynomial Features. |
| **Deep Learning** | |
| `pytorch_position_classifier.py` | PyTorch Neural Network for player position classification. |
| **Optimization** | |
| `linear_regression_cross_validation.py` | Implementation of K-Fold Cross-Validation. |
| `linear_regression_regularization.py` | Adding L1 (Lasso) and L2 (Ridge) penalties to the custom optimizer. |
| `imbalanced_data_resampling.py` | Handling imbalanced datasets via Oversampling/Undersampling. |
| `sklearn_hyperparameter_optimization.py` | Tuning hyperparameters using `GridSearchCV`. |

---

## 📊 Results Highlights

### 1. Regression Model Performance (MSE)
We compared standard models to predict player ratings. **SVR** performed best on the evaluation set.

| Model | Training MSE | Validation MSE | Evaluation MSE |
| :--- | :--- | :--- | :--- |
| Linear Regression | 14.47 | 14.50 | 14.30 |
| Decision Tree | 4.66 | 8.05 | 7.40 |
| **SVR (Best)** | **6.21** | **6.83** | **6.17** |

### 2. Manual vs. Library Implementation
My custom implementation of Linear Regression achieved nearly identical results to the Scikit-learn version, validating the mathematical logic.

- **Scikit-learn LR:** MSE 33.25
- **Custom Gradient Descent:** MSE 32.82
- **Custom Closed-Form:** MSE 32.82

### 3. PyTorch Performance (CPU vs. GPU)
Training the logistic regression model on a small dataset showed similar times, with a slight overhead for GPU data transfer.

| Device | Training Time (s) |
| :--- | :--- |
| CPU | 11.11 |
| GPU | 11.94 |

---

## 🛠 Tech Stack
* **Language:** Python 3.9+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, PyTorch
* **Visualization:** Matplotlib, Seaborn

## 💿 Installation

1. Clone the repository:
```bash
git clone [https://github.com/your-username/fifa-ml-analysis.git](https://github.com/your-username/fifa-ml-analysis.git)
```

2. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib torch
```


3. Run any script, e.g.:
```bash
python sklearn_model_comparison.py
```



---

*This project was developed as part of the "System and Decision Methods" course.*
