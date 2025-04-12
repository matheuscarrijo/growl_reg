import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import MultiTaskLassoCV, LassoCV
import matplotlib.pyplot as plt
import numpy as np
from growl import GrowlRegressor

np.random.seed(1997)


# ====================
# GrOWL Example
# ====================

# Simulate a toy example

n, p, r = 10, 20, 5
X = np.random.randn(n, p)
cv = 5 # Cross-validation folds

# Create high correlation between two columns
X[:, 5] = X[:, 7] + 0.1 * np.random.randn(n)

# True B matrix with some non-zero rows
B_true_toy = np.zeros((p, r))
B_true_toy[5, :] = np.random.randn(r)
B_true_toy[7, :] = B_true_toy[5, :] # correlated
B_true_toy[0, :] = np.random.randn(r)
B_true_toy[15, :] = np.random.randn(r)
B_true_toy[18, :] = np.random.randn(r)
B_true_toy[p-1, :] = np.random.randn(r)

Y_toy = X @ B_true_toy + 0.3 * np.random.randn(n, r)

# Define the GrOWL estimator and parameter grid
growl_estimator = GrowlRegressor()

param_grid = {'lambda_1': [0.1, 1.0, 5.0],
              'lambda_2': [0.0, 0.5, 1.0],
              'ramp_size': [0.0, 0.5, 1.0]}

grid_search = GridSearchCV(growl_estimator, param_grid, cv=cv, 
                           scoring='neg_mean_squared_error').fit(X, Y_toy)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Retrieve the best model and get the estimated B matrix
best_growl_model = grid_search.best_estimator_ 
B_est_growl = best_growl_model.coef_ # shape (p, r) 

# MultiTaskLasso for reference
multi_task_lasso = MultiTaskLassoCV(cv=cv, fit_intercept=False, 
                               max_iter=1000).fit(X, Y_toy)
B_est_lasso = multi_task_lasso.coef_.T  # shape (p, r)

# Plot side by side: True B, GrOWL estimate, MultiTaskLasso estimate
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# --- True B ---
im0 = axes[0].imshow(B_true_toy, cmap='coolwarm', aspect='auto')
axes[0].set_title("True B Matrix")
axes[0].set_xlabel("Output dimension (r)")
axes[0].set_ylabel("Features (p)")
fig.colorbar(im0, ax=axes[0])
# --- GrOWL Estimate ---
im1 = axes[1].imshow(B_est_growl, cmap='coolwarm', aspect='auto')
axes[1].set_title("Estimated B (Growl)")
axes[1].set_xlabel("Output dimension (r)")
axes[1].set_ylabel("Features (p)")
fig.colorbar(im1, ax=axes[1])
# --- MultiTaskLasso Estimate ---
im2 = axes[2].imshow(B_est_lasso, cmap='coolwarm', aspect='auto')
axes[2].set_title("Estimated B (MultiTaskLasso)")
axes[2].set_xlabel("Output dimension (r)")
axes[2].set_ylabel("Features (p)")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

# ====================
# OWL Example
# ====================

# ---------------------- 1) Generate data ----------------------
n, p = 100, 20

# Base random design
X_base = np.random.randn(n, p)
X = X_base.copy()

# Make some columns correlated
X[:, 1] = X[:, 0] + 0.02 * np.random.randn(n)
X[:, 3] = X[:, 2] + 0.01 * np.random.randn(n)
X[:, 4] = X[:, 5] + 0.02 * np.random.randn(n)
X[:, 6] = X[:, 7] + 0.07 * np.random.randn(n)

# True coefficient vector
b_true = np.random.randn(p)
b_true[0:2] = 2.0   # large positive
b_true[2:4] = -1.5  # negative
b_true[4:6] = 1.0
b_true[6:8] = -0.5
b_true[8] = 0
b_true[9] = 0
# The rest remain random

# Single-output target (shape (n,))
y = X @ b_true + 0.05 * np.random.randn(n)

# ---------------------- 2) Fit scikit-learn's Lasso ----------------------
lasso = LassoCV(cv=cv, fit_intercept=False, max_iter=10000).fit(X, y)
b_hat_lasso_sklearn = lasso.coef_  # shape (p,)

# ------------- 3) Fit GrowlRegressor (OWL style) with grid-search -------------
# Because GrowlRegressor uses (lambda_1, lambda_2, ramp_size) to build weights,
# we can explore different "OWL-like" or "OSCAR-like" penalty shapes via 
# ramp_size=1.0, etc.
param_grid = {
    # Try different lambda_1 levels
    'lambda_1': [0.1, 0.5, 1.0],
    # Try different lambda_2 levels
    'lambda_2': [0.0, 0.5, 1.0],
    # ramp_size = 1.0 => Full OSCAR-like shape; = 0 => Lasso-like shape
    'ramp_size': [0.0, 0.5, 1.0],
}

# Fit the GrowlRegressor with grid search
owl_estimator = GrowlRegressor(max_iter=2000, tol=1e-7, 
                               check_type='relative_cost')
grid_search = GridSearchCV(owl_estimator, 
                           param_grid=param_grid, 
                           scoring='neg_mean_squared_error', 
                           cv=5).fit(X, y)

# Print best parameters and score
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Retrieve best model and its coefficients
best_growl_model = grid_search.best_estimator_
b_hat_owl = best_growl_model.coef_  # shape (p,)

# ---------------------- 4) Visual comparison ----------------------
plt.figure(figsize=(15, 4))

# 1) True coefficients
plt.subplot(1, 3, 1)
plt.stem(b_true, linefmt="b-", markerfmt="bo", basefmt="r-")
plt.title("Original b")

# 2) Lasso (scikit-learn)
plt.subplot(1, 3, 2)
plt.stem(b_hat_lasso_sklearn, linefmt="b-", markerfmt="bo", basefmt="r-")
plt.title("LASSO (sklearn) b")

# 3) Best GrowlRegressor (OWL-like)
plt.subplot(1, 3, 3)
plt.stem(b_hat_owl, linefmt="b-", markerfmt="bo", basefmt="r-")
plt.title("OWL-like (GrowlRegressor) b")

plt.tight_layout()
plt.show()

