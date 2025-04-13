import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from .fista_solver import growl_fista 

class GrowlRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible GrOWL-based regression estimator.
    
    Parameters
    ----------
    lambda_1 : float
        Base level for weights in the GrOWL penalty.
    lambda_2 : float
        Amount of linear decay to apply to the first part of the weight vector.
    ramp_size : float in [0, 1]
        Fraction of features over which a linear ramp is applied
        (OSCAR-style decay). The remainder are assigned the constant lambda_1.
    max_iter : int
        Maximum number of FISTA iterations.
    tol : float
        Tolerance for stopping criterion.
    check_type : {'absolute_cost', 'relative_cost', 'solution_diff'}
        Defines the stopping criterion in the FISTA loop.
    scale_objective : bool
        Whether to divide the obj. function by ||Y||_F^2 (useful for large Y).
    verbose : bool
        If True, prints iteration info.
    """

    def __init__(self, lambda_1=1.0, lambda_2=0.0, ramp_size=0.0,
                 max_iter=10000, tol=1e-4, fit_intercept=False, 
                 check_type='relative_cost', scale_objective=False, 
                 verbose=False):
        
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.ramp_size = ramp_size
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.check_type = check_type
        self.scale_objective = scale_objective
        self.verbose = verbose

    def fit(self, X, Y):
        """
        Fit the GrOWL regressor to data (X, Y).
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Save input shapes for later validation
        self._n_samples_, self._n_features_ = X.shape
    
        # Make Y 2D if necessary
        if Y.ndim == 1:
            Y = Y[:, None]
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows."
                             f"Got X: {X.shape[0]}, Y: {Y.shape[0]}")

        if self.fit_intercept:
            self.X_mean_ = X.mean(axis=0)
            self.Y_mean_ = Y.mean(axis=0)
            X = X - self.X_mean_
            Y = Y - self.Y_mean_
        else:
            self.X_mean_ = np.zeros(X.shape[1])
            self.Y_mean_ = np.zeros(Y.shape[1])

        # Run the FISTA solver
        B, cost_hist = growl_fista(
            X, Y,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            ramp_size=self.ramp_size,
            max_iter=self.max_iter,
            tol=self.tol,
            check_type=self.check_type,
            scale_objective=self.scale_objective,
            verbose=self.verbose
        )

        # Flatten B if 1D response
        self.coef_ = B[:, 0] if B.shape[1] == 1 else B
        self.cost_history_ = cost_hist
        self.is_fitted_ = True  # for predict() or external checks

        return self
    
    def _check_is_fitted(self): # Helper method for predict()
        """Check if the estimator is fitted."""
        if not getattr(self, "is_fitted_", False):
            raise AttributeError("GrowlRegressor is not fitted yet. Please call 'fit()' first.")

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise AttributeError("This GrowlRegressor instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this method.")
    
        X = np.asarray(X)

        # Check dimensional consistency with training data
        if X.shape[1] != self.X_mean_.shape[0]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) does not match training data ({self.X_mean_.shape[0]}).")
    
        # Center X using training mean
        X_centered = X - self.X_mean_

        # Compute predictions
        pred = X_centered @ self.coef_

        # Add back Y mean
        if np.ndim(pred) > 1:
            return pred + self.Y_mean_
        else:
            return (pred + self.Y_mean_).ravel()
