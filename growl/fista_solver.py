import numpy as np
from .prox_operator import prox_growl

def growl_fista(X, Y, lambda_1=None, lambda_2=None, ramp_size=None,
                w=None, max_iter=10000, tol=1e-4, check_type='relative_cost',
                scale_objective=False, verbose=False):
    """
    Solve the GrOWL-regularized least-squares problem:

        min_B (1/2n) * ||Y - X B||_F^2  +  \sum_i w_i || \beta_{[i], \cdot} ||_2

    using a FISTA-type proximal splitting scheme.

    By default, the user can pass in a custom `w` (length-p array, 
    non-negative, non-increasing). Alternatively, set the parameter values 
    'lambda_1', 'lambda_2', 'ramp_size' to define the weight vector. The later 
    is a fraction of the design matrix X columns the user want to use to apply
    linear decay (oscar-type weights).
        
                    ramp_size = int(np.ceil(ramp_size * p))
                    For i = 1, ..., ramp_size:
                        w_i = lambda_1 + lambda_2*(ramp_size - i + 1)
                    For i = ramp_size + 1, ..., p:
                        w_i = lambda_1

    1. Lasso: set ramp_size = 0.
              w_i = lambda_1  for  i = 1, ..., p

    2. OSCAR: set ramp_size = p
       w_i = lambda_1 + lambda_2 * (p - i)
       i = 0, ..., p-1  (largest at i=0, smallest at i=p-1)

    3. Ramp: set ramp_size in (0, 1).
       For i = 1, ..., ramp_size = int(np.ceil(ramp_size * p)):
           w_i = lambda_1+lambda_2*(ramp_size = int(np.ceil(ramp_size * p))-i+1)
       For i = ramp_size = int(np.ceil(ramp_size * p)) + 1, ..., p:
           w_i = lambda_1

    Args:
    X : (n x p) numpy array
    Y : (n x r) numpy array
    w : (p,) array of non-negative, non-increasing weights, or None
    lambda_1 : float or None
        Used to construct w.
    lambda_2 : float or None
        Used to construct w.
    ramp_size : float in [0, 1] or None
        Used to build w if w is None. 
    max_iter : int
        Maximum number of FISTA iterations.
    tol : float
        Tolerance for stopping criterion.
    check_type : {'absolute_cost', 'relative_cost', 'solution_diff'}
        - 'absolute_cost': stop if |cost(k+1) - cost(k)| < tol
        - 'relative_cost': stop if |cost(k+1) - cost(k)| / cost(k) < tol
        - 'solution_diff': stop if ||B(k+1) - B(k)||_F < tol
    scale_objective : bool
        If True, the objective function is divided by ||Y||_F^2
        for better numerical scaling.

    Returns
    -------
    B : (p x r) numpy array
        The solution estimate.
    cost_hist : list of float
        The (possibly scaled) objective function values at each iteration.
    """

    # Ensure Y is 2D
    if Y.ndim == 1:
        Y = Y[:, None]

    n, p = X.shape
    _, r = Y.shape

    # First, validate arguments
    
    if check_type not in {'absolute_cost', 'relative_cost', 'solution_diff'}:
        raise ValueError(f"Invalid check_type: {check_type}")

    # Construct weights if not provided
    if w is None:
        if None in (lambda_1, lambda_2, ramp_size):
            raise ValueError(
                "If 'w' is not provided, you must specify all of "
                "'lambda_1', 'lambda_2', and 'ramp_size'."
            )
        
        ramp_size = int(np.ceil(ramp_size * p))
        ramp_size = max(0, min(p, ramp_size))

        # Set the weight vector using the parameters provided by user.  
        # ramp_size = 0       ---> LASSO procedure
        # ramp_size = 1       ---> OSCAR (linear decay) procedure
        # ramp_size = in (0, 1) ---> The first 20% (e.g. ramp_size = 0.2) 
        #                            follow linear decay, while the remaining
        #                            ones are equal to lambda_2.
        w = np.zeros(p)
        for i in range(p):
            if i < ramp_size:
                # linear decay portion
                w[i] = lambda_1 + lambda_2 * (ramp_size - i)
            else:
                # constant portion
                w[i] = lambda_1
    else:
        # User provided w, so validate w and ignore lambdas and ramp_size
        w = np.asarray(w)
        if len(w) != p:
            raise ValueError(f"Weight vector w must have length {p}.")
        if np.any(w < 0):
            raise ValueError("All weights w must be non-negative.")
        
        if any(param is not None for param in (lambda_1, lambda_2, ramp_size)):
            raise ValueError("If 'w' is provided, do not specify 'lambda_1', 'lambda_2', or 'ramp_size'.")
        
    # Ensure non-increasing sort if needed
    # (These formulas are already in non-increasing order if lambda_2 >= 0, ramp_delta >= 0)
    # But just to be safe, we can sort descending:
    w = np.sort(w)[::-1]

    # Lipschitz constant L for the gradient of f(B) = ||Y - X B||^2
    # The gradient is 2 X^T(X B - Y). So L = 2*spectral_norm(X^T X) = 2||X||^2
    # A simpler upper bound is 2 * (largest eigenvalue of X^T X). 
    # We can estimate or compute directly via power method:
    # L = (1/n) * np.linalg.norm(X, 2)**2
    L = (1/n) * (np.linalg.norm(X, 2)**2)

    # Initialize
    B = np.zeros((p, r))
    Z = B.copy()
    t = 1.0

    cost_hist = []

    # Precompute norm of Y if we want to scale the objective
    normY2 = np.sum(Y**2) if scale_objective else 1.0
    def cost_function(B_):
        # f(B) = ||Y - X B||_F^2
        # g(B) = sum_i w_i ||\beta_{i, \cdot}||_2
        residual = Y - X @ B_
        fval = (1/(2 * n)) * np.sum(residual**2) # Perform square Frobenius norm
        row_norms_ = np.linalg.norm(B_, axis=1)
        # Sort row_norms_ in descending order and multiply by w (assumed
        # sorted) to match sum_i w_i || \beta_{i·}||_2 with sorted row norms
        sorted_norms = np.sort(row_norms_)[::-1]  # descending
        gval = np.sum(w * sorted_norms)
        return (fval + gval) / normY2

    for it in range(max_iter):
        B_old = B.copy()

        # Gradient step on Z
        grad_f_Z = (1/n) * X.T @ (X @ Z - Y) # shape: (p, r)
        # Update with step size 1/L
        V = Z - (1 / L) * grad_f_Z  # shape: (p, r)

        # Prox step
        B = prox_growl(V, (1.0/L)*w)  # scale weights by (1/L)
    
        # FISTA update (Nesterov momentum step)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = B + ((t - 1.0) / t_new) * (B - B_old)
        t = t_new

        # Check stopping criterion
        cost_val = cost_function(B)
        cost_hist.append(cost_val)

        # Stopping criterion
        if it > 0:
            
            if check_type == 'absolute_cost': # Absolute change in cost
        
                if verbose:
                    print(f"Iteration: {it + 1}, Obj. func. value = {cost_val:.6f}")
                    
                if abs(cost_hist[-1] - cost_hist[-2]) < tol:
                    break
                    
            elif check_type == 'relative_cost':
                
                if verbose:
                    print(f"Iteration: {it + 1}, Obj. func. value = {cost_val:.6f}")
                    
                # Relative change in cost
                denom = abs(cost_hist[-2]) + 1e-12
                rel_diff = abs(cost_hist[-1] - cost_hist[-2]) / denom
                
                if rel_diff < tol:
                    break
                    
            elif check_type == 'solution_diff':
                # Check how much B changed
                diff_norm = np.linalg.norm(B - B_old, 'fro')

                if verbose:
                    print(f"Iteration: {it + 1}, change in B = {diff_norm:.6f}")
                                    
                if diff_norm < tol:
                    break
                    
            # else:
            #     if abs(cost_hist[-1] - cost_hist[-2]) < tol:
            #         break
        
    else: # Only runs if no 'break' occurred
        print("Max iterations in FISTA algorithm for growl reached without convergence.")
        print(f"{it}")

    return B, cost_hist