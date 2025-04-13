import numpy as np 

def prox_owl(v, w):
    """
    Calculates the proximal operator for the OWL norm on vectors. Given a 
    vector v and sorted non-increasing weights w, computes the OWL proximal 
    operator, given by 

        prox_{\Omega_w}(v) = argmin_x (1/2) || x - v ||_2^2 + \Omega_w(x),
 
    where Omega_w(x) = \sum_i w_i |x|_(i) is the OWL norm, and w is of the same
    dimension as x and sorted in non-increasing order: w_1 >= ... >= w_p >= 0. 
    Also |x|_(i) are the components of v reordered in non-increasing order (by
    absolute value).

    The solution of the min problem is given in equation (24) of the paper
    "The ordered weighted L1 norm - atomic formulation, projections, and 
    Algorithms" by Zeng and M. Figueiredo (2015). It relies on the following
    algorithm:

    1. Sorting |v| in descending order to get |v|_(i).
    2. Performing thresholding with w: compute z_i = max(|v|_(i) - w_i, 0).
    3. Applying an isotonic regression step to ensure the result remains
       non-increasing when ordered by absolute value.
    4. Restoring the original order and signs of v.

    The method ensures that the sparsity and ordering properties of the OWL
    norm are preserved, which generalizes the soft-thresholding operator used 
    in Lasso to a structured penalization framework.

    Args:
        v (np.ndarray): Input vector.
        w (np.ndarray): Non-increasing sequence of weights.

    Returns:
        np.ndarray: The result of applying the OWL proximal operator to v.
    """

    # 1. Sort |v| in descending order, keep track of the sort index
    abs_v = np.abs(v)
    sort_idx = np.argsort(-abs_v)  # indices that sort abs_v in descending order
    abs_v_sorted = abs_v[sort_idx]

    # 2. Apply Pool Adjacent Violators (PAV) algorithm for isotonic regression

    # Step 1: z_i = abs_v_sorted[i] - w[i], then sort z in descending order if
    # needed
    z = abs_v_sorted - w

    # Step 2: the "pool adjacent violators" for z in descending order
    #         ensuring the final vector is sorted and each entry >= 0
    z_proj = np.zeros_like(z)

    start = 0
    while start < len(z):
        end = start
        # average the block [start, end] if any negativity or non-increasing 
        # violation arises
        block_sum = z[start]
        # Merge blocks while the sorted order is violated (i.e. z[end+1] > z[end])
        while end + 1 < len(z) and z[end+1] > z[end]:
            end += 1
            block_sum += z[end]

        # block is from start to end
        avg = block_sum / (end - start + 1)
        for j in range(start, end + 1):
            z_proj[j] = max(avg, 0)
        start = end + 1

    # Re-map back to original order and restore sign
    v_final = np.zeros_like(v)
    for i in range(len(v)):
        idx = sort_idx[i]
        v_final[idx] = np.sign(v[idx]) * z_proj[i]

    return v_final

def prox_growl(V, w):
    """
    Calculates proximal operator given by 
    
       prox_G(V) = argmin_B (1/2)||B - V||_F^2 + sum_i w_i ||\beta_{i\cdot}||_2,
    
    given that w is sorted in non-increasing order (w_1 >= ... >= w_p >= 0).
    """
    p, r = V.shape
    # Compute row norms
    row_norms = np.linalg.norm(V, axis=1)

    # If a row is zero, we keep it zero (to avoid dividing by zero).
    # So handle those carefully.
    # Step 1: compute the prox for the row norms via the OWL prox
    shrunk_row_norms = prox_owl(row_norms, w)
    # Step 2: rescale each row
    out = np.zeros_like(V)
    for i in range(p):
        norm_i = row_norms[i]
        if norm_i > 0:
            out[i, :] = (shrunk_row_norms[i] / norm_i) * V[i, :]
        else:
            # row was zero, keep it zero
            out[i, :] = V[i, :]

    return out