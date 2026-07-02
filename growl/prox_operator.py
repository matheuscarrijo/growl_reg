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

    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    # 1. Sort |v| in descending order, keep track of the sort index
    abs_v = np.abs(v)
    sort_idx = np.argsort(-abs_v)  # indices that sort abs_v in descending order
    abs_v_sorted = abs_v[sort_idx]

    # 2. Threshold with the weights: z_i = |v|_(i) - w_i
    z = abs_v_sorted - w

    # 3. Project z onto the non-increasing cone with the Pool Adjacent
    #    Violators (PAV) algorithm. A stack of (block mean, block size) pairs
    #    is maintained; whenever a new block's mean exceeds the mean of the
    #    block before it, the two blocks are merged and the check is repeated
    #    against the (new) previous block. This backward re-merging is what
    #    makes the projection exact -- a single forward pass that only pools
    #    adjacent raw violations does not solve the isotonic problem.
    block_means = []
    block_sizes = []
    for val in z:
        mean = float(val)
        size = 1
        while block_means and block_means[-1] < mean:
            prev_mean = block_means.pop()
            prev_size = block_sizes.pop()
            mean = (mean * size + prev_mean * prev_size) / (size + prev_size)
            size += prev_size
        block_means.append(mean)
        block_sizes.append(size)

    z_proj = np.empty_like(z)
    pos = 0
    for mean, size in zip(block_means, block_sizes):
        z_proj[pos:pos + size] = mean
        pos += size

    # 4. Clip at zero (projection onto the non-increasing, non-negative cone),
    #    re-map back to the original order, and restore signs.
    z_proj = np.maximum(z_proj, 0.0)
    v_final = np.zeros_like(v)
    v_final[sort_idx] = z_proj

    return np.sign(v) * v_final

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