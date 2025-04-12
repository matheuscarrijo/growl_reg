# Group Ordered Weighted \(\ell_1\) Norm (GrOWL)

This repository provides a Python implementation of the **Group Ordered Weighted
 \(\ell_1\) Norm (GrOWL)** regularization using the **Proximal Operator** and 
 the **Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)**. It contains 
 various modules that together solve the following general optimization problem:

\[
\min_{\beta} \; \frac{1}{2}\| y - X\beta \|_2^2 + \text{GrOWL}(\beta; \mathbf{w}, G),
\]

where \(X \in \mathbb{R}^{n \times p}\) is the data matrix, 
\(y \in \mathbb{R}^{n}\) is the response vector, \(\beta \in \mathbb{R}^{p}\) 
are the coefficients we want to estimate, and 
\(\text{GrOWL}(\beta; \mathbf{w}, G)\) is the 
**Group Ordered Weighted \(\ell_1\) Norm** of \(\beta\), parameterized by 
weights \(\mathbf{w}\) and a grouping structure \(G\).

---

## Mathematical Background

### 1. Group Ordered Weighted \(\ell_1\) Norm (GrOWL)

The standard Ordered Weighted \(\ell_1\) (OWL) norm arranges the coefficients of
 \(\beta\) in non-increasing order of magnitude and applies descending weights 
 \(w_1 \ge w_2 \ge \cdots \ge w_p \ge 0\). The **group** extension, **GrOWL**, 
 extends this idea by:

1. Partitioning the coefficient vector \(\beta\) into predefined groups (blocks).
2. Computing the (sorted) norms of these groups.
3. Applying descending weights to the group norms instead of individual 
coefficients.

Formally, assume \(\beta\) is partitioned into \(G\) groups, 
\( \{\beta_1, \beta_2, \ldots, \beta_G\}\). Then, for each group \(g\), 
we compute the Euclidean norm \(\|\beta_g\|_2\). We reorder these group norms in
 a non-increasing manner:
\[
\|\beta_{(1)}\|_2 \ge \|\beta_{(2)}\|_2 \ge \cdots \ge \|\beta_{(G)}\|_2,
\]
where \((1),(2),\ldots,(G)\) is a permutation of \(1,2,\ldots,G\) that sorts the
norms from largest to smallest. Given a weight vector 
\(\mathbf{w} = (w_1, w_2, \ldots, w_G)\) with
\(w_1 \ge w_2 \ge \cdots \ge w_G \ge 0\), we define **GrOWL** as:

\[
\text{GrOWL}(\beta; \mathbf{w}, G) 
= \sum_{g=1}^{G} w_g \|\beta_{(g)}\|_2.
\]

### 2. Optimization Problem

We want to solve a regularized least squares problem of the form:

\[
\min_{\beta \in \mathbb{R}^p} \; \frac{1}{2}\|y - X\beta\|_2^2 + \lambda \, 
\text{GrOWL}(\beta; \mathbf{w}, G),
\]

where \(\lambda\) is a regularization parameter that controls the trade-off 
between the data fidelity term and the regularization term.

---

## Repository Structure

Below are the important modules in this project and their functionalities:

1. **`__init__.py`**  
   This file makes the folder into a Python package. It may contain import 
   statements that expose key functions or classes at the package level.

2. **`base.py`**  
   Defines base classes or utility functions used across different parts of the 
   solver. This can include:
   - Data structures for storing problem parameters (e.g., \(X\), \(y\), groups).
   - Common helper methods for logging and checks.

3. **`prox_operator.py`**  
   Implements the **Proximal Operator** associated with the GrOWL penalty. The 
   proximal operator is essential for iterative algorithms (like FISTA) that 
   solve optimization problems involving non-smooth terms.  
   - **Key function**: `prox_growl(...)`  
     This function performs the group-sorting, computes group norms, and applies
     the weighted shrinkage (or thresholding) needed to reflect the GrOWL 
     penalty in each iteration.

4. **`fista_solver.py`**  
   Implements the **Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)**, 
   an accelerated first-order method. The algorithm proceeds by:
   1. Computing the gradient of the smooth part of the objective 
   (\(\frac{1}{2}\|y - X\beta\|^2\)).
   2. Making a gradient-descent-like step.
   3. Applying the **proximal operator** to incorporate the GrOWL penalty.
   4. Updating an acceleration parameter to speed up convergence.  
   - **Key function**: `fista_growl(...)`  
     Solves the GrOWL-regularized problem using FISTA. Takes in:
     - Data \((X, y)\)
     - Regularization parameter \(\lambda\)
     - Weights \(\mathbf{w}\)
     - Group structure
     - Step size or Lipschitz constant
     - Maximum iterations and tolerance  
     Returns the estimated coefficient vector \(\beta\).

5. **`growl_example.py`**  
   Provides an example script showing how to:
   - Define or load input data \((X, y)\).
   - Specify groups and corresponding weights.
   - Call the FISTA solver with the GrOWL proximal operator.
   - Inspect or visualize the results (e.g., solution path or final \(\beta\)).

---

## Usage

1. **Install/Clone the repository**  
   Simply clone or download this repository to your local machine:
   ```bash
   git clone <this-repository-url>
