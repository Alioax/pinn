# Neural Operator Architecture: DeepONet for Parametric 1D Transport

This document explains the **DeepONet** (Deep Operator Network) architecture used in this repository to learn the mapping from Peclet number to the dimensionless concentration field. It covers what the branch and trunk do, what their inputs and outputs mean, how they combine to give the solution, and how training data is obtained in both **supervised** and **unsupervised** (physics-informed) settings.

---

## 1. What Problem Does the Neural Operator Solve?

We work with the **dimensionless** 1D advection–dispersion equation:

- **PDE:** \(\displaystyle \frac{\partial C^*}{\partial t^*} + \frac{\partial C^*}{\partial x^*} = \frac{1}{Pe}\,\frac{\partial^2 C^*}{\partial x^{*2}}\)
- **Domain:** \(x^* \in [0,1]\), \(t^* \in [0,1]\)
- **Initial condition:** \(C^*(x^*, 0) = 0\)
- **Boundary conditions:** \(C^*(0, t^*) = 1\) (inlet), \(C^*(1, t^*) = 0\) (outlet)

Here \(C^*\) is dimensionless concentration, and **Pe** (Péclet number) is the single physical parameter that controls how diffusion-dominated vs advection-dominated the transport is. We treat **log Pe** as the parameter and learn the **operator** that maps:

\[
\log Pe \;\longmapsto\; C^*(\cdot, \cdot)
\]

i.e., “given a Peclet number, what is the full concentration field \(C^*(x^*, t^*)\)?” The DeepONet is the neural network that approximates this operator.

---

## 2. High-Level Idea: Branch and Trunk

A DeepONet has two sub-networks that work together:

| Sub-network | Role | Intuition |
|-------------|------|-----------|
| **Branch** | Encodes the **parameter** (here: \(\log Pe\)) | “Which solution regime are we in?” |
| **Trunk** | Encodes the **query location** \((x^*, t^*)\) | “Where in space-time are we evaluating?” |

The final prediction is a **combination** of the two: the branch output and the trunk output are combined (via an inner product, then a nonlinearity) to produce the scalar \(C^*\) at that location for that parameter. So:

- The **branch** depends only on \(\log Pe\) and outputs a fixed-size vector (the “parameter code”).
- The **trunk** depends only on \((x^*, t^*)\) and outputs a vector of the **same size** (the “location code”).
- The **inner product** of these two vectors (plus sigmoid) gives \(C^*\).

This design separates “which solution” (branch) from “where to evaluate” (trunk), which is exactly what we need for an operator: one branch evaluation per Peclet number, and as many trunk evaluations as we want for different \((x^*, t^*)\).

---

## 3. Branch Network

### Input

- **What:** A single scalar per sample: **\(\log Pe\)**.
- **Shape:** Typically \((N, 1)\): \(N\) is the number of “samples” (e.g. number of Peclet values in a batch, or number of collocation points when each point has its own \(\log Pe\)).
- **Meaning:** \(\log Pe\) identifies the solution. Low \(\log Pe\) → more diffusion; high \(\log Pe\) → sharper advective front. Using \(\log Pe\) (instead of \(Pe\)) spreads the parameter range \([1, 10^5]\) more evenly and keeps scales manageable.

### Architecture

- A small **MLP** (e.g. 2 hidden layers, 64 neurons each, Tanh).
- **Input dimension:** 1.
- **Output dimension:** \(p\) (e.g. \(p = 64\)), the **latent dimension** shared with the trunk.
- **Output meaning:** A \(p\)-dimensional vector that summarizes “everything the network needs to know about this Peclet number” to generate the right concentration field. We call this vector **\(\mathbf{b}\)** (branch output).

### Output

- **Shape:** \((N, p)\) when given \((N, 1)\) inputs.
- **Meaning:** \(\mathbf{b}\) is the **parameter embedding**: one vector per \(\log Pe\). The same \(\log Pe\) always gives the same \(\mathbf{b}\) (for a fixed model), so one branch evaluation per Peclet number is enough to query the field at any number of \((x^*, t^*)\) points.

---

## 4. Trunk Network

### Input

- **What:** The **query point** in dimensionless space-time: **\((x^*, t^*)\)**.
- **Shape:** Typically \((M, 2)\): \(M\) is the number of query points, each with two coordinates \((x^*, t^*)\).
- **Meaning:** “At which dimensionless position and time do we want \(C^*\)?” So the trunk encodes **where** we are in the domain \([0,1] \times [0,1]\).

### Architecture

- A small **MLP** (e.g. 2 hidden layers, 64 neurons each, Tanh).
- **Input dimension:** 2.
- **Output dimension:** \(p\), the **same** latent dimension as the branch.
- **Output meaning:** A \(p\)-dimensional vector that encodes the location \((x^*, t^*)\). We call this vector **\(\mathbf{t}\)** (trunk output).

### Output

- **Shape:** \((M, p)\) when given \((M, 2)\) inputs.
- **Meaning:** \(\mathbf{t}\) is the **location embedding**: one vector per \((x^*, t^*)\). The trunk does not depend on \(\log Pe\); it only knows about space-time. So the same trunk can be reused for every Peclet number—only the branch output changes.

---

## 5. How Branch and Trunk Combine: The Final Output \(C^*\)

The scalar concentration \(C^*\) at one parameter and one location is computed as:

\[
C^* \;=\; \sigma\bigl( \langle \mathbf{b}, \mathbf{t} \rangle \bigr)
\;=\; \sigma\bigl( \mathbf{b}^\top \mathbf{t} \bigr)
\]

where \(\sigma\) is the **sigmoid** function, so \(C^* \in (0, 1)\). In code:

- **Pointwise (one value per row):** When we have \(N\) rows and each row has its own \((\log Pe, x^*, t^*)\), we get \(\mathbf{b}\) of shape \((N, p)\) and \(\mathbf{t}\) of shape \((N, p)\). Then we form the **diagonal** of \(\mathbf{b} \mathbf{t}^\top\), i.e. \((N,)\) values: \(\bigl( (\mathbf{b} * \mathbf{t}).\mathrm{sum}(\mathrm{dim}=-1) \bigr)\), and apply sigmoid.
- **Grid (batch of parameters × many points):** When we have \(B\) Peclet values and \(M\) query points, \(\mathbf{b}\) is \((B, p)\) and \(\mathbf{t}\) is \((M, p)\). Then \(\mathbf{b} \mathbf{t}^\top\) is \((B, M)\): entry \((i,j)\) is the inner product of branch \(i\) with trunk \(j\), i.e. \(C^*\) for parameter \(i\) at point \(j\). Apply sigmoid element-wise.

**Why inner product + sigmoid?**

- The inner product measures how well the “parameter code” and “location code” agree. The network learns branch and trunk so that this agreement is high where \(C^*\) should be large (e.g. near the inlet at early time) and low where \(C^*\) should be small (e.g. far downstream at early time).
- Sigmoid keeps \(C^*\) in \((0,1)\), which matches the dimensionless concentration range.

---

## 6. Summary Table: Inputs and Outputs

| Component | Input | Output | Meaning of output |
|-----------|--------|--------|--------------------|
| **Branch** | \(\log Pe\), shape \((N, 1)\) | \(\mathbf{b}\), shape \((N, p)\) | Parameter embedding: “solution regime” for each \(\log Pe\). |
| **Trunk** | \((x^*, t^*)\), shape \((M, 2)\) | \(\mathbf{t}\), shape \((M, p)\) | Location embedding: “where in space-time” for each query. |
| **Combination** | \(\mathbf{b}\), \(\mathbf{t}\) | \(C^* = \sigma(\mathbf{b}^\top \mathbf{t})\) | Dimensionless concentration at that parameter and location. |

Default sizes in this repo: \(p = 64\), branch/trunk each with 2 hidden layers and 64 neurons, Tanh activations, Xavier initialization.

---

## 7. How Training Data Is Obtained

We have two baselines: **supervised** (data from the analytical solution) and **unsupervised** (no data; only physics-informed losses). Both use the **same** DeepONet architecture and the **same** PDE and BCs; only the loss and the way “training data” is defined differ.

---

### 7.1 Supervised Training (Data-Based)

**Idea:** We **generate** solution fields from the **Ogata–Banks analytical solution** for many Peclet numbers, then train the DeepONet to match those fields. No PDE or BC terms in the loss—only a regression loss (e.g. MSE) between predicted and analytical \(C^*\).

**How the “training data” is built:**

1. **Choose training Peclet numbers.**  
   We take a set of values spanning the range of interest, e.g. \(Pe \in [1, 10^5]\). In practice we use **log-spaced** values (e.g. `np.logspace(0, 5, num=50)`) so that low and high Pe are both well represented. Each training sample is **one** Peclet number.

2. **Choose a fixed grid in \((x^*, t^*)\).**  
   For example \(x^*\) and \(t^*\) each in \([0, 1]\) with \(n_x \times n_t\) points (e.g. \(64 \times 64\)). This grid is the **same** for every Peclet number.

3. **For each training \(Pe\), compute \(C^*\) on the grid.**  
   We use the dimensionless convention: \(U = 1\), \(C_0 = 1\), \(D = 1/Pe\), and interpret \(x^*\) and \(t^*\) as physical space and time (with \(L = 1\), \(T = 1\)). The analytical solution is evaluated at each grid point (per time slice to handle \(t^* = 0\) correctly). That gives one **field** \(C^*(x^*, t^*)\) of shape \((n_x, n_t)\) per \(Pe\).

4. **Form the dataset.**  
   Each sample is a pair \((\log Pe, C^*_{\mathrm{grid}})\). So we have as many samples as we have training Peclet numbers. The target is the flattened grid \(C^*_{\mathrm{grid}}\) of length \(n_x \cdot n_t\).

**Training step:**

- Sample a **batch** of Peclet numbers (e.g. 8).
- For each, we already have the target field on the fixed grid.
- **Forward:** Branch takes the batch of \(\log Pe\) → \((B, p)\). Trunk takes the **same** grid points (same for all samples in the batch) → \((M, p)\) with \(M = n_x \cdot n_t\). Combine to get predicted \(C^*\) of shape \((B, M)\).
- **Loss:** MSE between predicted \((B, M)\) and target \((B, M)\).
- **Backward and update** branch and trunk.

So in supervised training, “training data” = a set of **(parameter, full solution field)** pairs generated from the analytical solution. The DeepONet learns to interpolate or generalize to new Peclet numbers from these examples.

---

### 7.2 Unsupervised Training (Physics-Informed, No Data)

**Idea:** We do **not** use any concentration data. We only impose the **PDE**, the **initial condition**, and the **boundary conditions** by defining losses at **collocation points**. The “training data” is just the choice of where we evaluate these residuals; the targets come from physics (residual = 0, IC/BC values), not from the analytical solution.

**How the “training data” is defined (each epoch):**

1. **PDE collocation points.**  
   We sample many points \((x^*, t^*, \log Pe)\) in the interior: \(x^* \sim U(0,1)\), \(t^* \sim U(0, t^*_{\mathrm{final}})\), \(\log Pe \sim U(\log Pe_{\min}, \log Pe_{\max})\). Typical count: e.g. \(250 \times 250\). At each point we will evaluate the **PDE residual** (using automatic differentiation of the DeepONet output w.r.t. \(x^*\) and \(t^*\)). No target concentration is provided—we only ask that the residual be zero.

2. **Initial condition (IC) points.**  
   We sample \(x^*\) and \(\log Pe\) (e.g. 250 points), and set \(t^* = 0\). The “target” is \(C^* = 0\) at these points.

3. **Inlet BC points.**  
   We sample \(t^*\) and \(\log Pe\), and set \(x^* = 0\). The target is \(C^* = 1\).

4. **Outlet BC points.**  
   We sample \(t^*\) and \(\log Pe\), and set \(x^* = 1\). The target is \(C^* = 0\).

So each epoch we have four sets of **(inputs, physics-based target)**:

- **PDE:** inputs = \((x^*, t^*, \log Pe)\); “target” = residual = 0 (so loss = mean residual²).
- **IC / inlet / outlet:** inputs = \((x^*, t^*, \log Pe)\); target = 0, 1, or 0 respectively (loss = MSE).

**Training step:**

- Resample all four sets (optional but common; we do it every epoch).
- **PDE:** Forward the DeepONet at PDE collocation points (pointwise: one \((\log Pe, x^*, t^*)\) per row). Compute \(C^*\), then \(\partial C^*/\partial t^*\), \(\partial C^*/\partial x^*\), \(\partial^2 C^*/\partial x^{*2}\) by autodiff. Form \(Pe = \exp(\log Pe)\) and the residual; PDE loss = mean(residual²).
- **IC / BC:** Forward at IC, inlet, and outlet points; IC loss = MSE(\(C^*\), 0), inlet loss = MSE(\(C^*\), 1), outlet loss = MSE(\(C^*\), 0).
- **Total loss** = weighted sum (e.g. all weights 1). Backward and update.

So in unsupervised training, there is **no** concentration dataset. The only “data” are the **positions** \((x^*, t^*)\) and \(\log Pe\) where we enforce the PDE and BCs. The network learns the operator purely from physics.

---

## 8. Side-by-Side Comparison

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| **Use of analytical solution** | To generate target fields \(C^*\) on a grid for each training \(Pe\). | Not used in training; only for optional plot overlay. |
| **“Training data”** | Pairs \((\log Pe, C^*_{\mathrm{grid}})\) from analytical solution. | Collocation points \((x^*, t^*, \log Pe)\) and physics-based targets (residual = 0, IC/BC values). |
| **Loss** | MSE between predicted and analytical \(C^*\) on the grid. | PDE residual² + IC MSE + inlet MSE + outlet MSE. |
| **Gradients** | Only w.r.t. network weights. | W.r.t. weights and (for PDE) w.r.t. \(x^*\), \(t^*\) for derivative computation. |
| **Typical use** | When we trust the analytical solution and want a fast surrogate. | When we want a solution that satisfies the PDE/BCs without using any concentration data. |

---

## 9. Where to Find the Code

- **Supervised:** `code/archive/legacy_mainline_wip/neural_operator_supervised/supervised_neural_operator_baseline.py`  
  - Data build: analytical \(C^*\) on a fixed grid for many \(Pe\); Dataset/DataLoader over \((\log Pe, C^*_{\mathrm{flat}})\).  
  - Forward: grid mode (batch of Pe × grid points).  
  - Loss: MSE.

- **Unsupervised:** `code/archive/legacy_mainline_wip/neural_operator_unsupervised/unsupervised_neural_operator_baseline.py`  
  - No dataset; each epoch resamples PDE, IC, inlet, outlet collocation points.  
  - Forward: pointwise at collocation points; PDE derivatives via `torch.autograd.grad`.  
  - Loss: PDE + IC + inlet + outlet.

**Current canonical PINO (Report 3):** `code/homogeneous_pe/pino/pinn1d_transport_parametric_neural_operator.py`  
**Current canonical PINO (Report 4, CFL):** `code/homogeneous_cfl/pino_cfl/pinn1d_cfl_pe_parametric_neural_operator.py`

Both archived and current scripts use the same branch/trunk layout, same latent dimension, and same combination (inner product + sigmoid). The architecture document you are reading applies to all of them.
