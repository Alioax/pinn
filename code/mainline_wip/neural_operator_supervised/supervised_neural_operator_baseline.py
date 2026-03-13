"""
Supervised Neural Operator (DeepONet) for 1D Advection-Dispersion (dimensionless).

Learns the operator log Pe -> C*(x*, t*) from analytical data for the dimensionless PDE:
    C*_t* + C*_x* - (1/Pe) C*_{x*x*} = 0

with boundary/initial conditions:
    C*(0, t*) = 1  (inlet)
    C*(1, t*) = 0  (outlet, far-field approximation)
    C*(x*, 0) = 0  (initial condition)

The DeepONet maps parameter (log Pe) to the full field C*(x*, t*) across a range of Peclet numbers.
All parameters are configurable at the top of the file.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path
from tqdm import tqdm

# Import analytical solution from sibling folder
sys.path.append(str(Path(__file__).parent.parent / "analytical_solution"))
from analytical_solution import analytical_solution

# Visualization settings
mpl.rcParams["figure.dpi"] = 800
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["#FF5F05", "#13294B", "#009FD4", "#FCB316",
           "#006230", "#007E8E", "#5C0E41", "#7D3E13"]
)

# ============================================================================
# Configuration - Edit parameters here
# ============================================================================

# Physics parameters (dimensionless)
Pe_min = 1
Pe_max = 1e5
t_final_star = 1.0

# Data (finer nx helps capture sharp fronts at high Pe)
num_pe_train = 100
nx = 128
nt = 64
train_fraction = 0.8

# Model architecture (DeepONet)
# Branch: encodes log Pe only (single scalar) -> keep small (1 layer, 6 neurons)
# Trunk: larger to represent sharp spatial fronts (3 layers, 32 neurons)
branch_layers = 1
branch_neurons = 6
trunk_layers = 3
trunk_neurons = 32
latent_dim = 32
activation_name = "Tanh"

# Training parameters
num_epochs = 2000
batch_size = 8
lr = 0.001

# Plotting parameters (dimensionless)
times_tstar = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
pe_values_to_plot = [10, 50, 1e2, 500, 1e3, 1e4, 1e5]
num_spatial_points = 500
plot_dpi = 800

# Derived parameters
logPe_min = np.log(Pe_min)
logPe_max = np.log(Pe_max)

# ============================================================================
# Build training data (analytical C* on grid for each Pe)
# ============================================================================

script_dir = Path(__file__).parent
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

torch.manual_seed(123456789)
np.random.seed(123456789)

x_star_1d = np.linspace(0, 1, nx).astype(np.float32)
t_star_1d = np.linspace(0, 1, nt).astype(np.float32)
X_star, T_star = np.meshgrid(x_star_1d, t_star_1d, indexing="xy")
# Flatten for analytical_solution: (nx*nt,) each
x_flat = X_star.ravel()
t_flat = T_star.ravel()
n_points = nx * nt
grid_points = np.stack([x_flat, t_flat], axis=1)

pe_train_all = np.logspace(np.log10(Pe_min), np.log10(Pe_max), num=num_pe_train).astype(np.float32)
log_pe_train_all = np.log(pe_train_all)

def build_c_star_grid(pe):
    """Compute dimensionless C* on the fixed grid for a given Pe (per time slice, t scalar)."""
    rows = []
    for t_val in t_star_1d:
        if t_val < 1e-10:
            rows.append(np.zeros(nx, dtype=np.float32))
        else:
            c = analytical_solution(
                x_star_1d, t_val,
                U_param=1.0,
                D_param=1.0 / pe,
                C_0_param=1.0,
            )
            rows.append(np.asarray(c, dtype=np.float32))
    return np.stack(rows, axis=0)

# Precompute all training fields
print("Building training data from analytical solution...")
C_star_list = [build_c_star_grid(pe) for pe in pe_train_all]
# Stack to (num_pe_train, nt, nx); flatten to (num_pe_train, n_points) for dataset
C_star_array = np.stack(C_star_list, axis=0).reshape(num_pe_train, n_points)

# Train/val split
n_train = int(num_pe_train * train_fraction)
indices = np.random.permutation(num_pe_train)
train_idx = indices[:n_train]
val_idx = indices[n_train:]

# ============================================================================
# Dataset
# ============================================================================

class PecletDataset(Dataset):
    """Dataset of (log_pe, C_star_grid_flat) for supervised DeepONet training."""

    def __init__(self, log_pe, C_star_flat):
        self.log_pe = torch.tensor(log_pe, dtype=torch.float32).unsqueeze(1)
        self.C_star_flat = torch.tensor(C_star_flat, dtype=torch.float32)

    def __len__(self):
        return len(self.log_pe)

    def __getitem__(self, i):
        return self.log_pe[i], self.C_star_flat[i]

train_dataset = PecletDataset(log_pe_train_all[train_idx], C_star_array[train_idx])
val_dataset = PecletDataset(log_pe_train_all[val_idx], C_star_array[val_idx])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

# Optional: plot training Peclet coverage
plt.figure(figsize=(6.5, 4))
plt.hist(log_pe_train_all, bins=20, color="C1", edgecolor="black", alpha=0.7)
plt.xlabel("log(Pe)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Training Peclet Numbers (log scale)", fontsize=14, pad=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(results_dir / "training_pe_coverage.png"), dpi=plot_dpi, bbox_inches="tight")
plt.savefig(str(results_dir / "training_pe_coverage.pdf"), format="pdf", bbox_inches="tight")
plt.close()
print(f"Training Peclet coverage plot saved to: {results_dir / 'training_pe_coverage.png'}")

# ============================================================================
# DeepONet model
# ============================================================================

activation_map = {
    "Tanh": nn.Tanh,
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
}
activation_cls = activation_map[activation_name]
gain = nn.init.calculate_gain("tanh")


class DeepONet(nn.Module):
    """DeepONet: branch(log Pe) and trunk(x*, t*) with inner product + sigmoid -> C*.
    Forward supports pointwise (training) and grid (plotting) modes.
    """

    def __init__(self):
        super(DeepONet, self).__init__()
        branch_layers_list = []
        in_f = 1
        for _ in range(branch_layers):
            branch_layers_list.append(nn.Linear(in_f, branch_neurons))
            branch_layers_list.append(activation_cls())
            in_f = branch_neurons
        branch_layers_list.append(nn.Linear(in_f, latent_dim))
        self.branch = nn.Sequential(*branch_layers_list)

        trunk_layers_list = []
        in_f = 2
        for _ in range(trunk_layers):
            trunk_layers_list.append(nn.Linear(in_f, trunk_neurons))
            trunk_layers_list.append(activation_cls())
            in_f = trunk_neurons
        trunk_layers_list.append(nn.Linear(in_f, latent_dim))
        self.trunk = nn.Sequential(*trunk_layers_list)

        self._init_weights()

    def _init_weights(self):
        for m in self.branch:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, log_pe, x_star, t_star):
        """
        Pointwise: log_pe (N, 1), x_star (N, 1), t_star (N, 1) -> (N,).
        Grid: log_pe (B, 1), x_star (P,), t_star (P,) -> (B, P).
        """
        x_ = x_star.squeeze(-1) if x_star.dim() > 1 else x_star
        t_ = t_star.squeeze(-1) if t_star.dim() > 1 else t_star
        points = torch.stack([x_, t_], dim=-1)
        b = self.branch(log_pe)
        t = self.trunk(points)
        if log_pe.shape[0] == points.shape[0]:
            return torch.sigmoid((b * t).sum(dim=-1))
        return torch.sigmoid(torch.mm(b, t.t()))


model = DeepONet()

# ============================================================================
# Training loop
# ============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

losses_train = []
losses_val = []
grid_points_device = grid_points_tensor

pbar = tqdm(range(num_epochs), desc="Training DeepONet")
for epoch in pbar:
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for log_pe_batch, C_target_batch in train_loader:
        optimizer.zero_grad()
        x_star = grid_points_tensor[:, 0]
        t_star = grid_points_tensor[:, 1]
        pred = model(log_pe_batch, x_star, t_star)
        loss = loss_fn(pred, C_target_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    losses_train.append(epoch_loss / n_batches)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        n_val = 0
        for log_pe_batch, C_target_batch in val_dataset:
            log_pe_batch = log_pe_batch.unsqueeze(0)
            C_target_batch = C_target_batch.unsqueeze(0)
            pred = model(log_pe_batch, x_star, t_star)
            val_loss += loss_fn(pred, C_target_batch).item()
            n_val += 1
        if n_val > 0:
            losses_val.append(val_loss / n_val)
        else:
            losses_val.append(0.0)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        pbar.set_postfix({
            "train": f"{losses_train[-1]:.4e}",
            "val": f"{losses_val[-1]:.4e}",
        })
pbar.close()

# Save trained model state
model_path = results_dir / "supervised_neural_operator_baseline_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# ============================================================================
# Loss plot
# ============================================================================

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(losses_train, label="Train")
ax.plot(losses_val, label="Val")
ax.set_ylabel("MSE")
ax.set_xlabel("Epoch")
ax.legend()
ax.spines[["right", "top"]].set_visible(False)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(str(results_dir / "loss.png"), dpi=plot_dpi, bbox_inches="tight")
plt.savefig(str(results_dir / "loss.pdf"), format="pdf", bbox_inches="tight")
plt.close()
print(f"Loss plot saved to: {results_dir / 'loss.png'}")

# ============================================================================
# Concentration profile plots (dimensionless)
# ============================================================================

x_star_plot = np.linspace(0, 1, num_spatial_points)
x_star_plot_tensor = torch.tensor(x_star_plot, dtype=torch.float32)

model.eval()
with torch.no_grad():
    for pe_plot in pe_values_to_plot:
        log_pe_plot = np.log(pe_plot)
        log_pe_plot_tensor = torch.tensor([[log_pe_plot]], dtype=torch.float32)

        plt.figure(figsize=(5, 4))
        plt.plot([], [], linewidth=2, linestyle="-", color="black", label="DeepONet")
        plt.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

        for idx, t_star in enumerate(reversed(times_tstar)):
            color = f"C{idx % 8}"

            t_star_plot_tensor = torch.full((num_spatial_points,), t_star, dtype=torch.float32)
            pred = model(log_pe_plot_tensor, x_star_plot_tensor, t_star_plot_tensor)
            C_star = pred.squeeze().cpu().numpy()
            plt.plot(x_star_plot, C_star, linewidth=2, linestyle="-", color=color)

            U_fake = 1.0
            C0_fake = 1.0
            D_fake = 1.0 / pe_plot
            C_analytical = analytical_solution(
                x_star_plot, t_star,
                U_param=U_fake, D_param=D_fake, C_0_param=C0_fake,
            )
            plt.plot(x_star_plot, C_analytical, linewidth=2, linestyle="--", color=color, alpha=0.7)
            plt.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"t* = {t_star}")

        plt.xlabel("x* (dimensionless)", fontsize=12)
        plt.ylabel("C* (dimensionless)", fontsize=12)
        plt.title(f"C* profiles (Pe = {pe_plot:g})", fontsize=12)

        legend = plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.6),
            ncol=4,
            frameon=False,
            fontsize=10,
            labelspacing=0.5,
            columnspacing=1.2,
        )
        for text in legend.get_texts():
            text.set_color("black")
            text.set_alpha(1.0)

        plt.xlim(0, 1)
        plt.ylim(-0.25, 1.1)

        ax_ = plt.gca()
        grid_alpha = 0.3
        grid_color = "black"
        grid_linewidth = 0.4

        ax_.grid(True, axis="x", alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)

        x_ticks = ax_.get_xticks()
        xgridlines = ax_.get_xgridlines()
        indices_to_hide = []
        for i, tick_pos in enumerate(x_ticks):
            if abs(tick_pos - 0.0) < 1e-6 or abs(tick_pos - 1.0) < 1e-6:
                indices_to_hide.append(i)
        for i in indices_to_hide:
            if i < len(xgridlines):
                xgridlines[i].set_visible(False)

        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)
        ax_.spines["bottom"].set_visible(True)
        ax_.spines["left"].set_visible(True)
        ax_.spines["bottom"].set_color(grid_color)
        ax_.spines["bottom"].set_alpha(grid_alpha)
        ax_.spines["bottom"].set_linewidth(grid_linewidth)
        ax_.spines["left"].set_color(grid_color)
        ax_.spines["left"].set_alpha(grid_alpha)
        ax_.spines["left"].set_linewidth(grid_linewidth)

        ax_.tick_params(axis="x", which="major",
                        colors=grid_color,
                        width=grid_linewidth,
                        length=0)
        ax_.tick_params(axis="y", which="major",
                        colors=grid_color,
                        width=grid_linewidth,
                        length=0)

        for label in ax_.get_xticklabels():
            label.set_color("black")
            label.set_alpha(1.0)
        for label in ax_.get_yticklabels():
            label.set_color("black")
            label.set_alpha(1.0)

        ax_.xaxis.label.set_color("black")
        ax_.xaxis.label.set_alpha(1.0)
        ax_.yaxis.label.set_color("black")
        ax_.yaxis.label.set_alpha(1.0)

        plt.tight_layout()

        plot_path = results_dir / f"{pe_plot} Cstar_profiles.png"
        plt.savefig(str(plot_path), dpi=plot_dpi, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")
        plot_pdf_path = results_dir / f"{pe_plot} Cstar_profiles.pdf"
        plt.savefig(str(plot_pdf_path), format="pdf", bbox_inches="tight")
        print(f"PDF saved to: {plot_pdf_path}")

        plt.close()

print("Done.")
