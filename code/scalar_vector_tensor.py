#!/usr/bin/env python3
"""
3-panel visualization:
1) scalar field (temperature-like)
2) vector field (velocity)
3) tensor field (symmetric part of velocity gradient) as ellipses
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

# ------------------------------------------------------------
# Domain and sampling
# ------------------------------------------------------------
L = 2.0
N = 25
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

# Fine grid for scalar field
N2 = 200
xx = np.linspace(-L, L, N2)
yy = np.linspace(-L, L, N2)
XX, YY = np.meshgrid(xx, yy)

# ------------------------------------------------------------
# Custom diverging colormap: cyan → black → orange
# ------------------------------------------------------------
diverging_colors = [
    (0.0, (0.2, 0.8, 1.0)),  # cyan
    (0.5, (0.0, 0.0, 0.0)),  # black
    (1.0, (1.0, 0.5, 0.0)),  # orange
]
scalar_cmap = mcolors.LinearSegmentedColormap.from_list(
    "cyan_black_orange", diverging_colors
)

# ------------------------------------------------------------
# Velocity field definition
# ------------------------------------------------------------
def velocity_field(x, y):
    # radial decay
    r2 = x**2 + y**2
    decay = np.exp(-0.4 * r2)

    # swirl component
    swirl_x = -y
    swirl_y = x

    # radial component with sinusoidal modulation
    radial = np.sin(1.5 * np.sqrt(r2 + 1e-6))  # avoid division by zero
    radial_x = radial * x
    radial_y = radial * y

    # shear modulation
    shear_x = 0.5 * np.sin(2 * y)
    shear_y = 0.5 * np.cos(2 * x)

    u = decay * (swirl_x + 0.7 * radial_x + shear_x)
    v = decay * (swirl_y + 0.7 * radial_y + shear_y)
    return u, v

# Velocity field on coarse grid (for quiver & tensor)
VX, VY = velocity_field(X, Y)

# ------------------------------------------------------------
# Scalar field (temperature-like) on fine grid
# ------------------------------------------------------------
scalar_field = np.exp(-0.1 * (XX**2 + YY**2)) * (
    0.7 * np.sin(2.8 * XX + 1.2 * YY)
    - 0.5 * np.cos(3.4 * YY - 1.1 * XX)
    + 0.6 * np.sin(2.2 * XX - 2.6 * YY) * np.cos(2.8 * XX + 1.9 * YY)
)

# Rescale to symmetric range, then to [0,1] for the colormap
scalar_min = scalar_field.min()
scalar_max = scalar_field.max()
scalar_field = 4.0 * ((scalar_field - scalar_min) / (scalar_max - scalar_min + 1e-9)) - 2.0
scalar_for_cmap = (scalar_field + 1.0) / 2.0  # [-1,1] → [0,1]
scalar_rgba = scalar_cmap(np.clip(scalar_for_cmap, 0.0, 1.0))

# ------------------------------------------------------------
# Tensor field: full symmetric part of velocity gradient
# (this gives genuine ellipse eccentricity)
# ------------------------------------------------------------
# np.gradient: first axis ~ y, second axis ~ x
dVx_dy, dVx_dx = np.gradient(VX, y, x)
dVy_dy, dVy_dx = np.gradient(VY, y, x)

# Symmetric part S = 0.5 (grad v + grad v^T)
Sxx = dVx_dx
Syy = dVy_dy
Sxy = 0.5 * (dVx_dy + dVy_dx)  # == Syx

# ------------------------------------------------------------
# Figure setup: 3 equal-width panels + one thin colorbar column
# ------------------------------------------------------------
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 5), facecolor="black")

gs = fig.add_gridspec(
    1, 3, width_ratios=[20, 20, 20], wspace=0.25
)

ax0 = fig.add_subplot(gs[0, 0])  # scalar
# cax = fig.add_subplot(gs[0, 1])  # colorbar removed
ax1 = fig.add_subplot(gs[0, 1])  # vector
ax2 = fig.add_subplot(gs[0, 2])  # tensor

axes = [ax0, ax1, ax2]
for ax in axes:
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

# ------------------------------------------------------------
# Panel 1: Scalar field
# ------------------------------------------------------------
im_s = ax0.imshow(
    scalar_rgba,
    origin="lower",
    extent=(-L, L, -L, L),
)
ax0.set_title("Scalar field", color="white")
ax0.set_xlabel("s", color="white")
ax0.set_ylabel("t", color="white")
ax0.set_aspect("equal")

scalar_mappable = plt.cm.ScalarMappable(
    norm=Normalize(vmin=-1, vmax=1),
    cmap=scalar_cmap,
)
scalar_mappable.set_array([])
# cbar = fig.colorbar(scalar_mappable, cax=cax)
# cbar.outline.set_edgecolor('white')

# ------------------------------------------------------------
# Panel 2: Vector field (quiver)
# ------------------------------------------------------------
ax1.set_title("Vector field", color="white")
ax1.set_xlabel("s", color="white")
ax1.set_ylabel("t", color="white")
ax1.set_aspect("equal")
ax1.set_xlim(-L, L)
ax1.set_ylim(-L, L)

ax1.quiver(X, Y, VX, VY, color="cyan", pivot="mid")

# ------------------------------------------------------------
# Panel 3: Tensor field (ellipses)
# ------------------------------------------------------------
ax2.set_title("Tensor field", color="white")
ax2.set_xlabel("s", color="white")
ax2.set_ylabel("t", color="white")
ax2.set_aspect("equal")
ax2.set_xlim(-L, L)
ax2.set_ylim(-L, L)

step = 2   # subsample grid to avoid clutter
eps = 1e-6

# Compute a global eigenvalue scale so ellipses are nicely sized
max_eig = 0.0
for i in range(0, N, step):
    for j in range(0, N, step):
        Txx = Sxx[i, j]
        Tyy = Syy[i, j]
        Txy = Sxy[i, j]
        T = np.array([[Txx, Txy],
                      [Txy, Tyy]])
        vals = np.linalg.eigvalsh(T)
        max_eig = max(max_eig, np.max(np.abs(vals)))

if max_eig < eps:
    max_eig = 1.0

scale = 0.5 / max_eig  # global scaling factor

for i in range(0, N, step):
    for j in range(0, N, step):
        Txx = Sxx[i, j]
        Tyy = Syy[i, j]
        Txy = Sxy[i, j]
        T = np.array([[Txx, Txy],
                      [Txy, Tyy]])

        vals, vecs = np.linalg.eig(T)
        lam1, lam2 = vals

        if abs(lam1) < eps and abs(lam2) < eps:
            continue

        # principal direction: eigenvector of largest |lambda|
        idx = np.argmax(np.abs(vals))
        v1 = vecs[:, idx]
        angle = np.degrees(np.arctan2(v1[1], v1[0]))

        width = scale * abs(lam1)
        height = scale * abs(lam2)
        if width < 0.02 and height < 0.02:
            continue

        det = lam1 * lam2
        # orange for area-preserving/positive det-like; cyan for saddle-ish
        edge_col = (1.0, 0.5, 0.0) if det >= 0 else (0.2, 0.8, 1.0)

        e = Ellipse(
            (X[i, j], Y[i, j]),
            width=width,
            height=height,
            angle=angle,
            edgecolor=edge_col,
            facecolor="none",
            lw=1.0,
            alpha=0.9,
        )
        ax2.add_patch(e)

plt.savefig("scalar_vector_tensor_fields.png", dpi=600, facecolor="black")
plt.show()
