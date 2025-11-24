#!/usr/bin/env python3
"""
Spin2Rotation visualization: 2D velocity field with three panels:
1) arrow plot of the velocity vectors,
2) heatmap of the x-component,
3) heatmap of the y-component.

The velocity field combines a vortical swirl, a radial component,
and a sinusoidal modulation to create nontrivial structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

# Domain and sampling
L = 2.0
N = 25
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

# Continuous grids for heatmaps
N2 = 200
xx = np.linspace(-L, L, N2)
yy = np.linspace(-L, L, N2)
XX, YY = np.meshgrid(xx, yy)

# Custom diverging colormap: cyan → black → orange
diverging_colors = [
    (0.0, (0.2, 0.8, 1.0)),
    (0.5, (0.0, 0.0, 0.0)),
    (1.0, (1.0, 0.5, 0.0)),
]
scalar_cmap = mcolors.LinearSegmentedColormap.from_list(
    "cyan_black_orange", diverging_colors
)

# Define a velocity field with swirl, radial, and sinusoidal components
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

VX, VY = velocity_field(X, Y)
VX_heat, VY_heat = velocity_field(XX, YY)

# Set up figure with black background and four panels (2x2 grid)
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=False, facecolor="black")
fig.subplots_adjust(left=0.05, right=0.78, wspace=0.35, hspace=0.35)
axes = axes.flatten()
for ax in axes:
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

# Non-symmetric scalar field with high-frequency positive/negative structure
scalar_field = np.exp(-0.1 * (XX**2 + YY**2)) * (
    0.7 * np.sin(2.8 * XX + 1.2 * YY)
    - 0.5 * np.cos(3.4 * YY - 1.1 * XX)
    + 0.6 * np.sin(2.2 * XX - 2.6 * YY) * np.cos(2.8 * XX + 1.9 * YY)
)
scalar_min = scalar_field.min()
scalar_max = scalar_field.max()
scalar_field = 2.0 * (scalar_field - scalar_min) / (scalar_max - scalar_min + 1e-9) - 1.0
scalar_rgba = scalar_cmap((scalar_field + 1.0) / 2.0)
scalar_rgba[..., 3] = np.clip((scalar_field + 1.0) / 2.0, 0, 1)
ax0 = axes[0]
ax0.imshow(
    scalar_rgba,
    origin="lower",
    extent=(-L, L, -L, L),
)
ax0.quiver(X, Y, VX, VY, color="cyan", pivot="mid")
ax0.set_title("Velocity field + temperature field", color="white")
ax0.set_xlabel("x", color="white")
ax0.set_ylabel("y", color="white")
ax0.set_aspect("equal")

# Panel 2: v_x-component heatmap
ax1 = axes[1]
im_vx = ax1.imshow(
    VX_heat,
    origin="lower",
    extent=(-L, L, -L, L),
    cmap=scalar_cmap,
)
ax1.set_title(r"$v_x(x,y)$", color="white")
ax1.set_xlabel("x", color="white")
ax1.set_ylabel("y", color="white")
cbar_vx = fig.colorbar(im_vx, ax=ax1, shrink=0.6)
cbar_vx.ax.yaxis.set_tick_params(color="white")
cbar_vx.outline.set_edgecolor("white")

# Panel 3: v_y-component heatmap
ax2 = axes[2]
im_vy = ax2.imshow(
    VY_heat,
    origin="lower",
    extent=(-L, L, -L, L),
    cmap=scalar_cmap,
)
ax2.set_title(r"$v_y(x,y)$", color="white")
ax2.set_xlabel("x", color="white")
ax2.set_ylabel("y", color="white")
cbar_vy = fig.colorbar(im_vy, ax=ax2, shrink=0.6)
cbar_vy.ax.yaxis.set_tick_params(color="white")
cbar_vy.outline.set_edgecolor("white")

# Panel 4: scalar field visualization
ax3 = axes[3]
im_s = ax3.imshow(
    scalar_rgba,
    origin="lower",
    extent=(-L, L, -L, L),
)
ax3.set_title("Temperature field", color="white")
ax3.set_xlabel("x", color="white")
ax3.set_ylabel("y", color="white")
scalar_mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=scalar_cmap)
scalar_mappable.set_array([])
cbar_s = fig.colorbar(
    scalar_mappable,
    ax=ax3,
    shrink=0.6,
    label="Intensity",
)
cbar_s.ax.yaxis.set_tick_params(color="white")
cbar_s.outline.set_edgecolor("white")
cbar_s.ax.yaxis.label.set_color("white")

fig.suptitle(
    "Nontrivial velocity field with vector & component plots",
    fontsize=14,
    color="white",
    x=0.05,
    ha="left",
)
plt.savefig("spin2rotation_velocity.png", dpi=200)
plt.show()