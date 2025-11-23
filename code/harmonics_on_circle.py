import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Parameters
m_values = [-2, -1, 0, 1, 2]
N = 800
radius = 1.0

theta = np.linspace(0, 2 * np.pi, N)

# Colormap (shared across all panels)
shared_cmap = plt.cm.coolwarm

# Figure with three rows: phase, |Re|, |Im|
fig, axes = plt.subplots(
    3, len(m_values),
    figsize=(15, 12),
    subplot_kw={"aspect": "equal"},
    facecolor="black",
    dpi=200,
)
fig.subplots_adjust(hspace=0.15, wspace=0.35)


def draw_coloured_circle(ax, colors):
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    for i in range(N - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], color=colors[i], linewidth=2)


def setup_axes(ax, title=None):
    ax.plot([-1.2, 1.2], [0, 0], color="white", linewidth=1)
    ax.plot([0, 0], [-1.2, 1.2], color="white", linewidth=1)
    ax.text(1.15, 0.05, "x", color="white", fontsize=10)
    ax.text(0.05, 1.15, "y", color="white", fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=16, color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_facecolor("black")


# Top row: phase
for col, m in enumerate(m_values):
    phase = (m * theta) % (2 * np.pi)
    norm_phase = phase / (2 * np.pi)
    colors = shared_cmap(norm_phase)
    draw_coloured_circle(axes[0, col], colors)
    setup_axes(axes[0, col], title=f"m = {m}")

# Row labels on the left
row_labels = [
    r"$\phi = m\theta$",
    r"$\Re\{e^{im\theta}\} = \cos(m\theta)$",
    r"$\Im\{e^{im\theta}\} = \sin(m\theta)$",
]
for idx, label in enumerate(row_labels):
    axes[idx, 0].text(
        -1.9,
        0.0,
        label,
        color="white",
        fontsize=16,
        ha="right",
        va="center",
    )

# Middle row: Re
for col, m in enumerate(m_values):
    real_vals = np.cos(m * theta)
    normed = (real_vals + 1) / 2
    colors = shared_cmap(normed)
    draw_coloured_circle(axes[1, col], colors)
    setup_axes(axes[1, col])

# Bottom row: Im
for col, m in enumerate(m_values):
    imag_vals = np.sin(m * theta)
    normed = (imag_vals + 1) / 2
    colors = shared_cmap(normed)
    draw_coloured_circle(axes[2, col], colors)
    setup_axes(axes[2, col])

# Row-specific colourbars placed left of each row
phase_norm = Normalize(vmin=0, vmax=2 * np.pi)
value_norm = Normalize(vmin=-1, vmax=1)

row_y_positions = [0.81, 0.51, 0.21]
for idx, y_pos in enumerate(row_y_positions):
    if idx == 0:
        sm = ScalarMappable(norm=phase_norm, cmap=shared_cmap)
        ticks = [0, np.pi, 2 * np.pi]
        ticklabels = [r"$0$", r"$\pi$", r"$2\pi$"]
    else:
        sm = ScalarMappable(norm=value_norm, cmap=shared_cmap)
        ticks = [-1, 0, 1]
        ticklabels = [r"$-1$", r"$0$", r"$1$"]
    sm.set_array([])
    cbar_ax = fig.add_axes([0.11, y_pos - 0.035, 0.12, 0.012])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color="white", fontsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label("", fontsize=0)

plt.tight_layout(rect=[0.08, 0.02, 1, 1])
plt.savefig("harmonics_on_circle.png", dpi=300)
# plt.show()
