from manim import *
import numpy as np
from math import factorial
from scipy.special import lpmv

IMAGE_NEG_HEX = "#00D5FF"
IMAGE_POS_HEX = "#F26D00"

SPHERE_RADIUS = 2.5

# Which harmonics to show (ℓ, m)
HARMONICS = [
    (l, m)
    for l in range(0, 7)  # up to l=6
    for m in range(-l, l + 1)
]


# ----------------- spherical harmonics -----------------
def real_spherical_harmonic(l, m, theta, phi):
    """
    Real-valued spherical harmonic Y_l^m(θ, φ) supporting any l,m.
    Normalization constant is included only for relative scaling; the colours
    are normalized later via compute_max_val.
    """
    x = np.cos(theta)
    abs_m = abs(m)

    # Associated Legendre polynomial P_l^m (includes Condon–Shortley phase)
    P_lm = lpmv(abs_m, l, x)

    # Normalization (not critical for colouring, but keeps magnitude reasonable)
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - abs_m) / factorial(l + abs_m))

    if m > 0:
        return np.sqrt(2.0) * norm * P_lm * np.cos(abs_m * phi)
    if m < 0:
        return np.sqrt(2.0) * norm * P_lm * np.sin(abs_m * phi)
    return norm * P_lm


def compute_max_val(l, m, n_theta=160, n_phi=320):
    thetas = np.linspace(1e-3, np.pi - 1e-3, n_theta)
    phis = np.linspace(0.0, TAU, n_phi)
    Theta, Phi = np.meshgrid(thetas, phis, indexing="ij")
    vals = real_spherical_harmonic(l, m, Theta, Phi)
    max_val = np.max(np.abs(vals))
    return float(max_val if max_val != 0 else 1.0)


# -------------- coloured spherical harmonic sphere --------------
def make_coloured_spherical_harmonic(
    l, m, radius=SPHERE_RADIUS, resolution=(80, 160)
):
    def sphere_param(u, v):
        theta = u
        phi = v
        r = radius
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    surface = Surface(
        sphere_param,
        u_range=[1e-3, PI - 1e-3],
        v_range=[0, TAU],
        resolution=resolution,
        fill_opacity=1.0,
        stroke_width=0.0,
        stroke_opacity=0.0,
        shade_in_3d=True,
    )
    surface.set_style(stroke_width=0.0, stroke_opacity=0.0)

    max_val = compute_max_val(l, m)
    pos_color = ManimColor(IMAGE_POS_HEX)
    neg_color = ManimColor(IMAGE_NEG_HEX)

    for patch in surface:
        x, y, z = patch.get_center()
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            continue

        theta = np.arccos(np.clip(z / r, -1.0, 1.0))
        phi = np.arctan2(y, x)

        val = real_spherical_harmonic(l, m, theta, phi)
        a = np.clip(abs(val) / max_val, 0.0, 1.0)
        base_color = pos_color if val >= 0 else neg_color
        vivid_factor = np.sqrt(a)
        color = interpolate_color(WHITE, base_color, vivid_factor)

        patch.set_fill(color, opacity=1.0)
        patch.set_stroke(width=0.0, opacity=0.0)

    return surface


# ----------------- Scene -----------------
class SphericalHarmonicsOnSphere(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-4, 4, 2],
            y_range=[-4, 4, 2],
            z_range=[-4, 4, 2],
        ).set_color(RED)

        x_label = MathTex("x'", color=RED).next_to(
            axes.x_axis.get_end(), RIGHT, buff=0.2
        )
        y_label = MathTex("y'", color=RED).next_to(
            axes.y_axis.get_end(), UP, buff=0.2
        )
        z_label = MathTex("z'", color=RED).next_to(
            axes.z_axis.get_end(), OUT, buff=0.2
        )

        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-45 * DEGREES,
            distance=9,
        )
        self.begin_ambient_camera_rotation(rate=0.15)

        self.add(axes, x_label, y_label, z_label)

        duration_per_harmonic = 1.0
        current_surface = None
        current_label = None

        for idx, (l, m) in enumerate(HARMONICS):
            sh_surface = make_coloured_spherical_harmonic(l, m)
            label = MathTex(
                fr"Y_{{{l}}}^{{{m}}}",
                color=WHITE,
            ).to_corner(UL).shift(IN * 0.5)

            if idx == 0:
                self.play(FadeIn(sh_surface, scale=0.8), FadeIn(label))
            else:
                self.play(
                    ReplacementTransform(current_surface, sh_surface),
                    ReplacementTransform(current_label, label),
                )

            current_surface = sh_surface
            current_label = label
            self.wait(duration_per_harmonic)

        self.wait(2)
