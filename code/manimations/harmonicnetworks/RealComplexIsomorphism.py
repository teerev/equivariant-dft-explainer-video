from manim import *
import numpy as np

# ------------------------------------------------------------
# Global config / colours
# ------------------------------------------------------------
SCARLET = ManimColor("#ff2400")
RADIUS = 1.5

C_CYAN   = np.array([0.0, 0.8, 1.0])
C_ORANGE = np.array([1.0, 0.35, 0.0])
C_BLACK  = np.array([0.0, 0.0, 0.0])

N_SAMPLES = 360
THETAS = np.linspace(0, TAU, N_SAMPLES)

THETA0 = 30 * DEGREES   # angle of p' from sigma axis
ALPHA  = 47 * DEGREES   # rotation angle for Stage 2

# ------------------------------------------------------------
# Colour mapping
# ------------------------------------------------------------
def value_to_rgb(v):
    v = np.clip(v, -1, 1)
    if v < 0:
        t = (v + 1.0)
        rgb = (1 - t) * C_CYAN + t * C_BLACK
    else:
        t = v
        rgb = (1 - t) * C_BLACK + t * C_ORANGE
    return rgb


# ------------------------------------------------------------
# Functions on S¹
# ------------------------------------------------------------
def real_fourier(theta):
    return (
        0.3 * np.cos(theta)
        + 0.7 * np.sin(theta)
        - 0.5 * np.cos(2 * theta)
        + 0.9 * np.sin(2 * theta)
    )


def complex_fourier(theta):
    c = {
        -2: (-0.4 + 0.2j),
        -1: ( 0.6 - 0.3j),
         0: ( 0.1 + 0.0j),
         1: ( 0.7 + 0.5j),
         2: (-0.2 - 0.6j),
    }
    return sum(c[m] * np.exp(1j * m * theta) for m in c)


# ------------------------------------------------------------
# Circle rendering helper
# ------------------------------------------------------------
def coloured_circle(center, radius, func):
    group = VGroup()
    for i in range(len(THETAS) - 1):
        th1 = THETAS[i]
        th2 = THETAS[i + 1]
        mid = 0.5 * (th1 + th2)
        val = func(mid)
        rgb = value_to_rgb(val)

        p1 = center + radius * np.array([np.cos(th1), np.sin(th1), 0])
        p2 = center + radius * np.array([np.cos(th2), np.sin(th2), 0])

        seg = Line(p1, p2, stroke_width=6)
        seg.set_color(ManimColor(rgb))
        group.add(seg)
    return group


# ------------------------------------------------------------
# Axes, p', θ annotations
# ------------------------------------------------------------
def make_axes(center):
    axis_len = 1.1 * RADIUS
    neg_stub = 0.25

    x_axis = Line(
        center + LEFT * neg_stub,
        center + RIGHT * axis_len,
        color=SCARLET,
        stroke_width=3,
    )
    y_axis = Line(
        center + DOWN * neg_stub,
        center + UP * axis_len,
        color=SCARLET,
        stroke_width=3,
    )

    sigma_label = MathTex(r"\sigma", color=SCARLET).scale(0.6)
    tau_label   = MathTex(r"\tau",   color=SCARLET).scale(0.6)

    def update_sigma(mob):
        mob.next_to(x_axis.get_end(), RIGHT, buff=0.1)
    sigma_label.add_updater(update_sigma)
    update_sigma(sigma_label)

    def update_tau(mob):
        mob.next_to(y_axis.get_end(), UP, buff=0.1)
    tau_label.add_updater(update_tau)
    update_tau(tau_label)

    axes_lines  = VGroup(x_axis, y_axis)
    axes_labels = VGroup(sigma_label, tau_label)
    return axes_lines, axes_labels


def make_pprime_and_theta(center):
    direction = np.array([np.cos(THETA0), np.sin(THETA0), 0.0])
    end = center + RADIUS * direction

    arrow = Arrow(
        start=center,
        end=end,
        buff=0,
        stroke_width=4,
        color=SCARLET,
        max_tip_length_to_length_ratio=0.15,
    )

    p_label = MathTex("p'", color=SCARLET).scale(0.7)

    def update_p_label(mob):
        mob.next_to(arrow.get_end(), UP + RIGHT, buff=0.15)

    p_label.add_updater(update_p_label)
    update_p_label(p_label)

    arc_theta = Arc(
        radius=0.6,
        start_angle=0.0,
        angle=THETA0,
        arc_center=center,
        color=SCARLET,
    )
    theta_label = MathTex(r"\theta", color=SCARLET).scale(0.6)
    theta_label.move_to(
        center
        + 0.9
        * np.array([np.cos(THETA0 / 2), np.sin(THETA0 / 2), 0.0])
    )

    p_group     = VGroup(arrow)
    theta_group = VGroup(arc_theta, theta_label)
    return p_group, theta_group, p_label


# ------------------------------------------------------------
# Build components for one circle
# ------------------------------------------------------------
def make_circle_components(center, func, label_tex):
    circle = coloured_circle(center, RADIUS, func)

    axes_lines, axes_labels = make_axes(center)
    p_group, theta_group, p_label = make_pprime_and_theta(center)

    func_label = MathTex(label_tex, color=WHITE).scale(0.7)
    func_label.next_to(circle, DOWN, buff=0.25)

    deco = VGroup(axes_lines, p_group, theta_group)
    return circle, deco, axes_labels, p_label, func_label


# ------------------------------------------------------------
# Full layout
# ------------------------------------------------------------
def layout_scene():
    top_center          = np.array([-4.0,  2.0, 0.0])
    bottom_left_center  = np.array([-5.0, -1.8, 0.0])
    bottom_right_center = np.array([-1.5, -1.8, 0.0])

    circle_top, deco_top, axes_lbl_top, p_lbl_top, func_lbl_top = make_circle_components(
        top_center,
        real_fourier,
        r"\Theta_{\text{real}}(\theta)"
    )

    circle_breal, deco_breal, axes_lbl_breal, p_lbl_breal, func_lbl_breal = make_circle_components(
        bottom_left_center,
        lambda th: np.real(complex_fourier(th)),
        r"\Re (\Theta_{\text{complex}}(\theta))"
    )

    circle_bimag, deco_bimag, axes_lbl_bimag, p_lbl_bimag, func_lbl_bimag = make_circle_components(
        bottom_right_center,
        lambda th: np.imag(complex_fourier(th)),
        r"\Im (\Theta_{\text{complex}}(\theta))"
    )

    return (
        circle_top, deco_top, axes_lbl_top, p_lbl_top, func_lbl_top,
        circle_breal, deco_breal, axes_lbl_breal, p_lbl_breal, func_lbl_breal,
        circle_bimag, deco_bimag, axes_lbl_bimag, p_lbl_bimag, func_lbl_bimag,
        top_center, bottom_left_center, bottom_right_center,
    )


# ------------------------------------------------------------
# Stage 1 — static base configuration
# ------------------------------------------------------------
class Stage1(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        (
            circle_top, deco_top, axes_lbl_top, p_lbl_top, func_lbl_top,
            circle_breal, deco_breal, axes_lbl_breal, p_lbl_breal, func_lbl_breal,
            circle_bimag, deco_bimag, axes_lbl_bimag, p_lbl_bimag, func_lbl_bimag,
            top_center, bl_center, br_center,
        ) = layout_scene()

        top_group   = VGroup(circle_top,   deco_top)
        breal_group = VGroup(circle_breal, deco_breal)
        bimag_group = VGroup(circle_bimag, deco_bimag)

        self.play(
            FadeIn(top_group),
            FadeIn(breal_group),
            FadeIn(bimag_group),

            FadeIn(axes_lbl_top),   FadeIn(p_lbl_top),   FadeIn(func_lbl_top),
            FadeIn(axes_lbl_breal), FadeIn(p_lbl_breal), FadeIn(func_lbl_breal),
            FadeIn(axes_lbl_bimag), FadeIn(p_lbl_bimag), FadeIn(func_lbl_bimag),

            run_time=1.5,
        )
        self.wait(1.0)


# ------------------------------------------------------------
# Stage 2 — Active rotation of f: circle rotates by +α,
#           p' fixed, show Q_α p' and algebra on the right.
# ------------------------------------------------------------
class Stage2(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        (
            circle_top, deco_top, axes_lbl_top, p_lbl_top, func_lbl_top,
            circle_breal, deco_breal, axes_lbl_breal, p_lbl_breal, func_lbl_breal,
            circle_bimag, deco_bimag, axes_lbl_bimag, p_lbl_bimag, func_lbl_bimag,
            top_center, bl_center, br_center,
        ) = layout_scene()

        # Group the three panels
        top_group   = VGroup(circle_top,   deco_top)
        breal_group = VGroup(circle_breal, deco_breal)
        bimag_group = VGroup(circle_bimag, deco_bimag)

        # Add the base configuration (same as end of Stage 1)
        # Note: deco_* includes axes_lines, p_group (arrow), theta_group (arc+label)
        # We must explicitly add axes_labels, p_label, func_label as they are separate
        self.add(
            top_group, breal_group, bimag_group,
            axes_lbl_top,   p_lbl_top,   func_lbl_top,
            axes_lbl_breal, p_lbl_breal, func_lbl_breal,
            axes_lbl_bimag, p_lbl_bimag, func_lbl_bimag,
        )
        self.wait(0.5)

        # -----------------------------
        # New arrow for Q_α p'(θ)
        # -----------------------------
        # Create q_arrow starting at p' position (copy of p_arrow)
        p_arrow = deco_top[1][0]
        q_arrow = p_arrow.copy().set_color(WHITE)

        # Calculate final position for label placement
        q_dir_final = np.array([
            np.cos(THETA0 + ALPHA),
            np.sin(THETA0 + ALPHA),
            0.0
        ])
        q_end_final = top_center + RADIUS * q_dir_final

        q_label = MathTex(r"Q_\alpha p'", color=WHITE).scale(0.7)
        q_label.next_to(q_end_final, UP + RIGHT, buff=0.15)

        # α arc between p' and Q_α p'
        alpha_arc = Arc(
            radius=0.8,
            start_angle=THETA0,
            angle=ALPHA,
            arc_center=top_center,
            color=WHITE,
        )
        alpha_label = MathTex(r"\alpha", color=WHITE).scale(0.6)
        mid_angle = THETA0 + ALPHA / 2.0
        alpha_label.move_to(
            top_center
            + 1.1 * np.array([np.cos(mid_angle), np.sin(mid_angle), 0.0])
        )

        # Animate: rotate only the top circle (function), add Q_α p' and α arc
        self.add(q_arrow)
        self.play(
            Rotate(circle_top, angle=ALPHA, about_point=top_center),
            Rotate(q_arrow, angle=ALPHA, about_point=top_center),
            FadeIn(alpha_arc),
            FadeIn(alpha_label),
            FadeIn(q_label),
            run_time=2.0,
        )
        self.wait(0.5)

        self.wait(2.0)

