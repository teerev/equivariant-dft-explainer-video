from manim import *
import numpy as np

# ------------------------------------------------------------
# Global config / colours
# ------------------------------------------------------------
SCARLET = ManimColor("#ff2400")
RADIUS = 1.35

C_CYAN   = np.array([0.0, 0.8, 1.0])
C_ORANGE = np.array([1.0, 0.35, 0.0])
C_BLACK  = np.array([0.0, 0.0, 0.0])

N_SAMPLES = 360
THETAS = np.linspace(0, TAU, N_SAMPLES)

THETA0 = 30 * DEGREES   # angle of p' from sigma axis
ALPHA  = 47 * DEGREES   # rotation angle for Stage 2/3

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

    sigma_label = MathTex(r"x'", color=SCARLET).scale(0.6)
    tau_label   = MathTex(r"y'",   color=SCARLET).scale(0.6)

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
    theta_label = MathTex(r"\theta'", color=SCARLET).scale(0.6)
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
    top_center          = np.array([-5.5,  0.0, 0.0])
    bottom_left_center  = np.array([-5.5,  2.0, 0.0])
    bottom_right_center = np.array([-5.5, -2.0, 0.0])

    circle_top, deco_top, axes_lbl_top, p_lbl_top, func_lbl_top = make_circle_components(
        top_center,
        real_fourier,
        r"\Theta_{\text{real}}(\theta')"
    )

    circle_breal, deco_breal, axes_lbl_breal, p_lbl_breal, func_lbl_breal = make_circle_components(
        bottom_left_center,
        lambda th: np.real(complex_fourier(th)),
        r"\Re(\Theta_{\text{complex}}(\theta'))"
    )

    circle_bimag, deco_bimag, axes_lbl_bimag, p_lbl_bimag, func_lbl_bimag = make_circle_components(
        bottom_right_center,
        lambda th: np.imag(complex_fourier(th)),
        r"\Im(\Theta_{\text{complex}}(\theta'))"
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
            FadeIn(axes_lbl_top),   FadeIn(p_lbl_top),   FadeIn(func_lbl_top),
            run_time=1.5,
        )
        self.wait(1.0)


# ------------------------------------------------------------
# Stage 2 — Active rotation of Θ_real(θ):
#           top circle rotates by +α,
#           p' fixed, show Q_α p' and α on the top panel.
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

        # Base configuration (as at end of Stage 1)
        self.add(
            top_group,
            axes_lbl_top,   p_lbl_top,   func_lbl_top,
        )
        self.wait(0.5)

        # ----------------------------------------------------
        # New arrow for Q_α p'(θ) on the TOP circle
        # ----------------------------------------------------
        # deco_top = VGroup(axes_lines, p_group, theta_group)
        p_arrow = deco_top[1][0]  # p_group[0]
        q_arrow = p_arrow.copy().set_color(WHITE)

        # End position of Q_α p' for label placement
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

        # ----------------------------------------------------
        # Animate: rotate only the TOP FUNCTION circle by +α
        #          p', axes and θ stay fixed; Q_α p' rotates
        #          with the function and then stays visible.
        # ----------------------------------------------------
        self.add(q_arrow)
        self.play(
            Rotate(circle_top, angle=ALPHA, about_point=top_center),
            Rotate(q_arrow,    angle=ALPHA, about_point=top_center),
            FadeIn(alpha_arc),
            FadeIn(alpha_label),
            FadeIn(q_label),
            run_time=2.0,
        )
        self.wait(2.0)


# ------------------------------------------------------------
# Stage 3 — Active rotation of Θ_complex(θ):
#           bottom Re/Im circles rotate by +α,
#           p' fixed everywhere,
#           show corresponding Q_α p' vectors on all panels.
# ------------------------------------------------------------
class Stage3(Scene):
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

        # Base configuration: Start with bottom circles visible, top invisible
        self.add(
            breal_group, bimag_group,
            axes_lbl_breal, p_lbl_breal, func_lbl_breal,
            axes_lbl_bimag, p_lbl_bimag, func_lbl_bimag,
        )

        self.wait(0.5)

        # ----------------------------------------------------
        # Corresponding Q_α p' vectors on the BOTTOM circles
        # ----------------------------------------------------
        # Bottom-left (Re)
        p_arrow_breal = deco_breal[1][0]
        q_arrow_breal = p_arrow_breal.copy().set_color(WHITE)

        # Bottom-right (Im)
        p_arrow_bimag = deco_bimag[1][0]
        q_arrow_bimag = p_arrow_bimag.copy().set_color(WHITE)

        self.add(q_arrow_breal, q_arrow_bimag)

        # ----------------------------------------------------
        # Labels and Arcs for Bottom-Left
        # ----------------------------------------------------
        q_dir_bl = np.array([
            np.cos(THETA0 + ALPHA),
            np.sin(THETA0 + ALPHA),
            0.0
        ])
        q_end_bl = bl_center + RADIUS * q_dir_bl
        q_label_bl = MathTex(r"Q_\alpha p'", color=WHITE).scale(0.7)
        q_label_bl.next_to(q_end_bl, UP + RIGHT, buff=0.15)

        alpha_arc_bl = Arc(
            radius=0.8,
            start_angle=THETA0,
            angle=ALPHA,
            arc_center=bl_center,
            color=WHITE,
        )
        alpha_label_bl = MathTex(r"\alpha", color=WHITE).scale(0.6)
        mid_angle_bl = THETA0 + ALPHA / 2.0
        alpha_label_bl.move_to(
            bl_center
            + 1.1 * np.array([np.cos(mid_angle_bl), np.sin(mid_angle_bl), 0.0])
        )

        # ----------------------------------------------------
        # Labels and Arcs for Bottom-Right
        # ----------------------------------------------------
        q_dir_br = np.array([
            np.cos(THETA0 + ALPHA),
            np.sin(THETA0 + ALPHA),
            0.0
        ])
        q_end_br = br_center + RADIUS * q_dir_br
        q_label_br = MathTex(r"Q_\alpha p'", color=WHITE).scale(0.7)
        q_label_br.next_to(q_end_br, UP + RIGHT, buff=0.15)

        alpha_arc_br = Arc(
            radius=0.8,
            start_angle=THETA0,
            angle=ALPHA,
            arc_center=br_center,
            color=WHITE,
        )
        alpha_label_br = MathTex(r"\alpha", color=WHITE).scale(0.6)
        mid_angle_br = THETA0 + ALPHA / 2.0
        alpha_label_br.move_to(
            br_center
            + 1.1 * np.array([np.cos(mid_angle_br), np.sin(mid_angle_br), 0.0])
        )

        # After rotation, these will end up at angle θ + α in each panel.
        # We animate both the circle and these arrows together.

        self.play(
            Rotate(circle_breal, angle=ALPHA, about_point=bl_center),
            Rotate(circle_bimag, angle=ALPHA, about_point=br_center),
            Rotate(q_arrow_breal, angle=ALPHA, about_point=bl_center),
            Rotate(q_arrow_bimag, angle=ALPHA, about_point=br_center),
            
            FadeIn(q_label_bl),
            FadeIn(alpha_arc_bl),
            FadeIn(alpha_label_bl),
            
            FadeIn(q_label_br),
            FadeIn(alpha_arc_br),
            FadeIn(alpha_label_br),
            
            run_time=2.0,
        )
        self.wait(2.0)
