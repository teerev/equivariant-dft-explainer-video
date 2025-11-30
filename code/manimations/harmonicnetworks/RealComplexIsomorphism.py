from manim import *
import numpy as np


# ------------------------------------------------------------
# Global config / colours
# ------------------------------------------------------------
SCARLET = ManimColor("#ff2400")  # scarlet red for axes
RADIUS = 1.5                     # circle radius
ANGLE_P0 = 20 * DEGREES          # initial vector angle wrt sigma-axis
ALPHA = 30 * DEGREES             # rotation angle for stages 2 and 3


# ------------------------------------------------------------
# Helper constructors
# ------------------------------------------------------------
def make_circle_with_axes(center: np.ndarray, coord_label: str):
    """
    Create a white circle with scarlet Cartesian axes crossing at 'center',
    and a label like '(\\sigma, \\tau)' or '(\\sigma, i\\tau)'.
    """
    circle = Circle(radius=RADIUS, color=WHITE, stroke_width=2).move_to(center)

    radius_axes = 1.1 * RADIUS
    # Short negative stub length
    neg_stub_len = 0.3
    
    x_axis = Line(
        center + LEFT * neg_stub_len,
        center + RIGHT * radius_axes,
        color=SCARLET,
        stroke_width=3,
    )
    y_axis = Line(
        center + DOWN * neg_stub_len,
        center + UP * radius_axes,
        color=SCARLET,
        stroke_width=3,
    )
    
    # Ticks at the ends of axes
    tick_length = 0.1
    
    # X-axis ticks (only on the positive right side now)
    x_tick_right = Line(
        center + RIGHT * radius_axes + UP * tick_length/2,
        center + RIGHT * radius_axes + DOWN * tick_length/2,
        color=SCARLET,
        stroke_width=3
    )
    
    # Y-axis ticks (only on the positive up side now)
    y_tick_up = Line(
        center + UP * radius_axes + LEFT * tick_length/2,
        center + UP * radius_axes + RIGHT * tick_length/2,
        color=SCARLET,
        stroke_width=3
    )

    axes = VGroup(x_axis, y_axis, x_tick_right, y_tick_up)
    group = VGroup(circle, axes)

    # Axis labels: sigma, tau
    # Determine if top (Real) or bottom (Complex) based on which circle it is.
    # We can distinguish by center y-coordinate or pass a flag.
    # Let's pass a flag or just deduce from 'coord_label' being empty but we want specific labels?
    # Actually, the user asked to remove "(\sigma, \tau)" main label but kept the axis labels.
    # Let's add sigma/tau axis labels back manually.
    
    sigma_label = MathTex(r"\sigma", color=SCARLET).scale(0.6).next_to(x_axis, RIGHT, buff=0.1)
    
    # Heuristic: if center y is positive -> real plane -> tau
    # if center y is negative -> complex plane -> i*tau
    if center[1] > 0:
        tau_label_text = r"\tau"
    else:
        tau_label_text = r"i\tau"
        
    tau_label = MathTex(tau_label_text, color=SCARLET).scale(0.6).next_to(y_axis, UP, buff=0.1)

    group.add(sigma_label, tau_label)

    if coord_label:
        # Coordinate label for the axes (e.g. "(\sigma,\tau)")
        coord_tex = MathTex(coord_label, color=WHITE).scale(0.7)
        coord_tex.next_to(circle, UP + RIGHT, buff=0.3)
        group.add(coord_tex)

    return group


def make_radial_vector(center: np.ndarray, angle: float, radius: float = RADIUS):
    """
    Arrow from 'center' to the circle circumference at given angle (radians).
    Angle measured from +sigma axis (i.e. +x axis).
    """
    direction = np.array([
        np.cos(angle),
        np.sin(angle),
        0.0,
    ])
    end = center + radius * direction
    arrow = Arrow(
        start=center,
        end=end,
        buff=0,
        stroke_width=4,
        max_tip_length_to_length_ratio=0.15,
        color=SCARLET,
    )
    return arrow


def layout_scene():
    """
    Shared layout for all stages:
    - Real plane circle at top-left
    - Complex (Argand) circle at bottom-left
    - Each with the same initial vector p' at 20 degrees.
    Returns:
        real_group, complex_group, real_arrow, complex_arrow
    """
    real_center = LEFT * 5.2 + UP * 2.0
    complex_center = LEFT * 5.2 + DOWN * 2.0

    # Circles + axes + labels
    real_group = make_circle_with_axes(real_center, "")
    complex_group = make_circle_with_axes(complex_center, "")

    # Identical p' vectors in both circles at 20 degrees
    real_arrow = make_radial_vector(real_center, ANGLE_P0)
    complex_arrow = make_radial_vector(complex_center, ANGLE_P0)

    # Label p' near the real vector
    p_prime_label_real = MathTex("p'", color=SCARLET).scale(0.7)
    p_prime_label_real.next_to(real_arrow.get_end(), UP + RIGHT, buff=0.15)

    # Label p' near the complex vector
    p_prime_label_complex = MathTex("p'", color=SCARLET).scale(0.7)
    p_prime_label_complex.next_to(complex_arrow.get_end(), UP + RIGHT, buff=0.15)

    real_group.add(real_arrow, p_prime_label_real)
    complex_group.add(complex_arrow, p_prime_label_complex)

    return real_group, complex_group, real_arrow, complex_arrow, p_prime_label_real, p_prime_label_complex


# ------------------------------------------------------------
# Stage 1: static picture with both circles and identical vectors
# ------------------------------------------------------------
class Stage1(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        real_group, complex_group, _, _, _, _ = layout_scene()

        # Bring everything in
        self.play(
            FadeIn(real_group, shift=RIGHT * 0.2),
            FadeIn(complex_group, shift=RIGHT * 0.2),
            run_time=1.5,
        )
        self.wait(1.5)


# ------------------------------------------------------------
# Stage 2: rotate the arrow in the REAL circle by +ALPHA
# ------------------------------------------------------------
class Stage2(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        real_group, complex_group, real_arrow, complex_arrow, p_label_real, _ = layout_scene()

        # Make sure all elements are on screen
        self.add(real_group, complex_group)

        real_center = real_arrow.get_start()  # this is the center for rotation

        # Brief pause before rotation
        self.wait(0.8)

        # Rotate ONLY the real-plane arrow and update label
        
        # Calculate correct end position for the target label
        end_arrow_tip = real_center + RADIUS * np.array([np.cos(ANGLE_P0 + ALPHA), np.sin(ANGLE_P0 + ALPHA), 0])
        
        q_label_real = MathTex(r"Q_\alpha p'", color=SCARLET).scale(0.7)
        q_label_real.next_to(end_arrow_tip, UP + RIGHT, buff=0.15)
        
        self.play(
            Rotate(real_arrow, angle=ALPHA, about_point=real_center),
            Transform(p_label_real, q_label_real, path_arc=ALPHA),
            run_time=1.5,
        )
        
        self.wait(1.0)


# ------------------------------------------------------------
# Stage 3: rotate the arrow in the COMPLEX circle by +ALPHA
# ------------------------------------------------------------
class Stage3(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        real_group, complex_group, real_arrow, complex_arrow, p_label_real, p_label_complex = layout_scene()

        # Ensure real arrow starts rotated (continuing from Stage 2)
        real_center = real_arrow.get_start()
        real_arrow.rotate(angle=ALPHA, about_point=real_center)
        
        # Update real label to Q_alpha p' and position it correctly
        q_label_real = MathTex(r"Q_\alpha p'", color=SCARLET).scale(0.7)
        q_label_real.next_to(real_arrow.get_end(), UP + RIGHT, buff=0.15)
        
        # We need to replace the old label in the group/scene with the new one
        # p_label_real was added to real_group in layout_scene()
        p_label_real.become(q_label_real)
        
        # Add everything (initially same as Stage 1)
        self.add(real_group, complex_group)

        complex_center = complex_arrow.get_start()

        # Brief pause before rotation
        self.wait(0.8)

        # Rotate ONLY the complex-plane arrow
        
        # Helper to update label position
        def update_label_complex(mob):
            mob.next_to(complex_arrow.get_end(), UP + RIGHT, buff=0.15)

        # Calculate correct end position for the target label
        end_arrow_tip = complex_center + RADIUS * np.array([np.cos(ANGLE_P0 + ALPHA), np.sin(ANGLE_P0 + ALPHA), 0])
        
        # New label after rotation: Q_alpha p'
        q_label_complex = MathTex(r"Q_\alpha p'", color=SCARLET).scale(0.7)
        q_label_complex.next_to(end_arrow_tip, UP + RIGHT, buff=0.15)

        # We remove updater before Transform so they don't conflict.
        # We use path_arc to make the label move in a curve similar to the arrow.
        self.play(
            Rotate(complex_arrow, angle=ALPHA, about_point=complex_center),
            Transform(p_label_complex, q_label_complex, path_arc=ALPHA),
            run_time=1.5,
        )
        
        self.wait(1.0)
