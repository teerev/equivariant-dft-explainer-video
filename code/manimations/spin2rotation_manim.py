"""
Manim animation illustrating how circular-harmonic coefficients transform
under an in-plane rotation.  It mirrors the numerical demo in
`code/spin2rotation.py`, but animates both the spatial field and the complex
coefficients simultaneously.

Run (example):
    manim -pql code/animations/spin2rotation_manim.py Spin2RotationScene
"""

from __future__ import annotations

import numpy as np
import matplotlib
from manim import (
    BLUE,
    GREEN,
    PI,
    RED,
    LEFT,
    RIGHT,
    UP,
    DOWN,
    Arrow,
    ComplexPlane,
    ImageMobject,
    MathTex,
    Scene,
    Tex,
    ValueTracker,
    VGroup,
    always_redraw,
)

class Spin2RotationScene(Scene):
    def construct(self) -> None:
        # ------------------------------------------------------------------
        # Parameters (mirrors spin2rotation.py)
        # ------------------------------------------------------------------
        L = 2.0
        N = 300
        sigma = 1.0
        m_vals = np.array([0, 1, 2])
        coeffs = np.array(
            [
                1.0 + 0.0j,
                0.7 + 0.4j,
                0.5 - 0.8j,
            ]
        )

        # Cartesian grid (fixed) and polar representation
        x = np.linspace(-L, L, N)
        y = np.linspace(-L, L, N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        gauss = np.exp(-R**2 / (2 * sigma**2))
        colour_map = matplotlib.colormaps.get_cmap("coolwarm")

        def complex_field(alpha: float) -> np.ndarray:
            """Return real part of rotated field on the grid."""
            # Rotate coefficients: c'_m = e^{-i m alpha} c_m
            coeffs_rot = coeffs * np.exp(-1j * m_vals * alpha)
            field = np.zeros_like(R, dtype=np.complex128)
            for m, c in zip(m_vals, coeffs_rot):
                field += c * np.exp(1j * m * Theta)
            return (gauss * field).real

        def array_to_image(arr: np.ndarray) -> ImageMobject:
            norm = (arr - arr.min()) / (np.ptp(arr) + 1e-9)
            rgb = (colour_map(norm)[..., :3] * 255).astype(np.uint8)
            return ImageMobject(rgb)

        def coeff_point(m_index: int, alpha: float) -> complex:
            c = coeffs[m_index] * np.exp(-1j * m_vals[m_index] * alpha)
            return c

        # ------------------------------------------------------------------
        # Spatial field panels
        # ------------------------------------------------------------------
        image_scale = 3.0 / N * 4.0  # keep roughly square of width 4 units
        original_field = array_to_image(complex_field(alpha=0)).scale(image_scale)
        original_field.shift(3.5 * LEFT + 1.5 * UP)

        alpha_tracker = ValueTracker(0.0)
        rotated_field = always_redraw(
            lambda: array_to_image(complex_field(alpha_tracker.get_value()))
            .scale(image_scale)
            .shift(3.5 * RIGHT + 1.5 * UP)
        )

        # Cartesian axes drawn on top
        axes_template = Tex(r"x", color=BLUE).next_to(rotated_field, RIGHT, buff=0.1)

        self.add(original_field, rotated_field)
        self.add(
            Tex(r"$\Re f(x,y)$", color=BLUE).scale(0.8).next_to(original_field, DOWN)
        )
        self.add(
            Tex(r"$\Re f'(x,y)$", color=BLUE)
            .scale(0.8)
            .next_to(rotated_field, DOWN)
        )

        # ------------------------------------------------------------------
        # Complex plane for coefficients
        # ------------------------------------------------------------------
        plane = ComplexPlane(x_range=[-1.8, 1.8, 1], y_range=[-1.8, 1.8, 1])
        plane.scale(1.3).to_edge(DOWN)
        plane.set_color("#888888")

        colors = [RED, GREEN, BLUE]
        arrows = VGroup()
        labels = VGroup()
        for idx, m in enumerate(m_vals):
            colour = colors[idx % len(colors)]
            arrow_original = Arrow(
                plane.n2p(0),
                plane.n2p(coeff_point(idx, alpha=0)),
                buff=0,
                max_stroke_width_to_length_ratio=4,
                max_tip_length_to_length_ratio=0.15,
                color=colour,
            )
            arrow_rot = always_redraw(
                lambda idx=idx, colour=colour: Arrow(
                    plane.n2p(0),
                    plane.n2p(coeff_point(idx, alpha_tracker.get_value())),
                    buff=0,
                    max_stroke_width_to_length_ratio=4,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=6,
                    color=colour,
                    stroke_opacity=0.4,
                )
            )
            arrows.add(arrow_original, arrow_rot)
            labels.add(
                MathTex(
                    rf"m={m}",
                    color=colour,
                ).scale(0.8)
                .next_to(arrow_original.get_end(), RIGHT, buff=0.1)
            )

        coeff_text = Tex(
            r"Rotated coefficients: $c_m' = e^{-i m \alpha}\,c_m$",
            color="#DDDDDD",
        ).scale(0.8)
        coeff_text.next_to(plane, UP, buff=0.4)

        self.add(plane, arrows, labels, coeff_text)

        # ------------------------------------------------------------------
        # Animate rotation
        # ------------------------------------------------------------------
        instructions = Tex(
            r"Rotate by $\alpha = 45^\circ$",
            color="#FFFFFF",
        ).next_to(original_field, UP, buff=0.4)
        self.play(instructions.animate.set_opacity(1.0))

        self.play(alpha_tracker.animate.set_value(np.deg2rad(45)), run_time=5)
        self.wait(1)


__all__ = ["Spin2RotationScene"]


