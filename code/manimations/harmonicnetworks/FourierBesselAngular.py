from manim import *
import numpy as np
from PIL import Image


class FourierBesselAngular(Scene):
    def construct(self):
        # Black background
        self.camera.background_color = BLACK

        # Scarlet red for axes, labels, vector
        BLUE = ManimColor("#0066F5")

        # Axes on the left
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE},
        ).to_edge(LEFT, buff=0.5)

        axes_labels = axes.get_axis_labels(r"x'", r"y'").set_color(BLUE)

        # Vector p' at 30 degrees
        alpha = 30 * DEGREES
        p_vec = np.array([3 * np.cos(alpha), 3 * np.sin(alpha), 0.0])
        vector = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(p_vec[0], p_vec[1]),
            buff=0,
            color=BLUE,
            max_tip_length_to_length_ratio=0.15,
        )
        vec_label = MathTex("p'").set_color(BLUE).next_to(vector.get_end(), RIGHT)

        # Draw static
        self.play(Create(axes), FadeIn(axes_labels))
        self.play(GrowArrow(vector), FadeIn(vec_label))
        self.wait(0.5)

        axes.set_z_index(10)
        axes_labels.set_z_index(11)
        vector.set_z_index(12)
        vec_label.set_z_index(12)

        # Initial equation
        equation = MathTex(
            r"\Re[e^{i m\theta'}] = \cos(m\theta'),\quad "
            r"\Im[e^{i m\theta'}] = \sin(m\theta')"
        ).scale(0.8)
        equation.next_to(axes, DOWN, buff=0.4)
        equation.set_z_index(13)
        self.play(FadeIn(equation, run_time=0.5))

        # Colour scheme
        C_CYAN   = np.array([0.0, 0.8, 1.0])
        C_ORANGE = np.array([1.0, 0.35, 0.0])

        def value_to_rgb_and_alpha(z):
            z_clipped = np.clip(z, -1.0, 1.0)
            mag = np.abs(z_clipped)
            gamma = 0.3
            mag_gamma = mag**gamma

            base = np.where(
                z_clipped[..., None] >= 0,
                C_ORANGE[None, None, :],
                C_CYAN[None, None, :],
            )

            rgb = base * mag_gamma[..., None]
            alpha = mag_gamma
            return rgb, alpha

        # Grid
        grid_size = 512
        r_max = 4.0

        xs = np.linspace(-r_max, r_max, grid_size)
        ys = np.linspace(-r_max, r_max, grid_size)
        X, Y = np.meshgrid(xs, ys)
        THETA = np.arctan2(Y, X)
        R = np.sqrt(X**2 + Y**2)
        CIRCLE_MASK = (R <= r_max).astype(float)

        max_m = 3  # m = 1,2,3

        for m in range(1, max_m + 1):
            # REAL part
            Z_real = np.cos(m * THETA)
            eq_real = MathTex(
                fr"\Re[e^{{i {m}\theta'}}] = \cos({m}\theta')"
            ).scale(0.9)
            eq_real.next_to(axes, DOWN, buff=0.4)
            eq_real.set_z_index(13)
            self.play(ReplacementTransform(equation, eq_real, run_time=0.35))
            equation = eq_real

            rgb, alpha = value_to_rgb_and_alpha(Z_real)
            rgb = rgb * CIRCLE_MASK[..., None]
            alpha = alpha * CIRCLE_MASK
            rgba = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
            rgba[..., 0:3] = (rgb * 255).astype(np.uint8)
            rgba[..., 3] = (alpha * 255).astype(np.uint8)

            img_real = Image.fromarray(rgba, mode="RGBA")
            img_mobj_real = ImageMobject(img_real).set_z_index(1)
            img_mobj_real.set_height(axes.height)
            img_mobj_real.move_to(axes.get_center())

            self.play(FadeIn(img_mobj_real, run_time=0.5))
            self.wait(0.4)
            self.play(FadeOut(img_mobj_real, run_time=0.5))

            # IMAG part
            Z_imag = np.sin(m * THETA)
            eq_imag = MathTex(
                fr"\Im[e^{{i {m}\theta'}}] = \sin({m}\theta')"
            ).scale(0.9)
            eq_imag.next_to(axes, DOWN, buff=0.4)
            eq_imag.set_z_index(13)
            self.play(ReplacementTransform(equation, eq_imag, run_time=0.35))
            equation = eq_imag

            rgb, alpha = value_to_rgb_and_alpha(Z_imag)
            rgb = rgb * CIRCLE_MASK[..., None]
            alpha = alpha * CIRCLE_MASK
            rgba = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
            rgba[..., 0:3] = (rgb * 255).astype(np.uint8)
            rgba[..., 3] = (alpha * 255).astype(np.uint8)

            img_imag = Image.fromarray(rgba, mode="RGBA")
            img_mobj_imag = ImageMobject(img_imag).set_z_index(1)
            img_mobj_imag.set_height(axes.height)
            img_mobj_imag.move_to(axes.get_center())

            self.play(FadeIn(img_mobj_imag, run_time=0.5))
            self.wait(0.4)
            self.play(FadeOut(img_mobj_imag, run_time=0.5))

        self.wait(1)
