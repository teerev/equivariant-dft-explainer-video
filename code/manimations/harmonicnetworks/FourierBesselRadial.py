from manim import *
import numpy as np
from scipy.special import jv, jn_zeros
from PIL import Image


class FourierBesselRadial(Scene):
    def construct(self):
        # Black background
        self.camera.background_color = BLACK

        # Scarlet red for axes, labels, vector
        scarlet = RED

        # Axes on the left
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": scarlet},
        ).to_edge(LEFT, buff=0.5)
        axes.shift(UP * 0.4)

        axes_labels = axes.get_axis_labels(r"x'", r"y'").set_color(scarlet)

        # Vector p' at 30 degrees
        alpha = 30 * DEGREES
        p_vec = np.array([3 * np.cos(alpha), 3 * np.sin(alpha), 0.0])
        vector = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(p_vec[0], p_vec[1]),
            buff=0,
            color=scarlet,
            max_tip_length_to_length_ratio=0.15,
        )
        vec_label = MathTex("p'").set_color(scarlet).next_to(vector.get_end(), RIGHT)

        # Draw static geometry
        self.play(Create(axes), FadeIn(axes_labels))
        self.play(GrowArrow(vector), FadeIn(vec_label))
        self.wait(0.5)

        axes.set_z_index(10)
        axes_labels.set_z_index(11)
        vector.set_z_index(12)
        vec_label.set_z_index(12)

        # Initial generic equation
        equation = MathTex(
            r"R_{m,n}(r) = J_m\!\left(\alpha_{m,n}\,\frac{r}{R}\right)"
        ).scale(0.8)
        equation.next_to(axes, DOWN, buff=0.25)
        equation.set_z_index(13)
        self.play(FadeIn(equation, run_time=0.5))

        # Colour scheme (vivid cyan/orange, gamma=0.3)
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

        # Grid on the plane
        grid_size = 512
        R_disk = 4.0  # this is "R" in the formula

        xs = np.linspace(-R_disk, R_disk, grid_size)
        ys = np.linspace(-R_disk, R_disk, grid_size)
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X**2 + Y**2)

        # Mask outside the disk (basis is defined on the disk)
        circle_mask = R <= R_disk

        # m = 0..3, n = 1..3
        max_m = 3
        max_n = 3

        for m in range(0, max_m + 1):
            # precompute first max_n zeros of J_m
            zeros_m = jn_zeros(m, max_n)
            for n_idx in range(max_n):
                n = n_idx + 1
                alpha_mn = zeros_m[n_idx]

                # Update equation text
                eq_tex = (
                    fr"R_{{{m},{n}}}(r) = "
                    fr"J_{{{m}}}\!\left(\alpha_{{{m},{n}}}\,\frac{{r}}{{R}}\right)"
                )
                new_eq = MathTex(eq_tex).scale(0.8)
                new_eq.next_to(axes, DOWN, buff=0.4)
                new_eq.set_z_index(13)
                self.play(ReplacementTransform(equation, new_eq, run_time=0.4))
                equation = new_eq

                # true radial eigenfunction on the disk
                arg = alpha_mn * (R / R_disk)
                Z = jv(m, arg)
                Z = np.where(circle_mask, Z, 0.0)

                rgb, alpha = value_to_rgb_and_alpha(Z)

                rgba = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
                rgba[..., 0:3] = (rgb * 255).astype(np.uint8)
                rgba[..., 3] = (alpha * 255).astype(np.uint8)

                img = Image.fromarray(rgba, mode="RGBA")
                img_mobj = ImageMobject(img).set_z_index(1)
                img_mobj.set_height(axes.height)
                img_mobj.move_to(axes.get_center())

                self.play(FadeIn(img_mobj, run_time=0.5))
                self.wait(0.4)
                self.play(FadeOut(img_mobj, run_time=0.5))

        self.wait(1)
