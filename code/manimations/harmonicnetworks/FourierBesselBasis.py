from manim import *
import numpy as np
from scipy.special import jv, jn_zeros
from PIL import Image


class FourierBesselBasis(Scene):
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

        axes_labels = axes.get_axis_labels(r"\sigma", r"\tau").set_color(scarlet)

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

        # Draw static
        self.play(Create(axes), FadeIn(axes_labels))
        self.play(GrowArrow(vector), FadeIn(vec_label))
        self.wait(0.5)

        axes.set_z_index(10)
        axes_labels.set_z_index(11)
        vector.set_z_index(12)
        vec_label.set_z_index(12)

        # Generic formula initially
        equation = MathTex(
            r"\psi_{m,n}(r,\theta) = "
            r"J_m\!\left(\alpha_{m,n}\frac{r}{R}\right)e^{im\theta}"
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

        # Grid on disk
        grid_size = 512
        R_disk = 4.0

        xs = np.linspace(-R_disk, R_disk, grid_size)
        ys = np.linspace(-R_disk, R_disk, grid_size)
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)

        circle_mask = R <= R_disk

        max_m = 3
        max_n = 3

        for m in range(0, max_m + 1):
            zeros_m = jn_zeros(m, max_n)
            for n_idx in range(max_n):
                n = n_idx + 1
                alpha_mn = zeros_m[n_idx]
                arg = alpha_mn * (R / R_disk)
                radial = jv(m, arg)
                radial = np.where(circle_mask, radial, 0.0)

                # REAL / cos part
                if m == 0:
                    # for m=0, cos(0θ)=1, so real part is just radial
                    Z_real = radial
                    eq_tex_real = (
                        fr"\psi_{{{m},{n}}}^{{(\cos)}}(r,\theta) = "
                        fr"J_{{{m}}}\!\left(\alpha_{{{m},{n}}}\frac{{r}}{{R}}\right)"
                    )
                else:
                    Z_real = radial * np.cos(m * THETA)
                    eq_tex_real = (
                        fr"\psi_{{{m},{n}}}^{{(\cos)}}(r,\theta) = "
                        fr"J_{{{m}}}\!\left(\alpha_{{{m},{n}}}\frac{{r}}{{R}}\right)"
                        fr"\cos({m}\theta)"
                    )

                eq_real = MathTex(eq_tex_real).scale(0.8)
                eq_real.next_to(axes, DOWN, buff=0.4)
                eq_real.set_z_index(13)
                self.play(ReplacementTransform(equation, eq_real, run_time=0.4))
                equation = eq_real

                rgb, alpha = value_to_rgb_and_alpha(Z_real)
                rgba = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
                rgba[..., 0:3] = (rgb * 255).astype(np.uint8)
                rgba[..., 3]   = (alpha * 255).astype(np.uint8)

                img_real = Image.fromarray(rgba, mode="RGBA")
                img_mobj_real = ImageMobject(img_real).set_z_index(1)
                img_mobj_real.set_height(axes.height)
                img_mobj_real.move_to(axes.get_center())

                self.play(FadeIn(img_mobj_real, run_time=0.5))
                self.wait(0.4)
                self.play(FadeOut(img_mobj_real, run_time=0.5))

                # IMAG / sin part: skip for m=0 (it's identically zero)
                if m == 0:
                    continue

                Z_imag = radial * np.sin(m * THETA)
                eq_tex_imag = (
                    fr"\psi_{{{m},{n}}}^{{(\sin)}}(r,\theta) = "
                    fr"J_{{{m}}}\!\left(\alpha_{{{m},{n}}}\frac{{r}}{{R}}\right)"
                    fr"\sin({m}\theta)"
                )

                eq_imag = MathTex(eq_tex_imag).scale(0.8)
                eq_imag.next_to(axes, DOWN, buff=0.4)
                eq_imag.set_z_index(13)
                self.play(ReplacementTransform(equation, eq_imag, run_time=0.4))
                equation = eq_imag

                rgb, alpha = value_to_rgb_and_alpha(Z_imag)
                rgba = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
                rgba[..., 0:3] = (rgb * 255).astype(np.uint8)
                rgba[..., 3]   = (alpha * 255).astype(np.uint8)

                img_imag = Image.fromarray(rgba, mode="RGBA")
                img_mobj_imag = ImageMobject(img_imag).set_z_index(1)
                img_mobj_imag.set_height(axes.height)
                img_mobj_imag.move_to(axes.get_center())

                self.play(FadeIn(img_mobj_imag, run_time=0.5))
                self.wait(0.4)
                self.play(FadeOut(img_mobj_imag, run_time=0.5))

        self.wait(1)
