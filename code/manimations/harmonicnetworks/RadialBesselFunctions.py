from manim import *
import numpy as np
from scipy.special import jv
from PIL import Image


class BesselAnimation(Scene):
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

        # Draw static geometry
        self.play(Create(axes), FadeIn(axes_labels))
        self.play(GrowArrow(vector), FadeIn(vec_label))
        self.wait(0.5)

        axes.set_z_index(10)
        axes_labels.set_z_index(11)
        vector.set_z_index(12)
        vec_label.set_z_index(12)

        # === LaTeX equation under the plot (animated per mode) ===
        equation = MathTex(r"R_1(r) = J_1(k r)").scale(0.9)
        equation.next_to(axes, DOWN, buff=0.4)
        equation.set_z_index(13)
        self.play(FadeIn(equation, run_time=0.5))

        # Colour scheme (saturated cyan/orange)
        C_CYAN   = np.array([0.0, 0.8, 1.0])     # vibrant cyan
        C_ORANGE = np.array([1.0, 0.35, 0.0])    # bright orange

        def value_to_rgb_and_alpha(z):
            z_clipped = np.clip(z, -1.0, 1.0)
            mag = np.abs(z_clipped)

            # gamma < 1 boosts mid-range visibility
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

        # Grid for Bessel fields
        grid_size = 512
        r_max = 4.0

        xs = np.linspace(-r_max, r_max, grid_size)
        ys = np.linspace(-r_max, r_max, grid_size)
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X**2 + Y**2)

        scale_factor = 4.0   # visual scaling of Bessel argument

        for n in range(1, 11):
            # Update Equation
            new_eq = MathTex(fr"R_{{{n}}}(r) = J_{{{n}}}(k r)").scale(0.9)
            new_eq.next_to(axes, DOWN, buff=0.4)
            new_eq.set_z_index(13)
            self.play(ReplacementTransform(equation, new_eq, run_time=0.35))
            equation = new_eq

            # Compute Bessel field
            Z = jv(n, scale_factor * R)
            rgb, alpha = value_to_rgb_and_alpha(Z)

            # Convert to RGBA
            rgba = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
            rgba[..., 0:3] = (rgb * 255).astype(np.uint8)
            rgba[..., 3]   = (alpha * 255).astype(np.uint8)

            img = Image.fromarray(rgba, mode="RGBA")
            img_mobj = ImageMobject(img).set_z_index(1)

            img_mobj.set_height(axes.height)
            img_mobj.move_to(axes.get_center())

            self.play(FadeIn(img_mobj, run_time=0.5))
            self.wait(0.5)
            self.play(FadeOut(img_mobj, run_time=0.5))

        self.wait(1)
