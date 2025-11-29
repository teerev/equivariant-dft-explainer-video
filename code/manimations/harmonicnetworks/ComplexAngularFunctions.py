from manim import *
import numpy as np
from PIL import Image


class AngularHarmonics(Scene):
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

        # === Equation below the plot ===
        equation = MathTex(r"\Re[e^{i1\theta}] = \cos(\theta)").scale(0.9)
        equation.next_to(axes, DOWN, buff=0.4)
        equation.set_z_index(13)
        self.play(FadeIn(equation, run_time=0.5))

        # Colour scheme (saturated cyan/orange)
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

        # Grid for angular harmonics
        grid_size = 512
        r_max = 4.0

        xs = np.linspace(-r_max, r_max, grid_size)
        ys = np.linspace(-r_max, r_max, grid_size)
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)

        # Angular modes
        max_m = 8   # up to m=8 (change if needed)

        for m in range(1, max_m + 1):
            # === REAL PART: cos(mθ) ===
            Z_real = np.cos(m * THETA)

            new_eq = MathTex(
                fr"\Re[e^{{i {m}\theta}}] = \cos({m}\theta)"
            ).scale(0.9)
            new_eq.next_to(axes, DOWN, buff=0.4)
            new_eq.set_z_index(13)
            self.play(ReplacementTransform(equation, new_eq, run_time=0.35))
            equation = new_eq

            rgb, alpha = value_to_rgb_and_alpha(Z_real)
            rgba = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
            rgba[..., :3] = (rgb * 255).astype(np.uint8)
            rgba[..., 3]  = (alpha * 255).astype(np.uint8)

            img_real = Image.new("RGBA", (grid_size, grid_size))
            img_real.putdata([tuple(p) for p in rgba.reshape(-1, 4)])
            img_mobj = ImageMobject(img_real).set_z_index(1)
            img_mobj.set_height(axes.height)
            img_mobj.move_to(axes.get_center())

            self.play(FadeIn(img_mobj, run_time=0.5))
            self.wait(0.4)
            self.play(FadeOut(img_mobj, run_time=0.5))

            # === IMAG PART: sin(mθ) ===
            Z_imag = np.sin(m * THETA)

            new_eq2 = MathTex(
                fr"\Im[e^{{i {m}\theta}}] = \sin({m}\theta)"
            ).scale(0.9)
            new_eq2.next_to(axes, DOWN, buff=0.4)
            new_eq2.set_z_index(13)
            self.play(ReplacementTransform(equation, new_eq2, run_time=0.35))
            equation = new_eq2

            rgb, alpha = value_to_rgb_and_alpha(Z_imag)
            rgba = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
            rgba[..., :3] = (rgb * 255).astype(np.uint8)
            rgba[..., 3]  = (alpha * 255).astype(np.uint8)

            img_imag = Image.new("RGBA", (grid_size, grid_size))
            img_imag.putdata([tuple(p) for p in rgba.reshape(-1, 4)])
            img_mobj2 = ImageMobject(img_imag).set_z_index(1)
            img_mobj2.set_height(axes.height)
            img_mobj2.move_to(axes.get_center())

            self.play(FadeIn(img_mobj2, run_time=0.5))
            self.wait(0.4)
            self.play(FadeOut(img_mobj2, run_time=0.5))

        self.wait(1)
