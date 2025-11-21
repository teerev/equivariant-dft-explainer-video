from manim import *
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene

class ConvolutionRotationalNonEquivariance(RightRegionScene):
    def construct(self):
        self.camera.background_color = BLACK

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={
                "color": GREY_B,
                "stroke_width": 2,
                "include_tip": False,
            },
        )
        axes.shift(LEFT * 1.5 + DOWN * 1.5)
        axes_shift = LEFT * 1.5 + DOWN * 1.0
        axes.shift(axes_shift)
        axes_labels = axes.get_axis_labels(MathTex("x"), MathTex("y"))

        input_image = self.create_input_image(resolution=360, span=6.5)
        input_image.set_width(axes.width * 3.2)
        input_image.set_z_index(-2)
        input_image.move_to(axes.get_center() - axes_shift)

        kernel_span = 3.0
        kernel_image = self.create_kernel_image(resolution=240, span=kernel_span)
        kernel_image.set_width(axes.width * 0.95)
        kernel_start = axes.c2p(-0.6, -0.4) - axes_shift
        kernel_image.move_to(kernel_start)
        kernel_image.set_z_index(-1)

        kernel_box = Rectangle(
            width=kernel_image.width * 0.4,
            height=kernel_image.height * 0.4,
            stroke_color=RED_E,
            stroke_width=1.5,
        )
        kernel_box.move_to(kernel_image)
        kernel_box.set_z_index(-1.1)
        kernel_group = Group(kernel_image, kernel_box) 

        title = Text(
            "Localized kernel intensity",
            font_size=30,
            color=GREY_B,
        ).to_edge(UP)

        self.play(FadeIn(input_image, run_time=2.0))
        self.play(FadeIn(kernel_group, run_time=1.2))
        self.play(Create(axes), Write(axes_labels))
        self.play(Write(title))
        right_target = axes.c2p(1.0, -0.4) - axes_shift
        up_target = axes.c2p(1.0, 1.2) - axes_shift

        self.play(
            kernel_group.animate.move_to(right_target),
            run_time=4,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.play(
            kernel_group.animate.move_to(up_target),
            run_time=4,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait(2)


    def create_input_image(self, resolution=320, span=6.0, num_centers=60):
        rng = np.random.default_rng(2024)
        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        field = np.zeros_like(grid_x)

        for _ in range(num_centers):
            weight = rng.uniform(0.6, 1.4)
            center_x = rng.uniform(-span * 0.4, span * 0.4)
            center_y = rng.uniform(-span * 0.4, span * 0.4)
            beta = rng.uniform(0.15, 0.5)

            radial_distance = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
            gumbel = np.exp(-(radial_distance / beta + np.exp(-radial_distance / beta)))
            field += weight * gumbel

        envelope = np.exp(-((grid_x**2 + grid_y**2) / (2 * (span * 0.6) ** 2)))
        field *= envelope

        field -= field.min()
        max_value = field.max() or 1.0
        field /= max_value

        rgba = np.zeros((resolution, resolution, 4))
        rgba[..., 2] = field
        rgba[..., 3] = np.clip(field * 1.05, 0, 1)

        input_image = ImageMobject(np.uint8(np.flipud(rgba) * 255))
        # ❌ DO NOT call input_image.set_opacity(...)
        return input_image


    def create_kernel_image(self, resolution=220, span=3.0, cutoff=0.15):
        np.random.seed(7)
        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        field = np.zeros_like(grid_x)

        num_blobs = 21
        for _ in range(num_blobs):
            amplitude = np.random.uniform(0.6, 1.0)
            center_x = np.random.uniform(-1.0, 1.0)
            center_y = np.random.uniform(-1.0, 1.0)
            sigma_x = np.random.uniform(0.08, 0.25)
            sigma_y = np.random.uniform(0.08, 0.25)
            theta = np.random.uniform(0, np.pi)

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotated_x = cos_t * (grid_x - center_x) + sin_t * (grid_y - center_y)
            rotated_y = -sin_t * (grid_x - center_x) + cos_t * (grid_y - center_y)

            blob = amplitude * np.exp(
                -0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2)
            )
            field += blob

        radial_mask = np.exp(-((grid_x**2 + grid_y**2) / (2 * (span / 1.5) ** 2)))
        field *= radial_mask

        field -= field.min()
        max_value = field.max() or 1.0
        field /= max_value

        # Only keep the “hot” parts of the kernel
        mask = field > cutoff

        rgba = np.zeros((resolution, resolution, 4), dtype=float)
        rgba[..., 0] = np.power(field, 0.9) * mask          # red channel
        rgba[..., 3] = np.clip(field * 1.4, 0, 1) * mask    # alpha only where mask == True

        kernel_image = ImageMobject(np.uint8(np.flipud(rgba) * 255))
        # ❌ DO NOT call kernel_image.set_opacity(...)
        return kernel_image