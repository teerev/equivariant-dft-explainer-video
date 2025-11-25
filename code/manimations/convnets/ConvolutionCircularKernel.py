from manim import *
from manim.utils.color import ManimColor
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene

class ConvolutionCircularKernel(RightRegionScene):
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
        axes_labels = axes.get_axis_labels(MathTex("s"), MathTex("t"))

        input_image = self.create_input_image(resolution=360, span=6.5)
        input_image.set_width(axes.width * 3.2)
        input_image.set_z_index(-2)
        input_image.move_to(axes.c2p(0, 0))

        kernel_span = 3.0
        kernel_image = self.create_kernel_image(resolution=240, span=kernel_span)
        kernel_image.set_width(axes.width * 0.95)
        kernel_start = axes.c2p(0.2, 0.6)
        kernel_image.move_to(kernel_start)
        kernel_image.set_z_index(-1)

        bold_red = ManimColor("#ff4b3e")
        kernel_size = min(kernel_image.width, kernel_image.height) * 0.38
        kernel_radius = kernel_size / 1.5
        kernel_circle = Circle(
            radius=kernel_radius,
            stroke_color=bold_red,
            stroke_width=1.5,
        )
        kernel_circle.move_to(kernel_image)
        kernel_circle.set_z_index(-1.1)
        circle_center = kernel_circle.get_center()

        origin_point = axes.c2p(0, 0)
        vector_color = GREY_A

        def create_vector_label(tex, color, arrow, offset=0.08):
            direction = arrow.get_end() - arrow.get_start()
            norm = np.linalg.norm(direction)
            if norm == 0:
                norm = 1.0
                direction = np.array([1.0, 0.0, 0.0])
            unit_direction = direction / norm
            label = MathTex(tex, color=color).scale(0.5)
            label.move_to(arrow.get_end() + unit_direction * offset)
            return label

        def global_vector_group():
            target = kernel_circle.get_center()
            arrow = Arrow(
                origin_point,
                target,
                buff=0,
                color=vector_color,
                stroke_width=2.2,
                tip_length=0.08,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)
            label = create_vector_label(r"\mathbf{p}", vector_color, arrow)
            return VGroup(arrow, label)

        def offset_vector_group():
            base = kernel_circle.get_center()
            offset = np.array(
                [kernel_size * -0.48, kernel_size * 0.25, 0.0]
            ) * 0.5
            arrow = Arrow(
                base,
                base + offset,
                buff=0,
                color=YELLOW_E,
                stroke_width=2.2,
                tip_length=0.08,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)
            label = create_vector_label(r"\mathbf{p}'", YELLOW_E, arrow)
            return VGroup(arrow, label)

        global_vector = always_redraw(global_vector_group)
        offset_vector = always_redraw(offset_vector_group)

        kernel_group = Group(kernel_image, kernel_circle)

        title = Text(
            "Localized kernel intensity",
            font_size=30,
            color=GREY_B,
        ).to_edge(UP)

        self.play(FadeIn(input_image, run_time=2.0))
        self.play(FadeIn(kernel_group, run_time=1.2))
        self.play(Create(axes), Write(axes_labels))
        self.play(Write(title))
        self.play(
            FadeIn(global_vector, run_time=0.8),
            FadeIn(offset_vector, run_time=0.8),
        )
        right_target = axes.c2p(1.6, 0.6)
        up_target = axes.c2p(1.6, 1.6)

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

        circle_mask = (grid_x**2 + grid_y**2) <= span**2  # inscribed circle
        field *= circle_mask.astype(float)

        field -= field.min()
        max_value = field.max() or 1.0
        field /= max_value

        # Only keep the “hot” parts of the kernel but ensure the support stays circular.
        mask = field > cutoff

        rgba = np.zeros((resolution, resolution, 4), dtype=float)
        rgba[..., 0] = np.power(field, 0.9) * mask          # red channel
        rgba[..., 3] = np.clip(field * 1.4, 0, 1) * circle_mask

        kernel_image = ImageMobject(np.uint8(np.flipud(rgba) * 255))
        # ❌ DO NOT call kernel_image.set_opacity(...)
        return kernel_image