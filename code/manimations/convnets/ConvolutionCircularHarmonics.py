from manim import *
from manim.utils.color import ManimColor
import numpy as np
import sys
from pathlib import Path
from scipy.special import jn_zeros, jv

sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene


class ConvolutionCircularHarmonics(RightRegionScene):
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
        angular_sequence = [
            (0, "cos"),
            (1, "cos"),
            (1, "sin"),
            (2, "cos"),
            (2, "sin"),
            (3, "cos"),
        ]
        basis_images, label_pairs = self.generate_circular_basis_images(
            num_radial=6,
            angular_sequence=angular_sequence,
            resolution=240,
            span=kernel_span,
        )
        kernel_width = axes.width * 0.98
        kernel_start = axes.c2p(0.2, 0.6)
        for image in basis_images:
            image.set_width(kernel_width)
            image.move_to(kernel_start)
            image.set_z_index(-1)

        bold_red = ManimColor("#ff4b3e")
        kernel_size = min(basis_images[0].width, basis_images[0].height) * 0.38
        kernel_radius = kernel_size / 0.78
        kernel_circle = Circle(
            radius=kernel_radius,
            stroke_color=bold_red,
            stroke_width=1.5,
        )
        kernel_circle.move_to(basis_images[0])
        kernel_circle.set_z_index(-1.1)

        up_target = axes.c2p(1.6, 1.6)
        basis_images[0].move_to(up_target)
        kernel_circle.move_to(up_target)

        basis_labels = [
            self.create_combined_label(r_idx, a_idx)
            for (r_idx, a_idx) in label_pairs
        ]
        label_position = axes.c2p(2.4, 4.6)
        for label in basis_labels:
            label.move_to(label_position)
            label.set_z_index(3)

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

        title = Text(
            "Localized kernel intensity",
            font_size=30,
            color=GREY_B,
        ).to_edge(UP)

        self.play(FadeIn(input_image, run_time=1.6))
        self.play(Create(axes), Write(axes_labels))
        self.play(
            FadeIn(basis_images[0], run_time=1.2),
            FadeIn(kernel_circle, run_time=1.2),
            FadeIn(basis_labels[0], run_time=1.2),
        )
        self.play(Write(title))
        self.play(
            FadeIn(global_vector, run_time=0.8),
            FadeIn(offset_vector, run_time=0.8),
        )
        self.wait(0.8)

        current_image = basis_images[0]
        current_label = basis_labels[0]
        for idx, next_image in enumerate(basis_images[1:], start=2):
            next_image.move_to(kernel_circle.get_center())
            next_label = basis_labels[idx - 1]
            self.play(
                ReplacementTransform(current_image, next_image),
                ReplacementTransform(current_label, next_label),
                run_time=1.2,
                rate_func=rate_functions.ease_in_out_sine,
            )
            current_image = next_image
            current_label = next_label
            self.wait(0.4)

        self.wait(1.5)

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
        return input_image

    def generate_circular_basis_images(
        self,
        num_radial=6,
        angular_sequence=None,
        resolution=220,
        span=3.0,
    ):
        if angular_sequence is None:
            angular_sequence = [
                (0, "cos"),
                (1, "cos"),
                (1, "sin"),
                (2, "cos"),
                (2, "sin"),
                (3, "cos"),
            ]

        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        radius = np.sqrt(grid_x**2 + grid_y**2)
        theta = np.mod(np.arctan2(grid_y, grid_x), 2.0 * np.pi)
        r_norm = np.clip(radius / span, 0.0, 1.0)
        mask = radius <= span

        zeros = jn_zeros(0, num_radial)
        radial_fields = []
        for zero in zeros:
            radial = jv(0, zero * r_norm)
            radial = np.where(mask, radial, 0.0)
            radial -= radial.min()
            max_value = radial.max() or 1.0
            radial /= max_value
            radial_fields.append(radial)

        angular_fields = []
        for order, kind in angular_sequence:
            if kind == "sin":
                field = np.sin(order * theta)
            else:
                field = np.cos(order * theta)
            if order == 0 and kind == "sin":
                field = np.zeros_like(theta)
            field = np.where(mask, field, 0.0)
            field -= field.min()
            max_value = field.max() or 1.0
            field /= max_value
            angular_fields.append(field)

        images = []
        label_pairs = []
        for radial_idx, radial in enumerate(radial_fields, start=1):
            for angular_idx, angular in enumerate(angular_fields, start=1):
                combined = radial * angular
                combined -= combined.min()
                max_value = combined.max() or 1.0
                combined /= max_value

                rgba = np.zeros((resolution, resolution, 4), dtype=float)
                rgba[..., 0] = combined
                rgba[..., 3] = np.clip(combined * 1.4, 0, 1) * mask

                image = ImageMobject(np.uint8(np.flipud(rgba) * 255))
                images.append(image)
                label_pairs.append((radial_idx, angular_idx))

        return images, label_pairs

    def create_combined_label(self, radial_idx, angular_idx):
        label = MathTex(
            r"K_{%d,%d}(r,\theta) = R_{%d}(r) A_{%d}(\theta)"
            % (radial_idx, angular_idx, radial_idx, angular_idx),
            color=GREY_B,
        ).scale(0.7)
        return label

