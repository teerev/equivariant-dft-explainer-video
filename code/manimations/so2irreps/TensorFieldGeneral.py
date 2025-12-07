# pyright: reportMissingImports=false
import manim as mn
import numpy as np
import sys
from pathlib import Path
from scipy.special import jv, jn_zeros

SCARLET = mn.ManimColor("#F20000")
IMAGE_NEG_HEX = "#00D5FF"
IMAGE_POS_HEX = "#F26D00"
IMAGE_NEG_COLOR = mn.ManimColor(IMAGE_NEG_HEX)
IMAGE_POS_COLOR = mn.ManimColor(IMAGE_POS_HEX)
IMAGE_NEG_RGB = np.array(IMAGE_NEG_COLOR.to_rgb())
IMAGE_POS_RGB = np.array(IMAGE_POS_COLOR.to_rgb())

PURE_BLUE = mn.ManimColor("#0000FF")
PURE_BLUE_RGB = np.array(PURE_BLUE.to_rgb())

sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene


class TensorFieldSceneBase(RightRegionScene):
    def build_scene_elements(self):
        self.camera.background_color = mn.BLACK

        axes = mn.Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={
                "color": mn.GREY_B,
                "stroke_width": 2,
                "include_tip": False,
            },
        )
        axes_shift = mn.LEFT * 5.9 + mn.DOWN * 3.0
        axes.shift(axes_shift)
        axes_labels = axes.get_axis_labels(mn.MathTex("x"), mn.MathTex("y"))

        # --- Pure blue background field ---
        input_image = self.create_input_image(resolution=360, span=6.5)
        input_image.set_width(axes.width * 3.2)
        input_image.set_z_index(-2)
        input_image.move_to(self.camera.frame_center)

        # --- Kernel setup ---
        kernel_span = 3.0
        kernel_image = self.create_bessel_image(
            2, 1, "real", resolution=240, span=kernel_span, shape="circle"
        )
        kernel_image.set_width(axes.width * 0.95)
        kernel_image.set_z_index(-1)
        kernel_image.move_to(axes.c2p(1.6, 1.6))

        kernel_box = mn.Circle(
            radius=(kernel_image.width * 0.4) / 2,
            color=SCARLET,
            stroke_width=2.0,
        )
        kernel_box.move_to(kernel_image)
        kernel_box.set_z_index(-0.9)

        # --- Kernel Axes (sigma, tau) moving with kernel ---
        def get_kernel_axes():
            center = kernel_box.get_center()
            w = kernel_box.width
            h = kernel_box.height

            k_axes = mn.Axes(
                x_range=[-0.4, 1.5, 1],
                y_range=[-0.4, 1.5, 1],
                x_length=w * 0.6,
                y_length=h * 0.6,
                axis_config={
                    "color": SCARLET,
                    "stroke_width": 2,
                    "include_tip": False,
                    "include_ticks": False,
                },
            )
            k_axes.shift(center - k_axes.c2p(0, 0))

            sigma_tick = mn.Line(
                mn.UP * 0.08, mn.DOWN * 0.08, color=SCARLET, stroke_width=2
            ).move_to(
                k_axes.x_axis.get_end()
            )
            tau_tick = mn.Line(
                mn.LEFT * 0.08, mn.RIGHT * 0.08, color=SCARLET, stroke_width=2
            ).move_to(
                k_axes.y_axis.get_end()
            )

            sigma_label = mn.MathTex(r"x'", color=SCARLET).scale(0.6)
            sigma_label.next_to(sigma_tick, mn.RIGHT, buff=0.05)

            tau_label = mn.MathTex(r"y'", color=SCARLET).scale(0.6)
            tau_label.next_to(tau_tick, mn.UP, buff=0.05)

            return mn.VGroup(k_axes, sigma_tick, tau_tick, sigma_label, tau_label)

        kernel_axes_group = mn.always_redraw(get_kernel_axes)
        kernel_axes_group.set_z_index(0)

        # --- Global Vector (s,t) -> p ---
        origin_point = axes.c2p(0, 0)
        vector_color = mn.GREY_A

        global_label_alpha = mn.ValueTracker(1.0)
        offset_label_alpha = mn.ValueTracker(1.0)
        vector_length_tracker = mn.ValueTracker(0.8)
        angle_tracker = mn.ValueTracker(0.0)

        def global_vector_group():
            target = kernel_box.get_center()
            arrow = mn.Arrow(
                origin_point,
                target,
                buff=0,
                color=vector_color,
                stroke_width=2.5,
                tip_length=0.08,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)

            alpha = global_label_alpha.get_value()

            lbl_st = mn.MathTex(r"(s,t)", color=vector_color).scale(0.5)
            lbl_p = mn.MathTex(r"\mathbf{p}", color=vector_color).scale(0.5)

            pos = target + mn.UP * 0.2
            lbl_st.move_to(pos).set_opacity(1 - alpha)
            lbl_p.move_to(pos).set_opacity(alpha)

            return mn.VGroup(arrow, lbl_st, lbl_p)

        global_vector = mn.always_redraw(global_vector_group)

        # --- Offset vector (sigma, tau) -> p' ---
        base_offset = np.array([kernel_box.width * 0.48, kernel_box.height * 0.25, 0.0]) * 0.8

        def offset_vector_group():
            base = kernel_box.get_center()
            angle = angle_tracker.get_value()
            scale = vector_length_tracker.get_value()

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotated_offset = np.array(
                [
                    base_offset[0] * cos_a - base_offset[1] * sin_a,
                    base_offset[0] * sin_a + base_offset[1] * cos_a,
                    base_offset[2],
                ]
            )
            offset = rotated_offset * scale

            arrow = mn.Arrow(
                base,
                base + offset,
                buff=0,
                color=SCARLET,
                stroke_width=2.2,
                tip_length=0.08,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)

            alpha = offset_label_alpha.get_value()

            lbl_st = mn.MathTex(r"(\sigma,\tau)", color=SCARLET).scale(0.5)
            lbl_p = mn.MathTex(r"\mathbf{p}'", color=SCARLET).scale(0.5)

            pos = base + offset + mn.UP * 0.2
            lbl_st.move_to(pos).set_opacity(1 - alpha)
            lbl_p.move_to(pos).set_opacity(alpha)

            return mn.VGroup(arrow, lbl_st, lbl_p)

        offset_vector = mn.always_redraw(offset_vector_group)

        # --- Field annotation (feature vector) ---
        annotation_point = self.camera.frame_center + mn.LEFT * 2.0 + mn.UP * 1.1
        annotation_dot = mn.Dot(annotation_point, color=mn.WHITE, radius=0.04).set_z_index(5)

        feature_vector_tex = (
            r"\begin{pmatrix}"
            r" T(p) \\[6pt]"
            r" \sigma^{(0)}(p) \\[6pt]"
            r" u^{(+1)}(p) \\[6pt]"
            r" u^{(-1)}(p) \\[6pt]"
            r" \sigma^{(+2)}(p) \\[6pt]"
            r" \sigma^{(-2)}(p)"
            r"\end{pmatrix}"
        )
        annotation_label = mn.MathTex(feature_vector_tex, color=mn.WHITE).scale(0.5)
        annotation_label.next_to(annotation_dot, mn.UP + mn.LEFT, buff=0.15)
        annotation_label.shift(mn.DOWN * 0.1)
        annotation_label.set_z_index(5)

        annotation_group = mn.VGroup(annotation_dot, annotation_label)
        annotation_group.set_z_index(5)

        base_mobjects = [
            input_image,
            kernel_image,
            kernel_box,
            axes,
            axes_labels,
            kernel_axes_group,
            global_vector,
            offset_vector,
        ]

        # Compute static endpoint of p' (offset vector tip) and p for potential annotations/points
        angle_init = angle_tracker.get_value()
        scale_init = vector_length_tracker.get_value()
        cos_a, sin_a = np.cos(angle_init), np.sin(angle_init)
        rotated_offset = np.array(
            [
                base_offset[0] * cos_a - base_offset[1] * sin_a,
                base_offset[0] * sin_a + base_offset[1] * cos_a,
                base_offset[2],
            ]
        )
        offset = rotated_offset * scale_init
        p_prime_point = kernel_box.get_center() + offset
        p_point = kernel_box.get_center()  # endpoint of global vector (p)

        return base_mobjects, annotation_group, input_image, p_prime_point, p_point

    def get_bessel_label(self, m, n, part):
        func_type = r"(\cos)" if part == "real" else r"(\sin)"
        trig = r"\cos" if part == "real" else r"\sin"

        if m == 0:
            return fr"F_{{{m},{n}}}(r',\theta') = J_{{{m}}}\!\left(\alpha_{{{m},{n}}}\frac{{r'}}{{R}}\right)"

        return (
            fr"F_{{{m},{n}}}^{{{func_type}}}(r',\theta') = "
            fr"J_{{{m}}}\!\left(\alpha_{{{m},{n}}}\frac{{r'}}{{R}}\right)"
            fr"\,{trig}({m}\theta')"
        )

    def create_bessel_image(self, m, n, part="real", resolution=240, span=3.0, shape="square"):
        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)

        R_disk = span * 0.4
        zeros_m = jn_zeros(m, n + 5)
        alpha_mn = zeros_m[n - 1]
        arg = alpha_mn * (R / R_disk)

        radial = jv(m, arg)
        radial = np.where(R <= R_disk, radial, 0.0)

        if part == "real":
            Z = radial if m == 0 else radial * np.cos(m * THETA)
        else:
            Z = np.zeros_like(radial) if m == 0 else radial * np.sin(m * THETA)

        z_clipped = np.clip(Z, -1.0, 1.0)
        mag = np.abs(z_clipped)
        gamma = 0.3
        mag_gamma = mag**gamma

        base = np.where(
            z_clipped[..., None] >= 0,
            IMAGE_POS_RGB[None, None, :],
            IMAGE_NEG_RGB[None, None, :],
        )
        rgb = base * mag_gamma[..., None]
        alpha = mag_gamma

        h, w = resolution, resolution
        cy, cx = h // 2, w // 2
        yy, xx = np.indices((h, w))

        if shape == "square":
            half_side = int((h * 0.4) / 2)
            mask = (np.abs(yy - cy) <= half_side) & (np.abs(xx - cx) <= half_side)
        else:
            radius = (h * 0.4) / 2
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

        alpha[~mask] = 0.0

        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[..., :3] = rgb
        rgba[..., 3] = alpha

        return mn.ImageMobject(np.uint8(np.flipud(rgba) * 255))

    def create_input_image(self, resolution=320, span=6.0, num_large=8, num_small=18):
        rng = np.random.default_rng(2025)

        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)

        field = np.zeros_like(X)

        for _ in range(num_large):
            amp = rng.uniform(-1.0, 1.0)
            cx = rng.uniform(-span * 0.4, span * 0.4)
            cy = rng.uniform(-span * 0.4, span * 0.4)
            sx = rng.uniform(1.0, 2.0)
            sy = rng.uniform(1.0, 2.0)
            theta = rng.uniform(0, np.pi)

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            Xc = X - cx
            Yc = Y - cy
            xr = cos_t * Xc + sin_t * Yc
            yr = -sin_t * Xc + cos_t * Yc

            gauss = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
            field += amp * gauss

        for _ in range(num_small):
            amp = rng.uniform(-0.7, 0.7)
            cx = rng.uniform(-span * 0.6, span * 0.6)
            cy = rng.uniform(-span * 0.6, span * 0.6)
            sx = rng.uniform(0.3, 0.9)
            sy = rng.uniform(0.3, 0.9)
            theta = rng.uniform(0, np.pi)

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            Xc = X - cx
            Yc = Y - cy
            xr = cos_t * Xc + sin_t * Yc
            yr = -sin_t * Xc + cos_t * Yc

            gauss = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
            field += amp * gauss

        k1x, k1y = 0.8, 0.5
        k2x, k2y = 1.3, -0.9
        sinusoidal = 0.25 * np.sin(k1x * X + k1y * Y) + 0.18 * np.cos(k2x * X + k2y * Y)
        field += sinusoidal

        env_sigma = span * 2.0
        envelope = np.exp(-(X**2 + Y**2) / (2.0 * env_sigma**2))
        field *= envelope

        field -= field.mean()
        max_abs = np.max(np.abs(field)) or 1.0
        v = field / max_abs
        v = np.clip(v, -1.0, 1.0)

        magnitude = np.abs(v)
        gamma = 0.7
        mag_gamma = magnitude**gamma

        rgba = np.zeros((resolution, resolution, 4), dtype=float)
        rgba[..., :3] = mag_gamma[..., None] * PURE_BLUE_RGB

        alpha = 0.2 + 0.8 * mag_gamma
        rgba[..., 3] = np.clip(alpha, 0.0, 1.0)

        input_image = mn.ImageMobject(np.uint8(np.flipud(rgba) * 255))
        return input_image

    def create_point_cloud(self, num_points=100, x_fraction=0.33, y_fraction=0.5):
        """Generate a blue point cloud in the lower-left region."""
        rng = np.random.default_rng(2042)
        width = mn.config.frame_width
        height = mn.config.frame_height

        x_min = -width / 2
        x_max = x_min + width * x_fraction
        y_min = -height / 2
        y_max = y_min + height * y_fraction

        dots = []
        for _ in range(num_points):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            dot = mn.Dot(point=[x, y, 0.0], radius=0.03, color=PURE_BLUE)
            dot.set_fill(PURE_BLUE, opacity=0.95)
            dot.set_stroke(color=PURE_BLUE, width=0.0)
            dots.append(dot)

        cloud = mn.VGroup(*dots)
        cloud.set_z_index(6)  # ensure dots render above other elements
        return cloud


class TensorFieldGeneralStage1(TensorFieldSceneBase):
    def construct(self):
        base_mobjects, _, _, _, _ = self.build_scene_elements()
        self.add(*base_mobjects)
        self.wait(5)


class TensorFieldGeneralStage2(TensorFieldSceneBase):
    def construct(self):
        base_mobjects, annotation_group, _, _, _ = self.build_scene_elements()
        self.add(*base_mobjects)
        self.wait(2)
        self.play(mn.FadeIn(annotation_group, shift=mn.UP * 0.2), run_time=1.2)
        self.wait(3)


class TensorFieldGeneralStage3(TensorFieldSceneBase):
    def construct(self):
        base_mobjects, annotation_group, input_image, p_prime_point, p_point = self.build_scene_elements()
        self.add(*base_mobjects)
        self.wait(1.5)

        point_cloud = self.create_point_cloud(num_points=120)
        tip_dot = mn.Dot(p_prime_point, radius=0.03, color=PURE_BLUE)
        tip_dot.set_fill(PURE_BLUE, opacity=0.98)
        tip_dot.set_stroke(color=PURE_BLUE, width=0.0)
        tip_dot.set_z_index(7)
        base_dot = mn.Dot(p_point, radius=0.03, color=PURE_BLUE)
        base_dot.set_fill(PURE_BLUE, opacity=0.98)
        base_dot.set_stroke(color=PURE_BLUE, width=0.0)
        base_dot.set_z_index(7)
        point_cloud.add(tip_dot, base_dot)
        # Crossfade to avoid alignment issues between ImageMobject and VGroup
        self.play(
            mn.FadeOut(input_image, run_time=1.2),
            mn.FadeIn(point_cloud, run_time=1.2),
        )
        self.bring_to_front(point_cloud)
        self.wait(2.0)

