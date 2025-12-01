from manim import *
from manim.utils.color import ManimColor
import numpy as np
import sys
from pathlib import Path
from scipy.special import jv, jn_zeros

SCARLET = ManimColor("#F20000")
IMAGE_NEG_HEX = "#00D5FF"
IMAGE_POS_HEX = "#F26D00"
IMAGE_NEG_COLOR = ManimColor(IMAGE_NEG_HEX)
IMAGE_POS_COLOR = ManimColor(IMAGE_POS_HEX)
IMAGE_NEG_RGB = np.array(IMAGE_NEG_COLOR.to_rgb())
IMAGE_POS_RGB = np.array(IMAGE_POS_COLOR.to_rgb())
ZERO_RGB = np.array([0.0, 0.0, 0.0])

sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene


class ConvolutionContinuousVariableChange(RightRegionScene):
    def construct(self):
        self.camera.background_color = BLACK

        # --- Equations ---
        # Final equation: convolution notation with vector p
        equation_rot = MathTex(
            r"(X * F)(p) = \int_{\mathbb{R}^2} X(p + p') F(p')\,\mathrm{d}p'",
            color=WHITE
        ).scale(0.55).to_edge(UP, buff=0.5)

        self.add(equation_rot)

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
        axes_shift = LEFT * 5.9 + DOWN * 3.5
        axes.shift(axes_shift)
        axes_labels = axes.get_axis_labels(MathTex("s"), MathTex("t"))

        # --- background image ---
        # Matched to ConvolutionContinuousShiftImage
        input_image = self.create_input_image(resolution=360, span=6.5)
        input_image.set_width(axes.width * 3.2)
        input_image.set_z_index(-2)
        input_image.move_to(self.camera.frame_center) # Matched centering

        # --- kernel setup ---
        kernel_span = 3.0
        
        # Start with Bessel (m=0, n=1) as CIRCLE
        kernel_image = self.create_bessel_image(0, 1, "real", resolution=240, span=kernel_span, shape="circle")
        kernel_image.set_width(axes.width * 0.95)
        
        # Position at the "final" location
        final_pos = axes.c2p(1.6, 1.6) 
        kernel_image.move_to(final_pos)
        kernel_image.set_z_index(-1)

        # Visible kernel boundary (Circle)
        # In original square logic: width = kernel_image.width * 0.4
        # Circle radius was half that width.
        kernel_box = Circle(
            radius=(kernel_image.width * 0.4) / 2,
            color=SCARLET,
            stroke_width=2.0,
        )
        kernel_box.move_to(kernel_image)
        kernel_box.set_z_index(-0.9)

        # --- Kernel Axes (sigma, tau) moving with kernel ---
        def get_kernel_axes():
            center = kernel_box.get_center()
            # Determine width/height based on current box shape
            w = kernel_box.width
            h = kernel_box.height
            
            k_axes = Axes(
                x_range=[0, 1.5, 1],
                y_range=[0, 1.5, 1],
                x_length=w * 0.6,
                y_length=h * 0.6,
                axis_config={
                    "color": SCARLET,
                    "stroke_width": 2,
                    "include_tip": False,
                    "include_ticks": False,
                },
            )
            k_axes.move_to(center, aligned_edge=DL)
            
            sigma_tick = Line(UP * 0.08, DOWN * 0.08, color=SCARLET, stroke_width=2).move_to(k_axes.x_axis.get_end())
            tau_tick = Line(LEFT * 0.08, RIGHT * 0.08, color=SCARLET, stroke_width=2).move_to(k_axes.y_axis.get_end())

            sigma_label = MathTex(r"\sigma", color=SCARLET).scale(0.6)
            sigma_label.next_to(sigma_tick, RIGHT, buff=0.05)
            
            tau_label = MathTex(r"\tau", color=SCARLET).scale(0.6)
            tau_label.next_to(tau_tick, UP, buff=0.05)
            
            return VGroup(k_axes, sigma_tick, tau_tick, sigma_label, tau_label)
            
        kernel_axes_group = always_redraw(get_kernel_axes)
        kernel_axes_group.set_z_index(0)

        # --- Global Vector (s,t) -> p ---
        origin_point = axes.c2p(0, 0)
        vector_color = GREY_A
        
        # Trackers
        # Initialize to final states (p, p') and shrunk vector
        global_label_alpha = ValueTracker(1.0)
        offset_label_alpha = ValueTracker(1.0)
        vector_length_tracker = ValueTracker(0.8)
        angle_tracker = ValueTracker(0.0)

        def global_vector_group():
            target = kernel_box.get_center()
            arrow = Arrow(
                origin_point,
                target,
                buff=0,
                color=vector_color,
                stroke_width=2.5,
                tip_length=0.08,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)
            
            alpha = global_label_alpha.get_value()
            
            lbl_st = MathTex(r"(s,t)", color=vector_color).scale(0.5)
            lbl_p = MathTex(r"\mathbf{p}", color=vector_color).scale(0.5)
            
            pos = target + UP * 0.2
            lbl_st.move_to(pos).set_opacity(1 - alpha)
            lbl_p.move_to(pos).set_opacity(alpha)
            
            return VGroup(arrow, lbl_st, lbl_p)

        global_vector = always_redraw(global_vector_group)

        # --- Offset vector (sigma, tau) -> p' ---
        base_offset = np.array(
            [kernel_box.width * 0.48, kernel_box.height * 0.25, 0.0]
        ) * 0.8

        def offset_vector_group():
            base = kernel_box.get_center()
            angle = angle_tracker.get_value()
            scale = vector_length_tracker.get_value()
            
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Rotate base offset
            rotated_offset = np.array([
                base_offset[0] * cos_a - base_offset[1] * sin_a,
                base_offset[0] * sin_a + base_offset[1] * cos_a,
                base_offset[2]
            ])
            
            # Scale offset
            offset = rotated_offset * scale
            
            arrow = Arrow(
                base,
                base + offset,
                buff=0,
                color=SCARLET,
                stroke_width=2.2,
                tip_length=0.08,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)
            
            alpha = offset_label_alpha.get_value()
            
            lbl_st = MathTex(r"(\sigma,\tau)", color=SCARLET).scale(0.5)
            lbl_p = MathTex(r"\mathbf{p}'", color=SCARLET).scale(0.5)
            
            pos = base + offset + UP * 0.2
            lbl_st.move_to(pos).set_opacity(1 - alpha)
            lbl_p.move_to(pos).set_opacity(alpha)
            
            return VGroup(arrow, lbl_st, lbl_p)

        offset_vector = always_redraw(offset_vector_group)

        # --- Colorbar ---
        colorbar = Rectangle(
            width=0.35,
            height=2.2,
            stroke_color=GREY_B,
            stroke_width=1.2,
        )
        colorbar.set_fill(
            color=[IMAGE_NEG_COLOR, BLACK, IMAGE_POS_COLOR],
            opacity=1.0,
        )

        neg_label = MathTex("-1", color=GREY_B).scale(0.5)
        neg_label.next_to(colorbar, LEFT, buff=0.1)
        neg_label.align_to(colorbar, DOWN)

        zero_label = MathTex("0", color=GREY_B).scale(0.5)
        zero_label.next_to(colorbar, LEFT, buff=0.1)
        zero_label.move_to(colorbar.get_center() + LEFT * 0.3)

        pos_label = MathTex("1", color=GREY_B).scale(0.5)
        pos_label.next_to(colorbar, LEFT, buff=0.1)
        pos_label.align_to(colorbar, UP)

        colorbar_group = VGroup(colorbar, neg_label, zero_label, pos_label)
        colorbar_group.set_z_index(10)

        # Position in top-left corner with offset, matching ConvolutionPullback.py
        colorbar_group.to_corner(UL, buff=0.5).shift(DOWN * 1.0)

        # --- Setup Scene ---
        self.add(input_image)
        self.add(kernel_image)
        self.add(kernel_box)
        self.add(axes, axes_labels)
        self.add(kernel_axes_group)
        self.add(global_vector)
        self.add(offset_vector)
        self.add(colorbar_group)
        
        self.wait(1)

        current_kernel = kernel_image

        # --- Sequence of Fourier-Bessel Functions ---
        max_m = 3
        max_n = 3

        for m in range(max_m + 1):
            for n in range(1, max_n + 1):
                if m == 0 and n == 1:
                    continue

                # Real part
                new_img = self.create_bessel_image(m, n, "real", resolution=240, span=kernel_span, shape="circle")
                new_img.set_width(current_kernel.width)
                new_img.move_to(current_kernel.get_center())
                new_img.set_z_index(current_kernel.get_z_index())

                self.play(ReplacementTransform(current_kernel, new_img), run_time=1.0)
                current_kernel = new_img
                self.wait(0.5)

                # Imag part (if m > 0)
                if m > 0:
                    new_img_imag = self.create_bessel_image(m, n, "imag", resolution=240, span=kernel_span, shape="circle")
                    new_img_imag.set_width(current_kernel.width)
                    new_img_imag.move_to(current_kernel.get_center())
                    new_img_imag.set_z_index(current_kernel.get_z_index())

                    self.play(ReplacementTransform(current_kernel, new_img_imag), run_time=1.0)
                    current_kernel = new_img_imag
                    self.wait(0.5)

        self.wait(2)



    def create_bessel_image(self, m, n, part="real", resolution=240, span=3.0, shape="square"):
        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)

        # R_disk matching the support ratio (0.4 of total span 6.0 = 2.4 diameter -> 1.2 radius)
        # span is half-width (3.0). Image width is 2*span.
        # support_ratio = 0.4.
        R_disk = span * 0.4

        zeros_m = jn_zeros(m, n + 5)
        alpha_mn = zeros_m[n - 1]
        arg = alpha_mn * (R / R_disk)
        
        radial = jv(m, arg)
        # Mask strictly to disk for function definition
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

        # Shape mask
        h, w = resolution, resolution
        cy, cx = h // 2, w // 2
        yy, xx = np.indices((h, w))
        
        if shape == "square":
            half_side = int((h * 0.4) / 2)
            mask = (np.abs(yy - cy) <= half_side) & (np.abs(xx - cx) <= half_side)
        else:
            radius = (h * 0.4) / 2
            mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2

        alpha[~mask] = 0.0
        
        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[..., :3] = rgb
        rgba[..., 3] = alpha

        return ImageMobject(np.uint8(np.flipud(rgba) * 255))

    def create_input_image(self, resolution=320, span=6.0,
                           num_large=8, num_small=18):
        rng = np.random.default_rng(2025)

        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)

        field = np.zeros_like(X)

        # large-scale blobs
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

        # small/medium-scale blobs
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

        # sinusoidal modulation
        k1x, k1y = 0.8, 0.5
        k2x, k2y = 1.3, -0.9
        sinusoidal = (
            0.25 * np.sin(k1x * X + k1y * Y)
            + 0.18 * np.cos(k2x * X + k2y * Y)
        )
        field += sinusoidal

        # broad envelope
        env_sigma = span * 2.0
        envelope = np.exp(-(X**2 + Y**2) / (2.0 * env_sigma**2))
        field *= envelope

        # remove DC, stretch to [-1,1], then flip sign to swap colours
        field -= field.mean()
        max_abs = np.max(np.abs(field)) or 1.0
        v = field / max_abs
        v = -v  # swap cyan/orange globally

        # Color mapping using global constants
        v = np.clip(v, -1.0, 1.0)
        rgba = np.zeros((resolution, resolution, 4), dtype=float)

        pos_mask = v > 0
        neg_mask = v < 0

        if np.any(pos_mask):
            t = v[pos_mask]
            rgba[pos_mask, :3] = (1 - t)[:, None] * ZERO_RGB + t[:, None] * IMAGE_POS_RGB

        if np.any(neg_mask):
            t = -v[neg_mask]
            rgba[neg_mask, :3] = (1 - t)[:, None] * ZERO_RGB + t[:, None] * IMAGE_NEG_RGB

        gamma = 0.7
        alpha = 0.2 + 0.8 * np.power(np.abs(v), gamma)
        rgba[..., 3] = np.clip(alpha, 0.0, 1.0)

        input_image = ImageMobject(np.uint8(np.flipud(rgba) * 255))
        return input_image

    def create_kernel_image(self, resolution=220, span=3.0, cutoff=None, shape="square"):
        np.random.seed(10)

        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)
        field = np.zeros_like(X)

        num_blobs = 50
        for _ in range(num_blobs):
            amplitude = np.random.uniform(-1.0, 1.0)
            center_x = np.random.uniform(-1.0, 1.0)
            center_y = np.random.uniform(-1.0, 1.0)
            sigma_x = np.random.uniform(0.08, 0.1)
            sigma_y = np.random.uniform(0.08, 0.1)
            theta = np.random.uniform(0, np.pi)

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            Xc = X - center_x
            Yc = Y - center_y
            rotated_x = cos_t * Xc + sin_t * Yc
            rotated_y = -sin_t * Xc + cos_t * Yc

            blob = np.exp(
                -0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2)
            )
            field += amplitude * blob

        radial_mask = np.exp(-((X**2 + Y**2) / (2 * (span / 1.5) ** 2)))
        field *= radial_mask

        field -= field.mean()
        max_abs = np.max(np.abs(field)) or 1.0
        v = field / max_abs
        v = np.clip(v, -1.0, 1.0)

        contrast_gamma = 0.6
        v = np.sign(v) * np.power(np.abs(v), contrast_gamma)

        # Support Mask
        h, w = v.shape
        cy, cx = h // 2, w // 2
        # box width is ~0.4 of image width. 
        support_ratio = 0.4
        
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        
        if shape == "square":
            half_side_y = int((h * support_ratio) / 2)
            half_side_x = int((w * support_ratio) / 2)
            support_mask = (
                (np.abs(yy - cy) <= half_side_y) &
                (np.abs(xx - cx) <= half_side_x)
            )
        else:
            # Circle support
            # Radius should match half the square side to be inscribed/circumscribed?
            # User said "canvas ... hide inside the circle", implies circle outlines the square boundary
            # We'll use radius = half_side (inscribed in the box area)
            radius = (h * support_ratio) / 2
            dist_sq = (yy - cy)**2 + (xx - cx)**2
            support_mask = dist_sq <= radius**2

        rgba = np.zeros((h, w, 4), dtype=float)

        pos_mask = (v > 0) & support_mask
        neg_mask = (v < 0) & support_mask

        if np.any(pos_mask):
            t = v[pos_mask]
            rgba[pos_mask, :3] = (1 - t)[:, None] * ZERO_RGB + t[:, None] * IMAGE_POS_RGB

        if np.any(neg_mask):
            t = -v[neg_mask]
            rgba[neg_mask, :3] = (1 - t)[:, None] * ZERO_RGB + t[:, None] * IMAGE_NEG_RGB

        alpha_gamma = 0.7
        alpha = np.zeros_like(v)
        alpha[support_mask] = np.power(np.abs(v[support_mask]), alpha_gamma)

        alpha[alpha < 0.02] = 0.0
        rgba[..., 3] = np.clip(alpha, 0.0, 1.0)

        kernel_image = ImageMobject(np.uint8(np.flipud(rgba) * 255))
        return kernel_image
