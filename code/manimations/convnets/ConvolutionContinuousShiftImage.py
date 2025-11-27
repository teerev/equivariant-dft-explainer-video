from manim import *
from manim.utils.color import ManimColor
import numpy as np
import sys
from pathlib import Path

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


class ConvolutionContinuousShiftImage(RightRegionScene):
    def construct(self):
        self.camera.background_color = BLACK

        # Create convolution equation at top of frame (finite limits initially)
        equation_finite = MathTex(
            r"Y(s,t) = (X * F)(s,t) = \int_{-A/2}^{A/2}\!\!\int_{-B/2}^{B/2} X(s + \sigma,\; t + \tau) F(\sigma,\tau)\,\mathrm{d}\sigma\mathrm{d}\tau",
            color=WHITE
        ).scale(0.55).to_edge(UP, buff=0.5)

        # Create equation with infinite limits for morphing
        equation_infinite = MathTex(
            r"Y(s,t) = (X * F)(s,t) = \int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty} X(s + \sigma,\; t + \tau) F(\sigma,\tau)\,\mathrm{d}\sigma\mathrm{d}\tau",
            color=WHITE
        ).scale(0.55).to_edge(UP, buff=0.5)

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

        # --- background image: richer signed field in bright cyan/orange ---
        input_image = self.create_input_image(resolution=360, span=6.5)
        input_image.set_width(axes.width * 3.2)
        input_image.set_z_index(-2)
        input_image.move_to(self.camera.frame_center)  # Position at screen center, not axes origin

        # --- kernel: bright cyan/orange, smaller ---
        kernel_span = 3.0
        kernel_image = self.create_kernel_image(resolution=240, span=kernel_span)
        kernel_image.set_width(axes.width * 0.95)
        kernel_start = axes.c2p(1.2, 1.2)  # Start at current ending position
        kernel_image.move_to(kernel_start)
        kernel_image.set_z_index(-1)

        # Visible kernel box
        kernel_box = Rectangle(
            width=kernel_image.width * 0.4,
            height=kernel_image.height * 0.4,
            stroke_color=SCARLET,
            stroke_width=2.0,
        )
        kernel_box.move_to(kernel_image)
        
        # --- Kernel Axes (sigma, tau) moving with kernel ---
        def get_kernel_axes():
            center = kernel_box.get_center()
            k_axes = Axes(
                x_range=[0, 1.5, 1],
                y_range=[0, 1.5, 1],
                x_length=kernel_box.width * 0.6,
                y_length=kernel_box.height * 0.6,
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

        origin_point = axes.c2p(0, 0)
        vector_color = GREY_A

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
            label = MathTex(r"(s,t)", color=vector_color).scale(0.5)
            label.move_to(target + UP * 0.2)
            return VGroup(arrow, label)

        base_offset = np.array(
            [kernel_box.width * 0.48, kernel_box.height * 0.25, 0.0]
        ) * 0.8
        angle_tracker = ValueTracker(0.0)

        def offset_vector_group():
            base = kernel_box.get_center()
            angle = angle_tracker.get_value()
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            offset = np.array(
                [
                    base_offset[0] * cos_a - base_offset[1] * sin_a,
                    base_offset[0] * sin_a + base_offset[1] * cos_a,
                    base_offset[2],
                ]
            )
            arrow = Arrow(
                base,
                base + offset,
                buff=0,
                color=SCARLET,
                stroke_width=2.2,
                tip_length=0.22,
                max_tip_length_to_length_ratio=0.06,
            ).set_z_index(2)
            label = MathTex(r"(\sigma,\tau)", color=SCARLET).scale(0.5)
            label.move_to(base + offset + UP * 0.2)
            
            return VGroup(arrow, label)

        global_vector = always_redraw(global_vector_group)
        offset_vector = always_redraw(offset_vector_group)

        kernel_group = Group(kernel_image, kernel_box) 

        # --- Colorbar Legend ---
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
        
        # Position relative to camera frame UL to ensure visibility
        # self.camera.frame_center already shifted in RightRegionScene
        frame_ul = self.camera.frame_center + (LEFT * config.frame_width / 2) + (UP * config.frame_height / 2)
        colorbar.move_to(frame_ul + RIGHT * 0.8 + DOWN * 1.5)

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
        colorbar_group.set_z_index(10) # Ensure on top

        # Add equation at the very beginning (it's already created above)
        self.add(equation_finite)

        self.play(FadeIn(input_image, run_time=2.0))
        self.wait(1.0)
        self.play(FadeIn(colorbar_group, run_time=0.8))
        self.play(Create(axes), Write(axes_labels))
        self.play(FadeIn(kernel_group, run_time=1.2))
        # Animate kernel axes appearing with the kernel
        self.play(Create(kernel_axes_group))
        self.play(
            FadeIn(global_vector, run_time=0.8),
            FadeIn(offset_vector, run_time=0.8),
        )

        # Define wiggle positions relative to kernel_start (1.2, 1.2)
        # Pure cardinal directions - only change one coordinate at a time
        up_pos = axes.c2p(1.2, 1.6)      # 0.4 units up (x=1.2, y=1.6)
        down_pos = axes.c2p(1.2, 0.8)    # 0.4 units down (x=1.2, y=0.8)
        right_pos = axes.c2p(1.6, 1.2)   # 0.4 units right (x=1.6, y=1.2)
        left_pos = axes.c2p(0.8, 1.2)    # 0.4 units left (x=0.8, y=1.2)

        # Up-down wiggle (up, down, up, down)
        wiggle_positions_ud = [up_pos, down_pos, up_pos, down_pos]
        for pos in wiggle_positions_ud:
            self.play(
                kernel_group.animate.move_to(pos),
                run_time=0.35,
                rate_func=rate_functions.ease_in_out_sine,
            )

        # Return to start position before right-left wiggle
        self.play(
            kernel_group.animate.move_to(kernel_start),
            run_time=0.35,
            rate_func=rate_functions.ease_in_out_sine,
        )

        # Right-left wiggle (right, left, right, left)
        wiggle_positions_rl = [right_pos, left_pos, right_pos, left_pos]
        for pos in wiggle_positions_rl:
            self.play(
                kernel_group.animate.move_to(pos),
                run_time=0.35,
                rate_func=rate_functions.ease_in_out_sine,
            )
        wiggle_angles = [-0.4, 0.35, -0.25, 0.2, 0.0]
        for angle in wiggle_angles:
            self.play(
                angle_tracker.animate.set_value(angle),
                run_time=0.35,
                rate_func=rate_functions.ease_in_out_sine,
            )
        self.wait(1.0)

        # Final animation: Expand kernel box to infinity and morph equation limits
        # Kernel box expands infinitely (representing infinite kernel support)
        # First, fix the kernel axes in place (don't let them move with the expanding box)
        kernel_axes_group.suspend_updating()  # Stop the always_redraw updater

        self.play(
            kernel_box.animate.scale(50),  # Scale up dramatically
            Transform(equation_finite, equation_infinite),  # Morph limits to infinity
            run_time=6.0,
            rate_func=smooth
        )

        self.wait(2)

    # ------------------------------------------------------------------
    # Rich, high-structure background field in bright cyan/orange
    # (deterministic, then multiplied by -1 to swap colours)
    # ------------------------------------------------------------------
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

        neg_rgb = np.array(ManimColor(IMAGE_NEG_HEX).to_rgb())  # cyan
        pos_rgb = np.array(ManimColor(IMAGE_POS_HEX).to_rgb())  # orange
        zero_rgb = np.array([0.0, 0.0, 0.0])                # black

        v = np.clip(v, -1.0, 1.0)
        rgba = np.zeros((resolution, resolution, 4), dtype=float)

        pos_mask = v > 0
        neg_mask = v < 0

        if np.any(pos_mask):
            t = v[pos_mask]
            rgba[pos_mask, :3] = (1 - t)[:, None] * zero_rgb + t[:, None] * pos_rgb

        if np.any(neg_mask):
            t = -v[neg_mask]
            rgba[neg_mask, :3] = (1 - t)[:, None] * zero_rgb + t[:, None] * neg_rgb

        gamma = 0.7
        alpha = 0.2 + 0.8 * np.power(np.abs(v), gamma)
        rgba[..., 3] = np.clip(alpha, 0.0, 1.0)

        input_image = ImageMobject(np.uint8(np.flipud(rgba) * 255))
        return input_image

    # ------------------------------------------------------------------
    # Kernel in bright cyan/orange with smooth fade,
    # square support aligned with the red kernel box
    # ------------------------------------------------------------------
    def create_kernel_image(self, resolution=220, span=3.0, cutoff=None):
        np.random.seed(10)

        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)
        field = np.zeros_like(X)

        num_blobs = 50
        for _ in range(num_blobs):
            amplitude = np.random.uniform(-1.0, 1.0)  # allow ± blobs
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

        # Optional gentle radial damping for aesthetics (doesn't define support)
        radial_mask = np.exp(-((X**2 + Y**2) / (2 * (span / 1.5) ** 2)))
        field *= radial_mask

        # Center and linear stretch to [-1, 1]
        field -= field.mean()
        max_abs = np.max(np.abs(field)) or 1.0
        v = field / max_abs
        v = np.clip(v, -1.0, 1.0)

        # Contrast hack: push mid-values toward extremes
        contrast_gamma = 0.6  # < 1 brightens mid-values
        v = np.sign(v) * np.power(np.abs(v), contrast_gamma)

        # ----- define square support matching the red kernel box -----
        # kernel_box.width = kernel_image.width * 0.4
        # → make support square be 40% of texture side length.
        h, w = v.shape
        cy, cx = h // 2, w // 2
        support_ratio = 0.4
        half_side_y = int((h * support_ratio) / 2)
        half_side_x = int((w * support_ratio) / 2)

        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        support_mask = (
            (np.abs(yy - cy) <= half_side_y) &
            (np.abs(xx - cx) <= half_side_x)
        )

        # Soften edges within the support box to avoid hard cuts
        # Normalized distance from center of support box (0 to 1 at edge)
        # Use max(dx, dy) for square shape to match the box
        dist_x = np.abs(xx - cx) / half_side_x
        dist_y = np.abs(yy - cy) / half_side_y
        max_dist = np.maximum(dist_x, dist_y)
        
        # Fade out smoothly in the last 40% of the support region
        # 1.0 when dist < 0.6, linearly drops to 0.0 at dist = 1.0
        edge_fade = np.clip((1.0 - max_dist) / 0.4, 0.0, 1.0)
        
        # Apply fade to the normalized field values
        v *= edge_fade

        neg_rgb = np.array(ManimColor(IMAGE_NEG_HEX).to_rgb())  # cyan
        pos_rgb = np.array(ManimColor(IMAGE_POS_HEX).to_rgb())  # orange
        zero_rgb = np.array([0.0, 0.0, 0.0])

        rgba = np.zeros((h, w, 4), dtype=float)

        pos_mask = (v > 0) & support_mask
        neg_mask = (v < 0) & support_mask

        if np.any(pos_mask):
            t = v[pos_mask]
            rgba[pos_mask, :3] = (1 - t)[:, None] * zero_rgb + t[:, None] * pos_rgb

        if np.any(neg_mask):
            t = -v[neg_mask]
            rgba[neg_mask, :3] = (1 - t)[:, None] * zero_rgb + t[:, None] * neg_rgb

        # Alpha: smooth fade INSIDE the support, zero outside
        alpha_gamma = 0.7
        alpha = np.zeros_like(v)
        alpha[support_mask] = np.power(np.abs(v[support_mask]), alpha_gamma)

        # kill tiny alpha so edges are clean
        alpha[alpha < 0.02] = 0.0
        rgba[..., 3] = np.clip(alpha, 0.0, 1.0)

        kernel_image = ImageMobject(np.uint8(np.flipud(rgba) * 255))
        return kernel_image
