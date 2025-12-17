from manim import *
from manim.utils.color import ManimColor
import numpy as np
import sys
from pathlib import Path

BLUE = ManimColor("#0066F5")
IMAGE_NEG_HEX = "#00D5FF"
IMAGE_POS_HEX = "#F26D00"
IMAGE_NEG_COLOR = ManimColor(IMAGE_NEG_HEX)
IMAGE_POS_COLOR = ManimColor(IMAGE_POS_HEX)
IMAGE_NEG_RGB = np.array(IMAGE_NEG_COLOR.to_rgb())
IMAGE_POS_RGB = np.array(IMAGE_POS_COLOR.to_rgb())
ZERO_RGB = np.array([0.0, 0.0, 0.0])

sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene


class PolarCoordsIntroduction(RightRegionScene):
    INITIAL_DELAY = 0.0

    def construct(self):
        self.camera.background_color = BLACK

        # --- Setup Axes ---
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
        
        # --- Kernel Setup ---
        kernel_span = 3.0
        final_pos = axes.c2p(1.6, 1.6)

        # Static kernel image
        kernel_image = self.create_kernel_image(resolution=240, span=kernel_span, shape="circle")
        kernel_image.set_width(axes.width * 0.95)
        kernel_image.move_to(final_pos)
        kernel_image.set_z_index(0)

        # --- Kernel Axes (sigma, tau) ---
        # Centered on the kernel
        k_axes = Axes(
            x_range=[0, 1.5, 1],
            y_range=[0, 1.5, 1],
            x_length=kernel_image.width * 0.4 * 0.6,
            y_length=kernel_image.height * 0.4 * 0.6,
            axis_config={
                "color": BLUE,
                "stroke_width": 2,
                "include_tip": False,
                "include_ticks": False,
            },
        )
        k_axes.move_to(kernel_image.get_center(), aligned_edge=DL)
        
        sigma_tick = Line(UP * 0.08, DOWN * 0.08, color=BLUE, stroke_width=2).move_to(k_axes.x_axis.get_end())
        tau_tick = Line(LEFT * 0.08, RIGHT * 0.08, color=BLUE, stroke_width=2).move_to(k_axes.y_axis.get_end())

        sigma_label = MathTex(r"x'", color=BLUE).scale(0.6)
        sigma_label.next_to(sigma_tick, RIGHT, buff=0.05)
        
        tau_label = MathTex(r"y'", color=BLUE).scale(0.6)
        tau_label.next_to(tau_tick, UP, buff=0.05)
        
        kernel_axes_group = VGroup(k_axes, sigma_tick, tau_tick, sigma_label, tau_label)
        kernel_axes_group.set_z_index(1)

        # --- Vector p' ---
        # Using trackers for animation
        # Initial values based on offset_vector [1.0, 0.6, 0]
        initial_offset = np.array([1.0, 0.6, 0.0])
        initial_radius = np.linalg.norm(initial_offset) # approx 1.166
        initial_angle = np.arctan2(initial_offset[1], initial_offset[0]) # approx 0.54 rad

        self.angle_tracker = ValueTracker(initial_angle)
        self.radius_tracker = ValueTracker(initial_radius)
        self.label_shift_tracker = ValueTracker(0.0)
        
        def get_p_vector_group():
            center = kernel_image.get_center()
            angle = self.angle_tracker.get_value()
            radius = self.radius_tracker.get_value()
            
            offset = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0
            ])
            
            target = center + offset
            
            # Vector arrow
            arrow = Arrow(
                center,
                target,
                buff=0,
                color=BLUE,
                stroke_width=2.5,
                tip_length=0.1,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)
            
            # Label p'
            lbl_p = MathTex(r"\mathbf{p}'", color=BLUE).scale(0.6)
            lbl_p.next_to(arrow.get_end(), UP, buff=0.1)
            lbl_p.shift(RIGHT * 0.15 * self.label_shift_tracker.get_value())
            
            # Angle arc (theta)
            # Reference line (sigma axis direction)
            line_sigma = Line(center, center + RIGHT * 0.5)
            line_vec = Line(center, target)
            
            angle_arc = Angle(
                line_sigma, line_vec,
                radius=0.4,
                other_angle=False,
                color=BLUE,
                stroke_width=1.5
            )
            
            theta_label = MathTex(r"\theta'", color=BLUE).scale(0.5)
            # Position theta label
            # Ideally at the midpoint of the arc, slightly outward
            mid_arc_angle = angle / 2
            arc_label_radius = 0.6
            theta_pos = center + np.array([
                arc_label_radius * np.cos(mid_arc_angle),
                arc_label_radius * np.sin(mid_arc_angle),
                0
            ])
            theta_label.move_to(theta_pos)
            
            # Radius label 'r'
            # Midpoint of vector, slightly offset
            midpoint = center + offset * 0.5
            # Perpendicular direction
            perp = np.array([-offset[1], offset[0], 0])
            if np.linalg.norm(perp) > 1e-6:
                perp = perp / np.linalg.norm(perp)
            
            r_label = MathTex("r'", color=BLUE).scale(0.5)
            r_label.move_to(midpoint + perp * 0.15)
            
            return VGroup(arrow, lbl_p, angle_arc, theta_label, r_label)
        
        p_vector_group = always_redraw(get_p_vector_group)

        # --- Setup Scene ---
        self.add(kernel_image)
        self.add(kernel_axes_group)
        self.add(p_vector_group)
        
        # Initial wait
        self.wait(1)
        
        # --- Wiggle Animations ---
        
        # 1. Angular wiggle: +theta, -theta, +theta, -theta
        base_theta = initial_angle
        delta_theta = 0.3
        
        self.play(
            self.angle_tracker.animate.set_value(base_theta + delta_theta),
            run_time=0.5, rate_func=smooth
        )
        self.play(
            self.angle_tracker.animate.set_value(base_theta - delta_theta),
            run_time=1.0, rate_func=smooth
        )
        self.play(
            self.angle_tracker.animate.set_value(base_theta + delta_theta),
            run_time=1.0, rate_func=smooth
        )
        self.play(
            self.angle_tracker.animate.set_value(base_theta - delta_theta),
            run_time=1.0, rate_func=smooth
        )
        self.play(
            self.angle_tracker.animate.set_value(base_theta),
            run_time=0.5, rate_func=smooth
        )
        
        self.wait(0.5)
        
        # 2. Radial wiggle: +r, -r, +r, -r
        base_r = initial_radius
        delta_r = 0.2
        
        self.play(
            self.radius_tracker.animate.set_value(base_r - delta_r),
            run_time=0.5, rate_func=smooth
        )
        self.play(
            self.radius_tracker.animate.set_value(base_r + delta_r),
            run_time=1.0, rate_func=smooth
        )
        self.play(
            self.radius_tracker.animate.set_value(base_r -1.5 * delta_r),
            run_time=1.0, rate_func=smooth
        )

        final_radius = self.radius_tracker.get_value()
        highlight_circle = Circle(
            radius=final_radius,
            color=WHITE,
            stroke_width=2.5,
        )
        highlight_circle.move_to(kernel_image.get_center())
        highlight_circle.set_z_index(1)

        self.play(
            FadeIn(highlight_circle),
            self.label_shift_tracker.animate.set_value(1.0),
            run_time=0.8,
        )
        #self.play(
        #    self.radius_tracker.animate.set_value(base_r - delta_r),
        #    run_time=1.0, rate_func=smooth
        #)
        #self.play(
        #    self.radius_tracker.animate.set_value(base_r),
        #    run_time=0.5, rate_func=smooth
        #)
        
        self.wait(2)


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

