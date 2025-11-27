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


class ConvolutionPullback(MovingCameraScene):
    def setup(self):
        super().setup()
        # RightRegionScene logic adapted for MovingCameraScene
        RIGHT_MARGIN_PX = 180
        px_to_coord = self.camera.frame_width / self.camera.pixel_width
        shift = (RIGHT_MARGIN_PX / 2) * px_to_coord
        self.camera.frame.shift(RIGHT * shift)

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
        axes_shift = LEFT * 1.5 + DOWN * 1.0
        axes.shift(axes_shift)
        axes_labels = axes.get_axis_labels(MathTex("s"), MathTex("t"))

        # --- background image ---
        # High resolution, rotated analytically frame-by-frame for perfect smoothness
        # using always_redraw (re-evaluating function) instead of image rotation
        theta_tracker = ValueTracker(0.0)
        
        def get_input_image():
            return self.create_input_image(
                resolution=1500, 
                span=6.0,
                rotation_angle=theta_tracker.get_value()
            ).set_width(18.0).set_z_index(-2).move_to(axes.c2p(0, 0))
            
        input_image = always_redraw(get_input_image)

        # --- Coordinate System Rotation Tracker ---
        coord_alpha_tracker = ValueTracker(0.0)

        # --- kernel setup ---
        kernel_span = 3.0
        
        # Start with CIRCLE kernel (final state of previous animation)
        # Using analytical rotation + always_redraw for smooth motion
        initial_kernel_center = axes.c2p(1.6, 1.6)
        target_kernel_width = axes.width * 0.95
        
        def get_kernel_image():
            angle = coord_alpha_tracker.get_value()
            
            # 1. Create kernel image with internal rotation (to align with local axes)
            # Resolution bumped for quality
            img = self.create_kernel_image(
                resolution=400, 
                span=kernel_span, 
                shape="circle",
                rotation_angle=angle
            )
            img.set_width(target_kernel_width)
            
            # 2. Calculate position: The kernel is "attached" to the rotating coordinate system.
            # Its position (1.6, 1.6) in the coordinate system rotates about the global origin (0,0).
            # If coord system rotates by `angle` (counter-clockwise relative to initial),
            # the point (1.6, 1.6) rotates by `angle`.
            
            # Get vector from origin to initial center
            origin = axes.c2p(0, 0)
            vec = initial_kernel_center - origin
            
            # Rotate vector by angle
            c, s = np.cos(angle), np.sin(angle)
            rx = vec[0] * c - vec[1] * s
            ry = vec[0] * s + vec[1] * c
            rotated_vec = np.array([rx, ry, 0])
            
            new_pos = origin + rotated_vec
            
            img.move_to(new_pos)
            img.set_z_index(-1)
            return img

        kernel_image = always_redraw(get_kernel_image)
        
        # Visible kernel outline (circle)
        kernel_outline = Circle(
            radius=(target_kernel_width * 0.4) / 2,
            color=SCARLET,
            stroke_width=2.0
        )
        # We need to position the outline manually initially for get_kernel_axes, 
        # but it will be updated by updater later.
        kernel_outline.move_to(initial_kernel_center)
        kernel_outline.set_z_index(-0.9)
        kernel_box = kernel_outline # For compatibility with helper functions if needed

        # --- Kernel Axes (sigma, tau) moving with kernel ---
        def get_kernel_axes():
            center = kernel_outline.get_center()
            # Determine width/height based on current outline shape
            w = kernel_outline.width
            h = kernel_outline.height
            
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
            
            g = VGroup(k_axes, sigma_tick, tau_tick, sigma_label, tau_label)
            
            # Rotate local axes to match global coordinate rotation
            # Rotate about the kernel center (which is the center of this local system approx)
            # Actually k_axes.move_to(center, aligned_edge=DL) puts origin at DL.
            # We want to rotate the whole group around the mobject center? 
            # No, the "up" direction of the axes should tilt.
            # So rotating around the group's center or the axes origin?
            # Usually we want the text to stay upright? 
            # User said "rotate along with the (s,t) coordinate system".
            # If (s,t) rotates, the whole paper rotates. Text usually rotates with the paper.
            # So rotate the whole group.
            g.rotate(coord_alpha_tracker.get_value(), about_point=center)
            
            return g
            
        kernel_axes_group = always_redraw(get_kernel_axes)
        kernel_axes_group.set_z_index(0)

        # --- Global Vector (s,t) -> p ---
        origin_point = axes.c2p(0, 0)
        vector_color = GREY_A
        
        # Trackers (Initialized to final values of previous scene)
        global_label_alpha = ValueTracker(1.0)
        offset_label_alpha = ValueTracker(1.0)
        vector_length_tracker = ValueTracker(0.8) # Shrunk value
        angle_tracker = ValueTracker(0.0)

        def global_vector_group():
            target = kernel_outline.get_center()
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
        # Base offset logic from previous scene
        # base_offset = np.array([kernel_box.width * -0.48, kernel_box.height * 0.25, 0.0]) * 0.8
        # Since kernel_box is now kernel_outline (Circle), width is same as original box width.
        base_offset = np.array(
            [kernel_outline.width * -0.48, kernel_outline.height * 0.25, 0.0]
        ) * 0.8

        def offset_vector_group():
            base = kernel_outline.get_center()
            angle = angle_tracker.get_value()
            coord_angle = coord_alpha_tracker.get_value()
            total_angle = angle + coord_angle
            scale = vector_length_tracker.get_value()

            cos_a, sin_a = np.cos(total_angle), np.sin(total_angle)
            
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
        
        # Position relative to frame for fixed_in_frame behavior
        # Original logic: frame_ul + RIGHT * 0.8 + DOWN * 1.5
        # We can simulate this by placing it initially relative to the camera frame
        # and then adding it as a fixed mobject.
        # However, to_corner(UL) works relative to the frame width/height.
        
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
        
        # Position in top-left corner with offset
        colorbar_group.to_corner(UL, buff=0.5).shift(DOWN * 1.0)
        
        # --- Add everything to scene ---
        self.add(input_image)
        self.add(kernel_image)
        self.add(kernel_outline)
        
        # Group axes and labels for easier rotation later
        axes_group = VGroup(axes, axes_labels)
        self.add(axes_group)
        
        self.add(kernel_axes_group)
        self.add(global_vector)
        self.add(offset_vector)
        
        # Add colorbar as fixed in frame so it doesn't rotate with the camera/scene
        # RightRegionScene inherits from Scene, so we don't have add_fixed_in_frame_mobjects.
        # However, Scene has `add_fixed_in_frame_mobjects` ONLY if it's a ThreeDScene or we treat it specially?
        # Standard Scene (v0.18.1) does NOT have `add_fixed_in_frame_mobjects`.
        # However, since we switched to MovingCameraScene, we can add it to the camera frame!
        # But `self.camera.frame` is a mobject. We can attach the colorbar to it.
        # Or better: `self.add_fixed_in_frame_mobjects(colorbar_group)` DOES exist in MovingCameraScene?
        # Actually, MovingCameraScene inherits from Scene.
        # Let's just add it to the scene and manually correct rotation if needed.
        # Wait, if we are NOT rotating the camera frame in Stage 4 anymore, but resetting coordinate rotation,
        # then the "camera frame" stays fixed. So simple `self.add()` works perfectly!
        # The colorbar will stay upright because the camera doesn't rotate.
        self.add(colorbar_group)
        
        self.wait(1)

        # --- Stage 1: Rotate background image (Active Rotation) ---
        # Rotate 20 degrees counter-clockwise about the global origin (axes origin)
        rotation_angle = 20 * DEGREES
        
        self.play(
            theta_tracker.animate.set_value(rotation_angle),
            run_time=2.0,
            rate_func=smooth
        )
        
        self.wait(1)
        
        # --- Stage 2: Rotate back to start ---
        self.play(
            theta_tracker.animate.set_value(0.0),
            run_time=1.5,
            rate_func=smooth
        )
        
        self.wait(1)
        
        # --- Stage 3: Rotate Coordinate System (Passive Transformation) ---
        # Rotate axes, kernel, and kernel axes by -alpha (clockwise) around global origin.
        # Background stays fixed (theta=0).
        # We need to rotate: axes_group, kernel_outline.
        # kernel_image, kernel_axes_group, global_vector, offset_vector update automatically via always_redraw.
        
        # Define objects to rotate
        objects_to_rotate = [axes_group, kernel_outline]
        for mob in objects_to_rotate:
            mob.save_state()
            
        def rotate_coords_updater(mob):
            mob.restore()
            mob.rotate(
                coord_alpha_tracker.get_value(),
                about_point=axes.c2p(0, 0)
            )

        # Attach updaters
        for mob in objects_to_rotate:
            mob.add_updater(rotate_coords_updater)

        self.play(
            coord_alpha_tracker.animate.set_value(-rotation_angle),
            run_time=2.0,
            rate_func=smooth
        )
        
        self.wait(1)
        
        # --- Stage 4: Reset Coordinate System ---
        # Rotate coordinates back to 0.
        # "Just change the final stage to just rotating the coordinate system back to where it was"
        
        self.play(
            coord_alpha_tracker.animate.set_value(0.0),
            run_time=2.0,
            rate_func=smooth
        )
        
        # Clean up updaters
        for mob in objects_to_rotate:
            mob.remove_updater(rotate_coords_updater)
        
        self.wait(2)


    def create_input_image(self, resolution=320, span=6.0, rotation_angle=0.0,
                           num_large=8, num_small=18):
        rng = np.random.default_rng(2025)

        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)

        # Apply rotation to coordinate grid if needed
        # Rotate coordinates by -angle to simulate rotating the function by +angle
        if rotation_angle != 0:
            c, s = np.cos(-rotation_angle), np.sin(-rotation_angle)
            X_rot = c * X - s * Y
            Y_rot = s * X + c * Y
            X, Y = X_rot, Y_rot

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

    def create_kernel_image(self, resolution=220, span=3.0, cutoff=None, shape="square", rotation_angle=0.0):
        np.random.seed(10)

        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)
        
        # Apply rotation
        if rotation_angle != 0:
            c, s = np.cos(-rotation_angle), np.sin(-rotation_angle)
            X_rot = c * X - s * Y
            Y_rot = s * X + c * Y
            X, Y = X_rot, Y_rot
            
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

