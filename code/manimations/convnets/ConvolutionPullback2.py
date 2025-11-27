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

class ConvolutionPullback2(MovingCameraScene):
    def setup(self):
        super().setup()
        # RightRegionScene logic adapted for MovingCameraScene
        RIGHT_MARGIN_PX = 180
        px_to_coord = self.camera.frame_width / self.camera.pixel_width
        shift = (RIGHT_MARGIN_PX / 2) * px_to_coord
        self.camera.frame.shift(RIGHT * shift)

    def construct(self):
        self.camera.background_color = BLACK

        # --- Global Axis ---
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
        
        # --- Coordinate System Rotation Tracker ---
        coord_alpha_tracker = ValueTracker(0.0)

        # --- Kernel Setup ---
        kernel_span = 3.0
        # Changing initial position to adjust angle of p.
        # Originally (1.6, 1.6), which is 45 degrees.
        # Target angle is 20 degrees with s-axis.
        # Radius was sqrt(1.6^2 + 1.6^2) approx 2.26
        # New coordinates: (r * cos(20deg), r * sin(20deg))
        radius = np.sqrt(1.6**2 + 1.6**2)
        # Actually, user asked to increase length of p by 180% (1.8x).
        # So new radius = 2.26 * 1.8 = 4.07
        # Wait, "180% as long" usually means 1.8 * original_length.
        # "Increase the length ... so it's 180% as long" -> factor 1.8.
        
        new_radius = radius * 1.4
        target_angle = 15 * DEGREES
        
        new_x = new_radius * np.cos(target_angle)
        new_y = new_radius * np.sin(target_angle)
        
        initial_kernel_center = axes.c2p(new_x, new_y)
        target_kernel_width = axes.width * 0.95

        def get_kernel_image():
            angle = coord_alpha_tracker.get_value()
            
            # 1. Create kernel image with internal rotation (to align with local axes)
            img = self.create_kernel_image(
                resolution=400, 
                span=kernel_span, 
                shape="circle",
                rotation_angle=angle
            )
            img.set_width(target_kernel_width)
            
            # 2. Calculate position
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
            img.set_opacity(0) # Hide kernel image but keep it
            return img

        kernel_image = always_redraw(get_kernel_image)

        # --- Background Image (s.png) ---
        project_root = Path(__file__).parent.parent.parent.parent
        s_png_path = project_root / "notes" / "s.png"
        
        if not s_png_path.exists():
             print(f"Warning: {s_png_path} does not exist.")

        s_image = ImageMobject(str(s_png_path))
        s_image.set_width(0.8)
        
        def get_s_image_pos():
             # Same logic as kernel position
             angle = coord_alpha_tracker.get_value()
             origin = axes.c2p(0, 0)
             vec = initial_kernel_center - origin
             
             c, s = np.cos(angle), np.sin(angle)
             rx = vec[0] * c - vec[1] * s
             ry = vec[0] * s + vec[1] * c
             rotated_vec = np.array([rx, ry, 0])
             
             new_pos = origin + rotated_vec
             
             s_img = s_image.copy()
             s_img.move_to(new_pos)
             s_img.set_z_index(-3) # Behind everything
             return s_img

        s_image_dynamic = always_redraw(get_s_image_pos)


        # --- Global Vector (s,t) -> p ---
        origin_point = axes.c2p(0, 0)
        vector_color = GREY_A
        
        # Trackers
        global_label_alpha = ValueTracker(1.0)
        offset_label_alpha = ValueTracker(1.0)
        vector_length_tracker = ValueTracker(1.5) 
        angle_tracker = ValueTracker(0.0)

        def global_vector_group():
            # final_target depends on the rotated kernel center
            angle = coord_alpha_tracker.get_value()
            origin = axes.c2p(0, 0)
            vec = initial_kernel_center - origin
            
            c, s = np.cos(angle), np.sin(angle)
            rx = vec[0] * c - vec[1] * s
            ry = vec[0] * s + vec[1] * c
            rotated_vec = np.array([rx, ry, 0])
            final_target = origin + rotated_vec
            
            arrow = Arrow(
                origin_point, 
                final_target,
                buff=0,
                color=vector_color,
                stroke_width=2.5,
                tip_length=0.08,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)
            
            alpha = global_label_alpha.get_value()
            
            lbl_st = MathTex(r"(s,t)", color=vector_color).scale(0.5)
            lbl_p = MathTex(r"\mathbf{p}", color=vector_color).scale(0.5)
            
            pos = final_target + UP * 0.2
            lbl_st.move_to(pos).set_opacity(1 - alpha)
            lbl_p.move_to(pos).set_opacity(alpha)
            
            return VGroup(arrow, lbl_st, lbl_p)

        global_vector = always_redraw(global_vector_group)

        # --- Offset vector (sigma, tau) -> p' ---
        kw = target_kernel_width * 0.4
        kh = target_kernel_width * 0.4
        
        # decrease the length of vector p' so it's half its current length
        # Previous length factor was implicitly around 0.45 * height.
        # Now multiply by 0.5
        base_offset = np.array(
            [0.0, kh * 0.45 * 0.5, 0.0]
        )

        def offset_vector_group():
            # Base position is simply the rotated kernel center
            angle = coord_alpha_tracker.get_value()
            origin = axes.c2p(0, 0)
            vec = initial_kernel_center - origin
            c, s = np.cos(angle), np.sin(angle)
            rx = vec[0] * c - vec[1] * s
            ry = vec[0] * s + vec[1] * c
            base = origin + np.array([rx, ry, 0])

            angle_val = angle_tracker.get_value()
            coord_angle = coord_alpha_tracker.get_value()
            total_angle = angle_val + coord_angle
            scale = vector_length_tracker.get_value()

            cos_a, sin_a = np.cos(total_angle), np.sin(total_angle)
            
            rotated_offset = np.array([
                base_offset[0] * cos_a - base_offset[1] * sin_a,
                base_offset[0] * sin_a + base_offset[1] * cos_a,
                base_offset[2]
            ])
            
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

        # Add everything to scene
        self.add(axes, axes_labels)
        self.add(kernel_image)
        self.add(s_image_dynamic)
        self.add(global_vector)
        self.add(offset_vector)
        
        self.wait(2)

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
