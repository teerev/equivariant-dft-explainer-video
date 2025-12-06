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

GLOBAL_SCALE = 1.05
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
        # Range modified to cut off negative ranges
        # Keep small negative amount: [-0.5, 3, 1]
        axes = Axes(
            x_range=[-0.5, 3, 1],
            y_range=[-0.5, 3, 1],
            x_length=3.5 * GLOBAL_SCALE, 
            y_length=3.5 * GLOBAL_SCALE, 
            axis_config={
                "color": WHITE, 
                "stroke_width": 2 * GLOBAL_SCALE,
                "include_tip": True, 
                "tip_length": 0.15 * GLOBAL_SCALE,
                "tip_width": 0.1 * GLOBAL_SCALE,
                "tip_height": 0.1 * GLOBAL_SCALE,
                "include_ticks": False, 
            },
        )
        axes_shift = LEFT * 4.5 + UP * 1.5
        axes.shift(axes_shift)
        axes_labels = axes.get_axis_labels(
            MathTex("x", color=WHITE).scale(GLOBAL_SCALE), 
            MathTex("y", color=WHITE).scale(GLOBAL_SCALE)
        ) 
        
        # --- Trackers ---
        scale_tracker = ValueTracker(GLOBAL_SCALE) # Global scale factor for everything
        coord_alpha_tracker = ValueTracker(0.0)
        vector_rotation_tracker = ValueTracker(0.0)
        axes_rotation_tracker = ValueTracker(0.0)
        global_label_alpha = ValueTracker(1.0)
        offset_label_alpha = ValueTracker(1.0)
        vector_length_tracker = ValueTracker(1.5) 
        angle_tracker = ValueTracker(0.0)
        offset_label_opacity_tracker = ValueTracker(0.0)
        separation_tracker = ValueTracker(0.0) # For pullback split
        rotated_axis_label_opacity_tracker = ValueTracker(0.0)
        right_panel_rotation_tracker = ValueTracker(0.0)
        
        # --- Positioning Configuration ---
        separation_vector = RIGHT * 4.5
        
        def get_right_origin():
            # Use value from separation_tracker (0 to 1) to interpolate
            # This allows controlling both distance and direction with one vector
            # separation_tracker will now go from 0 to 1
            # Note: separation vector itself is not scaled, as it defines relative placement
            # But the axes c2p will naturally handle scaling of coordinates
            return axes.c2p(0,0) + separation_vector * separation_tracker.get_value()

        # --- Kernel Setup ---
        kernel_span = 3.0
        radius = np.sqrt(1.6**2 + 1.6**2)
        new_radius = radius * 1.4
        target_angle = 15 * DEGREES
        new_x = new_radius * np.cos(target_angle)
        new_y = new_radius * np.sin(target_angle)
        
        # c2p will automatically handle the scaling since axes are scaled
        initial_kernel_center = axes.c2p(new_x, new_y)
        target_kernel_width = 5.7 * GLOBAL_SCALE

        def get_kernel_image():
            angle = coord_alpha_tracker.get_value()
            img = self.create_kernel_image(
                resolution=400, 
                span=kernel_span, 
                shape="circle",
                rotation_angle=angle
            )
            img.set_width(target_kernel_width)
            
            origin = axes.c2p(0, 0)
            vec = initial_kernel_center - origin
            c, s = np.cos(angle), np.sin(angle)
            rx = vec[0] * c - vec[1] * s
            ry = vec[0] * s + vec[1] * c
            rotated_vec = np.array([rx, ry, 0])
            new_pos = origin + rotated_vec
            
            img.move_to(new_pos)
            img.set_z_index(-1)
            img.set_opacity(0) 
            return img

        kernel_image = always_redraw(get_kernel_image)

        # --- Background Image (s.png) ---
        project_root = Path(__file__).parent.parent.parent.parent
        s_png_path = project_root / "notes" / "s.png"
        if not s_png_path.exists():
             print(f"Warning: {s_png_path} does not exist.")

        s_image = ImageMobject(str(s_png_path))
        s_image.set_width(0.8 * GLOBAL_SCALE)
        
        # Fixed S image (Right Panel)
        def s_image_static_group():
            # Moves with separation
            # Initial position relative to axes origin
            origin = axes.c2p(0,0)
            rel_pos = initial_kernel_center - origin
            
            right_origin = get_right_origin()
            
            # Apply panel rotation around right_origin
            panel_angle = right_panel_rotation_tracker.get_value()
            
            c, s = np.cos(panel_angle), np.sin(panel_angle)
            rx = rel_pos[0] * c - rel_pos[1] * s
            ry = rel_pos[0] * s + rel_pos[1] * c
            rotated_rel_pos = np.array([rx, ry, 0])
            
            pos = right_origin + rotated_rel_pos
            
            img = s_image.copy()
            img.move_to(pos)
            img.rotate(panel_angle)
            img.set_z_index(-4)
            return img

        s_image_static = always_redraw(s_image_static_group)
        
        # Dynamic S image (Left Panel)
        def get_s_image_pos():
             angle = vector_rotation_tracker.get_value()
             origin = axes.c2p(0, 0)
             vec = initial_kernel_center - origin
             c, s = np.cos(angle), np.sin(angle)
             rx = vec[0] * c - vec[1] * s
             ry = vec[0] * s + vec[1] * c
             rotated_vec = np.array([rx, ry, 0])
             new_pos = origin + rotated_vec
             
             s_img = s_image.copy()
             s_img.move_to(new_pos)
             s_img.rotate(angle)
             s_img.set_z_index(-3) 
             return s_img

        s_image_dynamic = always_redraw(get_s_image_pos)


        # --- Global Vector p (Right Panel) ---
        vector_color = GREY_A
        
        def global_vector_group():
            # Right Panel Origin
            origin = get_right_origin()
            
            vec = initial_kernel_center - axes.c2p(0,0) # Fixed relative vector
            
            # Apply panel rotation
            panel_angle = right_panel_rotation_tracker.get_value()
            c, s = np.cos(panel_angle), np.sin(panel_angle)
            rx = vec[0] * c - vec[1] * s
            ry = vec[0] * s + vec[1] * c
            rotated_vec = np.array([rx, ry, 0])
            
            final_target = origin + rotated_vec
            
            arrow = Arrow(
                origin, 
                final_target,
                buff=0,
                color=vector_color,
                stroke_width=2.5 * GLOBAL_SCALE,
                tip_length=0.08 * GLOBAL_SCALE, 
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)
            
            alpha = global_label_alpha.get_value()
            
            lbl_st = MathTex(r"(x,y)", color=vector_color).scale(0.5 * GLOBAL_SCALE)
            lbl_p = MathTex(r"\mathbf{p}", color=vector_color).scale(0.5 * GLOBAL_SCALE * 1.5)
            
            pos = final_target + UP * 0.2 * GLOBAL_SCALE
            lbl_st.move_to(pos).set_opacity(1 - alpha)
            lbl_p.move_to(pos).set_opacity(alpha)
            
            return VGroup(arrow, lbl_st, lbl_p)

        global_vector = always_redraw(global_vector_group)

        # --- Offset vector p' (Right Panel) ---
        kw = target_kernel_width * 0.4
        kh = target_kernel_width * 0.4
        base_offset = np.array([0.0, kh * 0.45 * 0.5, 0.0])

        def offset_vector_group():
            # Relative to global_vector tip in Right Panel
            origin = get_right_origin()
            vec = initial_kernel_center - axes.c2p(0,0)
            
            # Apply panel rotation
            panel_angle = right_panel_rotation_tracker.get_value()
            c, s = np.cos(panel_angle), np.sin(panel_angle)
            
            # Rotated p
            rx = vec[0] * c - vec[1] * s
            ry = vec[0] * s + vec[1] * c
            rotated_vec = np.array([rx, ry, 0])
            
            base = origin + rotated_vec

            # Rotated offset vector (p')
            offset = base_offset * vector_length_tracker.get_value()
            rx_off = offset[0] * c - offset[1] * s
            ry_off = offset[0] * s + offset[1] * c
            rotated_offset = np.array([rx_off, ry_off, 0])
            
            arrow = Arrow(
                base,
                base + rotated_offset,
                buff=0,
                color=SCARLET,
                stroke_width=2.2 * GLOBAL_SCALE,
                tip_length=0.08 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(2)
            
            alpha = offset_label_alpha.get_value()
            
            lbl_st = MathTex(r"(x',y')", color=SCARLET).scale(0.6 * GLOBAL_SCALE)
            lbl_p = MathTex(r"\mathbf{p}'", color=SCARLET).scale(0.6 * GLOBAL_SCALE * 1.5)
            
            pos = base + rotated_offset + UP * 0.2 * GLOBAL_SCALE
            # Rotate label pos offset if needed? Usually UP is fine but maybe better aligned
            # Keep it simple: just place above tip
            
            lbl_st.move_to(pos).set_opacity(1 - alpha)
            lbl_p.move_to(pos).set_opacity(alpha)
            
            return VGroup(arrow, lbl_st, lbl_p)

        offset_vector = always_redraw(offset_vector_group)

        # --- Rotated Vector Q_alpha p (Left Panel) ---
        def rotated_vector_group():
            # Left Panel Origin = axes.c2p(0,0) (Fixed)
            angle = vector_rotation_tracker.get_value()
            origin = axes.c2p(0, 0)
            vec = initial_kernel_center - origin 
            
            c, s = np.cos(angle), np.sin(angle)
            rx = vec[0] * c - vec[1] * s
            ry = vec[0] * s + vec[1] * c
            rotated_vec = np.array([rx, ry, 0])
            
            target = origin + rotated_vec
            
            arrow = Arrow(
                origin, 
                target,
                buff=0,
                color=WHITE, 
                stroke_width=2.5 * GLOBAL_SCALE,
                tip_length=0.08 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(3) 
            
            return arrow

        rotated_vector = always_redraw(rotated_vector_group)

        # --- Rotated Offset Vector Q_alpha p' (Left Panel) ---
        def rotated_offset_vector_group():
            angle = vector_rotation_tracker.get_value()
            origin = axes.c2p(0, 0)
            vec = initial_kernel_center - origin 
            
            c, s = np.cos(angle), np.sin(angle)
            rx = vec[0] * c - vec[1] * s
            ry = vec[0] * s + vec[1] * c
            rotated_vec = np.array([rx, ry, 0])
            base = origin + rotated_vec 
            
            kh = 5.7 * 0.4 * GLOBAL_SCALE # scaled kernel height
            # Note: base_offset calculation above already uses target_kernel_width which we scaled
            # But here we hardcoded 5.7. Let's make sure everything is consistent.
            # Actually, let's just use the same scale factor here.
            # p_prime definition in rotated_offset_vector_group was:
            # p_prime = np.array([0.0, kh * 0.45 * 0.5, 0.0]) * vector_length_tracker.get_value()
            
            # Re-calculating properly:
            target_kw_scaled = 5.7 * GLOBAL_SCALE
            kh_scaled = target_kw_scaled * 0.4
            p_prime = np.array([0.0, kh_scaled * 0.45 * 0.5, 0.0]) * vector_length_tracker.get_value()
            
            rx_off = p_prime[0] * c - p_prime[1] * s
            ry_off = p_prime[0] * s + p_prime[1] * c
            rotated_offset = np.array([rx_off, ry_off, 0])
            
            arrow = Arrow(
                base,
                base + rotated_offset,
                buff=0,
                color=SCARLET, 
                stroke_width=2.2 * GLOBAL_SCALE,
                tip_length=0.08 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            ).set_z_index(3)
        
            lbl = MathTex(r"\mathbf{Q}_\alpha \mathbf{p}'", color=SCARLET).scale(0.6 * GLOBAL_SCALE * 1.5)
            lbl.next_to(base + rotated_offset, UP, buff=0.1 * GLOBAL_SCALE)
            lbl.set_opacity(offset_label_opacity_tracker.get_value())
            
            return VGroup(arrow, lbl)

        rotated_offset_vector = always_redraw(rotated_offset_vector_group)

        # --- Rotated Axes (Right Panel) ---
        def rotated_axes_group():
            angle = axes_rotation_tracker.get_value()
            sep = separation_tracker.get_value()
            
            # Center moves with separation
            center = get_right_origin()
            
            new_axes = Axes(
                x_range=[-0.5, 3, 1],
                y_range=[-0.5, 3, 1],
                x_length=3.5 * GLOBAL_SCALE,
                y_length=3.5 * GLOBAL_SCALE,
                axis_config={
                    "color": WHITE, 
                    "stroke_width": 2 * GLOBAL_SCALE,
                    "include_tip": True,
                    "tip_length": 0.15 * GLOBAL_SCALE,
                    "tip_width": 0.1 * GLOBAL_SCALE,
                    "tip_height": 0.1 * GLOBAL_SCALE,
                    "include_ticks": False,
                },
            )
            # Ensure origin is at center
            origin_offset = new_axes.c2p(0,0)
            new_axes.shift(center - origin_offset)
            
            l_s = MathTex(r"\mathbf{Q}_{-\alpha} x", color=WHITE).scale(GLOBAL_SCALE)
            l_t = MathTex(r"\mathbf{Q}_{-\alpha} y", color=WHITE).scale(GLOBAL_SCALE)
            l_s.next_to(new_axes.x_axis.get_end(), RIGHT, buff=0.1 * GLOBAL_SCALE)
            l_t.next_to(new_axes.y_axis.get_end(), UP, buff=0.1 * GLOBAL_SCALE)
            
            l_s.set_opacity(rotated_axis_label_opacity_tracker.get_value())
            l_t.set_opacity(rotated_axis_label_opacity_tracker.get_value())

            g = VGroup(new_axes, l_s, l_t)
            g.rotate(-angle + right_panel_rotation_tracker.get_value(), about_point=center)
            
            return g

        rotated_axes = always_redraw(rotated_axes_group)
        
        # --- Negative Angle Indicator (Right Panel) ---
        def neg_angle_group():
            angle = axes_rotation_tracker.get_value()
            sep = separation_tracker.get_value()

            # Hide if separation starts (or angle is very small)
            if abs(angle) < 0.01 or sep > 0.1:
                return VGroup()

            origin = get_right_origin()
            
            # Reference T axis (vertical)
            t_point = origin + UP * 1.2
            
            # Rotated T axis
            c, s = np.cos(-angle), np.sin(-angle)
            t_vec = UP * 1.2
            rx = t_vec[0] * c - t_vec[1] * s
            ry = t_vec[0] * s + t_vec[1] * c
            qt_point = origin + np.array([rx, ry, 0])
            
            line_t = Line(origin, t_point)
            line_qt = Line(origin, qt_point)
            
            arc = Angle(line_qt, line_t, radius=0.5 * GLOBAL_SCALE, color=WHITE)
            lbl = MathTex(r"-\alpha", color=WHITE).scale(0.6 * GLOBAL_SCALE)
            pos = Angle(line_qt, line_t, radius=0.8 * GLOBAL_SCALE).point_from_proportion(0.5)
            lbl.move_to(pos)
            
            # Only show if angle > 0
            if angle < 0.01:
                arc.set_opacity(0)
                lbl.set_opacity(0)
            
            return VGroup(arc, lbl)

        neg_angle_indicator = always_redraw(neg_angle_group)


        # --- Equation ---
        # (\mathbf{Q}_\alpha \cdot X)(p) \;:=\; X\bigl(\mathbf{Q}_{-\alpha}\,p\bigr)
        
        # Position relative to axes origin
        # axes origin is at y=1.5. We want equation around y=-1.5 to -2.0.
        # So offset should be around DOWN * 3.0 to 3.5
        eq_y_offset = DOWN * 2.8
        
        eq_lhs = MathTex(r"(\mathbf{Q}_\alpha \cdot V)(p)", color=WHITE).scale(GLOBAL_SCALE)
        eq_eq = MathTex(r"\;:=", color=WHITE).scale(GLOBAL_SCALE)
        eq_rhs = MathTex(r"V\bigl(\mathbf{Q}_{-\alpha}\,p\bigr)", color=WHITE).scale(GLOBAL_SCALE)
        
        # Calculate positions
        # Note: separation_vector is used directly here as it defines the final translation
        # Use visual centers for midpoint calculation
        left_center = axes.get_center()
        final_right_center = left_center + separation_vector 
        
        midpoint = (left_center + final_right_center) / 2
        
        # Control equation spacing here:
        eq_spacing = 2.0 * GLOBAL_SCALE # Distance from center for LHS and RHS
        
        # Use left_center as Y reference for offset since all are on same Y level relative to panels
        y_pos = (left_center + eq_y_offset)[1]
        
        eq_lhs.move_to(np.array([(midpoint - RIGHT * eq_spacing)[0], y_pos, 0]))
        eq_eq.move_to(np.array([midpoint[0], y_pos, 0]))
        eq_rhs.move_to(np.array([(midpoint + RIGHT * eq_spacing)[0], y_pos, 0]))


        # Add everything to scene
        self.add(axes, axes_labels)
        self.add(kernel_image)
        self.add(s_image_static) 
        self.add(s_image_dynamic)
        self.add(global_vector)
        self.add(offset_vector)
        self.add(rotated_vector)
        self.add(rotated_offset_vector) 
        self.add(rotated_axes) # Initially at sep=0, angle=0 -> on top of axes
        self.add(neg_angle_indicator)
        
        self.wait(2)
        
        # Animation 1: Rotation
        self.play(
            vector_rotation_tracker.animate.set_value(20 * DEGREES),
            run_time=2.0
        )
        
        # Animation 2: Labels appear
        # Left panel labels (Angle, Qp, Qp')
        # We need to recreate these as always_redraw or simple objects?
        # They are static relative to Left Panel (which doesn't move).
        
        angle_rot = vector_rotation_tracker.get_value()
        origin = axes.c2p(0, 0)
        vec_p = initial_kernel_center - origin
        c, s = np.cos(angle_rot), np.sin(angle_rot)
        rx = vec_p[0] * c - vec_p[1] * s
        ry = vec_p[0] * s + vec_p[1] * c
        qp_point = origin + np.array([rx, ry, 0])
        
        line_p = Line(origin, initial_kernel_center)
        line_q = Line(origin, qp_point)
        angle_arc = Angle(line_p, line_q, radius=0.6 * GLOBAL_SCALE, color=WHITE)
        angle_label = MathTex(r"\alpha", color=WHITE).scale(0.6 * GLOBAL_SCALE)
        angle_label.move_to(Angle(line_p, line_q, radius=0.9 * GLOBAL_SCALE).point_from_proportion(0.5))
        label_qp = MathTex(r"\mathbf{Q}_\alpha \mathbf{p}", color=WHITE).scale(0.6 * GLOBAL_SCALE * 1.5)
        label_qp.next_to(qp_point, UP, buff=0.1 * GLOBAL_SCALE)
        
        self.play(
            Create(angle_arc),
            Write(angle_label),
            Write(label_qp),
            Write(eq_lhs),
            offset_label_opacity_tracker.animate.set_value(1.0),
            run_time=1.5
        )
        
        # Animation 3: Axes Rotation
        self.play(
            axes_rotation_tracker.animate.set_value(20 * DEGREES),
            run_time=2.0
        )
        
        self.play(
            rotated_axis_label_opacity_tracker.animate.set_value(1.0),
            run_time=1.0
        )
        
        self.wait(1)
        
        # Animation 3: Separation (Pullback)
        # Move Right Panel components to the right
        # separation_tracker -> 1.0 (full separation vector)
        
        self.play(
            separation_tracker.animate.set_value(1.0),
            FadeOut(angle_arc),
            FadeOut(angle_label),
            run_time=2.0
        )
        
        self.wait(1)
        
        # Animation 4: Right Panel Rotation
        self.play(
            right_panel_rotation_tracker.animate.set_value(20 * DEGREES),
            run_time=2.0
        )
        
        self.play(
            Write(eq_eq),
            Write(eq_rhs),
            run_time=1.5
        )
        
        self.wait(2)

    def create_kernel_image(self, resolution=220, span=3.0, cutoff=None, shape="square", rotation_angle=0.0):
        np.random.seed(10)

        xs = np.linspace(-span, span, resolution)
        ys = np.linspace(-span, span, resolution)
        X, Y = np.meshgrid(xs, ys)
        
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

        h, w = v.shape
        cy, cx = h // 2, w // 2
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
