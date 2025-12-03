from manim import *
import numpy as np

class TensorTransSO2Basis(Scene):
    def construct(self):
        # 4x2 grid configuration
        grid_w = config.frame_width / 4
        grid_h = config.frame_height / 2
        panel_size = min(grid_w, grid_h) * 0.85
        
        GLOBAL_SCALE = 0.9

        # --- Axes ---
        # Panel 1 (Spatial Domain)
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=panel_size, 
            y_length=panel_size, 
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
        
        axes_labels = axes.get_axis_labels(
            MathTex("s", color=WHITE).scale(GLOBAL_SCALE), 
            MathTex("t", color=WHITE).scale(GLOBAL_SCALE)
        )

        axes_group1 = VGroup(axes, axes_labels)

        # Panel 2 (Frequency +1)
        axes2 = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=panel_size, 
            y_length=panel_size, 
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
        
        # Vertical: Re(u^(+1)(p)), Horizontal: Im(u^(+1)(p))
        axes2_labels = axes2.get_axis_labels(
            MathTex(r"\Im(\sigma^{+2}(\mathbf{p}))", color=WHITE).scale(0.6 * GLOBAL_SCALE),
            MathTex(r"\Re(\sigma^{+2}(\mathbf{p}))", color=WHITE).scale(0.6 * GLOBAL_SCALE)
        )
        axes_group2 = VGroup(axes2, axes2_labels)

        # Panel 3 (Frequency -1)
        axes3 = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=panel_size, 
            y_length=panel_size, 
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
        
        # Vertical: Re(u^(-1)(p)), Horizontal: Im(u^(-1)(p))
        axes3_labels = axes3.get_axis_labels(
            MathTex(r"\Im(\sigma^{-2}(\mathbf{p}))", color=WHITE).scale(0.6 * GLOBAL_SCALE),
            MathTex(r"\Re(\sigma^{-2}(\mathbf{p}))", color=WHITE).scale(0.6 * GLOBAL_SCALE)
        )
        axes_group3 = VGroup(axes3, axes3_labels)

        # Positioning on 4x2 grid (top row, first 3 columns)
        # Top-left of grid is (-frame_width/2, frame_height/2)
        # Row 0 y-center: frame_height/2 - grid_h/2
        # Col 0 x-center: -frame_width/2 + grid_w/2
        
        row_y = config.frame_height / 2 - grid_h / 2
        col_0_x = -config.frame_width / 2 + grid_w / 2
        
        # Center the axes (coordinate systems) themselves on the grid points
        # rather than centering the groups (which include varying labels)
        
        # Panel 1
        target_c1 = np.array([col_0_x, row_y, 0])
        shift_v1 = target_c1 - axes.get_center()
        axes_group1.shift(shift_v1)
        
        # Panel 2
        target_c2 = np.array([col_0_x + grid_w, row_y, 0])
        shift_v2 = target_c2 - axes2.get_center()
        axes_group2.shift(shift_v2)

        # Panel 3
        target_c3 = np.array([col_0_x + 2 * grid_w, row_y, 0])
        shift_v3 = target_c3 - axes3.get_center()
        axes_group3.shift(shift_v3)

        self.add(axes_group1, axes_group2, axes_group3)

        # --- Vector p ---
        # Using a vector roughly visible in the first quadrant
        p_val = np.array([2.5, 0.4, 0])
        p_len = np.linalg.norm(p_val[:2])
        u_len = 0.75 * p_len  # 50% longer than before
        u_angle = 60 * DEGREES  # up-right, 30 degrees from vertical
        u_vec_base = np.array(
            [
                u_len * np.cos(u_angle),
                u_len * np.sin(u_angle),
                0.0,
            ]
        )
        
        vector_rotation_tracker = ValueTracker(0.0)
        
        # Origin in scene coordinates
        origin = axes.c2p(0.0, 0.0)
        
        # Reference vector (original p)
        p_target_orig = axes.c2p(*p_val[:2])
        original_arrow = Arrow(
            origin,
            p_target_orig,
            buff=0,
            color=GREY,
            stroke_width=2.5 * GLOBAL_SCALE,
            tip_length=0.08 * GLOBAL_SCALE,
            max_tip_length_to_length_ratio=1.0,
        )
        original_label = MathTex(r"\mathbf{p}", color=GREY).scale(0.5 * GLOBAL_SCALE)
        # original_label.next_to(p_target_orig, UP + RIGHT, buff=0.1 * GLOBAL_SCALE)
        # Put label at midpoint of the vector line
        original_label.move_to(original_arrow.get_center() + 0.2 * UP * GLOBAL_SCALE)
        
        # --- Stress tensor setup (copied from TensorTrans) ---
        sigma = np.array([[2.0, 0.8],
                          [0.8, 1.0]])
        tau_angle = -130 * DEGREES
        tau = np.array([np.cos(tau_angle), np.sin(tau_angle)])
        n_vec = np.array([-tau[1], tau[0]])
        beta_phase_offset = float(np.mod(np.arctan2(n_vec[1], n_vec[0]), TAU))
        line_len = 0.8

        def rot2(alpha):
            c, s = np.cos(alpha), np.sin(alpha)
            return np.array([[c, -s],
                             [s,  c]])

        def make_curved_line(point_coords, direction_vec, color, stroke_width, anchor_point):
            line_end_1 = point_coords - 0.5 * line_len * direction_vec
            line_end_2 = point_coords + 0.5 * line_len * direction_vec
            normal_dir = np.array([-direction_vec[1], direction_vec[0]])
            control_point = (line_end_1 + line_end_2) / 2 + 0.2 * line_len * normal_dir

            curve = VMobject(color=color, stroke_width=stroke_width)
            curve.set_points_smoothly(
                [
                    axes.c2p(*line_end_1),
                    axes.c2p(*control_point),
                    axes.c2p(*line_end_2),
                ]
            )
            curve.shift(anchor_point - axes.c2p(*control_point))
            return curve

        curve_rotation = 0.0
        t_vec = sigma @ n_vec

        tip_scene = original_arrow.get_end()
        base_line = make_curved_line(p_val[:2], tau, GREY_B, 2 * GLOBAL_SCALE, tip_scene)

        horiz_half = 0.35 * GLOBAL_SCALE
        horizontal_line_base = Line(
            tip_scene + horiz_half * LEFT,
            tip_scene + horiz_half * RIGHT,
            color=WHITE,
            stroke_width=1.5 * GLOBAL_SCALE,
        )

        n_arrow_base = Arrow(
            tip_scene,
            axes.c2p(*(p_val[:2] + 1.4 * n_vec)),
            buff=0,
            color=GREEN_C,
            stroke_width=2 * GLOBAL_SCALE,
            tip_length=0.07 * GLOBAL_SCALE,
        )
        n_label_base = MathTex(r"\hat{\mathbf{n}}(\mathbf{p})", color=GREEN_C).scale(0.5 * GLOBAL_SCALE)
        n_label_base.next_to(n_arrow_base.get_end(), UP + RIGHT, buff=0.06)

        t_arrow_base = Arrow(
            tip_scene,
            axes.c2p(*(p_val[:2] + 1.4 * t_vec)),
            buff=0,
            color=BLUE_C,
            stroke_width=2 * GLOBAL_SCALE,
            tip_length=0.07 * GLOBAL_SCALE,
        )
        t_label_base = MathTex(r"\mathbf{t}(\mathbf{p})", color=BLUE_C).scale(0.45 * GLOBAL_SCALE)
        t_label_base.next_to(t_arrow_base.get_end(), RIGHT + 0.2 * UP, buff=0.06)

        beta_angle_base = float(np.mod(np.arctan2(n_vec[1], n_vec[0]), TAU))
        beta_arc_base = Arc(
            radius=0.35 * GLOBAL_SCALE,
            start_angle=0.0,
            angle=beta_angle_base,
            arc_center=tip_scene,
            color=GREEN,
            stroke_width=2.0 * GLOBAL_SCALE,
        )
        beta_label_base = MathTex(r"\beta", color=GREEN).scale(0.45 * GLOBAL_SCALE)
        beta_offset_dir_base = UP if beta_angle_base <= PI else DOWN
        beta_label_base.move_to(
            beta_arc_base.point_from_proportion(0.5)
            + 0.05 * beta_offset_dir_base * GLOBAL_SCALE
        )

        self.add(
            original_arrow,
            original_label,
            base_line,
            horizontal_line_base,
            n_arrow_base,
            n_label_base,
            t_arrow_base,
            t_label_base,
            beta_arc_base,
            beta_label_base,
        )
        
        # --- Combined Rotating Group ---
        def get_rotating_group():
            angle = vector_rotation_tracker.get_value()
            psi = 2 * angle
            
            c, s = np.cos(angle), np.sin(angle)
            
            # --- 1. Rotated p vector ---
            # p starts at p_val
            rx = p_val[0] * c - p_val[1] * s
            ry = p_val[0] * s + p_val[1] * c
            p_target_rot = axes.c2p(rx, ry)
            
            arrow_p = Arrow(
                origin, 
                p_target_rot,
                buff=0,
                color=GREY, 
                stroke_width=2.5 * GLOBAL_SCALE,
                tip_length=0.08 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            )
            
            label_p = MathTex(r"\mathbf{p}", color=GREY).scale(0.5 * GLOBAL_SCALE)
            label_p.move_to(arrow_p.get_center() + 0.2 * UP * GLOBAL_SCALE)
            
            # --- Material line, normal, traction (rotated) ---
            p_rot = np.array([rx, ry])
            p_tip = p_target_rot
            tau_rot = rot2(angle) @ tau
            n_rot = rot2(angle) @ n_vec
            t_rot = rot2(angle) @ (sigma @ n_vec)

            line_rot = make_curved_line(p_rot, tau_rot, WHITE, 2.4 * GLOBAL_SCALE, p_tip)

            horiz_line_rot = Line(
                p_tip + horiz_half * LEFT,
                p_tip + horiz_half * RIGHT,
                color=WHITE,
                stroke_width=1.5 * GLOBAL_SCALE,
            )

            n_arrow_rot = Arrow(
                p_tip,
                axes.c2p(*(p_rot + 1.4 * n_rot)),
                buff=0,
                color=GREEN_C,
                stroke_width=2.2 * GLOBAL_SCALE,
                tip_length=0.07 * GLOBAL_SCALE,
            )
            n_label_rot = MathTex(r"\hat{\mathbf{n}}^\alpha(\mathbf{Q}_\alpha\mathbf{p})",
                                  color=GREEN_C).scale(0.45 * GLOBAL_SCALE)
            n_label_rot.next_to(n_arrow_rot.get_end(), UP + RIGHT, buff=0.06)

            t_arrow_rot = Arrow(
                p_tip,
                axes.c2p(*(p_rot + 1.4 * t_rot)),
                buff=0,
                color=BLUE_C,
                stroke_width=2.2 * GLOBAL_SCALE,
                tip_length=0.07 * GLOBAL_SCALE,
            )
            t_label_rot = MathTex(
                r"\mathbf{t}^\alpha(\mathbf{Q}_\alpha\mathbf{p})",
                color=BLUE_C
            ).scale(0.45 * GLOBAL_SCALE)
            t_label_rot.next_to(t_arrow_rot.get_end(), RIGHT + 0.2 * UP, buff=0.06)

            beta_angle_dynamic = float(np.mod(np.arctan2(n_rot[1], n_rot[0]), TAU))
            beta_arc_dynamic = Arc(
                radius=0.35 * GLOBAL_SCALE,
                start_angle=0.0,
                angle=beta_angle_dynamic,
                arc_center=p_tip,
                color=GREEN,
                stroke_width=2.0 * GLOBAL_SCALE,
            )
            beta_label_dynamic = MathTex(r"\beta", color=GREEN).scale(0.45 * GLOBAL_SCALE)
            beta_offset_dir_dynamic = UP if beta_angle_dynamic <= PI else DOWN
            beta_label_dynamic.move_to(
                beta_arc_dynamic.point_from_proportion(0.5)
                + 0.05 * beta_offset_dir_dynamic * GLOBAL_SCALE
            )
            
            # --- 5. Vectors on Panel 2 and 3 ---
            # New vectors emanate from origin of their respective plots and rotate with beta
            beta_vector_angle = u_angle + psi - beta_phase_offset + PI  # additional 180° shift
            beta_arc_angle = (beta_vector_angle % TAU + TAU) % TAU  # keep arc in [0, 2π)
            beta_arc_angle_neg = -beta_arc_angle
            
            # Vector length? Same as u?
            # Let's use u_len.
            
            # Panel 2 Vector (Freq +1?)
            # Origin of axes2
            origin2 = axes2.c2p(0,0)
            vec2_end_local = np.array([
                u_len * np.cos(beta_vector_angle),
                u_len * np.sin(beta_vector_angle),
                0
            ])
            vec2_end = axes2.c2p(vec2_end_local[0], vec2_end_local[1])
            
            arrow2 = Arrow(
                origin2,
                vec2_end,
                buff=0,
                color=BLUE_C, # Match u color?
                stroke_width=2.0 * GLOBAL_SCALE,
                tip_length=0.07 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            )
            label_u_plus = MathTex(r"\sigma^{+2}", color=BLUE_C).scale(0.5 * GLOBAL_SCALE)
            label_u_plus.next_to(arrow2.get_end(), RIGHT + 0.2 * UP, buff=0.08 * GLOBAL_SCALE)
            
            # Panel 3 Vector (Freq -1?)
            # Request says "beta must be the same as the angle beta on the left plot".
            # So both use beta_val?
            # Usually +1 and -1 frequencies rotate in opposite directions?
            # But the request specifically says "beta must be the same...".
            # So I will make them identical for now as requested.
            
            origin3 = axes3.c2p(0,0)
            vec3_end_local = np.array([
                u_len * np.cos(-beta_vector_angle),
                u_len * np.sin(-beta_vector_angle),
                0
            ])
            vec3_end = axes3.c2p(vec3_end_local[0], vec3_end_local[1])
            
            arrow3 = Arrow(
                origin3,
                vec3_end,
                buff=0,
                color=BLUE_C,
                stroke_width=2.0 * GLOBAL_SCALE,
                tip_length=0.07 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            )
            label_u_minus = MathTex(r"\sigma^{-2}", color=BLUE_C).scale(0.5 * GLOBAL_SCALE)
            label_u_minus.next_to(arrow3.get_end(), LEFT + 0.2 * DOWN, buff=0.08 * GLOBAL_SCALE)
            
            # Also add beta angles for these new vectors?
            # "they must subtend an angle beta with their horizontal axes"
            
            # Helper for return
            def pack_group(beta2_angle_parts=None, beta3_angle_parts=None):
                if beta2_angle_parts is None:
                    beta2_angle_parts = []
                if beta3_angle_parts is None:
                    beta3_angle_parts = []
                return VGroup(
                    line_rot,
                    horiz_line_rot,
                    beta_arc_dynamic, beta_label_dynamic,
                    n_arrow_rot, n_label_rot,
                    t_arrow_rot, t_label_rot,
                    arrow_p, label_p,
                    arrow2, label_u_plus,
                    arrow3, label_u_minus,
                    *beta2_angle_parts,
                    *beta3_angle_parts
                )
            
            # Use arcs for beta indicators to avoid flicker/glitches
            beta2_arc = Arc(
                radius=0.4 * GLOBAL_SCALE,
                start_angle=0.0,
                angle=beta_arc_angle,
                arc_center=origin2,
                color=GREEN,
                stroke_width=2.0 * GLOBAL_SCALE,
            )
            if abs(beta_arc_angle) > 1e-3:
                beta2_label = MathTex(r"2\alpha", color=GREEN).scale(0.5 * GLOBAL_SCALE)
                beta2_label.move_to(beta2_arc.point_from_proportion(0.5) + 0.05 * UP * GLOBAL_SCALE)
                beta2_parts = [beta2_arc, beta2_label]
            else:
                beta2_parts = [beta2_arc]

            beta3_arc = Arc(
                radius=0.4 * GLOBAL_SCALE,
                start_angle=0.0,
                angle=beta_arc_angle_neg,
                arc_center=origin3,
                color=GREEN,
                stroke_width=2.0 * GLOBAL_SCALE,
            )
            beta3_label = MathTex(r"-2\alpha", color=GREEN).scale(0.5 * GLOBAL_SCALE)
            beta3_label.move_to(
                beta3_arc.point_from_proportion(0.5) + 0.05 * DOWN * GLOBAL_SCALE
            )
            beta3_parts = [beta3_arc, beta3_label]

            return pack_group(beta2_parts, beta3_parts)

        rotating_group = always_redraw(get_rotating_group)
        self.add(rotating_group)
        
        # Remove static initial objects since they are now covered by the updater at angle=0
        self.remove(
            original_arrow, original_label,
            base_line, horizontal_line_base,
            n_arrow_base, n_label_base,
            t_arrow_base, t_label_base,
            beta_arc_base, beta_label_base
        )
        
        self.wait(1)
        
        # Animate rotation: repeat the stable 1-rotation segment five times
        per_rotation_time = 4.0  # Matches the previous angular speed
        rotations = 5
        for i in range(rotations):
            self.play(
                vector_rotation_tracker.animate.set_value(TAU),
                run_time=per_rotation_time,
                rate_func=linear,
            )
            # Reset tracker so each subsequent pass reuses the stable range [0, 2π)
            if i < rotations - 1:
                vector_rotation_tracker.set_value(0)

        self.wait(1)

