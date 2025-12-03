from manim import *
import numpy as np

class VectorTransSO2Basis(Scene):
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
            MathTex(r"\Im(u^{+1}(\mathbf{p}))", color=WHITE).scale(0.6 * GLOBAL_SCALE),
            MathTex(r"\Re(u^{+1}(\mathbf{p}))", color=WHITE).scale(0.6 * GLOBAL_SCALE)
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
            MathTex(r"\Im(u^{-1}(\mathbf{p}))", color=WHITE).scale(0.6 * GLOBAL_SCALE),
            MathTex(r"\Re(u^{-1}(\mathbf{p}))", color=WHITE).scale(0.6 * GLOBAL_SCALE)
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
            color=WHITE,
            stroke_width=2.5 * GLOBAL_SCALE,
            tip_length=0.08 * GLOBAL_SCALE,
            max_tip_length_to_length_ratio=1.0,
        )
        original_label = MathTex(r"\mathbf{p}", color=WHITE).scale(0.5 * GLOBAL_SCALE)
        # original_label.next_to(p_target_orig, UP + RIGHT, buff=0.1 * GLOBAL_SCALE)
        # Put label at midpoint of the vector line
        original_label.move_to(original_arrow.get_center() + 0.2 * UP * GLOBAL_SCALE)
        
        original_u_start = p_val[:2]
        original_u_end = original_u_start + u_vec_base[:2]
        original_u_arrow = Arrow(
            axes.c2p(*original_u_start),
            axes.c2p(*original_u_end),
            buff=0,
            color=BLUE_C,
            stroke_width=2.0 * GLOBAL_SCALE,
            tip_length=0.07 * GLOBAL_SCALE,
            max_tip_length_to_length_ratio=1.0,
        )
        original_u_label = MathTex(r"\mathbf{u}(\mathbf{p})", color=BLUE_C).scale(0.5 * GLOBAL_SCALE)
        original_u_label.next_to(original_u_arrow.get_end(), RIGHT + 0.2 * UP, buff=0.08 * GLOBAL_SCALE)
        
        self.add(original_arrow, original_label, original_u_arrow, original_u_label)
        
        # --- Angles ---
        # Angle alpha: subtended by s-axis and vector p
        # Create a horizontal line along s-axis for reference
        s_axis_end = axes.c2p(3.0, 0.0)
        s_line = Line(origin, s_axis_end, stroke_width=0).set_opacity(0) # invisible reference line
        
        angle_alpha = Angle(
            s_line,
            original_arrow,
            radius=0.6 * GLOBAL_SCALE,
            other_angle=False,
            color=WHITE
        )
        label_alpha = MathTex(r"\alpha", color=WHITE).scale(0.6 * GLOBAL_SCALE)
        label_alpha.next_to(angle_alpha, RIGHT, buff=0.05 * GLOBAL_SCALE)
        
        # Angle beta: subtended by vector u with a small horizontal line at p's end
        # Horizontal reference line at p's end
        
        # Horizontal line length
        h_line_len = 0.8 * GLOBAL_SCALE
        p_end_point = original_arrow.get_end()
        
        # Line extending horizontally to the right from p's tip
        horiz_line = Line(
            p_end_point,
            p_end_point + np.array([h_line_len, 0, 0]),
            color=WHITE,
            stroke_width=1.5 * GLOBAL_SCALE
        )
        
        angle_beta = Angle(
            horiz_line,
            original_u_arrow,
            radius=0.5 / 3 * GLOBAL_SCALE,
            other_angle=False,
            color=BLUE
        )
        label_beta = MathTex(r"\beta", color=BLUE).scale(0.6 * GLOBAL_SCALE)
        # Position beta label closer to arc
        label_beta.next_to(angle_beta, RIGHT, buff=0.05 * GLOBAL_SCALE)
        label_beta.shift(0.05 * UP * GLOBAL_SCALE)

        self.add(angle_alpha, label_alpha, horiz_line, angle_beta, label_beta)
        
        # --- Combined Rotating Group ---
        def get_rotating_group():
            angle = vector_rotation_tracker.get_value()
            
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
                color=WHITE, 
                stroke_width=2.5 * GLOBAL_SCALE,
                tip_length=0.08 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            )
            
            label_p = MathTex(r"\mathbf{p}", color=WHITE).scale(0.5 * GLOBAL_SCALE)
            label_p.move_to(arrow_p.get_center() + 0.2 * UP * GLOBAL_SCALE)
            
            # --- 2. Rotated u(p) vector ---
            # Original u vector base (relative to p's tip)
            rotated_u_vec = np.array(
                [
                    u_vec_base[0] * c - u_vec_base[1] * s,
                    u_vec_base[0] * s + u_vec_base[1] * c,
                    0.0,
                ]
            )
            
            u_start = np.array([rx, ry, 0.0])
            u_end = u_start + rotated_u_vec
            
            arrow_u = Arrow(
                axes.c2p(u_start[0], u_start[1]),
                axes.c2p(u_end[0], u_end[1]),
                buff=0,
                color=BLUE_C,
                stroke_width=2.0 * GLOBAL_SCALE,
                tip_length=0.07 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            )
            
            label_u = MathTex(r"\mathbf{u}(\mathbf{p})", color=BLUE_C).scale(0.5 * GLOBAL_SCALE)
            label_u.next_to(arrow_u.get_end(), RIGHT + 0.2 * UP, buff=0.08 * GLOBAL_SCALE)
            
            # --- 3. Angle Alpha (s-axis to p) ---
            # Recreate s_line inside the function or access it from outer scope if defined before.
            # It was defined outside, but maybe not captured correctly if it was removed from scene?
            # No, removing from scene doesn't delete the variable.
            # However, Python scoping in nested functions usually captures variables.
            # The traceback says "name 's_line' is not defined".
            # This implies it wasn't defined in the scope where get_rotating_group runs.
            # But s_line IS defined above in construct. 
            # Wait, maybe the variable was defined in a block that I accidentally overwrote or I'm misreading.
            # Let's redefine s_line here to be safe, using the same logic.
            
            s_axis_end_local = axes.c2p(3.0, 0.0)
            s_line_local = Line(origin, s_axis_end_local, stroke_width=0).set_opacity(0)

            # We need a line from origin to current p for the Angle mobject
            line_to_p = Line(origin, p_target_rot, stroke_width=0)
            
            # Let's construct alpha between fixed s-axis and current p.
            angle_alpha_mob = Angle(
                s_line_local,
                line_to_p,
                radius=0.6 * GLOBAL_SCALE,
                other_angle=False,
                color=WHITE
            )
            # Ensure label alpha follows the arc
            label_alpha_mob = MathTex(r"\alpha", color=WHITE).scale(0.6 * GLOBAL_SCALE)
            # Use point_from_proportion to place label
            label_alpha_mob.move_to(
                Angle(s_line_local, line_to_p, radius=0.6 * GLOBAL_SCALE + 0.25).point_from_proportion(0.5)
            )
            
            # --- 4. Angle Beta (horizontal from p to u) ---
            # Horizontal line at current p tip
            h_line_len = 0.8 * GLOBAL_SCALE
            p_tip = p_target_rot
            
            horiz_line_mob = Line(
                p_tip,
                p_tip + np.array([h_line_len, 0, 0]),
                color=WHITE,
                stroke_width=1.5 * GLOBAL_SCALE
            )
            
            # Angle Beta: between this horizontal line and u
            # u extends from p_tip to u_end
            line_u = Line(p_tip, axes.c2p(u_end[0], u_end[1]), stroke_width=0)
            
            # Manim's Angle class tries to find the intersection of the two lines to determine the vertex.
            # If line_u is perfectly horizontal, it might be parallel to horiz_line_mob.
            # This happens if u has 0 y-component relative to p.
            # To fix "lines are parallel", we can slightly perturbe one if needed, or ensure they share a start point.
            # Angle(l1, l2) assumes l1 and l2 share a vertex if they aren't intersecting elsewhere?
            # Actually, Angle expects lines to intersect. If they share a start point, that IS the intersection.
            # However, floating point precision might make them seem parallel/disjoint if they are collinear.
            # If u is horizontal, angle is 0 or 180.
            # If angle is 0, Angle mobject might fail or return empty.
            
            # Check if u is parallel to horizontal
            # u vector in scene coords:
            u_vec_scene = axes.c2p(u_end[0], u_end[1]) - p_tip
            # Horizontal vector: [1, 0, 0]
            
            # If u is very close to horizontal, we might need to handle it.
            # But usually Angle handles collinear lines by returning 0 arc? 
            # No, the error is explicitly "The lines are parallel, there is no unique intersection point."
            # This comes from finding the intersection.
            
            # Explicitly setting the lines to share the exact same start point helps Manim find the intersection?
            # They already share p_tip.
            # The issue is likely when they are collinear (angle 0 or 180).
            
            # Fallback: if collinear, just return empty VGroup for the angle arc?
            # Or perturb u slightly.
            
            # Simple fix: if u_vec_base angle is 0 or pi, we might have issues.
            # In our case u_vec_base is at 60 degrees.
            # But as it rotates, it might align horizontally?
            # No, u rotates WITH p?
            # Wait, `rotated_u_vec` rotates by `angle`.
            # `angle` goes from 0 to 10pi.
            # `u_vec_base` starts at 60 deg.
            # So `rotated_u_vec` angle is 60 + alpha.
            # At some point 60 + alpha = 0 or 180 (mod 360).
            # Yes, it will align horizontally multiple times.
            
            # Workaround: Use Arc instead of Angle if we know the angles.
            # Or catch the error.
            
            # --- 4. Angle Beta (horizontal from p to u) ---
            # ... (existing code for beta) ...
            
            # --- 5. Vectors on Panel 2 and 3 ---
            # New vectors emanate from origin of their respective plots and rotate with beta
            beta_vector_angle = u_angle + angle  # running beta (can exceed 2π)
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
            label_u_plus = MathTex(r"\mathbf{u}^{+}(\mathbf{p})", color=BLUE_C).scale(0.5 * GLOBAL_SCALE)
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
            label_u_minus = MathTex(r"\mathbf{u}^{-}(\mathbf{p})", color=BLUE_C).scale(0.5 * GLOBAL_SCALE)
            label_u_minus.next_to(arrow3.get_end(), LEFT + 0.2 * DOWN, buff=0.08 * GLOBAL_SCALE)
            
            # Also add beta angles for these new vectors?
            # "they must subtend an angle beta with their horizontal axes"
            
            # Helper for return
            def pack_group(beta_angle_parts, beta2_angle_parts=None, beta3_angle_parts=None):
                if beta2_angle_parts is None:
                    beta2_angle_parts = []
                if beta3_angle_parts is None:
                    beta3_angle_parts = []
                return VGroup(
                    arrow_p, label_p, 
                    arrow_u, label_u, 
                    angle_alpha_mob, label_alpha_mob,
                    horiz_line_mob, 
                    *beta_angle_parts,
                    arrow2, label_u_plus,
                    arrow3, label_u_minus,
                    *beta2_angle_parts,
                    *beta3_angle_parts
                )
            
            # Use arcs for beta indicators to avoid flicker/glitches
            beta_arc = Arc(
                radius=0.5 / 3 * GLOBAL_SCALE,
                start_angle=0.0,
                angle=beta_arc_angle,
                arc_center=p_tip,
                color=BLUE,
                stroke_width=2.0 * GLOBAL_SCALE,
            )
            beta_label_dynamic = MathTex(r"\beta", color=BLUE).scale(0.5 * GLOBAL_SCALE)
            beta_label_dynamic.move_to(
                beta_arc.point_from_proportion(0.5) + 0.05 * UP * GLOBAL_SCALE
            )
            beta_parts = [beta_arc, beta_label_dynamic]

            beta2_arc = Arc(
                radius=0.4 * GLOBAL_SCALE,
                start_angle=0.0,
                angle=beta_arc_angle,
                arc_center=origin2,
                color=BLUE,
                stroke_width=2.0 * GLOBAL_SCALE,
            )
            if abs(beta_arc_angle) > 1e-3:
                beta2_label = MathTex(r"\beta", color=BLUE).scale(0.5 * GLOBAL_SCALE)
                beta2_label.move_to(beta2_arc.point_from_proportion(0.5) + 0.05 * UP * GLOBAL_SCALE)
                beta2_parts = [beta2_arc, beta2_label]
            else:
                beta2_parts = [beta2_arc]

            beta3_arc = Arc(
                radius=0.4 * GLOBAL_SCALE,
                start_angle=0.0,
                angle=beta_arc_angle_neg,
                arc_center=origin3,
                color=BLUE,
                stroke_width=2.0 * GLOBAL_SCALE,
            )
            beta3_label = MathTex(r"-\beta", color=BLUE).scale(0.5 * GLOBAL_SCALE)
            beta3_label.move_to(
                beta3_arc.point_from_proportion(0.5) + 0.05 * DOWN * GLOBAL_SCALE
            )
            beta3_parts = [beta3_arc, beta3_label]

            return pack_group(beta_parts, beta2_parts, beta3_parts)

        rotating_group = always_redraw(get_rotating_group)
        self.add(rotating_group)
        
        # Remove static initial objects since they are now covered by the updater at angle=0
        self.remove(
            original_arrow, original_label, original_u_arrow, original_u_label,
            angle_alpha, label_alpha, horiz_line, angle_beta, label_beta
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

