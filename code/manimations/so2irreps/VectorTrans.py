from manim import *
import numpy as np

class VectorTrans(Scene):
    def construct(self):
        GLOBAL_SCALE = 1.05

        # --- Axes ---
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            x_length=3 * GLOBAL_SCALE, 
            y_length=3 * GLOBAL_SCALE, 
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

        axes_group = VGroup(axes, axes_labels)
        axes_group.to_corner(UL, buff=0.6)
        self.add(axes_group)

        # --- Vector p ---
        # Using a vector roughly visible in the first quadrant
        p_val = np.array([2.5, 0.4, 0])
        p_len = np.linalg.norm(p_val[:2])
        u_len = 0.5 * p_len
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
        original_label.next_to(p_target_orig, UP + RIGHT, buff=0.1 * GLOBAL_SCALE)
        
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
        
        def get_rotated_vector_group():
            angle = vector_rotation_tracker.get_value()
            
            c, s = np.cos(angle), np.sin(angle)
            # Rotating p vector
            rx = p_val[0] * c - p_val[1] * s
            ry = p_val[0] * s + p_val[1] * c
            
            target = axes.c2p(rx, ry)
            
            arrow = Arrow(
                origin, 
                target,
                buff=0,
                color=WHITE, 
                stroke_width=2.5 * GLOBAL_SCALE,
                tip_length=0.08 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            )
            
            # Label for rotated vector
            label_tex = r"\mathbf{Q}_\alpha \mathbf{p}"
            label = MathTex(label_tex, color=WHITE).scale(0.5 * GLOBAL_SCALE)
            label.next_to(target, UP + RIGHT, buff=0.1 * GLOBAL_SCALE)
            
            # Only show label if separated enough to avoid clutter
            if abs(angle) < 0.1:
                label.set_opacity(0)
            else:
                label.set_opacity(1)
            
            rotated_u_vec = np.array(
                [
                    u_vec_base[0] * c - u_vec_base[1] * s,
                    u_vec_base[0] * s + u_vec_base[1] * c,
                    0.0,
                ]
            )
            u_start = np.array([rx, ry, 0.0])
            u_end = u_start + rotated_u_vec
            rotated_u_arrow = Arrow(
                axes.c2p(u_start[0], u_start[1]),
                axes.c2p(u_end[0], u_end[1]),
                buff=0,
                color=BLUE_C,
                stroke_width=2.0 * GLOBAL_SCALE,
                tip_length=0.07 * GLOBAL_SCALE,
                max_tip_length_to_length_ratio=1.0,
            )
            rotated_u_label = MathTex(r"\mathbf{u}^\alpha(\mathbf{Q}_\alpha \mathbf{p})", color=BLUE_C).scale(
                0.5 * GLOBAL_SCALE
            )
            rotated_u_label.next_to(rotated_u_arrow.get_end(), RIGHT + 0.2 * UP, buff=0.06 * GLOBAL_SCALE)
            if abs(angle) < 0.1:
                rotated_u_label.set_opacity(0)
            else:
                rotated_u_label.set_opacity(1)
            
            return VGroup(arrow, label, rotated_u_arrow, rotated_u_label)

        vector_group = always_redraw(get_rotated_vector_group)
        self.add(vector_group)
        
        # --- Angle Indicator ---
        def angle_indicator_group():
            angle = vector_rotation_tracker.get_value()
            if abs(angle) < 0.1:
                return VGroup()
                
            # Current rotated p direction
            c, s = np.cos(angle), np.sin(angle)
            rx = p_val[0] * c - p_val[1] * s
            ry = p_val[0] * s + p_val[1] * c
            p_target_rot = axes.c2p(rx, ry)
            
            # Lines for Angle mobject
            # We recreate them to ensure Angle works correctly
            l1 = Line(origin, p_target_orig)
            l2 = Line(origin, p_target_rot)
            
            arc = Angle(l1, l2, radius=0.8 * GLOBAL_SCALE, color=WHITE)
            lbl = MathTex(r"\alpha", color=WHITE).scale(0.6 * GLOBAL_SCALE)
            
            # Position label
            lbl.move_to(Angle(l1, l2, radius=1.1 * GLOBAL_SCALE).point_from_proportion(0.5))
            
            return VGroup(arc, lbl)

        angle_indicator = always_redraw(angle_indicator_group)
        self.add(angle_indicator)
        
        self.wait(1)
        
        # Animate rotation
        self.play(
            vector_rotation_tracker.animate.set_value(45 * DEGREES),
            run_time=2.0
        )

        final_angle = vector_rotation_tracker.get_value()
        c_f, s_f = np.cos(final_angle), np.sin(final_angle)
        rx_f = p_val[0] * c_f - p_val[1] * s_f
        ry_f = p_val[0] * s_f + p_val[1] * c_f
        rotated_u_vec_final = np.array(
            [
                u_vec_base[0] * c_f - u_vec_base[1] * s_f,
                u_vec_base[0] * s_f + u_vec_base[1] * c_f,
                0.0,
            ]
        )
        final_u_start = np.array([rx_f, ry_f, 0.0])
        final_u_end = final_u_start + rotated_u_vec_final

        dotted_u = DashedLine(
            origin,
            axes.c2p(*original_u_end),
            dash_length=0.07,
            stroke_width=1.2 * GLOBAL_SCALE,
            color=GREY_B,
        ).add_tip(
            tip_length=0.06 * GLOBAL_SCALE,
            tip_width=0.04 * GLOBAL_SCALE,
        )

        dotted_rotated_u = DashedLine(
            origin,
            axes.c2p(final_u_end[0], final_u_end[1]),
            dash_length=0.07,
            stroke_width=1.2 * GLOBAL_SCALE,
            color=GREY_B,
        ).add_tip(
            tip_length=0.06 * GLOBAL_SCALE,
            tip_width=0.04 * GLOBAL_SCALE,
        )

        dotted_group = VGroup(dotted_u, dotted_rotated_u)
        self.play(FadeIn(dotted_group), run_time=1.0)
        
        self.wait(1)

