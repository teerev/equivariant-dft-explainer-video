from manim import *
import numpy as np

class DataVsAnalyticQuadratics(Scene):
    def construct(self):
        # -----------------------
        # Two panels (no axes)
        # -----------------------
        panel_w = 6.2
        panel_h = 5.4
        inset = 0.55  # inner padding where we actually draw trajectories
        xy_readout_gap = 0.25  # extra spacing between x-value and y-label

        left_panel = RoundedRectangle(width=panel_w, height=panel_h, corner_radius=0.2)
        right_panel = RoundedRectangle(width=panel_w, height=panel_h, corner_radius=0.2)
        panels = VGroup(left_panel, right_panel).arrange(RIGHT, buff=0.6).to_edge(DOWN, buff=0.6)

        # NOTE: We keep these panel rectangles only as invisible layout anchors.
        # (User requested removing the visible boxes.)

        # Inner "viewport" rectangles (invisible, used for mapping)
        left_view = Rectangle(
            width=panel_w - 2 * inset,
            height=panel_h - 2 * inset,
        ).move_to(left_panel.get_center()).set_opacity(0.0)

        right_view = left_view.copy().move_to(right_panel.get_center()).set_opacity(0.0)

        # -----------------------
        # Data-space ranges (used for coordinate mapping)
        # -----------------------
        x_min, x_max = -3.0, 3.0
        y_min, y_max = -1.0, 6.0

        def map_to_view(x, y, view_rect: Rectangle):
            """Map (x,y) in data-space to a point in the given view_rect."""
            u = (x - x_min) / (x_max - x_min)
            v = (y - y_min) / (y_max - y_min)
            # clamp for safety
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)

            left = view_rect.get_left()[0]
            right = view_rect.get_right()[0]
            bottom = view_rect.get_bottom()[1]
            top = view_rect.get_top()[1]

            X = left + u * (right - left)
            Y = bottom + v * (top - bottom)
            return np.array([X, Y, 0.0])

        # -----------------------
        # Left: coordinate readout
        # -----------------------
        coord_label = VGroup(
            MathTex("x="),
            DecimalNumber(0, num_decimal_places=2),
            MathTex(",\\; y="),
            DecimalNumber(0, num_decimal_places=2),
        ).arrange(RIGHT, buff=0.12).scale(0.8)

        coord_label.move_to(left_panel.get_corner(UL) + RIGHT * 2.0 + DOWN * 0.65)
        # Prevent overlap when x is negative by adding extra spacing before the y readout
        coord_label[2:].shift(xy_readout_gap * RIGHT)
        x_num = coord_label[1]
        y_num = coord_label[3]
        self.add(coord_label)

        # -----------------------
        # Right: equation text (updated per trajectory)
        # -----------------------
        eq_display = MathTex("y = ax^2 + bx + c").scale(0.85)
        eq_display.move_to(right_panel.get_corner(UL) + RIGHT * 2.2 + DOWN * 0.65)
        self.add(eq_display)

        # -----------------------
        # Trajectories (a,b,c)
        # -----------------------
        trajectories = [
            (-0.45,  0.25, 4.50),
            (-0.35, -0.40, 4.20),
            (-0.55,  0.60, 3.80),
        ]

        travel_x0, travel_x1 = -2.6, 2.6
        t = ValueTracker(0.0)

        # Styling for curves: SAME on left & right
        curve_stroke_width = 5

        def y_of_x(x, a, b, c):
            return a * x * x + b * x + c

        def make_curve(a, b, c, view_rect, n=140):
            xs = np.linspace(travel_x0, travel_x1, n)
            pts = [map_to_view(x, y_of_x(x, a, b, c), view_rect) for x in xs]
            m = VMobject()
            m.set_points_smoothly(pts)
            m.set_stroke(width=curve_stroke_width)
            return m

        def eq_tex(a, b, c):
            # compact sign-aware formatting
            def fmt(v):
                s = f"{v:.2f}"
                s = s.rstrip("0").rstrip(".") if "." in s else s
                return s

            a_s = fmt(a)
            b_s = fmt(abs(b))
            c_s = fmt(abs(c))
            b_sign = "+" if b >= 0 else "-"
            c_sign = "+" if c >= 0 else "-"

            return MathTex(
                r"y = " + a_s + r"x^2 " + b_sign + r" " + b_s + r"x " + c_sign + r" " + c_s
            ).scale(0.85)

        # -----------------------
        # Animate each trajectory (show line once, then fade out)
        # -----------------------
        for i, (a, b, c) in enumerate(trajectories):
            t.set_value(0.0)

            # Curves (identical style on both panels)
            left_curve = make_curve(a, b, c, left_view)
            right_curve = make_curve(a, b, c, right_view)

            # Dots (one at a time per trajectory)
            def dot_pos(view_rect):
                x_val = np.interp(t.get_value(), [0, 1], [travel_x0, travel_x1])
                y_val = y_of_x(x_val, a, b, c)
                return map_to_view(x_val, y_val, view_rect)

            left_dot = always_redraw(lambda: Dot(dot_pos(left_view), radius=0.085))
            right_dot = always_redraw(lambda: Dot(dot_pos(right_view), radius=0.085))

            # Update coordinate readout (data-space)
            def update_readout(_):
                x_val = np.interp(t.get_value(), [0, 1], [travel_x0, travel_x1])
                y_val = y_of_x(x_val, a, b, c)
                x_num.set_value(x_val)
                y_num.set_value(y_val)

            coord_label.add_updater(update_readout)

            # Update equation on the right
            new_eq = eq_tex(a, b, c).move_to(eq_display.get_center())
            self.play(Transform(eq_display, new_eq), run_time=0.45)

            # Show both curves ONCE for this trajectory
            self.play(
                Create(left_curve),
                Create(right_curve),
                FadeIn(left_dot),
                FadeIn(right_dot),
                run_time=0.8,
            )

            # Move particle along the curve
            self.play(t.animate.set_value(1.0), run_time=1.8, rate_func=linear)
            self.wait(0.15)

            # Fade out everything for this trajectory, then proceed
            coord_label.remove_updater(update_readout)
            self.play(
                FadeOut(left_dot),
                FadeOut(right_dot),
                FadeOut(left_curve),
                FadeOut(right_curve),
                run_time=0.6,
            )

        self.wait(0.4)
