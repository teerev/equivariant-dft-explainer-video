from manim import *
from manim.utils.color import ManimColor
import numpy as np


class RealFourierOnCircle(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        # ------------------------------------------------------------
        # 1. Colour scheme: cyan (-1), black (0), orange (+1)
        # ------------------------------------------------------------
        C_CYAN   = np.array([0.0, 0.8, 1.0])
        C_ORANGE = np.array([1.0, 0.35, 0.0])
        C_BLACK  = np.array([0.0, 0.0, 0.0])
        BLUE  = ManimColor("#0066F5")

        def value_to_rgb(v):
            """
            Map value in [-1, 1] to linear RGB between
            cyan -> black -> orange.
            """

            # If you want gamma correction later, uncomment this:
            # v = np.sign(v) * (abs(v) ** 0.6)

            v = np.clip(v, -1, 1)

            if v < 0:
                # negative: interpolate cyan (-1) -> black (0)
                t = (v + 1.0) / 1.0   # v=-1 -> 0, v=0 -> 1
                rgb = (1 - t) * C_CYAN + t * C_BLACK
            else:
                # positive: interpolate black (0) -> orange (+1)
                t = v / 1.0           # v=0 -> 0, v=1 -> 1
                rgb = (1 - t) * C_BLACK + t * C_ORANGE

            return rgb

        # ------------------------------------------------------------
        # 2. Coefficient trackers
        # ------------------------------------------------------------
        a0 = ValueTracker(0.0)
        a1 = ValueTracker(0.0)
        b1 = ValueTracker(0.0)
        a2 = ValueTracker(0.0)
        b2 = ValueTracker(0.0)

        # ------------------------------------------------------------
        # 3. Layout positions
        # ------------------------------------------------------------
        top_y = 2.5
        x_positions = [-5.5, -3.2, -0.9, 1.4]
        small_radius = 0.9
        big_radius = 2.0  # Reduced by another ~10% from 2.24
        center_big = np.array([-3.2, -1.0, 0.0])

        # sample angular grid
        N = 360
        thetas = np.linspace(0, TAU, N)

        # ------------------------------------------------------------
        # 4. Helper: get coloured ring points for a basis function
        # ------------------------------------------------------------
        def get_axes(center, radius):
            axes_group = VGroup()
            pos_len = radius * 1.1
            neg_len = 0.25
            
            x_ax = Line(center + LEFT*neg_len, center + RIGHT*pos_len, color=BLUE, stroke_width=2)
            y_ax = Line(center + DOWN*neg_len, center + UP*pos_len, color=BLUE, stroke_width=2)
            
            x_lab = MathTex(r"x'", color=BLUE).scale(0.5).next_to(x_ax, RIGHT, buff=0.1)
            y_lab = MathTex(r"y'", color=BLUE).scale(0.5).next_to(y_ax, UP, buff=0.1)
            
            axes_group.add(x_ax, y_ax, x_lab, y_lab)
            return axes_group

        def coloured_circle(center, radius, func):
            """
            func(theta) returns scalar in [-1,1]
            Produces a VGroup of line segments with stroke matching colour along the circle.
            """
            group = VGroup()
            
            # Create small segments
            for i in range(len(thetas) - 1):
                th1 = thetas[i]
                th2 = thetas[i+1]
                
                # Use midpoint value for better coloring approximation
                mid_th = (th1 + th2) / 2.0
                val = func(mid_th)
                rgb = value_to_rgb(val)
                
                p1 = center + radius * np.array([np.cos(th1), np.sin(th1), 0.0])
                p2 = center + radius * np.array([np.cos(th2), np.sin(th2), 0.0])
                
                seg = Line(p1, p2, stroke_width=6)
                seg.set_color(ManimColor(rgb))
                group.add(seg)

            return group

        # ------------------------------------------------------------
        # 5. Basis functions (top-row circles)
        # ------------------------------------------------------------
        # Fixed basis functions (amplitude = 1) independent of trackers
        def basis_cos1(th): return np.cos(th)
        def basis_sin1(th): return np.sin(th)
        def basis_cos2(th): return np.cos(2 * th)
        def basis_sin2(th): return np.sin(2 * th)

        basis_funcs = [basis_cos1, basis_sin1, basis_cos2, basis_sin2]
        basis_labels_tex = [
            r"\cos\theta'",
            r"\sin\theta'",
            r"\cos 2\theta'",
            r"\sin 2\theta'",
        ]

        basis_mobjects = []
        basis_labels = []

        for i, func in enumerate(basis_funcs):
            x = x_positions[i]
            center = np.array([x, top_y, 0.0])

            # Axes
            axes = get_axes(center, small_radius)

            # Static rings for basis functions
            ring = coloured_circle(
                center=center,
                radius=small_radius,
                func=func,
            )
            label = MathTex(basis_labels_tex[i]).scale(0.7)
            label.next_to(center, DOWN, buff=0.2)

            basis_mobjects.append(VGroup(axes, ring))
            basis_labels.append(label)

        # ------------------------------------------------------------
        # 6. Main function on circle (bottom)
        # ------------------------------------------------------------
        def full_function(th):
            return (
                a0.get_value()
                + a1.get_value() * np.cos(th)
                + b1.get_value() * np.sin(th)
                + a2.get_value() * np.cos(2 * th)
                + b2.get_value() * np.sin(2 * th)
            )

        main_axes = get_axes(center_big, big_radius)
        main_coloured_circle = always_redraw(
            lambda: coloured_circle(
                center=center_big,
                radius=big_radius,
                func=full_function,
            )
        )

        # ------------------------------------------------------------
        # 7. Coefficient numeric labels
        # ------------------------------------------------------------
        # Colorbar
        colorbar = Rectangle(
            width=0.35,
            height=2.2,
            stroke_color=GRAY_B,
            stroke_width=1.2,
        )
        # Gradient fill from cyan (-1) -> black (0) -> orange (+1)
        # Manim's set_fill with list creates gradient
        colorbar.set_fill(color=[ManimColor(C_CYAN), ManimColor(C_BLACK), ManimColor(C_ORANGE)], opacity=1.0)
        
        # Position colorbar to the left of the main circle
        colorbar.next_to(center_big, LEFT, buff=2.5)

        neg_label = MathTex("-1", color=GRAY_B).scale(0.5)
        neg_label.next_to(colorbar, LEFT, buff=0.1)
        neg_label.align_to(colorbar, DOWN)

        zero_label = MathTex("0", color=GRAY_B).scale(0.5)
        zero_label.next_to(colorbar, LEFT, buff=0.1)
        # Align zero vertically with center
        zero_label.move_to([neg_label.get_center()[0], colorbar.get_center()[1], 0])

        pos_label = MathTex("1", color=GRAY_B).scale(0.5)
        pos_label.next_to(colorbar, LEFT, buff=0.1)
        pos_label.align_to(colorbar, UP)
        
        colorbar_group = VGroup(colorbar, neg_label, zero_label, pos_label)

        def coeff_label(name, tracker, position):
            label = MathTex(fr"{name} = {tracker.get_value():.2f}").scale(0.7)
            label.move_to(position)

            def updater(mob):
                new = MathTex(fr"{name} = {tracker.get_value():.2f}").scale(0.7)
                new.move_to(position)
                mob.become(new)

            label.add_updater(updater)
            return label

        coeff_group = VGroup(
            coeff_label("a_0", a0, np.array([1.5,  0.5, 0.0])),
            coeff_label("a_1", a1, np.array([1.5, -0.2, 0.0])),
            coeff_label("b_1", b1, np.array([1.5, -0.9, 0.0])),
            coeff_label("a_2", a2, np.array([1.5, -1.6, 0.0])),
            coeff_label("b_2", b2, np.array([1.5, -2.3, 0.0])),
        )

        # ------------------------------------------------------------
        # 8. Analytic expression for f(theta)
        # ------------------------------------------------------------
        full_expr = MathTex(
            r"\Theta(\theta') = a_0"
            r" + a_1 \cos\theta'"
            r" + b_1 \sin\theta'"
            r" + a_2 \cos 2\theta'"
            r" + b_2 \sin 2\theta'"
        ).scale(0.8)
        full_expr.move_to(np.array([-2.0, -3.5, 0.0]))

        # ------------------------------------------------------------
        # 9. Draw everything
        # ------------------------------------------------------------
        self.play(*[FadeIn(m) for m in basis_mobjects])
        self.play(*[FadeIn(lbl) for lbl in basis_labels])
        self.play(FadeIn(main_axes), FadeIn(main_coloured_circle), FadeIn(colorbar_group))
        self.play(FadeIn(coeff_group), FadeIn(full_expr))
        self.wait(1.0)

        # ------------------------------------------------------------
        # 10. Animate coefficients
        # ------------------------------------------------------------
        # Swing through many states
        target_sets = [
            {a0: 0.5, a1: 0.8, b1: -0.7, a2: -0.9, b2: 0.85},
            {a0: -0.4, a1: -0.9, b1: 0.6, a2: 0.8, b2: -0.9},
            {a0: 0.2, a1: 0.3, b1: -0.2, a2: -0.5, b2: 0.4},
            {a0: 0.8, a1: -0.2, b1: 0.9, a2: 0.3, b2: -0.7},
            {a0: -0.6, a1: 0.7, b1: -0.8, a2: -0.4, b2: 0.6},
            {a0: 0.0, a1: 1.0, b1: 0.0, a2: -1.0, b2: 0.0},
            {a0: 0.3, a1: -0.5, b1: 0.5, a2: 0.9, b2: -0.2},
            {a0: -0.2, a1: 0.4, b1: -0.9, a2: -0.8, b2: 0.7},
            {a0: 0.4, a1: 0.6, b1: 0.2, a2: -0.3, b2: -0.5},
        ]

        for targets in target_sets:
            self.play(
                *[tracker.animate.set_value(v) for tracker, v in targets.items()],
                run_time=2.0,
                rate_func=smooth,
            )
            self.wait(0.2)

        # Stay visible at the end
        self.wait(2.0)
