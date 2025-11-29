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
        x_positions = [-4.0, -1.3, 1.3, 4.0]
        small_radius = 0.9
        big_radius = 2.8
        center_big = np.array([0.0, -1.0, 0.0])

        # sample angular grid
        N = 360
        thetas = np.linspace(0, TAU, N)

        # ------------------------------------------------------------
        # 4. Helper: get coloured ring points for a basis function
        # ------------------------------------------------------------
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
            r"\cos\theta",
            r"\sin\theta",
            r"\cos 2\theta",
            r"\sin 2\theta",
        ]

        basis_mobjects = []
        basis_labels = []

        for i, func in enumerate(basis_funcs):
            x = x_positions[i]

            # Static rings for basis functions
            ring = coloured_circle(
                center=np.array([x, top_y, 0.0]),
                radius=small_radius,
                func=func,
            )
            label = MathTex(basis_labels_tex[i]).scale(0.7)
            label.next_to(np.array([x, top_y, 0.0]), DOWN, buff=0.2)

            basis_mobjects.append(ring)
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
            coeff_label("a_0", a0, np.array([4.5, 1.5, 0.0])),
            coeff_label("a_1", a1, np.array([4.5, 0.8, 0.0])),
            coeff_label("b_1", b1, np.array([4.5, 0.1, 0.0])),
            coeff_label("a_2", a2, np.array([4.5, -0.6, 0.0])),
            coeff_label("b_2", b2, np.array([4.5, -1.3, 0.0])),
        )

        # ------------------------------------------------------------
        # 8. Analytic expression for f(theta)
        # ------------------------------------------------------------
        full_expr = MathTex(
            r"f(\theta) = a_0"
            r" + a_1 \cos\theta"
            r" + b_1 \sin\theta"
            r" + a_2 \cos 2\theta"
            r" + b_2 \sin 2\theta"
        ).scale(0.8)
        full_expr.next_to(center_big, DOWN, buff=0.7)

        # ------------------------------------------------------------
        # 9. Draw everything
        # ------------------------------------------------------------
        self.play(*[FadeIn(m) for m in basis_mobjects])
        self.play(*[FadeIn(lbl) for lbl in basis_labels])
        self.play(FadeIn(main_coloured_circle))
        self.play(FadeIn(coeff_group), FadeIn(full_expr))
        self.wait(1.0)

        # ------------------------------------------------------------
        # 10. Animate coefficients
        # ------------------------------------------------------------
        targets_1 = {a0: 0.3, a1: 0.7, b1: -0.5, a2: 0.4, b2: 0.25}
        targets_2 = {a0: -0.2, a1: -0.6, b1: 0.4, a2: -0.3, b2: 0.5}

        self.play(
            *[tracker.animate.set_value(v) for tracker, v in targets_1.items()],
            run_time=3.0,
            rate_func=smooth,
        )
        self.wait(1.0)

        self.play(
            *[tracker.animate.set_value(v) for tracker, v in targets_2.items()],
            run_time=3.0,
            rate_func=smooth,
        )
        self.wait(1.0)

        self.play(
            FadeOut(
                VGroup(
                    *basis_mobjects,
                    *basis_labels,
                    main_coloured_circle,
                    coeff_group,
                    full_expr,
                )
            ),
            run_time=1.5,
        )
        self.wait(0.5)
