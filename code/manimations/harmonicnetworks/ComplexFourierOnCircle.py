from manim import *
from manim.utils.color import ManimColor
import numpy as np


class ComplexFourierOnCircle(Scene):
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
        # 2. Coefficient trackers for complex modes m = -2..2
        #    c_m = a_m + i b_m
        # ------------------------------------------------------------
        # Naming: a_mneg2 = Re(c_{-2}), etc.
        a_mneg2 = ValueTracker(0.0)
        b_mneg2 = ValueTracker(0.0)

        a_mneg1 = ValueTracker(0.0)
        b_mneg1 = ValueTracker(0.0)

        a_m0 = ValueTracker(0.0)
        b_m0 = ValueTracker(0.0)

        a_m1 = ValueTracker(0.0)
        b_m1 = ValueTracker(0.0)

        a_m2 = ValueTracker(0.0)
        b_m2 = ValueTracker(0.0)

        # For convenience, collect them in a list
        complex_modes = [
            (-2, a_mneg2, b_mneg2),
            (-1, a_mneg1, b_mneg1),
            ( 0, a_m0,    b_m0   ),
            ( 1, a_m1,    b_m1   ),
            ( 2, a_m2,    b_m2   ),
        ]

        # ------------------------------------------------------------
        # 3. Layout positions (Three Panel Layout)
        # ------------------------------------------------------------
        # ... (keep existing code) ...
        # Right Panel (25%): Coefficients
        
        SCARLET = ManimColor("#ff2400")

        # Vertical positions for the 5 modes (m = -2 to 2)
        # Spreading them vertically from top to bottom
        y_spacing = 1.5
        y_start = 3.0
        y_positions = [y_start - i * y_spacing for i in range(5)]
        
        small_radius = 0.6
        big_radius = 1.6

        # X coordinates
        # Left Panel: Real parts
        col_real_x = -6.0
        big_real_x = -3.2
        
        # Centre Panel: Imaginary parts
        # Shifted left by big_radius (1.6)
        big_imag_x = -0.6 - big_radius      # -2.2
        col_imag_x = 2.2 - big_radius       # 0.6
        
        col_coeff_x = 5.5      # Column of coefficients
        
        # Centers for big circles
        # Real moved UP (y=1.5), Imag moved DOWN (y=-2.5)
        center_big_real = np.array([big_real_x, 1.5, 0.0])
        center_big_imag = np.array([big_imag_x, -2.3, 0.0])

        # sample angular grid
        N = 360
        thetas = np.linspace(0, TAU, N)

        # ------------------------------------------------------------
        # 3.5 Helper: Axis generator
        # ------------------------------------------------------------
        def make_axes(center, radius):
            axis_len = 1.2 * radius
            neg_stub = 0.25 * radius

            x_axis = Line(
                center + LEFT * neg_stub,
                center + RIGHT * axis_len,
                color=SCARLET,
                stroke_width=2,
            )
            y_axis = Line(
                center + DOWN * neg_stub,
                center + UP * axis_len,
                color=SCARLET,
                stroke_width=2,
            )

            sigma_label = MathTex(r"x'", color=SCARLET).scale(0.4)
            tau_label   = MathTex(r"y'",   color=SCARLET).scale(0.4)

            sigma_label.next_to(x_axis.get_end(), RIGHT, buff=0.05)
            tau_label.next_to(y_axis.get_end(), UP, buff=0.05)

            return VGroup(x_axis, y_axis, sigma_label, tau_label)

        # ------------------------------------------------------------
        # 4. Helper: coloured circle for a scalar function on the circle
        # ------------------------------------------------------------
        def coloured_circle(center, radius, func):
            """
            func(theta) returns scalar in [-1,1]
            Produces a VGroup of line segments with stroke matching colour along the circle.
            """
            group = VGroup()
            for i in range(len(thetas) - 1):
                th1 = thetas[i]
                th2 = thetas[i + 1]

                mid_th = (th1 + th2) / 2.0
                val = func(mid_th)
                rgb = value_to_rgb(val)

                p1 = center + radius * np.array([np.cos(th1), np.sin(th1), 0.0])
                p2 = center + radius * np.array([np.cos(th2), np.sin(th2), 0.0])

                seg = Line(p1, p2, stroke_width=5)
                seg.set_color(ManimColor(rgb))
                group.add(seg)

            return group

        # ------------------------------------------------------------
        # 5. Basis functions for complex modes m = -2..2
        #    For plotting we split Re and Im parts.
        # ------------------------------------------------------------
        # Fixed complex basis: exp(i m θ)
        def complex_basis(m, th):
            return np.exp(1j * m * th)

        basis_mobjects = []
        basis_labels = []
        basis_axes = []

        # For each m, make two circles: Re(e^{i m θ}) and Im(e^{i m θ})
        for (idx, (m, _, _)) in enumerate(complex_modes):
            y = y_positions[idx]

            # Real part (Left Panel)
            center_re = np.array([col_real_x, y, 0.0])
            ring_re = coloured_circle(
                center=center_re,
                radius=small_radius,
                func=lambda th, m=m: np.real(complex_basis(m, th)),
            )
            # Axes for small circle
            axes_re = make_axes(center_re, small_radius)
            basis_axes.append(axes_re)

            # Label to the left of the circle to save space? Or tiny below?
            # Let's put it tiny below as before but scale it down
            label_re = MathTex(r"\Re\left(e^{i \left(" + f"{m}" + r"\theta'\right)}\right)").scale(0.5)
            label_re.next_to(center_re, DOWN, buff=0.1 + small_radius*0.2) # Adjust buff to clear axis

            basis_mobjects.append(ring_re)
            basis_labels.append(label_re)

            # Imaginary part (Centre Panel)
            center_im = np.array([col_imag_x, y, 0.0])
            ring_im = coloured_circle(
                center=center_im,
                radius=small_radius,
                func=lambda th, m=m: np.imag(complex_basis(m, th)),
            )
            # Axes for small circle
            axes_im = make_axes(center_im, small_radius)
            basis_axes.append(axes_im)

            label_im = MathTex(r"\Im\left(e^{i \left(" + f"{m}" + r"\theta'\right)}\right)").scale(0.5)
            label_im.next_to(center_im, DOWN, buff=0.1 + small_radius*0.2)

            basis_mobjects.append(ring_im)
            basis_labels.append(label_im)

        # ------------------------------------------------------------
        # 6. Full complex function on the circle
        #    f(θ) = Σ_{m=-2}^{2} c_m e^{i m θ},  c_m = a_m + i b_m
        # ------------------------------------------------------------
        def full_complex_function(th):
            val = 0.0 + 0.0j
            for (m, a_tr, b_tr) in complex_modes:
                c_m = a_tr.get_value() + 1j * b_tr.get_value()
                val += c_m * np.exp(1j * m * th)
            return val

        # Real part of f(θ)
        main_circle_real = always_redraw(
            lambda: coloured_circle(
                center=center_big_real,
                radius=big_radius,
                func=lambda th: np.real(full_complex_function(th)),
            )
        )
        main_axes_real = make_axes(center_big_real, big_radius)

        # Imaginary part of f(θ)
        main_circle_imag = always_redraw(
            lambda: coloured_circle(
                center=center_big_imag,
                radius=big_radius,
                func=lambda th: np.imag(full_complex_function(th)),
            )
        )
        main_axes_imag = make_axes(center_big_imag, big_radius)

        # Labels for the big circles
        label_f_real = MathTex(r"\Re\big(\Theta(\theta')\big)").scale(0.7)
        label_f_real.next_to(center_big_real, DOWN, buff=0.25 + big_radius*0.1)

        label_f_imag = MathTex(r"\Im\big(\Theta(\theta')\big)").scale(0.7)
        label_f_imag.next_to(center_big_imag, DOWN, buff=0.25 + big_radius*0.1)

        # ------------------------------------------------------------
        # 7. Coefficient numeric labels for c_m = a_m + i b_m
        # ------------------------------------------------------------
        # Anchor positions for coefficient labels (right side of screen)
        # Contained in upper right corner (width 25%, height 25%)
        # Frame width ~14 (x from -7 to 7), height ~8 (y from -4 to 4)
        # Right 25%: x in [3.5, 7.0]. Center approx 5.25.
        # Upper 25%: y in [2.0, 4.0].
        
        # Let's tighten the spacing significantly
        y_start_coeff = 2.8
        y_spacing_coeff = 0.2  # Tighter spacing
        coeff_x_pos = 5.25     # Centered in the right 25% panel
        
        coeff_positions = [
            np.array([coeff_x_pos, y_start_coeff - i * y_spacing_coeff, 0.0]) for i in range(5)
        ]

        def coeff_label_complex(m, a_tracker, b_tracker, position):
            """
            Show c_m = a_m + i b_m with live numeric values.
            """
            def make_tex():
                a_val = a_tracker.get_value()
                b_val = b_tracker.get_value()
                return MathTex(
                    fr"c_{{{m}}} = "
                    fr"{a_val:+.2f}"
                    r" + i\,"
                    fr"({b_val:+.2f})"
                ).scale(0.5) # Reduced scale

            label = make_tex()
            label.move_to(position)

            def updater(mob):
                new = make_tex()
                new.move_to(position)
                mob.become(new)

            label.add_updater(updater)
            return label

        coeff_label_group = VGroup(
            coeff_label_complex(-2, a_mneg2, b_mneg2, coeff_positions[0]),
            coeff_label_complex(-1, a_mneg1, b_mneg1, coeff_positions[1]),
            coeff_label_complex( 0, a_m0,    b_m0,    coeff_positions[2]),
            coeff_label_complex( 1, a_m1,    b_m1,    coeff_positions[3]),
            coeff_label_complex( 2, a_m2,    b_m2,    coeff_positions[4]),
        )

        # ------------------------------------------------------------
        # 8. Analytic expression for f(θ)
        # ------------------------------------------------------------
        full_expr = MathTex(
            r"\Theta(\theta') \;=\; \sum_{m=-2}^{2} c_m\,e^{i \left(m \theta'\right)}"
        ).scale(0.6) # Reduced scale
        
        # Position above the coefficients in the same panel
        full_expr.move_to(np.array([coeff_x_pos, 3.5, 0.0]))

        # ------------------------------------------------------------
        # 9. Draw everything
        # ------------------------------------------------------------
        self.play(
            *[FadeIn(m) for m in basis_mobjects],
            *[FadeIn(ax) for ax in basis_axes],
            run_time=1.5
        )
        self.play(*[FadeIn(lbl) for lbl in basis_labels], run_time=1.0)

        self.play(
            FadeIn(main_circle_real),
            FadeIn(main_axes_real),
            FadeIn(main_circle_imag),
            FadeIn(main_axes_imag),
            FadeIn(label_f_real),
            FadeIn(label_f_imag),
            run_time=1.5,
        )

        self.play(
            FadeIn(coeff_label_group),
            FadeIn(full_expr),
            run_time=1.5,
        )
        self.wait(1.0)

        # ------------------------------------------------------------
        # 10. Animate coefficients (arbitrary demo values)
        # ------------------------------------------------------------
        targets_1 = {
            a_mneg2:  0.4, b_mneg2: -0.3,
            a_mneg1: -0.7, b_mneg1:  0.5,
            a_m0:     0.6, b_m0:    -0.2,
            a_m1:    -0.5, b_m1:     0.8,
            a_m2:     0.9, b_m2:    -0.6,
        }

        targets_2 = {
            a_mneg2: -0.6, b_mneg2:  0.2,
            a_mneg1:  0.8, b_mneg1: -0.4,
            a_m0:    -0.3, b_m0:     0.7,
            a_m1:     0.1, b_m1:    -0.9,
            a_m2:     0.3, b_m2:     0.5,
        }

        targets_3 = {
            a_mneg2:  0.2, b_mneg2:  0.2,
            a_mneg1: -0.3, b_mneg1: -0.3,
            a_m0:     0.0, b_m0:     0.0,
            a_m1:     0.4, b_m1:     0.1,
            a_m2:    -0.4, b_m2:     0.4,
        }

        self.play(
            *[tracker.animate.set_value(v) for tracker, v in targets_1.items()],
            run_time=3.0,
            rate_func=smooth,
        )
        self.wait(0.5)

        self.play(
            *[tracker.animate.set_value(v) for tracker, v in targets_2.items()],
            run_time=3.0,
            rate_func=smooth,
        )
        self.wait(0.5)

        self.play(
            *[tracker.animate.set_value(v) for tracker, v in targets_3.items()],
            run_time=3.0,
            rate_func=smooth,
        )
        self.wait(1.0)

        # ------------------------------------------------------------
        # 11. Fade out
        # ------------------------------------------------------------
        self.play(
            FadeOut(
                VGroup(
                    *basis_mobjects,
                    *basis_labels,
                    *basis_axes,
                    main_circle_real,
                    main_axes_real,
                    main_circle_imag,
                    main_axes_imag,
                    label_f_real,
                    label_f_imag,
                    coeff_label_group,
                    full_expr,
                )
            ),
            run_time=1.5,
        )
        self.wait(0.5)
