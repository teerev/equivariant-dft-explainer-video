from manim import *
import numpy as np

class UniversalApproxSingleLayer(Scene):
    def construct(self):
        # ----------------------------
        # Layout tuning knobs (edit these)
        # ----------------------------
        AXES_SCALE = 0.9
        AXES_LEFT_BUFF = 0.1
        AXES_X_WIDTH_SCALE = 0.6   # < 1 shrinks only in left-right direction
        AXES_EXTRA_SHIFT = 1.2 * LEFT

        NET_POS = RIGHT * 5       # move network right/left
        NET_BRACE_DIR = LEFT      # put the N brace on LEFT to avoid overlap
        NET_BRACE_BUFF = 1.4
        NET_N_LABEL_SCALE = 0.55

        # ----------------------------
        # Target function (continuous)
        # ----------------------------
        def f(x):
            # Smooth-ish but nontrivial target
            return 0.6*np.sin(2.2*x) + 0.25*np.sin(6.0*x) + 0.15*np.cos(0.9*x)

        # Sigmoid activation (nonlinear)
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        # --------------------------------
        # Fit 1-hidden-layer sum of sigmoids
        # y ≈ sum_j a_j * sigma(b_j*(x-c_j)) + d
        # We'll fix (b_j, c_j), solve linear least squares for (a_j, d)
        # --------------------------------
        def fit_single_hidden_layer(xs, ys, N, b=4.0, seed=0):
            rng = np.random.default_rng(seed + N)
            # centers spread across domain with small jitter
            c = np.linspace(xs.min(), xs.max(), N)
            c = c + 0.04*(xs.max()-xs.min())*rng.standard_normal(N)

            # build design matrix: [sigma(b*(x-c_j)) ... , 1]
            Phi = np.zeros((len(xs), N + 1))
            for j in range(N):
                Phi[:, j] = sigmoid(b*(xs - c[j]))
            Phi[:, -1] = 1.0  # bias term

            # least squares solve
            w, *_ = np.linalg.lstsq(Phi, ys, rcond=None)
            a = w[:-1]
            d = w[-1]
            return a, c, d, b

        def make_approx_function(a, c, d, b):
            def g(x):
                z = sigmoid(b*(x - c))
                return float(np.dot(a, z) + d)
            return g

        # ----------------------------
        # Axes / curves
        # ----------------------------
        x_min, x_max = -3.0, 3.0
        axes = Axes(
            x_range=[x_min, x_max, 1],
            y_range=[-1.3, 1.3, 0.5],
            tips=False,
        ).scale(AXES_SCALE).to_edge(LEFT, buff=AXES_LEFT_BUFF)
        axes.stretch(AXES_X_WIDTH_SCALE, dim=0)
        axes.shift(AXES_EXTRA_SHIFT)

        x_label = axes.get_x_axis_label(Tex("x"), edge=RIGHT, direction=RIGHT, buff=0.2)
        y_label = axes.get_y_axis_label(Tex("y"), edge=UP, direction=UP, buff=0.2)

        target_graph = axes.plot(f, x_range=[x_min, x_max], stroke_width=4)
        target_label = Tex(r"Target $f(x)$").scale(0.6).next_to(axes, UP, buff=0.2).align_to(axes, LEFT)

        # sample points for fitting
        xs = np.linspace(x_min, x_max, 300)
        ys = np.array([f(x) for x in xs])

        # More intermediate widths + slower pacing to make this ~30s total.
        widths = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96]
        # Speed tweak: target ~25s total runtime (was ~30s)
        step_transform_time = 1.45
        step_wait_time = 0.4

        # initial approx
        a, c, d, b = fit_single_hidden_layer(xs, ys, widths[0], b=4.0, seed=2)
        g = make_approx_function(a, c, d, b)
        approx_graph = axes.plot(g, x_range=[x_min, x_max], stroke_width=4)
        approx_label = Tex(r"Approx $\hat f_N(x)$").scale(0.6).next_to(target_label, DOWN, buff=0.15).align_to(target_label, LEFT)

        # ----------------------------
        # Network diagram (ball-and-stick)
        # ----------------------------
        def network_diagram(N, pos=NET_POS):
            # 1 - N - 1
            layer_x = [0.0, 1.4, 2.8]
            input_y = [0.0]
            hidden_y = np.linspace(-2.8, 2.8, N) if N > 1 else np.array([0.0])
            output_y = [0.0]

            def nodes(x, ys, r=0.07):
                return VGroup(*[Circle(radius=r).move_to([x, y, 0]) for y in ys])

            inp = nodes(layer_x[0], input_y, r=0.09)
            hid = nodes(layer_x[1], hidden_y, r=0.06 if N <= 20 else 0.035)
            out = nodes(layer_x[2], output_y, r=0.09)

            # connections (thin; avoid clutter for large N)
            lines = VGroup()
            for h in hid:
                lines.add(Line(inp[0].get_center(), h.get_center(), stroke_width=1.5, stroke_opacity=0.7))
                lines.add(Line(h.get_center(), out[0].get_center(), stroke_width=1.5, stroke_opacity=0.7))

            net = VGroup(lines, inp, hid, out).move_to(pos)

            # Put the brace + N label on the LEFT (prevents overlap with output layer)
            brace = Brace(hid, direction=NET_BRACE_DIR, buff=NET_BRACE_BUFF)
            n_text = Tex(fr"$N={N}$").scale(NET_N_LABEL_SCALE).next_to(brace, NET_BRACE_DIR, buff=0.12)
            return VGroup(net, brace, n_text)

        net = network_diagram(widths[0])

        # ----------------------------
        # Build scene
        # ----------------------------
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.play(Create(target_graph), FadeIn(target_label))
        self.play(Create(approx_graph), FadeIn(approx_label))
        self.play(FadeIn(net))
        self.wait(0.4)

        # animate increasing width and curve refinement
        for N in widths[1:]:
            a2, c2, d2, b2 = fit_single_hidden_layer(xs, ys, N, b=4.0, seed=2)
            g2 = make_approx_function(a2, c2, d2, b2)
            new_graph = axes.plot(g2, x_range=[x_min, x_max], stroke_width=4)

            new_net = network_diagram(N)

            self.play(
                Transform(approx_graph, new_graph),
                Transform(net, new_net),
                run_time=step_transform_time
            )
            self.wait(step_wait_time)
