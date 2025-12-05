from manim import *


class ClebschGordanSO2(Scene):
    """
    Visualize the tensor-product matrix between physical fields and SO(2)
    filter modes. The animation highlights, spin-by-spin, the entries that
    survive the convolution integral according to the CG rule
    m_out = m_field + m_filter.
    """

    def construct(self):
        row_fields = [
            {"symbol": r"T_{(r,\theta)}", "spin": 0},
            {"symbol": r"\sigma^{(0)}_{(r,\theta)}", "spin": 0},
            {"symbol": r"u^{(+1)}_{(r,\theta)}", "spin": +1},
            {"symbol": r"u^{(-1)}_{(r,\theta)}", "spin": -1},
            {"symbol": r"\sigma^{(+2)}_{(r,\theta)}", "spin": +2},
            {"symbol": r"\sigma^{(-2)}_{(r,\theta)}", "spin": -2},
        ]

        col_filters = [
            {"symbol": r"F^{(-2)}_1", "spin": -2},
            {"symbol": r"F^{(-1)}_1", "spin": -1},
            {"symbol": r"F^{(0)}_1", "spin": 0},
            {"symbol": r"F^{(+1)}_1", "spin": +1},
            {"symbol": r"F^{(+2)}_1", "spin": +2},
        ]

        cell_scale = 0.55
        cells, cell_lookup = self._build_cell_grid(row_fields, col_filters, cell_scale)

        row_labels_block = self._make_row_spin_labels(row_fields, cell_lookup)
        col_labels_block = self._make_col_spin_labels(col_filters, cell_lookup)

        matrix_group = VGroup(cells, row_labels_block, col_labels_block)
        target_width = config.frame_width * 0.8
        if matrix_group.width > target_width:
            matrix_group.scale_to_fit_width(target_width)
        matrix_group.shift(0.2 * DOWN)

        rule_text = MathTex(r"m+n=k").scale(0.7)
        rule_text.to_edge(RIGHT)
        self.play(Write(rule_text))
        self.play(FadeIn(matrix_group, shift=DOWN * 0.2))

        combos_by_spin = self._compute_spin_combos(row_fields, col_filters)
        focus_spins = list(range(-4, 5))
        spin_colors = {
            -4: TEAL_E,
            -3: BLUE_E,
            -2: BLUE_D,
            -1: BLUE_B,
            0: YELLOW,
            1: ORANGE,
            2: RED,
            3: MAROON_B,
            4: MAROON_A,
        }

        highlight_history = []
        equation_display = None

        for target_spin in focus_spins:
            combos = combos_by_spin.get(target_spin, [])
            if not combos:
                continue
            highlight_history.append((target_spin, combos))

            equations = self._build_equation_block(
                combos, row_fields, col_filters, target_spin, spin_colors[target_spin]
            )
            equations.next_to(rule_text, DOWN, buff=0.3)
            if equation_display is None:
                equation_display = equations
                self.play(FadeIn(equation_display, shift=0.1 * DOWN), run_time=0.5)
            else:
                self.play(Transform(equation_display, equations), run_time=0.5)

            highlight_rects = VGroup(
                *[
                    SurroundingRectangle(
                        cell_lookup[(r, c)],
                        buff=0.12,
                        color=spin_colors[target_spin],
                        corner_radius=0.05,
                        stroke_width=4,
                    )
                    for (r, c) in combos
                ]
            )

            colored_cells = [cell_lookup[(r, c)] for (r, c) in combos]

            self.play(
                LaggedStart(*[Create(rect) for rect in highlight_rects], lag_ratio=0.08),
                run_time=0.8,
            )
            self.play(
                LaggedStart(
                    *[cell.animate.set_color(spin_colors[target_spin]) for cell in colored_cells],
                    lag_ratio=0.05,
                ),
                run_time=0.6,
            )
            self.wait(0.4)
            self.play(
                LaggedStart(
                    *[cell.animate.set_color(WHITE) for cell in colored_cells],
                    lag_ratio=0.05,
                ),
                FadeOut(highlight_rects),
                run_time=0.8,
            )

        if equation_display is not None:
            self.play(FadeOut(equation_display, shift=0.1 * DOWN), run_time=0.4)

        final_rects = VGroup()
        final_color_anims = []
        for target_spin, combos in highlight_history:
            color = spin_colors[target_spin]
            for (r, c) in combos:
                cell = cell_lookup[(r, c)]
                final_rects.add(
                    SurroundingRectangle(
                        cell,
                        buff=0.12,
                        color=color,
                        corner_radius=0.05,
                        stroke_width=4,
                    )
                )
                final_color_anims.append(cell.animate.set_color(color))

        self.play(
            AnimationGroup(
                *final_color_anims,
                *[Create(rect) for rect in final_rects],
                lag_ratio=0,
            ),
            run_time=1.2,
        )
        self.wait(2.0)

    def _build_cell_grid(self, row_fields, col_filters, scale_factor):
        """Create MathTex entries laid out in a grid with lookup for highlighting."""
        rows = len(row_fields)
        cols = len(col_filters)
        cell_lookup = {}
        flat_cells = []

        for r_idx, row in enumerate(row_fields):
            for c_idx, col in enumerate(col_filters):
                entry = MathTex(
                    rf"{row['symbol']}\,{col['symbol']}",
                    color=WHITE,
                ).scale(scale_factor)
                cell_lookup[(r_idx, c_idx)] = entry
                flat_cells.append(entry)

        grid = VGroup(*flat_cells).arrange_in_grid(
            rows=rows,
            cols=cols,
            buff=(0.35, 0.3),
        )
        return grid, cell_lookup

    def _make_row_spin_labels(self, row_fields, cell_lookup):
        labels = VGroup()
        for r_idx, row in enumerate(row_fields):
            label = MathTex(
                rf"n = {self._fmt_spin(row['spin'], include_plus=False)}"
            ).scale(0.55)
            label.next_to(cell_lookup[(r_idx, 0)], LEFT, buff=0.45)
            labels.add(label)
        return labels

    def _make_col_spin_labels(self, col_filters, cell_lookup):
        labels = VGroup()
        for c_idx, col in enumerate(col_filters):
            label = MathTex(
                rf"m = {self._fmt_spin(col['spin'], include_plus=False)}"
            ).scale(0.55)
            label.next_to(cell_lookup[(0, c_idx)], UP, buff=0.35)
            labels.add(label)
        return labels

    def _compute_spin_combos(self, row_fields, col_filters):
        combos = {}
        for r_idx, row in enumerate(row_fields):
            for c_idx, col in enumerate(col_filters):
                total = row["spin"] + col["spin"]
                combos.setdefault(total, []).append((r_idx, c_idx))
        return combos

    def _fmt_spin(self, value, include_plus=True):
        if value > 0:
            return f"+{value}" if include_plus else f"{value}"
        if value == 0:
            return "0"
        return f"{value}"

    def _build_equation_block(self, combos, row_fields, col_filters, target_spin, color):
        eq_lines = VGroup()
        for r_idx, c_idx in combos:
            row_spin = row_fields[r_idx]["spin"]
            col_spin = col_filters[c_idx]["spin"]
            eq = MathTex(
                rf"{self._fmt_spin(row_spin)} + ({self._fmt_spin(col_spin)})"
                rf" = {self._fmt_spin(target_spin, include_plus=False)}",
                color=color,
            ).scale(0.6)
            eq_lines.add(eq)

        eq_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        return eq_lines
