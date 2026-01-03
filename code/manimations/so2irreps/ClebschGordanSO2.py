# pyright: reportMissingImports=false
import manim as mn


MATRIX_WIDTH_FRACTION = 0.92
MATRIX_ADDITIONAL_SCALE = 0.75  # Shrink matrix to 75% of previous size
MATRIX_VERTICAL_SHIFT = -1.6 * mn.DOWN

EQUATION_TEXT_BASE_SCALE = 0.52
EQUATION_BLOCK_SCALE = 0.75  # Shrink spin-sum block to 75% of previous size
EQUATION_BUFF_FROM_RULE = 0.3
EQUATION_HORIZONTAL_SHIFT = 3.4 * mn.LEFT  # Move equation block laterally
EQUATION_VERTICAL_SHIFT = -0.6 * mn.DOWN       # Optional vertical nudging for equations
EQUATION_APPEAR_SHIFT = -0.5 * mn.DOWN
RULE_TEXT_HORIZONTAL_SHIFT = 3.5 * mn.LEFT
RULE_TEXT_VERTICAL_SHIFT = mn.ORIGIN + 0.8 * mn.DOWN



class ClebschGordanSO2(mn.Scene):
    """
    Visualize the tensor-product matrix between physical fields and SO(2)
    filter modes. The animation highlights, spin-by-spin, the entries that
    survive the convolution integral according to the CG rule
    m_out = m_field + m_filter.
    """

    def construct(self):
        row_fields = [
            {"symbol": r"\sigma^{(-2)}", "spin": -2},
            {"symbol": r"u^{(-1)}", "spin": -1},
            {"symbol": r"T", "spin": 0},
            {"symbol": r"\sigma^{(0)}", "spin": 0},
            {"symbol": r"u^{(+1)}", "spin": +1},
            {"symbol": r"\sigma^{(+2)}", "spin": +2},
        ]

        col_filters = [
            {"symbol": r"F^{(-2)}_1", "spin": -2},            
            {"symbol": r"F^{(-2)}_2", "spin": -2},
            {"symbol": r"F^{(-1)}_1", "spin": -1},
            {"symbol": r"F^{(0)}_1", "spin": 0},
            {"symbol": r"F^{(+1)}_1", "spin": +1},
            {"symbol": r"F^{(+1)}_2", "spin": +1},
            {"symbol": r"F^{(+2)}_1", "spin": +2},
        ]

        cell_scale = 0.55
        cells, cell_lookup = self._build_cell_grid(row_fields, col_filters, cell_scale)

        row_labels_block = self._make_row_spin_labels(row_fields, cell_lookup)
        col_labels_block = self._make_col_spin_labels(col_filters, cell_lookup)

        matrix_group = mn.VGroup(cells, row_labels_block, col_labels_block)
        target_width = mn.config.frame_width * MATRIX_WIDTH_FRACTION
        matrix_group.scale_to_fit_width(target_width)
        matrix_group.scale(MATRIX_ADDITIONAL_SCALE)
        matrix_group.shift(MATRIX_VERTICAL_SHIFT)

        rule_text = mn.MathTex(r"m_i + m_f = m_o").scale(0.7)
        rule_text.to_edge(mn.RIGHT)
        rule_text.shift(RULE_TEXT_HORIZONTAL_SHIFT + RULE_TEXT_VERTICAL_SHIFT)
        self.play(mn.Write(rule_text), run_time=0.5)
        self.play(mn.FadeIn(matrix_group, shift=mn.DOWN * 0.2), run_time=0.5)

        combos_by_spin = self._compute_spin_combos(row_fields, col_filters)
        focus_spins = list(range(-4, 5))
        spin_colors = {
            -4: mn.TEAL_E,
            -3: mn.BLUE_E,
            -2: mn.BLUE_D,
            -1: mn.BLUE_B,
            0: mn.YELLOW,
            1: mn.ORANGE,
            2: mn.RED,
            3: mn.MAROON_B,
            4: mn.MAROON_A,
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
            equations.scale(EQUATION_BLOCK_SCALE)
            equations.next_to(rule_text, mn.DOWN, buff=EQUATION_BUFF_FROM_RULE)
            equations.shift(EQUATION_HORIZONTAL_SHIFT + EQUATION_VERTICAL_SHIFT)
            if equation_display is None:
                equation_display = equations
                self.play(mn.FadeIn(equation_display, shift=EQUATION_APPEAR_SHIFT), run_time=0.25)
            else:
                self.play(mn.Transform(equation_display, equations), run_time=0.25)

            highlight_rects = mn.VGroup(
                *[
                    mn.SurroundingRectangle(
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
                mn.LaggedStart(*[mn.Create(rect) for rect in highlight_rects], lag_ratio=0.08),
                run_time=0.4,
            )
            self.play(
                mn.LaggedStart(
                    *[cell.animate.set_color(spin_colors[target_spin]) for cell in colored_cells],
                    lag_ratio=0.05,
                ),
                run_time=0.3,
            )
            self.wait(0.2)
            self.play(
                mn.LaggedStart(
                    *[cell.animate.set_color(mn.WHITE) for cell in colored_cells],
                    lag_ratio=0.05,
                ),
                mn.FadeOut(highlight_rects),
                run_time=0.4,
            )

        if equation_display is not None:
            self.play(mn.FadeOut(equation_display, shift=EQUATION_APPEAR_SHIFT), run_time=0.2)

        final_rects = mn.VGroup()
        final_color_anims = []
        for target_spin, combos in highlight_history:
            color = spin_colors[target_spin]
            for (r, c) in combos:
                cell = cell_lookup[(r, c)]
                final_rects.add(
                    mn.SurroundingRectangle(
                        cell,
                        buff=0.12,
                        color=color,
                        corner_radius=0.05,
                        stroke_width=4,
                    )
                )
                final_color_anims.append(cell.animate.set_color(color))

        self.play(
            mn.AnimationGroup(
                *final_color_anims,
                *[mn.Create(rect) for rect in final_rects],
                lag_ratio=0,
            ),
            run_time=0.6,
        )
        self.wait(1.0)

    def _build_cell_grid(self, row_fields, col_filters, scale_factor):
        """Create MathTex entries laid out in a grid with lookup for highlighting."""
        rows = len(row_fields)
        cols = len(col_filters)
        cell_lookup = {}
        flat_cells = []

        for r_idx, row in enumerate(row_fields):
            for c_idx, col in enumerate(col_filters):
                entry = mn.MathTex(
                    rf"{row['symbol']}\,{col['symbol']}",
                    color=mn.WHITE,
                ).scale(scale_factor)
                cell_lookup[(r_idx, c_idx)] = entry
                flat_cells.append(entry)

        grid = mn.VGroup(*flat_cells).arrange_in_grid(
            rows=rows,
            cols=cols,
            buff=(0.35, 0.3),
        )
        return grid, cell_lookup

    def _make_row_spin_labels(self, row_fields, cell_lookup):
        labels = mn.VGroup()
        for r_idx, row in enumerate(row_fields):
            label = mn.MathTex(
                rf"m_i = {self._fmt_spin(row['spin'], include_plus=False)}"
            ).scale(0.55)
            label.next_to(cell_lookup[(r_idx, 0)], mn.LEFT, buff=0.45)
            labels.add(label)
        return labels

    def _make_col_spin_labels(self, col_filters, cell_lookup):
        labels = mn.VGroup()
        for c_idx, col in enumerate(col_filters):
            label = mn.MathTex(
                rf"m_f = {self._fmt_spin(col['spin'], include_plus=False)}"
            ).scale(0.55)
            label.next_to(cell_lookup[(0, c_idx)], mn.UP, buff=0.35)
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
        eq_lines = mn.VGroup()
        for r_idx, c_idx in combos:
            row_spin = row_fields[r_idx]["spin"]
            col_spin = col_filters[c_idx]["spin"]
            row_sym = row_fields[r_idx]["symbol"]
            col_sym = col_filters[c_idx]["symbol"]
            eq = mn.MathTex(
                rf"{row_sym}\,(m_i={self._fmt_spin(row_spin)}) + "
                rf"{col_sym}\,(m_f={self._fmt_spin(col_spin)})"
                rf" = k={self._fmt_spin(target_spin, include_plus=False)}",
                color=color,
            ).scale(EQUATION_TEXT_BASE_SCALE)
            eq_lines.add(eq)

        eq_lines.arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.12)
        return eq_lines
