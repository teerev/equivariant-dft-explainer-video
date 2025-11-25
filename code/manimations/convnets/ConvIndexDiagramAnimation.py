from manim import *
import numpy as np

class ConvIndexDiagramAnimation(Scene):
    def construct(self):
        cell = 0.4  # side length of each square

        # ------------------------------------------------------------------
        # Helper to build a rectangular grid with given number of cols/rows.
        # Bottom-left corner initially at (0, 0); we'll shift later.
        # ------------------------------------------------------------------
        def make_grid(cols, rows, color=WHITE, stroke_width=2):
            lines = VGroup()
            width = cols * cell
            height = rows * cell

            # Vertical lines
            for i in range(cols + 1):
                x = i * cell
                lines.add(
                    Line(
                        np.array([x, 0, 0]),
                        np.array([x, height, 0]),
                        stroke_color=color,
                        stroke_width=stroke_width,
                    )
                )

            # Horizontal lines
            for j in range(rows + 1):
                y = j * cell
                lines.add(
                    Line(
                        np.array([0, y, 0]),
                        np.array([width, y, 0]),
                        stroke_color=color,
                        stroke_width=stroke_width,
                    )
                )

            return lines

        # ------------------------------------------------------------------
        # Base positions
        # ------------------------------------------------------------------
        # Bottom-left of the WHITE (input) grid
        origin_white = np.array([-2.0, -1.0, 0.0])

        # Grid sizes: (cols, rows)
        white_size = (9, 8)
        green_size = (7, 4)
        red_size = (3, 5)

        # Relative offsets (in grid cells)
        # green: 2 right, 1 up from white
        origin_green = origin_white + np.array([1 * cell, 2 * cell, 0.0])
        # red starts aligned with the input grid (bottom-left corners coincide)
        origin_red = origin_white.copy()

        # ------------------------------------------------------------------
        # Make grids
        # ------------------------------------------------------------------
        white_grid = make_grid(*white_size, color=WHITE, stroke_width=4)
        white_grid.shift(origin_white)

        green_grid = make_grid(*green_size, color=GREEN_C, stroke_width=12)
        green_grid.shift(origin_green)

        red_grid = make_grid(*red_size, color=RED, stroke_width=4)
        red_grid.shift(origin_red)

        red_group = VGroup(red_grid)

        # ------------------------------------------------------------------
        # Index labels
        # ------------------------------------------------------------------
        # j labels (white grid, vertical)
        j_labels = VGroup()
        x_j = origin_white[0] - 0.8 * cell
        for j in range(white_size[1]):
            y = origin_white[1] + (j + 0.5) * cell
            lab = MathTex(rf"j={j}", color=WHITE)
            lab.scale(0.4)
            lab.move_to(np.array([x_j, y, 0.0]))
            j_labels.add(lab)

        # i labels (white grid, horizontal)
        i_labels = VGroup()
        y_i_base = origin_white[1] - 0.6 * cell
        for i in range(white_size[0]):
            x = origin_white[0] + (i + 0.5) * cell
            lab = MathTex(rf"i={i}", color=WHITE)
            lab.scale(0.4)
            lab.rotate(90 * DEGREES)
            lab.move_to(np.array([x, y_i_base - 0.2 * cell, 0.0]))
            i_labels.add(lab)

        # v labels (red grid, vertical)
        v_labels = VGroup()
        x_v = origin_white[0] - 2.3 * cell
        v_values = np.arange(-(red_size[1] // 2), red_size[1] // 2 + 1)
        for idx, v_val in enumerate(v_values):
            y = origin_red[1] + (idx + 0.5) * cell
            lab = MathTex(rf"v={v_val}", color=RED)
            lab.scale(0.4)
            lab.move_to(np.array([x_v, y, 0.0]))
            v_labels.add(lab)

        # u labels (red grid, horizontal)
        u_labels = VGroup()
        y_u = y_i_base# - 0.7 * cell
        u_values = np.arange(-(red_size[0] // 2), red_size[0] // 2 + 1)
        for idx, u_val in enumerate(u_values):
            x = origin_red[0] + (idx + 0.5) * cell
            lab = MathTex(rf"u={u_val}", color=RED)
            lab.scale(0.4)
            lab.rotate(90 * DEGREES)
            lab.move_to(np.array([x, y_u - 1.6 * cell, 0.0]))
            u_labels.add(lab)

        # ------------------------------------------------------------------
        # Add everything to the scene (order matters for layering)
        # ------------------------------------------------------------------
        teal_square = Square(
            side_length=cell,
            fill_color=TEAL,
            fill_opacity=1.0,
            stroke_width=0,
        )
        teal_center_x = origin_red[0] + (1 + 0.5) * cell
        teal_center_y = origin_red[1] + (2 + 0.5) * cell
        teal_square.move_to(np.array([teal_center_x, teal_center_y, 0.0]))

        self.add(
            green_grid,
            white_grid,
            red_group,
            j_labels,
            i_labels,
            v_labels,
            u_labels,
            teal_square,
        )

        move_sequence = [RIGHT, UP, RIGHT, UP, LEFT]
        for direction in move_sequence:
            shift_vec = cell * direction
            self.play(FadeOut(teal_square), run_time=0.2)

            animations = [red_group.animate.shift(shift_vec)]
            if np.array_equal(direction, RIGHT) or np.array_equal(direction, LEFT):
                animations.append(u_labels.animate.shift(shift_vec))
            if np.array_equal(direction, UP) or np.array_equal(direction, DOWN):
                animations.append(v_labels.animate.shift(shift_vec))
            self.play(*animations, run_time=2, rate_func=smooth)
            teal_square.shift(shift_vec)
            self.play(FadeIn(teal_square, run_time=0.2))

        self.wait(0.5)