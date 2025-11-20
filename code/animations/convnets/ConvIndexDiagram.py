from manim import *
import numpy as np

class ConvIndexDiagram(Scene):
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
        red_size   = (3, 5)

        # Relative offsets (in grid cells)
        # green: 2 right, 1 up from white
        origin_green = origin_white + np.array([1 * cell, 2 * cell, 0.0])
        # red: 1 left, 2 up from white
        origin_red   = origin_white + np.array([2 * cell, 1 * cell, 0.0])

        # ------------------------------------------------------------------
        # Make grids
        # ------------------------------------------------------------------
        white_grid = make_grid(*white_size, color=WHITE, stroke_width=4)
        white_grid.shift(origin_white)

        green_grid = make_grid(*green_size, color=GREEN_C, stroke_width=12)
        green_grid.shift(origin_green)

        red_grid = make_grid(*red_size, color=RED, stroke_width=4)
        red_grid.shift(origin_red)

        # ------------------------------------------------------------------
        # Highlight central cell inside the red grid (u=1, v=2)
        # ------------------------------------------------------------------
        center_square = Square(
            side_length=cell,
            stroke_width=0,
            fill_color=TEAL,
            fill_opacity=1.0,
        )
        center_x = origin_red[0] + (1 + 0.5) * cell   # u = 1
        center_y = origin_red[1] + (2 + 0.5) * cell   # v = 2
        center_square.move_to(np.array([center_x, center_y, 0.0]))

        # ------------------------------------------------------------------
        # Index labels
        # ------------------------------------------------------------------
        # j' labels (white grid, vertical)
        jprime_labels = VGroup()
        x_jprime = origin_white[0] - 0.8 * cell
        for j in range(white_size[1]):
            y = origin_white[1] + (j + 0.5) * cell
            lab = MathTex(rf"j'={j}", color=WHITE)
            lab.scale(0.4)
            lab.move_to(np.array([x_jprime, y, 0.0]))
            jprime_labels.add(lab)

        # i' labels (white grid, horizontal)
        iprime_labels = VGroup()
        y_iprime = origin_white[1] - 0.6 * cell
        for i in range(white_size[0]):
            x = origin_white[0] + (i + 0.5) * cell
            lab = MathTex(rf"i'={i}", color=WHITE)
            lab.scale(0.4)
            lab.rotate(90 * DEGREES)
            lab.move_to(np.array([x, y_iprime - 0.2 * cell, 0.0]))
            iprime_labels.add(lab)

        # j labels (green grid, vertical)
        j_labels = VGroup()
        x_j = origin_white[0] - 2.6 * cell
        for j in range(green_size[1]):
            y = origin_green[1] + (j + 0.5) * cell
            lab = MathTex(rf"j={j}", color=GREEN_B)
            lab.scale(0.4)
            lab.move_to(np.array([x_j, y, 0.0]))
            j_labels.add(lab)

        # v labels (red grid, vertical)
        v_labels = VGroup()
        x_v = origin_white[0] - 4.6 * cell
        for v in range(red_size[1]):
            y = origin_red[1] + (v + 0.5) * cell
            lab = MathTex(rf"v={v}", color=RED)
            lab.scale(0.4)
            lab.move_to(np.array([x_v, y, 0.0]))
            v_labels.add(lab)

        # i labels (green grid, horizontal)
        i_labels = VGroup()
        y_i = y_iprime - 0.7 * cell
        for i in range(green_size[0]):
            x = origin_green[0] + (i + 0.5) * cell
            lab = MathTex(rf"i={i}", color=GREEN_B)
            lab.scale(0.4)
            lab.rotate(90 * DEGREES)
            lab.move_to(np.array([x, y_i - 0.8 * cell, 0.0]))
            i_labels.add(lab)

        # u labels (red grid, horizontal)
        u_labels = VGroup()
        y_u = y_i - 0.7 * cell
        for u in range(red_size[0]):
            x = origin_red[0] + (u + 0.5) * cell
            lab = MathTex(rf"u={u}", color=RED)
            lab.scale(0.4)
            lab.rotate(90 * DEGREES)
            lab.move_to(np.array([x, y_u - 1.6 * cell, 0.0]))
            u_labels.add(lab)

        # ------------------------------------------------------------------
        # Add everything to the scene (order matters for layering)
        # ------------------------------------------------------------------
        self.add(
            green_grid,
            white_grid,
            red_grid,
            center_square,
            jprime_labels,
            iprime_labels,
            j_labels,
            v_labels,
            i_labels,
            u_labels,
        )
