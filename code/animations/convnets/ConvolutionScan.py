from manim import *
import numpy as np
import sys
from pathlib import Path
# Add parent directory to path to import base_scene
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_scene import RightRegionScene

class ConvolutionScan(RightRegionScene):
    def construct(self):

        # ============================================================
        # 1. PARAMETERS
        # ============================================================
        input_size = 6
        kernel_size = 3
        output_size = input_size - kernel_size + 1  # 4×4 map

        rng = np.random.default_rng(seed=2025)
        input_values = rng.integers(low=0, high=256, size=(input_size, input_size))
        kernel_values = np.array([
            [0.8, 0.1, -0.6],
            [0.2, 0.05, -0.35],
            [0.45, -0.2, -0.7],
        ])

        cell_size = 0.5

        # ============================================================
        # 2. CREATE THE INPUT GRID (6×6)
        # ============================================================
        input_grid = VGroup()
        for i in range(input_size):
            row = VGroup()
            for j in range(input_size):
                square = Square(side_length=cell_size)
                square.set_stroke(color=GRAY, width=1)
                square.move_to(np.array([j*cell_size, -i*cell_size, 0]))
                value_label = Integer(int(input_values[i, j]), font_size=18)
                value_label.scale(0.7)
                value_label.move_to(square.get_center())
                cell = VGroup(square, value_label)
                row.add(cell)
            input_grid.add(row)

        input_grid.move_to(LEFT*2.5)
        input_label = Text("6×6 Input Image").scale(0.5).next_to(input_grid, UP)
        self.play(Create(input_grid), FadeIn(input_label))
        self.wait(0.5)

        # ============================================================
        # 3. CREATE THE OUTPUT GRID (4×4)
        # ============================================================
        output_grid = VGroup()
        for i in range(output_size):
            row = VGroup()
            for j in range(output_size):
                square = Square(side_length=cell_size)
                square.set_stroke(color=BLUE_D, width=2)
                square.move_to(np.array([j*cell_size, -i*cell_size, 0]))
                row.add(square)
            output_grid.add(row)

        output_grid.move_to(RIGHT*2.5)
        output_label = Text("4×4 Feature Map").scale(0.5).next_to(output_grid, UP)
        self.play(Create(output_grid), FadeIn(output_label))
        self.wait(0.5)

        # ============================================================
        # 4. CREATE THE KERNEL (3×3)
        # ============================================================
        kernel = VGroup()
        for i in range(kernel_size):
            row = VGroup()
            for j in range(kernel_size):
                square = Square(side_length=cell_size)
                square.set_fill(color=RED, opacity=0.4)
                square.set_stroke(color=RED_E, width=2)
                square.move_to(np.array([j*cell_size, -i*cell_size, 0]))
                row.add(square)
            kernel.add(row)

        # Position starting kernel over input grid
        kernel.move_to(input_grid[0][0])
        kernel.shift(RIGHT*(cell_size) + DOWN*(cell_size))

        #kernel_label = Text("3×3 Kernel").scale(0.5).next_to(kernel, UP)
        #self.play(FadeIn(kernel), FadeIn(kernel_label))
        self.wait(0.25)

        # ============================================================
        # 4b. ELEMENTWISE PRODUCT PANEL
        # ============================================================
        def create_matrix_display(matrix, stroke_color=WHITE, text_color=WHITE):
            display = VGroup()
            for r in range(kernel_size):
                row = VGroup()
                for c in range(kernel_size):
                    square = Square(side_length=0.35)
                    square.set_stroke(color=stroke_color, width=1.5)
                    square.set_fill(color=BLACK, opacity=0.1)
                    value = matrix[r, c]
                    if float(value).is_integer():
                        value_label = Integer(int(value), font_size=18)
                    else:
                        value_label = DecimalNumber(float(value), num_decimal_places=2, font_size=18)
                    value_label.scale(0.6)
                    value_label.set_color(text_color)
                    value_label.move_to(square.get_center())
                    cell = VGroup(square, value_label)
                    row.add(cell)
                row.arrange(RIGHT, buff=0)
                display.add(row)
            display.arrange(DOWN, buff=0)
            return display

        patch_display = create_matrix_display(np.zeros((kernel_size, kernel_size)), stroke_color=WHITE)
        product_display = create_matrix_display(np.zeros((kernel_size, kernel_size)), stroke_color=GREEN, text_color=GREEN)
        kernel_reference_display = create_matrix_display(kernel_values, stroke_color=RED, text_color=RED_E)

        patch_column = VGroup(Text("Input patch").scale(0.35), patch_display).arrange(DOWN, buff=0.08)
        kernel_column = VGroup(Text("Kernel").scale(0.35), kernel_reference_display).arrange(DOWN, buff=0.08)
        product_column = VGroup(Text("Elementwise product").scale(0.35), product_display).arrange(DOWN, buff=0.08)

        multiply_symbol = MathTex("\\odot").scale(0.6)
        equals_symbol = MathTex("=").scale(0.6)

        elementwise_panel = VGroup(
            patch_column,
            multiply_symbol,
            kernel_column,
            equals_symbol,
            product_column
        ).arrange(RIGHT, buff=0.25)

        multiply_symbol.set_y(patch_display.get_center()[1])
        equals_symbol.set_y(kernel_reference_display.get_center()[1])
        elementwise_panel.to_edge(DOWN, buff=0.6)

        self.play(FadeIn(elementwise_panel))
        self.wait(0.25)

        # Storage for output value labels (start empty so numbers appear only when computed)
        output_value_labels = [
            [None for _ in range(output_size)]
            for _ in range(output_size)
        ]

        # ============================================================
        # 5. SCANNING ANIMATION
        # ============================================================

        # For each output cell (i, j), move kernel to correct spot
        # and highlight output cell
        for out_i in range(output_size):
            for out_j in range(output_size):

                # Compute absolute input-grid coordinates
                top_left = input_grid[out_i][out_j].get_center()

                # Compute kernel shift target relative to its internal offset
                shift_target = top_left + np.array([cell_size, -cell_size, 0])

                # Animate kernel moving to this position
                patch_vals = input_values[out_i:out_i+kernel_size, out_j:out_j+kernel_size]
                product_vals = patch_vals * kernel_values

                new_patch_display = create_matrix_display(patch_vals, stroke_color=WHITE)
                new_patch_display.move_to(patch_display.get_center())

                new_product_display = create_matrix_display(product_vals, stroke_color=GREEN, text_color=GREEN)
                new_product_display.move_to(product_display.get_center())

                self.play(
                    kernel.animate.move_to(shift_target),
                    Transform(patch_display, new_patch_display),
                    Transform(product_display, new_product_display),
                    run_time=0.35
                )

                # Highlight the active output square
                active_square = output_grid[out_i][out_j]
                highlight = active_square.copy()
                highlight.set_fill(color=GREEN, opacity=0.6)

                # Compute the convolution result for this position (keep float precision)
                output_value = float(np.sum(product_vals))
                new_value_label = DecimalNumber(
                    output_value, num_decimal_places=2, font_size=20, color=WHITE
                ).scale(0.6)
                new_value_label.move_to(active_square.get_center())

                # Update output cell value as it lights up
                existing_label = output_value_labels[out_i][out_j]
                if existing_label is None:
                    output_value_labels[out_i][out_j] = new_value_label
                    active_square.add(new_value_label)
                    self.play(
                        FadeIn(highlight, run_time=0.2),
                        FadeIn(new_value_label, run_time=0.2),
                    )
                else:
                    self.play(
                        FadeIn(highlight, run_time=0.2),
                        Transform(existing_label, new_value_label),
                    )
                self.play(FadeOut(highlight, run_time=0.2))

        # ============================================================
        # 6. END
        # ============================================================
        self.wait(1)
