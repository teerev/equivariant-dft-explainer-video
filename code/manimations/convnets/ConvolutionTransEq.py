from manim import *
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene

class ConvolutionTransEq(RightRegionScene):
    def construct(self):

        # Boxed equation at top left
        equation_text = MathTex(r"g f(X) = f(g X)")
        equation_box = SurroundingRectangle(equation_text, color=WHITE, buff=0.2, stroke_width=2)
        equation_group = VGroup(equation_text, equation_box)
        equation_group.set_x(-2.0).set_y(3)
    
        # Supporting text to the right of equation
        where_text = Text("where g is a translation", font_size=24)
        where_text.next_to(equation_group, RIGHT, buff=0.5)
        
        # Two vertical planes (rectangles)
        left_plane = Rectangle(width=2.5, height=4, color=BLUE, stroke_width=2)
        left_plane.set_x(-2).set_y(0)
        
        right_plane = Rectangle(width=2.5, height=4, color=BLUE, stroke_width=2)
        right_plane.set_x(2).set_y(0)
        
        # Shape X (wavy/abstract shape) in top-left of left plane
        X_shape = self.create_wavy_shape()
        X_shape.move_to(left_plane.get_corner(UL) + RIGHT * 0.5 + DOWN * 0.5).scale(0.3)
        X_label = MathTex("X", font_size=32)
        X_label.move_to(left_plane.get_right() + LEFT * 0.2).align_to(X_shape, DOWN)

        # Shape gX in bottom-left of left plane (translated version, still clear)
        gX_shape = self.create_wavy_shape()
        gX_shape.move_to(left_plane.get_corner(DL) + RIGHT * 0.5 + UP * 0.5).scale(0.3)
        gX_label = MathTex("gX", font_size=32)
        gX_label.move_to(left_plane.get_right() + LEFT * 0.2).align_to(gX_shape, UP)
        
        # Arrow g pointing down in left plane (adjusted to avoid label)
        g_arrow_left = Arrow(
            X_shape.get_bottom() + DOWN * 0.15,
            gX_shape.get_top() + UP * 0.15,
            color=WHITE,
            buff=0.1,
            stroke_width=3
        )
        g_label_left = MathTex("g", font_size=28, color=WHITE)
        g_label_left.move_to(left_plane.get_left() + RIGHT * 0.25).move_to([g_label_left.get_x(), g_arrow_left.get_center()[1], 0])
        
        # Shape f(X) in top-right of right plane (convolved - fuzzy/blurred)
        fX_shape = self.create_fuzzy_shape()
        fX_shape.move_to(right_plane.get_corner(UR) + LEFT * 0.5 + DOWN * 0.5).scale(0.3)
        fX_label = MathTex("f(X)", font_size=32)
        fX_label.move_to(right_plane.get_left() + RIGHT * 0.2).align_to(fX_shape, DOWN)
        
        # Shape at bottom-right (final result - also fuzzy)
        final_shape = self.create_fuzzy_shape()
        final_shape.move_to(right_plane.get_corner(DR) + LEFT * 0.5 + UP * 0.5).scale(0.3)
        final_label = MathTex(r"g f(X) = f(g X)", font_size=32)
        final_label.move_to(right_plane.get_left() + RIGHT * 0.4).align_to(final_shape, UP).shift(UP * 0.4)
        
        # Arrow g pointing down in right plane (adjusted to avoid label)
        g_arrow_right = Arrow(
            fX_shape.get_bottom() + DOWN * 0.15,
            final_shape.get_top() + UP * 0.15,
            color=WHITE,
            buff=0.1,
            stroke_width=3
        )
        g_label_right = MathTex("g", font_size=32, color=WHITE)
        g_label_right.move_to(right_plane.get_right() + LEFT * 0.25).move_to([g_label_right.get_x(), g_arrow_right.get_center()[1], 0])
        
        # Horizontal arrow f from X to f(X) (top path) - curved to avoid overlap
        f_arc_top = ArcBetweenPoints(
            X_shape.get_right(),
            fX_shape.get_left(),
            angle=-PI/4,
            color=YELLOW,
            stroke_width=3
        )
        f_arrow_top = Arrow(
            f_arc_top.get_end() + UP * 0.05,
            fX_shape.get_left(),
            color=YELLOW,
            buff=0.05,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        f_path_top = VGroup(f_arc_top, f_arrow_top)
        f_label_top = MathTex("f", font_size=32, color=YELLOW)
        f_label_top.move_to((X_shape.get_right() + fX_shape.get_left()) / 2 + UP * 1.0)

        # Horizontal arrow f from gX to bottom of right plane (bottom path) - curved
        f_arc_bottom = ArcBetweenPoints(
            gX_shape.get_right(),
            final_shape.get_left(),
            angle=PI/4,
            color=YELLOW,
            stroke_width=3
        )
        f_arrow_bottom = Arrow(
            f_arc_bottom.get_end() + DOWN * 0.05,
            final_shape.get_left(),
            color=YELLOW,
            buff=0.05,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        f_path_bottom = VGroup(f_arc_bottom, f_arrow_bottom)
        f_label_bottom = MathTex("f", font_size=28, color=YELLOW)
        f_label_bottom.move_to((gX_shape.get_right() + final_shape.get_left()) / 2 + DOWN * 1.0)
        
        # Animate everything
        self.play(Write(equation_text))
        self.play(Create(equation_box))
        self.play(Write(where_text))
        
        self.play(Create(left_plane), Create(right_plane))
        
        # Left plane contents
        self.play(Create(X_shape), Write(X_label))
        self.play(Create(g_arrow_left), Write(g_label_left))
        self.play(Create(gX_shape), Write(gX_label))
        
        # Right plane contents and paths
        self.play(Create(f_path_top), Write(f_label_top))
        self.play(Create(fX_shape), Write(fX_label))
        self.play(Create(g_arrow_right), Write(g_label_right))
        
        self.play(Create(f_path_bottom), Write(f_label_bottom))
        self.play(Create(final_shape), Write(final_label))
        
        self.wait(2)
    
    def create_wavy_shape(self):
        """Create a wavy/abstract blob shape (clear input image)"""
        points = []
        angles = np.linspace(0, 2 * PI, 8)
        radii = [1 + 0.3 * np.sin(3 * angle) for angle in angles]
        for angle, radius in zip(angles, radii):
            points.append([radius * np.cos(angle), radius * np.sin(angle), 0])
        return Polygon(*points, color=GREEN, fill_opacity=0.5, stroke_width=2)
    
    def create_fuzzy_shape(self):
        """Create a fuzzy/blurred shape (convolved feature map)"""
        # Create base shape with more distortion/noise
        base_points = []
        angles = np.linspace(0, 2 * PI, 8)
        radii = [1 + 0.3 * np.sin(3 * angle) for angle in angles]
        for angle, radius in zip(angles, radii):
            base_points.append([radius * np.cos(angle), radius * np.sin(angle), 0])
        
        # Create multiple overlapping shapes with slight offsets to simulate blur
        shapes = VGroup()
        for offset in [0, 0.15, -0.15]:
            for angle_offset in [0, 0.2, -0.2]:
                noisy_points = []
                for px, py, pz in base_points:
                    noisy_points.append([
                        px + offset + 0.1 * np.sin(px * 5),
                        py + angle_offset + 0.1 * np.cos(py * 5),
                        0
                    ])
                shape = Polygon(*noisy_points, color=GREEN, fill_opacity=0.2, stroke_width=1.5)
                shapes.add(shape)
        
        # Make it look scrambled/fuzzy but keep green color like input
        shapes.set_stroke(opacity=0.6).set_fill(opacity=0.15)
        return shapes

