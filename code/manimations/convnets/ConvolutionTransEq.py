from manim import *
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene

class ConvolutionTransEq(RightRegionScene):
    def construct(self):

        input_image_path = "/Users/user/repos/equivariant-dft-explainer-video/notes/im_sample.png"
        conv_image_path = "/Users/user/repos/equivariant-dft-explainer-video/notes/con_sample.png"

        # Boxed equation at top left
        equation_text = MathTex(r"g f(V) = f(g V)")
        equation_box = SurroundingRectangle(equation_text, color=WHITE, buff=0.2, stroke_width=2)
        equation_group = VGroup(equation_text, equation_box)
        equation_group.set_x(-2.0).set_y(3.3)
    
        # Supporting text to the right of equation
        where_text = Text("where g is a translation", font_size=24)
        where_text.next_to(equation_group, RIGHT, buff=0.5)
        
        # Two vertical planes (rectangles)
        left_plane = Rectangle(width=3.8, height=6.0, color=WHITE, stroke_width=2)
        left_plane.set_x(-2).set_y(-0.4)
        
        right_plane = Rectangle(
            width=3.8,
            height=6.0,
            color="#32CD32",  # bright grass green
            stroke_width=2,
        )
        right_plane.set_x(2).set_y(-0.4)
        
        # Shape X (wavy/abstract shape) in top-left of left plane
        V_shape = self.create_mnist_image(input_image_path)
        V_shape.move_to(left_plane.get_corner(UL) + RIGHT * 0.8 + DOWN * 0.8)
        V_label = MathTex("V", font_size=32)
        V_label.next_to(V_shape, LEFT, buff=0.25)

        # Shape gX in bottom-left of left plane (translated version, still clear)
        gV_shape = self.create_mnist_image(input_image_path)
        gV_shape.move_to(left_plane.get_corner(DL) + RIGHT * 0.8 + UP * 0.8)
        gV_label = MathTex("gV", font_size=32)
        gV_label.next_to(gV_shape, LEFT, buff=0.25)
        
        # Arrow g pointing down in left plane (adjusted to avoid label)
        g_arrow_left = Arrow(
            V_shape.get_bottom() + DOWN * 0.15,
            gV_shape.get_top() + UP * 0.15,
            color=WHITE,
            buff=0.1,
            stroke_width=3
        )
        g_label_left = MathTex("g", font_size=28, color=WHITE)
        g_label_left.move_to(left_plane.get_left() + RIGHT * 0.25).move_to([g_label_left.get_x(), g_arrow_left.get_center()[1], 0])
        
        # Shape f(X) in top-right of right plane (convolved - fuzzy/blurred)
        fV_shape = self.create_mnist_image(conv_image_path)
        fV_shape.move_to(right_plane.get_corner(UR) + LEFT * 0.8 + DOWN * 0.8)
        fV_label = MathTex("f(V)", font_size=32)
        fV_label.next_to(fV_shape, RIGHT, buff=0.25)
        
        # Shape at bottom-right (final result - also fuzzy)
        final_shape = self.create_mnist_image(conv_image_path)
        final_shape.move_to(right_plane.get_corner(DR) + LEFT * 0.8 + UP * 0.8)
        final_label = MathTex(r"gf(V)=fg(V)", font_size=32)
        final_label.next_to(final_shape, RIGHT, buff=0.25)
        
        # Arrow g pointing down in right plane (adjusted to avoid label)
        g_arrow_right = Arrow(
            fV_shape.get_bottom() + DOWN * 0.15,
            final_shape.get_top() + UP * 0.15,
            color=WHITE,
            buff=0.1,
            stroke_width=3
        )
        g_label_right = MathTex("g", font_size=32, color=WHITE)
        g_label_right.move_to(right_plane.get_right() + LEFT * 0.25).move_to([g_label_right.get_x(), g_arrow_right.get_center()[1], 0])

        fgV_label = MathTex("gf(V)=fg(V)", font_size=28, color=WHITE)
        fgV_label.next_to(final_shape, RIGHT, buff=0.25)
        
        # Horizontal arrow f from X to f(X) (top path) - curved to avoid overlap
        f_arrow_top = Arrow(
            V_shape.get_right(),
            fV_shape.get_left(),
            color=WHITE,
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        f_path_top = VGroup(f_arrow_top)
        f_label_top = MathTex("f", font_size=32, color=WHITE)
        f_label_top.next_to(f_arrow_top, DOWN, buff=0.1)
        f_label_top.shift(LEFT * 0.3)

        # Horizontal arrow f from gX to bottom of right plane (bottom path) - curved
        f_arrow_bottom = Arrow(
            gV_shape.get_right(),
            final_shape.get_left(),
            color=WHITE,
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        f_path_bottom = VGroup(f_arrow_bottom)
        f_label_bottom = MathTex("f", font_size=28, color=WHITE)
        f_label_bottom.next_to(f_arrow_bottom, UP, buff=0.1)
        f_label_bottom.shift(LEFT * 0.3)
        
        # Animate everything
        self.play(Write(equation_text))
        self.play(Create(equation_box))
        self.play(Write(where_text))
        
        self.play(Create(left_plane), Create(right_plane))
        
        # Left plane contents
        self.play(FadeIn(V_shape), Write(V_label))
        self.play(Create(g_arrow_left), Write(g_label_left))
        self.play(FadeIn(gV_shape), Write(gV_label))
        
        # Right plane contents and paths
        self.play(Create(f_path_top), Write(f_label_top))
        self.play(FadeIn(fV_shape), Write(fV_label))
        self.play(Create(g_arrow_right), Write(g_label_right))
        
        self.play(Create(f_path_bottom), Write(f_label_bottom))
        self.play(FadeIn(final_shape), Write(fgV_label))
        
        self.wait(2)
    
    def create_mnist_image(self, image_path, height=1.3):
        img = ImageMobject(str(image_path))
        img.scale_to_fit_height(height)
        return img

