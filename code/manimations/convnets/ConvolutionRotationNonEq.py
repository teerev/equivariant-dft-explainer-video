from manim import *
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base_scene import RightRegionScene

class ConvolutionRotationNonEq(RightRegionScene):
    def construct(self):

        input_image_path = "/Users/user/repos/equivariant-dft-explainer-video/notes/im_sample.png"
        conv_image_path = "/Users/user/repos/equivariant-dft-explainer-video/notes/con_sample.png"

        # Boxed equation at top left: g f(V) = f(g V)
        equation_text = MathTex(r"g f(V) \not= f(g V)")
        equation_box = SurroundingRectangle(equation_text, color=WHITE, buff=0.2, stroke_width=2)
        equation_group = VGroup(equation_text, equation_box)
        # Roughly place it high up first
        equation_group.set_x(-2.0).set_y(3.3)
    
        # Supporting text to the right of equation
        where_text = Text("where g is a rotation", font_size=24)
        where_text.next_to(equation_group, RIGHT, buff=0.5)
        
        # Two vertical planes (rectangles)
        # Increased width to 3.8 to match the request "Make both X and Y boxes wider"
        left_plane = Rectangle(width=3.8, height=6.0, color=WHITE, stroke_width=2)
        left_plane.set_x(-2.2).set_y(-0.4)
        
        right_plane = Rectangle(width=3.8, height=6.0, color="#32CD32", stroke_width=2)
        right_plane.set_x(2.2).set_y(-0.4)
        
        # Align equation box left/right edges with left_plane (V box)
        # Scale box width to match left_plane width? Or just align edges?
        # User said: "X box is perfectly lined up on the left and right sides with the equivariance condition box"
        # This implies equation_box width should MATCH left_plane width.
        equation_box.set_width(left_plane.width)
        # Re-center text inside the now-wider box
        equation_text.move_to(equation_box.get_center())
        
        # Align horizontal position of equation group to match left_plane
        equation_group.set_x(left_plane.get_x())
        
        # Re-position supporting text
        where_text.next_to(equation_group, RIGHT, buff=0.5)

        # ---------------------------------------------------------
        # Setup Static Elements
        # ---------------------------------------------------------
        self.play(
            Write(equation_text),
            Create(equation_box),
            Write(where_text),
            Create(left_plane), 
            Create(right_plane)
        )
        self.wait(0.5)

        # Shared arrow style
        arrow_kwargs = {"buff": 0.1, "color": WHITE, "stroke_width": 3, "max_tip_length_to_length_ratio": 0.15}
        rot_angle = -PI/2

        # ---------------------------------------------------------
        # SEQUENCE 1: Convolve then Rotate (Top path concept)
        # V -> f(V) -> g f(V)
        # ---------------------------------------------------------
        
        # 1. Show V centered in Left Plane
        V_img = self.create_mnist_image(input_image_path).move_to(left_plane.get_center())
        V_label = MathTex("V", font_size=32).next_to(V_img, UP, buff=0.2)
        
        self.play(FadeIn(V_img), Write(V_label))
        self.wait(0.5)
        
        # 2. Convolve: V (Left) -> f(V) (Right)
        fV_img = self.create_mnist_image(conv_image_path).move_to(right_plane.get_center())
        fV_label = MathTex("f(V)", font_size=32).next_to(fV_img, UP, buff=0.2)
        
        f_arrow_1 = Arrow(left_plane.get_right() + LEFT*0.75, right_plane.get_left() + RIGHT*0.75, **arrow_kwargs)
        f_text_1 = MathTex("f", font_size=28).next_to(f_arrow_1, UP, buff=0.1)
        
        self.play(
            Create(f_arrow_1), Write(f_text_1),
            FadeIn(fV_img),
            Write(fV_label)
        )
        self.wait(0.5)
        
        # 3. Rotate in place (Right Plane): f(V) -> g f(V)
        gfV_label = MathTex(r"g f(V)", font_size=32).next_to(fV_img, UP, buff=0.2)
        
        self.play(
            Rotate(fV_img, rot_angle),
            ReplacementTransform(fV_label, gfV_label)
        )
        self.wait(1.5)
        
        # Clear Sequence 1 to make room for Sequence 2
        self.play(
            FadeOut(V_img), FadeOut(V_label),
            FadeOut(fV_img), FadeOut(gfV_label),
            FadeOut(f_arrow_1), FadeOut(f_text_1)
        )
        self.wait(0.5)
        
        # ---------------------------------------------------------
        # SEQUENCE 2: Rotate then Convolve (Bottom path concept)
        # V -> g V -> f(g V) ?? -> ?
        # ---------------------------------------------------------
        
        # 1. Show V centered in Left Plane again
        V_img_2 = self.create_mnist_image(input_image_path).move_to(left_plane.get_center())
        V_label_2 = MathTex("V", font_size=32).next_to(V_img_2, UP, buff=0.2)
        
        self.play(FadeIn(V_img_2), Write(V_label_2))
        self.wait(0.5)
        
        # 2. Rotate in place (Left Plane): V -> g V
        gV_label = MathTex(r"g V", font_size=32).next_to(V_img_2, UP, buff=0.2)
        
        self.play(
            Rotate(V_img_2, rot_angle),
            ReplacementTransform(V_label_2, gV_label)
        )
        self.wait(0.5)
        
        # 3. Convolve: g V (Left) -> ? (Right)
        q_mark = MathTex("?", color=RED, font_size=96).move_to(right_plane.get_center())
        fgV_label = MathTex(r"f(g V)", font_size=32)
        fgV_label.move_to(right_plane.get_center())
        fgV_label.match_y(gV_label)
        
        f_arrow_2 = Arrow(left_plane.get_right() + LEFT*0.75, right_plane.get_left() + RIGHT*0.75, **arrow_kwargs)
        f_text_2 = MathTex("f", font_size=28).next_to(f_arrow_2, UP, buff=0.1)
        
        self.play(
            Create(f_arrow_2), Write(f_text_2),
            FadeIn(q_mark),
            Write(fgV_label)
        )
        
        self.wait(2)
    
    def create_mnist_image(self, image_path, height=1.3):
        img = ImageMobject(str(image_path))
        img.scale_to_fit_height(height)
        return img
