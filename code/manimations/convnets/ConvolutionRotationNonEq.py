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

        # Boxed equation at top left: g f(X) = f(g X)
        equation_text = MathTex(r"g f(X) = f(g X)")
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
        
        right_plane = Rectangle(width=3.8, height=6.0, color=GREEN, stroke_width=2)
        right_plane.set_x(2.2).set_y(-0.4)
        
        # Align equation box left/right edges with left_plane (X box)
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
        # X -> f(X) -> g f(X)
        # ---------------------------------------------------------
        
        # 1. Show X centered in Left Plane
        X_img = self.create_mnist_image(input_image_path).move_to(left_plane.get_center())
        X_label = MathTex("X", font_size=32).next_to(X_img, UP, buff=0.2)
        
        self.play(FadeIn(X_img), Write(X_label))
        self.wait(0.5)
        
        # 2. Convolve: X (Left) -> f(X) (Right)
        fX_img = self.create_mnist_image(conv_image_path).move_to(right_plane.get_center())
        fX_label = MathTex("f(X)", font_size=32).next_to(fX_img, UP, buff=0.2)
        
        f_arrow_1 = Arrow(left_plane.get_center(), right_plane.get_center(), **arrow_kwargs)
        f_text_1 = MathTex("f", font_size=28).next_to(f_arrow_1, UP, buff=0.1)
        
        self.play(
            Create(f_arrow_1), Write(f_text_1),
            FadeIn(fX_img),
            Write(fX_label)
        )
        self.wait(0.5)
        
        # 3. Rotate in place (Right Plane): f(X) -> g f(X)
        gfX_label = MathTex(r"g f(X)", font_size=32).next_to(fX_img, UP, buff=0.2)
        
        self.play(
            Rotate(fX_img, rot_angle),
            ReplacementTransform(fX_label, gfX_label)
        )
        self.wait(1.5)
        
        # Clear Sequence 1 to make room for Sequence 2
        self.play(
            FadeOut(X_img), FadeOut(X_label),
            FadeOut(fX_img), FadeOut(gfX_label),
            FadeOut(f_arrow_1), FadeOut(f_text_1)
        )
        self.wait(0.5)
        
        # ---------------------------------------------------------
        # SEQUENCE 2: Rotate then Convolve (Bottom path concept)
        # X -> g X -> f(g X) ?? -> ?
        # ---------------------------------------------------------
        
        # 1. Show X centered in Left Plane again
        X_img_2 = self.create_mnist_image(input_image_path).move_to(left_plane.get_center())
        X_label_2 = MathTex("X", font_size=32).next_to(X_img_2, UP, buff=0.2)
        
        self.play(FadeIn(X_img_2), Write(X_label_2))
        self.wait(0.5)
        
        # 2. Rotate in place (Left Plane): X -> g X
        gX_label = MathTex(r"g X", font_size=32).next_to(X_img_2, UP, buff=0.2)
        
        self.play(
            Rotate(X_img_2, rot_angle),
            ReplacementTransform(X_label_2, gX_label)
        )
        self.wait(0.5)
        
        # 3. Convolve: g X (Left) -> ? (Right)
        q_mark = MathTex("?", color=RED, font_size=96).move_to(right_plane.get_center())
        
        f_arrow_2 = Arrow(left_plane.get_center(), right_plane.get_center(), **arrow_kwargs)
        f_text_2 = MathTex("f", font_size=28).next_to(f_arrow_2, UP, buff=0.1)
        
        self.play(
            Create(f_arrow_2), Write(f_text_2),
            FadeIn(q_mark)
        )
        
        self.wait(2)
    
    def create_mnist_image(self, image_path, height=1.3):
        img = ImageMobject(str(image_path))
        img.scale_to_fit_height(height)
        return img
