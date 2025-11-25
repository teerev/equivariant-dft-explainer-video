from manim import *
import numpy as np


class RightRegionScene(Scene):
    def setup(self):
        # Space to reserve on the right (pixels)
        RIGHT_MARGIN_PX = 180

        # Convert px → manim coords
        px_to_coord = self.camera.frame_width / self.camera.pixel_width

        # To keep content centered within the 1560px region, shift the camera
        # center to the right by half the reserved margin. Moving the camera
        # right makes the visible content appear leftward.
        shift = (RIGHT_MARGIN_PX / 2) * px_to_coord

        self.camera.frame_center = self.camera.frame_center + np.array([shift, 0, 0])
