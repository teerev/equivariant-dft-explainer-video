from manim import *
import numpy as np

# ---------- Global render config: 1080p ----------
config.pixel_width  = 3840
config.pixel_height = 2160
config.frame_rate   = 60
config.background_color = BLACK

ROWS = 2
COLS = 4

S_COLOR       = "#00D5FF"
KERNEL_COLOR  = "#00D5FF"   # bluish; you can replace with ImageMobject later
P_COLOR       = WHITE
PPRIME_COLOR  = RED
TEXT_COLOR    = WHITE
ALPHA         = PI / 12     # 15 degrees

# ============================================================
# Layout helpers
# ============================================================

def get_panel_center(row: int, col: int) -> np.ndarray:
    """
    Return the center coordinates for a panel in a ROWS x COLS grid.

    row: 0 = top row, ROWS-1 = bottom row
    col: 0 = leftmost, COLS-1 = rightmost
    """
    fw = config.frame_width
    fh = config.frame_height

    cell_w = fw / COLS
    cell_h = fh / ROWS

    x = -fw / 2 + (col + 0.5) * cell_w
    
    # Shift panels to the left, proportionally to their column index
    # (col=0 moves 1 unit left, col=3 moves 4 units left)
    shift_unit = 0.03 * fw
    x -= (col + 1) * shift_unit

    y =  fh / 2 - (row + 0.5) * cell_h + 0.3
    return np.array([x, y, 0.0])


def make_axes(center: np.ndarray,
              s_label="s",
              t_label="t",
              prefix=None,
              rotate_angle=0) -> VGroup:
    """
    Draw a simple pair of axes centred at `center`.
    Optionally rotate them and prefix the labels with, e.g., Q_{-α}.
    """
    x_axis = Arrow(ORIGIN, RIGHT * 2.3, buff=0, stroke_width=3, tip_length=0.15)
    y_axis = Arrow(ORIGIN, UP * 2.3, buff=0, stroke_width=3, tip_length=0.15)
    axes_lines = VGroup(x_axis, y_axis)

    if rotate_angle != 0:
        axes_lines.rotate(rotate_angle, about_point=ORIGIN)

    # Use shift instead of move_to so that the axes origin (intersection)
    # ends up exactly at `center`.
    axes_lines.shift(center)

    # Axis labels
    x_lab_tex = s_label
    y_lab_tex = t_label
    if prefix is not None:
        x_lab_tex = rf"{prefix}{s_label}"
        y_lab_tex = rf"{prefix}{t_label}"

    x_label = MathTex(x_lab_tex, color=TEXT_COLOR).scale(0.7)
    y_label = MathTex(y_lab_tex, color=TEXT_COLOR).scale(0.7)

    x_label.next_to(axes_lines[0].get_end(), DR, buff=0.1)
    y_label.next_to(axes_lines[1].get_end(), UL, buff=0.1)

    return VGroup(axes_lines, x_label, y_label)


def make_S(center: np.ndarray, offset_vector=RIGHT * 2.0 + UP * 0.6) -> Mobject:
    """
    A blue 'S' at a fixed offset within the panel.
    """
    s_obj = ImageMobject("/Users/user/repos/equivariant-dft-explainer-video/notes/s.png")
    s_obj.scale_to_fit_height(0.9)
    s_obj.move_to(center + offset_vector)
    return s_obj


def make_kernel_blob(center: np.ndarray,
                     offset_vector=RIGHT * 1.5 + DOWN * 0.3) -> Mobject:
    """
    Placeholder "kernel response" blob.
    """
    blob = ImageMobject("/Users/user/repos/equivariant-dft-explainer-video/notes/s_conv.png")
    blob.scale_to_fit_height(1.5)
    blob.move_to(center + offset_vector)
    return blob


def arrow_p(origin: np.ndarray, target: np.ndarray, label="p") -> VGroup:
    arr = Arrow(origin, target, buff=0, color=P_COLOR, stroke_width=3, tip_length=0.15)
    lab = MathTex(label, color=TEXT_COLOR).scale(0.8)
    lab.next_to(arr.get_end(), UR, buff=0.1)
    return VGroup(arr, lab)


def arrow_pprime(base_point: np.ndarray,
                 direction=UP,
                 label=r"p'",
                 color=PPRIME_COLOR) -> VGroup:
    arr = Arrow(base_point,
                base_point + direction * 1.0,
                buff=0,
                color=color,
                stroke_width=3,
                tip_length=0.15)
    lab = MathTex(label, color=color).scale(0.8)
    lab.next_to(arr.get_end(), RIGHT, buff=0.1)
    return VGroup(arr, lab)


# ============================================================
# Panel factory
# ============================================================

def make_panel(index: int) -> Mobject:
    """
    Build the panel for a given index 1..8.

    Layout (rows, cols):
      1 2 3 4
      5 6 7 8
    """
    idx = index - 1
    row = idx // COLS     # 0 or 1
    col = idx %  COLS     # 0..3
    center = get_panel_center(row, col)

    # Shift entire figure left and down by removed axis lengths (1.0 each)
    center += LEFT * 1.0 + DOWN * 1.0

    # We'll assemble each panel as a Group (not VGroup, because we have ImageMobjects)
    g = Group()

    # Panel-specific content
    
    # Define p_offset (vector p) relative to center
    # This controls the initial angle of p with the x axis
    # Default is RIGHT * 1.5 + DOWN * 0.3
    # You can change this to control p's initial angle/length
    p_initial_offset = RIGHT * 2.0 + UP * 0.6
    
    if index == 1:
        # --- Panel 1: original coordinates, image X, point p ---
        axes = make_axes(center, s_label="s", t_label="t")
        s_obj = make_S(center, offset_vector=p_initial_offset)
        origin = center  # intersection of axes

        p_vec = arrow_p(origin, s_obj.get_center(), label="p")
        pprime = arrow_pprime(s_obj.get_center(), direction=UP, label=r"p'")
        g.add(axes, s_obj, p_vec, pprime)

    elif index == 2:
        # --- Panel 2: same as 1, plus p' at S ---
        # Rotate p and p' rigidly by ALPHA
        axes = make_axes(center, s_label="s", t_label="t")
        
        rot_offset = rotate_vector(p_initial_offset, ALPHA)
        # Create S at the rotated position
        s_obj = make_S(center, offset_vector=rot_offset)
        # Rotate S about its own center by -ALPHA
        s_obj.rotate(ALPHA, about_point=s_obj.get_center())

        origin = center

        p_vec = arrow_p(origin, s_obj.get_center(), label=r"Q_\alpha p")
        
        rot_direction = rotate_vector(UP, ALPHA)
        pprime = arrow_pprime(s_obj.get_center(), direction=rot_direction, label=r"Q_\alpha p'")
        g.add(axes, s_obj, p_vec, pprime)

    elif index == 3:
        # --- Panel 3: rotated coordinates Q_{-α}, still showing S and p' ---
        axes = make_axes(center,
                         s_label="s",
                         t_label="t",
                         prefix=r"Q_{-\alpha}",
                         rotate_angle=-ALPHA)
        s_obj = make_S(center, offset_vector=p_initial_offset)
        origin = center

        p_vec = arrow_p(origin, s_obj.get_center(), label="p")
        pprime = arrow_pprime(s_obj.get_center(), direction=UP, label=r"p'")
        g.add(axes, s_obj, p_vec, pprime)

    elif index == 4:
        # --- Panel 4: rotated coordinates, kernel response at p, p' arrow ---
        axes = make_axes(center,
                         s_label="s",
                         t_label="t",
                         prefix=r"Q_{-\alpha}",
                         rotate_angle=-ALPHA)
        blob = make_kernel_blob(center, offset_vector=p_initial_offset)
        origin = center

        p_vec = arrow_p(origin, blob.get_center(), label="p")
        pprime = arrow_pprime(blob.get_center(), direction=UP, label=r"p'")
        g.add(axes, blob, p_vec, pprime)

    elif index == 5:
        # --- Panel 5: original coordinates, kernel response at p (X*F)(p) ---
        axes = make_axes(center, s_label="s", t_label="t")
        blob = make_kernel_blob(center, offset_vector=p_initial_offset)
        origin = center

        p_vec = arrow_p(origin, blob.get_center(), label="p")
        g.add(axes, blob, p_vec)

    elif index == 6:
        # --- Panel 6: original coordinates, blob at Q_α p ---
        axes = make_axes(center, s_label="s", t_label="t")
        # place blob as if p has been rotated by +α
        
        # Here we rotate the original p_offset by ALPHA
        rot_offset = rotate_vector(p_initial_offset, ALPHA)
        blob = make_kernel_blob(center, offset_vector=rot_offset)
        blob.rotate(ALPHA, about_point=blob.get_center())
        origin = center

        p_vec = arrow_p(origin, blob.get_center(), label=r"Q_\alpha p")
        g.add(axes, blob, p_vec)

    elif index == 7:
        # --- Panel 7: rotated coordinates, blob at p in Q_{-α} frame ---
        axes = make_axes(center,
                         s_label="s",
                         t_label="t",
                         prefix=r"Q_{-\alpha}",
                         rotate_angle=-ALPHA)
        blob = make_kernel_blob(center, offset_vector=p_initial_offset)
        origin = center

        p_vec = arrow_p(origin, blob.get_center(), label="p")
        g.add(axes, blob, p_vec)

    elif index == 8:
        # --- Panel 8: rotated coordinates, highlight p' mismatch ---
        axes = make_axes(center,
                         s_label="s",
                         t_label="t",
                         prefix=r"Q_{-\alpha}",
                         rotate_angle=-ALPHA)
        blob = make_kernel_blob(center, offset_vector=p_initial_offset)
        origin = center

        p_vec = arrow_p(origin, blob.get_center(), label="p")
        # show a "different" p' (e.g. slightly rotated) to emphasise mismatch
        pprime = arrow_pprime(blob.get_center(),
                              direction=rotate_vector(UP, -ALPHA),
                              label=r"p''",
                              color=PPRIME_COLOR)

        g.add(axes, blob, p_vec, pprime)

    else:
        # Fallback placeholder
        rect = Rectangle(
            width=config.frame_width / COLS * 0.9,
            height=config.frame_height / ROWS * 0.9,
            stroke_color=WHITE,
            stroke_width=2,
        ).move_to(center)
        label = Text(str(index)).scale(0.7).move_to(center)
        g.add(rect, label)

    return g


# ============================================================
# Scene base + 8 cumulative scenes
# ============================================================

class BaseGridScene(Scene):
    """
    Base class for all 8 scenes.
    It creates the first `panels_to_show` panels in a 4x2 grid.
    """

    panels_to_show: int = 1

    def construct(self):
        panels = []
        for k in range(1, self.panels_to_show + 1):
            p = make_panel(k)
            panels.append(p)

        if panels:
            if len(panels) > 1:
                self.add(*panels[:-1])
            self.play(FadeIn(panels[-1]))
            self.wait(0.5)


class EquivariancePanels1(BaseGridScene):
    panels_to_show = 1

class EquivariancePanels2(BaseGridScene):
    panels_to_show = 2

class EquivariancePanels3(BaseGridScene):
    panels_to_show = 3

class EquivariancePanels4(BaseGridScene):
    panels_to_show = 4

class EquivariancePanels5(BaseGridScene):
    panels_to_show = 5

class EquivariancePanels6(BaseGridScene):
    panels_to_show = 6

class EquivariancePanels7(BaseGridScene):
    panels_to_show = 7

class EquivariancePanels8(BaseGridScene):
    panels_to_show = 8
