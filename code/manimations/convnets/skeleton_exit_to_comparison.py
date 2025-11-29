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
PPRIME_COLOR  = "#FF0000"
TEXT_COLOR    = WHITE
ALPHA         = PI / 12     # 15 degrees
KERNEL_CIRCLE = True

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

    # Fix the leftmost column center
    # Left edge is -fw/2. We add a bit of padding (e.g., 2.2 units).
    # Note: Manim coordinate height is 8.0, width is ~14.2 for 16:9.
    LEFT_START_X = -fw / 2 + 1.2
    
    # Spacing between columns (squeezed relative to uniform spacing)
    PANEL_SPACING_X = 2.5

    x = LEFT_START_X + col * PANEL_SPACING_X

    y =  fh / 2 - (row + 0.5) * cell_h + 0.85
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


def arrow_p(origin: np.ndarray, target: np.ndarray, label="p") -> VGroup:
    arr = Arrow(origin, target, buff=0, color=P_COLOR, stroke_width=3, tip_length=0.15)
    lab = MathTex(label, color=TEXT_COLOR).scale(0.8)
    # Place label at midpoint, shifted slightly down/right to avoid overlap
    midpoint = (origin + target) / 2
    # Direction perpendicular to arrow could be calculated, but a simple shift might suffice
    # given most arrows are generally pointing right/down.
    # Let's try putting it BELOW the midpoint.
    lab.move_to(midpoint + UP * 0.3 + LEFT * 0.1)
    return VGroup(arr, lab)


def arrow_pprime(base_point: np.ndarray,
                 direction=UP,
                 label=r"p'",
                 color=PPRIME_COLOR) -> VGroup:
    arr = Arrow(base_point,
                base_point + direction * 0.5,
                buff=0,
                color=color,
                stroke_width=3,
                tip_length=0.15)
    lab = MathTex(label, color=color).scale(0.8)
    # Move label up relative to the arrow tip
    lab.next_to(arr.get_end(), UP, buff=0.1)
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
    
    # Global scale factor for everything inside the panel
    # Shrink by 20% means scale = 0.8
    PANEL_SCALE = 0.8

    # Panel-specific content
    
    # Define p_offset (vector p) relative to center
    # This controls the initial angle of p with the x axis
    # Default is RIGHT * 1.5 + DOWN * 0.3
    # You can change this to control p's initial angle/length
    p_initial_offset = RIGHT * 2.0 + UP * 0.6
    
    if index == 3:
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
        if KERNEL_CIRCLE:
            c = Circle(radius=0.5, color=PPRIME_COLOR, stroke_width=2)
            c.move_to(s_obj.get_center())
            g.add(c)

    elif index == 8:
        # --- Panel 8: rotated coordinates, highlight p' mismatch ---
        axes = make_axes(center,
                         s_label="s",
                         t_label="t",
                         prefix=r"Q_{-\alpha}",
                         rotate_angle=-ALPHA)
        blob = make_S(center, offset_vector=p_initial_offset)
        origin = center

        p_vec = arrow_p(origin, blob.get_center(), label="p")
        # show a "different" p' (e.g. slightly rotated) to emphasise mismatch
        pprime = arrow_pprime(blob.get_center(),
                              direction=rotate_vector(UP, -ALPHA),
                              label=r"p''",
                              color=PPRIME_COLOR)

        g.add(axes, blob, p_vec, pprime)
        if KERNEL_CIRCLE:
            c = Circle(radius=0.5, color=PPRIME_COLOR, stroke_width=2)
            c.move_to(blob.get_center())
            g.add(c)

    # Apply the global scale to the entire panel group
    g.scale(PANEL_SCALE, about_point=center)

    return g


# ============================================================
# Scene base + 8 cumulative scenes
# ============================================================

class ExitToComparison(Scene):
    """
    Shows panels 3 and 8, then rearranges them.
    """
    def construct(self):
        # Create the two panels
        p3 = make_panel(3)
        p8 = make_panel(8)

        # Add them to the scene immediately (no fade in)
        self.add(p3, p8)
        self.wait(1.0)

        # Target positions
        # We want them side-by-side near the top.
        # Let's use the Y level of panel 3 (row 0).
        target_y = p3.get_y()
        
        # Center them around X=0 with some spacing?
        # Current spacing is 2.5. Let's stick to that or slightly wider.
        # Let's place p3 at x = -1.5 and p8 at x = +1.5?
        # That gives spacing of 3.0.
        
        target_p3 = np.array([-5.0, target_y, 0.5])
        target_p8 = np.array([ -1.0, target_y, 0])

        self.play(
            p3.animate.move_to(target_p3),
            p8.animate.move_to(target_p8),
            run_time=2.0
        )
        self.wait(1.0)

        # Rotate both panels by +15 degrees about their centers
        # But keep labels upright.
        anims = []
        for p in [p3, p8]:
            center = p.get_center()
            
            # Deconstruct panel components based on make_panel structure:
            # p[0]: axes group (lines, xlabel, ylabel)
            # p[1]: S object / blob
            # p[2]: p vector group (arrow, label)
            # p[3]: p' vector group (arrow, label)
            # p[4]: Circle (since KERNEL_CIRCLE=True)
            
            axes_group = p[0]
            s_obj = p[1]
            p_vec_group = p[2]
            pprime_group = p[3]
            
            # Identify parts to rotate normally (geometry)
            rotatable_parts = [
                axes_group[0],   # axes lines
                s_obj,           # image
                p_vec_group[0],  # p arrow
                pprime_group[0], # p' arrow
            ]
            if len(p) > 4:
                rotatable_parts.append(p[4]) # circle
            
            # Identify labels to move (position only) but not rotate (orientation)
            labels = [
                axes_group[1],   # x label
                axes_group[2],   # y label
                p_vec_group[1],  # p label
                pprime_group[1], # p' label
            ]
            
            # Animate geometry rotation
            for part in rotatable_parts:
                anims.append(Rotate(part, angle=ALPHA, about_point=center))
                
            # Animate label positions
            for lab in labels:
                curr_pos = lab.get_center()
                new_pos = center + rotate_vector(curr_pos - center, ALPHA)
                anims.append(lab.animate.move_to(new_pos))
        
        self.play(*anims, run_time=1.5)
        self.wait(1.0)
