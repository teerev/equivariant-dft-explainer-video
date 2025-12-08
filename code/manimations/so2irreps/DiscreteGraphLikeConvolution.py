# pyright: reportMissingImports=false
import manim as mn
import numpy as np
import sys
from pathlib import Path
from scipy.special import jv, jn_zeros

# Colors
SCARLET = mn.ManimColor("#F20000")
IMAGE_NEG_HEX = "#00D5FF"
IMAGE_POS_HEX = "#F26D00"
IMAGE_NEG_COLOR = mn.ManimColor(IMAGE_NEG_HEX)
IMAGE_POS_COLOR = mn.ManimColor(IMAGE_POS_HEX)
IMAGE_NEG_RGB = np.array(IMAGE_NEG_COLOR.to_rgb())
IMAGE_POS_RGB = np.array(IMAGE_POS_COLOR.to_rgb())
PURE_BLUE = mn.ManimColor("#0000FF")
BRIGHT_GREEN = mn.ManimColor("#8f02fa")
GREEN_C = mn.ManimColor("#88CC88")
KERNEL_IMAGE_SCALE = 0.475
KERNEL_CIRCLE_RADIUS_SCALE = 0.5

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))
try:
    from base_scene import RightRegionScene
except ImportError:
    class RightRegionScene(mn.Scene): pass

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def create_bessel_image(m, n, part="real", resolution=240, span=3.0, shape="square"):
    xs = np.linspace(-span, span, resolution)
    ys = np.linspace(-span, span, resolution)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X**2 + Y**2)
    THETA = np.arctan2(Y, X)

    R_disk = span * 0.4
    zeros_m = jn_zeros(m, n + 5)
    alpha_mn = zeros_m[n - 1]
    arg = alpha_mn * (R / R_disk)

    radial = jv(m, arg)
    radial = np.where(R <= R_disk, radial, 0.0)

    if part == "real":
        Z = radial if m == 0 else radial * np.cos(m * THETA)
    else:
        Z = np.zeros_like(radial) if m == 0 else radial * np.sin(m * THETA)

    z_clipped = np.clip(Z, -1.0, 1.0)
    mag = np.abs(z_clipped)
    gamma = 0.3
    mag_gamma = mag**gamma

    base = np.where(
        z_clipped[..., None] >= 0,
        IMAGE_POS_RGB[None, None, :],
        IMAGE_NEG_RGB[None, None, :],
    )
    rgb = base * mag_gamma[..., None]
    alpha = mag_gamma

    h, w = resolution, resolution
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))

    if shape == "square":
        half_side = int((h * 0.4) / 2)
        mask = (np.abs(yy - cy) <= half_side) & (np.abs(xx - cx) <= half_side)
    else:
        radius = (h * 0.4) / 2
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

    alpha[~mask] = 0.0

    rgba = np.zeros((h, w, 4), dtype=float)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha

    return mn.ImageMobject(np.uint8(np.flipud(rgba) * 255))

def make_grid(cols, rows, cell_size, color=mn.WHITE, stroke_width=2):
    lines = mn.VGroup()
    width = cols * cell_size
    height = rows * cell_size
    for i in range(cols + 1):
        x = i * cell_size
        lines.add(mn.Line(np.array([x, 0, 0]), np.array([x, height, 0]), stroke_color=color, stroke_width=stroke_width))
    for j in range(rows + 1):
        y = j * cell_size
        lines.add(mn.Line(np.array([0, y, 0]), np.array([width, y, 0]), stroke_color=color, stroke_width=stroke_width))
    return lines

# -------------------------------------------------------------------------
# Main Scene
# -------------------------------------------------------------------------

class SplitPanelConvolution(mn.Scene):
    def construct(self):
        self.camera.background_color = mn.BLACK
        
        # 1. Setup Elements
        top_group, top_refs = self.setup_top_panel()
        bottom_group, bottom_refs = self.setup_bottom_panel()
        
        # 2. Layout & Scaling
        # Top Panel
        top_scale = 0.6
        top_group.scale(top_scale)
        top_group.to_edge(mn.UP, buff=0.5)
        
        # Bottom Panel
        bottom_scale = 0.6
        bottom_group.scale(bottom_scale)
        bottom_group.to_edge(mn.DOWN, buff=0.5)
        
        # Alignment: Move top panel rightwards so its LHS aligns with bottom panel origin
        axes = bottom_refs['axes']
        # Calculate origin position in scene coordinates
        # Note: axes is inside bottom_group, which is scaled and moved.
        # We need the global position of the origin point (0,0,0) of the axes.
        bottom_origin = axes.c2p(0, 0, 0)
        
        top_left_x = top_group.get_left()[0]
        shift_vector = np.array([bottom_origin[0] - top_left_x, 0, 0])
        top_group.shift(shift_vector)
        
        # Add highlight updater *after* scaling, so it uses current positions/sizes
        # We need to reference the scaled objects
        kernel_box = bottom_refs['kernel_box']
        kernel_group = bottom_refs['kernel_group']
        point_cloud = bottom_refs['point_cloud']
        highlight_group = bottom_refs['highlight_group']
        
        def update_hl(mob):
            mob.scale(0)
            mob.remove(*mob)
            mob.scale(1)
            
            center = kernel_group.get_center()
            # Visual radius is half the visual width of the kernel box
            radius = kernel_box.width / 2.0 
            
            # Find points inside
            inside_dots = []
            for dot in point_cloud:
                if np.linalg.norm(dot.get_center() - center) <= radius:
                    inside_dots.append(dot)
            
            lines = []
            dots = []
            for d in inside_dots:
                # Create a fresh highlight dot at the target dot's visual center
                # Radius half of current: 0.11 / 2 = 0.055 * bottom_scale
                hl_dot = mn.Dot(point=d.get_center(), radius=0.055 * bottom_scale, color=BRIGHT_GREEN)
                hl_dot.set_z_index(21)
                dots.append(hl_dot)
                
                # Line
                if np.linalg.norm(d.get_center() - center) > 1e-6:
                    l = mn.Line(center, d.get_center(), color=BRIGHT_GREEN, stroke_width=2.0)
                    l.set_z_index(20)
                    lines.append(l)
            
            mob.add(*lines, *dots)

        highlight_group.add_updater(update_hl)
        
        self.add(top_group)
        self.add(bottom_group)
        
        # 3. Create Animations (using scaled units)
        top_anim = self.create_top_animation(top_refs, top_scale)
        bottom_anim = self.create_bottom_animation(bottom_refs)
        
        # 4. Play in Parallel
        self.play(
            mn.AnimationGroup(top_anim, bottom_anim),
        )
        self.wait(1)

    def setup_top_panel(self):
        # Stretch cell size instead of number of squares
        cell = 0.6 # Increased from 0.4 (50% bigger)
        
        origin_white = np.array([-2.0, -1.0, 0.0])
        # Revert to original grid dimensions
        white_size = (9, 8) 
        red_size = (3, 3)
        origin_red = origin_white.copy()

        white_grid = make_grid(*white_size, cell, color=mn.WHITE, stroke_width=4)
        white_grid.shift(origin_white)

        # Red grid
        red_grid = make_grid(*red_size, cell, color=mn.RED, stroke_width=4)
        red_grid.shift(origin_red)
        
        # We replace the teal square with dots in the white grid.
        # But first, group red_grid.
        red_group = mn.VGroup(red_grid)

        # Create dots for the white grid cells
        white_grid_dots = mn.VGroup()
        white_dots_list = [] # Keep track for updating
        
        for i in range(white_size[0]): # cols
            for j in range(white_size[1]): # rows
                local_pos = np.array([(i + 0.5) * cell, (j + 0.5) * cell, 0.0])
                abs_pos = origin_white + local_pos
                
                # Radius 3x bigger: 0.05 * cell * 3 = 0.15 * cell
                dot = mn.Dot(point=abs_pos, radius=0.15 * cell, color=PURE_BLUE)
                # Ensure dot is on top of white grid and red grid lines
                dot.set_z_index(10) 
                white_grid_dots.add(dot)
                white_dots_list.append(dot)

        # Highlight group for lines
        top_highlight_group = mn.VGroup().set_z_index(20)

        # Add highlight updater to the dots
        def update_top_hl(mob):
            # mob is top_highlight_group
            mob.scale(0)
            mob.remove(*mob)
            mob.scale(1)
            
            # Use white_grid_dots, red_group from outer scope
            # Check bounds
            rg_center = red_group.get_center()
            rg_width = red_group.width
            rg_height = red_group.height
            
            left = rg_center[0] - rg_width / 2
            right = rg_center[0] + rg_width / 2
            bottom = rg_center[1] - rg_height / 2
            top = rg_center[1] + rg_height / 2
            
            eps = 1e-4
            
            inside_dots = []
            for dot in white_grid_dots:
                p = dot.get_center()
                if (left + eps <= p[0] <= right - eps) and (bottom + eps <= p[1] <= top - eps):
                    dot.set_color(BRIGHT_GREEN)
                    inside_dots.append(dot)
                else:
                    dot.set_color(PURE_BLUE)
            
            # If we have inside dots, draw lines from center
            # Center of kernel is rg_center. Find closest dot.
            if inside_dots:
                # Find dot closest to rg_center
                center_dot = min(inside_dots, key=lambda d: np.linalg.norm(d.get_center() - rg_center))
                
                lines = []
                for d in inside_dots:
                    if d is center_dot: continue
                    # Draw line from center dot to d
                    l = mn.Line(center_dot.get_center(), d.get_center(), color=BRIGHT_GREEN, stroke_width=2.0)
                    lines.append(l)
                
                mob.add(*lines)

        top_highlight_group.add_updater(update_top_hl)

        top_group = mn.VGroup(
            white_grid, white_grid_dots, red_group, top_highlight_group
        )
        
        refs = {
            'red_group': red_group,
            'cell': cell,
            'origin_red': origin_red
        }
        return top_group, refs

    def create_top_animation(self, refs, scale_factor):
        red_group = refs['red_group']
        eff_cell = refs['cell'] * scale_factor
        
        # Grid dimensions from setup:
        # white_size = (9, 8)  (cols, rows)
        # red_size = (3, 3)
        # bottom-left of white grid is origin_white
        # origin_white = (-2.0, -1.0, 0.0) -> scaled by 0.6 + shifted to UP edge
        # But we are working in the local coordinates of `top_group` BEFORE it was scaled/shifted in construct.
        # Wait, `refs` contains original objects. 
        # But in construct: `top_group.scale(top_scale).to_edge(...)`.
        # This transforms the objects in place.
        # So `red_group.get_center()` returns the SCALED and SHIFTED position.
        
        # We need to calculate the path in the transformed space.
        # Or simpler: Calculate relative shifts.
        # A relative shift of `cell * scale` moves by one cell visual distance.
        
        # Grid dimensions:
        W, H = 9, 8
        K_w, K_h = 3, 3
        
        # Valid positions for kernel bottom-left relative to white grid bottom-left:
        # col index i from 0 to W - K_w
        # row index j from 0 to H - K_h
        
        # Range of valid top-left positions for the kernel (scanning from top-left):
        # Rows: from H - K_h down to 0
        # Cols: 0 to W - K_w
        
        # Let's verify initial position.
        # Setup puts red_grid at origin_white (aligned bottom-left).
        # So red_grid starts at (0, 0) relative to white grid.
        # We want to start at TOP-LEFT valid position.
        # Top valid row index = H - K_h = 8 - 3 = 5.
        # Left valid col index = 0.
        # So we need to move red_group from (0,0) to (0, 5) * relative_cell_size.
        
        # Movement vector for one cell:
        right_step = mn.RIGHT * eff_cell
        down_step = mn.DOWN * eff_cell
        up_step = mn.UP * eff_cell
        
        # Current logic assumes we are at (0,0) relative to white grid?
        # In setup_top_panel: `red_grid.shift(origin_red)`. `origin_red = origin_white`.
        # So initially, red is at bottom-left corner (0,0).
        
        # 1. Move to Top-Left Start (0, 5)
        # We construct the animation sequence.
        
        anims = []
        
        # Initial jump to top-left (col 0, row 5)
        # 5 steps UP from current position (0,0)
        start_shift = 5 * up_step
        
        # But `red_group` has already been transformed by scene layout.
        # `animate.shift` works relative to current.
        # BUT `Succession` accumulates animations.
        # If we use `ApplyMethod` with absolute coords, we need to track current pos.
        
        current_pos = red_group.get_center()
        
        # Move to (0, 5) relative to current (0,0)
        start_pos = current_pos + start_shift
        
        # Instant move to start
        red_group.move_to(start_pos)
        current_pos = start_pos
        
        # Now traverse:
        # Rows from 5 down to 0.
        # For even rows (relative index from top? No, let's track j)
        # j goes 5, 4, 3, 2, 1, 0.
        
        # Pattern:
        # j=5: Scan Right (0 -> 6)
        # Down
        # j=4: Scan Left (6 -> 0)
        # Down
        # ...
        
        max_col = W - K_w # 9 - 3 = 6
        min_col = 0
        
        # We are at (0, 5).
        # Scan direction variable
        scan_right = True # Initially scanning right
        
        for j in range(5, -1, -1):
            # Scan across the row
            # If scanning right: 0 to 6
            # If scanning left: 6 to 0
            
            # We are currently at start of row (either left or right end).
            # We make discrete steps across.
            
            steps = max_col - min_col # 6 steps
            
            for _ in range(steps):
                direction = right_step if scan_right else -right_step # Left is negative right
                target = current_pos + direction
                
                anims.append(
                    mn.ApplyMethod(
                        red_group.move_to,
                        target,
                        run_time=0.4, # Faster steps
                        rate_func=mn.linear
                    )
                )
                # Pause at each grid-aligned position
                anims.append(mn.Wait(0.5))
                
                current_pos = target
            
            # If not the last row, move down
            if j > 0:
                target = current_pos + down_step
                anims.append(
                    mn.ApplyMethod(
                        red_group.move_to,
                        target,
                        run_time=0.4,
                        rate_func=mn.linear
                    )
                )
                # Pause after moving down
                anims.append(mn.Wait(0.5))
                
                current_pos = target
                
                # Flip scan direction
                scan_right = not scan_right
            
        return mn.Succession(*anims)

    def setup_bottom_panel(self):
        axes = mn.Axes(
            x_range=[-0.5, 5.5, 1], y_range=[-0.5, 5.5, 1], x_length=6, y_length=6,
            axis_config={"color": mn.GREY_B, "stroke_width": 2, "include_tip": True},
        )
        axes_shift = mn.RIGHT * 0.1 + mn.DOWN * 3.0
        axes.shift(axes_shift)
        axes_labels = axes.get_axis_labels(mn.MathTex("x"), mn.MathTex("y"))
        # nudge the x label closer to the x axis so it matches the y label placement
        axes_labels[0].shift(mn.DOWN * 1.25)

        # Point Cloud
        rng = np.random.default_rng(2042)
        width = mn.config.frame_width
        height = mn.config.frame_height
        scale_factor = 0.7
        
        # New requirements:
        # Stretch distribution as well?
        # Range was [0.2, 2.2]. Stretched 50%: [0.2, 3.3].
        # Further stretching to fill axes more.
        
        x_min_ax, x_max_ax = 0.2, 4.8
        y_min_ax, y_max_ax = 0.2, 4.8
        
        point_cloud_group = mn.VGroup()
        # Halve number of points: 60 -> 30
        for _ in range(60):
            x_ax = rng.uniform(x_min_ax, x_max_ax)
            y_ax = rng.uniform(y_min_ax, y_max_ax)
            pt = axes.c2p(x_ax, y_ax)
            
            # Radius half of current: 0.09 / 2 = 0.045
            dot = mn.Dot(point=pt, radius=0.045, color=PURE_BLUE)
            dot.set_fill(PURE_BLUE, opacity=0.95)
            dot.set_stroke(color=PURE_BLUE, width=0.0)
            dot.set_z_index(6)
            point_cloud_group.add(dot)

        # Kernel
        kernel_span = 6.0
        kernel_image = create_bessel_image(2, 1, resolution=240, span=kernel_span)
        kernel_image.set_width(axes.width * KERNEL_IMAGE_SCALE)
        kernel_image.set_z_index(-1)
        
        kernel_box = mn.Circle(
            radius=axes.width * KERNEL_CIRCLE_RADIUS_SCALE*0.2, 
            color=SCARLET, 
            stroke_width=2.0
        )
        kernel_box.set_z_index(-0.9)
        
        # Simple kernel axes
        k_ax = mn.Axes(
            x_range=[-0.4, 1.5, 1], y_range=[-0.4, 1.5, 1], 
            x_length=kernel_box.width * 0.6, y_length=kernel_box.height * 0.6,
            axis_config={"color": SCARLET, "stroke_width": 2, "include_tip": False, "include_ticks": False},
        )
        # Shift k_ax to center
        k_ax.shift(kernel_box.get_center() - k_ax.c2p(0, 0))
        
        kernel_axes_group = mn.VGroup(k_ax) # Simplified axes group
        kernel_group = mn.Group(kernel_image, kernel_box, kernel_axes_group)
        kernel_group.set_z_index(5)

        highlight_group = mn.VGroup().set_z_index(20)

        bottom_group = mn.Group(axes, axes_labels, point_cloud_group, kernel_group, highlight_group)
        
        refs = {
            'kernel_group': kernel_group,
            'kernel_box': kernel_box,
            'point_cloud': point_cloud_group,
            'highlight_group': highlight_group,
            'axes': axes
        }
        return bottom_group, refs

    def create_bottom_animation(self, refs):
        kernel_group = refs['kernel_group']
        point_cloud = refs['point_cloud']
        axes = refs['axes'] # Used for coordinate check logic (or we can use screen coords)
        
        # Get actual positions of dots (which are already positioned in scene)
        # NOTE: This runs AFTER scene scaling/positioning in construct
        dots = list(point_cloud)
        positions = [d.get_center() for d in dots]
        
        # Find start index (top-left)
        # We need logic to find top-left.
        # Since 'axes' object is also scaled/shifted, using c2p logic might be tricky if we don't account for scale.
        # But we can just use screen coordinates.
        # Top-left of the BUNCH -> min x, max y?
        # User logic was: x in [-0.5, -2.5] relative to axes.
        # Let's just find the point with minimal x + maximal y score?
        # Or stick to logic: x < mean_x and y > mean_y?
        
        # Let's map positions back to axes coords if possible? 
        # Or just pick visually.
        # Simplest: Sort by (y - x) -> max (top-left).
        # positions[i][1] - positions[i][0] is maximized for Top (high y) Left (low x).
        
        scores = [(pos[1] - pos[0], i) for i, pos in enumerate(positions)]
        scores.sort(key=lambda x: x[0], reverse=True)
        start_idx = scores[0][1]
        
        # Calculate Path
        path_indices = [start_idx]
        visited = {start_idx}
        curr = start_idx
        
        while len(visited) < len(positions):
            candidates = []
            for i, pos in enumerate(positions):
                if i in visited: continue
                dist = np.linalg.norm(positions[curr] - pos)
                candidates.append((dist, i))
            
            if not candidates: break
            candidates.sort(key=lambda x: x[0])
            nxt = candidates[0][1]
            path_indices.append(nxt)
            visited.add(nxt)
            curr = nxt
            
        # Build Animation Sequence
        anims = []
        
        # Set start pos immediately
        kernel_group.move_to(positions[start_idx])
        
        for idx in path_indices[1:]:
            target = positions[idx]
            # Use ApplyMethod
            anims.append(
                mn.ApplyMethod(
                    kernel_group.move_to, 
                    target, 
                    run_time=0.4, 
                    rate_func=mn.linear
                )
            )
            anims.append(mn.Wait(0.5))
            
        return mn.Succession(*anims)
