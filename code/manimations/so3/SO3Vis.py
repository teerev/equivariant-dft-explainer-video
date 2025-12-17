from math import factorial
from manim import *
from manim.mobject.three_d.three_dimensions import Surface as ParametricSurface
import numpy as np


# --------------------------------------------------------------
# Real spherical harmonics helper functions
# --------------------------------------------------------------

def associated_legendre(l: int, m: int, x):
    """
    Associated Legendre P_l^m(x), vectorised for numpy arrays.
    Uses standard recursive definition (Numerical Recipes style).
    """
    x = np.array(x)
    m = abs(m)
    if m > l:
        return np.zeros_like(x)

    # P_m^m
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = (1.0 - x) * (1.0 + x)
        fact = 1.0
        for _ in range(1, m + 1):
            pmm *= -fact * np.sqrt(somx2)
            fact += 2.0

    if l == m:
        return pmm

    # P_{m+1}^m
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1

    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll

    return pll


def real_sph_harm(l: int, m: int, theta, phi):
    """
    Real-valued spherical harmonics Y_l^m(θ, φ).

    m > 0:  √2 N_lm P_l^m(cosθ) cos(mφ)
    m < 0:  √2 N_l|m| P_l^|m|(cosθ) sin(|m|φ)
    m = 0:  N_l0 P_l^0(cosθ)
    """
    theta = np.array(theta)
    phi = np.array(phi)
    m_abs = abs(m)

    x = np.cos(theta)
    P = associated_legendre(l, m_abs, x)

    N = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m_abs) / factorial(l + m_abs))

    if m > 0:
        return np.sqrt(2.0) * N * P * np.cos(m * phi)
    elif m < 0:
        return np.sqrt(2.0) * N * P * np.sin(m_abs * phi)
    else:
        return N * P


class RealSphericalHarmonicKernel(ParametricSurface):
    def __init__(
        self,
        l: int = 2,
        m: int = 0,
        radius: float = 1.0,
        opacity: float = 0.2,
        color_neg: str = "#00D5FF",
        color_pos: str = "#F26D00",
        resolution=(48, 48),
        **kwargs,
    ):
        self.l = l
        self.m = m
        self.radius = radius

        # Precompute a rough max |Y_lm| on a grid to normalise the radius
        thetas = np.linspace(1e-3, np.pi - 1e-3, 120)
        phis = np.linspace(0.0, 2 * np.pi, 240)
        th_grid, ph_grid = np.meshgrid(thetas, phis, indexing="ij")
        Y_vals = real_sph_harm(l, m, th_grid, ph_grid)
        maxY = np.max(np.abs(Y_vals))
        if maxY == 0:
            maxY = 1.0
        self.maxY = maxY

        def param_func(u, v):
            # u ∈ [0, 2π] → φ, v ∈ [0, π] → θ
            phi = u
            theta = v
            Y = real_sph_harm(l, m, theta, phi)
            r = radius * np.abs(Y) / self.maxY
            # avoid exact zeros so we don't get degenerate points
            r = np.where(r < 1e-3, 1e-3, r)

            sin_th = np.sin(theta)
            x = r * sin_th * np.cos(phi)
            y = r * sin_th * np.sin(phi)
            z = r * np.cos(theta)
            return np.array([x, y, z])

        super().__init__(
            param_func,
            u_range=[0.0, TAU],
            v_range=[0.0, PI],
            resolution=resolution,
            **kwargs,
        )

        # Overall transparency + no visible stroke mesh
        self.set_fill(WHITE, opacity=opacity)
        self.set_stroke(width=0.0, opacity=0.0)

        # Color lobes by sign of Y_lm(θ, φ)
        def value_func(x, y, z):
            r = np.sqrt(x * x + y * y + z * z)
            # For very small r, define Y = 0
            if np.isscalar(r):
                if r < 1e-6:
                    return 0.0
            else:
                r = np.where(r < 1e-6, 1e-6, r)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)
            phi = np.where(phi < 0, phi + 2 * np.pi, phi)
            return real_sph_harm(l, m, theta, phi)

        # With 2 colors, negative values → color_neg, positive → color_pos
        for face in self.family_members_with_points():
            midpoint = face.get_center()
            val = value_func(*midpoint)
            face.set_color(color_pos if val >= 0 else color_neg)


# --------------------------------------------------------------
# Main scene
# --------------------------------------------------------------

class KernelPointCloud(ThreeDScene):
    def construct(self):
        # --------------------------------------------------------------------
        # 1. CONFIGURATION
        # --------------------------------------------------------------------
        NUM_POINTS      = 100
        POINT_SPREAD    = 3.0
        KERNEL_RADIUS   = 1.2
        NUM_CENTERS     = 15   # number of kernel positions to visit

        # spherical harmonic parameters (tweak these!)
        L_HARM          = 2
        M_HARM          = 0
        HARM_OPACITY    = 0.25  # transparency of the harmonic lobes

        BASE_POINT_COLOR    = BLUE_C
        HIGHLIGHT_COLOR     = GREEN_C
        CENTER_POINT_COLOR  = YELLOW
        EDGE_COLOR          = GREEN_A

        POINT_RADIUS    = 0.05
        EDGE_WIDTH      = 2.5

        np.random.seed(1)

        # --------------------------------------------------------------------
        # 2. AXES + CAMERA
        # --------------------------------------------------------------------
        axes = ThreeDAxes(
            x_range=[-POINT_SPREAD, POINT_SPREAD, 1],
            y_range=[-POINT_SPREAD, POINT_SPREAD, 1],
            z_range=[-POINT_SPREAD, POINT_SPREAD, 1],
        ).set_color(WHITE)

        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.15)

        self.play(Create(axes), run_time=2)

        # --------------------------------------------------------------------
        # 3. POINT CLOUD
        # --------------------------------------------------------------------
        points = np.random.uniform(
            -POINT_SPREAD, POINT_SPREAD, size=(NUM_POINTS, 3)
        )

        dots = VGroup(*[
            Dot3D(
                point=points[i],
                radius=POINT_RADIUS,
                color=BASE_POINT_COLOR,
            )
            for i in range(NUM_POINTS)
        ])

        self.add(dots)

        # --------------------------------------------------------------------
        # 4. REAL SPHERICAL HARMONIC "KERNEL"
        # --------------------------------------------------------------------
        kernel = RealSphericalHarmonicKernel(
            l=L_HARM,
            m=M_HARM,
            radius=KERNEL_RADIUS,
            opacity=HARM_OPACITY,
            color_neg="#00D5FF",
            color_pos="#F26D00",
            resolution=(48, 48),
        )

        # place initially at first point; actual path comes next
        kernel.move_to(points[0])
        self.play(FadeIn(kernel), run_time=1.0)

        # --------------------------------------------------------------------
        # 5. NEIGHBOUR HELPERS
        # --------------------------------------------------------------------
        def neighbor_indices(center_idx):
            """All indices within KERNEL_RADIUS of center_idx (including itself)."""
            center = points[center_idx]
            dists = np.linalg.norm(points - center, axis=1)
            return np.where(dists <= KERNEL_RADIUS + 1e-6)[0]

        def build_kernel_path(num_steps):
            """
            Path of indices: each step tries to move to the nearest unvisited
            neighbour within KERNEL_RADIUS; otherwise jumps to the closest
            unvisited point in the cloud.
            """
            if num_steps <= 0:
                return []

            # start at point closest to origin (deterministic)
            dists_from_origin = np.linalg.norm(points, axis=1)
            current = int(np.argmin(dists_from_origin))
            path = [current]
            visited = {current}

            for _ in range(num_steps - 1):
                current_point = points[current]
                dists = np.linalg.norm(points - current_point, axis=1)

                within_radius = np.where(
                    (dists > 0) & (dists <= KERNEL_RADIUS)
                )[0]
                within_unvisited = [i for i in within_radius if i not in visited]

                if within_unvisited:
                    next_idx = min(within_unvisited, key=lambda i: dists[i])
                else:
                    # no close unvisited neighbours; jump to closest unvisited overall
                    all_unvisited = [i for i in range(NUM_POINTS) if i not in visited]
                    if not all_unvisited:
                        break
                    dists_global = np.linalg.norm(points[all_unvisited], axis=1)
                    next_idx = all_unvisited[int(np.argmin(dists_global))]

                path.append(next_idx)
                visited.add(next_idx)
                current = next_idx

            return path

        center_sequence = build_kernel_path(NUM_CENTERS)

        # --------------------------------------------------------------------
        # 6. ANIMATION LOOP
        # --------------------------------------------------------------------
        current_edges = VGroup()
        current_neighbor_idxs = []

        for center_idx in center_sequence:
            center_point = points[center_idx]
            new_neighbors = list(neighbor_indices(center_idx))

            # edges from center to neighbours
            new_edges = VGroup()
            for j in new_neighbors:
                if j == center_idx:
                    continue
                new_edges.add(
                    Line(
                        start=center_point,
                        end=points[j],
                        color=EDGE_COLOR,
                        stroke_width=EDGE_WIDTH,
                    )
                )

            anims = [kernel.animate.move_to(center_point)]

            if len(current_edges) > 0:
                anims.append(FadeOut(current_edges, run_time=0.3))

            # reset previous neighbour colours
            for j in current_neighbor_idxs:
                anims.append(dots[j].animate.set_color(BASE_POINT_COLOR))

            # highlight new neighbours
            for j in new_neighbors:
                anims.append(dots[j].animate.set_color(HIGHLIGHT_COLOR))

            # emphasise current center
            anims.append(dots[center_idx].animate.set_color(CENTER_POINT_COLOR))

            anims.append(FadeIn(new_edges, run_time=0.3))

            self.play(*anims, run_time=1.0, rate_func=smooth)

            current_edges = new_edges
            current_neighbor_idxs = new_neighbors

        self.wait(2)
