from manim import *
import numpy as np


class KernelPointCloud(ThreeDScene):
    def construct(self):
        # --------------------------------------------------------------------
        # 1. CONFIGURATION
        # --------------------------------------------------------------------
        NUM_POINTS      = 100
        POINT_SPREAD    = 3.0
        KERNEL_RADIUS   = 1.2
        NUM_CENTERS     = 15   # number of kernel positions to visit

        BASE_POINT_COLOR    = BLUE_C
        HIGHLIGHT_COLOR     = GREEN_C
        CENTER_POINT_COLOR  = YELLOW
        EDGE_COLOR          = GREEN_A
        KERNEL_COLOR        = GREEN_C

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
        # 4. TRANSPARENT KERNEL SPHERE (no wireframe)
        # --------------------------------------------------------------------
        kernel_sphere = Sphere(
            center=ORIGIN,
            radius=KERNEL_RADIUS,
            resolution=(48, 48),
        )
        # translucent shell, stroke disabled so no mesh lines
        kernel_sphere.set_fill(KERNEL_COLOR, opacity=0.15)
        kernel_sphere.set_stroke(KERNEL_COLOR, width=0.0, opacity=0.0)

        # place initially at first point; actual path comes next
        kernel_sphere.move_to(points[0])
        self.play(FadeIn(kernel_sphere), run_time=1.0)

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

            anims = [kernel_sphere.animate.move_to(center_point)]

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
