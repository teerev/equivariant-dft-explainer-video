from manim import *
import numpy as np
import random


class ConstrainedHypothesesBranchingValleys(Scene):
    """
    No outlines. A light-blue hypothesis-space blob.
    Then a highly-branching, programmatically-generated "valley subset" appears,
    highlighted as filled electric-blue ribbons (a thin branching manifold inside the blob).

    Target runtime: ~7–9 seconds.
    """

    # -----------------------
    # Geometry: blob + helpers
    # -----------------------
    def make_blob_polygon(self, R=2.55, n=28, jitter=0.18, seed=1):
        """
        Deterministic blobby polygon (points only). We'll fill it and remove outlines.
        """
        rng = random.Random(seed)
        pts = []
        for k in range(n):
            th = TAU * k / n
            r = R * (1.0 + jitter * rng.uniform(-1, 1))
            # Slight anisotropy so it reads like a "space" not a circle
            x = 1.05 * r * np.cos(th)
            y = 0.85 * r * np.sin(th)
            pts.append(np.array([x, y, 0.0]))
        return pts

    def point_in_poly(self, p, poly_pts):
        """
        Ray-casting point-in-polygon for 2D points.
        poly_pts: list of np arrays (x,y,0).
        p: np array (x,y,0).
        """
        x, y = p[0], p[1]
        inside = False
        n = len(poly_pts)
        for i in range(n):
            x0, y0 = poly_pts[i][0], poly_pts[i][1]
            x1, y1 = poly_pts[(i + 1) % n][0], poly_pts[(i + 1) % n][1]
            # Check edge crosses horizontal ray to +inf
            cond = ((y0 > y) != (y1 > y))
            if cond:
                x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
                if x_int > x:
                    inside = not inside
        return inside

    def ribbon_from_points(self, pts, half_width):
        """
        Create a filled ribbon polygon around a polyline, with NO outlines.
        """
        pts = [np.array(p) for p in pts]
        if len(pts) < 2:
            return VMobject()

        # Tangents
        tangents = []
        for i in range(len(pts)):
            if i == 0:
                t = pts[1] - pts[0]
            elif i == len(pts) - 1:
                t = pts[-1] - pts[-2]
            else:
                t = pts[i + 1] - pts[i - 1]
            norm = np.linalg.norm(t)
            if norm < 1e-9:
                t = np.array([1.0, 0.0, 0.0])
            else:
                t = t / norm
            tangents.append(t)

        # Normals
        normals = []
        for t in tangents:
            n = np.array([-t[1], t[0], 0.0])
            n /= (np.linalg.norm(n) + 1e-9)
            normals.append(n)

        upper = [p + half_width * n for p, n in zip(pts, normals)]
        lower = [p - half_width * n for p, n in zip(pts, normals)][::-1]

        poly = VMobject()
        poly.set_points_as_corners(upper + lower + [upper[0]])
        poly.close_path()
        poly.set_stroke(width=0, opacity=0)  # no outline
        return poly

    # -----------------------
    # Branch generator (programmatic)
    # -----------------------
    def generate_branching_polylines(
        self,
        blob_poly_pts,
        seed=7,
        max_depth=4,
        base_step=0.22,
        trunk_steps=12,
        branch_prob=0.70,
        max_children=3,
    ):
        """
        Returns a list of polylines (each is a list of points) representing a branching valley structure.

        Strategy:
        - Grow a trunk from near the bottom toward the top with slight curvature noise.
        - Recursively spawn branches along the way.
        - Reject steps that leave the blob polygon.
        Deterministic via seed.
        """
        rng = random.Random(seed)

        polylines = []

        # Start near bottom center
        start = np.array([0.0, -1.85, 0.0])
        # Initial direction mostly upward
        dir0 = np.array([0.0, 1.0, 0.0])

        def jitter_dir(d, ang_sigma=0.35):
            """
            Rotate 2D direction by a small random angle.
            """
            ang = rng.uniform(-ang_sigma, ang_sigma)
            c, s = np.cos(ang), np.sin(ang)
            x, y = d[0], d[1]
            out = np.array([c * x - s * y, s * x + c * y, 0.0])
            norm = np.linalg.norm(out)
            return out / (norm + 1e-12)

        def grow_path(p0, d0, steps, step_scale, depth):
            """
            Grow one polyline; return list of points.
            """
            pts = [p0.copy()]
            p = p0.copy()
            d = d0.copy()
            for _ in range(steps):
                d = jitter_dir(d, ang_sigma=0.18 + 0.05 * depth)
                step = step_scale * (0.85 + 0.35 * rng.random())
                cand = p + step * d

                # keep inside the blob; if outside, try a few alternative jitters
                ok = self.point_in_poly(cand, blob_poly_pts)
                tries = 0
                while not ok and tries < 10:
                    d = jitter_dir(d, ang_sigma=0.65)
                    cand = p + step * d
                    ok = self.point_in_poly(cand, blob_poly_pts)
                    tries += 1
                if not ok:
                    break

                p = cand
                pts.append(p.copy())
            return pts

        def spawn(p, d, depth):
            if depth > max_depth:
                return

            # Shorter and thinner as depth increases
            steps = max(6, int(trunk_steps * (0.60 ** depth)))
            step_scale = base_step * (0.92 ** depth)

            pts = grow_path(p, d, steps=steps, step_scale=step_scale, depth=depth)
            if len(pts) >= 4:
                polylines.append(pts)

            # Choose multiple spawn points along this path for richer branching
            if depth < max_depth:
                candidate_indices = list(range(max(2, len(pts)//4), len(pts)-2, max(2, len(pts)//6)))
                rng.shuffle(candidate_indices)
                # spawn up to a few children
                num_spawn_sites = min(3, len(candidate_indices))
                for si in candidate_indices[:num_spawn_sites]:
                    if rng.random() > branch_prob:
                        continue

                    parent = pts[si]
                    # approximate local direction from neighboring points
                    local = pts[si + 1] - pts[si - 1]
                    local /= (np.linalg.norm(local) + 1e-12)

                    children = 1 + (1 if rng.random() < 0.55 else 0) + (1 if rng.random() < 0.25 else 0)
                    children = min(children, max_children)

                    # Spread angles for children
                    base_angles = []
                    if children == 1:
                        base_angles = [rng.uniform(-0.9, 0.9)]
                    elif children == 2:
                        base_angles = [rng.uniform(-1.1, -0.25), rng.uniform(0.25, 1.1)]
                    else:
                        base_angles = [rng.uniform(-1.2, -0.55), rng.uniform(-0.35, 0.35), rng.uniform(0.55, 1.2)]

                    for a in base_angles:
                        c, s = np.cos(a), np.sin(a)
                        x, y = local[0], local[1]
                        child_dir = np.array([c * x - s * y, s * x + c * y, 0.0])
                        child_dir /= (np.linalg.norm(child_dir) + 1e-12)
                        spawn(parent, child_dir, depth + 1)

        # Grow a trunk first, then branch recursively
        trunk = grow_path(start, dir0, steps=trunk_steps, step_scale=base_step * 1.05, depth=0)
        if len(trunk) >= 6:
            polylines.append(trunk)

        # Spawn from multiple points along the trunk for rich branching
        trunk_indices = list(range(len(trunk)//4, len(trunk)-3, 3))
        rng.shuffle(trunk_indices)
        for idx in trunk_indices[:6]:
            p = trunk[idx]
            local = trunk[idx + 1] - trunk[idx - 1]
            local /= (np.linalg.norm(local) + 1e-12)
            # two primary branch directions from trunk
            for a in [rng.uniform(-1.05, -0.35), rng.uniform(0.35, 1.05)]:
                c, s = np.cos(a), np.sin(a)
                x, y = local[0], local[1]
                d = np.array([c * x - s * y, s * x + c * y, 0.0])
                d /= (np.linalg.norm(d) + 1e-12)
                spawn(p, d, depth=1)

        return polylines

    # -----------------------
    # Scene
    # -----------------------
    def construct(self):
        LIGHT_BLUE_FILL = "#9ED8FF"      # light blue blob
        ELECTRIC_BLUE = "#00B8FF"        # electric blue valley subset

        # Top annotations
        label_all = Tex("All possible hypotheses", color=LIGHT_BLUE_FILL).scale(0.75)
        label_all.to_edge(UP, buff=0.35)
        label_valid = Tex("Physically valid hypotheses", color=ELECTRIC_BLUE).scale(0.75)
        label_valid.next_to(label_all, DOWN, buff=0.15)

        # Blob (filled only; no outline)
        blob_pts = self.make_blob_polygon(R=2.60, n=30, jitter=0.17, seed=2)
        blob = VMobject()
        blob.set_points_smoothly(blob_pts + [blob_pts[0]])
        blob.close_path()
        blob.set_fill(LIGHT_BLUE_FILL, opacity=0.30)
        blob.set_stroke(width=0, opacity=0)

        # Generate a more richly branching valley structure (programmatic)
        polylines = self.generate_branching_polylines(
            blob_poly_pts=blob_pts,
            seed=11,
            max_depth=6,
            base_step=0.23,
            trunk_steps=12,
            branch_prob=0.45,
            max_children=3,
        )

        # Convert polylines to filled ribbons; widths taper with depth by length heuristic
        ribbons = VGroup()
        for pts in polylines:
            # thinner for shorter branches
            L = np.sum(np.linalg.norm(np.diff(np.array(pts)[:, :2], axis=0), axis=1))
            half_w = 0.10
            if L < 1.2:
                half_w = 0.045
            elif L < 2.0:
                half_w = 0.060
            elif L < 3.0:
                half_w = 0.075
            else:
                half_w = 0.090

            rib = self.ribbon_from_points(pts, half_width=half_w)
            # Use full opacity so overlaps don't "add up" in brightness.
            rib.set_fill(ELECTRIC_BLUE, opacity=1.0)
            rib.set_stroke(width=0, opacity=0)  # no outline
            ribbons.add(rib)

        # --- animation (~21s) ---
        # 1) Show hypothesis space (blob + label) first
        self.play(FadeIn(blob, scale=1.01), FadeIn(label_all), run_time=0.8)
        # Total time for this first state: ~10s
        self.wait(9.2)

        # 2) After ~10s, reveal physically valid subset + its label
        self.play(FadeIn(label_valid), FadeIn(ribbons), run_time=1.2)
        # Hold until end: ~11s after the reveal starts
        self.wait(9.8)
