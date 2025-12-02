from manim import *

config.background_color = BLACK

M_VALUES = [-3, -2, -1, 0, 1, 2, 3]   # harmonic orders
HARMONIC_SIGN = -1             # from e^{-i m alpha}
DURATION = 16.0                 # seconds for one full input rotation


class HarmonicFeatureMapBehaviour(Scene):
    def construct(self):
        # --------------------------------------------------------------
        # Top row: input image (left) and example convolved feature map
        # --------------------------------------------------------------
        input_img = ImageMobject("s.png")
        input_img.height = 2.2
        input_frame = SurroundingRectangle(input_img, buff=0.1, color=GREY_A)
        input_label = Tex("Input $X$", font_size=32)
        input_label.next_to(input_frame, UP, buff=0.15)
        input_panel = Group(input_frame, input_img, input_label)

        top_row = Group(input_panel).arrange(RIGHT, buff=1.5)
        top_row.to_edge(UP, buff=1.0)

        eq1 = MathTex(
            r"Y^{\alpha}\bigl(\mathbf{Q}_{\alpha} p\bigr) = D(\alpha)\,Y(p)",
            font_size=32
        )
        
        matrix_content = (
            r"D(\alpha) ="
            r"\begin{pmatrix}"
            r"e^{-3\mathrm{i}\alpha} & 0 & 0 & 0 & 0 & 0 & 0 \\"
            r"0 & e^{-2\mathrm{i}\alpha} & 0 & 0 & 0 & 0 & 0 \\"
            r"0 & 0 & e^{-\mathrm{i}\alpha} & 0 & 0 & 0 & 0 \\"
            r"0 & 0 & 0 & 1 & 0 & 0 & 0 \\"
            r"0 & 0 & 0 & 0 & e^{\mathrm{i}\alpha} & 0 & 0 \\"
            r"0 & 0 & 0 & 0 & 0 & e^{2\mathrm{i}\alpha} & 0 \\"
            r"0 & 0 & 0 & 0 & 0 & 0 & e^{3\mathrm{i}\alpha}"
            r"\end{pmatrix}"
        )
        eq2 = MathTex(matrix_content, font_size=20)
        
        equation = VGroup(eq1, eq2).arrange(DOWN, buff=0.2)
        equation.next_to(top_row, RIGHT, buff=0.4)

        # --------------------------------------------------------------
        # Bottom: grid of Re/Im panels for m = -2,-1,0,1,2
        # --------------------------------------------------------------
        columns = []
        feature_entries = []  # will hold (img_re, img_im, m)

        base_tile_img = ImageMobject("s_conv.png")
        base_tile_img.height = 1.0

        for m in M_VALUES:
            # --- Real part tile ---
            img_re = base_tile_img.copy()
            frame_re = SurroundingRectangle(img_re, buff=0.06, color=GREY_B)
            label_re = Tex(r"$\Re$", font_size=26)
            label_re.next_to(frame_re, UP, buff=0.05)
            re_tile = Group(frame_re, img_re, label_re)

            # --- Imag part tile (90° phase offset) ---
            img_im = base_tile_img.copy()
            img_im.rotate(PI / 2)
            frame_im = SurroundingRectangle(img_im, buff=0.06, color=GREY_B)
            label_im = Tex(r"$\Im$", font_size=26)
            label_im.next_to(frame_im, UP, buff=0.05)
            im_tile = Group(frame_im, img_im, label_im)

            # Stack Re over Im
            column_core = Group(re_tile, im_tile).arrange(DOWN, buff=0.4)

            # m label
            m_label = MathTex(rf"m={m}", font_size=28)
            m_label.next_to(column_core, DOWN, buff=0.15)

            full_column = Group(column_core, m_label)
            columns.append(full_column)

            # store **images only** for rotation
            feature_entries.append((img_re, img_im, m))

        bottom_grid = Group(*columns).arrange(RIGHT, buff=0.4)
        bottom_grid.to_edge(DOWN, buff=0.3)

        # Shift everything left by one column width
        # Approx width of one column = image width + buff
        # Image height 1.0 -> width depends on aspect ratio but likely similar
        # Let's assume aspect ratio ~1.0 for square tiles, plus padding.
        # Visually shifting LEFT * 1.5 should be safe based on layout.
        
        # User requested to move image and equation to the left a bit more
        # Original shift was LEFT * 1.8
        # I will shift top_row and equation further left by adding an extra shift
        
        shift_vec = LEFT * 1.8
        top_row_extra_shift = LEFT * 1.1 # Tweak this value to move image/equation further left
        
        top_row.shift(shift_vec + top_row_extra_shift)
        equation.next_to(top_row, RIGHT, buff=0.4)
        bottom_grid.shift(shift_vec)

        # --------------------------------------------------------------
        # Add everything before attaching updaters
        # --------------------------------------------------------------
        self.add(equation, top_row, bottom_grid)

        # --------------------------------------------------------------
        # Updater factory: rotate about a fixed centre
        # --------------------------------------------------------------
        # Four full rotations (4 * TAU) over the same DURATION
        omega_input = 4 * TAU / DURATION  
        
        def make_rot_updater(omega, center):
            def updater(mob, dt):
                mob.rotate(omega * dt, about_point=center)
            return updater

        # Input rotates once over DURATION, frame + label stay still
        rotate_input = True
        if rotate_input:
            input_center = input_img.get_center()
            # Reverse direction of input rotation (-omega_input)
            input_img.add_updater(make_rot_updater(-omega_input, input_center))

        # Each harmonic: -m × input frequency (e^{-i m alpha}),
        # images spin inside static frames.
        for img_re, img_im, m in feature_entries:
            omega_m = HARMONIC_SIGN * m * omega_input
            center_re = img_re.get_center()
            center_im = img_im.get_center()
            img_re.add_updater(make_rot_updater(omega_m, center_re))
            img_im.add_updater(make_rot_updater(omega_m, center_im))

        # --------------------------------------------------------------
        # Let it run
        # --------------------------------------------------------------
        self.wait(DURATION)

        # Optional: stop everything at the end
        if rotate_input:
            input_img.clear_updaters()
        for img_re, img_im, m in feature_entries:
            img_re.clear_updaters()
            img_im.clear_updaters()
        self.wait(1)
