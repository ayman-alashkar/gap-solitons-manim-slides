"""
Opening scene: how the water-wave problem is set up, ideas only.

    A. A floating ball: the crest crosses the tank while the ball only goes round a
       loop (the wave travels, the water does not). Then dye blobs beneath, one at a
       time: smaller loops deeper, sliding at the bottom, never turning.
       Optional (SHOW_DRIFT): a particle moved by the flow at its own position creeps
       forward each loop: Stokes drift, the next order of the same expansion.
       Then the Eulerian view: fixed points.
    B. Assumptions, each shown on the parcels: area stays the same (no compression),
       no friction, cabins don't turn (no vorticity) -> the flow is the slope of one
       landscape phi -> inside the water phi is the average of its neighbours (Laplace).
    C. Boundary conditions: bottom z = -h(x) (no flow through it; flat is the special
       case), surface (KBC, DBC from Bernoulli) on the unknown surface.
    D. Small waves: Taylor to the flat level, then order by order (Stokes).

Wave used for the motion: linear theory at finite depth (g = 1, kh = 1.6):
    omega^2 = g k tanh(kh),  theta = kx - omega t
    eta = a cos(theta)
    phi = (a omega / k) cosh(k(z+h)) / sinh(kh) * sin(theta)
    u = phi_x = a omega cosh(k(z+h))/sinh(kh) cos(theta)
    w = phi_z = a omega sinh(k(z+h))/sinh(kh) sin(theta)
    parcel at (x0, z0):  x = x0 - a cosh(k(z0+h))/sinh(kh) sin(theta0)
                         z = z0 + a sinh(k(z0+h))/sinh(kh) cos(theta0)
    -> ellipses; vertical size 0 at the bottom. To this order a parcel does not rotate.
Equal scale in x and z on screen, so arrow directions and ellipse shapes are true.
Amplitude a is exaggerated for visibility.

Part C, rippled bottom: arrows there are schematic (flat-bottom speed, drawn along
the local bottom slope) to show "flow slides along the bottom".

Two versions in this file:
    FluidSetup      full version (~80 s): parcels, Eulerian view, all three assumptions,
                    phi, Laplace, bottom, surface, Taylor, orders
    FluidSetupLite  short version for a general audience (~45 s): parcels, phi,
                    bottom, surface, orders. One caption per beat; few math tags.

Pauses ("beats"): by default each beat is a short hold (PAUSE seconds) in a normal
video. Optional click-through: set USE_SLIDES = True, then
                  manim-slides render fluid_setup_scene.py FluidSetup
                  manim-slides FluidSetup
(needs manim-slides and Manim versions that work together; 5.7 + 0.22 do not).

Render:   manim -pql fluid_setup_scene.py FluidSetup        (full, preview)
          manim -pql fluid_setup_scene.py FluidSetupLite    (short, preview)
          use -pqh instead of -pql for the final quality
"""

import numpy as np
from manim import *

# Click-through slides are OPT-IN. manim-slides 5.7 does not work with Manim 0.22
# (it crashes on import of the scene), so leave this False unless your versions match.
USE_SLIDES = False
HAS_SLIDES = False
BaseScene = Scene
if USE_SLIDES:
    try:
        from manim_slides import Slide as BaseScene
        HAS_SLIDES = True
    except Exception:
        BaseScene = Scene

SHOW_MATH = True
SHOW_DRIFT = True                      # optional beat: loops creep forward (Stokes drift)
PAUSE = 1.5                            # hold at each beat when not using manim-slides
MATH = "#9FC9FF"
BALL = "#F5A623"
CORAL = "#D85A30"
TEAL = "#1D9E75"
DYE = "#E8C547"

G, H = 1.0, 3.0
K = 1.6 / H                                   # kh = 1.6
W = np.sqrt(G * K * np.tanh(K * H))
A = 0.5
LAM = 2 * np.pi / K
X_RANGE = (0.0, LAM)                          # one wavelength
Z_RANGE = (-H - 0.4, 0.9)                     # room below for the sandbars
X_LEN = 12.0
Z_LEN = X_LEN * (Z_RANGE[1] - Z_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])   # equal scale


def math_tag(markup, size=20):
    return MarkupText(markup, font_size=size, color=MATH)


def live(draw):
    """A redrawn (moving) object that can fade in/out WITHOUT pausing its motion.
    Manim's FadeIn/FadeOut pause a live object's updates during the fade, which makes
    it freeze and then jump. Here the transparency is applied inside the redraw."""
    alpha = ValueTracker(1.0)

    def redraw():
        m = draw()
        a = alpha.get_value()
        if a < 1:
            m.fade(1 - a)
        return m

    mob = always_redraw(redraw)
    mob.alpha = alpha
    return mob


def orbit_sizes(z0):
    return (A * np.cosh(K * (z0 + H)) / np.sinh(K * H),
            A * np.sinh(K * (z0 + H)) / np.sinh(K * H))


class FluidSetup(BaseScene):
    def beat(self):
        """A pause for talking: a click with manim-slides, otherwise a short hold."""
        if HAS_SLIDES:
            self.next_slide()
        else:
            self.wait(PAUSE)

    def appear(self, *mobs):
        """Animations that fade live objects in while they keep moving."""
        for m in mobs:
            m.alpha.set_value(0)
            self.add(m)
        return [m.alpha.animate.set_value(1) for m in mobs]

    def vanish(self, *mobs):
        """Animations that fade live objects out while they keep moving."""
        return [m.alpha.animate.set_value(0) for m in mobs]

    def clear_all(self, keep):
        """Fade everything except `keep`; live objects fade without freezing."""
        lives = [m for m in self.mobjects if hasattr(m, "alpha")]
        others = [m for m in self.mobjects
                  if m is not keep and not hasattr(m, "alpha") and not isinstance(m, ValueTracker)]
        self.play(*self.vanish(*lives), *[FadeOut(m) for m in others])
        self.remove(*lives)

    def live_landscape(self, t):
        alpha = ValueTracker(1.0)
        mob = always_redraw(lambda: self.phi_image(t.get_value(), alpha.get_value()))
        mob.alpha = alpha
        return mob

    def construct(self):
        ax = Axes(x_range=[*X_RANGE], y_range=[*Z_RANGE], x_length=X_LEN, y_length=Z_LEN,
                  tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0}
                  ).move_to([0, -1.15, 0])
        self.ax = ax
        t = ValueTracker(0.0)
        self.add(t)                       # wave driven step by step in part A
        bump = ValueTracker(0.0)          # 0 = flat bottom, 1 = rippled bottom

        xs = np.linspace(*X_RANGE, 300)

        def theta(x):
            return K * x - W * t.get_value()

        def eta(x):
            return A * np.cos(theta(x))

        def f_bed(x):                     # sandbars: z = -H + f(x)
            return 0.3 * np.sin(3 * 2 * np.pi * x / LAM)

        def bed(x):
            return -H + bump.get_value() * f_bed(x)

        water = live(lambda: Polygon(
            *[ax.c2p(x, eta(x)) for x in xs], *[ax.c2p(x, bed(x)) for x in xs[::-1]],
            stroke_width=0, fill_color=BLUE_E, fill_opacity=0.45))
        surface = live(lambda: VMobject(stroke_color=BLUE_B, stroke_width=3)
                                .set_points_smoothly([ax.c2p(x, eta(x)) for x in xs]))
        bottom = live(lambda: VMobject(stroke_color=GREY_B, stroke_width=3)
                               .set_points_smoothly([ax.c2p(x, bed(x)) for x in xs]))

        # ================= A. a floating ball, then the water beneath =================
        title = Text("How we describe moving water", font_size=32).to_edge(UP)
        self.play(Write(title))
        self.play(*self.appear(water, surface, bottom))

        T = 2 * np.pi / W                       # one wave period (simulation time)
        x0 = LAM / 2
        cap = None

        def say(text, color=WHITE):
            """Replace the caption under the title (one caption at a time)."""
            nonlocal cap
            new = Text(text, font_size=22, color=color).next_to(title, DOWN, buff=0.25)
            if cap:                                  # old caption out first: no overlap
                self.play(FadeOut(cap), run_time=0.4)
            self.play(FadeIn(new), run_time=0.5)
            cap = new

        def advance(periods, run_time, *anims):
            """Move the wave forward by a whole number of periods (no jumps)."""
            self.play(t.animate.increment_value(periods * T), *anims,
                      run_time=run_time, rate_func=linear)

        # surface particle orbit: horizontal A/tanh(kh), vertical A
        def ball_pos():
            th = K * x0 - W * t.get_value()
            return ax.c2p(x0 - A / np.tanh(K * H) * np.sin(th), A * np.cos(th))

        ball = live(lambda: Circle(radius=0.14, stroke_color=WHITE, stroke_width=1.5,
                                            fill_color=BALL, fill_opacity=1).move_to(ball_pos()))

        # marker riding one crest: crest at x = omega t / k, crosses the tank in one period
        def crest_marker():
            xc = W * t.get_value() / K
            tip = ax.c2p(xc, A + 0.12)
            tri = Triangle(color=WHITE, fill_opacity=1, stroke_width=0).scale(0.09).rotate(PI)
            tri.move_to(tip + UP * 0.1)
            lab = Text("crest", font_size=15).next_to(tri, UP, buff=0.05)
            return VGroup(tri, lab)

        marker = live(crest_marker)
        self.play(*self.appear(ball, marker))
        say("watch the ball, and the crest of the wave", BALL)
        self.beat()

        trail = TracedPath(ball.get_center, stroke_color=BALL, stroke_width=2)
        self.add(trail)
        advance(1, 8)                           # crest crosses the whole tank
        self.play(*self.vanish(marker))
        say("the wave moved on; the ball only went round a small loop", BALL)
        self.beat()

        # the water beneath: dye blobs, one at a time, top to bottom
        depths = [-1.0, -1.6, -2.2, -2.8]       # the last one sits just above the bed
        size = 0.3

        def parcel(z0):
            # four corners, each moved by its own orbit -> true shape and orientation
            def draw():
                pts = []
                for dx, dz in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    xa, za = x0 + dx * size / 2, z0 + dz * size / 2
                    ox, oz = orbit_sizes(za)
                    th = K * xa - W * t.get_value()
                    pts.append(ax.c2p(xa - ox * np.sin(th), za + oz * np.cos(th)))
                sq = Polygon(*pts, stroke_color=DYE, stroke_width=2,
                             fill_color=DYE, fill_opacity=0.75)
                mark = Line(sq.get_center(), (pts[1] + pts[2]) / 2, color=BLACK, stroke_width=3)
                return VGroup(sq, mark)
            return live(draw)

        def orbit_path(z0, color=DYE):
            ox, oz = orbit_sizes(z0)
            s_ = np.linspace(0, 2 * np.pi, 80)
            return VMobject(stroke_color=color, stroke_width=1.5, stroke_opacity=0.6
                            ).set_points_smoothly([ax.c2p(x0 + ox * np.cos(a), z0 + oz * np.sin(a))
                                                   for a in s_])

        blobs = [parcel(z) for z in depths]
        blob_paths = [orbit_path(z) for z in depths]
        ball_path = orbit_path(0.0, BALL)

        def blob_centre(z0):
            ox, oz = orbit_sizes(z0)
            return lambda: ax.c2p(x0 - ox * np.sin(K * x0 - W * t.get_value()),
                                  z0 + oz * np.cos(K * x0 - W * t.get_value()))

        say("the water underneath does the same: imagine small blobs of dye", DYE)
        self.remove(trail)                      # swap the growing trail for the same,
        self.add(ball_path)                     # fixed loop (no visible change)
        blob_trails = []
        for b, z0 in zip(blobs, depths):        # each blob draws its own loop as it moves
            tr = TracedPath(blob_centre(z0), stroke_color=DYE, stroke_width=1.5,
                            stroke_opacity=0.6)
            blob_trails.append(tr)
            self.add(tr)
            advance(0.5, 3, *self.appear(b))
        advance(1, 7)                           # every blob has now done a full loop
        self.remove(*blob_trails)               # swap trails for identical fixed loops
        self.add(*blob_paths)
        say("smaller loops deeper down; at the bottom the water only slides")
        advance(1, 7)
        self.beat()
        say("and the blobs never turn (watch the black ticks)")
        advance(1, 7)
        self.beat()


        parcels = VGroup(ball, *blobs)
        paths = VGroup(ball_path, *blob_paths)

        if SHOW_DRIFT:
            self.drift_beat(ax, t, say, advance, T, parcels, paths)
        cap1 = cap

        # from here on the wave runs continuously
        t.add_updater(lambda m, dt: m.increment_value(1.2 * dt))

        # Eulerian view
        grid_x = np.linspace(X_RANGE[0] + 0.9, X_RANGE[1] - 0.9, 9)
        grid_z = [-0.75, -1.45, -2.15, -2.75]

        def velocity(x, z):
            th = theta(x)
            return (A * W * np.cosh(K * (z + H)) / np.sinh(K * H) * np.cos(th),
                    A * W * np.sinh(K * (z + H)) / np.sinh(K * H) * np.sin(th))

        def arrows():
            g = VGroup()
            for xg in grid_x:
                for zg in grid_z:
                    if abs(xg - x0) < 1.0:
                        continue                        # keep the parcel column clear
                    u, w = velocity(xg, zg)
                    s = 3.0                             # display scale
                    g.add(Arrow(ax.c2p(xg, zg), ax.c2p(xg + s * u, zg + s * w), buff=0,
                                color=WHITE, stroke_width=2, tip_length=0.1,
                                max_tip_length_to_length_ratio=0.4))
                    g.add(Dot(ax.c2p(xg, zg), radius=0.025, color=GREY_A))
            return g

        field = live(arrows)
        cap3 = Text("or stand still and record the flow passing each point (Eulerian)",
                    font_size=22).next_to(title, DOWN, buff=0.25)
        self.play(FadeOut(cap1), run_time=0.4)
        self.play(FadeIn(cap3), *self.appear(field))
        self.wait(3)
        if SHOW_MATH:
            tag = math_tag("D/Dt = ∂/∂t + <b>u</b>·∇   links the two views", 17)
            tag.next_to(ax, DOWN, buff=0.15)
            self.play(FadeIn(tag))
            self.wait(2.5)
            self.play(FadeOut(tag))
        self.play(*self.vanish(field), FadeOut(cap3))

        # ================= B. assumptions =================
        chips = [("no compression", "parcels keep their area", "∇·<b>u</b> = 0"),
                 ("no friction", "nothing slows the flow", "inviscid"),
                 ("no vorticity", "parcels never turn, like Ferris-wheel cabins", "∇×<b>u</b> = 0")]
        chip_mobs = VGroup()
        for main, plain, m in chips:
            parts = [Text(main, font_size=19, weight=BOLD), Text(plain, font_size=14, color=GREY_A)]
            if SHOW_MATH:
                parts.append(math_tag(m, 14))
            chip_mobs.add(VGroup(*parts).arrange(DOWN, buff=0.05))
        chip_mobs.arrange(RIGHT, buff=0.6).next_to(title, DOWN, buff=0.2)
        for c in chip_mobs:
            self.play(FadeIn(c, shift=DOWN * 0.15), run_time=0.7)
            self.wait(1.5)

        # phi as a landscape: colour map + arrows (its slope)
        self.play(FadeOut(chip_mobs))
        landscape = self.live_landscape(t)
        cap4 = VGroup(Text("no vorticity means all the arrows are the slope of ONE landscape φ",
                           font_size=20),
                      Text("(colour = height of φ; the water flows uphill on it)",
                           font_size=15, color=GREY_A)
                      ).arrange(DOWN, buff=0.06).next_to(title, DOWN, buff=0.2)
        self.play(*self.vanish(*parcels), FadeOut(paths), *self.appear(landscape), FadeIn(cap4))
        self.add(surface)                       # keep the surface line on top
        field = live(arrows)
        self.play(*self.appear(field))
        if SHOW_MATH:
            tag = math_tag("<b>u</b> = ∇φ     (φ: velocity potential)", 17).next_to(ax, DOWN, buff=0.15)
            self.play(FadeIn(tag))
        self.wait(4)

        cap5 = VGroup(Text("no compression: what flows into a spot flows out", font_size=20),
                      Text("→ every point of φ is the average of its neighbours: no bumps inside",
                           font_size=17, color=YELLOW)
                      ).arrange(DOWN, buff=0.06).next_to(title, DOWN, buff=0.2)
        self.play(FadeOut(cap4), run_time=0.4)
        self.play(FadeIn(cap5))
        if SHOW_MATH:
            tag2 = math_tag("∇²φ = 0   inside the water   (Laplace)", 17).next_to(ax, DOWN, buff=0.15)
            self.play(Transform(tag, tag2))
        self.wait(4)
        self.play(FadeOut(cap5), *self.vanish(field, landscape),
                  *([FadeOut(tag)] if SHOW_MATH else []))

        # ================= C. boundary conditions =================
        bottom_glow = always_redraw(lambda: VMobject(stroke_color=TEAL, stroke_width=6)
                                    .set_points_smoothly([ax.c2p(x, bed(x)) for x in xs]))
        bot_lab = VGroup(Text("bottom: no water flows through it", font_size=20, color=TEAL),
                         math_tag("φ<sub>z</sub> = 0  at  z = −h", 16) if SHOW_MATH else VGroup()
                         ).arrange(RIGHT, buff=0.4).next_to(ax, DOWN, buff=0.15)
        self.play(FadeIn(bottom_glow), FadeIn(bot_lab))
        self.wait(2)

        # real seabeds are not flat: z = -h(x)
        t.clear_updaters()                                 # freeze the wave
        bot_lab2 = VGroup(Text("a real seabed: z = −h(x)", font_size=20, color=TEAL),
                          math_tag("φ<sub>z</sub> + h<sub>x</sub>φ<sub>x</sub> = 0  at  z = −h(x)", 16)
                          if SHOW_MATH else VGroup()
                          ).arrange(RIGHT, buff=0.4).next_to(ax, DOWN, buff=0.15)
        self.play(bump.animate.set_value(1), Transform(bot_lab, bot_lab2), run_time=2)

        def slide_arrows():
            g = VGroup()
            for xb in np.linspace(1.0, LAM - 1.0, 12):
                u_b = A * W / np.sinh(K * H) * np.cos(theta(xb))    # flat-bottom speed
                slope = bump.get_value() * 0.3 * 3 * 2 * np.pi / LAM * np.cos(3 * 2 * np.pi * xb / LAM)
                d = np.array([1.0, slope]) / np.hypot(1.0, slope)
                zb = bed(xb) + 0.12
                L = 3.0 * u_b
                g.add(Arrow(ax.c2p(xb, zb), ax.c2p(xb + L * d[0], zb + L * d[1]), buff=0,
                            color=WHITE, stroke_width=2, tip_length=0.1,
                            max_tip_length_to_length_ratio=0.4))
            return g

        slide = slide_arrows()
        flat_note = Text("the flow slides along it; a flat bottom is the special case h_x = 0",
                         font_size=17, color=GREY_A).next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(slide), FadeIn(flat_note))
        self.wait(3)
        self.play(FadeOut(slide), FadeOut(flat_note))

        # surface
        surf_glow = VMobject(stroke_color=CORAL, stroke_width=6).set_points_smoothly(
            [ax.c2p(x, eta(x)) for x in xs])
        kbc = VGroup(Text("surface moves with the water", font_size=19, color=CORAL),
                     math_tag("η<sub>t</sub> + φ<sub>x</sub>η<sub>x</sub> = φ<sub>z</sub>", 16)
                     if SHOW_MATH else VGroup()).arrange(DOWN, buff=0.05)
        dbc = VGroup(Text("pressure there = air pressure (Bernoulli)", font_size=19, color=CORAL),
                     math_tag("φ<sub>t</sub> + ½|∇φ|² + gη = 0", 16)
                     if SHOW_MATH else VGroup()).arrange(DOWN, buff=0.05)
        surf_lab = VGroup(kbc, dbc).arrange(RIGHT, buff=1.0).next_to(title, DOWN, buff=0.2)
        on_eta = Text("both applied at z = η, the surface itself", font_size=16, color=GREY_A)
        on_eta.next_to(surf_lab, DOWN, buff=0.08)
        self.play(Create(surf_glow), FadeIn(surf_lab), FadeIn(on_eta))
        self.wait(3)

        hard = Text("nonlinear, and on a surface we don't know yet  →  no exact solution",
                    font_size=21, color=YELLOW)
        hard_bg = BackgroundRectangle(hard, fill_opacity=0.75, buff=0.15)
        VGroup(hard_bg, hard).move_to(ax.c2p(LAM / 2, -1.4))
        self.play(FadeIn(hard_bg), FadeIn(hard))
        self.wait(3)

        # ================= D. small waves: Taylor =================
        self.play(FadeOut(hard_bg), FadeOut(hard), FadeOut(surf_lab), FadeOut(on_eta))
        flat = DashedLine(ax.c2p(X_RANGE[0], 0), ax.c2p(X_RANGE[1], 0), color=WHITE)
        idea = VGroup(Text("if the waves are small: move the surface conditions to the flat level",
                           font_size=20),
                      math_tag("f(η) = f(0) + η f′(0) + …   (Taylor)", 16) if SHOW_MATH else VGroup()
                      ).arrange(DOWN, buff=0.08).next_to(title, DOWN, buff=0.25)
        self.play(Create(flat), FadeIn(idea))
        self.wait(3)

        self.clear_all(keep=title)
        self.stokes_orders(title)

    # ------------------------------------------------------------------
    def phi_image(self, time, fade=1.0):
        """Colour map of phi (flat bottom) inside the water; transparent above the surface."""
        ax = self.ax
        nx, nz = 240, 80
        x = np.linspace(*X_RANGE, nx)
        z = np.linspace(Z_RANGE[1], Z_RANGE[0], nz)          # image rows go top -> bottom
        X, Z = np.meshgrid(x, z)
        th = K * X - W * time
        phi = np.cosh(K * (Z + H)) / np.sinh(K * H) * np.sin(th)     # (a w / k) dropped: shape only
        v = phi / (np.cosh(K * H) / np.sinh(K * H))                  # scale to [-1, 1]
        lo, hi = np.array([40, 90, 200]), np.array([235, 120, 60])  # blue (low) -> orange (high)
        rgb = ((1 - v[..., None]) / 2 * lo + (1 + v[..., None]) / 2 * hi)
        inside = (Z <= A * np.cos(K * X - W * time)) & (Z >= -H)     # between bottom and surface
        alpha = np.where(inside, 200 * fade, 0)
        img = np.dstack([rgb, alpha]).astype(np.uint8)
        im = ImageMobject(img)
        im.stretch_to_fit_width(X_LEN).stretch_to_fit_height(Z_LEN)
        im.move_to(ax.c2p((X_RANGE[0] + X_RANGE[1]) / 2, (Z_RANGE[0] + Z_RANGE[1]) / 2))
        return im

    # ------------------------------------------------------------------
    def drift_beat(self, ax, t, say, advance, T, parcels, paths, periods=2):
        """Particles moved by the flow AT THEIR OWN POSITION (dx/dt = u(x, z, t),
        integrated with RK4). Nothing is added by hand: the forward creep is Stokes
        drift, second order in steepness. Checked: for small waves the drift matches
        Stokes' formula a^2 w k cosh(2k(z+h)) / (2 sinh^2 kh) within a few percent.
        Here the amplitude is exaggerated, so the drift is only qualitative.
        The first-order ball and blobs are hidden during this beat: in reality every
        particle drifts, so showing closed loops beside drifting ones would be wrong."""
        def vel(x, z, tt):
            th = K * x - W * tt
            return (A * W * np.cosh(K * (z + H)) / np.sinh(K * H) * np.cos(th),
                    A * W * np.sinh(K * (z + H)) / np.sinh(K * H) * np.sin(th))

        t0 = t.get_value()
        n = 400                                       # RK4 steps per period

        def integrate(x, z):
            dt = T / n
            times, pts = [t0], [(x, z)]
            for i in range(int(periods * n)):
                tt = t0 + i * dt
                k1 = vel(x, z, tt)
                k2 = vel(x + dt / 2 * k1[0], z + dt / 2 * k1[1], tt + dt / 2)
                k3 = vel(x + dt / 2 * k2[0], z + dt / 2 * k2[1], tt + dt / 2)
                k4 = vel(x + dt * k3[0], z + dt * k3[1], tt + dt)
                x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                z += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                times.append(tt + dt)
                pts.append((x, z))
            return np.array(times), np.array(pts)

        def follower(times, pts, color):
            def pos():
                tt = np.clip(t.get_value(), times[0], times[-1])
                return ax.c2p(np.interp(tt, times, pts[:, 0]), np.interp(tt, times, pts[:, 1]))
            return live(lambda: Dot(pos(), radius=0.09, color=color)), pos

        xs0 = LAM / 3
        starts = [(xs0, A * np.cos(K * xs0 - W * t0), RED),      # on the surface
                  (xs0, -1.6, "#FF9E9E")]                       # deeper down
        dots, trails = [], []
        for xs, zs, col in starts:
            times, pts = integrate(xs, zs)
            d, pos = follower(times, pts, col)
            dots.append(d)
            trails.append(TracedPath(pos, stroke_color=col, stroke_width=2))
        start_mark = DashedLine(ax.c2p(xs0, -2.0), ax.c2p(xs0, 0.75), color=GREY_B,
                                stroke_width=1.5)

        say("closer look: two particles that follow the flow wherever they are", RED)
        self.play(*self.vanish(*parcels), FadeOut(paths))
        self.play(*self.appear(*dots), Create(start_mark))
        self.add(*trails)
        advance(periods, periods * 7)
        say("their loops do not close: a slow forward drift, weaker deeper down", RED)
        tag = (math_tag("Stokes drift: second order in steepness, ∝ (ka)²", 16)
               .next_to(ax, DOWN, buff=0.15)) if SHOW_MATH else VGroup()
        self.play(FadeIn(tag))
        self.beat()
        self.wait(1)
        self.play(*self.vanish(*dots), FadeOut(VGroup(*trails)), FadeOut(start_mark), FadeOut(tag))
        self.remove(*dots)
        say("so the closed loops are the first-order picture")
        self.play(*self.appear(*parcels), FadeIn(paths))
        self.wait(1)

    # ------------------------------------------------------------------
    def stokes_orders(self, title, math=True):
        """The expansion as layers: first order + second order = steeper crests."""
        new_title = Text("Then solve order by order", font_size=32).to_edge(UP)
        self.play(Transform(title, new_title))

        a, k = 0.5, 1.0                                  # deep water, exaggerated ka = 0.5
        th = np.linspace(0, 4 * np.pi, 400)
        layers = [("order ε:  a simple wave", a * np.cos(th), BLUE_B),
                  ("order ε²:  a smaller wave, twice as short", 0.5 * k * a**2 * np.cos(2 * th), TEAL),
                  ("sum:  sharper crests, flatter troughs",
                   a * np.cos(th) + 0.5 * k * a**2 * np.cos(2 * th), WHITE)]
        rows = VGroup()
        for label, y, color in layers:
            ax = Axes(x_range=[0, 4 * np.pi], y_range=[-0.8, 0.8], x_length=7.5, y_length=1.1,
                      tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0.25})
            curve = ax.plot_line_graph(th, y, add_vertex_dots=False, line_color=color, stroke_width=3)
            lab = Text(label, font_size=18, color=color)
            rows.add(VGroup(ax, curve, lab))
        for row in rows:
            row[2].next_to(row[0], RIGHT, buff=0.3)
        rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(new_title, DOWN, buff=0.4).to_edge(LEFT, buff=0.6)
        plus = Text("+", font_size=30).move_to((rows[0][0].get_bottom() + rows[1][0].get_top()) / 2)
        eq = Text("=", font_size=30).move_to((rows[1][0].get_bottom() + rows[2][0].get_top()) / 2)

        self.play(FadeIn(rows[0]))
        self.play(FadeIn(plus), FadeIn(rows[1]))
        self.play(FadeIn(eq), FadeIn(rows[2]))
        if SHOW_MATH and math:
            tag = math_tag("η = εη<sub>1</sub> + ε²η<sub>2</sub> + ε³η<sub>3</sub> + …     (ε = steepness)", 18)
            tag.next_to(rows, DOWN, buff=0.3).align_to(rows, LEFT)
            self.play(FadeIn(tag))
        self.wait(2)

        out1 = VGroup(Text("order ε  →  dispersion relation", font_size=22, color=BLUE_B),
                      math_tag("ω² = gk tanh(kh)", 16) if (SHOW_MATH and math) else VGroup()
                      ).arrange(DOWN, buff=0.06)
        out3 = VGroup(Text("order ε³  →  NLS", font_size=22, color=YELLOW),
                      Text("the envelope equation", font_size=15, color=GREY_A)
                      ).arrange(DOWN, buff=0.06)
        VGroup(out1, out3).arrange(RIGHT, buff=1.5).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(out1, shift=UP * 0.15))
        self.wait(1)
        self.play(FadeIn(out3, shift=UP * 0.15))
        self.wait(3)


# ======================================================================
# Short version for a general audience
# ======================================================================
class FluidSetupLite(FluidSetup):
    """Same physics and drawing as FluidSetup, fewer beats, one caption at a time."""

    def caption(self, old, text, color=WHITE):
        new = Text(text, font_size=22, color=color).next_to(self.title, DOWN, buff=0.25)
        if old is not None:                          # old caption out first: no overlap
            self.play(FadeOut(old), run_time=0.4)
        self.play(FadeIn(new), run_time=0.5)
        return new

    def construct(self):
        ax = Axes(x_range=[*X_RANGE], y_range=[*Z_RANGE], x_length=X_LEN, y_length=Z_LEN,
                  tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0}
                  ).move_to([0, -1.15, 0])
        self.ax = ax
        t = ValueTracker(0.0)
        t.add_updater(lambda m, dt: m.increment_value(1.5 * dt))
        self.add(t)
        bump = ValueTracker(0.0)
        xs = np.linspace(*X_RANGE, 300)

        def theta(x):
            return K * x - W * t.get_value()

        def eta(x):
            return A * np.cos(theta(x))

        def bed(x):
            return -H + bump.get_value() * 0.3 * np.sin(3 * 2 * np.pi * x / LAM)

        water = live(lambda: Polygon(
            *[ax.c2p(x, eta(x)) for x in xs], *[ax.c2p(x, bed(x)) for x in xs[::-1]],
            stroke_width=0, fill_color=BLUE_E, fill_opacity=0.45))
        surface = live(lambda: VMobject(stroke_color=BLUE_B, stroke_width=3)
                                .set_points_smoothly([ax.c2p(x, eta(x)) for x in xs]))
        bottom = live(lambda: VMobject(stroke_color=GREY_B, stroke_width=3)
                               .set_points_smoothly([ax.c2p(x, bed(x)) for x in xs]))

        self.title = Text("What the water does under a wave", font_size=32).to_edge(UP)
        self.play(Write(self.title))
        self.play(*self.appear(water, surface, bottom))

        # ---- 1. parcels: orbits, and they never turn ----
        x0, size = LAM / 2, 0.32
        depths = [-0.65, -1.35, -2.05, -2.75]

        def parcel(z0):
            def draw():
                pts = []
                for dx, dz in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    xa, za = x0 + dx * size / 2, z0 + dz * size / 2
                    ox, oz = orbit_sizes(za)
                    th = K * xa - W * t.get_value()
                    pts.append(ax.c2p(xa - ox * np.sin(th), za + oz * np.cos(th)))
                sq = Polygon(*pts, stroke_color=DYE, stroke_width=2, fill_color=DYE, fill_opacity=0.7)
                return VGroup(sq, Line(sq.get_center(), (pts[1] + pts[2]) / 2,
                                       color=BLACK, stroke_width=3))
            return live(draw)

        def orbit_path(z0):
            ox, oz = orbit_sizes(z0)
            s_ = np.linspace(0, 2 * np.pi, 80)
            return VMobject(stroke_color=DYE, stroke_width=1.5, stroke_opacity=0.5
                            ).set_points_smoothly([ax.c2p(x0 + ox * np.cos(a), z0 + oz * np.sin(a))
                                                   for a in s_])

        parcels = VGroup(*[parcel(z) for z in depths])
        paths = VGroup(*[orbit_path(z) for z in depths])
        cap = self.caption(None, "each blob of water goes round in a loop", DYE)
        self.play(*self.appear(*parcels), Create(paths))
        self.wait(3)
        cap = self.caption(cap, "smaller loops deeper down, and the blobs never turn")
        self.wait(3)

        # ---- 2. one landscape phi; inside, one simple rule ----
        def arrows():
            g = VGroup()
            for xg in np.linspace(X_RANGE[0] + 0.9, X_RANGE[1] - 0.9, 9):
                for zg in [-0.75, -1.45, -2.15, -2.75]:
                    th = theta(xg)
                    u = A * W * np.cosh(K * (zg + H)) / np.sinh(K * H) * np.cos(th)
                    w = A * W * np.sinh(K * (zg + H)) / np.sinh(K * H) * np.sin(th)
                    g.add(Arrow(ax.c2p(xg, zg), ax.c2p(xg + 3 * u, zg + 3 * w), buff=0,
                                color=WHITE, stroke_width=2, tip_length=0.1,
                                max_tip_length_to_length_ratio=0.4))
            return g

        landscape = self.live_landscape(t)
        field = live(arrows)
        cap = self.caption(cap, "so the whole flow follows the slope of one landscape (colour)")
        self.play(*self.vanish(*parcels), FadeOut(paths), *self.appear(landscape))
        self.add(surface)
        self.play(*self.appear(field))
        self.wait(3)
        cap = self.caption(cap, "inside the water, that landscape obeys one simple rule")
        tag = math_tag("∇²φ = 0", 22).next_to(ax, DOWN, buff=0.15) if SHOW_MATH else VGroup()
        self.play(FadeIn(tag))
        self.wait(3)
        self.play(*self.vanish(field, landscape), FadeOut(tag))

        # ---- 3. bottom ----
        cap = self.caption(cap, "bottom: water cannot flow through it, even over sandbars", TEAL)
        bottom_glow = always_redraw(lambda: VMobject(stroke_color=TEAL, stroke_width=6)
                                    .set_points_smoothly([ax.c2p(x, bed(x)) for x in xs]))
        self.play(FadeIn(bottom_glow))
        self.play(bump.animate.set_value(1), run_time=2)
        tag_b = math_tag("φ<sub>z</sub> + h<sub>x</sub>φ<sub>x</sub> = 0  at  z = −h(x)", 16
                         ).next_to(ax, DOWN, buff=0.15) if SHOW_MATH else VGroup()
        self.play(FadeIn(tag_b))
        self.wait(2)

        # ---- 4. surface: the hard part ----
        cap = self.caption(cap, "surface: moves with the water, and feels only air pressure", CORAL)
        surf_glow = live(lambda: VMobject(stroke_color=CORAL, stroke_width=6)
                         .set_points_smoothly([ax.c2p(x, eta(x)) for x in xs]))
        tag_s = (math_tag("η<sub>t</sub> + φ<sub>x</sub>η<sub>x</sub> = φ<sub>z</sub>      "
                          "φ<sub>t</sub> + ½|∇φ|² + gη = 0      at z = η", 16)
                 .next_to(cap, DOWN, buff=0.1)) if SHOW_MATH else VGroup()
        self.play(*self.appear(surf_glow), FadeIn(tag_s))
        self.wait(3)
        cap = self.caption(cap, "the hard part: these rules sit on a surface we don't know yet", YELLOW)
        self.play(FadeOut(tag_s))
        self.wait(3)

        # ---- 5. step by step ----
        self.clear_all(keep=self.title)
        self.stokes_orders(self.title, math=False)
