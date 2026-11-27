"""
Part 1 of the talk as interactive slides: the water-wave problem, how we solve it,
and the first result (dispersion).
"""

import numpy as np
from manim import *
from manim_slides import Slide
from fluid_setup_scene import (FluidSetup as _Helpers, live, orbit_sizes, math_tag,
                               CORAL, TEAL, DYE, G, H, K, W, A, LAM, X_RANGE, Z_RANGE,
                               X_LEN, Z_LEN)

SAND = "#C2A267"


def omega_deep(k):
    return np.sqrt(G * k)


class SetupToDispersion(_Helpers, Slide):
    def say(self, text, color=WHITE):
        """One caption at a time under the title; the old one leaves first."""
        new = Text(text, font_size=22, color=color).next_to(self.title, DOWN, buff=0.22)
        if getattr(self, "cap", None) is not None:
            self.play(FadeOut(self.cap), run_time=0.35)
        self.play(FadeIn(new), run_time=0.45)
        self.cap = new

    def construct(self):
        self.cap = None
        self.title = Text("How we describe moving water", font_size=32).to_edge(UP)
        self.play(Write(self.title))
        self.water_problem()
        self.orders()
        self.dispersion()

    # ==================================================================
    def water_problem(self):
        ax = Axes(x_range=[*X_RANGE], y_range=[*Z_RANGE], x_length=X_LEN, y_length=Z_LEN,
                  tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0}
                  ).move_to([0, -1.15, 0])
        self.ax = ax
        t = ValueTracker(0.0)
        t.add_updater(lambda m, dt: m.increment_value(1.2 * dt))
        self.add(t)
        bump = ValueTracker(0.0)
        xs = np.linspace(*X_RANGE, 300)

        def theta(x):
            return K * x - W * t.get_value()

        def eta(x):
            return A * np.cos(theta(x))

        def bed(x):
            return -H + bump.get_value() * 0.3 * np.sin(3 * 2 * np.pi * x / LAM)

        def velocity(x, z):
            th = theta(x)
            return (A * W * np.cosh(K * (z + H)) / np.sinh(K * H) * np.cos(th),
                    A * W * np.sinh(K * (z + H)) / np.sinh(K * H) * np.sin(th))

        water = live(lambda: Polygon(*[ax.c2p(x, eta(x)) for x in xs],
                                     *[ax.c2p(x, bed(x)) for x in xs[::-1]],
                                     stroke_width=0, fill_color=BLUE_E, fill_opacity=0.45))
        surface = live(lambda: VMobject(stroke_color=BLUE_B, stroke_width=3)
                       .set_points_smoothly([ax.c2p(x, eta(x)) for x in xs]))
        bottom = live(lambda: VMobject(stroke_color=GREY_B, stroke_width=3)
                      .set_points_smoothly([ax.c2p(x, bed(x)) for x in xs]))
        self.play(*self.appear(water, surface, bottom))

        # ---- 1. one fluid particle, briefly ----
        x0, z0, size = LAM / 2, -1.2, 0.34

        def blob_draw():
            pts = []
            for dx, dz in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                xa, za = x0 + dx * size / 2, z0 + dz * size / 2
                ox, oz = orbit_sizes(za)
                th = K * xa - W * t.get_value()
                pts.append(ax.c2p(xa - ox * np.sin(th), za + oz * np.cos(th)))
            sq = Polygon(*pts, stroke_color=DYE, stroke_width=2, fill_color=DYE, fill_opacity=0.8)
            c = sq.get_center()
            u, w = velocity(x0, z0)
            tip = c + np.array([u, w, 0]) * 3.5 * ax.x_axis.get_unit_size()
            arr = Arrow(c, tip, buff=0, color=WHITE, stroke_width=3, tip_length=0.14,
                        max_tip_length_to_length_ratio=0.4)
            return VGroup(sq, arr)

        blob = live(blob_draw)
        self.say("a fluid particle: a tiny blob of water, with one velocity", DYE)
        self.play(*self.appear(blob))
        self.wait(2.5)
        self.next_slide()

        def arrows():
            g = VGroup()
            for xg in np.linspace(X_RANGE[0] + 0.9, X_RANGE[1] - 0.9, 9):
                for zg in [-0.75, -1.45, -2.15, -2.75]:
                    u, w = velocity(xg, zg)
                    g.add(Arrow(ax.c2p(xg, zg), ax.c2p(xg + 3 * u, zg + 3 * w), buff=0,
                                color=WHITE, stroke_width=2, tip_length=0.1,
                                max_tip_length_to_length_ratio=0.4))
            return g

        field = live(arrows)
        self.say("we want the velocity at every point of the water")
        self.play(*self.vanish(blob), *self.appear(field))
        self.remove(blob)
        self.wait(2.0)
        self.next_slide()

        # ---- 2. assumptions -> landscape -> Laplace ----
        self.say("three assumptions make this much simpler")
        chips = VGroup()
        for main, m in [("no compression", "∇·<b>u</b> = 0"),
                        ("no friction", "inviscid"),
                        ("no vorticity", "∇×<b>u</b> = 0")]:
            chips.add(VGroup(Text(main, font_size=19, weight=BOLD), math_tag(m, 15)
                             ).arrange(DOWN, buff=0.06))
        chips.arrange(RIGHT, buff=1.0).next_to(self.cap, DOWN, buff=0.25)
        for c in chips:
            self.play(FadeIn(c, shift=DOWN * 0.15), run_time=0.6)
            self.wait(0.4)
        self.next_slide()

        landscape = self.live_landscape(t)
        self.say("no vorticity: all the arrows are the slope of one landscape φ (colour)")
        self.play(*self.appear(landscape))
        self.add(surface, field)
        self.wait(2.0)
        self.next_slide()

        laplace = VGroup(math_tag("∇²φ = 0", 26),
                         Text("every point = average of its neighbours", font_size=15, color=GREY_A)
                         ).arrange(DOWN, buff=0.06)
        lap_bg = BackgroundRectangle(laplace, fill_opacity=0.65, buff=0.12)
        lap = VGroup(lap_bg, laplace).move_to(ax.c2p(LAM / 2, -1.55))
        self.say("no compression: inside the water, φ obeys one simple rule")
        self.play(ReplacementTransform(chips, lap), run_time=1.5)
        self.wait(1.5)
        self.next_slide()
        self.play(*self.vanish(field, landscape))
        self.remove(field, landscape)

        # ---- 3. boundary conditions ----
        bottom_glow = live(lambda: VMobject(stroke_color=TEAL, stroke_width=6)
                           .set_points_smoothly([ax.c2p(x, bed(x)) for x in xs]))
        bot_tag = math_tag("φ<sub>z</sub> = 0", 17).next_to(ax, DOWN, buff=0.12)
        self.say("bottom: no water flows through it", TEAL)
        self.play(*self.appear(bottom_glow), FadeIn(bot_tag))
        self.wait(1.0)
        bot_tag2 = math_tag("φ<sub>z</sub> + h<sub>x</sub>φ<sub>x</sub> = 0   at z = −h(x)", 17
                            ).next_to(ax, DOWN, buff=0.12)
        self.say("even when the bottom is not flat", TEAL)
        self.play(bump.animate.set_value(1), FadeOut(bot_tag), run_time=1.2)
        self.play(FadeIn(bot_tag2), run_time=0.6)
        bot_tag = bot_tag2
        self.next_slide()

        surf_glow = live(lambda: VMobject(stroke_color=CORAL, stroke_width=6)
                         .set_points_smoothly([ax.c2p(x, eta(x)) for x in xs]))
        kbc = math_tag("η<sub>t</sub> + φ<sub>x</sub>η<sub>x</sub> = φ<sub>z</sub>", 17)
        dbc = math_tag("φ<sub>t</sub> + ½|∇φ|² + gη = 0", 17)
        kbc_l = Text("moves with the water", font_size=15, color=CORAL)
        dbc_l = Text("feels only air pressure (Bernoulli)", font_size=15, color=CORAL)
        surf_tags = VGroup(VGroup(kbc_l, kbc).arrange(DOWN, buff=0.05),
                           VGroup(dbc_l, dbc).arrange(DOWN, buff=0.05)
                           ).arrange(RIGHT, buff=1.5).next_to(ax, UP, buff=0.15)
        self.say("surface: two conditions, applied at z = η", CORAL)
        self.play(*self.appear(surf_glow), FadeIn(surf_tags))
        self.next_slide()

        # ---- gather: the tank shrinks left, the equations become one system ----
        box = RoundedRectangle(corner_radius=0.15, width=6.4, height=3.6,
                               stroke_color=GREY_B, fill_color=GREY_E, fill_opacity=0.3)
        box.move_to([3.55, -0.9, 0])
        box_title = Text("the full problem", font_size=20, weight=BOLD).next_to(box.get_top(), DOWN, buff=0.15)

        def line(label, eq, color):
            return VGroup(Text(label, font_size=15, color=color), math_tag(eq, 17)
                          ).arrange(RIGHT, buff=0.25)

        sys_lines = VGroup(
            line("inside", "∇²φ = 0", WHITE),
            line("bottom", "φ<sub>z</sub> + h<sub>x</sub>φ<sub>x</sub> = 0", TEAL),
            line("surface", "η<sub>t</sub> + φ<sub>x</sub>η<sub>x</sub> = φ<sub>z</sub>", CORAL),
            line("surface", "φ<sub>t</sub> + ½|∇φ|² + gη = 0", CORAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(box_title, DOWN, buff=0.3)

        self.say("together: the full water-wave problem")
        self.play(ax.animate.scale(0.5).move_to([-3.6, -0.9, 0]),
                  FadeIn(box), FadeIn(box_title),
                  ReplacementTransform(lap, sys_lines[0]),
                  ReplacementTransform(bot_tag, sys_lines[1]),
                  ReplacementTransform(VGroup(kbc_l, kbc), sys_lines[2]),
                  ReplacementTransform(VGroup(dbc_l, dbc), sys_lines[3]),
                  run_time=2)
        hard = Text("nonlinear, and on a surface we don't know yet: no exact solution",
                    font_size=19, color=YELLOW).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(hard))
        self.next_slide()

        # ---- the way in: small waves ----
        self.say("but if the waves are small, we can solve it step by step")
        series = math_tag("η = εη<sub>1</sub> + ε²η<sub>2</sub> + ε³η<sub>3</sub> + …     ε = steepness", 18
                          ).move_to(hard)
        self.play(ReplacementTransform(hard, series))
        self.next_slide()

        t.clear_updaters()
        lives = [m for m in self.mobjects if hasattr(m, "alpha")]
        self.play(*self.vanish(*lives),
                  FadeOut(VGroup(box, box_title, sys_lines)),
                  series.animate.next_to(self.title, DOWN, buff=0.9))
        self.remove(*lives)
        self.series = series

    # ==================================================================
    def orders(self):
        self.say("what each order gives")
        a = 0.45
        th = np.linspace(0, 4 * np.pi, 300)

        def mini(y, color):
            ax = Axes(x_range=[0, 4 * np.pi], y_range=[-0.8, 0.8], x_length=3.2, y_length=0.9,
                      tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0.2})
            return VGroup(ax, ax.plot_line_graph(th, y, add_vertex_dots=False,
                                                  line_color=color, stroke_width=3))

        def row(order, picture, text, eq, color):
            o = Text(order, font_size=22, color=color, weight=BOLD)
            right = VGroup(Text(text, font_size=19, color=color),
                           math_tag(eq, 15) if eq else VGroup()).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            return VGroup(o, picture, right).arrange(RIGHT, buff=0.5)

        r1 = row("order ε", mini(a * np.cos(th), BLUE_B),
                 "a simple wave, and how fast it travels", "ω² = gk tanh(kh)", BLUE_B)
        r2 = row("order ε²", mini(a * np.cos(th) + 0.5 * a**2 * np.cos(2 * th), TEAL),
                 "sharper crests, flatter troughs", "", TEAL)

        def taller_ahead():
            g = mini(0.3 * np.cos(th), GREY_B)
            g.add(g[0].plot_line_graph(th, 0.6 * np.cos(th - 0.7), add_vertex_dots=False,
                                       line_color=YELLOW, stroke_width=3))
            return g

        r3 = row("order ε³", taller_ahead(),
                 "taller waves: crests travel a little faster  →  NLS", "", YELLOW)
        rows = VGroup(r1, r2, r3).arrange(DOWN, aligned_edge=LEFT, buff=0.45).next_to(self.series, DOWN, buff=0.5)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.8)
            self.next_slide()

        # spotlight on order eps: the rest dims
        self.say("first: what order ε tells us")
        self.play(r2.animate.set_opacity(0.2), r3.animate.set_opacity(0.2), self.series.animate.set_opacity(0.3))
        self.next_slide()
        self.disp_eq = r1[2][1]
        self.fade_rest = VGroup(r1[0], r1[1], r1[2][0], r2, r3, self.series)

    # ==================================================================
    def dispersion(self):
        new_title = Text("Dispersion: speed depends on wavelength", font_size=32).to_edge(UP)
        eq_target = math_tag("ω² = gk tanh(kh)", 22).next_to(new_title, DOWN, buff=0.25)
        self.play(FadeOut(self.cap), run_time=0.3)
        self.cap = None
        self.play(FadeOut(self.title), run_time=0.3)
        self.title = new_title
        self.play(FadeIn(new_title), FadeOut(self.fade_rest),
                  self.disp_eq.animate.become(eq_target), run_time=1.5)

        h_deep = 10.0

        def c_of(lam):
            k = 2 * np.pi / lam
            return np.sqrt(G / k * np.tanh(k * h_deep))

        axc = Axes(x_range=[0, 30, 10], y_range=[0, 2.5, 1], x_length=7, y_length=3.6,
                   tips=False, axis_config={"include_ticks": False, "color": GREY_B}
                   ).next_to(eq_target, DOWN, buff=0.5).shift(LEFT * 2.2)
        xl = Text("wavelength", font_size=16, color=GREY_A).next_to(axc, DOWN, buff=0.15)
        yl = Text("speed", font_size=16, color=GREY_A).rotate(PI / 2).next_to(axc, LEFT, buff=0.15)
        curve = axc.plot(c_of, x_range=[0.5, 30], color=BLUE_B, stroke_width=4)
        self.play(Create(axc), FadeIn(xl), FadeIn(yl))
        self.play(Create(curve), run_time=1.5)

        lam_s, lam_l = 2 * np.pi, 8 * np.pi
        d_s = Dot(axc.c2p(lam_s, c_of(lam_s)), color=ORANGE, radius=0.1)
        d_l = Dot(axc.c2p(lam_l, c_of(lam_l)), color=BLUE, radius=0.1)
        l_s = Text("short", font_size=16, color=ORANGE).next_to(d_s, DOWN + RIGHT, buff=0.05)
        l_l = Text("long", font_size=16, color=BLUE).next_to(d_l, UP, buff=0.1)
        msg = Text("longer waves travel faster", font_size=22, color=YELLOW).next_to(axc, RIGHT, buff=0.3)
        self.play(FadeIn(d_s, scale=1.5), FadeIn(d_l, scale=1.5), FadeIn(l_s), FadeIn(l_l))
        self.play(FadeIn(msg))
        self.next_slide()

        # --- the two points become two waves ---
        t2 = ValueTracker(0.0)
        x_max = 45
        ax_long = Axes(x_range=[0, x_max, 5], y_range=[-1.3, 1.3, 1], x_length=12, y_length=2.0,
                       tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0})
        ax_short = ax_long.copy()
        ax_long.move_to([0, 0.75, 0])
        ax_short.move_to([0, -2.0, 0])

        def wave(axw, k, color):
            return always_redraw(lambda: axw.plot(
                lambda x: np.cos(k * x - omega_deep(k) * t2.get_value()),
                x_range=[0, x_max, 0.05], color=color))

        def crest(axw, k):
            c = omega_deep(k) / k
            return always_redraw(lambda: Dot(axw.c2p(c * t2.get_value(), 1.0),
                                             color=YELLOW, radius=0.09))

        w_l, w_s = wave(ax_long, 0.25, BLUE), wave(ax_short, 1.0, ORANGE)
        lab_long = Text("long wave", font_size=20, color=BLUE).next_to(ax_long, UP, buff=0.05).align_to(ax_long, LEFT)
        lab_short = Text("short wave", font_size=20, color=ORANGE).next_to(ax_short, UP, buff=0.05).align_to(ax_short, LEFT)
        start = DashedLine(ax_long.c2p(0, 1.3), ax_short.c2p(0, -1.3), color=GREY)

        self.play(FadeOut(VGroup(axc, xl, yl, curve, msg)),
                  ReplacementTransform(d_l, w_l), ReplacementTransform(d_s, w_s),
                  ReplacementTransform(l_l, lab_long), ReplacementTransform(l_s, lab_short),
                  run_time=1.8)
        c_l, c_s = crest(ax_long, 0.25), crest(ax_short, 1.0)
        self.play(FadeIn(c_l), FadeIn(c_s), Create(start))
        self.play(t2.animate.set_value(20), run_time=6, rate_func=linear)
        self.next_slide()

        # --- a packet of many wavelengths spreads ---
        self.play(FadeOut(VGroup(w_l, w_s, c_l, c_s, start, lab_long, lab_short)))
        sub = Text("a packet holds many wavelengths, so it spreads out", font_size=22
                   ).next_to(self.disp_eq, DOWN, buff=0.2)
        self.play(FadeIn(sub))

        k0, sigma_k, n = 1.0, 0.3, 60
        k = np.linspace(k0 - 3 * sigma_k, k0 + 3 * sigma_k, n)
        amp = np.exp(-(k - k0) ** 2 / (2 * sigma_k ** 2))
        amp /= amp.sum()
        w = omega_deep(k)

        def ph(x, tt):
            return np.outer(np.atleast_1d(x), k) - w * tt

        def eta(x, tt):
            return np.cos(ph(x, tt)) @ amp

        def env(x, tt):
            return np.abs(np.exp(1j * ph(x, tt)) @ amp)

        tp = ValueTracker(0.0)
        x_min, x_max = -10, 75
        axp = Axes(x_range=[x_min, x_max, 10], y_range=[-1.2, 1.2, 0.5], x_length=12.5, y_length=4.2,
                   tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0}
                   ).next_to(sub, DOWN, buff=0.35)
        surf = always_redraw(lambda: axp.plot(lambda x: eta(x, tp.get_value())[0],
                                              x_range=[x_min, x_max, 0.05], color=BLUE))
        top = always_redraw(lambda: axp.plot(lambda x: env(x, tp.get_value())[0],
                                             x_range=[x_min, x_max, 0.1], color=YELLOW, stroke_opacity=0.7))
        bot = always_redraw(lambda: axp.plot(lambda x: -env(x, tp.get_value())[0],
                                             x_range=[x_min, x_max, 0.1], color=YELLOW, stroke_opacity=0.7))
        self.play(Create(surf))
        self.play(FadeIn(top), FadeIn(bot))

        def ghost(tf):
            return axp.plot(lambda x, tf=tf: env(x, tf)[0], x_range=[x_min, x_max, 0.1],
                            color=YELLOW, stroke_opacity=0.55, stroke_width=2)

        xg = np.linspace(x_min, x_max, 1500)

        def peak():
            e = env(xg, tp.get_value())
            i = np.argmax(e)
            return axp.c2p(xg[i], e[i])

        pdot = always_redraw(lambda: Dot(peak(), color=RED, radius=0.07))
        trail = TracedPath(pdot.get_center, stroke_color=RED, stroke_width=3)
        self.add(trail, pdot, ghost(0.0))
        for tn in [20, 40, 60, 80]:
            self.play(tp.animate.set_value(tn), run_time=2.5, rate_func=linear)
            self.add(ghost(tn))
        note = Text("long waves ahead, short waves behind", font_size=20, color=GREY_A).next_to(axp, DOWN, buff=0.1)
        self.play(FadeIn(note))
        self.next_slide()