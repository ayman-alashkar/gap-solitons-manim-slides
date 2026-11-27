"""
Part 2 of the talk as interactive slides: from dispersion to the NLS and its tree,
ending on a full-screen soliton.
"""

import numpy as np
from manim import *
from manim_slides import Slide
from nls_visual_tree import (clip_mi, clip_peregrine, clip_bump, clip_dark, clip_shock,
                             make_panel, live_curve, caption, elbow, math_tag,
                             CORAL, TEAL, solve_nls, grid)

G = 1.0
MATHC = "#9FC9FF"


def omega_deep(k):
    return np.sqrt(G * k)


K0, SIGMA_K, NMODES = 1.0, 0.3, 60
_k = np.linspace(K0 - 3 * SIGMA_K, K0 + 3 * SIGMA_K, NMODES)
_amp = np.exp(-(_k - K0) ** 2 / (2 * SIGMA_K ** 2))
_amp /= _amp.sum()
_w = omega_deep(_k)


def _ph(x, tt):
    return np.outer(np.atleast_1d(x), _k) - _w * tt


def packet_eta(x, tt):
    return np.cos(_ph(x, tt)) @ _amp


def packet_env(x, tt):
    return np.abs(np.exp(1j * _ph(x, tt)) @ _amp)


def part1_last_frame():
    title = Text("Dispersion: speed depends on wavelength", font_size=32).to_edge(UP)
    eq = MarkupText("ω² = gk tanh(kh)", font_size=22, color=MATHC).next_to(title, DOWN, buff=0.25)
    sub = Text("a packet holds many wavelengths, so it spreads out", font_size=22).next_to(eq, DOWN, buff=0.2)
    x_min, x_max = -10, 75
    axp = Axes(x_range=[x_min, x_max, 10], y_range=[-1.2, 1.2, 0.5], x_length=12.5, y_length=4.2,
               tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0}
               ).next_to(sub, DOWN, buff=0.35)
    T = 80.0
    surf = axp.plot(lambda x: packet_eta(x, T)[0], x_range=[x_min, x_max, 0.05], color=BLUE)
    top = axp.plot(lambda x: packet_env(x, T)[0], x_range=[x_min, x_max, 0.1], color=YELLOW, stroke_opacity=0.7)
    bot = axp.plot(lambda x: -packet_env(x, T)[0], x_range=[x_min, x_max, 0.1], color=YELLOW, stroke_opacity=0.7)
    ghosts = VGroup(*[axp.plot(lambda x, tf=tf: packet_env(x, tf)[0], x_range=[x_min, x_max, 0.1],
                               color=YELLOW, stroke_opacity=0.55, stroke_width=2)
                      for tf in [0, 20, 40, 60, 80]])
    xg = np.linspace(x_min, x_max, 1500)
    pts = []
    for tt in np.linspace(0, T, 160):
        e = packet_env(xg, tt)
        i = np.argmax(e)
        pts.append(axp.c2p(xg[i], e[i]))
    trail = VMobject(stroke_color=RED, stroke_width=3).set_points_as_corners(pts)
    pdot = Dot(pts[-1], color=RED, radius=0.07)
    note = Text("long waves ahead, short waves behind", font_size=20, color=GREY_A).next_to(axp, DOWN, buff=0.1)
    return dict(title=title, eq=eq, sub=sub, axp=axp, surf=surf, top=top, bot=bot,
                ghosts=ghosts, trail=trail, pdot=pdot, note=note)


def clip_bright_long():
    x = grid(40, 512)
    return x, solve_nls((1 / np.cosh(x + 6)) * np.exp(0.5j * x), 40, +1, 24.0)


def term(markup, size=30):
    return MarkupText(markup, font_size=size, color=MATHC)


class DispersionToNLS(Slide):
    def say(self, text, color=WHITE, size=22):
        new = Text(text, font_size=size, color=color).next_to(self.title, DOWN, buff=0.2)
        if getattr(self, "cap", None) is not None:
            self.play(FadeOut(self.cap), run_time=0.35)
        self.play(FadeIn(new), run_time=0.45)
        self.cap = new

    def construct(self):
        self.cap = None
        F = part1_last_frame()
        self.add(*[F[k] for k in ["title", "eq", "sub", "ghosts", "surf", "top", "bot", "trail", "pdot", "note"]])
        self.title = F["title"]
        self.envelope_idea(F)
        self.dispersion_term(F)
        self.nonlinear_term()
        self.assemble()
        self.tree()
        self.soliton_balance()

    # ------------------------------------------------------------------
    def envelope_idea(self, F):
        new_title = Text("An equation for the envelope", font_size=32).to_edge(UP)
        self.play(FadeOut(self.title), FadeOut(VGroup(F["eq"], F["sub"], F["note"], F["trail"], F["pdot"])),
                  run_time=0.5)
        self.title = new_title
        self.play(FadeIn(new_title), run_time=0.5)
        self.say("a packet: a fast wave inside a slow envelope A")
        A_lab = term("A", 30).next_to(F["top"].get_top(), UP, buff=0.1)
        self.play(F["surf"].animate.set_stroke(opacity=0.25),
                  F["top"].animate.set_stroke(width=6, opacity=1), FadeIn(A_lab))
        self.next_slide()
        self.say("we want a rule for how A moves; it has two ingredients")
        self.next_slide()
        self.A_lab = A_lab

    # ------------------------------------------------------------------
    def dispersion_term(self, F):
        self.say("1. dispersion", BLUE_B)
        left = VGroup(F["ghosts"], F["surf"], F["top"], F["bot"], self.A_lab)
        self.play(left.animate.scale_to_fit_width(6.6).move_to([-3.3, 0.4, 0]), run_time=1.5)

        axw = Axes(x_range=[0.2, 2.0, 0.5], y_range=[0.3, 1.6, 0.5], x_length=4.6, y_length=2.8,
                   tips=False, axis_config={"include_ticks": False, "color": GREY_B}).move_to([3.9, 0.5, 0])
        xl = Text("k", font_size=18, color=GREY_A).next_to(axw, DOWN, buff=0.1)
        yl = Text("ω", font_size=18, color=GREY_A).next_to(axw, LEFT, buff=0.1)
        curve = axw.plot(omega_deep, x_range=[0.2, 2.0], color=BLUE_B, stroke_width=4)
        tang = axw.plot(lambda k: 1 + 0.5 * (k - 1), x_range=[0.2, 2.0], color=GREY_B, stroke_width=2)
        tang = DashedVMobject(tang)
        k0dot = Dot(axw.c2p(1, 1), color=WHITE, radius=0.06)
        self.play(Create(axw), FadeIn(xl), FadeIn(yl), Create(curve))
        self.play(Create(tang), FadeIn(k0dot))
        bend_txt = VGroup(Text("a straight line would mean no dispersion;", font_size=15, color=GREY_A),
                          Text("the bending is what spreads the packet", font_size=15, color=GREY_A)
                          ).arrange(DOWN, buff=0.04).next_to(axw, DOWN, buff=0.45)
        self.play(FadeIn(bend_txt))
        P_lab = VGroup(term("P", 30), Text("= how bent ω(k) is  (P ∝ ω″)", font_size=16, color=MATHC)
                       ).arrange(RIGHT, buff=0.15).next_to(bend_txt, DOWN, buff=0.25)
        self.play(FadeIn(P_lab))
        self.next_slide()

        Axx_lab = VGroup(term("A<sub>ξξ</sub>", 30), Text("= how bent the envelope is", font_size=16, color=MATHC)
                         ).arrange(RIGHT, buff=0.15).next_to(left, DOWN, buff=0.5)
        self.play(Indicate(F["top"], color=YELLOW, scale_factor=1.03), FadeIn(Axx_lab))
        self.next_slide()

        disp_term = term("P A<sub>ξξ</sub>", 34).move_to([0, -2.6, 0])
        disp_lab = Text("dispersion: spreads the envelope", font_size=17, color=BLUE_B).next_to(disp_term, DOWN, buff=0.1)
        self.play(ReplacementTransform(VGroup(P_lab[0], Axx_lab[0]), disp_term),
                  FadeOut(VGroup(P_lab[1], Axx_lab[1])), run_time=1.5)
        self.play(FadeIn(disp_lab))
        self.next_slide()
        self.play(FadeOut(VGroup(left, axw, xl, yl, curve, tang, k0dot, bend_txt)),
                  VGroup(disp_term, disp_lab).animate.scale(0.8).move_to([-4.6, -3.1, 0]))
        self.disp = VGroup(disp_term, disp_lab)

    # ------------------------------------------------------------------
    def nonlinear_term(self):
        self.say("2. nonlinearity: the wave's own height sets its crest speed", YELLOW)
        ax = Axes(x_range=[-14, 14, 7], y_range=[-1.1, 1.1, 0.5], x_length=11, y_length=3.2,
                  tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0}).move_to([0, 0.3, 0])
        k, alpha = 2.0, 1.2

        def A(x):
            return 0.9 * np.exp(-x ** 2 / 18)

        t = ValueTracker(0.0)
        wave = always_redraw(lambda: ax.plot(
            lambda x: A(x) * np.cos(k * x - t.get_value() * (1 + alpha * A(x) ** 2)),
            x_range=[-14, 14, 0.03], color=BLUE_B, stroke_width=3))
        env_t = ax.plot(A, x_range=[-14, 14], color=YELLOW, stroke_width=3)
        env_b = ax.plot(lambda x: -A(x), x_range=[-14, 14], color=YELLOW, stroke_width=3)
        A2 = DashedVMobject(ax.plot(lambda x: A(x) ** 2, x_range=[-14, 14], color=ORANGE, stroke_width=2.5),
                            num_dashes=50)
        A2_lab = Text("extra speed ∝ |A|²", font_size=17, color=ORANGE).next_to(ax.c2p(4.5, 0.55), RIGHT, buff=0.1)
        frame_note = Text("(seen moving with the packet)", font_size=14, color=GREY_B, slant=ITALIC)
        frame_note.next_to(ax, UP, buff=0.0).align_to(ax, RIGHT)
        self.play(Create(wave), Create(env_t), Create(env_b), FadeIn(frame_note))
        self.play(Create(A2), FadeIn(A2_lab))
        note = Text("crests under the tall middle run ahead: squeezed in front, stretched behind",
                    font_size=17, color=GREY_A).next_to(ax, DOWN, buff=0.15)
        self.play(t.animate.set_value(9), FadeIn(note), run_time=6, rate_func=linear)
        self.next_slide()

        alone = VGroup(Text("alone, it does not change the envelope's height: it twists the wave inside it",
                            font_size=17, color=WHITE),
                       Text("together with dispersion, that twist reshapes the envelope",
                            font_size=17, color=YELLOW)).arrange(DOWN, buff=0.06).move_to(note)
        self.play(FadeOut(note), run_time=0.35)
        self.play(FadeIn(alone))
        self.next_slide()

        pieces = VGroup(term("Q", 30), term("|A|²", 30), term("A", 30)).arrange(RIGHT, buff=0.6).next_to(alone, DOWN, buff=0.25)
        self.play(FadeIn(pieces[0]), TransformFromCopy(A2_lab, pieces[1]), TransformFromCopy(env_t, pieces[2]))
        nl_term = term("Q |A|²A", 34).move_to(pieces)
        nl_lab = Text("nonlinearity: height changes crest speed", font_size=17, color=YELLOW).next_to(nl_term, DOWN, buff=0.1)
        self.play(ReplacementTransform(pieces, nl_term), run_time=1.2)
        self.play(FadeIn(nl_lab))
        self.next_slide()
        self.play(FadeOut(VGroup(wave, env_t, env_b, A2, A2_lab, alone, frame_note)),
                  VGroup(nl_term, nl_lab).animate.scale(0.8).move_to([4.6, -3.1, 0]))
        self.nl = VGroup(nl_term, nl_lab)

    # ------------------------------------------------------------------
    def assemble(self):
        self.say("together: the nonlinear Schrödinger equation (NLS)")
        p_t = term("i A<sub>τ</sub>", 40)
        plus1, plus2, eq0 = term("+", 40), term("+", 40), term("= 0", 40)
        d_t, n_t = term("P A<sub>ξξ</sub>", 40), term("Q |A|²A", 40)
        row = VGroup(p_t, plus1, d_t, plus2, n_t, eq0).arrange(RIGHT, buff=0.3).move_to([0, 0.6, 0])
        labs = VGroup(Text("how A changes in time", font_size=15, color=GREY_A).next_to(p_t, DOWN, buff=0.2),
                      Text("dispersion", font_size=15, color=BLUE_B).next_to(d_t, DOWN, buff=0.2),
                      Text("nonlinearity", font_size=15, color=YELLOW).next_to(n_t, DOWN, buff=0.2))
        self.play(ReplacementTransform(self.disp[0], d_t), ReplacementTransform(self.nl[0], n_t),
                  FadeOut(self.disp[1]), FadeOut(self.nl[1]), run_time=1.5)
        self.play(FadeIn(p_t), FadeIn(plus1), FadeIn(plus2), FadeIn(eq0))
        self.play(FadeIn(labs))
        self.next_slide()
        self.equation = VGroup(row, labs)
        self.eq_row = row

    # ------------------------------------------------------------------
    def tree(self):
        self.play(FadeOut(self.cap), FadeOut(self.title), FadeOut(self.equation[1]), run_time=0.6)
        self.cap = None
        top_box = RoundedRectangle(corner_radius=0.12, width=6.2, height=0.95, stroke_color=GREY_B,
                                   fill_color=GREY_B, fill_opacity=0.12).move_to([0, 3.15, 0])
        top_lab = Text("NLS: the rule for the envelope", font_size=14, color=GREY_A).next_to(top_box, DOWN, buff=0.06)
        self.play(self.eq_row.animate.scale(0.62).move_to(top_box), FadeIn(top_box), run_time=1.3)
        self.play(FadeIn(top_lab))

        def split_node(t1, t2, color, xpos):
            box = RoundedRectangle(corner_radius=0.1, width=4.4, height=0.66, stroke_color=color,
                                   stroke_width=2, fill_color=color, fill_opacity=0.15)
            lab = VGroup(Text(t1, font_size=19, weight=BOLD), Text(t2, font_size=14, color=GREY_A)
                         ).arrange(DOWN, buff=0.05)
            box.move_to([xpos, 1.45, 0])
            return VGroup(box, lab.move_to(box))

        foc = split_node("Focusing   PQ > 0", "nonlinearity fights the spreading", CORAL, -3.7)
        defoc = split_node("Defocusing   PQ < 0", "nonlinearity adds to the spreading", TEAL, 3.7)
        self.play(Create(elbow(top_lab.get_bottom(), foc.get_top(), CORAL)),
                  Create(elbow(top_lab.get_bottom(), defoc.get_top(), TEAL)),
                  FadeIn(foc, shift=DOWN * 0.15), FadeIn(defoc, shift=DOWN * 0.15))
        self.next_slide()

        LEAF_Y = -0.55

        def leaves(specs, parent, color):
            out = []
            for (x, f, y_range, xpos, text, name, highlight, m, tracker) in specs:
                box, ax = make_panel(x, y_range, color, 2.05, 1.0, [xpos, LEAF_Y, 0])
                cap = caption(text, name, YELLOW if highlight else WHITE, m).next_to(box, DOWN, buff=0.08)
                out.append(dict(box=box, ax=ax, cap=cap, curve=live_curve(ax, x, f, tracker, color),
                                arrow=elbow(parent.get_bottom(), box.get_top(), color), x=x, f=f))
            return out

        s_f, s_b, s_d = ValueTracker(0), ValueTracker(0), ValueTracker(0)
        xb, fb = clip_bright_long()
        Lf = leaves([
            (*clip_mi(), [0, 2.7], -5.9, "uniform wave\nbreaks into pulses", "modulational instability", False, "max growth rate = Q|A|²", s_f),
            (xb, fb, [0, 1.2], -3.7, "a hump that\nholds its shape", "bright soliton", False, "|A| ∝ sech", s_b),
            (*clip_peregrine(), [0, 3.2], -1.5, "a giant wave appears\nand vanishes", "rogue wave", True, "peak = 3 × background", s_f),
        ], foc, CORAL)
        self.play(*[Create(L["arrow"]) for L in Lf], *[FadeIn(L["box"]) for L in Lf],
                  *[FadeIn(L["cap"]) for L in Lf])
        self.add(*[L["curve"] for L in Lf])
        self.play(s_f.animate.set_value(1), s_b.animate.set_value(0.5), run_time=6, rate_func=linear)
        self.next_slide()

        Ld = leaves([
            (*clip_bump(), [0.8, 1.2], 1.5, "a bump just\ntravels away", "stable modulation", False, "ripple speed = |A|√(−2PQ)", s_d),
            (*clip_dark(), [0, 1.2], 3.7, "a hole that\nholds its shape", "dark soliton", False, "|A| ∝ tanh", s_d),
            (*clip_shock(), [0, 1.5], 5.9, "a step breaks\ninto ripples", "dispersive shock wave", False, "steep front → ripple train", s_d),
        ], defoc, TEAL)
        self.play(*[Create(L["arrow"]) for L in Ld], *[FadeIn(VGroup(L["box"], L["cap"])) for L in Ld])
        self.add(*[L["curve"] for L in Ld])
        self.play(s_d.animate.set_value(1), run_time=6, rate_func=linear)
        self.next_slide()
        self.soliton = Lf[1]
        self.s_b = s_b

    # ------------------------------------------------------------------
    def soliton_balance(self):
        L, s_b = self.soliton, self.s_b
        keep = {L["box"], L["curve"]}
        others = [m for m in self.mobjects if m not in keep and not isinstance(m, ValueTracker)]

        big_box = RoundedRectangle(corner_radius=0.2, width=12.4, height=4.6, stroke_color=CORAL,
                                   stroke_width=3, fill_color=CORAL, fill_opacity=0.1).move_to([0, -0.4, 0])
        self.play(*[FadeOut(m) for m in others],
                  L["box"].animate.become(big_box),
                  L["ax"].animate.stretch_to_fit_width(12.0).stretch_to_fit_height(4.2).move_to(big_box),
                  s_b.animate.set_value(0.68), run_time=2.5, rate_func=linear)

        title = Text("A soliton: two effects in balance", font_size=32).to_edge(UP)
        x, f = L["x"], L["f"]

        def peak():
            i = int(round(s_b.get_value() * (len(f) - 1)))
            j = int(np.argmax(f[i]))
            return x[j], f[i][j]

        def arrows():
            xp, h = peak()
            ax = L["ax"]
            y = 0.45 * h
            g = VGroup()
            for side in (-1, 1):
                g.add(Arrow(ax.c2p(xp + side * 1.4, y + 0.18), ax.c2p(xp + side * 4.2, y + 0.18),
                            buff=0, color=BLUE_B, stroke_width=5, tip_length=0.2))
                g.add(Arrow(ax.c2p(xp + side * 4.2, y - 0.18), ax.c2p(xp + side * 1.4, y - 0.18),
                            buff=0, color=YELLOW, stroke_width=5, tip_length=0.2))
            return g

        arr = always_redraw(arrows)
        legend = VGroup(
            VGroup(Arrow(LEFT * 0.4, RIGHT * 0.4, buff=0, color=BLUE_B, stroke_width=5, tip_length=0.18),
                   Text("dispersion spreads it", font_size=18, color=BLUE_B)).arrange(RIGHT, buff=0.2),
            VGroup(Arrow(RIGHT * 0.4, LEFT * 0.4, buff=0, color=YELLOW, stroke_width=5, tip_length=0.18),
                   Text("nonlinearity squeezes it", font_size=18, color=YELLOW)).arrange(RIGHT, buff=0.2),
        ).arrange(RIGHT, buff=1.2).next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(title), FadeIn(arr), FadeIn(legend))
        self.play(s_b.animate.set_value(1), run_time=5, rate_func=linear)
        result = Text("so it travels without changing shape", font_size=22).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(result))
        self.next_slide()