"""
Part 3 of the talk as interactive slides: optical fibre -> Bragg grating ->
gap soliton -> water tank with sandbars.
"""

import numpy as np
from manim import *
from manim_slides import Slide
from nls_visual_tree import live_curve
from story_nls import clip_bright_long, CORAL
from bragg_scene import (solve_cme, make_axes, fibre_background, water_background,
                         live_wave, side_label, LG, GRATING, SURF_Y)


def part2_last_frame():
    x, f = clip_bright_long()
    big_box = RoundedRectangle(corner_radius=0.2, width=12.4, height=4.6, stroke_color=CORAL,
                               stroke_width=3, fill_color=CORAL, fill_opacity=0.1).move_to([0, -0.4, 0])
    ax = Axes(x_range=[x[0], x[-1]], y_range=[0, 1.2], x_length=12.0, y_length=4.2, tips=False,
              axis_config={"include_ticks": False, "stroke_opacity": 0}).move_to(big_box)
    s = ValueTracker(1.0)
    curve = live_curve(ax, x, f, s, CORAL)

    j = int(np.argmax(f[-1]))
    xp, h = x[j], f[-1][j]
    y = 0.45 * h
    arrows = VGroup()
    for side in (-1, 1):
        arrows.add(Arrow(ax.c2p(xp + side * 1.4, y + 0.18), ax.c2p(xp + side * 4.2, y + 0.18),
                         buff=0, color=BLUE_B, stroke_width=5, tip_length=0.2))
        arrows.add(Arrow(ax.c2p(xp + side * 4.2, y - 0.18), ax.c2p(xp + side * 1.4, y - 0.18),
                         buff=0, color=YELLOW, stroke_width=5, tip_length=0.2))
    title = Text("A soliton: two effects in balance", font_size=32).to_edge(UP)
    legend = VGroup(
        VGroup(Arrow(LEFT * 0.4, RIGHT * 0.4, buff=0, color=BLUE_B, stroke_width=5, tip_length=0.18),
               Text("dispersion spreads it", font_size=18, color=BLUE_B)).arrange(RIGHT, buff=0.2),
        VGroup(Arrow(RIGHT * 0.4, LEFT * 0.4, buff=0, color=YELLOW, stroke_width=5, tip_length=0.18),
               Text("nonlinearity squeezes it", font_size=18, color=YELLOW)).arrange(RIGHT, buff=0.2),
    ).arrange(RIGHT, buff=1.2).next_to(title, DOWN, buff=0.3)
    result = Text("so it travels without changing shape", font_size=22).to_edge(DOWN, buff=0.35)
    return dict(box=big_box, ax=ax, curve=curve, arrows=arrows, title=title, legend=legend,
                result=result)


class SolitonToBragg(Slide):
    def set_title(self, text):
        new = Text(text, font_size=30).to_edge(UP)
        self.play(FadeOut(self.title), run_time=0.3)
        self.play(FadeIn(new), run_time=0.45)
        self.title = new

    def set_note(self, text):
        if getattr(self, "note", None) is not None:
            self.play(FadeOut(self.note), run_time=0.35)
            self.note = None
        if text:
            new = Text(text, font_size=20, color=GREY_A).to_edge(DOWN, buff=0.3)
            self.play(FadeIn(new), run_time=0.45)
            self.note = new

    def run(self, ax1, ax2, top, bottom, color, labels, hand_off=None):
        (x, f1, sc1), (_, f2, sc2) = top, bottom
        s = ValueTracker(0)
        w1 = live_wave(ax1, x, f1, s, sc1, color=color)
        w2 = live_wave(ax2, x, f2, s, sc2, color=color)
        lab1, lab2 = labels
        outgoing = []
        if getattr(self, "waves", None) is not None:
            outgoing.append(FadeOut(self.waves))
        if getattr(self, "lab1", None) is not None:
            outgoing += [FadeOut(self.lab1), FadeOut(self.lab2)]
        if outgoing:
            self.play(*outgoing, run_time=0.4)
        self.play(FadeIn(lab1), FadeIn(lab2), run_time=0.5)
        self.lab1, self.lab2 = lab1, lab2
        if hand_off is not None:
            self.remove(hand_off)
            self.add(w1)
            self.play(FadeIn(w2), run_time=0.5)
        else:
            self.add(w1, w2)
        self.play(s.animate.set_value(1), run_time=9, rate_func=linear)
        self.waves = VGroup(w1, w2)

    def construct(self):
        # ---- simulations ----
        x, in_gap = solve_cme(0.3, q=0.0, nonlinear=False)
        _, off_gap = solve_cme(0.3, q=2.5, nonlinear=False)
        _, weak = solve_cme(0.3, q=0.0, nonlinear=True)
        _, strong = solve_cme(1.0, q=0.0, nonlinear=True)
        lin = 0.9 / np.abs(in_gap).max()
        nl = 1.0 / np.abs(strong).max()

        # ---- 0. last frame of part 2 ----
        P = part2_last_frame()
        self.add(P["box"], P["curve"], P["arrows"], P["title"], P["legend"], P["result"])
        self.title = P["title"]
        self.note = None
        self.lab1 = self.lab2 = None
        self.waves = None

        # ---- 1. the soliton's box becomes a fibre ----
        ax1 = make_axes([0, 1.1, 0])
        ax2 = make_axes([0, -1.9, 0])
        fib1, fib2 = fibre_background(ax1), fibre_background(ax2)
        core1, stripes1 = fib1
        core2, stripes2 = fib2

        self.play(FadeOut(P["arrows"]), FadeOut(P["legend"]), FadeOut(P["result"]), run_time=0.6)
        self.set_title("The same equation describes light pulses in a glass fibre")
        pulse = VMobject(stroke_color=BLUE_B, stroke_width=2.5).set_points_as_corners(
            [ax1.c2p(a, SURF_Y + lin * b) for a, b in zip(x, in_gap[0])])
        self.play(P["box"].animate.stretch_to_fit_width(core1.width).stretch_to_fit_height(core1.height)
                  .move_to(core1).set_stroke(GREY_B, width=1.5).set_fill(GREY_E, opacity=0.4),
                  ReplacementTransform(P["curve"], pulse), run_time=2.2)
        self.play(FadeIn(core1), FadeOut(P["box"]), run_time=0.3)
        self.add(pulse)
        self.next_slide()

        # ---- 2. a grating appears; a second fibre joins ----
        self.set_title("Light: make the fibre periodic (a grating)")
        tag = Text("grating", font_size=16, color=GRATING).next_to(ax1.c2p(LG / 2, 1.5), UP, buff=0.05)
        self.play(LaggedStart(*[FadeIn(st) for st in stripes1], lag_ratio=0.08), FadeIn(tag), run_time=1.5)
        self.remove(*stripes1)
        self.add(stripes1)
        self.add(pulse)
        self.play(FadeIn(core2), FadeIn(stripes2))
        self.set_note("each stripe reflects a little; with the right spacing all the echoes add up")
        self.run(ax1, ax2, (x, in_gap, lin), (x, off_gap, lin), BLUE_B,
                 (side_label(ax1, "wavelength = 2 × spacing:  reflected", YELLOW),
                  side_label(ax2, "other wavelength:  passes through")), hand_off=pulse)
        self.next_slide()

        # ---- 3. strong vs weak in the gap ----
        self.set_title("Light: a strong pulse gets through the gap")
        self.set_note("")
        self.run(ax1, ax2, (x, weak, nl), (x, strong, nl), BLUE_B,
                 (side_label(ax1, "weak pulse in the gap:  reflected"),
                  side_label(ax2, "strong pulse in the gap:  travels through as a gap soliton", YELLOW)))
        self.set_note("observed in optical-fibre gratings (Eggleton et al., 1996)")
        self.next_slide()

        # ---- 4. the fibre becomes water; the stripes melt into sandbars ----
        wat1, wat2 = water_background(ax1), water_background(ax2)
        self.set_title("Water: sandbars play the role of the grating")
        self.set_note("")
        self.play(FadeOut(self.waves), FadeOut(tag),
                  ReplacementTransform(core1, wat1[0]), ReplacementTransform(stripes1, wat1[1]),
                  ReplacementTransform(core2, wat2[0]), ReplacementTransform(stripes2, wat2[1]),
                  run_time=2.2)
        self.waves = None
        self.run(ax1, ax2, (x, in_gap, lin), (x, off_gap, lin), WHITE,
                 (side_label(ax1, "sandbar spacing = half the wavelength:  reflected", YELLOW),
                  side_label(ax2, "other wavelength:  passes over")))
        self.set_note("the same mathematics: a band of wavelengths that cannot pass (the band gap)")
        self.next_slide()

        self.set_title("Water: a strong wave should get through too")
        self.set_note("")
        self.run(ax1, ax2, (x, weak, nl), (x, strong, nl), WHITE,
                 (side_label(ax1, "weak wave in the gap:  reflected"),
                  side_label(ax2, "strong wave in the gap:  travels through as a gap soliton", YELLOW)))
        self.set_note("predicted for water waves (Ruban, 2008): what my experiments aim to test")
        self.next_slide()