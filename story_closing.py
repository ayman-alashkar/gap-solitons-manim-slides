"""
Part 4 (closing) as interactive slides, starting from the last frame of part 3.
"""

import numpy as np
from manim import *
from manim_slides import Slide
from bragg_scene import solve_cme, make_axes, fibre_background, water_background, live_wave, side_label

CORAL = "#D85A30"
TEAL = "#1D9E75"
GREEN_OK = "#5DCAA5"
AMBER = "#EF9F27"
SAND = "#C2A267"


def item(text, done, size=18):
    mark = Dot(radius=0.08, color=GREEN_OK) if done else Circle(radius=0.08, color=AMBER, stroke_width=2.5)
    return VGroup(mark, Text(text, font_size=size)).arrange(RIGHT, buff=0.25)


class BraggToClosing(Slide):
    def set_title(self, text, size=32):
        new = Text(text, font_size=size).to_edge(UP)
        self.play(FadeOut(self.title), run_time=0.3)
        self.play(FadeIn(new), run_time=0.45)
        self.title = new

    def construct(self):
        # ---------------- 0. part 3's last frame ----------------
        x, weak = solve_cme(0.3, q=0.0, nonlinear=True)
        _, strong = solve_cme(1.0, q=0.0, nonlinear=True)
        nl = 1.0 / np.abs(strong).max()
        ax1, ax2 = make_axes([0, 1.1, 0]), make_axes([0, -1.9, 0])
        wat1, wat2 = water_background(ax1), water_background(ax2)
        done = ValueTracker(1.0)
        w1 = live_wave(ax1, x, weak, done, nl, color=WHITE)
        w2 = live_wave(ax2, x, strong, done, nl, color=WHITE)
        lab1 = side_label(ax1, "weak wave in the gap:  reflected")
        lab2 = side_label(ax2, "strong wave in the gap:  travels through as a gap soliton", YELLOW)
        self.title = Text("Water: a strong wave should get through too", font_size=30).to_edge(UP)
        note = Text("predicted for water waves (Ruban, 2008): what my experiments aim to test",
                    font_size=20, color=GREY_A).to_edge(DOWN, buff=0.3)
        self.add(wat1, wat2, w1, w2, lab1, lab2, self.title, note)

        # ---------------- 1. where I stand ----------------
        water_icon = VGroup(wat1, wat2, w1.copy().clear_updaters(), w2.copy().clear_updaters())
        self.remove(w1, w2)
        self.add(water_icon)

        def card(title, status, ref, color, pos):
            box = RoundedRectangle(corner_radius=0.15, width=5.0, height=3.0, stroke_color=color,
                                   fill_color=color, fill_opacity=0.1).move_to(pos)
            text = VGroup(Text(title, font_size=24, weight=BOLD), Text(status, font_size=17, color=color),
                          Text(ref, font_size=13, color=GREY_A)).arrange(DOWN, buff=0.06)
            text.next_to(box.get_bottom(), UP, buff=0.15)
            return box, text

        w_box, w_text = card("Water", "gap solitons predicted", "Ruban, 2008", AMBER, [3.0, 1.0, 0])
        l_box, l_text = card("Light", "gap solitons observed", "Eggleton et al., 1996", GREEN_OK, [-3.0, 1.0, 0])
        icon_slot_w = w_box.get_top() + DOWN * 0.95
        icon_slot_l = l_box.get_top() + DOWN * 0.95

        self.play(FadeOut(VGroup(lab1, lab2, note)), run_time=0.4)
        self.set_title("Where I stand")
        self.play(water_icon.animate.scale_to_fit_width(4.4).move_to(icon_slot_w), FadeIn(w_box), run_time=1.6)
        self.play(FadeIn(w_text))

        bx1, bx2 = make_axes([0, 1.1, 0]), make_axes([0, -1.9, 0])
        light_icon = VGroup(fibre_background(bx1), fibre_background(bx2),
                            live_wave(bx1, x, weak, done, nl, color=BLUE_B).clear_updaters(),
                            live_wave(bx2, x, strong, done, nl, color=BLUE_B).clear_updaters())
        light_icon.scale_to_fit_width(4.4).move_to(icon_slot_l)
        self.play(FadeIn(l_box), FadeIn(light_icon), FadeIn(l_text))
        self.next_slide()

        have = VGroup(Text("what I have", font_size=20, color=GREY_A),
                      item("the theory, built up to NLS", True),
                      item("the Bragg picture, in light and in water", True),
                      item("model simulations (shown in this talk)", True)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        not_yet = VGroup(Text("not yet", font_size=20, color=GREY_A),
                         item("experiments", False),
                         item("Ruban's coefficients, derived myself", False)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        lists = VGroup(have, not_yet).arrange(RIGHT, buff=1.2, aligned_edge=UP).move_to([0, -2.3, 0])
        if lists.width > 12.5:
            lists.scale_to_fit_width(12.5)
        self.play(FadeIn(have))
        self.next_slide()
        self.play(FadeIn(not_yet))
        self.play(Indicate(not_yet[1], color=AMBER, scale_factor=1.08))
        self.next_slide()

        # ---------------- 2. the bottleneck ----------------
        L = 20.0
        axb = Axes(x_range=[0, L, 5], y_range=[0, 1.2, 0.5], x_length=10, y_length=3.4, tips=False,
                   axis_config={"include_ticks": False, "color": GREY_B}).move_to([0, -0.5, 0])
        xs = np.linspace(0, L, 400)
        bars = VMobject(stroke_color=SAND, stroke_width=3).set_points_smoothly(
            [axb.c2p(xv, 0) + DOWN * (0.12 + 0.08 * np.sin(2 * np.pi * xv / 1.4)) for xv in xs])
        xl = Text("distance across the sandbars", font_size=16, color=GREY_A).next_to(axb, DOWN, buff=0.4)
        yl = Text("wave amplitude", font_size=16, color=GREY_A).rotate(PI / 2).next_to(axb, LEFT, buff=0.15)
        sub = VGroup(Text("viscous boundary layers (bottom, sidewalls, surface films)", font_size=19),
                     Text("take a little energy on every sandbar period", font_size=19)
                     ).arrange(DOWN, buff=0.06)

        self.play(FadeOut(VGroup(have, not_yet, l_box, l_text, light_icon, w_text, w_box)), run_time=0.6)
        self.set_title("The bottleneck: friction is back")
        sub.next_to(self.title, DOWN, buff=0.2)
        self.play(ReplacementTransform(water_icon, bars), FadeIn(sub), run_time=1.6)
        self.play(Create(axb), FadeIn(xl), FadeIn(yl))

        thr = 0.6
        thr_line = DashedLine(axb.c2p(0, thr), axb.c2p(L, thr), color=YELLOW)
        thr_lab = Text("threshold to get through the gap", font_size=15, color=YELLOW)
        thr_lab.next_to(axb.c2p(L, thr), UP, buff=0.08).align_to(thr_line, RIGHT)
        tag = Text("schematic", font_size=14, color=GREY_B, slant=ITALIC).next_to(axb, RIGHT, buff=0.1).align_to(axb, UP)
        self.play(Create(thr_line), FadeIn(thr_lab), FadeIn(tag))
        low = axb.plot(lambda v: np.exp(-0.015 * v), x_range=[0, L], color=GREEN_OK, stroke_width=4)
        high = axb.plot(lambda v: np.exp(-0.05 * v), x_range=[0, L], color=CORAL, stroke_width=4)
        lab_low = Text("little friction: stays above, gets through", font_size=16, color=GREEN_OK)
        lab_low.next_to(axb.c2p(L, np.exp(-0.015 * L)), UP, buff=0.4).align_to(axb, RIGHT)
        cross = Dot(axb.c2p(np.log(1 / thr) / 0.05, thr), color=CORAL, radius=0.09)
        lab_high = Text("too much friction: drops below, reflected", font_size=16, color=CORAL)
        lab_high.next_to(axb.c2p(L, np.exp(-0.05 * L)), DOWN, buff=0.12).align_to(axb, RIGHT)
        self.play(Create(low), run_time=2.5, rate_func=linear)
        self.play(FadeIn(lab_low))
        self.play(Create(high), run_time=2.5, rate_func=linear)
        self.play(FadeIn(cross, scale=1.5), FadeIn(lab_high))
        loop = Text("the first assumption we made (no friction) is the one the tank will test",
                    font_size=19, color=GREY_A).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(loop))
        self.next_slide()

        # ---------------- 3. what would settle it ----------------
        def box(title, subt, color, pos):
            r = RoundedRectangle(corner_radius=0.15, width=4.8, height=1.6, stroke_color=color,
                                 fill_color=color, fill_opacity=0.12).move_to(pos)
            t = VGroup(Text(title, font_size=23, weight=BOLD), Text(subt, font_size=16, color=GREY_A)
                       ).arrange(DOWN, buff=0.1).move_to(r)
            return VGroup(r, t)

        tank = box("Tank experiments", "waves over a built periodic bottom", TEAL, [-3.2, 0.4, 0])
        solver = box("My own solver", "Ruban's coupled-mode equations", BLUE_B, [3.2, 0.4, 0])
        plot = VGroup(axb, bars, xl, yl, thr_line, thr_lab, tag, low, high, lab_low, lab_high, cross)
        self.play(FadeOut(VGroup(sub, loop)), run_time=0.4)
        self.set_title("What would settle it")
        self.play(plot.animate.scale_to_fit_width(3.6).move_to(tank[0]).set_opacity(0.0),
                  FadeIn(tank), run_time=1.6)
        self.remove(plot)
        self.play(FadeIn(solver, shift=LEFT * 0.3))
        arrow = DoubleArrow(tank.get_right(), solver.get_left(), buff=0.15, color=WHITE)
        comp = Text("compare", font_size=18).next_to(arrow, UP, buff=0.1)
        self.play(GrowArrow(arrow), FadeIn(comp))
        out = Text("does a strong wave really get through, and for how long?",
                   font_size=21, color=YELLOW).next_to(VGroup(tank, solver), DOWN, buff=0.8)
        self.play(FadeIn(out))
        self.next_slide()

        # ---------------- 4. the new question ----------------
        self.play(FadeOut(VGroup(tank, solver, arrow, comp)), out.animate.next_to(self.title, DOWN, buff=1.0))
        self.set_title("A question I did not have before this course", size=30)
        q = VGroup(Text("and if we can trap a wave in the gap,", font_size=30),
                   Text("could we harvest its energy", font_size=30),
                   Text("without un-trapping it?", font_size=30, color=YELLOW)).arrange(DOWN, buff=0.18)
        q.next_to(out, DOWN, buff=0.5)
        for line in q:
            self.play(FadeIn(line), run_time=0.7)
        self.next_slide()
        notes = VGroup(
            Text("trapping concentrates energy; it does not create it", font_size=17, color=GREY_A),
            Text("taking energy out is itself a loss: too much, and the trap opens", font_size=17, color=GREY_A),
            Text("Bragg focusing for wave energy has been studied (linear);", font_size=17, color=GREY_A),
            Text("I did not find the self-trapped (nonlinear) version", font_size=17, color=GREY_A),
        ).arrange(DOWN, buff=0.08).next_to(q, DOWN, buff=0.5)
        for n in notes:
            self.play(FadeIn(n), run_time=0.5)
        self.next_slide()

        # ---------------- 5. a personal note ----------------
        self.play(*[FadeOut(m) for m in self.mobjects if m is not self.title])
        self.set_title("A personal note", size=26)
        meaning = Text("“He who made for you, out of the green tree, fire — and see, you kindle from it.”",
                       font_size=22, slant=ITALIC)
        ref = Text("Qur’an, Ya-Sin 36:80", font_size=16, color=GREY_B)
        block = VGroup(meaning, ref).arrange(DOWN, buff=0.25).shift(UP * 0.9)
        self.play(FadeIn(block), run_time=1.2)
        self.next_slide()
        reflection = VGroup(
            Text("I read it many times without feeling it.", font_size=21),
            Text("After this course, it meant something.", font_size=21),
            Text("How much complexity hides in simple things,", font_size=21, color=YELLOW),
            Text("and how much I still do not know.", font_size=21, color=YELLOW),
        ).arrange(DOWN, buff=0.1).next_to(block, DOWN, buff=0.7)
        self.play(FadeIn(reflection[:2]))
        self.next_slide()
        self.play(FadeIn(reflection[2:]))
        self.next_slide()
        self.play(*[FadeOut(m) for m in self.mobjects])
        thanks = Text("Thank you", font_size=44)
        self.play(FadeIn(thanks))
        self.next_slide()