"""
NLS as a visual tree: every box is a short moving clip, not a word.

    [ dispersion: packet spreads ]   [ nonlinearity: crests steepen ]
                         \\             /
               [ NLS: one rule for the envelope ]
                  /                          \\
   Focusing: nonlinearity             Defocusing: nonlinearity
   fights the spreading               adds to the spreading
    /        |        \\               /         |          \\
 pulses   soliton   rogue wave     bump      dark soliton   shock
                                   travels                  (ripples)

Envelope equation used for the leaves (scaled units; the 1/2 and |s| = 1 are
choices of scaling):
        i psi_t + (1/2) psi_xx + s |psi|^2 psi = 0,   s = +1 focusing, -1 defocusing
Clips show |psi| (the envelope height).

Where each clip comes from:
    split-step simulation : dispersion only, modulational instability, moving bright
                            soliton, bump on a background, step (dispersive shock)
    exact formulas        : Peregrine breather (rogue wave), grey dark soliton
                            (exact solutions are cleaner than simulating them on a
                            periodic box)
    top-right clip        : NOT NLS. Inviscid Burgers u_t + u u_x = 0, solved exactly
                            before breaking; used only to show "taller parts move
                            faster", which is what nonlinearity means here.

Render:   manim -pql nls_visual_tree.py NLSVisualTree    (preview)
          manim -pqh nls_visual_tree.py NLSVisualTree    (final)
"""

import numpy as np
from manim import *

CORAL = "#D85A30"
TEAL = "#1D9E75"
TRUNK = GREY_B
N_FRAMES = 120
SHOW_MATH = True          # set False to hide the math tags
MATH = "#9FC9FF"          # colour of the math tags

# Math tags use the notation of the derivation notes:
#     i A_tau + P A_xixi + Q |A|^2 A = 0
# P comes from dispersion (P ∝ ω''(k)), Q from the nonlinear frequency shift.
# Checked against the scaled simulations (P = 1/2, Q = ±1, |A| = 1):
#   max ripple growth rate (focusing)  = Q|A|^2      -> 1, fastest ripple K = sqrt(2) ✓
#   ripple speed (defocusing)          = |A| sqrt(-2PQ) -> 1, bump halves move at speed 1 ✓
#   Peregrine peak                     = 3 x background ✓


# ======================================================================
# Physics: all clips are precomputed as arrays  frames[time_index, x_index]
# ======================================================================
def solve_nls(psi0, L, s, t_end, dt=0.002):
    """Split-step Fourier (Strang). Returns |psi| at N_FRAMES times in [0, t_end]."""
    N = psi0.size
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    half_linear = np.exp(-1j * k**2 * dt / 4)          # exp(-i k^2 (dt/2) / 2)
    n_steps = int(round(t_end / dt))
    save_at = set(np.linspace(0, n_steps, N_FRAMES).round().astype(int))

    psi = psi0.astype(complex)
    frames = [np.abs(psi)]
    for step in range(1, n_steps + 1):
        psi = np.fft.ifft(half_linear * np.fft.fft(psi))
        psi = psi * np.exp(1j * s * np.abs(psi)**2 * dt)
        psi = np.fft.ifft(half_linear * np.fft.fft(psi))
        if step in save_at:
            frames.append(np.abs(psi))
    return np.array(frames)


def grid(L, N):
    return np.linspace(-L / 2, L / 2, N, endpoint=False)


def clip_dispersion_only():
    x = grid(60, 512)
    return x, solve_nls(np.exp(-x**2), 60, 0, 6.0)


def clip_burgers():
    # u_t + u u_x = 0, u(x,0) = sin x.  Exact: u = sin(x - u t), valid for t < 1.
    # Solve the implicit relation with Newton at each time.
    x = np.linspace(0, 4 * np.pi, 400)
    frames = []
    for t in np.linspace(0, 0.9, N_FRAMES):
        u = np.sin(x)
        for _ in range(50):
            f = u - np.sin(x - u * t)
            u = u - f / (1 + t * np.cos(x - u * t))
        frames.append(u)
    return x, np.array(frames)


def clip_carrier():
    # carrier wave under a fixed envelope: eta = sech(x/1.5) cos(6x - phase)
    x = np.linspace(-6, 6, 500)
    env = 1 / np.cosh(x / 1.5)
    frames = np.array([env * np.cos(6 * x - p)
                       for p in np.linspace(0, 6 * np.pi, N_FRAMES)])
    return x, frames, env


def clip_mi():
    # uniform wave + tiny ripple; K = sqrt(2) is the fastest-growing ripple for amplitude 1
    K = np.sqrt(2)
    L = 4 * 2 * np.pi / K
    x = grid(L, 256)
    return x, solve_nls(1 + 0.01 * np.cos(K * x), L, +1, 8.0)


def clip_bright():
    # sech(x) e^{i v x}: a bright soliton moving at speed v = 1
    x = grid(40, 512)
    return x, solve_nls((1 / np.cosh(x + 6)) * np.exp(1j * x), 40, +1, 12.0)


def clip_peregrine():
    # exact Peregrine breather of the focusing equation:
    # psi = [1 - 4(1 + 2it) / (1 + 4x^2 + 4t^2)] e^{it};  peak |psi| = 3 at x = t = 0
    x = np.linspace(-8, 8, 400)
    frames = []
    for t in np.linspace(-3, 3, N_FRAMES):
        psi = 1 - 4 * (1 + 2j * t) / (1 + 4 * x**2 + 4 * t**2)
        frames.append(np.abs(psi))
    return x, np.array(frames)


def clip_bump():
    # small bump on a uniform background; defocusing: it splits and travels away
    # (its panel is zoomed to heights 0.8-1.2 so the small bump is visible)
    x = grid(60, 512)
    return x, solve_nls(1 + 0.15 * np.exp(-(x / 3)**2), 60, -1, 12.0)


def clip_dark():
    # exact grey soliton of the defocusing equation (background 1):
    # psi = (B tanh(B(x - v t)) + i v) e^{-it},  B^2 + v^2 = 1
    v = 0.5
    B = np.sqrt(1 - v**2)
    x = np.linspace(-10, 10, 400)
    frames = []
    for t in np.linspace(-10, 10, N_FRAMES):
        psi = B * np.tanh(B * (x - v * t)) + 1j * v
        frames.append(np.abs(psi))
    return x, np.array(frames)


def clip_shock():
    # high plateau (|psi| = 1) next to a low one (0.5); the step breaks into ripples
    x = grid(80, 1024)
    psi0 = 0.5 + 0.25 * (np.tanh(x + 20) - np.tanh(x - 20))
    t_end = 6.0
    frames = solve_nls(psi0, 80, -1, t_end)
    view = (x > 10) & (x < 35)          # show the right-hand step only
    return x[view], frames[:, view]


# ======================================================================
# Drawing helpers
# ======================================================================
def make_panel(x, y_range, color, w, h, pos):
    box = RoundedRectangle(corner_radius=0.1, width=w, height=h,
                           stroke_color=color, stroke_width=2,
                           fill_color=color, fill_opacity=0.1).move_to(pos)
    ax = Axes(x_range=[x[0], x[-1]], y_range=y_range,
              x_length=w - 0.2, y_length=h - 0.2, tips=False,
              axis_config={"include_ticks": False, "stroke_opacity": 0.0})
    ax.move_to(box)
    return box, ax


def live_curve(ax, x, frames, s, color, width=3):
    """Curve showing frames[i], with i set by tracker s in [0, 1]."""
    step = max(1, len(x) // 300)                  # at most ~300 points per curve
    xs = x[::step]
    def draw():
        i = int(round(s.get_value() * (len(frames) - 1)))
        pts = [ax.c2p(a, b) for a, b in zip(xs, frames[i][::step])]
        return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts)
    return always_redraw(draw)


def math_tag(markup, size=16):
    return MarkupText(markup, font_size=size, color=MATH)


def caption(text, name, color=WHITE, math=None):
    # one Text per line so every line is centred
    lines = VGroup(*[Text(t, font_size=15, color=color) for t in text.split("\n")])
    lines.arrange(DOWN, buff=0.05)
    b = Text(name, font_size=12, color=GREY_B, slant=ITALIC)
    parts = [lines, b]
    if math and SHOW_MATH:
        parts.append(math_tag(math, size=12))
    return VGroup(*parts).arrange(DOWN, buff=0.06)


def elbow(start, end, color=TRUNK):
    mid_y = (start[1] + end[1]) / 2
    p1, p2 = np.array([start[0], mid_y, 0]), np.array([end[0], mid_y, 0])
    path = VMobject(stroke_color=color, stroke_width=2).set_points_as_corners([start, p1, p2])
    tip = Arrow(p2, end, buff=0, color=color, stroke_width=2, tip_length=0.13)
    return VGroup(path, tip)


# ======================================================================
# Scene
# ======================================================================
class NLSVisualTree(Scene):
    def construct(self):
        # ---------------- row 1: the two effects ----------------
        s_top = ValueTracker(0)
        x, f = clip_dispersion_only()
        box_d, ax_d = make_panel(x, [0, 1.1], TRUNK, 3.4, 1.1, [-2.6, 2.95, 0])
        cap_d = caption("dispersion: a packet spreads", "different wavelengths travel at different speeds")
        cap_d.next_to(box_d, DOWN, buff=0.08)
        curve_d = live_curve(ax_d, x, f, s_top, BLUE)

        x, f = clip_burgers()
        box_n, ax_n = make_panel(x, [-1.2, 1.2], TRUNK, 3.4, 1.1, [2.6, 2.95, 0])
        cap_n = caption("nonlinearity: crests steepen", "taller parts move faster")
        cap_n.next_to(box_n, DOWN, buff=0.08)
        curve_n = live_curve(ax_n, x, f, s_top, BLUE)

        tags_top = []
        if SHOW_MATH:
            tag_d = VGroup(math_tag("P A<sub>ξξ</sub>", 20), math_tag("P ∝ ω″(k)", 14)
                           ).arrange(DOWN, buff=0.08).next_to(box_d, LEFT, buff=0.3)
            tag_n = VGroup(math_tag("Q |A|²A", 20), math_tag("Q ∝ frequency shift", 14)
                           ).arrange(DOWN, buff=0.08).next_to(box_n, RIGHT, buff=0.3)
            tags_top = [FadeIn(tag_d), FadeIn(tag_n)]
        self.play(FadeIn(box_d), FadeIn(box_n), FadeIn(cap_d), FadeIn(cap_n), *tags_top)
        self.add(curve_d, curve_n)
        self.play(s_top.animate.set_value(1), run_time=5, rate_func=linear)
        self.wait(0.5)

        # ---------------- row 2: NLS = rule for the envelope ----------------
        s_mid = ValueTracker(0)
        x, f, env = clip_carrier()
        box_e, ax_e = make_panel(x, [-1.2, 1.2], TRUNK, 3.8, 0.9, [0, 1.0, 0])
        cap_e = caption("NLS: one rule for how the envelope (yellow) moves",
                        "nonlinear Schrödinger equation")
        cap_e.next_to(box_e, DOWN, buff=0.08)
        env_top = ax_e.plot_line_graph(x, env, add_vertex_dots=False,
                                       line_color=YELLOW, stroke_width=3)
        env_bot = ax_e.plot_line_graph(x, -env, add_vertex_dots=False,
                                       line_color=YELLOW, stroke_width=3)
        carrier = live_curve(ax_e, x, f, s_mid, BLUE_B, width=2)

        a1 = elbow(cap_d.get_bottom(), box_e.get_top() + LEFT * 0.6)
        a2 = elbow(cap_n.get_bottom(), box_e.get_top() + RIGHT * 0.6)
        tags_mid = []
        if SHOW_MATH:
            tag_e = math_tag("i A<sub>τ</sub> + P A<sub>ξξ</sub> + Q |A|²A = 0", 18)
            tag_e.next_to(box_e, RIGHT, buff=0.3)
            tags_mid = [FadeIn(tag_e)]
        self.play(Create(a1), Create(a2), FadeIn(box_e), FadeIn(cap_e), *tags_mid)
        self.add(carrier)
        self.play(Create(env_top), Create(env_bot))
        self.play(s_mid.animate.set_value(1), run_time=3, rate_func=linear)

        # ---------------- row 3: the split ----------------
        def split_node(title, sub, color, xpos):
            box = RoundedRectangle(corner_radius=0.1, width=4.2, height=0.62,
                                   stroke_color=color, stroke_width=2,
                                   fill_color=color, fill_opacity=0.15)
            label = VGroup(Text(title, font_size=19, weight=BOLD),
                           Text(sub, font_size=14, color=GREY_A)).arrange(DOWN, buff=0.05)
            box.move_to([xpos, -0.65, 0])
            label.move_to(box)
            return VGroup(box, label)

        foc = split_node("Focusing", "nonlinearity fights the spreading", CORAL, -3.7)
        defoc = split_node("Defocusing", "nonlinearity adds to the spreading", TEAL, 3.7)
        self.play(Create(elbow(cap_e.get_bottom(), foc.get_top(), CORAL)),
                  Create(elbow(cap_e.get_bottom(), defoc.get_top(), TEAL)),
                  FadeIn(foc, shift=DOWN * 0.15), FadeIn(defoc, shift=DOWN * 0.15))
        if SHOW_MATH:
            tag_f = math_tag("PQ > 0", 20).next_to(foc, RIGHT, buff=0.2)
            tag_df = math_tag("PQ &lt; 0", 20).next_to(defoc, LEFT, buff=0.2)
            self.play(FadeIn(tag_f), FadeIn(tag_df))
        self.wait(1)

        # ---------------- row 4: leaves ----------------
        def build_leaves(specs, parent, color, s):
            parts, curves, arrows = [], [], []
            for (x, f, y_range, xpos, text, name, highlight, math) in specs:
                box, ax = make_panel(x, y_range, color, 2.05, 1.0, [xpos, -2.05, 0])
                cap = caption(text, name, YELLOW if highlight else WHITE, math)
                cap.next_to(box, DOWN, buff=0.08)
                parts.append(VGroup(box, cap))
                curves.append(live_curve(ax, x, f, s, color))
                arrows.append(elbow(parent.get_bottom(), box.get_top(), color))
            return parts, curves, arrows

        s_foc = ValueTracker(0)
        foc_specs = [
            (*clip_mi(), [0, 2.7], -5.9, "uniform wave\nbreaks into pulses", "modulational instability", False, "max growth rate = Q|A|²"),
            (*clip_bright(), [0, 1.2], -3.7, "a hump that\nholds its shape", "bright soliton", False, "|A| ∝ sech"),
            (*clip_peregrine(), [0, 3.2], -1.5, "a giant wave appears\nand vanishes", "rogue wave", True, "peak = 3 × background"),
        ]
        parts, curves, arrows = build_leaves(foc_specs, foc, CORAL, s_foc)
        self.play(*[Create(a) for a in arrows], *[FadeIn(p) for p in parts])
        self.add(*curves)
        self.play(s_foc.animate.set_value(1), run_time=6, rate_func=linear)
        self.wait(0.5)

        s_def = ValueTracker(0)
        def_specs = [
            (*clip_bump(), [0.8, 1.2], 1.5, "a bump just\ntravels away", "stable modulation", False, "ripple speed = |A|√(−2PQ)"),
            (*clip_dark(), [0, 1.2], 3.7, "a hole that\nholds its shape", "dark soliton", False, "|A| ∝ tanh"),
            (*clip_shock(), [0, 1.5], 5.9, "a step breaks\ninto ripples", "dispersive shock wave", False, "steep front → ripple train"),
        ]
        parts, curves, arrows = build_leaves(def_specs, defoc, TEAL, s_def)
        self.play(*[Create(a) for a in arrows], *[FadeIn(p) for p in parts])
        self.add(*curves)
        self.play(s_def.animate.set_value(1), run_time=6, rate_func=linear)
        self.wait(0.5)

        # ---------------- replay everything together ----------------
        trackers = [s_top, s_mid, s_foc, s_def]
        for s in trackers:
            s.set_value(0)
        self.play(*[s.animate.set_value(1) for s in trackers], run_time=7, rate_func=linear)
        self.wait(2)
