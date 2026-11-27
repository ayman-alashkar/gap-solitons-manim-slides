"""
Bragg scene: optics first, then water, then the gap soliton.

    Optics, then water, each told in full:
      a. linear: a wave whose wavelength is twice the grating spacing is reflected;
         another wavelength passes.
      b. nonlinear: a weak wave in the gap is reflected, a strong one gets through
         as a gap soliton (observed in fibre gratings; predicted for water).
    The water bottom is drawn as a side view (cross-section along the wave's path).

Model: nonlinear coupled-mode equations for a forward envelope u and a backward
envelope v (scaled units, group speed outside the grating = 1):

    i(u_t + u_x) + kappa v + (|u|^2 + 2|v|^2) u = 0
    i(v_t - v_x) + kappa u + (|v|^2 + 2|u|^2) v = 0

kappa = coupling strength (nonzero only inside the grating). The nonlinear
coefficients (1, 2) are the optics (Kerr) form. Ruban's water-wave equations have
the same structure with different coefficients; the behaviour shown here is the
qualitative one they share. This is a model, not a simulation of the full water-wave
equations.

Numerics: split-step, each piece solved exactly
    advection  u(x,t+dt) = u(x-dt),  v(x,t+dt) = v(x+dt)      (Fourier shift)
    coupling   u <- cos(kappa dt) u + i sin(kappa dt) v,  v <- i sin(kappa dt) u + cos(kappa dt) v
    nonlinear  u <- u exp(i(|u|^2 + 2|v|^2) dt),  v likewise
Checked: weak pulse in the gap ~97% reflected; amplitude 1.0 ~80% transmitted.

What is drawn is a wave built from the two envelopes:
    eta = |u| cos(k (x - t)) + |v| cos(k (x + t)),   k = k0 + q
(Ruban's surface formula is Re[u e^{i(kx - wt)} + v e^{-i(kx + wt)}]; the envelope
phase is dropped here for readability, since nonlinearity chirps it strongly.)
k0 (carrier wavenumber) and the carrier speed 1 are display choices. The grating
period is pi/k0 = half the carrier wavelength (the Bragg condition).

Render:   manim -pql bragg_scene.py Bragg    (preview)
          manim -pqh bragg_scene.py Bragg    (final)
"""

import numpy as np
from manim import *

K0 = 2.0                       # carrier wavenumber (display choice); wavelength = pi
KAPPA, LG = 1.0, 20.0          # coupling strength, grating length (grating on 0 < x < LG)
X_VIEW = (-40.0, 45.0)         # part of the domain that is drawn
N_FRAMES = 150
T_END = 55.0

GRATING = "#7F77DD"
WATER = BLUE_D
SAND = "#C2A267"


# ======================================================================
# Physics
# ======================================================================
def solve_cme(A, q=0.0, nonlinear=True, L=160.0, N=4096, dt=0.01):
    """Return x_view and eta(x, t) at N_FRAMES times for an incident pulse of amplitude A.
    q shifts the pulse's wavenumber away from the Bragg value (q = 0: centre of the gap)."""
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kap = np.where((x > 0) & (x < LG), KAPPA, 0.0)

    u = A * np.exp(-((x + 20) / 4) ** 2) * np.exp(1j * q * x)   # starts at x = -20
    v = np.zeros_like(u)

    half_adv_u = np.exp(-1j * k * dt / 2)     # u moves right by dt/2
    half_adv_v = np.exp(+1j * k * dt / 2)     # v moves left by dt/2
    c, s = np.cos(kap * dt), np.sin(kap * dt)

    view = (x >= X_VIEW[0]) & (x <= X_VIEW[1])
    xv = x[view]
    n_steps = int(round(T_END / dt))
    save_at = set(np.linspace(0, n_steps, N_FRAMES).round().astype(int))

    kd = K0 + q                          # displayed carrier wavenumber

    # The model has no dispersion term, so strong runs grow envelope features finer
    # than one carrier wavelength, where the model itself is not valid (it assumes the
    # envelope varies slowly). For display only, |u| and |v| are smoothed with a
    # Gaussian of width 1 (about a third of a wavelength).
    smooth = np.exp(-0.5 * (k * 1.0) ** 2)

    def env(a):
        return np.abs(np.fft.ifft(smooth * np.fft.fft(np.abs(a))))[view]

    def eta(u, v, t):
        # carrier drawn from the envelope HEIGHTS |u|, |v| only. The envelope phase
        # (a strong chirp in the nonlinear runs) is left out so the wave stays readable.
        return env(u) * np.cos(kd * (xv - t)) + env(v) * np.cos(kd * (xv + t))

    frames = [eta(u, v, 0.0)]
    for step in range(1, n_steps + 1):
        u = np.fft.ifft(half_adv_u * np.fft.fft(u))
        v = np.fft.ifft(half_adv_v * np.fft.fft(v))
        u, v = c * u + 1j * s * v, 1j * s * u + c * v
        if nonlinear:
            pu, pv = np.abs(u) ** 2, np.abs(v) ** 2
            u = u * np.exp(1j * (pu + 2 * pv) * dt)
            v = v * np.exp(1j * (pv + 2 * pu) * dt)
        u = np.fft.ifft(half_adv_u * np.fft.fft(u))
        v = np.fft.ifft(half_adv_v * np.fft.fft(v))
        if step in save_at:
            frames.append(eta(u, v, step * dt))
    return xv, np.array(frames)


# ======================================================================
# Drawing helpers
# ======================================================================
W, H = 12.0, 1.7               # panel size on screen
SURF_Y = 0.35                  # where the undisturbed wave line sits (axis units)


def make_axes(pos):
    return Axes(x_range=[*X_VIEW], y_range=[-1.6, 1.6], x_length=W, y_length=H,
                tips=False, axis_config={"include_ticks": False, "stroke_opacity": 0}
                ).move_to(pos)


def fibre_background(ax):
    """Glass fibre with stripes in the grating (one stripe per half period)."""
    core = Polygon(ax.c2p(X_VIEW[0], -1.0), ax.c2p(X_VIEW[1], -1.0),
                   ax.c2p(X_VIEW[1], 1.5), ax.c2p(X_VIEW[0], 1.5),
                   stroke_color=GREY_B, stroke_width=1.5, fill_color=GREY_E, fill_opacity=0.4)
    period = np.pi / K0
    stripes = VGroup()
    for x0 in np.arange(0, LG, period):
        stripes.add(Polygon(ax.c2p(x0, -1.0), ax.c2p(x0 + period / 2, -1.0),
                            ax.c2p(x0 + period / 2, 1.5), ax.c2p(x0, 1.5),
                            stroke_width=0, fill_color=GRATING, fill_opacity=0.35))
    return VGroup(core, stripes)


def bottom_y(x):
    """Flat bottom with sinusoidal sandbars of period pi/K0 inside the grating."""
    ripple = np.where((x > 0) & (x < LG), 0.22 * np.sin(2 * K0 * x), 0.0)
    return -1.25 + ripple


def water_background(ax):
    xs = np.linspace(*X_VIEW, 800)
    bed = [ax.c2p(a, b) for a, b in zip(xs, bottom_y(xs))]
    water = Polygon(ax.c2p(X_VIEW[0], SURF_Y), ax.c2p(X_VIEW[1], SURF_Y), *bed[::-1],
                    stroke_width=0, fill_color=WATER, fill_opacity=0.25)
    sand = Polygon(*bed, ax.c2p(X_VIEW[1], -1.6), ax.c2p(X_VIEW[0], -1.6),
                   stroke_color=SAND, stroke_width=2, fill_color=SAND, fill_opacity=0.35)
    return VGroup(water, sand)


def live_wave(ax, x, frames, s, scale, color=BLUE_B):
    """Wave line eta(x, t) drawn on top of the background; frame chosen by tracker s."""
    def draw():
        i = int(round(s.get_value() * (len(frames) - 1)))
        pts = [ax.c2p(a, SURF_Y + scale * b) for a, b in zip(x, frames[i])]
        return VMobject(stroke_color=color, stroke_width=2.5).set_points_as_corners(pts)
    return always_redraw(draw)


def side_label(ax, text, color=WHITE):
    return Text(text, font_size=18, color=color).next_to(ax, UP, buff=0.05).align_to(ax, LEFT)


# ======================================================================
# Scene: the full story in optics, then the full story in water
# ======================================================================
class Bragg(Scene):
    def construct(self):
        # all simulations first
        x, in_gap = solve_cme(0.3, q=0.0, nonlinear=False)
        _, off_gap = solve_cme(0.3, q=2.5, nonlinear=False)
        _, weak = solve_cme(0.3, q=0.0, nonlinear=True)
        _, strong = solve_cme(1.0, q=0.0, nonlinear=True)

        lin_scale = 0.9 / np.abs(in_gap).max()        # linear runs: amplitude is arbitrary
        nl_scale = 1.0 / np.abs(strong).max()         # weak and strong share one scale

        # ---------------- optics ----------------
        self.two_panels(
            "Light: a periodic grating reflects one wavelength", fibre_background,
            (x, in_gap, lin_scale, "wavelength = 2 × spacing:  reflected", YELLOW),
            (x, off_gap, lin_scale, "other wavelength:  passes through", WHITE),
            "each stripe reflects a little; with this spacing all the echoes add up",
            show_grating_tag=True)
        self.two_panels(
            "Light: a strong pulse gets through the gap", fibre_background,
            (x, weak, nl_scale, "weak pulse in the gap:  reflected", WHITE),
            (x, strong, nl_scale, "strong pulse in the gap:  travels through as a gap soliton", YELLOW),
            "observed in optical-fibre gratings (Eggleton et al., 1996)")

        # ---------------- water ----------------
        self.two_panels(
            "Water: sandbars do the same to surface waves", water_background,
            (x, in_gap, lin_scale, "sandbar spacing = half the wavelength:  reflected", YELLOW),
            (x, off_gap, lin_scale, "other wavelength:  passes over", WHITE),
            "same mathematics: a band of wavelengths that cannot pass (the band gap)",
            wave_color=WHITE)
        self.two_panels(
            "Water: a strong wave should get through too", water_background,
            (x, weak, nl_scale, "weak wave in the gap:  reflected", WHITE),
            (x, strong, nl_scale, "strong wave in the gap:  travels through as a gap soliton", YELLOW),
            "predicted for water waves (Ruban, 2008): what my experiments aim to test",
            wave_color=WHITE)

    # ------------------------------------------------------------------
    def two_panels(self, title_text, background, top, bottom, note_text,
                   wave_color=BLUE_B, show_grating_tag=False):
        """Two stacked runs over the same medium, played together, then a note."""
        self.clear()
        title = Text(title_text, font_size=30).to_edge(UP)
        ax1 = make_axes([0, 1.1, 0])
        ax2 = make_axes([0, -1.9, 0])
        (x1, f1, sc1, t1, c1), (x2, f2, sc2, t2, c2) = top, bottom
        lab1 = side_label(ax1, t1, c1)
        lab2 = side_label(ax2, t2, c2)

        self.play(Write(title))
        extras = []
        if show_grating_tag:
            tag = Text("grating", font_size=16, color=GRATING)
            tag.next_to(ax1.c2p(LG / 2, 1.5), UP, buff=0.05)
            extras.append(FadeIn(tag))
        self.play(FadeIn(background(ax1)), FadeIn(background(ax2)),
                  FadeIn(lab1), FadeIn(lab2), *extras)

        s = ValueTracker(0)
        self.add(live_wave(ax1, x1, f1, s, sc1, color=wave_color),
                 live_wave(ax2, x2, f2, s, sc2, color=wave_color))
        self.play(s.animate.set_value(1), run_time=9, rate_func=linear)

        note = Text(note_text, font_size=20, color=GREY_A).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(2)
