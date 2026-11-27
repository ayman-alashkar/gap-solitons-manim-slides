"""
The whole talk as ONE interactive manim-slides presentation with animated slide numbers.

Render:   manim-slides render -q l full_talk.py FullTalk   (fast preview, 480p)
          manim-slides render -q h full_talk.py FullTalk   (final, 1080p)
Export:   manim-slides convert FullTalk talk.html --open
"""

import types
from manim import *
from manim_slides import Slide
from story_setup_dispersion import SetupToDispersion
from story_nls import DispersionToNLS
from story_bragg import SolitonToBragg
from story_closing import BraggToClosing

PARTS = [SetupToDispersion, DispersionToNLS, SolitonToBragg, BraggToClosing]

STATE = ["title", "cap", "note", "lab1", "lab2", "waves", "ax", "series",
         "disp_eq", "fade_rest", "A_lab", "disp", "nl", "equation", "eq_row",
         "soliton", "s_b"]


class FullTalk(Slide):
    def _make_slide_label(self, number):
        return (
            Text(str(number), font_size=20, color=GREY_B)
            .to_corner(DR, buff=0.25)
            .set_z_index(100)
        )

    def next_slide(self, *args, **kwargs):
        super().next_slide(*args, **kwargs)
        self.pending_slide_tick = True

    def play(self, *args, **kwargs):
        # Keep bulk FadeOuts in story_nls / story_closing from erasing the slide number
        filtered_animations = [
            anim for anim in args
            if not (isinstance(anim, FadeOut) and getattr(anim, "mobject", None) is self.slide_number_label)
        ]

        if self.pending_slide_tick:
            self.pending_slide_tick = False
            previous_label = self.slide_number_label
            self.slide_counter += 1
            updated_label = self._make_slide_label(self.slide_counter)
            self.slide_number_label = updated_label

            # Odometer roll-up animation
            super().play(
                FadeOut(previous_label, shift=UP * 0.25),
                FadeIn(updated_label, shift=UP * 0.25),
                run_time=0.35,
            )

        if filtered_animations:
            return super().play(*filtered_animations, **kwargs)

    def use_part(self, part):
        manim_classes = set(Scene.__mro__) | set(MovingCameraScene.__mro__) | set(Slide.__mro__)
        for cls in reversed(part.__mro__):
            if cls in manim_classes or cls is object:
                continue
            for name, fn in vars(cls).items():
                if isinstance(fn, types.FunctionType) and name != "construct":
                    setattr(self, name, types.MethodType(fn, self))

    def construct(self):
        self.slide_counter = 1
        self.pending_slide_tick = False
        self.slide_number_label = self._make_slide_label(self.slide_counter)
        self.add(self.slide_number_label)

        for i, part in enumerate(PARTS):
            if i > 0:
                self.clear()
                self.add(self.slide_number_label)
            for name in STATE:
                if name in self.__dict__:
                    delattr(self, name)
            self.use_part(part)
            part.construct(self)