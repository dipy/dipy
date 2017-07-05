import numpy as np
from dipy.viz import window, actor


class TimeLineManager(object):
    def __init__(self, show_m, actors,
                 video_fname,
                 size=(1280, 720),
                 default_font_size=24,
                 large_font_size=42,
                 small_font_size=20, reset_clipping=True):
        self.states = []
        self.second = 0
        self.actors = actors
        self.subs = []
        self.frame = 0
        self.repeat_frame = 10
        self.fps = 25
        self.show_m = show_m
        self.events = []
        self.size = size
        self.default_font_size = default_font_size
        self.large_font_size = large_font_size
        self.small_font_size = small_font_size
        self.reset_clipping = reset_clipping
        self.initialize_subs()

        for act in self.actors:
            act.VisibilityOff()

        self.movie_writer = window.MovieWriter(video_fname, show_m.window)
        self.movie_writer.start()

    def initialize_subs(self):

        title = actor.text_overlay(
            ' ',
            position=(self.size[0] / 2, self.size[1] / 2),
            color=(0, 0, 0),
            font_size=self.large_font_size, bold=True, justification='center')

        top_right = actor.text_overlay(
            ' ',
            position=(self.size[0]/2 + 250, 650),
            color=(0, 0, 0),
            font_size=self.small_font_size,
            font_family='Times', bold=True)

        top_left = actor.text_overlay(
            ' ',
            position=(10, 650),
            color=(0, 0, 0),
            font_size=self.small_font_size,
            font_family='Times', bold=True)

        sub = actor.text_overlay(
            ' ',
            position=(self.size[0]/2, 25),
            color=(0, 0, 0),
            font_size=self.default_font_size + 10,
            justification='center',
            bold=True)

        self.top_left = top_left
        self.top_right = top_right
        self.title = title
        self.sub = sub

        self.show_m.ren.add(self.top_left)
        self.show_m.ren.add(self.top_right)
        self.show_m.ren.add(self.title)
        self.show_m.ren.add(self.sub)

        self.top_left.VisibilityOff()
        self.top_right.VisibilityOff()
        self.title.VisibilityOff()
        self.sub.VisibilityOff()

    def add_state(self, second, actors, states):

        self.states.append((second, actors, states))

    def add_sub(self, second, positions, messages):

        self.subs.append((second, positions, messages))

    def add_event(self, second, duration, functions, args):

        self.events.append((second, duration, functions, args))

    def execute(self):
        for state in self.states:
            if self.second == state[0]:
                actors = state[1]
                states = state[2]
                for act, state in zip(actors, states):
                    if state == 'on':
                        act.VisibilityOn()
                    if state == 'off':
                        act.VisibilityOff()

        for sub in self.subs:
            if self.second == sub[0]:
                positions = sub[1]
                messages = sub[2]
                for position, message in zip(positions, messages):
                    if position == 'top_left':
                        self.top_left.set_message(message)
                        self.top_left.Modified()
                        self.top_left.VisibilityOn()
                    if position == 'top_right':
                        self.top_right.set_message(message)
                        self.top_right.Modified()
                        self.top_right.VisibilityOn()
                    if position == 'title':
                        self.title.set_message(message)
                        self.title.Modified()
                        self.title.VisibilityOn()
                    if position == 'sub':
                        self.sub.set_message(message)
                        self.sub.Modified()
                        self.sub.VisibilityOn()

        for event in self.events:
            second = event[0]
            duration = event[1]
            functions = event[2]
            args = event[3]

            if self.second >= second and self.second < second + duration:
                for func, args in zip(functions, args):
                    func(*args)

        if self.reset_clipping:
            self.show_m.ren.reset_clipping_range()
        self.show_m.render()
        self.movie_writer.write()

        self.frame += 1
        self.second = self.frame / np.float(self.fps)
        print('Second %0.2f' % (self.second,))
