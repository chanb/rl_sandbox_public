import numpy as np
import pygame

from rl_sandbox.constants import RGB_ARRAY, WINDOW


class Renderer:
    def __init__(self, env, render_h=256, render_w=256):
        self._env = env
        self.render_h = render_h
        self.render_w = render_w
        self.pitch = self.render_w * -3

    def render(self, **kwargs):
        if not hasattr(self, WINDOW):
            pygame.init()
            self.window = pygame.display.set_mode((self.render_w, self.render_h), pygame.RESIZABLE)
            pygame.display.set_caption('Gym')
        pixels = np.transpose(self._env.render(render_mode), (1, 0, 2))
        
        cur_size = self.window.get_size()
        render_img_rect = pygame.Rect((0, 0), (cur_size[0], cur_size[1]))
        disp_img = pygame.transform.scale(pygame.surfarray.make_surface(pixels),
                                            render_img_rect.size).convert()
        self.window.blit(disp_img, (0, 0))
        pygame.display.flip()

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self, **kwargs):
        raise NotImplementedError

    def seed(self, seed):
        pass


class DMControlRenderer(Renderer):
    def __init__(self, env, render_h=256, render_w=256):
        super().__init__(env, render_h, render_w)

    def render(self, **kwargs):
        if not hasattr(self, WINDOW):
            pygame.init()
            self.window = pygame.display.set_mode((self.render_w, self.render_h), pygame.RESIZABLE)
            pygame.display.set_caption('DM Control')
        pixels = np.transpose(self._env.physics.render(camera_id=0, height=self.render_h, width=self.render_w), (1, 0, 2))
        
        cur_size = self.window.get_size()
        render_img_rect = pygame.Rect((0, 0), (cur_size[0], cur_size[1]))
        disp_img = pygame.transform.scale(pygame.surfarray.make_surface(pixels),
                                            render_img_rect.size).convert()
        self.window.blit(disp_img, (0, 0))
        pygame.display.flip()


class GymRenderer(Renderer):
    def __init__(self, env, render_mode=RGB_ARRAY, render_h=256, render_w=256):
        super().__init__(env, render_h, render_w)
        self.render_mode = render_mode

    def render(self, **kwargs):
        if not hasattr(self, WINDOW):
            pygame.init()
            self.window = pygame.display.set_mode((self.render_w, self.render_h), pygame.RESIZABLE)
            pygame.display.set_caption('Gym')
        pixels = np.transpose(self._env.render(self.render_mode), (1, 0, 2))
        
        cur_size = self.window.get_size()
        render_img_rect = pygame.Rect((0, 0), (cur_size[0], cur_size[1]))
        disp_img = pygame.transform.scale(pygame.surfarray.make_surface(pixels),
                                            render_img_rect.size).convert()
        self.window.blit(disp_img, (0, 0))
        pygame.display.flip()

    def seed(self, seed):
        self._env.seed(seed)
