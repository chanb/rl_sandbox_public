import cv2
import numpy as np
import torch

from rl_sandbox.transforms.general_transforms import Transform


""" RAD: https://arxiv.org/abs/2004.14990
"""
class RandomCrop(Transform):
    def __init__(self, img_dim, height=64, width=64):
        assert len(img_dim) == 3
        self.img_dim = img_dim
        self.height = height
        self.width = width

    def __call__(self, imgs):
        imgs = imgs.reshape(-1, *self.img_dim)
        n, c, h, w = imgs.shape
        w1 = torch.randint(0, w - self.width + 1, (n,))
        h1 = torch.randint(0, h - self.height + 1, (n,))
        cropped = torch.empty((n, c, self.height, self.width), dtype=imgs.dtype, device=imgs.device)
        for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
            cropped[i][:] = img[:, h11:h11 + self.height, w11:w11 + self.width]
        return cropped


class CenterCrop(Transform):
    def __init__(self, img_dim, height=64, width=64):
        assert len(img_dim) == 3
        self.img_dim = img_dim
        self.height = height
        self.width = width
        self.top = (img_dim[1] - height) // 2
        self.left = (img_dim[2] - width) // 2

    def __call__(self, imgs):
        imgs = imgs.reshape(-1, *self.img_dim)
        n, c, h, w = imgs.shape
        cropped = torch.empty((n, c, self.height, self.width), dtype=imgs.dtype, device=imgs.device)
        for i, img in enumerate(imgs):
            cropped[i][:] = img[:, self.top:self.top + self.height, self.left:self.left + self.width]
        return cropped


class NumPyCenterCrop(Transform):
    def __init__(self, img_dim, height=64, width=64):
        assert len(img_dim) == 3
        self.img_dim = img_dim
        self.height = height
        self.width = width
        self.top = (img_dim[1] - height) // 2
        self.left = (img_dim[2] - width) // 2

    def __call__(self, imgs):
        imgs = imgs.reshape(-1, *self.img_dim)
        n, c, h, w = imgs.shape
        cropped = np.empty((n, c, self.height, self.width), dtype=imgs.dtype)
        for i, img in enumerate(imgs):
            cropped[i][:] = img[:, self.top:self.top + self.height, self.left:self.left + self.width]
        return cropped


class NumpyGrayscale(Transform):
    def __call__(self, imgs):
        return cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)[..., None]
