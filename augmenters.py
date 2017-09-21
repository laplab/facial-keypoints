from abc import abstractmethod

import numpy as np
from skimage.transform import rotate


class Augmenter(object):
    @abstractmethod
    def augment(self, img, coords):
        pass


class NoChangesAugmenter(Augmenter):
    def augment(self, img, coords):
        yield img, coords


class FlipAugmenter(Augmenter):
    def augment(self, img, coords):
        new_img = np.fliplr(img)
        new_coords = coords.copy()

        height, width = img.shape
        new_coords[::2] = width - new_coords[::2]

        permutation = np.array([
            (3, 0), (2, 1), (1, 2), (0, 3), (9, 4), (8, 5), (7, 6),
            (6, 7), (5, 8), (4, 9), (10, 10), (13, 11), (12, 12), (11, 13)
        ])
        flipped = np.zeros_like(new_coords)

        pos_from = 2 * permutation[:, 0]
        pos_to = 2 * permutation[:, 1]
        flipped[pos_to] = new_coords[pos_from]
        flipped[pos_to + 1] = new_coords[pos_from + 1]

        yield new_img, flipped


class RotateAugmenter(Augmenter):
    def __init__(self, radius, repeat):
        self.radius = radius
        self.repeat = repeat

    def augment(self, img, coords):
        for i in range(self.repeat):
            angle = np.random.uniform(-self.radius, self.radius)
            theta = np.pi * angle / 180
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

            height, width = img.shape
            rotation_center = np.array([height / 2, width / 2])
            new_img = rotate(img, -angle, center=rotation_center)
            new_coords = coords.copy()

            for i in range(0, len(coords), 2):
                pos = np.array([coords[i] - rotation_center[1],
                                coords[i + 1] - rotation_center[0]])
                new_pos = rotation_matrix.dot(pos)
                new_coords[i] = new_pos[0] + rotation_center[1]
                new_coords[i + 1] = new_pos[1] + rotation_center[0]

            yield new_img, new_coords
