"""
use dataset transforms following SparseRCNN
"""
import random

import numpy as np
import cv2
import torch
from . import cv2_transform as cv2_transform


def resize_shortest_edge(images, short_edge_length, max_size, boxes=None, sample_style="range"):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (list): list of images to perform scale jitter. Dimension is
            `height` x `width` x `channel`.
        short_edge_length (list[int]): If ``sample_style=="range"``,
            a [min, max] interval from which to sample the shortest edge length.
            If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
        max_size (int): maximum allowed longest edge length.
        boxes (list or None): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
        sample_style (str): either "range" or "choice".

    Returns:
        (list): the list of scaled images with dimension of
            `new height` x `new width` x `channel`.
        (ndarray or None): the scaled boxes with dimension of
            `num boxes` x 4.
    """
    assert sample_style in ["range", "choice"], sample_style
    is_range = sample_style == "range"
    if isinstance(short_edge_length, int):
        short_edge_length = (short_edge_length, short_edge_length)
    if is_range:
        assert len(short_edge_length) == 2, (
            "short_edge_length must be two values using 'range' sample style."
            f" Got {short_edge_length}!"
        )
    height = images[0].shape[0]
    width = images[0].shape[1]

    if is_range:
        size = np.random.randint(short_edge_length[0], short_edge_length[1] + 1)
    else:
        size = np.random.choice(short_edge_length)

    if (width <= height <= max_size and width == size) or (
            height <= width <= max_size and height == size
    ):
        return images, boxes

    scale_ratio = size * 1.0 / min(height, width)
    box_scale = scale_ratio
    if height < width:
        new_height, new_width = size, width * scale_ratio
    else:
        new_height, new_width = height * scale_ratio, size

    if max(new_height, new_width) > max_size:
        scale_ratio = max_size * 1.0 / max(new_height, new_width)
        box_scale *= scale_ratio
        new_height = new_height * scale_ratio
        new_width = new_width * scale_ratio

    new_height = int(new_height + 0.5)
    new_width = int(new_width + 0.5)

    if boxes is not None:
        boxes = [proposal * box_scale for proposal in boxes]

    return (
        [
            cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            for image in images
        ],
        boxes,
    )


class SparseDatasetMapper:
    def __init__(self, cfg, is_train=True):

        self.is_train = is_train

        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR

        if is_train:
            self._min_size = cfg.AVA.MIN_SIZE_TRAIN
            self._max_size = cfg.AVA.MAX_SIZE_TRAIN
            self._sample_style = cfg.AVA.MIN_SIZE_TRAIN_SAMPLING
            self._crop_enable = cfg.AVA.CROP_ENABLED
            self._crop_size = cfg.AVA.CROP_SIZE
            self._random_horizontal_flip = cfg.DATA.RANDOM_FLIP

            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.AVA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.AVA.TRAIN_PCA_EIGVEC

        else:
            self._min_size = cfg.AVA.MIN_SIZE_TEST
            self._max_size = cfg.AVA.MAX_SIZE_TEST
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP
            self._sample_style = "choice"

    def __call__(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self.is_train:  # "train"
            if self._random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
            imgs, boxes = resize_shortest_edge(
                imgs, self._min_size, self._max_size, boxes, self._sample_style,
            )
            if self._crop_enable:
                raise NotImplementedError("Not Implement logic")

        else:
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs, boxes = resize_shortest_edge(
                imgs, self._min_size, self._max_size, boxes, self._sample_style
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self.is_train and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes  # boxes:  (x1,y1,x2,y2)

