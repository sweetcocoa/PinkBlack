import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia


class RatioScale:
    """
    이미지를 한쪽 길이가 max_size 인 이미지로 resize한 후 그리고 정사각형 모양으로 Padding
    Resizes an image with keeping aspect ratio, and adds black-padding so that the shape of the image is square.
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.hw = iaa.Scale({'height': self.max_size, 'width':"keep-aspect-ratio"}, ia.ALL)
        self.wh = iaa.Scale({'width': self.max_size, 'height':"keep-aspect-ratio"}, ia.ALL)

    def __call__(self, x):
        assert (len(x.shape) == 2 or len(x.shape) == 3) and isinstance(x, np.ndarray)
        if x.shape[0] > x.shape[1]:
            return self.hw.augment_image(x)
        else:
            return self.wh.augment_image(x)


def get_crop_from_center(img, center, size):
    """
    :param img: PIL Image
    :param center: ( x, y ) - A point on image array
    :param size: (width, height)
    :return:
    """
    half_w = size[0] // 2
    half_h = size[1] // 2

    left = int(center[0])
    top = int(center[1])

    output_img = img.crop((left - half_w,
                           top - half_h,
                           left + half_w,
                           top + half_h,
                           ))
    return output_img
