from PIL import Image
import cv2
import numpy as np
from skimage.transform import SimilarityTransform


class RatioScale:
    """
    이미지를 한쪽 길이가 max_size 인 이미지로 resize한 후 그리고 정사각형 모양으로 Padding
    Resizes an image with keeping aspect ratio, and adds black-padding so that the shape of the image is square.
    """
    def __init__(self, max_size, method=Image.ANTIALIAS):
        self.max_size = max_size
        self.method = method

    def __call__(self, image):
        image.thumbnail(self.max_size, self.method)
        offset = (int((self.max_size[0] - image.size[0]) / 2), int((self.max_size[1] - image.size[1]) / 2))
        back = Image.new("RGB", self.max_size, "black")
        back.paste(image, offset)
        return back


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


def align_face(img, landmark, origin=(0, 0)):
    """
    얼굴 랜드마크 좌표 (눈, 코, 입)을 중심으로 이미지를 정렬함.

    일단은 정사각형만 지원.
    It only supports squares, not rectangles.
    :param img:  np array image (H, W, C)
    :param landmark:  np array coordinates 5 x 2. 눈1, 눈2, 코, 입 양 끝.
    68개 랜드마크 기준으로
    [36:42 평균,
    42:48 평균,
    30,
    48,
    54]
    :param origin:
    :return:
    """
    dst_size = 112
    new_dst_size = 224
    src_img_size = max(img.size)
    dst = np.array([
        [30.2946 + 8, 51.6963],
        [65.5318 + 8, 51.5014],
        [48.0252 + 8, 71.7366],
        [33.5493 + 8, 92.3655],
        [62.7299 + 8, 92.2041]], dtype=np.float32 )  * new_dst_size / dst_size
    p = src_img_size / new_dst_size
    dst = dst * p

    src = landmark - np.array(origin)
    tform = SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    out = cv2.warpAffine(img, M, (src_img_size, src_img_size), borderValue=0.0)
    return out


