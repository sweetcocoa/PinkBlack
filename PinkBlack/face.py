from .PinkModule.sfd import S3FD
from .PinkModule.fan import FAN3D
import numpy as np
import cv2
from skimage.transform import SimilarityTransform


def align_face(img, landmark, destination_size=(96, 112)):
    # TODO :: 원하는 사이즈의 이미지로 align
    """
    얼굴 랜드마크 좌표 (눈, 코, 입)을 중심으로 이미지를 정렬함.

    :param img:  np array image (H, W, C)
    :param landmark:  np array coordinates 5 x 2. 눈1, 눈2, 코, 입 양 끝.

    if len(landmark) == 68:
        It will be converted to 5x2

    68개 랜드마크 기준으로
    [36:42 평균,
    42:48 평균,
    30,
    48,
    54]

    :return:

    """

    if len(landmark) == 68:
        landmark = np.array(landmark)
        landmark = np.array([
            landmark[36:42, :2].mean(axis=0),
            landmark[42:48, :2].mean(axis=0),
            landmark[30, :2],
            landmark[48, :2],
            landmark[54, :2]
        ])

    ref_size = np.array([96, 112])

    # dst = np.array([
    #     [30.2946, 51.6963],
    #     [65.5318, 51.5014],
    #     [48.0252, 71.7366],
    #     [33.5493, 92.3655],
    #     [62.7299, 92.2041]], dtype=np.float32)

    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    dst_size = np.array(destination_size)

    p = dst_size / ref_size
    dst = dst * p

    src = landmark
    tform = SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    out = cv2.warpAffine(img, M, destination_size, borderValue=0.0)
    return out


def get_landmarks(img, sfd, fan):
    """
    :param img: cv2 bgr image
    :param sfd:
    :param fan:
    :return: boxes, ldmk
    """
    boxes = sfd.get_face_boxes(img)
    if len(boxes) == 0:
        return [], None

    ldmk = fan.get_landmarks(img, boxes[0])
    return boxes, ldmk