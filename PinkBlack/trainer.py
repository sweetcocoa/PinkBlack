import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm


def prepare(mode="binary",
            data="image",
            need_auroc=False,
            transforms=None,
            num_labels=None,
            ):
    """
    mode = {"classification": classification
                if classification, num_label is needed
            "regression" : regression
            "multilabel": multilabel binary classification
            }

    data = {"image": 이미지
            "text" : 텍스트 }

    need_auroc = binary classification인 경우 (구현X)

    :param mode:
    :param data:
    :param need_auroc:
    :param transforms:
    :return:
    """
    pass


def train():
    pass


def valid():
    pass


