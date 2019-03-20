from .network import FAN
from .utils import *
from .tr_helper import tr
import numpy as np
import torch


class FAN3D(object):
    def __init__(self, fan_weight):
        self.net = FAN(num_modules=4)
        self.net.load_state_dict(torch.load(fan_weight, map_location="cpu"))
        self.net.cuda()
        self.net.eval()

    def get_landmarks_batch(self, imgs: np.ndarray, boxes: np.ndarray):
        # imgs: [N, H, W, 3]
        # boxes: cropped for face region
        assert (len(imgs.shape) == 4)
        assert (imgs.shape[3] == 3)

        center = np.stack(
            [boxes[:, 2] - (boxes[:, 2] - boxes[:, 0]) / 2.0,
             boxes[:, 3] - (boxes[:, 3] - boxes[:, 1]) / 2.0], axis=1)
        center[:, 1] -= (boxes[:, 3] - boxes[:, 1]) * 0.12
        scale = (boxes[:, 2] - boxes[:, 0] + boxes[:, 3] - boxes[:, 1]) / 195.0

        # batch
        croped = crop(imgs, center, scale)

        croped = torch.from_numpy(croped).cuda()

        with torch.no_grad():
            inp = croped.permute(0, 3, 1, 2).float() / 255.0
            heatmaps = self.net(inp)
        heatmap = heatmaps[-1].cpu()

        _, points = get_preds_fromhm(heatmap, center, scale)
        return points.numpy()  # [N, 2]

    def get_landmarks(self, img: np.ndarray, boxes: np.ndarray):
        # imgs: [H, W, 3]
        # boxes: cropped for face region
        boxes = np.array
        assert (len(img.shape) == 3)
        assert (img.shape[2] == 3)

        imgs = np.expand_dims(img, axis=0)
        boxes = np.expand_dims(boxes, axis=0)

        center = np.stack(
            [boxes[:, 2] - (boxes[:, 2] - boxes[:, 0]) / 2.0,
             boxes[:, 3] - (boxes[:, 3] - boxes[:, 1]) / 2.0], axis=1)
        center[:, 1] -= (boxes[:, 3] - boxes[:, 1]) * 0.12
        scale = (boxes[:, 2] - boxes[:, 0] + boxes[:, 3] - boxes[:, 1]) / 195.0

        # batch
        croped = crop(imgs, center, scale)

        croped = torch.from_numpy(croped).cuda()

        with torch.no_grad():
            inp = croped.permute(0, 3, 1, 2).float() / 255.0
            heatmaps = self.net(inp)
        heatmap = heatmaps[-1].cpu()

        _, points = get_preds_fromhm(heatmap, center, scale)
        return points.numpy()  # [N, 2]

    @staticmethod
    def get_tr(fov, ldmk, sz):
        return tr(fov, ldmk, sz, y_shrink=1.)


