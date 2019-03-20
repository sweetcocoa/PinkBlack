from . import network
import torch
import numpy as np
import cv2

class S3FD:
    def __init__(self, weight_file:str):
        self.net = network.S3FD()
        self.bgr_mean = torch.Tensor([104.0, 117.0, 123.0]).cuda()
        self.net.load_state_dict(torch.load(weight_file))
        self.net.cuda()
        self.net.eval()

    def run(self, imgs:np.ndarray):
        # imgs: bgr [N, H, W, 3]
        assert(len(imgs.shape)==4)
        assert(imgs.shape[3]==3)
        imgs = torch.from_numpy(imgs).float().cuda()
        imgs = imgs - self.bgr_mean
        imgs = imgs.permute(0, 3, 1, 2)

        with torch.no_grad():
            olist = self.net(imgs)

        out = []
        n = imgs.shape[0]
        for batch in range(n):
            bboxlist = []
            for i in range(len(olist)//2):
                ocls, oreg = olist[i*2].data.cpu(), olist[i*2+1].data.cpu()
                _, _, FH, FW = ocls.size()  # feature map size
                stride = 2**(i+2)    # 4,8,16,32,64,128

                for y in range(FH):
                    for x in range(FW):
                        score = ocls[batch, 1, y, x]
                        if score < 0.5:
                            continue
                        axc, ayc = stride/2+x*stride, stride/2+y*stride
                        loc = oreg[batch, :, y, x].contiguous().view(1, 4)

                        priors = torch.Tensor([[axc/1.0, ayc/1.0, stride*4/1.0, stride*4/1.0]])
                        variances = [0.1, 0.2]
                        box = decode(loc, priors, variances)
                        x1, y1, x2, y2 = box[0]*1.0
                        bboxlist.append([x1, y1, x2, y2, score])

            bboxlist = np.array(bboxlist)

            if len(bboxlist)>0:
                keep = nms(bboxlist, 0.3)
                bboxlist = bboxlist[keep, :]

            out.append(bboxlist)

        return out

    def get_face_boxes(self, img):
        # 얼굴 좌표와 confidence값 리턴
        # [rect1, rect2, rect3..]
        # 1개의 rect : [x1, y1, x2, y2, confidence(0~1)]
        # (주의) 원래 mongodb 로 쓰던 좌표계 (y1, x1, weight, height, confidence) 와 다름
        # 소수점 값
        width = img.shape[1]
        height = img.shape[0]
        resized = cv2.resize(img, (256, 256))
        imgs = np.expand_dims(resized, axis=0)
        boxes = self.run(imgs)[0]
        faces = []

        for i, box in enumerate(boxes):
            # y1 x1 y2 x2
            box[0] *= width / 256
            box[1] *= height / 256
            box[2] *= width / 256
            box[3] *= height / 256
            sfd_box = box
            faces.append(sfd_box)

        return faces


def nms(dets, thresh):
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w*h / (areas[i] + areas[order[1:]] - w*h)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep




def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


