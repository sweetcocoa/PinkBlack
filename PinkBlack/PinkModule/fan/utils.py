import os
import sys
import time
import torch
import math
import numpy as np
import torch
import cv2


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image


# point: Nx2
# center : Nx2
# scale : N
def transform(points, centers, scales, resolution, invert=False):
    # [N,3]

    out = []
    for point, center, scale in zip(points, centers, scales):
        _pt = torch.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = torch.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if invert:
            t = torch.inverse(t)

        new_point = (torch.matmul(t, _pt))[0:2]
        out.append(new_point.int())

    return torch.stack(out, dim=0)


# input: NxHxWx3
# center : Nx2
# scale : N
def crop(images:np.ndarray, center, scale, resolution=256.0):
    # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    n = images.shape[0]

    ul = transform(torch.tensor([1, 1]).repeat(n,1), center, scale, resolution, True)
    br = transform(torch.tensor([resolution, resolution]).repeat(n,1), center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)


    ht = images.shape[1]
    wd = images.shape[2]
    newX = np.stack([np.maximum(1, -ul[:,0] + 1), np.minimum(br[:,0], wd) - ul[:,0]], axis=1).astype(np.int32)
    newY = np.stack([np.maximum(1, -ul[:,1] + 1), np.minimum(br[:,1], ht) - ul[:,1]], axis=1).astype(np.int32)
    oldX = np.stack([np.maximum(1, ul[:,0] + 1), np.minimum(br[:,0], wd)], axis=1).astype(np.int32)
    oldY = np.stack([np.maximum(1, ul[:,1] + 1), np.minimum(br[:,1], ht)], axis=1).astype(np.int32)

    out = []
    for i in range(n):
        newImg = np.zeros([br[i,1] - ul[i,1], br[i,0] - ul[i,0], 3], dtype=np.uint8)
        newImg[newY[i, 0] - 1:newY[i, 1], newX[i, 0] - 1:newX[i, 1], :] = \
            images[i, oldY[i, 0] - 1:oldY[i, 1], oldX[i, 0] - 1:oldX[i, 1], :]
        out.append(
            cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                            interpolation=cv2.INTER_LINEAR)
        )
    out = np.stack(out, axis=0)
    return out


# input: Nx68x64x64
# center : Nx2
# scale : N
# ret1 : Nx68x2 (coord in face img)
# ret2 : Nx68x2 (coord in orig img)
def get_preds_fromhm(hm, center=None, scale=None):
    # [N, 68, 64*64] -> [N, 68]
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1) # x
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1) # y
    # preds : [N, 68, 2]

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :] # [64*64]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.tensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())
    # for i in range(hm.size(0)):  # batch index
    for j in range(hm.size(1)):  # 68
        preds_orig[:, j] = transform(preds[:, j], center, scale, 64, True)

    return preds, preds_orig


# input : landmark
def tr(fov, landmark_2d, sz, y_shrink):
    landmark_2d = np.ascontiguousarray(landmark_2d[:, 0:2])
    feature_point, t, r = get_tr(landmark_2d,
                                 fov,
                                 width=sz[0],
                                 height=sz[1],
                                 use_subset=False,
                                 y_shrink=y_shrink)
    render_t, render_r = convert_for_render(fov, sz[0], sz[1],
                                            feature_point, t, r)
    return render_t, render_r


def get_tr(landmark_2d, fov, width, height, use_subset, y_shrink):
    mean_3d = get_mean_3d(y_shrink)

    # camera parameter
    fx, fy, cx, cy = fov_to_camera_parameter(fov, width, height)
    camera_metrix = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape((3, 3))

    vec_trans = np.zeros((1, 3))
    vec_trans = np.array([[0.0, 0.0, 500]])
    vec_rot = np.zeros((1, 3))
    vec_rot = np.array([[0.0, 0.0, 0.0]])
    dist_coeffs = np.array([])


    index = [27] + list(range(68))
    mean_3d_v1 = np.ascontiguousarray(mean_3d[index, :])
    landmark_2d_v1 = np.ascontiguousarray(landmark_2d[index, :])
    #print(index)
    retval, rvec, tvec = cv2.solvePnP(mean_3d_v1, landmark_2d_v1, camera_metrix,
                                      dist_coeffs, vec_rot, vec_trans, True)
    rvec_v1, tvec_v1 = np.copy(rvec), np.copy(tvec)
    rvec = rvec_v1[0]
    tvec = tvec[0]
    rvec = AxisAngle2Euler(rvec)

    face_feature_point = mean_3d[27, :]
    return face_feature_point, tvec, rvec


def convert_for_render(fov, width, height, face_feature_point, tvec, rvec):
    #fx, fy, cx, cy = fov_to_camera_parameter(fov, width, height)

    gt = obj2world(face_feature_point, rvec, tvec)
    # print('fov:', math.atan(cy / fy)*2.0 * 180.0 / math.pi)
    return [gt[0] / 10.0, gt[1]*-1.0 / 10.0, gt[2]*-1.0 / 10.0 + 20.0], [rvec[0], rvec[1]*-1.0, rvec[2]*-1.0]


def camera_parameter(fx, fy, cx, cy, cols, rows):
    cx_undefined = False
    fx_undefined = False
    if cx == 0 or cy == 0:
        cx_undefined = True
    if fx == 0 or fy == 0:
        fx_undefined = True

    if (cx_undefined):
        cx = cols / 2.0
        cy = rows / 2.0

    # Use a rough guess-timate of focal length
    if (fx_undefined):
        fx = 500 * (cols / 640.0) * 1.2
        fy = 500 * (rows / 480.0) * 1.2
        fx = (fx + fy) / 2.0
        fy = fx

    return fx, fy, cx, cy


def Euler2RotationMatrix(eulerAngles):
    rotation_matrix = np.zeros((3, 3), dtype=np.float)

    s1 = math.sin(eulerAngles[0])
    s2 = math.sin(eulerAngles[1])
    s3 = math.sin(eulerAngles[2])

    c1 = math.cos(eulerAngles[0])
    c2 = math.cos(eulerAngles[1])
    c3 = math.cos(eulerAngles[2])

    rotation_matrix[0, 0] = c2 * c3
    rotation_matrix[0, 1] = -c2 * s3
    rotation_matrix[0, 2] = s2
    rotation_matrix[1, 0] = c1 * s3 + c3 * s1 * s2
    rotation_matrix[1, 1] = c1 * c3 - s1 * s2 * s3
    rotation_matrix[1, 2] = -c2 * s1
    rotation_matrix[2, 0] = s1 * s3 - c1 * c3 * s2
    rotation_matrix[2, 1] = c3 * s1 + c1 * s2 * s3
    rotation_matrix[2, 2] = c1 * c2
    #print('rotation_matrix', rotation_matrix)

    return rotation_matrix


def obj2world(obj_point, rotation, translation):
    ret = np.array(obj_point, dtype=np.float)
    #print('obj_point', ret)

    rot = Euler2RotationMatrix(rotation)
    #print('rot', rot)

    ret = np.matmul(rot, ret)
    #print('rot * ret', ret)
    ret = np.transpose(ret)
    # print(ret)

    ret[:, 0] = ret[:, 0] + translation[0]
    ret[:, 1] = ret[:, 1] + translation[1]
    ret[:, 2] = ret[:, 2] + translation[2]
    # print(ret)

    return ret[0]

def obj2world_list(obj_points, rotation, translation):
    ret = np.array(obj_points, dtype=np.float)
    #print('obj_point', ret)

    rot = Euler2RotationMatrix(rotation)
    #print('rot', rot)

    ret = np.transpose(ret)
    ret = np.matmul(rot, ret)
    #print('rot * ret', ret)
    ret = np.transpose(ret)
    # print(ret)

    ret[:, 0] = ret[:, 0] + translation[0]
    ret[:, 1] = ret[:, 1] + translation[1]
    ret[:, 2] = ret[:, 2] + translation[2]
    # print(ret)

    return ret


def RotationMatrix2Euler(rotation_matrix):
    #print('--', rotation_matrix)
    q0 = math.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    q1 = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4.0*q0)
    q2 = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4.0*q0)
    q3 = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4.0*q0)

    yaw = math.asin(2.0 * (q0*q2 + q1*q3))
    pitch = math.atan2(2.0 * (q0*q1-q2*q3), q0*q0-q1*q1-q2*q2+q3*q3)
    roll = math.atan2(2.0 * (q0*q3-q1*q2), q0*q0+q1*q1-q2*q2-q3*q3)
    return np.array([pitch, yaw, roll])


def AxisAngle2Euler(axis_angle):
    rotation_matrix = cv2.Rodrigues(axis_angle)
    return RotationMatrix2Euler(rotation_matrix[0])


def degree_to_radian(degree):
    return degree * math.pi / 180


def fov_to_forcal_lenth(fov, width, height):
    if 0 == fov:
        return 0

    cy = height / 2

    return cy / math.tan(degree_to_radian(fov / 2))


def fov_to_camera_parameter(fov_y, width, height):
    fy = fov_to_forcal_lenth(fov_y, width, height)
    return camera_parameter(fy, fy, width / 2, height / 2, width, height)


