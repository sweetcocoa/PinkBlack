import numpy as np
import math
import cv2


def get_mean_3d(y_shrink):
  landmark_3d =     [ -73.393523 ,- 72.775014 ,- 70.533638 ,- 66.850058 ,- 59.790187
      ,- 48.368973 ,- 34.121101 ,- 17.875411 ,0.098749 ,17.477031 ,32.648966
      ,46.372358 ,57.343480 ,64.388482 ,68.212038 ,70.486405 ,71.375822 ,- 61.119406
      ,- 51.287588 ,- 37.804800 ,- 24.022754 ,- 11.635713 ,12.056636 ,25.106256
      ,38.338588 ,51.191007 ,60.053851 ,0.653940 ,0.804809 ,0.992204 ,1.226783 ,
      - 14.772472 ,- 7.180239 ,0.555920 ,8.272499 ,15.214351 ,- 46.047290 ,- 37.674688
      ,- 27.883856 ,- 19.648268 ,- 28.272965 ,- 38.082418 ,19.265868 ,27.894191
      ,37.437529 ,45.170805 ,38.196454 ,28.764989 ,- 28.916267 ,- 17.533194 ,
      - 6.684590 ,0.381001 ,8.375443 ,18.876618 ,28.794412 ,19.057574 ,8.956375
      ,0.381549 ,- 7.428895 ,- 18.160634 ,- 24.377490 ,- 6.897633 ,0.340663 ,8.444722
      ,24.474473 ,8.449166 ,0.205322 ,- 7.198266 ,- 29.801432 ,- 10.949766 ,7.929818
      ,26.074280 ,42.564390 ,56.481080 ,67.246992 ,75.056892 ,77.061286 ,74.758448
      ,66.929021 ,56.311389 ,42.419126 ,25.455880 ,6.990805 ,- 11.666193 ,- 30.365191
      ,- 49.361602 ,- 58.769795 ,- 61.996155 ,- 61.033399 ,- 56.686759 ,- 57.391033
      ,- 61.902186 ,- 62.777713 ,- 59.302347 ,- 50.190255 ,- 42.193790 ,- 30.993721
      ,- 19.944596 ,- 8.414541 ,2.598255 ,4.751589 ,6.562900 ,4.661005 ,2.643046 ,
      - 37.471411 ,- 42.730510 ,- 42.711517 ,- 36.754742 ,- 35.134493 ,- 34.919043 ,
      - 37.032306 ,- 43.342445 ,- 43.110822 ,- 38.086515 ,- 35.532024 ,- 35.484289
      ,28.612716 ,22.172187 ,19.029051 ,20.721118 ,19.035460 ,22.394109 ,28.079924
      ,36.298248 ,39.634575 ,40.395647 ,39.836405 ,36.677899 ,28.677771 ,25.475976
      ,26.014269 ,25.326198 ,28.323008 ,30.596216 ,31.408738 ,30.844876 ,47.667532
      ,45.909403 ,44.842580 ,43.141114 ,38.635298 ,30.750622 ,18.456453 ,3.609035 ,
      - 0.881698 ,5.181201 ,19.176563 ,30.770570 ,37.628629 ,40.886309 ,42.281449
      ,44.142567 ,47.140426 ,14.254422 ,7.268147 ,0.442051 ,- 6.606501 ,- 11.967398
      ,- 12.051204 ,- 7.315098 ,- 1.022953 ,5.349435 ,11.615746 ,- 13.380835 ,
      - 21.150853 ,- 29.284036 ,- 36.948060 ,- 20.132003 ,- 23.536684 ,- 25.944448 ,
      - 23.695741 ,- 20.858157 ,7.037989 ,3.021217 ,1.353629 ,- 0.111088 ,- 0.147273
      ,1.476612 ,- 0.665746 ,0.247660 ,1.696435 ,4.894163 ,0.282961 ,- 1.172675 ,
      - 2.240310 ,- 15.934335 ,- 22.611355 ,- 23.748437 ,- 22.721995 ,- 15.610679 ,
      - 3.217393 ,- 14.987997 ,- 22.554245 ,- 23.591626 ,- 22.406106 ,- 15.121907 ,
      - 4.785684 ,- 20.893742 ,- 22.220479 ,- 21.025520 ,- 5.712776 ,- 20.671489 ,
      - 21.903670 ,- 20.328022]

  landmark_3d = np.array(landmark_3d, dtype=np.float)
  landmark_3d = np.transpose(landmark_3d.reshape(3,68))
  landmark_3d = landmark_3d.reshape(68, 3, 1)

  face_min_y = landmark_3d[:,1,0].min()
  landmark_3d[:,1,0] = face_min_y + (landmark_3d[:,1,0] - face_min_y) * y_shrink

  return landmark_3d

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


