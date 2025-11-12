#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import cv2
from math import cos, sin, degrees

from .base import CameraBase

class Camera(CameraBase):
    """
    Describes a monocular camera. It stores it parameters and performs 
    projection operations between world frame, camera frame and image place.
    Notice that no distorsion is now considered, neither in storing the 
    parameters nor in projecting points.
    """

    def __init__(self, cameraData):
        """
        Initializes the camera object with its intrinsic and extrinsic parameters.
        :param cameraData: object structured as follows
        camera_data = {
            'intrinsic': {
                'fx': fx
                'fy': fy,
                'cx': cx,
                'cy': cy
            },
            'extrinsic': {
                'x': x,
                'y': y,
                'z': z,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll
            }
        }
        """

        # Extract params
        self.cameraData = cameraData
        self.yaw = cameraData['extrinsic']['yaw']
        self.pitch = cameraData['extrinsic']['pitch']
        self.roll = cameraData['extrinsic']['roll']
        self.x = cameraData['extrinsic']['x']
        self.y = cameraData['extrinsic']['y']
        self.z = cameraData['extrinsic']['z']
        self.fx = cameraData['intrinsic']['fx']
        self.fy = cameraData['intrinsic']['fy']
        self.cx = cameraData['intrinsic']['cx']
        self.cy = cameraData['intrinsic']['cy']
        self.computeCameraMatrix()

    def copy(self):
        return Camera(self.cameraData)

    __copy__ = copy  # Now works with copy.copy too

    def reinitialize(
        self,
        intrinsics_fx=None,
        intrinsics_fy=None,
        intrinsics_cx=None,
        intrinsics_cy=None,
        extrinsic_x=None,
        extrinsic_y=None,
        extrinsic_z=None,
        extrinsic_yaw=None,
        extrinsic_pitch=None,
        extrinsic_roll=None
        ):

        self.yaw = self.cameraData['extrinsic']['yaw'] = extrinsic_yaw if extrinsic_yaw is not None else self.yaw
        self.pitch = self.cameraData['extrinsic']['pitch'] = extrinsic_pitch if extrinsic_pitch is not None else self.pitch
        self.roll = self.cameraData['extrinsic']['roll'] = extrinsic_roll if extrinsic_roll is not None else self.roll
        self.x = self.cameraData['extrinsic']['x'] = extrinsic_x if extrinsic_x is not None else self.x
        self.y = self.cameraData['extrinsic']['y'] = extrinsic_y if extrinsic_y is not None else self.y
        self.z = self.cameraData['extrinsic']['z'] = extrinsic_z if extrinsic_z is not None else self.z
        self.fx = self.cameraData['intrinsic']['fx'] = intrinsics_fx if intrinsics_fx is not None else self.fx
        self.fy = self.cameraData['intrinsic']['fy'] = intrinsics_fy if intrinsics_fy is not None else self.fy
        self.cx = self.cameraData['intrinsic']['cx'] = intrinsics_cx if intrinsics_cx is not None else self.cx
        self.cy = self.cameraData['intrinsic']['cy'] = intrinsics_cy if intrinsics_cy is not None else self.cy
        self.computeCameraMatrix()

    def computeCameraMatrix(self):
        # cos/sin for yaw, pitch, roll
        c_y = cos(self.yaw)
        s_y = sin(self.yaw)
        c_p = cos(self.pitch)
        s_p = sin(self.pitch)
        c_r = cos(self.roll)
        s_r = sin(self.roll)
        # from cityscape guide
        R_c_to_v = np.float32([[c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
                               [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
                               [- s_p, c_p * s_r, c_p * c_r]])
        t_c_to_v = np.float32([self.x, self.y, self.z])[np.newaxis].T

        # compute intrisic matrix
        self.K = np.float32([[self.fx, 0, self.cx],
                             [0, self.fy, self.cy],
                             [0,        0,          1]])
        M_camera_to_video = np.float32([[0, -1, 0],
                                        [0, 0, -1],
                                        [1, 0, 0]])
        C = self.K.dot(M_camera_to_video) # @
        R = R_c_to_v.T
        rpy, _ = cv2.Rodrigues(R_c_to_v)
        self.t = - R.dot(t_c_to_v)
        self.Rt = np.hstack((R, self.t))

        # compute projection matrix
        self.P = C.dot(self.Rt)

    def __str__(self) -> str:
        res = f"fx: {self.fx}, fy: {self.fy}\ncx: {self.cx}, cy: {self.cy}\n"
        res += f"x: {self.x}, y: {self.y}, z: {self.z}\n"
        res += f"roll (deg): {degrees(self.roll)}, pitch (deg): {degrees(self.pitch)}, yaw (deg): {degrees(self.yaw)}"
        return res
    