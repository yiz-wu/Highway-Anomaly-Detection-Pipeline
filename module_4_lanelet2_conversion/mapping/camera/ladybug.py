import numpy as np
from math import cos, sin, pi, degrees, radians, atan2, sqrt

from .base import CameraBase

class LadybugCamera(CameraBase):
    """
    Describes a single camera in a ladybug system
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        bugPosition,
        cameraPostion,
        cameraRotation
        ):
        """
        Initializes the camera object with its intrinsic and extrinsic parameters.

        Args:
            fx: focal x * size x
            fy: focal y * size y
            cx: center u * size x
            cy: center v * size y
            bugPosition: imu/veicle to ladybug translation as [x,y,z] in meters.
                Should be [0,0, vehicle_heigh + imu2bug.z]
            cameraPosition: camera to ladybug translation as [x,y,z] in meters
            cameraRotation: camera to ladybug rotation as euler [z,y,x] in radians    
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.bugPosition = np.asarray(bugPosition, dtype=np.float32)
        self.cameraPostion = np.asarray(cameraPostion, dtype=np.float32)
        self.cameraRotation = np.asarray(cameraRotation, dtype=np.float32)
        self.staticCameraRotation = np.asarray(cameraRotation, dtype=np.float32)
        self.computeCameraMatrix()

    def computeCameraMatrix(self):
        
        # eulerZYX -> RPY swaps 0,2 indexes
        # http://web.mit.edu/2.05/www/Handout/HO2.PDF

        # bug coordinate system is the camera toward sky (?)
        Rbug2vehicle = self.make_rotation_matrix(yaw=0, pitch=radians(-90), roll=0)

        Rcam2bug = self.make_rotation_matrix(
            self.cameraRotation[2],
            self.cameraRotation[1],
            self.cameraRotation[0]
        )
        Rcam2vehicle = Rcam2bug @ Rbug2vehicle
        Rvehicle2cam = Rcam2vehicle.T # vechicle2cam (imu2cam)

        # used for camera to ladybug system rotation, which is fixed ??
        RstaticCam2bug = self.make_rotation_matrix(
            self.staticCameraRotation[2],
            self.staticCameraRotation[1],
            self.staticCameraRotation[0]
        ) 
        RstaticCam2vehicle = RstaticCam2bug @ Rbug2vehicle
        RstaticVehicle2cam = RstaticCam2vehicle.T

        # world to bug translation : [0,0, vehicle_heigh + imu2bug.z]
        wdelta = - Rvehicle2cam @ self.bugPosition[np.newaxis].T # np.newaxis make it become a row vector
        # bug to camera translation : [0,0, cameraPostion.z]
        cdelta = - RstaticVehicle2cam @ self.cameraPostion[np.newaxis].T
        # final translation vector
        self.t = wdelta + cdelta

        # extrinsic matrix
        self.Rt = np.hstack((Rvehicle2cam, self.t))

        # intrisic matrix
        self.K = np.float32([[self.fx, 0, self.cx],
                             [0, self.fy, self.cy],
                             [0,       0,       1]])

        M_camera_to_video = np.float32([[0, 0, -1],
                                        [0, 1, 0],
                                        [1, 0, 0]])

        self.P = self.K @ M_camera_to_video @ self.Rt

        
        # for debug purposes I compute roll, pitch, yaw that can be printed out
        # http://web.mit.edu/2.05/www/Handout/HO2.PDF
        R = Rcam2vehicle
        self.roll = atan2(R[2,1],R[2,2])
        self.pitch = atan2(-R[2,0], sqrt(R[0,0]**2 + R[1,0]**2))
        self.yaw = atan2(R[1,0], R[0,0])  
        

    def set_camera_rotation(self, rotation):
        self.cameraRotation = rotation
        self.computeCameraMatrix()
    
    def make_rotation_matrix(self, yaw, pitch, roll):
        """Compute a rotation matrix from yaw, pitch, roll according to
        https://en.wikipedia.org/wiki/Rotation_matrix

        Args:
            yaw: 
            pitch: 
            roll: 

        Returns:
            rotation matrix
        """
        # https://en.wikipedia.org/wiki/Rotation_matrix
        Ryaw = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw), cos(yaw), 0],
            [0,0,1]
        ])
        Rpitch = np.array([
            [cos(pitch), 0, sin(pitch)],
            [0, 1, 0],
            [-sin(pitch), 0, cos(pitch)]
        ])
        Rroll = np.array([
            [1,0,0],
            [0, cos(roll), -sin(roll)],
            [0, sin(roll), cos(roll)]
        ])
        return Ryaw @ Rpitch @ Rroll

    def __str__(self) -> str:
        pos = np.squeeze(-self.t)

        res = f"fx: {self.fx}, fy: {self.fy}\ncx: {self.cx}, cy: {self.cy}\n"
        res += f"x (m): {pos[0]}, y (m): {pos[1]}, z (m): {pos[2]}\n"
        res += f"(cam->vehicle) roll (deg): {degrees(self.roll)}, pitch (deg):"\
            f"{degrees(self.pitch)}, yaw (deg): {degrees(self.yaw)}"
        return res
     