import numpy as np

class CameraBase:
    """Base class for all bev-compatible cameras
    """
    
    def __init__(self) -> None:
        self.k = None
        self.P = None
        self.rotation_warn = False

    def getIntrinsicMatrix(self):
        """Get camera intrisic matrix

        Returns:
            camera instrinsic matrix
        """
        return self.K

    def getWorldToCameraProjection(self):
        """Get world to camera matrix

        Returns:
            world to camera matrix
        """
        return self.P

    def getCameraToWorldDirProjection(self):
        return np.linalg.inv(self.P[:,0:3]) # if P = (M|m), return M

    def projectImagePointToWorldDirection(self, image_point):
        image_point_h = np.float32([image_point[0], image_point[1], 1])
        world_direction = self.getCameraToWorldDirProjection().dot(image_point_h)
        return world_direction

    def projectWorldPointsToImagePoints(self, world_points):
        # world_points = [[x1, y1, z1],...,[xn,yn,zn]]
        world_points_h = np.hstack((world_points, np.ones((world_points.shape[0], 1)))).T
        image_points_h = self.getWorldToCameraProjection().dot(world_points_h) # @
        image_points = image_points_h[0:2,:] / image_points_h[2,:]
        image_points = image_points.T
        return image_points

    def computeCameraMatrix(self):
        """Compute world to camera matrix using camera instrisincs and extrinsics
        """
        raise NotImplementedError
    
    def set_camera_rotation(self, rotation):
        """Update camera rotation and recompute world to camera matrix

        Args:
            rotation: new camera rotation
        """
        if self.rotation_warn:
            print("Warning: You are trying to update camera rotation with a"\
                "camera that does not support this feature"
            )
            self.rotation_warn = True
