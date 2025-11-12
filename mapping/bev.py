#
# Author: Paolo Cudrano (archnnj)
# Modified by: Riccardo (rickycorte)
#

import numpy as np
import cv2

class Bev:
    """
    Bird's Eye View class. Constructed on top of a Camera instance, it enhances 
    it with the capability to construct a BEV image and project points between 
    the BEV plane and the world frame.
    Notice that the world coordinate frame is at ground level; the x axis is 
    pointing forward, while the y axis is pointing towards the left of the car. 
    The z axis is, obviously, pointing upwards.
    Notice moreover that, each time a method requires some points as parameters,
     they are passed as row vectors (stacked if necessary).
    """

    def __init__(self, camera, outView, resolution, shear=None):
        """
        Initializes the BEV object with a camera object, the world coordinates 
        to be projected and the size of the BEV output image to be produced, all
        necessary information for the projection of images and points between
        the two frames.

        Args: 
            camera: an instance of Camera
            outView: [dist_min, dist_max, -right_max, +left_max]; distances 
                measured in meters in the world frame, they define the portion 
                of world ground plane to be centered on.
            resolution: Resolution in meter/pixel used to calculate the output
                image size
            shear: [shearx, sheary] applied to H matrix
        """
        self.camera = camera
        self.outView = outView
        self.resolution = resolution
        self.shear = shear

        #if resolution is None:
        #    self.resolution = outImageSize[0] / (outView[1] - outView[0])

        self.y = int( (outView[1] - outView[0]) / resolution)
        self.x = int( (outView[3] - outView[2]) / resolution)

        self.outImageSize = (self.y, self.x)
        self.computeBevProjection(outView, self.outImageSize)

    def get_resolution(self):
        """Get bev resolution in meter/pixel
        
        Returns:
            output resolution in meter/pixel
        """
        return self.resolution

    def bev_shape(self):
        """Compute bev image shape

        Returns:
            RGB bev shape. If not RGB ignore the last shape dimension
        """
        return (*self.outImageSize, 3)

    def updateCamera(self, camera):
        """Change camera for this instance

        Args:
            camera: new camera to use
        """
        self.camera = camera
        self.computeBevProjection(self.outView, self.outImageSize)
    
    def set_camera_rotation(self, rotation):
        """Update camera rotation.

        Args:
            rotation: new camera rotation
        """
        self.camera.set_camera_rotation(rotation)
        self.computeBevProjection(self.outView, self.outImageSize)

    def getCamera(self):
        """Get current camera for this instance

        Returns:
            current camera object
        """
        return self.camera

    def getCameraToBevProjection(self):
        return self.H_bev

    def getBevToCameraProjection(self):
        return np.linalg.inv(self.H_bev)

    def getWorldToBevProjection(self):
        return self.H_bev.dot(self.camera.getWorldToCameraProjection()) # @

    def getBevToWorldDirProjection(self):
        # if H = (M|m), return M
        return np.linalg.inv(self.getWorldToBevProjection()[:,0:3]) 

    # compute the scaling factor [pixel/meter] given the view [meter] and the output image size [pixel]
    def computeSize(self, outView, outImageSize):
        # compute width and height of the view in meters
        worldHW = np.abs([outView[1] - outView[0], outView[3] - outView[2]])

        # if all elements in outImageSize is not negative
        if not np.any(np.bitwise_or(np.isnan(outImageSize), np.asarray(outImageSize) < 0)):
            # compute scaling factor
            self.scaleXY = np.flip((np.array(outImageSize) - 1) / worldHW)  # pixel/meter
            self.outSize = outImageSize
        # if any element in outImageSize is negative or not specified
        else:
            invalidSizeIndex = np.argwhere(
                np.bitwise_or(np.isnan(outImageSize), np.asarray(outImageSize) < 0)
            )[0][0]

            validSizeIndex = np.argwhere(
                np.logical_not(
                    np.bitwise_or(np.isnan(outImageSize), np.asarray(outImageSize) < 0))
            )[0][0]

            # use only the valid size for the scaling factor
            scale = float(outImageSize[validSizeIndex] - 1) / worldHW[validSizeIndex]
            self.scaleXY = [scale, scale]
            # estimate the other dimension by multiplying its length [meter] by the scaling factor [pixel/meter]
            otherDim = int(round(scale * worldHW[invalidSizeIndex]) + 1)
            self.outSize = outImageSize
            self.outSize[invalidSizeIndex] = otherDim

    def computeBevProjection(self, outView, outImageSize):
        """Compute bev homography matrix

        Args:
            outView: out view in meters  [dist_min, dist_max, -right_max, +left_max]
            outImageSize: output image size in pixels [height, width]
        """
        self.computeSize(outView, outImageSize)

        # get projection matrix from world (vehicle) to camera 3x4
        P = self.camera.getWorldToCameraProjection()
        tform2D_toimage = P[:, [0, 1, 3]] # drop Z, 2d homography
        # compute the inverse -> from camera to 2d world (vehicle)
        tform2D_tovehicle = np.linalg.inv(tform2D_toimage)
        
        # adjust the orientation of the BEV -> invert x and y and take opposite 
        # because in world reference we have x pointing to the forward direction of vehicle, and y pointing to its left
        # working with matrix, row goes down and column goes right
        adjTform = np.float32([[0, -1, 0],
                               [-1, 0, 0],
                               [0, 0, 1]]).T
        bevTform = adjTform.dot(tform2D_tovehicle) # tform2D_tovehicle @ adjTform 

        # position of vehicle wrt top left corner of extentx in meters
        dYdXVehicle = np.float32([outView[3], outView[1]]) 
        tXY = self.scaleXY * dYdXVehicle  # X,Y translation in pixels
        # this viewMatrix will scale the x and y position from meter to pixel, and add the traslation components
        viewMatrix = np.float32([[self.scaleXY[0], 0, 0],
                                 [0, self.scaleXY[1], 0],
                                 [tXY[0] + 1, tXY[1] + 1, 1]]).T
        self.H_bev = viewMatrix.dot(bevTform)  # bevTform @ viewMatrix;

        if self.shear is not None:
            sx = self.shear[0]
            sy = self.shear[1]
            M = np.array(
                [
                    [1, sx, sx * - self.x / 2],
                    [sy, 1, sy * -self.y/ 2],
                    [0,0,1]
                ], dtype=np.float32)

            self.H_bev = M @ self.H_bev 

    def getBevImageSize(self):
        return tuple(np.flip(self.outSize))

    def computeBev(self, img, flags=cv2.INTER_CUBIC):
        """Compite bev for an image

        Args:
            img: image to warp
            flags: cv2 interpolation. Defaults to cv2.INTER_CUBIC.

        Returns:
            bev of img input
        """
        return cv2.warpPerspective(img, self.H_bev, self.getBevImageSize(), flags=flags)

    def projectImagePointsToBevPoints(self, image_points):
        image_points_h = np.hstack(
            (image_points, np.ones((image_points.shape[0], 1)))
        ).T
        bev_points_h = self.H_bev.dot(image_points_h)   # @
        bev_points = bev_points_h[0:2] / bev_points_h[2]
        bev_points = bev_points.T
        return bev_points

    def projectBevPointsToImagePoints(self, bev_points):
        bev_points_h = np.hstack((bev_points, np.ones((bev_points.shape[0], 1)))).T
        image_points_h = np.linalg.inv(self.H_bev).dot(bev_points_h) # @
        image_points = image_points_h[0:2,:] / image_points_h[2,:]
        image_points = image_points.T
        return image_points

    def projectWorldPointsToBevPoints(self, world_points): 
        # world_points = [[x1, y1, z1],...,[xn,yn,zn]]
        image_points = self.camera.projectWorldPointsToImagePoints(world_points)
        bev_points = self.projectImagePointsToBevPoints(image_points)
        return bev_points

    # sara questa la trasformazione a coordinate globali?
    def projectBevPointsToWorldGroundPlane(self, bev_points):
        world_points_z0_h = np.linalg.inv(self.camera.getWorldToCameraProjection()[:, [0, 1, 3]]).dot(
                                    np.insert(self.projectBevPointsToImagePoints(bev_points), 2, 1, axis=1).T
                                    ).T
         # insert z=0 column
        world_points_h = np.insert(world_points_z0_h, 2, 0, axis=1) 
        world_points = world_points_h[:,0:3] / world_points_h[:,[3]]
        return world_points

    def __str__(self) -> str:
        res = f"View size(m): {self.outView} - Image size(px): {self.bev_shape()}\n"
        res += str(self.camera) + "\n"
        res += f"Shear: {self.shear}"
        return res