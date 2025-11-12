import numpy as np
from numpy.typing import NDArray
from tqdm import trange
import math
import cv2
import uuid
import os
import zipfile
import json
import copy
from time import perf_counter

from . import AngleMode, BlendMode, SmoothMode
from .dataloader import DataLoader
from .predictors import Predictor
from .rgb import palette, map2D_to_RGB
from .location import MapExtents, OffsetDirection

def angle_rad(delta):
    return math.atan2(delta[1], delta[0])

class PixelChunk:
    """Rectangular region of a pixel map. Each pixel represents a square in the
    world if resolution x resolution.

    TODO:
        save raw as img
    """
    def __init__(self) -> None:
        self.indices = None 
        self.data = None

        self.map_extents = None

        self.start_frame = None
        self.end_frame = None

        self.start_pos = None
        self.last_pos = None

        self.resolution = None
        self.shape = None       # [width, height, depth]

        self.needs_reload = False
        self.disk_file = None

    # Create a pixel map chunk from dataset. 
    def from_dataset(
        self,
        predictor: Predictor,
        dataset: DataLoader,
        start_frame: int,
        end_frame : int,
        symmetric_offset: float = 0, 
        angle_mode = AngleMode.ESTIMATED,
        smooth_mode = SmoothMode.NONE,
        distance_weigh_mask: NDArray = None,
        blend_mode: BlendMode = BlendMode.SUM,
        interpolation= cv2.INTER_NEAREST,
        snapshot_path=None,
        performance_mode = False
        ):
        """
        Create a pixel map chunk from dataset. 

        Args:
            predictor: predictor used to generete the bird eye views
            dataset: dataset to use
            start_frame: first frame of the dataset to use to generate this chunk,
            end_frame: last frame of the dataset used to generate this chunk,
            symmetric_offset: offset in METER added to both direction. This may
                be used to center the chunk result or fit cropped parts
            angle_mode: angle to use to rotate and place bev
            smooth_mode: enable smoothing,
            distance_weigh_mask: distance mask multiplied to the bev (must have 
                the same shape of a bev). Leave none if no mask whould be applied
            blend_mode: mode used to fuse the bird eye views
            interpolation: cv2 interpolation used when warping
            snapshot_path: folder where save a rgb snaphot of the chunk after
                every frame, usefull only for visualization only
        """
        p_start_time = perf_counter()
        p_disk_load_times = []
        p_iter_times = []
        p_pred_times = []

        self.start_frame = start_frame
        if end_frame == -1:
            end_frame = len(dataset)
        self.end_frame = end_frame

        self.resolution = predictor.get_map_resolution()

        # Compute chunk map EXTENTS (anchors) from GPS location of dataset
        extents = MapExtents(self.resolution)
        extents.find(dataset, start_frame, end_frame)
        # Add symmetric offset on both directions -> (x,y), then convert from meters to pixels 
        extra_offset = np.array([symmetric_offset] * 2) / self.resolution
        # offset equals to the half of the bev size + extra offset
        extents.set_offset(
            OffsetDirection.LEFT_TOP,
            predictor.get_bev_size()[::-1] / 2 + extra_offset,  # switch the dimensions to (width, height)
            inpixel=True
        )
        extents.set_offset(
            OffsetDirection.RIGHT_BOTTOM,
            predictor.get_bev_size()[::-1] / 2 + extra_offset,
            inpixel=True
        )
        self.map_extents = extents


        # get shape (height, width, depth) in pixel and start generating chunk
        self.shape = (*extents.size(inpixel=True), predictor.get_map_depth())
        num_classes = self.shape[-1]    # depth

        # initialize the output pixel map
        output_map = np.zeros(self.shape)

        if snapshot_path is not None:
            os.makedirs(snapshot_path, exist_ok=True)
        
        # logging
        if not performance_mode:
            print(f"Frame: {start_frame} -> {end_frame}\n" \
                f" Anchor: {self.map_extents.get_top_left_anchor()}." \
                f"\n Output image size: {self.shape}")

        # check if can use old position to inizialize angle
        if angle_mode == AngleMode.ESTIMATED and start_frame > 0:
            # position before the first frame
            prev_pos = dataset.position(start_frame - 1)[:2]    # GPS position (x,y) 
        else:
            prev_pos = None
        
        self.start_pos = dataset.position(start_frame)
        self.last_pos = dataset.position(end_frame - 1)

        # inference loop : predict and integrate it to chunk
        for i in trange(end_frame - start_frame, disable=performance_mode): # progress bar, stepping i of 1
            p_iter_start = perf_counter()

            # offset to correct index
            i += start_frame 

            # load image, position and rotation
            p_disk_start = perf_counter()
            image, pos, rot = dataset[i]
            p_disk_load_times.append((p_disk_start, perf_counter()))
            pos = np.array(pos)[:2] # GPS position (x,y) at frame i

            if smooth_mode == SmoothMode.POSTION or smooth_mode == SmoothMode.ALL:
                pos = dataset.smooth_position(i)    # probably WRONG: double offset correction to index i, here and inside the dataset.smooth_position(i) method

            # first image is skipped and used as reference for the first angle
            if i == 0 and prev_pos is None:
                prev_pos = pos
            
            # recalculate camera calibration at every frame
            predictor.before_predicion(dataset, i)

            # perform inference
            p_pred_start = perf_counter()
            pred = predictor.predict_bev(image)
            p_pred_times.append((p_pred_start, perf_counter()))

            # post process prediction (filtering out small block of predictions)
            pred = predictor.post_process_bev(pred)
            
            # check dimension
            assert pred.ndim == 2 or pred.ndim == 3, \
                f"Unsupported shape received. Expected number of dimensions" \
                f" is 2 or 3 but got {pred.ndim}"

            # if needed transform 2d prediction into 3d prediction
            if pred.ndim == 2:
                # check dtype
                assert pred.dtype == np.uint8, f"Expected a 2D uint8 prediction but got {pred.shape} of type {pred.dtype}"

                ''' 
                transform to One-Hot encode the labels in prediction (https://stackoverflow.com/questions/36960320/)
                    2D: (x,y) -> 3D: (x,y,one-hot encoding)
                    [[1,3],
                     [2,4]] ->  [[1,0,0,0], [0,0,1,0],
                                 [0,1,0,0], [0,0,0,1]]
                '''
                pred = (np.arange(predictor.get_map_depth()) == pred[...,None]) # [..., None] syntax adds a new axis at the end
            else:
                # 3d map is returned, no need to add depth
                # check depth
                assert pred.shape[-1] == predictor.get_map_depth(), \
                    f"Expected bev depth of {predictor.get_map_depth()} but got {pred.shape[-1]}"

            pred = pred.astype(np.float32)

            # apply weight to prediction
            if distance_weigh_mask is not None:
                # replicate 2d weight for every channel
                alpha = np.broadcast_to(            # broadcast(copy) the 1st array parameter to the shape of 2nd parameter
                    distance_weigh_mask[...,None],  # 2D weight mask -> 3D mask with depth 1
                    distance_weigh_mask.shape + (num_classes, ) # 3D mask with (depth == num_classes)
                )
                pred *= alpha



            # yaw offset in radians : turning left/right of the car 
            if smooth_mode == SmoothMode.ANGLE or smooth_mode == SmoothMode.ALL:
                angle = dataset.smooth_heading(i)
            else:      
                angle = angle_rad(pos-prev_pos) if angle_mode == AngleMode.ESTIMATED else dataset.rotation(i)[0]
            
            if angle_mode == AngleMode.ONLY_OFFSET:
                # angle in radians needed to rotate the bev to look at forward direction of vehicle (usually x axis)
                angle = predictor.get_bev_to_forward_angle()
            else:
                angle = angle + predictor.get_bev_to_forward_angle()

            # convert angle from radians to degrees
            angle = math.degrees(angle)

            # get the position of center of rotation of bev in pixel: (x, y, 1)
            bev_rotation_center = predictor.get_bev_center()

            # Compute rotation matrix R needed to rotate bev images around a center point (car position)
            #   has shape (2, 3) -> (2x3) matrix
            #   first two columns are the rotation component, last column is the traslation component (from center point)
            R = cv2.getRotationMatrix2D(bev_rotation_center, angle, 1.0) 
            
            # delta = offset vector from top left anchor to the current position, with vertical axis inverted (convention of origin at left top)
            #   - pos -> GPS position (x,y) at frame i
            #   - extents.get_top_left_anchor() -> GPS position of top left anchor of the chunk [start_frame, end_frame] 
            delta = (pos - extents.get_top_left_anchor()) * [1, -1] 

            # check and warn for precision loss due to conversions (where??)
            if (delta.dtype == np.float64 and 
                np.all(np.absolute(delta - np.float32(delta)) > 0.005)):

                print(f"Warning: You may have lost positioning precision due" \
                        f" to conversions at frame {start_frame + i}")

            # Compute traslation component T due to offset from anchor of chunck (delta) of the bev's center point (car position)
            T = np.array([  [0, 0, delta[0] / self.resolution - bev_rotation_center[0] ],
                            [0, 0, delta[1] / self.resolution - bev_rotation_center[1] ]  ])

            # Apply rotation and traslation to the predicted bev to get objective bev image in chunck
            warped = cv2.warpAffine(
                pred,
                R + T,
                self.shape[:2][::-1],   # invert shape to (width, height)
                flags=interpolation
            )

            # Add bev image to the chunk image (output_map)
            if blend_mode == BlendMode.SUM:
                output_map += warped
            elif blend_mode == BlendMode.AVERAGE:
                # Blend only overlapping pixels, sum other points

                # Compute a mask of overlapping pixels in the two images
                # sum along last axis (depth) to get 2D masks of pixels that are not zero
                mask_warp = warped.sum(axis=-1)
                mask_output_map = output_map.sum(axis=-1)
                true_overlap = (mask_warp != 0) * (mask_output_map != 0)
                
                # Blend two images with equal weight
                blended = cv2.addWeighted(
                    output_map.astype(np.float32),    0.5,
                    warped.astype(np.float32),  0.5,    gamma=0)

                # Sum the two images and replace the overlapping pixels with the blended ones
                output_map += warped
                output_map[true_overlap, :] = blended[true_overlap, :]
            elif blend_mode == BlendMode.ENHANCE_OVERLAP:
                mask_warp = warped.sum(axis=-1)
                mask_output_map = output_map.sum(axis=-1)
                true_overlap = (mask_warp != 0) * (mask_output_map != 0)
                
                warped[true_overlap] *= 2 # weight more points that did overlap
                output_map += warped
            else:
                raise Exception("Invalid blend mode")

            prev_pos = pos

            # save snapshot of current chucnk
            if snapshot_path is not None:
                self.indices = np.nonzero(output_map) 
                self.data = output_map[self.indices] 
                self.save_rgb(os.path.join(snapshot_path, f"{i}.png"), scale=0.5)

            p_iter_times.append((p_iter_start, perf_counter()))
        # end loop

        # save chunck to self.data
        self.indices = np.nonzero(output_map) 
        self.data = output_map[self.indices]   
        self.needs_reload = False

        # logging
        p_end_time = perf_counter()

        if performance_mode:
            total_time = p_end_time - p_start_time

            iter_deltas = np.array([p[1]-p[0] for p in p_iter_times])
            disk_deltas = np.array([p[1]-p[0] for p in p_disk_load_times])
            pred_deltas = np.array([p[1]-p[0] for p in p_pred_times])

            if disk_deltas.shape[0] > iter_deltas.shape[0]:
                disk_deltas = disk_deltas[1:]

            avg_iter_no_disk = np.mean(iter_deltas- disk_deltas)

            processing_deltas = iter_deltas - disk_deltas - pred_deltas

            print("-" * 30)
            print("Chunk generation performance:")
            print(f"Total time: {total_time} s".replace(".", ","))
            print(f"Total disk time disk: {np.sum(disk_deltas)} s".replace(".", ","))
            print(f"(iter) Avg time: {np.mean(iter_deltas)} s".replace(".", ","))
            print(f"(iter) Avg time (no disk): {avg_iter_no_disk} s".replace(".", ","))
            print(f"(iter) Avg pred time: {np.mean(pred_deltas)} s".replace(".", ","))
            print(f"(iter) Avg processing time: {np.mean(processing_deltas)} s".replace(".", ","))


    
    def extents(self) -> MapExtents:
        """Get chunk extents and location

        Returns:
            map extents of this chunk
        """
        return copy.deepcopy(self.map_extents)

    
    def depth(self):
        """Get chunk depth

        Returns:
            depth/number of channels of each pixel
        """
        return self.shape[-1]

    # return a np array of shape (width, height, depth) with values equal to the data of the chunk
    def dense(self, scale: float = 1.0) -> NDArray:
        """Get a dense representation of this chunk. This representation may use
        a huge amout of RAM so be careful. Use the scale factor to reduce ram 
        usage at the cost of output resolution

        Args:
            scale: rescale output. Defaults to 1.0.

        Returns:
            dense matrix representing the chunk
        """

        # reload data from disk if was cleared to keep free space
        if self.needs_reload:
            self.load(self.disk_file)

        map = np.zeros(self.shape, dtype=np.float32)
        map[self.indices] = self.data

        if scale != 1:
            new_shape = (np.array(self.shape[:2]) * scale).astype(np.int32)[::-1]
            map = cv2.resize(map, new_shape, interpolation=cv2.INTER_NEAREST)

        return map

    # save the chunk as an archive with the indices, data and chunk info
    def save(self, path, name=None) -> str:
        """Save on disk this chunk

        Args:
            path: path where this file should be written
            name: name of this file, if none an automatic name is picked 
                and returned

        Returns:
            name of the saved file 
        """

        if self.data is None:
            raise Exception("Cannot save empty chunk. You may need to call" \
                " load() manually or generate the chunk with from_dataset()")

        if name is None:
            name = str(uuid.uuid4())
        
        file_name = f"{name}.block"

        os.makedirs(path, exist_ok=True)

        indices_file = os.path.join(path, f"i{name}.npz")
        # transform tuple (dim1, dim2, ...) into a vertical array 
        # that can be stored in one block
        np.savez_compressed(indices_file, np.transpose(self.indices)) 

        data_file = os.path.join(path, f"d{name}.npz")
        np.savez_compressed(data_file, self.data)

        info = {
            "version": "1.0",
            "extents": self.map_extents.to_json(),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "start_pos": self.start_pos.tolist(),
            "end_pos": self.last_pos.tolist(),
            "resolution": float(self.resolution),
            "shape": [int(i) for i in self.shape],
        }
        
        with zipfile.ZipFile(os.path.join(path, file_name), mode="w") as archive:
            archive.write(indices_file, "indices")
            archive.write(data_file, "data")
            archive.writestr("info.json", json.dumps(info))

        os.remove(indices_file)
        os.remove(data_file)

        self.disk_file = os.path.join(path, file_name)
        return file_name

    # load only the metadata of the chunk saved in the file
    def load_info(self, file):
        """Load only the metadata of the chunk

        Args:
            file: name of the file to load
        """
        with zipfile.ZipFile(file, mode="r") as archive:       
            
            info = json.loads(archive.read("info.json"))

        self.map_extents = MapExtents.from_json(info["extents"])
        self.start_frame = info["start_frame"]
        self.end_frame = info["end_frame"]
        self.start_pos = np.array(info["start_pos"])
        self.last_pos = np.array(info["end_pos"])
        self.resolution = info["resolution"]
        self.shape = tuple(info["shape"])

        self.disk_file = file
        self.needs_reload = True
        
    # load the data of the chunk saved in the file, data under the form of "indices" and "data"
    def load(self, file):
        """Load the chunk from file

        Args:
            file: name of the file to load
        """

        self.load_info(file)

        with zipfile.ZipFile(file, mode="r") as archive:
            
            with archive.open("indices") as fp:
                self.indices = np.load(fp)["arr_0"]
                self.indices = tuple(self.indices.T)
            
            with archive.open("data") as fp:
                self.data = np.load(fp)["arr_0"]
        
        self.needs_reload = False

    # delete the indices and data
    def free_ram(self):
        """Clear ram for this chuck. Next operation will require a reload from disk"""
        
        self.needs_reload = True
        self.indices = None
        self.data = None

        del self.indices
        del self.data

    # get the rgb matrix of the chunk
    def dense_rgb(
        self,
        scale: float =1.0,
        drivable_weight = 0.1,
        show_drivable=True,
        threshold:float = None,
        bg_weight=1e-5
        ) -> NDArray:
        """Compute an rgb dense version of the current chunk using the palette 
        defined in rgb module. This function is designed to work with class maps
        and offset an easy and clean way to visualize predictions as rgb images.

        If you did generate a RGB chunk use dense() to retrive the rgb format. 

        Args:
            scale: scale factor of the output. Defaults to 1.0.
            drivable_weight: weight to multiple to drivable class that is assume
                to be the last one in depth. Defaults to 0.1.
            show_drivable: set to true to show the drivable in the output.
                Defaults to True.
            threshold: set a min value for a prediction to be valid.
                Defaults to None.
            bg_weight: Weight of the background class. Assumed to be depth 0.
                Defaults to 1e-5.

        Returns:
            Dense rgb matrix representing the chunk
        """
        map = self.dense(scale)

        if bg_weight is not None:
            # set a little value to keep background only 
            # where no other prediction is available
            map[:,:,0] *= bg_weight 
        
        if drivable_weight is not None:
            drivable_class = map.shape[-1] - 1
            map[:,:, drivable_class] *= drivable_weight # weight drivable 

        if threshold is not None:
            invalid_pixels = map.max(axis=-1) < threshold
        
        # get the 2D index of the max value along last axis (depth)
        map = np.argmax(map, axis=-1)   
        
        # if the max value is equal to drivable area class, set it to 0
        if not show_drivable and drivable_weight is not None:
            map[map == drivable_class] = 0
        
        if threshold is not None:
            map[invalid_pixels] = 0

        output = np.ones((*map.shape, 3), dtype=np.uint8)
        classes = np.unique(map)
        for c in classes:
            output[map == c] = np.array(palette[c], dtype=np.uint8)
        
        return output

    # save the output of dense_rgb() as an image
    def save_rgb(
        self,
        file_name:str,
        scale: float =1.0,
        drivable_weight = 0.1,
        show_drivable=False,
        threshold:float = None,
        white_bg = False
        ):
        """Save an rgb image of this class prediction chunk. 
        This function is a shortcut to save the output of dense_rgb()

        Args:
            file_name: output file name
            scale: scale factor of the output. Defaults to 1.0.
            drivable_weight: weight to multiple to drivable class that is assume
                to be the last one in depth. Defaults to 0.1.
            show_drivable: set to true to show the drivable in the output.
                Defaults to True.
            threshold: set a min value for a prediction to be valid.
                Defaults to None.
            bg_weight: Weight of the background class. Assumet to be depth 0.
                Defaults to 1e-5.
        """
        path, _ = os.path.split(file_name)
        if path != "":
            os.makedirs(path, exist_ok=True)

        output = self.dense_rgb(scale, drivable_weight, show_drivable, threshold)

        if white_bg:
            output[output.sum(-1) == 0] = [255, 255, 255]
        
        output =  cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_name, output)

    def __str__(self) -> str:
        res = f"Chunk {self.start_frame} -> {self.end_frame}:\n"
        res += f" Anchor: {self.map_extents.get_top_left_anchor()}\n"
        res += f" Size (y,x): {self.map_extents.size()}m\n"
        res += f" Shape: {self.shape}"

        return res


    def contains(self, position: NDArray) -> bool:
        """Check if a chunk contains a world position

        This is the same as calling extents().contains(...)

        Args:
            position: (x,y) position in metes to check 

        Returns:
            true if the specified position is inside this chunk 
        """
        return self.map_extents.contains(position)