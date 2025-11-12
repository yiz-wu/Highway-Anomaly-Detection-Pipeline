from typing import List, Tuple, Union
from .dataloader import DataLoader
from .pixelchunk import PixelChunk, AngleMode, SmoothMode, BlendMode
from numpy.typing import NDArray
import numpy as np
import os
import json
import cv2
from .rgb import palette
from .predictors import Predictor
from .location import MapExtents
from .exceptions import OutOfBoundError

class PixelMap:
    """Pixel map with high precision representig a dataset
    """
    def __init__(self) -> None:
        self.chunks: List[PixelChunk] = []
        self.chunk_ids: List[str] = []     
        self.folder = None

        self.max_loaded_chunks: int = 2
        self.loaded_chunks: List[int] = []

    def from_dataset(
        self,
        predictor: Predictor,
        dataset: DataLoader,
        save_path: str,
        symmetric_offset: float = 0,
        angle_mode = AngleMode.ESTIMATED,
        smooth_mode = SmoothMode.NONE,
        distance_weigh_mask: NDArray = None,
        split_after_frames:int = 100,
        limit_chunks = None,
        blend_mode: BlendMode = BlendMode.SUM,
        interpolation= cv2.INTER_NEAREST,
        performance_mode = False
        ):
        """
        Create a map from a dataset

        Args:
            predictor: predictor used to generete the bird eye views
            dataset: dataset to use
            save_path: path where the output map should be saved
            symmetric_offset: offset in meters added to any direction. This may
             be used to center the chunk result or fit cropped parts
            angle_mode: angle to use to rotate and place bev
            smooth_mode: enable smoothing for both position and rotation
            distance_weigh_mask: distance mask multiplied to the bev (must have 
                the same shape of a bev). Leave none if no mask should be applied
            split_after_frames: number of frames for every chunk. 
                This parameter impacts the size of the single chunk and its 
                memory requirements. 
                Lower values mean a huge amount of chunks that use little memory.
            limit_chunks: Limit the number of chunks generates, can be usefull
                to generate only a piece of the map
            blend_mode: mode used to fuse the bird eye views (do not use 
                AVERAGE mode with classes. AVERAGE mode is designed to stitch 
                rgb bevs)
            interpolation: cv2 interpolation used when warping (leave NEAREST 
                for predictions)
        """
        
        self.folder = save_path
        os.makedirs(save_path, exist_ok=True)

        # Computer number of frames per chunk
        frame_per_chunk = split_after_frames

        # Set the number of chunks : equals to the input limit or as many as it needs 
        num_iter = int(len(dataset) / frame_per_chunk) + 1
        if limit_chunks is not None and limit_chunks > 0:
            num_iter = limit_chunks

        # logging
        print(f"Generating map for dataset: {dataset.__class__.__name__}")
        print(f"Generating {num_iter}chuncks @ {frame_per_chunk}frames" \
            f" for {len(dataset)} dataset frames")

        # Generate chunks
        for i in range(num_iter):
            start_frame = frame_per_chunk * i
            end_frame = min(frame_per_chunk * (i+1), len(dataset))

            # create and compose a new chunk
            map_chunk = PixelChunk()
            map_chunk.from_dataset(
                predictor,
                dataset,
                start_frame,
                end_frame,
                symmetric_offset=symmetric_offset,
                distance_weigh_mask = distance_weigh_mask,
                smooth_mode= smooth_mode,
                angle_mode= angle_mode,
                blend_mode= blend_mode,
                interpolation=interpolation,
                performance_mode=performance_mode
            )

            # Save chunk to disk and free chunk data (but not the metadata)
            self.chunks.append(map_chunk)
            self.chunk_ids.append(map_chunk.save(save_path, name=str(i)))
            map_chunk.free_ram()

        # Save filenames of chunks in json
        info = { "ids": self.chunk_ids }
        with open(os.path.join(save_path, "info.json"), 'w') as fp:
            json.dump(info, fp)
        
        print(f"Map saved: {save_path}")
    

    def load(self, path: str):
        """Load map from disk

        Args:
            path: path to map folder
        """
        self.folder = path

        with open(os.path.join(path, "info.json"), "r") as fp:
            info = json.load(fp)
            self.chunk_ids = info["ids"]
        
        for id in self.chunk_ids:
            c = PixelChunk()
            c.load_info(os.path.join(path, id))
            self.chunks.append(c)
        
        if len(self.chunks) <= 0:
            print("No chunks found")
    
    def extents(self) -> MapExtents:
        """Compute map size and anchor that define its world position
        Raises:
            Exception: no chunk can be loaded

        Returns:
            MapExtents containing the map location and extents
        """
        if len(self) <= 0:
            raise Exception("No loaded chunks for this map")

        return self.__compute_extents(self.chunks, self.get_resolution())


    def get_resolution(self) -> float:
        """Return resolution of the current map

        Raises:
            Exception: no chunk found

        Returns:
            map resolution in meter/pixel
        """
        if len(self) <= 0:
            raise Exception("No loaded chunks for this map")
        
        return self.chunks[0].resolution
    

    def dense(self, scale: float =1.0, blend=False):
        """Compute dense representation of the whole map. This operation uses
        a disguisting amount of RAM so be carefull and scale the map to fit your
        machine available resources!

        Args:
            scale: scale factor of the output. Defaults to 1.0.
            blend: set to true if the map is made of rgb chunks that should be
                blended to obtaing a good result. Defaults to False.

        Returns:
            dense matrix representing the full map
        """
        # clear loaded chunks to freeup space
        for i in self.loaded_chunks:
            self.chunks[i].free_ram()
        
        self.loaded_chunks.clear()

        map = self.__stitch_chunks(self.chunks, self.extents(), scale, blend)

        for c in self.chunks:
            c.free_ram()
        
        return map

    def dense_rgb(
        self,
        scale: float =1.0,
        drivable_weight = 0.1,
        show_drivable=False
        ):
        """Return classes as a dense rgb matrix. Be careful with RAM usage. See
        dense() for details. Do not use this function if the map is already rgb, 
        in that case use dense()

        Args:
            scale: scale factor of the output. Defaults to 1.0.
            drivable_weight: weight of drivable class (last class in depth).
                Defaults to 0.1.
            show_drivable: set to true to include the drivable area in the output.
                Defaults to False.

        Returns:
            dense rgb representation of the map
        """
        map = self.dense(scale)

        # postprocess the result set a little value to keep background only 
        # where no other prediction is available
        map[:,:,0] = 1e-5 
        if drivable_weight is not None:
            drivable_class = map.shape[-1] - 1
            map[:,:, drivable_class] *= drivable_weight # weight drivable 
        
        map = np.argmax(map, axis=-1)
        
        if not show_drivable and drivable_weight is not None:
            map[map == drivable_class] = 0

        output = np.ones((*map.shape, 3), dtype=np.uint8)
        classes = np.unique(map)
        for c in classes:
            output[map == c] = np.array(palette[c], dtype=np.uint8)
        
        return output


    def save_rgb(
        self,
        file_name:str,
        scale: float = 0.1,
        drivable_weight = 0.1,
        show_drivable=False
        ):
        """Save output of dense_rgb() as ab image. 
        Do not use this function if the map is already rgb. In that case save 
        the dense() matrix

        Args:
            file_name: output file name
            scale: scale factor of the output. Defaults to 0.1.
            drivable_weight: weight of drivable class (last class in depth).
                Defaults to 0.1.
            show_drivable: set to true to include the drivable area in the output.
                Defaults to False.
        """
        path, n_ = os.path.split(file_name)
        if path != "":
            os.makedirs(path, exist_ok=True)

        output = self.dense_rgb(scale, drivable_weight, show_drivable)
        
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_name, output)



    def save_chunks_rgb(
        self,
        folder:str,
        scale: float =1.0,
        drivable_weight = 0.0,
        show_drivable=False
        ):
        """Save rgb class representation of every chunk that compose this map

        Args:
            folder: folder where images are saved
            scale: scale factor of every chunk output. Defaults to 1.0.
            drivable_weight: _description_. Defaults to 0.0.
            drivable_weight: weight of drivable class (last class in depth).
                Defaults to 0.1.
            show_drivable: set to true to include the drivable area in the output.
                Defaults to False.

        Raises:
            Exception: This map has no chunks
        """
        if len(self) <= 0:
            raise Exception("No loaded chunks for this map")

        os.makedirs(folder, exist_ok=True)

        for c in self.chunks:
            file_name = os.path.join(folder, f"{c.start_frame}-{c.end_frame}.png")
            c.save_rgb(file_name, scale, drivable_weight, show_drivable)
            c.free_ram()
    
    def _clean_junk(
        self,
        dense,
        connectivity,
        min_area,
        drivable_class,
        ):
        # scale to find decent results
        dense[...,0] = .1
        if drivable_class is not None:
            dense[...,drivable_class] *= .1

        mask = np.argmax(dense, axis=-1)
        # remove drivable
        if drivable_class is not None:
            dc = dense.shape[-1] -1 if drivable_class == -1 else drivable_class
            mask[mask == dc] = 0
                

        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
            (mask != 0).astype(np.uint8),
            connectivity,
            cv2.CV_32S
        )
        sizes = stats[1:, -1]
        nb_components = nb_components - 1 
        # clean junk
        for j in range(0, nb_components):
            if sizes[j] <= min_area:
                dense[output == j + 1] *= 0
        
        # revert scaling
        if drivable_class is not None:
            dense[...,drivable_class] /= .1

        return dense

    
    def region_around(
        self,
        position: NDArray,
        size: NDArray,
        autofix: False,
        min_area_size = None,
        drivable_class = -1,
        ) -> Tuple[NDArray, NDArray]:
        """Get a dense rectangular region of the map centerd in position and with
        dimension size.

        If the required region exceeds the available map an exception is thrown

        Args:
            position: (x,y) position of the center of the retuned region in meters
            size: total size of the region as (x, y) in meters
            autofix: when possible try to fix top left corner to fit the required
                region inside a valid part of the map
            min_area_size: delete block with area under this value. Use None to disable
                Defaults to None
        
        Raises:
            OutOfBoundError if the required area is not fully contained inside 
                the map

        Returns:
            dense NDArray containing the required map region, and the top left
                corner in world coordinates

            None if no region can be found
        """
        # search containing chunks
        containing_chunks = []
        for i in range(len(self.chunks)):
            if self.chunks[i].contains(position):
                containing_chunks.append(i)

        # compure region bound points
        rtl = position + [-1, 1] * np.asarray(size) / 2
        rbl = position + [+1, -1] * np.asarray(size) / 2

        # compute the dense where we sample the region
        if len(containing_chunks) < 1:
            return None
        
        # check that the region is fully contained in the avaivlable extents
        extents = None
        chunks = None

        if len(containing_chunks) == 1:
            chunks = [self.chunks[containing_chunks[0]]]
            extents = chunks[0].extents()
        else:
            chunks = [self.chunks[cid] for cid in containing_chunks]
            extents = self.__compute_extents(chunks, self.get_resolution())
        
        # try to move the required region inside the available space
        if autofix:
            # fix right 
            delta_r = rbl[0] - extents.get_bottom_right_anchor()[0]
            if delta_r > 0:
                rtl[0] -= delta_r; rbl[0]-= delta_r
            # fix left
            delta_l = extents.get_top_left_anchor()[0] - rtl[0] 
            if delta_l > 0:
                rtl[0] += delta_l; rbl[0] += delta_l
            # fix bottom
            delta_b = extents.get_bottom_right_anchor()[1] - rbl[1]
            if delta_b > 0:
                rtl[1] += delta_b; rbl[1] += delta_b
            # fix top
            delta_t = rtl[1] - extents.get_top_left_anchor()[1]
            if delta_t > 0:
                rtl[1] -= delta_t; rbl[1] -= delta_t

        if not extents.contains(rtl) or not extents.contains(rbl):
            raise OutOfBoundError # TODO: add better error
        
        # compute the sampling region
        dense = None

        self.__update_chunk_cache(containing_chunks)

        if len(containing_chunks) == 1:
            dense = chunks[0].dense()
        else:         
            dense = self.__stitch_chunks(chunks, extents, 1, False)
            # we may have allocated a few more chunks so we remove them
            self.__update_chunk_cache(containing_chunks)

        if min_area_size is not None:
            dense = self._clean_junk(dense, 8, min_area_size, drivable_class)

        # sample region and return result by computing translating posistions
        # into pixels
        size = np.asarray(size)
        size = (size / self.get_resolution()).astype(np.int32)
        tl = (rtl - extents.get_top_left_anchor()) / self.get_resolution()
        tl = (tl * [1, -1]).astype(np.int32) # y in images is flipped
        br = tl + size

        return dense[tl[1]: br[1], tl[0] : br[0], :], rtl


    def free_ram(self):
        """Free ram that may be used by loaded chunks
        """
        for c in self.chunks:
            c.free_ram()
        
        self.loaded_chunks.clear()

    def __len__(self):
        return len(self.chunks)


    def __update_chunk_cache(self, chunk_id: Union[int, List[int]]):
        if not isinstance(chunk_id, List):
            chunk_id = [chunk_id]
        
        for cid in chunk_id:
            if cid in self.loaded_chunks:
                # mremove chunk from old expiration list
                self.loaded_chunks.pop(self.loaded_chunks.index(cid))
            else:
                # keep ram clean by removin unused chunks from ram
                if len(self.loaded_chunks) > self.max_loaded_chunks:
                    unload = self.loaded_chunks.pop()
                    self.chunks[unload].free_ram()

            # add chunk to cached ones. Actual loading is done at first dense call
            self.loaded_chunks.append(cid)
    

    def __compute_extents(self, chunks: List[PixelChunk], resolution: float):
        extents = MapExtents(resolution)
     
        # iterate all chunks to add both anchors
        for c in chunks:
            extents.put_point(
                c.extents().get_top_left_anchor()
            )
            extents.put_point(
                c.extents().get_bottom_right_anchor() 
            )

        return extents

    
    def __stitch_chunks(self, chunks, extents, scale = 1.0, blend = False):
        # first we need to find the full size required to fit the whole map
        scaled_size = (extents.size(inpixel=True) * scale).astype(np.int32)

        # stitch toghether the scaled blocks
        sz = (*scaled_size, chunks[0].depth()) 
        map = np.zeros(sz)

        for c in chunks:
            delta_meters = (
                c.extents().get_top_left_anchor() - 
                extents.get_top_left_anchor()
            )
            delta = delta_meters * scale * [1, -1] / extents.resolution

            T = np.array([
                [scale, 0, delta[0]],
                [0, scale, delta[1]]
            ])

            warped = cv2.warpAffine(
                c.dense(),
                T,
                sz[:2][::-1],
                flags=cv2.INTER_NEAREST
            )

            if blend:
                mask_warp = warped.sum(axis=-1) 
                mask_dest = map.sum(axis=-1)
                true_overlap = (mask_warp != 0) * (mask_dest != 0)

                map[true_overlap, :] = cv2.addWeighted(
                    warped[true_overlap].astype(np.float32),
                    0.5,
                    map[true_overlap].astype(np.float32),
                    0.5,
                    0
                )

                warped[true_overlap, :] = 0
                map += warped
            else:
                map += warped

        return map
        
