import fcntl
import glob
import json
import logging
import os
from functools import lru_cache
from typing import Tuple


import geopandas as gpd
import numpy as np
import numpy.typing as npt

# import pyogrio
# import rasterio

# local lib
from devkit.sim_engine.map_manager.map_expansion.bit_map import BitMap
from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMap

# from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMapExplorer


# class MineMapsJsonPng:
#     """组合 BitMap and TgScenesMap(semanticMap) ."""

#     def __init__(
#         self,
#         data_root: str = None,
#         map_root: str = None,
#         location: str = None,
#         map_version: str = "v_1_5",
#         utm_local_range: Tuple[float, float, float, float] = None,
#     ) -> None:
#         self.map_version = map_version
#         self.data_root = data_root
#         self.map_root = map_root
#         self.location = location
#         self.utm_local_range = utm_local_range
#         self.semantic_map: TgScenesMap = None
#         self.bit_map_mask: BitMap = None

#         self._get_dataset_root()
#         self._get_maps()

#     def _get_dataset_root(self):
#         if self.data_root is not None:
#             _data_root = self.data_root
#         if self.map_root is not None:
#             _data_root = os.path.dirname(self.map_root)
#         self.data_root = _data_root

#     def _get_maps(self):
#         self.semantic_map = TgScenesMap(self.data_root, self.location)
#         self.bit_map_mask = BitMap(self.data_root, self.location, "bitmap_mask")
#         # self._bit_map_rgb = BitMap(self.data_root, self.location, "bitmap_rgb", is_transform_gray=True)
#         self.bit_map_mask.image_ndarray_local = self.bit_map_mask.load_bitmap_using_utm_local_range(
#             utm_local_range=self.utm_local_range, x_margin=20, y_margin=20
#         )

#         self.map_name = self.location + "_semanticMap" + "_BitMap"

#     # The size of the cache was derived from testing with the Raster Model
#     #   on our cluster to balance memory usage and performance.
#     # @lru_cache(maxsize=16)
#     # def s(self, location: str, layer_name: str) -> gpd.geodataframe:
#     #     """Inherited, see superclass."""
#     #     # TODO: Remove temporary workaround once map_version is cleaned
#     #     location = location.replace(".gpkg", "")

#     #     rel_path = self._get_gpkg_file_path(location)
#     #     path_on_disk = os.path.join(self._map_root, rel_path)

#     #     if not os.path.exists(path_on_disk):
#     #         layer_lock_file = f"{self._map_lock_dir}/{location}_{layer_name}.lock"
#     #         self._safe_save_layer(layer_lock_file, rel_path)
#     #     self._wait_for_expected_filesize(path_on_disk, location)

#     #     with warnings.catch_warnings():
#     #         # Suppress the warnings from the GPKG operations below so that they don't spam the training logs.
#     #         warnings.filterwarnings("ignore")

#     #         # The projected coordinate system depends on which UTM zone the mapped location is in.
#     #         map_meta = gpd.read_file(path_on_disk, layer="meta", engine="pyogrio")
#     #         projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

#     #         gdf_in_pixel_coords = pyogrio.read_dataframe(path_on_disk, layer=layer_name, fid_as_index=True)
#     #         gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)

#     #         # For backwards compatibility, cast the index to string datatype.
#     #         #   and mirror it to the "fid" column.
#     #         gdf_in_utm_coords.index = gdf_in_utm_coords.index.map(str)
#     #         gdf_in_utm_coords["fid"] = gdf_in_utm_coords.index

#     #     return gdf_in_utm_coords
