
import matplotlib.pyplot as plt
import numpy as np
from lytools import *
from rasterio.transform import from_bounds
import xarray as xr
from netCDF4 import Dataset
from rasterio.warp import Resampling, calculate_default_transform, reproject
import pyproj
from pprint import pprint
T = Tools()

from __Global__ import *


tif_template= rf'D:\Western_US_IAV\Data\basedata\200902.tif'
D=DIC_and_TIF(tif_template=tif_template)

class My_functions:
    def __init__(self):
        pass


    def align_tif(self,fpath,reference_path,outpath):

        with rasterio.open(reference_path) as ref:
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_width = ref.width
            ref_height = ref.height

        with rasterio.open(fpath) as src:
            out_meta = src.meta.copy()
            out_meta.update({
                'crs': ref_crs,
                'transform': ref_transform,
                'width': ref_width,
                'height': ref_height,
                'nodata': src.nodata if src.nodata is not None else -9999
            })

            with rasterio.open(outpath, 'w', **out_meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.nearest
                    )

