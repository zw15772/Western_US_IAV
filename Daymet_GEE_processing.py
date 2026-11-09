# coding='utf-8'
"""
Daymet precipitation indices exported from GEE (gee_daymet_precip_indices.js).

Input : D:\\Western_US_IAV\\Data\\Daymet_GEE\\daymet_<index>_1982_2024_tmpl*.tif
        one GeoTIFF per index, 43 bands (y1982 ... y2024), grid = template_align.tif
        (EPSG:4326, 0.05 deg, 449 cols x 468 rows, UL corner -124.529167, 48.979167)
Output: D:\\Western_US_IAV\\Result2\\00_aligned\\Daymet_GEE\\<index>.npy
        dic[(row, col)] = np.array of 43 annual values (1982-2024)
        (row, col) matches the template / LAI dicts directly.

Indices: PRCPTOT, WETDAYS, R5mm, LIGHT, Rx1day, SDII, R95p, R95pTOT
Parameter notes: coworkplayground\\docs\\daymet_precip_indices_README.md
"""
import os
import glob

import numpy as np
import rasterio

from __Global__ import *


class Daymet_GEE:

    def __init__(self):
        self.datadir = os.path.join(data_root, 'Daymet_GEE')
        self.outdir = os.path.join(this_root, 'Result2', '00_aligned', 'Daymet_GEE')
        os.makedirs(self.outdir, exist_ok=True)
        self.start_year = 1982
        self.end_year = 2024
        self.indices = ['PRCPTOT', 'WETDAYS', 'R5mm', 'LIGHT',
                        'Rx1day', 'SDII', 'R95p', 'R95pTOT']

    def run(self):
        self.tif_to_dict()
        # self.p95_to_dict()

    def _read_stack(self, fpath):
        with rasterio.open(fpath) as src:
            arr = src.read().astype(np.float32)
            nodata = src.nodata
            ## GEE may export one extra row on top; cut rows to match the template (origin 48.979, 468 rows)
            row0 = int(round((src.transform.f - 48.97916666666667) / 0.05))
        arr = arr[:, row0:row0 + 468, :]
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr

    def _find_file(self, name):
        flist = sorted(glob.glob(os.path.join(self.datadir, f'daymet_{name}_*tmpl*.tif')))
        if len(flist) == 0:
            print(f'[skip] no file for {name}')
            return None
        if len(flist) > 1:
            # GEE splits large exports into tiles; mosaic first, e.g.
            #   gdalbuildvrt daymet_<name>.vrt <tiles>  -> then gdal_translate to .tif
            print(f'[warn] {name}: {len(flist)} tiles, mosaic them first')
            return None
        return flist[0]

    def tif_to_dict(self):
        n_years = self.end_year - self.start_year + 1
        for name in self.indices:
            fpath = self._find_file(name)
            if fpath is None:
                continue
            arr = self._read_stack(fpath)
            assert arr.shape[0] == n_years, f'{name}: {arr.shape[0]} bands, expected {n_years}'
            _, nr, nc = arr.shape
            dic = {}
            for r in range(nr):
                for c in range(nc):
                    v = arr[:, r, c]
                    if np.all(np.isnan(v)):
                        continue
                    dic[(r, c)] = v
            outf = os.path.join(self.outdir, f'{name}.npy')
            np.save(outf, dic)
            print(f'[ok] {name}: {len(dic)} pixels -> {outf}')

    def p95_to_dict(self):
        fpath = self._find_file('p95_baseline')
        if fpath is None:
            return
        arr = self._read_stack(fpath)[0]
        dic = {(r, c): arr[r, c] for r in range(arr.shape[0]) for c in range(arr.shape[1])
               if not np.isnan(arr[r, c])}
        np.save(os.path.join(self.outdir, 'p95_baseline_1982_1999.npy'), dic)


def main():
    Daymet_GEE().run()


if __name__ == '__main__':
    main()
