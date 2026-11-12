# coding='utf-8'
from __Global__ import *
import os
import time
from osgeo import gdal


class RAP_processing():
    ## RAP v3 vegetation cover (Rangeland Analysis Platform, 30 m, one Cloud Optimized GeoTIFF per year)
    ## read directly from the RAP server with GDAL /vsicurl/ (no full download), averaged to the 0.05 deg template grid
    ## does not use Google Earth Engine (GEE project is in restricted mode)
    ## cover types (% cover): AFG annual forbs and grasses, PFG perennial forbs and grasses, SHR shrubs, TRE trees,
    ##                        LTR litter, BGR bare ground
    ## RAP does not mask cropland: use our MCD12C1 cropland mask in the analysis
    def __init__(self):
        self.url = 'http://rangeland.ntsg.umt.edu/data/rap/rap-vegetation-cover/vegetation-cover-v3-{year}.tif'
        self.outdir = data_root + rf'RAP\tmpl\\'
        T.mk_dir(self.outdir, force=True)
        self.year_list = list(range(1986, 2025))
        ## template grid (same as LAI / SPEI / Daymet dicts): 449 cols x 468 rows, 0.05 deg
        self.x0 = -124.52916666666665
        self.y0 = 48.97916666666667
        self.res = 0.05
        self.ncol = 449
        self.nrow = 468
        self.bounds = [self.x0, self.y0 - self.nrow * self.res, self.x0 + self.ncol * self.res, self.y0]   ## xmin ymin xmax ymax
        ## GDAL settings for reading remote COGs
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')
        gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '5')
        gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '5')
        gdal.SetConfigOption('VSI_CACHE', 'TRUE')
        pass

    def run(self):
        ## step 1 (test): file info + one small window with and without the COG overviews
        self.check_file(1986)
        self.test_overview(1986)
        ## step 2: one full year, check the time and values
        # self.warp_year(1986, use_overview=False)
        # self.check_output(1986)
        ## step 3: all years
        # for year in self.year_list:
        #     self.warp_year(year, use_overview=False)
        pass

    def check_file(self, year):
        ## band names, data type, nodata, projection and pixel size of the remote file
        ds = gdal.Open('/vsicurl/' + self.url.format(year=year))
        if ds is None:
            print('cannot open', self.url.format(year=year))
            return
        print('size', ds.RasterXSize, ds.RasterYSize, 'bands', ds.RasterCount)
        print('geotransform', ds.GetGeoTransform())
        print('projection', ds.GetProjection()[:120])
        for i in range(1, ds.RasterCount + 1):
            b = ds.GetRasterBand(i)
            print(i, 'description:', b.GetDescription(), '| type:', gdal.GetDataTypeName(b.DataType),
                  '| nodata:', b.GetNoDataValue(), '| overviews:', b.GetOverviewCount())
        pass

    def warp(self, src, dst, bounds, use_overview):
        ## average of the 30 m pixels in each 0.05 deg cell; use_overview=False reads the full-resolution pixels
        opt = gdal.WarpOptions(
            outputBounds=bounds, xRes=self.res, yRes=self.res, dstSRS='EPSG:4326',
            resampleAlg='average', outputType=gdal.GDT_Float32, dstNodata=np.nan,
            overviewLevel='AUTO' if use_overview else 'NONE',
            multithread=True, warpMemoryLimit=2048,
            creationOptions=['COMPRESS=LZW', 'TILED=YES'],
        )
        gdal.Warp(dst, src, options=opt)

    def test_overview(self, year):
        ## 1 x 1 deg window in central Nevada (on the template grid): full resolution vs overviews, time and difference
        c0 = 150   ## template column of the window's west edge
        r0 = 170   ## template row of the window's north edge
        n = 20     ## 20 x 20 cells = 1 x 1 deg
        bounds = [self.x0 + c0 * self.res, self.y0 - (r0 + n) * self.res,
                  self.x0 + (c0 + n) * self.res, self.y0 - r0 * self.res]
        print('test window', np.round(bounds, 3))
        src = '/vsicurl/' + self.url.format(year=year)
        result = {}
        for use_overview in [False, True]:
            dst = self.outdir + f'test_{year}_overview_{use_overview}.tif'
            t0 = time.time()
            self.warp(src, dst, bounds, use_overview)
            arr = gdal.Open(dst).ReadAsArray()
            result[use_overview] = arr
            print('use_overview', use_overview, f'{time.time() - t0:.1f} s', 'band means', np.round(np.nanmean(arr, axis=(1, 2)), 2))
        diff = result[True] - result[False]
        print('overview - full: mean abs diff per band', np.round(np.nanmean(np.abs(diff), axis=(1, 2)), 3),
              'max abs diff', np.round(np.nanmax(np.abs(diff), axis=(1, 2)), 3))
        pass

    def warp_year(self, year, use_overview=False):
        dst = self.outdir + f'RAP_cover_{year}_tmpl.tif'
        if os.path.isfile(dst):
            print(year, 'exists')
            return
        t0 = time.time()
        self.warp('/vsicurl/' + self.url.format(year=year), dst, self.bounds, use_overview)
        print(year, f'done {time.time() - t0:.0f} s')

    def check_output(self, year):
        ## shape must be 6 x 468 x 449; value in central Nevada (-116.5, 40.5) should be close to the GEE test
        ## (SHR about 11% in 2000; 1986 will differ a little)
        ds = gdal.Open(self.outdir + f'RAP_cover_{year}_tmpl.tif')
        arr = ds.ReadAsArray()
        print('shape', arr.shape, 'geotransform', ds.GetGeoTransform())
        r = int((self.y0 - 40.5) / self.res)
        c = int((-116.5 - self.x0) / self.res)
        print('central Nevada', np.round(arr[:, r, c], 2))
        pass


class RAP_GEE_download():
    ## RAP v3 cover averaged to the 0.05 deg template grid ON GEE, downloaded straight to the local disk
    ## (same idea as gee_download_kabul.py: ee Python API, no Google Drive, no batch task queue)
    ## NOTE: the averaging still runs on GEE, so it still counts against the project's EECU quota (restricted mode
    ##   slows it down); it only skips the batch queue
    ## aggregation: 30 m -> 0.01 deg (mean) -> 0.05 deg (mean), exact mean of the 30 m pixels
    ## requests are 20 x 20 template cells (1 x 1 deg, 6 bands): tested 2026-11-12 in the Code Editor, about 7 s each
    ##   and within the interactive memory limit (larger tiles ran out of memory)
    ## output: Data\RAP\tmpl\RAP_cover_{year}_tmpl.tif, 6 bands in the order of self.band_list, NaN outside RAP
    def __init__(self):
        self.ee_project = 'starproject-487101'
        self.asset = 'projects/rap-data-365417/assets/vegetation-cover-v3'
        self.band_list = ['AFG', 'PFG', 'SHR', 'TRE', 'LTR', 'BGR']
        self.outdir = data_root + rf'RAP\tmpl\\'
        T.mk_dir(self.outdir, force=True)
        self.year_list = list(range(1986, 2025))
        ## template grid
        self.x0 = -124.52916666666665
        self.y0 = 48.97916666666667
        self.res = 0.05
        self.fine = 0.01
        self.ncol = 449
        self.nrow = 468
        self.tile = 20            ## template cells per request side
        self.n_thread = 2         ## parallel requests (restricted mode: 8 hit 'Too Many Requests', 2026-11-12)
        self.only_study_box = True   ## only tiles touching lon -125 to -105, lat 30 to 45 (saves about 30% of the quota)
        pass

    def run(self):
        import ee
        try:
            ee.Initialize(project=self.ee_project)
        except Exception:
            ee.Authenticate()   ## first time only: opens a browser, log in with the account that owns the project
            ee.Initialize(project=self.ee_project)
        ## test one year first, then switch to the full loop
        # self.download_year(2000)
        # self.check_output(2000)
        for year in self.year_list:
            self.download_year(year)
        pass

    def year_image(self, year):
        import ee
        img = ee.ImageCollection(self.asset).filter(ee.Filter.eq('year', year)).first()
        img = ee.Image(img).select(self.band_list).toFloat()
        native = img.projection()
        fine = img.setDefaultProjection(native) \
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=4096) \
            .reproject(crs='EPSG:4326', crsTransform=[self.fine, 0, self.x0, 0, -self.fine, self.y0])
        out = fine.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=4096) \
            .reproject(crs='EPSG:4326', crsTransform=[self.res, 0, self.x0, 0, -self.res, self.y0])
        return out.unmask(-9999)   ## no RAP data (ocean, Mexico) -> -9999, set to NaN after download

    def tile_list(self):
        ## (row0, col0, height, width) of each request on the template grid
        nrow, ncol = self.nrow, self.ncol
        if self.only_study_box:
            ncol = int(np.ceil((-105 - self.x0) / self.res))   ## east edge of the study box
            nrow = int(np.ceil((self.y0 - 30) / self.res))     ## south edge of the study box
        result = []
        for r0 in range(0, nrow, self.tile):
            for c0 in range(0, ncol, self.tile):
                result.append((r0, c0, min(self.tile, nrow - r0), min(self.tile, ncol - c0)))
        return result

    def get_tile(self, image, r0, c0, h, w):
        import ee
        request = {
            'expression': image,
            'fileFormat': 'NUMPY_NDARRAY',
            'grid': {
                'dimensions': {'width': w, 'height': h},
                'affineTransform': {'scaleX': self.res, 'shearX': 0, 'translateX': self.x0 + c0 * self.res,
                                    'shearY': 0, 'scaleY': -self.res, 'translateY': self.y0 - r0 * self.res},
                'crsCode': 'EPSG:4326',
            },
        }
        for attempt in range(10):
            try:
                arr = ee.data.computePixels(request)
                return np.stack([arr[b] for b in self.band_list]).astype(np.float32)   ## band x h x w
            except Exception as e:
                msg = str(e)
                if 'Too Many Requests' in msg or 'concurrency' in msg:
                    time.sleep(30)   ## concurrency limit: wait and retry, not a real failure
                    continue
                print(f'  tile ({r0}, {c0}) attempt {attempt + 1}: {msg[:120]}')
                time.sleep(10 * (attempt + 1))
        return None

    def download_year(self, year):
        from concurrent.futures import ThreadPoolExecutor
        dst = self.outdir + f'RAP_cover_{year}_tmpl.tif'
        if os.path.isfile(dst):
            print(year, 'exists')
            return
        image = self.year_image(year)
        tiles = self.tile_list()
        out = np.full((len(self.band_list), self.nrow, self.ncol), np.nan, dtype=np.float32)
        failed = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.n_thread) as pool:
            futures = {pool.submit(self.get_tile, image, *t): t for t in tiles}
            for i, f in enumerate(futures):
                r0, c0, h, w = futures[f]
                arr = f.result()
                if arr is None:
                    failed.append((r0, c0, h, w))
                    continue
                out[:, r0:r0 + h, c0:c0 + w] = arr
                if (i + 1) % 50 == 0:
                    print(f'  {year}: {i + 1}/{len(tiles)} tiles, {time.time() - t0:.0f} s')
        if len(failed) > 0:
            print(year, 'FAILED tiles, file not written:', failed)
            return
        out[out <= -9999] = np.nan
        self.write_tif(out, dst)
        print(year, f'done {time.time() - t0:.0f} s')

    def write_tif(self, arr, dst):
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(dst, self.ncol, self.nrow, arr.shape[0], gdal.GDT_Float32, options=['COMPRESS=LZW'])
        ds.SetGeoTransform([self.x0, self.res, 0, self.y0, 0, -self.res])
        ds.SetProjection('EPSG:4326')
        for i, b in enumerate(self.band_list):
            band = ds.GetRasterBand(i + 1)
            band.WriteArray(arr[i])
            band.SetDescription(b)
            band.SetNoDataValue(np.nan)
        ds = None

    def check_output(self, year):
        ## central Nevada (-116.5, 40.5) in 2000 should be SHR about 11.5% (exact GEE test, 2026-11-12)
        ds = gdal.Open(self.outdir + f'RAP_cover_{year}_tmpl.tif')
        arr = ds.ReadAsArray()
        r = int((self.y0 - 40.5) / self.res)
        c = int((-116.5 - self.x0) / self.res)
        print('shape', arr.shape, '| central Nevada', dict(zip(self.band_list, np.round(arr[:, r, c], 2))))
        print('valid cells per band', np.sum(np.isfinite(arr), axis=(1, 2)))
        pass


def main():
    # RAP_processing().run()   ## alternative without GEE: read the RAP server files with GDAL
    RAP_GEE_download().run()


if __name__ == '__main__':
    main()
