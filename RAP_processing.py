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


class RAP_dict_map():
    ## RAP cover (Data\RAP\tmpl\RAP_cover_{year}_tmpl.tif, 6 bands, 1986-2024) -> dicts, dataframe, spatial change maps
    ## dicts: Result2\00_aligned\RAP\{band}.npy   {pix: 39 values}
    ## dataframe: Result\Dataframe\RAP\RAP_cover.df   (pix, year, lon, lat, RAP_AFG ... RAP_BGR), study region only
    ## maps (Result2\00_aligned\RAP\map\), from RAP_cover.df, cropland_pct > 50% (MCD12C1) left blank:
    ##   period mean difference (2000-2024) - (1986-1999), percentage points
    ##   linear trend per decade, 1986-2024 and 2000-2024
    def __init__(self):
        self.datadir = data_root + rf'RAP\tmpl\\'
        self.outdir = this_root + rf'Result2\00_aligned\RAP\\'
        self.df_dir = result_root + rf'\Dataframe\RAP\\'
        T.mk_dir(self.outdir, force=True)
        T.mk_dir(self.df_dir, force=True)
        self.band_list = ['AFG', 'PFG', 'SHR', 'TRE', 'LTR', 'BGR']
        self.name_dic = {'AFG': 'Annual forbs and grasses', 'PFG': 'Perennial forbs and grasses', 'SHR': 'Shrubs',
                         'TRE': 'Trees', 'LTR': 'Litter', 'BGR': 'Bare ground'}
        self.year_list = list(range(1986, 2025))
        self.pre, self.post = (1986, 1999), (2000, 2024)
        self.trend_period_list = [(1986, 2024), (2000, 2024)]
        self.crop_threshold = 50
        self.x0 = -124.52916666666665
        self.y0 = 48.97916666666667
        self.res = 0.05
        pass

    def run(self):
        # self.tif_to_dict()   ## done 2026-11-14
        # df = self.dict_to_df()   ## done; do not rerun: Build_framework added cropland_pct and landcover_type to this df
        # T.save_df(df, self.df_dir + 'RAP_cover.df')
        # T.df_to_excel(df.head(2000), self.df_dir + 'RAP_cover_head')
        self.change_map()
        pass

    def tif_to_dict(self):
        stack = []
        for year in self.year_list:
            ds = gdal.Open(self.datadir + f'RAP_cover_{year}_tmpl.tif')
            stack.append(ds.ReadAsArray().astype(np.float32))   ## band x row x col
        stack = np.array(stack)   ## year x band x row x col
        for i, b in enumerate(self.band_list):
            arr = stack[:, i]
            spatial_dic = {}
            for r in range(arr.shape[1]):
                for c in range(arr.shape[2]):
                    vals = arr[:, r, c]
                    if np.all(np.isnan(vals)):
                        continue
                    spatial_dic[(r, c)] = vals
            np.save(self.outdir + f'{b}.npy', spatial_dic)
            print(b, len(spatial_dic), 'pixels')
        pass

    def dict_to_df(self):
        dic_all = {b: T.load_npy(self.outdir + f'{b}.npy') for b in self.band_list}
        pix_region = []
        for pix in dic_all['SHR']:
            r, c = pix
            lon = self.x0 + self.res * (c + 0.5)
            lat = self.y0 - self.res * (r + 0.5)
            if -125 < lon < -105 and 30 < lat < 45:
                pix_region.append((pix, lon, lat))
        n_year = len(self.year_list)
        df = pd.DataFrame({
            'pix': [p for p, _, _ in pix_region for _ in range(n_year)],
            'year': self.year_list * len(pix_region),
            'lon': np.repeat([lon for _, lon, _ in pix_region], n_year),
            'lat': np.repeat([lat for _, _, lat in pix_region], n_year),
        })
        for b in self.band_list:
            df[f'RAP_{b}'] = np.concatenate([dic_all[b][p] for p, _, _ in pix_region]).astype(np.float32)
        print('df', df.shape, 'pixels', len(pix_region))
        return df

    def ols_rows(self, x, y):
        from scipy import stats
        n = len(x)
        xm = x - x.mean()
        ym = y - y.mean(axis=1, keepdims=True)
        slope = ym @ xm / np.sum(xm ** 2)
        resid = ym - slope[:, None] * xm[None, :]
        se = np.sqrt(np.sum(resid ** 2, axis=1) / (n - 2) / np.sum(xm ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            t = slope / se
        p = 2 * stats.t.sf(np.abs(t), n - 2)
        return slope, p

    def change_map(self):
        from SPEI_LAI_correlation import SPEI_SNU_LAI_correlation, D
        mapdir = self.outdir + rf'map\\'
        T.mk_dir(mapdir, force=True)
        mapper = SPEI_SNU_LAI_correlation()
        mapper.outdir = mapdir   ## plot_map saves into mapper.outdir
        years = np.array(self.year_list)
        ## RAP_cover.df with cropland_pct from Build_framework.add_MODIS_LUCC_to_df; remove cropland > 50%
        df = T.load_df(self.df_dir + 'RAP_cover.df')
        n_pix_all = df['pix'].nunique()
        df = df[df['cropland_pct'] <= self.crop_threshold]   ## NaN cropland_pct (no MODIS land cover) also removed
        print('pixels:', n_pix_all, '-> after removing cropland >', self.crop_threshold, '%:', df['pix'].nunique())
        for b in self.band_list:
            mat = df.pivot(index='pix', columns='year', values=f'RAP_{b}').reindex(columns=self.year_list)
            mat = mat[mat.notna().all(axis=1)]
            pix_list = list(mat.index)
            arr = mat.values.astype(float)   ## pixels x 39 years
            lat = np.array([self.y0 - self.res * (pix[0] + 0.5) for pix in pix_list])
            lon = np.array([self.x0 + self.res * (pix[1] + 0.5) for pix in pix_list])
            area_w = np.cos(np.deg2rad(lat))
            in_box = (lon > -125) & (lon < -105) & (lat > 30) & (lat < 45)
            ## 1) period mean difference
            m_pre = (years >= self.pre[0]) & (years <= self.pre[1])
            m_post = (years >= self.post[0]) & (years <= self.post[1])
            diff = arr[:, m_post].mean(axis=1) - arr[:, m_pre].mean(axis=1)
            fname = f'{b}_diff_{self.post[0]}_{self.post[1]}_minus_{self.pre[0]}_{self.pre[1]}'
            self.save_and_plot(mapper, D, pix_list, diff, None, fname, in_box, area_w,
                               f'{b} cover change (percentage points)')
            ## 2) linear trends
            for start, end in self.trend_period_list:
                m = (years >= start) & (years <= end)
                slope, p = self.ols_rows(years[m].astype(float), arr[:, m])
                fname = f'{b}_trend_{start}_{end}'
                self.save_and_plot(mapper, D, pix_list, slope * 10, p, fname, in_box, area_w,
                                   f'{b} cover trend (percentage points per decade)')
        pass

    def save_and_plot(self, mapper, D, pix_list, vals, p, fname, in_box, area_w, cbar_label):
        arr = D.pix_dic_to_spatial_arr(dict(zip(pix_list, vals)))
        arr[arr <= -9999] = np.nan   ## pixels not in the dict are filled with -9999
        D.arr_to_tif(arr.copy(), self.outdir + rf'map\\' + f'{fname}.tif')   ## copy: arr_to_tif changes NaN in place
        vlim = np.nanpercentile(np.abs(vals[in_box]), 90)
        mapper.plot_map(arr, fname, 'RdBu', -vlim, vlim, cbar_label, vals[in_box], area_w[in_box])
        w = area_w[in_box]
        v = vals[in_box]
        line = f'{fname}: area-weighted mean {np.sum(w * v) / np.sum(w):+.2f}, area > +5: {np.sum(w * (v > 5)) / np.sum(w) * 100:.1f}%, ' \
               f'area < -5: {np.sum(w * (v < -5)) / np.sum(w) * 100:.1f}%'
        if p is not None:
            pin = p[in_box]
            line += f', significant increase {np.sum(w * ((v > 0) & (pin < 0.05))) / np.sum(w) * 100:.1f}%, ' \
                    f'significant decrease {np.sum(w * ((v < 0) & (pin < 0.05))) / np.sum(w) * 100:.1f}%'
        print(line)


def main():
    # RAP_processing().run()   ## alternative without GEE: read the RAP server files with GDAL
    # RAP_GEE_download().run()   ## done 2026-11-13: 1986-2024 in Data\RAP\tmpl
    RAP_dict_map().run()


if __name__ == '__main__':
    main()
