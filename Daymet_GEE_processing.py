# coding='utf-8'
from __Global__ import *
import os
import glob
from osgeo import gdal


class Daymet_GEE_seasonal():
    ## seasonal Daymet precipitation indices from GEE (coworkplayground\scripts\gee\gee_daymet_precip_indices_seasonal.js)
    ## seasons follow the dataframe definition:
    ##   summer = JASO (Jul-Oct), spring = MAM (Mar-May),
    ##   winter = NDJF (Nov of the previous year - Feb), labelled with the following year
    ## each tif: 43 bands = years 1982-2024; GEE exports 469 rows (origin 49.029) -> cut to the template 468 rows (origin 48.979)
    ## output: one dict per season and index  {pix: 43 values}, and one dataframe (pix, year, lon, lat, <season>_<index> ...)
    def __init__(self):
        self.datadir = data_root + rf'Daymet_GEE\\'
        self.outdir = this_root + rf'Result2\00_aligned\Daymet_GEE\\'
        T.mk_dir(self.outdir, force=True)
        self.year_list = list(range(1982, 2025))
        self.tag_dic = {'summer': 'JASO', 'spring': 'MAM', 'winter': 'NDJF'}
        self.ylabel_dic = {'PRCPTOT': 'PRCPTOT (mm)', 'WETDAYS': 'WETDAYS (days)', 'R5mm': 'R5mm (days)',
                           'LIGHT': 'LIGHT (days)', 'Rx1day': 'Rx1day (mm)', 'SDII': 'SDII (mm/day)',
                           'R95p': 'R95p (mm)', 'R95pTOT': 'R95pTOT (%)', 'CDD': 'CDD (days)',
                           'SWE_max': 'SWE_max (mm)', 'P_LIGHT': 'P_LIGHT (mm)', 'P_MOD': 'P_MOD (mm)'}
        self.index_dic = {
            ## P_LIGHT = rain from 1-5 mm days, P_MOD = rain from 5 mm - p95 days (added 2026-11-11 for LAI_precip_regression)
            'summer': ['PRCPTOT', 'WETDAYS', 'R5mm', 'LIGHT', 'Rx1day', 'SDII', 'R95p', 'R95pTOT', 'CDD', 'P_LIGHT', 'P_MOD'],
            'spring': ['PRCPTOT', 'WETDAYS', 'R5mm', 'LIGHT', 'Rx1day', 'SDII', 'R95p', 'R95pTOT', 'CDD', 'P_LIGHT', 'P_MOD'],
            'winter': ['PRCPTOT', 'SWE_max'],
        }
        ## template grid (same as LAI / SPEI dicts)
        self.x0 = -124.52916666666665
        self.y0 = 48.97916666666667
        self.res = 0.05
        self.nrow = 468
        self.trend_period_list = [(1982, 2024), (2000, 2024)]
        pass

    def run(self):
        self.tif_to_dict()   ## re-run 2026-11-12 to add P_LIGHT and P_MOD
        df = self.dict_to_df()
        df_dir = result_root + rf'\Dataframe\Daymet\\'
        T.mk_dir(df_dir, force=True)
        T.save_df(df, df_dir + 'Daymet_seasonal.df')
        T.df_to_excel(df.head(2000), df_dir + 'Daymet_seasonal_head')   ## first rows only, for checking
        self.check_bins(df)
        self.trend_map()   ## already done
        pass

    def check_bins(self, df):
        ## P_LIGHT + P_MOD + R95p should equal PRCPTOT minus the rain from days < 1 mm (a few mm at most)
        for season in ['summer', 'spring']:
            total = df[f'{season}_PRCPTOT']
            bins = df[f'{season}_P_LIGHT'] + df[f'{season}_P_MOD'] + df[f'{season}_R95p']
            diff = total - bins
            print(season, 'mean PRCPTOT', round(total.mean(), 1), 'mean sum of bins', round(bins.mean(), 1),
                  'PRCPTOT - bins: median', round(diff.median(), 2), 'min', round(diff.min(), 2),
                  'max', round(diff.max(), 2))

    def find_tif(self, season, index):
        ## files may sit in sub-folders after unzipping
        tag = self.tag_dic[season]
        flist = glob.glob(self.datadir + rf'**\daymet_{tag}_{index}_1982_2024_tmpl*.tif', recursive=True)
        if len(flist) != 1:
            print(f'[check] {season} {index}: {len(flist)} files found')
            return None
        return flist[0]

    def read_tif(self, fpath):
        ds = gdal.Open(fpath)
        arr = ds.ReadAsArray().astype(np.float32)   ## (band, row, col)
        origin_y = ds.GetGeoTransform()[3]
        row0 = int(round((origin_y - self.y0) / self.res))   ## 1 for the 469-row GEE exports
        arr = arr[:, row0:row0 + self.nrow, :]
        return arr

    def tif_to_dict(self):
        for season in self.index_dic:
            for index in self.index_dic[season]:
                fpath = self.find_tif(season, index)
                if fpath is None:
                    continue
                arr = self.read_tif(fpath)
                assert arr.shape[0] == len(self.year_list), f'{season} {index}: {arr.shape[0]} bands'
                spatial_dic = {}
                for r in range(arr.shape[1]):
                    for c in range(arr.shape[2]):
                        vals = arr[:, r, c]
                        if np.all(np.isnan(vals)):
                            continue
                        spatial_dic[(r, c)] = vals
                np.save(self.outdir + f'{season}_{index}.npy', spatial_dic)
                print(season, index, arr.shape, 'done')
        pass

    def dict_to_df(self):
        ## land mask: pixels with summer precipitation > 0 in at least one year (GEE writes 0 over the ocean)
        land_dic = T.load_npy(self.outdir + 'summer_PRCPTOT.npy')
        pix_list = [pix for pix in land_dic if np.nansum(land_dic[pix]) > 0]
        ## study region, same as df_clean in the analysis scripts
        pix_region = []
        for pix in pix_list:
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
        for season in self.index_dic:
            for index in self.index_dic[season]:
                f = self.outdir + f'{season}_{index}.npy'
                if not os.path.isfile(f):
                    continue
                spatial_dic = T.load_npy(f)
                vals = np.concatenate([spatial_dic.get(p, np.full(n_year, np.nan)) for p, _, _ in pix_region])
                df[f'{season}_{index}'] = vals.astype(np.float32)
                print(season, index, 'added to df')
        return df


    def ols_rows(self, x, y):
        ## linear trend of each row of y (pixels x years) against x (years): slope and two-sided p
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

    def trend_map(self):
        ## per-pixel linear trend (per decade) of every seasonal index, 1982-2024 and 2000-2024
        ## output: slope and p tifs + map in the SPEI_LAI_correlation style (cartopy, state borders, study box, histogram)
        from SPEI_LAI_correlation import SPEI_SNU_LAI_correlation, D
        outdir = self.outdir + rf'trend\\'
        T.mk_dir(outdir, force=True)
        mapper = SPEI_SNU_LAI_correlation()
        mapper.outdir = outdir   ## plot_map saves into mapper.outdir
        years = np.array(self.year_list)
        land_dic = T.load_npy(self.outdir + 'summer_PRCPTOT.npy')   ## GEE writes 0 over the ocean
        land_pix = set(pix for pix in land_dic if np.nansum(land_dic[pix]) > 0 and np.nanmin(land_dic[pix]) > -9999)
        for season in self.index_dic:
            for index in self.index_dic[season]:
                f = self.outdir + f'{season}_{index}.npy'
                if not os.path.isfile(f):
                    continue
                spatial_dic = T.load_npy(f)
                pix_list = [pix for pix in spatial_dic if pix in land_pix]
                arr = np.array([spatial_dic[pix] for pix in pix_list], dtype=float)   ## pixels x 43 years
                lat = np.array([self.y0 - self.res * (pix[0] + 0.5) for pix in pix_list])
                lon = np.array([self.x0 + self.res * (pix[1] + 0.5) for pix in pix_list])
                area_w = np.cos(np.deg2rad(lat))
                in_box = (lon > -125) & (lon < -105) & (lat > 30) & (lat < 45)   ## study region
                for start, end in self.trend_period_list:
                    m = (years >= start) & (years <= end)
                    slope, p = self.ols_rows(years[m].astype(float), arr[:, m])
                    slope = slope * 10   ## per decade
                    slope_dic = dict(zip(pix_list, slope))
                    p_dic = dict(zip(pix_list, p))
                    fname = f'{season}_{index}_{start}_{end}'
                    arr_slope = D.pix_dic_to_spatial_arr(slope_dic)
                    arr_slope[arr_slope <= -9999] = np.nan   ## pixels not in the dict (ocean) are filled with -9999
                    arr_p = D.pix_dic_to_spatial_arr(p_dic)
                    arr_p[arr_p <= -9999] = np.nan
                    ## arr_to_tif writes NaN back as -999999 IN PLACE, so save copies (otherwise the ocean is plotted dark red)
                    D.arr_to_tif(arr_slope.copy(), outdir + f'{fname}_slope.tif')
                    D.arr_to_tif(arr_p.copy(), outdir + f'{fname}_p.tif')
                    ## symmetric colour range from the 90th percentile of |slope| inside the study region
                    ## (Mexico outside the box has much larger values and would wash out the colours)
                    vlim = np.nanpercentile(np.abs(slope[in_box]), 90)
                    cmap = 'RdBu_r' if index == 'CDD' else 'RdBu'   ## blue = wetter (fewer dry days for CDD)
                    mapper.plot_map(arr_slope, fname, cmap, -vlim, vlim,
                                    f'{season.capitalize()} {self.ylabel_dic[index]} trend (per decade)',
                                    slope[in_box], area_w[in_box])
                    sig = np.nansum(area_w[in_box] * (p[in_box] < 0.05)) / np.nansum(area_w[in_box]) * 100
                    print(fname, f'significant area (p<0.05): {sig:.1f}%')
        pass


def main():
    Daymet_GEE_seasonal().run()


if __name__ == '__main__':
    main()
