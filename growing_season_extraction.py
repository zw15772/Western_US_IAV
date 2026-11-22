# coding='utf-8'
from __Global__ import *
from scipy import stats
tif_template = data_root + rf'basedata\200902.tif'   ## 0.05 deg template grid, 449 x 468
D = DIC_and_TIF(tif_template=tif_template)


class Map_plotter():
    ## map style of the project (same as plot_map in SPEI_LAI_correlation.py): cartopy, state lines, study box,
    ##   area-weighted histogram inset; saves {outdir}{fname}_map.png / .pdf
    def __init__(self, outdir):
        self.outdir = outdir

    def plot_map(self, arr, fname, cmap, vmin, vmax, cbar_label, hist_vals, hist_w, discrete_levels=None):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patches as mpatches
        from matplotlib.colors import BoundaryNorm, Normalize
        import matplotlib.ticker as mticker
        if discrete_levels is not None:
            levels = np.array(discrete_levels)
            bounds = np.concatenate([[levels[0] - 1], (levels[:-1] + levels[1:]) / 2, [levels[-1] + 1]])
            cmap_obj = plt.get_cmap(cmap, len(levels))
            norm = BoundaryNorm(bounds, cmap_obj.N)
        else:
            cmap_obj = plt.get_cmap(cmap)
            norm = Normalize(vmin=vmin, vmax=vmax)
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.imshow(arr, cmap=cmap_obj, norm=norm, extent=[-124.529, -102.079, 25.579, 48.979],
                       transform=ccrs.PlateCarree(), interpolation='nearest')   ## template grid: 449 x 468, 0.05 deg
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_patch(mpatches.Rectangle((-125, 30), 20, 15, linewidth=1.5, edgecolor='black', facecolor='none',
                                        transform=ccrs.PlateCarree(), zorder=10))   ## study region box
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(range(-125, -100, 5))
        gl.ylocator = mticker.FixedLocator(range(25, 50, 5))
        if discrete_levels is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=levels, spacing='uniform')
            cbar.ax.tick_params(labelsize=6)
        else:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label)
        ## histogram inset (area-weighted %), lower-left over the Pacific
        ax_in = ax.inset_axes([0.10, 0.05, 0.30, 0.25])
        ok = np.isfinite(hist_vals) & np.isfinite(hist_w)
        vals, w = hist_vals[ok], hist_w[ok] / np.sum(hist_w[ok]) * 100
        if discrete_levels is not None:
            area = [np.sum(w[vals == lev]) for lev in levels]
            ax_in.bar(np.arange(len(levels)), area, color=[cmap_obj(i) for i in range(len(levels))], width=0.9)
            ax_in.set_xticks([0, len(levels) // 2, len(levels) - 1])
            ax_in.set_xticklabels([levels[0], levels[len(levels) // 2], levels[-1]])
        else:
            bins = np.linspace(vmin, vmax, 41)
            area, edges = np.histogram(vals, bins=bins, weights=w)
            centers = (edges[:-1] + edges[1:]) / 2
            ax_in.bar(centers, area, width=edges[1] - edges[0], color=cmap_obj(norm(centers)))
            ax_in.axvline(0, color='grey', lw=0.5)
        ax_in.set_title('Area (%)', fontsize=6, pad=2)
        ax_in.tick_params(labelsize=6)
        ax_in.set_zorder(20)
        ax_in.patch.set_facecolor('white')
        ax_in.patch.set_alpha(1)
        plt.title(fname)
        plt.savefig(self.outdir + f'{fname}_map.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'{fname}_map.pdf', bbox_inches='tight')
        plt.close()


class Growing_season_extraction():
    ## Growing season of every pixel, and LAI / SPEI / climate of that growing season in each year (2026-11-15)
    ## pixels: template pixels in the study box (lon -125 to -105, lat 30 to 45) with MODIS LAI, cropland <= 50%
    ##   (MCD12C1 multi-year mean, IGBP 12 + 14), same rules as Build_framework / df_clean; years 2003-2024
    ## growing season, from the 2003-2024 mean of each calendar month of MODIS LAI (12 values, treated as a circle):
    ##   main season: around the highest month; neighbouring months are added while LAI keeps falling and stays
    ##     above min + 50% x (peak - min)  (half of the seasonal amplitude)
    ##   second season: the most prominent other peak, if its prominence (height above the higher of the two lows
    ##     that separate it from the main peak) is >= 30% of the amplitude and it lasts >= 2 months
    ##     (one-month bumps in Dec / Jan, mostly north of 38 N, are taken as winter noise)
    ##   peaks in Nov, Dec or Jan are not growing seasons (winter noise of the unfiltered monthly maximum, mostly
    ##     snow areas north of 38 N): the main peak and the second peak are looked for in Feb-Oct only
    ##   season 1 / season 2 = calendar order of the two peaks (two-season pixels only)
    ##   the analysis uses the MAIN season: single-season pixels: their season; two-season pixels: the season with
    ##   the higher peak. The window is the same months every year; windows across the new year are left out.
    ## variables of each year (window = the pixel's main season):
    ##   GS_LAI        MODIS LAI, mean of the window months (all months needed), m2/m2
    ##                 (monthly max of MCD15A3H 4-day, Data\MODIS_LAI_0706\dic, Oct 2022 fixed)
    ##   GS_LAI_rel    % difference from the pixel's 2003-2024 mean of GS_LAI
    ##   GS_SPEI03 ... GS_SPEI48   mean of the monthly SPEI in the window (TerraClimate SPEI, NOAA method)
    ##   GS_P          precipitation sum in the window (mm);  GS_P_antecedent  sum in the 8 months before it
    ##   GS_VPD, GS_Tmax   mean in the window (kPa, deg C)  (TerraClimate monthly)
    ## per pixel: gs_start, gs_end, gs_peak, gs_length, n_season, s1_*/s2_* (months 1-12), GS_mean_LAI,
    ##   monsoon_share (Jul-Sep P / annual P, TerraClimate, 2003-2024)
    ## pixels: study box, cropland <= threshold, US only (cell centre inside the US, Natural Earth 50 m states)
    ## 2026-11-21: + Annual NLCD cropland mask: pixels with >= 5% cropland (81 pasture/hay + 82 crops, max of 2003,
    ##   2004, 2024; Data_processing_NLCD_cropland) are removed. MCD12C1 <= 50% alone kept pixels with centre-pivot
    ##   fields (NAIP site G08). Pixels without NLCD are kept.
    ## per pixel also: lon, lat (cell centre), area_weight = cos(lat), cropland_pct, IGBP_majority, landcover_type
    ##   (Forest 1-5, Shrub 6-7, Savanna 8-9, Grass 10, Cropland 12/14, Other), from the MCD12C1 template tifs
    ## per year also: RAP_AFG / PFG / SHR / TRE from Result\Dataframe\RAP\RAP_cover.df (if the file exists)
    ## outputs: Result\Dataframe\growing_season\GS.df (pixel-year), growing_season.df (pixel, with the 12 monthly
    ##   means clim_01 ... clim_12); maps and tables of the seasons: Growing_season_definition
    def __init__(self):
        self.rap_dff = result_root + rf'\Dataframe\RAP\RAP_cover.df'
        self.mcd12_dir = data_root + rf'\basedata\MCD12\\'
        self.crop_threshold = 50
        self.nlcd_crop_dff = result_root + rf'\Dataframe\NLCD\NLCD_cropland.df'
        self.nlcd_crop_threshold = 0.05   ## remove pixels with NLCD cropland share >= 5% (2026-11-21)
        self.x0, self.y0, self.res, self.nrow, self.ncol = -124.52916666666665, 48.97916666666667, 0.05, 468, 449
        self.df_dir = result_root + rf'\Dataframe\growing_season\\'
        self.dff = self.df_dir + 'GS.df'
        T.mk_dir(self.df_dir, force=True)
        self.year_list = list(range(2003, 2025))
        self.spei_scale_list = list(range(3, 49, 3))
        self.prom_ratio = 0.3
        self.half = 0.5
        self.min_len_second = 2
        self.n_antecedent = 8
        self.LAI_threshold = 0.1
        self.rap_col_list = ['RAP_AFG', 'RAP_PFG', 'RAP_SHR', 'RAP_TRE']
        pass

    def run(self):
        pix_info = self.pixel_info()
        pix_list = pix_info.index
        ## growing seasons
        lai_month = self.monthly_values(pix_list, data_root + rf'MODIS_LAI_0706\dic\\', 2003)
        clim = np.nanmean(lai_month[:, 12:].reshape(len(pix_list), -1, 12), axis=1)   ## pixel x 12, 2003-2024
        gs = pd.DataFrame([self.define(c) for c in clim], index=pix_list)
        gs['gs_length'] = (gs['gs_end'] - gs['gs_start']) % 12 + 1
        use = (gs['n_season'].values > 0) & ~(gs['gs_start'] > gs['gs_end']).values
        print('pixels:', len(gs), '| no season:', (gs['n_season'] == 0).sum(), '| two seasons:',
              (gs['n_season'] == 2).sum(), '| main window across the new year (left out):',
              ((gs['n_season'] > 0) & (gs['gs_start'] > gs['gs_end'])).sum())
        start = np.where(use, gs['gs_start'].values - 1, 0).astype(int)
        length = np.where(use, gs['gs_length'].values, 1).astype(int)
        ## yearly variables of the main season
        var_dic = {}
        lai = self.window(lai_month, start, length, 'mean')
        lai[~use] = np.nan
        var_dic['GS_LAI'] = lai
        gs['GS_mean_LAI'] = np.nanmean(lai, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            var_dic['GS_LAI_rel'] = (lai / gs['GS_mean_LAI'].values[:, None] - 1) * 100
        ppt = self.monthly_values(pix_list, data_root + rf'TerraClimate\ppt\dic\\', None)
        var_dic['GS_P'] = self.window(ppt, start, length, 'sum')
        var_dic['GS_P_antecedent'] = self.window(ppt, start - self.n_antecedent, np.full(len(start), self.n_antecedent),
                                                 'sum')
        p_2003 = ppt[:, 12:].reshape(len(pix_list), -1, 12)
        gs['monsoon_share'] = p_2003[:, :, 6:9].sum(axis=2).mean(axis=1) / p_2003.sum(axis=2).mean(axis=1)
        del ppt
        for var, name in [('vpd', 'GS_VPD'), ('tmax', 'GS_Tmax')]:
            arr = self.monthly_values(pix_list, data_root + rf'TerraClimate\{var}\dic\\', None)
            var_dic[name] = self.window(arr, start, length, 'mean')
        for scale in self.spei_scale_list:
            arr = self.monthly_values(pix_list, data_root + rf'TerraClimate\SPEI\SPEI_{scale}_NOAA\dic\\', None)
            var_dic[f'GS_SPEI{scale:02d}'] = self.window(arr, start, length, 'mean')
        for name in var_dic:
            if name != 'GS_LAI' and name != 'GS_LAI_rel':
                var_dic[name][~use] = np.nan
        ## pixel-year table
        n_year = len(self.year_list)
        df = pd.DataFrame({'pix': [p for p in pix_list for _ in range(n_year)], 'year': np.tile(self.year_list, len(pix_list))})
        for name, arr in var_dic.items():
            df[name] = arr.ravel()
        df = df.merge(pix_info, left_on='pix', right_index=True, how='left')
        df = df.merge(gs, left_on='pix', right_index=True, how='left')
        if os.path.isfile(self.rap_dff):
            df_rap = T.load_df(self.rap_dff)
            rap_cols = [c for c in self.rap_col_list if c in df_rap.columns]
            df = df.merge(df_rap[['pix', 'year'] + rap_cols], on=['pix', 'year'], how='left')
        df = df[df['n_season'] > 0]   ## pixels without MODIS LAI or without a season
        for name in var_dic:
            print(name, 'non-NaN rows:', df[name].notna().sum(), '/', len(df))
        T.save_df(df, self.dff)
        T.df_to_excel(df.head(2000), self.df_dir + 'GS_head')   ## first rows only, for checking
        for k in range(12):   ## 2003-2024 mean LAI of each calendar month (used by Growing_season_definition)
            gs[f'clim_{k + 1:02d}'] = clim[:, k]
        T.save_df(gs.join(pix_info), self.df_dir + 'growing_season.df')
        pass

    def pixel_info(self):
        ## study-box pixels of the template grid with cropland <= threshold; land cover from MCD12C1
        from osgeo import gdal

        def read(fname):
            ds = gdal.Open(self.mcd12_dir + fname)
            arr = ds.ReadAsArray().astype(float)
            ## GEE exported one extra row on top (origin 49.029); cut rows to match the template (origin 48.979)
            row0 = int(round((ds.GetGeoTransform()[3] - self.y0) / self.res))
            return arr[..., row0:row0 + self.nrow, :]

        pct = read('MCD12C1_IGBP_percent_mean_tmpl.tif')       ## 17 x rows x cols, % cover per IGBP class
        major = read('MCD12C1_IGBP_majority_mode_tmpl.tif')    ## rows x cols, IGBP class
        crop = pct[12] + pct[14]
        crop[(crop < 0) | (crop > 100)] = np.nan
        type_dic = {1: 'Forest', 2: 'Forest', 3: 'Forest', 4: 'Forest', 5: 'Forest', 6: 'Shrub', 7: 'Shrub',
                    8: 'Savanna', 9: 'Savanna', 10: 'Grass', 12: 'Cropland', 14: 'Cropland'}
        rows, cols = np.meshgrid(np.arange(self.nrow), np.arange(self.ncol), indexing='ij')
        lon = self.x0 + (cols + 0.5) * self.res
        lat = self.y0 - (rows + 0.5) * self.res
        box = (lon > -125) & (lon < -105) & (lat > 30) & (lat < 45) & ~(crop > self.crop_threshold)
        r, c = np.where(box)
        us = self.us_mask(lon[r, c], lat[r, c])   ## US only: RAP and the paper cover the US, not Mexico
        print('study-box pixels outside the US removed:', (~us).sum())
        r, c = r[us], c[us]
        igbp = major[r, c]
        info = pd.DataFrame({'lon': lon[r, c], 'lat': lat[r, c], 'area_weight': np.cos(np.deg2rad(lat[r, c])),
                             'cropland_pct': crop[r, c], 'IGBP_majority': igbp,
                             'landcover_type': [type_dic.get(int(v), 'Other') if np.isfinite(v) else np.nan
                                                for v in igbp]},
                            index=pd.Index(list(zip(r.tolist(), c.tolist())), name='pix', tupleize_cols=False))   ## tuples, not a MultiIndex
        print('study-box pixels with cropland <=', self.crop_threshold, '%:', len(info))
        ## NLCD cropland mask (2026-11-21)
        nlcd = T.load_df(self.nlcd_crop_dff)['nlcd_cropland_max'].reindex(info.index)
        info['nlcd_cropland'] = nlcd.values
        drop = (nlcd >= self.nlcd_crop_threshold).values
        print('NLCD cropland >=', self.nlcd_crop_threshold * 100, '% removed:', int(drop.sum()),
              '| no NLCD (kept):', int(nlcd.isna().sum()))
        info = info[~drop]
        print('pixels after the NLCD cropland mask:', len(info))
        return info

    def us_mask(self, lon, lat):
        ## True if the cell centre is in the US (Natural Earth 50 m states, the same data as the map state lines)
        import cartopy.io.shapereader as shpreader
        import shapely
        from shapely.ops import unary_union
        shp = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_1_states_provinces_lakes')
        geoms = [rec.geometry for rec in shpreader.Reader(shp).records()
                 if rec.attributes.get('adm0_a3') == 'USA' or rec.attributes.get('admin') == 'United States of America']
        us = unary_union(geoms)
        if hasattr(shapely, 'contains_xy'):   ## shapely >= 2.0
            return shapely.contains_xy(us, lon, lat)
        from shapely.prepared import prep
        from shapely.geometry import Point
        us = prep(us)
        return np.array([us.contains(Point(x, y)) for x, y in zip(lon, lat)])

    def monthly_values(self, pix_list, fdir, first_year):
        ## pixel x month, Jan 2002 ... Dec 2024 (2002 for the antecedent months); first_year None = series ends Dec 2024
        spatial_dic = T.load_npy_dir(fdir)
        n = max(len(v) for v in list(spatial_dic.values())[:1000])
        if first_year is None:
            first_year = 2025 - n // 12
        out = np.full((len(pix_list), n), np.nan)
        for i, pix in enumerate(pix_list):
            if pix in spatial_dic and len(spatial_dic[pix]) == n:
                out[i] = np.array(spatial_dic[pix], dtype=float)
        del spatial_dic
        print(fdir, '| first year', first_year, '| months', n)
        k0 = (2002 - first_year) * 12
        if k0 < 0:   ## MODIS LAI starts in 2003: pad 2002 with NaN
            out = np.concatenate([np.full((len(pix_list), -k0), np.nan), out], axis=1)
            k0 = 0
        return out[:, k0:k0 + 23 * 12]

    def window(self, arr, start, length, stat):
        ## per pixel and year 2003-2024: mean or sum over the months start ... start + length - 1 (index 0 = Jan of the
        ##   year; negative start = months of the year before); NaN if any month is missing
        n_year = len(self.year_list)
        k = np.arange(12)[None, None, :]
        base = (np.arange(n_year)[None, :] + 1) * 12 + start[:, None]   ## index 0 of arr = Jan 2002
        idx = np.clip(base[:, :, None] + k, 0, arr.shape[1] - 1)
        in_win = k < length[:, None, None]
        v = np.take_along_axis(arr, idx.reshape(len(arr), -1), axis=1).reshape(idx.shape)
        v = np.where(in_win, v, np.nan)
        complete = np.all(np.isfinite(v) | ~in_win, axis=2)
        out = np.nansum(v, axis=2) if stat == 'sum' else np.nanmean(v, axis=2)
        return np.where(complete, out, np.nan)

    def define(self, c):
        ## seasons of one pixel from its 12 monthly means; months returned as 1-12
        out = {'n_season': 0, 's1_start': np.nan, 's1_end': np.nan, 's1_peak': np.nan,
               's2_start': np.nan, 's2_end': np.nan, 's2_peak': np.nan,
               'gs_start': np.nan, 'gs_end': np.nan, 'gs_peak': np.nan}
        if np.any(~np.isfinite(c)) or c.max() <= c.min():
            return out
        n = 12
        lo, amp = c.min(), c.max() - c.min()
        allowed = np.zeros(n, dtype=bool)
        allowed[1:10] = True   ## peaks only in Feb-Oct
        main = int(np.argmax(np.where(allowed, c, -np.inf)))

        def expand(p, thr):
            s = p
            while True:
                q = (s - 1) % n
                if q == p or c[q] < thr or c[q] > c[s]:
                    break
                s = q
            e = p
            while True:
                q = (e + 1) % n
                if q == p or c[q] < thr or c[q] > c[e]:
                    break
                e = q
            return s, e

        main_season = (main,) + expand(main, lo + self.half * amp)
        season_list = [main_season]
        best = None
        for p in range(n):
            if p == main or not allowed[p] or not (c[p] >= c[(p - 1) % n] and c[p] > c[(p + 1) % n]):
                continue
            fwd = [c[(p + k) % n] for k in range(1, (main - p) % n)]
            bwd = [c[(p - k) % n] for k in range(1, (p - main) % n)]
            low = max(min(fwd) if fwd else c[p], min(bwd) if bwd else c[p])
            prom = c[p] - low
            if prom >= self.prom_ratio * amp and (best is None or prom > best[0]):
                best = (prom, p, low)
        if best is not None:
            prom, p, low = best
            s, e = expand(p, low + self.half * prom)
            if (e - s) % n + 1 >= self.min_len_second:
                season_list.append((p, s, e))
        out['gs_peak'], out['gs_start'], out['gs_end'] = [v + 1 for v in main_season]
        season_list = sorted(season_list, key=lambda x: x[0])   ## calendar order of the peaks
        out['n_season'] = len(season_list)
        for i, (p, s, e) in enumerate(season_list):
            out[f's{i + 1}_peak'], out[f's{i + 1}_start'], out[f's{i + 1}_end'] = p + 1, s + 1, e + 1
        return out


class Growing_season_definition(Growing_season_extraction):
    ## Growing-season (phenology) results: one or two seasons, which months (2026-11-15)
    ## input: Result\Dataframe\growing_season\growing_season.df (Growing_season_extraction must run first)
    ## pixels: growing-season mean LAI >= 0.1
    ## regions: all, monsoon / non-monsoon (Jul-Sep share of annual P >= 0.3); classes: All + 4 vegetation types
    ## maps: number of seasons; main season start / end / peak / length; two-season pixels: season 1 and season 2
    ##   start / end
    ## tables:
    ##   phenology_summary.csv       area with two seasons; peak month of single-season pixels (% of area)
    ##   phenology_windows.csv       most common windows: single-season (main window), two-season (season 1 + 2)
    ##   phenology_sensitivity.csv   area with two seasons if the second peak needs 20 / 30 / 50% of the amplitude,
    ##                               and the area where a second peak lasts only one month (not counted)
    ## output: Result2\02_vegetation\phenology\
    def __init__(self):
        super().__init__()
        self.outdir = this_root + rf'Result2\02_vegetation\phenology\\'
        T.mk_dir(self.outdir, force=True)
        self.mapper = Map_plotter(self.outdir)
        self.class_list = ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']
        self.month_name_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pass

    def run(self):
        gs = T.load_df(self.df_dir + 'growing_season.df')
        gs = gs[(gs['n_season'] > 0) & (gs['GS_mean_LAI'] >= self.LAI_threshold)]
        gs = gs.copy()
        if 'clim_01' not in gs.columns:   ## growing_season.df made before clim_ was added: monthly means from the dict
            lai_month = self.monthly_values(gs.index, data_root + rf'MODIS_LAI_0706\dic\\', 2003)
            clim = np.nanmean(lai_month[:, 12:].reshape(len(gs), -1, 12), axis=1)
            for k in range(12):
                gs[f'clim_{k + 1:02d}'] = clim[:, k]
        ## seasons again from the monthly means with the current define() (e.g. Feb-Oct peak rule, 2026-11-15);
        ##   GS.df keeps the windows of the last Growing_season_extraction run
        clim = gs[[f'clim_{k:02d}' for k in range(1, 13)]].values
        new = pd.DataFrame([self.define(c) for c in clim], index=gs.index)
        for col in new.columns:
            gs[col] = new[col].values
        gs['gs_length'] = (gs['gs_end'] - gs['gs_start']) % 12 + 1
        gs = gs[gs['n_season'] > 0]
        gs['region'] = np.where(gs['monsoon_share'] >= 0.3, 'monsoon', 'non-monsoon')
        print('pixels:', len(gs), '| two-season pixels:', (gs['n_season'] == 2).sum())
        self.maps(gs)
        self.summary(gs)
        self.windows(gs)
        self.sensitivity(gs)
        pass

    def group_mask(self, gs):
        ## (region, class) -> mask
        dic = {}
        for region in ['all', 'monsoon', 'non-monsoon']:
            mr = np.ones(len(gs), dtype=bool) if region == 'all' else (gs['region'] == region).values
            for lc in self.class_list:
                m = mr & ((gs['landcover_type'] == lc).values if lc != 'All' else True)
                if m.sum() >= 30:
                    dic[(region, lc)] = m
        return dic

    def maps(self, gs):
        w = gs['area_weight'].values
        two = gs['n_season'].values == 2
        month_levels = list(range(1, 13))
        for col, label, cmap, levels, only_two in [
                ('n_season', 'Number of growing seasons', 'Reds', [1, 2], False),
                ('gs_start', 'Main growing season: start month', 'twilight_shifted', month_levels, False),
                ('gs_end', 'Main growing season: end month', 'twilight_shifted', month_levels, False),
                ('gs_peak', 'Main growing season: peak month', 'twilight_shifted', month_levels, False),
                ('gs_length', 'Main growing season: length (months)', 'viridis', list(range(1, 9)), False),
                ('s1_start', 'Two-season pixels, season 1: start month', 'twilight_shifted', month_levels, True),
                ('s1_end', 'Two-season pixels, season 1: end month', 'twilight_shifted', month_levels, True),
                ('s2_start', 'Two-season pixels, season 2: start month', 'twilight_shifted', month_levels, True),
                ('s2_end', 'Two-season pixels, season 2: end month', 'twilight_shifted', month_levels, True)]:
            vals = gs[col].values.astype(float)
            if only_two:
                vals = np.where(two, vals, np.nan)
            arr = D.pix_dic_to_spatial_arr(dict(zip(gs.index, vals)))
            arr[arr <= -9999] = np.nan
            ok = np.isfinite(vals)
            self.mapper.plot_map(arr, col, cmap, None, None, label, vals[ok], w[ok], discrete_levels=levels)
        pass

    def summary(self, gs):
        ## area with two seasons; peak month of single-season pixels (% of the single-season area)
        w_all = gs['area_weight'].values
        result = []
        for (region, lc), m in self.group_mask(gs).items():
            w = w_all[m]
            two = gs['n_season'].values[m] == 2
            peak = gs['gs_peak'].values[m]
            row = [region, lc, m.sum(), np.sum(w[two]) / np.sum(w) * 100]
            for k in range(1, 13):
                row.append(np.sum(w[~two & (peak == k)]) / np.sum(w[~two]) * 100)
            result.append(row)
        cols = ['region', 'class', 'n_pix', 'two_season_area_%'] + \
               [f'single_peak_{m}_%' for m in self.month_name_list]
        df = pd.DataFrame(result, columns=cols)
        df.to_csv(self.outdir + 'phenology_summary.csv', index=False)
        print(df.iloc[:, :4].round(1))
        pass

    def windows(self, gs):
        ## most common windows (top 6, % of the area of that group)
        M = self.month_name_list
        win = lambda s, e: f'{M[int(s) - 1]}-{M[int(e) - 1]}'
        two_all = gs['n_season'].values == 2
        label = np.array([win(s1, e1) + ' + ' + win(s2, e2) if t else win(s, e) for t, s, e, s1, e1, s2, e2 in
                          zip(two_all, gs['gs_start'], gs['gs_end'], gs['s1_start'], gs['s1_end'], gs['s2_start'],
                              gs['s2_end'])])
        w_all = gs['area_weight'].values
        result = []
        for (region, lc), m in self.group_mask(gs).items():
            for n_s, name in [(1, 'one season'), (2, 'two seasons')]:
                mm = m & (gs['n_season'].values == n_s)
                if mm.sum() == 0:
                    continue
                s = pd.Series(w_all[mm], index=label[mm]).groupby(level=0).sum()
                for k, v in (s / s.sum() * 100).sort_values(ascending=False).head(6).items():
                    result.append([region, lc, name, k, v])
        df = pd.DataFrame(result, columns=['region', 'class', 'type', 'window', 'area_%_of_type'])
        df.to_csv(self.outdir + 'phenology_windows.csv', index=False)
        print(df[(df['class'] == 'All')].round(1))
        pass

    def sensitivity(self, gs):
        ## two-season area for other thresholds of the second peak; one-month second peaks (not counted)
        clim = gs[[f'clim_{k:02d}' for k in range(1, 13)]].values
        w_all = gs['area_weight'].values
        prom_now, min_len_now = self.prom_ratio, self.min_len_second
        res = {}
        for prom in [0.2, 0.3, 0.5]:
            for min_len in [1, 2]:
                self.prom_ratio, self.min_len_second = prom, min_len
                res[(prom, min_len)] = np.array([self.define(c)['n_season'] for c in clim]) == 2
        self.prom_ratio, self.min_len_second = prom_now, min_len_now
        result = []
        for (region, lc), m in self.group_mask(gs).items():
            w = w_all[m]
            row = [region, lc, m.sum()]
            for prom in [0.2, 0.3, 0.5]:
                row.append(np.sum(w[res[(prom, 2)][m]]) / np.sum(w) * 100)
            one_month = res[(0.3, 1)][m] & ~res[(0.3, 2)][m]
            row.append(np.sum(w[one_month]) / np.sum(w) * 100)
            result.append(row)
        df = pd.DataFrame(result, columns=['region', 'class', 'n_pix', 'two_season_%_prom20', 'two_season_%_prom30',
                                           'two_season_%_prom50', 'one_month_second_peak_%_prom30'])
        df.to_csv(self.outdir + 'phenology_sensitivity.csv', index=False)
        print(df.round(1))
        pass


class Growing_season_trend():
    ## parent class: trend of growing-season variables (main season, GS.df), 2003-2024 (2026-11-15)
    ## children: GS_LAI_trend, GS_SPEI_trend, GS_climate_trend
    ## pixels (same for all three, so LAI / SPEI / climate trends are comparable):
    ##   complete 2003-2024 GS_LAI series and growing-season mean LAI >= 0.1
    ## trend: OLS per pixel, per decade, not detrended
    ## per variable: {var}_trend map, {var}_trend_sig map (p < 0.05 only), tif of both
    ## table: area-weighted mean trend with 95% CI (1 x 1 deg block bootstrap) and area with a significant increase /
    ##   decrease, per vegetation type, for all pixels, monsoon / non-monsoon (Jul-Sep share of annual P >= 0.3)
    ##   and two-season pixels
    def __init__(self, outdir):
        self.dff = result_root + rf'\Dataframe\growing_season\GS.df'
        self.outdir = outdir
        T.mk_dir(self.outdir, force=True)
        self.year_list = list(range(2003, 2025))
        self.tag = f'{self.year_list[0]}_{self.year_list[-1]}'
        self.LAI_threshold = 0.1
        self.class_list = ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']
        self.mapper = Map_plotter(self.outdir)
        pass

    def load(self):
        ## GS.df, the selected pixels and their info
        df = T.load_df(self.dff)
        lai = df.pivot(index='pix', columns='year', values='GS_LAI').reindex(columns=self.year_list)
        ok = lai.notna().all(axis=1) & (lai.mean(axis=1) >= self.LAI_threshold)
        pix_list = lai.index[ok]
        info = df.groupby('pix')[['area_weight', 'landcover_type', 'lon', 'lat', 'n_season',
                                  'monsoon_share']].first().reindex(pix_list)
        print('pixels:', len(info))
        return df, info

    def matrix(self, df, info, var, relative=False):
        ## pix x year matrix of one variable; relative=True: % of the pixel's 2003-2024 mean
        arr = df.pivot(index='pix', columns='year', values=var).reindex(index=info.index, columns=self.year_list).values
        if relative:
            mean = np.nanmean(arr, axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                arr = np.where(mean > 0, (arr / mean - 1) * 100, np.nan)
        return arr

    def trend(self, arr):
        ## OLS slope per decade and two-sided p, per row; rows with any NaN give NaN
        x = np.array(self.year_list, dtype=float)
        xm = x - x.mean()
        ym = arr - arr.mean(axis=1, keepdims=True)
        slope = ym @ xm / np.sum(xm ** 2)
        resid = ym - slope[:, None] * xm[None, :]
        se = np.sqrt(np.sum(resid ** 2, axis=1) / (len(x) - 2) / np.sum(xm ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            p = 2 * stats.t.sf(np.abs(slope / se), len(x) - 2)
        return slope * 10, p

    def save_map(self, info, vals, fname, label, vlim, cmap):
        arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, vals)))
        arr[arr <= -9999] = np.nan
        D.arr_to_tif(arr.copy(), self.outdir + f'{fname}.tif')   ## copy: arr_to_tif changes NaN in place
        ok = np.isfinite(vals)
        self.mapper.plot_map(arr, fname, cmap, -vlim, vlim, label, vals[ok], info['area_weight'].values[ok])

    def map_pair(self, info, tr, p, fname, label, vlim, cmap):
        ## trend map and the p < 0.05 map with the same colour range
        self.save_map(info, tr, fname, label, vlim, cmap)
        self.save_map(info, np.where(p < 0.05, tr, np.nan), fname.replace('_trend', '_trend_sig'),
                      label + ', p < 0.05 only', vlim, cmap)

    def block_ci(self, info, v, w):
        ## 95% CI of an area-weighted mean, resampling 1 x 1 deg blocks
        block = (np.floor(info['lon'].values) * 1000 + np.floor(info['lat'].values)).astype(int)
        ids = np.unique(block)
        idx = {b: np.where(block == b)[0] for b in ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(200):
            pick = np.concatenate([idx[b] for b in rng.choice(ids, size=len(ids))])
            boot.append(np.sum(v[pick] * w[pick]) / np.sum(w[pick]))
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    def table(self, info, res_list, fname):
        ## res_list: [(variable, unit, trend, p, pixel mean of the variable), ...]
        region_dic = {'all': np.ones(len(info), dtype=bool),
                      'monsoon': (info['monsoon_share'] >= 0.3).values,
                      'non-monsoon': (info['monsoon_share'] < 0.3).values,
                      'two-season': (info['n_season'] == 2).values}
        w_all = info['area_weight'].values
        result = []
        for var, unit, tr, p, mean_val in res_list:
            for region, mr in region_dic.items():
                for lc in self.class_list:
                    m = mr & ((info['landcover_type'] == lc).values if lc != 'All' else True) & np.isfinite(tr)
                    if m.sum() < 30:
                        continue
                    w = w_all[m]
                    lo, hi = self.block_ci(info[m], tr[m], w)
                    result.append([var, unit, region, lc, m.sum(), np.sum(tr[m] * w) / np.sum(w), lo, hi,
                                   np.sum(w * ((tr[m] > 0) & (p[m] < 0.05))) / np.sum(w) * 100,
                                   np.sum(w * ((tr[m] < 0) & (p[m] < 0.05))) / np.sum(w) * 100,
                                   np.nansum(mean_val[m] * w) / np.sum(w)])
        df_t = pd.DataFrame(result, columns=['variable', 'unit', 'region', 'class', 'n_pix', 'trend_per_decade',
                                             'ci_low', 'ci_high', 'sig_increase_area_%', 'sig_decrease_area_%',
                                             'mean_value'])
        df_t.to_csv(self.outdir + f'{fname}_{self.tag}.csv', index=False)
        print(df_t.round(2).to_string())
        pass


class GS_LAI_trend(Growing_season_trend):
    ## growing-season LAI trend: % of the pixel's mean per decade and m2/m2 per decade
    ## output: Result2\02_vegetation\growing_season\trend\
    def __init__(self):
        super().__init__(this_root + rf'Result2\02_vegetation\growing_season\trend\\')

    def run(self):
        df, info = self.load()
        res_list = []
        for var, unit, rel in [('GS_LAI', '%', True), ('GS_LAI', 'm2/m2', False)]:
            arr = self.matrix(df, info, var, rel)
            tr, p = self.trend(arr)
            mean_val = np.nanmean(self.matrix(df, info, var), axis=1)
            fname = f'GS_LAI_trend_{self.tag}' if rel else f'GS_LAI_trend_abs_{self.tag}'
            self.map_pair(info, tr, p, fname, f'Growing-season LAI trend ({unit} per decade)',
                          np.nanpercentile(np.abs(tr), 90), 'RdBu')
            res_list.append([var, unit, tr, p, mean_val])
        self.table(info, res_list, 'GS_LAI_trend_summary')
        pass


class GS_SPEI_trend(Growing_season_trend):
    ## growing-season SPEI trend (SPEI units per decade; negative = drier)
    ## maps: SPEI03, SPEI12, SPEI24 with one shared colour range; table: all scales 03-48
    ## output: Result2\01_drought\growing_season\trend\
    def __init__(self):
        super().__init__(this_root + rf'Result2\01_drought\growing_season\trend\\')
        self.map_scale_list = [3,6,9, 12, 24, 36,48]
        self.table_scale_list = list(range(3, 49, 3))

    def run(self):
        df, info = self.load()
        tr_dic, p_dic = {}, {}
        for scale in self.table_scale_list:
            tr_dic[scale], p_dic[scale] = self.trend(self.matrix(df, info, f'GS_SPEI{scale:02d}'))
        vlim = np.nanpercentile(np.abs(np.concatenate([tr_dic[s] for s in self.map_scale_list])), 90)
        for scale in self.map_scale_list:
            self.map_pair(info, tr_dic[scale], p_dic[scale], f'GS_SPEI{scale:02d}_trend_{self.tag}',
                          f'Growing-season SPEI{scale:02d} trend (per decade)', vlim, 'RdBu')
        res_list = [[f'GS_SPEI{s:02d}', 'SPEI', tr_dic[s], p_dic[s],
                     np.nanmean(self.matrix(df, info, f'GS_SPEI{s:02d}'), axis=1)] for s in self.table_scale_list]
        self.table(info, res_list, 'GS_SPEI_trend_summary')
        pass


class GS_climate_trend(Growing_season_trend):
    ## growing-season climate trend
    ##   GS_P, GS_P_antecedent: % of the pixel's mean per decade (BrBG, positive = wetter)
    ##   GS_VPD: per decade, in the unit of the VPD dict (RdBu_r);  GS_Tmax: deg C per decade (RdBu_r)
    ## output: Result2\01_drought\growing_season\trend\ (same folder as the SPEI trend)
    def __init__(self):
        super().__init__(this_root + rf'Result2\01_drought\trend\\')
        ## variable, unit, relative to the mean, colour map
        self.var_list = [('GS_P', '%', True, 'BrBG'),
                         ('GS_P_antecedent', '%', True, 'BrBG'),
                         ('GS_VPD', 'kPa', False, 'RdBu_r'),
                         ('GS_Tmax', 'deg C', False, 'RdBu_r')]

    def run(self):
        df, info = self.load()
        res_list = []
        for var, unit, rel, cmap in self.var_list:
            tr, p = self.trend(self.matrix(df, info, var, rel))
            self.map_pair(info, tr, p, f'{var}_trend_{self.tag}', f'Growing-season {var[3:]} trend ({unit} per decade)',
                          np.nanpercentile(np.abs(tr), 90), cmap)
            res_list.append([var, unit, tr, p, np.nanmean(self.matrix(df, info, var), axis=1)])
        self.table(info, res_list, 'GS_climate_trend_summary')
        pass


class GS_LAI_long_term(Growing_season_extraction):
    ## growing-season LAI of a long monthly LAI record (2026-11-21)
    ## pixels : the pixels of growing_season.df (same as GS.df: study box, MCD12C1 cropland <= 50%, NLCD cropland < 5%,
    ##          US only) with a main season inside one calendar year
    ## window : the MODIS growing season of each pixel (gs_start - gs_end of growing_season.df), the same months every
    ##          year; the phenology is NOT redefined from the coarse long record, so long and MODIS series are comparable
    ## per year: GS_LAI = mean of the window months (all months needed), m2/m2
    ##           GS_LAI_rel = % difference from the pixel's mean GS_LAI over all years of the record
    ## per pixel: lon, lat, area_weight, landcover_type, gs_start, gs_end, n_season, GS_mean_LAI (this record),
    ##           MODIS_GS_mean_LAI (GS_mean_LAI of growing_season.df)
    ## input  : monthly dict per pixel, Jan of first_year ... Dec of last_year
    ## output : Result\Dataframe\growing_season\GS_{name}.df (pixel-year)
    def __init__(self, name, fdir, first_year, last_year):
        super().__init__()
        self.name = name
        self.fdir = fdir
        self.first_year = first_year
        self.last_year = last_year
        self.year_list = list(range(first_year, last_year + 1))
        self.gs_dff = self.df_dir + 'growing_season.df'
        self.dff_out = self.df_dir + f'GS_{name}.df'

    def run(self):
        gs = T.load_df(self.gs_dff)
        gs = gs[(gs['n_season'] > 0) & (gs['gs_start'] <= gs['gs_end'])]
        pix_list = gs.index
        print(self.name, '| pixels:', len(pix_list))
        lai_month = self.load_monthly(pix_list)
        start = gs['gs_start'].values.astype(int) - 1
        length = (gs['gs_end'].values - gs['gs_start'].values).astype(int) + 1
        lai = self.window_long(lai_month, start, length)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(lai, axis=1)
            rel = (lai / mean[:, None] - 1) * 100
        n_year = len(self.year_list)
        df = pd.DataFrame({'pix': [p for p in pix_list for _ in range(n_year)],
                           'year': np.tile(self.year_list, len(pix_list)),
                           'GS_LAI': lai.ravel(), 'GS_LAI_rel': rel.ravel()})
        info = gs[['lon', 'lat', 'area_weight', 'landcover_type', 'gs_start', 'gs_end', 'n_season']].copy()
        info['MODIS_GS_mean_LAI'] = gs['GS_mean_LAI'].values
        info['GS_mean_LAI'] = mean
        df = df.merge(info, left_on='pix', right_index=True, how='left')
        print(self.name, '| GS_LAI non-NaN rows:', df['GS_LAI'].notna().sum(), '/', len(df),
              '| pixels with all years:', int(np.isfinite(lai).all(axis=1).sum()))
        T.save_df(df, self.dff_out)
        T.df_to_excel(df.head(2000), self.df_dir + f'GS_{self.name}_head')   ## first rows only, for checking
        pass

    def load_monthly(self, pix_list):
        ## pixel x month, Jan first_year ... Dec last_year; pixels with a different series length -> NaN
        spatial_dic = T.load_npy_dir(self.fdir)
        n = (self.last_year - self.first_year + 1) * 12
        out = np.full((len(pix_list), n), np.nan)
        n_bad = 0
        for i, pix in enumerate(pix_list):
            if pix not in spatial_dic:
                continue
            v = np.array(spatial_dic[pix], dtype=float)
            if len(v) != n:
                n_bad += 1
                continue
            out[i] = v
        del spatial_dic
        out[out < 0] = np.nan   ## fill values
        print(self.fdir, '| months', n, '| pixels with another length:', n_bad)
        return out

    def window_long(self, arr, start, length):
        ## per pixel and year: mean over the months start ... start + length - 1 (index 0 = Jan of first_year);
        ##   NaN if any month is missing
        import warnings
        n_year = len(self.year_list)
        k = np.arange(12)[None, None, :]
        base = np.arange(n_year)[None, :] * 12 + start[:, None]
        idx = np.clip(base[:, :, None] + k, 0, arr.shape[1] - 1)
        in_win = k < length[:, None, None]
        v = np.take_along_axis(arr, idx.reshape(len(arr), -1), axis=1).reshape(idx.shape)
        v = np.where(in_win, v, np.nan)
        complete = np.all(np.isfinite(v) | ~in_win, axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            out = np.nanmean(v, axis=2)
        return np.where(complete, out, np.nan)


class GS_LAI_GIMMS(GS_LAI_long_term):
    ## GIMMS LAI4g, monthly (MVC of the half-month values), 1982-2020 -> GS_GIMMS.df (2026-11-21)
    def __init__(self):
        super().__init__('GIMMS', data_root + rf'GIMMS4g\dic\\', 1982, 2020)


class GS_LAI_SNU(GS_LAI_long_term):
    ## SNU LAI, monthly, 1982-2024 -> GS_SNU.df (2026-11-21)
    def __init__(self):
        super().__init__('SNU', data_root + rf'SNU_LAI\dic\\', 1982, 2024)


def main():
    ## 2026-11-15: US only -> rerun everything in this order
    # Growing_season_extraction().run()   ## GS.df, growing_season.df
    # Growing_season_definition().run()   ## one / two seasons, months: maps and tables in 02_vegetation\phenology
    # GS_LAI_trend().run()                ## GS LAI trend maps and table (02_vegetation)
    # GS_SPEI_trend().run()               ## GS SPEI trend maps and table (01_drought)
    # GS_climate_trend().run()            ## GS P, VPD, Tmax trend maps and table (01_drought)
    ## 2026-11-21: long LAI records, MODIS growing-season window
    GS_LAI_GIMMS().run()                ## GS_GIMMS.df, 1982-2020
    GS_LAI_SNU().run()                  ## GS_SNU.df, 1982-2024


if __name__ == '__main__':
    main()
