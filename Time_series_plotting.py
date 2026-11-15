from __Global__ import *
import os
import warnings
from scipy import stats
from matplotlib.ticker import MaxNLocator


class Time_series_base():
    ## shared plotting; column name in df = f'{season}_{var}', e.g. spring_SPEI03, summer_SNU_LAI
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        self.dff = result_root + rf'\Dataframe\SPEI\SPEI.df'
        self.result2 = this_root + rf'\Result2\\'
        self.season_list = ['spring', 'summer']
        self.color_dic = {'spring': '#EA6E88', 'summer': '#48526F'}
        self.split_year = 2000
        self.end_year = 2024
        self.crop_threshold = 50   ## % cropland cover above which a pixel is excluded
        ## trend periods: (start, end, linestyle)
        self.trend_period_list = [(1982, 2024, ':'), (2000, 2024, '-')]
        self.ylabel_dic = {}   ## y-axis label per variable; default = variable name
        pass

    def load_df(self):
        df = T.load_df(self.dff)
        df = self.df_clean(df)
        return df

    def df_clean(self, df):
        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 45]
        ## exclude cropland: mean MCD12C1 cropland cover (IGBP 12 + 14) > 50%
        df = df[~(df['cropland_pct'] > self.crop_threshold)]
        return df

    def load_GS_df(self):
        ## growing_season_extraction.py -> GS.df: each pixel's main growing season, 2003-2024
        ## pixels: complete GS LAI 2003-2024 and growing-season mean LAI >= 0.1 (same as Growing_season_trend)
        ## columns GS_X are renamed growing_X so that plot_time_series finds them with season 'growing'
        df = T.load_df(result_root + rf'\Dataframe\growing_season\GS.df')
        df = df[(df['year'] >= 2003) & (df['year'] <= 2024)]
        n_ok = df['GS_LAI'].notna().groupby(df['pix']).transform('sum')
        df = df[(n_ok == 22) & (df['GS_mean_LAI'] >= 0.1)]
        df = df.rename(columns={c: 'growing_' + c[3:] for c in df.columns if c.startswith('GS_')})
        self.season_list = ['growing']
        self.color_dic = {'growing': '#2E7D32'}
        self.trend_period_list = [(2003, 2024, '-')]
        print('growing-season pixels:', df['pix'].nunique())
        return df

    def plot_time_series(self, df, var_list, start_year, outdir, fname):
        T.mk_dir(outdir, force=True)
        years = np.arange(start_year, self.end_year + 1)
        trend_list = []

        fig, axes = plt.subplots(len(var_list), 1, sharex=True, squeeze=False,
                                 figsize=(self.map_width * 1.5, self.map_height * 0.6 * max(len(var_list), 1.5)))
        axes = axes[:, 0]

        for ax, var in zip(axes, var_list):
            for season in self.season_list:
                col = f'{season}_{var}'
                if not col in df.columns:
                    continue
                mean_list = []
                std_list = []
                for year in years:
                    df_i = df[df['year'] == year]
                    vals = np.array(df_i[col].tolist(), dtype=float)
                    weight = np.array(df_i['area_weight'].tolist(), dtype=float)
                    weighted_mean = np.nansum(vals * weight) / np.nansum(weight * np.isfinite(vals))
                    weighted_var = np.nansum(weight * (vals - weighted_mean) ** 2) / np.nansum(weight * np.isfinite(vals))
                    mean_list.append(weighted_mean)
                    std_list.append(np.sqrt(weighted_var))
                mean_list = np.array(mean_list)
                std_list = np.array(std_list)
                color = self.color_dic[season]

                ax.plot(years, mean_list, color=color, lw=1.5, label=season.capitalize())
                ax.fill_between(years, mean_list - std_list, mean_list + std_list, color=color, alpha=0.15)

                ## linear trends: 1982-2024 (dotted) and 2000-2024 (solid)
                text = f'{season.capitalize()}'
                for t_start, t_end, ls in self.trend_period_list:
                    mask = (years >= t_start) & (years <= t_end) & np.isfinite(mean_list)
                    slope, intercept, r, p, se = stats.linregress(years[mask], mean_list[mask])
                    ax.plot(years[mask], slope * years[mask] + intercept, color=color, lw=2, linestyle=ls)
                    text += f'  {t_start}-{t_end}: {slope * 10:.2f}/10yr, p={p:.2f}'
                    trend_list.append([var, season, f'{t_start}-{t_end}', slope * 10, p])
                ax.text(0.01, 0.95 if season == 'spring' else 0.84, text,
                        transform=ax.transAxes, color=color, fontsize=7.5, va='top')

            if 'SPEI' in var:
                ax.axhline(0, color='grey', lw=0.5)
            if self.split_year > start_year:
                ax.axvspan(self.split_year, self.end_year, color='grey', alpha=0.1)
            ax.set_ylabel(self.ylabel_dic.get(var, var), fontsize=10)
            ax.set_xlim(start_year, self.end_year)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[0].legend(loc='upper right', frameon=False, ncol=2)
        axes[-1].set_xlabel('Year', fontsize=10)
        plt.tight_layout()
        plt.savefig(outdir + f'{fname}.png', dpi=300)
        plt.savefig(outdir + f'{fname}.pdf')
        plt.show()
        plt.close()

        df_trend = pd.DataFrame(trend_list, columns=['variable', 'season', 'period', 'slope_per_decade', 'p_value'])
        df_trend.to_csv(outdir + f'{fname}_trend.csv', index=False)
        print(df_trend)


class SPEI_time_series_MODIS(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'01_drought\SPEI_timeseries\\'
        self.start_year = 1982
        self.main_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI48']
        self.other_list = ['SPEI01', 'SPEI09', 'SPEI15', 'SPEI18', 'SPEI21', 'SPEI27',
                           'SPEI30', 'SPEI33', 'SPEI36', 'SPEI39', 'SPEI42', 'SPEI45']
        pass



    def run(self):
        ## SPEI averaged over each pixel's main growing season (GS.df), 2003-2024
        df = self.load_GS_df()
        outdir = self.outdir
        other_list = [v for v in self.other_list if v != 'SPEI01']   ## GS.df has SPEI03 ... SPEI48
        self.plot_time_series(df, self.main_list, 2003, outdir, 'Fig_SPEI_main_GS')
        self.plot_time_series(df, other_list, 2003, outdir, 'FigS_SPEI_other_GS')
        pass


class SNU_LAI_time_series(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'02_vegetation\SNU_LAI_timeseries\\'
        self.start_year = 1982
        self.var_list = ['SNU_LAI_relative_change']
        self.ylabel_dic = {'SNU_LAI_relative_change': 'Relative Change (%)'}
        self.LAI_threshold = 0.3   ## drop pixels whose multi-year mean LAI < 0.3
        pass

    def run(self):
        df = self.load_df()
        df = self.mask_low_LAI(df)
        self.plot_time_series(df, self.var_list, self.start_year, self.outdir, 'Fig_SNU_LAI')
        pass

    def mask_low_LAI(self, df):
        for season in self.season_list:
            pix_mean = df.groupby('pix')[f'{season}_SNU_LAI'].transform('mean')   ## multi-year mean per pixel
            df.loc[pix_mean < self.LAI_threshold, f'{season}_SNU_LAI_relative_change'] = np.nan
        return df


class MODIS_LAI_time_series(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'02_vegetation\MODIS_LAI_timeseries\\'
        self.start_year = 2003
        self.var_list = ['MODIS_LAI_relative_change']
        self.ylabel_dic = {'MODIS_LAI_relative_change': 'Relative Change (%)'}
        self.trend_period_list = [(2003, 2024, '-')]   ## MODIS starts 2003: one trend only
        self.LAI_threshold = 0.3   ## drop pixels whose multi-year mean LAI < 0.3
        pass



    def run(self):
        ## MODIS LAI of each pixel's main growing season (GS.df), % difference from the pixel's 2003-2024 mean
        df = self.load_GS_df()
        self.ylabel_dic = {'LAI_rel': 'Relative Change (%)'}
        self.plot_time_series(df, ['LAI_rel'], 2003, self.outdir + rf'\\', 'Fig_MODIS_LAI_GS')
        pass

    def mask_low_LAI(self, df):
        ## df has no raw MODIS LAI, so take the multi-year mean (2003-2024) from the npy
        for season in self.season_list:
            val_dic = T.load_npy(result_root + rf'\MODIS_LAI\{season}_LAI.npy')
            mean_dic = {pix: np.nanmean(np.array(vals, dtype=float)) for pix, vals in val_dic.items()}
            pix_mean = df['pix'].map(mean_dic)
            df.loc[~(pix_mean >= self.LAI_threshold), f'{season}_MODIS_LAI_relative_change'] = np.nan
        return df


class Daymet_time_series(Time_series_base):
    ## seasonal Daymet precipitation indices (Daymet_GEE_processing.py -> Daymet_seasonal.df)
    ## summer = JASO (Jul-Oct), spring = MAM (Mar-May), winter = NDJF (Nov of previous year - Feb)
    def __init__(self):
        super().__init__()
        self.dff = result_root + rf'\Dataframe\Daymet\Daymet_seasonal.df'
        self.outdir = self.result2 + rf'01_drought\Daymet_timeseries\\'
        self.start_year = 1982
        self.color_dic = {'spring': '#EA6E88', 'summer': '#48526F', 'winter': '#4A90C2'}
        self.main_list = ['PRCPTOT', 'WETDAYS', 'R5mm', 'SDII', 'R95pTOT', 'CDD']
        self.other_list = ['LIGHT', 'Rx1day', 'R95p']
        self.winter_list = ['PRCPTOT', 'SWE_max']
        self.ylabel_dic = {'PRCPTOT': 'PRCPTOT (mm)', 'WETDAYS': 'WETDAYS (days)', 'R5mm': 'R5mm (days)',
                           'LIGHT': 'LIGHT (days)', 'Rx1day': 'Rx1day (mm)', 'SDII': 'SDII (mm/day)',
                           'R95p': 'R95p (mm)', 'R95pTOT': 'R95pTOT (%)', 'CDD': 'CDD (days)',
                           'SWE_max': 'SWE_max (mm)'}
        pass

    def run(self):
        df = self.load_df()
        df=self.df_clean(df)  ## same region and cropland exclusion as the SPEI / LAI time series
        ## spring and summer in the same panel
        self.season_list = ['spring', 'summer']
        self.plot_time_series(df, self.main_list, self.start_year, self.outdir, 'Fig_Daymet_main')
        self.plot_time_series(df, self.other_list, self.start_year, self.outdir, 'FigS_Daymet_other')
        self.compare_periods(df, self.main_list + self.other_list, 'Daymet_spring_summer')
        ## winter
        self.season_list = ['winter']
        self.plot_time_series(df, self.winter_list, self.start_year, self.outdir, 'FigS_Daymet_winter')
        self.compare_periods(df, self.winter_list, 'Daymet_winter')
        pass

    def load_df(self):
        df = T.load_df(self.dff)
        df['area_weight'] = np.cos(np.deg2rad(df['lat']))
        df['cropland_pct'] = df['pix'].map(self.cropland_dic())
        df = self.df_clean(df)   ## same region and cropland exclusion as the SPEI / LAI time series
        return df

    def cropland_dic(self):
        ## MCD12C1 multi-year mean cropland % (IGBP 12 + 14), same as Build_framework.add_MODIS_LUCC_to_df
        from osgeo import gdal
        tiff = data_root + rf'\basedata\MCD12\MCD12C1_IGBP_percent_mean_tmpl.tif'
        ds = gdal.Open(tiff)
        array = ds.ReadAsArray().astype(float)
        row0 = int(round((ds.GetGeoTransform()[3] - 48.97916666666667) / 0.05))   ## 469-row GEE export -> 468
        array = array[:, row0:row0 + 468, :]
        crop_arr = array[12] + array[14]
        crop_arr[(crop_arr < 0) | (crop_arr > 100)] = np.nan
        return DIC_and_TIF().spatial_arr_to_dic(crop_arr)

    def compare_periods(self, df, var_list, fname):
        ## regional area-weighted mean per year; 1982-1999 vs 2000-2024 (mean difference, Mann-Whitney p)
        T.mk_dir(self.outdir, force=True)
        years = np.arange(self.start_year, self.end_year + 1)
        result_list = []
        for var in var_list:
            for season in self.season_list:
                col = f'{season}_{var}'
                if not col in df.columns:
                    continue
                mean_list = []
                for year in years:
                    df_i = df[df['year'] == year]
                    vals = df_i[col].values.astype(float)
                    w = df_i['area_weight'].values.astype(float)
                    ok = np.isfinite(vals)
                    mean_list.append(np.sum(vals[ok] * w[ok]) / np.sum(w[ok]))
                mean_list = np.array(mean_list)
                before = mean_list[years < self.split_year]
                after = mean_list[years >= self.split_year]
                stat, p = stats.mannwhitneyu(after, before)
                result_list.append([var, season, before.mean(), after.mean(), after.mean() - before.mean(),
                                    (after.mean() / before.mean() - 1) * 100, p])
        df_out = pd.DataFrame(result_list, columns=['variable', 'season', 'mean_1982_1999', 'mean_2000_2024',
                                                    'difference', 'change_pct', 'p_mannwhitney'])
        df_out.to_csv(self.outdir + f'{fname}_period_comparison.csv', index=False)
        print(df_out.round(3))
        pass


class SPEI_long_term_plot(Time_series_base):
    ## growing-season SPEI 1958-2024 (2026-11-15)
    ## per pixel: SPEI averaged over the pixel's main growing season (growing_season.df, same window every year),
    ##   SPEI03 ... SPEI48 and SPEIbest (the pixel's best scale from bivariate.py: highest r between detrended
    ##   GS LAI and detrended GS SPEI, 2003-2024). Long-term series and trends are NOT detrended.
    ## pixels: same growing-season pixels as GS.df (n_season > 0, window within one calendar year, GS mean LAI >= 0.1)
    ## time series: area-weighted mean +- std, trends 1958-2024 (dotted) and 2000-2024 (solid), per vegetation type
    ##   main figure SPEI03, 06, 12, 24, 48; SPEIbest in its own figure (one panel per vegetation type)
    ## maps: trend 1958-2024, trend 2000-2024 (+ p < 0.05 only) and step change (mean 2000-2024 - mean 1958-1999)
    ## tables: period comparison and area-weighted trends per scale x vegetation type
    ## output: Result2\01_drought\SPEI_long_term\
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'01_drought\SPEI_long_term\\'
        T.mk_dir(self.outdir, force=True)
        self.start_year = 1958
        self.year_list = list(range(1958, 2025))
        self.scale_list = list(range(3, 49, 3))
        self.map_scale_list = [3, 6, 9, 12, 24, 36, 48]
        self.main_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI48']   ## main figure, same as SPEI_time_series
        self.class_list = ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']
        self.season_list = ['growing']
        self.color_dic = {'growing': '#D9480F'}   ## orange-red
        self.trend_period_list = [(1958, 2024, ':'), (2000, 2024, '-')]
        self.ylabel_dic = {'SPEIbest': 'SPEI (best scale)'}
        self.gs_dff = result_root + rf'\Dataframe\growing_season\growing_season.df'
        self.biv_dff = this_root + rf'Result2\03_response\LAI_SPEI_bivariate\LAI_SPEI_bivariate_2003_2024.df'
        self.dff_out = result_root + rf'\Dataframe\SPEI_GS_1958_2024.df'  ## pixel x year growing-season SPEI
        pass

    def run(self):
        if not os.path.isfile(self.dff_out):
            self.build_df()
        # df = T.load_df(self.dff_out)
        # self.time_series(df)
        # self.period_table(df)
        # self.trend_maps(df)
        pass

    def build_df(self):
        ## pixel x year growing-season SPEI 1958-2024 for every scale, plus SPEIbest
        gs = T.load_df(self.gs_dff)
        gs = gs[(gs['n_season'] > 0) & (gs['gs_start'] <= gs['gs_end']) & (gs['GS_mean_LAI'] >= 0.1)]
        pix_list = list(gs.index)
        n_pix, n_year = len(pix_list), len(self.year_list)
        mon = np.arange(12)[None, :]
        in_win = (mon >= gs['gs_start'].values[:, None] - 1) & (mon <= gs['gs_end'].values[:, None] - 1)
        out = {}
        for scale in self.scale_list:
            spatial_dic = T.load_npy_dir(data_root + rf'TerraClimate\SPEI\SPEI_{scale}_NOAA\dic\\')
            n = max(len(v) for v in list(spatial_dic.values())[:1000])
            first_year = 2025 - n // 12
            arr = np.full((n_pix, n), np.nan)
            for i, pix in enumerate(pix_list):
                if pix in spatial_dic and len(spatial_dic[pix]) == n:
                    arr[i] = np.array(spatial_dic[pix], dtype=float)
            del spatial_dic
            k0 = (self.year_list[0] - first_year) * 12
            arr = arr[:, k0:k0 + n_year * 12].reshape(n_pix, n_year, 12)
            v = np.where(in_win[:, None, :], arr, np.nan)
            complete = np.all(np.isfinite(v) | ~in_win[:, None, :], axis=2)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                out[f'growing_SPEI{scale:02d}'] = np.where(complete, np.nanmean(v, axis=2), np.nan)
            print(f'SPEI{scale:02d} | first year {first_year} | months {n}')
        ## best scale per pixel (bivariate.py)
        best = T.load_df(self.biv_dff)['best_scale'].reindex(pix_list).values
        sb = np.full((n_pix, n_year), np.nan)
        for scale in self.scale_list:
            m = best == scale
            sb[m] = out[f'growing_SPEI{scale:02d}'][m]
        out['growing_SPEIbest'] = sb
        df = pd.DataFrame({'pix': [p for p in pix_list for _ in range(n_year)], 'year': self.year_list * n_pix})
        for col, arr in out.items():
            df[col] = arr.ravel()
        for col in ['lon', 'lat', 'area_weight', 'landcover_type', 'gs_start', 'gs_end']:
            df[col] = np.repeat(gs[col].values, n_year)
        df['best_scale'] = np.repeat(best, n_year)
        T.save_df(df, self.dff_out)
        T.df_to_excel(df, self.dff_out.replace('.df', '.xlsx'))
        print('pixels:', n_pix, '| with best scale:', np.isfinite(best).sum())

    def class_df(self, df, lc):
        return df if lc == 'All' else df[df['landcover_type'] == lc]

    def time_series(self, df):
        ## main figure (SPEI03, 06, 12, 24, 48) per vegetation type; other scales for All only;
        ## best-scale SPEI: one figure, one panel per vegetation type
        outdir = self.outdir + rf'time_series\\'
        for lc in self.class_list:
            self.plot_time_series(self.class_df(df, lc), self.main_list, self.start_year, outdir,
                                  f'Fig_SPEI_long_term_{lc}')
        other_list = [f'SPEI{s:02d}' for s in self.scale_list if f'SPEI{s:02d}' not in self.main_list]
        self.plot_time_series(df, other_list, self.start_year, outdir, 'FigS_SPEI_long_term_other_All')
        df = df.copy()
        best_list = []
        for lc in self.class_list:
            m = np.ones(len(df), dtype=bool) if lc == 'All' else (df['landcover_type'] == lc).values
            df[f'growing_SPEIbest_{lc}'] = np.where(m, df['growing_SPEIbest'], np.nan)
            best_list.append(f'SPEIbest_{lc}')
            self.ylabel_dic[f'SPEIbest_{lc}'] = f'SPEI best scale\n{lc}'
        self.plot_time_series(df, best_list, self.start_year, outdir, 'Fig_SPEIbest_long_term')

    def wmean_series(self, d, col):
        ## area-weighted mean per year
        g = d.assign(wv=d[col] * d['area_weight'], w=d['area_weight'].where(d[col].notna()))
        g = g.groupby('year')[['wv', 'w']].sum()
        return (g['wv'] / g['w']).reindex(self.year_list).values

    def period_table(self, df):
        ## regional mean 1958-1999 vs 2000-2024 (difference, Mann-Whitney p), per scale x vegetation type
        years = np.array(self.year_list)
        result = []
        for lc in self.class_list:
            d = self.class_df(df, lc)
            for var in [f'SPEI{s:02d}' for s in self.scale_list] + ['SPEIbest']:
                y = self.wmean_series(d, f'growing_{var}')
                before, after = y[years < self.split_year], y[years >= self.split_year]
                p = stats.mannwhitneyu(after[np.isfinite(after)], before[np.isfinite(before)])[1]
                result.append([lc, var, np.nanmean(before), np.nanmean(after), np.nanmean(after) - np.nanmean(before), p])
        df_t = pd.DataFrame(result, columns=['class', 'variable', 'mean_1958_1999', 'mean_2000_2024', 'difference',
                                             'p_mannwhitney'])
        df_t.to_csv(self.outdir + 'SPEI_long_term_period_comparison.csv', index=False)
        print(df_t.round(3).to_string())

    def trend(self, arr, years):
        ## OLS slope per decade and two-sided p, per row (rows with NaN -> NaN)
        x = np.array(years, dtype=float)
        xm = x - x.mean()
        ym = arr - arr.mean(axis=1, keepdims=True)
        slope = ym @ xm / np.sum(xm ** 2)
        resid = ym - slope[:, None] * xm[None, :]
        se = np.sqrt(np.sum(resid ** 2, axis=1) / (len(x) - 2) / np.sum(xm ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            p = 2 * stats.t.sf(np.abs(slope / se), len(x) - 2)
        return slope * 10, p

    def trend_maps(self, df):
        ## per scale: trend 1958-2024, trend 2000-2024 (+ p < 0.05) and step change maps; table of mean trends
        from growing_season_extraction import Map_plotter, D
        outdir = self.outdir + rf'trend_map\\'
        T.mk_dir(outdir, force=True)
        mapper = Map_plotter(outdir)
        info = df.groupby('pix')[['area_weight', 'landcover_type']].first()
        w_all = info['area_weight'].values
        var_list = [f'SPEI{s:02d}' for s in self.scale_list] + ['SPEIbest']
        map_var_list = [f'SPEI{s:02d}' for s in self.map_scale_list] + ['SPEIbest']
        res = {}
        for var in var_list:
            mat = df.pivot(index='pix', columns='year', values=f'growing_{var}').reindex(index=info.index,
                                                                                        columns=self.year_list)
            years = np.array(self.year_list)
            for t0, t1 in [(1958, 2024), (2000, 2024)]:
                m = (years >= t0) & (years <= t1)
                res[(var, f'{t0}_{t1}')] = self.trend(mat.values[:, m], years[m])
            res[(var, 'step')] = (np.nanmean(mat.values[:, years >= self.split_year], axis=1) -
                                  np.nanmean(mat.values[:, years < self.split_year], axis=1), None)
        ## maps: one colour range per map type, shared by all scales
        for key, label in [('1958_2024', 'trend 1958-2024 (per decade)'), ('2000_2024', 'trend 2000-2024 (per decade)'),
                           ('step', 'mean 2000-2024 minus mean 1958-1999')]:
            vlim = np.nanpercentile(np.abs(np.concatenate([res[(v, key)][0] for v in map_var_list])), 90)
            for var in map_var_list:
                vals, p = res[(var, key)]
                maps = [(vals, f'GS_{var}_{key}')]
                if p is not None:
                    maps.append((np.where(p < 0.05, vals, np.nan), f'GS_{var}_{key}_sig'))
                for v, fname in maps:
                    arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, v)))
                    arr[arr <= -9999] = np.nan
                    D.arr_to_tif(arr.copy(), outdir + f'{fname}.tif')
                    ok = np.isfinite(v)
                    mapper.plot_map(arr, fname, 'RdBu', -vlim, vlim, f'Growing-season {var} {label}', v[ok], w_all[ok])
        ## table: area-weighted mean and area with significant decrease / increase
        result = []
        for lc in self.class_list:
            mlc = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
            for var in var_list:
                for key in ['1958_2024', '2000_2024', 'step']:
                    vals, p = res[(var, key)]
                    m = mlc & np.isfinite(vals)
                    w = w_all[m]
                    row = [lc, var, key, m.sum(), np.sum(vals[m] * w) / np.sum(w)]
                    if p is None:
                        row += [np.nan, np.nan]
                    else:
                        row += [np.sum(w * ((vals[m] < 0) & (p[m] < 0.05))) / np.sum(w) * 100,
                                np.sum(w * ((vals[m] > 0) & (p[m] < 0.05))) / np.sum(w) * 100]
                    result.append(row)
        df_t = pd.DataFrame(result, columns=['class', 'variable', 'period', 'n_pix', 'mean_value',
                                             'sig_decrease_area_%', 'sig_increase_area_%'])
        df_t.to_csv(self.outdir + 'SPEI_long_term_trend_summary.csv', index=False)
        print(df_t[df_t['class'] == 'All'].round(3).to_string())


## add a new variable: copy SNU_LAI_time_series, change outdir / start_year / var_list




def main():

    # SNU_LAI_time_series().run()
    # Daymet_time_series().run()
    # MODIS_LAI_time_series().run()

    # SPEI_time_series_MODIS().run()     ## modis record
    # MODIS_LAI_time_series().run()   ## growing-season MODIS LAI (GS.df)
    SPEI_long_term_plot().run()     ## growing-season SPEI 1958-2024: time series and trend maps


if __name__ == '__main__':
    main()
