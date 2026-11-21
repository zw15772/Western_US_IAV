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
        df = df[df['landcover_type'].isin(['Forest', 'Shrub', 'Grass', 'Savanna'])]   ## same pixels as the other figures (2026-11-20)
        print('pixels (four vegetation types):', df['pix'].nunique())
        self.ylabel_dic = {'LAI_rel': 'Relative Change (%)'}
        self.plot_time_series(df, ['LAI_rel'], 2003, self.outdir + rf'\\', 'Fig_MODIS_LAI_GS')
        self.trend_bar(df)   ## 2026-11-20: LAI trend per vegetation type, bar plot
        pass

    def trend_bar(self, df, var='growing_LAI_rel'):
        ## per pixel OLS trend of the growing-season LAI (% of the pixel's 2003-2024 mean per decade),
        ## area-weighted mean per vegetation type with 95% CI (1 x 1 deg block bootstrap, 200 draws)
        ## output: Fig_MODIS_LAI_GS_trend_bar.png / .pdf / .csv in self.outdir
        class_list = ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']
        color_dic = {'All': '#555555', 'Forest': '#1B5E20', 'Shrub': '#B8860B', 'Grass': '#7CB342', 'Savanna': '#8D6E63'}
        years = np.arange(2003, 2025)
        df = df[df['landcover_type'].isin(class_list[1:])]
        mat = df.pivot_table(index='pix', columns='year', values=var).reindex(columns=years)
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first().loc[mat.index]
        x = years.astype(float) - years.mean()
        y = mat.values
        ok = np.isfinite(y).all(axis=1)
        y, info = y[ok], info[ok]
        ym = y - y.mean(axis=1, keepdims=True)
        slope = ym @ x / np.sum(x ** 2)
        resid = ym - slope[:, None] * x[None, :]
        se = np.sqrt(np.sum(resid ** 2, axis=1) / (len(x) - 2) / np.sum(x ** 2))
        p = 2 * stats.t.sf(np.abs(slope / se), len(x) - 2)
        trend = slope * 10
        result = []
        for lc in class_list:
            m = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
            w = info['area_weight'].values[m]
            v = trend[m]
            lo, hi = self.block_ci_bar(info['lon'].values[m], info['lat'].values[m], v, w)
            result.append([lc, m.sum(), np.sum(v * w) / np.sum(w), lo, hi,
                           np.sum(w * ((trend[m] > 0) & (p[m] < 0.05))) / np.sum(w) * 100,
                           np.sum(w * ((trend[m] < 0) & (p[m] < 0.05))) / np.sum(w) * 100])
        tab = pd.DataFrame(result, columns=['class', 'n_pix', 'trend_per_decade', 'ci_low', 'ci_high',
                                            'sig_increase_area_%', 'sig_decrease_area_%'])
        tab.to_csv(self.outdir + 'Fig_MODIS_LAI_GS_trend_bar.csv', index=False)
        print(tab.round(2).to_string())
        fig, ax = plt.subplots(figsize=(self.map_width * 0.9, self.map_height * 0.8))
        xx = np.arange(len(tab))
        err = np.array([tab['trend_per_decade'] - tab['ci_low'], tab['ci_high'] - tab['trend_per_decade']])
        ax.bar(xx, tab['trend_per_decade'], color=[color_dic[c] for c in tab['class']], width=0.65,
               yerr=err, capsize=3, error_kw={'lw': 0.8})
        for xi, v, lo, hi in zip(xx, tab['trend_per_decade'], tab['ci_low'], tab['ci_high']):
            ax.text(xi, hi + 0.2 if v >= 0 else lo - 0.2, f'{v:+.2f}', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=7)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(xx)
        ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(tab['class'], tab['n_pix'])], fontsize=7)
        ax.set_ylabel('LAI trend (% per decade)', fontsize=9)
        ax.set_title('Growing-season LAI trend, 2003-2024', fontsize=9)
        lo_lim = min(0, tab['ci_low'].min()) * 1.25
        hi_lim = max(0, tab['ci_high'].max()) * 1.25
        ax.set_ylim(lo_lim - 0.5, hi_lim + 0.5)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(self.outdir + 'Fig_MODIS_LAI_GS_trend_bar.png', dpi=300)
        plt.savefig(self.outdir + 'Fig_MODIS_LAI_GS_trend_bar.pdf')
        plt.close()

    def block_ci_bar(self, lon, lat, v, w):
        block = (np.floor(lon) * 1000 + np.floor(lat)).astype(int)
        ids = np.unique(block)
        idx = {b: np.where(block == b)[0] for b in ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(200):
            pick = np.concatenate([idx[b] for b in rng.choice(ids, size=len(ids))])
            boot.append(np.sum(v[pick] * w[pick]) / np.sum(w[pick]))
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    def mask_low_LAI(self, df):
        ## df has no raw MODIS LAI, so take the multi-year mean (2003-2024) from the npy
        for season in self.season_list:
            val_dic = T.load_npy(result_root + rf'\MODIS_LAI\{season}_LAI.npy')
            mean_dic = {pix: np.nanmean(np.array(vals, dtype=float)) for pix, vals in val_dic.items()}
            pix_mean = df['pix'].map(mean_dic)
            df.loc[~(pix_mean >= self.LAI_threshold), f'{season}_MODIS_LAI_relative_change'] = np.nan
        return df


class Daymet_time_series(Time_series_base):
    ## rainfall structure from Daymet daily precipitation in each pixel's main growing season, 2003-2024 (2026-11-15)
    ## data: Data_processing.py -> Data_processing_Daymet.daymet_gs_metrics -> Result\Dataframe\Daymet\Daymet_GS_metrics.df
    ##   GS_*   : growing season (gs_start - gs_end of growing_season.df)
    ##   Pant_* : 8 months before the growing season (same window as GS_P_antecedent)
    ##   Ptot (mm), wetdays (days >= 1 mm), R5 / R10 / R20 (days >= 5 / 10 / 20 mm), SDII (mm per wet day),
    ##   heavy_frac (% of the total from days >= 10 mm), CDD (longest dry spell, days), GS_early_share (% in the first
    ##   half of the growing season)
    ## pixels: growing-season pixels (n_season > 0, window within one calendar year, GS mean LAI >= 0.1), US only
    ## time series: area-weighted mean +- std per year, trend 2003-2024, per vegetation type
    ## maps: per-pixel OLS trend 2003-2024 (Ptot in % of the pixel mean per decade, others in units per decade) and
    ##   p < 0.05 only; table: area-weighted mean trend (95% CI, 1 x 1 deg block bootstrap) and significant area
    ## output: Result2\01_drought\Daymet_timeseries\growing_season\
    def __init__(self):
        super().__init__()
        self.dff = result_root + rf'\Dataframe\Daymet\Daymet_GS_metrics.df'
        self.gs_dff = result_root + rf'\Dataframe\growing_season\growing_season.df'
        self.outdir = self.result2 + rf'01_drought\Daymet_timeseries\growing_season\\'
        T.mk_dir(self.outdir, force=True)
        self.start_year = 2003
        self.split_year = 2003   ## no grey shading
        self.year_list = list(range(2003, 2025))
        self.season_list = ['GS', 'Pant']
        self.color_dic = {'GS': '#2E7D32', 'Pant': '#1F618D'}
        self.trend_period_list = [(2003, 2024, '-')]
        self.class_list = ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']
        self.main_list = ['Ptot', 'wetdays', 'SDII', 'heavy_frac', 'CDD']
        self.other_list = ['R5', 'R10', 'R20', 'early_share']
        self.ylabel_dic = {'Ptot': 'Ptot (mm)', 'wetdays': 'Wet days (>= 1 mm)', 'SDII': 'SDII (mm/wet day)',
                           'heavy_frac': 'Heavy (>= 10 mm) share (%)', 'CDD': 'CDD (days)', 'R5': 'R5 (days)',
                           'R10': 'R10 (days)', 'R20': 'R20 (days)', 'early_share': 'Early-GS share (%)'}
        pass

    def run(self):
        df = self.load_df()
        ## one figure per window (GS, Pant) and vegetation type
        for season in ['GS', 'Pant']:
            self.season_list = [season]
            for lc in self.class_list:
                d = df if lc == 'All' else df[df['landcover_type'] == lc]
                self.plot_time_series(d, self.main_list, self.start_year, self.outdir + rf'time_series\\',
                                      f'Fig_Daymet_{season}_{lc}')
            self.plot_time_series(df, self.other_list, self.start_year, self.outdir + rf'time_series\\',
                                  f'FigS_Daymet_{season}_other_All')
        self.season_list = ['GS', 'Pant']
        self.trend_maps(df)
        pass

    def load_df(self):
        df = T.load_df(self.dff)
        gs = T.load_df(self.gs_dff)
        info = gs[['lon', 'lat', 'area_weight', 'landcover_type']]
        df = df.join(info, on='pix')
        df = df[(df['year'] >= 2003) & (df['year'] <= 2024)]
        print('pixels:', df['pix'].nunique())
        return df

    def trend(self, arr):
        ## OLS slope per decade and two-sided p, per row (rows with NaN -> NaN)
        x = np.array(self.year_list, dtype=float)
        xm = x - x.mean()
        ym = arr - arr.mean(axis=1, keepdims=True)
        slope = ym @ xm / np.sum(xm ** 2)
        resid = ym - slope[:, None] * xm[None, :]
        se = np.sqrt(np.sum(resid ** 2, axis=1) / (len(x) - 2) / np.sum(xm ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            p = 2 * stats.t.sf(np.abs(slope / se), len(x) - 2)
        return slope * 10, p

    def block_ci(self, lon, lat, v, w):
        block = (np.floor(lon) * 1000 + np.floor(lat)).astype(int)
        ids = np.unique(block)
        idx = {b: np.where(block == b)[0] for b in ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(200):
            pick = np.concatenate([idx[b] for b in rng.choice(ids, size=len(ids))])
            boot.append(np.sum(v[pick] * w[pick]) / np.sum(w[pick]))
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    def trend_maps(self, df):
        from growing_season_extraction import Map_plotter, D
        outdir = self.outdir + rf'trend_map\\'
        T.mk_dir(outdir, force=True)
        mapper = Map_plotter(outdir)
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
        w_all = info['area_weight'].values
        result = []
        for season in self.season_list:
            for var in self.main_list + self.other_list:
                col = f'{season}_{var}'
                if col not in df.columns:
                    continue
                mat = df.pivot(index='pix', columns='year', values=col).reindex(index=info.index,
                                                                                columns=self.year_list).values
                unit = 'per decade'
                if var == 'Ptot':
                    mean = np.nanmean(mat, axis=1, keepdims=True)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        mat = np.where(mean > 0, (mat / mean - 1) * 100, np.nan)
                    unit = '% per decade'
                tr, p = self.trend(mat)
                vlim = np.nanpercentile(np.abs(tr), 90)
                cmap = 'RdBu_r' if var == 'CDD' else 'RdBu'   ## blue = wetter
                for vals, fname in [(tr, f'{col}_trend_2003_2024'), (np.where(p < 0.05, tr, np.nan),
                                                                     f'{col}_trend_sig_2003_2024')]:
                    arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, vals)))
                    arr[arr <= -9999] = np.nan
                    D.arr_to_tif(arr.copy(), outdir + f'{fname}.tif')
                    ok = np.isfinite(vals)
                    mapper.plot_map(arr, fname, cmap, -vlim, vlim, f'{col} trend ({unit})', vals[ok], w_all[ok])
                for lc in self.class_list:
                    m = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
                    m = m & np.isfinite(tr)
                    if m.sum() < 30:
                        continue
                    w = w_all[m]
                    lo, hi = self.block_ci(info['lon'].values[m], info['lat'].values[m], tr[m], w)
                    result.append([season, var, unit, lc, m.sum(), np.sum(tr[m] * w) / np.sum(w), lo, hi,
                                   np.sum(w * ((tr[m] < 0) & (p[m] < 0.05))) / np.sum(w) * 100,
                                   np.sum(w * ((tr[m] > 0) & (p[m] < 0.05))) / np.sum(w) * 100])
        df_t = pd.DataFrame(result, columns=['window', 'variable', 'unit', 'class', 'n_pix', 'trend', 'ci_low',
                                             'ci_high', 'sig_decrease_area_%', 'sig_increase_area_%'])
        df_t.to_csv(self.outdir + 'Daymet_GS_trend_summary_2003_2024.csv', index=False)
        print(df_t[df_t['class'].isin(['All', 'Grass'])].round(2).to_string())


class SPEI_long_term_plot(Time_series_base):
    ## growing-season SPEI 1958-2024 (2026-11-15)
    ## per pixel: SPEI03 ... SPEI48 averaged over the pixel's main growing season (growing_season.df, same window
    ##   every year). Long-term series and trends are NOT detrended.
    ## pixels: same growing-season pixels as GS.df (n_season > 0, window within one calendar year, GS mean LAI >= 0.1)
    ## periods: 1958-2024 (long term) and 2003-2024 (= MODIS period, same as the LAI analysis)
    ## time series: area-weighted mean +- std, trends 1958-2024 (dotted) and 2003-2024 (solid), per vegetation type
    ##   main figure SPEI03, 06, 12, 24, 48
    ## maps per scale: trend 1958-2024 (+ p < 0.05 only), trend 2003-2024 (+ p < 0.05 only) and the differences:
    ##   mean_diff  = mean 2003-2024 - mean 1958-2002   (shift of the level)
    ##   trend_diff = trend 2003-2024 - trend 1958-2024 (change of the rate)
    ## tables: period comparison and area-weighted trends per scale x vegetation type
    ## output: Result2\01_drought\SPEI_long_term\
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'01_drought\SPEI_long_term\\'
        T.mk_dir(self.outdir, force=True)
        self.start_year = 1958
        self.split_year = 2003   ## start of the recent period (MODIS)
        self.year_list = list(range(1958, 2025))
        self.scale_list = list(range(3, 49, 3))
        self.map_scale_list = [3, 6, 9, 12, 24, 36, 48]
        self.main_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI48']   ## main figure, same as SPEI_time_series
        self.class_list = ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']
        self.season_list = ['growing']
        self.color_dic = {'growing': '#D9480F'}   ## orange-red
        self.trend_period_list = [(1958, 2024, ':'), (2003, 2024, '-')]
        self.gs_dff = result_root + rf'\Dataframe\growing_season\growing_season.df'
        self.dff_out = result_root + rf'\Dataframe\SPEI_long_term\SPEI_GS_1958_2024.df'  ## pixel x year growing-season SPEI
        pass

    def run(self):
        if not os.path.isfile(self.dff_out):
            self.build_df()
        df = T.load_df(self.dff_out)
        ## same pixels as the LAI analysis: the four vegetation types only ('Other' IGBP classes left out)
        df = df[df['landcover_type'].isin(self.class_list[1:])]
        print('pixels (four vegetation types):', df['pix'].nunique())
        self.time_series(df)
        self.period_table(df)
        self.trend_maps(df)
        pass

    def build_df(self):
        ## pixel x year growing-season SPEI 1958-2024 for every scale
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
        df = pd.DataFrame({'pix': [p for p in pix_list for _ in range(n_year)], 'year': self.year_list * n_pix})
        for col, arr in out.items():
            df[col] = arr.ravel()
        for col in ['lon', 'lat', 'area_weight', 'landcover_type', 'gs_start', 'gs_end']:
            df[col] = np.repeat(gs[col].values, n_year)
        T.mk_dir(os.path.dirname(self.dff_out), force=True)
        T.save_df(df, self.dff_out)
        T.df_to_excel(df, self.dff_out.replace('.df', '.xlsx'))
        print('pixels:', n_pix)

    def class_df(self, df, lc):
        return df if lc == 'All' else df[df['landcover_type'] == lc]

    def time_series(self, df):
        ## main figure (SPEI03, 06, 12, 24, 48) per vegetation type; other scales for All only
        outdir = self.outdir + rf'time_series\\'
        for lc in self.class_list:
            self.plot_time_series(self.class_df(df, lc), self.main_list, self.start_year, outdir,
                                  f'Fig_SPEI_long_term_{lc}')
        other_list = [f'SPEI{s:02d}' for s in self.scale_list if f'SPEI{s:02d}' not in self.main_list]
        self.plot_time_series(df, other_list, self.start_year, outdir, 'FigS_SPEI_long_term_other_All')

    def wmean_series(self, d, col):
        ## area-weighted mean per year
        g = d.assign(wv=d[col] * d['area_weight'], w=d['area_weight'].where(d[col].notna()))
        g = g.groupby('year')[['wv', 'w']].sum()
        return (g['wv'] / g['w']).reindex(self.year_list).values

    def period_table(self, df):
        ## regional mean 1958-2002 vs 2003-2024 (difference, Mann-Whitney p), per scale x vegetation type
        years = np.array(self.year_list)
        result = []
        for lc in self.class_list:
            d = self.class_df(df, lc)
            for var in [f'SPEI{s:02d}' for s in self.scale_list]:
                y = self.wmean_series(d, f'growing_{var}')
                before, after = y[years < self.split_year], y[years >= self.split_year]
                p = stats.mannwhitneyu(after[np.isfinite(after)], before[np.isfinite(before)])[1]
                result.append([lc, var, np.nanmean(before), np.nanmean(after), np.nanmean(after) - np.nanmean(before), p])
        df_t = pd.DataFrame(result, columns=['class', 'variable', 'mean_1958_2002', 'mean_2003_2024', 'difference',
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

    def trend_nan(self, arr, years, min_frac=0.9):
        ## OLS slope per decade and two-sided p per row, on the years with data;
        ## NaN if a row has data in fewer than min_frac of the years
        ## (SPEI12 starts in 1959 and SPEI48 in 1962, and some pixels miss single years)
        x = np.array(years, dtype=float)
        ok = np.isfinite(arr)
        n = ok.sum(axis=1).astype(float)
        X = np.where(ok, x[None, :], 0.)
        Y = np.where(ok, arr, 0.)
        with np.errstate(divide='ignore', invalid='ignore'):
            xm = X.sum(axis=1) / n
            ym = Y.sum(axis=1) / n
            dx = np.where(ok, x[None, :] - xm[:, None], 0.)
            dy = np.where(ok, arr - ym[:, None], 0.)
            sxx = np.sum(dx ** 2, axis=1)
            slope = np.sum(dx * dy, axis=1) / sxx
            resid = np.where(ok, dy - slope[:, None] * dx, 0.)
            se = np.sqrt(np.sum(resid ** 2, axis=1) / (n - 2) / sxx)
            p = 2 * stats.t.sf(np.abs(slope / se), n - 2)
        bad = n < min_frac * len(x)
        slope[bad], p[bad] = np.nan, np.nan
        return slope * 10, p

    def trend_maps(self, df):
        ## per scale: trend 1958-2024 and 2003-2024 (+ p < 0.05), mean_diff and trend_diff maps; table
        from growing_season_extraction import Map_plotter, D
        outdir = self.outdir + rf'trend_map\\'
        T.mk_dir(outdir, force=True)
        mapper = Map_plotter(outdir)
        info = df.groupby('pix')[['area_weight', 'landcover_type']].first()
        w_all = info['area_weight'].values
        var_list = [f'SPEI{s:02d}' for s in self.scale_list]
        map_var_list = [f'SPEI{s:02d}' for s in self.map_scale_list]
        years = np.array(self.year_list)
        res = {}
        for var in var_list:
            mat = df.pivot(index='pix', columns='year', values=f'growing_{var}').reindex(index=info.index,
                                                                                        columns=self.year_list).values
            for t0, t1 in [(1958, 2024), (2003, 2024)]:
                m = (years >= t0) & (years <= t1)
                res[(var, f'{t0}_{t1}')] = self.trend_nan(mat[:, m], years[m])   ## skips missing years (long scales start later)
            res[(var, 'mean_diff')] = (np.nanmean(mat[:, years >= self.split_year], axis=1) -
                                       np.nanmean(mat[:, years < self.split_year], axis=1), None)
            res[(var, 'trend_diff')] = (res[(var, '2003_2024')][0] - res[(var, '1958_2024')][0], None)
        key_list = [('1958_2024', 'trend 1958-2024 (per decade)'), ('2003_2024', 'trend 2003-2024 (per decade)'),
                    ('mean_diff', 'mean 2003-2024 minus mean 1958-2002'),
                    ('trend_diff', 'trend 2003-2024 minus trend 1958-2024 (per decade)')]
        ## maps: one colour range per map type, shared by all scales
        for key, label in key_list:
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
        ## table: area-weighted mean and area with significant decrease / increase (trends only)
        result = []
        for lc in self.class_list:
            mlc = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
            for var in var_list:
                for key, _ in key_list:
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
    # Daymet_time_series().run()      ## 2026-11-15: rainfall structure in the growing season (Daymet daily)
    # MODIS_LAI_time_series().run()

    # SPEI_time_series_MODIS().run()     ## modis record
    MODIS_LAI_time_series().run()   ## growing-season MODIS LAI (GS.df); 2026-11-20: + trend bar per vegetation type
    # SPEI_long_term_plot().run()     ## growing-season SPEI 1958-2024: time series and trend maps


if __name__ == '__main__':
    main()
