from __Global__ import *
from SPEI_LAI_correlation import SPEI_SNU_LAI_correlation, D   ## reuse df_clean / to_matrix / detrend_rows / plot_map


class SPEI_background_sensitivity(SPEI_SNU_LAI_correlation):
    ## method 1: response of LAI anomaly to current-season SPEI03, grouped by background state (previous year's SPEI48)
    ## LAI anomaly = per-pixel linearly detrended SNU LAI, as % of the pixel's multi-year mean
    ## SPEI03 and SPEI48 (background) are raw values, NOT detrended (drought year = raw SPEI03 < -1)
    ## three metrics per background group:
    ##   'slope'   = sensitivity, area-weighted pooled regression of LAI anomaly on SPEI03 (all years)
    ##   'drought' = mean LAI anomaly in drought years (SPEI03 < -1)
    ##   'wet'     = mean LAI anomaly in wet years (SPEI03 > 1)
    ## 95% CI from bootstrapping years (same resamples for all groups, so Dry - Wet can be tested)
    def __init__(self):
        super().__init__()   ## same dff / years / LAI and cropland thresholds / df_clean as the correlation
        self.outdir = this_root + rf'\Result2\03_response\SPEI_background_sensitivity\\'
        T.mk_dir(self.outdir, force=True)
        self.season_list = ['summer']
        self.short_var = 'SPEI03'        ## short-term drought: current season
        self.short_col = 'SPEI_short'    ## raw SPEI03; 'SPEI_short_detrend' = per-pixel detrended
        self.background_list = ['SPEI12', 'SPEI24', 'SPEI36', 'SPEI48']   ## background: previous year, same season
        self.background_var = 'SPEI48'   ## current background in the loop (set in run)
        self.drought_threshold = -1      ## SPEI03 < -1: drought year
        self.wet_year_threshold = 1      ## SPEI03 > 1: wet year
        ## grouping: fixed thresholds, SPEI48 > 1 Wet, -1 to 1 Normal, < -1 Dry
        self.group_method_list = ['fixed']
        self.wet_threshold = 1
        self.dry_threshold = -1
        self.group_list = ['Wet', 'Normal', 'Dry']
        self.group_color_dic = {'Wet': '#2166ac', 'Normal': 'grey', 'Dry': '#b2182b'}
        self.period_list = [(1983, 2024), (2000, 2024)]   ## 1982 has no previous-year SPEI48; groups re-split per period
        self.metric_list = ['slope', 'drought', 'wet']
        self.n_boot = 1000
        pass

    def run(self):
        # df = T.load_df(self.dff)   ## only needed when rebuilding pixel_year.df
        # df = self.df_clean(df)
        # df_pair = self.build_pixel_year(df)
        # T.save_df(df_pair, self.outdir + 'pixel_year.df')
        df_pair = T.load_df(self.outdir + 'pixel_year.df')
        df_pair = self.detrend_short(df_pair)
        for background_var in self.background_list:
            self.background_var = background_var   ## labels, grouping column and file names follow this
            outdir_bg = self.outdir + f'{background_var}\\'
            T.mk_dir(outdir_bg, force=True)
            ## ---- main text: unpaired comparison of background groups ----
            df_result = self.cal_response(df_pair)
            df_result.to_csv(outdir_bg + 'response.csv', index=False)
            for method in self.group_method_list:
                self.plot_response(df_pair, df_result, method)
            df_lc_result = self.cal_response_by_landcover(df_pair)
            df_lc_result.to_csv(outdir_bg + 'response_by_landcover.csv', index=False)
            self.plot_response_by_landcover(df_lc_result)
            ## ---- SI: paired test within the same pixel (removes differences between regions) ----
            df_pix, df_paired = self.cal_paired_SI(df_pair)
            df_paired.to_csv(outdir_bg + 'SI_paired_dry_minus_wet.csv', index=False)
            print(background_var)
            print(df_paired)
            self.plot_paired_SI(df_pix, df_paired)
        pass

    def build_pixel_year(self, df):
        ## one row per pixel-year: LAI anomaly, current SPEI03 (raw), previous-year SPEI48
        result_list = []
        for season in self.season_list:
            lai = self.to_matrix(df, f'{season}_{self.lai_var}')
            lai = lai[lai.mean(axis=1) >= self.LAI_threshold]
            lai_mean = lai.mean(axis=1).values
            anomaly = self.detrend_rows(lai.values) / lai_mean[:, None] * 100   ## detrended, % of mean
            df_anomaly = pd.DataFrame(anomaly, index=lai.index, columns=lai.columns)
            df_anomaly = df_anomaly.stack().reset_index()
            df_anomaly.columns = ['pix', 'year', 'LAI_anomaly']

            df_short = df[['pix', 'year', 'area_weight', self.landcover_col, f'{season}_{self.short_var}']]
            df_short = df_short.rename(columns={f'{season}_{self.short_var}': 'SPEI_short'})
            bg_col_list = [f'{season}_{bg}' for bg in self.background_list]
            df_bg = df[['pix', 'year'] + bg_col_list].copy()
            df_bg['year'] = df_bg['year'] + 1   ## previous year's value lines up with the current year
            df_bg = df_bg.rename(columns={f'{season}_{bg}': f'bg_{bg}' for bg in self.background_list})

            df_s = df_anomaly.merge(df_short, on=['pix', 'year']).merge(df_bg, on=['pix', 'year'])
            df_s = df_s.dropna(subset=['LAI_anomaly', 'SPEI_short', 'area_weight'])
            df_s['season'] = season
            result_list.append(df_s)
        df_pair = pd.concat(result_list, ignore_index=True)
        return df_pair

    def detrend_short(self, df_pair):
        ## per-pixel linear detrend of SPEI03 (same as the correlation analysis)
        result_list = []
        for season in self.season_list:
            df_s = df_pair[df_pair['season'] == season]
            spei = self.to_matrix(df_s, 'SPEI_short')
            df_d = pd.DataFrame(self.detrend_rows(spei.values), index=spei.index, columns=spei.columns)
            df_d = df_d.stack().reset_index()
            df_d.columns = ['pix', 'year', 'SPEI_short_detrend']
            df_s = df_s.merge(df_d, on=['pix', 'year'])
            result_list.append(df_s)
        df_pair = pd.concat(result_list, ignore_index=True)
        return df_pair

    def add_group(self, df_p):
        ## background group within the selected period
        df_p = df_p.copy()
        df_p['group'] = 'Normal'
        bg = df_p[f'bg_{self.background_var}']   ## previous-year background SPEI, raw
        df_p.loc[bg > self.wet_threshold, 'group'] = 'Wet'
        df_p.loc[bg < self.dry_threshold, 'group'] = 'Dry'
        df_p.loc[bg.isna(), 'group'] = np.nan
        return df_p

    def select_years(self, df_g, metric):
        x = df_g[self.short_col]
        if metric == 'drought':
            return df_g[x < self.drought_threshold]
        if metric == 'wet':
            return df_g[x > self.wet_year_threshold]
        return df_g

    def year_stats(self, df_g, years):
        ## weighted sums per year: w, wx, wy, wxx, wxy (x = SPEI03, y = LAI anomaly); bootstrap resamples these rows
        w = df_g['area_weight'].values
        x = df_g[self.short_col].values
        y = df_g['LAI_anomaly'].values
        df_tmp = pd.DataFrame({'year': df_g['year'].values, 'w': w, 'wx': w * x, 'wy': w * y,
                               'wxx': w * x * x, 'wxy': w * x * y})
        df_sum = df_tmp.groupby('year').sum().reindex(years).fillna(0)
        return df_sum[['w', 'wx', 'wy', 'wxx', 'wxy']].values

    def value_from_sums(self, s, metric):
        ## slope: weighted OLS slope (and intercept); drought / wet: weighted mean LAI anomaly
        sw, sx, sy, sxx, sxy = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
        if metric == 'slope':
            slope = (sxy - sx * sy / sw) / (sxx - sx * sx / sw)
            intercept = (sy - slope * sx) / sw
            return slope, intercept
        return sy / sw, np.nan * sw

    def cal_response(self, df_pair):
        rng = np.random.default_rng(42)
        result_list = []
        for season in self.season_list:
            for method in self.group_method_list:
                for start, end in self.period_list:
                    df_p = df_pair[(df_pair['season'] == season) & (df_pair['year'] >= start) & (df_pair['year'] <= end)]
                    df_p = self.add_group(df_p)
                    years = np.arange(start, end + 1)
                    idx = rng.integers(0, len(years), (self.n_boot, len(years)))   ## same year resamples for all groups
                    for metric in self.metric_list:
                        boot_dic = {}
                        for group in self.group_list:
                            df_g = self.select_years(df_p[df_p['group'] == group], metric)
                            sums = self.year_stats(df_g, years)
                            value, intercept = self.value_from_sums(sums.sum(axis=0), metric)
                            value_boot, _ = self.value_from_sums(sums[idx].sum(axis=1), metric)
                            boot_dic[group] = value_boot
                            result_list.append([season, method, f'{start}-{end}', metric, group, value, intercept,
                                                np.nanpercentile(value_boot, 2.5), np.nanpercentile(value_boot, 97.5),
                                                len(df_g), np.nan])
                        ## Dry vs Wet, one-sided bootstrap p for a stronger response under a dry background:
                        ## slope: Dry steeper; drought: Dry more negative; wet: Dry more positive
                        diff = boot_dic['Dry'] - boot_dic['Wet']
                        diff = diff[np.isfinite(diff)]   ## empty group -> no test
                        if len(diff) == 0:
                            p = np.nan
                        elif metric == 'drought':
                            p = np.mean(diff >= 0)
                        else:
                            p = np.mean(diff <= 0)
                        result_list.append([season, method, f'{start}-{end}', metric, 'Dry-Wet',
                                            np.nanmedian(diff), np.nan,
                                            np.nanpercentile(diff, 2.5), np.nanpercentile(diff, 97.5), np.nan, p])
        df_result = pd.DataFrame(result_list, columns=['season', 'method', 'period', 'metric', 'group', 'value',
                                                       'intercept', 'ci_low', 'ci_high', 'n_pixel_year',
                                                       'p_dry_stronger'])
        return df_result

    def cal_response_by_landcover(self, df_pair):
        ## same three metrics, separately for each vegetation type; first period only (1983-2024), fixed +-1 groups
        rng = np.random.default_rng(42)
        start, end = self.period_list[0]
        years = np.arange(start, end + 1)
        idx = rng.integers(0, len(years), (self.n_boot, len(years)))   ## same year resamples for all groups
        result_list = []
        for season in self.season_list:
            df_p = df_pair[(df_pair['season'] == season) & (df_pair['year'] >= start) & (df_pair['year'] <= end)]
            df_p = self.add_group(df_p)
            for lc in self.landcover_list:
                df_lc = df_p[df_p[self.landcover_col] == lc]
                for metric in self.metric_list:
                    boot_dic = {}
                    for group in self.group_list:
                        df_g = self.select_years(df_lc[df_lc['group'] == group], metric)
                        sums = self.year_stats(df_g, years)
                        value, _ = self.value_from_sums(sums.sum(axis=0), metric)
                        value_boot, _ = self.value_from_sums(sums[idx].sum(axis=1), metric)
                        boot_dic[group] = value_boot
                        result_list.append([season, lc, metric, group, value,
                                            np.nanpercentile(value_boot, 2.5), np.nanpercentile(value_boot, 97.5),
                                            len(df_g), np.nan])
                    diff = boot_dic['Dry'] - boot_dic['Wet']
                    diff = diff[np.isfinite(diff)]   ## empty group -> no test
                    if len(diff) == 0:
                        p = np.nan
                    elif metric == 'drought':
                        p = np.mean(diff >= 0)
                    else:
                        p = np.mean(diff <= 0)
                    result_list.append([season, lc, metric, 'Dry-Wet', np.nanmedian(diff),
                                        np.nanpercentile(diff, 2.5), np.nanpercentile(diff, 97.5), np.nan, p])
        df_lc_result = pd.DataFrame(result_list, columns=['season', 'landcover', 'metric', 'group', 'value',
                                                          'ci_low', 'ci_high', 'n_pixel_year', 'p_dry_stronger'])
        return df_lc_result

    def plot_response_by_landcover(self, df_lc_result):
        ## x = vegetation type, one marker per background group; * = Dry vs Wet p < 0.05
        start, end = self.period_list[0]
        label_dic = {'Wet': f'Wet (> {self.wet_threshold})',
                     'Normal': f'Normal ({self.dry_threshold} to {self.wet_threshold})',
                     'Dry': f'Dry (< {self.dry_threshold})'}
        title_dic = {'slope': f'(a) Sensitivity: slope of LAI anomaly on {self.short_var}',
                     'drought': f'(b) Drought years ({self.short_var} < {self.drought_threshold})',
                     'wet': f'(c) Wet years ({self.short_var} > {self.wet_year_threshold})'}
        ylabel_dic = {'slope': f'% LAI per unit {self.short_var}', 'drought': 'Mean LAI anomaly (%)',
                      'wet': 'Mean LAI anomaly (%)'}
        offset_dic = {'Wet': -0.22, 'Normal': 0, 'Dry': 0.22}
        for season in self.season_list:
            fig, axes = plt.subplots(1, 3, figsize=(24 * centimeter_factor, 7.5 * centimeter_factor))
            for ax, metric in zip(axes, self.metric_list):
                df_m = df_lc_result[(df_lc_result['season'] == season) & (df_lc_result['metric'] == metric)]
                for i, lc in enumerate(self.landcover_list):
                    for group in self.group_list:
                        row = df_m[(df_m['landcover'] == lc) & (df_m['group'] == group)].iloc[0]
                        if row['n_pixel_year'] < 1000:   ## too few pixel-years to show
                            continue
                        ax.errorbar(i + offset_dic[group], row['value'],
                                    yerr=[[row['value'] - row['ci_low']], [row['ci_high'] - row['value']]],
                                    fmt='o', ms=4, color=self.group_color_dic[group], capsize=2,
                                    label=label_dic[group] if i == 0 else None)
                    n_dry = df_m[(df_m['landcover'] == lc) & (df_m['group'] == 'Dry')]['n_pixel_year'].iloc[0]
                    n_wet = df_m[(df_m['landcover'] == lc) & (df_m['group'] == 'Wet')]['n_pixel_year'].iloc[0]
                    row = df_m[(df_m['landcover'] == lc) & (df_m['group'] == 'Dry-Wet')].iloc[0]
                    if row['p_dry_stronger'] < 0.05 and n_dry >= 1000 and n_wet >= 1000:
                        ax.text(i, 1.0, '*', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                                fontsize=12)
                ax.axhline(0, color='grey', lw=0.5)
                ax.set_xticks(range(len(self.landcover_list)))
                ax.set_xticklabels(self.landcover_list, fontsize=8)
                ax.set_xlim(-0.5, len(self.landcover_list) - 0.5)
                ax.set_ylabel(ylabel_dic[metric])
                ax.set_title(title_dic[metric] + '\n', fontsize=8, loc='left')
            axes[0].legend(frameon=False, fontsize=6.5, title=f'Previous-year {self.background_var}',
                           title_fontsize=6.5)
            plt.suptitle(f'{season.capitalize()} {start}-{end}, by vegetation type (* Dry vs Wet, p < 0.05)',
                         fontsize=9)
            plt.tight_layout()
            plt.savefig(self.outdir + f'{self.background_var}\\{season}_background_response_by_landcover.png', dpi=300)
            plt.savefig(self.outdir + f'{self.background_var}\\{season}_background_response_by_landcover.pdf')
            plt.show()
            plt.close()

    def cal_paired_SI(self, df_pair):
        ## SI: paired test within the same pixel: for each pixel with drought (or wet) years under BOTH Dry and Wet
        ## backgrounds, diff = mean LAI anomaly (Dry background) - mean LAI anomaly (Wet background)
        ## 95% CI: block bootstrap over 1 deg blocks (20 x 20 pixels), area-weighted mean of pixel diffs
        rng = np.random.default_rng(42)
        start, end = self.period_list[0]
        pix_list = []
        summary_list = []
        for season in self.season_list:
            df_p = df_pair[(df_pair['season'] == season) & (df_pair['year'] >= start) & (df_pair['year'] <= end)]
            df_p = self.add_group(df_p)
            for metric in ['drought', 'wet']:
                df_m = self.select_years(df_p, metric)
                df_m = df_m[df_m['group'].isin(['Dry', 'Wet'])]
                df_mean = df_m.groupby(['pix', 'group'])['LAI_anomaly'].mean().unstack('group')
                df_n = df_m.groupby(['pix', 'group'])['LAI_anomaly'].count().unstack('group')
                df_mean = df_mean.dropna()   ## pixels with both Dry and Wet background years
                df_d = pd.DataFrame({'pix': df_mean.index, 'diff': (df_mean['Dry'] - df_mean['Wet']).values,
                                     'n_dry': df_n.loc[df_mean.index, 'Dry'].values,
                                     'n_wet': df_n.loc[df_mean.index, 'Wet'].values})
                df_info = df_p.drop_duplicates('pix')[['pix', 'area_weight', self.landcover_col]]
                df_d = df_d.merge(df_info, on='pix')
                df_d['block'] = [(pix[0] // 20, pix[1] // 20) for pix in df_d['pix']]
                df_d['season'] = season
                df_d['metric'] = metric
                pix_list.append(df_d)
                for lc in ['All'] + self.landcover_list:
                    df_lc = df_d if lc == 'All' else df_d[df_d[self.landcover_col] == lc]
                    if len(df_lc) == 0:
                        continue
                    w = df_lc['area_weight'].values
                    d = df_lc['diff'].values
                    df_b = pd.DataFrame({'block': df_lc['block'].values, 'w': w, 'wd': w * d})
                    sums = df_b.groupby('block')[['w', 'wd']].sum().values
                    idx = rng.integers(0, len(sums), (self.n_boot, len(sums)))
                    boot = sums[idx].sum(axis=1)
                    boot = boot[:, 1] / boot[:, 0]
                    value = np.sum(w * d) / np.sum(w)
                    p = np.mean(boot >= 0) if metric == 'drought' else np.mean(boot <= 0)
                    summary_list.append([season, metric, lc, value, np.percentile(boot, 2.5),
                                         np.percentile(boot, 97.5), len(df_lc), p])
        df_pix = pd.concat(pix_list, ignore_index=True)
        df_summary = pd.DataFrame(summary_list, columns=['season', 'metric', 'landcover', 'paired_diff',
                                                         'ci_low', 'ci_high', 'n_pixel', 'p_dry_stronger'])
        return df_pix, df_summary

    def plot_paired_SI(self, df_pix, df_summary):
        ## SI figure; left two panels: paired Dry - Wet difference by vegetation type; right: maps of the pixel differences
        start, end = self.period_list[0]
        title_dic = {'drought': f'(a) Drought years ({self.short_var} < {self.drought_threshold})',
                     'wet': f'(b) Wet years ({self.short_var} > {self.wet_year_threshold})'}
        lc_list = ['All'] + self.landcover_list
        for season in self.season_list:
            fig, axes = plt.subplots(1, 2, figsize=(16 * centimeter_factor, 7 * centimeter_factor))
            for ax, metric in zip(axes, ['drought', 'wet']):
                df_m = df_summary[(df_summary['season'] == season) & (df_summary['metric'] == metric)]
                tick_list = []
                for i, lc in enumerate(lc_list):
                    df_row = df_m[df_m['landcover'] == lc]
                    if len(df_row) == 0:
                        tick_list.append(lc)
                        continue
                    row = df_row.iloc[0]
                    ax.errorbar(i, row['paired_diff'],
                                yerr=[[row['paired_diff'] - row['ci_low']], [row['ci_high'] - row['paired_diff']]],
                                fmt='o', ms=5, color='k' if lc == 'All' else self.landcover_color_dic[lc], capsize=3)
                    tick_list.append(f'{lc}\nn={int(row["n_pixel"])}')   ## n = paired pixels
                    if row['p_dry_stronger'] < 0.05:
                        ax.text(i, 1.0, '*', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                                fontsize=12)
                ax.axhline(0, color='grey', lw=0.5)
                ax.set_xticks(range(len(lc_list)))
                ax.set_xticklabels(tick_list, fontsize=7)
                ax.set_xlim(-0.5, len(lc_list) - 0.5)
                ax.set_ylabel('Dry - Wet background, same pixel\n(LAI anomaly, %)', fontsize=8)
                ax.set_title(title_dic[metric] + '\n', fontsize=8, loc='left')
            plt.suptitle(f'{season.capitalize()} {start}-{end}, paired within pixel '
                         f'(* Dry stronger than Wet, p < 0.05)', fontsize=9)
            plt.tight_layout()
            plt.savefig(self.outdir + f'{self.background_var}\\SI_{season}_paired_dry_minus_wet.png', dpi=300)
            plt.savefig(self.outdir + f'{self.background_var}\\SI_{season}_paired_dry_minus_wet.pdf')
            plt.show()
            plt.close()

            for metric in ['drought', 'wet']:
                df_m = df_pix[(df_pix['season'] == season) & (df_pix['metric'] == metric)]
                arr = D.pix_dic_to_spatial_arr(T.df_to_spatial_dic(df_m, 'diff'))
                self.plot_map(arr, f'{self.background_var}\\SI_{season}_paired_{metric}_dry_minus_wet', 'RdBu', -30, 30,
                              'Dry - Wet background (LAI anomaly, %)', df_m['diff'].values,
                              df_m['area_weight'].values)

    def get_row(self, df_result, season, method, period, metric, group):
        return df_result[(df_result['season'] == season) & (df_result['method'] == method) &
                         (df_result['period'] == period) & (df_result['metric'] == metric) &
                         (df_result['group'] == group)].iloc[0]

    def plot_response(self, df_pair, df_result, method):
        ## (a) binned mean LAI anomaly vs SPEI03 with the regression line (first period); drought / wet years shaded
        ## (b) mean LAI anomaly in drought years (SPEI03 < -1), (c) in wet years (SPEI03 > 1), both periods, 95% CI
        ## (d) share of each background group per year (shows the time confounding)
        bins = np.arange(-2.5, 2.51, 0.25)
        ## group labels show how each group is defined
        label_dic = {'Wet': f'Wet (> {self.wet_threshold})',
                     'Normal': f'Normal ({self.dry_threshold} to {self.wet_threshold})',
                     'Dry': f'Dry (< {self.dry_threshold})'}
        panel_dic = {'drought': ('(b)', f'Drought years ({self.short_var} < {self.drought_threshold})',
                                 'more negative'),
                     'wet': ('(c)', f'Wet years ({self.short_var} > {self.wet_year_threshold})', 'more positive')}
        for season in self.season_list:
            fig, axes = plt.subplots(2, 2, figsize=(18 * centimeter_factor, 14 * centimeter_factor))
            axes = axes.flatten()
            start, end = self.period_list[0]
            period = f'{start}-{end}'
            df_p = df_pair[(df_pair['season'] == season) & (df_pair['year'] >= start) & (df_pair['year'] <= end)]
            df_p = self.add_group(df_p)

            ax = axes[0]
            ax.axvspan(-2.5, self.drought_threshold, color='#fbebeb', zorder=0)
            ax.axvspan(self.wet_year_threshold, 2.5, color='#e7f1fb', zorder=0)
            for group in self.group_list:
                df_g = df_p[df_p['group'] == group]
                x = df_g[self.short_col].values
                y = df_g['LAI_anomaly'].values
                w = df_g['area_weight'].values
                bin_idx = np.digitize(x, bins)
                xc, yc = [], []
                for i in range(1, len(bins)):
                    m = bin_idx == i
                    if m.sum() < 100:   ## skip sparse bins
                        continue
                    xc.append((bins[i - 1] + bins[i]) / 2)
                    yc.append(np.sum(y[m] * w[m]) / np.sum(w[m]))
                if len(xc) == 0:   ## group too small to plot
                    continue
                color = self.group_color_dic[group]
                row = self.get_row(df_result, season, method, period, 'slope', group)
                xl = np.array([min(xc), max(xc)])
                ax.scatter(xc, yc, s=10, color=color)
                ax.plot(xl, row['value'] * xl + row['intercept'], color=color, lw=1.5,
                        label=f'{label_dic[group]}, slope={row["value"]:.1f}')
            row = self.get_row(df_result, season, method, period, 'slope', 'Dry-Wet')
            ax.text(0.98, 0.03, f'Dry steeper than Wet, p={row["p_dry_stronger"]:.3f}',
                    transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom')
            ax.set_xlim(-2.5, 2.5)
            ax.axhline(0, color='grey', lw=0.5)
            ax.axvline(0, color='grey', lw=0.5)
            ax.set_xlabel(f'{season.capitalize()} {self.short_var}' + (' (detrended)' if 'detrend' in self.short_col else ''))
            ax.set_ylabel('LAI anomaly (%)')
            ax.set_title(f'(a) {season.capitalize()} {period}\n', fontsize=9, loc='left')
            ax.legend(frameon=False, fontsize=6, title=f'Previous-year {self.background_var}', title_fontsize=6,
                      loc='upper left')

            for ax, metric in [(axes[1], 'drought'), (axes[2], 'wet')]:
                label, title, direction = panel_dic[metric]
                offset_dic = {0: -0.12, 1: 0.12}
                p_text_list = []
                for k, (start_k, end_k) in enumerate(self.period_list):
                    period_k = f'{start_k}-{end_k}'
                    for i, group in enumerate(self.group_list):
                        row = self.get_row(df_result, season, method, period_k, metric, group)
                        color = self.group_color_dic[group]
                        ax.errorbar(i + offset_dic[k], row['value'],
                                    yerr=[[row['value'] - row['ci_low']], [row['ci_high'] - row['value']]],
                                    fmt='o' if k == 0 else 's', color=color,
                                    mfc=color if k == 0 else 'white', capsize=3,
                                    label=period_k if i == 0 else None)
                    row = self.get_row(df_result, season, method, period_k, metric, 'Dry-Wet')
                    p_text_list.append(f'{period_k} p={row["p_dry_stronger"]:.3f}')
                ax.axhline(0, color='grey', lw=0.5)
                ax.set_xticks(range(len(self.group_list)))
                ax.set_xticklabels([label_dic[g].replace(' (', '\n(') for g in self.group_list], fontsize=7)
                ax.set_xlim(-0.5, len(self.group_list) - 0.5)
                ax.set_xlabel(f'Background (previous-year {self.background_var})')
                ax.set_ylabel('Mean LAI anomaly (%)')
                ax.set_title(f'{label} {title}\n', fontsize=9, loc='left')
                ax.text(0.0, 1.0, f'Dry {direction} than Wet: ' + ', '.join(p_text_list),
                        transform=ax.transAxes, fontsize=6, va='bottom', ha='left')
                ax.margins(y=0.25)
                ax.legend(frameon=True, framealpha=1, fontsize=6.5, loc='best')

            ax = axes[3]
            df_share = df_p.groupby(['year', 'group'])['area_weight'].sum().unstack('group').fillna(0)
            df_share = df_share.div(df_share.sum(axis=1), axis=0) * 100
            bottom = np.zeros(len(df_share))
            for group in self.group_list:
                if not group in df_share.columns:
                    continue
                ax.bar(df_share.index, df_share[group].values, bottom=bottom, color=self.group_color_dic[group],
                       width=0.9, label=label_dic[group])
                bottom += df_share[group].values
            ax.set_xlabel('Year')
            ax.set_ylabel('Area share (%)')
            ax.set_ylim(0, 100)
            ax.set_title(f'(d) Background group by year ({period})\n', fontsize=9, loc='left')
            ax.legend(frameon=True, fontsize=6, ncol=3, loc='upper left', framealpha=1)

            plt.suptitle(f'Grouping: fixed thresholds of previous-year {self.background_var}', fontsize=9)
            plt.tight_layout()
            plt.savefig(self.outdir + f'{self.background_var}\\{season}_background_response_{method}.png', dpi=300)
            plt.savefig(self.outdir + f'{self.background_var}\\{season}_background_response_{method}.pdf')
            plt.show()
            plt.close()




def main():
    SPEI_background_sensitivity().run()   ## main text (unpaired) + SI (paired within pixel)



if __name__ == '__main__':
    main()
