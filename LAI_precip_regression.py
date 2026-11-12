from __Global__ import *
from SPEI_LAI_correlation import SPEI_SNU_LAI_correlation, D   ## reuse df_clean / to_matrix / detrend_rows / plot_map


class LAI_precip_structure(SPEI_SNU_LAI_correlation):
    ## Does 1 mm of heavy rain change LAI more than 1 mm of light or moderate rain?
    ## model (same season, same window for LAI and precipitation; summer = Jul-Oct, spring = Mar-May):
    ##   LAI anomaly (% of pixel mean) = b_light * P_LIGHT + b_mod * P_MOD + b_heavy * P_HEAVY
    ##   P_LIGHT = rain from days 1-5 mm, P_MOD = rain from days 5 mm - p95, P_HEAVY = R95p (days > p95), all in mm
    ##   two versions, X and Y treated the same way:
    ##     'detrend' = LAI and precipitation both linearly detrended per pixel
    ##     'raw'     = LAI and precipitation both as anomalies from the pixel mean (no detrending)
    ## M1 = the three summer bins only
    ## M2 = M1 + spring PRCPTOT (P_SPRING, Mar-May) + winter SWE_max (SWE_WINTER, Nov-May max), both same year, in mm
    ##      -> does the summer-bin result hold after controlling antecedent water, and how large is the carryover?
    ## M1 and M2 use the same pixel-years, so the coefficients can be compared directly
    ## main text: area-weighted pooled regression of all pixel-years, 95% CI by bootstrapping years
    ## SI: the same regression per pixel, coefficient maps
    def __init__(self):
        super().__init__()   ## same dff / years / LAI and cropland thresholds / df_clean as the correlation
        self.daymet_dff = result_root + rf'\Dataframe\Daymet\Daymet_seasonal.df'
        self.outdir = this_root + rf'Result2\03_response\LAI_precip_regression\\'
        T.mk_dir(self.outdir, force=True)
        self.season_list = ['summer']   ## summer LAI only (spring LAI dropped 2026-11-12)
        ## 3 bins need P_LIGHT and P_MOD from GEE; until they are downloaded use 2 bins (rain below / above p95)
        self.n_bins = 3
        self.version_list = ['detrend', 'raw']
        self.control_list = ['P_SPRING', 'SWE_WINTER']
        self.model_list = ['M1', 'M2']
        self.model = 'M1'   ## current model in the loop (set in run)
        self.n_boot = 1000
        pass

    def run(self):
        self.move_old_outputs()
        df = self.build_pixel_year()
        T.save_df(df, self.outdir + f'pixel_year_{self.n_bins}bins.df')
        # df = T.load_df(self.outdir + f'pixel_year_{self.n_bins}bins.df')
        result_list = []
        for model in self.model_list:
            self.model = model   ## var_list, file names follow this
            df_result = self.cal_pooled(df)
            df_result.insert(0, 'model', model)
            result_list.append(df_result)
            df_si = self.cal_per_pixel_SI(df)
            T.save_df(df_si, self.outdir + f'SI_per_pixel_{model}_{self.n_bins}bins.df')
            self.plot_per_pixel_SI(df_si)
        df_result = pd.concat(result_list, ignore_index=True)
        df_result.to_csv(self.outdir + f'pooled_{self.n_bins}bins.csv', index=False)
        print(df_result)
        self.plot_pooled(df_result)
        pass

    def move_old_outputs(self):
        ## SI maps from the first run (spring + summer, no model in the file name) -> old\ (moved, not deleted)
        import glob, shutil, os
        old_dir = self.outdir + 'old\\'
        flist = glob.glob(self.outdir + 'SI_summer_*') + glob.glob(self.outdir + 'SI_spring_*') + \
                glob.glob(self.outdir + f'SI_per_pixel_{self.n_bins}bins.df')
        if len(flist) == 0:
            return
        T.mk_dir(old_dir, force=True)
        for f in flist:
            shutil.move(f, old_dir + os.path.basename(f))
        print(len(flist), 'old files moved to', old_dir)

    def bin_list(self):
        if self.n_bins == 3:
            return ['P_LIGHT', 'P_MOD', 'P_HEAVY']
        return ['P_BELOW95', 'P_HEAVY']

    def var_list(self):
        ## predictors of the current model
        if self.model == 'M2':
            return self.bin_list() + self.control_list
        return self.bin_list()

    def build_pixel_year(self):
        ## one row per pixel-year and season: LAI anomaly and precipitation bins, both versions
        df_lai = self.df_clean(T.load_df(self.dff))
        df_p = T.load_df(self.daymet_dff)
        result_list = []
        for season in self.season_list:
            ## precipitation bins (mm)
            df_bin = df_p[['pix', 'year']].copy()
            df_bin['P_HEAVY'] = df_p[f'{season}_R95p']
            if self.n_bins == 3:
                df_bin['P_LIGHT'] = df_p[f'{season}_P_LIGHT']
                df_bin['P_MOD'] = df_p[f'{season}_P_MOD']
            else:
                df_bin['P_BELOW95'] = df_p[f'{season}_PRCPTOT'] - df_p[f'{season}_R95p']
            ## M2 controls: spring rain and winter snow of the same year (both before the summer window)
            df_bin['P_SPRING'] = df_p['spring_PRCPTOT']
            df_bin['SWE_WINTER'] = df_p['winter_SWE_max']
            ## LAI, low-LAI pixels masked
            lai = self.to_matrix(df_lai, f'{season}_{self.lai_var}')
            lai = lai[lai.mean(axis=1) >= self.LAI_threshold]
            lai_mean = lai.mean(axis=1).values[:, None]
            col_dic = {
                'LAI_detrend': self.detrend_rows(lai.values) / lai_mean * 100,
                'LAI_raw': (lai.values - lai_mean) / lai_mean * 100,
            }
            for b in self.bin_list() + self.control_list:
                x = self.to_matrix(df_bin, b).reindex(index=lai.index, columns=lai.columns).values
                col_dic[f'{b}_detrend'] = self.detrend_rows(x)
                col_dic[f'{b}_raw'] = x - np.nanmean(x, axis=1, keepdims=True)
            df_s = pd.DataFrame({'pix': np.repeat(lai.index.values, lai.shape[1]),
                                 'year': np.tile(lai.columns.values, lai.shape[0])})
            for col in col_dic:
                df_s[col] = col_dic[col].ravel()
            info = df_lai.groupby('pix')[['area_weight', self.landcover_col]].first()
            df_s = df_s.join(info, on='pix')
            df_s = df_s.dropna()   ## all variables, so M1 and M2 use the same pixel-years
            df_s['season'] = season
            result_list.append(df_s)
            print(season, 'pixel-years:', len(df_s), 'pixels:', df_s['pix'].nunique())
        return pd.concat(result_list, ignore_index=True)

    def year_sums(self, df_g, version, years):
        ## per year: weighted X'X and X'y (no intercept: both sides are anomalies); bootstrap resamples years
        var_list = self.var_list()
        X = df_g[[f'{b}_{version}' for b in var_list]].values
        y = df_g[f'LAI_{version}'].values
        w = df_g['area_weight'].values
        k = len(var_list)
        xx = np.einsum('n,ni,nj->nij', w, X, X).reshape(len(w), k * k)
        xy = (w * y)[:, None] * X
        df_tmp = pd.DataFrame(np.hstack([xx, xy]))
        df_tmp['year'] = df_g['year'].values
        s = df_tmp.groupby('year').sum().reindex(years).fillna(0).values
        return s, k

    def coef_from_sums(self, s, k):
        XX = s[..., :k * k].reshape(s.shape[:-1] + (k, k))
        Xy = s[..., k * k:]
        return np.linalg.solve(XX, Xy[..., None])[..., 0]

    def pooled_one(self, df_g, version, rng_idx):
        years = np.sort(df_g['year'].unique())
        s, k = self.year_sums(df_g, version, years)
        coef = self.coef_from_sums(s.sum(axis=0), k)
        coef_boot = self.coef_from_sums(s[rng_idx].sum(axis=1), k)   ## n_boot x k
        return coef, coef_boot

    def cal_pooled(self, df):
        ## all pixels and by land cover; Heavy - Light and Heavy - Mod tested with the same bootstrap draws
        var_list = self.var_list()
        i_heavy = var_list.index('P_HEAVY')
        result = []
        for season in self.season_list:
            df_s = df[df['season'] == season]
            n_year = df_s['year'].nunique()
            rng_idx = np.random.default_rng(0).integers(0, n_year, size=(self.n_boot, n_year))
            group_dic = {'All': df_s}
            for lc in self.landcover_list:
                group_dic[lc] = df_s[df_s[self.landcover_col] == lc]
            for version in self.version_list:
                ## collinearity check (all pixels only): VIF of each bin
                if True:
                    X = df_s[[f'{b}_{version}' for b in var_list]].values
                    vif = np.diag(np.linalg.inv(np.corrcoef(X.T)))
                    print(self.model, season, version, 'VIF', dict(zip(var_list, np.round(vif, 2))))
                for group in group_dic:
                    df_g = group_dic[group]
                    if df_g['pix'].nunique() < 30:
                        continue
                    coef, coef_boot = self.pooled_one(df_g, version, rng_idx)
                    for i, b in enumerate(var_list):
                        lo, hi = np.percentile(coef_boot[:, i], [2.5, 97.5])
                        result.append([season, version, group, b, coef[i], lo, hi, np.nan, df_g['pix'].nunique()])
                    ## heavy minus the other summer bins, one-sided bootstrap p for heavy > other
                    for i, b in enumerate(self.bin_list()):
                        if b == 'P_HEAVY':
                            continue
                        diff = coef[i_heavy] - coef[i]
                        diff_boot = coef_boot[:, i_heavy] - coef_boot[:, i]
                        lo, hi = np.percentile(diff_boot, [2.5, 97.5])
                        p = np.mean(diff_boot <= 0)
                        result.append([season, version, group, f'P_HEAVY - {b}', diff, lo, hi, p,
                                       df_g['pix'].nunique()])
        return pd.DataFrame(result, columns=['season', 'version', 'group', 'term', 'coef', 'ci_low', 'ci_high',
                                             'p_heavy_gt', 'n_pix'])

    def plot_pooled(self, df_result):
        ## coefficient (% LAI per mm) of each predictor with 95% CI; rows = model, columns = version, groups along x
        color_dic = {'P_LIGHT': '#9ecae1', 'P_MOD': '#4292c6', 'P_HEAVY': '#08306b', 'P_BELOW95': '#6baed6',
                     'P_SPRING': '#74c476', 'SWE_WINTER': '#9e9ac8'}
        group_list = ['All'] + self.landcover_list
        season = self.season_list[0]
        fig, axes = plt.subplots(len(self.model_list), len(self.version_list), sharey=True, squeeze=False,
                                 figsize=(16 * centimeter_factor, 6 * len(self.model_list) * centimeter_factor))
        for i, model in enumerate(self.model_list):
            self.model = model
            var_list = self.var_list()
            width = 0.8 / len(var_list)
            for j, version in enumerate(self.version_list):
                ax = axes[i, j]
                df_v = df_result[(df_result['model'] == model) & (df_result['season'] == season) &
                                 (df_result['version'] == version)]
                for k, b in enumerate(var_list):
                    for g, group in enumerate(group_list):
                        row = df_v[(df_v['group'] == group) & (df_v['term'] == b)]
                        if len(row) == 0:
                            continue
                        row = row.iloc[0]
                        x = g + (k - (len(var_list) - 1) / 2) * width
                        ax.bar(x, row['coef'], width, color=color_dic[b], label=b if g == 0 else None)
                        ax.errorbar(x, row['coef'], yerr=[[row['coef'] - row['ci_low']], [row['ci_high'] - row['coef']]],
                                    color='k', lw=0.8, capsize=1.5)
                ax.axhline(0, color='k', lw=0.5)
                ax.set_xticks(range(len(group_list)))
                ax.set_xticklabels(group_list, fontsize=7)
                ax.set_title(f'{model}, {season.capitalize()} LAI, {version}', fontsize=8)
                if j == 0:
                    ax.set_ylabel('LAI anomaly (%) per mm')
            axes[i, 0].legend(frameon=False, fontsize=6)
        plt.tight_layout()
        plt.savefig(self.outdir + f'pooled_{self.n_bins}bins.png', dpi=300)
        plt.savefig(self.outdir + f'pooled_{self.n_bins}bins.pdf')
        plt.show()
        plt.close()

    def cal_per_pixel_SI(self, df):
        ## SI: same regression per pixel (no intercept), coefficient and two-sided p of each predictor
        ## a predictor with no variation in a pixel (e.g. SWE_WINTER = 0 every year where it never snows) is left out there
        from scipy import stats
        var_list = self.var_list()
        result = []
        for season in self.season_list:
            df_s = df[df['season'] == season]
            for version in self.version_list:
                for pix, df_p in df_s.groupby('pix'):
                    X_all = df_p[[f'{b}_{version}' for b in var_list]].values
                    y = df_p[f'LAI_{version}'].values
                    n = len(y)
                    if n < 20:
                        continue
                    use = X_all.std(axis=0) > 1e-6
                    X = X_all[:, use]
                    k = X.shape[1]
                    XX = X.T @ X
                    if np.linalg.cond(XX) > 1e10:
                        continue
                    XX_inv = np.linalg.inv(XX)
                    coef = np.full(len(var_list), np.nan)
                    p = np.full(len(var_list), np.nan)
                    coef[use] = XX_inv @ X.T @ y
                    resid = y - X @ coef[use]
                    se = np.sqrt(np.diag(XX_inv) * np.sum(resid ** 2) / (n - k))
                    p[use] = 2 * stats.t.sf(np.abs(coef[use] / se), n - k)
                    row = [season, version, pix, df_p['area_weight'].iloc[0]]
                    result.append(row + list(coef) + list(p))
                print(self.model, season, version, 'per pixel done')
        columns = ['season', 'version', 'pix', 'area_weight'] + [f'coef_{b}' for b in var_list] + \
                  [f'p_{b}' for b in var_list]
        return pd.DataFrame(result, columns=columns)

    def plot_per_pixel_SI(self, df_si):
        ## SI maps: coefficient of each predictor, and heavy minus the next bin; one map per model, season and version
        bins = self.bin_list()
        var_list = self.var_list()
        for season in self.season_list:
            for version in self.version_list:
                df_v = df_si[(df_si['season'] == season) & (df_si['version'] == version)].copy()
                df_v['coef_heavy_minus'] = df_v['coef_P_HEAVY'] - df_v[f'coef_{bins[-2]}']
                w = df_v['area_weight'].values
                for col in [f'coef_{b}' for b in var_list] + ['coef_heavy_minus']:
                    vals = df_v[col].values
                    if np.all(np.isnan(vals)):
                        continue
                    vlim = np.nanpercentile(np.abs(vals), 90)
                    arr = D.pix_dic_to_spatial_arr(dict(zip(df_v['pix'], vals)))
                    arr[arr <= -9999] = np.nan   ## pixels not in the dict are filled with -9999
                    fname = f'SI_{self.model}_{season}_{version}_{col}_{self.n_bins}bins'
                    label = 'P_HEAVY minus ' + bins[-2] if col == 'coef_heavy_minus' else col.replace('coef_', '')
                    ok = np.isfinite(vals)
                    self.plot_map(arr, fname, 'RdBu', -vlim, vlim, f'{label}: LAI anomaly (%) per mm', vals[ok], w[ok])
        pass


class LAI_precip_period(LAI_precip_structure):
    ## M2 only, fitted separately in three periods to see whether the dependence of summer LAI on each water source
    ## changed after 2000: 1982-2024 (reference), 1982-1999, 2000-2024
    ## LAI and predictors are detrended (or mean-removed for 'raw') WITHIN each period, so the 2000 step does not
    ##   enter the year-to-year variation
    ## only pixels valid in both 1982-1999 and 2000-2024 are used, so the periods compare the same vegetation
    ## period difference = (2000-2024) - (1982-1999); each period bootstrapped over its own years, 95% CI of the difference
    def __init__(self):
        super().__init__()
        self.model = 'M2'
        self.model_list = ['M2']
        self.period_list = [(1982, 2024), (1982, 1999), (2000, 2024)]
        self.pre, self.post = (1982, 1999), (2000, 2024)
        self.outdir = self.outdir + 'period\\'
        T.mk_dir(self.outdir, force=True)
        pass

    def run(self):
        df = self.build_period_df()
        T.save_df(df, self.outdir + f'pixel_year_period_{self.n_bins}bins.df')
        # df = T.load_df(self.outdir + f'pixel_year_period_{self.n_bins}bins.df')
        df_result = self.cal_period(df)
        df_result.to_csv(self.outdir + f'pooled_M2_period_{self.n_bins}bins.csv', index=False)
        print(df_result[df_result['group'] == 'All'])
        for version in self.version_list:
            self.plot_period(df_result, version)
        pass

    def build_period_df(self):
        ## pixel-year table per period; df_clean keeps self.start_year - self.end_year, so detrending is within the period
        result_list = []
        for start, end in self.period_list:
            self.start_year, self.end_year = start, end
            df_p = self.build_pixel_year()
            df_p['period'] = f'{start}-{end}'
            result_list.append(df_p)
        self.start_year, self.end_year = 1982, 2024
        df = pd.concat(result_list, ignore_index=True)
        ## same pixels in both sub-periods
        pix_pre = set(df.loc[df['period'] == f'{self.pre[0]}-{self.pre[1]}', 'pix'])
        pix_post = set(df.loc[df['period'] == f'{self.post[0]}-{self.post[1]}', 'pix'])
        pix_both = pix_pre & pix_post
        df = df[df['pix'].isin(pix_both)]
        print('pixels in both periods:', len(pix_both))
        return df

    def cal_period(self, df):
        var_list = self.var_list()
        season = self.season_list[0]
        result = []
        boot_dic = {}   ## (period, version, group) -> coef_boot, for the period difference
        for start, end in self.period_list:
            period = f'{start}-{end}'
            df_s = df[(df['period'] == period) & (df['season'] == season)]
            n_year = df_s['year'].nunique()
            rng_idx = np.random.default_rng(0).integers(0, n_year, size=(self.n_boot, n_year))
            group_dic = {'All': df_s}
            for lc in self.landcover_list:
                group_dic[lc] = df_s[df_s[self.landcover_col] == lc]
            for version in self.version_list:
                X = df_s[[f'{b}_{version}' for b in var_list]].values
                vif = np.diag(np.linalg.inv(np.corrcoef(X.T)))
                print(period, version, 'VIF', dict(zip(var_list, np.round(vif, 2))))
                for group in group_dic:
                    df_g = group_dic[group]
                    if df_g['pix'].nunique() < 30:
                        continue
                    coef, coef_boot = self.pooled_one(df_g, version, rng_idx)
                    boot_dic[(period, version, group)] = (coef, coef_boot)
                    for i, b in enumerate(var_list):
                        lo, hi = np.percentile(coef_boot[:, i], [2.5, 97.5])
                        result.append([period, version, group, b, coef[i], lo, hi, np.nan, df_g['pix'].nunique()])
        ## (2000-2024) - (1982-1999); two-sided bootstrap p
        pre, post = f'{self.pre[0]}-{self.pre[1]}', f'{self.post[0]}-{self.post[1]}'
        for (period, version, group) in list(boot_dic):
            if period != post or (pre, version, group) not in boot_dic:
                continue
            coef_post, boot_post = boot_dic[(post, version, group)]
            coef_pre, boot_pre = boot_dic[(pre, version, group)]
            for i, b in enumerate(var_list):
                diff = coef_post[i] - coef_pre[i]
                diff_boot = boot_post[:, i] - boot_pre[:, i]
                lo, hi = np.percentile(diff_boot, [2.5, 97.5])
                p = 2 * min(np.mean(diff_boot <= 0), np.mean(diff_boot >= 0))
                result.append(['post - pre', version, group, b, diff, lo, hi, p, np.nan])
        return pd.DataFrame(result, columns=['period', 'version', 'group', 'term', 'coef', 'ci_low', 'ci_high',
                                             'p_diff', 'n_pix'])

    def plot_period(self, df_result, version):
        ## one panel per predictor; x = vegetation group, bars = periods; * above a group = post - pre p < 0.05
        var_list = self.var_list()
        group_list = ['All'] + self.landcover_list
        period_list = [f'{s}-{e}' for s, e in self.period_list]
        color_dic = {period_list[0]: 'grey', period_list[1]: '#92c5de', period_list[2]: '#b2182b'}
        width = 0.8 / len(period_list)
        df_v = df_result[df_result['version'] == version]
        fig, axes = plt.subplots(1, len(var_list), sharey=True,
                                 figsize=(4.5 * len(var_list) * centimeter_factor, 6 * centimeter_factor))
        for ax, b in zip(axes, var_list):
            for k, period in enumerate(period_list):
                for g, group in enumerate(group_list):
                    row = df_v[(df_v['period'] == period) & (df_v['group'] == group) & (df_v['term'] == b)]
                    if len(row) == 0:
                        continue
                    row = row.iloc[0]
                    x = g + (k - (len(period_list) - 1) / 2) * width
                    ax.bar(x, row['coef'], width, color=color_dic[period], label=period if g == 0 else None)
                    ax.errorbar(x, row['coef'], yerr=[[row['coef'] - row['ci_low']], [row['ci_high'] - row['coef']]],
                                color='k', lw=0.6, capsize=1)
            for g, group in enumerate(group_list):
                row = df_v[(df_v['period'] == 'post - pre') & (df_v['group'] == group) & (df_v['term'] == b)]
                if len(row) > 0 and row.iloc[0]['p_diff'] < 0.05:
                    ax.text(g, ax.get_ylim()[1] * 0.9, '*', ha='center', fontsize=10)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(range(len(group_list)))
            ax.set_xticklabels(group_list, fontsize=6, rotation=45)
            ax.set_title(b, fontsize=8)
        axes[0].set_ylabel('Summer LAI anomaly (%) per mm')
        axes[0].legend(frameon=False, fontsize=6)
        plt.suptitle(f'M2, {version}', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.outdir + f'pooled_M2_period_{version}_{self.n_bins}bins.png', dpi=300)
        plt.savefig(self.outdir + f'pooled_M2_period_{version}_{self.n_bins}bins.pdf')
        plt.show()
        plt.close()


def main():
    # LAI_precip_structure().run()   ## M1 and M2, 1982-2024: done
    LAI_precip_period().run()


if __name__ == '__main__':
    main()
