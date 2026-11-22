## Model 1 on the main growing season (GS.df), 2003-2024 (2026-11-15)
## expected LAI trend = beta (LAI vs SPEI at the pixel's best scale) x SPEI trend (best scale)
## compared with the observed LAI trend; gap = observed - expected
## uses the trend tools of growing_season_extraction.py and the best scale from bivariate.py
from growing_season_extraction import *
## 2026-11-21: larger fonts in all figures (default sizes; explicit fontsize values were also enlarged x1.6)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'legend.fontsize': 11, 'legend.title_fontsize': 11})
from scipy.signal import detrend


class Attribution_model1(Growing_season_trend):
    ## inputs : GS.df; best SPEI scale per pixel from LAI_SPEI_bivariate_2003_2024.df (bivariate.py)
    ## beta   : slope of detrended GS_LAI_rel (% of the pixel's mean) on detrended SPEI at the best scale (% per SPEI unit)
    ##          main  = pooled beta per vegetation type (all pixels of the type stacked)
    ##          check = per-pixel beta
    ## trends : not detrended; LAI in % per decade, SPEI per decade (best scale)
    ## expected = beta x SPEI trend; gap = observed - expected
    ## output : Result2\03_response\attribution_model1\
    def __init__(self):
        super().__init__(this_root + rf'Result2\03_response\attribution_model1\\')
        self.biv_dff = this_root + rf'Result2\03_response\LAI_SPEI_bivariate\LAI_SPEI_bivariate_2003_2024.df'
        self.scale_list = list(range(3, 49, 3))
        self.color_dic = {'observed': '#2E7D32', 'expected': '#C0392B', 'gap': '#7F8C8D'}

    def run(self):
        df, info = self.load()
        biv = T.load_df(self.biv_dff).reindex(info.index)
        best_scale = biv['best_scale'].values
        lai_rel = self.matrix(df, info, 'GS_LAI_rel')
        spei = np.full(lai_rel.shape, np.nan)
        for scale in self.scale_list:
            m = best_scale == scale
            spei[m] = self.matrix(df, info, f'GS_SPEI{scale:02d}')[m]
        ok = np.isfinite(lai_rel).all(axis=1) & np.isfinite(spei).all(axis=1)
        info, lai_rel, spei, best_scale = info[ok], lai_rel[ok], spei[ok], best_scale[ok]
        print('pixels:', len(info))
        lai_d, spei_d = detrend(lai_rel, axis=1), detrend(spei, axis=1)
        ## beta
        beta_pixel = np.sum(lai_d * spei_d, axis=1) / np.sum(spei_d ** 2, axis=1)
        beta_pooled = np.full(len(info), np.nan)
        beta_class = {}
        for lc in self.class_list[1:]:
            m = (info['landcover_type'] == lc).values
            beta_class[lc] = np.sum(lai_d[m] * spei_d[m]) / np.sum(spei_d[m] ** 2)
            beta_pooled[m] = beta_class[lc]
        print('pooled beta (% LAI per SPEI unit):', {k: round(v, 2) for k, v in beta_class.items()})
        ## trends
        tr_lai, p_lai = self.trend(lai_rel)
        tr_spei, p_spei = self.trend(spei)
        exp_pooled = beta_pooled * tr_spei
        exp_pixel = beta_pixel * tr_spei
        gap_pooled = tr_lai - exp_pooled
        gap_pixel = tr_lai - exp_pixel
        ## maps (same colour range for observed, expected, gap)
        vlim = np.nanpercentile(np.abs(tr_lai), 90)
        for vals, name, label in [(tr_lai, 'observed', 'Observed LAI trend'),
                                  (exp_pooled, 'expected', 'Expected LAI trend (pooled beta x SPEI trend)'),
                                  (gap_pooled, 'gap', 'Gap = observed - expected (pooled beta)'),
                                  (exp_pixel, 'expected_pixel_beta', 'Expected LAI trend (pixel beta x SPEI trend)'),
                                  (gap_pixel, 'gap_pixel_beta', 'Gap = observed - expected (pixel beta)')]:
            self.save_map(info, vals, f'LAI_trend_{name}_{self.tag}', label + ' (% per decade)', vlim, 'RdBu')
        self.save_map(info, beta_pixel, f'beta_pixel_{self.tag}', 'beta (% LAI per SPEI unit)',
                      np.nanpercentile(np.abs(beta_pixel), 90), 'RdBu')
        ## outputs
        res = pd.DataFrame({'best_scale': best_scale, 'beta_pooled': beta_pooled, 'beta_pixel': beta_pixel,
                            'SPEI_trend': tr_spei, 'SPEI_trend_p': p_spei, 'LAI_trend': tr_lai, 'LAI_trend_p': p_lai,
                            'expected_pooled': exp_pooled, 'gap_pooled': gap_pooled,
                            'expected_pixel': exp_pixel, 'gap_pixel': gap_pixel}, index=info.index)
        T.save_df(info.join(res), self.outdir + f'attribution_model1_{self.tag}.df')
        df_t = self.table_model1(info, res, beta_class)
        self.bar(df_t)
        self.scatter_obs_exp(info, tr_lai, exp_pooled, f'LAI_trend_observed_vs_expected_{self.tag}')
        pass

    def table_model1(self, info, res, beta_class):
        ## per region x class: area-weighted observed, expected and gap with 95% CI (1 x 1 deg block bootstrap)
        region_dic = {'all': np.ones(len(info), dtype=bool)}   ## no monsoon / non-monsoon split
        w_all = info['area_weight'].values
        result = []
        for region, mr in region_dic.items():
            for lc in self.class_list:
                ## All = the four vegetation types (pixels of other types have no pooled beta)
                m = mr & ((info['landcover_type'] == lc).values if lc != 'All'
                          else info['landcover_type'].isin(self.class_list[1:]).values)
                if m.sum() < 30:
                    continue
                w = w_all[m]
                row = [region, lc, m.sum(), beta_class.get(lc, np.nan),
                       np.sum(res['SPEI_trend'].values[m] * w) / np.sum(w)]
                for col in ['LAI_trend', 'expected_pooled', 'gap_pooled', 'expected_pixel', 'gap_pixel']:
                    v = res[col].values[m]
                    lo, hi = self.block_ci(info[m], v, w)
                    row += [np.sum(v * w) / np.sum(w), lo, hi]
                result.append(row)
        cols = ['region', 'class', 'n_pix', 'beta_pooled', 'SPEI_trend_per_decade']
        for name in ['observed', 'expected', 'gap', 'expected_pixel_beta', 'gap_pixel_beta']:
            cols += [name, f'{name}_ci_low', f'{name}_ci_high']
        df_t = pd.DataFrame(result, columns=cols)
        df_t.to_csv(self.outdir + f'attribution_model1_summary_{self.tag}.csv', index=False)
        print(df_t[['region', 'class', 'n_pix', 'beta_pooled', 'SPEI_trend_per_decade', 'observed', 'expected',
                    'gap', 'expected_pixel_beta', 'gap_pixel_beta']].round(2).to_string())
        return df_t

    def bar(self, df_t):
        ## observed / expected / gap per class, one panel per region, pooled beta, 95% CI
        region_list = list(df_t['region'].unique())
        fig, axs = plt.subplots(1, len(region_list), figsize=(4 * len(region_list), 3.2), sharey=True)
        for ax, region in zip(np.atleast_1d(axs), region_list):
            d = df_t[df_t['region'] == region].reset_index(drop=True)
            x = np.arange(len(d))
            for k, name in enumerate(['observed', 'expected', 'gap']):
                err = np.array([d[name] - d[f'{name}_ci_low'], d[f'{name}_ci_high'] - d[name]])
                ax.bar(x + (k - 1) * 0.27, d[name], width=0.27, color=self.color_dic[name], yerr=err,
                       capsize=2, error_kw={'lw': 0.6}, label=name)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(d['class'], fontsize=13)
            ax.set_title(region, fontsize=14.5)
        np.atleast_1d(axs)[0].set_ylabel('LAI trend (% per decade)', fontsize=13)
        np.atleast_1d(axs)[0].legend(fontsize=11, frameon=False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'attribution_model1_bar_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'attribution_model1_bar_{self.tag}.pdf', bbox_inches='tight')
        plt.close()

    def scatter_obs_exp(self, info, obs, exp, fname):
        ## per vegetation type: observed vs expected LAI trend, area-weighted hexbin, 1:1 line
        fig, axs = plt.subplots(1, len(self.class_list), figsize=(3 * len(self.class_list), 3), sharex=True,
                                sharey=True)
        lim = np.nanpercentile(np.abs(np.concatenate([obs, exp])), 99)
        for ax, lc in zip(axs, self.class_list):
            m = np.isfinite(obs) & np.isfinite(exp)
            if lc != 'All':
                m &= (info['landcover_type'] == lc).values
            w = info['area_weight'].values[m]
            ax.hexbin(exp[m], obs[m], C=w, reduce_C_function=np.sum, gridsize=40, cmap='Greys', bins='log',
                      mincnt=1, extent=[-lim, lim, -lim, lim])
            ax.plot([-lim, lim], [-lim, lim], color='#C0392B', lw=0.8)
            ax.axhline(0, color='k', lw=0.5)
            ax.axvline(0, color='k', lw=0.5)
            above = np.sum(w[obs[m] > exp[m]]) / np.sum(w) * 100
            ax.text(0.03, 0.97, f'obs > exp: {above:.0f}%', transform=ax.transAxes, ha='left', va='top', fontsize=13)
            ax.set_title(f'{lc} (n={m.sum()})', fontsize=14.5)
            ax.set_xlabel('Expected LAI trend (% /decade)', fontsize=13)
        axs[0].set_ylabel('Observed LAI trend (% /decade)', fontsize=13)
        plt.tight_layout()
        plt.savefig(self.outdir + f'{fname}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'{fname}.pdf', bbox_inches='tight')
        plt.close()


class Attribution_model2(Growing_season_trend):
    ## Model 2 on the main growing season (GS.df), 2003-2024 (2026-11-15)
    ## layer 1 (climate, year to year, detrended, pooled per vegetation type):
    ##     LAI_rel' = b1 * GS_P_rel' + b2 * GS_P_antecedent_rel' + b3 * GS_VPD' + e      (' = detrended)
    ##     GS_P_rel, GS_P_antecedent_rel: % of the pixel's 2003-2024 mean;  GS_VPD: kPa
    ##     A_k = b_k x trend_k (not detrended);  A = A_P + A_Pant + A_VPD
    ## layer 2 (composition, across pixels within a vegetation type, area-weighted):
    ##     observed - A = g0 + g1 * dAFG + g2 * dPFG + g3 * dSHR + g4 * dTRE + u     (dX = RAP cover trend, % per decade)
    ##     RAP trend: OLS on the available years, at least self.rap_min_year of 22 (RAP has missing years)
    ##     B = sum g_k * dX_k;  C = observed - A - B  (= g0 + u)
    ## observed = A_P + A_Pant + A_VPD + B + C for every pixel; All = the four vegetation types
    ## no monsoon / non-monsoon split (the growing season is already defined per pixel)
    ## CI: 1 x 1 deg block bootstrap of the pixel values (b and g fixed); g CI: block bootstrap refit
    ## output: Result2\03_response\attribution_model2\
    def __init__(self):
        super().__init__(this_root + rf'Result2\03_response\attribution_model2\\')
        self.clim_list = [('GS_P', 'P', True), ('GS_P_antecedent', 'Pant', True), ('GS_VPD', 'VPD', False)]
        self.rap_list = ['RAP_AFG', 'RAP_PFG', 'RAP_SHR', 'RAP_TRE']
        self.rap_min_year = 18
        self.model1_dff = this_root + rf'Result2\03_response\attribution_model1\attribution_model1_{self.tag}.df'
        self.comp_list = ['observed', 'A_P', 'A_Pant', 'A_VPD', 'A', 'B', 'C']
        self.color_dic = {'observed': '#2E7D32', 'A_P': '#5DADE2', 'A_Pant': '#1F618D', 'A_VPD': '#C0392B',
                          'A': '#E67E22', 'B': '#8E44AD', 'C': '#7F8C8D'}

    def run(self):
        df, info = self.load()
        info = info[info['landcover_type'].isin(self.class_list[1:])]
        lai = self.matrix(df, info, 'GS_LAI_rel')
        X = {name: self.matrix(df, info, var, rel) for var, name, rel in self.clim_list}
        rap = {c: self.matrix(df, info, c) for c in self.rap_list}
        self.rap_missing_check(info, rap)
        ok = np.isfinite(lai).all(axis=1)
        for arr in X.values():
            ok &= np.isfinite(arr).all(axis=1)
        for arr in rap.values():
            ok &= np.isfinite(arr).sum(axis=1) >= self.rap_min_year
        ## which condition removes pixels (printed)
        print('four vegetation types:', len(info))
        print('  incomplete LAI:', (~np.isfinite(lai).all(axis=1)).sum())
        for k, arr in X.items():
            print(f'  incomplete {k}:', (~np.isfinite(arr).all(axis=1)).sum())
        for k, arr in rap.items():
            print(f'  {k} < {self.rap_min_year} years:', (np.isfinite(arr).sum(axis=1) < self.rap_min_year).sum())
        print('  kept:', ok.sum())
        info, lai = info[ok], lai[ok]
        X = {k: v[ok] for k, v in X.items()}
        rap = {k: v[ok] for k, v in rap.items()}
        print('pixels:', len(info))
        res = pd.DataFrame(index=info.index)
        res['observed'], res['LAI_trend_p'] = self.trend(lai)
        ## layer 1
        coef = self.layer1(info, lai, X, res)
        ## layer 2
        for c in self.rap_list:
            res[f'{c}_trend'] = self.trend_nan(rap[c])
        gamma = self.layer2(info, res)
        res['C'] = res['observed'] - res['A'] - res['B']
        if os.path.isfile(self.model1_dff):
            res['expected_model1'] = T.load_df(self.model1_dff)['expected_pooled'].reindex(info.index)
        T.save_df(info.join(res), self.outdir + f'attribution_model2_{self.tag}.df')
        coef.to_csv(self.outdir + f'attribution_model2_coef_{self.tag}.csv', index=False)
        gamma.to_csv(self.outdir + f'attribution_model2_gamma_{self.tag}.csv', index=False)
        print(coef.round(3).to_string())
        print(gamma.round(3).to_string())
        ## maps
        vlim = np.nanpercentile(np.abs(res['observed']), 90)
        for name in ['A', 'A_Pant', 'A_P', 'A_VPD', 'B', 'C']:
            self.save_map(info, res[name].values, f'model2_{name}_{self.tag}', f'Model 2 {name} (% per decade)',
                          vlim, 'RdBu')
        df_t = self.table_model2(info, res)
        self.bar(df_t)
        pass

    def trend_nan(self, arr):
        ## OLS slope per decade per row, using the available years only
        x = np.array(self.year_list, dtype=float)
        ok = np.isfinite(arr)
        n = ok.sum(axis=1)
        xx = np.where(ok, x[None, :], 0.)
        yy = np.where(ok, arr, 0.)
        xm = xx.sum(axis=1) / n
        ym = yy.sum(axis=1) / n
        sxy = np.sum(np.where(ok, (x[None, :] - xm[:, None]) * (arr - ym[:, None]), 0.), axis=1)
        sxx = np.sum(np.where(ok, (x[None, :] - xm[:, None]) ** 2, 0.), axis=1)
        return sxy / sxx * 10

    def rap_missing_check(self, info, rap):
        ## how many RAP years are missing, per vegetation type and per year (printed and saved)
        miss = np.zeros(next(iter(rap.values())).shape, dtype=bool)
        for arr in rap.values():
            miss |= ~np.isfinite(arr)
        n_miss = miss.sum(axis=1)
        result = []
        for lc in self.class_list:
            m = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
            result.append([lc, m.sum(), np.mean(n_miss[m] == 0) * 100, np.mean(n_miss[m] == 22) * 100,
                           np.mean((n_miss[m] > 0) & (n_miss[m] <= 22 - self.rap_min_year)) * 100,
                           np.mean((n_miss[m] > 22 - self.rap_min_year) & (n_miss[m] < 22)) * 100])
        df_c = pd.DataFrame(result, columns=['class', 'n_pix', 'complete_%', 'no_RAP_%',
                                             f'kept_with_gaps_%', 'dropped_partial_%'])
        df_y = pd.DataFrame({'year': self.year_list, 'missing_pix_%': miss.mean(axis=0) * 100})
        df_c.to_csv(self.outdir + 'RAP_missing_check_class.csv', index=False)
        df_y.to_csv(self.outdir + 'RAP_missing_check_year.csv', index=False)
        print(df_c.round(1).to_string())
        print(df_y.round(1).to_string())

    def layer1(self, info, lai, X, res):
        ## pooled OLS of detrended LAI_rel on detrended climate, per vegetation type; VIF and R2
        lai_d = detrend(lai, axis=1)
        X_d = {k: detrend(v, axis=1) for k, v in X.items()}
        X_tr = {k: self.trend(v)[0] for k, v in X.items()}
        names = list(X.keys())
        for k in names:
            res[f'{k}_trend'] = X_tr[k]
        A = {k: np.full(len(info), np.nan) for k in names}
        result = []
        for lc in self.class_list[1:]:
            m = (info['landcover_type'] == lc).values
            y = lai_d[m].ravel()
            M = np.column_stack([X_d[k][m].ravel() for k in names])
            b, *_ = np.linalg.lstsq(M, y, rcond=None)
            r2 = 1 - np.sum((y - M @ b) ** 2) / np.sum(y ** 2)
            vif = []
            for j in range(len(names)):
                others = np.delete(M, j, axis=1)
                g, *_ = np.linalg.lstsq(others, M[:, j], rcond=None)
                r2_j = 1 - np.sum((M[:, j] - others @ g) ** 2) / np.sum(M[:, j] ** 2)
                vif.append(1 / (1 - r2_j))
            for j, k in enumerate(names):
                A[k][m] = b[j] * X_tr[k][m]
            result.append([lc, m.sum(), r2] + list(b) + vif)
        for k in names:
            res[f'A_{k}'] = A[k]
        res['A'] = sum(A[k] for k in names)
        return pd.DataFrame(result, columns=['class', 'n_pix', 'R2'] + [f'beta_{k}' for k in names] +
                                            [f'VIF_{k}' for k in names])

    def wls(self, Z, y, w):
        ## area-weighted OLS with intercept; returns [g0, g1, ...]
        Z1 = np.column_stack([np.ones(len(y)), Z])
        sw = np.sqrt(w)
        g, *_ = np.linalg.lstsq(Z1 * sw[:, None], y * sw, rcond=None)
        return g

    def layer2(self, info, res):
        ## across pixels within a vegetation type: (observed - A) on RAP cover trends
        res['B'] = np.nan
        cols = [f'{c}_trend' for c in self.rap_list]
        rng = np.random.default_rng(0)
        result = []
        for lc in self.class_list[1:]:
            m = (info['landcover_type'] == lc).values
            Z = res.loc[m, cols].values
            y = (res['observed'] - res['A']).values[m]
            w = info['area_weight'].values[m]
            g = self.wls(Z, y, w)
            res.loc[m, 'B'] = Z @ g[1:]
            ## gamma CI: refit on 1 x 1 deg block bootstrap samples
            block = (np.floor(info['lon'].values[m]) * 1000 + np.floor(info['lat'].values[m])).astype(int)
            ids = np.unique(block)
            idx = {bk: np.where(block == bk)[0] for bk in ids}
            boot = []
            for _ in range(200):
                pick = np.concatenate([idx[bk] for bk in rng.choice(ids, size=len(ids))])
                boot.append(self.wls(Z[pick], y[pick], w[pick]))
            lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
            for j, name in enumerate(['intercept'] + self.rap_list):
                result.append([lc, name, g[j], lo[j], hi[j]])
        return pd.DataFrame(result, columns=['class', 'term', 'gamma', 'ci_low', 'ci_high'])

    def table_model2(self, info, res):
        ## per region x class: area-weighted mean of each component with 95% CI
        region_dic = {'all': np.ones(len(info), dtype=bool)}
        extra = [c for c in ['expected_model1'] if c in res.columns]
        w_all = info['area_weight'].values
        result = []
        for region, mr in region_dic.items():
            for lc in self.class_list:
                m = mr & ((info['landcover_type'] == lc).values if lc != 'All' else True)
                if m.sum() < 30:
                    continue
                w = w_all[m]
                row = [region, lc, m.sum()]
                for name in self.comp_list + extra:
                    v = res[name].values[m]
                    f = np.isfinite(v)   ## expected_model1 can be missing for a few pixels
                    lo, hi = self.block_ci(info[m][f], v[f], w[f])
                    row += [np.sum(v[f] * w[f]) / np.sum(w[f]), lo, hi]
                result.append(row)
        cols = ['region', 'class', 'n_pix']
        for name in self.comp_list + extra:
            cols += [name, f'{name}_ci_low', f'{name}_ci_high']
        df_t = pd.DataFrame(result, columns=cols)
        df_t.to_csv(self.outdir + f'attribution_model2_summary_{self.tag}.csv', index=False)
        print(df_t[['region', 'class', 'n_pix'] + self.comp_list + extra].round(2).to_string())
        return df_t

    def bar(self, df_t):
        ## observed and its components per class, one panel per region, 95% CI
        name_list = [n for n in self.comp_list if n != 'A']   ## observed, A_P, A_Pant, A_VPD (, A_SWE), B, C
        region_list = list(df_t['region'].unique())
        fig, axs = plt.subplots(1, len(region_list), figsize=(5 * len(region_list), 3.4), sharey=True)
        width = 0.8 / len(name_list)
        for ax, region in zip(np.atleast_1d(axs), region_list):
            d = df_t[df_t['region'] == region].reset_index(drop=True)
            x = np.arange(len(d))
            for k, name in enumerate(name_list):
                err = np.array([d[name] - d[f'{name}_ci_low'], d[f'{name}_ci_high'] - d[name]])
                ax.bar(x + (k - (len(name_list) - 1) / 2) * width, d[name], width=width,
                       color=self.color_dic[name], yerr=err, capsize=1.5, error_kw={'lw': 0.5}, label=name)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(d['class'], fontsize=13)
            ax.set_title(region, fontsize=14.5)
        np.atleast_1d(axs)[0].set_ylabel('LAI trend (% per decade)', fontsize=13)
        np.atleast_1d(axs)[0].legend(fontsize=9.5, frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(self.outdir + f'attribution_model2_bar_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'attribution_model2_bar_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


class Model2_annual_perennial(Attribution_model2):
    ## step 3: does C shrink when annual-grass-dominated pixels get their own climate sensitivity? (2026-11-21)
    ## problem: layer 1 of Model 2 fits ONE set of beta per vegetation type. Pixels dominated by annual grass
    ##   (e.g. NAIP site G01, cheatgrass) boom in wet years and crash in dry years, much more than the pooled beta;
    ##   the part of their trend that follows the climate is then under-estimated in A and ends up in C
    ## change: for Grass and Shrub, layer 1 is fitted per vegetation type x herbaceous group (6 sets of beta)
    ##   annual    : 2003-2024 mean RAP AFG cover > mean RAP PFG cover
    ##   perennial : otherwise
    ##   Forest and Savanna keep one pooled beta (LAI mostly from trees, herbaceous layer matters little)
    ##   (a group with < 30 pixels uses the pooled beta of its vegetation type)
    ## 2026-11-21: this is the main Model 2 of the paper (Attribution_figures reads its output)
    ## everything else as Model 2: same pixels, same climate variables, layer 2 per vegetation type -> B, C
    ## compare: C of this model vs standard Model 2, per vegetation type and per herbaceous group
    ## output: Result2\03_response\model2_annual_perennial\
    def __init__(self):
        super().__init__()
        self.outdir = this_root + rf'Result2\03_response\model2_annual_perennial\\'
        T.mk_dir(self.outdir, force=True)
        self.mapper = Map_plotter(self.outdir)
        self.m2_dff = this_root + rf'Result2\03_response\attribution_model2\attribution_model2_{self.tag}.df'
        self.group_list = ['annual', 'perennial']
        self.split_class_list = ['Grass', 'Shrub']   ## vegetation types split into annual / perennial
        self.herb_group = None

    def run(self):
        super().run()   ## Model 2 with the layer 1 below
        self.compare()
        pass

    def rap_missing_check(self, info, rap):
        ## used here as the hook where the RAP cover of every pixel is available: herbaceous group per pixel
        super().rap_missing_check(info, rap)
        with np.errstate(invalid='ignore'):
            afg = np.nanmean(rap['RAP_AFG'], axis=1)
            pfg = np.nanmean(rap['RAP_PFG'], axis=1)
        self.herb_group = pd.Series(np.where(afg > pfg, 'annual', 'perennial'), index=info.index)
        self.herb_cover = pd.DataFrame({'AFG_mean': afg, 'PFG_mean': pfg}, index=info.index)
        for lc in self.class_list[1:]:
            m = (info['landcover_type'] == lc).values
            print(f'{lc:8s} annual-dominated pixels: {np.mean(self.herb_group.values[m] == "annual") * 100:.1f}%')

    def layer1(self, info, lai, X, res):
        ## as Model 2, but pooled OLS per vegetation type x herbaceous group
        lai_d = detrend(lai, axis=1)
        X_d = {k: detrend(v, axis=1) for k, v in X.items()}
        X_tr = {k: self.trend(v)[0] for k, v in X.items()}
        names = list(X.keys())
        for k in names:
            res[f'{k}_trend'] = X_tr[k]
        grp = self.herb_group.reindex(info.index).values
        res['herb_group'] = grp
        A = {k: np.full(len(info), np.nan) for k in names}
        result = []
        for lc in self.class_list[1:]:
            m_lc = (info['landcover_type'] == lc).values
            b_pool, r2_pool = self.fit(lai_d, X_d, names, m_lc)
            if lc not in self.split_class_list:
                for j, k in enumerate(names):
                    A[k][m_lc] = b_pool[j] * X_tr[k][m_lc]
                result.append([lc, 'pooled', int(m_lc.sum()), r2_pool] + list(b_pool))
                continue
            for g in self.group_list:
                m = m_lc & (grp == g)
                if m.sum() < 30:
                    b, r2 = b_pool, np.nan
                else:
                    b, r2 = self.fit(lai_d, X_d, names, m)
                for j, k in enumerate(names):
                    A[k][m] = b[j] * X_tr[k][m]
                result.append([lc, g, int(m.sum()), r2] + list(b))
        for k in names:
            res[f'A_{k}'] = A[k]
        res['A'] = sum(A[k] for k in names)
        coef = pd.DataFrame(result, columns=['class', 'herb_group', 'n_pix', 'R2'] + [f'beta_{k}' for k in names])
        print(coef.round(3).to_string())
        return coef

    def fit(self, lai_d, X_d, names, m):
        y = lai_d[m].ravel()
        M = np.column_stack([X_d[k][m].ravel() for k in names])
        b, *_ = np.linalg.lstsq(M, y, rcond=None)
        r2 = 1 - np.sum((y - M @ b) ** 2) / np.sum(y ** 2)
        return b, r2

    def compare(self):
        ## standard Model 2 vs this model: A, B, C per vegetation type and herbaceous group
        new = T.load_df(self.outdir + f'attribution_model2_{self.tag}.df')
        std = T.load_df(self.m2_dff).reindex(new.index)
        df = new[['lon', 'lat', 'area_weight', 'landcover_type', 'herb_group', 'observed']].copy()
        for k in ['A', 'B', 'C']:
            df[f'{k}_standard'] = std[k].values
            df[f'{k}_new'] = new[k].values
        for k in ['A_P', 'A_Pant', 'A_VPD']:   ## climate parts of this model (for Attribution_figures)
            df[f'{k}_new'] = new[k].values
        df = df.join(self.herb_cover)
        T.save_df(df, self.outdir + f'model2_annual_perennial_compare_{self.tag}.df')
        name_list = ['observed', 'A_standard', 'A_new', 'B_standard', 'B_new', 'C_standard', 'C_new',
                     'A_P_new', 'A_Pant_new', 'A_VPD_new']
        result = []
        for lc in self.class_list:
            m_lc = np.ones(len(df), dtype=bool) if lc == 'All' else (df['landcover_type'] == lc).values
            for g in ['all'] + self.group_list:
                m = m_lc & (np.ones(len(df), dtype=bool) if g == 'all' else (df['herb_group'] == g).values)
                m &= np.isfinite(df['C_standard'].values) & np.isfinite(df['C_new'].values)
                if m.sum() < 30:
                    continue
                w = df['area_weight'].values[m]
                row = [lc, g, int(m.sum()), np.sum(w) / np.sum(df['area_weight'].values[m_lc]) * 100]
                for name in name_list:
                    v = df[name].values[m]
                    lo, hi = self.block_ci(df[m], v, w)
                    row += [np.sum(v * w) / np.sum(w), lo, hi]
                result.append(row)
        cols = ['class', 'herb_group', 'n_pix', 'area_%_of_class']
        for name in name_list:
            cols += [name, f'{name}_ci_low', f'{name}_ci_high']
        tab = pd.DataFrame(result, columns=cols)
        tab.to_csv(self.outdir + f'model2_annual_perennial_compare_{self.tag}.csv', index=False)
        print(tab[['class', 'herb_group', 'n_pix', 'area_%_of_class'] + name_list].round(2).to_string())
        self.plot_compare(tab)

    def plot_compare(self, tab):
        ## C standard vs C new: left all pixels per vegetation type, right grass by herbaceous group (A and C)
        fig, axs = plt.subplots(1, 2, figsize=(18 * centimeter_factor, 7 * centimeter_factor),
                                gridspec_kw={'width_ratios': [5, 3]})
        d = tab[tab['herb_group'] == 'all'].set_index('class').reindex(self.class_list)
        x = np.arange(len(d))
        for k, (name, label, color) in enumerate([('C_standard', 'C, standard Model 2', '#9E9E9E'),
                                                  ('C_new', 'C, beta per annual / perennial', '#1F2A33')]):
            err = np.array([d[name] - d[f'{name}_ci_low'], d[f'{name}_ci_high'] - d[name]])
            axs[0].bar(x + (k - 0.5) * 0.38, d[name], width=0.38, color=color, yerr=err, capsize=2,
                       error_kw={'lw': 0.6}, label=label)
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(self.class_list, fontsize=12)
        axs[0].legend(fontsize=10.5, frameon=False, loc='upper left')
        d = tab[(tab['class'] == 'Grass') & (tab['herb_group'].isin(self.group_list))].set_index('herb_group') \
            .reindex(self.group_list)
        x = np.arange(len(d))
        bars = [('A_standard', 'A standard', '#F4A582'), ('A_new', 'A new', '#C0392B'),
                ('C_standard', 'C standard', '#9E9E9E'), ('C_new', 'C new', '#1F2A33')]
        for k, (name, label, color) in enumerate(bars):
            err = np.array([d[name] - d[f'{name}_ci_low'], d[f'{name}_ci_high'] - d[name]])
            axs[1].bar(x + (k - 1.5) * 0.2, d[name], width=0.2, color=color, yerr=err, capsize=1.5,
                       error_kw={'lw': 0.5}, label=label)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels([f'{g}\n({a:.0f}% of grass)' for g, a in zip(self.group_list, d['area_%_of_class'])],
                               fontsize=12)
        axs[1].set_title('Grass', fontsize=13)
        axs[1].legend(fontsize=9.5, frameon=False, ncol=2)
        for ax in axs:
            ax.axhline(0, color='grey', lw=0.5)
            ax.tick_params(labelsize=11)
            ax.spines[['top', 'right']].set_visible(False)
        axs[0].set_ylabel('LAI trend (% per decade)', fontsize=13)
        plt.tight_layout()
        plt.savefig(self.outdir + f'model2_annual_perennial_compare_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'model2_annual_perennial_compare_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


class Phenology_C(Growing_season_trend):
    ## where in the season does the greening sit, and is C (Model 2 remainder) linked to phenology? (2026-11-15)
    ## data  : monthly MODIS LAI 2003-2024 (Data\MODIS_LAI_0706\dic), growing_season.df (gs_start, gs_end),
    ##         Model 2 pixels and C (attribution_model2_2003_2024.df)
    ## monthly LAI is expressed as % of the pixel's growing-season mean LAI, so the mean of the window-month trends
    ##   equals the GS LAI trend (% per decade) of Model 2 ("observed")
    ## per pixel, trend (% per decade, not detrended) of:
    ##   pre   : month before the growing season        post : month after the growing season
    ##   early : first half of the window months        late : second half of the window months (odd length: middle
    ##           month in neither)
    ##   timing: LAI-weighted centre of the season (window +- 2 months, within the calendar year), days per decade;
    ##           negative = the season's LAI mass moves earlier
    ## profile: area-weighted mean monthly trend aligned to the season start (-3 ... +3) and end (-3 ... +3), per class
    ## C link : area-weighted correlation across pixels between C and each metric, per vegetation type
    ## output : Result2\03_response\phenology_C\
    def __init__(self):
        super().__init__(this_root + rf'Result2\03_response\phenology_C\\')
        self.gs_dff = result_root + rf'\Dataframe\growing_season\growing_season.df'
        self.m2_dff = this_root + rf'Result2\03_response\attribution_model2\attribution_model2_{self.tag}.df'
        self.lai_dir = data_root + rf'MODIS_LAI_0706\dic\\'
        self.metric_list = ['observed', 'early', 'late', 'pre', 'post', 'timing']
        self.offset_list = list(range(-3, 4))

    def run(self):
        m2 = T.load_df(self.m2_dff)
        gs = T.load_df(self.gs_dff).reindex(m2.index)
        pix_list = list(m2.index)
        start = gs['gs_start'].values.astype(int) - 1   ## 0 = Jan
        end = gs['gs_end'].values.astype(int) - 1
        ## pixel x year x month, % of the GS mean LAI
        arr = Growing_season_extraction().monthly_values(pix_list, self.lai_dir, 2003)[:, 12:]   ## drop 2002
        arr = arr.reshape(len(pix_list), len(self.year_list), 12)
        arr = arr / gs['GS_mean_LAI'].values[:, None, None] * 100
        tr_month = np.full((len(pix_list), 12), np.nan)
        for k in range(12):
            tr_month[:, k] = self.trend(arr[:, :, k])[0]
        res = self.metrics(arr, tr_month, start, end)
        res['C'] = m2['C'].values
        print('check: GS trend from months vs Model 2 observed, mean abs diff =',
              np.nanmean(np.abs(res['observed'] - m2['observed'].values)).round(4))
        res = pd.DataFrame(res, index=m2.index)
        T.save_df(m2[['area_weight', 'landcover_type', 'lon', 'lat']].join(res), self.outdir + f'phenology_C_{self.tag}.df')
        info = m2[['area_weight', 'landcover_type', 'lon', 'lat']]
        self.table(info, res)
        self.profile(info, tr_month, start, end)
        vlim = np.nanpercentile(np.abs(res['timing']), 90)
        self.save_map(info, res['timing'].values, f'timing_trend_{self.tag}',
                      'Season centre trend (days per decade, negative = earlier)', vlim, 'RdBu_r')
        for name in ['pre', 'post', 'early', 'late']:
            self.save_map(info, res[name].values, f'{name}_trend_{self.tag}',
                          f'LAI trend, {name} (% of GS mean per decade)', np.nanpercentile(np.abs(res['observed']), 90),
                          'RdBu')
        pass

    def metrics(self, arr, tr_month, start, end):
        n = len(start)
        length = (end - start) % 12 + 1
        idx = np.arange(n)
        res = {}
        win = [(start + j) % 12 for j in range(12)]   ## window months in order, j < length
        tr_win = np.stack([tr_month[idx, win[j]] for j in range(12)], axis=1)
        j = np.arange(12)[None, :]
        res['observed'] = np.nanmean(np.where(j < length[:, None], tr_win, np.nan), axis=1)
        half = length // 2
        res['early'] = np.nanmean(np.where(j < half[:, None], tr_win, np.nan), axis=1)
        res['late'] = np.nanmean(np.where((j >= (length - half)[:, None]) & (j < length[:, None]), tr_win, np.nan), axis=1)
        res['pre'] = tr_month[idx, (start - 1) % 12]
        res['post'] = tr_month[idx, (end + 1) % 12]
        ## timing: LAI-weighted centre of the window +- 2 months (clipped to Jan-Dec), per year -> trend
        m = np.arange(12)[None, :]
        lo, hi = np.clip(start - 2, 0, 11), np.clip(end + 2, 0, 11)
        wmask = (m >= lo[:, None]) & (m <= hi[:, None])
        lai = np.where(wmask[:, None, :], np.clip(arr, 0, None), np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            centre = np.nansum(lai * m[None, :, :], axis=2) / np.nansum(lai, axis=2)
        centre = np.where(np.isfinite(lai).sum(axis=2) == wmask.sum(axis=1)[:, None], centre, np.nan)
        res['timing'] = self.trend(centre)[0] * 30.4   ## months -> days per decade
        return res

    def wcorr(self, x, y, w):
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, w = x[ok], y[ok], w[ok]
        xm, ym = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
        return np.sum(w * (x - xm) * (y - ym)) / np.sqrt(np.sum(w * (x - xm) ** 2) * np.sum(w * (y - ym) ** 2))

    def table(self, info, res):
        ## per class: area-weighted mean (95% CI) of each metric, and its correlation with C across pixels
        w_all = info['area_weight'].values
        result = []
        for lc in self.class_list:
            m = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
            for name in self.metric_list + ['C']:
                v = res[name].values[m]
                f = np.isfinite(v)
                w = w_all[m][f]
                lo, hi = self.block_ci(info[m][f], v[f], w)
                r = self.wcorr(res[name].values[m], res['C'].values[m], w_all[m]) if name != 'C' else np.nan
                result.append([lc, name, f.sum(), np.sum(v[f] * w) / np.sum(w), lo, hi, r])
        df_t = pd.DataFrame(result, columns=['class', 'metric', 'n_pix', 'mean', 'ci_low', 'ci_high', 'r_with_C'])
        df_t.to_csv(self.outdir + f'phenology_C_summary_{self.tag}.csv', index=False)
        print(df_t.round(2).to_string())

    def profile(self, info, tr_month, start, end):
        ## area-weighted mean monthly LAI trend aligned to the season start and end, one line per class
        idx = np.arange(len(start))
        w_all = info['area_weight'].values
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, anchor, name in [(axs[0], start, 'start'), (axs[1], end, 'end')]:
            for lc in self.class_list:
                m = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
                y = []
                for off in self.offset_list:
                    v = tr_month[idx, (anchor + off) % 12][m]
                    f = np.isfinite(v)
                    y.append(np.sum(v[f] * w_all[m][f]) / np.sum(w_all[m][f]))
                ax.plot(self.offset_list, y, marker='o', ms=3, lw=1.5 if lc == 'All' else 1, label=lc,
                        color='k' if lc == 'All' else None)
            ax.axhline(0, color='grey', lw=0.5)
            ax.axvspan(0 if name == 'start' else -3.3, 3.3 if name == 'start' else 0, color='#2E7D32', alpha=0.08)
            ax.set_xticks(self.offset_list)
            ax.set_xlabel(f'Month relative to growing-season {name}', fontsize=13)
            ax.set_title(f'aligned to season {name} (green = inside the season)', fontsize=13)
        axs[0].set_ylabel('LAI trend (% of GS mean per decade)', fontsize=13)
        axs[0].legend(fontsize=11, frameon=False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'phenology_profile_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'phenology_profile_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


class Dry_wet_trend(Growing_season_trend):
    ## does the greening also hold in drought years? (2026-11-15)
    ## data: GS.df (GS_LAI_rel, % of the pixel's 2003-2024 mean) and GS SPEI at the pixel's best scale (bivariate.py);
    ##   SPEI keeps its long-term (1958-2024) baseline: 0 = long-term normal, -1 = moderate drought
    ## 1) interaction model, pooled per vegetation type, pixel fixed effects (within-pixel demeaned), area-weighted:
    ##      LAI_rel = a_pix + b * t + beta * SPEI + g * t * SPEI + e        (t in decades, centred on 2013.5)
    ##    LAI trend in a year with SPEI = s: b + g * s  (% per decade); g < 0: the gain is larger in wet years and
    ##    smaller in dry years (the same as the LAI-SPEI sensitivity increasing over time)
    ##    CI: 1 x 1 deg block bootstrap (refit)
    ## 2) categories (fixed SPEI thresholds): dry SPEI <= -1, normal -1 < SPEI < 1, wet SPEI >= 1;
    ##    mean LAI_rel per category in 2003-2013 and 2014-2024 (+ mean SPEI, to see if dry years got drier)
    ## output: Result2\03_response\dry_wet_trend\
    def __init__(self):
        super().__init__(this_root + rf'Result2\03_response\dry_wet_trend\\')
        self.biv_dff = this_root + rf'Result2\03_response\LAI_SPEI_bivariate\LAI_SPEI_bivariate_2003_2024.df'
        self.scale_list = list(range(3, 49, 3))
        self.s_list = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
        self.cat_list = [('dry', -np.inf, -1), ('normal', -1, 1), ('wet', 1, np.inf)]
        self.period_list = [('2003-2013', 2003, 2013), ('2014-2024', 2014, 2024)]
        self.cat_color = {'dry': '#C0392B', 'normal': '#F5B041', 'wet': '#2471A3'}

    def run(self):
        df, info = self.load()
        info = info[info['landcover_type'].isin(self.class_list[1:])]
        best = T.load_df(self.biv_dff)['best_scale'].reindex(info.index).values
        lai = self.matrix(df, info, 'GS_LAI_rel')
        spei = np.full(lai.shape, np.nan)
        for scale in self.scale_list:
            m = best == scale
            if m.any():
                spei[m] = self.matrix(df, info.iloc[np.where(m)[0]], f'GS_SPEI{scale:02d}')
        ok = np.isfinite(lai).all(axis=1) & np.isfinite(spei).all(axis=1)
        info, lai, spei = info[ok], lai[ok], spei[ok]
        print('pixels:', len(info))
        df_m = self.interaction(info, lai, spei)
        df_c = self.categories(info, lai, spei)
        self.plot(df_m, df_c)
        pass

    def interaction(self, info, lai, spei):
        t = (np.array(self.year_list, dtype=float) - np.mean(self.year_list)) / 10   ## decades
        X = np.stack([np.broadcast_to(t, lai.shape), spei, t[None, :] * spei], axis=2)   ## pix x year x 3
        Xd = X - X.mean(axis=1, keepdims=True)
        yd = lai - lai.mean(axis=1, keepdims=True)
        ## per-pixel normal-equation pieces, so the bootstrap only sums them
        XtX = np.einsum('pyi,pyj->pij', Xd, Xd)
        Xty = np.einsum('pyi,py->pi', Xd, yd)
        rng = np.random.default_rng(0)
        result = []
        for lc in self.class_list:
            m = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
            w = info['area_weight'].values[m]
            A, B = XtX[m], Xty[m]
            coef = np.linalg.solve(np.einsum('p,pij->ij', w, A), np.einsum('p,pi->i', w, B))
            block = (np.floor(info['lon'].values[m]) * 1000 + np.floor(info['lat'].values[m])).astype(int)
            ids = np.unique(block)
            idx = {bk: np.where(block == bk)[0] for bk in ids}
            boot = []
            for _ in range(200):
                pick = np.concatenate([idx[bk] for bk in rng.choice(ids, size=len(ids))])
                boot.append(np.linalg.solve(np.einsum('p,pij->ij', w[pick], A[pick]),
                                            np.einsum('p,pi->i', w[pick], B[pick])))
            boot = np.array(boot)
            row = [lc, m.sum(), np.mean(spei[m]), coef[0], coef[1], coef[2],
                   np.percentile(boot[:, 2], 2.5), np.percentile(boot[:, 2], 97.5)]
            for s in self.s_list:
                tb = boot[:, 0] + boot[:, 2] * s
                row += [coef[0] + coef[2] * s, np.percentile(tb, 2.5), np.percentile(tb, 97.5)]
            result.append(row)
        cols = ['class', 'n_pix', 'mean_SPEI', 'b_trend_at_SPEI0', 'beta_SPEI', 'g_interaction', 'g_ci_low',
                'g_ci_high']
        for s in self.s_list:
            cols += [f'trend_at_SPEI_{s}', f'trend_at_SPEI_{s}_ci_low', f'trend_at_SPEI_{s}_ci_high']
        df_m = pd.DataFrame(result, columns=cols)
        df_m.to_csv(self.outdir + f'dry_wet_interaction_{self.tag}.csv', index=False)
        print(df_m[['class', 'n_pix', 'mean_SPEI', 'beta_SPEI', 'g_interaction', 'g_ci_low', 'g_ci_high',
                    'trend_at_SPEI_-1.5', 'trend_at_SPEI_-1', 'trend_at_SPEI_0', 'trend_at_SPEI_0.5']].round(2).to_string())
        return df_m

    def categories(self, info, lai, spei):
        ## mean LAI_rel per SPEI category and period (pixel means first, then area-weighted across pixels)
        years = np.array(self.year_list)
        result = []
        for lc in self.class_list:
            mlc = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
            for cat, lo_s, hi_s in self.cat_list:
                for per, y0, y1 in self.period_list:
                    sel = (spei <= hi_s) if cat == 'dry' else (spei >= lo_s) if cat == 'wet' else \
                        (spei > lo_s) & (spei < hi_s)
                    sel = sel & ((years >= y0) & (years <= y1))[None, :]
                    n_y = sel.sum(axis=1)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        pix_lai = np.nansum(np.where(sel, lai, 0), axis=1) / n_y
                        pix_spei = np.nansum(np.where(sel, spei, 0), axis=1) / n_y
                    m = mlc & (n_y > 0)
                    if m.sum() < 30:
                        continue
                    w = info['area_weight'].values[m]
                    lo, hi = self.block_ci(info[m], pix_lai[m], w)
                    result.append([lc, cat, per, m.sum(), np.sum(n_y[m] * w) / np.sum(w),
                                   np.sum(pix_spei[m] * w) / np.sum(w), np.sum(pix_lai[m] * w) / np.sum(w), lo, hi])
        df_c = pd.DataFrame(result, columns=['class', 'category', 'period', 'n_pix', 'mean_n_years', 'mean_SPEI',
                                             'LAI_rel', 'ci_low', 'ci_high'])
        df_c.to_csv(self.outdir + f'dry_wet_categories_{self.tag}.csv', index=False)
        print(df_c.round(2).to_string())
        return df_c

    def plot(self, df_m, df_c):
        n = len(self.class_list)
        fig, axs = plt.subplots(2, n, figsize=(3 * n, 6))
        for k, lc in enumerate(self.class_list):
            ## top: LAI trend as a function of the year's SPEI
            ax = axs[0, k]
            d = df_m[df_m['class'] == lc]
            if len(d):
                d = d.iloc[0]
                y = [d[f'trend_at_SPEI_{s}'] for s in self.s_list]
                lo = [d[f'trend_at_SPEI_{s}_ci_low'] for s in self.s_list]
                hi = [d[f'trend_at_SPEI_{s}_ci_high'] for s in self.s_list]
                ax.fill_between(self.s_list, lo, hi, color='#7F8C8D', alpha=0.3)
                ax.plot(self.s_list, y, color='k', marker='o', ms=3)
            ax.axhline(0, color='grey', lw=0.5)
            ax.axvline(-1, color='#C0392B', lw=0.5, ls='--')
            ax.set_title(lc, fontsize=14.5)
            ax.set_xlabel('SPEI of the year (best scale)', fontsize=13)
            ## bottom: LAI_rel by category, early vs late period
            ax = axs[1, k]
            d = df_c[df_c['class'] == lc]
            for j, (cat, _, _) in enumerate(self.cat_list):
                for i, (per, _, _) in enumerate(self.period_list):
                    r = d[(d['category'] == cat) & (d['period'] == per)]
                    if len(r) == 0:
                        continue
                    r = r.iloc[0]
                    ax.bar(j + (i - 0.5) * 0.38, r['LAI_rel'], width=0.36, color=self.cat_color[cat],
                           alpha=0.45 if i == 0 else 1,
                           yerr=[[r['LAI_rel'] - r['ci_low']], [r['ci_high'] - r['LAI_rel']]], capsize=2,
                           error_kw={'lw': 0.6})
            ax.axhline(0, color='grey', lw=0.5)
            ax.set_xticks(range(len(self.cat_list)))
            ax.set_xticklabels([c for c, _, _ in self.cat_list], fontsize=13)
        axs[0, 0].set_ylabel('LAI trend (% per decade)', fontsize=13)
        axs[1, 0].set_ylabel('LAI_rel (%), light = 2003-2013, dark = 2014-2024', fontsize=11)
        plt.tight_layout()
        plt.savefig(self.outdir + f'dry_wet_trend_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'dry_wet_trend_{self.tag}.pdf', bbox_inches='tight')
        plt.close()




class Fire_check(Growing_season_trend):
    ## does fire explain the Model 2 results? (2026-11-15)
    ## 1) grass C: is the unexplained greening post-fire recovery? -> C and its aridity hump in never-burned pixels
    ## 2) forest B: how much of the tree-cover loss is inside fire perimeters?
    ## inputs: attribution_model2_2003_2024.df, C_aridity_2003_2024.df (AI), Result\Dataframe\MTBS\MTBS_fire.df
    ## fire groups (share of the pixel inside MTBS perimeters, union over the period):
    ##   unburned           : 1984-2024 < 5%
    ##   burned before 2003 : 1984-2002 >= 10% and 2003-2024 < 5%
    ##   burned 2003-2008 / 2009-2014 / 2015-2024 : 2003-2024 >= 10%, by the last burn year
    ##   minor              : everything else (5-10% burned)
    ## output: Result2\03_response\fire_check\
    def __init__(self):
        super().__init__(this_root + rf'Result2\03_response\fire_check\\')
        self.m2_dff = this_root + rf'Result2\03_response\attribution_model2\attribution_model2_{self.tag}.df'
        self.ca_dff = this_root + rf'Result2\03_response\C_aridity\C_aridity_{self.tag}.df'
        self.fire_dff = result_root + rf'\Dataframe\MTBS\MTBS_fire.df'
        self.group_list = ['unburned', 'burned before 2003', 'burned 2003-2008', 'burned 2009-2014',
                           'burned 2015-2024', 'minor']
        self.bin_edges = [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 10]

    def run(self):
        m2 = T.load_df(self.m2_dff)
        fire = T.load_df(self.fire_dff).reindex(m2.index)
        ai = T.load_df(self.ca_dff)['AI'].reindex(m2.index)
        df = m2.join(fire).assign(AI=ai.values)
        df['fire_group'] = self.fire_group(df)
        T.save_df(df, self.outdir + f'fire_check_{self.tag}.df')
        df_g = self.group_table(df)
        df_b = self.forest_share(df)
        df_ai = self.grass_aridity(df)
        self.plot(df_g, df_ai)
        pass

    def fire_group(self, df):
        all_ = df['burn_frac_1984_2024'].fillna(0).values
        pre = df['burn_frac_1984_2002'].fillna(0).values
        dur = df['burn_frac_2003_2024'].fillna(0).values
        last = df['last_burn_year'].values
        g = np.full(len(df), 'minor', dtype=object)
        g[all_ < 0.05] = 'unburned'
        g[(pre >= 0.1) & (dur < 0.05)] = 'burned before 2003'
        during = dur >= 0.1
        g[during & (last >= 2003) & (last <= 2008)] = 'burned 2003-2008'
        g[during & (last >= 2009) & (last <= 2014)] = 'burned 2009-2014'
        g[during & (last >= 2015)] = 'burned 2015-2024'
        return g

    def group_table(self, df):
        ## per class x fire group: area share and area-weighted observed, A, B, C (95% CI)
        result = []
        for lc in self.class_list:
            mlc = (df['landcover_type'] == lc).values if lc != 'All' else np.ones(len(df), dtype=bool)
            w_lc = df['area_weight'].values[mlc].sum()
            for g in self.group_list + ['all pixels']:
                m = mlc & ((df['fire_group'] == g).values if g != 'all pixels' else True)
                if m.sum() < 30:
                    continue
                w = df['area_weight'].values[m]
                row = [lc, g, m.sum(), w.sum() / w_lc * 100]
                for name in ['observed', 'A', 'B', 'C']:
                    v = df[name].values[m]
                    lo, hi = self.block_ci(df[m], v, w)
                    row += [np.sum(v * w) / np.sum(w), lo, hi]
                result.append(row)
        cols = ['class', 'fire_group', 'n_pix', 'area_%']
        for name in ['observed', 'A', 'B', 'C']:
            cols += [name, f'{name}_ci_low', f'{name}_ci_high']
        df_t = pd.DataFrame(result, columns=cols)
        df_t.to_csv(self.outdir + f'fire_group_summary_{self.tag}.csv', index=False)
        print(df_t[['class', 'fire_group', 'n_pix', 'area_%', 'observed', 'B', 'C', 'C_ci_low', 'C_ci_high']].round(2).to_string())
        return df_t

    def forest_share(self, df):
        ## share of the area-weighted LAI loss (observed < 0), of B and of the tree-cover loss inside burned pixels
        result = []
        burned = df['fire_group'].isin(['burned 2003-2008', 'burned 2009-2014', 'burned 2015-2024']).values
        for lc in self.class_list:
            m = (df['landcover_type'] == lc).values if lc != 'All' else np.ones(len(df), dtype=bool)
            w = df['area_weight'].values
            row = [lc, np.sum(w[m & burned]) / np.sum(w[m]) * 100]
            for name in ['observed', 'B', 'RAP_TRE_trend']:
                loss = np.where(df[name].values < 0, df[name].values, 0) * w
                row.append(np.sum(loss[m & burned]) / np.sum(loss[m]) * 100 if np.sum(loss[m]) != 0 else np.nan)
            result.append(row)
        df_b = pd.DataFrame(result, columns=['class', 'burned_2003_2024_area_%', 'share_of_LAI_loss_in_burned_%',
                                             'share_of_negative_B_in_burned_%', 'share_of_tree_cover_loss_in_burned_%'])
        df_b.to_csv(self.outdir + f'fire_loss_share_{self.tag}.csv', index=False)
        print(df_b.round(1).to_string())
        return df_b

    def grass_aridity(self, df):
        ## grass C per aridity bin: all pixels vs never-burned pixels
        d = df[df['landcover_type'] == 'Grass']
        b = np.digitize(d['AI'].values, self.bin_edges) - 1
        result = []
        for sub in ['all pixels', 'unburned']:
            msub = np.ones(len(d), dtype=bool) if sub == 'all pixels' else (d['fire_group'] == 'unburned').values
            for k in range(len(self.bin_edges) - 1):
                m = msub & (b == k) & np.isfinite(d['C'].values)
                if m.sum() < 30:
                    continue
                w = d['area_weight'].values[m]
                v = d['C'].values[m]
                lo, hi = self.block_ci(d[m], v, w)
                result.append(['Grass', sub, self.bin_edges[k], self.bin_edges[k + 1], m.sum(),
                               np.sum(d['AI'].values[m] * w) / np.sum(w), np.sum(v * w) / np.sum(w), lo, hi])
        df_ai = pd.DataFrame(result, columns=['class', 'subset', 'AI_low', 'AI_high', 'n_pix', 'AI_mean', 'C', 'ci_low',
                                              'ci_high'])
        df_ai.to_csv(self.outdir + f'fire_grass_C_aridity_{self.tag}.csv', index=False)
        print(df_ai.round(2).to_string())
        return df_ai

    def plot(self, df_g, df_ai):
        n = len(self.class_list)
        fig, axs = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.6))
        color = {'observed': '#2E7D32', 'B': '#8E44AD', 'C': '#7F8C8D'}
        for k, lc in enumerate(self.class_list):
            ax = axs[k]
            d = df_g[(df_g['class'] == lc) & (df_g['fire_group'] != 'minor')].reset_index(drop=True)
            x = np.arange(len(d))
            for j, name in enumerate(['observed', 'B', 'C']):
                err = np.array([d[name] - d[f'{name}_ci_low'], d[f'{name}_ci_high'] - d[name]])
                ax.bar(x + (j - 1) * 0.27, d[name], width=0.27, color=color[name], yerr=err, capsize=1.5,
                       error_kw={'lw': 0.5}, label=name)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{g}\n({a:.0f}%)" for g, a in zip(d['fire_group'], d['area_%'])], fontsize=9,
                               rotation=45, ha='right')
            ax.set_title(lc, fontsize=14.5)
        axs[0].set_ylabel('LAI trend (% per decade)', fontsize=13)
        axs[0].legend(fontsize=9.5, frameon=False)
        ax = axs[n]
        for sub, c in [('all pixels', 'k'), ('unburned', '#D9480F')]:
            d = df_ai[df_ai['subset'] == sub]
            ax.errorbar(d['AI_mean'], d['C'], yerr=[d['C'] - d['ci_low'], d['ci_high'] - d['C']], marker='o', ms=3,
                        capsize=2, color=c, label=sub)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Aridity index P / PET', fontsize=13)
        ax.set_title('Grass C vs aridity', fontsize=14.5)
        ax.legend(fontsize=11, frameon=False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'fire_check_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'fire_check_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


class Fire_grass(Growing_season_trend):
    ## grass only: LAI trend (observed) and C vs the share of the pixel burned (2026-11-15)
    ## burned share = share of the pixel inside MTBS perimeters (union over the period)
    ##   panel 1: burned in 2003-2024 (fire during the study period: loss, then recovery)
    ##   panel 2: burned in 1984-2002 only (not burned 2003-2024; recovery from older fires)
    ## bins (fixed): 0, 0-5, 5-10, 10-25, 25-50, 50-75, 75-100 %
    ## input : fire_check_2003_2024.df (Fire_check)
    ## output: Result2\03_response\fire_check\grass\
    def __init__(self):
        super().__init__(this_root + rf'Result2\03_response\fire_check\grass\\')
        self.fc_dff = this_root + rf'Result2\03_response\fire_check\fire_check_{self.tag}.df'
        self.lc = 'Grass'
        self.bin_edges = [-0.001, 0.0001, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0001]
        self.bin_label = ['0', '0-5', '5-10', '10-25', '25-50', '50-75', '75-100']
        self.color = {'observed': '#2E7D32', 'C': '#7F8C8D'}

    def run(self):
        df = T.load_df(self.fc_dff)
        df = df[df['landcover_type'] == self.lc]
        dur = df['burn_frac_2003_2024'].fillna(0)
        pre = df['burn_frac_1984_2002'].fillna(0)
        result = []
        for panel, x, m_panel in [('burned 2003-2024', dur, np.ones(len(df), dtype=bool)),
                                  ('burned 1984-2002 only', pre, (dur < 0.0001).values)]:
            b = np.digitize(x.values, self.bin_edges) - 1
            for k, lab in enumerate(self.bin_label):
                m = m_panel & (b == k)
                if m.sum() < 30:
                    continue
                w = df['area_weight'].values[m]
                row = [self.lc, panel, lab, m.sum(), np.sum(w) / df['area_weight'].sum() * 100]
                for name in ['observed', 'A', 'B', 'C']:
                    v = df[name].values[m]
                    lo, hi = self.block_ci(df[m], v, w)
                    row += [np.sum(v * w) / np.sum(w), lo, hi]
                result.append(row)
        cols = ['class', 'panel', 'burned_%', 'n_pix', 'area_%']
        for name in ['observed', 'A', 'B', 'C']:
            cols += [name, f'{name}_ci_low', f'{name}_ci_high']
        df_t = pd.DataFrame(result, columns=cols)
        df_t.to_csv(self.outdir + f'fire_grass_burned_share_{self.tag}.csv', index=False)
        print(df_t[['panel', 'burned_%', 'n_pix', 'area_%', 'observed', 'A', 'B', 'C', 'C_ci_low',
                    'C_ci_high']].round(2).to_string())
        self.plot(df_t)
        pass

    def plot(self, df_t):
        panel_list = list(df_t['panel'].unique())
        fig, axs = plt.subplots(1, len(panel_list), figsize=(5 * len(panel_list), 3.6), sharey=True)
        for ax, panel in zip(np.atleast_1d(axs), panel_list):
            d = df_t[df_t['panel'] == panel].reset_index(drop=True)
            x = np.arange(len(d))
            for j, name in enumerate(['observed', 'C']):
                err = np.array([d[name] - d[f'{name}_ci_low'], d[f'{name}_ci_high'] - d[name]])
                ax.bar(x + (j - 0.5) * 0.38, d[name], width=0.36, color=self.color[name], yerr=err, capsize=2,
                       error_kw={'lw': 0.6}, label=name)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(d['burned_%'], d['n_pix'])], fontsize=11)
            ax.set_xlabel(f'Share of pixel {panel} (%)', fontsize=13)
            ax.set_title(f'{self.lc}: {panel}', fontsize=14.5)
        np.atleast_1d(axs)[0].set_ylabel('LAI trend (% per decade)', fontsize=13)
        np.atleast_1d(axs)[0].legend(fontsize=11, frameon=False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'fire_grass_burned_share_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'fire_grass_burned_share_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


class Model2_robustness(Attribution_model2):
    ## is C an artefact of a linear, constant-sensitivity climate model? (2026-11-15)
    ## A is computed as the trend of the climate-predicted LAI: A = trend(LAI_hat), LAI_hat = sum b_j (F_j - mean F_j),
    ##   b fitted on detrended data (pooled per vegetation type), F = climate features (raw, not detrended)
    ## versions of layer 1 (climate: P_rel, Pant_rel, VPD as in Model 2):
    ##   linear    : F = X                       (= Model 2, check)
    ##   nonlinear : F = X, X^2                  (curvature: floor in dry years, saturation in wet years)
    ##   time      : F = X, (X - mean X) * t     (sensitivity changing over time, t in decades centred on 2013.5;
    ##               X centred per pixel first, otherwise X * t carries a pure time trend, e.g. VPD ~ 1.5 kPa x t)
    ##               A_clim = sum b_k * trend(X_k) (sensitivity at mid-period), A_sens = A - A_clim
    ## layer 2 as Model 2 (observed - A on RAP cover trends, across pixels per vegetation type) -> B, C per version
    ## rolling sensitivity: 11-year windows (2003-2013 ... 2014-2024), pooled linear b per vegetation type
    ## output: Result2\03_response\model2_robustness\
    def __init__(self):
        super().__init__()
        self.outdir = this_root + rf'Result2\03_response\model2_robustness\\'
        T.mk_dir(self.outdir, force=True)
        self.version_list = ['linear', 'nonlinear', 'time']
        self.window = 11

    def run(self):
        df, info = self.load()
        info = info[info['landcover_type'].isin(self.class_list[1:])]
        lai = self.matrix(df, info, 'GS_LAI_rel')
        X = {name: self.matrix(df, info, var, rel) for var, name, rel in self.clim_list}
        rap = {c: self.matrix(df, info, c) for c in self.rap_list}
        ok = np.isfinite(lai).all(axis=1)
        for arr in X.values():
            ok &= np.isfinite(arr).all(axis=1)
        for arr in rap.values():
            ok &= np.isfinite(arr).sum(axis=1) >= self.rap_min_year
        info, lai = info[ok], lai[ok]
        Xs = np.stack([X[k][ok] for k in X], axis=2)   ## pix x year x 3
        rap = {k: v[ok] for k, v in rap.items()}
        print('pixels:', len(info))
        res = pd.DataFrame(index=info.index)
        res['observed'], _ = self.trend(lai)
        for c in self.rap_list:
            res[f'{c}_trend'] = self.trend_nan(rap[c])
        t = (np.array(self.year_list, dtype=float) - np.mean(self.year_list)) / 10
        feat = {'linear': Xs,
                'nonlinear': np.concatenate([Xs, Xs ** 2], axis=2),
                'time': np.concatenate([Xs, (Xs - Xs.mean(axis=1, keepdims=True)) * t[None, :, None]], axis=2)}
        coef_rows = []
        for v in self.version_list:
            A, A_clim, b_dic = self.climate_A(info, lai, feat[v], Xs, v)
            res[f'A_{v}'] = A
            if v == 'time':
                res['A_time_clim'] = A_clim
                res['A_time_sens'] = A - A_clim
            B = self.layer2_fit(info, res, res['observed'].values - A)
            res[f'B_{v}'] = B
            res[f'C_{v}'] = res['observed'].values - A - B
            for lc, b in b_dic.items():
                coef_rows.append([v, lc] + list(np.round(b, 4)))
        pd.DataFrame(coef_rows).to_csv(self.outdir + f'model2_robustness_coef_{self.tag}.csv', index=False, header=False)
        T.save_df(info.join(res), self.outdir + f'model2_robustness_{self.tag}.df')
        df_t = self.table_rob(info, res)
        df_r = self.rolling(info, lai, Xs)
        self.plot(df_t, df_r)
        pass

    def climate_A(self, info, lai, F, Xs, version):
        ## pooled fit per vegetation type on detrended data, A = trend of the predicted LAI (raw features)
        from scipy.signal import detrend
        n, ny, nf = F.shape
        y_d = detrend(lai, axis=1)
        F_d = detrend(F, axis=1)
        F_c = F - F.mean(axis=1, keepdims=True)
        A = np.full(n, np.nan)
        A_clim = np.full(n, np.nan)
        b_dic = {}
        tr_X = np.stack([self.trend(Xs[:, :, k])[0] for k in range(Xs.shape[2])], axis=1)
        for lc in self.class_list[1:]:
            m = (info['landcover_type'] == lc).values
            b, *_ = np.linalg.lstsq(F_d[m].reshape(-1, nf), y_d[m].ravel(), rcond=None)
            b_dic[lc] = b
            A[m] = self.trend(F_c[m] @ b)[0]
            A_clim[m] = tr_X[m] @ b[:Xs.shape[2]]
        return A, A_clim, b_dic

    def layer2_fit(self, info, res, gap):
        ## across pixels per vegetation type: gap on RAP cover trends -> B (no bootstrap)
        B = np.full(len(info), np.nan)
        cols = [f'{c}_trend' for c in self.rap_list]
        for lc in self.class_list[1:]:
            m = (info['landcover_type'] == lc).values
            Z = res.loc[m, cols].values
            g = self.wls(Z, gap[m], info['area_weight'].values[m])
            B[m] = Z @ g[1:]
        return B

    def table_rob(self, info, res):
        name_list = ['observed'] + [f'{p}_{v}' for v in self.version_list for p in ['A', 'B', 'C']] + \
                    ['A_time_clim', 'A_time_sens']
        result = []
        for lc in self.class_list:
            m = (info['landcover_type'] == lc).values if lc != 'All' else np.ones(len(info), dtype=bool)
            w = info['area_weight'].values[m]
            row = [lc, m.sum()]
            for name in name_list:
                v = res[name].values[m]
                lo, hi = self.block_ci(info[m], v, w)
                row += [np.sum(v * w) / np.sum(w), lo, hi]
            result.append(row)
        cols = ['class', 'n_pix']
        for name in name_list:
            cols += [name, f'{name}_ci_low', f'{name}_ci_high']
        df_t = pd.DataFrame(result, columns=cols)
        df_t.to_csv(self.outdir + f'model2_robustness_summary_{self.tag}.csv', index=False)
        print(df_t[['class', 'observed', 'A_linear', 'C_linear', 'A_nonlinear', 'C_nonlinear', 'A_time', 'A_time_clim',
                    'A_time_sens', 'C_time']].round(2).to_string())
        return df_t

    def rolling(self, info, lai, Xs):
        ## pooled linear sensitivity per 11-year window
        from scipy.signal import detrend
        name_list = [name for _, name, _ in self.clim_list]
        result = []
        for i0 in range(len(self.year_list) - self.window + 1):
            sl = slice(i0, i0 + self.window)
            y_d = detrend(lai[:, sl], axis=1)
            F_d = detrend(Xs[:, sl, :], axis=1)
            centre = self.year_list[i0] + self.window // 2
            for lc in self.class_list[1:]:
                m = (info['landcover_type'] == lc).values
                b, *_ = np.linalg.lstsq(F_d[m].reshape(-1, Xs.shape[2]), y_d[m].ravel(), rcond=None)
                result.append([lc, centre] + list(b))
        df_r = pd.DataFrame(result, columns=['class', 'window_centre'] + [f'beta_{k}' for k in name_list])
        df_r.to_csv(self.outdir + f'model2_rolling_beta_{self.tag}.csv', index=False)
        print(df_r.round(3).to_string())
        return df_r

    def plot(self, df_t, df_r):
        n = len(self.class_list)
        fig, axs = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
        name_list = [('C_linear', 'C linear', '#7F8C8D'), ('C_nonlinear', 'C nonlinear', '#566573'),
                     ('C_time', 'C time-varying', '#2C3E50'), ('A_time_sens', 'A from sensitivity change', '#D9480F')]
        for k, lc in enumerate(self.class_list):
            ax = axs[0, k]
            d = df_t[df_t['class'] == lc].iloc[0]
            for j, (name, label, c) in enumerate(name_list):
                ax.bar(j, d[name], color=c, yerr=[[d[name] - d[f'{name}_ci_low']], [d[f'{name}_ci_high'] - d[name]]],
                       capsize=2, error_kw={'lw': 0.6}, label=label)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks([])
            ax.set_title(f"{lc} (observed {d['observed']:.1f})", fontsize=14.5)
            ax = axs[1, k]
            if lc != 'All':
                d = df_r[df_r['class'] == lc]
                for (_, name, _), c in zip(self.clim_list, ['#5DADE2', '#1F618D', '#C0392B']):
                    ax.plot(d['window_centre'], d[f'beta_{name}'] / d[f'beta_{name}'].abs().mean(), marker='o', ms=3,
                            color=c, label=name)
                ax.axhline(0, color='grey', lw=0.5)
                ax.set_xlabel('Centre of 11-year window', fontsize=13)
            else:
                ax.axis('off')
        axs[0, 0].set_ylabel('% per decade', fontsize=13)
        axs[0, 0].legend(fontsize=9.5, frameon=False)
        axs[1, 1].set_ylabel('beta / mean |beta|', fontsize=13)
        axs[1, 1].legend(fontsize=11, frameon=False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'model2_robustness_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'model2_robustness_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


## Attribution_model2_SWE: not used any more (2026-11-20), kept commented out
# class Attribution_model2_SWE(Attribution_model2):
#     ## Model 2 with snow: layer 1 = P_rel, Pant_rel, VPD + GS_SWE (2026-11-15)
#     ## GS_SWE: peak monthly SWE (mm, ERA5, Data\SWE\dic, 2002-2024) from 8 months before the growing-season start to
#     ##   the start month (same window as GS_P_antecedent); absolute mm, not % (no-snow pixels have 0)
#     ## everything else as Model 2 (layer 2 RAP, B, C, tables, maps)
#     ## output: Result2\03_response\attribution_model2_SWE\
#     def __init__(self):
#         super().__init__()
#         self.outdir = this_root + rf'Result2\03_response\attribution_model2_SWE\\'
#         T.mk_dir(self.outdir, force=True)
#         self.mapper = Map_plotter(self.outdir)   ## maps into this folder, not the Model 2 folder
#         self.clim_list = self.clim_list + [('GS_SWE', 'SWE', False)]
#         self.comp_list = ['observed', 'A_P', 'A_Pant', 'A_VPD', 'A_SWE', 'A', 'B', 'C']
#         self.color_dic['A_SWE'] = '#85C1E9'
#         self.swe_dir = data_root + rf'SWE\dic\\'
#         self.gs_dff = result_root + rf'\Dataframe\growing_season\growing_season.df'

#     def load(self):
#         df, info = super().load()
#         gs = T.load_df(self.gs_dff).reindex(info.index)
#         pix_list = list(info.index)
#         swe = Growing_season_extraction().monthly_values(pix_list, self.swe_dir, 2002)   ## Jan 2002 ... Dec 2024
#         swe[swe < -9000] = np.nan
#         swe = np.clip(swe, 0, None)
#         start = gs['gs_start'].values.astype(int) - 1
#         out = np.full((len(pix_list), len(self.year_list)), np.nan)
#         for j, year in enumerate(self.year_list):
#             k_start = (year - 2002) * 12 + start   ## index of the growing-season start month
#             idx = k_start[:, None] + np.arange(-8, 1)[None, :]
#             v = np.take_along_axis(swe, idx, axis=1)
#             out[:, j] = np.where(np.isfinite(v).all(axis=1), np.nanmax(v, axis=1), np.nan)
#         long = pd.DataFrame({'pix': [p for p in pix_list for _ in self.year_list],
#                              'year': self.year_list * len(pix_list), 'GS_SWE': out.ravel()})
#         df = df.drop(columns=['GS_SWE'], errors='ignore').merge(long, on=['pix', 'year'], how='left')
#         w = info['area_weight'].values
#         snow = np.nanmean(out, axis=1) > 1
#         print(f'GS_SWE: pixels with mean peak SWE > 1 mm: {np.sum(w[snow]) / np.sum(w) * 100:.1f}% of area')
#         return df, info


## Attribution_model3: not used any more (2026-11-20), kept commented out
# class Attribution_model3(Attribution_model2):
#     ## Model 3 = Model 2 + rainfall structure (Daymet daily precipitation, growing-season window) (2026-11-15)
#     ## layer 1: LAI_rel' = b1 P_rel' + b2 Pant_rel' + b3 VPD' + b4 wetdays' + b5 SDII' + b6 CDD' + e   (' = detrended)
#     ##   GS_wetdays (days >= 1 mm), GS_SDII (mm per wet day), GS_CDD (longest dry spell, days): absolute units
#     ##   P (TerraClimate total) stays in, so b4-b6 are the effect of how the rain falls at a given total
#     ##   check VIF (wetdays and P are related)
#     ## layer 2, B, C, tables and maps as Model 2
#     ## input : Result\Dataframe\Daymet\Daymet_GS_metrics.df (Data_processing.py, Data_processing_Daymet)
#     ## output: Result2\03_response\attribution_model3\
#     def __init__(self):
#         super().__init__()
#         self.outdir = this_root + rf'Result2\03_response\attribution_model3\\'
#         T.mk_dir(self.outdir, force=True)
#         self.mapper = Map_plotter(self.outdir)   ## maps into this folder, not the Model 2 folder
#         self.daymet_dff = result_root + rf'\Dataframe\Daymet\Daymet_GS_metrics.df'
#         self.rain_list = [('GS_wetdays', 'wet', False), ('GS_SDII', 'SDII', False), ('GS_CDD', 'CDD', False)]
#         self.clim_list = self.clim_list + self.rain_list
#         self.comp_list = ['observed', 'A_P', 'A_Pant', 'A_VPD', 'A_wet', 'A_SDII', 'A_CDD', 'A', 'B', 'C']
#         self.color_dic.update({'A_wet': '#76D7C4', 'A_SDII': '#F7DC6F', 'A_CDD': '#E59866'})

#     def load(self):
#         df, info = super().load()
#         dm = T.load_df(self.daymet_dff)[['pix', 'year'] + [v for v, _, _ in self.rain_list]]
#         df = df.drop(columns=[v for v, _, _ in self.rain_list], errors='ignore').merge(dm, on=['pix', 'year'],
#                                                                                       how='left')
#         print('Daymet rainfall structure merged:', dm['pix'].nunique(), 'pixels')
#         return df, info


class C_exclusion(Growing_season_trend):
    ## does C survive after removing land-use change, fire and vegetation-type change? (2026-11-20)
    ## main line (cumulative, each step removes pixels from the previous one):
    ##   0 all           : Model 2 pixels
    ##   1 - land use    : NLCD human_change (to / from developed, hay, crops) >= 5% of the pixel, 2003-2024
    ##   2 - fire        : MTBS burned share 2003-2024 >= 5%
    ##   3 - type change : NLCD natural_change (between natural classes) >= 10%
    ##   4 - AFG change  : 10% largest |dAFG| (RAP annual forb & grass cover trend) within each vegetation type
    ##                     -> 'clean' pixels (strong cheatgrass gain / loss removed; AFG is also in B, layer 2)
    ## vegetation-type change: grass C vs NLCD shrub_to_grass / grass_to_shrub, on step 2 pixels (natural change kept)
    ## inputs: fire_check_2003_2024.df (Model 2 + MTBS + AI), Result\Dataframe\NLCD\NLCD_change_2003_2024.df
    ## output: Result2\03_response\C_exclusion\
    def __init__(self):
        super().__init__(this_root + rf'Result2\03_response\C_exclusion\\')
        self.fc_dff = this_root + rf'Result2\03_response\fire_check\fire_check_{self.tag}.df'
        self.nlcd_dff = result_root + rf'\Dataframe\NLCD\NLCD_change_2003_2024.df'
        self.human_thr = 0.05
        self.fire_thr = 0.05
        self.natural_thr = 0.10
        self.afg_pct = 90
        self.step_list = ['0 all', '1 - land use', '2 - fire', '3 - type change', '4 - AFG change (clean)']
        self.comp_list = ['observed', 'A', 'B', 'C']
        self.type_bin_edges = [-0.001, 0.0001, 0.01, 0.05, 0.1, 0.25, 1.0001]
        self.type_bin_label = ['0', '0-1', '1-5', '5-10', '10-25', '25-100']
        self.color = {'observed': '#2E7D32', 'A': '#E67E22', 'B': '#8E44AD', 'C': '#7F8C8D'}

    def run(self):
        df = self.load_all()
        step = self.steps(df)
        T.save_df(df.assign(**{f'keep_{k}': v for k, v in enumerate(step.values())}),
                  self.outdir + f'C_exclusion_{self.tag}.df')
        df_t = self.step_table(df, step)
        df_type = self.type_change(df, step[self.step_list[2]])
        self.plot(df_t, df_type)
        self.maps(df, step)
        pass

    def load_all(self):
        df = T.load_df(self.fc_dff)
        nlcd = T.load_df(self.nlcd_dff)
        nlcd = nlcd[[c for c in nlcd.columns if c.startswith('nlcd_') and c != 'nlcd_years']]
        df = df.join(nlcd)
        print('Model 2 pixels:', len(df), '| with NLCD:', int(df['nlcd_changed'].notna().sum()))
        return df

    def steps(self, df):
        ## boolean 'kept' masks per step; NLCD missing -> treated as no change (printed)
        hum = df['nlcd_human_change'].fillna(0).values
        nat = df['nlcd_natural_change'].fillna(0).values
        fire = df['burn_frac_2003_2024'].fillna(0).values
        s0 = np.ones(len(df), dtype=bool)
        s1 = s0 & (hum < self.human_thr)
        s2 = s1 & (fire < self.fire_thr)
        s3 = s2 & (nat < self.natural_thr)
        ## AFG: top 10% |dAFG| within each vegetation type, among the step 3 pixels
        s4 = s3.copy()
        afg = np.abs(df['RAP_AFG_trend'].values)
        for lc in self.class_list[1:]:
            m = s3 & (df['landcover_type'] == lc).values & np.isfinite(afg)
            if m.sum() == 0:
                continue
            thr = np.percentile(afg[m], self.afg_pct)
            s4[m & (afg > thr)] = False
        return dict(zip(self.step_list, [s0, s1, s2, s3, s4]))

    def step_table(self, df, step):
        ## per class x step: n, share of the class area kept, area-weighted observed / A / B / C with 95% CI
        result = []
        for lc in self.class_list:
            mlc = (df['landcover_type'] == lc).values if lc != 'All' else np.ones(len(df), dtype=bool)
            w_lc = df['area_weight'].values[mlc].sum()
            for name, keep in step.items():
                m = mlc & keep
                if m.sum() < 30:
                    continue
                w = df['area_weight'].values[m]
                row = [lc, name, m.sum(), w.sum() / w_lc * 100]
                for comp in self.comp_list:
                    v = df[comp].values[m]
                    lo, hi = self.block_ci(df[m], v, w)
                    row += [np.sum(v * w) / np.sum(w), lo, hi]
                result.append(row)
        cols = ['class', 'step', 'n_pix', 'area_kept_%']
        for comp in self.comp_list:
            cols += [comp, f'{comp}_ci_low', f'{comp}_ci_high']
        df_t = pd.DataFrame(result, columns=cols)
        df_t.to_csv(self.outdir + f'C_exclusion_steps_{self.tag}.csv', index=False)
        print(df_t[['class', 'step', 'n_pix', 'area_kept_%', 'observed', 'B', 'C', 'C_ci_low', 'C_ci_high']].round(2)
              .to_string())
        return df_t

    def type_change(self, df, keep):
        ## grass C (and observed) per bin of NLCD shrub_to_grass / grass_to_shrub, after land use and fire removed
        d = df[keep & (df['landcover_type'] == 'Grass').values]
        result = []
        for var in ['nlcd_shrub_to_grass', 'nlcd_grass_to_shrub', 'nlcd_natural_change']:
            b = np.digitize(d[var].fillna(0).values, self.type_bin_edges) - 1
            for k, lab in enumerate(self.type_bin_label):
                m = b == k
                if m.sum() < 30:
                    continue
                w = d['area_weight'].values[m]
                row = ['Grass', var, lab, m.sum(), w.sum() / d['area_weight'].sum() * 100]
                for comp in ['observed', 'C']:
                    v = d[comp].values[m]
                    lo, hi = self.block_ci(d[m], v, w)
                    row += [np.sum(v * w) / np.sum(w), lo, hi]
                result.append(row)
        cols = ['class', 'variable', 'share_%', 'n_pix', 'area_%']
        for comp in ['observed', 'C']:
            cols += [comp, f'{comp}_ci_low', f'{comp}_ci_high']
        df_type = pd.DataFrame(result, columns=cols)
        df_type.to_csv(self.outdir + f'C_type_change_grass_{self.tag}.csv', index=False)
        print(df_type.round(2).to_string())
        return df_type

    def plot(self, df_t, df_type):
        ## left: C (and observed, B) per step for each class; right: grass C vs type change share
        cls = self.class_list
        var_list = list(df_type['variable'].unique())
        fig, axs = plt.subplots(1, len(cls) + len(var_list), figsize=(3.0 * (len(cls) + len(var_list)), 3.8))
        for ax, lc in zip(axs, cls):
            d = df_t[df_t['class'] == lc].reset_index(drop=True)
            x = np.arange(len(d))
            for j, comp in enumerate(['observed', 'B', 'C']):
                err = np.array([d[comp] - d[f'{comp}_ci_low'], d[f'{comp}_ci_high'] - d[comp]])
                ax.bar(x + (j - 1) * 0.27, d[comp], width=0.27, color=self.color[comp], yerr=err, capsize=1.5,
                       error_kw={'lw': 0.5}, label=comp)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s}\n({a:.0f}%)" for s, a in zip(d['step'], d['area_kept_%'])], fontsize=9,
                               rotation=45, ha='right')
            ax.set_title(lc, fontsize=14.5)
        axs[0].set_ylabel('LAI trend (% per decade)', fontsize=13)
        axs[0].legend(fontsize=9.5, frameon=False)
        for ax, var in zip(axs[len(cls):], var_list):
            d = df_type[df_type['variable'] == var].reset_index(drop=True)
            x = np.arange(len(d))
            for j, comp in enumerate(['observed', 'C']):
                err = np.array([d[comp] - d[f'{comp}_ci_low'], d[f'{comp}_ci_high'] - d[comp]])
                ax.bar(x + (j - 0.5) * 0.38, d[comp], width=0.36, color=self.color[comp], yerr=err, capsize=1.5,
                       error_kw={'lw': 0.5}, label=comp)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(d['share_%'], d['n_pix'])], fontsize=9.5)
            ax.set_xlabel(var.replace('nlcd_', '') + ' (% of pixel)', fontsize=13)
            ax.set_title('Grass, land use & fire removed', fontsize=13)
        plt.tight_layout()
        plt.savefig(self.outdir + f'C_exclusion_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'C_exclusion_{self.tag}.pdf', bbox_inches='tight')
        plt.close()

    def maps(self, df, step):
        ## C maps: all pixels and clean pixels (same colour range); the clean map is the base for NAIP hotspots
        vlim = np.nanpercentile(np.abs(df['C']), 90)
        clean = step[self.step_list[4]]
        for name, keep in [('all', np.ones(len(df), dtype=bool)), ('clean', clean)]:
            info = df[keep]
            self.save_map(info, info['C'].values, f'C_{name}_{self.tag}', f'Model 2 C, {name} pixels (% per decade)',
                          vlim, 'RdBu')
        removed = df.loc[~clean, ['landcover_type']].copy()
        removed['reason'] = np.where(~step[self.step_list[1]][~clean], 'land use',
                                     np.where(~step[self.step_list[2]][~clean], 'fire',
                                              np.where(~step[self.step_list[3]][~clean], 'type change', 'AFG')))
        print(removed.groupby(['landcover_type', 'reason']).size().unstack(fill_value=0))


class Attribution_figures():
    ## paper figures drawn from saved results only (no model is rerun) (2026-11-20)
    ## one function per figure; the file name = the function name + period
    ##   decouple_SPEI_LAI              : Model 1, observed vs drought-expected LAI trend (beta x SPEI trend)
    ##   attribute_LAI_trend_bar        : Model 2, grouped bars of climate (P, Pant, VPD), composition, unexplained, observed
    ##                                    Grass and Shrub also split into annual / perennial (2026-11-21)
    ## Model 2 = Model2_annual_perennial (beta per annual / perennial for Grass and Shrub, 2026-11-21);
    ##   set self.m2_dir = 'attribution_model2' for the standard Model 2
    ##   attribute_LAI_trend_maps       : Model 2, maps of climate / composition / unexplained
    ##   robustness_grass_unexplained   : grass unexplained trend under every check (forest plot)
    ## output: Result2\03_response\attribution_figures\
    def __init__(self):
        self.outdir = this_root + rf'Result2\03_response\attribution_figures\\'
        T.mk_dir(self.outdir, force=True)
        self.tag = '2003_2024'
        self.class_list = ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']
        self.resp = this_root + rf'Result2\03_response\\'
        self.m2_dir = 'model2_annual_perennial'   ## main Model 2 (2026-11-21); 'attribution_model2' = standard
        self.group_list = ['annual', 'perennial']
        ## Model 2 terms: (column, label, colour)
        self.term_list = [('A_P', 'Climate: growing-season P', '#F4A261'),
                          ('A_Pant', 'Climate: antecedent P', '#E9C46A'),
                          ('A_VPD', 'Climate: VPD', '#C2410C'),
                          ('B', 'Vegetation composition (RAP)', '#7B5EA7'),
                          ('C', 'Unexplained', '#8C8C8C')]
        ## ternary map palettes, end colours (climate, composition, unexplained); try several, keep one (2026-11-20)
        self.ternary_palette = {
            'blue_magenta_green': [[0.05, 0.25, 0.90], [0.90, 0.05, 0.60], [0.00, 0.65, 0.20]],
            'blue_magenta_orange': [[0.05, 0.25, 0.90], [0.90, 0.05, 0.60], [0.95, 0.50, 0.00]],
            'blue_red_yellow': [[0.10, 0.30, 0.85], [0.85, 0.10, 0.10], [0.98, 0.85, 0.05]],
            'teal_purple_orange': [[0.11, 0.62, 0.47], [0.46, 0.44, 0.70], [0.85, 0.37, 0.01]],
            'red_green_blue': [[0.90, 0.10, 0.10], [0.10, 0.30, 0.90], [0.10, 0.75, 0.15]],
        }
        self.palette = 'blue_magenta_green'

    def run(self):
        # self.decouple_SPEI_LAI()
        self.attribute_LAI_trend_bar()
        # self.attribute_LAI_trend_maps()
        # self.attribute_LAI_trend_ternary_map()   ## dropped 2026-11-20: pixel-level C is mostly trend noise
        # self.robustness_grass_unexplained()
        pass

    def decouple_SPEI_LAI(self):
        f = self.resp + rf'attribution_model1\attribution_model1_summary_{self.tag}.csv'
        d = pd.read_csv(f)
        d = d[d['region'] == 'all'].set_index('class').reindex(self.class_list)
        comp = [('observed', 'Observed', '#2E7D32'), ('expected', 'Expected from drought (β × SPEI trend)', '#C2410C')]
        x = np.arange(len(d))
        wid = 0.36
        fig, ax = plt.subplots(figsize=(13 * centimeter_factor, 7.5 * centimeter_factor))
        for j, (col, lab, c) in enumerate(comp):
            v = d[col].values
            err = np.array([v - d[f'{col}_ci_low'].values, d[f'{col}_ci_high'].values - v])
            ax.bar(x + (j - 0.5) * wid, v, width=wid, color=c, label=lab, yerr=err, capsize=2, error_kw={'lw': 0.7})
        for xi, (_, r) in zip(x, d.iterrows()):
            top = max(r['observed_ci_high'], r['expected_ci_high'], 0)
            ax.text(xi, top + 0.4, f"gap {r['gap']:+.1f}", ha='center', va='bottom', fontsize=10.5,
                    color='#1F2A33', fontweight='bold' if r['gap_ci_low'] * r['gap_ci_high'] > 0 else 'normal')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c}\n(n={int(n)})' for c, n in zip(d.index, d['n_pix'])], fontsize=11)
        ax.set_ylabel('LAI trend (% per decade)', fontsize=13)
        ax.tick_params(axis='y', labelsize=11)
        ax.legend(fontsize=10.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
        ax.set_ylim(min(d['observed_ci_low'].min(), d['expected_ci_low'].min()) - 0.8,
                    max(d['observed_ci_high'].max(), d['expected_ci_high'].max()) + 2.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title('Observed vs drought-expected LAI trend, 2003-2024', fontsize=13)   ## gap in bold: 95% CI excludes 0
        plt.tight_layout()
        self.save(f'decouple_SPEI_LAI_{self.tag}')
        print(d[['n_pix', 'observed', 'expected', 'gap', 'gap_ci_low', 'gap_ci_high']].round(2).to_string())

    def attribute_LAI_trend_bar(self):
        ## grouped bars (2026-11-20, replaces the waterfall): per vegetation type, each Model 2 term and the observed
        ## trend drawn from 0 with its 95% CI (1 x 1 deg block bootstrap)
        ## 2026-11-21: Shrub and Grass: all pixels, then annual- and perennial-dominated pixels (RAP AFG vs PFG mean)
        d = self.model2_bar_table()
        bar_list = self.term_list + [('observed', 'Observed', '#2E7D32')]
        nb = len(bar_list)
        wid = 0.8 / nb
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(22 * centimeter_factor, 8.5 * centimeter_factor))
        for k, (col, lab, c) in enumerate(bar_list):
            v = d[col].values
            err = np.array([v - d[f'{col}_ci_low'].values, d[f'{col}_ci_high'].values - v])
            ax.bar(x + (k - (nb - 1) / 2) * wid, v, width=wid, color=c, label=lab, yerr=err, capsize=1.0,
                   error_kw={'lw': 0.5}, edgecolor='white', linewidth=0.3)
        ax.axhline(0, color='k', lw=0.6)
        for xi, (a, b) in zip(x[:-1], zip(d['class'].values[:-1], d['class'].values[1:])):
            ax.axvline(xi + 0.5, color='#BBBBBB' if a != b else '#EEEEEE', lw=0.6 if a != b else 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(d['label'], fontsize=10.5)
        ax.set_ylabel('Contribution to LAI trend (% per decade)', fontsize=13)
        ax.tick_params(axis='y', labelsize=11)
        ax.legend(fontsize=10.5, frameon=False, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.2))   ## below the axis
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title('Attribution of the LAI trend, 2003-2024 (Model 2, beta per annual / perennial for Grass and Shrub)',
                     fontsize=13)
        plt.tight_layout()
        self.save(f'attribute_LAI_trend_bar_{self.tag}')
        d.round(3).to_csv(self.outdir + f'attribute_LAI_trend_bar_{self.tag}.csv', index=False)
        print(d[['label', 'n_pix', 'observed'] + [c for c, _, _ in self.term_list]].round(2).to_string())

    def model2_bar_table(self):
        ## one row per bar group: class rows from the Model 2 summary, annual / perennial rows from the compare table
        cols = ['observed'] + [c for c, _, _ in self.term_list]
        s = pd.read_csv(self.resp + rf'{self.m2_dir}\attribution_model2_summary_{self.tag}.csv')
        s = s[s['region'] == 'all'].set_index('class')
        g = None
        if self.m2_dir == 'model2_annual_perennial':
            g = pd.read_csv(self.resp + rf'{self.m2_dir}\model2_annual_perennial_compare_{self.tag}.csv')
        rows = []
        for lc in self.class_list:
            split = g is not None and lc in ('Shrub', 'Grass')
            if not split:   ## 2026-11-21: Shrub and Grass shown only as their annual / perennial groups
                r = {'class': lc, 'group': 'all', 'n_pix': s.loc[lc, 'n_pix'],
                     'label': f'{lc}\n(n={int(s.loc[lc, "n_pix"])})'}
                for c in cols:
                    for suf in ['', '_ci_low', '_ci_high']:
                        r[c + suf] = s.loc[lc, c + suf]
                rows.append(r)
                continue
            for grp in self.group_list:
                q = g[(g['class'] == lc) & (g['herb_group'] == grp)].iloc[0]
                r = {'class': lc, 'group': grp, 'n_pix': q['n_pix'],
                     'label': f'{lc} {grp}\n({q["area_%_of_class"]:.0f}% of {lc.lower()})'}
                for c in cols:
                    src = c if c == 'observed' else f'{c}_new'
                    for suf in ['', '_ci_low', '_ci_high']:
                        r[c + suf] = q[src + suf]
                rows.append(r)
        return pd.DataFrame(rows)

    def attribute_LAI_trend_maps(self):
        ## maps of the three parts, same discrete colour steps (% per decade), area-weighted mean in the title
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        from matplotlib.colors import BoundaryNorm
        df = T.load_df(self.resp + rf'{self.m2_dir}\attribution_model2_{self.tag}.df')
        df = df[df['landcover_type'].isin(self.class_list[1:])]
        w = df['area_weight'].values
        bounds = [-10, -6, -3, -1, 1, 3, 6, 10]
        cmap = plt.get_cmap('RdBu', len(bounds) - 1)
        norm = BoundaryNorm(bounds, cmap.N)
        part_list = [('A', 'Climate (P + antecedent P + VPD)'), ('B', 'Vegetation composition (RAP)'),
                     ('C', 'Unexplained')]
        fig = plt.figure(figsize=(24 * centimeter_factor, 9 * centimeter_factor))
        for k, (col, title) in enumerate(part_list):
            ax = fig.add_subplot(1, 3, k + 1, projection=ccrs.PlateCarree())
            arr = D.pix_dic_to_spatial_arr(dict(zip(df.index, df[col].values)))
            arr[arr <= -9999] = np.nan
            im = ax.imshow(arr, cmap=cmap, norm=norm, extent=[-124.529, -102.079, 25.579, 48.979],
                           transform=ccrs.PlateCarree(), interpolation='nearest')
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
            ax.set_extent([-125, -105, 30, 45], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.FixedLocator(range(-125, -100, 5))
            gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
            gl.xlabel_style = {'size': 9.5}
            gl.ylabel_style = {'size': 9.5}
            v = df[col].values
            ok = np.isfinite(v)
            ax.set_title(f'{title}\nmean {np.sum(v[ok] * w[ok]) / np.sum(w[ok]):+.2f}', fontsize=12)
        cax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds, extend='both')
        cb.set_label('Contribution to LAI trend (% per decade)', fontsize=11)
        cb.ax.tick_params(labelsize=10.5)
        plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.18, wspace=0.12)
        self.save(f'attribute_LAI_trend_maps_{self.tag}')

    def robustness_grass_unexplained(self):
        ## grass unexplained trend (C) with 95% CI under every check; band = baseline CI
        def row(f, cond, col='C'):
            d = pd.read_csv(self.resp + f)
            for k, v in cond.items():
                d = d[d[k] == v]
            r = d.iloc[0]
            return r[col], r[f'{col}_ci_low'], r[f'{col}_ci_high']
        g = {'class': 'Grass'}
        m2 = rf'attribution_model2\attribution_model2_summary_{self.tag}.csv'
        rob = rf'model2_robustness\model2_robustness_summary_{self.tag}.csv'
        exc = rf'C_exclusion\C_exclusion_steps_{self.tag}.csv'
        check_list = [
            ('Baseline (Model 2)', row(m2, {**g, 'region': 'all'})),
            ## SWE version dropped (2026-11-20)
            ## Model 3 (rainfall structure) dropped (2026-11-20)
            ('Nonlinear climate response', row(rob, g, 'C_nonlinear')),
            ('Climate sensitivity changing in time', row(rob, g, 'C_time')),
            ('Unburned pixels only (MTBS)', row(rf'fire_check\fire_group_summary_{self.tag}.csv',
                                                {**g, 'fire_group': 'unburned'})),
            ('Land-use change removed (NLCD)', row(exc, {**g, 'step': '1 - land use'})),
            ('Land use, fire, type change, cheatgrass removed', row(exc, {**g, 'step': '4 - AFG change (clean)'})),
        ]
        names = [n for n, _ in check_list]
        vals = np.array([v for _, v in check_list], dtype=float)
        y = np.arange(len(names))[::-1]
        fig, ax = plt.subplots(figsize=(13 * centimeter_factor, 7.5 * centimeter_factor))
        ax.axvspan(vals[0, 1], vals[0, 2], color='#8C8C8C', alpha=0.15, lw=0)
        ax.axvline(vals[0, 0], color='#8C8C8C', lw=0.6, ls='--')
        ax.axvline(0, color='k', lw=0.6)
        ax.errorbar(vals[:, 0], y, xerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], fmt='o', color='#1F2A33',
                    ms=3.5, lw=0.9, capsize=2)
        for yy, v in zip(y, vals):
            ax.text(v[2] + 0.1, yy, f'{v[0]:.2f}', va='center', fontsize=10.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlim(0, max(vals[:, 2]) + 0.8)
        ax.set_xlabel('Grass unexplained LAI trend (% per decade, 95% CI)', fontsize=12)
        ax.tick_params(axis='x', labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        self.save(f'robustness_grass_unexplained_{self.tag}')
        pd.DataFrame(vals, index=names, columns=['C', 'ci_low', 'ci_high']).round(3).to_csv(
            self.outdir + f'robustness_grass_unexplained_{self.tag}.csv')
        print(pd.DataFrame(vals, index=names, columns=['C', 'ci_low', 'ci_high']).round(2).to_string())

    def attribute_LAI_trend_ternary_map(self, min_total=1.0):
        ## ternary map (2026-11-20): each pixel coloured by the shares of |climate|, |composition|, |unexplained|
        ## in |A| + |B| + |C| (which part dominates the pixel, not its sign); pixels with |A| + |B| + |C| < min_total
        ## (% per decade) are grey (all parts small). Right: colour key triangle + dominant part per vegetation type
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        df = T.load_df(self.resp + rf'attribution_model2\attribution_model2_{self.tag}.df')
        df = df[df['landcover_type'].isin(self.class_list[1:])]
        mag = np.abs(df[['A', 'B', 'C']].values)
        tot = mag.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            share = mag / tot[:, None]
        weak = ~(tot >= min_total)
        rgb = self.ternary_rgb(np.nan_to_num(share))
        rgb[weak] = [0.85, 0.85, 0.85]
        ny, nx = 468, 449   ## template grid
        img = np.ones((ny, nx, 4))
        img[..., 3] = 0
        for (r, c), col, ok in zip(df.index, rgb, np.isfinite(tot)):
            if ok:
                img[r, c, :3] = col
                img[r, c, 3] = 1
        fig = plt.figure(figsize=(10, 4.8))
        ax = fig.add_axes([0.02, 0.06, 0.56, 0.88], projection=ccrs.PlateCarree())
        ax.imshow(img, extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(),
                  interpolation='nearest', origin='upper')
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
        ax.set_extent([-125, -105, 30, 45], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(range(-125, -100, 5))
        gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
        ax.set_title('Which part dominates the LAI trend, 2003-2024', fontsize=14.5)
        self.ternary_legend(fig.add_axes([0.62, 0.48, 0.34, 0.46]))
        ## dominant part (largest share) per vegetation type, area %
        w = df['area_weight'].values
        dom = np.where(weak, 3, np.argmax(np.nan_to_num(share), axis=1))
        lab = ['Climate', 'Composition', 'Unexplained', 'All small']
        col = [list(self.ternary_rgb(np.eye(3)[k:k + 1])[0]) for k in range(3)] + [[0.85, 0.85, 0.85]]
        tab = []
        for lc in self.class_list:
            m = (df['landcover_type'] == lc).values if lc != 'All' else np.ones(len(df), dtype=bool)
            tab.append([np.sum(w[m & (dom == k)]) / np.sum(w[m]) * 100 for k in range(4)])
        tab = np.array(tab)
        pd.DataFrame(tab, index=self.class_list, columns=lab).round(1).to_csv(
            self.outdir + f'attribute_LAI_trend_ternary_dominant_{self.tag}.csv')
        self.dominant_bar(fig.add_axes([0.70, 0.08, 0.27, 0.32]), tab, lab, col)
        self.save(f'attribute_LAI_trend_ternary_map_{self.palette}_{self.tag}')
        print(pd.DataFrame(tab, index=self.class_list, columns=lab).round(1).to_string())

    def dominant_bar(self, ax2, tab, lab, col):
        y = np.arange(len(self.class_list))[::-1]
        left = np.zeros(len(self.class_list))
        for k in range(len(lab)):
            ax2.barh(y, tab[:, k], left=left, color=col[k], height=0.7, edgecolor='white', linewidth=0.5, label=lab[k])
            for yy, l0, v in zip(y, left, tab[:, k]):
                if v >= 10:
                    ax2.text(l0 + v / 2, yy, f'{v:.0f}', ha='center', va='center', fontsize=9,
                             color='white' if k in (0, 1, 2) else 'black')
            left += tab[:, k]
        ax2.set_yticks(y)
        ax2.set_yticklabels(self.class_list, fontsize=10.5)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Area where the part is the largest (%)', fontsize=10.5)
        ax2.tick_params(labelsize=9.5)
        ax2.spines[['top', 'right']].set_visible(False)

    def ternary_rgb(self, share):
        ## share: n x 3 (climate, composition, unexplained), rows sum to 1 -> colour from self.palette;
        ## mixed pixels fade to a light centre
        end = np.array(self.ternary_palette[self.palette], dtype=float)
        share = np.asarray(share, dtype=float)
        mix = share @ end
        dist = np.sqrt(np.sum((share - 1 / 3) ** 2, axis=1, keepdims=True)) / (np.sqrt(6) / 3)
        light = np.clip(1 - dist, 0, 1)   ## 1 at the centre, 0 at a corner (smooth, no seams)
        return np.clip(mix + 0.45 * light ** 2 * (1 - mix), 0, 1)   ## stronger colours, light only near the centre

    def ternary_legend(self, ax, n=60):
        ## colour key: climate bottom-left, composition bottom-right, unexplained top
        h = np.sqrt(3) / 2
        pts, shares = [], []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                a, b = i / n, j / n
                c = 1 - a - b
                shares.append([a, b, c])
                pts.append([b + 0.5 * c, h * c])
        pts, shares = np.array(pts), np.array(shares)
        ax.scatter(pts[:, 0], pts[:, 1], c=self.ternary_rgb(shares), s=9, marker='h', linewidths=0)
        ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color='k', lw=0.6)
        ax.text(-0.02, -0.06, 'Climate', ha='center', va='top', fontsize=11)
        ax.text(1.02, -0.06, 'Composition', ha='center', va='top', fontsize=11)
        ax.text(0.5, h + 0.04, 'Unexplained', ha='center', va='bottom', fontsize=11)
        ax.text(0.5, -0.17, 'share of |climate| + |composition| + |unexplained|', ha='center', va='top', fontsize=9,
                color='#555555')
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.25, h + 0.12)
        ax.set_aspect('equal')
        ax.axis('off')

    def save(self, fname):
        plt.savefig(self.outdir + f'{fname}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'{fname}.pdf', bbox_inches='tight')
        plt.close()


def main():
    # Attribution_model1().run()   ## needs bivariate.py first (best SPEI scale)
    # Attribution_model2().run()
    # Model2_annual_perennial().run()  ## main Model 2: beta per annual / perennial for Grass and Shrub (needs Model 2)
    Attribution_figures().run()      ## paper figures from saved results (reads Model2_annual_perennial)
    # Phenology_C().run()            ## needs Model 2 output
    # Dry_wet_trend().run()          ## needs bivariate.py output (best SPEI scale)
    # Water_threshold().run()        ## needs bivariate.py and SPEI_long_term_plot outputs
    # Fire_check().run()             ## needs MTBS_processing.py, Model 2 and the old C_aridity output (AI)
    # Fire_grass().run()             ## needs Fire_check output
    # Model2_robustness().run()      ## C with nonlinear / time-varying climate sensitivity
    # Attribution_model2_SWE().run() ## Model 2 + snow (GS_SWE)
    # Attribution_model3().run()     ## Model 2 + rainfall structure (Daymet: wet days, SDII, CDD)
    # C_exclusion().run()            ## needs Fire_check and Data_processing_Annual_NLCD outputs
    # Attribution_figures().run()    ## paper figures from saved results


if __name__ == '__main__':
    main()
