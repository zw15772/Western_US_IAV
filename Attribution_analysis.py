## Model 1 on the main growing season (GS.df), 2003-2024 (2026-11-15)
## expected LAI trend = beta (LAI vs SPEI at the pixel's best scale) x SPEI trend (best scale)
## compared with the observed LAI trend; gap = observed - expected
## uses the trend tools of growing_season_extraction.py and the best scale from bivariate.py
from growing_season_extraction import *
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
            ax.set_xticklabels(d['class'], fontsize=8)
            ax.set_title(region, fontsize=9)
        np.atleast_1d(axs)[0].set_ylabel('LAI trend (% per decade)', fontsize=8)
        np.atleast_1d(axs)[0].legend(fontsize=7, frameon=False)
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
            ax.text(0.03, 0.97, f'obs > exp: {above:.0f}%', transform=ax.transAxes, ha='left', va='top', fontsize=8)
            ax.set_title(f'{lc} (n={m.sum()})', fontsize=9)
            ax.set_xlabel('Expected LAI trend (% /decade)', fontsize=8)
        axs[0].set_ylabel('Observed LAI trend (% /decade)', fontsize=8)
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
        name_list = ['observed', 'A_P', 'A_Pant', 'A_VPD', 'B', 'C']
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
            ax.set_xticklabels(d['class'], fontsize=8)
            ax.set_title(region, fontsize=9)
        np.atleast_1d(axs)[0].set_ylabel('LAI trend (% per decade)', fontsize=8)
        np.atleast_1d(axs)[0].legend(fontsize=6, frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(self.outdir + f'attribution_model2_bar_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'attribution_model2_bar_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


def main():
    Attribution_model1().run()   ## needs bivariate.py first (best SPEI scale)
    Attribution_model2().run()


if __name__ == '__main__':
    main()
