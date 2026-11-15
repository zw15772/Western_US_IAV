# coding='utf-8'
from __Global__ import *
from scipy import stats
from SPEI_LAI_correlation import SPEI_SNU_LAI_correlation, D   ## reuse df_clean / to_matrix / detrend_rows / plot_map


class Attribution_dataframe():
    ## one pixel-year table for the attribution analysis (summer = Jul-Oct, spring = Mar-May, winter = Nov-Feb)
    ##   SPEI.df:            summer_SNU_LAI, summer_SPEI03 ... SPEI48, cropland_pct, landcover_type, area_weight
    ##   Daymet_seasonal.df: summer_PRCPTOT, spring_PRCPTOT, winter_SWE_max, summer_P_LIGHT / P_MOD / R95p
    ##   TerraClimate dicts: summer mean VPD (kPa) and Tmax (degC), from monthly dicts 1982-2024
    ##   RAP_cover.df:       RAP_AFG, RAP_PFG, RAP_SHR, RAP_TRE (1986-2024, NaN before)
    ## output: Result\Dataframe\attribution\attribution.df
    def __init__(self):
        self.outdir = result_root + rf'\Dataframe\attribution\\'
        T.mk_dir(self.outdir, force=True)
        self.dff = self.outdir + 'attribution.df'
        self.year_list = list(range(1982, 2025))
        self.spei_scale_list = ['03', '06', '09', '12', '15', '18', '21', '24', '27', '30', '33', '36', '39', '42',
                                '45', '48']
        self.daymet_col_list = ['summer_PRCPTOT', 'spring_PRCPTOT', 'winter_SWE_max',
                                'summer_P_LIGHT', 'summer_P_MOD', 'summer_R95p']
        self.rap_col_list = ['RAP_AFG', 'RAP_PFG', 'RAP_SHR', 'RAP_TRE']
        pass

    def run(self):
        # df = self.build()   ## done 2026-11-14
        # T.save_df(df, self.dff)
        # T.df_to_excel(df.head(2000), self.outdir + 'attribution_head')   ## first rows only, for checking
        self.add_modis_lai()   ## again 2026-11-14: summer_LAI.npy remade after the Oct 2022 fix
        # self.add_winter_june_precip()   ## done 2026-11-14
        pass

    def add_winter_june_precip(self):
        ## two water inputs missing from model 2 (added 2026-11-14):
        ##   winter_PRCPTOT: Daymet Nov (previous year) - Feb total, rain + snow (SWE_max misses winter rain in low deserts)
        ##   June_PRCP:      TerraClimate June precipitation (June is in neither spring Mar-May nor summer Jul-Oct)
        df = T.load_df(self.dff)
        df_d = T.load_df(result_root + rf'\Dataframe\Daymet\Daymet_seasonal.df')
        ## TerraClimate ppt monthly dict, Jan 1958 ... Dec 2024
        spatial_dic = T.load_npy_dir(data_root + rf'TerraClimate\ppt\dic\\')
        pix_list, val_list = [], []
        for pix in spatial_dic:
            vals = np.array(spatial_dic[pix], dtype=float)
            if len(vals) % 12 != 0 or np.all(np.isnan(vals)):
                continue
            start_year = 2025 - len(vals) // 12
            june = vals.reshape(-1, 12)[:, 5]
            years = np.arange(start_year, 2025)
            m = np.isin(years, self.year_list)
            pix_list.append(pix)
            val_list.append(june[m])
        print('TerraClimate ppt first year:', start_year)
        df_j = pd.DataFrame({'pix': [p for p in pix_list for _ in self.year_list],
                             'year': self.year_list * len(pix_list),
                             'June_PRCP': np.concatenate(val_list)})
        df = df.drop(columns=[c for c in ['winter_PRCPTOT', 'June_PRCP'] if c in df.columns])
        df = df.merge(df_d[['pix', 'year', 'winter_PRCPTOT']], on=['pix', 'year'], how='left')
        df = df.merge(df_j, on=['pix', 'year'], how='left')
        for col in ['winter_PRCPTOT', 'June_PRCP']:
            print(col, 'non-NaN rows:', df[col].notna().sum(), '/', len(df))
        T.save_df(df, self.dff)
        pass

    def add_modis_lai(self):
        ## MODIS summer LAI (Jul-Oct mean, 2003-2024), Result\MODIS_LAI\summer_LAI.npy {pix: 22 values}
        df = T.load_df(self.dff)
        spatial_dic = T.load_npy(result_root + rf'\MODIS_LAI\summer_LAI.npy')
        modis_years = list(range(2003, 2025))
        pix_list = [pix for pix in spatial_dic if len(spatial_dic[pix]) == len(modis_years)]
        df_m = pd.DataFrame({'pix': [p for p in pix_list for _ in modis_years],
                             'year': modis_years * len(pix_list),
                             'summer_MODIS_LAI': np.concatenate([np.array(spatial_dic[p], dtype=float) for p in pix_list])})
        if 'summer_MODIS_LAI' in df.columns:
            df = df.drop(columns=['summer_MODIS_LAI'])
        df = df.merge(df_m, on=['pix', 'year'], how='left')
        print('summer_MODIS_LAI non-NaN rows:', df['summer_MODIS_LAI'].notna().sum(), '/', len(df))
        T.save_df(df, self.dff)
        pass

    def build(self):
        ## base: SPEI.df (LAI, SPEI, land cover), study region and cropland removed as in the correlation analysis
        mapper = SPEI_SNU_LAI_correlation()
        df = T.load_df(mapper.dff)
        df = mapper.df_clean(df)
        col_list = ['pix', 'year', 'lon', 'lat', 'area_weight', 'cropland_pct', 'landcover_type', 'summer_SNU_LAI'] + \
                   [f'summer_SPEI{s}' for s in self.spei_scale_list]
        df = df[col_list].copy()
        print('SPEI.df', df.shape, 'pixels', df['pix'].nunique())
        ## Daymet seasonal precipitation
        df_d = T.load_df(result_root + rf'\Dataframe\Daymet\Daymet_seasonal.df')
        df = df.merge(df_d[['pix', 'year'] + self.daymet_col_list], on=['pix', 'year'], how='left')
        ## TerraClimate summer VPD and Tmax
        for var in ['vpd', 'tmax']:
            df_t = self.terraclimate_summer(var)
            df = df.merge(df_t, on=['pix', 'year'], how='left')
        ## RAP cover
        df_r = T.load_df(result_root + rf'\Dataframe\RAP\RAP_cover.df')
        df = df.merge(df_r[['pix', 'year'] + self.rap_col_list], on=['pix', 'year'], how='left')
        for col in self.daymet_col_list + ['summer_VPD', 'summer_Tmax'] + self.rap_col_list:
            print(col, 'non-NaN rows:', df[col].notna().sum(), '/', len(df))
        return df

    def terraclimate_summer(self, var):
        ## monthly dict {pix: 516 values, Jan 1982 ... Dec 2024} -> summer (Jul-Oct) mean per year
        spatial_dic = T.load_npy_dir(data_root + rf'TerraClimate\{var}\dic\\')
        n_year = len(self.year_list)
        pix_list, val_list = [], []
        for pix in spatial_dic:
            vals = np.array(spatial_dic[pix], dtype=float)
            if len(vals) != n_year * 12 or np.all(np.isnan(vals)):
                continue
            summer = vals.reshape(n_year, 12)[:, 6:10].mean(axis=1)   ## Jul, Aug, Sep, Oct
            pix_list.append(pix)
            val_list.append(summer)
        name = {'vpd': 'summer_VPD', 'tmax': 'summer_Tmax'}[var]
        df_t = pd.DataFrame({'pix': [p for p in pix_list for _ in range(n_year)],
                             'year': self.year_list * len(pix_list),
                             name: np.concatenate(val_list)})
        print(var, 'pixels', len(pix_list))
        return df_t


class Attribution_analysis():
    ## Why did SPEI decline while summer LAI did not?  (summer LAI as % of the pixel's 1982-2024 mean)
    ## Model 1 (SPEI gap): per pixel, the SPEI scale (03-48 months) with the highest r with detrended LAI;
    ##   beta = slope of detrended LAI on detrended SPEI (1982-2024)
    ##   expected LAI trend E = beta x trend(SPEI); gap G = observed LAI trend - E
    ## Model 2 (attribution):
    ##   step 1: per vegetation type, pooled regression (area-weighted, 95% CI by bootstrapping years)
    ##           detrended LAI = b1 summer P + b2 spring P + b3 winter SWE_max + b4 summer VPD   (all detrended, 1982-2024)
    ##   step 2: climate part A_k = b_k x trend(X_k) per pixel;  R = observed LAI trend - sum(A_k)
    ##   step 3: per vegetation type, across pixels: R = g0 + sum g_j trend(RAP cover_j)  (AFG, PFG, SHR, TRE)
    ##           composition part B = sum g_j trend_j;  remainder C = R - B
    ## trends: OLS per decade, 1986-2024 (RAP period) so that A + B + C = observed trend; model 1 also 1982-2024
    def __init__(self):
        self.dff = result_root + rf'\Dataframe\attribution\attribution.df'
        self.outdir = this_root + rf'Result2\04_attribution\\'
        T.mk_dir(self.outdir, force=True)
        self.year_list = list(range(1982, 2025))
        self.trend_period = (1986, 2024)
        self.model1_period_list = [(1986, 2024), (1982, 2024)]
        self.spei_scale_list = ['03', '06', '09', '12', '15', '18', '21', '24', '27', '30', '33', '36', '39', '42',
                                '45', '48']
        self.clim_list = ['summer_PRCPTOT', 'spring_PRCPTOT', 'winter_SWE_max', 'summer_VPD']
        self.rap_list = ['RAP_AFG', 'RAP_PFG', 'RAP_SHR', 'RAP_TRE']
        self.LAI_threshold = 0.3
        self.lai_col = 'summer_SNU_LAI'
        self.landcover_col = 'landcover_type'
        self.landcover_list = ['Forest', 'Shrub', 'Grass', 'Savanna']
        self.n_boot = 1000
        self.mapper = SPEI_SNU_LAI_correlation()
        self.mapper.outdir = self.outdir   ## plot_map saves into mapper.outdir
        pass

    def run(self):
        df = T.load_df(self.dff)
        pix_info, mat = self.prepare(df)
        self.model1(pix_info, mat)
        self.model2(pix_info, mat)
        pass

    # ------------------------------------------------------------------ data
    def prepare(self, df):
        ## pixel x year matrices; LAI as % of the pixel mean; pixels with complete LAI and climate data
        years = self.year_list
        to_mat = lambda col: df.pivot(index='pix', columns='year', values=col).reindex(columns=years)
        lai = to_mat(self.lai_col)
        lai = lai[lai.notna().all(axis=1) & (lai.mean(axis=1) >= self.LAI_threshold)]
        mat = {'LAI': lai.values / lai.mean(axis=1).values[:, None] * 100}
        pix = lai.index
        for col in [f'summer_SPEI{s}' for s in self.spei_scale_list] + self.clim_list + self.rap_list:
            mat[col] = to_mat(col).reindex(index=pix).values
        ok = np.ones(len(pix), dtype=bool)
        for col in [f'summer_SPEI{s}' for s in self.spei_scale_list] + self.clim_list:
            ok &= np.all(np.isfinite(mat[col]), axis=1)
        m_rap = np.array([y >= 1986 for y in years])
        for col in self.rap_list:
            ok &= np.all(np.isfinite(mat[col][:, m_rap]), axis=1)
        for col in mat:
            mat[col] = mat[col][ok]
        info = df.groupby('pix')[['area_weight', self.landcover_col, 'lon', 'lat']].first().reindex(pix[ok])
        print('pixels used:', ok.sum(), 'of', len(pix))
        return info, mat

    def detrend(self, arr):
        return self.mapper.detrend_rows(arr)

    def trend(self, arr, period):
        ## OLS slope per decade for each row over the period
        years = np.array(self.year_list)
        m = (years >= period[0]) & (years <= period[1])
        x = years[m].astype(float)
        y = arr[:, m]
        xm = x - x.mean()
        return (y - y.mean(axis=1, keepdims=True)) @ xm / np.sum(xm ** 2) * 10

    def wmean(self, v, w):
        ok = np.isfinite(v)
        return np.sum(v[ok] * w[ok]) / np.sum(w[ok])

    def group_dic(self, info):
        dic = {'All': np.ones(len(info), dtype=bool)}
        for lc in self.landcover_list:
            dic[lc] = (info[self.landcover_col] == lc).values
        return dic

    def save_map(self, info, vals, fname, cbar_label, cmap='RdBu', vlim=None, discrete_levels=None):
        arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, vals)))
        arr[arr <= -9999] = np.nan   ## pixels not in the dict are filled with -9999
        D.arr_to_tif(arr.copy(), self.outdir + f'{fname}.tif')   ## copy: arr_to_tif changes NaN in place
        w = info['area_weight'].values
        ok = np.isfinite(vals)
        if discrete_levels is not None:
            self.mapper.plot_map(arr, fname, cmap, None, None, cbar_label, vals[ok], w[ok],
                                 discrete_levels=discrete_levels)
            return
        if vlim is None:
            vlim = np.nanpercentile(np.abs(vals), 90)
        self.mapper.plot_map(arr, fname, cmap, -vlim, vlim, cbar_label, vals[ok], w[ok])

    # ------------------------------------------------------------------ model 1
    def model1(self, info, mat):
        lai_d = self.detrend(mat['LAI'])
        n = len(info)
        r_all = np.full((n, len(self.spei_scale_list)), np.nan)
        beta_all = np.full((n, len(self.spei_scale_list)), np.nan)
        for i, s in enumerate(self.spei_scale_list):
            x = self.detrend(mat[f'summer_SPEI{s}'])
            r_all[:, i] = np.sum(x * lai_d, axis=1) / np.sqrt(np.sum(x ** 2, axis=1) * np.sum(lai_d ** 2, axis=1))
            beta_all[:, i] = np.sum(x * lai_d, axis=1) / np.sum(x ** 2, axis=1)   ## % LAI per SPEI unit
        ## per pixel: scale with the highest r (r > 0 only)
        ## pixels where r is NaN at every scale (e.g. SPEI or LAI missing in some year) -> no best scale
        print('model 1: pixels with r NaN at all scales:', np.all(np.isnan(r_all), axis=1).sum(), 'of', n)
        i_best = np.argmax(np.where(np.isnan(r_all), -np.inf, r_all), axis=1)
        r_best = r_all[np.arange(n), i_best]
        valid = r_best > 0   ## NaN > 0 is False, so all-NaN pixels are dropped
        scale_best = np.array([int(self.spei_scale_list[i]) for i in i_best], dtype=float)
        beta = beta_all[np.arange(n), i_best]
        scale_best[~valid] = np.nan
        beta[~valid] = np.nan
        w = info['area_weight'].values
        groups = self.group_dic(info)
        self.save_map(info, scale_best, 'M1_best_SPEI_scale', 'SPEI scale with max r (months)', cmap='Spectral_r',
                      discrete_levels=[int(s) for s in self.spei_scale_list])
        self.save_map(info, beta, 'M1_beta', 'LAI (% of mean) per SPEI unit')
        print('model 1: pixels with max r > 0:', valid.sum(), 'of', n)
        result = []
        for period in self.model1_period_list:
            tag = f'{period[0]}_{period[1]}'
            obs = self.trend(mat['LAI'], period)
            spei_trend = np.array([self.trend(mat[f'summer_SPEI{s}'], period) for s in self.spei_scale_list]).T
            E = beta * spei_trend[np.arange(n), i_best]
            G = obs - E
            obs[~valid] = np.nan
            vlim = np.nanpercentile(np.abs(np.concatenate([obs, E])), 90)
            self.save_map(info, obs, f'M1_LAI_trend_{tag}', 'Observed LAI trend (% per decade)', vlim=vlim)
            self.save_map(info, E, f'M1_expected_{tag}', 'SPEI-expected LAI trend (% per decade)', vlim=vlim)
            self.save_map(info, G, f'M1_gap_{tag}', 'Observed - expected (% per decade)', vlim=vlim)
            for g in groups:
                m = groups[g] & valid
                result.append([tag, 'best scale', g, self.wmean(obs[m], w[m]), self.wmean(E[m], w[m]),
                               self.wmean(G[m], w[m]), m.sum()])
            ## SI: every pixel with the same scale
            for i, s in enumerate(self.spei_scale_list):
                E_s = beta_all[:, i] * spei_trend[:, i]
                obs_s = self.trend(mat['LAI'], period)
                for g in groups:
                    m = groups[g]
                    result.append([tag, f'SPEI{s}', g, self.wmean(obs_s[m], w[m]), self.wmean(E_s[m], w[m]),
                                   self.wmean(obs_s[m] - E_s[m], w[m]), m.sum()])
        df_result = pd.DataFrame(result, columns=['period', 'scale', 'group', 'observed', 'expected', 'gap', 'n_pix'])
        df_result.to_csv(self.outdir + 'M1_summary.csv', index=False)
        print(df_result[df_result['scale'] == 'best scale'])
        self.plot_model1_bar(df_result)
        ## area-weighted mean r per scale (shows which scales LAI follows)
        df_r = pd.DataFrame({'scale': self.spei_scale_list,
                             'mean_r': [self.wmean(r_all[:, i], w) for i in range(len(self.spei_scale_list))]})
        df_r.to_csv(self.outdir + 'M1_mean_r_by_scale.csv', index=False)
        pass

    def plot_model1_bar(self, df_result):
        group_list = ['All'] + self.landcover_list
        fig, axes = plt.subplots(1, len(self.model1_period_list), sharey=True, squeeze=False,
                                 figsize=(16 * centimeter_factor, 6 * centimeter_factor))
        for j, period in enumerate(self.model1_period_list):
            ax = axes[0, j]
            tag = f'{period[0]}_{period[1]}'
            df_p = df_result[(df_result['period'] == tag) & (df_result['scale'] == 'best scale')].set_index('group')
            x = np.arange(len(group_list))
            ax.bar(x - 0.2, df_p.loc[group_list, 'observed'], 0.4, color='k', label='Observed')
            ax.bar(x + 0.2, df_p.loc[group_list, 'expected'], 0.4, color='#d6604d', label='SPEI-expected')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(group_list, fontsize=7)
            ax.set_title(f'{period[0]}-{period[1]}', fontsize=8)
        axes[0, 0].set_ylabel('Summer LAI trend (% per decade)')
        axes[0, 0].legend(frameon=False, fontsize=7)
        plt.tight_layout()
        plt.savefig(self.outdir + 'M1_observed_vs_expected.png', dpi=300)
        plt.savefig(self.outdir + 'M1_observed_vs_expected.pdf')
        plt.close()

    # ------------------------------------------------------------------ model 2
    def model2(self, info, mat):
        w = info['area_weight'].values
        groups = self.group_dic(info)
        k = len(self.clim_list)
        ## step 1: detrended LAI ~ detrended climate, pooled per vegetation type, bootstrap over years
        y = self.detrend(mat['LAI'])
        X = np.stack([self.detrend(mat[c]) for c in self.clim_list], axis=2)   ## pix x year x k
        X_flat = X.reshape(-1, k)
        vif = np.diag(np.linalg.inv(np.corrcoef(X_flat[np.all(np.isfinite(X_flat), axis=1)].T)))
        print('model 2 VIF', dict(zip(self.clim_list, np.round(vif, 2))))
        n_year = len(self.year_list)
        rng_idx = np.random.default_rng(0).integers(0, n_year, size=(self.n_boot, n_year))
        beta_dic, step1 = {}, []
        for g in groups:
            m = groups[g]
            if m.sum() < 30:
                continue
            ## per-year weighted X'X and X'y, then resample years
            xx = np.einsum('p,pyi,pyj->yij', w[m], X[m], X[m])
            xy = np.einsum('p,pyi,py->yi', w[m], X[m], y[m])
            coef = np.linalg.solve(xx.sum(axis=0), xy.sum(axis=0))
            coef_boot = np.linalg.solve(xx[rng_idx].sum(axis=1), xy[rng_idx].sum(axis=1)[..., None])[..., 0]
            beta_dic[g] = coef
            ## standardized coefficient: beta x sd(X) / sd(LAI)
            sd_x = np.sqrt(np.average(X[m].reshape(-1, k) ** 2, axis=0, weights=np.repeat(w[m], n_year)))
            sd_y = np.sqrt(np.average(y[m].reshape(-1) ** 2, weights=np.repeat(w[m], n_year)))
            for i, c in enumerate(self.clim_list):
                lo, hi = np.percentile(coef_boot[:, i], [2.5, 97.5])
                step1.append([g, c, coef[i], lo, hi, coef[i] * sd_x[i] / sd_y, m.sum()])
        df1 = pd.DataFrame(step1, columns=['group', 'term', 'beta', 'ci_low', 'ci_high', 'beta_std', 'n_pix'])
        df1.to_csv(self.outdir + 'M2_step1_beta.csv', index=False)
        print(df1)
        ## step 2: climate part, beta of the pixel's vegetation type x the pixel's own trend
        period = self.trend_period
        tag = f'{period[0]}_{period[1]}'
        obs = self.trend(mat['LAI'], period)
        A = np.full((len(info), k), np.nan)
        for lc in self.landcover_list:
            if lc not in beta_dic:
                continue
            m = groups[lc]
            for i, c in enumerate(self.clim_list):
                A[m, i] = beta_dic[lc][i] * self.trend(mat[c][m], period)
        A_sum = A.sum(axis=1)
        R = obs - A_sum
        ## step 3: R ~ trends of RAP cover, across pixels within each vegetation type
        rap_trend = np.stack([self.trend(mat[c], period) for c in self.rap_list], axis=1)
        B = np.full(len(info), np.nan)
        step3 = []
        for lc in self.landcover_list:
            m = groups[lc] & np.isfinite(R)
            if m.sum() < 30:
                continue
            Z = np.column_stack([np.ones(m.sum()), rap_trend[m]])
            sw = np.sqrt(w[m])
            gam = np.linalg.lstsq(Z * sw[:, None], R[m] * sw, rcond=None)[0]
            B[m] = rap_trend[m] @ gam[1:]
            gam_lo, gam_hi = self.block_bootstrap(info[m], Z, R[m], w[m])
            for j, name in enumerate(['intercept'] + self.rap_list):
                step3.append([lc, name, gam[j], gam_lo[j], gam_hi[j], m.sum()])
        df3 = pd.DataFrame(step3, columns=['group', 'term', 'gamma', 'ci_low', 'ci_high', 'n_pix'])
        df3.to_csv(self.outdir + 'M2_step3_gamma.csv', index=False)
        print(df3)
        C = R - B
        ## maps (same colour range for all parts)
        vlim = np.nanpercentile(np.abs(obs), 90)
        self.save_map(info, obs, f'M2_LAI_trend_{tag}', 'Observed LAI trend (% per decade)', vlim=vlim)
        for i, c in enumerate(self.clim_list):
            self.save_map(info, A[:, i], f'M2_A_{c}_{tag}', f'{c} part (% per decade)', vlim=vlim)
        self.save_map(info, A_sum, f'M2_A_climate_{tag}', 'Climate part A (% per decade)', vlim=vlim)
        self.save_map(info, B, f'M2_B_composition_{tag}', 'Composition part B (% per decade)', vlim=vlim)
        self.save_map(info, C, f'M2_C_remainder_{tag}', 'Remainder C (% per decade)', vlim=vlim)
        ## summary per group
        summary = []
        for g in groups:
            m = groups[g] & np.isfinite(B)
            row = [g, self.wmean(obs[m], w[m])] + [self.wmean(A[m, i], w[m]) for i in range(k)] + \
                  [self.wmean(B[m], w[m]), self.wmean(C[m], w[m]), m.sum()]
            summary.append(row)
        df_s = pd.DataFrame(summary, columns=['group', 'observed'] + [f'A_{c}' for c in self.clim_list] +
                                             ['B_composition', 'C_remainder', 'n_pix'])
        df_s.to_csv(self.outdir + f'M2_summary_{tag}.csv', index=False)
        print(df_s)
        self.plot_model2_bar(df_s, tag)
        pass

    def block_bootstrap(self, info, Z, R, w):
        ## 95% CI of the step-3 coefficients, resampling 1 x 1 deg blocks
        block = (np.floor(info['lon'].values) * 1000 + np.floor(info['lat'].values)).astype(int)
        block_ids = np.unique(block)
        idx_dic = {b: np.where(block == b)[0] for b in block_ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(200):
            pick = np.concatenate([idx_dic[b] for b in rng.choice(block_ids, size=len(block_ids))])
            sw = np.sqrt(w[pick])
            boot.append(np.linalg.lstsq(Z[pick] * sw[:, None], R[pick] * sw, rcond=None)[0])
        boot = np.array(boot)
        return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)

    def plot_model2_bar(self, df_s, tag):
        ## observed trend and its parts for All and each vegetation type
        part_list = ['observed'] + [f'A_{c}' for c in self.clim_list] + ['B_composition', 'C_remainder']
        color_dic = {'summer_PRCPTOT': '#4292c6', 'spring_PRCPTOT': '#74c476', 'winter_SWE_max': '#9e9ac8',
                     'summer_VPD': '#d6604d', 'summer_Tmax': '#fd8d3c', 'winter_PRCPTOT': '#6a51a3',
                     'June_PRCP': '#41b6c4'}
        color_list = ['k'] + [color_dic[c] for c in self.clim_list] + ['#d6a64f', 'grey']
        group_list = ['All'] + self.landcover_list
        df_s = df_s.set_index('group')
        width = 0.8 / len(part_list)
        plt.figure(figsize=(18 * centimeter_factor, 7 * centimeter_factor))
        for i, part in enumerate(part_list):
            x = np.arange(len(group_list)) + (i - (len(part_list) - 1) / 2) * width
            plt.bar(x, df_s.loc[group_list, part], width, color=color_list[i], label=part.replace('A_', 'A: '))
        plt.axhline(0, color='k', lw=0.5)
        plt.xticks(np.arange(len(group_list)), group_list, fontsize=8)
        plt.ylabel('Summer LAI trend (% per decade)')
        plt.title(f'Attribution of the summer LAI trend, {tag.replace("_", "-")}', fontsize=9)
        plt.legend(frameon=False, fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(self.outdir + f'M2_attribution_bar_{tag}.png', dpi=300)
        plt.savefig(self.outdir + f'M2_attribution_bar_{tag}.pdf')
        plt.close()


class Attribution_MODIS_check(Attribution_analysis):
    ## Is the SNU greening (and the large remainder C) also seen in MODIS?
    ## same pixels, same years (2003-2024), same models; LAI = SNU or MODIS
    ##   1) per pixel: correlation of detrended SNU and MODIS LAI anomalies, and the two LAI trends
    ##   2) model 1 and model 2 rerun with each product (betas and trends all from 2003-2024)
    ## output: Result2\04_attribution\MODIS_check\{SNU, MODIS}\ and comparison files in MODIS_check\
    def __init__(self):
        super().__init__()
        self.year_list = list(range(2003, 2025))
        self.trend_period = (2003, 2024)
        self.model1_period_list = [(2003, 2024)]
        ## MODIS LAI is low in sparse dry shrublands: 0.3 removed ~23,000 pixels (2026-11-14) -> 0.1 for both products
        self.LAI_threshold = 0.1
        self.check_dir = self.outdir + rf'MODIS_check_LAI{self.LAI_threshold}\\'
        T.mk_dir(self.check_dir, force=True)
        self.product_dic = {'SNU': 'summer_SNU_LAI', 'MODIS': 'summer_MODIS_LAI'}
        pass

    def run(self):
        df = T.load_df(self.dff)
        df = df[(df['year'] >= self.year_list[0]) & (df['year'] <= self.year_list[-1])]
        ## both products, then keep the pixels valid in both
        data = {}
        for name, col in self.product_dic.items():
            self.lai_col = col
            data[name] = self.prepare(df)
        pix_common = data['SNU'][0].index.intersection(data['MODIS'][0].index)
        print('pixels valid in both products:', len(pix_common))
        for name in data:
            info, mat = data[name]
            keep = info.index.isin(pix_common)
            data[name] = (info[keep], {k: v[keep] for k, v in mat.items()})
        self.compare_products(data)
        for name in data:
            info, mat = data[name]
            self.outdir = self.check_dir + rf'{name}\\'
            T.mk_dir(self.outdir, force=True)
            self.mapper.outdir = self.outdir
            self.model1(info, mat)
            self.model2(info, mat)
        self.compare_summary()
        pass

    def compare_products(self, data):
        ## per pixel: r of detrended anomalies, trend of each product; maps and a summary table
        self.outdir = self.check_dir
        self.mapper.outdir = self.check_dir
        info = data['SNU'][0]
        lai_snu, lai_modis = data['SNU'][1]['LAI'], data['MODIS'][1]['LAI']
        a, b = self.detrend(lai_snu), self.detrend(lai_modis)
        r = np.sum(a * b, axis=1) / np.sqrt(np.sum(a ** 2, axis=1) * np.sum(b ** 2, axis=1))
        tr_snu = self.trend(lai_snu, self.trend_period)
        tr_modis = self.trend(lai_modis, self.trend_period)
        p_snu = self.trend_p(lai_snu, self.trend_period)
        p_modis = self.trend_p(lai_modis, self.trend_period)
        w = info['area_weight'].values
        self.save_map(info, r, 'r_SNU_MODIS_detrended', 'r, detrended summer LAI (SNU vs MODIS)', vlim=1)
        vlim = np.nanpercentile(np.abs(np.concatenate([tr_snu, tr_modis])), 90)
        self.save_map(info, tr_snu, 'trend_SNU_2003_2024', 'SNU LAI trend (% per decade)', vlim=vlim)
        self.save_map(info, tr_modis, 'trend_MODIS_2003_2024', 'MODIS LAI trend (% per decade)', vlim=vlim)
        self.save_map(info, tr_snu - tr_modis, 'trend_SNU_minus_MODIS_2003_2024', 'SNU - MODIS trend (% per decade)',
                      vlim=vlim)
        result = []
        for g, m in self.group_dic(info).items():
            ok = m & np.isfinite(r)
            r_sp = np.corrcoef(tr_snu[ok], tr_modis[ok])[0, 1]
            ## area (%) with a significant (p < 0.05) increase / decrease
            sig = lambda tr, p, sign: np.sum(w[ok] * ((sign * tr[ok] > 0) & (p[ok] < 0.05))) / np.sum(w[ok]) * 100
            result.append([g, self.wmean(r[ok], w[ok]), self.wmean(tr_snu[ok], w[ok]), self.wmean(tr_modis[ok], w[ok]),
                           r_sp, sig(tr_snu, p_snu, 1), sig(tr_snu, p_snu, -1),
                           sig(tr_modis, p_modis, 1), sig(tr_modis, p_modis, -1), ok.sum()])
        df_c = pd.DataFrame(result, columns=['group', 'mean_r_detrended', 'trend_SNU', 'trend_MODIS',
                                             'spatial_r_of_trends', 'SNU_sig_increase_%', 'SNU_sig_decrease_%',
                                             'MODIS_sig_increase_%', 'MODIS_sig_decrease_%', 'n_pix'])
        df_c.to_csv(self.check_dir + 'SNU_vs_MODIS.csv', index=False)
        print(df_c)
        ## scatter of per-pixel trends
        plt.figure(figsize=(8 * centimeter_factor, 8 * centimeter_factor))
        plt.hexbin(tr_snu, tr_modis, gridsize=60, extent=(-vlim, vlim, -vlim, vlim), mincnt=1, bins='log', cmap='Greys')
        plt.plot([-vlim, vlim], [-vlim, vlim], 'r-', lw=0.8)
        plt.axhline(0, color='k', lw=0.3)
        plt.axvline(0, color='k', lw=0.3)
        plt.xlabel('SNU LAI trend (% per decade)')
        plt.ylabel('MODIS LAI trend (% per decade)')
        plt.title('2003-2024', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.check_dir + 'trend_SNU_vs_MODIS_scatter.png', dpi=300)
        plt.savefig(self.check_dir + 'trend_SNU_vs_MODIS_scatter.pdf')
        plt.close()

    def trend_p(self, arr, period):
        ## two-sided p of the OLS trend for each row
        years = np.array(self.year_list)
        m = (years >= period[0]) & (years <= period[1])
        x = years[m].astype(float)
        y = arr[:, m]
        n = len(x)
        xm = x - x.mean()
        ym = y - y.mean(axis=1, keepdims=True)
        slope = ym @ xm / np.sum(xm ** 2)
        resid = ym - slope[:, None] * xm[None, :]
        se = np.sqrt(np.sum(resid ** 2, axis=1) / (n - 2) / np.sum(xm ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            t = slope / se
        return 2 * stats.t.sf(np.abs(t), n - 2)

    def compare_summary(self):
        ## model 2 summary of both products side by side
        tag = f'{self.trend_period[0]}_{self.trend_period[1]}'
        df_list = []
        for name in self.product_dic:
            df_s = pd.read_csv(self.check_dir + rf'{name}\M2_summary_{tag}.csv')
            df_s.insert(0, 'product', name)
            df_list.append(df_s)
        df_all = pd.concat(df_list, ignore_index=True)
        df_all.to_csv(self.check_dir + f'M2_summary_SNU_vs_MODIS_{tag}.csv', index=False)
        print(df_all)


class Attribution_MODIS(Attribution_MODIS_check):
    ## main analysis: MODIS summer LAI only, 2003-2024 (the dry "new normal"), LAI threshold 0.1
    ## same models as Attribution_analysis (model 1: SPEI gap; model 2: VPD / precipitation / composition / remainder)
    ## plus the observed LAI trend with significance
    ## output: Result2\04_attribution\MODIS_main\
    def __init__(self):
        super().__init__()   ## years 2003-2024, trend period 2003-2024, LAI threshold 0.1, trend_p
        self.lai_col = 'summer_MODIS_LAI'
        self.outdir = this_root + rf'Result2\04_attribution\MODIS_main\\'
        T.mk_dir(self.outdir, force=True)
        self.mapper.outdir = self.outdir
        self.main_dir = self.outdir
        ## model 2 temperature versions: VPD only / summer Tmax only / both
        precip = ['summer_PRCPTOT', 'spring_PRCPTOT', 'winter_SWE_max']
        self.clim_version_dic = {'VPD': precip + ['summer_VPD'],
                                 'Tmax': precip + ['summer_Tmax'],
                                 'VPD_Tmax': precip + ['summer_VPD', 'summer_Tmax']}
        self.clim_list = precip + ['summer_VPD', 'summer_Tmax']   ## prepare() requires complete data for all of them
        pass

    def run(self):
        df = T.load_df(self.dff)
        df = df[(df['year'] >= self.year_list[0]) & (df['year'] <= self.year_list[-1])]
        info, mat = self.prepare(df)   ## same pixels for every version
        self.lai_trend(info, mat)
        self.model1(info, mat)
        tag = f'{self.trend_period[0]}_{self.trend_period[1]}'
        df_list = []
        for version, clim_list in self.clim_version_dic.items():
            self.clim_list = clim_list
            self.outdir = self.main_dir + rf'model2_{version}\\'
            T.mk_dir(self.outdir, force=True)
            self.mapper.outdir = self.outdir
            print('model 2 version:', version)
            self.model2(info, mat)
            df_s = pd.read_csv(self.outdir + f'M2_summary_{tag}.csv')
            df_s.insert(0, 'version', version)
            df_list.append(df_s)
        self.outdir = self.main_dir
        self.mapper.outdir = self.main_dir
        df_all = pd.concat(df_list, ignore_index=True)
        df_all.to_csv(self.main_dir + f'M2_summary_versions_{tag}.csv', index=False)
        print(df_all)
        pass

    def lai_trend(self, info, mat):
        ## observed MODIS LAI trend, significance, area-weighted summary per vegetation type
        tr = self.trend(mat['LAI'], self.trend_period)
        p = self.trend_p(mat['LAI'], self.trend_period)
        w = info['area_weight'].values
        tag = f'{self.trend_period[0]}_{self.trend_period[1]}'
        self.save_map(info, p, f'LAI_trend_p_{tag}', 'p of LAI trend', cmap='Greys_r', vlim=1)
        result = []
        for g, m in self.group_dic(info).items():
            ww = w[m]
            result.append([g, self.wmean(tr[m], ww),
                           np.sum(ww * ((tr[m] > 0) & (p[m] < 0.05))) / np.sum(ww) * 100,
                           np.sum(ww * ((tr[m] < 0) & (p[m] < 0.05))) / np.sum(ww) * 100, m.sum()])
        df_t = pd.DataFrame(result, columns=['group', 'mean_trend_%_per_decade', 'sig_increase_area_%',
                                             'sig_decrease_area_%', 'n_pix'])
        df_t.to_csv(self.outdir + f'LAI_trend_summary_{tag}.csv', index=False)
        print(df_t)


class Remainder_C_drivers(Attribution_MODIS):
    ## What is the remainder C (model 2, VPD version, MODIS 2003-2024)?  Two hypotheses tested across pixels:
    ##   H1 CO2 / water-use efficiency: C larger where it is drier  -> aridity index AI = mean annual P / mean annual PET
    ##   H2 monsoon rain:               C larger where more rain falls in Jul-Sep -> monsoon share = Jul-Sep P / annual P
    ##   both are climatologies over 2003-2024, TerraClimate ppt and pet monthly dicts
    ## (H3, annual grass cover, is not tested here: C is already the part left after the RAP AFG trend)
    ## outputs in MODIS_main\C_drivers\:
    ##   maps of AI and monsoon share; C_drivers.df (one row per pixel)
    ##   C by AI class (UNEP classes) and by monsoon-share bin (0.1 steps), per vegetation type,
    ##     area-weighted mean with 95% CI from a 1 x 1 deg block bootstrap
    ##   C ~ z(log10 AI) + z(monsoon share) per vegetation type, coefficients with 1 deg block bootstrap CI
    def __init__(self):
        super().__init__()
        self.version = 'VPD'   ## main model 2 version
        self.model2_dir = self.main_dir + rf'model2_{self.version}\\'
        self.outdir = self.main_dir + rf'C_drivers\\'
        T.mk_dir(self.outdir, force=True)
        self.mapper.outdir = self.outdir
        self.clim_period = (2003, 2024)
        ## UNEP aridity classes (P / PET)
        self.AI_bin_list = [0, 0.05, 0.2, 0.5, 0.65, 10]
        self.AI_label_list = ['hyper-arid\n<0.05', 'arid\n0.05-0.2', 'semi-arid\n0.2-0.5', 'dry sub-humid\n0.5-0.65',
                              'humid\n>0.65']
        self.monsoon_bin_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        self.monsoon_label_list = ['<0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '>0.5']
        pass

    def run(self):
        df = T.load_df(self.dff)
        df = df[(df['year'] >= self.year_list[0]) & (df['year'] <= self.year_list[-1])]
        info, mat = self.prepare(df)   ## same pixels as Attribution_MODIS
        tag = f'{self.trend_period[0]}_{self.trend_period[1]}'
        info = info.copy()
        info['C'] = self.read_tif_values(info, self.model2_dir + f'M2_C_remainder_{tag}.tif')
        info['observed'] = self.read_tif_values(info, self.model2_dir + f'M2_LAI_trend_{tag}.tif')
        info['AI'], info['monsoon_share'] = self.climatology(info)
        T.save_df(info, self.outdir + 'C_drivers.df')
        self.driver_maps(info)
        self.C_by_bin(info, 'AI', self.AI_bin_list, self.AI_label_list, 'Aridity index (P / PET)')
        self.C_by_bin(info, 'monsoon_share', self.monsoon_bin_list, self.monsoon_label_list,
                      'Monsoon share (Jul-Sep P / annual P)')
        self.C_regression(info)
        pass

    def read_tif_values(self, info, fpath):
        ## value of a template-grid tif at each pixel (pix = (row, col))
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        arr = np.array(arr, dtype=float)
        arr[arr < -9999] = np.nan
        return np.array([arr[pix] for pix in info.index])

    def climatology(self, info):
        ## mean annual P, PET and Jul-Sep P over the period, from monthly dicts that end in Dec 2024
        dic = {}
        for var in ['ppt', 'pet']:
            spatial_dic = T.load_npy_dir(data_root + rf'TerraClimate\{var}\dic\\')
            n = len(spatial_dic[info.index[0]])
            start_year = 2025 - n // 12
            print(var, 'monthly values per pixel:', n, '-> first year', start_year, '(assumes the last month is Dec 2024)')
            years = np.arange(start_year, 2025)
            m = (years >= self.clim_period[0]) & (years <= self.clim_period[1])
            annual, jas = [], []
            for pix in info.index:
                vals = np.array(spatial_dic.get(pix, np.full(n, np.nan)), dtype=float).reshape(-1, 12)[m]
                annual.append(np.nanmean(vals.sum(axis=1)))
                jas.append(np.nanmean(vals[:, 6:9].sum(axis=1)))   ## Jul, Aug, Sep
            dic[var] = np.array(annual)
            dic[var + '_JAS'] = np.array(jas)
            del spatial_dic
        with np.errstate(divide='ignore', invalid='ignore'):
            AI = dic['ppt'] / dic['pet']
            monsoon_share = dic['ppt_JAS'] / dic['ppt']
        print('AI percentiles 5/50/95:', np.nanpercentile(AI, [5, 50, 95]).round(3))
        print('monsoon share percentiles 5/50/95:', np.nanpercentile(monsoon_share, [5, 50, 95]).round(3))
        return AI, monsoon_share

    def driver_maps(self, info):
        w = info['area_weight'].values
        for col, cmap, vmin, vmax, label in [('AI', 'YlGnBu', 0, 1, 'Aridity index (P / PET), 2003-2024'),
                                             ('monsoon_share', 'YlGnBu', 0, 0.6, 'Jul-Sep share of annual P, 2003-2024')]:
            vals = info[col].values
            arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, vals)))
            arr[arr <= -9999] = np.nan
            ok = np.isfinite(vals)
            self.mapper.plot_map(arr, f'{col}_map', cmap, vmin, vmax, label, vals[ok], w[ok])

    def C_by_bin(self, info, col, bin_list, label_list, xlabel):
        ## area-weighted mean C per bin and vegetation type, 95% CI from a 1 deg block bootstrap
        w = info['area_weight'].values
        idx = np.digitize(info[col].values, bin_list) - 1   ## bin number, -1 / len-1 = outside
        group_list = ['All'] + self.landcover_list
        groups = self.group_dic(info)
        result = []
        for g in group_list:
            for b, label in enumerate(label_list):
                m = groups[g] & (idx == b) & np.isfinite(info['C'].values)
                if m.sum() < 30:
                    continue
                lo, hi = self.block_mean_ci(info[m], info['C'].values[m], w[m])
                result.append([g, label.replace('\n', ' '), b, self.wmean(info['C'].values[m], w[m]), lo, hi,
                               self.wmean(info['observed'].values[m], w[m]), m.sum()])
        df_r = pd.DataFrame(result, columns=['group', 'bin', 'bin_id', 'C_mean', 'C_ci_low', 'C_ci_high',
                                             'observed_mean', 'n_pix'])
        df_r.to_csv(self.outdir + f'C_by_{col}.csv', index=False)
        print(df_r)
        ## figure: one line per vegetation type
        color_dic = {'All': 'k', 'Forest': '#1b7837', 'Shrub': '#bf812d', 'Grass': '#dfc27d', 'Savanna': '#80cdc1'}
        plt.figure(figsize=(12 * centimeter_factor, 8 * centimeter_factor))
        n_g = len(group_list)
        for i, g in enumerate(group_list):
            d = df_r[df_r['group'] == g]
            x = d['bin_id'].values + (i - (n_g - 1) / 2) * 0.08
            plt.errorbar(x, d['C_mean'], yerr=[d['C_mean'] - d['C_ci_low'], d['C_ci_high'] - d['C_mean']],
                         color=color_dic[g], marker='o', ms=3, lw=1.2 if g == 'All' else 0.8, capsize=2, label=g)
        plt.axhline(0, color='k', lw=0.5)
        plt.xticks(range(len(label_list)), label_list, fontsize=6)
        plt.xlabel(xlabel, fontsize=8)
        plt.ylabel('Remainder C (% per decade)', fontsize=8)
        plt.legend(frameon=False, fontsize=6)
        plt.tight_layout()
        plt.savefig(self.outdir + f'C_by_{col}.png', dpi=300)
        plt.savefig(self.outdir + f'C_by_{col}.pdf')
        plt.close()

    def block_mean_ci(self, info, v, w):
        ## 95% CI of an area-weighted mean, resampling 1 x 1 deg blocks
        block = (np.floor(info['lon'].values) * 1000 + np.floor(info['lat'].values)).astype(int)
        block_ids = np.unique(block)
        idx_dic = {b: np.where(block == b)[0] for b in block_ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(200):
            pick = np.concatenate([idx_dic[b] for b in rng.choice(block_ids, size=len(block_ids))])
            boot.append(np.sum(v[pick] * w[pick]) / np.sum(w[pick]))
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    def C_regression(self, info):
        ## per vegetation type: C = g0 + g1 z(log10 AI) + g2 z(monsoon share); g = change in C per 1 SD
        w_all = info['area_weight'].values
        x_dic = {'log10_AI': np.log10(info['AI'].values), 'monsoon_share': info['monsoon_share'].values}
        result = []
        for g, m in self.group_dic(info).items():
            m = m & np.isfinite(info['C'].values)
            for x in x_dic.values():
                m &= np.isfinite(x)
            Z = [np.ones(m.sum())]
            for x in x_dic.values():
                xm = x[m]
                Z.append((xm - xm.mean()) / xm.std())
            Z = np.column_stack(Z)
            C = info['C'].values[m]
            w = w_all[m]
            sw = np.sqrt(w)
            gam = np.linalg.lstsq(Z * sw[:, None], C * sw, rcond=None)[0]
            lo, hi = self.block_bootstrap(info[m], Z, C, w)
            r_pair = np.corrcoef(Z[:, 1], Z[:, 2])[0, 1]
            for j, name in enumerate(['intercept'] + list(x_dic)):
                result.append([g, name, gam[j], lo[j], hi[j], r_pair, m.sum()])
        df_g = pd.DataFrame(result, columns=['group', 'term', 'gamma_per_SD', 'ci_low', 'ci_high',
                                             'r_logAI_monsoon', 'n_pix'])
        df_g.to_csv(self.outdir + 'C_regression.csv', index=False)
        print(df_g)


class Monsoon_precip_check(Remainder_C_drivers):
    ## C is largest where the Jul-Sep share of annual P is high. Did summer (Jul-Oct) rain change there, and does LAI care?
    ## region: monsoon = Jul-Sep P / annual P >= 0.3 (no seasonality would be 3/12 = 0.25), from C_drivers.df
    ## summer Daymet indices, 2003-2024:
    ##   amount:    PRCPTOT, P_LIGHT (1-5 mm days), P_MOD (5 mm - p95 days), R95p (> p95 days)          [mm]
    ##   frequency: WETDAYS (>= 1 mm), LIGHT (1-5 mm), R5mm (>= 5 mm, includes heavy days)               [days]
    ##   intensity / dry spells: SDII (mm per wet day), Rx1day (mm), CDD (longest dry run, days)
    ##   (no count of > p95 days was exported; R95p and Rx1day stand for heavy rain)
    ## 1) trend per pixel (per decade, absolute and % of the pixel mean) and p; summary per region x vegetation type
    ## 2) screening: one index at a time, detrended LAI ~ detrended index, pooled per region x vegetation type
    ##    (area-weighted, 95% CI by bootstrapping years); implied LAI trend = beta x index trend, compared with C
    ## output: MODIS_main\C_drivers\monsoon_check\
    def __init__(self):
        super().__init__()
        self.c_dir = self.outdir   ## C_drivers\
        self.outdir = self.c_dir + rf'monsoon_check\\'
        T.mk_dir(self.outdir, force=True)
        self.mapper.outdir = self.outdir
        self.monsoon_threshold = 0.3
        self.index_list = ['PRCPTOT', 'P_LIGHT', 'P_MOD', 'R95p', 'WETDAYS', 'LIGHT', 'R5mm', 'SDII', 'Rx1day', 'CDD']
        self.unit_dic = {'PRCPTOT': 'mm', 'P_LIGHT': 'mm', 'P_MOD': 'mm', 'R95p': 'mm', 'WETDAYS': 'days',
                         'LIGHT': 'days', 'R5mm': 'days', 'SDII': 'mm/day', 'Rx1day': 'mm', 'CDD': 'days'}
        pass

    def run(self):
        df = T.load_df(self.dff)
        df = df[(df['year'] >= self.year_list[0]) & (df['year'] <= self.year_list[-1])]
        info, mat = self.prepare(df)   ## LAI (% of mean) for the same pixels as model 2
        df_c = T.load_df(self.c_dir + 'C_drivers.df')
        info = info.copy()
        for col in ['C', 'monsoon_share']:
            info[col] = df_c[col].reindex(info.index).values
        info['region'] = np.where(info['monsoon_share'] >= self.monsoon_threshold, 'monsoon', 'non-monsoon')
        info.loc[info['monsoon_share'].isna(), 'region'] = np.nan
        print(info['region'].value_counts())
        pmat = self.daymet_matrices(info)
        df_t = self.trend_table(info, pmat)
        df_b = self.beta_table(info, mat, pmat)
        self.plot_summary(df_t, df_b)
        pass

    def daymet_matrices(self, info):
        ## pixel x year matrices of the summer Daymet indices, same pixel order as info
        df_d = T.load_df(result_root + rf'\Dataframe\Daymet\Daymet_seasonal.df')
        df_d = df_d[(df_d['year'] >= self.year_list[0]) & (df_d['year'] <= self.year_list[-1])]
        pmat = {}
        for index in self.index_list:
            col = f'summer_{index}'
            pmat[index] = df_d.pivot(index='pix', columns='year', values=col).reindex(
                index=info.index, columns=self.year_list).values.astype(float)
            print(col, 'pixels with complete data:', np.all(np.isfinite(pmat[index]), axis=1).sum(), 'of', len(info))
        return pmat

    def groups_by_region(self, info):
        ## (region, vegetation type) -> mask
        dic = {}
        for region in ['monsoon', 'non-monsoon']:
            for g, m in self.group_dic(info).items():
                dic[(region, g)] = m & (info['region'] == region).values
        return dic

    def trend_table(self, info, pmat):
        w = info['area_weight'].values
        result = []
        for index in self.index_list:
            x = pmat[index]
            ok_pix = np.all(np.isfinite(x), axis=1)
            tr = self.trend(x, self.trend_period)
            p = self.trend_p(x, self.trend_period)
            mean = np.nanmean(x, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                tr_rel = np.where(mean > 0, tr / mean * 100, np.nan)
            for (region, g), m in self.groups_by_region(info).items():
                m = m & ok_pix
                if m.sum() < 30:
                    continue
                ww = w[m]
                lo, hi = self.block_mean_ci(info[m], tr[m], ww)
                ## trend of the area-weighted regional mean series
                series = np.sum(x[m] * ww[:, None], axis=0) / np.sum(ww)
                p_reg = self.trend_p(series[None, :], self.trend_period)[0]
                result.append([index, self.unit_dic[index], region, g, self.wmean(mean[m], ww),
                               self.wmean(tr[m], ww), lo, hi, self.wmean(tr_rel[m], ww), p_reg,
                               np.sum(ww * ((tr[m] > 0) & (p[m] < 0.05))) / np.sum(ww) * 100,
                               np.sum(ww * ((tr[m] < 0) & (p[m] < 0.05))) / np.sum(ww) * 100, m.sum()])
        df_t = pd.DataFrame(result, columns=['index', 'unit', 'region', 'group', 'mean', 'trend_per_decade',
                                             'trend_ci_low', 'trend_ci_high', 'trend_%_of_mean_per_decade',
                                             'p_regional_mean_series', 'sig_increase_area_%', 'sig_decrease_area_%',
                                             'n_pix'])
        df_t.to_csv(self.outdir + 'precip_trend_by_region.csv', index=False)
        print(df_t[df_t['group'] == 'All'])
        return df_t

    def beta_table(self, info, mat, pmat):
        ## one index at a time: detrended LAI = beta x detrended index, pooled per region x vegetation type
        w = info['area_weight'].values
        y = self.detrend(mat['LAI'])
        n_year = len(self.year_list)
        rng_idx = np.random.default_rng(0).integers(0, n_year, size=(self.n_boot, n_year))
        result = []
        for index in self.index_list:
            ok_pix = np.all(np.isfinite(pmat[index]), axis=1)
            x = np.full_like(pmat[index], np.nan)
            x[ok_pix] = self.detrend(pmat[index][ok_pix])
            tr = self.trend(pmat[index], self.trend_period)
            for (region, g), m in self.groups_by_region(info).items():
                m = m & ok_pix
                if m.sum() < 30:
                    continue
                xy = np.einsum('p,py,py->y', w[m], x[m], y[m])   ## per year
                xx = np.einsum('p,py,py->y', w[m], x[m], x[m])
                beta = xy.sum() / xx.sum()
                beta_boot = xy[rng_idx].sum(axis=1) / xx[rng_idx].sum(axis=1)
                lo, hi = np.percentile(beta_boot, [2.5, 97.5])
                r = np.sum(w[m][:, None] * x[m] * y[m]) / np.sqrt(np.sum(w[m][:, None] * x[m] ** 2) *
                                                                   np.sum(w[m][:, None] * y[m] ** 2))
                implied = self.wmean(beta * tr[m], w[m])   ## % LAI per decade
                C_mean = self.wmean(info['C'].values[m], w[m])
                result.append([index, region, g, beta, lo, hi, r, implied, C_mean, m.sum()])
        df_b = pd.DataFrame(result, columns=['index', 'region', 'group', 'beta_%LAI_per_unit', 'ci_low', 'ci_high',
                                             'r_pooled', 'implied_LAI_trend_%_per_decade', 'C_mean', 'n_pix'])
        df_b.to_csv(self.outdir + 'precip_beta_by_region.csv', index=False)
        print(df_b[df_b['group'] == 'All'])
        return df_b

    def plot_summary(self, df_t, df_b):
        ## rows: All, Grass, Shrub, Savanna; left: index trend (% of mean per decade); right: implied LAI trend vs C
        group_list = ['All', 'Grass', 'Shrub', 'Savanna']
        color_dic = {'monsoon': '#2166ac', 'non-monsoon': '#d6a64f'}
        x = np.arange(len(self.index_list))
        fig, axes = plt.subplots(len(group_list), 2, figsize=(18 * centimeter_factor, 20 * centimeter_factor),
                                 sharex=True)
        for i, g in enumerate(group_list):
            for j, region in enumerate(['monsoon', 'non-monsoon']):
                dx = (j - 0.5) * 0.38
                d = df_t[(df_t['group'] == g) & (df_t['region'] == region)].set_index('index').reindex(self.index_list)
                axes[i, 0].bar(x + dx, d['trend_%_of_mean_per_decade'], 0.38, color=color_dic[region], label=region)
                d = df_b[(df_b['group'] == g) & (df_b['region'] == region)].set_index('index').reindex(self.index_list)
                axes[i, 1].bar(x + dx, d['implied_LAI_trend_%_per_decade'], 0.38, color=color_dic[region],
                               label=region)
                C_mean = d['C_mean'].dropna()
                if len(C_mean) > 0:
                    axes[i, 1].axhline(C_mean.iloc[0], color=color_dic[region], ls='--', lw=0.8,
                                       label=f'C, {region}')
            for k in range(2):
                axes[i, k].axhline(0, color='k', lw=0.5)
                axes[i, k].tick_params(labelsize=6)
            axes[i, 0].set_ylabel(f'{g}\nindex trend\n(% of mean / decade)', fontsize=7)
            axes[i, 1].set_ylabel('implied LAI trend\n(% / decade)', fontsize=7)
        axes[0, 0].legend(frameon=False, fontsize=6)
        axes[0, 1].legend(frameon=False, fontsize=6)
        for k in range(2):
            axes[-1, k].set_xticks(x)
            axes[-1, k].set_xticklabels(self.index_list, rotation=60, fontsize=6)
        axes[0, 0].set_title('Summer (Jul-Oct) rain index trend, 2003-2024', fontsize=8)
        axes[0, 1].set_title('beta x index trend (one index at a time)', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.outdir + 'monsoon_precip_check.png', dpi=300)
        plt.savefig(self.outdir + 'monsoon_precip_check.pdf')
        plt.close()


class Attribution_MODIS_water(Remainder_C_drivers):
    ## model 2 (MODIS, 2003-2024, VPD version) with the missing water inputs; does C get smaller, especially in the
    ## monsoon region (Jul-Sep share of annual P >= 0.3)?
    ##   base:          summer P (Jul-Oct), spring P (Mar-May), winter SWE_max, summer VPD   (= the current main model)
    ##   winterP:       winter SWE_max -> winter P (Nov-Feb, rain + snow, Daymet)
    ##   winterP_June:  winterP + June P (TerraClimate)
    ## output: Result2\04_attribution\MODIS_water\{version}\ and the comparison tables in MODIS_water\
    def __init__(self):
        super().__init__()
        self.c_dir = self.outdir   ## MODIS_main\C_drivers\ (C_drivers.df with monsoon_share)
        self.main_dir = this_root + rf'Result2\04_attribution\MODIS_water\\'
        T.mk_dir(self.main_dir, force=True)
        self.outdir = self.main_dir
        self.mapper.outdir = self.main_dir
        self.clim_version_dic = {
            'base': ['summer_PRCPTOT', 'spring_PRCPTOT', 'winter_SWE_max', 'summer_VPD'],
            'winterP': ['summer_PRCPTOT', 'spring_PRCPTOT', 'winter_PRCPTOT', 'summer_VPD'],
            'winterP_June': ['summer_PRCPTOT', 'spring_PRCPTOT', 'winter_PRCPTOT', 'June_PRCP', 'summer_VPD'],
        }
        ## prepare() keeps pixels with complete data for all of them -> same pixels in every version
        self.clim_list = ['summer_PRCPTOT', 'spring_PRCPTOT', 'winter_SWE_max', 'winter_PRCPTOT', 'June_PRCP',
                          'summer_VPD']
        pass

    def run(self):
        df = T.load_df(self.dff)
        df = df[(df['year'] >= self.year_list[0]) & (df['year'] <= self.year_list[-1])]
        info, mat = self.prepare(df)
        info = info.copy()
        df_c = T.load_df(self.c_dir + 'C_drivers.df')
        info['monsoon_share'] = df_c['monsoon_share'].reindex(info.index).values
        info['region'] = np.where(info['monsoon_share'] >= 0.3, 'monsoon', 'non-monsoon')
        tag = f'{self.trend_period[0]}_{self.trend_period[1]}'
        df_list, region_list = [], []
        for version, clim_list in self.clim_version_dic.items():
            self.clim_list = clim_list
            self.outdir = self.main_dir + rf'{version}\\'
            T.mk_dir(self.outdir, force=True)
            self.mapper.outdir = self.outdir
            print('model 2 version:', version)
            self.model2(info, mat)
            df_s = pd.read_csv(self.outdir + f'M2_summary_{tag}.csv')
            df_s.insert(0, 'version', version)
            df_list.append(df_s)
            region_list.append(self.parts_by_region(info, version, tag))
        self.outdir = self.main_dir
        self.mapper.outdir = self.main_dir
        df_all = pd.concat(df_list, ignore_index=True)
        df_all.to_csv(self.main_dir + f'M2_summary_versions_{tag}.csv', index=False)
        print(df_all)
        df_r = pd.concat(region_list, ignore_index=True)
        df_r.to_csv(self.main_dir + f'M2_parts_by_region_{tag}.csv', index=False)
        print(df_r)
        self.plot_C_by_region(df_r)
        pass

    def parts_by_region(self, info, version, tag):
        ## area-weighted observed, A (each term), B and C per region x vegetation type, read back from the model 2 maps
        w = info['area_weight'].values
        part_dic = {'observed': f'M2_LAI_trend_{tag}', 'B_composition': f'M2_B_composition_{tag}',
                    'C_remainder': f'M2_C_remainder_{tag}'}
        for c in self.clim_list:
            part_dic[f'A_{c}'] = f'M2_A_{c}_{tag}'
        vals = {k: self.read_tif_values(info, self.outdir + f'{f}.tif') for k, f in part_dic.items()}
        result = []
        for region in ['monsoon', 'non-monsoon']:
            for g, m in self.group_dic(info).items():
                m = m & (info['region'] == region).values & np.isfinite(vals['C_remainder'])
                if m.sum() < 30:
                    continue
                lo, hi = self.block_mean_ci(info[m], vals['C_remainder'][m], w[m])
                row = {'version': version, 'region': region, 'group': g, 'n_pix': m.sum()}
                for k in part_dic:
                    row[k] = self.wmean(vals[k][m], w[m])
                row['C_ci_low'], row['C_ci_high'] = lo, hi
                result.append(row)
        return pd.DataFrame(result)

    def plot_C_by_region(self, df_r):
        ## C per version, region and vegetation type
        group_list = ['All', 'Grass', 'Shrub', 'Savanna', 'Forest']
        version_list = list(self.clim_version_dic)
        color_dic = {'base': '#bdbdbd', 'winterP': '#6baed6', 'winterP_June': '#08519c'}
        fig, axes = plt.subplots(1, 2, figsize=(18 * centimeter_factor, 7 * centimeter_factor), sharey=True)
        width = 0.8 / len(version_list)
        for ax, region in zip(axes, ['monsoon', 'non-monsoon']):
            for i, version in enumerate(version_list):
                d = df_r[(df_r['version'] == version) & (df_r['region'] == region)].set_index('group').reindex(
                    group_list)
                x = np.arange(len(group_list)) + (i - (len(version_list) - 1) / 2) * width
                ax.bar(x, d['C_remainder'], width, color=color_dic[version], label=version,
                       yerr=[d['C_remainder'] - d['C_ci_low'], d['C_ci_high'] - d['C_remainder']],
                       error_kw={'lw': 0.6, 'capsize': 1.5})
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(np.arange(len(group_list)))
            ax.set_xticklabels(group_list, fontsize=7)
            ax.set_title(region, fontsize=8)
        axes[0].set_ylabel('Remainder C (% per decade)', fontsize=8)
        axes[0].legend(frameon=False, fontsize=6)
        plt.tight_layout()
        plt.savefig(self.main_dir + 'C_by_region_versions.png', dpi=300)
        plt.savefig(self.main_dir + 'C_by_region_versions.pdf')
        plt.close()


class C_check(Remainder_C_drivers):
    ## two checks of the remainder C of the main model (MODIS_water\winterP_June: winter P, spring P, June P,
    ## summer P, summer VPD; MODIS 2003-2024)
    ## 1) denominator: LAI is in % of the pixel mean, so a small absolute change looks large where LAI is low.
    ##    C_abs = C (%) x mean LAI / 100 (m2/m2 per decade; exact, because every part of model 2 is linear in LAI %)
    ##    C (%) and C_abs by mean-LAI class and by monsoon-share bin
    ## 2) CO2 / water limitation: CO2 should help most where growth is most limited by water.
    ##    water limitation per pixel = r (and beta) of detrended LAI with detrended water-year P
    ##    (Nov-Feb + Mar-May + Jun + Jul-Oct = Nov-Oct); C by r class; C ~ z(r) + z(monsoon share) per vegetation type
    ##    (r uses year-to-year variation, C uses the trend, so the two are not the same signal)
    ## output: Result2\04_attribution\MODIS_water\C_check\
    def __init__(self):
        super().__init__()
        self.c_dir = self.outdir   ## MODIS_main\C_drivers\ (C_drivers.df with monsoon_share)
        self.model2_dir = this_root + rf'Result2\04_attribution\MODIS_water\winterP_June\\'
        self.outdir = this_root + rf'Result2\04_attribution\MODIS_water\C_check\\'
        T.mk_dir(self.outdir, force=True)
        self.mapper.outdir = self.outdir
        ## same variables as Attribution_MODIS_water, so prepare() keeps the same pixels
        self.clim_list = ['summer_PRCPTOT', 'spring_PRCPTOT', 'winter_SWE_max', 'winter_PRCPTOT', 'June_PRCP',
                          'summer_VPD']
        self.water_year_list = ['winter_PRCPTOT', 'spring_PRCPTOT', 'June_PRCP', 'summer_PRCPTOT']
        self.lai_bin_list = [0, 0.25, 0.5, 1, 2, 100]
        self.lai_label_list = ['<0.25', '0.25-0.5', '0.5-1', '1-2', '>2']
        self.r_bin_list = [-1, 0, 0.2, 0.4, 0.6, 1.01]
        self.r_label_list = ['<0', '0-0.2', '0.2-0.4', '0.4-0.6', '>0.6']
        pass

    def run(self):
        df = T.load_df(self.dff)
        df = df[(df['year'] >= self.year_list[0]) & (df['year'] <= self.year_list[-1])]
        info, mat = self.prepare(df)
        info = info.copy()
        tag = f'{self.trend_period[0]}_{self.trend_period[1]}'
        info['C'] = self.read_tif_values(info, self.model2_dir + f'M2_C_remainder_{tag}.tif')
        info['observed'] = self.read_tif_values(info, self.model2_dir + f'M2_LAI_trend_{tag}.tif')
        ## mean MODIS summer LAI (m2/m2), 2003-2024
        lai = df.pivot(index='pix', columns='year', values=self.lai_col).reindex(index=info.index)
        info['mean_LAI'] = lai.mean(axis=1).values
        info['C_abs'] = info['C'] * info['mean_LAI'] / 100
        info['observed_abs'] = info['observed'] * info['mean_LAI'] / 100
        df_c = T.load_df(self.c_dir + 'C_drivers.df')
        info['monsoon_share'] = df_c['monsoon_share'].reindex(info.index).values
        ## water limitation: r and beta of detrended LAI (%) with detrended water-year P
        p_wy = sum(mat[c] for c in self.water_year_list)
        x, y = self.detrend(p_wy), self.detrend(mat['LAI'])
        info['r_water'] = np.sum(x * y, axis=1) / np.sqrt(np.sum(x ** 2, axis=1) * np.sum(y ** 2, axis=1))
        info['beta_water'] = np.sum(x * y, axis=1) / np.sum(x ** 2, axis=1) * 100   ## % LAI per 100 mm
        info['P_water_year'] = np.mean(p_wy, axis=1)
        T.save_df(info, self.outdir + 'C_check.df')
        self.maps(info)
        ## 1) denominator
        self.bin_plot(info, 'C', 'mean_LAI', self.lai_bin_list, self.lai_label_list, 'Mean summer LAI (m2/m2)',
                      'C (% per decade)', 'C_pct_by_meanLAI')
        self.bin_plot(info, 'C_abs', 'mean_LAI', self.lai_bin_list, self.lai_label_list, 'Mean summer LAI (m2/m2)',
                      'C (m2/m2 per decade)', 'C_abs_by_meanLAI')
        self.bin_plot(info, 'C', 'monsoon_share', self.monsoon_bin_list, self.monsoon_label_list,
                      'Monsoon share (Jul-Sep P / annual P)', 'C (% per decade)', 'C_pct_by_monsoon_share')
        self.bin_plot(info, 'C_abs', 'monsoon_share', self.monsoon_bin_list, self.monsoon_label_list,
                      'Monsoon share (Jul-Sep P / annual P)', 'C (m2/m2 per decade)', 'C_abs_by_monsoon_share')
        ## 2) water limitation
        self.bin_plot(info, 'C', 'r_water', self.r_bin_list, self.r_label_list,
                      'r (detrended LAI vs water-year P)', 'C (% per decade)', 'C_pct_by_r_water')
        self.bin_plot(info, 'C_abs', 'r_water', self.r_bin_list, self.r_label_list,
                      'r (detrended LAI vs water-year P)', 'C (m2/m2 per decade)', 'C_abs_by_r_water')
        self.regression(info)
        pass

    def maps(self, info):
        w = info['area_weight'].values
        for col, cmap, vlim, label in [('C_abs', 'RdBu', None, 'C (m2/m2 per decade)'),
                                       ('r_water', 'RdBu', 1, 'r, detrended LAI vs water-year P'),
                                       ('mean_LAI', 'YlGn', None, 'Mean summer LAI (m2/m2)')]:
            vals = info[col].values
            arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, vals)))
            arr[arr <= -9999] = np.nan
            ok = np.isfinite(vals)
            if col == 'mean_LAI':
                self.mapper.plot_map(arr, f'{col}_map', cmap, 0, 2, label, vals[ok], w[ok])
                continue
            if vlim is None:
                vlim = np.nanpercentile(np.abs(vals), 90)
            self.mapper.plot_map(arr, f'{col}_map', cmap, -vlim, vlim, label, vals[ok], w[ok])

    def bin_plot(self, info, val_col, bin_col, bin_list, label_list, xlabel, ylabel, fname):
        ## area-weighted mean of val_col per bin of bin_col and vegetation type, 95% CI from a 1 deg block bootstrap
        w = info['area_weight'].values
        v = info[val_col].values
        idx = np.digitize(info[bin_col].values, bin_list) - 1
        group_list = ['All'] + self.landcover_list
        groups = self.group_dic(info)
        result = []
        for g in group_list:
            for b, label in enumerate(label_list):
                m = groups[g] & (idx == b) & np.isfinite(v)
                if m.sum() < 30:
                    continue
                lo, hi = self.block_mean_ci(info[m], v[m], w[m])
                result.append([g, label, b, self.wmean(v[m], w[m]), lo, hi, m.sum(),
                               np.sum(w[m]) / np.sum(w[groups[g] & np.isfinite(v)]) * 100])
        df_r = pd.DataFrame(result, columns=['group', 'bin', 'bin_id', 'mean', 'ci_low', 'ci_high', 'n_pix',
                                             'area_%_of_group'])
        df_r.to_csv(self.outdir + f'{fname}.csv', index=False)
        print(fname)
        print(df_r)
        color_dic = {'All': 'k', 'Forest': '#1b7837', 'Shrub': '#bf812d', 'Grass': '#dfc27d', 'Savanna': '#80cdc1'}
        plt.figure(figsize=(12 * centimeter_factor, 8 * centimeter_factor))
        n_g = len(group_list)
        for i, g in enumerate(group_list):
            d = df_r[df_r['group'] == g]
            x = d['bin_id'].values + (i - (n_g - 1) / 2) * 0.08
            plt.errorbar(x, d['mean'], yerr=[d['mean'] - d['ci_low'], d['ci_high'] - d['mean']],
                         color=color_dic[g], marker='o', ms=3, lw=1.2 if g == 'All' else 0.8, capsize=2, label=g)
        plt.axhline(0, color='k', lw=0.5)
        plt.xticks(range(len(label_list)), label_list, fontsize=6)
        plt.xlabel(xlabel, fontsize=8)
        plt.ylabel(ylabel, fontsize=8)
        plt.legend(frameon=False, fontsize=6)
        plt.tight_layout()
        plt.savefig(self.outdir + f'{fname}.png', dpi=300)
        plt.savefig(self.outdir + f'{fname}.pdf')
        plt.close()

    def regression(self, info):
        ## per vegetation type: C = g0 + g1 z(r_water) + g2 z(monsoon share), for C in % and in m2/m2
        ## and C = g0 + g2 z(monsoon share) alone: does the monsoon effect shrink once water limitation is in?
        w_all = info['area_weight'].values
        model_dic = {'monsoon_only': ['monsoon_share'], 'r_only': ['r_water'],
                     'r_and_monsoon': ['r_water', 'monsoon_share']}
        result = []
        for y_col in ['C', 'C_abs']:
            for g, m0 in self.group_dic(info).items():
                m = m0 & np.isfinite(info[y_col].values) & np.isfinite(info['r_water'].values) & \
                    np.isfinite(info['monsoon_share'].values)
                if m.sum() < 30:
                    continue
                yy, w = info[y_col].values[m], w_all[m]
                sw = np.sqrt(w)
                for model, x_list in model_dic.items():
                    Z = [np.ones(m.sum())]
                    for c in x_list:
                        xm = info[c].values[m]
                        Z.append((xm - xm.mean()) / xm.std())
                    Z = np.column_stack(Z)
                    gam = np.linalg.lstsq(Z * sw[:, None], yy * sw, rcond=None)[0]
                    lo, hi = self.block_bootstrap(info[m], Z, yy, w)
                    pred = Z @ gam
                    r2 = 1 - np.sum(w * (yy - pred) ** 2) / np.sum(w * (yy - np.sum(w * yy) / np.sum(w)) ** 2)
                    for j, name in enumerate(['intercept'] + x_list):
                        result.append([y_col, g, model, name, gam[j], lo[j], hi[j], r2, m.sum()])
        df_g = pd.DataFrame(result, columns=['C_unit', 'group', 'model', 'term', 'gamma_per_SD', 'ci_low', 'ci_high',
                                             'R2_weighted', 'n_pix'])
        df_g.to_csv(self.outdir + 'C_regression_r_monsoon.csv', index=False)
        print(df_g)


class Phenology_check(C_check):
    ## Is the extra greening (C) in the monsoon region a change in the seasonal cycle?
    ##   e.g. earlier green-up (Jun / Jul) or later senescence (Oct / Nov), or a higher peak (Aug / Sep)
    ## data: MODIS monthly LAI (monthly max of MCD15A3H, Jan 2003 ... Dec 2024), Data\MODIS_LAI_0706\dic
    ##   (the summer LAI used so far = mean of Jul-Oct of these monthly values)
    ## pixels, vegetation type, area weight, monsoon share and C: MODIS_water\C_check\C_check.df
    ## region: monsoon = Jul-Sep share of annual P >= 0.3
    ## 1) trend of each calendar month, 2003-2024 (m2/m2 per decade), per region x vegetation type, 1 deg block CI
    ## 2) share of the summer trend from each of Jul, Aug, Sep, Oct:
    ##    summer trend (% of summer mean) = sum over the 4 months of  trend_month / summer mean / 4 x 100
    ## 3) mean seasonal cycle 2003-2013 vs 2014-2024
    ## output: Result2\04_attribution\MODIS_water\phenology\
    def __init__(self):
        super().__init__()
        self.check_dir = self.outdir   ## MODIS_water\C_check\
        self.outdir = this_root + rf'Result2\04_attribution\MODIS_water\phenology\\'
        T.mk_dir(self.outdir, force=True)
        self.mapper.outdir = self.outdir
        self.month_name_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.summer_month_list = [6, 7, 8, 9]   ## index of Jul ... Oct
        self.period_dic = {'2003-2013': (2003, 2013), '2014-2024': (2014, 2024)}
        self.group_list = ['All', 'Grass', 'Shrub', 'Savanna', 'Forest']
        self.min_year = 18   ## a monthly trend needs data in at least 18 of the 22 years
        pass

    def run(self):
        info = T.load_df(self.check_dir + 'C_check.df')
        if 'pix' in info.columns:   ## in case the pixel index was saved as a column
            info = info.set_index('pix')
        info['region'] = np.where(info['monsoon_share'] >= 0.3, 'monsoon', 'non-monsoon')
        lai = self.monthly_lai(info)   ## pixel x year x month
        df_t = self.monthly_trend(info, lai)
        self.plot_monthly_trend(df_t)
        self.plot_summer_share(df_t)
        df_c = self.seasonal_cycle(info, lai)
        self.plot_seasonal_cycle(df_c)
        pass

    def monthly_lai(self, info):
        spatial_dic = T.load_npy_dir(data_root + rf'MODIS_LAI_0706\dic\\')
        n_year = len(self.year_list)
        lai = np.full((len(info), n_year, 12), np.nan)
        for i, pix in enumerate(info.index):
            if pix not in spatial_dic:
                continue
            vals = np.array(spatial_dic[pix], dtype=float)
            if len(vals) != n_year * 12:
                continue
            lai[i] = vals.reshape(n_year, 12)
        del spatial_dic
        ## missing data: share of pixels with NaN, per month and year (only cases above 1% are printed)
        nan_share = np.mean(np.isnan(lai), axis=0)   ## year x month
        for j in range(12):
            for k, y in enumerate(self.year_list):
                if nan_share[k, j] > 0.01:
                    print(f'NaN: {self.month_name_list[j]} {y}: {nan_share[k, j] * 100:.1f}% of pixels')
        summer = np.nanmean(lai[:, :, 6:10], axis=2)
        ## check: should equal the summer LAI used in the model (mean_LAI = its 2003-2024 mean)
        diff = np.nanmean(summer, axis=1) - info['mean_LAI'].values
        print('monthly dict -> summer mean minus mean_LAI in C_check.df: median', np.nanmedian(diff).round(4),
              'max abs', np.nanmax(np.abs(diff)).round(4))
        return lai

    def groups_by_region(self, info):
        dic = {}
        for region in ['monsoon', 'non-monsoon']:
            for g, m in self.group_dic(info).items():
                dic[(region, g)] = m & (info['region'] == region).values
        return dic

    def trend_nan(self, arr):
        ## OLS slope per decade for each row, using only the years with data
        x = np.array(self.year_list, dtype=float)[None, :]
        ok = np.isfinite(arr)
        n = ok.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            xm = np.sum(np.where(ok, x, 0), axis=1) / n
            ym = np.sum(np.where(ok, arr, 0), axis=1) / n
            dx = np.where(ok, x - xm[:, None], 0)
            dy = np.where(ok, arr - ym[:, None], 0)
            slope = np.sum(dx * dy, axis=1) / np.sum(dx ** 2, axis=1) * 10
        slope[n < self.min_year] = np.nan
        return slope

    def monthly_trend(self, info, lai):
        w = info['area_weight'].values
        summer_mean = info['mean_LAI'].values
        result = []
        for month in range(12):
            x = lai[:, :, month]
            ok = np.sum(np.isfinite(x), axis=1) >= self.min_year   ## a few missing years are allowed
            tr = self.trend_nan(x)   ## m2/m2 per decade
            share = tr / summer_mean / 4 * 100     ## % of summer mean per decade (summer months add up)
            for (region, g), m in self.groups_by_region(info).items():
                m = m & ok
                if m.sum() < 30:
                    continue
                lo, hi = self.block_mean_ci(info[m], tr[m], w[m])
                result.append([region, g, month + 1, self.month_name_list[month], self.wmean(tr[m], w[m]), lo, hi,
                               self.wmean(share[m], w[m]), self.wmean(np.nanmean(x[m], axis=1), w[m]),
                               self.wmean(info['C'].values[m], w[m]), self.wmean(info['observed'].values[m], w[m]),
                               m.sum()])
        df_t = pd.DataFrame(result, columns=['region', 'group', 'month', 'month_name', 'trend_m2m2_per_decade',
                                             'ci_low', 'ci_high', 'part_of_summer_trend_%', 'mean_LAI_month',
                                             'C_mean', 'observed_summer_trend_%', 'n_pix'])
        df_t.to_csv(self.outdir + 'monthly_trend_by_region.csv', index=False)
        print(df_t[df_t['group'].isin(['All', 'Grass'])])
        return df_t

    def plot_monthly_trend(self, df_t):
        ## trend of each month, monsoon vs non-monsoon; Jul-Oct shaded
        color_dic = {'monsoon': '#2166ac', 'non-monsoon': '#d6a64f'}
        fig, axes = plt.subplots(1, len(self.group_list), figsize=(24 * centimeter_factor, 6 * centimeter_factor))
        for ax, g in zip(axes, self.group_list):
            ax.axvspan(6.5, 10.5, color='0.92', zorder=0)
            for region in ['monsoon', 'non-monsoon']:
                d = df_t[(df_t['group'] == g) & (df_t['region'] == region)]
                if len(d) == 0:
                    continue
                ax.plot(d['month'], d['trend_m2m2_per_decade'], color=color_dic[region], marker='o', ms=2.5,
                        lw=1, label=region)
                ax.fill_between(d['month'], d['ci_low'], d['ci_high'], color=color_dic[region], alpha=0.2, lw=0)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([m[0] for m in self.month_name_list], fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_title(g, fontsize=8)
        axes[0].set_ylabel('Monthly LAI trend, 2003-2024\n(m2/m2 per decade)', fontsize=7)
        axes[0].legend(frameon=False, fontsize=6)
        plt.tight_layout()
        plt.savefig(self.outdir + 'monthly_LAI_trend.png', dpi=300)
        plt.savefig(self.outdir + 'monthly_LAI_trend.pdf')
        plt.close()

    def plot_summer_share(self, df_t):
        ## part of the summer trend (% per decade) from each of Jul ... Oct; dashed line = C of the region
        color_dic = {7: '#c7e9c0', 8: '#74c476', 9: '#31a354', 10: '#006d2c'}
        region_list = ['monsoon', 'non-monsoon']
        fig, axes = plt.subplots(1, 2, figsize=(18 * centimeter_factor, 7 * centimeter_factor), sharey=True)
        for ax, region in zip(axes, region_list):
            for i, g in enumerate(self.group_list):
                d = df_t[(df_t['group'] == g) & (df_t['region'] == region) & (df_t['month'].between(7, 10))]
                if len(d) < 4:
                    continue
                bottom_pos, bottom_neg = 0, 0
                for _, row in d.iterrows():
                    v = row['part_of_summer_trend_%']
                    if v >= 0:
                        ax.bar(i, v, 0.6, bottom=bottom_pos, color=color_dic[row['month']],
                               label=row['month_name'] if i == 0 else None)
                        bottom_pos += v
                    else:
                        ax.bar(i, v, 0.6, bottom=bottom_neg, color=color_dic[row['month']],
                               label=row['month_name'] if i == 0 else None)
                        bottom_neg += v
                ax.plot(i, d['observed_summer_trend_%'].iloc[0], 'k_', ms=14, mew=1.5,
                        label='observed summer trend' if i == 0 else None)
                ax.plot(i, d['C_mean'].iloc[0], 'r_', ms=14, mew=1.5, label='C' if i == 0 else None)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(range(len(self.group_list)))
            ax.set_xticklabels(self.group_list, fontsize=7)
            ax.set_title(region, fontsize=8)
        axes[0].set_ylabel('Part of the summer LAI trend\n(% of summer mean per decade)', fontsize=7)
        axes[0].legend(frameon=False, fontsize=6)
        plt.tight_layout()
        plt.savefig(self.outdir + 'summer_trend_by_month.png', dpi=300)
        plt.savefig(self.outdir + 'summer_trend_by_month.pdf')
        plt.close()

    def seasonal_cycle(self, info, lai):
        ## area-weighted mean monthly LAI in each period
        w = info['area_weight'].values
        years = np.array(self.year_list)
        result = []
        for (region, g), m in self.groups_by_region(info).items():
            if m.sum() < 30:
                continue
            for p, (y0, y1) in self.period_dic.items():
                my = (years >= y0) & (years <= y1)
                clim = np.nanmean(lai[m][:, my, :], axis=1)   ## pixel x month
                for month in range(12):
                    v = clim[:, month]
                    result.append([region, g, p, month + 1, self.wmean(v, w[m])])
        df_c = pd.DataFrame(result, columns=['region', 'group', 'period', 'month', 'mean_LAI'])
        df_c.to_csv(self.outdir + 'seasonal_cycle_by_period.csv', index=False)
        return df_c

    def plot_seasonal_cycle(self, df_c):
        ## rows: monsoon, non-monsoon; columns: vegetation types; grey = 2003-2013, colour = 2014-2024
        region_list = ['monsoon', 'non-monsoon']
        color_dic = {'2003-2013': '0.6', '2014-2024': '#1a9850'}
        fig, axes = plt.subplots(2, len(self.group_list), figsize=(24 * centimeter_factor, 10 * centimeter_factor))
        for r, region in enumerate(region_list):
            for c, g in enumerate(self.group_list):
                ax = axes[r, c]
                ax.axvspan(6.5, 10.5, color='0.92', zorder=0)
                for p in self.period_dic:
                    d = df_c[(df_c['region'] == region) & (df_c['group'] == g) & (df_c['period'] == p)]
                    if len(d) == 0:
                        continue
                    ax.plot(d['month'], d['mean_LAI'], color=color_dic[p], marker='o', ms=2, lw=1, label=p)
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels([m[0] for m in self.month_name_list], fontsize=6)
                ax.tick_params(labelsize=6)
                ax.set_title(f'{g}, {region}', fontsize=7)
            axes[r, 0].set_ylabel('Mean LAI (m2/m2)', fontsize=7)
        axes[0, 0].legend(frameon=False, fontsize=6)
        plt.tight_layout()
        plt.savefig(self.outdir + 'seasonal_cycle_by_period.png', dpi=300)
        plt.savefig(self.outdir + 'seasonal_cycle_by_period.pdf')
        plt.close()


def main():
    # Attribution_dataframe().run()   ## done: MODIS summer LAI added to attribution.df
    # Attribution_analysis().run()   ## done 2026-11-14: SNU, 1982/1986-2024
    # Attribution_MODIS_check().run()   ## done: SNU vs MODIS, 2003-2024
    # Attribution_MODIS().run()   ## done 2026-11-14: main analysis, MODIS only, 2003-2024, LAI >= 0.1
    # Remainder_C_drivers().run()   ## done 2026-11-14: C rises with monsoon share, not with aridity
    # Monsoon_precip_check().run()   ## done 2026-11-14: monsoon summer rain did not increase -> does not explain C
    # Attribution_dataframe().run()   ## done 2026-11-14: added winter_PRCPTOT and June_PRCP to attribution.df
    # Attribution_MODIS_water().run()   ## done 2026-11-14: main model = winterP_June; C only ~25% smaller in monsoon grass
    # C_check().run()   ## done 2026-11-14: not a denominator effect; water limitation does not explain C
    # Phenology_check().run()   ## done 2026-11-14 (before the Oct 2022 fix)
    ## 2026-11-14: rerun after the Oct 2022 MODIS fix (Data_processing_MODIS_LAI.remove_broken_tif), in this order
    Attribution_dataframe().run()   ## new summer_MODIS_LAI into attribution.df
    Attribution_MODIS_water().run()   ## main model (winterP_June)
    C_check().run()
    Phenology_check().run()


if __name__ == '__main__':
    main()
