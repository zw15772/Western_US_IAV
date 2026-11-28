# coding='utf-8'
## Why are LAI and SPEI decoupled in some places? Step 2: driver model (2026-11-27)
## question: among the pixels dry after 2000 (decoupled + resistant + coupled of decoupling_regions.py), why does LAI rise
##   in some places and fall in others? (spatial question; a driver that is the same everywhere, e.g. CO2, is not seen)
## target : dLAI = mean 2000-2024 - mean 1982-1999 of GS LAI (% of the pixel's mean), from decoupling_regions_{scale}.df
## drivers (per pixel; pre = 1982-1999, post = 2000-2024; GS = the pixel's SNU growing-season months):
##   background (pre-period state)
##     P_pre          annual P (mm/yr), TerraClimate ppt
##     AI_pre         annual P / annual PET
##     T_pre          annual mean Tmax (deg C) (TerraClimate tmin only from 2003)
##     summerP_share  Jul-Sep P / annual P (monsoon)
##     snow_share     share of the annual P falling in months with Tmax < 5 deg C (snow proxy; no SWE, no tmin before 2003)
##     soil_pre       GS root-zone soil moisture 0-100 cm (m3 m-3), ERA5-Land swvl1-3 weighted by depth (7, 21, 72 cm)
##                    (TerraClimate soil only from 2003)
##     LAI_pre        GS LAI (m2 m-2), SNU
##     AFG/PFG/SHR/TRE_pre  RAP cover 1986-1999 (%), annual forbs and grasses / perennial / shrub / tree
##   change (post - pre)
##     dSPEI12        drought dose (dSPEI at the scale of decoupling_regions)
##     dP             annual P (% of pre);  dT  annual T (deg C);  dVPD  GS VPD (kPa);  dSoil  GS root-zone soil moisture (m3 m-3)
##     dAFG/dPFG/dSHR/dTRE  RAP cover (% points; post 2000-2024 - pre 1986-1999)
##     burn_frac      MTBS burned fraction 1984-2024
## two driver sets:
##   'climate': background + climate change + burn_frac (no cover change: cover change may be part of the response)
##   'full'   : + RAP cover change
## models (per driver set):
##   A  RF regression of dLAI (area-weighted), 1 x 1 deg spatial block CV (GroupKFold 5): out-of-fold R2; random-CV R2
##      printed beside it (the gap = optimism from spatial autocorrelation); SHAP (TreeExplainer) on a random subsample:
##      mean |SHAP| importance and dependence of the top 6 drivers
##   check  standardised linear regression (z-scores, area-weighted), 95% CI by 1 deg block bootstrap, VIF
##   B  RF classification decoupled (1) vs coupled (0), block-CV AUC and mean |SHAP| (csv only)
## also: Spearman correlation of the drivers (collinearity), driver means per class
## no lon / lat as drivers; associations, not causes
## needs shap (pip install shap)
## output : {R2}\03_response\decoupling_drivers\
##   decoupling_drivers_{scale}.df            per pixel: class, dLAI, all drivers (cache; delete to rebuild)
##   decoupling_driver_class_means_{scale}.csv  area-weighted mean / median of each driver per class
##   decoupling_driver_corr_{scale}.csv, Fig_decoupling_driver_corr_{scale}
##   decoupling_RF_{set}_{scale}.csv          per driver: mean |SHAP| (%), direction, std. linear coef + CI, VIF
##   decoupling_RF_skill_{scale}.csv          R2 (block CV, random CV, train), AUC of model B, n
##   decoupling_RF_B_{set}_{scale}.csv        model B mean |SHAP| (%)
##   Fig_decoupling_RF_{set}_{scale}          a importance, b linear check, c observed vs predicted, d-i dependence
from decoupling_regions import *
import warnings


class Decoupling_drivers(Decoupling_regions):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2w + rf'03_response\decoupling_drivers\\'
        T.mk_dir(self.outdir, force=True)
        self.rap_dff = result_root + rf'\Dataframe\RAP\RAP_cover.df'
        self.fire_dff = result_root + rf'\Dataframe\MTBS\MTBS_fire.df'
        self.regions_dff = self.result2w + rf'03_response\decoupling_regions\decoupling_regions_{self.scale}.df'
        self.drivers_dff = self.outdir + f'decoupling_drivers_{self.scale}.df'
        self.rap_pre_years = (1986, 1999)   ## RAP starts in 1986
        self.rap_min_pre, self.rap_min_post = 8, 18
        self.snow_tmax = 5   ## snow proxy: P of the months with mean Tmax < 5 deg C (about Tmean < 0; no tmin before 2003)
        self.dry_classes = ['decoupled', 'resistant', 'coupled']
        self.background = ['P_pre', 'AI_pre', 'T_pre', 'summerP_share', 'snow_share', 'soil_pre', 'LAI_pre',
                           'AFG_pre', 'PFG_pre', 'SHR_pre', 'TRE_pre']
        self.change_clim = [f'd{self.scale}', 'dP', 'dT', 'dVPD', 'dSoil', 'burn_frac']
        self.change_cover = ['dAFG', 'dPFG', 'dSHR', 'dTRE']
        self.set_dic = {'climate': self.background + self.change_clim,
                        'full': self.background + self.change_clim + self.change_cover}
        self.label_dic = {'P_pre': 'P pre (mm/yr)', 'AI_pre': 'P/PET pre', 'T_pre': 'Tmax pre (°C)',
                          'summerP_share': 'Jul-Sep P share', 'snow_share': 'snow share of P',
                          'soil_pre': 'GS soil moisture pre (m³ m⁻³)', 'LAI_pre': 'LAI pre (m² m⁻²)',
                          'AFG_pre': 'annual herb cover pre (%)', 'PFG_pre': 'perennial herb cover pre (%)',
                          'SHR_pre': 'shrub cover pre (%)', 'TRE_pre': 'tree cover pre (%)',
                          f'd{self.scale}': f'Δ{self.scale}', 'dP': 'ΔP (%)', 'dT': 'ΔTmax (°C)',
                          'dVPD': 'ΔGS VPD (kPa)', 'dSoil': 'ΔGS soil moisture (m³ m⁻³)', 'burn_frac': 'burned fraction',
                          'dAFG': 'Δannual herb cover (%)', 'dPFG': 'Δperennial herb cover (%)',
                          'dSHR': 'Δshrub cover (%)', 'dTRE': 'Δtree cover (%)'}
        self.kind_colors = {'background': '#8C6D31', 'climate change': '#3182BD', 'fire': '#E6550D',
                            'cover change': '#31A354'}
        self.rf_params = dict(n_estimators=300, min_samples_leaf=20, max_features=0.33, n_jobs=-1, random_state=0)
        self.n_fold = 5
        self.n_shap = 4000
        self.n_boot = 200
        self.font = {'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 20, 'xtick.labelsize': 18,
                     'ytick.labelsize': 18}

    def run(self):
        if not os.path.isfile(self.drivers_dff):
            self.build_drivers()
        df = T.load_df(self.drivers_dff)
        df = df[df['class'].isin(self.dry_classes)]
        self.class_means(df)
        self.driver_corr(df)
        skill = []
        for name, feats in self.set_dic.items():
            d = self.complete(df, feats, name)
            res, sk = self.model_A(d, feats, name)
            skill.append(sk + [self.model_B(d, feats, name)])
        skill = pd.DataFrame(skill, columns=['set', 'n_pix', 'R2_block_cv', 'R2_random_cv', 'R2_train', 'AUC_B_block_cv'])
        skill.to_csv(self.outdir + f'decoupling_RF_skill_{self.scale}.csv', index=False)
        print(skill.round(3).to_string())

    ## ---------- drivers ----------
    def build_drivers(self):
        reg = T.load_df(self.regions_dff)
        reg = reg[reg['class'].isin(self.dry_classes)]
        pix_list = list(reg.index)
        lai = self.load()
        lai = lai[lai['pix'].isin(set(pix_list))]
        gs = lai.groupby('pix')[['gs_start', 'gs_end']].first().reindex(pix_list)
        mon = np.arange(12)[None, :]
        in_win = (mon >= gs['gs_start'].values[:, None] - 1) & (mon <= gs['gs_end'].values[:, None] - 1)
        pre, post = self.years < self.split, self.years >= self.split
        out = reg[['lon', 'lat', 'area_weight', 'landcover_type', 'group', 'class', 'dLAI', 'dLAI_p', 'SPEI_pre',
                   'SPEI_post']].copy()
        out[f'd{self.scale}'] = reg['dSPEI'].values
        ## LAI pre (absolute)
        lai_pre = lai[lai['year'] < self.split].groupby('pix')['growing_LAI'].mean()
        out['LAI_pre'] = lai_pre.reindex(pix_list).values
        ## precipitation and PET (annual sums)
        ppt = self.monthly('ppt', pix_list)
        p_ann = ppt.sum(axis=2)
        out['P_pre'] = p_ann[:, pre].mean(axis=1)
        out['dP'] = (p_ann[:, post].mean(axis=1) / out['P_pre'].values - 1) * 100
        out['summerP_share'] = ppt[:, pre, 6:9].sum(axis=(1, 2)) / p_ann[:, pre].sum(axis=1)
        pet = self.monthly('pet', pix_list)
        out['AI_pre'] = out['P_pre'].values / pet.sum(axis=2)[:, pre].mean(axis=1)
        del pet
        ## temperature, snow share (2026-11-27: TerraClimate tmin is only downloaded from 2003 -> Tmax only)
        tmax = self.monthly('tmax', pix_list)
        t_ann = tmax.mean(axis=2)
        out['T_pre'] = t_ann[:, pre].mean(axis=1)
        out['dT'] = t_ann[:, post].mean(axis=1) - out['T_pre'].values
        out['snow_share'] = np.where(tmax[:, pre] < self.snow_tmax, ppt[:, pre], 0).sum(axis=(1, 2)) / \
                            p_ann[:, pre].sum(axis=1)
        del ppt, tmax
        ## GS VPD
        gs_v = self.gs_mean(self.monthly('vpd', pix_list), in_win)
        out['dVPD'] = gs_v[:, post].mean(axis=1) - gs_v[:, pre].mean(axis=1)
        ## GS root-zone soil moisture, ERA5-Land 0-100 cm (2026-11-27: TerraClimate soil only from 2003)
        sm = sum(self.monthly(f'swvl{k}', pix_list, data_root + rf'SM\L{k}\dic\\') * d
                 for k, d in [(1, 7), (2, 21), (3, 72)]) / 100 / 1000   ## dic values are m3 m-3 x 1000
        gs_v = self.gs_mean(sm, in_win)
        out['soil_pre'] = gs_v[:, pre].mean(axis=1)
        out['dSoil'] = gs_v[:, post].mean(axis=1) - out['soil_pre'].values
        del sm, gs_v
        ## RAP cover
        rap = T.load_df(self.rap_dff)
        rap = rap[rap['pix'].isin(set(pix_list))]
        ry = np.arange(self.rap_pre_years[0], self.years[-1] + 1)
        r_pre, r_post = ry <= self.rap_pre_years[1], ry >= self.split
        for c in ['AFG', 'PFG', 'SHR', 'TRE']:
            arr = rap.pivot_table(index='pix', columns='year', values=f'RAP_{c}').reindex(index=pix_list, columns=ry).values
            ok = (np.isfinite(arr[:, r_pre]).sum(axis=1) >= self.rap_min_pre) & \
                 (np.isfinite(arr[:, r_post]).sum(axis=1) >= self.rap_min_post)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                m_pre, m_post = np.nanmean(arr[:, r_pre], axis=1), np.nanmean(arr[:, r_post], axis=1)
            out[f'{c}_pre'] = np.where(ok, m_pre, np.nan)
            out[f'd{c}'] = np.where(ok, m_post - m_pre, np.nan)
        del rap
        ## fire (no MTBS record = not burned)
        fire = T.load_df(self.fire_dff)
        out['burn_frac'] = fire['burn_frac_1984_2024'].reindex(pix_list).fillna(0).values
        T.save_df(out, self.drivers_dff)
        T.df_to_excel(out.head(2000), self.drivers_dff.replace('.df', '_head'))
        print('drivers saved:', len(out), 'pixels')

    def monthly(self, var, pix_list, fdir=None, y0=None):
        ## pixel x year x month, 1982-2024, TerraClimate dic (or the dic folder fdir)
        ## y0: first year of the output (default 1982; 2026-11-28: 1981 for the antecedent months)
        y0 = self.years[0] if y0 is None else y0
        spatial_dic = T.load_npy_dir(fdir if fdir else data_root + rf'TerraClimate\{var}\dic\\')
        n = max(len(v) for v in list(spatial_dic.values())[:1000])
        first_year = 2025 - n // 12
        k0 = (y0 - first_year) * 12
        nm = (self.years[-1] - y0 + 1) * 12
        if k0 < 0 or k0 + nm > n:
            raise ValueError(f'{var}: dic covers {first_year}-2024 only, {y0}-{self.years[-1]} needed')
        out = np.full((len(pix_list), nm), np.nan, dtype=np.float32)
        for i, pix in enumerate(pix_list):
            if pix in spatial_dic and len(spatial_dic[pix]) == n:
                out[i] = np.array(spatial_dic[pix], dtype=float)[k0:k0 + nm]
        del spatial_dic
        out[out < (-100 if var in ['tmax', 'tmin'] else 0)] = np.nan   ## fill values
        print(var, '| first year', first_year, '| months', n, '| pixels with data:', int(np.isfinite(out).all(axis=1).sum()),
              'of', len(pix_list))
        return out.reshape(len(pix_list), self.years[-1] - y0 + 1, 12)

    def gs_mean(self, arr, in_win):
        ## pixel x year mean over the GS months; NaN if a month is missing
        v = np.where(in_win[:, None, :], arr, np.nan)
        complete = np.all(np.isfinite(v) | ~in_win[:, None, :], axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return np.where(complete, np.nanmean(v, axis=2), np.nan)

    def complete(self, df, feats, name):
        ok = np.isfinite(df[feats + ['dLAI']].values.astype(float)).all(axis=1)
        print(f'[{name}] pixels: {len(df)} | complete drivers: {int(ok.sum())}')
        for f in feats:
            n_bad = int((~np.isfinite(df[f].values.astype(float))).sum())
            if n_bad:
                print(f'   missing {f}: {n_bad}')
        return df[ok]

    def kind(self, f):
        if f in self.background:
            return 'background'
        if f == 'burn_frac':
            return 'fire'
        return 'cover change' if f in self.change_cover else 'climate change'

    ## ---------- description ----------
    def class_means(self, df, feats=None):
        feats = (self.set_dic['full'] if feats is None else feats) + ['dLAI', 'SPEI_post']
        rows = []
        for c in self.dry_classes:
            d = df[df['class'] == c]
            w = d['area_weight'].values
            for f in feats:
                v = d[f].values.astype(float)
                ok = np.isfinite(v)
                rows.append([c, f, int(ok.sum()), np.sum(v[ok] * w[ok]) / np.sum(w[ok]), np.median(v[ok])])
        tab = pd.DataFrame(rows, columns=['class', 'driver', 'n', 'mean', 'median'])
        tab = tab.pivot_table(index='driver', columns='class', values=['mean', 'median']).reindex(feats)
        tab.to_csv(self.outdir + f'decoupling_driver_class_means_{self.scale}.csv')
        print(tab.round(3).to_string())

    def driver_corr(self, df, feats=None):
        from matplotlib.colors import BoundaryNorm
        feats = self.set_dic['full'] if feats is None else feats
        r = df[feats].astype(float).corr(method='spearman')
        r.to_csv(self.outdir + f'decoupling_driver_corr_{self.scale}.csv')
        hi = [(a, b, r.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if abs(r.loc[a, b]) >= 0.7]
        print('driver pairs with |Spearman r| >= 0.7:', [(a, b, round(v, 2)) for a, b, v in hi])
        bounds = np.arange(-1, 1.01, 0.2)
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(18, 16))
            im = ax.imshow(r.values, cmap='RdBu_r', norm=BoundaryNorm(bounds, 256))
            for i in range(len(feats)):
                for j in range(len(feats)):
                    if i != j:
                        ax.text(j, i, f'{r.values[i, j]:.1f}', ha='center', va='center', fontsize=12,
                                color='white' if abs(r.values[i, j]) >= 0.7 else 'black')
            labels = [self.label_dic[f] for f in feats]
            ax.set_xticks(range(len(feats)))
            ax.set_xticklabels(labels, rotation=90, fontsize=16)
            ax.set_yticks(range(len(feats)))
            ax.set_yticklabels(labels, fontsize=16)
            plt.colorbar(im, ax=ax, shrink=0.8, ticks=bounds, label='Spearman r')
            ax.set_title(f'Drivers, pixels dry after 2000 (n = {len(df)})', loc='left')
            plt.savefig(self.outdir + f'Fig_decoupling_driver_corr_{self.scale}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_decoupling_driver_corr_{self.scale}.pdf', bbox_inches='tight')
            plt.close()

    ## ---------- models ----------
    def blocks(self, d):
        return (np.floor(d['lon'].values) * 1000 + np.floor(d['lat'].values)).astype(int)

    def wr2(self, y, yhat, w):
        return 1 - np.sum(w * (y - yhat) ** 2) / np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2)

    def oof(self, est, X, y, w, groups, proba=False):
        ## out-of-fold predictions; groups None = random folds
        from sklearn.model_selection import GroupKFold, KFold
        from sklearn.base import clone
        pred = np.full(len(y), np.nan)
        split = GroupKFold(self.n_fold).split(X, y, groups) if groups is not None else \
            KFold(self.n_fold, shuffle=True, random_state=0).split(X)
        for tr, te in split:
            m = clone(est).fit(X[tr], y[tr], sample_weight=w[tr])
            pred[te] = m.predict_proba(X[te])[:, 1] if proba else m.predict(X[te])
        return pred

    def shap_values(self, model, X, classifier=False):
        import shap
        rng = np.random.default_rng(0)
        pick = rng.choice(len(X), size=min(self.n_shap, len(X)), replace=False)
        sv = shap.TreeExplainer(model).shap_values(X[pick])
        if classifier:
            sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
        return pick, np.asarray(sv)

    def model_A(self, d, feats, name):
        X = d[feats].values.astype(float)
        y = d['dLAI'].values.astype(float)
        w = d['area_weight'].values.astype(float)
        rf = RandomForestRegressor(**self.rf_params)
        pred = self.oof(rf, X, y, w, self.blocks(d))
        r2_block = self.wr2(y, pred, w)
        r2_rand = self.wr2(y, self.oof(rf, X, y, w, None), w)
        rf.fit(X, y, sample_weight=w)
        r2_train = self.wr2(y, rf.predict(X), w)
        print(f'[{name}] model A R2: block CV {r2_block:.3f} | random CV {r2_rand:.3f} | train {r2_train:.3f}')
        pick, sv = self.shap_values(rf, X)
        imp = np.abs(sv).mean(axis=0)
        res = pd.DataFrame({'driver': feats, 'label': [self.label_dic[f] for f in feats],
                            'kind': [self.kind(f) for f in feats],
                            'mean_abs_SHAP': imp, 'SHAP_%': imp / imp.sum() * 100,
                            'direction_r': [stats.spearmanr(X[pick, j], sv[:, j])[0] for j in range(len(feats))]})
        lin = self.linear_check(d, feats)
        res = res.merge(lin, on='driver').sort_values('SHAP_%', ascending=False).reset_index(drop=True)
        res.to_csv(self.outdir + f'decoupling_RF_{name}_{self.scale}.csv', index=False)
        print(res.drop(columns=['label']).round(3).to_string())
        self.figure(d, res, X[pick], sv, feats, y, pred, w, r2_block, r2_rand, name)
        return res, [name, len(d), r2_block, r2_rand, r2_train]

    def linear_check(self, d, feats):
        ## standardised area-weighted linear regression, 1 deg block bootstrap CI, VIF
        X = d[feats].values.astype(float)
        y = d['dLAI'].values.astype(float)
        w = d['area_weight'].values.astype(float)
        Z = (X - X.mean(axis=0)) / X.std(axis=0)
        yz = (y - y.mean()) / y.std()

        def fit(Zs, ys, ws):
            Z1 = np.column_stack([np.ones(len(ys)), Zs])
            sw = np.sqrt(ws)
            b, *_ = np.linalg.lstsq(Z1 * sw[:, None], ys * sw, rcond=None)
            return b[1:]

        b = fit(Z, yz, w)
        r2 = self.wr2(yz, Z @ b + np.sum(w * (yz - Z @ b)) / np.sum(w), w)
        block = self.blocks(d)
        ids = np.unique(block)
        idx = {k: np.where(block == k)[0] for k in ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(self.n_boot):
            pk = np.concatenate([idx[k] for k in rng.choice(ids, size=len(ids))])
            boot.append(fit(Z[pk], yz[pk], w[pk]))
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        vif = []
        for j in range(len(feats)):
            others = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
            g, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
            vif.append(1 / (1 - (1 - np.sum((Z[:, j] - others @ g) ** 2) / np.sum((Z[:, j] - Z[:, j].mean()) ** 2))))
        print(f'   linear check R2 = {r2:.3f}')
        return pd.DataFrame({'driver': feats, 'lin_coef': b, 'lin_ci_low': lo, 'lin_ci_high': hi, 'VIF': vif,
                             'lin_R2': r2})

    def model_B(self, d, feats, name):
        ## decoupled (1) vs coupled (0)
        d = d[d['class'].isin(['decoupled', 'coupled'])]
        from sklearn.metrics import roc_auc_score
        X = d[feats].values.astype(float)
        y = (d['class'].values == 'decoupled').astype(int)
        w = d['area_weight'].values.astype(float)
        rf = RandomForestClassifier(class_weight='balanced', **self.rf_params)
        prob = self.oof(rf, X, y, w, self.blocks(d), proba=True)
        auc = roc_auc_score(y, prob, sample_weight=w)
        rf.fit(X, y, sample_weight=w)
        _, sv = self.shap_values(rf, X, classifier=True)
        imp = np.abs(sv).mean(axis=0)
        res = pd.DataFrame({'driver': feats, 'kind': [self.kind(f) for f in feats], 'SHAP_%': imp / imp.sum() * 100})
        res = res.sort_values('SHAP_%', ascending=False)
        res.to_csv(self.outdir + f'decoupling_RF_B_{name}_{self.scale}.csv', index=False)
        print(f'[{name}] model B decoupled vs coupled: n = {len(d)} ({int(y.sum())} decoupled) | block-CV AUC {auc:.3f}')
        print(res.round(1).head(8).to_string())
        return auc

    ## ---------- figure ----------
    def figure(self, d, res, Xs, sv, feats, y, pred, w, r2_block, r2_rand, name):
        import matplotlib.patches as mpatches
        with plt.rc_context(self.font):
            fig = plt.figure(figsize=(26, 24))
            gsp = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.42, height_ratios=[1.5, 1, 1])
            n = len(res)
            yy = np.arange(n)[::-1]
            ## a: mean |SHAP|
            ax = fig.add_subplot(gsp[0, 0])
            ax.barh(yy, res['SHAP_%'], color=[self.kind_colors[k] for k in res['kind']], height=0.75)
            for yi, v, r in zip(yy, res['SHAP_%'], res['direction_r']):
                ax.text(v + 0.3, yi, '+' if r > 0.3 else ('−' if r < -0.3 else '±'), va='center', fontsize=18)
            ax.set_yticks(yy)
            ax.set_yticklabels(res['label'], fontsize=16)
            ax.set_xlabel('mean |SHAP| (% of total)')
            ax.set_title('a  RF importance', loc='left', fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            handles = [mpatches.Patch(color=c, label=k) for k, c in self.kind_colors.items()
                       if k in set(res['kind'])]
            ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=16)
            ## b: linear check, same order
            ax = fig.add_subplot(gsp[0, 1])
            err = np.array([res['lin_coef'] - res['lin_ci_low'], res['lin_ci_high'] - res['lin_coef']])
            ax.barh(yy, res['lin_coef'], xerr=err, color=[self.kind_colors[k] for k in res['kind']], height=0.75,
                    error_kw={'lw': 1.2, 'capsize': 3})
            ax.axvline(0, color='k', lw=0.8)
            ax.set_yticks(yy)
            ax.set_yticklabels([])
            ax.set_xlabel('standardised coefficient')
            ax.set_title(f'b  linear check (R² = {res["lin_R2"].iloc[0]:.2f})', loc='left', fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            ## c: observed vs predicted (block CV)
            ax = fig.add_subplot(gsp[0, 2])
            lim = np.nanpercentile(np.abs(np.concatenate([y, pred])), 99)
            ax.hexbin(pred, y, C=w, reduce_C_function=np.sum, gridsize=45, cmap='Greys', bins='log', mincnt=1,
                      extent=[-lim, lim, -lim, lim])
            ax.plot([-lim, lim], [-lim, lim], color='#C0392B', lw=1.2)
            ax.axhline(0, color='k', lw=0.5)
            ax.axvline(0, color='k', lw=0.5)
            ax.set_xlabel('predicted ΔLAI (%)')
            ax.set_ylabel('observed ΔLAI (%)')
            ax.set_title('c  1° block CV', loc='left', fontweight='bold')
            ax.text(0.03, 0.97, f'R² = {r2_block:.2f}\nrandom CV R² = {r2_rand:.2f}\nn = {len(d)}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=18)
            ax.set_aspect('equal')
            ## d-i: dependence of the top 6 drivers
            for k in range(6):
                ax = fig.add_subplot(gsp[1 + k // 3, k % 3])
                f = res['driver'].iloc[k]
                j = feats.index(f)
                x, s = Xs[:, j], sv[:, j]
                lo, hi = np.percentile(x, [1, 99])
                m = (x >= lo) & (x <= hi)
                ax.scatter(x[m], s[m], s=6, color=self.kind_colors[self.kind(f)], alpha=0.25, lw=0)
                edges = np.unique(np.percentile(x[m], np.linspace(0, 100, 11)))
                if len(edges) > 2:
                    b = np.clip(np.digitize(x[m], edges[1:-1]), 0, len(edges) - 2)
                    xc = [np.median(x[m][b == i]) for i in range(len(edges) - 1) if np.any(b == i)]
                    yc = [np.median(s[m][b == i]) for i in range(len(edges) - 1) if np.any(b == i)]
                    ax.plot(xc, yc, color='k', lw=2.5, marker='o', ms=6)
                ax.axhline(0, color='k', lw=0.6)
                ax.set_xlabel(self.label_dic[f])
                if k % 3 == 0:
                    ax.set_ylabel('SHAP (ΔLAI, %)')
                ax.set_title(f'{"defghi"[k]}', loc='left', fontweight='bold')
                ax.spines[['top', 'right']].set_visible(False)
            fig.suptitle(f'Drivers of ΔLAI (2000-2024 − 1982-1999), pixels dry after 2000, driver set: {name}',
                         x=0.08, ha='left', y=0.94, fontsize=24)
            plt.savefig(self.outdir + f'Fig_decoupling_RF_{name}_{self.scale}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_decoupling_RF_{name}_{self.scale}.pdf', bbox_inches='tight')
            plt.close()
        print('saved', f'Fig_decoupling_RF_{name}_{self.scale}')


class Decoupling_drivers_selected(Decoupling_drivers):
    ## 2026-11-28: drivers selected one by one (one per mechanism, |r| < 0.7), same models as Decoupling_drivers
    ## main set (11):
    ##   water background      AI_pre
    ##   seasonality           summerP_share, snow_share
    ##   vegetation structure  TRE_pre, SHR_pre, PFG_pre, AFG_pre (RAP 1986-1999; replaces LAI_pre)
    ##   drought change        dP_GS, dP_ant (% of 1982-1999), dT (Tmax)
    ##   fire                  burn_frac
    ##   dP_GS : P summed over the GS months; dP_ant : P of the 8 months before the GS (as GS_P_antecedent in GS.df)
    ##   left out: P_pre, soil_pre, LAI_pre, T_pre, dSPEI12, dP, dVPD, dSoil, cover change
    ##   soil water capacity and slope will be added when downloaded
    ## checks (one swap each): snow_share -> T_pre;  dT -> dVPD;  dP_GS + dP_ant + dT -> dSPEI12
    ## target: relative dLAI (% of mean), as before
    ## grouped SHAP: sum of SHAP_% per mechanism (collinear members are read together)
    ## output : {R2}\03_response\decoupling_drivers\selected\  (same file names as Decoupling_drivers, + groups csv)
    ##   the driver cache decoupling_drivers_{scale}.df (parent folder) gets the columns dP_GS and dP_ant
    def __init__(self):
        super().__init__()
        self.outdir = self.outdir + rf'selected\\'
        T.mk_dir(self.outdir, force=True)
        self.n_ant = 8
        main = ['AI_pre', 'summerP_share', 'snow_share', 'TRE_pre', 'SHR_pre', 'PFG_pre', 'AFG_pre', 'dP_GS', 'dP_ant',
                'dT', 'burn_frac']
        self.set_dic = {'main': main,
                        'check_Tpre': [('T_pre' if f == 'snow_share' else f) for f in main],
                        'check_VPD': [('dVPD' if f == 'dT' else f) for f in main],
                        'check_SPEI': [f for f in main if f not in ['dP_GS', 'dP_ant', 'dT']] + [f'd{self.scale}']}
        self.group_dic = {'AI_pre': 'water background', 'summerP_share': 'seasonality', 'snow_share': 'seasonality',
                          'T_pre': 'seasonality', 'TRE_pre': 'vegetation structure', 'SHR_pre': 'vegetation structure',
                          'PFG_pre': 'vegetation structure', 'AFG_pre': 'vegetation structure',
                          'dP_GS': 'drought change', 'dP_ant': 'drought change', 'dT': 'drought change',
                          'dVPD': 'drought change', f'd{self.scale}': 'drought change', 'burn_frac': 'fire'}
        self.kind_colors = {'water background': '#2166AC', 'seasonality': '#6A51A3', 'vegetation structure': '#4DAC26',
                            'drought change': '#B2182B', 'fire': '#FF7F00'}
        self.label_dic['dP_GS'] = 'ΔGS P (%)'
        self.label_dic['dP_ant'] = 'Δantecedent P (%)'

    def kind(self, f):
        return self.group_dic[f]

    def run(self):
        df = T.load_df(self.drivers_dff)
        if 'dP_GS' not in df.columns:
            df = self.add_gs_precip(df)
        df = df[df['class'].isin(self.dry_classes)]
        all_feats = list(dict.fromkeys(sum(self.set_dic.values(), [])))
        self.class_means(df, all_feats)
        self.driver_corr(df, all_feats)
        skill, groups = [], []
        for name, feats in self.set_dic.items():
            d = self.complete(df, feats, name)
            res, sk = self.model_A(d, feats, name)
            skill.append(sk + [self.model_B(d, feats, name)])
            g = res.groupby('kind')['SHAP_%'].sum()
            groups += [[name, k, g.get(k, np.nan)] for k in self.kind_colors]
        skill = pd.DataFrame(skill, columns=['set', 'n_pix', 'R2_block_cv', 'R2_random_cv', 'R2_train', 'AUC_B_block_cv'])
        skill.to_csv(self.outdir + f'decoupling_RF_skill_{self.scale}.csv', index=False)
        groups = pd.DataFrame(groups, columns=['set', 'group', 'SHAP_%']).pivot(index='group', columns='set',
                                                                                 values='SHAP_%')
        groups.to_csv(self.outdir + f'decoupling_RF_groups_{self.scale}.csv')
        print(skill.round(3).to_string())
        print(groups.round(1).to_string())

    def add_gs_precip(self, df):
        ## dP_GS, dP_ant (% of the 1982-1999 mean) from TerraClimate ppt (1981 needed for the antecedent months)
        pix_list = list(df.index)
        gs = T.load_df(result_root + rf'\Dataframe\growing_season\{GS_TABLE}')[['gs_start', 'gs_end']].reindex(pix_list)
        ppt = self.monthly('ppt', pix_list, y0=self.years[0] - 1).reshape(len(pix_list), -1)   ## index 0 = Jan 1981
        start = gs['gs_start'].values.astype(int) - 1
        length = (gs['gs_end'].values - gs['gs_start'].values).astype(int) + 1
        base = (np.arange(len(self.years))[None, :] + 1) * 12 + start[:, None]   ## pixel x year, first GS month
        k = np.arange(12)[None, None, :]
        idx = np.clip(base[:, :, None] + k, 0, ppt.shape[1] - 1)
        v = np.take_along_axis(ppt, idx.reshape(len(ppt), -1), axis=1).reshape(idx.shape)
        p_gs = np.where(k < length[:, None, None], v, 0).sum(axis=2)
        ka = np.arange(-self.n_ant, 0)[None, None, :]
        idx = base[:, :, None] + ka
        p_ant = np.take_along_axis(ppt, idx.reshape(len(ppt), -1), axis=1).reshape(idx.shape).sum(axis=2)
        pre, post = self.years < self.split, self.years >= self.split
        for name, arr in [('dP_GS', p_gs), ('dP_ant', p_ant)]:
            m_pre = arr[:, pre].mean(axis=1)
            df[name] = np.where(m_pre >= 1, (arr[:, post].mean(axis=1) / m_pre - 1) * 100, np.nan)
            print(name, '| median %:', np.nanmedian(df[name].values).round(1), '| NaN:', int(np.isnan(df[name].values).sum()))
        T.save_df(df, self.drivers_dff)
        return df


def main():
    # Decoupling_drivers().run()
    Decoupling_drivers_selected().run()   ## 2026-11-28


if __name__ == '__main__':
    main()
