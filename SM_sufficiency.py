# coding='utf-8'
## Is soil water still sufficient after 2000? (2026-11-23)
## SPEI is the atmospheric balance (P - PET), not the water the plants can reach. Here: ERA5-Land soil moisture in
##   the MODIS growing-season window, a soil-moisture threshold found from the LAI response itself (critical soil
##   moisture), and how often the growing season falls below it.
## data  : ERA5-Land monthly volumetric soil water, Data\SM\L{1..4}\dic (per-pixel monthly series on the 0.05 deg grid,
##         Jan first_year ... Dec last_year; values as stored, the analysis uses only relative values)
##         L1 0-7 cm, L2 7-28 cm, L3 28-100 cm, L4 100-289 cm; root zone = L1-L3 thickness-weighted (0-100 cm),
##         deep = L4
##         SNU GS LAI (Result\Dataframe\growing_season\GS_SNU.df)
## pixels: the MODIS figure pixels (model2_annual_perennial_compare) with a main season inside one calendar year
## groups: 7 groups; Shrub / Grass split by RAP 1986-2024 mean AFG > PFG (as the long records)
## steps :
##   1 build      : per pixel-year GS / pre-season mean root-zone SM and its percentile among the pixel's own years
##                  (GS_P, PRE_P; v3 2026-11-23, no rescaling). Earlier versions: v2 W rescaled by GS-month p5 / p95;
##                  v1 W = (SM - p5) / (p95 - p5) with p5 / p95 = the pixel's
##                  5th / 95th percentile of all months first_year-last_year (root zone); per pixel-year: GS mean
##                  root-zone SM, GS mean deep SM, GS mean W, and the GS months of W (for the stress fraction)
##   1b pre-season: mean W and root-zone SM of the 3 months before gs_start (water stored before the season;
##                  winter / spring recharge); first year -> NaN if the months fall before Jan of the dic (2026-11-23)
##   2 threshold  : per group, all pixel-years: SNU GS LAI anomaly (detrended per pixel, % of the pixel's mean) vs
##                  GS SM percentile P; area-weighted means in P bins; hinge fit LAI = c + s * min(P - P_crit, 0)
##                  (LAI rises with P below P_crit, flat above); P_crit by grid search, 95% CI by 1 x 1 deg block
##                  bootstrap; the two halves of the record are shown as P histograms
##                  done for GS SM percentile P ('GS') and pre-season W ('PRE'); R2 of the hinge fit on the pixel-years tells which
##                  one LAI follows more closely (SM_threshold_compare_{tag}.csv)
##   3 stress     : per pixel-year, GS SM percentile P below the P_crit of the group (stress = 100 / 0); per group: % of area
##                  below P_crit each year with OLS trend, first half vs second half (Welch p);
##                  trend of GS root-zone and deep SM (% of the pixel's mean per decade)
## reading: SM declines but the area below P_crit stays low and does not rise -> water still sufficient;
##          the area below P_crit rises -> the soil does dry into the range where LAI responds
## caveat : ERA5-Land soil water is modelled from P and PET without CO2 effects on transpiration
## output : Result2\01_drought\SM_sufficiency_{y0}_{y1}\
from growing_season_extraction import *
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15,
                     'ytick.labelsize': 15, 'legend.fontsize': 15, 'legend.title_fontsize': 15})


class SM_sufficiency(GS_LAI_long_term):
    def __init__(self, first_year=1982, last_year=2024):
        ## 2026-11-23: the Data\SM dic series were rebuilt and now run Jan 1982 - Dec 2024 (516 months; 2003-2024
        ##   identical to the earlier dic, no jump at 2000/2001); self.dic_years = years of the dic, the analysis
        ##   period first_year-last_year is cut from it
        self.dic_years = (1982, 2024)
        super().__init__('SM', None, first_year, last_year)
        self.tag = f'{first_year}_{last_year}'
        self.outdir = this_root + rf'Result2\01_drought\SM_sufficiency_{self.tag}\\'
        T.mk_dir(self.outdir, force=True)
        self.layer_dic = {'L1': 7., 'L2': 21., 'L3': 72., 'L4': 189.}   ## thickness, cm
        self.cmp_dff = this_root + \
            rf'Result2\03_response\model2_annual_perennial\model2_annual_perennial_compare_2003_2024.df'
        self.snu_dff = self.df_dir + 'GS_SNU.df'
        self.rap_dff = result_root + rf'\Dataframe\RAP\RAP_cover.df'
        ## v3 (2026-11-23): x = percentile of the year's GS / pre-season root-zone SM within the pixel's own record;
        ##   the v1 / v2 files (rescaled W) are not read
        self.dff_out = self.outdir + f'GS_SM_v3_{self.tag}.df'
        self.npy_win = self.outdir + f'GS_SM_root_months_v3_{self.tag}.npy'
        self.group_list = ['All', 'Forest', 'Shrub annual', 'Shrub perennial', 'Grass annual', 'Grass perennial',
                           'Savanna']
        self.bp_grid = np.arange(5, 96, 1.)   ## percentile
        self.bins = np.arange(0, 100.1, 5)
        ## two periods: split at 2000 when the record starts before 2000 (as the SPEI / LAI step), else two halves
        if first_year < 2000 < last_year:
            self.half_list = [(first_year, 1999), (2000, last_year)]
        else:
            half = (last_year - first_year + 1) // 2
            self.half_list = [(first_year, first_year + half - 1), (first_year + half, last_year)]
        self.half_color = ['#8FB8DE', '#C0392B']
        self.n_pre = 3   ## months before gs_start for the pre-season soil moisture
        self.var_dic = {'GS': ('GS_P', 'GS root-zone SM percentile'),
                        'PRE': ('PRE_P', f'Pre-season ({self.n_pre} months) root-zone SM percentile')}

    def run(self):
        if not os.path.isfile(self.dff_out):
            self.build()
        df = T.load_df(self.dff_out)
        df = self.add_lai(df)
        comp = []
        for key in self.var_dic:
            thr = self.threshold(df, key)
            self.stress(df, thr, key)
            comp.append(thr.assign(variable=key))
        comp = pd.concat(comp)
        comp.to_csv(self.outdir + f'SM_threshold_compare_{self.tag}.csv', index=False)
        print(comp.pivot_table(index='group', columns='variable', values=['P_crit', 'R2']).round(3).to_string())

    ## ---------- 1 build ----------
    def build(self):
        cmp = T.load_df(self.cmp_dff)
        gs = T.load_df(self.gs_dff)
        gs = gs[(gs['n_season'] > 0) & (gs['gs_start'] <= gs['gs_end'])]
        gs = gs[gs.index.isin(set(cmp.index))]
        pix_list = gs.index
        print('SM pixels (MODIS figure pixels, main season in one year):', len(pix_list))
        month = {}
        y_a, y_b = self.first_year, self.last_year
        self.first_year, self.last_year = self.dic_years   ## load_monthly checks the full dic length
        for L in self.layer_dic:
            self.fdir = data_root + rf'SM\{L}\dic\\'
            month[L] = self.load_monthly(pix_list)[:, (y_a - self.dic_years[0]) * 12:(y_b - self.dic_years[0] + 1) * 12]
        self.first_year, self.last_year = y_a, y_b
        th = self.layer_dic
        root = (month['L1'] * th['L1'] + month['L2'] * th['L2'] + month['L3'] * th['L3']) / \
               (th['L1'] + th['L2'] + th['L3'])
        deep = month['L4']
        del month
        start = gs['gs_start'].values.astype(int) - 1
        length = (gs['gs_end'].values - gs['gs_start'].values).astype(int) + 1
        win = self.window_months(root, start, length)   ## pixel x year x 12, GS months of root-zone SM
        n_year = len(self.year_list)
        df = pd.DataFrame({'pix': [p for p in pix_list for _ in range(n_year)],
                           'i_pix': np.repeat(np.arange(len(pix_list)), n_year),   ## row of the window npy
                           'year': np.tile(self.year_list, len(pix_list)),
                           'GS_SM_root': self.window_long(root, start, length).ravel(),
                           'GS_SM_deep': self.window_long(deep, start, length).ravel(),
                           'PRE_SM_root': self.pre_months(root, start).ravel()})
        ## v3 (2026-11-23): no rescaling; the x variable is the percentile of the pixel-year's GS (or pre-season) mean
        ##   root-zone SM among the pixel's own years first_year-last_year (0 = driest year, 100 = wettest)
        for col, out in [('GS_SM_root', 'GS_P'), ('PRE_SM_root', 'PRE_P')]:
            g = df.groupby('pix')[col]
            df[out] = (g.rank(method='average') - 0.5) / g.transform('count') * 100
        info = gs[['lon', 'lat', 'area_weight', 'gs_start', 'gs_end']].copy()
        info['landcover_type'] = cmp.loc[pix_list, 'landcover_type'].values
        info['group'] = self.groups(info)
        df = df.merge(info, left_on='pix', right_index=True, how='left')
        print('GS root SM non-NaN pixel-years:', df['GS_SM_root'].notna().sum(), '/', len(df))
        T.save_df(df, self.dff_out)
        np.save(self.npy_win, win.astype(np.float32))

    def pre_month_values(self, arr, start):
        ## per pixel and year: the n_pre months before gs_start (pixel x year x n_pre); NaN before the first month
        n_year = len(self.year_list)
        k = np.arange(-self.n_pre, 0)[None, None, :]
        idx = np.arange(n_year)[None, :, None] * 12 + start[:, None, None] + k
        v = np.take_along_axis(arr, np.clip(idx, 0, arr.shape[1] - 1).reshape(len(arr), -1), axis=1).reshape(idx.shape)
        return np.where(idx < 0, np.nan, v)

    def pre_months(self, arr, start):
        ## per pixel and year: mean of the n_pre months before gs_start (may reach into the previous year);
        ##   NaN if any of them is missing or falls before the first month of arr
        n_year = len(self.year_list)
        k = np.arange(-self.n_pre, 0)[None, None, :]
        idx = np.arange(n_year)[None, :, None] * 12 + start[:, None, None] + k
        bad = idx < 0
        v = np.take_along_axis(arr, np.clip(idx, 0, arr.shape[1] - 1).reshape(len(arr), -1), axis=1).reshape(idx.shape)
        v = np.where(bad, np.nan, v)
        return np.mean(v, axis=2)

    def window_months(self, arr, start, length):
        ## per pixel and year: the 12 months start ... start + 11 (index 0 = Jan of first_year); months outside the
        ##   window -> NaN
        n_year = len(self.year_list)
        k = np.arange(12)[None, None, :]
        base = np.arange(n_year)[None, :] * 12 + start[:, None]
        idx = np.clip(base[:, :, None] + k, 0, arr.shape[1] - 1)
        v = np.take_along_axis(arr, idx.reshape(len(arr), -1), axis=1).reshape(idx.shape)
        return np.where(k < length[:, None, None], v, np.nan)

    def groups(self, info):
        ## group name per pixel: Forest, Savanna, Shrub / Grass x annual / perennial (RAP 1986-2024 mean AFG > PFG)
        rap = T.load_df(self.rap_dff)
        rap = rap[(rap['year'] >= 1986) & (rap['year'] <= 2024)]
        m = rap.groupby('pix')[['RAP_AFG', 'RAP_PFG']].mean().reindex(info.index)
        herb = np.where(m['RAP_AFG'] > m['RAP_PFG'], 'annual', 'perennial')
        lc = info['landcover_type'].values
        return np.where(np.isin(lc, ['Shrub', 'Grass']) & m['RAP_AFG'].notna().values,
                        lc.astype(object) + ' ' + herb, lc)

    def group_mask(self, df, g):
        return np.ones(len(df), dtype=bool) if g == 'All' else (df['group'] == g).values

    ## ---------- LAI anomaly ----------
    def add_lai(self, df):
        snu = T.load_df(self.snu_dff)
        snu = snu[(snu['year'] >= self.year_list[0]) & (snu['year'] <= self.year_list[-1])]
        ## detrended anomaly per pixel (pixels with all years), % of the pixel's mean
        mat = snu.pivot(index='pix', columns='year', values='GS_LAI').reindex(columns=self.year_list)
        y = mat.values
        x = np.array(self.year_list, dtype=float)
        xm = x - x.mean()
        mean = y.mean(axis=1, keepdims=True)
        slope = (y - mean) @ xm / np.sum(xm ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            anom = np.where(mean > 0, (y - mean - slope[:, None] * xm[None, :]) / mean * 100, np.nan)
        ny = len(self.year_list)
        an = pd.DataFrame({'pix': [p for p in mat.index for _ in range(ny)],
                           'year': np.tile(self.year_list, len(mat)), 'LAI_anom': anom.ravel()})
        df = df.merge(an, on=['pix', 'year'], how='left')
        print('pixel-years with SNU LAI anomaly:', int(np.isfinite(df['LAI_anom']).sum()))
        return df

    ## ---------- 2 threshold ----------
    def hinge_fit(self, x, y, w):
        ## y = c + s * min(x - bp, 0); grid search of bp (weighted LS); returns bp, c, s, sse
        best = (np.nan, np.nan, np.nan, np.inf)
        for bp in self.bp_grid:
            h = np.minimum(x - bp, 0)
            X = np.column_stack([np.ones_like(h), h])
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
            sse = np.sum(w * (y - X @ coef) ** 2)
            if sse < best[3]:
                best = (bp, coef[0], coef[1], sse)
        return best

    def threshold(self, df, key, n_boot=200):
        var, xlabel = self.var_dic[key]
        d_all = df[np.isfinite(df['LAI_anom']) & np.isfinite(df[var])]
        nb = len(self.bins) - 1
        xc = (self.bins[:-1] + self.bins[1:]) / 2
        rng = np.random.default_rng(0)
        rows, curves = [], {}
        for g in self.group_list:
            d = d_all[self.group_mask(d_all, g)]
            b = np.clip(np.digitize(d[var].values, self.bins) - 1, 0, nb - 1)
            w = d['area_weight'].values
            y = d['LAI_anom'].values
            blk = (np.floor(d['lon'].values) * 1000 + np.floor(d['lat'].values)).astype(int)
            ids, inv = np.unique(blk, return_inverse=True)
            SW = np.zeros((len(ids), nb))
            SY = np.zeros((len(ids), nb))
            np.add.at(SW, (inv, b), w)
            np.add.at(SY, (inv, b), w * y)

            def fit(sw, sy):
                ok = sw > sw.sum() * 0.002   ## bins with >= 0.2% of the weight (sparse tail bins left out)
                return self.hinge_fit(xc[ok], sy[ok] / sw[ok], sw[ok])

            sw0, sy0 = SW.sum(0), SY.sum(0)
            bp, c, s, _ = fit(sw0, sy0)
            boot = []
            for _ in range(n_boot):
                pick = rng.integers(0, len(ids), len(ids))
                boot.append(fit(SW[pick].sum(0), SY[pick].sum(0))[0])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            ## R2 of the hinge on the pixel-years (area-weighted)
            pred = c + s * np.minimum(d[var].values - bp, 0)
            ym = np.sum(w * y) / np.sum(w)
            r2 = 1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - ym) ** 2)
            rows.append([g, len(d), bp, lo, hi, c, s, r2])
            curves[g] = (sw0, sy0, d)
        thr = pd.DataFrame(rows, columns=['group', 'n_pixel_year', 'P_crit', 'ci_low', 'ci_high', 'c', 'slope', 'R2'])
        thr.to_csv(self.outdir + f'SM_threshold_{key}_{self.tag}.csv', index=False)
        print(thr.round(3).to_string())
        ## figure: binned LAI anomaly vs P, hinge fit, P_crit with CI; P histograms of the two halves (right axis)
        fig, axs = plt.subplots(2, 4, figsize=(22, 10), sharex=True)
        for ax, g in zip(axs.ravel(), self.group_list):
            sw0, sy0, d = curves[g]
            r = thr[thr['group'] == g].iloc[0]
            ok = sw0 > sw0.sum() * 0.002
            ax.plot(xc[ok], sy0[ok] / sw0[ok], 'o', color='#333333', ms=6)
            xx = np.linspace(xc[ok].min(), xc[ok].max(), 200)
            ax.plot(xx, r['c'] + r['slope'] * np.minimum(xx - r['P_crit'], 0), color='#2E7D32', lw=2.5)
            ax.axvspan(r['ci_low'], r['ci_high'], color='#C8E6C9', alpha=0.6, lw=0)
            ax.axvline(r['P_crit'], color='#2E7D32', ls='--', lw=1.5)
            ax.axhline(0, color='grey', lw=0.8)
            ax.text(0.03, 0.96, f"P$_{{crit}}$ = {r['P_crit']:.0f}\n[{r['ci_low']:.0f}, {r['ci_high']:.0f}]"
                                f"\nR$^2$ = {r['R2']:.3f}",
                    transform=ax.transAxes, va='top', fontsize=15)
            ax2 = ax.twinx()
            for (y0, y1), col in zip(self.half_list, self.half_color):
                m = (d['year'] >= y0) & (d['year'] <= y1)
                ax2.hist(np.clip(d[var][m], self.bins[0], self.bins[-1] - 1e-6), bins=self.bins,
                         weights=d['area_weight'][m] / d['area_weight'][m].sum() * 100, histtype='step', lw=2,
                         color=col, label=f'{y0}-{y1}')
            ax2.set_ylim(0, None)
            ax2.tick_params(labelsize=12, colors='#777777')
            ax.set_title(g)
            ax.spines[['top']].set_visible(False)
        for ax in axs[:, 0]:
            ax.set_ylabel('SNU LAI anomaly (%)')
        for ax in axs[-1]:
            ax.set_xlabel(xlabel)
        axs[-1, -1].axis('off')
        h2, l2 = ax2.get_legend_handles_labels()
        axs[-1, -1].legend(h2, l2, title='Distribution (right axis, %)', loc='center', frameon=False, fontsize=17)
        plt.tight_layout()
        plt.savefig(self.outdir + f'SM_threshold_{key}_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'SM_threshold_{key}_{self.tag}.pdf', bbox_inches='tight')
        plt.close()
        return thr

    ## ---------- 3 stress ----------
    def stress(self, df, thr, key):
        ## stress per pixel-year: GS SM percentile P below the P_crit of the pixel's group (100 / 0), the same quantity the
        ##   threshold was fitted on; the area-weighted yearly mean = % of the group's area below P_crit
        ##   (All uses each group's own P_crit); pixels without a group threshold -> NaN
        ## key 'GS' / 'PRE': which SM percentile (and its P_crit) defines the stress; root / deep SM trends are shown for
        ##   the GS window in both, plus the pre-season root-zone SM
        var = self.var_dic[key][0]
        df = df.copy()
        crit = df['group'].map(thr.set_index('group')['P_crit']).values.astype(float)
        df['stress_frac'] = np.where(np.isfinite(crit) & np.isfinite(df[var]),
                                     (df[var].values < crit) * 100., np.nan)
        for col in ['GS_SM_root', 'GS_SM_deep', 'PRE_SM_root']:
            mean = df.groupby('pix')[col].transform('mean')
            df[col + '_rel'] = (df[col] / mean - 1) * 100
        years = np.array(self.year_list)
        rows, ts = [], {}
        for g in self.group_list:
            d = df[self.group_mask(df, g)]
            ser = {}
            for col in ['stress_frac', 'GS_SM_root_rel', 'GS_SM_deep_rel', 'PRE_SM_root_rel', var]:
                gg = d.assign(wv=d[col] * d['area_weight'], ww=d['area_weight'].where(d[col].notna()))
                gg = gg.groupby('year')[['wv', 'ww']].sum()
                ser[col] = (gg['wv'] / gg['ww']).reindex(years).values
                ok = np.isfinite(ser[col])
                sl, ic, _, p, _ = stats.linregress(years[ok], ser[col][ok])
                row = [g, col, sl * 10, p]
                for y0, y1 in self.half_list:
                    m = (years >= y0) & (years <= y1)
                    row.append(np.nanmean(ser[col][m]))
                ## 95% CI of the half difference from the yearly values (Welch)
                a = ser[col][(years >= self.half_list[0][0]) & (years <= self.half_list[0][1])]
                bb = ser[col][(years >= self.half_list[1][0]) & (years <= self.half_list[1][1])]
                row.append(stats.ttest_ind(bb, a, equal_var=False, nan_policy='omit').pvalue)
                rows.append(row)
            ts[g] = ser
        h = [f'{y0}-{y1}' for y0, y1 in self.half_list]
        tab = pd.DataFrame(rows, columns=['group', 'variable', 'trend_per_decade', 'p', f'mean_{h[0]}',
                                          f'mean_{h[1]}', 'p_half_diff'])
        tab.to_csv(self.outdir + f'SM_stress_{key}_{self.tag}.csv', index=False)
        print(tab.round(3).to_string())
        pd.DataFrame({f'{g} | {c}': v for g, ser in ts.items() for c, v in ser.items()}, index=years).to_csv(
            self.outdir + f'SM_stress_timeseries_{key}_{self.tag}.csv')
        ## figure 1: per group, yearly stress fraction (bars) and GS root-zone SM (% of mean, line, right axis)
        fig, axs = plt.subplots(2, 4, figsize=(22, 10), sharex=True)
        for ax, g in zip(axs.ravel(), self.group_list):
            ser = ts[g]
            t = tab[(tab['group'] == g)].set_index('variable')
            ax.bar(years, ser['stress_frac'], color='#E59866', width=0.8)
            for (y0, y1), col in zip(self.half_list, self.half_color):
                m = (years >= y0) & (years <= y1)
                ax.hlines(np.nanmean(ser['stress_frac'][m]), y0 - 0.4, y1 + 0.4, color=col, lw=3)
            r = t.loc['stress_frac']
            rr = t.loc['GS_SM_root_rel' if key == 'GS' else 'PRE_SM_root_rel']
            ax.text(0.02, 0.97, f"stress {r[f'mean_{h[0]}']:.0f}% -> {r[f'mean_{h[1]}']:.0f}% (p={r['p_half_diff']:.2f})"
                                f"\n{key} root SM {rr['trend_per_decade']:+.1f}%/dec (p={rr['p']:.2f})",
                    transform=ax.transAxes, va='top', fontsize=14)
            ax.set_ylim(0, 100)
            ax2 = ax.twinx()
            ax2.plot(years, ser['GS_SM_root_rel' if key == 'GS' else 'PRE_SM_root_rel'], color='#1F618D', lw=2)
            ax2.axhline(0, color='#1F618D', lw=0.6, ls=':')
            ax2.tick_params(labelsize=12, colors='#1F618D')
            ax.set_title(g)
        for ax in axs[:, 0]:
            ax.set_ylabel('Area below P$_{crit}$ (%)')
        axs[-1, -1].axis('off')
        axs[-1, -1].text(0.05, 0.5, f'bars: % of area with {key}\nSM percentile below P$_{{crit}}$\nhorizontal lines: mean of\n'
                                    f'{h[0]} (blue), {h[1]} (red)\nblue line (right axis):\n'
                                    f'{key} root-zone SM, % of mean', fontsize=16, va='center',
                         transform=axs[-1, -1].transAxes)
        plt.tight_layout()
        plt.savefig(self.outdir + f'SM_stress_timeseries_{key}_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'SM_stress_timeseries_{key}_{self.tag}.pdf', bbox_inches='tight')
        plt.close()
        ## figure 2: summary bars: a) trend of GS root-zone and deep SM; b) stress fraction, two halves
        x = np.arange(len(self.group_list))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 7))
        for k, (col, name, color) in enumerate([('PRE_SM_root_rel', f'pre-season root zone', '#7D3C98'),
                                                ('GS_SM_root_rel', 'GS root zone 0-100 cm', '#1F618D'),
                                                ('GS_SM_deep_rel', 'GS deep 100-289 cm', '#85C1E9')]):
            v = [tab[(tab['group'] == g) & (tab['variable'] == col)]['trend_per_decade'].iloc[0]
                 for g in self.group_list]
            a1.bar(x + (k - 1) * 0.27, v, width=0.27, color=color, label=name)
        a1.axhline(0, color='k', lw=1)
        a1.set_ylabel('Soil moisture trend\n(% of mean per decade)')
        a1.set_title(f'a) Soil moisture trend {self.year_list[0]}-{self.year_list[-1]}')
        a1.legend(frameon=False)
        for k, (col, hh) in enumerate(zip(self.half_color, h)):
            v = [tab[(tab['group'] == g) & (tab['variable'] == 'stress_frac')][f'mean_{hh}'].iloc[0]
                 for g in self.group_list]
            a2.bar(x + (k - 0.5) * 0.38, v, width=0.38, color=col, label=hh)
            for xx, vv in zip(x + (k - 0.5) * 0.38, v):
                a2.text(xx, vv + 1, f'{vv:.0f}', ha='center', fontsize=13)
        a2.set_ylabel('Area below P$_{crit}$ (%)')
        a2.set_title(f'b) Area below P$_{{crit}}$ ({key}), two periods')
        a2.legend(frameon=False)
        for ax in (a1, a2):
            ax.set_xticks(x)
            ax.set_xticklabels([g.replace(' ', '\n') for g in self.group_list])
            ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'SM_summary_{key}_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'SM_summary_{key}_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


def main():
    SM_sufficiency(1982, 2024).run()   ## 2026-11-23: ERA5-Land 1982-2024 (dic rebuilt); periods 1982-1999 / 2000-2024


if __name__ == '__main__':
    main()
