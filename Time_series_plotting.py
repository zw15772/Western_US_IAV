from __Global__ import *
from window_config import *   ## 2026-11-23: GS_WINDOW, WIN, GS_TABLE, R2
## 2026-11-21: larger fonts in all figures (default sizes; explicit fontsize values were also enlarged x1.6)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'legend.fontsize': 11, 'legend.title_fontsize': 11})
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
        self.result2w = this_root + rf'\{R2}\\'   ## 2026-11-23: outputs of the long records with the active window
        self.cbar_label_values = None   ## 2026-11-23: trend_sig_map labels only these colour-bar ticks (None = all)
        self.season_list = ['spring', 'summer']
        self.color_dic = {'spring': '#EA6E88', 'summer': '#48526F'}
        self.split_year = 2000
        self.end_year = 2024
        self.crop_threshold = 50   ## % cropland cover above which a pixel is excluded
        ## trend periods: (start, end, linestyle)
        self.trend_period_list = [(1982, 2024, ':'), (2000, 2024, '-')]
        self.ylabel_dic = {}   ## y-axis label per variable; default = variable name
        ## herbaceous group per pixel (annual: 2003-2024 mean RAP AFG > PFG), from Model2_annual_perennial (2026-11-21)
        self.herb_dff = self.result2 + rf'03_response\model2_annual_perennial\model2_annual_perennial_compare_2003_2024.df'
        ## herb_years None: groups of Model 2 (RAP 2003-2024 mean); (y0, y1): RAP mean AFG > PFG over y0-y1
        ##   (long records use (1986, 2024), the whole RAP record) (2026-11-21)
        self.herb_years = None
        self.rap_dff = result_root + rf'\Dataframe\RAP\RAP_cover.df'
        self._herb_dic = None
        pass

    def herb_group_dic(self):
        ## pix -> 'annual' / 'perennial' ('' = no RAP)
        if self._herb_dic is not None:
            return self._herb_dic
        if self.herb_years is None:
            hg = T.load_df(self.herb_dff)
            self._herb_dic = dict(zip(hg.index, hg['herb_group'].values))
            return self._herb_dic
        y0, y1 = self.herb_years
        rap = T.load_df(self.rap_dff)
        rap = rap[(rap['year'] >= y0) & (rap['year'] <= y1)]
        m = rap.groupby('pix')[['RAP_AFG', 'RAP_PFG']].mean()
        grp = np.where(m.isna().any(axis=1), '', np.where(m['RAP_AFG'] > m['RAP_PFG'], 'annual', 'perennial'))
        self._herb_dic = dict(zip(m.index, grp))
        print(f'herbaceous groups from RAP {y0}-{y1} mean: annual', int(np.sum(grp == 'annual')), '| perennial',
              int(np.sum(grp == 'perennial')))
        return self._herb_dic

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
                                 figsize=(self.map_width * 2.0, self.map_height * 0.7 * max(len(var_list), 1.5)))
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
                        transform=ax.transAxes, color=color, fontsize=10, va='top')   ## one long line: 10

            if 'SPEI' in var:
                ax.axhline(0, color='grey', lw=0.5)
            if self.split_year > start_year:
                ax.axvspan(self.split_year, self.end_year, color='grey', alpha=0.1)
            ax.set_ylabel(self.ylabel_dic.get(var, var), fontsize=16)
            ax.set_xlim(start_year, self.end_year)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[0].legend(loc='upper right', frameon=False, ncol=2)
        axes[-1].set_xlabel('Year', fontsize=16)
        plt.tight_layout()
        plt.savefig(outdir + f'{fname}.png', dpi=300)
        plt.savefig(outdir + f'{fname}.pdf')
        plt.show()
        plt.close()

        df_trend = pd.DataFrame(trend_list, columns=['variable', 'season', 'period', 'slope_per_decade', 'p_value'])
        df_trend.to_csv(outdir + f'{fname}_trend.csv', index=False)
        print(df_trend)

    def pixel_trend(self, df, var, years=np.arange(2003, 2025)):
        ## per-pixel OLS trend (per decade) and two-sided p of df[var], pixels with all years only (2026-11-21)
        mat = df.pivot_table(index='pix', columns='year', values=var).reindex(columns=years)
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first().loc[mat.index]
        y = mat.values
        ok = np.isfinite(y).all(axis=1)
        y, info = y[ok], info[ok]
        x = years.astype(float) - years.mean()
        ym = y - y.mean(axis=1, keepdims=True)
        slope = ym @ x / np.sum(x ** 2)
        resid = ym - slope[:, None] * x[None, :]
        se = np.sqrt(np.sum(resid ** 2, axis=1) / (len(x) - 2) / np.sum(x ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            p = 2 * stats.t.sf(np.abs(slope / se), len(x) - 2)
        return info, slope * 10, p

    def step_change(self, df, var, years, split, outdir, fname, label, bounds, cmap, cat_names, ylabel, **kw):
        ## step change at the drought shift (2026-11-22): per pixel mean(split ... end) - mean(start ... split - 1)
        ##   of df[var]; p of Welch t test (unequal variances) between the two periods; pixels with all years only
        ## outputs: map with dots p < 0.05 + inset + 7-group area bar (trend_sig_map, fname), and
        ##   {fname}_group_mean.png / .csv: area-weighted mean difference per group with 95% CI (1 deg block bootstrap)
        mat = df.pivot_table(index='pix', columns='year', values=var).reindex(columns=years)
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first().loc[mat.index]
        y = mat.values
        ok = np.isfinite(y).all(axis=1)
        y, info = y[ok], info[ok]
        a, b = y[:, years < split], y[:, years >= split]
        diff = b.mean(axis=1) - a.mean(axis=1)
        va, vb = a.var(axis=1, ddof=1) / a.shape[1], b.var(axis=1, ddof=1) / b.shape[1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = diff / np.sqrt(va + vb)
            dof = (va + vb) ** 2 / (va ** 2 / (a.shape[1] - 1) + vb ** 2 / (b.shape[1] - 1))
            p = 2 * stats.t.sf(np.abs(t), dof)
        self.trend_sig_map(info, diff, p, outdir, fname, label, bounds, cmap, cat_names, split_herb=True, **kw)
        self.group_mean_bar(info, diff, p, outdir, f'{fname}_group_mean', ylabel)
        return info, diff, p

    def block_ci_xy(self, lon, lat, v, w, n_boot=200):
        block = (np.floor(lon) * 1000 + np.floor(lat)).astype(int)
        ids = np.unique(block)
        idx = {bb: np.where(block == bb)[0] for bb in ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(n_boot):
            pick = np.concatenate([idx[bb] for bb in rng.choice(ids, size=len(ids))])
            boot.append(np.sum(v[pick] * w[pick]) / np.sum(w[pick]))
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    def group_mean_bar(self, info, v, p, outdir, fname, ylabel):
        ## area-weighted mean of v per group (7 groups) with 95% CI; csv also has the area % with p < 0.05 by sign
        color_dic = {'All': '#555555', 'Forest': '#1B5E20', 'Shrub annual': '#E0B84C', 'Shrub perennial': '#8B6508',
                     'Grass annual': '#C5E1A5', 'Grass perennial': '#558B2F', 'Savanna': '#8D6E63'}
        w_all = info['area_weight'].values
        rows = []
        for g, m in self.group_masks(info).items():
            m = m & np.isfinite(v)
            w = w_all[m]
            lo, hi = self.block_ci_xy(info['lon'].values[m], info['lat'].values[m], v[m], w)
            rows.append([g, int(m.sum()), np.sum(v[m] * w) / np.sum(w), lo, hi,
                         np.sum(w * ((v[m] > 0) & (p[m] < 0.05))) / np.sum(w) * 100,
                         np.sum(w * ((v[m] < 0) & (p[m] < 0.05))) / np.sum(w) * 100])
        tab = pd.DataFrame(rows, columns=['class', 'n_pix', 'mean', 'ci_low', 'ci_high', 'sig_increase_area_%',
                                          'sig_decrease_area_%'])
        tab.to_csv(outdir + f'{fname}.csv', index=False)
        print(fname)
        print(tab.round(2).to_string())
        with plt.rc_context({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 15}):
            fig, ax = plt.subplots(figsize=(10, 4.4))
            xx = np.arange(len(tab))
            vv = tab['mean'].values
            ax.bar(xx, vv, color=[color_dic[c] for c in tab['class']], width=0.65,
                   yerr=[vv - tab['ci_low'].values, tab['ci_high'].values - vv], capsize=4, error_kw={'lw': 1.2})
            ax.axhline(0, color='k', lw=0.8)
            ax.set_xticks(xx)
            ax.set_xticklabels([c.replace(' ', '\n') for c in tab['class']])
            ax.set_ylabel(ylabel)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(outdir + f'{fname}.png', dpi=300, bbox_inches='tight')
            plt.savefig(outdir + f'{fname}.pdf', bbox_inches='tight')
            plt.close()
        return tab

    def group_masks(self, info):
        ## 7 groups as Fig 5 (2026-11-21): All, Forest, Shrub annual / perennial, Grass annual / perennial, Savanna
        ## Shrub and Grass split by the herbaceous group of Model2_annual_perennial (mean AFG > mean PFG -> annual)
        ## 2026-11-21: or by the RAP mean over self.herb_years (long records: 1986-2024), see herb_group_dic
        grp_dic = self.herb_group_dic()
        grp = np.array([grp_dic.get(pix, '') for pix in info.index])
        lc = info['landcover_type'].values
        out = {'All': np.ones(len(info), dtype=bool), 'Forest': lc == 'Forest'}
        for c in ['Shrub', 'Grass']:
            for g in ['annual', 'perennial']:
                out[f'{c} {g}'] = (lc == c) & (grp == g)
        out['Savanna'] = lc == 'Savanna'
        return out

    def group_area_bar(self, tab, cats, outdir, fname):
        ## stacked horizontal bars per group: area % of the 4 classes; area-weighted mean trend on the right (2026-11-21)
        ## the two non-sig classes in two greys (dark = negative, light = positive)
        import matplotlib.patches as mpatches
        colors = [cats[0][2], '#A6A6A6', '#D9D9D9', cats[3][2]]
        names = [n for n, _, _ in cats]
        fig, ax = plt.subplots(figsize=(self.map_width * 1.5, self.map_height * 1.1))
        y = np.arange(len(tab))[::-1]
        left = np.zeros(len(tab))
        for k, (c, n) in enumerate(zip(colors, names)):
            v = tab[n + ' area_%'].values
            ax.barh(y, v, left=left, color=c, height=0.7, edgecolor='white', linewidth=0.5)
            for yy, l0, vv in zip(y, left, v):
                if vv >= 8:
                    ax.text(l0 + vv / 2, yy, f'{vv:.0f}', ha='center', va='center', fontsize=9.5,
                            color='white' if k in (0, 3) else 'black')
            left += v
        for yy, mt in zip(y, tab['mean_trend'].values):
            ax.text(102, yy, f'{mt:+.2f}', va='center', fontsize=10.5)
        ax.text(102, y[0] + 0.55, 'Mean', va='bottom', fontsize=10.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{c} (n={n})' for c, n in zip(tab['class'], tab['n_pix'])], fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Area (%)', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        handles = [mpatches.Patch(color=c, label=n) for c, n in zip(colors, names)]
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        plt.tight_layout(rect=[0, 0.07, 0.92, 1])
        plt.savefig(outdir + f'{fname}_by_group.png', dpi=300, bbox_inches='tight')
        plt.savefig(outdir + f'{fname}_by_group.pdf', bbox_inches='tight')
        plt.close()

    def level_change(self, years, y, bp, n_lag=None):
        ## level change at a break year (2026-11-21), segmented regression on one series:
        ##   y = b0 + b1 t + b2 D + b3 t D,  t = year - bp,  D = 1 for year >= bp
        ##   b0 = line before bp extrapolated to bp; b0 + b2 = line after bp at bp -> b2 = level change at bp
        ##   b1 = slope before, b1 + b3 = slope after (per year)
        ## 95% CI and p of b2 with Newey-West (HAC) standard errors (year-to-year autocorrelation),
        ##   lag = floor(4 (n / 100) ^ (2 / 9)), Bartlett weights, small-sample factor n / (n - 4), t with n - 4 df
        ok = np.isfinite(y)
        x, y = years[ok].astype(float), np.asarray(y, dtype=float)[ok]
        t = x - bp
        D = (x >= bp).astype(float)
        X = np.column_stack([np.ones_like(t), t, D, t * D])
        n, k = X.shape
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        L = n_lag if n_lag is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for l in range(1, L + 1):
            G = Xe[l:].T @ Xe[:-l]
            S += (1 - l / (L + 1)) * (G + G.T)
        XtXi = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtXi @ S @ XtXi) * n / (n - k))
        tc = stats.t.ppf(0.975, n - k)
        return {'b2': b[2], 'ci_low': b[2] - tc * se[2], 'ci_high': b[2] + tc * se[2],
                'p': 2 * stats.t.sf(abs(b[2] / se[2]), n - k), 'slope_before': b[1] * 10,
                'slope_after': (b[1] + b[3]) * 10, 'b0': b[0], 'b1': b[1], 'b3': b[3], 'n': n, 'lag': L}

    def level_change_groups(self, df, col, years, bp_list, outdir, fname, ylabel):
        ## level change b2 per group (7 groups) and break year; regional area-weighted mean series per group
        ## figure {fname}.png: left: All series with the two lines at bp 2000 (dashed: before-line extrapolated) and
        ##   the gap at 2000; right: b2 with 95% CI per group, one marker per break year
        ## table {fname}.csv
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
        series, rows = {}, []
        for g, m in self.group_masks(info).items():
            d = df[df['pix'].isin(set(info.index[m]))]
            gg = d.assign(wv=d[col] * d['area_weight'], w=d['area_weight'].where(d[col].notna()))
            gg = gg.groupby('year')[['wv', 'w']].sum()
            y = (gg['wv'] / gg['w']).reindex(years).values
            series[g] = y
            for bp in bp_list:
                r = self.level_change(years, y, bp)
                rows.append([g, int(m.sum()), bp] + [r[k] for k in ['b2', 'ci_low', 'ci_high', 'p', 'slope_before',
                                                                    'slope_after', 'n', 'lag']])
        tab = pd.DataFrame(rows, columns=['class', 'n_pix', 'break_year', 'level_change', 'ci_low', 'ci_high', 'p',
                                          'slope_before_per_decade', 'slope_after_per_decade', 'n_year', 'hac_lag'])
        tab.to_csv(outdir + f'{fname}.csv', index=False)
        print(fname)
        print(tab.round(2).to_string())
        fig, axs = plt.subplots(1, 2, figsize=(self.map_width * 2.6, self.map_height * 1.1),
                                gridspec_kw={'width_ratios': [3, 2]})
        ax = axs[0]
        y = series['All']
        r = self.level_change(years, y, 2000)
        ax.plot(years, y, color='#555555', lw=1.2)
        tb, ta = years[years < 2000], years[years >= 2000]
        ax.plot(tb, r['b0'] + r['b1'] * (tb - 2000), color='#1F618D', lw=2.2, label='before 2000')
        ax.plot([tb[-1], 2000], [r['b0'] + r['b1'] * (tb[-1] - 2000), r['b0']], color='#1F618D', lw=2.2, ls='--')
        ax.plot(ta, r['b0'] + r['b2'] + (r['b1'] + r['b3']) * (ta - 2000), color='#C0392B', lw=2.2, label='2000 on')
        ax.plot([2000, 2000], [r['b0'], r['b0'] + r['b2']], color='k', lw=2.5)
        ax.text(2000.6, r['b0'] + r['b2'] / 2, f"level change {r['b2']:+.2f}\n[{r['ci_low']:.2f}, {r['ci_high']:.2f}]",
                va='center', fontsize=11)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Year')
        ax.legend(frameon=False, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        ax = axs[1]
        groups = list(series)
        color_list = ['#9E9E9E', '#1F618D', '#C0392B', '#E67E22']
        for k, bp in enumerate(bp_list):
            d = tab[tab['break_year'] == bp].set_index('class').reindex(groups)
            yy = np.arange(len(groups))[::-1] + (k - (len(bp_list) - 1) / 2) * 0.18
            ax.errorbar(d['level_change'], yy, xerr=[d['level_change'] - d['ci_low'], d['ci_high'] - d['level_change']],
                        fmt='o', ms=5, color=color_list[k % 4], capsize=2, lw=1, label=str(bp))
        ax.axvline(0, color='k', lw=0.6)
        ax.set_yticks(np.arange(len(groups))[::-1])
        ax.set_yticklabels(groups)
        ax.set_xlabel('Level change at break year')
        ax.legend(frameon=False, title='Break year', fontsize=10, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(outdir + f'{fname}.png', dpi=300, bbox_inches='tight')
        plt.savefig(outdir + f'{fname}.pdf', bbox_inches='tight')
        plt.close()
        return tab

    def trend_sig_map(self, info, trend, p, outdir, fname, label, bounds, cmap, cat_names, stride=3, lighten=0.,
                      dot_size=0.6, split_herb=False):
        ## trend map with discrete colours, dots = p < 0.05 (every `stride`-th pixel in both directions, so the dots
        ## stay readable), colour bar under the map; small bar chart in the lower-left corner of the map
        ## (Pacific, outside the study pixels) = area share of 4 classes: sig / non-sig negative,
        ## non-sig / sig positive (layout 2026-11-21)
        ## cat_names: (negative word, positive word), e.g. ('drying', 'wetting') or ('browning', 'greening')
        ## also saves {fname}_area.csv: the 4 classes per vegetation type
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        from matplotlib.colors import BoundaryNorm
        T.mk_dir(outdir, force=True)
        ny, nx = 468, 449
        arr = np.full((ny, nx), np.nan)
        rows = np.array([q[0] for q in info.index])
        cols = np.array([q[1] for q in info.index])
        arr[rows, cols] = trend
        if isinstance(cmap, (list, tuple)):   ## a list of hex colours, one per class (2026-11-21)
            from matplotlib.colors import ListedColormap
            cmap_obj = ListedColormap(list(cmap))
            cmap_obj.set_under(cmap[0])
            cmap_obj.set_over(cmap[-1])
        else:
            cmap_obj = plt.get_cmap(cmap, len(bounds) - 1)
        if lighten > 0:   ## mix the colours with white so the dots stay visible on the dark classes (2026-11-21)
            from matplotlib.colors import ListedColormap
            rgba = cmap_obj(np.arange(cmap_obj.N))
            rgba[:, :3] = rgba[:, :3] * (1 - lighten) + lighten
            cmap_obj = ListedColormap(rgba)
            cmap_obj.set_under(rgba[0])
            cmap_obj.set_over(rgba[-1])
        norm = BoundaryNorm(bounds, cmap_obj.N)
        neg, pos = cat_names
        cats = [(f'Sig. {neg}', (trend < 0) & (p < 0.05), cmap_obj(0.0)),
            (f'Non-sig. {neg}', (trend < 0) & ~(p < 0.05), '#BDBDBD'),  ## non-sig: grey
        (f'Non-sig. {pos}', (trend >= 0) & ~(p < 0.05), '#BDBDBD'),
                (f'Sig. {pos}', (trend >= 0) & (p < 0.05), cmap_obj(0.999))]
        w = info['area_weight'].values
        area = [np.sum(w[m]) / np.sum(w) * 100 for _, m, _ in cats]
        fig = plt.figure(figsize=(self.map_width, self.map_height))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.imshow(arr, cmap=cmap_obj, norm=norm, extent=[-124.529, -102.079, 25.579, 48.979],
                       transform=ccrs.PlateCarree(), interpolation='nearest')
        sig = (p < 0.05) & (rows % stride == 0) & (cols % stride == 0)
        ax.scatter(info['lon'].values[sig], info['lat'].values[sig], s=dot_size, c='k', marker='.', lw=0,
                   transform=ccrs.PlateCarree(), zorder=5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
        ax.set_extent([-125, -102, 28, 49], crs=ccrs.PlateCarree())   ## room in the lower right for the bar chart
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(range(-120, -100, 10))   ## every 10 deg: 5 deg labels overlapped (2026-11-21)
        gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
        gl.xlabel_style = {'size': 9.5}
        gl.ylabel_style = {'size': 9.5}
        ax.set_title(label, fontsize=11)
        cb = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, ticks=bounds, extend='both', pad=0.08,
                          aspect=35)
        cb.ax.tick_params(labelsize=9.5)
        if self.cbar_label_values is not None:   ## 2026-11-23: fewer labels so the numbers do not overlap
            cb.set_ticklabels([f'{b:g}' if np.any(np.isclose(b, self.cbar_label_values)) else '' for b in bounds])
        cb.set_label('Dots: p < 0.05', fontsize=9.5)
        ## small bar chart in the lower-left corner of the map (Pacific, no study pixels there), no axis labels:
        ##   bars left to right = sig negative, non-sig negative, non-sig positive, sig positive (explained in caption)
        ax_b = ax.inset_axes([0.02, 0.03, 0.30, 0.17])   ## wider so the 4 labels do not overlap (2026-11-21)
        ax_b.bar(range(4), area, color=[c for _, _, c in cats], width=0.75, edgecolor='#333333', linewidth=0.25)
        for i_, a in enumerate(area):
            ax_b.text(i_, a + max(area) * 0.04, f'{a:.0f}%', ha='center', va='bottom', fontsize=6)
        ax_b.set_ylim(0, max(area) * 1.3 + 1)
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        ax_b.spines[['top', 'right', 'left']].set_visible(False)
        ax_b.spines['bottom'].set_linewidth(0.4)
        ax_b.patch.set_alpha(0)
        ax_b.set_zorder(20)
        plt.savefig(outdir + f'{fname}.png', dpi=300, bbox_inches='tight')
        plt.savefig(outdir + f'{fname}.pdf', bbox_inches='tight')
        plt.close()
        ## table per vegetation type; split_herb: 7 groups as Fig 5 + stacked bar figure {fname}_by_group (2026-11-21)
        if split_herb:
            groups = self.group_masks(info)
        else:
            lc_all = info['landcover_type'].values
            groups = {lc: (np.ones(len(info), dtype=bool) if lc == 'All' else lc_all == lc)
                      for lc in ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']}
        result = []
        for lc, m in groups.items():
            if m.sum() == 0:
                continue
            row = [lc, int(m.sum()), np.sum(trend[m] * w[m]) / np.sum(w[m])]
            row += [np.sum(w[m & mm]) / np.sum(w[m]) * 100 for _, mm, _ in cats]
            result.append(row)
        tab = pd.DataFrame(result, columns=['class', 'n_pix', 'mean_trend'] + [n + ' area_%' for n, _, _ in cats])
        tab.to_csv(outdir + f'{fname}_area.csv', index=False)
        print(fname)
        print(tab.round(2).to_string())
        if split_herb:
            self.group_area_bar(tab, cats, outdir, fname)


class SPEI_time_series_MODIS(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir =  self.result2 + rf'01_drought\SPEI_short_term\\'   ## 2026-11-21: renamed from SPEI_timeseries
        self.start_year = 1982
        self.main_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI48']
        self.other_list = ['SPEI01', 'SPEI09', 'SPEI15', 'SPEI18', 'SPEI21', 'SPEI27',
                           'SPEI30', 'SPEI33', 'SPEI36', 'SPEI39', 'SPEI42', 'SPEI45']
        self.short_list = ['SPEI03', 'SPEI06', 'SPEI09', 'SPEI12']   ## 2026-11-21: 3-12 months, 2003-2024
        self.spei_bounds = [-0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6]   ## SPEI per decade, discrete colours
        pass



    def run(self):
        ## SPEI averaged over each pixel's main growing season (GS.df), 2003-2024
        df = self.load_GS_df()
        df = df[df['landcover_type'].isin(['Forest', 'Shrub', 'Grass', 'Savanna'])]   ## four vegetation types (2026-11-21)
        print('pixels (four vegetation types):', df['pix'].nunique())
        self.color_dic = {'growing': '#D9480F'}   ## SPEI in orange-red, as in SPEI_long_term_plot (load_GS_df sets LAI green)
        outdir = self.outdir
        # other_list = [v for v in self.other_list if v != 'SPEI01']   ## GS.df has SPEI03 ... SPEI48
        # self.plot_time_series(df, self.main_list, 2003, outdir, 'Fig_SPEI_main_GS')
        # self.plot_time_series(df, other_list, 2003, outdir, 'FigS_SPEI_other_GS')
        ## 2026-11-21: SPEI03-12, 2003-2024: time series + trend maps (dots p < 0.05, inset: 4 classes)
        self.plot_time_series(df, self.short_list, 2003, outdir, 'Fig_SPEI03_12_GS')
        for var in self.short_list:
            info, tr, p = self.pixel_trend(df, f'growing_{var}')
            self.trend_sig_map(info, tr, p, outdir + 'trend_map\\', f'GS_{var}_trend_sig_dots_2003_2024',
                               f'Growing-season {var} trend 2003-2024 (per decade)', self.spei_bounds, 'RdBu',
                               ('drying', 'wetting'), split_herb=True)   ## + 7-group bar (2026-11-21)
        pass


class Long_LAI_time_series(Time_series_base):
    ## long growing-season LAI records (GIMMS LAI4g 1982-2020, SNU LAI 1982-2024) (2026-11-21)
    ## data   : growing_season_extraction.py -> GS_LAI_GIMMS / GS_LAI_SNU -> Result\Dataframe\growing_season\GS_{name}.df
    ##          (MODIS growing-season window of each pixel, the same months every year)
    ## pixels : the MODIS figure pixels (model2_annual_perennial_compare: four vegetation types, NLCD cropland < 5%,
    ##          complete MODIS GS LAI) with GS LAI in every year of the long record
    ## periods: whole record (dotted), 1982-1999 (dashed), 2000-end (solid); 2000 = onset of the megadrought (Fig 1)
    ## figures:
    ##   Fig_{name}_LAI_GS          regional mean GS_LAI_rel +- 1 SD with the three trends
    ##   Fig_{name}_vs_MODIS        overlap with MODIS (2003-end): both re-centred on the overlap mean, r, trends
    ##   GS_LAI_trend_sig_dots_{period}  trend map (dots p < 0.05, inset) + 7-group bar, per period
    ##   Fig_{name}_trend_bar       7 groups x 3 periods, area-weighted mean trend with 95% CI (1 deg block bootstrap)
    ## output : Result2\02_vegetation\{name}_LAI_timeseries\
    def __init__(self, name, first_year, last_year):
        super().__init__()
        self.name = name
        self.first_year = first_year
        self.end_year = last_year
        self.dff = result_root + rf'\Dataframe\growing_season\GS_{name}{WIN}.df'   ## 2026-11-23: window_config
        self.modis_dff = result_root + rf'\Dataframe\growing_season\GS.df'
        self.outdir = self.result2w + rf'02_vegetation\{name}_LAI_timeseries\\'
        T.mk_dir(self.outdir, force=True)
        self.split_year = 2000
        self.season_list = ['growing']
        self.color_dic = {'growing': '#2E7D32'}
        self.period_list = [(first_year, last_year, ':'), (first_year, 1999, '--'), (2000, last_year, '-')]
        self.trend_period_list = self.period_list
        self.ylabel_dic = {'LAI_rel': f'{name} GS LAI (% of mean)'}
        self.veg_colors = ['#7B3F13', '#C0692A', '#EBA14A', '#F5D38E', '#CFE3D1', '#87B8CF', '#3D7FA6', '#0B4F6C']
        self.lai_bounds = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        self.herb_years = (1986, 2024)   ## 2026-11-21: annual / perennial from the whole RAP record, not 2003-2024
        pass

    def run(self):
        df = self.load()
        self.plot_time_series(df, ['LAI_rel'], self.first_year, self.outdir, f'Fig_{self.name}_LAI_GS')
        self.vs_modis(df)
        ## 2026-11-21: level change at the break year (1999-2002), regional mean GS_LAI_rel per group
        self.level_change_groups(df, 'growing_LAI_rel', np.arange(self.first_year, self.end_year + 1),
                                 [1999, 2000, 2001, 2002], self.outdir, f'Fig_{self.name}_level_change',
                                 f'{self.name} GS LAI (% of mean)')
        ## 2026-11-22: step change at 2000: mean 2000-end - mean 1982-1999 of GS_LAI_rel (% of the pixel's mean)
        self.step_change(df, 'growing_LAI_rel', np.arange(self.first_year, self.end_year + 1), 2000,
                         self.outdir + 'step_change\\', f'GS_LAI_step_2000',
                         f'{self.name} LAI: mean 2000-{self.end_year} - mean 1982-1999 (% of mean)', self.lai_bounds,
                         self.veg_colors, ('lower', 'higher'), f'{self.name} LAI step (% of mean)', stride=4,
                         dot_size=2.5)
        tab_list = []
        for t0, t1, _ in self.period_list:
            years = np.arange(t0, t1 + 1)
            info, tr, p = self.pixel_trend(df[(df['year'] >= t0) & (df['year'] <= t1)], 'growing_LAI_rel', years)
            self.trend_sig_map(info, tr, p, self.outdir + 'trend_map\\', f'GS_LAI_trend_sig_dots_{t0}_{t1}',
                               f'{self.name} growing-season LAI trend {t0}-{t1} (% per decade)', self.lai_bounds,
                               self.veg_colors, ('browning', 'greening'), stride=4, dot_size=2.5, split_herb=True)
            tab_list.append(self.group_trend(info, tr, p, f'{t0}-{t1}'))
        tab = pd.concat(tab_list)
        tab.to_csv(self.outdir + f'Fig_{self.name}_trend_bar.csv', index=False)
        print(tab.round(2).to_string())
        self.trend_bar_periods(tab)
        pass

    def load(self):
        df = T.load_df(self.dff)
        df = df[(df['year'] >= self.first_year) & (df['year'] <= self.end_year)]
        pix_now = set(T.load_df(self.herb_dff).index)   ## same pixels as the MODIS figures
        df = df[df['pix'].isin(pix_now)]
        n_ok = df['GS_LAI'].notna().groupby(df['pix']).transform('sum')
        df = df[n_ok == self.end_year - self.first_year + 1]
        df = df.rename(columns={'GS_LAI_rel': 'growing_LAI_rel', 'GS_LAI': 'growing_LAI'})
        print(self.name, '| pixels (MODIS figure pixels, all years):', df['pix'].nunique())
        return df

    def wmean(self, d, col, years):
        g = d.assign(wv=d[col] * d['area_weight'], w=d['area_weight'].where(d[col].notna()))
        g = g.groupby('year')[['wv', 'w']].sum()
        return (g['wv'] / g['w']).reindex(years).values

    def vs_modis(self, df):
        ## overlap with MODIS (2003 - end): GS LAI of the same pixels, each re-centred on its own overlap mean
        ##   (% of the pixel's overlap mean), area-weighted regional mean; r of the two series and both trends
        t0, t1 = 2003, self.end_year
        years = np.arange(t0, t1 + 1)
        m = T.load_df(self.modis_dff)
        m = m[(m['year'] >= t0) & (m['year'] <= t1) & m['pix'].isin(set(df['pix']))]
        out = {}
        for label, d, col in [('MODIS', m, 'GS_LAI'), (self.name, df[(df['year'] >= t0) & (df['year'] <= t1)],
                                                        'growing_LAI')]:
            mean = d.groupby('pix')[col].transform('mean')
            d = d.assign(rel=(d[col] / mean - 1) * 100)
            out[label] = self.wmean(d, 'rel', years)
        r, p_r = stats.pearsonr(out['MODIS'], out[self.name])
        fig, ax = plt.subplots(figsize=(self.map_width * 1.6, self.map_height * 0.9))
        text = f'r = {r:.2f} (p = {p_r:.3f})'
        for label, color in [('MODIS', '#555555'), (self.name, '#2E7D32')]:
            y = out[label]
            sl, ic, _, p, _ = stats.linregress(years, y)
            ax.plot(years, y, color=color, lw=1.5, label=label)
            ax.plot(years, sl * years + ic, color=color, lw=1, ls='--')
            text += f'\n{label}: {sl * 10:+.2f} % per decade (p = {p:.2f})'
        ax.text(0.01, 0.97, text, transform=ax.transAxes, fontsize=11, va='top')
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_ylabel('GS LAI (% of 2003-end mean)', fontsize=14.5)
        ax.legend(frameon=False, fontsize=11, loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'Fig_{self.name}_vs_MODIS.png', dpi=300)
        plt.savefig(self.outdir + f'Fig_{self.name}_vs_MODIS.pdf')
        plt.close()
        print(f'{self.name} vs MODIS {t0}-{t1}:', text.replace('\n', ' | '))

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

    def group_trend(self, info, tr, p, period):
        ## per group (7 groups): area-weighted mean trend, 95% CI, area % with sig. greening / browning
        result = []
        for g, m in self.group_masks(info).items():
            m = m & np.isfinite(tr)
            w = info['area_weight'].values[m]
            lo, hi = self.block_ci(info['lon'].values[m], info['lat'].values[m], tr[m], w)
            result.append([period, g, int(m.sum()), np.sum(tr[m] * w) / np.sum(w), lo, hi,
                           np.sum(w * ((tr[m] > 0) & (p[m] < 0.05))) / np.sum(w) * 100,
                           np.sum(w * ((tr[m] < 0) & (p[m] < 0.05))) / np.sum(w) * 100])
        return pd.DataFrame(result, columns=['period', 'class', 'n_pix', 'trend_per_decade', 'ci_low', 'ci_high',
                                             'sig_increase_area_%', 'sig_decrease_area_%'])

    def trend_bar_periods(self, tab):
        ## grouped bars: 7 groups x 3 periods (whole record, before 2000, 2000-end), 95% CI
        period_list = list(dict.fromkeys(tab['period']))
        color_list = ['#BDBDBD', '#8FB8DE', '#2E7D32']
        groups = list(dict.fromkeys(tab['class']))
        xx = np.arange(len(groups))
        wd = 0.26
        fig, ax = plt.subplots(figsize=(self.map_width * 1.8, self.map_height * 1.0))
        for k, (per, color) in enumerate(zip(period_list, color_list)):
            d = tab[tab['period'] == per].set_index('class').reindex(groups)
            v = d['trend_per_decade'].values
            err = np.array([v - d['ci_low'].values, d['ci_high'].values - v])
            ax.bar(xx + (k - 1) * wd, v, width=wd, color=color, yerr=err, capsize=1.5, error_kw={'lw': 0.6},
                   label=per)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(xx)
        ax.set_xticklabels([g.replace(' ', '\n') for g in groups], fontsize=11)
        ax.set_ylabel('LAI trend (% per decade)', fontsize=14.5)
        ax.set_title(f'{self.name} growing-season LAI trend by group', fontsize=14.5)
        ax.legend(frameon=False, fontsize=11, ncol=3)
        ax.tick_params(labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'Fig_{self.name}_trend_bar.png', dpi=300)
        plt.savefig(self.outdir + f'Fig_{self.name}_trend_bar.pdf')
        plt.close()


class GIMMS_LAI_time_series(Long_LAI_time_series):
    ## GIMMS LAI4g, 1982-2020 (2026-11-21)
    def __init__(self):
        super().__init__('GIMMS', 1982, 2020)


class SNU_LAI_time_series(Long_LAI_time_series):
    ## SNU LAI, 1982-2024 (2026-11-21; was the spring / summer version on SPEI.df)
    def __init__(self):
        super().__init__('SNU', 1982, 2024)


class Best_scale_SPEI_1982(Long_LAI_time_series):
    ## growing-season SPEI at each pixel's best scale, 1982-2024, three periods as the SNU LAI figures (2026-11-23)
    ## best scale: Best_scale_long('SNU', 1982, 2024) (bivariate.py): scale (3-48 months) with the highest r between
    ##   detrended SNU GS LAI and detrended GS SPEI -> best_scale_long_SNU_1982_2024.df
    ## SPEI : SPEI_GS_1958_2024.df (MODIS growing-season window of each pixel), column growing_SPEI{best scale}
    ## pixels: pixels of best_scale_long_SNU_1982_2024.df (= SNU LAI pixels) with SPEI in every year 1982-2024
    ## groups: RAP 1986-2024 (as the long LAI figures)
    ## figures (Result2\01_drought\SPEI_best_scale_1982_2024\):
    ##   Fig_SPEI_best_GS       regional mean +- 1 SD, trends 1982-2024 (dotted), 1982-1999 (dashed), 2000-2024 (solid)
    ##   Fig_SPEI_best_trend_bar 7 groups x 3 periods, area-weighted mean trend with 95% CI (1 deg block bootstrap)
    ##   trend_map\GS_SPEI_best_trend_sig_dots_{t0}_{t1}  per period, dots p < 0.05 + 7-group area bar
    ##   csv of each figure
    def __init__(self):
        super().__init__('SNU', 1982, 2024)
        self.name = 'SPEI_best'
        self.outdir = self.result2w + rf'01_drought\SPEI_best_scale_1982_2024\\'   ## 2026-11-23: window_config
        T.mk_dir(self.outdir, force=True)
        self.spei_dff = result_root + rf'\Dataframe\SPEI_long_term\SPEI_GS_1958_2024{WIN}.df'
        self.best_dff = self.result2w + rf'03_response\best_scale_long\best_scale_long_SNU_1982_2024.df'
        self.spei_bounds = [-1.5, -1, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 1, 1.5]   ## 2026-11-23: +-1.5, finer near 0 (was +-0.8)
        self.cbar_label_values = [-1.5, -1, -0.4, 0, 0.4, 1, 1.5]   ## 2026-11-23: labelled ticks only
        self.big_font = {'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15,
                         'ytick.labelsize': 15, 'legend.fontsize': 15}

    def run(self):
        df = self.load()
        self.plot_ts(df)
        tab_list = []
        for t0, t1, _ in self.period_list:
            years = np.arange(t0, t1 + 1)
            info, tr, p = self.pixel_trend(df[(df['year'] >= t0) & (df['year'] <= t1)], 'growing_SPEI_best', years)
            self.trend_sig_map(info, tr, p, self.outdir + 'trend_map\\', f'GS_SPEI_best_trend_sig_dots_{t0}_{t1}',
                               f'Growing-season SPEI (best scale) trend {t0}-{t1} (per decade)', self.spei_bounds,
                               'RdBu', ('drying', 'wetting'), split_herb=True)
            tab_list.append(self.group_trend(info, tr, p, f'{t0}-{t1}'))
        tab = pd.concat(tab_list)
        tab.to_csv(self.outdir + 'Fig_SPEI_best_trend_bar.csv', index=False)
        print(tab.round(2).to_string())
        self.plot_bar(tab)

    def load(self):
        best = T.load_df(self.best_dff)
        best = best[np.isfinite(best['best_scale'])]
        df = T.load_df(self.spei_dff)
        df = df[(df['year'] >= self.first_year) & (df['year'] <= self.end_year) & df['pix'].isin(set(best.index))]
        scale = df['pix'].map(best['best_scale']).astype(int).values
        val = np.full(len(df), np.nan)
        for sc in np.unique(scale):
            col = f'growing_SPEI{sc:02d}'
            if col not in df.columns:
                print('missing column', col)
                continue
            m = scale == sc
            val[m] = df[col].values[m]
        df = df.assign(growing_SPEI_best=val, best_scale=scale)
        n_ok = df['growing_SPEI_best'].notna().groupby(df['pix']).transform('sum')
        df = df[n_ok == self.end_year - self.first_year + 1]
        print('best-scale SPEI pixels:', df['pix'].nunique(), '| median best scale:',
              np.median(df.groupby('pix')['best_scale'].first()))
        return df

    def plot_ts(self, df):
        years = np.arange(self.first_year, self.end_year + 1)
        mean = self.wmean(df, 'growing_SPEI_best', years)
        d = df.assign(dev2=(df['growing_SPEI_best'] - df['year'].map(dict(zip(years, mean)))) ** 2)
        sd = np.sqrt(self.wmean(d, 'dev2', years))
        rows = []
        color = '#D84315'
        with plt.rc_context(self.big_font):
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(years, mean, color=color, lw=2)
            ax.fill_between(years, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)
            text = []
            for t0, t1, ls in self.period_list:
                m = (years >= t0) & (years <= t1)
                sl, ic, _, p, _ = stats.linregress(years[m], mean[m])
                ax.plot(years[m], sl * years[m] + ic, color=color, lw=2.5, ls=ls)
                text.append(f'{t0}-{t1}: {sl * 10:+.2f} / decade (p = {p:.2f})')
                rows.append([f'{t0}-{t1}', sl * 10, p])
            ax.text(0.01, 0.03, '\n'.join(text), transform=ax.transAxes, va='bottom', fontsize=15, color=color)
            ax.axhline(0, color='grey', lw=0.8)
            ax.axvline(2000, color='grey', lw=1, ls=':')
            ax.set_xlim(years[0], years[-1])
            ax.set_ylabel('SPEI (best scale)')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + 'Fig_SPEI_best_GS.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_SPEI_best_GS.pdf', bbox_inches='tight')
            plt.close()
        out = pd.DataFrame(rows, columns=['period', 'slope_per_decade', 'p_value'])
        out.to_csv(self.outdir + 'Fig_SPEI_best_GS_trend.csv', index=False)
        pd.DataFrame({'year': years, 'mean': mean, 'sd': sd}).to_csv(self.outdir + 'Fig_SPEI_best_GS.csv', index=False)
        print(out.round(3).to_string())

    def plot_bar(self, tab):
        period_list = list(dict.fromkeys(tab['period']))
        color_list = ['#BDBDBD', '#F4A582', '#B2182B']
        groups = list(dict.fromkeys(tab['class']))
        xx = np.arange(len(groups))
        wd = 0.26
        with plt.rc_context(self.big_font):
            fig, ax = plt.subplots(figsize=(13, 5))
            for k, (per, color) in enumerate(zip(period_list, color_list)):
                d = tab[tab['period'] == per].set_index('class').reindex(groups)
                v = d['trend_per_decade'].values
                err = np.array([v - d['ci_low'].values, d['ci_high'].values - v])
                ax.bar(xx + (k - 1) * wd, v, width=wd, color=color, yerr=err, capsize=3, label=per)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_xticks(xx)
            ax.set_xticklabels([g.replace(' ', '\n') for g in groups])
            ax.set_ylabel('SPEI trend (per decade)')
            ax.legend(frameon=False, ncol=3, loc='lower left')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + 'Fig_SPEI_best_trend_bar.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_SPEI_best_trend_bar.pdf', bbox_inches='tight')
            plt.close()


class Decoupling_quadrant(Best_scale_SPEI_1982):
    ## per pixel: sign of the SNU GS LAI trend x sign of the best-scale GS SPEI trend, 1982-1999 and 2000-2024 (2026-11-24)
    ##   drying, greening   SPEI trend < 0, LAI trend >= 0  (decoupled from drought)
    ##   drying, browning   SPEI trend < 0, LAI trend < 0   (follows drought)
    ##   wetting, greening  SPEI trend >= 0, LAI trend >= 0 (follows water)
    ##   wetting, browning  SPEI trend >= 0, LAI trend < 0  (decoupled, other direction)
    ## version 'all': every pixel by sign; 'sig': only pixels with both trends p < 0.05 keep their class, the rest grey
    ## data: SNU GS LAI (GS_SNU{WIN}.df, % of mean) and best-scale GS SPEI (Best_scale_SPEI_1982.load), same pixels;
    ##   growing-season window and output tree from window_config; groups RAP 1986-2024 (7 groups)
    ## output: {R2}\03_response\decoupling_quadrant\
    ##   Decoupling_quadrant_{all / sig}_map.png / .pdf : maps of the two periods (top), area % by group (bottom)
    ##   Decoupling_quadrant.csv : area % of each class per period, group and version
    def __init__(self, period_list=((1982, 1999), (2000, 2024))):
        super().__init__()
        self.outdir = self.result2w + rf'03_response\decoupling_quadrant\\'
        T.mk_dir(self.outdir, force=True)
        self.q_periods = list(period_list)
        self.q_names = ['drying, greening', 'drying, browning', 'wetting, greening', 'wetting, browning']
        self.q_colors = ['#4DAC26', '#A6611A', '#2C7FB8', '#D01C8B']
        self.ns_name = 'not both p < 0.05'
        self.ns_color = '#D9D9D9'

    def run(self):
        spei = self.load()
        lai = Long_LAI_time_series.load(self)
        res = {}
        rows = []
        for t0, t1 in self.q_periods:
            years = np.arange(t0, t1 + 1)
            info_l, tr_l, p_l = self.pixel_trend(lai[(lai['year'] >= t0) & (lai['year'] <= t1)], 'growing_LAI_rel',
                                                 years)
            info_s, tr_s, p_s = self.pixel_trend(spei[(spei['year'] >= t0) & (spei['year'] <= t1)],
                                                 'growing_SPEI_best', years)
            common = info_l.index.intersection(info_s.index)
            il = pd.Series(np.arange(len(info_l)), index=info_l.index).loc[common].values
            js = pd.Series(np.arange(len(info_s)), index=info_s.index).loc[common].values
            info = info_l.loc[common]
            tl, pl, ts, ps = tr_l[il], p_l[il], tr_s[js], p_s[js]
            q = np.where(ts < 0, np.where(tl >= 0, 0, 1), np.where(tl >= 0, 2, 3))
            sig = (pl < 0.05) & (ps < 0.05)
            res[(t0, t1)] = (info, q, sig)
            w = info['area_weight'].values
            for g, mg in self.group_masks(info).items():
                for ver in ['all', 'sig']:
                    row = [f'{t0}-{t1}', g, ver, int(mg.sum())]
                    for c in range(4):
                        sel = mg & (q == c) & (sig if ver == 'sig' else True)
                        row.append(np.sum(w[sel]) / np.sum(w[mg]) * 100)
                    rows.append(row)
            print(f'{t0}-{t1}: pixels {len(info)}, both trends p < 0.05: {int(sig.sum())}')
        tab = pd.DataFrame(rows, columns=['period', 'group', 'version', 'n_pix'] + self.q_names)
        tab[self.ns_name] = np.where(tab['version'] == 'sig', 100 - tab[self.q_names].sum(axis=1), 0.)
        tab.to_csv(self.outdir + 'Decoupling_quadrant.csv', index=False)
        print(tab.round(1).to_string())
        for ver in ['all', 'sig']:
            self.figure(res, tab, ver)

    def figure(self, res, tab, ver):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
        from matplotlib.colors import ListedColormap, BoundaryNorm
        colors = self.q_colors + [self.ns_color]
        names = self.q_names + ([self.ns_name] if ver == 'sig' else [])
        ## 2026-11-24: larger fonts for the slides (figure shown at ~1000 px wide)
        font = {'font.size': 26, 'axes.titlesize': 30, 'axes.labelsize': 28, 'xtick.labelsize': 24, 'ytick.labelsize': 26}
        with plt.rc_context(font):
            fig = plt.figure(figsize=(20, 16))
            for j, (t0, t1) in enumerate(self.q_periods):
                info, q, sig = res[(t0, t1)]
                k = q.astype(float)
                if ver == 'sig':
                    k[~sig] = 4
                arr = np.full((468, 449), np.nan)
                arr[np.array([p[0] for p in info.index]), np.array([p[1] for p in info.index])] = k
                ax = fig.add_axes([0.01 + j * 0.49, 0.43, 0.47, 0.52], projection=ccrs.PlateCarree())
                ax.imshow(arr, cmap=ListedColormap(colors), norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5),
                          extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(),
                          interpolation='nearest')
                ax.add_feature(cfeature.STATES, linewidth=0.3)
                ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
                ax.set_extent([-125, -102, 28, 49], crs=ccrs.PlateCarree())
                gl = ax.gridlines(draw_labels=True, linewidth=0)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlocator = mticker.FixedLocator(range(-120, -100, 10))
                gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
                gl.xlabel_style = {'size': 24}
                gl.ylabel_style = {'size': 24}
                ax.set_title(f'{t0}-{t1}', fontsize=32, fontweight='bold')
                ax2 = fig.add_axes([0.17 + j * 0.50, 0.11, 0.31, 0.27])
                t = tab[(tab['period'] == f'{t0}-{t1}') & (tab['version'] == ver)].set_index('group')
                y = np.arange(len(t))[::-1]
                left = np.zeros(len(t))
                for c, (col, name) in enumerate(zip(colors, names)):
                    v = t[name].values
                    ax2.barh(y, v, left=left, color=col, height=0.72, edgecolor='white', linewidth=0.5)
                    for yy, l0, vv in zip(y, left, v):
                        if vv >= 8:
                            ax2.text(l0 + vv / 2, yy, f'{vv:.0f}', ha='center', va='center', fontsize=22,
                                     color='black' if col == self.ns_color else 'white')
                    left += v
                ax2.set_yticks(y)
                ax2.set_yticklabels(list(t.index))
                ax2.set_xlim(0, 100)
                ax2.set_xlabel('Area (%)')
                ax2.spines[['top', 'right']].set_visible(False)
            handles = [mpatches.Patch(color=c, label=n) for c, n in zip(colors, names)]
            fig.legend(handles=handles, loc='lower center', ncol=3 if len(names) > 4 else 4, frameon=False,
                       bbox_to_anchor=(0.5, -0.04), fontsize=26, handlelength=1.2, columnspacing=1.2)
            sub = ' (both trends p < 0.05)' if ver == 'sig' else ' (sign of the trends)'
            fig.suptitle(f'SNU GS LAI trend vs best-scale GS SPEI trend{sub}', fontsize=28, y=1.04)
            plt.savefig(self.outdir + f'Decoupling_quadrant_{ver}_map.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Decoupling_quadrant_{ver}_map.pdf', bbox_inches='tight')
            plt.close()


class MODIS_LAI_time_series(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'02_vegetation\MODIS_LAI_timeseries\\'
        self.start_year = 2003
        self.var_list = ['MODIS_LAI_relative_change']
        self.ylabel_dic = {'MODIS_LAI_relative_change': 'Relative Change (%)'}
        self.trend_period_list = [(2003, 2024, '-')]   ## MODIS starts 2003: one trend only
        ## vegetation trend palette (2026-11-21): brown (browning) -> pale green-grey -> blue-teal (greening), 8 classes
        self.veg_colors = ['#7B3F13', '#C0692A', '#EBA14A', '#F5D38E', '#CFE3D1', '#87B8CF', '#3D7FA6', '#0B4F6C']
        self.LAI_threshold = 0.3   ## drop pixels whose multi-year mean LAI < 0.3
        pass



    def run(self):
        ## MODIS LAI of each pixel's main growing season (GS.df), % difference from the pixel's 2003-2024 mean
        df = self.load_GS_df()
        df = df[df['landcover_type'].isin(['Forest', 'Shrub', 'Grass', 'Savanna'])]   ## same pixels as the other figures (2026-11-20)
        print('pixels (four vegetation types):', df['pix'].nunique())
        self.ylabel_dic = {'LAI_rel': 'Relative Change (%)'}
        self.plot_time_series(df, ['LAI_rel'], 2003, self.outdir + rf'\\', 'Fig_MODIS_LAI_GS')
        ## 2026-11-21: LAI trend map, dots p < 0.05, inset: 4 classes (non-sig / sig browning, greening)
        info, tr, p = self.pixel_trend(df, 'growing_LAI_rel')
        self.trend_sig_map(info, tr, p, self.outdir, 'GS_LAI_trend_sig_dots_2003_2024',
                           'Growing-season LAI trend 2003-2024 (% per decade)', [-20, -15, -10, -5,  0,  5, 10, 15,20],
                           self.veg_colors, ('browning', 'greening'), stride=4, dot_size=2.5,
                           split_herb=True)   ## -20..20, bigger dots; + 7-group bar (2026-11-21)
        self.trend_bar(df)   ## 2026-11-20: LAI trend per vegetation type, bar plot
        pass

    def trend_bar(self, df, var='growing_LAI_rel'):
        ## per pixel OLS trend of the growing-season LAI (% of the pixel's 2003-2024 mean per decade),
        ## area-weighted mean per vegetation type with 95% CI (1 x 1 deg block bootstrap, 200 draws)
        ## output: Fig_MODIS_LAI_GS_trend_bar.png / .pdf / .csv in self.outdir
        ## 2026-11-21: 7 groups as Fig 5 (Shrub and Grass split into annual / perennial)
        color_dic = {'All': '#555555', 'Forest': '#1B5E20', 'Shrub annual': '#E0B84C', 'Shrub perennial': '#8B6508',
                     'Grass annual': '#C5E1A5', 'Grass perennial': '#558B2F', 'Savanna': '#8D6E63'}
        years = np.arange(2003, 2025)
        df = df[df['landcover_type'].isin(['Forest', 'Shrub', 'Grass', 'Savanna'])]
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
        for lc, m in self.group_masks(info).items():
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
        fig, ax = plt.subplots(figsize=(self.map_width * 1.6, self.map_height * 1.0))
        xx = np.arange(len(tab))
        err = np.array([tab['trend_per_decade'] - tab['ci_low'], tab['ci_high'] - tab['trend_per_decade']])
        ax.bar(xx, tab['trend_per_decade'], color=[color_dic[c] for c in tab['class']], width=0.65,
               yerr=err, capsize=3, error_kw={'lw': 0.8})
        for xi, v, lo, hi in zip(xx, tab['trend_per_decade'], tab['ci_low'], tab['ci_high']):
            ax.text(xi, hi + 0.2 if v >= 0 else lo - 0.2, f'{v:+.2f}', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=11)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(xx)
        ax.set_xticklabels([f'{c.replace(" ", chr(10))}\n(n={n})' for c, n in zip(tab['class'], tab['n_pix'])],
                           fontsize=11)
        ax.set_ylabel('LAI trend (% per decade)', fontsize=14.5)
        ax.set_title('Growing-season LAI trend, 2003-2024', fontsize=14.5)
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
        ## one figure per window (GS, Pant) and group; 2026-11-21: 7 groups as Fig 5
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
        groups = self.group_masks(info)
        for season in ['GS', 'Pant']:
            self.season_list = [season]
            for g, mg in groups.items():
                d = df[df['pix'].isin(set(info.index[mg]))]
                self.plot_time_series(d, self.main_list, self.start_year, self.outdir + rf'time_series\\',
                                      f'Fig_Daymet_{season}_{g.replace(" ", "_")}')
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
        ## 2026-11-21: same pixels as the SPEI short-term / attribution figures: four vegetation types, NLCD cropland
        ##   < 5%, complete GS LAI (= pixels of model2_annual_perennial_compare)
        pix_now = set(T.load_df(self.herb_dff).index)
        df = df[df['pix'].isin(pix_now)]
        print('pixels:', df['pix'].nunique())
        return df

    def nice_bounds(self, tr):
        ## symmetric discrete colour bounds: 90th percentile of |trend| rounded up to 1, 2, 2.5 or 5 x 10^k (2026-11-21)
        v = np.nanpercentile(np.abs(tr), 90)
        k = 10 ** np.floor(np.log10(v))
        vlim = k * min([s for s in [1, 2, 2.5, 5, 10] if s * k >= v])
        return [float(b) for b in np.round(vlim * np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]), 6)]

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
                cmap = 'RdBu_r' if var == 'CDD' else 'RdBu'   ## blue = wetter
                arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, tr)))
                arr[arr <= -9999] = np.nan
                D.arr_to_tif(arr.copy(), outdir + f'{col}_trend_2003_2024.tif')
                ## 2026-11-21: paper style as SPEI / LAI: discrete colours, dots p < 0.05, inset of 4 classes,
                ##   + 7-group bar ({fname}_by_group) and _area.csv
                ok = np.isfinite(tr) & np.isfinite(p)
                self.trend_sig_map(info[ok], tr[ok], p[ok], outdir, f'{col}_trend_sig_dots_2003_2024',
                                   f'{col} trend 2003-2024 ({unit})', self.nice_bounds(tr), cmap,
                                   ('decrease', 'increase'), split_herb=True)
                for lc, m in self.group_masks(info).items():   ## 7 groups (2026-11-21)
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
        print(df_t[df_t['class'].isin(['All', 'Grass perennial'])].round(2).to_string())
        self.trend_bar_groups(df_t)

    def trend_bar_groups(self, df_t):
        ## 2026-11-21: area-weighted mean trend per group (7 groups as Fig 5) with 95% CI, one panel per variable;
        ## one figure per window (GS, Pant) and variable list; bars faded when the CI includes 0
        ## output: Fig_Daymet_{window}_trend_bar_{main / other}.png / .pdf in self.outdir
        color_dic = {'All': '#555555', 'Forest': '#1B5E20', 'Shrub annual': '#E0B84C', 'Shrub perennial': '#8B6508',
                     'Grass annual': '#C5E1A5', 'Grass perennial': '#558B2F', 'Savanna': '#8D6E63'}
        for season in ['GS', 'Pant']:
            for tag, var_list in [('main', self.main_list), ('other', self.other_list)]:
                var_list = [v for v in var_list if ((df_t['window'] == season) & (df_t['variable'] == v)).any()]
                if len(var_list) == 0:
                    continue
                fig, axs = plt.subplots(len(var_list), 1, sharex=True, squeeze=False,
                                        figsize=(self.map_width * 1.7, self.map_height * 0.6 * len(var_list)))
                axs = axs[:, 0]
                for ax, var in zip(axs, var_list):
                    d = df_t[(df_t['window'] == season) & (df_t['variable'] == var)].reset_index(drop=True)
                    xx = np.arange(len(d))
                    v, lo, hi = d['trend'].values, d['ci_low'].values, d['ci_high'].values
                    sig = (lo > 0) | (hi < 0)
                    ax.bar(xx, v, color=[color_dic.get(c, '#999999') for c in d['class']], width=0.65,
                           alpha=1, edgecolor='none')
                    ax.bar(xx[~sig], v[~sig], color='white', width=0.65, alpha=0.55, edgecolor='none')   ## fade n.s.
                    ax.errorbar(xx, v, yerr=[v - lo, hi - v], fmt='none', ecolor='k', elinewidth=0.6, capsize=2)
                    for xi, vi, l0, h0 in zip(xx, v, lo, hi):
                        ax.text(xi, h0 if vi >= 0 else l0, f'{vi:+.2f}', ha='center', va='bottom' if vi >= 0 else 'top',
                                fontsize=9)
                    ax.axhline(0, color='k', lw=0.5)
                    unit = d['unit'].iloc[0]
                    ax.set_ylabel(f'{var}\n({unit})', fontsize=11)
                    ax.tick_params(labelsize=10.5)
                    ax.spines[['top', 'right']].set_visible(False)
                    ymin, ymax = min(0, np.nanmin(lo)), max(0, np.nanmax(hi))
                    pad = (ymax - ymin) * 0.25 + 1e-9
                    ax.set_ylim(ymin - pad, ymax + pad)
                axs[-1].set_xticks(xx)
                axs[-1].set_xticklabels([c.replace(' ', '\n') for c in d['class']], fontsize=10.5)
                axs[0].set_title(f'Daymet {season} trend 2003-2024 by group (faded: 95% CI includes 0)', fontsize=13)
                plt.tight_layout()
                plt.savefig(self.outdir + f'Fig_Daymet_{season}_trend_bar_{tag}.png', dpi=300, bbox_inches='tight')
                plt.savefig(self.outdir + f'Fig_Daymet_{season}_trend_bar_{tag}.pdf', bbox_inches='tight')
                plt.close()


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
        self.outdir = self.result2w + rf'01_drought\SPEI_long_term\\'   ## 2026-11-23: window_config
        T.mk_dir(self.outdir, force=True)
        self.start_year = 1958
        self.split_year = 2003   ## start of the recent period (MODIS)
        self.year_list = list(range(1958, 2025))
        self.scale_list = list(range(3, 49, 3))
        self.map_scale_list = [3, 6, 9, 12, 24, 36, 48]
        self.main_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI48']   ## main figure, same as SPEI_time_series
        self.sig_bounds = [-0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6]   ## SPEI per decade, dots maps (2026-11-21)
        self.class_list = ['All', 'Forest', 'Shrub', 'Grass', 'Savanna']
        self.season_list = ['growing']
        self.color_dic = {'growing': '#D9480F'}   ## orange-red
        self.trend_period_list = [(1958, 2024, ':'), (2003, 2024, '-')]
        self.gs_dff = result_root + rf'\Dataframe\growing_season\{GS_TABLE}'   ## 2026-11-23: window_config
        self.dff_out = result_root + rf'\Dataframe\SPEI_long_term\SPEI_GS_1958_2024{WIN}.df'  ## pixel x year growing-season SPEI
        pass

    def run(self):
        if not os.path.isfile(self.dff_out):
            self.build_df()
        df = T.load_df(self.dff_out)
        ## same pixels as the LAI analysis: the four vegetation types only ('Other' IGBP classes left out)
        df = df[df['landcover_type'].isin(self.class_list[1:])]
        ## 2026-11-21: keep only the pixels of the current growing_season.df (NLCD cropland < 5% mask); the cached
        ##   SPEI_GS_1958_2024.df was built before the mask
        pix_now = set(T.load_df(self.gs_dff).index)
        df = df[df['pix'].isin(pix_now)]
        print('pixels (four vegetation types):', df['pix'].nunique())
        # self.time_series(df)
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
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
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
                ## 2026-11-21: trend maps in the paper style: discrete colours, dots = p < 0.05, area panel with
                ##   non-sig / sig drying and non-sig / sig wetting (+ _area.csv per vegetation type)
                if p is not None:
                    ok = np.isfinite(vals) & np.isfinite(p)
                    self.trend_sig_map(info[ok], vals[ok], p[ok], outdir, f'GS_{var}_{key}_sig_dots',
                                       f'Growing-season {var} {label}', self.sig_bounds, 'RdBu', ('drying', 'wetting'))
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


class SPEI_1982_plot(SPEI_long_term_plot):
    ## growing-season SPEI 1982-2024, the period of the long LAI records (GIMMS, SNU) (2026-11-21)
    ## data   : SPEI_GS_1958_2024.df of SPEI_long_term_plot (MODIS growing-season window of each pixel)
    ## pixels : the MODIS figure pixels (model2_annual_perennial_compare), as the GIMMS / SNU LAI figures
    ## periods: 1982-2024 (dotted), 1982-1999 (dashed), 2000-2024 (solid), as the long LAI figures
    ## time series: SPEI03, 06, 12, 24, area-weighted mean +- 1 SD
    ## maps   : SPEI03 and SPEI12 trend per period, dots p < 0.05, inset of 4 classes, + 7-group bar (split_herb;
    ##          the groups are the 2003-2024 RAP groups, as in the long LAI figures)
    ## output : Result2\01_drought\SPEI_1982_2024\
    def __init__(self):
        super().__init__()
        self.outdir = self.result2w + rf'01_drought\SPEI_1982_2024\\'   ## 2026-11-23: window_config
        T.mk_dir(self.outdir, force=True)
        self.start_year = 1982
        self.split_year = 2000
        self.ts_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI36']   ## 2026-11-23: main-line scales
        self.map_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI36']   ## 2026-11-23: all five mapped
        self.period_list = [(1982, 2024, ':'), (1982, 1999, '--'), (2000, 2024, '-')]
        self.trend_period_list = self.period_list
        self.herb_years = (1986, 2024)   ## 2026-11-21: annual / perennial from the whole RAP record, as the long LAI
        self.sig_bounds = [-1.5, -1, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 1, 1.5]   ## 2026-11-23: +-1.5 as the best-scale maps (was +-0.6)
        self.cbar_label_values = [-1.5, -1, -0.4, 0, 0.4, 1, 1.5]   ## 2026-11-23: labelled ticks only

    def run(self):
        if not os.path.isfile(self.dff_out):
            self.build_df()
        df = T.load_df(self.dff_out)
        df = df[(df['year'] >= 1982) & (df['year'] <= 2024)]
        pix_now = set(T.load_df(self.herb_dff).index)   ## same pixels as the MODIS / GIMMS / SNU LAI figures
        df = df[df['pix'].isin(pix_now)]
        print('pixels:', df['pix'].nunique())
        self.plot_time_series(df, self.ts_list, 1982, self.outdir, 'Fig_SPEI_1982_2024')
        ## 2026-11-21: level change of SPEI at the break year (1999-2002), regional mean per group
        for var in ['SPEI03', 'SPEI12']:
            self.level_change_groups(df, f'growing_{var}', np.arange(1982, 2025), [1999, 2000, 2001, 2002],
                                     self.outdir, f'Fig_{var}_level_change', f'Growing-season {var}')
        for var in self.map_list:
            for t0, t1, _ in self.period_list:
                years = np.arange(t0, t1 + 1)
                info, tr, p = self.pixel_trend(df[(df['year'] >= t0) & (df['year'] <= t1)], f'growing_{var}', years)
                self.trend_sig_map(info, tr, p, self.outdir + 'trend_map\\', f'GS_{var}_trend_sig_dots_{t0}_{t1}',
                                   f'Growing-season {var} trend {t0}-{t1} (per decade)', self.sig_bounds, 'RdBu',
                                   ('drying', 'wetting'), split_herb=True)
        ## 2026-11-22: step change at 2000: mean 2000-2024 - mean 1982-1999, Welch t, map + 7-group means
        for var in self.map_list:
            self.step_change(df, f'growing_{var}', np.arange(1982, 2025), 2000, self.outdir + 'step_change\\',
                             f'GS_{var}_step_2000', f'{var}: mean 2000-2024 - mean 1982-1999',
                             [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], 'RdBu', ('drier', 'wetter'),
                             f'{var} step (2000-2024 - 1982-1999)')
        pass


class LAI_same_period_compare(Time_series_base):
    ## GIMMS vs SNU over the same years 2000-2020 (2026-11-21)
    ## question: do the two long records agree after 2000 (forest, annual shrub, level change at 2000)?
    ## data  : GS_GIMMS.df, GS_SNU.df: GS_LAI in the MODIS growing-season window of each pixel
    ## pixels: the MODIS figure pixels (model2_annual_perennial_compare) with GS LAI in every year 2000-2020 in both
    ##         records (same pixels for the two)
    ## trend : per pixel OLS of GS LAI in % of the pixel's own 2000-2020 mean, per decade
    ## groups: 7 groups, Shrub / Grass split by RAP 1986-2024 (as the long LAI figures)
    ## table : per group and record: area-weighted mean trend, 95% CI (1 deg block bootstrap), area % sig. greening /
    ##         browning; r of the pixel trends between GIMMS and SNU (spatial agreement)
    ## output: Result2\02_vegetation\LAI_same_period_compare\ : LAI_same_period_2000_2020.csv / _bar.png / _r.csv
    def __init__(self, y0=2000, y1=2020):   ## 2026-11-22: period as parameters
        super().__init__()
        self.outdir = self.result2w + rf'02_vegetation\LAI_same_period_compare\\'   ## 2026-11-23: window_config
        T.mk_dir(self.outdir, force=True)
        self.years = np.arange(y0, y1 + 1)
        self.tag = f'{self.years[0]}_{self.years[-1]}'
        self.data_dic = {'GIMMS': result_root + rf'\Dataframe\growing_season\GS_GIMMS{WIN}.df',
                         'SNU': result_root + rf'\Dataframe\growing_season\GS_SNU{WIN}.df'}
        self.color_dic = {'GIMMS': '#8FB8DE', 'SNU': '#2E7D32'}
        self.herb_years = (1986, 2024)   ## groups as the long LAI figures

    def run(self):
        pix_now = set(T.load_df(self.herb_dff).index)
        x = self.years.astype(float) - self.years.mean()
        tr_dic, p_dic, info = {}, {}, None
        for name, f in self.data_dic.items():
            df = T.load_df(f)
            df = df[df['pix'].isin(pix_now) & (df['year'] >= self.years[0]) & (df['year'] <= self.years[-1])]
            mat = df.pivot_table(index='pix', columns='year', values='GS_LAI').reindex(columns=self.years)
            mat = mat[mat.notna().all(axis=1) & (mat.mean(axis=1) > 0)]
            y = (mat.values / mat.values.mean(axis=1, keepdims=True) - 1) * 100
            ym = y - y.mean(axis=1, keepdims=True)
            slope = ym @ x / np.sum(x ** 2)
            resid = ym - slope[:, None] * x[None, :]
            se = np.sqrt(np.sum(resid ** 2, axis=1) / (len(x) - 2) / np.sum(x ** 2))
            tr_dic[name] = pd.Series(slope * 10, index=mat.index)
            p_dic[name] = pd.Series(2 * stats.t.sf(np.abs(slope / se), len(x) - 2), index=mat.index)
            if info is None:
                info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
            print(name, '| pixels with all years:', len(mat))
        common = set.intersection(*[set(s.index) for s in tr_dic.values()])
        info = info[np.array([pix in common for pix in info.index])]
        print('pixels in all three records:', len(info))
        tr = {n: s.reindex(info.index).values for n, s in tr_dic.items()}
        pv = {n: s.reindex(info.index).values for n, s in p_dic.items()}
        w_all = info['area_weight'].values
        rows, r_rows = [], []
        for g, m in self.group_masks(info).items():
            w = w_all[m]
            for name in self.data_dic:
                v = tr[name][m]
                lo, hi = self.block_ci(info['lon'].values[m], info['lat'].values[m], v, w)
                rows.append([g, name, int(m.sum()), np.sum(v * w) / np.sum(w), lo, hi,
                             np.sum(w * ((v > 0) & (pv[name][m] < 0.05))) / np.sum(w) * 100,
                             np.sum(w * ((v < 0) & (pv[name][m] < 0.05))) / np.sum(w) * 100])
            r_rows.append([g, int(m.sum()), np.corrcoef(tr['GIMMS'][m], tr['SNU'][m])[0, 1]])
        tab = pd.DataFrame(rows, columns=['class', 'record', 'n_pix', 'trend_per_decade', 'ci_low', 'ci_high',
                                          'sig_increase_area_%', 'sig_decrease_area_%'])
        tab.to_csv(self.outdir + f'LAI_same_period_{self.tag}.csv', index=False)
        r_tab = pd.DataFrame(r_rows, columns=['class', 'n_pix', 'r_GIMMS_SNU'])
        r_tab.to_csv(self.outdir + f'LAI_same_period_{self.tag}_r.csv', index=False)
        print(tab.pivot(index='class', columns='record', values='trend_per_decade').round(2).to_string())
        print(r_tab.round(2).to_string())
        self.bar(tab)
        self.agreement_map(info, tr['GIMMS'], tr['SNU'])   ## 2026-11-21: sign agreement map
        self.bivariate_4x4(info, tr['GIMMS'], pv['GIMMS'], tr['SNU'], pv['SNU'])   ## 2026-11-22: 4 x 4 classes
        pass

    def agreement_map(self, info, tr_a, tr_b, name_a='GIMMS', name_b='SNU'):
        ## sign agreement of the pixel LAI trends of two records (2026-11-21):
        ##   ++ both greening, +- only name_a greening, -+ only name_b greening, -- both browning
        ## left: map; right: area % of the four classes per group (7 groups); legend below
        ## output: LAI_sign_agreement_{tag}_map.png / .pdf / .csv
        with plt.rc_context({'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15}):   ## large fonts (2026-11-21)
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import matplotlib.patches as mpatches
            import matplotlib.ticker as mticker
            from matplotlib.colors import ListedColormap, BoundaryNorm
            labels = [f'++ both greening', f'+- {name_a} greening, {name_b} browning',
                      f'-+ {name_a} browning, {name_b} greening', '-- both browning']
            colors = ['#3D7FA6', '#B39DDB', '#A5D6A7', '#C0692A']
            k = np.where(tr_a >= 0, np.where(tr_b >= 0, 1, 2), np.where(tr_b >= 0, 3, 4)).astype(float)
            k[~(np.isfinite(tr_a) & np.isfinite(tr_b))] = np.nan
            ny, nx = 468, 449
            arr = np.full((ny, nx), np.nan)
            rows = np.array([q[0] for q in info.index])
            cols = np.array([q[1] for q in info.index])
            arr[rows, cols] = k
            w = info['area_weight'].values
            groups = self.group_masks(info)
            share = np.array([[np.sum(w[mg & (k == c)]) / np.sum(w[mg & np.isfinite(k)]) * 100 for c in [1, 2, 3, 4]]
                              for mg in groups.values()])
            pd.DataFrame(share, index=list(groups), columns=['++', '+-', '-+', '--']).round(1).to_csv(
                self.outdir + f'LAI_sign_agreement_{self.tag}.csv')
            print(pd.DataFrame(share, index=list(groups), columns=['++', '+-', '-+', '--']).round(1).to_string())
            fig = plt.figure(figsize=(12, 5.6))
            ax = fig.add_axes([0.02, 0.16, 0.50, 0.78], projection=ccrs.PlateCarree())
            ax.imshow(arr, cmap=ListedColormap(colors), norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4),
                      extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(), interpolation='nearest')
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
            ax.set_extent([-125, -102, 28, 49], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.FixedLocator(range(-120, -100, 10))
            gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
            gl.xlabel_style = {'size': 15}
            gl.ylabel_style = {'size': 15}
            ax.set_title(f'LAI trend sign {self.years[0]}-{self.years[-1]}: {name_a} / {name_b}')
            ax2 = fig.add_axes([0.66, 0.22, 0.31, 0.66])
            y = np.arange(len(groups))[::-1]
            left = np.zeros(len(groups))
            for c in range(4):
                ax2.barh(y, share[:, c], left=left, color=colors[c], height=0.7, edgecolor='white', linewidth=0.5)
                for yy, l0, v in zip(y, left, share[:, c]):
                    if v >= 8:
                        ax2.text(l0 + v / 2, yy, f'{v:.0f}', ha='center', va='center', fontsize=14,
                                 color='white' if c in (0, 3) else 'black')
                left += share[:, c]
            ax2.set_yticks(y)
            ax2.set_yticklabels(list(groups))
            ax2.set_xlim(0, 100)
            ax2.set_xlabel('Area (%)')
            ax2.spines[['top', 'right']].set_visible(False)
            handles = [mpatches.Patch(color=c, label=l) for l, c in zip(labels, colors)]
            fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
            plt.savefig(self.outdir + f'LAI_sign_agreement_{self.tag}_map.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'LAI_sign_agreement_{self.tag}_map.pdf', bbox_inches='tight')
            plt.close()

    def bivariate_4x4(self, info, tr_a, p_a, tr_b, p_b, name_a='GIMMS', name_b='SNU'):
        ## 2026-11-22: each record in 4 classes (sig. browning, n.s. browning, n.s. greening, sig. greening; p < 0.05)
        ##   -> 4 x 4 bivariate map; colour: hue from the mean class of the two records (brown -> blue), greyed with
        ##   their disagreement; right: area % matrix (rows name_a, columns name_b) = agreement table
        ## output: LAI_bivariate4x4_{tag}_map.png / .pdf, LAI_bivariate4x4_{tag}.csv (All) and _by_group.csv
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
        names = ['sig. browning', 'n.s. browning', 'n.s. greening', 'sig. greening']
        score = [-2, -1, 1, 2]

        def cls(tr, p):
            sig = p < 0.05
            return np.where(tr < 0, np.where(sig, 0, 1), np.where(sig, 3, 2))

        ca, cb = cls(tr_a, p_a), cls(tr_b, p_b)
        k = ca * 4 + cb   ## 0 ... 15
        base = {-2: '#7B3F13', -1.5: '#C0692A', -1: '#EBA14A', -0.5: '#F5D38E', 0: '#BDBDBD', 0.5: '#CFE3D1',
                1: '#87B8CF', 1.5: '#3D7FA6', 2: '#0B4F6C'}
        colors = []
        for i in range(4):
            for j in range(4):
                m = (score[i] + score[j]) / 2
                d = abs(score[i] - score[j]) / 4   ## 0 (same class) ... 1 (opposite significant)
                c = np.array(to_rgb(base[m]))
                colors.append(tuple(c * (1 - 0.6 * d) + np.array(to_rgb('#BDBDBD')) * 0.6 * d))
        w = info['area_weight'].values
        mat = np.array([[np.sum(w[(ca == i) & (cb == j)]) for j in range(4)] for i in range(4)]) / np.sum(w) * 100
        pd.DataFrame(mat, index=[f'{name_a} {n}' for n in names], columns=[f'{name_b} {n}' for n in names]).round(
            1).to_csv(self.outdir + f'LAI_bivariate4x4_{self.tag}.csv')
        rows = []
        for g, mg in self.group_masks(info).items():
            wg = w[mg]
            rows.append([g, int(mg.sum()),
                         np.sum(wg[ca[mg] == cb[mg]]) / np.sum(wg) * 100,
                         np.sum(wg[(ca[mg] == 3) & (cb[mg] == 3)]) / np.sum(wg) * 100,
                         np.sum(wg[(ca[mg] == 0) & (cb[mg] == 0)]) / np.sum(wg) * 100,
                         np.sum(wg[((ca[mg] == 3) & (cb[mg] <= 1)) | ((ca[mg] == 0) & (cb[mg] >= 2))]) / np.sum(wg) * 100,
                         np.sum(wg[((cb[mg] == 3) & (ca[mg] <= 1)) | ((cb[mg] == 0) & (ca[mg] >= 2))]) / np.sum(wg) * 100])
        gtab = pd.DataFrame(rows, columns=['class', 'n_pix', 'same_class_%', 'both_sig_greening_%',
                                           'both_sig_browning_%', f'{name_a}_sig_opposite_to_{name_b}_%',
                                           f'{name_b}_sig_opposite_to_{name_a}_%'])
        gtab.to_csv(self.outdir + f'LAI_bivariate4x4_{self.tag}_by_group.csv', index=False)
        print(pd.DataFrame(mat, index=names, columns=names).round(1).to_string())
        print(gtab.round(1).to_string())
        rows_i = np.array([q[0] for q in info.index])
        cols_i = np.array([q[1] for q in info.index])
        arr = np.full((468, 449), np.nan)
        arr[rows_i, cols_i] = k
        with plt.rc_context({'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                             'ytick.labelsize': 13}):
            fig = plt.figure(figsize=(14, 6.2))
            ax = fig.add_axes([0.0, 0.06, 0.55, 0.86], projection=ccrs.PlateCarree())
            ax.imshow(arr, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(-0.5, 16.5, 1), 16),
                      extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(),
                      interpolation='nearest')
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
            ax.set_extent([-125, -102, 30, 46], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.FixedLocator(range(-120, -100, 10))
            gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
            gl.xlabel_style = {'size': 13}
            gl.ylabel_style = {'size': 13}
            ax.set_title(f'LAI trend {self.years[0]}-{self.years[-1]}: {name_a} x {name_b}')
            ## legend = area matrix, cells coloured as on the map
            ax2 = fig.add_axes([0.66, 0.20, 0.30, 0.62])
            for i in range(4):
                for j in range(4):
                    ax2.add_patch(plt.Rectangle((j, 3 - i), 1, 1, color=colors[i * 4 + j], ec='white', lw=1.5))
                    ax2.text(j + 0.5, 3 - i + 0.5, f'{mat[i, j]:.0f}', ha='center', va='center', fontsize=14,
                             color='white' if (i * 4 + j) in (0, 15) else 'black')
            ax2.set_xlim(0, 4)
            ax2.set_ylim(0, 4)
            short = ['sig. -', 'n.s. -', 'n.s. +', 'sig. +']
            ax2.set_xticks(np.arange(4) + 0.5)
            ax2.set_xticklabels(short)
            ax2.set_yticks(np.arange(4) + 0.5)
            ax2.set_yticklabels(short[::-1])
            ax2.set_xlabel(f'{name_b} LAI trend')
            ax2.set_ylabel(f'{name_a} LAI trend')
            ax2.set_title('Area (%)')
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.tick_params(length=0)
            plt.savefig(self.outdir + f'LAI_bivariate4x4_{self.tag}_map.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'LAI_bivariate4x4_{self.tag}_map.pdf', bbox_inches='tight')
            plt.close()

    def ts_overlay(self):
        ## 2026-11-23: robustness of the time series: SNU vs GIMMS over self.years (1982-2020), same pixels
        ## per record: GS LAI in % of each pixel's own mean over self.years, area-weighted mean per group (7 groups)
        ## each panel: both series, their OLS trends (% per decade, p) and r of the detrended series
        ## output: LAI_ts_overlay_{tag}.png / .pdf / .csv (yearly series)
        pix_now = set(T.load_df(self.herb_dff).index)
        mats, info = {}, None
        for name, f in self.data_dic.items():
            df = T.load_df(f)
            df = df[df['pix'].isin(pix_now) & (df['year'] >= self.years[0]) & (df['year'] <= self.years[-1])]
            mat = df.pivot_table(index='pix', columns='year', values='GS_LAI').reindex(columns=self.years)
            mats[name] = mat[mat.notna().all(axis=1) & (mat.mean(axis=1) > 0)]
            if info is None:
                info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
        common = set.intersection(*[set(m.index) for m in mats.values()])
        info = info[np.array([pix in common for pix in info.index])]
        rel = {}
        for name, mat in mats.items():
            y = mat.reindex(info.index).values
            rel[name] = (y / y.mean(axis=1, keepdims=True) - 1) * 100
        w = info['area_weight'].values
        groups = self.group_masks(info)
        out = {}
        for g, m in groups.items():
            for name in self.data_dic:
                out[f'{g} | {name}'] = np.sum(rel[name][m] * w[m, None], axis=0) / np.sum(w[m])
        pd.DataFrame(out, index=self.years).to_csv(self.outdir + f'LAI_ts_overlay_{self.tag}.csv')
        x = self.years.astype(float)
        with plt.rc_context({'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 14, 'xtick.labelsize': 12,
                             'ytick.labelsize': 12, 'legend.fontsize': 13}):
            fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
            for ax, g in zip(axs.ravel(), groups):
                text = []
                det = {}
                for name in self.data_dic:
                    y = out[f'{g} | {name}']
                    sl, ic, _, p, _ = stats.linregress(x, y)
                    det[name] = y - (sl * x + ic)
                    ax.plot(self.years, y, color=self.color_dic[name], lw=2, label=name)
                    ax.plot(self.years, sl * x + ic, color=self.color_dic[name], lw=1.2, ls='--')
                    text.append(f'{name} {sl * 10:+.1f} (p={p:.2f})')
                names = list(self.data_dic)
                r = np.corrcoef(det[names[0]], det[names[1]])[0, 1]
                text.append(f'r detrended = {r:.2f}')
                ax.text(0.02, 0.98, '\n'.join(text), transform=ax.transAxes, va='top', fontsize=11)
                ax.axhline(0, color='grey', lw=0.5)
                ax.set_title(f'{g} (n={int(groups[g].sum())})')
                ax.spines[['top', 'right']].set_visible(False)
            for ax in axs[:, 0]:
                ax.set_ylabel('GS LAI (% of mean)')
            axs[0, 0].legend(frameon=False, loc='lower right')
            for ax in axs.ravel()[len(groups):]:
                ax.set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + f'LAI_ts_overlay_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'LAI_ts_overlay_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def bar_two_periods(self, tag_list=('1982_2020', '2000_2020')):
        ## 2026-11-22: 7-group trend with 95% CI, GIMMS and SNU, one panel per period (reads the csv of run())
        ## output: LAI_same_period_two_periods_bar.png / .pdf
        with plt.rc_context({'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 16, 'xtick.labelsize': 13,
                             'ytick.labelsize': 14, 'legend.fontsize': 14}):
            fig, axs = plt.subplots(len(tag_list), 1, figsize=(11, 4.0 * len(tag_list)), sharex=True)
            for ax, tag in zip(np.atleast_1d(axs), tag_list):
                tab = pd.read_csv(self.outdir + f'LAI_same_period_{tag}.csv')
                groups = list(dict.fromkeys(tab['class']))
                xx = np.arange(len(groups))
                for k, name in enumerate(self.data_dic):
                    d = tab[tab['record'] == name].set_index('class').reindex(groups)
                    v = d['trend_per_decade'].values
                    err = np.array([v - d['ci_low'].values, d['ci_high'].values - v])
                    ax.bar(xx + (k - (len(self.data_dic) - 1) / 2) * 0.36, v, width=0.36, color=self.color_dic[name],
                           yerr=err, capsize=3, error_kw={'lw': 1}, label=name)
                ax.axhline(0, color='k', lw=0.6)
                ax.set_title(tag.replace('_', '-'))
                ax.set_ylabel('LAI trend (% / decade)')
                ax.spines[['top', 'right']].set_visible(False)
            ax.set_xticks(xx)
            ax.set_xticklabels([g.replace(' ', '\n') for g in groups])
            np.atleast_1d(axs)[0].legend(frameon=False, ncol=2)
            plt.tight_layout()
            plt.savefig(self.outdir + 'LAI_same_period_two_periods_bar.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'LAI_same_period_two_periods_bar.pdf', bbox_inches='tight')
            plt.close()

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

    def bar(self, tab):
        ## grouped bars: 7 groups x records, 95% CI
        with plt.rc_context({'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15}):   ## large fonts (2026-11-21)
            groups = list(dict.fromkeys(tab['class']))
            xx = np.arange(len(groups))
            wd = 0.36
            fig, ax = plt.subplots(figsize=(11, 4.4))
            for k, name in enumerate(self.data_dic):
                d = tab[tab['record'] == name].set_index('class').reindex(groups)
                v = d['trend_per_decade'].values
                err = np.array([v - d['ci_low'].values, d['ci_high'].values - v])
                ax.bar(xx + (k - (len(self.data_dic) - 1) / 2) * wd, v, width=wd, color=self.color_dic[name], yerr=err,
                       capsize=2, error_kw={'lw': 0.8}, label=name)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks(xx)
            ax.set_xticklabels([g.replace(' ', '\n') for g in groups])
            ax.set_ylabel('LAI trend (% per decade)')
            ax.set_title(f'Growing-season LAI trend {self.years[0]}-{self.years[-1]}, same pixels')
            ax.legend(frameon=False, ncol=3)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + f'LAI_same_period_{self.tag}_bar.png', dpi=300)
            plt.savefig(self.outdir + f'LAI_same_period_{self.tag}_bar.pdf')
            plt.close()


## add a new variable: copy SNU_LAI_time_series, change outdir / start_year / var_list




def main():

    # SNU_LAI_time_series().run()
    # Daymet_time_series().run()      ## 2026-11-15: rainfall structure in the growing season (Daymet daily)
    # MODIS_LAI_time_series().run()

    # SPEI_time_series_MODIS().run()     ## modis record
    # MODIS_LAI_time_series().run()   ## growing-season MODIS LAI (GS.df); 2026-11-20: + trend bar per vegetation type (done, NLCD mask)
    # SPEI_long_term_plot().run()     ## growing-season SPEI 1958-2024: time series and trend maps (+ dots maps 2026-11-21) (done)
    # SPEI_time_series_MODIS().run()     ## 2026-11-21: SPEI03-12 2003-2024 time series + trend maps with dots (done)
    # SPEI_time_series_MODIS().run()     ## 2026-11-21: + 7-group bar (Shrub / Grass annual, perennial) (done)
    # MODIS_LAI_time_series().run()      ## + LAI trend map with dots (2026-11-21); + 7-group bars (done)
    # Daymet_time_series().run()         ## 2026-11-21: same pixels, 7 groups, dots maps (done)
    ## 2026-11-21: long records with annual / perennial from RAP 1986-2024
    # GIMMS_LAI_time_series().run()      ## GS LAI 1982-2020 (needs GS_GIMMS.df) (done)
    # SNU_LAI_time_series().run()        ## GS LAI 1982-2024 (needs GS_SNU.df) (done)
    # SPEI_1982_plot().run()             ## GS SPEI 1982-2024, three periods, dots maps (done)
    ## 2026-11-22: main line SNU 1982-2024, steps 1-3 (GIMMS = robustness)
    # SPEI_1982_plot().run()                        ## 1: SPEI03-12 time series, trend maps, step change at 2000 (done)
    # SNU_LAI_time_series().run()                   ## 2: SNU LAI trend maps / bars, step change at 2000 (done)
    # GIMMS_LAI_time_series().run()                 ## 3: GIMMS, same figures (robustness) (done)
    ## 2026-11-23: main line SNU LAI + SPEI03, 06, 12, 24, 36 (1982-2024); GIMMS = robustness
    # SPEI_1982_plot().run()                               ## SPEI03, 06, 12, 24, 36: time series, maps, step (done)
    # LAI_same_period_compare(1982, 2020).ts_overlay()     ## robustness 1: SNU vs GIMMS time series 1982-2020 (done)
    # LAI_same_period_compare(2000, 2020).run()            ## robustness 2: spatial consistency 2000-2020, 4 x 4 map (done)
    ## 2026-11-23: SPEI trend maps with the +-1.5 colour bar (Fig 1 and Fig 4 of the slides); window from window_config;
    ##   2026-11-23: colour-bar labels at -1.5, -1, -0.4, 0, 0.4, 1, 1.5 only (the numbers overlapped)
    # Best_scale_SPEI_1982().run()   ## best-scale SPEI (SNU), three periods (done)
    # SPEI_1982_plot().run()         ## SPEI03-36 time series, trend maps, step change (done)
    ## 2026-11-24: drying / wetting x greening / browning per pixel, 1982-1999 and 2000-2024; + version with both p < 0.05
    Decoupling_quadrant().run()
    # LAI_same_period_compare(1982, 2020).run()            ## (done 2026-11-22)
    # LAI_same_period_compare(1982, 1999).run()            ## (done 2026-11-22)
    # LAI_same_period_compare().bar_two_periods(('1982_2020', '2000_2020'))   ## (done in-session 2026-11-22)


if __name__ == '__main__':
    main()
