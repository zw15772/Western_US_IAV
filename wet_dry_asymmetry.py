# coding='utf-8'
## Why does LAI keep rising after 2000 although it is drier? (2026-11-26)
## period : 2000-2024 only (before 2000 it was overall wet and vegetation was not water-limited)
## data   : SNU GS LAI (growing_LAI_rel, % of the pixel's mean; Long_LAI_time_series pixels = MODIS figure pixels) and
##          GS SPEI (SPEI_GS_1958_2024{WIN}.df, original SPEI, no re-standardisation); main SPEI12, SPEI03 as check
## versions: raw (LAI and SPEI as they are) and detrended (both detrended per pixel over 2000-2024), X and Y alike
## 1 asymmetry (hinge regression, per pixel):
##     LAI = a + b_wet * max(SPEI, 0) + b_dry * min(SPEI, 0)
##     b_wet: LAI gain per unit SPEI above 0, b_dry: LAI loss per unit SPEI below 0; b_wet > b_dry = wet years gain
##     more than dry years lose; pixels with >= 5 years on each side; per group area-weighted mean, paired t-test
##     (raw: 0 = the SPEI normal; detrended: 0 = the pixel's own 2000-2024 level, i.e. wetter / drier than usual)
##     composites on raw SPEI (fixed thresholds): mean LAI in wet (> 1), normal (-1 to 1), dry (< -1) pixel-years
## 2 decomposition of the LAI trend 2000-2024 (raw version): per pixel-year LAI = wet part b_wet * max(SPEI, 0)
##     + dry part b_dry * min(SPEI, 0) + background (a + residual); regional mean of each part per year and its
##     trend; trend(LAI) = trend(wet) + trend(dry) + trend(background)
## 3 drought sensitivity in moving windows: per pixel slope of LAI on SPEI in 10-year windows from 1982
##     (1982-1991 ... 2015-2024; 2026-11-27: from 1982, pixels complete 1982-2024), raw and detrended within the
##     window; area-weighted mean per group; trend over all windows and over the windows after 2000 (start >= 2000)
##     (overlapping windows: the p value is optimistic, read as descriptive)
## groups : 7 groups (All, Forest, Shrub annual / perennial, Grass annual / perennial, Savanna), RAP 1986-2024
## output : {R2}\03_response\wet_dry_asymmetry\
##   asymmetry_{scale}.csv, composite_{scale}.csv, LAI_trend_decomposition_{scale}.csv, sensitivity_window_{scale}.csv
##   Fig_asymmetry_{scale}, Fig_LAI_trend_decomposition_{scale}, Fig_sensitivity_window_{scale} (talk-size fonts)
from Time_series_plotting import *


class Wet_dry_asymmetry(Long_LAI_time_series):
    def __init__(self):
        super().__init__('SNU', 2000, 2024)
        self.outdir = self.result2w + rf'03_response\wet_dry_asymmetry\\'
        T.mk_dir(self.outdir, force=True)
        self.spei_dff = result_root + rf'\Dataframe\SPEI_long_term\SPEI_GS_1958_2024{WIN}.df'
        self.scale_list = ['SPEI12', 'SPEI03']
        self.years = np.arange(2000, 2025)
        self.window_years = np.arange(1982, 2025)   ## 2026-11-27: moving-window sensitivity from 1982
        self.min_side = 5        ## years needed on each side of SPEI = 0
        self.window = 10         ## years per moving window
        self.groups = ['All', 'Forest', 'Shrub annual', 'Shrub perennial', 'Grass annual', 'Grass perennial', 'Savanna']
        self.font = {'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
                     'legend.fontsize': 14}

    def run(self):
        info, lai, spei = self.load_arrays(self.years)
        info_w, lai_w, spei_w = self.load_arrays(self.window_years)   ## 2026-11-27: 1982-2024 for the windows
        for sc in self.scale_list:
            fit = {}
            rows = []
            for ver in ['raw', 'detrended']:
                y, x = (lai, spei[sc]) if ver == 'raw' else (self.detrend(lai), self.detrend(spei[sc]))
                fit[ver] = self.hinge_fit(y, x)
                rows.append(self.asymmetry_table(info, fit[ver], ver))
            tab = pd.concat(rows)
            tab.to_csv(self.outdir + f'asymmetry_{sc}.csv', index=False)
            print(sc, 'asymmetry:')
            print(tab.round(3).to_string())
            self.composite(info, lai, spei[sc], sc)
            dec = self.decomposition(info, lai, spei[sc], fit['raw'], sc)
            sen = self.sensitivity_window(info_w, lai_w, spei_w[sc], sc)
            self.fig_asymmetry(tab, sc)
            self.fig_decomposition(dec, sc)
            self.fig_sensitivity(sen, sc)

    def load_arrays(self, years):
        ## pixel x year arrays of GS LAI (% of mean) and GS SPEI at each scale; pixels complete over the years
        df = Long_LAI_time_series('SNU', int(years[0]), int(years[-1])).load()
        sp = T.load_df(self.spei_dff)
        sp = sp[(sp['year'] >= years[0]) & (sp['year'] <= years[-1])]
        cols = [f'growing_{sc}' for sc in self.scale_list]
        df = df.merge(sp[['pix', 'year'] + cols], on=['pix', 'year'], how='inner')
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
        lai = df.pivot_table(index='pix', columns='year', values='growing_LAI_rel').reindex(
            index=info.index, columns=years).values
        spei = {sc: df.pivot_table(index='pix', columns='year', values=f'growing_{sc}').reindex(
            index=info.index, columns=years).values for sc in self.scale_list}
        ok = np.isfinite(lai).all(axis=1) & np.all([np.isfinite(v).all(axis=1) for v in spei.values()], axis=0)
        print(f'pixels with LAI and SPEI in every year {years[0]}-{years[-1]}: {int(ok.sum())} of {len(ok)}')
        return info[ok], lai[ok], {sc: v[ok] for sc, v in spei.items()}

    def detrend(self, v):
        ## linear detrend of each row (pixel) over the years
        t = self.years - self.years.mean()
        b = ((v - v.mean(axis=1, keepdims=True)) * t).sum(axis=1) / (t ** 2).sum()
        return v - v.mean(axis=1, keepdims=True) - b[:, None] * t

    def hinge_fit(self, y, x):
        ## per pixel OLS of y on [1, max(x, 0), min(x, 0)] (batched normal equations)
        X = np.stack([np.ones_like(x), np.maximum(x, 0), np.minimum(x, 0)], axis=2)   ## pix x year x 3
        n_wet, n_dry = (x > 0).sum(axis=1), (x < 0).sum(axis=1)
        ok = (n_wet >= self.min_side) & (n_dry >= self.min_side)
        XtX = np.einsum('pyi,pyj->pij', X[ok], X[ok])
        Xty = np.einsum('pyi,py->pi', X[ok], y[ok])
        coef = np.full((len(y), 3), np.nan)
        coef[ok] = np.linalg.solve(XtX, Xty[..., None])[..., 0]
        print(f'hinge fit: pixels with >= {self.min_side} years on each side: {int(ok.sum())} of {len(ok)}')
        return {'a': coef[:, 0], 'b_wet': coef[:, 1], 'b_dry': coef[:, 2], 'n_wet': n_wet, 'n_dry': n_dry, 'ok': ok}

    def wmean_ci(self, v, w):
        ## area-weighted mean and 95% CI (weighted SE with effective n)
        m = np.isfinite(v)
        v, w = v[m], w[m]
        if len(v) < 3:
            return np.nan, np.nan, np.nan
        mu = np.sum(w * v) / np.sum(w)
        n_eff = np.sum(w) ** 2 / np.sum(w ** 2)
        se = np.sqrt(np.sum(w * (v - mu) ** 2) / np.sum(w)) / np.sqrt(n_eff)
        return mu, mu - 1.96 * se, mu + 1.96 * se

    def asymmetry_table(self, info, fit, ver):
        w = info['area_weight'].values
        rows = []
        for g, m in self.group_masks(info).items():
            m = m & fit['ok']
            if m.sum() < 30:
                continue
            bw, bw_lo, bw_hi = self.wmean_ci(fit['b_wet'][m], w[m])
            bd, bd_lo, bd_hi = self.wmean_ci(fit['b_dry'][m], w[m])
            t, p = stats.ttest_rel(fit['b_wet'][m], fit['b_dry'][m])
            rows.append([ver, g, int(m.sum()), bw, bw_lo, bw_hi, bd, bd_lo, bd_hi, bw / bd if bd != 0 else np.nan, p])
        return pd.DataFrame(rows, columns=['version', 'group', 'n_pix', 'b_wet', 'b_wet_lo', 'b_wet_hi', 'b_dry',
                                           'b_dry_lo', 'b_dry_hi', 'ratio_wet_dry', 'paired_p'])

    def composite(self, info, lai, x, sc):
        ## mean LAI (% of mean) in wet / normal / dry pixel-years, fixed thresholds on the raw SPEI
        w = np.repeat(info['area_weight'].values[:, None], lai.shape[1], axis=1)
        rows = []
        for g, m in self.group_masks(info).items():
            for cls, sel in [('wet (> 1)', x > 1), ('normal (-1 to 1)', (x >= -1) & (x <= 1)), ('dry (< -1)', x < -1)]:
                s = sel & m[:, None]
                rows.append([g, cls, int(s.sum()), s.sum() / (m.sum() * lai.shape[1]) * 100,
                             np.sum(lai[s] * w[s]) / np.sum(w[s]) if s.any() else np.nan])
        tab = pd.DataFrame(rows, columns=['group', 'class', 'n_pixel_years', 'share_%', 'mean_LAI_rel'])
        tab.to_csv(self.outdir + f'composite_{sc}.csv', index=False)
        return tab

    def decomposition(self, info, lai, x, fit, sc):
        ## raw version: LAI = wet part + dry part + background; regional mean per year and trend of each part
        ok = fit['ok']
        wet = fit['b_wet'][:, None] * np.maximum(x, 0)
        dry = fit['b_dry'][:, None] * np.minimum(x, 0)
        back = lai - wet - dry
        w = info['area_weight'].values
        t = self.years.astype(float)
        rows, ts = [], []
        for g, m in self.group_masks(info).items():
            m = m & ok
            if m.sum() < 30:
                continue
            row = [g, int(m.sum())]
            for name, v in [('LAI', lai), ('wet part', wet), ('dry part', dry), ('background', back)]:
                s = np.sum(v[m] * w[m, None], axis=0) / np.sum(w[m])
                r = stats.linregress(t, s)
                row += [r.slope * 10, r.pvalue]
                ts.append(pd.DataFrame({'group': g, 'part': name, 'year': self.years, 'value': s}))
            rows.append(row)
        cols = ['group', 'n_pix']
        for name in ['LAI', 'wet part', 'dry part', 'background']:
            cols += [f'{name} trend_%_dec', f'{name} p']
        tab = pd.DataFrame(rows, columns=cols)
        tab.to_csv(self.outdir + f'LAI_trend_decomposition_{sc}.csv', index=False)
        pd.concat(ts).to_csv(self.outdir + f'LAI_trend_decomposition_series_{sc}.csv', index=False)
        print(sc, 'LAI trend decomposition (% per decade):')
        print(tab.round(3).to_string())
        return tab

    def sensitivity_window(self, info, lai, x, sc):
        ## per pixel slope of LAI on SPEI in moving windows; area-weighted mean per group
        w = info['area_weight'].values
        masks = self.group_masks(info)
        rows = []
        for k in range(len(self.window_years) - self.window + 1):
            sl = slice(k, k + self.window)
            yrs = self.window_years[sl]
            for ver in ['raw', 'detrended']:
                yy, xx = lai[:, sl], x[:, sl]
                if ver == 'detrended':
                    t = yrs - yrs.mean()
                    yy = yy - yy.mean(axis=1, keepdims=True) - (((yy - yy.mean(axis=1, keepdims=True)) * t).sum(axis=1) /
                                                                 (t ** 2).sum())[:, None] * t
                    xx = xx - xx.mean(axis=1, keepdims=True) - (((xx - xx.mean(axis=1, keepdims=True)) * t).sum(axis=1) /
                                                                 (t ** 2).sum())[:, None] * t
                xc = xx - xx.mean(axis=1, keepdims=True)
                var = (xc ** 2).sum(axis=1)
                slope = np.where(var > 0, (xc * (yy - yy.mean(axis=1, keepdims=True))).sum(axis=1) / np.where(var > 0, var, 1),
                                 np.nan)
                for g, m in masks.items():
                    mu, lo, hi = self.wmean_ci(slope[m], w[m])
                    rows.append([ver, g, f'{yrs[0]}-{yrs[-1]}', yrs.mean(), mu, lo, hi])
        tab = pd.DataFrame(rows, columns=['version', 'group', 'window', 'centre', 'slope', 'lo', 'hi'])
        tab.to_csv(self.outdir + f'sensitivity_window_{sc}.csv', index=False)
        return tab

    def fig_asymmetry(self, tab, sc):
        with plt.rc_context(self.font):
            fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
            top = np.nanmax(tab[['b_wet_hi', 'b_dry_hi']].values)   ## same y range in both panels
            for ax, ver in zip(axs, ['raw', 'detrended']):
                d = tab[tab['version'] == ver].set_index('group').reindex(self.groups).dropna(subset=['b_wet'])
                xx = np.arange(len(d))
                for off, col, c, lab in [(-0.2, 'b_wet', '#2171B5', 'wet side (SPEI > 0)'),
                                         (0.2, 'b_dry', '#D9480F', 'dry side (SPEI < 0)')]:
                    err = np.array([d[col] - d[col + '_lo'], d[col + '_hi'] - d[col]])
                    ax.bar(xx + off, d[col], width=0.4, color=c, yerr=err, capsize=3, label=lab)
                for i, (r, p) in enumerate(zip(d['ratio_wet_dry'], d['paired_p'])):
                    ax.text(i, top * 1.08, f'{r:.2f}\n{self.p_text(p)}', ha='center', va='bottom', fontsize=11)
                ax.set_ylim(top=top * 1.4)
                ax.axhline(0, color='k', lw=0.8)
                ax.set_xticks(xx)
                ax.set_xticklabels([g.replace(' ', '\n') for g in d.index], fontsize=13)
                ax.set_title(f'{ver} (wet / dry ratio, paired p)', fontsize=16)
                ax.spines[['top', 'right']].set_visible(False)
            axs[0].set_ylabel(f'LAI change per unit {sc}\n(% of mean)')
            h, l = axs[0].get_legend_handles_labels()
            fig.legend(h, l, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.06))
            plt.tight_layout()
            plt.savefig(self.outdir + f'Fig_asymmetry_{sc}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_asymmetry_{sc}.pdf', bbox_inches='tight')
            plt.close()

    def fig_decomposition(self, tab, sc):
        parts = [('LAI', '#2E7D32'), ('wet part', '#2171B5'), ('dry part', '#D9480F'), ('background', '#999999')]
        d = tab.set_index('group').reindex(self.groups).dropna(subset=['n_pix'])
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(13, 5.5))
            xx = np.arange(len(d))
            for k, (name, c) in enumerate(parts):
                v = d[f'{name} trend_%_dec'].values
                p = d[f'{name} p'].values
                bars = ax.bar(xx + (k - 1.5) * 0.2, v, width=0.2, color=c, label=name)
                for b, vv, pp in zip(bars, v, p):
                    if pp < 0.05:
                        ax.text(b.get_x() + b.get_width() / 2, vv, '*', ha='center', va='bottom' if vv >= 0 else 'top',
                                fontsize=18)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_xticks(xx)
            ax.set_xticklabels([g.replace(' ', '\n') for g in d.index])
            ax.set_ylabel(f'Trend 2000-2024 (% per decade)\n* p < 0.05')
            ax.legend(frameon=False, ncol=4, loc='upper left')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + f'Fig_LAI_trend_decomposition_{sc}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_LAI_trend_decomposition_{sc}.pdf', bbox_inches='tight')
            plt.close()

    def fig_sensitivity(self, tab, sc):
        colors = dict(zip(self.groups, ['k', '#1B7837', '#C0692A', '#7B3F13', '#EBA14A', '#87B8CF', '#3D7FA6']))
        with plt.rc_context(self.font):
            fig, axs = plt.subplots(1, 2, figsize=(13, 8.5), sharey=True)
            for ax, ver in zip(axs, ['raw', 'detrended']):
                lines = []
                for g in self.groups:
                    d = tab[(tab['version'] == ver) & (tab['group'] == g)].dropna(subset=['slope'])
                    if d.empty:
                        continue
                    r = stats.linregress(d['centre'], d['slope'])
                    d2 = d[d['window'].str[:4].astype(int) >= 2000]   ## windows starting in 2000 or later
                    r2 = stats.linregress(d2['centre'], d2['slope'])
                    ax.plot(d['centre'], d['slope'], color=colors[g], lw=3.5 if g == 'All' else 2,
                            label=f'{g}: all {r.slope * 10:+.2f} ({self.p_text(r.pvalue)}) | '
                                  f'post-2000 {r2.slope * 10:+.2f} ({self.p_text(r2.pvalue)})')
                    if g == 'All':
                        ax.fill_between(d['centre'], d['lo'], d['hi'], color='k', alpha=0.1, lw=0)
                ax.axhline(0, color='grey', lw=0.8)
                ax.axvline(2000 + (self.window - 1) / 2, color='grey', ls=':', lw=1.2)   ## first window starting in 2000
                ax.set_xlabel(f'Window centre ({self.window}-year windows)')
                ax.set_title(ver, fontsize=16)
                ax.legend(frameon=False, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1)
                ax.spines[['top', 'right']].set_visible(False)
            axs[0].set_ylabel(f'LAI sensitivity to {sc}\n(% of mean per unit SPEI)')
            fig.text(0.5, 0.97, 'trend of the sensitivity per decade: all windows | windows starting 2000 or later',
                     ha='center', fontsize=14)
            plt.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.45, wspace=0.12)
            plt.savefig(self.outdir + f'Fig_sensitivity_window_{sc}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_sensitivity_window_{sc}.pdf', bbox_inches='tight')
            plt.close()

    def p_text(self, p):
        return 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'


def main():
    Wet_dry_asymmetry().run()


if __name__ == '__main__':
    main()
