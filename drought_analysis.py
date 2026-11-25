# coding='utf-8'
## Drought (and wet) analysis of the growing-season SPEI, 1958-2024 (2026-11-24)
## question: was 2000-2024 a megadrought? The regional mean GS SPEI stays within -1 ... 1: is the drought mild, or do
##   the averages (over the growing-season months and over pixels) hide it? Dry and wet sides are counted the same way,
##   so a change can be read as more droughts, fewer wet years, or both
## data  : SPEI_GS_1958_2024{WIN}.df (TerraClimate P - PET; climate_indices SPEI, gamma, calibration 1958-1999; GS SPEI =
##         mean of the monthly SPEI over the pixel's growing-season window, window from window_config)
## pixels: the LAI-figure pixels (herb_dff), Forest / Shrub / Grass / Savanna, complete 1958-2024
## groups: 7 groups, Shrub / Grass split by RAP 1986-2024 (as the long LAI figures)
## values: the GS SPEI as it is, no re-standardisation (2026-11-24, Wen); classes on the standard SPEI thresholds.
##         drought_sd_ref.csv only reports the SD of the GS SPEI in 1958-1999 (< 1 because it averages several months),
##         so the reader knows the thresholds are conservative.
## scales: SPEI03, SPEI06, SPEI12, SPEI24, SPEI36
## analyses:
##   1 area per year: % of area with SPEI < -0.5 (D0 abnormally dry), < -1 (D1 moderate), < -1.5 (D2 severe), < -2 (D3
##     extreme), and the wet mirror > 0.5 (W0), > 1 (W1), > 1.5 (W2), > 2 (W3), 1958-2024; period means 1958-1981,
##     1982-1999, 2000-2024
##   2 frequency: per pixel % of years with SPEI < -1 / -1.5 / -2 and > 1 / 1.5 / 2 in 1982-1999 and 2000-2024 (maps)
##   3 persistence: per pixel % of years with SPEI < 0 / > 0, and the longest run of consecutive years with SPEI < 0 / > 0
##     per period (the periods have 18 and 25 years; the % of years is comparable, the run length favours the longer one)
##   4 regional (area-weighted) mean series of the GS SPEI with the period means
## output: {R2}\01_drought\drought_analysis\
##   Fig_drought_area.png / .pdf        : 1, one row per scale, dry up, wet down
##   Fig_drought_mean_series.png / .pdf : 4
##   Fig_drought_freq_{scale}.png / .pdf: 2, maps (rows -1 / -1.5 / -2, dry left, wet right) and nested drought bars
##   drought_area_by_year.csv, drought_period_summary.csv, drought_sd_ref.csv
from Time_series_plotting import *


class Drought_analysis(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2w + rf'01_drought\drought_analysis\\'
        T.mk_dir(self.outdir, force=True)
        self.spei_dff = result_root + rf'\Dataframe\SPEI_long_term\SPEI_GS_1958_2024{WIN}.df'
        self.herb_years = (1986, 2024)
        self.year_list = np.arange(1958, 2025)
        self.ref = (1958, 1999)
        self.period_dic = {'1958-1981': (1958, 1981), '1982-1999': (1982, 1999), '2000-2024': (2000, 2024)}
        self.scale_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI36']   ## 2026-11-24
        self.class_list = [('D0 abnormally dry, SPEI < -0.5', -0.5, '#FFE08A'), ('D1 moderate, SPEI < -1', -1., '#F4A259'),
                           ('D2 severe, SPEI < -1.5', -1.5, '#D9480F'), ('D3 extreme, SPEI < -2', -2., '#7F1D1D')]
        self.wet_list = [('W0 abnormally wet, SPEI > 0.5', 0.5, '#C6DBEF'), ('W1 moderate, SPEI > 1', 1., '#6BAED6'),
                         ('W2 severe, SPEI > 1.5', 1.5, '#2171B5'), ('W3 extreme, SPEI > 2', 2., '#08306B')]
        self.extent = [-124.529, -102.079, 25.579, 48.979]
        self.font = {'font.size': 24, 'axes.titlesize': 27, 'axes.labelsize': 25, 'xtick.labelsize': 22,
                     'ytick.labelsize': 22, 'legend.fontsize': 21}

    def run(self):
        info, raw = self.load()
        self.sd_check(info, raw)
        area = self.area_by_year(info, raw)
        summ = self.period_summary(info, raw, area)
        self.fig_area(area, summ)
        self.fig_mean_series(info, raw)
        for s in self.scale_list:
            self.fig_freq(info, raw[s], summ, s)

    def load(self):
        df = T.load_df(self.spei_dff)
        pix_now = set(T.load_df(self.herb_dff).index)
        df = df[df['pix'].isin(pix_now) & df['landcover_type'].isin(['Forest', 'Shrub', 'Grass', 'Savanna'])]
        df = df[(df['year'] >= self.year_list[0]) & (df['year'] <= self.year_list[-1])]
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
        raw = {}
        for s in self.scale_list:
            raw[s] = df.pivot_table(index='pix', columns='year', values=f'growing_{s}').reindex(
                index=info.index, columns=self.year_list).values
        ## 2026-11-24: TerraClimate starts in 1958, so SPEI24 / SPEI36 have no value in the first 1-3 years (the
        ##   accumulation window reaches back before 1958). Start all scales at the first year from which every scale
        ##   has a value at >= 99 % of the pixels, then keep the pixels complete from that year on
        print('pixels:', len(info))
        start = self.year_list[0]
        for s, v in raw.items():
            frac = np.isfinite(v).mean(axis=0)
            good = np.array([np.all(frac[k:] >= 0.99) for k in range(len(frac))])
            if not good.any():
                raise ValueError(f'{s}: no year from which >= 99 % of the pixels have values; finite share per year: '
                                 f'{dict(zip(self.year_list, frac.round(2)))}')
            s0 = self.year_list[np.argmax(good)]
            print(f'  {s}: first year with values at >= 99 % of the pixels: {s0}')
            start = max(start, s0)
        keep = self.year_list >= start
        self.year_list = self.year_list[keep]
        self.ref = (start, self.ref[1])
        self.period_dic = {f'{start}-1981': (start, 1981), '1982-1999': (1982, 1999), '2000-2024': (2000, 2024)}
        raw = {s: v[:, keep] for s, v in raw.items()}
        ok = np.all([np.isfinite(v).all(axis=1) for v in raw.values()], axis=0)
        print(f'analysis years {start}-{self.year_list[-1]} (reference {self.ref[0]}-{self.ref[1]}) | pixels complete '
              f'for all scales: {int(ok.sum())}')
        if ok.sum() == 0:
            raise ValueError('no pixel complete for all scales; check the SPEI_GS df')
        return info[ok], {s: v[ok] for s, v in raw.items()}

    def sd_check(self, info, raw):   ## report only, the analyses use the GS SPEI as it is
        ref = (self.year_list >= self.ref[0]) & (self.year_list <= self.ref[1])
        w = info['area_weight'].values
        rows = []
        for s, v in raw.items():
            mu = v[:, ref].mean(axis=1, keepdims=True)
            sd = v[:, ref].std(axis=1, ddof=1, keepdims=True)
            for g, m in self.group_masks(info).items():
                rows.append([s, g, np.sum(sd[m, 0] * w[m]) / np.sum(w[m]), np.sum(mu[m, 0] * w[m]) / np.sum(w[m])])
        tab = pd.DataFrame(rows, columns=['scale', 'group', 'SD_ref_raw', 'mean_ref_raw'])
        tab.to_csv(self.outdir + 'drought_sd_ref.csv', index=False)
        print(f'SD of the raw GS SPEI in {self.ref[0]}-{self.ref[1]} (1 = no compression):')
        print(tab.pivot_table(index='group', columns='scale', values='SD_ref_raw').round(2).to_string())
        return tab

    def area_by_year(self, info, spei):
        w = info['area_weight'].values
        rows = []
        for s, v in spei.items():
            for g, m in self.group_masks(info).items():
                ww = w[m]
                for k, yr in enumerate(self.year_list):
                    row = [s, g, yr]
                    for name, thr, _ in self.class_list:
                        row.append(np.sum(ww * (v[m, k] < thr)) / np.sum(ww) * 100)
                    for name, thr, _ in self.wet_list:
                        row.append(np.sum(ww * (v[m, k] > thr)) / np.sum(ww) * 100)
                    row.append(np.sum(ww * v[m, k]) / np.sum(ww))
                    rows.append(row)
        tab = pd.DataFrame(rows, columns=['scale', 'group', 'year'] + [c[0] for c in self.class_list] +
                           [c[0] for c in self.wet_list] + ['mean_SPEI'])
        tab.to_csv(self.outdir + 'drought_area_by_year.csv', index=False)
        return tab

    def longest_run(self, mask):
        cur = np.zeros(mask.shape[0])
        best = np.zeros(mask.shape[0])
        for t in range(mask.shape[1]):
            cur = np.where(mask[:, t], cur + 1, 0)
            best = np.maximum(best, cur)
        return best

    def pixel_stats(self, v, t0, t1):
        sel = (self.year_list >= t0) & (self.year_list <= t1)
        x = v[:, sel]
        return {'freq_D1': (x < -1).mean(axis=1) * 100, 'freq_W1': (x > 1).mean(axis=1) * 100,
                'freq_D2': (x < -1.5).mean(axis=1) * 100, 'freq_W2': (x > 1.5).mean(axis=1) * 100,
                'freq_D3': (x < -2).mean(axis=1) * 100, 'freq_W3': (x > 2).mean(axis=1) * 100,
                'dry_years': (x < 0).mean(axis=1) * 100, 'wet_years': (x > 0).mean(axis=1) * 100,
                'longest_dry_run': self.longest_run(x < 0), 'longest_wet_run': self.longest_run(x > 0),
                'mean_SPEI': x.mean(axis=1)}

    def period_summary(self, info, spei, area):
        w = info['area_weight'].values
        keys = ['freq_D1', 'freq_W1', 'dry_years', 'wet_years', 'longest_dry_run', 'longest_wet_run', 'mean_SPEI',
                'freq_D2', 'freq_W2', 'freq_D3', 'freq_W3']
        cls = [c[0] for c in self.class_list] + [c[0] for c in self.wet_list]
        rows = []
        for s, v in spei.items():
            for p, (t0, t1) in self.period_dic.items():
                st = self.pixel_stats(v, t0, t1)
                a = area[(area['scale'] == s) & (area['year'] >= t0) & (area['year'] <= t1)]
                for g, m in self.group_masks(info).items():
                    row = [s, p, g, int(m.sum())]
                    row += [a[a['group'] == g][c].mean() for c in cls]
                    row += [np.sum(st[k][m] * w[m]) / np.sum(w[m]) for k in keys]
                    rows.append(row)
        tab = pd.DataFrame(rows, columns=['scale', 'period', 'group', 'n_pix'] + cls +
                           ['freq_D1_%years', 'freq_W1_%years', 'dry_years_%', 'wet_years_%', 'longest_dry_run_years',
                            'longest_wet_run_years', 'mean_SPEI', 'freq_D2_%years', 'freq_W2_%years', 'freq_D3_%years',
                            'freq_W3_%years'])
        tab.to_csv(self.outdir + 'drought_period_summary.csv', index=False)
        show = ['scale', 'period', self.class_list[1][0], self.wet_list[1][0], 'freq_D1_%years', 'freq_W1_%years',
                'longest_dry_run_years', 'longest_wet_run_years', 'mean_SPEI']
        print(tab[tab['group'] == 'All'][show].round(2).to_string())
        return tab

    def fig_area(self, area, summ):
        with plt.rc_context(self.font):
            fig, axs = plt.subplots(len(self.scale_list), 1, figsize=(22, 6.4 * len(self.scale_list)), sharex=True)
            for ax, s in zip(np.atleast_1d(axs), self.scale_list):
                a = area[(area['scale'] == s) & (area['group'] == 'All')].sort_values('year')
                for name, thr, col in self.class_list:   ## nested: SPEI < -0.5 contains SPEI < -1 ...
                    ax.fill_between(a['year'], 0, a[name], color=col, step='mid', label=name, lw=0)
                for name, thr, col in self.wet_list:     ## wet side drawn downwards
                    ax.fill_between(a['year'], 0, -a[name], color=col, step='mid', label=name, lw=0)
                for p, (t0, t1) in self.period_dic.items():
                    t = summ[(summ['scale'] == s) & (summ['period'] == p) & (summ['group'] == 'All')]
                    for col_name, sign, va in [(self.class_list[1][0], 1, 'bottom'), (self.wet_list[1][0], -1, 'top')]:
                        v = t[col_name].values[0]
                        ax.hlines(sign * v, t0 - 0.5, t1 + 0.5, color='k', lw=3)
                        ax.text((t0 + t1) / 2, sign * (v + 3), f'{v:.0f}%', ha='center', va=va, fontsize=22,
                                fontweight='bold')
                for yb in [1981.5, 1999.5]:
                    ax.axvline(yb, color='grey', ls=':', lw=2)
                ax.axhline(0, color='k', lw=1)
                ax.set_ylim(-100, 100)
                ax.set_yticks([-100, -50, 0, 50, 100])
                ax.set_yticklabels(['100', '50', '0', '50', '100'])
                ax.set_xlim(self.year_list[0] - 0.5, self.year_list[-1] + 0.5)
                ax.set_ylabel(f'{s}\nwet     area %     dry')
                ax.spines[['top', 'right']].set_visible(False)
            np.atleast_1d(axs)[0].legend(frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.02))
            np.atleast_1d(axs)[-1].set_xlabel('Year (black lines: mean area with SPEI < -1 (up) and SPEI > 1 (down) in '
                                              f'{", ".join(self.period_dic)})')
            plt.tight_layout()
            plt.savefig(self.outdir + 'Fig_drought_area.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_drought_area.pdf', bbox_inches='tight')
            plt.close()

    def fig_mean_series(self, info, raw):
        w = info['area_weight'].values
        with plt.rc_context(self.font):
            fig, axs = plt.subplots(len(self.scale_list), 1, figsize=(22, 5.0 * len(self.scale_list)), sharex=True)
            for ax, s in zip(np.atleast_1d(axs), self.scale_list):
                for v, lab, col, lw in [(raw[s], 'GS SPEI, area-weighted mean', '#D9480F', 3.5)]:
                    ts = (v * w[:, None]).sum(axis=0) / w.sum()
                    ax.plot(self.year_list, ts, color=col, lw=lw, marker='o', ms=6, label=lab)
                    for p, (t0, t1) in self.period_dic.items():
                        sel = (self.year_list >= t0) & (self.year_list <= t1)
                        ax.hlines(ts[sel].mean(), t0, t1, color=col, lw=2, ls='--')
                for yv in [-1, -0.5, 0]:
                    ax.axhline(yv, color='k', lw=1 if yv == 0 else 0.8, ls='-' if yv == 0 else ':')
                ax.set_ylabel(f'{s}\nregional mean')
                ax.spines[['top', 'right']].set_visible(False)
            np.atleast_1d(axs)[-1].set_xlabel(f'Year (dashed: period means {", ".join(self.period_dic)})')
            plt.tight_layout()
            plt.savefig(self.outdir + 'Fig_drought_mean_series.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_drought_mean_series.pdf', bbox_inches='tight')
            plt.close()

    def map_panel(self, fig, rect, info, vals, cmap, bounds, title, cb_label, cbar=True):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.colors import BoundaryNorm
        arr = np.full((468, 449), np.nan)
        arr[np.array([p[0] for p in info.index]), np.array([p[1] for p in info.index])] = vals
        ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
        cm = plt.get_cmap(cmap, len(bounds) - 1)
        im = ax.imshow(arr, cmap=cm, norm=BoundaryNorm(bounds, cm.N), extent=self.extent,
                       transform=ccrs.PlateCarree(), interpolation='nearest')
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
        ax.set_extent([-125, -102, 30, 46], crs=ccrs.PlateCarree())
        ax.set_title(title, fontweight='bold')
        if cbar:
            cb = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.85, pad=0.04, ticks=bounds)
            cb.set_label(cb_label)
        return ax, im

    def fig_freq(self, info, v, summ, s):
        ## 2026-11-24: no difference maps; rows SPEI < -1 / -1.5 / -2 (left two columns, 1982-1999 and 2000-2024) and the
        ##   wet mirror > 1 / 1.5 / 2 (right two columns); one shared colorbar for the dry and one for the wet maps
        p1, p2 = (1982, 1999), (2000, 2024)
        st = {p1: self.pixel_stats(v, *p1), p2: self.pixel_stats(v, *p2)}
        groups = list(self.group_masks(info).keys())
        fb = [0, 5, 10, 20, 30, 40, 50, 60, 70]
        rows = [('freq_D1', 'SPEI < -1', 'freq_W1', 'SPEI > 1'), ('freq_D2', 'SPEI < -1.5', 'freq_W2', 'SPEI > 1.5'),
                ('freq_D3', 'SPEI < -2', 'freq_W3', 'SPEI > 2')]
        x0 = [0.0, 0.245, 0.51, 0.755]
        with plt.rc_context(self.font):
            fig = plt.figure(figsize=(32, 36))
            im_dry = im_wet = None
            for r, (kd, ld, kw, lw_) in enumerate(rows):
                y0 = 0.745 - r * 0.215
                for c, (key, lab, cm, per) in enumerate([(kd, ld, 'YlOrRd', p1), (kd, ld, 'YlOrRd', p2),
                                                         (kw, lw_, 'YlGnBu', p1), (kw, lw_, 'YlGnBu', p2)]):
                    _, im = self.map_panel(fig, [x0[c], y0, 0.235, 0.2], info, st[per][key], cm, fb,
                                           f'{lab}, {per[0]}-{per[1]}', '', cbar=False)
                    if c == 0:
                        im_dry = im
                    if c == 2:
                        im_wet = im
            for im, xl, lab in [(im_dry, x0[0], '% of years with SPEI below the threshold (dry)'),
                                (im_wet, x0[2], '% of years with SPEI above the threshold (wet)')]:
                cax = fig.add_axes([xl + 0.03, 0.305, 0.42, 0.012])
                cb = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=fb)
                cb.set_label(lab)
            ## 2026-11-24: bars only for drought, nested like the area figure: SPEI < -1 contains < -1.5 contains < -2;
            ##   one panel per period, same x axis
            y = np.arange(len(groups))[::-1]
            xmax = 0
            axs = []
            for k, p in enumerate(['1982-1999', '2000-2024']):
                ax = fig.add_axes([0.09 + k * 0.47, 0.03, 0.4, 0.2])
                t = summ[(summ['scale'] == s) & (summ['period'] == p)].set_index('group').reindex(groups)
                for (name, thr, col), key in zip(self.class_list[1:], ['freq_D1_%years', 'freq_D2_%years',
                                                                       'freq_D3_%years']):
                    ax.barh(y, t[key].values, height=0.7, color=col, label=name.split(',')[1].strip())
                    xmax = max(xmax, np.nanmax(t[key].values))
                ax.set_yticks(y)
                ax.set_yticklabels(groups if k == 0 else [''] * len(groups))
                ax.set_xlabel('% of years in drought')
                ax.set_title(p, fontweight='bold')
                ax.spines[['top', 'right']].set_visible(False)
                axs.append(ax)
            for ax in axs:
                ax.set_xlim(0, xmax * 1.05)
            axs[1].legend(frameon=False, ncol=3, loc='lower right', bbox_to_anchor=(1.0, 1.06))
            fig.suptitle(f'{s}: dry and wet frequency (growing-season SPEI)', fontsize=34, y=0.97)
            plt.savefig(self.outdir + f'Fig_drought_freq_{s}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_drought_freq_{s}.pdf', bbox_inches='tight')
            plt.close()


def main():
    Drought_analysis().run()   ## 2026-11-24, GS SPEI as it is (no re-standardisation); SPEI03-36; dry + wet


if __name__ == '__main__':
    main()
