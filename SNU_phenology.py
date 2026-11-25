# coding='utf-8'
## Growing season and phenology from the SNU monthly LAI, 1982-2024 (2026-11-23)
## 1) SNU_window_compare: is the MODIS growing-season window right for the SNU record?
## 2) SNU_phenology     : did the season start earlier / end later, and how much of the GS LAI trend sits at the edges
##                        of the fixed window (season extension) rather than in its core (more leaf)?
## outputs: Result2\02_vegetation\phenology_SNU\
from growing_season_extraction import *
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'legend.fontsize': 11, 'legend.title_fontsize': 11})


class SNU_season_base(Growing_season_trend):
    ## common parts: pixels, groups, SNU monthly LAI, trend with NaN, group means
    ## pixels: growing_season.df (study box, cropland masks, US only) with a MODIS main season inside one calendar year
    ## groups: Forest, Shrub / Grass x annual / perennial, Savanna of the long attribution table (RAP 1986-2024
    ##         AFG > PFG = annual); 'All' = all pixels with a group
    def __init__(self):
        Growing_season_trend.__init__(self, this_root + rf'Result2\02_vegetation\phenology_SNU\\')
        self.year_list = list(range(1982, 2025))
        self.tag = '1982_2024'
        self.gs_dff = result_root + rf'\Dataframe\growing_season\growing_season.df'
        self.group_dff = this_root + \
            rf'Result2\03_response\long_attribution_1986_2024\long_attribution_data_v2_1986_2024.df'
        self.group_list = ['All', 'Forest', 'Shrub annual', 'Shrub perennial', 'Grass annual', 'Grass perennial',
                           'Savanna']
        self.min_years = 35   ## years with a value needed for a pixel trend (of 43)
        self.big_font = {'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15,
                         'ytick.labelsize': 15, 'legend.fontsize': 15}

    def load_base(self):
        gs = T.load_df(self.gs_dff)
        gs = gs[(gs['n_season'] > 0) & (gs['gs_start'] <= gs['gs_end'])].copy()
        grp = T.load_df(self.group_dff).groupby('pix')['group'].first()
        gs['group'] = grp.reindex(gs.index).values
        gs = gs[gs['group'].notna()]
        self.loader = GS_LAI_SNU()
        mon = self.loader.load_monthly(list(gs.index))   ## pixel x 516, Jan 1982 ... Dec 2024
        print('pixels with a MODIS season and a group:', len(gs))
        return gs, mon

    def masks(self, gs):
        g = gs['group'].values
        return {k: (np.ones(len(gs), dtype=bool) if k == 'All' else g == k) for k in self.group_list}

    def trend_nan(self, arr, years=None):
        ## OLS slope per decade and p per row, NaN years skipped; NaN if fewer than min_years values
        x = np.array(self.year_list if years is None else years, dtype=float)
        ok = np.isfinite(arr)
        n = ok.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            xbar = np.nansum(np.where(ok, x, 0), axis=1) / n
            ybar = np.nansum(np.where(ok, arr, 0), axis=1) / n
            dx = np.where(ok, x[None, :] - xbar[:, None], 0)
            dy = np.where(ok, arr - ybar[:, None], 0)
            sxx = np.sum(dx ** 2, axis=1)
            slope = np.sum(dx * dy, axis=1) / sxx
            resid = np.where(ok, dy - slope[:, None] * dx, 0)
            se = np.sqrt(np.sum(resid ** 2, axis=1) / (n - 2) / sxx)
            p = 2 * stats.t.sf(np.abs(slope / se), n - 2)
        bad = n < self.min_years
        slope[bad], p[bad] = np.nan, np.nan
        return slope * 10, p

    def group_mean(self, gs, vals, name):
        ## area-weighted mean per group with 95% CI (1 x 1 deg block bootstrap)
        rows = []
        for g, m in self.masks(gs).items():
            m = m & np.isfinite(vals)
            if m.sum() < 30:
                continue
            d = gs[m]
            v, w = vals[m], d['area_weight'].values
            lo, hi = self.block_ci(d, v, w)
            rows.append([g, name, int(m.sum()), np.sum(v * w) / np.sum(w), lo, hi])
        return rows

    def map_discrete(self, gs, vals, fname, label, levels, cmap):
        arr = D.pix_dic_to_spatial_arr(dict(zip(gs.index, vals)))
        arr[arr <= -9999] = np.nan
        ok = np.isfinite(vals)
        self.mapper.plot_map(arr, fname, cmap, levels[0], levels[-1], label, vals[ok], gs['area_weight'].values[ok],
                             discrete_levels=levels)

    def map_cont(self, gs, vals, fname, label, vlim, cmap):
        arr = D.pix_dic_to_spatial_arr(dict(zip(gs.index, vals)))
        arr[arr <= -9999] = np.nan
        ok = np.isfinite(vals)
        self.mapper.plot_map(arr, fname, cmap, -vlim, vlim, label, vals[ok], gs['area_weight'].values[ok])


class SNU_window_compare(SNU_season_base):
    ## the growing season defined from the SNU record with the MODIS rule (Growing_season_extraction.define: main
    ##   season around the highest Feb-Oct month, neighbouring months while LAI falls and stays above half the
    ##   seasonal amplitude; 1982-2024 mean of each calendar month), compared with the MODIS window (2003-2024 MODIS)
    ## per pixel: start, end, peak, length difference SNU - MODIS (months); pixels whose SNU main season crosses the new
    ##   year or has no season are counted and left out
    ## GS LAI under both windows (mean of the window months, % of the pixel's mean), trend 1982-2024 and 1986-2024;
    ##   per group area-weighted mean and 95% CI, and the paired difference SNU window - MODIS window
    ## output: Fig_GS_window_compare.png / .pdf (a: start difference, b: end difference, c: GS LAI trend both windows),
    ##   GS_window_compare_shift.csv, GS_window_compare_trend.csv, maps GS_start_SNU_minus_MODIS, GS_end_SNU_minus_MODIS,
    ##   pixel-year table Result\Dataframe\growing_season\GS_SNU_snuwin_check.df (GS LAI with the SNU window, for
    ##   checking; the file used downstream, GS_SNU_snuwin.df, is made by GS_LAI_SNU with window_config)
    def run(self):
        gs, mon = self.load_base()
        snu = self.define_snu(gs, mon)
        shift = self.shift_table(gs, snu)
        trend = self.trend_compare(gs, snu, mon)
        self.figure(shift, trend)
        self.map_discrete(gs, np.clip(snu['d_start'].values, -3, 3), 'GS_start_SNU_minus_MODIS',
                          'GS start: SNU - MODIS (months)', [-3, -2, -1, 0, 1, 2, 3], 'RdBu_r')
        self.map_discrete(gs, np.clip(snu['d_end'].values, -3, 3), 'GS_end_SNU_minus_MODIS',
                          'GS end: SNU - MODIS (months)', [-3, -2, -1, 0, 1, 2, 3], 'RdBu_r')

    def define_snu(self, gs, mon):
        clim = np.nanmean(mon.reshape(len(gs), -1, 12), axis=1)   ## pixel x 12, 1982-2024
        snu = pd.DataFrame([self.loader.define(c) for c in clim], index=gs.index)
        use = (snu['n_season'] > 0) & (snu['gs_start'] <= snu['gs_end'])
        print('SNU season: none', int((snu['n_season'] == 0).sum()), '| across the new year',
              int(((snu['n_season'] > 0) & (snu['gs_start'] > snu['gs_end'])).sum()), '| used', int(use.sum()))
        for k in ['start', 'end', 'peak']:
            d = snu[f'gs_{k}'] - gs[f'gs_{k}']
            snu[f'd_{k}'] = np.where(use, (d + 6) % 12 - 6, np.nan)   ## circular, -6 ... 5
        snu['d_length'] = np.where(use, (snu['gs_end'] - snu['gs_start']) - (gs['gs_end'] - gs['gs_start']), np.nan)
        snu['use'] = use
        snu['amplitude'] = np.nanmax(clim, axis=1) - np.nanmin(clim, axis=1)
        return snu

    def shift_table(self, gs, snu):
        rows = []
        w = gs['area_weight'].values
        for g, m in self.masks(gs).items():
            m = m & snu['use'].values
            for k in ['start', 'end', 'peak', 'length']:
                d = snu[f'd_{k}'].values[m]
                ww = w[m]
                row = [g, k, int(m.sum()), np.sum(d * ww) / np.sum(ww), np.sum(np.abs(d) * ww) / np.sum(ww)]
                for c in [-2, -1, 0, 1, 2]:
                    sel = (d <= -2) if c == -2 else ((d >= 2) if c == 2 else (d == c))
                    row.append(np.sum(ww[sel]) / np.sum(ww) * 100)
                rows.append(row)
        same = snu['use'] & (snu['d_start'] == 0) & (snu['d_end'] == 0)
        print('pixels with the same window (start and end):', round(np.sum(w[same]) / np.sum(w[snu['use']]) * 100, 1),
              '% of area')
        tab = pd.DataFrame(rows, columns=['group', 'edge', 'n_pix', 'mean_diff', 'mean_abs_diff', '<=-2', '-1', '0',
                                          '+1', '>=+2'])
        tab.to_csv(self.outdir + 'GS_window_compare_shift.csv', index=False)
        print(tab.round(2).to_string())
        return tab

    def trend_compare(self, gs, snu, mon):
        u = snu['use'].values
        start_m = gs['gs_start'].values.astype(int) - 1
        len_m = (gs['gs_end'].values - gs['gs_start'].values).astype(int) + 1
        start_s = np.where(u, snu['gs_start'].values, 1).astype(int) - 1
        len_s = np.where(u, snu['gs_end'].values - snu['gs_start'].values + 1, 1).astype(int)
        lai = {'MODIS window': self.loader.window_long(mon, start_m, len_m),
               'SNU window': self.loader.window_long(mon, start_s, len_s)}
        lai['SNU window'][~u] = np.nan
        rel = {}
        for k, v in lai.items():
            with np.errstate(invalid='ignore', divide='ignore'):
                rel[k] = (v / np.nanmean(v, axis=1, keepdims=True) - 1) * 100
        self.save_snu_window(gs, snu, lai['SNU window'], rel['SNU window'])
        rows = []
        for t0 in [1982, 1986]:
            yrs = list(range(t0, 2025))
            sl = slice(t0 - 1982, None)
            tr = {k: self.trend_nan(v[:, sl], yrs)[0] for k, v in rel.items()}
            both = np.isfinite(tr['MODIS window']) & np.isfinite(tr['SNU window'])
            for k in tr:
                rows += [[f'{t0}-2024'] + r for r in self.group_mean(gs, np.where(both, tr[k], np.nan), k)]
            rows += [[f'{t0}-2024'] + r for r in self.group_mean(gs, np.where(both, tr['SNU window'] -
                                                                               tr['MODIS window'], np.nan),
                                                                  'SNU - MODIS')]
        tab = pd.DataFrame(rows, columns=['period', 'group', 'window', 'n_pix', 'trend', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + 'GS_window_compare_trend.csv', index=False)
        print(tab.pivot_table(index=['period', 'group'], columns='window', values='trend').round(2).to_string())
        return tab

    def save_snu_window(self, gs, snu, lai, rel):
        ny = len(self.year_list)
        df = pd.DataFrame({'pix': [p for p in gs.index for _ in range(ny)], 'year': np.tile(self.year_list, len(gs)),
                           'GS_LAI': lai.ravel(), 'GS_LAI_rel': rel.ravel()})
        info = gs[['lon', 'lat', 'area_weight', 'landcover_type', 'n_season']].copy()
        info['gs_start'] = snu['gs_start'].values
        info['gs_end'] = snu['gs_end'].values
        info['GS_mean_LAI'] = np.nanmean(lai, axis=1)
        df = df.merge(info, left_on='pix', right_index=True, how='left')
        df = df[df['pix'].isin(set(gs.index[snu['use'].values]))]
        T.save_df(df, result_root + rf'\Dataframe\growing_season\GS_SNU_snuwin_check.df')   ## 2026-11-23: renamed; GS_SNU_snuwin.df is made by GS_LAI_SNU

    def figure(self, shift, trend):
        cls = ['<=-2', '-1', '0', '+1', '>=+2']
        col = ['#2166AC', '#92C5DE', '#E0E0E0', '#F4A582', '#B2182B']
        groups = [g for g in self.group_list if g in set(shift['group'])]
        y = np.arange(len(groups))[::-1]
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(1, 3, figsize=(26, 8), gridspec_kw={'width_ratios': [1, 1, 1.3]})
            for ax, edge, title in [(axs[0], 'start', 'a) GS start, SNU - MODIS'),
                                    (axs[1], 'end', 'b) GS end, SNU - MODIS')]:
                t = shift[shift['edge'] == edge].set_index('group').reindex(groups)
                left = np.zeros(len(groups))
                for c, cc in zip(cls, col):
                    ax.barh(y, t[c].values, left=left, color=cc, label=f'{c} month' if c not in ['0'] else 'same')
                    left += t[c].values
                ax.set_yticks(y)
                ax.set_yticklabels(groups)
                ax.set_xlim(0, 100)
                ax.set_xlabel('% of area')
                ax.set_title(title)
                ax.spines[['top', 'right']].set_visible(False)
            axs[0].legend(frameon=False, ncol=5, loc='upper left', bbox_to_anchor=(0.0, -0.12), fontsize=14)
            ax = axs[2]
            t = trend[trend['period'] == '1982-2024']
            x = np.arange(len(groups))
            for k, (wn, c) in enumerate([('MODIS window', '#7F8C8D'), ('SNU window', '#2E7D32')]):
                d = t[t['window'] == wn].set_index('group').reindex(groups)
                v = d['trend'].values
                ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=c, label=wn,
                       yerr=[v - d['ci_low'].values, d['ci_high'].values - v], capsize=4)
            dd = t[t['window'] == 'SNU - MODIS'].set_index('group').reindex(groups)
            top = ax.get_ylim()[1]
            ax.set_ylim(top=top * 1.12)
            for xx, v in zip(x, dd['trend']):
                if np.isfinite(v):
                    ax.text(xx, top * 1.08, f'{v:+.2f}', ha='center', va='top', fontsize=14)   ## CI in the csv
            ax.axhline(0, color='k', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([g.replace(' ', '\n') for g in groups], fontsize=14)
            ax.set_ylabel('GS LAI trend 1982-2024 (% of mean / decade)')
            ax.set_title('c) GS LAI trend with each window (text: SNU - MODIS)')
            ax.legend(frameon=False, loc='lower right')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + 'Fig_GS_window_compare.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_GS_window_compare.pdf', bbox_inches='tight')
            plt.close()


class SNU_phenology(SNU_season_base):
    ## yearly start (SOS), end (EOS) and length (LOS) of the main season from the SNU monthly LAI, 1982-2024
    ## per pixel-year (12 monthly values, value of month i placed at its middle, day (i + 0.5) x 30.44):
    ##   peak  = highest month within +-2 months of the pixel's SNU climatological peak (MODIS peak if SNU has none)
    ##   level = min of the year + half x (peak - min of the year)  (half amplitude of that year's curve)
    ##   SOS   = upward crossing of the level before the peak, linear interpolation between months; searched back to
    ##           3 months before the MODIS window start (and not before January)
    ##   EOS   = downward crossing after the peak, searched up to 3 months after the window end (not after December)
    ##   no crossing in the search range -> NaN; pixels with a seasonal amplitude < amp_min (m2/m2, 1982-2024 mean
    ##   curve) are left out: the timing of a flat curve is noise
    ##   monthly data: one month = 30 days, so trends of a few days per decade are near the resolution
    ## trends 1982-2024 (days per decade; >= min_years years); per group area-weighted mean, 95% CI
    ## edge vs core: GS LAI with the MODIS window (the one of all analyses) = mean of the window months; its trend =
    ##   sum of the monthly trends / window length; edge = first and last month, core = the months between
    ##   (windows of >= 3 months); a large edge share = greening from a longer season, a large core share = more leaf
    ## output: Fig_phenology.png / .pdf (a: SOS, EOS, LOS trends; b: GS LAI trend edge + core; c: SOS / EOS anomaly
    ##   time series, All), phenology_trend.csv, phenology_edge_core.csv, maps SOS_trend, EOS_trend, pixel-year table
    ##   phenology_SNU_{tag}.df
    ## 2026-11-23: season window = window_config GS_TABLE (search range of SOS / EOS, edge vs core), outputs in R2
    def __init__(self):
        super().__init__()
        self.outdir = this_root + rf'{R2}\02_vegetation\phenology_SNU\\'
        T.mk_dir(self.outdir, force=True)
        self.mapper = Map_plotter(self.outdir)
        self.gs_dff = result_root + rf'\Dataframe\growing_season\{GS_TABLE}'
        self.group_dff = this_root + \
            rf'{R2}\03_response\long_attribution_1986_2024\long_attribution_data_v2_1986_2024.df'
        self.amp_min = 0.05
        self.search = 3
        self.day = 30.44

    def run(self):
        gs, mon = self.load_base()
        sos, eos, peak_v = self.timing(gs, mon)
        res = self.trends(gs, sos, eos, peak_v)
        edge = self.edge_core(gs, mon)
        self.figure(res, edge, gs, sos, eos)
        tr_sos = self.trend_nan(sos)[0]
        tr_eos = self.trend_nan(eos)[0]
        self.map_cont(gs, tr_sos, 'SOS_trend_1982_2024', 'SOS trend 1982-2024 (days / decade, - = earlier)', 15,
                      'RdBu')
        self.map_cont(gs, tr_eos, 'EOS_trend_1982_2024', 'EOS trend 1982-2024 (days / decade, + = later)', 15,
                      'RdBu_r')
        self.run_maps()

    def run_maps(self):
        ## 2026-11-23: trend maps 1982-2024 of the season length (LOS, days / decade) and of the peak LAI (% of the pixel's
        ##   mean per decade) from the saved pixel-year table phenology_SNU_{tag}.df (made by run); same pixels as run
        gs = T.load_df(self.gs_dff)
        gs = gs[(gs['n_season'] > 0) & (gs['gs_start'] <= gs['gs_end'])].copy()
        grp = T.load_df(self.group_dff).groupby('pix')['group'].first()
        gs['group'] = grp.reindex(gs.index).values
        gs = gs[gs['group'].notna()]
        df = T.load_df(self.outdir + f'phenology_SNU_{self.tag}.df')

        def piv(col):
            return df.pivot(index='pix', columns='year', values=col).reindex(index=gs.index,
                                                                             columns=self.year_list).values
        los, peak = piv('LOS'), piv('peak_LAI')
        with np.errstate(invalid='ignore', divide='ignore'):
            peak_rel = (peak / np.nanmean(peak, axis=1, keepdims=True) - 1) * 100
        tr_los, tr_peak = self.trend_nan(los)[0], self.trend_nan(peak_rel)[0]
        print('pixels with a LOS trend:', int(np.isfinite(tr_los).sum()), '| with a peak LAI trend:',
              int(np.isfinite(tr_peak).sum()), 'of', len(gs))
        self.map_cont(gs, tr_los, 'LOS_trend_1982_2024', 'Season length trend 1982-2024 (days / decade)', 15,
                      'RdBu_r')
        self.map_cont(gs, tr_peak, 'peak_LAI_trend_1982_2024', 'Peak LAI trend 1982-2024 (% of mean / decade)', 10,
                      'BrBG')

    def timing(self, gs, mon):
        npx, ny = len(gs), len(self.year_list)
        clim = np.nanmean(mon.reshape(npx, ny, 12), axis=1)
        snu = pd.DataFrame([self.loader.define(c) for c in clim], index=gs.index)
        pk = np.where(snu['n_season'].values > 0, snu['gs_peak'].values, gs['gs_peak'].values).astype(int) - 1
        amp = np.nanmax(clim, axis=1) - np.nanmin(clim, axis=1)
        keep = amp >= self.amp_min
        print('pixels with a seasonal amplitude >=', self.amp_min, ':', int(keep.sum()), 'of', npx)
        lo = np.clip(gs['gs_start'].values.astype(int) - 1 - self.search, 0, 11)
        hi = np.clip(gs['gs_end'].values.astype(int) - 1 + self.search, 0, 11)
        V = mon.reshape(npx * ny, 12)
        pk_r, lo_r, hi_r = np.repeat(pk, ny), np.repeat(lo, ny), np.repeat(hi, ny)
        k = np.arange(12)[None, :]
        near = np.abs(k - pk_r[:, None]) <= 2
        with np.errstate(invalid='ignore'):
            ipk = np.argmax(np.where(near & np.isfinite(V), V, -np.inf), axis=1)
            vmin = np.nanmin(V, axis=1)
        vpk = V[np.arange(len(V)), ipk]
        thr = vmin + 0.5 * (vpk - vmin)
        below = V < thr[:, None]
        ks = np.where(below & (k < ipk[:, None]) & (k >= lo_r[:, None]), k, -1).max(axis=1)
        ke = np.where(below & (k > ipk[:, None]) & (k <= hi_r[:, None]), k, 99).min(axis=1)
        rows = np.arange(len(V))
        sos = np.full(len(V), np.nan)
        eos = np.full(len(V), np.nan)
        m = ks >= 0
        a, b = V[rows[m], ks[m]], V[rows[m], ks[m] + 1]
        with np.errstate(invalid='ignore', divide='ignore'):
            sos[m] = ks[m] + (thr[m] - a) / (b - a)
        m2 = ke < 99
        a, b = V[rows[m2], ke[m2] - 1], V[rows[m2], ke[m2]]
        with np.errstate(invalid='ignore', divide='ignore'):
            eos[m2] = ke[m2] - 1 + (a - thr[m2]) / (a - b)
        bad = ~np.isfinite(V).all(axis=1) | ~np.repeat(keep, ny)
        sos[bad], eos[bad] = np.nan, np.nan
        sos = ((sos + 0.5) * self.day).reshape(npx, ny)   ## day of year
        eos = ((eos + 0.5) * self.day).reshape(npx, ny)
        peak_v = np.where(bad, np.nan, vpk).reshape(npx, ny)
        print('pixel-years with SOS:', int(np.isfinite(sos).sum()), '| with EOS:', int(np.isfinite(eos).sum()),
              'of', npx * ny)
        df = pd.DataFrame({'pix': [p for p in gs.index for _ in range(ny)], 'year': np.tile(self.year_list, npx),
                           'SOS': sos.ravel(), 'EOS': eos.ravel(), 'LOS': (eos - sos).ravel(),
                           'peak_LAI': peak_v.ravel()})
        T.save_df(df, self.outdir + f'phenology_SNU_{self.tag}.df')
        return sos, eos, peak_v

    def trends(self, gs, sos, eos, peak_v):
        rows = []
        with np.errstate(invalid='ignore', divide='ignore'):
            peak_rel = (peak_v / np.nanmean(peak_v, axis=1, keepdims=True) - 1) * 100
        for name, arr in [('SOS', sos), ('EOS', eos), ('LOS', eos - sos), ('peak LAI (%)', peak_rel)]:
            tr, p = self.trend_nan(arr)
            rows += self.group_mean(gs, tr, name)
            for g, m in self.masks(gs).items():
                ok = m & np.isfinite(tr)
                w = gs['area_weight'].values[ok]
                rows.append([g, name + ' sig. earlier/lower %', int(ok.sum()),
                             np.sum(w[(tr[ok] < 0) & (p[ok] < 0.05)]) / np.sum(w) * 100 if ok.sum() else np.nan,
                             np.nan, np.nan])
                rows.append([g, name + ' sig. later/higher %', int(ok.sum()),
                             np.sum(w[(tr[ok] > 0) & (p[ok] < 0.05)]) / np.sum(w) * 100 if ok.sum() else np.nan,
                             np.nan, np.nan])
        tab = pd.DataFrame(rows, columns=['group', 'variable', 'n_pix', 'value', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + 'phenology_trend.csv', index=False)
        print(tab.pivot_table(index='group', columns='variable', values='value').round(2).to_string())
        return tab

    def edge_core(self, gs, mon):
        npx, ny = len(gs), len(self.year_list)
        M = mon.reshape(npx, ny, 12)
        start = gs['gs_start'].values.astype(int) - 1
        L = (gs['gs_end'].values - gs['gs_start'].values).astype(int) + 1
        gs_lai = self.loader.window_long(mon, start, L)
        mean = np.nanmean(gs_lai, axis=1)
        edge_c = np.full(npx, np.nan)
        core_c = np.full(npx, np.nan)
        for Lk in np.unique(L):
            if Lk < 3:
                continue
            idx = np.where(L == Lk)[0]
            edge_c[idx], core_c[idx] = 0., 0.
            for j in range(Lk):
                mm = M[idx, :, :][np.arange(len(idx)), :, start[idx] + j]   ## pixel x year, month j of the window
                with np.errstate(invalid='ignore', divide='ignore'):
                    tr = self.trend_nan(mm / mean[idx, None] * 100)[0] / Lk   ## % of GS mean per decade
                if j in (0, Lk - 1):
                    edge_c[idx] += tr   ## NaN if a month has too few years
                else:
                    core_c[idx] += tr
        with np.errstate(invalid='ignore', divide='ignore'):
            total = self.trend_nan((gs_lai / mean[:, None] - 1) * 100)[0]
        print('pixels with a window of < 3 months (left out of edge / core):', int((L < 3).sum()))
        rows = []
        for name, v in [('GS LAI', total), ('edge months', edge_c), ('core months', core_c)]:
            v = np.where(L >= 3, v, np.nan)
            rows += self.group_mean(gs, v, name)
        tab = pd.DataFrame(rows, columns=['group', 'part', 'n_pix', 'trend', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + 'phenology_edge_core.csv', index=False)
        print(tab.pivot_table(index='group', columns='part', values='trend').round(2).to_string())
        return tab

    def figure(self, res, edge, gs, sos, eos):
        groups = [g for g in self.group_list if g in set(res['group'])]
        x = np.arange(len(groups))
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(1, 3, figsize=(28, 8), gridspec_kw={'width_ratios': [1.3, 1.1, 1]})
            ax = axs[0]
            for k, (name, c) in enumerate([('SOS', '#1F77B4'), ('EOS', '#D62728'), ('LOS', '#2E7D32')]):
                d = res[res['variable'] == name].set_index('group').reindex(groups)
                v = d['value'].values
                ax.bar(x + (k - 1) * 0.27, v, 0.27, color=c, label=name,
                       yerr=[v - d['ci_low'].values, d['ci_high'].values - v], capsize=3)
            ax.axhline(0, color='k', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([g.replace(' ', '\n') for g in groups], fontsize=14)
            ax.set_ylabel('Trend 1982-2024 (days / decade)')
            ax.set_title('a) season timing (SOS - = earlier, EOS + = later)')
            ax.legend(frameon=False)
            ax.spines[['top', 'right']].set_visible(False)
            ax = axs[1]
            e = edge.set_index(['group', 'part'])
            for xx, g in zip(x, groups):
                if (g, 'edge months') not in e.index:
                    continue
                ed, co, tot = (e.loc[(g, p), 'trend'] for p in ['edge months', 'core months', 'GS LAI'])
                ax.bar(xx, co, 0.6, color='#2E8B57', label='core months' if xx == 0 else None)
                ax.bar(xx, ed, 0.6, bottom=co if (co >= 0) == (ed >= 0) else 0, color='#F0B27A',
                       label='edge months (first, last)' if xx == 0 else None)
                ax.scatter(xx, tot, color='k', s=45, zorder=3, label='GS LAI trend' if xx == 0 else None)
            ax.axhline(0, color='k', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([g.replace(' ', '\n') for g in groups], fontsize=14)
            ax.set_ylabel('% of GS mean / decade')
            ax.set_title('b) GS LAI trend (MODIS window): edge vs core')
            ax.legend(frameon=False)
            ax.spines[['top', 'right']].set_visible(False)
            ax = axs[2]
            w = gs['area_weight'].values
            for arr, name, c in [(sos, 'SOS', '#1F77B4'), (eos, 'EOS', '#D62728')]:
                an = arr - np.nanmean(arr, axis=1, keepdims=True)
                ok = np.isfinite(an)
                ts = np.nansum(np.where(ok, an, 0) * w[:, None], axis=0) / np.sum(np.where(ok, w[:, None], 0), axis=0)
                ax.plot(self.year_list, ts, 'o-', color=c, lw=2, ms=5, label=name)
                tr = np.polyfit(self.year_list, ts, 1)
                ax.plot(self.year_list, np.polyval(tr, self.year_list), ':', color=c, lw=2,
                        label=f'{name} {tr[0] * 10:+.1f} d / decade')
            ax.axhline(0, color='k', lw=1)
            ax.set_xlabel('Year')
            ax.set_ylabel('Anomaly (days)')
            ax.set_title('c) All pixels: area-weighted anomaly')
            ax.legend(frameon=False, fontsize=13)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + 'Fig_phenology.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_phenology.pdf', bbox_inches='tight')
            plt.close()


def main():
    # SNU_window_compare().run()   ## 1: SNU vs MODIS growing-season window, GS LAI trend with both (done)
    # SNU_phenology().run()        ## 2: SOS / EOS / LOS trends, edge vs core of the GS LAI trend (done)
    SNU_phenology().run_maps()     ## 3: LOS and peak LAI trend maps from the saved table (2026-11-23)


if __name__ == '__main__':
    main()
