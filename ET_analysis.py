# coding='utf-8'
## GLEAM transpiration (Et) at its own 0.1 deg grid, batch 1 (2026-11-25)
## why  : water side of "the canopy held up": did transpiration fall when SPEI fell after 2000?
## caveat: GLEAM4 computes evaporative stress with a neural network whose inputs include LAI and VOD, and Et scales with
##        the vegetation cover fraction, so Et is not independent of LAI; it is read as "the water use implied by the
##        observed vegetation and climate", not as an independent check of the greening
## data : Data\GLEAM\dic\per_pix_dic_*.npy (Wen's tif_to_dic of extract_tif), GLEAM v4.3a monthly Et (mm per month),
##        study area, 0.1 deg, 1982-01 .. 2024-12 (516 values per pixel); grid read from one extract_tif file
## grid : native 0.1 deg, no resampling. Each 0.1 deg cell gets, from the 0.05 deg LAI pixels inside it
##        (Coarse_lookup, reusable for any coarse grid):
##          n_pix, share of each vegetation group (area weighted), dominant group (>= 50 %, else 'mixed'),
##          growing season = most common SNU gs_start / gs_end of its pixels (window_config, SNU window)
##        cells with fewer than 2 of the 4 possible 0.05 deg pixels are dropped
## variables per cell and year: Et_GS = mean monthly Et over the growing season (mm per month);
##                              Et_ann = calendar-year total (mm per year)
## analyses: trend 1982-2024, 1982-1999, 2000-2024 (OLS, % of the cell's 1982-2024 mean per decade, p < 0.05),
##           period change 2000-2024 vs 1982-1999 (%), group time series (cos-lat weighted mean)
## output: {R2}\04_water_carbon\GLEAM_Et\
##   lookup_0.1deg.csv, Et_GS_by_year.csv (cell x year), Et_trend.csv, Et_group_summary.csv,
##   Fig_Et_ts.png, Fig_Et_trend_map.png, Fig_Et_period_change.png
from Time_series_plotting import *
import calendar
import warnings


class Coarse_lookup(Time_series_base):
    ## 0.05 deg LAI pixels -> coarse cells: vegetation group shares, dominant group, growing season
    def __init__(self):
        super().__init__()
        self.herb_years = (1986, 2024)
        self.gs_dff = result_root + rf'\Dataframe\growing_season\\' + GS_TABLE

    def pixel_table(self):
        gs = T.load_df(self.gs_dff)
        pix_now = set(T.load_df(self.herb_dff).index)
        gs = gs[gs.index.isin(pix_now) & gs['landcover_type'].isin(['Forest', 'Shrub', 'Grass', 'Savanna'])]
        gs = gs[(gs['n_season'] > 0) & (gs['gs_start'] <= gs['gs_end'])]
        info = gs[['lon', 'lat', 'area_weight', 'landcover_type', 'gs_start', 'gs_end']].copy()
        masks = self.group_masks(info)
        info['group'] = 'other'
        for g in ['Forest', 'Shrub annual', 'Shrub perennial', 'Grass annual', 'Grass perennial', 'Savanna']:
            info.loc[masks[g], 'group'] = g
        return info

    def build(self, lon0, lat0, res, nrow, ncol, min_pix):
        ## lon0 / lat0: upper-left corner of the coarse grid
        info = self.pixel_table()
        info['r'] = np.floor((lat0 - info['lat']) / res).astype(int)
        info['c'] = np.floor((info['lon'] - lon0) / res).astype(int)
        info = info[(info['r'] >= 0) & (info['r'] < nrow) & (info['c'] >= 0) & (info['c'] < ncol)]
        groups = ['Forest', 'Shrub annual', 'Shrub perennial', 'Grass annual', 'Grass perennial', 'Savanna']
        rows = []
        for (r, c), d in info.groupby(['r', 'c']):
            w = d['area_weight'].values
            share = {g: w[d['group'].values == g].sum() / w.sum() for g in groups}
            gmax = max(share, key=share.get)
            rows.append([r, c, len(d), lat0 - (r + 0.5) * res, lon0 + (c + 0.5) * res,
                         gmax if share[gmax] >= 0.5 else 'mixed',
                         int(d['gs_start'].mode().iloc[0]), int(d['gs_end'].mode().iloc[0])] +
                        [share[g] for g in groups])
        tab = pd.DataFrame(rows, columns=['r', 'c', 'n_pix', 'lat', 'lon', 'group', 'gs_start', 'gs_end'] +
                           [f'share_{g}' for g in groups])
        tab = tab[(tab['n_pix'] >= min_pix) & (tab['gs_start'] <= tab['gs_end'])]
        print(f'coarse grid {res} deg: cells {len(tab)} (>= {min_pix} LAI pixels) | dominant group:')
        print(tab['group'].value_counts().to_string())
        return tab


class GLEAM_Et(Time_series_base):
    def __init__(self):
        super().__init__()
        self.tif_dir = data_root + rf'GLEAM\extract_tif\\'   ## only for the grid (origin, pixel size)
        self.dic_dir = data_root + rf'GLEAM\dic\\'
        self.outdir = self.result2w + rf'04_water_carbon\GLEAM_Et\\'
        T.mk_dir(self.outdir, force=True)
        self.year_list = np.arange(1982, 2025)
        self.period_dic = {'1982-2024': (1982, 2024), '1982-1999': (1982, 1999), '2000-2024': (2000, 2024)}
        self.group_list = ['All', 'Forest', 'Shrub annual', 'Shrub perennial', 'Grass annual', 'Grass perennial',
                           'Savanna']
        self.color_dic = {'All': '#222222', 'Forest': '#1B7837', 'Shrub annual': '#E08214',
                          'Shrub perennial': '#8C510A', 'Grass annual': '#DFC27D', 'Grass perennial': '#80CDC1',
                          'Savanna': '#762A83'}
        self.font = {'font.size': 26, 'axes.titlesize': 30, 'axes.labelsize': 28, 'xtick.labelsize': 24,
                     'ytick.labelsize': 24, 'legend.fontsize': 24}

    def run(self):
        cube, geo = self.read_cube()
        lon0, res, lat0 = geo[0], geo[1], geo[3]
        nrow, ncol = cube.shape[2], cube.shape[3]
        look = Coarse_lookup().build(lon0, lat0, res, nrow, ncol, min_pix=2)
        look.to_csv(self.outdir + f'lookup_{res:.1f}deg.csv', index=False)
        gs_et, ann_et = self.season_values(cube, look)
        tr = self.trends(look, gs_et)
        summ = self.group_summary(look, gs_et, ann_et, tr)
        self.fig_ts(look, gs_et)
        self.fig_ts_all(look, gs_et, 'Et', 'GLEAM GS Et (% of mean)', '#1F78B4',
                        [(1982, 2024, ':'), (1982, 1999, '--'), (2000, 2024, '-')], 'Fig_Et_ts_all')   ## 2026-11-25
        ## 2026-11-25: the LAI figures' layout (Fig 2 right, Fig 4) for Et
        self.group_trend_bar(look, tr, 'Et', 'Fig_Et_trend_bar')
        self.lai_style_maps(look, tr, geo, nrow, ncol, 'Et', 'GLEAM growing-season Et trend',
                            [-20, -15, -10, -5, 0, 5, 10, 15, 20])
        self.fig_trend_map(look, tr, geo, nrow, ncol)
        # self.fig_period_change(look, tr, summ, geo, nrow, ncol)   ## 2026-11-25: not needed (Wen)

    def read_cube(self):
        ## per-pixel dic (Wen's tif_to_dic): Data\GLEAM\dic\per_pix_dic_*.npy, key (row, col) of the extract_tif grid,
        ##   value = monthly Et 1982-01 .. 2024-12 (516 values, in the sorted order of the tif names)
        ## the grid (origin, pixel size, rows, cols) is read from one extract_tif file
        tifs = sorted(f for f in os.listdir(self.tif_dir) if f.endswith('.tif'))
        array, x0, y0, pw, ph = ToRaster().raster2array(self.tif_dir + tifs[0])
        nrow, ncol = np.array(array).shape
        geo = (x0, abs(pw), 0, y0, 0, -abs(ph))   ## north-up grid: y0 = upper edge
        dic = T.load_npy_dir(self.dic_dir)
        n = len(self.year_list) * 12
        cube = np.full((len(self.year_list), 12, nrow, ncol), np.nan)
        bad = 0
        for (r, c), v in dic.items():
            v = np.array(v, dtype=float)
            if len(v) != n:
                bad += 1
                continue
            v[(v < 0) | ~np.isfinite(v)] = np.nan
            cube[:, :, r, c] = v.reshape(len(self.year_list), 12)
        if bad:
            print(f'warning: {bad} pixels without {n} months were skipped (dic made from all {n} tifs 1982-2024?)')
        if bad == len(dic):
            raise ValueError(f'no pixel has {n} monthly values in {self.dic_dir}')
        print('Et cube (year, month, row, col):', cube.shape, '| grid', geo)
        return cube, geo

    def season_values(self, cube, look):
        r, c = look['r'].values, look['c'].values
        x = cube[:, :, r, c]                     ## (year, month, cell)
        mon = np.arange(1, 13)[:, None]
        win = (mon >= look['gs_start'].values[None, :]) & (mon <= look['gs_end'].values[None, :])   ## (month, cell)
        gs_et = np.nansum(x * win[None], axis=1) / win.sum(axis=0)[None]    ## mean monthly Et in the season
        gs_et[np.isnan(x).all(axis=1)] = np.nan
        ann_et = np.nansum(x, axis=1)
        ann_et[np.isnan(x).any(axis=1)] = np.nan
        out = pd.DataFrame(gs_et.T, columns=self.year_list)
        out.insert(0, 'c', look['c'].values)
        out.insert(0, 'r', look['r'].values)
        out.to_csv(self.outdir + 'Et_GS_by_year.csv', index=False)
        return gs_et.T, ann_et.T                 ## (cell, year)

    def ols(self, y, yrs):
        ## y (cell, year) -> slope, p for cells complete in these years
        from scipy import stats
        t = yrs - yrs.mean()
        ok = np.isfinite(y).all(axis=1)
        slope = np.full(len(y), np.nan)
        p = np.full(len(y), np.nan)
        yy = y[ok]
        b = (yy - yy.mean(axis=1, keepdims=True)) @ t / np.sum(t ** 2)
        res = yy - yy.mean(axis=1, keepdims=True) - b[:, None] * t[None]
        se = np.sqrt(np.sum(res ** 2, axis=1) / (len(t) - 2) / np.sum(t ** 2))
        slope[ok], p[ok] = b, 2 * stats.t.sf(np.abs(b / se), len(t) - 2)
        return slope, p

    def trends(self, look, gs_et):
        mean_all = np.nanmean(gs_et, axis=1)
        tr = look[['r', 'c', 'lat', 'lon', 'group']].copy()
        tr['Et_GS_mean'] = mean_all
        for p, (y0, y1) in self.period_dic.items():
            sel = (self.year_list >= y0) & (self.year_list <= y1)
            s, pv = self.ols(gs_et[:, sel], self.year_list[sel].astype(float))
            tr[f'trend_{p}'] = s * 10 / mean_all * 100        ## % of the mean per decade
            tr[f'p_{p}'] = pv
        m1 = gs_et[:, (self.year_list >= 1982) & (self.year_list <= 1999)].mean(axis=1)
        m2 = gs_et[:, (self.year_list >= 2000) & (self.year_list <= 2024)].mean(axis=1)
        ## 2026-11-25: step change as for LAI (Fig 2): mean 2000-2024 minus mean 1982-1999 of the relative series,
        ##   relative = (value / cell mean 1982-2024 - 1) * 100, i.e. (m2 - m1) / whole-record mean * 100
        tr['step_pct'] = (m2 - m1) / mean_all * 100
        tr.to_csv(self.outdir + 'Et_trend.csv', index=False)
        return tr

    def masks(self, look):
        g = look['group'].values
        return {k: (np.ones(len(g), bool) if k == 'All' else g == k) for k in self.group_list}

    def group_summary(self, look, gs_et, ann_et, tr):
        w = np.cos(np.deg2rad(look['lat'].values))
        rows = []
        for k, m in self.masks(look).items():
            if m.sum() == 0:
                continue
            row = [k, int(m.sum())]
            for p, (y0, y1) in [('1982-1999', (1982, 1999)), ('2000-2024', (2000, 2024))]:
                sel = (self.year_list >= y0) & (self.year_list <= y1)
                row += [np.nansum(np.nanmean(gs_et[m][:, sel], axis=1) * w[m]) / w[m].sum(),
                        np.nansum(np.nanmean(ann_et[m][:, sel], axis=1) * w[m]) / w[m].sum()]
            for p in self.period_dic:
                v, pv = tr[f'trend_{p}'].values[m], tr[f'p_{p}'].values[m]
                ok = np.isfinite(v)
                row += [np.sum(v[ok] * w[m][ok]) / w[m][ok].sum(),
                        np.sum(w[m][ok] * ((pv[ok] < 0.05) & (v[ok] > 0))) / w[m][ok].sum() * 100,
                        np.sum(w[m][ok] * ((pv[ok] < 0.05) & (v[ok] < 0))) / w[m][ok].sum() * 100]
            ch = tr['step_pct'].values[m]
            ok = np.isfinite(ch)
            row.append(np.sum(ch[ok] * w[m][ok]) / w[m][ok].sum())
            rows.append(row)
        cols = ['group', 'n_cells', 'Et_GS_1982_1999', 'Et_ann_1982_1999', 'Et_GS_2000_2024', 'Et_ann_2000_2024']
        for p in self.period_dic:
            cols += [f'trend_{p}', f'sig_up_%_{p}', f'sig_down_%_{p}']
        cols.append('step_pct')
        tab = pd.DataFrame(rows, columns=cols)
        tab.to_csv(self.outdir + 'Et_group_summary.csv', index=False)
        print(tab.round(2).to_string())
        return tab

    def fig_ts(self, look, gs_et):
        w = np.cos(np.deg2rad(look['lat'].values))
        groups = [g for g in self.group_list if self.masks(look)[g].sum() > 0]
        with plt.rc_context(self.font):
            fig, axs = plt.subplots(len(groups), 1, figsize=(20, 4.2 * len(groups)), sharex=True)
            for ax, g in zip(np.atleast_1d(axs), groups):
                m = self.masks(look)[g]
                x = gs_et[m]
                ok = np.isfinite(x).all(axis=1)
                ts = np.sum(x[ok] * w[m][ok, None], axis=0) / w[m][ok].sum()
                ax.plot(self.year_list, ts, color=self.color_dic[g], lw=3.5)
                for (y0, y1) in [(1982, 1999), (2000, 2024)]:
                    sel = (self.year_list >= y0) & (self.year_list <= y1)
                    ax.hlines(ts[sel].mean(), y0, y1, color=self.color_dic[g], lw=2.5, ls='--')
                ax.axvline(1999.5, color='grey', ls=':', lw=2)
                ax.set_ylabel('Et (mm/mo)')
                ax.set_title(f'{g} (n = {int(ok.sum())})', loc='left', fontweight='bold')
                ax.spines[['top', 'right']].set_visible(False)
            np.atleast_1d(axs)[-1].set_xlabel('Year (dashed: period means 1982-1999, 2000-2024)')
            fig.suptitle('GLEAM v4.3a transpiration, growing-season mean (SNU window), 0.1 deg', y=1.0)
            plt.tight_layout()
            plt.savefig(self.outdir + 'Fig_Et_ts.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_Et_ts.pdf', bbox_inches='tight')
            plt.close()

    def fig_ts_all(self, look, vals, var, ylab, color, period_list, fname):
        ## 2026-11-25: one panel like Fig 1 / the LAI figure: cos-lat weighted mean of all cells, band = +- 1 spatial SD,
        ##   OLS lines for the whole record (dotted), the early (dashed) and the late period (solid), trend in the
        ##   variable's unit and in % of the whole-record mean per decade
        from scipy import stats
        w = np.cos(np.deg2rad(look['lat'].values))
        years = self.year_list
        ## 2026-11-25: relative series as LAI in Fig 2: (value / cell mean of the whole record - 1) * 100
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            vals = (vals / np.nanmean(vals, axis=1, keepdims=True) - 1) * 100
        mean, sd = [], []
        for j in range(len(years)):
            v = vals[:, j]
            ok = np.isfinite(v)
            m = np.sum(v[ok] * w[ok]) / w[ok].sum()
            mean.append(m)
            sd.append(np.sqrt(np.sum(w[ok] * (v[ok] - m) ** 2) / w[ok].sum()))
        mean, sd = np.array(mean), np.array(sd)
        rows, text = [], []
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(20, 8))
            ax.plot(years, mean, color=color, lw=3.5)
            ax.fill_between(years, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)
            for t0, t1, ls in period_list:
                sel = (years >= t0) & (years <= t1)
                sl, ic, _, p, _ = stats.linregress(years[sel], mean[sel])
                ax.plot(years[sel], sl * years[sel] + ic, color=color, lw=3, ls=ls)
                text.append(f'{t0}-{t1}: {sl * 10:+.2f} % / decade (p = {p:.2f})')
                rows.append([f'{t0}-{t1}', sl * 10, p])
            ax.text(0.01, 0.03, '\n'.join(text), transform=ax.transAxes, va='bottom', fontsize=22, color=color)
            ax.axvline(1999.5, color='grey', lw=1.5, ls=':')
            ax.axhline(0, color='grey', lw=1)
            ax.set_xlim(years[0], years[-1])
            ax.set_ylabel(ylab)
            ax.set_xlabel('Year')
            ax.set_title(f'{var}, growing-season mean of all cells (n = {len(look)}), band = +- 1 SD', loc='left')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + fname + '.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + fname + '.pdf', bbox_inches='tight')
            plt.close()
        out = pd.DataFrame(rows, columns=['period', 'pct_per_decade', 'p_value'])
        out.to_csv(self.outdir + fname + '_trend.csv', index=False)
        pd.DataFrame({'year': years, 'mean': mean, 'sd': sd}).to_csv(self.outdir + fname + '.csv', index=False)
        print(out.round(3).to_string())

    def block_ci(self, lon, lat, v, w):
        ## 95% CI of the weighted mean, 1 deg block bootstrap (as Long_LAI_time_series.block_ci)
        block = (np.floor(lon) * 1000 + np.floor(lat)).astype(int)
        ids = np.unique(block)
        idx = {b: np.where(block == b)[0] for b in ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(200):
            pick = np.concatenate([idx[b] for b in rng.choice(ids, size=len(ids))])
            boot.append(np.sum(v[pick] * w[pick]) / np.sum(w[pick]))
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    def group_trend_bar(self, look, tr, var, fname):
        ## 2026-11-25: as Fig 2 right (Fig_SNU_trend_bar): 7 groups x 3 periods, cos-lat weighted mean of the cell
        ##   trends (% of the cell mean per decade), 95% CI from a 1 deg block bootstrap
        w_all = np.cos(np.deg2rad(look['lat'].values))
        rows = []
        for per in self.period_dic:
            v_all = tr[f'trend_{per}'].values
            for g, m in self.masks(look).items():
                m = m & np.isfinite(v_all)
                if m.sum() < 3:
                    continue
                v, w = v_all[m], w_all[m]
                lo, hi = self.block_ci(look['lon'].values[m], look['lat'].values[m], v, w)
                rows.append([per, g, int(m.sum()), np.sum(v * w) / np.sum(w), lo, hi])
        tab = pd.DataFrame(rows, columns=['period', 'class', 'n_cells', 'trend_per_decade', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + fname + '.csv', index=False)
        period_list = list(self.period_dic)
        color_list = ['#BDBDBD', '#8FB8DE', '#2E7D32']   ## the LAI figure's colours
        groups = [g for g in self.group_list if g in set(tab['class'])]
        xx = np.arange(len(groups))
        wd = 0.26
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(20, 8))
            for k, (per, color) in enumerate(zip(period_list, color_list)):
                d = tab[tab['period'] == per].set_index('class').reindex(groups)
                v = d['trend_per_decade'].values
                err = np.array([v - d['ci_low'].values, d['ci_high'].values - v])
                ax.bar(xx + (k - 1) * wd, v, width=wd, color=color, yerr=err, capsize=4, label=per)
            ax.axhline(0, color='k', lw=1)
            ax.set_xticks(xx)
            ax.set_xticklabels([g.replace(' ', '\n') for g in groups])
            ax.set_ylabel(f'{var} trend (% per decade)')
            ax.set_title(f'{var} growing-season trend by group')
            ax.legend(frameon=False, ncol=3)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + fname + '.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + fname + '.pdf', bbox_inches='tight')
            plt.close()
        print(tab.round(2).to_string())

    def lai_style_maps(self, look, tr, geo, nrow, ncol, var, title, bounds):
        ## 2026-11-25: as Fig 4 (trend_sig_map of the LAI figures): one map per period, the LAI colours, colour bar
        ##   with arrows, dots p < 0.05, inset = area share of sig / non-sig negative, non-sig / sig positive
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        from matplotlib.colors import BoundaryNorm, ListedColormap
        outdir = self.outdir + 'trend_map\\'
        T.mk_dir(outdir, force=True)
        colors = ['#7B3F13', '#C0692A', '#EBA14A', '#F5D38E', '#CFE3D1', '#87B8CF', '#3D7FA6', '#0B4F6C']
        cmap = ListedColormap(colors)
        cmap.set_under(colors[0])
        cmap.set_over(colors[-1])
        norm = BoundaryNorm(bounds, cmap.N)
        ext = [geo[0], geo[0] + ncol * geo[1], geo[3] + nrow * geo[5], geo[3]]
        w = np.cos(np.deg2rad(look['lat'].values))
        rows = []
        for per in self.period_dic:
            trend, p = tr[f'trend_{per}'].values, tr[f'p_{per}'].values
            ok = np.isfinite(trend)
            arr = np.full((nrow, ncol), np.nan)
            arr[look['r'].values, look['c'].values] = trend
            cats = [((trend < 0) & (p < 0.05) & ok, colors[0]), ((trend < 0) & ~(p < 0.05) & ok, '#BDBDBD'),
                    ((trend >= 0) & ~(p < 0.05) & ok, '#BDBDBD'), ((trend >= 0) & (p < 0.05) & ok, colors[-1])]
            area = [np.sum(w[m]) / np.sum(w[ok]) * 100 for m, _ in cats]
            rows.append([per] + area)
            with plt.rc_context({'font.size': 18}):
                fig = plt.figure(figsize=(9, 9))
                ax = plt.axes(projection=ccrs.PlateCarree())
                im = ax.imshow(arr, cmap=cmap, norm=norm, extent=ext, transform=ccrs.PlateCarree(),
                               interpolation='nearest')
                sig = (p < 0.05) & ok
                ax.scatter(look['lon'].values[sig], look['lat'].values[sig], s=3 if geo[1] < 0.2 else 9, c='k',
                           marker='.', lw=0, transform=ccrs.PlateCarree(), zorder=5)
                ax.add_feature(cfeature.STATES, linewidth=0.3)
                ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
                ax.set_extent([-125, -102, 28, 49], crs=ccrs.PlateCarree())
                gl = ax.gridlines(draw_labels=True, linewidth=0)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlocator = mticker.FixedLocator(range(-120, -100, 10))
                gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
                ax.set_title(f'{title} {per} (% per decade)', fontsize=18)
                cb = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.75, ticks=bounds, extend='both',
                                  pad=0.07, aspect=35)
                cb.set_label('Dots: p < 0.05')
                ax_b = ax.inset_axes([0.02, 0.03, 0.30, 0.17])
                ax_b.bar(range(4), area, color=[c for _, c in cats], width=0.75, edgecolor='#333333', linewidth=0.25)
                for i_, a in enumerate(area):
                    ax_b.text(i_, a + max(area) * 0.04, f'{a:.0f}%', ha='center', va='bottom', fontsize=11)
                ax_b.set_ylim(0, max(area) * 1.35 + 1)
                ax_b.set_xticks([])
                ax_b.set_yticks([])
                ax_b.spines[['top', 'right', 'left']].set_visible(False)
                ax_b.patch.set_alpha(0)
                ax_b.set_zorder(20)
                fn = f'GS_{var}_trend_sig_dots_{per.replace("-", "_")}'
                plt.savefig(outdir + fn + '.png', dpi=300, bbox_inches='tight')
                plt.savefig(outdir + fn + '.pdf', bbox_inches='tight')
                plt.close()
        tab = pd.DataFrame(rows, columns=['period', 'sig_neg_%', 'nonsig_neg_%', 'nonsig_pos_%', 'sig_pos_%'])
        tab.to_csv(outdir + f'GS_{var}_trend_area.csv', index=False)
        print(tab.round(1).to_string())

    def grid_map(self, ax, look, vals, geo, nrow, ncol, cmap, bounds, sig=None):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.colors import BoundaryNorm
        arr = np.full((nrow, ncol), np.nan)
        arr[look['r'].values, look['c'].values] = vals
        ext = [geo[0], geo[0] + ncol * geo[1], geo[3] + nrow * geo[5], geo[3]]
        cm = plt.get_cmap(cmap, len(bounds) - 1)
        im = ax.imshow(arr, cmap=cm, norm=BoundaryNorm(bounds, cm.N), extent=ext, transform=ccrs.PlateCarree(),
                       interpolation='nearest')
        if sig is not None:
            s = sig & np.isfinite(vals)
            ## 2026-11-25: larger dots (one per 0.1 deg cell), drawn on every second cell so the colour stays visible
            s = s & (look['r'].values % 2 == 0) & (look['c'].values % 2 == 0)
            ax.scatter(look['lon'].values[s], look['lat'].values[s], s=4, color='k', transform=ccrs.PlateCarree(),
                       linewidths=0)
        ax.add_feature(cfeature.STATES, linewidth=0.4)
        ## 2026-11-25: map extent = the cells with data, small margin
        ax.set_extent([look['lon'].min() - 0.3, look['lon'].max() + 0.3, look['lat'].min() - 0.3,
                       look['lat'].max() + 0.3], crs=ccrs.PlateCarree())
        return im

    def fig_trend_map(self, look, tr, geo, nrow, ncol):
        import cartopy.crs as ccrs
        bounds = [-15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15]
        with plt.rc_context(self.font):
            fig = plt.figure(figsize=(33, 11))
            for k, p in enumerate(self.period_dic):
                ax = fig.add_axes([k * 0.333, 0.12, 0.32, 0.8], projection=ccrs.PlateCarree())
                im = self.grid_map(ax, look, tr[f'trend_{p}'].values, geo, nrow, ncol, 'BrBG', bounds,
                                   sig=tr[f'p_{p}'].values < 0.05)
                ax.set_title(p, fontweight='bold')
            cax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds)
            cb.set_label('Et trend (% of the 1982-2024 mean per decade), dots p < 0.05 (every second cell)')
            plt.savefig(self.outdir + 'Fig_Et_trend_map.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_Et_trend_map.pdf', bbox_inches='tight')
            plt.close()

    def fig_period_change(self, look, tr, summ, geo, nrow, ncol):
        import cartopy.crs as ccrs
        bounds = [-30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30]
        with plt.rc_context(self.font):
            fig = plt.figure(figsize=(26, 11))
            ax = fig.add_axes([0.0, 0.12, 0.5, 0.8], projection=ccrs.PlateCarree())
            im = self.grid_map(ax, look, tr['step_pct'].values, geo, nrow, ncol, 'BrBG', bounds)
            ax.set_title('Et step: mean 2000-2024 - mean 1982-1999', fontweight='bold')
            cax = fig.add_axes([0.05, 0.06, 0.4, 0.03])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds)
            cb.set_label('step (% of the 1982-2024 mean)')
            bx = fig.add_axes([0.62, 0.16, 0.36, 0.72])
            t = summ.set_index('group').reindex([g for g in self.group_list if g in summ['group'].values])
            y = np.arange(len(t))[::-1]
            bx.barh(y, t['step_pct'].values, color=[self.color_dic[g] for g in t.index], height=0.7)
            for yi, v in zip(y, t['step_pct'].values):
                bx.text(v, yi, f' {v:+.1f}% ', va='center', ha='left' if v >= 0 else 'right', fontsize=22)
            ## 2026-11-25: room for the labels beyond the bar ends (they overlapped the group names)
            vmin, vmax = np.nanmin(t['step_pct'].values), np.nanmax(t['step_pct'].values)
            span = max(vmax, 0) - min(vmin, 0)
            bx.set_xlim(min(vmin, 0) - 0.3 * span, max(vmax, 0) + 0.3 * span)
            bx.axvline(0, color='k', lw=1)
            bx.set_yticks(y)
            bx.set_yticklabels(t.index)
            bx.set_xlabel('Et step, 2000-2024 vs 1982-1999 (% of mean)')
            bx.spines[['top', 'right']].set_visible(False)
            plt.savefig(self.outdir + 'Fig_Et_period_change.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_Et_period_change.pdf', bbox_inches='tight')
            plt.close()


def main():
    GLEAM_Et().run()   ## 2026-11-25, needs Data\GLEAM\dic (516 months 1982-2024 per pixel)


if __name__ == '__main__':
    main()
