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
        self.fig_trend_map(look, tr, geo, nrow, ncol)
        self.fig_period_change(look, tr, summ, geo, nrow, ncol)

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
        tr['change_pct'] = (m2 - m1) / m1 * 100
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
            ch = tr['change_pct'].values[m]
            ok = np.isfinite(ch)
            row.append(np.sum(ch[ok] * w[m][ok]) / w[m][ok].sum())
            rows.append(row)
        cols = ['group', 'n_cells', 'Et_GS_1982_1999', 'Et_ann_1982_1999', 'Et_GS_2000_2024', 'Et_ann_2000_2024']
        for p in self.period_dic:
            cols += [f'trend_{p}', f'sig_up_%_{p}', f'sig_down_%_{p}']
        cols.append('change_pct')
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
            im = self.grid_map(ax, look, tr['change_pct'].values, geo, nrow, ncol, 'BrBG', bounds)
            ax.set_title('Et, 2000-2024 vs 1982-1999', fontweight='bold')
            cax = fig.add_axes([0.05, 0.06, 0.4, 0.03])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds)
            cb.set_label('change (%)')
            bx = fig.add_axes([0.62, 0.16, 0.36, 0.72])
            t = summ.set_index('group').reindex([g for g in self.group_list if g in summ['group'].values])
            y = np.arange(len(t))[::-1]
            bx.barh(y, t['change_pct'].values, color=[self.color_dic[g] for g in t.index], height=0.7)
            for yi, v in zip(y, t['change_pct'].values):
                bx.text(v, yi, f' {v:+.1f}% ', va='center', ha='left' if v >= 0 else 'right', fontsize=22)
            ## 2026-11-25: room for the labels beyond the bar ends (they overlapped the group names)
            vmin, vmax = np.nanmin(t['change_pct'].values), np.nanmax(t['change_pct'].values)
            span = max(vmax, 0) - min(vmin, 0)
            bx.set_xlim(min(vmin, 0) - 0.3 * span, max(vmax, 0) + 0.3 * span)
            bx.axvline(0, color='k', lw=1)
            bx.set_yticks(y)
            bx.set_yticklabels(t.index)
            bx.set_xlabel('Et change, 2000-2024 vs 1982-1999 (%)')
            bx.spines[['top', 'right']].set_visible(False)
            plt.savefig(self.outdir + 'Fig_Et_period_change.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_Et_period_change.pdf', bbox_inches='tight')
            plt.close()


def main():
    GLEAM_Et().run()   ## 2026-11-25, needs Data\GLEAM\dic (516 months 1982-2024 per pixel)


if __name__ == '__main__':
    main()
