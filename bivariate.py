## LAI trend vs SPEI trend at each pixel's best SPEI scale (2026-11-15)
## uses GS.df and the trend tools of growing_season_extraction.py (same pixels, same trend, same map style)
from growing_season_extraction import *


class GS_LAI_SPEI_bivariate(Growing_season_trend):
    ## LAI trend vs SPEI trend at each pixel's best SPEI scale (2026-11-15)
    ## step 1  best scale: per pixel, Pearson r between detrended GS_LAI and detrended GS_SPEI{scale}, 2003-2024,
    ##         scale = 03, 06, ..., 48; best scale = the one with the highest r
    ## step 2  trends (not detrended): GS_LAI (% of the pixel's mean per decade) and GS_SPEI at the best scale
    ## step 3  bivariate map, 2 x 2 by sign: drier + browning, drier + greening, wetter + greening, wetter + browning
    ## pixels: same as Growing_season_trend; tables also for the pixels whose best r has p < 0.05
    ## output: Result2\03_response\growing_season\LAI_SPEI_bivariate\
    def __init__(self):
        super().__init__(this_root + rf'Result2\03_response\LAI_SPEI_bivariate\\')
        self.scale_list = list(range(3, 49, 3))
        self.quad_list = ['drier_browning', 'drier_greening', 'wetter_greening', 'wetter_browning']
        self.quad_color = ['#C0392B', '#27AE60', '#2471A3', '#E67E22']

    def run(self):
        df, info = self.load()
        info = info[info['landcover_type'].isin(self.class_list[1:])]   ## four vegetation types; 'All' = their sum (2026-11-20)
        print('pixels (four vegetation types):', len(info))
        lai = self.matrix(df, info, 'GS_LAI')
        best_scale, best_r, best_p = self.best_scale(df, info, lai)
        spei_best = np.full(lai.shape, np.nan)
        for scale in self.scale_list:
            m = best_scale == scale
            spei_best[m] = self.matrix(df, info, f'GS_SPEI{scale:02d}')[m]
        tr_lai, p_lai = self.trend(self.matrix(df, info, 'GS_LAI', relative=True))
        tr_spei, p_spei = self.trend(spei_best)
        quad = np.where(tr_spei < 0, np.where(tr_lai < 0, 1, 2), np.where(tr_lai >= 0, 3, 4)).astype(float)
        quad[~(np.isfinite(tr_spei) & np.isfinite(tr_lai))] = np.nan
        ## maps
        ## discrete classes, distinct colours (2026-11-20; was a 16-step viridis / continuous RdBu)
        self.class_map(info, best_scale.astype(float), f'best_SPEI_scale_{self.tag}',
                       [0.5, 4.5, 7.5, 10.5, 13.5, 25.5, 37.5, 49.5],
                       ['3', '6', '9', '12', '15-24', '27-36', '39-48'],
                       ['#D73027', '#FC8D59', '#FEE08B', '#A6D96A', '#1A9850', '#4575B4', '#313695'],
                       'Best SPEI scale (months)')
        self.class_map(info, best_r, f'best_SPEI_r_{self.tag}', [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       ['< 0', '0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '> 0.8'],
                       ['#BDBDBD', '#FFFFCC', '#A1DAB4', '#41B6C4', '#2C7FB8', '#253494'],
                       'r (detrended LAI vs best SPEI)')
        self.map_pair(info, tr_spei, p_spei, f'GS_SPEIbest_trend_{self.tag}',
                      'Growing-season SPEI trend at the best scale (per decade)',
                      np.nanpercentile(np.abs(tr_spei), 90), 'RdBu')
        # self.bivariate_map(info, quad, f'LAI_SPEI_bivariate_{self.tag}')
        ## 2026-11-20: same layout as the best-scale figure (map + area by vegetation type + legend)
        self.class_map(info, quad, f'LAI_SPEI_bivariate_{self.tag}', [0.5, 1.5, 2.5, 3.5, 4.5],
                       ['Drier + browning', 'Drier + greening', 'Wetter + greening', 'Wetter + browning'],
                       self.quad_color, 'SPEI trend + LAI trend, 2003-2024')
        self.scatter(info, tr_spei, tr_lai, best_p, f'LAI_SPEI_trend_scatter_{self.tag}')
        ## tables
        res = pd.DataFrame({'best_scale': best_scale, 'best_r': best_r, 'best_p': best_p, 'SPEI_trend': tr_spei,
                            'SPEI_trend_p': p_spei, 'LAI_trend': tr_lai, 'LAI_trend_p': p_lai, 'quadrant': quad},
                           index=info.index)
        T.save_df(info.join(res), self.outdir + f'LAI_SPEI_bivariate_{self.tag}.df')
        self.table_best_scale(info, best_scale, best_r, best_p)
        self.table_quadrant(info, quad, tr_spei, tr_lai, best_p)
        pass

    def best_scale(self, df, info, lai):
        ## per pixel: scale with the highest r between detrended LAI and detrended SPEI
        from scipy.signal import detrend
        n = len(self.year_list)
        lai_d = detrend(lai, axis=1)
        lai_d = (lai_d - lai_d.mean(axis=1, keepdims=True)) / lai_d.std(axis=1, keepdims=True)
        r_all = []
        for scale in self.scale_list:
            spei = self.matrix(df, info, f'GS_SPEI{scale:02d}')
            ok = np.isfinite(spei).all(axis=1)
            spei_d = np.full(spei.shape, np.nan)
            spei_d[ok] = detrend(spei[ok], axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                spei_d = (spei_d - spei_d.mean(axis=1, keepdims=True)) / spei_d.std(axis=1, keepdims=True)
                r_all.append(np.mean(lai_d * spei_d, axis=1))
        r_all = np.array(r_all).T                     ## pix x scale
        i_best = np.argmax(np.where(np.isnan(r_all), -np.inf, r_all), axis=1)
        best_r = r_all[np.arange(len(r_all)), i_best]
        best_scale = np.array(self.scale_list)[i_best]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = best_r * np.sqrt((n - 3) / (1 - best_r ** 2))   ## n - 3: detrended series
            best_p = 2 * stats.t.sf(np.abs(t), n - 3)
        best_scale = np.where(np.isfinite(best_r), best_scale, 0)
        return best_scale, best_r, best_p

    def save_discrete(self, info, vals, fname, cmap, levels, label):
        arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, vals)))
        arr[arr <= 0] = np.nan
        D.arr_to_tif(arr.copy(), self.outdir + f'{fname}.tif')
        ok = vals > 0
        self.mapper.plot_map(arr, fname, cmap, None, None, label, vals[ok], info['area_weight'].values[ok],
                             discrete_levels=levels)

    def class_map(self, info, vals, fname, bounds, labels, colors, cbar_label):
        ## classes (bounds), one distinct colour each (2026-11-20):
        ## left: map; right: area % of each class per vegetation type (stacked bars); legend of the classes below
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
        from matplotlib.colors import ListedColormap, BoundaryNorm
        n = len(labels)
        vals = np.array(vals, dtype=float)
        bad = ~np.isfinite(vals) | (vals < bounds[0])   ## e.g. best scale 0 = no value
        vals = np.where(bad, np.nan, vals)
        k = np.digitize(vals, bounds[1:-1]).astype(float)   ## class index 0 .. n-1
        k[bad] = np.nan
        D.arr_to_tif(D.pix_dic_to_spatial_arr(dict(zip(info.index, vals))), self.outdir + f'{fname}.tif')
        arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, k + 1)))
        arr[arr <= 0] = np.nan
        ## area % per vegetation type x class
        w = info['area_weight'].values
        lc_all = info['landcover_type'].values
        share = []
        for lc in self.class_list:
            m = np.isfinite(k) & ((lc_all == lc) if lc != 'All' else True)
            share.append([np.sum(w[m & (k == c)]) / np.sum(w[m]) * 100 for c in range(n)])
        share = np.array(share)
        pd.DataFrame(share, index=self.class_list, columns=labels).round(1).to_csv(
            self.outdir + f'{fname}_area_by_vegetation.csv')
        fig = plt.figure(figsize=(10, 4.8))
        ax = fig.add_axes([0.02, 0.14, 0.56, 0.80], projection=ccrs.PlateCarree())
        ax.imshow(arr, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(0.5, n + 1.5), n),
                  extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(), interpolation='nearest')
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_patch(mpatches.Rectangle((-125, 30), 20, 15, linewidth=1.5, edgecolor='black', facecolor='none',
                                        transform=ccrs.PlateCarree(), zorder=10))
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(range(-125, -100, 5))
        gl.ylocator = mticker.FixedLocator(range(25, 50, 5))
        ax.set_title(fname, fontsize=9)
        ## stacked bars per vegetation type
        ax2 = fig.add_axes([0.66, 0.22, 0.31, 0.66])
        y = np.arange(len(self.class_list))[::-1]
        left = np.zeros(len(self.class_list))
        for c in range(n):
            ax2.barh(y, share[:, c], left=left, color=colors[c], height=0.7, edgecolor='white', linewidth=0.5)
            for yy, l0, v in zip(y, left, share[:, c]):
                if v >= 8:
                    ax2.text(l0 + v / 2, yy, f'{v:.0f}', ha='center', va='center', fontsize=6,
                             color='white' if c in (0, n - 2, n - 1) else 'black')
            left += share[:, c]
        ax2.set_yticks(y)
        ax2.set_yticklabels(self.class_list, fontsize=8)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Area (%)', fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.spines[['top', 'right']].set_visible(False)
        ## legend of the classes
        handles = [mpatches.Patch(color=c, label=l) for l, c in zip(labels, colors)]
        fig.legend(handles=handles, loc='lower center', ncol=n, fontsize=7, title=cbar_label, title_fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        plt.savefig(self.outdir + f'{fname}_map.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'{fname}_map.pdf', bbox_inches='tight')
        plt.close()

    def bivariate_map(self, info, quad, fname):
        ## 2 x 2 map with its own legend: area % of each quadrant
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap, BoundaryNorm
        arr = D.pix_dic_to_spatial_arr(dict(zip(info.index, quad)))
        arr[arr <= 0] = np.nan
        D.arr_to_tif(arr.copy(), self.outdir + f'{fname}.tif')
        w = info['area_weight'].values
        ok = np.isfinite(quad)
        area = [np.sum(w[ok & (quad == q)]) / np.sum(w[ok]) * 100 for q in [1, 2, 3, 4]]
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.imshow(arr, cmap=ListedColormap(self.quad_color), norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4),
                  extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(), interpolation='nearest')
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_patch(mpatches.Rectangle((-125, 30), 20, 15, linewidth=1.5, edgecolor='black', facecolor='none',
                                        transform=ccrs.PlateCarree(), zorder=10))
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        handles = [mpatches.Patch(color=c, label=f'{q.replace("_", " + ")} ({a:.0f}%)')
                   for q, c, a in zip(self.quad_list, self.quad_color, area)]
        ax.legend(handles=handles, loc='lower left', fontsize=6, title='SPEI trend + LAI trend (area)',
                  title_fontsize=6, framealpha=1)
        plt.title(fname)
        plt.savefig(self.outdir + f'{fname}_map.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'{fname}_map.pdf', bbox_inches='tight')
        plt.close()

    def scatter(self, info, tr_spei, tr_lai, best_p, fname):
        ## per vegetation type: LAI trend vs SPEI trend (best scale); area % per quadrant in the corners
        fig, axs = plt.subplots(1, len(self.class_list), figsize=(3 * len(self.class_list), 3), sharex=True,
                                sharey=True)
        xl = np.nanpercentile(np.abs(tr_spei), 99)
        yl = np.nanpercentile(np.abs(tr_lai), 99)
        for ax, lc in zip(axs, self.class_list):
            m = np.isfinite(tr_spei) & np.isfinite(tr_lai)
            if lc != 'All':
                m &= (info['landcover_type'] == lc).values
            w = info['area_weight'].values[m]
            x, y = tr_spei[m], tr_lai[m]
            ax.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=40, cmap='Greys', bins='log', mincnt=1,
                      extent=[-xl, xl, -yl, yl])
            ax.axhline(0, color='k', lw=0.5)
            ax.axvline(0, color='k', lw=0.5)
            for (sx, sy), c, ha, va in zip([(-1, -1), (-1, 1), (1, 1), (1, -1)], self.quad_color,
                                           ['left', 'left', 'right', 'right'], ['bottom', 'top', 'top', 'bottom']):
                a = np.sum(w[(np.sign(x) == sx) & (np.where(y >= 0, 1, -1) == sy)]) / np.sum(w) * 100
                ax.text(0.03 if sx < 0 else 0.97, 0.03 if sy < 0 else 0.97, f'{a:.0f}%', color=c, ha=ha, va=va,
                        transform=ax.transAxes, fontsize=8, fontweight='bold')
            ax.set_title(f'{lc} (n={m.sum()})', fontsize=9)
            ax.set_xlabel('SPEI trend, best scale (/decade)', fontsize=8)
        axs[0].set_ylabel('LAI trend (% /decade)', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.outdir + f'{fname}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'{fname}.pdf', bbox_inches='tight')
        plt.close()

    def region_dic(self, info, best_p):
        return {'all': np.ones(len(info), dtype=bool),
                'monsoon': (info['monsoon_share'] >= 0.3).values,
                'non-monsoon': (info['monsoon_share'] < 0.3).values,
                'all_r_sig': best_p < 0.05,
                'monsoon_r_sig': (info['monsoon_share'] >= 0.3).values & (best_p < 0.05),
                'non-monsoon_r_sig': (info['monsoon_share'] < 0.3).values & (best_p < 0.05)}

    def table_best_scale(self, info, best_scale, best_r, best_p):
        ## per region x class: area % of each best scale, mean best r, area % with r > 0 and p < 0.05
        w_all = info['area_weight'].values
        result = []
        for region, mr in self.region_dic(info, best_p).items():
            if 'r_sig' in region:
                continue
            for lc in self.class_list:
                m = mr & ((info['landcover_type'] == lc).values if lc != 'All' else True) & np.isfinite(best_r)
                if m.sum() < 30:
                    continue
                w = w_all[m]
                row = [region, lc, m.sum(), np.sum(best_r[m] * w) / np.sum(w),
                       np.sum(w * ((best_r[m] > 0) & (best_p[m] < 0.05))) / np.sum(w) * 100]
                row += [np.sum(w[best_scale[m] == s]) / np.sum(w) * 100 for s in self.scale_list]
                result.append(row)
        df_t = pd.DataFrame(result, columns=['region', 'class', 'n_pix', 'mean_best_r', 'r_pos_sig_area_%'] +
                                            [f'SPEI{s:02d}_area_%' for s in self.scale_list])
        df_t.to_csv(self.outdir + f'best_SPEI_scale_summary_{self.tag}.csv', index=False)
        print(df_t.round(1).to_string())

    def table_quadrant(self, info, quad, tr_spei, tr_lai, best_p):
        ## per region x class: area % of each quadrant and mean trends
        w_all = info['area_weight'].values
        result = []
        for region, mr in self.region_dic(info, best_p).items():
            for lc in self.class_list:
                m = mr & ((info['landcover_type'] == lc).values if lc != 'All' else True) & np.isfinite(quad)
                if m.sum() < 30:
                    continue
                w = w_all[m]
                row = [region, lc, m.sum(), np.sum(tr_spei[m] * w) / np.sum(w), np.sum(tr_lai[m] * w) / np.sum(w)]
                row += [np.sum(w[quad[m] == q]) / np.sum(w) * 100 for q in [1, 2, 3, 4]]
                result.append(row)
        df_t = pd.DataFrame(result, columns=['region', 'class', 'n_pix', 'SPEI_trend_per_decade',
                                             'LAI_trend_%_per_decade'] + [f'{q}_area_%' for q in self.quad_list])
        df_t.to_csv(self.outdir + f'LAI_SPEI_quadrant_summary_{self.tag}.csv', index=False)
        print(df_t.round(2).to_string())


class LAI_SPEI_scale_heatmap(GS_LAI_SPEI_bivariate):
    ## vegetation response to SPEI at each scale: heat maps (2026-11-20)
    ## per pixel and scale: Pearson r between detrended GS_LAI and detrended GS_SPEI{scale}, 2003-2024
    ##   (same r as GS_LAI_SPEI_bivariate.best_scale; p with n - 3 df because both series are detrended)
    ## heat map a: area-weighted mean r, vegetation type x scale (best scale of each row outlined)
    ## heat map b: area % with r > 0 and p < 0.05
    ## heat map c: area % of pixels whose best scale is this scale
    ## output: Result2\03_response\LAI_SPEI_scale_heatmap\
    def __init__(self):
        super().__init__()
        self.outdir = this_root + rf'Result2\03_response\LAI_SPEI_scale_heatmap\\'
        T.mk_dir(self.outdir, force=True)

    def run(self):
        df, info = self.load()
        info = info[info['landcover_type'].isin(self.class_list[1:])]   ## four vegetation types; 'All' = their sum (2026-11-20)
        print('pixels (four vegetation types):', len(info))
        lai = self.matrix(df, info, 'GS_LAI')
        r_all, p_all = self.r_matrix(df, info, lai)
        res = pd.DataFrame(r_all, index=info.index, columns=[f'r_SPEI{s:02d}' for s in self.scale_list])
        T.save_df(info.join(res), self.outdir + f'LAI_SPEI_r_by_scale_{self.tag}.df')
        tab = self.table(info, r_all, p_all)
        self.heatmap(tab)
        pass

    def r_matrix(self, df, info, lai):
        ## pix x scale: r and p between detrended LAI and detrended SPEI
        from scipy.signal import detrend
        n = len(self.year_list)
        lai_d = detrend(lai, axis=1)
        lai_d = (lai_d - lai_d.mean(axis=1, keepdims=True)) / lai_d.std(axis=1, keepdims=True)
        r_all = []
        for scale in self.scale_list:
            spei = self.matrix(df, info, f'GS_SPEI{scale:02d}')
            ok = np.isfinite(spei).all(axis=1)
            spei_d = np.full(spei.shape, np.nan)
            spei_d[ok] = detrend(spei[ok], axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                spei_d = (spei_d - spei_d.mean(axis=1, keepdims=True)) / spei_d.std(axis=1, keepdims=True)
                r_all.append(np.mean(lai_d * spei_d, axis=1))
        r_all = np.array(r_all).T
        with np.errstate(divide='ignore', invalid='ignore'):
            t = r_all * np.sqrt((n - 3) / (1 - r_all ** 2))
            p_all = 2 * stats.t.sf(np.abs(t), n - 3)
        return r_all, p_all

    def table(self, info, r_all, p_all):
        ## long table: class x scale -> mean r, area % r > 0 & p < 0.05, area % best scale
        w_all = info['area_weight'].values
        ok_all = np.isfinite(r_all).any(axis=1)
        i_best = np.argmax(np.where(np.isnan(r_all), -np.inf, r_all), axis=1)
        result = []
        for lc in self.class_list:
            mlc = ok_all & ((info['landcover_type'] == lc).values if lc != 'All' else True)
            for k, scale in enumerate(self.scale_list):
                r = r_all[:, k]
                m = mlc & np.isfinite(r)
                w = w_all[m]
                result.append([lc, scale, m.sum(), np.sum(r[m] * w) / np.sum(w),
                               np.sum(w * ((r[m] > 0) & (p_all[m, k] < 0.05))) / np.sum(w) * 100,
                               np.sum(w_all[mlc & (i_best == k)]) / np.sum(w_all[mlc]) * 100])
        tab = pd.DataFrame(result, columns=['class', 'scale', 'n_pix', 'mean_r', 'r_pos_sig_area_%',
                                            'best_scale_area_%'])
        tab.to_csv(self.outdir + f'LAI_SPEI_r_by_scale_{self.tag}.csv', index=False)
        print(tab.pivot(index='class', columns='scale', values='mean_r').reindex(self.class_list).round(2).to_string())
        return tab

    def heatmap(self, tab):
        from matplotlib.patches import Rectangle
        from matplotlib.colors import BoundaryNorm
        ## discrete colour steps (2026-11-20): (column, title, cmap, bounds, label format)
        panel = [('mean_r', 'Mean r (detrended LAI vs SPEI)', 'YlGnBu', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], '{:.2f}'),
                 ('r_pos_sig_area_%', 'Area with r > 0, p < 0.05 (%)', 'YlGnBu', [0, 20, 40, 60, 80, 100], '{:.0f}'),
                 ('best_scale_area_%', 'Area where this scale is the best (%)', 'Purples', [0, 5, 10, 15, 20, 25, 30],
                  '{:.0f}')]
        fig, axs = plt.subplots(len(panel), 1, figsize=(9, 2.3 * len(panel)))
        for ax, (col, title, cmap, bounds, fmt) in zip(axs, panel):
            mat = tab.pivot(index='class', columns='scale', values=col).reindex(self.class_list)
            v = mat.values
            im = ax.imshow(v, cmap=plt.get_cmap(cmap, len(bounds) - 1), norm=BoundaryNorm(bounds, len(bounds) - 1),
                           aspect='auto')
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    hi = v[i, j] > bounds[0] + 0.6 * (bounds[-1] - bounds[0])
                    ax.text(j, i, fmt.format(v[i, j]), ha='center', va='center', fontsize=6,
                            color='white' if hi else 'black')
                if col == 'mean_r':
                    j = int(np.nanargmax(v[i]))
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='#C2410C', lw=1.5))
            ax.set_xticks(range(v.shape[1]))
            ax.set_xticklabels([f'{s}' for s in mat.columns], fontsize=7)
            ax.set_yticks(range(v.shape[0]))
            ax.set_yticklabels(mat.index, fontsize=8)
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=bounds, extend='max')
        axs[-1].set_xlabel('SPEI scale (months)', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.outdir + f'LAI_SPEI_scale_heatmap_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'LAI_SPEI_scale_heatmap_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


def main():
    GS_LAI_SPEI_bivariate().run()
    # LAI_SPEI_scale_heatmap().run()   ## detrended LAI vs SPEI r by scale and vegetation type


if __name__ == '__main__':
    main()
