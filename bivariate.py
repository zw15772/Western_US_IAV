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
        self.save_discrete(info, best_scale.astype(float), f'best_SPEI_scale_{self.tag}', 'viridis_r',
                           self.scale_list, 'Best SPEI scale (months)')
        self.save_map(info, best_r, f'best_SPEI_r_{self.tag}', 'r (detrended LAI vs best SPEI)', 0.8, 'RdBu')
        self.map_pair(info, tr_spei, p_spei, f'GS_SPEIbest_trend_{self.tag}',
                      'Growing-season SPEI trend at the best scale (per decade)',
                      np.nanpercentile(np.abs(tr_spei), 90), 'RdBu')
        self.bivariate_map(info, quad, f'LAI_SPEI_bivariate_{self.tag}')
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


def main():
    GS_LAI_SPEI_bivariate().run()


if __name__ == '__main__':
    main()
