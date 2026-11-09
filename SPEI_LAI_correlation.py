from __Global__ import *
from scipy import stats, signal
tif_template= rf'D:\Western_US_IAV\Data\basedata\200902.tif'
D=DIC_and_TIF(tif_template=tif_template)



class SPEI_SNU_LAI_correlation():
    ## pixel-wise Pearson r between detrended SNU LAI and detrended SPEI01-48, spring and summer
    def __init__(self):
        self.dff = result_root + rf'\Dataframe\SPEI\SPEI.df'
        self.outdir = this_root + rf'\Result2\03_response\SPEI_SNU_LAI_correlation\\'
        T.mk_dir(self.outdir, force=True)
        self.start_year = 1982
        self.end_year = 2024
        self.season_list = ['spring', 'summer']
        self.scale_list = ['01', '03', '06', '09', '12', '15', '18', '21', '24',
                           '27', '30', '33', '36', '39', '42', '45', '48']
        self.lai_var = 'SNU_LAI'
        self.LAI_threshold = 0.3   ## drop pixels whose multi-year mean LAI < 0.3
        self.crop_threshold = 50   ## drop pixels whose mean cropland cover (IGBP 12 + 14) > 50%
        self.landcover_col = 'landcover_type'
        self.landcover_list = ['Forest', 'Shrub', 'Grass', 'Savanna']
        self.landcover_color_dic = {'Forest': '#1b7837', 'Shrub': '#d6a64f', 'Grass': '#7fbc41', 'Savanna': 'magenta'}
        pass

    def run(self):
        # df = T.load_df(self.dff)   ## only needed when recomputing the correlation
        # df = self.df_clean(df)
        # df_corr = self.cal_correlation(df)
        # T.save_df(df_corr, self.outdir + 'correlation.df')
        # T.df_to_excel(df_corr, self.outdir + 'correlation')
        df_corr = T.load_df(self.outdir + 'correlation.df')
        self.plot_r_vs_scale(df_corr)
        self.plot_r_vs_scale_by_landcover(df_corr)
        self.max_scale_map(df_corr)  ## max value and max scale of r for each pixel, one map per season
        pass

    def df_clean(self, df):
        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 45]
        df = df[(df['year'] >= self.start_year) & (df['year'] <= self.end_year)]
        ## exclude cropland: mean MCD12C1 cropland cover (IGBP 12 + 14) > 50%
        df = df[~(df['cropland_pct'] > self.crop_threshold)]
        return df

    def to_matrix(self, df, col):
        ## rows = pix, columns = year
        return df.pivot(index='pix', columns='year', values=col)

    def detrend_rows(self, arr):
        ## linear detrend of each pixel's time series; rows with any NaN stay NaN
        out = np.full(arr.shape, np.nan)
        ok = np.all(np.isfinite(arr), axis=1)
        out[ok] = signal.detrend(arr[ok], axis=1)
        return out

    def pearson_rows(self, x, y):
        ## row-wise Pearson r and p (both already detrended, no NaN in valid rows)
        n = x.shape[1]
        xa = x - x.mean(axis=1, keepdims=True)
        ya = y - y.mean(axis=1, keepdims=True)
        r = (xa * ya).sum(axis=1) / np.sqrt((xa ** 2).sum(axis=1) * (ya ** 2).sum(axis=1))
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
        p = 2 * stats.t.sf(np.abs(t), n - 2)
        return r, p

    def cal_correlation(self, df):
        weight = df.groupby('pix')['area_weight'].first()
        landcover = df.groupby('pix')[self.landcover_col].first()
        result_list = []
        for season in self.season_list:
            lai = self.to_matrix(df, f'{season}_{self.lai_var}')
            ## mask low-LAI pixels
            lai = lai[lai.mean(axis=1) >= self.LAI_threshold]
            lai_detrend = self.detrend_rows(lai.values)
            for scale in self.scale_list:
                col = f'{season}_SPEI{scale}'
                if not col in df.columns:
                    continue
                spei = self.to_matrix(df, col).reindex(index=lai.index, columns=lai.columns)
                spei_detrend = self.detrend_rows(spei.values)
                r, p = self.pearson_rows(lai_detrend, spei_detrend)
                df_i = pd.DataFrame({'pix': lai.index, 'season': season, 'scale': scale, 'r': r, 'p': p})
                df_i['area_weight'] = df_i['pix'].map(weight)
                df_i[self.landcover_col] = df_i['pix'].map(landcover)
                result_list.append(df_i.dropna(subset=['r']))
                print(season, scale, 'done')
        return pd.concat(result_list, ignore_index=True)

    def plot_r_vs_scale(self, df_corr):
        color_dic = {'spring': '#EA6E88', 'summer': '#48526F'}
        fig, axes = plt.subplots(1, 2, figsize=(16 * centimeter_factor, 6 * centimeter_factor))
        summary_list = []
        for season in self.season_list:
            df_s = df_corr[df_corr['season'] == season]
            mean_r_list = []
            pos_sig_list = []
            for scale in self.scale_list:
                df_i = df_s[df_s['scale'] == scale]
                w = df_i['area_weight'].values
                mean_r = np.sum(df_i['r'].values * w) / np.sum(w)
                pos_sig = np.sum(w[(df_i['r'] > 0) & (df_i['p'] < 0.05)]) / np.sum(w) * 100
                mean_r_list.append(mean_r)
                pos_sig_list.append(pos_sig)
                summary_list.append([season, scale, mean_r, pos_sig])
            x = [int(s) for s in self.scale_list]
            axes[0].plot(x, mean_r_list, marker='o', color=color_dic[season], label=season.capitalize())
            axes[1].plot(x, pos_sig_list, marker='o', color=color_dic[season], label=season.capitalize())
        axes[0].set_ylabel('Area-weighted mean r')
        axes[1].set_ylabel('Area with r > 0, p < 0.05 (%)')
        for ax in axes:
            ax.set_xlabel('SPEI time scale (months)')
            ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
        axes[0].legend(frameon=False)
        plt.tight_layout()
        plt.savefig(self.outdir + 'r_vs_scale.png', dpi=300)
        plt.savefig(self.outdir + 'r_vs_scale.pdf')
        plt.show()
        plt.close()
        df_summary = pd.DataFrame(summary_list, columns=['season', 'scale', 'mean_r', 'area_pos_sig_pct'])
        df_summary.to_csv(self.outdir + 'r_vs_scale.csv', index=False)
        print(df_summary)

    def plot_r_vs_scale_by_landcover(self, df_corr):
        ## area-weighted mean r vs SPEI scale for Forest / Shrub / Grass (+ All), one panel per season
        x = [int(s) for s in self.scale_list]
        fig, axes = plt.subplots(1, 2, figsize=(16 * centimeter_factor, 6 * centimeter_factor), sharey=True)
        summary_list = []
        for ax, season in zip(axes, self.season_list):
            df_s = df_corr[df_corr['season'] == season]
            for lc in self.landcover_list:
                df_lc = df_s[df_s[self.landcover_col] == lc]
                mean_r_list = []
                for scale in self.scale_list:
                    df_i = df_lc[df_lc['scale'] == scale]
                    w = df_i['area_weight'].values
                    mean_r = np.sum(df_i['r'].values * w) / np.sum(w)
                    mean_r_list.append(mean_r)
                    summary_list.append([season, lc, scale, mean_r, len(df_i)])
                n_pix = df_lc['pix'].nunique()
                ax.plot(x, mean_r_list, marker='o', ms=3, color=self.landcover_color_dic[lc], lw=1.5,
                        label=f'{lc} (n={n_pix})')
            ax.set_title(season.capitalize())
            ax.set_xlabel('SPEI time scale (months)')
            ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel('Area-weighted mean r')
        plt.tight_layout()
        plt.savefig(self.outdir + 'r_vs_scale_by_landcover.png', dpi=300)
        plt.savefig(self.outdir + 'r_vs_scale_by_landcover.pdf')
        plt.show()
        plt.close()
        df_summary = pd.DataFrame(summary_list, columns=['season', 'landcover', 'scale', 'mean_r', 'n_pix'])
        df_summary.to_csv(self.outdir + 'r_vs_scale_by_landcover.csv', index=False)

    def max_scale_map(self, df_corr):
        ## for each pixel: the SPEI scale with the highest r, and that r
        for season in self.season_list:
            df_s = df_corr[df_corr['season'] == season]
            df_max = df_s.loc[df_s.groupby('pix')['r'].idxmax()]
            df_max['scale_num'] = df_max['scale'].astype(int)
            for col in ['scale_num', 'r']:
                spatial_dic = T.df_to_spatial_dic(df_max, col)
                arr = D.pix_dic_to_spatial_arr(spatial_dic)
                D.arr_to_tif(arr, self.outdir + f'{season}_max_{col}.tif')
            arr_scale = D.pix_dic_to_spatial_arr(T.df_to_spatial_dic(df_max, 'scale_num'))
            arr_r = D.pix_dic_to_spatial_arr(T.df_to_spatial_dic(df_max, 'r'))
            w = df_max['area_weight'].values
            ## scale map: only pixels with max r > 0 (if r < 0 at all scales, the "max" scale is meaningless); r <= 0 in grey
            df_pos = df_max[df_max['r'] > 0]
            arr_scale = D.pix_dic_to_spatial_arr(T.df_to_spatial_dic(df_pos, 'scale_num'))
            arr_grey = arr_r.copy()
            arr_grey[arr_r > 0] = np.nan
            self.plot_map(arr_scale, f'{season}_max_scale', 'Spectral', 1, 48,
                          'SPEI scale with max r (months)', df_pos['scale_num'].values, df_pos['area_weight'].values,
                          discrete_levels=[int(s) for s in self.scale_list], arr_grey=arr_grey)
            self.plot_map(arr_r, f'{season}_max_r', 'RdBu', -1, 1, 'Max r', df_max['r'].values, w)

    def plot_map(self, arr, fname, cmap, vmin, vmax, cbar_label, hist_vals, hist_w, discrete_levels=None, arr_grey=None):
        ## same map style as SPEI_processing.trend_analysis, plus an area-weighted histogram inset
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patches as mpatches
        from matplotlib.colors import BoundaryNorm, Normalize, ListedColormap
        import matplotlib.ticker as mticker

        if discrete_levels is not None:
            ## one color per SPEI scale
            levels = np.array(discrete_levels)
            bounds = np.concatenate([[levels[0] - 1], (levels[:-1] + levels[1:]) / 2, [levels[-1] + 1]])
            cmap_obj = plt.get_cmap(cmap, len(levels))
            norm = BoundaryNorm(bounds, cmap_obj.N)
        else:
            cmap_obj = plt.get_cmap(cmap)
            norm = Normalize(vmin=vmin, vmax=vmax)

        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        if arr_grey is not None:
            ## masked pixels (max r <= 0) in grey
            grey = np.where(np.isfinite(arr_grey), 1., np.nan)
            ax.imshow(grey, cmap=ListedColormap(['lightgrey']), vmin=0, vmax=1,
                      extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(),
                      interpolation='nearest')
        im = ax.imshow(
            arr,
            cmap=cmap_obj,
            norm=norm,
            extent=[-124.529, -102.079, 25.579, 48.979],   ## template grid: 449 x 468, 0.05 deg
            transform=ccrs.PlateCarree(),
            interpolation='nearest',
        )
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.STATES, linewidth=0.3)

        ## study region box
        rect = mpatches.Rectangle((-125, 30), 20, 15, linewidth=1.5, edgecolor='black',
                                  facecolor='none', transform=ccrs.PlateCarree(), zorder=10)
        ax.add_patch(rect)

        ## longitude / latitude labels
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(range(-125, -100, 5))   ## 5 deg spacing
        gl.ylocator = mticker.FixedLocator(range(25, 50, 5))

        if discrete_levels is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=levels, spacing='uniform')
            cbar.ax.tick_params(labelsize=6)
        else:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label)

        ## histogram inset (area-weighted %), lower-left over the Pacific
        ax_in = ax.inset_axes([0.10, 0.05, 0.30, 0.25])   ## shifted right, clear of the 30N label
        ok = np.isfinite(hist_vals) & np.isfinite(hist_w)
        vals, w = hist_vals[ok], hist_w[ok] / np.sum(hist_w[ok]) * 100
        if discrete_levels is not None:
            area = [np.sum(w[vals == lev]) for lev in levels]
            ax_in.bar(np.arange(len(levels)), area, color=[cmap_obj(i) for i in range(len(levels))], width=0.9)
            ax_in.set_xticks([0, len(levels) // 2, len(levels) - 1])
            ax_in.set_xticklabels([levels[0], levels[len(levels) // 2], levels[-1]])
        else:
            bins = np.linspace(-1, 1, 41)   ## full r range, no clipping
            area, edges = np.histogram(vals, bins=bins, weights=w)
            centers = (edges[:-1] + edges[1:]) / 2
            ax_in.bar(centers, area, width=edges[1] - edges[0], color=cmap_obj(norm(centers)))
            ax_in.axvline(0, color='grey', lw=0.5)
        ax_in.set_title('Area (%)', fontsize=6, pad=2)
        ax_in.tick_params(labelsize=6)
        ax_in.set_zorder(20)   ## above the study-region box line
        ax_in.patch.set_facecolor('white')
        ax_in.patch.set_alpha(1)
        plt.title(fname)
        plt.savefig(self.outdir + f'{fname}_map.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'{fname}_map.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

def main():
    SPEI_SNU_LAI_correlation().run()


if __name__ == '__main__':
    main()
