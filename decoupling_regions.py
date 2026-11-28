# coding='utf-8'
## Where are LAI and SPEI decoupled? Step 1: define the regions (2026-11-27)
## data   : SNU GS LAI (growing_LAI_rel, % of mean; pixels complete 1982-2024) and GS SPEI12 (SPEI_GS_1958_2024{WIN}.df)
## states : mean SPEI12 of each period (1982-1999, 2000-2024): wet > 0.5, normal -0.5 to 0.5, dry < -0.5
## classes (pixels dry after 2000 only, i.e. mean SPEI12 2000-2024 < -0.5):
##   decoupled  : dLAI > 0, p < 0.05   (dry after 2000, LAI higher than before)
##   coupled    : dLAI < 0, p < 0.05   (dry after 2000, LAI lower)
##   resistant  : dLAI not significant
##   not dry    : mean SPEI12 2000-2024 >= -0.5 (background, not analysed further)
##   dLAI = mean 2000-2024 - mean 1982-1999 of LAI (% of mean); p of Welch t test (unequal variances)
##   (the size of the SPEI drop is not a condition; it is kept in the table for later steps)
## output : {R2}\03_response\decoupling_regions\
##   decoupling_regions_{scale}.df     per pixel: lon, lat, group, SPEI pre / post / state, dLAI, p, class
##   decoupling_regions_area_{scale}.csv   area % of each class per group
##   decoupling_transition_{scale}.csv     area % of the pre -> post SPEI state transitions (All)
##   Fig_decoupling_regions_{scale}        map of the classes, transition matrix, area % by group
from Time_series_plotting import *


class Decoupling_regions(Long_LAI_time_series):
    def __init__(self):
        super().__init__('SNU', 1982, 2024)
        self.outdir = self.result2w + rf'03_response\decoupling_regions\\'
        T.mk_dir(self.outdir, force=True)
        self.spei_dff = result_root + rf'\Dataframe\SPEI_long_term\SPEI_GS_1958_2024{WIN}.df'
        self.scale = 'SPEI12'
        self.years = np.arange(1982, 2025)
        self.split = 2000
        self.wet_thr, self.dry_thr = 0.5, -0.5
        self.alpha = 0.05
        self.class_names = ['decoupled', 'coupled', 'resistant', 'not dry']
        self.class_labels = ['decoupled: dry after 2000, LAI up', 'coupled: dry after 2000, LAI down',
                             'resistant: dry after 2000, LAI no change', 'not dry after 2000']
        self.class_colors = ['#4DAC26', '#A6611A', '#E6AB02', '#D9D9D9']
        self.states = ['wet', 'normal', 'dry']
        self.groups = ['All', 'Forest', 'Shrub annual', 'Shrub perennial', 'Grass annual', 'Grass perennial', 'Savanna']

    def run(self):
        info, lai, spei = self.load_arrays()
        df = self.classify(info, lai, spei)
        self.area_table(df)
        trans = self.transition(df)
        self.figure(df, trans)

    def load_arrays(self):
        ## pixel x year arrays of GS LAI (% of mean) and GS SPEI, pixels complete 1982-2024
        df = self.load()
        sp = T.load_df(self.spei_dff)
        sp = sp[(sp['year'] >= self.years[0]) & (sp['year'] <= self.years[-1])]
        col = f'growing_{self.scale}'
        df = df.merge(sp[['pix', 'year', col]], on=['pix', 'year'], how='inner')
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type']].first()
        lai = df.pivot_table(index='pix', columns='year', values='growing_LAI_rel').reindex(
            index=info.index, columns=self.years).values
        spei = df.pivot_table(index='pix', columns='year', values=col).reindex(index=info.index, columns=self.years).values
        ok = np.isfinite(lai).all(axis=1) & np.isfinite(spei).all(axis=1)
        print(f'pixels with LAI and {self.scale} in every year {self.years[0]}-{self.years[-1]}: {int(ok.sum())} of {len(ok)}')
        return info[ok], lai[ok], spei[ok]

    def state(self, v):
        return np.where(v > self.wet_thr, 'wet', np.where(v < self.dry_thr, 'dry', 'normal'))

    def classify(self, info, lai, spei):
        pre, post = self.years < self.split, self.years >= self.split
        s_pre, s_post = spei[:, pre].mean(axis=1), spei[:, post].mean(axis=1)
        d_lai = lai[:, post].mean(axis=1) - lai[:, pre].mean(axis=1)
        p_lai = stats.ttest_ind(lai[:, post], lai[:, pre], axis=1, equal_var=False).pvalue
        dry = s_post < self.dry_thr
        cls = np.where(~dry, 'not dry',
                       np.where((d_lai > 0) & (p_lai < self.alpha), 'decoupled',
                                np.where((d_lai < 0) & (p_lai < self.alpha), 'coupled', 'resistant')))
        df = info.copy()
        masks = self.group_masks(info)
        df['group'] = ''
        for g in self.groups[1:]:
            if g in masks:
                df.loc[masks[g], 'group'] = g
        df['SPEI_pre'], df['SPEI_post'] = s_pre, s_post
        df['dSPEI'] = s_post - s_pre
        df['state_pre'], df['state_post'] = self.state(s_pre), self.state(s_post)
        df['dLAI'], df['dLAI_p'] = d_lai, p_lai
        df['class'] = cls
        T.save_df(df, self.outdir + f'decoupling_regions_{self.scale}.df')
        return df

    def area_table(self, df):
        w = df['area_weight'].values
        rows = []
        masks = self.group_masks(df)
        for g in self.groups:
            m = masks[g] if g in masks else np.ones(len(df), bool)
            row = [g, int(m.sum())]
            for c in self.class_names:
                row.append(np.sum(w[m & (df['class'].values == c)]) / np.sum(w[m]) * 100)
            dry = m & (df['class'].values != 'not dry')
            for c in self.class_names[:3]:   ## share among the pixels dry after 2000
                row.append(np.sum(w[dry & (df['class'].values == c)]) / np.sum(w[dry]) * 100 if dry.any() else np.nan)
            rows.append(row)
        tab = pd.DataFrame(rows, columns=['group', 'n_pix'] + [f'{c}_%' for c in self.class_names] +
                                         [f'{c}_%_of_dry' for c in self.class_names[:3]])
        tab.to_csv(self.outdir + f'decoupling_regions_area_{self.scale}.csv', index=False)
        print(tab.round(1).to_string())
        return tab

    def transition(self, df):
        ## area % of each pre -> post SPEI state (All pixels)
        w = df['area_weight'].values
        mat = pd.DataFrame(0., index=self.states, columns=self.states)
        for a in self.states:
            for b in self.states:
                sel = (df['state_pre'].values == a) & (df['state_post'].values == b)
                mat.loc[a, b] = np.sum(w[sel]) / np.sum(w) * 100
        mat.index.name = 'pre (1982-1999)'
        mat.to_csv(self.outdir + f'decoupling_transition_{self.scale}.csv')
        print('SPEI state transition (area %), rows 1982-1999, columns 2000-2024:')
        print(mat.round(1).to_string())
        return mat

    def figure(self, df, trans):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
        from matplotlib.colors import ListedColormap, BoundaryNorm
        font = {'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 22, 'xtick.labelsize': 20, 'ytick.labelsize': 20}
        tab = pd.read_csv(self.outdir + f'decoupling_regions_area_{self.scale}.csv').set_index('group')
        with plt.rc_context(font):
            fig = plt.figure(figsize=(20, 15))
            ## a: map of the classes
            k = pd.Series(range(len(self.class_names)), index=self.class_names).loc[df['class'].values].values.astype(float)
            arr = np.full((468, 449), np.nan)
            arr[np.array([p[0] for p in df.index]), np.array([p[1] for p in df.index])] = k
            ax = fig.add_axes([0.0, 0.40, 0.55, 0.56], projection=ccrs.PlateCarree())
            ax.imshow(arr, cmap=ListedColormap(self.class_colors),
                      norm=BoundaryNorm(np.arange(-0.5, len(self.class_names)), len(self.class_names)),
                      extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(), interpolation='nearest')
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
            ax.plot([-125, -105, -105, -125, -125], [30, 30, 45, 45, 30], color='k', lw=1.5, transform=ccrs.PlateCarree())
            ax.set_extent([-125, -102, 28, 49], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.FixedLocator(range(-120, -100, 10))
            gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
            ax.set_title(f'a  {self.scale}: dry after 2000 = mean 2000-2024 < {self.dry_thr}; '
                         f'LAI change p < {self.alpha}', loc='left', fontsize=20)
            ## b: SPEI state transition matrix
            ax = fig.add_axes([0.64, 0.52, 0.30, 0.36])
            v = trans.values
            ax.imshow(v, cmap='Oranges', vmin=0, vmax=max(v.max(), 1))
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=22,
                            color='white' if v[i, j] > v.max() * 0.6 else 'black')
            ax.set_xticks(range(3))
            ax.set_xticklabels(self.states)
            ax.set_yticks(range(3))
            ax.set_yticklabels(self.states)
            ax.set_xlabel(f'2000-2024 mean {self.scale}')
            ax.set_ylabel(f'1982-1999 mean {self.scale}')
            ax.set_title('b  state change (area %)', loc='left', fontsize=20)
            ## c: area % by group
            ax = fig.add_axes([0.20, 0.06, 0.62, 0.28])
            t = tab.reindex(self.groups)
            y = np.arange(len(t))[::-1]
            left = np.zeros(len(t))
            for c, col in zip(self.class_names, self.class_colors):
                vv = t[f'{c}_%'].values
                ax.barh(y, vv, left=left, color=col, height=0.72, edgecolor='white', linewidth=0.5)
                for yy, l0, x in zip(y, left, vv):
                    if x >= 7:
                        ax.text(l0 + x / 2, yy, f'{x:.0f}', ha='center', va='center', fontsize=18,
                                color='black' if c in ['not dry', 'resistant'] else 'white')
                left += vv
            ax.set_yticks(y)
            ax.set_yticklabels(list(t.index))
            ax.set_xlim(0, 100)
            ax.set_xlabel('Area (%)')
            ax.set_title('c', loc='left', fontsize=20, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            handles = [mpatches.Patch(color=c, label=n) for c, n in zip(self.class_colors, self.class_labels)]
            fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.11),
                       fontsize=20)
            plt.savefig(self.outdir + f'Fig_decoupling_regions_{self.scale}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_decoupling_regions_{self.scale}.pdf', bbox_inches='tight')
            plt.close()
        print('saved', f'Fig_decoupling_regions_{self.scale}')


class Decoupling_LAI_trend(Decoupling_regions):
    ## 2026-11-28: within the three classes of decoupling_regions (which already compare the level 1982-1999 vs
    ##   2000-2024), is LAI still changing during the drought period? per pixel OLS of GS LAI (% of mean) on year,
    ##   2000-2024: increasing (slope > 0, p < 0.05), declining (slope < 0, p < 0.05), no trend (p >= 0.05)
    ## checks (csv only): sign only (slope > 0 / <= 0); end in 2022 (2023 was an exceptionally wet year)
    ## output : {R2}\03_response\decoupling_regions\LAI_trend\
    ##   decoupling_LAI_trend_{scale}.df          per pixel: class, slope (% per decade), p, trend; same for 2000-2022
    ##   decoupling_LAI_trend_area_{scale}.csv    area % increasing / no trend / declining per class (3 versions)
    ##   Fig_decoupling_LAI_trend_{scale}         a-c map of the 2000-2024 trend in each class, d area % per class
    ##   decoupling_LAI_trend_veg_{scale}.csv     2026-11-28: the same per vegetation group x class (3 versions), n, median slope
    ##   Fig_decoupling_LAI_trend_veg_{scale}     2026-11-28: a-c one panel per class, bars per vegetation group (p < 0.05);
    ##                                            groups with fewer than self.min_n pixels are pale and marked with n
    def __init__(self):
        super().__init__()
        self.regions_dff = self.outdir + f'decoupling_regions_{self.scale}.df'
        self.outdir = self.outdir + rf'LAI_trend\\'
        T.mk_dir(self.outdir, force=True)
        self.dry_classes = ['decoupled', 'resistant', 'coupled']
        self.trend_names = ['increasing', 'no trend', 'declining']
        self.trend_colors = ['#1B7837', '#BDBDBD', '#B35806']
        self.check_end = 2022
        self.min_n = 100   ## vegetation group x class cells with fewer pixels are shown pale

    def run(self):
        info, lai, spei = self.load_arrays()
        reg = T.load_df(self.regions_dff)
        keep = reg.reindex(info.index)['class'].isin(self.dry_classes).values
        info, lai = info[keep], lai[keep]
        df = reg.reindex(info.index)[['lon', 'lat', 'area_weight', 'landcover_type', 'group', 'class', 'dLAI']].copy()
        for tag, end in [('', self.years[-1]), (f'_{self.check_end}', self.check_end)]:
            m = (self.years >= self.split) & (self.years <= end)
            b, p = self.seg_trend(lai[:, m], self.years[m].astype(float))
            df[f'slope{tag}'], df[f'p{tag}'] = b * 10, p
            df[f'trend{tag}'] = np.where(p >= self.alpha, 'no trend', np.where(b > 0, 'increasing', 'declining'))
        df['trend_sign'] = np.where(df['slope'] > 0, 'increasing', 'declining')
        T.save_df(df, self.outdir + f'decoupling_LAI_trend_{self.scale}.df')
        tab = self.area_table(df)
        self.figure(df, tab)
        self.veg_figure(self.veg_table(df))   ## 2026-11-28

    def seg_trend(self, y, years):
        ## OLS slope (per year) and two-sided p of every row of y on years
        x = years - years.mean()
        sxx = np.sum(x ** 2)
        b = (y - y.mean(axis=1)[:, None]) @ x / sxx
        res = y - y.mean(axis=1)[:, None] - b[:, None] * x[None, :]
        se = np.sqrt(np.sum(res ** 2, axis=1) / (len(years) - 2) / sxx)
        return b, 2 * stats.t.sf(np.abs(b / se), len(years) - 2)

    def area_table(self, df):
        w = df['area_weight'].values
        rows = []
        for version, col in [(f'{self.split}-{self.years[-1]}, p < {self.alpha}', 'trend'),
                             (f'{self.split}-{self.years[-1]}, sign only', 'trend_sign'),
                             (f'{self.split}-{self.check_end}, p < {self.alpha}', f'trend_{self.check_end}')]:
            for c in self.dry_classes:
                m = df['class'].values == c
                slope = df[f'slope_{self.check_end}' if col == f'trend_{self.check_end}' else 'slope'].values
                row = [version, c, int(m.sum()), np.median(slope[m])]
                for t in self.trend_names:
                    row.append(np.sum(w[m & (df[col].values == t)]) / np.sum(w[m]) * 100)
                rows.append(row)
        tab = pd.DataFrame(rows, columns=['version', 'class', 'n_pix', 'median_slope_%_per_decade'] +
                                         [f'{t}_%' for t in self.trend_names])
        tab.to_csv(self.outdir + f'decoupling_LAI_trend_area_{self.scale}.csv', index=False)
        print(tab.round(1).to_string())
        return tab

    def figure(self, df, tab):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
        from matplotlib.colors import ListedColormap, BoundaryNorm
        font = {'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 22, 'xtick.labelsize': 20, 'ytick.labelsize': 20}
        rows = np.array([p[0] for p in df.index])
        cols = np.array([p[1] for p in df.index])
        with plt.rc_context(font):
            fig = plt.figure(figsize=(26, 17))
            ## a-c: 2000-2024 trend of the pixels of each class (other dry pixels in pale grey)
            for kk, c in enumerate(self.dry_classes):
                arr = np.full((468, 449), np.nan)
                arr[rows, cols] = 3   ## other classes
                m = df['class'].values == c
                arr[rows[m], cols[m]] = pd.Series(range(3), index=self.trend_names).loc[df['trend'].values[m]].values
                ax = fig.add_axes([0.0 + kk * 0.33, 0.40, 0.32, 0.55], projection=ccrs.PlateCarree())
                ax.imshow(arr, cmap=ListedColormap(self.trend_colors + ['#F4F4F4']), norm=BoundaryNorm(np.arange(-0.5, 4), 4),
                          extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(), interpolation='nearest')
                ax.add_feature(cfeature.STATES, linewidth=0.3)
                ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
                ax.plot([-125, -105, -105, -125, -125], [30, 30, 45, 45, 30], color='k', lw=1.5,
                        transform=ccrs.PlateCarree())
                ax.set_extent([-125, -102, 28, 49], crs=ccrs.PlateCarree())
                gl = ax.gridlines(draw_labels=True, linewidth=0)
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = kk == 0
                gl.xlocator = mticker.FixedLocator(range(-120, -100, 10))
                gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
                ax.set_title(f'{"abc"[kk]}  {c}', loc='left', fontweight='bold')
            ## d: area % per class
            ax = fig.add_axes([0.22, 0.06, 0.56, 0.25])
            t = tab[tab['version'] == f'{self.split}-{self.years[-1]}, p < {self.alpha}'].set_index('class').reindex(
                self.dry_classes)
            y = np.arange(len(t))[::-1]
            left = np.zeros(len(t))
            for name, col in zip(self.trend_names, self.trend_colors):
                vv = t[f'{name}_%'].values
                ax.barh(y, vv, left=left, color=col, height=0.7, edgecolor='white', linewidth=0.5)
                for yy, l0, x in zip(y, left, vv):
                    if x >= 5:
                        ax.text(l0 + x / 2, yy, f'{x:.0f}', ha='center', va='center', fontsize=20,
                                color='black' if name == 'no trend' else 'white')
                left += vv
            for yy, sl in zip(y, t['median_slope_%_per_decade'].values):
                ax.text(101, yy, f'median {sl:+.1f} %/decade', va='center', fontsize=18)
            ax.set_yticks(y)
            ax.set_yticklabels(list(t.index))
            ax.set_xlim(0, 100)
            ax.set_xlabel('Area (%)')
            ax.set_title(f'd  LAI trend {self.split}-{self.years[-1]} (p < {self.alpha})', loc='left', fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            handles = [mpatches.Patch(color=c, label=n) for c, n in zip(self.trend_colors, self.trend_names)]
            handles.append(mpatches.Patch(color='#F4F4F4', label='other classes'))
            fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06),
                       fontsize=20)
            plt.savefig(self.outdir + f'Fig_decoupling_LAI_trend_{self.scale}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_decoupling_LAI_trend_{self.scale}.pdf', bbox_inches='tight')
            plt.close()
        print('saved', f'Fig_decoupling_LAI_trend_{self.scale}')

    def veg_table(self, df):
        ## area % increasing / no trend / declining per vegetation group x class, 3 versions
        w = df['area_weight'].values
        rows = []
        for version, col, scol in [(f'{self.split}-{self.years[-1]}, p < {self.alpha}', 'trend', 'slope'),
                                   (f'{self.split}-{self.years[-1]}, sign only', 'trend_sign', 'slope'),
                                   (f'{self.split}-{self.check_end}, p < {self.alpha}', f'trend_{self.check_end}',
                                    f'slope_{self.check_end}')]:
            for g in self.groups[1:]:
                for c in self.dry_classes:
                    m = (df['group'].values == g) & (df['class'].values == c)
                    if m.sum() == 0:
                        continue
                    row = [version, g, c, int(m.sum()), np.median(df[scol].values[m])]
                    for t in self.trend_names:
                        row.append(np.sum(w[m & (df[col].values == t)]) / np.sum(w[m]) * 100)
                    rows.append(row)
        tab = pd.DataFrame(rows, columns=['version', 'group', 'class', 'n_pix', 'median_slope_%_per_decade'] +
                                         [f'{t}_%' for t in self.trend_names])
        tab.to_csv(self.outdir + f'decoupling_LAI_trend_veg_{self.scale}.csv', index=False)
        print(tab[tab['version'] == tab['version'].iloc[0]].round(1).to_string())
        return tab

    def veg_figure(self, tab):
        import matplotlib.patches as mpatches
        font = {'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 22, 'xtick.labelsize': 20, 'ytick.labelsize': 20}
        t_all = tab[tab['version'] == f'{self.split}-{self.years[-1]}, p < {self.alpha}']
        with plt.rc_context(font):
            fig, axs = plt.subplots(1, 3, figsize=(27, 9), sharey=True)
            for kk, (ax, c) in enumerate(zip(axs, self.dry_classes)):
                t = t_all[t_all['class'] == c].set_index('group').reindex(self.groups[1:])
                y = np.arange(len(t))[::-1]
                left = np.zeros(len(t))
                small = (t['n_pix'].fillna(0).values < self.min_n)
                for name, col in zip(self.trend_names, self.trend_colors):
                    vv = t[f'{name}_%'].fillna(0).values
                    for yy, l0, x, sm in zip(y, left, vv, small):
                        ax.barh(yy, x, left=l0, color=col, height=0.7, edgecolor='white', linewidth=0.5,
                                alpha=0.35 if sm else 1)
                        if x >= 8 and not sm:
                            ax.text(l0 + x / 2, yy, f'{x:.0f}', ha='center', va='center', fontsize=18,
                                    color='black' if name == 'no trend' else 'white')
                    left += vv
                for yy, sl, n, sm in zip(y, t['median_slope_%_per_decade'].values, t['n_pix'].fillna(0).values, small):
                    txt = f'n = {int(n)}' if sm else f'{sl:+.1f}'
                    ax.text(102, yy, txt, va='center', fontsize=18, color='grey' if sm else 'black')
                ax.set_yticks(y)
                ax.set_yticklabels(list(t.index))
                ax.set_xlim(0, 100)
                ax.set_xlabel('Area (%)')
                ax.set_title(f'{"abc"[kk]}  {c}', loc='left', fontweight='bold')
                ax.spines[['top', 'right']].set_visible(False)
            handles = [mpatches.Patch(color=c, label=n) for c, n in zip(self.trend_colors, self.trend_names)]
            fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=20)
            fig.suptitle(f'LAI trend {self.split}-{self.years[-1]} (p < {self.alpha}) by vegetation group; '
                         f'right: median slope (% per decade)', x=0.05, ha='left', y=1.02, fontsize=22)
            plt.subplots_adjust(wspace=0.32)
            plt.savefig(self.outdir + f'Fig_decoupling_LAI_trend_veg_{self.scale}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_decoupling_LAI_trend_veg_{self.scale}.pdf', bbox_inches='tight')
            plt.close()
        print('saved', f'Fig_decoupling_LAI_trend_veg_{self.scale}')

def main():
    # Decoupling_regions().run()
    Decoupling_LAI_trend().run()   ## 2026-11-28


if __name__ == '__main__':
    main()
