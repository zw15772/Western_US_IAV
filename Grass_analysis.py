# coding='utf-8'
## Grass analysis: where and why grass greens beyond climate and composition (Model 2 unexplained term C)
## all outputs in Result2\Grass_attribution\ (one sub-folder per class)
## 2026-11-20
from growing_season_extraction import *
## 2026-11-21: larger fonts in all figures (default sizes; explicit fontsize values were also enlarged x1.6)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'legend.fontsize': 11, 'legend.title_fontsize': 11})
from scipy import ndimage
import matplotlib.ticker


class Grass_smooth_trend():
    ## grass unexplained LAI trend (Model 2 C, % per decade), raw and smoothed
    ## smoothing: for every grass pixel, area-weighted mean of C over the grass pixels in a 5 x 5 window (0.25 x 0.25 deg,
    ##   about 25 km); other vegetation types are not used; shown only where the window has >= min_n grass pixels
    ## why: one pixel's 22-year trend has about +-5 % per decade random error (larger than the signal, about +3),
    ##   independent errors cancel in the window, a regional signal does not
    ## pixel set: all grass pixels of Model 2 (no fire, land-use or other exclusion here)
    ## output: Result2\Grass_attribution\smooth_trend\
    def __init__(self):
        self.outdir = this_root + rf'Result2\05_Grass_attribution\smooth_trend\\'
        T.mk_dir(self.outdir, force=True)
        self.tag = '2003_2024'
        self.m2_dff = this_root + rf'Result2\03_response\attribution_model2\attribution_model2_{self.tag}.df'
        self.window = 5      ## pixels, 5 x 0.05 = 0.25 deg
        self.min_n = 5       ## min grass pixels in the window
        self.bounds = [-10, -6, -3, -1, 1, 3, 6, 10]   ## discrete colour steps, % per decade
        self.ny, self.nx = 468, 449                    ## template grid
        pass

    def run(self):
        df = self.load()
        res = {}
        res['raw_all'] = self.to_grid(df, 'C')
        res['smooth_all'] = self.smooth(df)
        self.save_result(df, res)
        self.plot(res, df)
        pass

    def load(self):
        df = T.load_df(self.m2_dff)
        df = df[df['landcover_type'] == 'Grass']
        print('grass pixels:', len(df))
        return df

    def to_grid(self, df, col):
        arr = np.full((self.ny, self.nx), np.nan)
        for (r, c), v in zip(df.index, df[col].values):
            arr[r, c] = v
        return arr

    def smooth(self, df):
        ## area-weighted moving-window mean of C over grass pixels, written back on the grass pixels of df
        v = self.to_grid(df, 'C')
        w = self.to_grid(df, 'area_weight')
        ok = np.isfinite(v) & np.isfinite(w)
        vw = np.where(ok, v * w, 0.)
        ww = np.where(ok, w, 0.)
        k = np.ones((self.window, self.window))
        s_vw = ndimage.convolve(vw, k, mode='constant', cval=0.)
        s_w = ndimage.convolve(ww, k, mode='constant', cval=0.)
        s_n = ndimage.convolve(ok.astype(float), k, mode='constant', cval=0.)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = s_vw / s_w
        out[~ok | (s_n < self.min_n)] = np.nan
        return out

    def save_result(self, df, res):
        rows = np.array([p[0] for p in df.index])
        cols = np.array([p[1] for p in df.index])
        out = df[['lon', 'lat', 'area_weight', 'C']].copy() if 'lon' in df.columns else df[['area_weight', 'C']].copy()
        for name, arr in res.items():
            out[f'C_{name}'] = arr[rows, cols]
            D.arr_to_tif(arr.copy(), self.outdir + f'grass_C_{name}_{self.tag}.tif')
        T.save_df(out, self.outdir + f'grass_C_smooth_{self.tag}.df')
        ## how much of the pixel-to-pixel spread is noise: SD of C before and after smoothing
        w = out['area_weight'].values
        for name in res:
            x = out[f'C_{name}'].values
            m = np.isfinite(x)
            mu = np.sum(x[m] * w[m]) / np.sum(w[m])
            sd = np.sqrt(np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m]))
            print(f'{name:13s} n={m.sum():6d}  mean={mu:+.2f}  SD={sd:.2f}  '
                  f'area C>0: {np.sum(w[m] * (x[m] > 0)) / np.sum(w[m]) * 100:.0f}%')

    def plot(self, res, df):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        from matplotlib.colors import BoundaryNorm
        cmap = plt.get_cmap('BrBG', len(self.bounds) - 1)   ## brown = browning, green = greening
        norm = BoundaryNorm(self.bounds, cmap.N)
        title = {'raw_all': 'Raw (0.05°)',
                 'smooth_all': 'Smoothed (0.25° window)'}
        fig = plt.figure(figsize=(17 * centimeter_factor, 9 * centimeter_factor))
        for k, name in enumerate(res):
            ax = fig.add_subplot(1, 2, k + 1, projection=ccrs.PlateCarree())
            im = ax.imshow(res[name], cmap=cmap, norm=norm, extent=[-124.529, -102.079, 25.579, 48.979],
                           transform=ccrs.PlateCarree(), interpolation='nearest')
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
            ax.set_extent([-125, -105, 30, 45], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.FixedLocator(range(-125, -100, 5))
            gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
            gl.xlabel_style = {'size': 9.5}
            gl.ylabel_style = {'size': 9.5}
            ax.set_title(title[name], fontsize=12)
        cax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=self.bounds, extend='both')
        cb.set_label('Grass unexplained LAI trend (% per decade)', fontsize=11)
        cb.ax.tick_params(labelsize=10.5)
        plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.18, wspace=0.12)
        plt.savefig(self.outdir + f'grass_C_raw_vs_smooth_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'grass_C_raw_vs_smooth_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


class Grass_aridity(Growing_season_trend):
    ## step 1b: is the grass unexplained trend (Model 2 C) larger in drier places? (moved from Attribution_analysis
    ##   C_aridity + C_aridity_AFG, grass only, 2026-11-20)
    ## CO2 raises water-use efficiency, so the gain is expected where water limits growth most (but not where it is
    ##   too dry to grow at all) -> hump along aridity
    ## aridity index AI = mean annual P / mean annual PET, 2003-2024 (TerraClimate ppt, pet); lower = drier
    ## C in % of the pixel's GS mean LAI per decade; C_abs = C x GS mean LAI / 100 (m2/m2 per decade), so a pattern is
    ##   not only the small denominator of dry pixels
    ## bins (UNEP-like): <0.1, 0.1-0.15, 0.15-0.2, 0.2-0.3, 0.3-0.4, 0.4-0.5, 0.5-0.65, >0.65
    ## lines: all grass pixels, and grass split by the RAP annual forb & grass trend (cheatgrass check):
    ##   AFG stable |dAFG| < 1, AFG increase dAFG >= 1, AFG decrease dAFG <= -1 (% cover per decade)
    ## mean per bin is area-weighted, 95% CI by 1 x 1 deg block bootstrap
    ## output: Result2\05_Grass_attribution\aridity\
    def __init__(self):
        super().__init__(this_root + rf'Result2\05_Grass_attribution\aridity\\')
        self.gs_dff = result_root + rf'\Dataframe\growing_season\growing_season.df'
        self.m2_dff = this_root + rf'Result2\03_response\attribution_model2\attribution_model2_{self.tag}.df'
        self.bin_edges = [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 10]
        self.afg_threshold = 1.0
        self.group_color = {'All grass': '#1F2A33'}   ## all grass only (2026-11-20)
        ## AFG split kept for later: add 'AFG stable': '#2E7D32', 'AFG increase': '#D9480F', 'AFG decrease': '#1F618D'
        self.var_list = ['C', 'C_abs', 'observed', 'A', 'B', 'RAP_AFG_trend']

    def run(self):
        df = self.load()
        tab = self.bin_table(df)
        self.plot(tab)
        pass

    def load(self):
        m2 = T.load_df(self.m2_dff)
        m2 = m2[m2['landcover_type'] == 'Grass']
        gs = T.load_df(self.gs_dff).reindex(m2.index)
        pix_list = list(m2.index)
        gse = Growing_season_extraction()
        ppt = gse.monthly_values(pix_list, data_root + rf'TerraClimate\ppt\dic\\', None)[:, 12:]   ## 2003-2024
        pet = gse.monthly_values(pix_list, data_root + rf'TerraClimate\pet\dic\\', None)[:, 12:]
        ai = np.nanmean(ppt, axis=1) / np.nanmean(pet, axis=1)
        print('grass pixels:', len(m2), '| AI median', np.nanmedian(ai).round(3),
              '| 5-95%', np.nanpercentile(ai, [5, 95]).round(3))
        df = m2[['area_weight', 'landcover_type', 'lon', 'lat', 'C', 'observed', 'A', 'B', 'RAP_AFG_trend']].copy()
        df['AI'] = ai
        df['C_abs'] = df['C'].values * gs['GS_mean_LAI'].values / 100
        d_afg = df['RAP_AFG_trend'].values
        df['AFG_group'] = np.where(np.abs(d_afg) < self.afg_threshold, 'AFG stable',
                                   np.where(d_afg >= self.afg_threshold, 'AFG increase', 'AFG decrease'))
        T.save_df(df, self.outdir + f'grass_aridity_{self.tag}.df')
        self.save_map(df, np.log10(ai), f'aridity_index_log10_grass_{self.tag}', 'log10(P / PET), 2003-2024', 1,
                      'BrBG')
        return df

    def bin_table(self, df):
        w_all = df['area_weight'].values
        b = np.digitize(df['AI'].values, self.bin_edges) - 1
        result = []
        for g in self.group_color:
            mg = np.ones(len(df), dtype=bool) if g == 'All grass' else (df['AFG_group'] == g).values
            for k in range(len(self.bin_edges) - 1):
                m = mg & (b == k) & np.isfinite(df['C'].values)
                if m.sum() < 30:
                    continue
                w = w_all[m]
                row = [g, self.bin_edges[k], self.bin_edges[k + 1], m.sum(),
                       np.sum(df['AI'].values[m] * w) / np.sum(w), np.sum(w) / np.sum(w_all[mg]) * 100]
                for name in self.var_list:
                    v = df[name].values[m]
                    lo, hi = self.block_ci(df[m], v, w)
                    row += [np.sum(v * w) / np.sum(w), lo, hi]
                result.append(row)
        cols = ['group', 'AI_low', 'AI_high', 'n_pix', 'AI_mean', 'area_%_of_group']
        for name in self.var_list:
            cols += [name, f'{name}_ci_low', f'{name}_ci_high']
        tab = pd.DataFrame(result, columns=cols)
        tab.to_csv(self.outdir + f'grass_C_aridity_bins_{self.tag}.csv', index=False)
        print(tab[['group', 'AI_low', 'AI_high', 'n_pix', 'C', 'C_ci_low', 'C_ci_high', 'C_abs']].round(3).to_string())
        return tab

    def plot(self, tab):
        ## C and C_abs per aridity bin: all grass (black) and the three AFG groups
        fig, axs = plt.subplots(1, 2, figsize=(18 * centimeter_factor, 7 * centimeter_factor))
        for ax, name, label in [(axs[0], 'C', 'Unexplained LAI trend (% per decade)'),
                                (axs[1], 'C_abs', 'Unexplained LAI trend (m$^2$ m$^{-2}$ per decade)')]:
            for g, color in self.group_color.items():
                d = tab[tab['group'] == g]
                if len(d) == 0:
                    continue
                err = np.array([d[name] - d[f'{name}_ci_low'], d[f'{name}_ci_high'] - d[name]])
                ax.errorbar(d['AI_mean'], d[name], yerr=err, marker='o', ms=3, capsize=2, color=color,
                            lw=1.8 if g == 'All grass' else 1, label=f'{g} (n={d["n_pix"].sum()})')
            ax.axhline(0, color='grey', lw=0.5)
            ax.axvspan(0.15, 0.3, color='#F2C14E', alpha=0.15, lw=0)
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))   ## plain decimals
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlabel('Aridity index P / PET (drier ←)', fontsize=13)
            ax.set_ylabel(label, fontsize=13)
            ax.tick_params(labelsize=11)
            ax.spines[['top', 'right']].set_visible(False)
        if len(self.group_color) > 1:
            axs[0].legend(fontsize=10.5, frameon=False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'grass_C_aridity_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'grass_C_aridity_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


class Grass_NAIP_sites():
    ## step 2: pick sites for a visual check with NAIP (1 m aerial photos, 2003-2024): what does the unexplained grass
    ##   greening look like on the ground?
    ## four site groups, at most one site per 2 x 2 deg block and >= 0.5 deg between any two sites:
    ##   hotspot = smoothed C in the top 10% and the pixel itself C in the top 10% of grass pixels, then split by the
    ##     observed LAI trend, because C = observed - A - B is high for two different reasons:
    ##   greening : hotspot and observed >= 10 (% per decade) -> LAI really went up, should be visible on NAIP
    ##   resisting: hotspot and observed < 5 -> LAI stayed flat although climate (mostly VPD) predicts a decline;
    ##              NAIP should show little change
    ##   browning : smoothed C and pixel C both in the bottom 10% and < 0
    ##   control  : |smoothed C| < 1, |pixel C| < 1 and |observed| < 2 (no change at all), AI within the 10-90% range
    ##              of the greening + resisting sites, block with >= 20 such pixels (not an isolated pixel); random
    ##   (percentiles over all grass pixels with smoothed C, so the rule does not depend on the colour bar)
    ##   greening / resisting / browning: blocks ranked by hotspot area, in each block the most extreme pixel that is
    ##     >= 0.5 deg from the sites already chosen
    ## outputs (Result2\05_Grass_attribution\NAIP_sites\):
    ##   grass_NAIP_sites.csv            site table (lon, lat, C, A, B, AI, AFG trend)
    ##   grass_NAIP_sites.kml            open in Google Earth Pro, historical imagery slider = NAIP in the US
    ##   grass_NAIP_sites_gee.js         paste into code.earthengine.google.com: swipe NAIP early vs late per site
    ##   grass_NAIP_sites_map.png        sites on the smoothed C map
    def __init__(self):
        self.outdir = this_root + rf'Result2\05_Grass_attribution\NAIP_sites\\'
        T.mk_dir(self.outdir, force=True)
        self.tag = '2003_2024'
        self.smooth_dff = this_root + rf'Result2\05_Grass_attribution\smooth_trend\grass_C_smooth_{self.tag}.df'
        self.aridity_dff = this_root + rf'Result2\05_Grass_attribution\aridity\grass_aridity_{self.tag}.df'
        self.block = 2.0                    ## deg, one site per block
        self.pct = 10                       ## top / bottom percent of C for greening / browning sites
        self.obs_green = 10                 ## observed LAI trend >= this: greening (% per decade)
        self.obs_resist = 5                 ## observed LAI trend < this: resisting
        self.obs_control = 2                ## |observed| < this for control sites
        self.min_dist = 0.5                 ## deg, minimum distance between two sites
        self.min_control_pix = 20           ## control block needs >= this many no-change pixels
        self.n_site = {'greening': 12, 'resisting': 6, 'browning': 6, 'control': 6}
        self.group_color = {'greening': '#003C30', 'resisting': '#35978F', 'browning': '#8C510A',
                            'control': '#7A7A7A'}
        self.group_marker = {'greening': '^', 'resisting': 's', 'browning': 'v', 'control': 'o'}
        self.res = 0.05                     ## pixel size, deg
        self.seed = 1
        self.ny, self.nx = 468, 449

    def run(self):
        df = self.load()
        sites = self.pick_sites(df)
        sites.to_csv(self.outdir + 'grass_NAIP_sites.csv', index=False)
        print(sites[['site', 'group', 'lon', 'lat', 'C_smooth', 'C_raw', 'observed', 'A', 'B', 'AI', 'RAP_AFG_trend']]
              .round(2).to_string())
        self.write_kml(sites)
        self.write_gee(sites)
        self.plot(df, sites)

    def load(self):
        sm = T.load_df(self.smooth_dff)
        ar = T.load_df(self.aridity_dff)
        df = ar[['lon', 'lat', 'area_weight', 'observed', 'A', 'B', 'AI', 'RAP_AFG_trend']].copy()
        df['C_raw'] = sm['C_raw_all'].reindex(df.index).values
        df['C_smooth'] = sm['C_smooth_all'].reindex(df.index).values
        df = df[np.isfinite(df['C_raw']) & np.isfinite(df['C_smooth'])]
        df['block'] = [f'{int(np.floor(x / self.block))}_{int(np.floor(y / self.block))}'
                       for x, y in zip(df['lon'].values, df['lat'].values)]
        print('grass pixels with smoothed C:', len(df))
        return df

    def pick_sites(self, df):
        rng = np.random.default_rng(self.seed)
        cs, cr, obs = df['C_smooth'].values, df['C_raw'].values, df['observed'].values
        s_hi, s_lo = np.percentile(cs, [100 - self.pct, self.pct])
        r_hi, r_lo = np.percentile(cr, [100 - self.pct, self.pct])
        print(f'top {self.pct}%: smoothed C >= {s_hi:.2f}, pixel C >= {r_hi:.2f} | '
              f'bottom {self.pct}%: smoothed C <= {s_lo:.2f}, pixel C <= {r_lo:.2f} (% per decade)')
        hot = (cs >= s_hi) & (cr >= r_hi)
        pools = {'greening': df[hot & (obs >= self.obs_green)],
                 'resisting': df[hot & (obs < self.obs_resist)],
                 'browning': df[(cs <= min(s_lo, 0)) & (cr <= min(r_lo, 0))]}
        out = []
        for g in ['greening', 'resisting', 'browning']:
            d = pools[g]
            area = d.groupby('block')['area_weight'].sum().sort_values(ascending=False)
            used = {r['block'] for r in out}
            n = 0
            for b in area.index:
                if n >= self.n_site[g]:
                    break
                if b in used:
                    continue
                db = d[d['block'] == b]
                order = np.argsort(-db['C_smooth'].values if g != 'browning' else db['C_smooth'].values)
                row = self.first_far(db, order, out)   ## most extreme pixel >= min_dist from chosen sites
                if row is None:
                    continue
                row['group'] = g
                row['hotspot_pixels_in_block'] = len(db)
                out.append(row)
                n += 1
            print(f'{g}: {len(d)} candidate pixels in {len(area)} blocks, {n} sites')
        ai = [r['AI'] for r in out if r['group'] in ('greening', 'resisting')]
        ai_lo, ai_hi = np.percentile(ai, [10, 90])
        d = df[(np.abs(cs) < 1) & (np.abs(cr) < 1) & (np.abs(obs) < self.obs_control) &
               (df['AI'] >= ai_lo) & (df['AI'] <= ai_hi)]
        cnt = d.groupby('block').size()
        used = {r['block'] for r in out}
        cnt = cnt[[b not in used for b in cnt.index]].sort_values(ascending=False)
        blocks = list(rng.permutation(cnt.index[cnt >= self.min_control_pix]))
        blocks += list(cnt.index[cnt < self.min_control_pix])   ## fallback: largest smaller blocks, if too few
        n = 0
        for b in blocks:
            if n >= self.n_site['control']:
                break
            db = d[d['block'] == b]
            row = self.first_far(db, rng.permutation(len(db)), out)
            if row is None:
                continue
            row['group'] = 'control'
            row['hotspot_pixels_in_block'] = len(db)
            out.append(row)
            n += 1
        print(f'control: {len(d)} candidate pixels, {int((cnt >= self.min_control_pix).sum())} blocks with '
              f'>= {self.min_control_pix}, {n} sites')
        sites = pd.DataFrame(out)
        sites['pix'] = [str(p) for p in sites.index]
        sites = sites.reset_index(drop=True)
        sites['site'] = [f'{g[0].upper()}{k + 1:02d}' for k, g in
                         zip(sites.groupby('group').cumcount(), sites['group'])]
        return sites

    def first_far(self, db, order, out):
        ## first pixel of db (in the given positional order) that is >= min_dist deg from every chosen site
        if len(out) == 0:
            return db.iloc[int(order[0])].copy()
        xy = np.array([[r['lon'], r['lat']] for r in out])
        lon, lat = db['lon'].values, db['lat'].values
        for i in order:
            if np.min(np.hypot(xy[:, 0] - lon[i], xy[:, 1] - lat[i])) >= self.min_dist:
                return db.iloc[int(i)].copy()
        return None

    def write_kml(self, sites):
        style = {'greening': 'ff303c00', 'resisting': 'ff8f9735', 'browning': 'ff0a518c',
                 'control': 'ff7a7a7a'}   ## KML colour = aabbggrr
        h = self.res / 2
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
                 '<name>Grass NAIP sites</name>']
        for g, c in style.items():
            lines.append(f'<Style id="{g}"><LineStyle><color>{c}</color><width>3</width></LineStyle>'
                         f'<PolyStyle><fill>0</fill></PolyStyle><IconStyle><color>{c}</color></IconStyle></Style>')
        for g in style:
            lines.append(f'<Folder><name>{g}</name>')
            for _, s in sites[sites['group'] == g].iterrows():
                x, y = s['lon'], s['lat']
                desc = (f'C smoothed {s["C_smooth"]:+.1f}, C pixel {s["C_raw"]:+.1f}, observed {s["observed"]:+.1f}, '
                        f'A {s["A"]:+.1f}, B {s["B"]:+.1f} (% per decade); AI {s["AI"]:.2f}; '
                        f'RAP AFG trend {s["RAP_AFG_trend"]:+.1f}')
                ring = f'{x-h},{y-h},0 {x+h},{y-h},0 {x+h},{y+h},0 {x-h},{y+h},0 {x-h},{y-h},0'
                lines.append(f'<Placemark><name>{s["site"]}</name><description>{desc}</description>'
                             f'<styleUrl>#{g}</styleUrl><MultiGeometry><Point><coordinates>{x},{y},0</coordinates>'
                             f'</Point><Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates>'
                             f'</LinearRing></outerBoundaryIs></Polygon></MultiGeometry></Placemark>')
            lines.append('</Folder>')
        lines.append('</Document></kml>')
        with open(self.outdir + 'grass_NAIP_sites.kml', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def write_gee(self, sites):
        ## Earth Engine Code Editor app: pick a site, swipe between two NAIP years (left / right)
        js_sites = ',\n'.join(
            f"  {{id: '{s['site']}', group: '{s['group']}', lon: {s['lon']:.4f}, lat: {s['lat']:.4f}, "
            f"info: 'C smooth {s['C_smooth']:+.1f} | C pixel {s['C_raw']:+.1f} | obs {s['observed']:+.1f} | "
            f"A {s['A']:+.1f} | B {s['B']:+.1f} | AI {s['AI']:.2f} | dAFG {s['RAP_AFG_trend']:+.1f}'}}"
            for _, s in sites.iterrows())
        js = f"""// Grass NAIP sites: swipe NAIP early (left) vs late (right). Generated by Grass_analysis.py
// white box = the 0.05 deg MODIS-grid pixel; yellow box = the 0.25 deg smoothing window
var sites = [
{js_sites}
];
var naip = ee.ImageCollection('USDA/NAIP/DOQQ');
var vis = {{bands: ['R', 'G', 'B'], min: 0, max: 255}};
var years = [];
for (var y = 2003; y <= 2024; y++) years.push(String(y));
var left = ui.Map(), right = ui.Map();
left.setOptions('SATELLITE'); right.setOptions('SATELLITE');
ui.Map.Linker([left, right]);
var split = ui.SplitPanel({{firstPanel: left, secondPanel: right, wipe: true, style: {{stretch: 'both'}}}});
var siteSel = ui.Select({{items: sites.map(function(s) {{return s.id + ' (' + s.group + ')';}}), placeholder: 'site'}});
var y1 = ui.Select({{items: years, value: '2006'}});
var y2 = ui.Select({{items: years, value: '2022'}});
var info = ui.Label(''), avail = ui.Label('');
var panel = ui.Panel([ui.Label('Grass NAIP sites', {{fontWeight: 'bold'}}), siteSel,
  ui.Panel([ui.Label('left'), y1, ui.Label('right'), y2], ui.Panel.Layout.flow('horizontal')), info, avail],
  null, {{width: '330px'}});
ui.root.widgets().reset([panel, split]);

function current() {{ var i = siteSel.items().indexOf(siteSel.getValue()); return i < 0 ? null : sites[i]; }}
function boxes(s) {{
  var px = ee.Geometry.Rectangle([s.lon - {self.res / 2}, s.lat - {self.res / 2}, s.lon + {self.res / 2}, s.lat + {self.res / 2}]);
  var win = ee.Geometry.Rectangle([s.lon - 0.125, s.lat - 0.125, s.lon + 0.125, s.lat + 0.125]);
  return [ui.Map.Layer(ee.Image().paint(px, 0, 2), {{palette: ['white']}}, 'pixel'),
          ui.Map.Layer(ee.Image().paint(win, 0, 2), {{palette: ['yellow']}}, 'window')];
}}
function layer(s, y) {{
  var pt = ee.Geometry.Point([s.lon, s.lat]);
  var img = naip.filterBounds(pt.buffer(3000)).filterDate(y + '-01-01', (Number(y) + 1) + '-01-01').mosaic();
  return ui.Map.Layer(img, vis, 'NAIP ' + y);
}}
function refresh() {{
  var s = current(); if (!s) return;
  left.layers().reset([layer(s, y1.getValue())].concat(boxes(s)));
  right.layers().reset([layer(s, y2.getValue())].concat(boxes(s)));
}}
siteSel.onChange(function() {{
  var s = current(); info.setValue(s.info);
  left.setCenter(s.lon, s.lat, 14);
  avail.setValue('NAIP years here: ...');
  naip.filterBounds(ee.Geometry.Point([s.lon, s.lat])).aggregate_array('system:time_start')
    .map(function(t) {{return ee.Date(t).format('YYYY-MM');}}).distinct().sort()
    .evaluate(function(l) {{avail.setValue('NAIP here: ' + l.join(', '));}});
  refresh();
}});
y1.onChange(refresh); y2.onChange(refresh);
"""
        with open(self.outdir + 'grass_NAIP_sites_gee.js', 'w', encoding='utf-8') as f:
            f.write(js)

    def plot(self, df, sites):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.colors import BoundaryNorm
        bounds = [-10, -6, -3, -1, 1, 3, 6, 10]
        arr = np.full((self.ny, self.nx), np.nan)
        for (r, c), v in zip(df.index, df['C_smooth'].values):
            arr[r, c] = v
        cmap = plt.get_cmap('BrBG', len(bounds) - 1)
        fig = plt.figure(figsize=(12 * centimeter_factor, 10 * centimeter_factor))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        im = ax.imshow(arr, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), extent=[-124.529, -102.079, 25.579, 48.979],
                       transform=ccrs.PlateCarree(), interpolation='nearest', alpha=0.6)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.set_extent([-125, -102, 30, 49], crs=ccrs.PlateCarree())
        for g, color in self.group_color.items():
            d = sites[sites['group'] == g]
            ax.scatter(d['lon'], d['lat'], marker=self.group_marker[g], s=28, facecolor=color, edgecolor='white', lw=0.6,
                       transform=ccrs.PlateCarree(), label=g, zorder=5)
            for _, s in d.iterrows():
                ax.text(s['lon'] + 0.25, s['lat'] + 0.15, s['site'], fontsize=9, transform=ccrs.PlateCarree(),
                        zorder=6)
        ax.legend(fontsize=10.5, frameon=False, loc='lower left')
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', ticks=bounds, extend='both', fraction=0.04, pad=0.04)
        cb.set_label('Grass unexplained LAI trend, smoothed (% per decade)', fontsize=11)
        cb.ax.tick_params(labelsize=10.5)
        plt.savefig(self.outdir + 'grass_NAIP_sites_map.png', dpi=300, bbox_inches='tight')
        plt.close()


class Cropland_threshold_C(Growing_season_trend):
    ## step 3: how much of the unexplained trend C comes from cropland inside the pixels? (2026-11-21)
    ## why: NAIP site G08 showed centre-pivot fields inside a "grass" pixel; the current mask removes only pixels with
    ##   > 50% cropland, so pixels with a few fields stay in and their crop / fallow switches end up in C
    ## cropland share = nlcd_cropland_max = max over Annual NLCD years (2003, 2004, 2024) of the 0.05 deg pixel share
    ##   of 81 pasture/hay + 82 cultivated crops (Data_processing_NLCD_cropland)
    ## masks (keep pixels with cropland share): no extra mask, < 10%, < 5%, < 1%, = 0
    ## per mask and vegetation type: area-weighted mean of observed, A, B, C (% per decade), 95% CI by 1 x 1 deg block
    ##   bootstrap, and the area kept (% of the type's Model 2 area)
    ## also: cropland share of the NAIP sites
    ## output: Result2\05_Grass_attribution\cropland_threshold\
    def __init__(self):
        super().__init__(this_root + rf'Result2\05_Grass_attribution\cropland_threshold\\')
        self.m2_dff = this_root + rf'Result2\03_response\attribution_model2\attribution_model2_{self.tag}.df'
        self.crop_dff = result_root + rf'\Dataframe\NLCD\NLCD_cropland.df'
        self.site_f = this_root + rf'Result2\05_Grass_attribution\NAIP_sites\grass_NAIP_sites.csv'
        self.mask_list = [('no extra mask', None), ('< 10%', 0.10), ('< 5%', 0.05), ('< 1%', 0.01), ('= 0', 0.)]
        self.var_list = ['observed', 'A', 'B', 'C']
        self.color = {'All': '#1F2A33', 'Forest': '#2E7D32', 'Shrub': '#B8860B', 'Grass': '#D9480F',
                      'Savanna': '#7B5EA7'}

    def run(self):
        df = self.load()
        tab = self.table_masks(df)
        self.plot(tab)
        self.site_check(df)
        pass

    def load(self):
        df = T.load_df(self.m2_dff)
        df = df[df['landcover_type'].isin(self.class_list[1:])]
        crop = T.load_df(self.crop_dff)
        df['cropland'] = crop['nlcd_cropland_max'].reindex(df.index).values
        n_nan = int(np.isnan(df['cropland'].values).sum())
        print('Model 2 pixels:', len(df), '| without NLCD (kept, share set to 0):', n_nan)
        df['cropland'] = df['cropland'].fillna(0.)
        for lc in self.class_list[1:]:
            v = df.loc[df['landcover_type'] == lc, 'cropland'].values
            print(f'{lc:8s} pixels with any cropland: {np.mean(v > 0) * 100:5.1f}% | >= 1%: {np.mean(v >= 0.01) * 100:5.1f}%'
                  f' | >= 5%: {np.mean(v >= 0.05) * 100:5.1f}% | >= 10%: {np.mean(v >= 0.10) * 100:5.1f}%')
        T.save_df(df[['lon', 'lat', 'area_weight', 'landcover_type', 'cropland'] + self.var_list],
                  self.outdir + f'cropland_C_{self.tag}.df')
        return df

    def keep(self, df, thr):
        v = df['cropland'].values
        if thr is None:
            return np.ones(len(df), dtype=bool)
        return v <= 0 if thr == 0 else v < thr

    def table_masks(self, df):
        result = []
        for mask, thr in self.mask_list:
            mk = self.keep(df, thr)
            for lc in self.class_list:
                ml = np.ones(len(df), dtype=bool) if lc == 'All' else (df['landcover_type'] == lc).values
                m = mk & ml
                w = df['area_weight'].values[m]
                row = [mask, lc, int(m.sum()), np.sum(w) / np.sum(df['area_weight'].values[ml]) * 100]
                for var in self.var_list:
                    v = df[var].values[m]
                    lo, hi = self.block_ci(df[m], v, w)
                    row += [np.sum(v * w) / np.sum(w), lo, hi]
                result.append(row)
        cols = ['mask', 'class', 'n_pix', 'area_kept_%']
        for var in self.var_list:
            cols += [var, f'{var}_ci_low', f'{var}_ci_high']
        tab = pd.DataFrame(result, columns=cols)
        tab.to_csv(self.outdir + f'cropland_threshold_C_{self.tag}.csv', index=False)
        print(tab[['mask', 'class', 'n_pix', 'area_kept_%', 'observed', 'C', 'C_ci_low', 'C_ci_high']]
              .round(2).to_string())
        return tab

    def plot(self, tab):
        ## left: C per mask with 95% CI, one line per vegetation type; right: area kept
        x = np.arange(len(self.mask_list))
        labels = [m for m, _ in self.mask_list]
        fig, axs = plt.subplots(1, 2, figsize=(18 * centimeter_factor, 7.5 * centimeter_factor))
        for k, lc in enumerate(self.class_list):
            d = tab[tab['class'] == lc].set_index('mask').reindex(labels)
            off = (k - 2) * 0.06
            err = np.array([d['C'] - d['C_ci_low'], d['C_ci_high'] - d['C']])
            axs[0].errorbar(x + off, d['C'], yerr=err, marker='o', ms=3, capsize=2, color=self.color[lc],
                            lw=2 if lc in ('Grass', 'All') else 1, label=lc)
            axs[1].plot(x, d['area_kept_%'], marker='o', ms=3, color=self.color[lc],
                        lw=2 if lc in ('Grass', 'All') else 1, label=lc)
        axs[0].axhline(0, color='grey', lw=0.5)
        axs[0].set_ylabel('Unexplained LAI trend C (% per decade)', fontsize=13)
        axs[1].set_ylabel('Area kept (% of Model 2 area)', fontsize=13)
        axs[1].set_ylim(0, 105)
        for ax in axs:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_xlabel('Cropland share allowed in a pixel (NLCD 81 + 82)', fontsize=13)
            ax.tick_params(labelsize=11)
            ax.spines[['top', 'right']].set_visible(False)
        axs[0].legend(fontsize=10.5, frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(self.outdir + f'cropland_threshold_C_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'cropland_threshold_C_{self.tag}.pdf', bbox_inches='tight')
        plt.close()

    def site_check(self, df):
        ## cropland share of the NAIP sites (G08 should be high)
        import ast
        if not isfile(self.site_f):
            return
        s = pd.read_csv(self.site_f)
        pix = [tuple(ast.literal_eval(p)) for p in s['pix']]
        s['cropland_%'] = df['cropland'].reindex(pd.Index(pix, tupleize_cols=False)).values * 100
        s[['site', 'group', 'lon', 'lat', 'observed', 'C_raw', 'cropland_%']].round(2) \
            .to_csv(self.outdir + 'NAIP_sites_cropland.csv', index=False)
        print(s[['site', 'group', 'observed', 'C_raw', 'cropland_%']].round(1).to_string())


class C_by_herb_cover(Growing_season_trend):
    ## step 4: LAI trend and its components along the herbaceous composition, before calling the greening
    ##   "perennial" (2026-11-21); descriptive, no annual / perennial threshold
    ## pixels: Grass and Shrub of Model2_annual_perennial (NLCD cropland < 5% mask)
    ## x axes (RAP, 2003-2024 mean cover of the pixel):
    ##   annual share = AFG / (AFG + PFG)   (0 = all perennial, 1 = all annual), bins 0-0.1-0.2-0.35-0.5-0.65-1
    ##   PFG cover (%)                     bins 0-5-10-20-30-45-100
    ## per bin: area-weighted mean of observed (model-free), A, B, C of the model with beta per annual / perennial
    ##   (C_new) and of the standard Model 2 (C_standard), 95% CI by 1 x 1 deg block bootstrap, area share
    ## note: C_new uses a different beta below / above annual share 0.5, so it can jump there; observed and
    ##   C_standard do not depend on the split
    ## output: Result2\05_Grass_attribution\herb_cover\
    def __init__(self):
        super().__init__(this_root + rf'Result2\05_Grass_attribution\herb_cover\\')
        self.cmp_dff = this_root + \
            rf'Result2\03_response\model2_annual_perennial\model2_annual_perennial_compare_{self.tag}.df'
        self.lc_list = ['Grass', 'Shrub']
        self.axis_dic = {'annual_share': ([0, 0.1, 0.2, 0.35, 0.5, 0.65, 1.0001],
                                          'Annual share of herbaceous cover, AFG / (AFG + PFG)'),
                         'PFG_mean': ([0, 5, 10, 20, 30, 45, 100], 'Perennial forb & grass cover, RAP mean (%)')}
        self.var_dic = {'observed': ('Observed', '#2E7D32', '-'), 'A_new': ('A climate', '#C0392B', '-'),
                        'B_new': ('B composition', '#8E44AD', '-'), 'C_new': ('C unexplained', '#1F2A33', '-'),
                        'C_standard': ('C, standard Model 2', '#9E9E9E', '--')}

    def run(self):
        df = self.load()
        tab = self.bin_table(df)
        self.plot(tab)
        pass

    def load(self):
        df = T.load_df(self.cmp_dff)
        df = df[df['landcover_type'].isin(self.lc_list)].copy()
        with np.errstate(invalid='ignore', divide='ignore'):
            df['annual_share'] = df['AFG_mean'] / (df['AFG_mean'] + df['PFG_mean'])
        print('pixels:', len(df), '| no herbaceous cover:', int(df['annual_share'].isna().sum()))
        return df

    def bin_table(self, df):
        result = []
        for lc in self.lc_list:
            m_lc = (df['landcover_type'] == lc).values
            w_lc = df['area_weight'].values[m_lc].sum()
            for xname, (edges, _) in self.axis_dic.items():
                b = np.digitize(df[xname].values, edges) - 1
                for k in range(len(edges) - 1):
                    m = m_lc & (b == k)
                    if m.sum() < 30:
                        continue
                    w = df['area_weight'].values[m]
                    row = [lc, xname, edges[k], min(edges[k + 1], edges[-1]), int(m.sum()),
                           np.sum(w) / w_lc * 100, np.sum(df[xname].values[m] * w) / np.sum(w)]
                    for var in self.var_dic:
                        v = df[var].values[m]
                        lo, hi = self.block_ci(df[m], v, w)
                        row += [np.sum(v * w) / np.sum(w), lo, hi]
                    result.append(row)
        cols = ['class', 'x', 'x_low', 'x_high', 'n_pix', 'area_%_of_class', 'x_mean']
        for var in self.var_dic:
            cols += [var, f'{var}_ci_low', f'{var}_ci_high']
        tab = pd.DataFrame(result, columns=cols)
        tab.to_csv(self.outdir + f'LAI_trend_by_herb_cover_{self.tag}.csv', index=False)
        print(tab[['class', 'x', 'x_low', 'x_high', 'n_pix', 'area_%_of_class'] + list(self.var_dic)]
              .round(2).to_string())
        return tab

    def plot(self, tab):
        ## rows: Grass, Shrub; columns: annual share, PFG cover; bars below = area share of the bin
        fig, axs = plt.subplots(2, 2, figsize=(18 * centimeter_factor, 13 * centimeter_factor), sharey='row')
        for i, lc in enumerate(self.lc_list):
            for j, (xname, (_, xlabel)) in enumerate(self.axis_dic.items()):
                ax = axs[i, j]
                d = tab[(tab['class'] == lc) & (tab['x'] == xname)]
                for var, (label, color, ls) in self.var_dic.items():
                    err = np.array([d[var] - d[f'{var}_ci_low'], d[f'{var}_ci_high'] - d[var]])
                    ax.errorbar(d['x_mean'], d[var], yerr=err, marker='o', ms=2.5, capsize=1.5, color=color,
                                ls=ls, lw=1.6 if var in ('observed', 'C_new') else 1, label=label)
                ax.axhline(0, color='grey', lw=0.5)
                if xname == 'annual_share':
                    ax.axvline(0.5, color='grey', lw=0.5, ls=':')
                ax2 = ax.twinx()
                ax2.bar(d['x_mean'], d['area_%_of_class'], width=(d['x_high'] - d['x_low']) * 0.8,
                        color='#D9D9D9', alpha=0.5, zorder=0)
                ax2.set_ylim(0, 100)
                ax2.tick_params(labelsize=9.5)
                if j == 1:
                    ax2.set_ylabel('Area (% of class)', fontsize=11)
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
                ax.set_title(lc, fontsize=13)
                ax.set_xlabel(xlabel, fontsize=11)
                ax.tick_params(labelsize=11)
            axs[i, 0].set_ylabel('LAI trend (% per decade)', fontsize=13)
        axs[0, 0].legend(fontsize=9.5, frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(self.outdir + f'LAI_trend_by_herb_cover_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'LAI_trend_by_herb_cover_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


class Observed_trend_by_herb_cover(Growing_season_trend):
    ## observed LAI trend (model-free) along the herbaceous composition at the START and at the END of the record
    ##   (2026-11-21); answers: did grasslands that already had much perennial grass green more?
    ## pixels: Grass and Shrub of Model2_annual_perennial (NLCD cropland < 5% mask); observed = OLS trend of GS LAI
    ##   (% of the pixel mean per decade, 2003-2024), the same as Fig 3 and Model 2
    ## composition from RAP yearly cover (GS.df), mean over the years of self.period_dic:
    ##   now (2026-11-21): one line, 2003-2024 mean cover (>= 18 RAP years)
    ##   option kept below: early = 2003-2006 / late = 2021-2024 (>= 2 RAP years); early vs late gave the same
    ##   pattern for grass (2 -> 4 % per decade from PFG < 10% to PFG 30-45%)
    ## x axes: annual share AFG / (AFG + PFG), bins 0-0.1-0.2-0.35-0.5-0.65-1; PFG cover (%), bins 0-5-10-20-30-45-100
    ## per bin: area-weighted mean observed trend, 95% CI (1 x 1 deg block bootstrap), n and area share
    ## output: Result2\05_Grass_attribution\herb_cover\
    def __init__(self):
        super().__init__(this_root + rf'Result2\05_Grass_attribution\herb_cover\\')
        self.cmp_dff = this_root + \
            rf'Result2\03_response\model2_annual_perennial\model2_annual_perennial_compare_{self.tag}.df'
        self.lc_list = ['Grass', 'Shrub']
        self.period_dic = {'mean': (list(range(2003, 2025)), '2003-2024 mean', '-')}
        # self.period_dic = {'early': ([2003, 2004, 2005, 2006], '2003-2006', '-'),
        #                    'late': ([2021, 2022, 2023, 2024], '2021-2024', '--')}
        self.axis_dic = {'annual_share': ([0, 0.1, 0.2, 0.35, 0.5, 0.65, 1.0001],
                                          'Annual share of herbaceous cover, AFG / (AFG + PFG)'),
                         'PFG': ([0, 5, 10, 20, 30, 45, 100], 'Perennial forb & grass cover, RAP (%)')}
        self.color = {'Grass': '#D9480F', 'Shrub': '#7B5EA7'}

    def run(self):
        df = self.load()
        tab = self.bin_table(df)
        self.plot(tab)
        pass

    def load(self):
        df = T.load_df(self.cmp_dff)
        df = df[df['landcover_type'].isin(self.lc_list)][['lon', 'lat', 'area_weight', 'landcover_type', 'observed']]
        gs = T.load_df(self.dff)
        afg = self.matrix(gs, df, 'RAP_AFG')
        pfg = self.matrix(gs, df, 'RAP_PFG')
        del gs
        for p, (years, _, _) in self.period_dic.items():
            col = [self.year_list.index(y) for y in years]
            ok = np.isfinite(afg[:, col]).sum(axis=1) >= (2 if len(years) <= 4 else 18)
            with np.errstate(invalid='ignore', divide='ignore'):
                a = np.where(ok, np.nanmean(afg[:, col], axis=1), np.nan)
                f = np.where(ok, np.nanmean(pfg[:, col], axis=1), np.nan)
                df[f'PFG_{p}'] = f
                df[f'annual_share_{p}'] = a / (a + f)
            print(f'{p}: pixels with RAP cover {int(np.isfinite(f).sum())} / {len(df)}')
        T.save_df(df, self.outdir + f'observed_trend_by_herb_cover_{self.tag}.df')
        return df

    def bin_table(self, df):
        result = []
        for lc in self.lc_list:
            m_lc = (df['landcover_type'] == lc).values
            w_lc = df['area_weight'].values[m_lc].sum()
            for p in self.period_dic:
                for xname, (edges, _) in self.axis_dic.items():
                    x = df[f'{xname}_{p}'].values
                    b = np.digitize(x, edges) - 1
                    for k in range(len(edges) - 1):
                        m = m_lc & (b == k) & np.isfinite(df['observed'].values)
                        if m.sum() < 30:
                            continue
                        w = df['area_weight'].values[m]
                        v = df['observed'].values[m]
                        lo, hi = self.block_ci(df[m], v, w)
                        result.append([lc, p, xname, edges[k], min(edges[k + 1], edges[-1]), int(m.sum()),
                                       np.sum(w) / w_lc * 100, np.sum(x[m] * w) / np.sum(w),
                                       np.sum(v * w) / np.sum(w), lo, hi])
        tab = pd.DataFrame(result, columns=['class', 'period', 'x', 'x_low', 'x_high', 'n_pix', 'area_%_of_class',
                                            'x_mean', 'observed', 'observed_ci_low', 'observed_ci_high'])
        tab.to_csv(self.outdir + f'observed_trend_by_herb_cover_{self.tag}.csv', index=False)
        print(tab.round(2).to_string())
        return tab

    def plot(self, tab):
        ## rows: Grass, Shrub; columns: annual share, PFG cover; solid = early cover, dashed = late cover
        fig, axs = plt.subplots(2, 2, figsize=(17 * centimeter_factor, 12 * centimeter_factor), sharey='row')
        for i, lc in enumerate(self.lc_list):
            for j, (xname, (_, xlabel)) in enumerate(self.axis_dic.items()):
                ax = axs[i, j]
                for p, (_, plabel, ls) in self.period_dic.items():
                    d = tab[(tab['class'] == lc) & (tab['period'] == p) & (tab['x'] == xname)]
                    err = np.array([d['observed'] - d['observed_ci_low'], d['observed_ci_high'] - d['observed']])
                    ax.errorbar(d['x_mean'], d['observed'], yerr=err, marker='o' if p != 'late' else 's', ms=3,
                                capsize=2, color=self.color[lc], ls=ls, lw=1.6 if p != 'late' else 1,
                                alpha=1 if p != 'late' else 0.7, label=f'cover {plabel}')
                ax.axhline(0, color='grey', lw=0.5)
                ax.set_title(lc, fontsize=13)
                ax.set_xlabel(xlabel, fontsize=11)
                ax.tick_params(labelsize=11)
                ax.spines[['top', 'right']].set_visible(False)
            axs[i, 0].set_ylabel('Observed LAI trend (% per decade)', fontsize=13)
            if len(self.period_dic) > 1:
                axs[i, 0].legend(fontsize=10.5, frameon=False)
        plt.tight_layout()
        plt.savefig(self.outdir + f'observed_trend_by_herb_cover_{self.tag}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.outdir + f'observed_trend_by_herb_cover_{self.tag}.pdf', bbox_inches='tight')
        plt.close()


def main():
    ## step 1: where grass greens beyond climate and composition
    # Grass_smooth_trend().run()   ## 1a: raw vs smoothed (0.25 deg) unexplained trend maps (done)
    # Grass_aridity().run()        ## 1b: unexplained trend along aridity (done)
    ## step 2: look at the hotspots on NAIP
    # Grass_NAIP_sites().run()     ## 2a: pick greening / resisting / browning / control sites, write csv, kml and GEE swipe app (done)
    ## step 3: cropland inside the pixels (needs Data_processing_NLCD_cropland first)
    # Cropland_threshold_C().run()  ## 3a: C under stricter cropland masks, all vegetation types
    ## step 4: trend components along the herbaceous composition (needs Model2_annual_perennial output)
    # C_by_herb_cover().run()       ## 4a: observed, A, B, C by annual share and by PFG cover, Grass and Shrub (done)
    Observed_trend_by_herb_cover().run()   ## 4b: observed trend by cover of 2003-2006 and of 2021-2024


if __name__ == '__main__':
    main()
