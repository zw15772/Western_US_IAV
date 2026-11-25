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


class PFG_change_vs_LAI_trend(Growing_season_trend):
    ## long-term greening of grass: perennial grass expansion or more leaf at the same cover? (2026-11-21)
    ## years : 1986-2024 (SNU LAI and RAP overlap)
    ## pixels: Grass pixels of the MODIS figures (model2_annual_perennial_compare; NLCD cropland < 5%) with SNU GS LAI,
    ##         RAP PFG and growing-season SPEI12 in every year 1986-2024
    ## per pixel (OLS, not detrended):
    ##   LAI_trend  : SNU GS LAI (MODIS growing-season window), % of the pixel's 1986-2024 mean per decade
    ##   PFG_trend  : RAP perennial forb & grass cover, percentage points per decade
    ##   SPEI_trend : growing-season SPEI12 per decade (drying = negative)
    ##   PFG_init   : mean PFG cover 1986-1990
    ## bins of PFG_trend (pp per decade): area-weighted mean LAI_trend, 95% CI (1 x 1 deg block bootstrap)
    ## regression LAI_trend = a + b PFG_trend (area-weighted): a = LAI trend where PFG cover did not change,
    ##   b = extra LAI trend per 1 pp per decade of PFG increase; 95% CI by the same block bootstrap
    ## stable cover: |PFG_trend| < 0.5 pp per decade, mean LAI trend with 95% CI
    ## strata (similar drying and similar initial PFG): terciles of SPEI_trend x terciles of PFG_init
    ## output: Result2\05_Grass_attribution\PFG_change_SNU\ :
    ##   PFG_change_vs_LAI_trend_{tag}.png (all grass), _strata.png (3 x 3), .csv (bins), _fit.csv (a, b, stable), .df
    def __init__(self):
        super().__init__(this_root + rf'{R2}\05_Grass_attribution\PFG_change_SNU\\')   ## 2026-11-23: window_config
        self.year_list = list(range(1986, 2025))
        self.tag = f'{self.year_list[0]}_{self.year_list[-1]}'
        self.cmp_dff = this_root + \
            rf'Result2\03_response\model2_annual_perennial\model2_annual_perennial_compare_2003_2024.df'
        self.snu_dff = result_root + rf'\Dataframe\growing_season\GS_SNU{WIN}.df'
        self.rap_dff = result_root + rf'\Dataframe\RAP\RAP_cover.df'
        self.spei_dff = result_root + rf'\Dataframe\SPEI_long_term\SPEI_GS_1958_2024{WIN}.df'
        self.bin_edges = [-100, -2, -1, -0.5, 0.5, 1, 2, 4, 100]   ## PFG trend, pp per decade
        self.stable = 0.5   ## |PFG trend| < 0.5 pp per decade = stable cover
        self.n_min = 30
        ## large fonts for these figures (readable on a slide, 2026-11-21)
        self.big_font = {'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15,
                         'ytick.labelsize': 15, 'legend.fontsize': 15}

    def run(self):
        df = self.load()
        tab, fit = self.bin_fit(df)
        self.plot_all(df, tab, fit)
        self.plot_strata(tab, fit)
        self.plot_maps(df)   ## 2026-11-21: pixel maps of the LAI and PFG trends
        pass

    def load(self):
        cmp = T.load_df(self.cmp_dff)
        info = cmp[cmp['landcover_type'] == 'Grass'][['lon', 'lat', 'area_weight', 'landcover_type']]
        pix_set = set(info.index)
        snu = T.load_df(self.snu_dff)
        snu = snu[snu['pix'].isin(pix_set)]
        lai = self.matrix(snu, info, 'GS_LAI', relative=True)   ## % of the pixel's 1986-2024 mean
        del snu
        rap = T.load_df(self.rap_dff)
        rap = rap[rap['pix'].isin(pix_set)]
        pfg = self.matrix(rap, info, 'RAP_PFG')
        del rap
        spei = T.load_df(self.spei_dff)
        spei = spei[spei['pix'].isin(pix_set)]
        spei12 = self.matrix(spei, info, 'growing_SPEI12')
        del spei
        ok = np.isfinite(lai).all(axis=1) & np.isfinite(pfg).all(axis=1) & np.isfinite(spei12).all(axis=1)
        print('grass pixels:', len(info), '| with all years of SNU LAI, RAP PFG and SPEI12:', int(ok.sum()))
        df = info[ok].copy()
        df['LAI_trend'], df['LAI_trend_p'] = self.trend(lai[ok])
        df['PFG_trend'], df['PFG_trend_p'] = self.trend(pfg[ok])
        df['SPEI_trend'] = self.trend(spei12[ok])[0]
        df['PFG_init'] = np.mean(pfg[ok][:, :5], axis=1)   ## 1986-1990
        df['PFG_mean'] = np.mean(pfg[ok], axis=1)
        df['dry_tercile'] = pd.qcut(df['SPEI_trend'], 3, labels=['strong drying', 'medium drying', 'weak drying'])
        df['init_tercile'] = pd.qcut(df['PFG_init'], 3, labels=['low PFG 1986-90', 'mid PFG 1986-90',
                                                               'high PFG 1986-90'])
        print(df[['LAI_trend', 'PFG_trend', 'SPEI_trend', 'PFG_init']].describe().round(2).to_string())
        T.save_df(df, self.outdir + f'PFG_change_vs_LAI_trend_{self.tag}.df')
        return df

    def boot_fit(self, d, n_boot=200):
        ## area-weighted LS of LAI_trend on PFG_trend; 95% CI by 1 x 1 deg block bootstrap
        def wls(x, y, w):
            X = np.column_stack([np.ones_like(x), x])
            W = np.sqrt(w)[:, None]
            return np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)[0]
        x, y, w = d['PFG_trend'].values, d['LAI_trend'].values, d['area_weight'].values
        ab = wls(x, y, w)
        block = (np.floor(d['lon'].values) * 1000 + np.floor(d['lat'].values)).astype(int)
        ids = np.unique(block)
        idx = {b: np.where(block == b)[0] for b in ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(n_boot):
            pick = np.concatenate([idx[b] for b in rng.choice(ids, size=len(ids))])
            boot.append(wls(x[pick], y[pick], w[pick]))
        lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
        return ab, lo, hi

    def bin_fit(self, df):
        strata = [('All grass', np.ones(len(df), dtype=bool))]
        for dry in df['dry_tercile'].cat.categories:
            for ini in df['init_tercile'].cat.categories:
                strata.append((f'{dry} | {ini}', ((df['dry_tercile'] == dry) & (df['init_tercile'] == ini)).values))
        b = np.digitize(df['PFG_trend'].values, self.bin_edges) - 1
        rows, fit_rows = [], []
        for name, ms in strata:
            for k in range(len(self.bin_edges) - 1):
                m = ms & (b == k)
                if m.sum() < self.n_min:
                    continue
                d = df[m]
                w = d['area_weight'].values
                v = d['LAI_trend'].values
                lo, hi = self.block_ci(d, v, w)
                rows.append([name, self.bin_edges[k], self.bin_edges[k + 1], int(m.sum()),
                             np.sum(d['PFG_trend'].values * w) / np.sum(w), np.sum(v * w) / np.sum(w), lo, hi])
            d = df[ms]
            if len(d) < self.n_min:
                continue
            st = d[np.abs(d['PFG_trend'].values) < self.stable]
            stable = [np.nan, np.nan, np.nan, len(st)]
            if len(st) >= self.n_min:
                w = st['area_weight'].values
                stable = [np.sum(st['LAI_trend'].values * w) / np.sum(w)] + \
                         list(self.block_ci(st, st['LAI_trend'].values, w)) + [len(st)]
            (a, bb), lo, hi = self.boot_fit(d)
            w = d['area_weight'].values
            fit_rows.append([name, len(d), np.sum(d['PFG_trend'].values * w) / np.sum(w),
                             np.sum(d['LAI_trend'].values * w) / np.sum(w),
                             np.sum(d['SPEI_trend'].values * w) / np.sum(w), a, lo[0], hi[0], bb, lo[1], hi[1]]
                            + stable)
        tab = pd.DataFrame(rows, columns=['stratum', 'bin_low', 'bin_high', 'n_pix', 'PFG_trend_mean', 'LAI_trend',
                                          'ci_low', 'ci_high'])
        fit = pd.DataFrame(fit_rows, columns=['stratum', 'n_pix', 'PFG_trend_mean', 'LAI_trend_mean', 'SPEI_trend_mean',
                                              'a_LAI_trend_at_PFG0', 'a_ci_low', 'a_ci_high',
                                              'b_per_pp_PFG', 'b_ci_low', 'b_ci_high',
                                              'stable_LAI_trend', 'stable_ci_low', 'stable_ci_high', 'stable_n'])
        tab.to_csv(self.outdir + f'PFG_change_vs_LAI_trend_{self.tag}.csv', index=False)
        fit.to_csv(self.outdir + f'PFG_change_vs_LAI_trend_{self.tag}_fit.csv', index=False)
        print(fit.round(2).to_string())
        return tab, fit

    def draw(self, ax, d, f):
        ## binned means with 95% CI, fitted line (dashed), grey band = stable cover
        err = np.array([d['LAI_trend'] - d['ci_low'], d['ci_high'] - d['LAI_trend']])
        ax.errorbar(d['PFG_trend_mean'], d['LAI_trend'], yerr=err, marker='o', ms=7, capsize=4, color='#2E7D32',
                    lw=2.2)
        if len(d):
            xx = np.linspace(d['PFG_trend_mean'].min(), d['PFG_trend_mean'].max(), 10)
            ax.plot(xx, f['a_LAI_trend_at_PFG0'] + f['b_per_pp_PFG'] * xx, color='k', lw=1, ls='--')
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        ax.axvspan(-self.stable, self.stable, color='#BDBDBD', alpha=0.3, lw=0)
        ax.spines[['top', 'right']].set_visible(False)

    def plot_all(self, df, tab, fit):
        ## left: binned LAI trend vs PFG trend (all grass); right: area histogram of the PFG trend
        ## 2026-11-21: compact figure, large fonts (readable on a slide)
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [3, 2]})
            d = tab[tab['stratum'] == 'All grass']
            f = fit[fit['stratum'] == 'All grass'].iloc[0]
            self.draw(axs[0], d, f)
            axs[0].text(0.02, 0.98, f"stable PFG: {f['stable_LAI_trend']:+.1f} "
                                    f"[{f['stable_ci_low']:.1f}, {f['stable_ci_high']:.1f}]\n"
                                    f"b = {f['b_per_pp_PFG']:+.2f} [{f['b_ci_low']:.2f}, {f['b_ci_high']:.2f}] per pp",
                        transform=axs[0].transAxes, va='top', fontsize=15)
            axs[0].set_xlabel('PFG cover trend (pp / decade)')
            axs[0].set_ylabel('LAI trend (% / decade)')
            axs[0].set_title(f'Grass, SNU 1986-2024 (n = {int(f["n_pix"])})')
            x = np.clip(df['PFG_trend'].values, -6, 8)
            axs[1].hist(x, bins=np.arange(-6, 8.5, 0.5),
                        weights=df['area_weight'].values / df['area_weight'].sum() * 100, color='#8FB8DE',
                        edgecolor='white')
            axs[1].axvspan(-self.stable, self.stable, color='#BDBDBD', alpha=0.3, lw=0)
            axs[1].set_xlabel('PFG cover trend (pp / decade)')
            axs[1].set_ylabel('Area (%)')
            axs[1].spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + f'PFG_change_vs_LAI_trend_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'PFG_change_vs_LAI_trend_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def plot_maps(self, df):
        ## pixel-level maps, grass only (2026-11-21): a) SNU GS LAI trend (% / decade), b) RAP PFG cover trend
        ##   (pp / decade), dots = p < 0.05 (every 3rd pixel); c) four classes from significance (p < 0.05):
        ##   1 sig. greening + sig. PFG increase, 2 sig. greening without sig. PFG increase (stable or declining),
        ##   3 LAI n.s., 4 sig. browning; area % of each class in the legend
        ##   (2026-11-21: was the four sign combinations LAI +/- x PFG +/-)
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap, BoundaryNorm
        rows = np.array([q[0] for q in df.index])
        cols = np.array([q[1] for q in df.index])
        w = df['area_weight'].values

        def grid(v):
            arr = np.full((468, 449), np.nan)
            arr[rows, cols] = v
            return arr

        lai_sig = df['LAI_trend_p'].values < 0.05
        green = lai_sig & (df['LAI_trend'].values > 0)
        brown = lai_sig & (df['LAI_trend'].values < 0)
        pfg_up = (df['PFG_trend_p'].values < 0.05) & (df['PFG_trend'].values > 0)
        pfg_down = (df['PFG_trend_p'].values < 0.05) & (df['PFG_trend'].values < 0)
        quad = np.where(green & pfg_up, 1, np.where(green, 2, np.where(brown, 4, 3))).astype(float)
        q_labels = ['Sig. greening, sig. PFG increase', 'Sig. greening, no sig. PFG increase', 'LAI n.s.',
                    'Sig. browning']
        q_colors = ['#0B4F6C', '#5FB3C4', '#E0E0E0', '#A0522D']
        q_area = [np.sum(w[quad == k]) / np.sum(w) * 100 for k in [1, 2, 3, 4]]
        print('grass area %: sig. greening with sig. PFG decrease:',
              round(np.sum(w[green & pfg_down]) / np.sum(w) * 100, 1))
        panels = [('LAI_trend', 'a) SNU LAI trend (% / decade)', [-20, -15, -10, -5, 0, 5, 10, 15, 20],
                   ['#7B3F13', '#C0692A', '#EBA14A', '#F5D38E', '#CFE3D1', '#87B8CF', '#3D7FA6', '#0B4F6C']),
                  ('PFG_trend', 'b) RAP PFG cover trend (pp / decade)', [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                   ['#8C510A', '#BF812D', '#DFC27D', '#F6E8C3', '#C7EAE5', '#80CDC1', '#35978F', '#01665E']),
                  ('quad', 'c) LAI trend vs PFG trend (p < 0.05)', None, q_colors)]
        with plt.rc_context(self.big_font):
            fig = plt.figure(figsize=(18, 6.8))
            for k, (col, title, bounds, colors) in enumerate(panels):
                ax = fig.add_axes([0.01 + k * 0.33, 0.22, 0.31, 0.70], projection=ccrs.PlateCarree())
                if col == 'quad':
                    im = ax.imshow(grid(quad), cmap=ListedColormap(colors), norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4),
                                   extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(),
                                   interpolation='nearest')
                else:
                    cmap = ListedColormap(colors)
                    cmap.set_under(colors[0])
                    cmap.set_over(colors[-1])
                    im = ax.imshow(grid(df[col].values), cmap=cmap, norm=BoundaryNorm(bounds, cmap.N),
                                   extent=[-124.529, -102.079, 25.579, 48.979], transform=ccrs.PlateCarree(),
                                   interpolation='nearest')
                    sig = (df[f'{col}_p'].values < 0.05) & (rows % 3 == 0) & (cols % 3 == 0)
                    ax.scatter(df['lon'].values[sig], df['lat'].values[sig], s=1.5, c='k', marker='.', lw=0,
                               transform=ccrs.PlateCarree(), zorder=5)
                    cax = fig.add_axes([0.03 + k * 0.33, 0.14, 0.27, 0.025])
                    cb = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds, extend='both')
                    cb.set_label('dots: p < 0.05')
                ax.add_feature(cfeature.STATES, linewidth=0.3)
                ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4)
                ax.set_extent([-125, -102, 30, 46], crs=ccrs.PlateCarree())
                gl = ax.gridlines(draw_labels=True, linewidth=0)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlocator = mticker.FixedLocator(range(-120, -100, 10))
                gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
                gl.xlabel_style = {'size': 14}
                gl.ylabel_style = {'size': 14}
                ax.set_title(title)
            handles = [mpatches.Patch(color=c, label=f'{l} ({a:.0f}%)') for l, c, a in zip(q_labels, q_colors, q_area)]
            fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.69, 0.20), ncol=1, frameon=False,
                       fontsize=14)
            plt.savefig(self.outdir + f'PFG_LAI_trend_maps_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'PFG_LAI_trend_maps_{self.tag}.pdf', bbox_inches='tight')
            plt.close()
        print('grass area % by sign class:', dict(zip(q_labels, np.round(q_area, 1))))

    def light_blue_pfg(self):
        ## PFG cover trend of the light-blue class of plot_maps c) (2026-11-22): sig. LAI greening (p < 0.05) without
        ##   a sig. PFG increase. "No sig. PFG increase" mixes ecologically different cases, so split it:
        ##   sig. PFG decline (p < 0.05, < 0), n.s. decline, n.s. increase
        ## figure: area-weighted histogram of the PFG trend (pp / decade), stacked by the three cases; zero line,
        ##   weighted median (solid) and quartiles (dashed); grey band = |trend| < 0.5 (stable, as Fig 10)
        ## input : PFG_change_vs_LAI_trend_{tag}.df (written by run / load; needs LAI_trend_p and PFG_trend_p)
        ## output: light_blue_PFG_trend_{tag}.png / .pdf / .csv
        df = T.load_df(self.outdir + f'PFG_change_vs_LAI_trend_{self.tag}.df')
        green = (df['LAI_trend_p'] < 0.05) & (df['LAI_trend'] > 0)
        pfg_up = (df['PFG_trend_p'] < 0.05) & (df['PFG_trend'] > 0)
        d = df[green & ~pfg_up]
        x = d['PFG_trend'].values
        w = d['area_weight'].values
        p = d['PFG_trend_p'].values
        cases = [('Sig. PFG decline', (p < 0.05) & (x < 0), '#8C510A'),
                 ('n.s. PFG decline', (p >= 0.05) & (x < 0), '#DFC27D'),
                 ('n.s. PFG increase', (p >= 0.05) & (x >= 0), '#80CDC1')]

        def wq(q):
            o = np.argsort(x)
            cw = np.cumsum(w[o]) / np.sum(w)
            return x[o][np.searchsorted(cw, q)]

        q1, med, q3 = wq(0.25), wq(0.5), wq(0.75)
        all_w = df['area_weight'].sum()
        rows = [['light blue (all)', len(d), np.sum(w) / all_w * 100, 100.0]]
        for name, m, _ in cases:
            rows.append([name, int(m.sum()), np.sum(w[m]) / all_w * 100, np.sum(w[m]) / np.sum(w) * 100])
        stable = np.abs(x) < self.stable
        rows.append([f'|PFG trend| < {self.stable}', int(stable.sum()), np.sum(w[stable]) / all_w * 100,
                     np.sum(w[stable]) / np.sum(w) * 100])
        tab = pd.DataFrame(rows, columns=['case', 'n_pix', 'area_%_of_grass', 'area_%_of_light_blue'])
        tab['PFG_q1'], tab['PFG_median'], tab['PFG_q3'] = q1, med, q3
        tab.to_csv(self.outdir + f'light_blue_PFG_trend_{self.tag}.csv', index=False)
        print(tab.round(2).to_string())
        bins = np.arange(-6, 4.25, 0.25)
        xc = np.clip(x, bins[0] + 1e-6, bins[-1] - 1e-6)
        with plt.rc_context(self.big_font):
            fig, ax = plt.subplots(figsize=(11, 5))
            bottom = np.zeros(len(bins) - 1)
            for name, m, color in cases:
                h = np.histogram(xc[m], bins=bins, weights=w[m] / np.sum(w) * 100)[0]
                share = np.sum(w[m]) / np.sum(w) * 100
                ax.bar(bins[:-1], h, width=0.25, align='edge', bottom=bottom, color=color, edgecolor='white',
                       linewidth=0.4, label=f'{name} ({share:.0f}%)')
                bottom += h
            ax.axvspan(-self.stable, self.stable, color='#BDBDBD', alpha=0.3, lw=0)
            ax.axvline(0, color='k', lw=1.5)
            ax.axvline(med, color='#C0392B', lw=2.5)
            for q in [q1, q3]:
                ax.axvline(q, color='#C0392B', lw=1.8, ls='--')
            ax.text(0.02, 0.97, f'median {med:+.2f}\nIQR [{q1:+.2f}, {q3:+.2f}] pp / decade',
                    transform=ax.transAxes, va='top', fontsize=15, color='#C0392B')
            ax.set_xlabel('PFG cover trend 1986-2024 (pp / decade)')
            ax.set_ylabel('Area (% of light-blue class)')
            ax.set_title(f'Sig. greening without sig. PFG increase ({np.sum(w) / all_w * 100:.0f}% of grassland)')
            ax.legend(frameon=False, loc='upper right')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + f'light_blue_PFG_trend_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'light_blue_PFG_trend_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def plot_strata(self, tab, fit):
        ## 3 x 3: rows = drying tercile (SPEI12 trend), columns = initial PFG tercile (1986-1990)
        ## 2026-11-21: compact figure, large fonts; column titles on top, row labels on the left
        rows = ['strong drying', 'medium drying', 'weak drying']
        cols = ['low PFG 1986-90', 'mid PFG 1986-90', 'high PFG 1986-90']
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(3, 3, figsize=(11, 9.5), sharex=True, sharey=True)
            for i, r in enumerate(rows):
                for j, c in enumerate(cols):
                    ax = axs[i, j]
                    name = f'{r} | {c}'
                    ff = fit[fit['stratum'] == name]
                    if len(ff) == 0:
                        continue
                    f = ff.iloc[0]
                    self.draw(ax, tab[tab['stratum'] == name], f)
                    ax.text(0.03, 0.97, f"stable {f['stable_LAI_trend']:+.1f}\nb {f['b_per_pp_PFG']:+.2f}",
                            transform=ax.transAxes, va='top', fontsize=15)
                    if i == 0:
                        ax.set_title(c.replace(' 1986-90', ''))
                axs[i, 0].set_ylabel(f'{r}\nLAI trend')
            for j in range(3):
                axs[-1, j].set_xlabel('PFG trend')
            fig.text(0.5, 1.0, 'Initial perennial cover (1986-1990)', ha='center', va='bottom', fontsize=17)
            plt.tight_layout()
            plt.savefig(self.outdir + f'PFG_change_vs_LAI_trend_{self.tag}_strata.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'PFG_change_vs_LAI_trend_{self.tag}_strata.pdf', bbox_inches='tight')
            plt.close()


class Greening_cover_change(PFG_change_vs_LAI_trend):
    ## inside the pixels that green (SNU LAI 1986-2024, p < 0.05): did RAP bare ground decline, and which cover
    ##   changed? (2026-11-23)
    ## years : 1986-2024 (SNU LAI and RAP overlap)
    ## pixels: Grass, Shrub and Savanna pixels of the MODIS figures (model2_annual_perennial_compare; NLCD cropland
    ##         < 5%) with SNU GS LAI and all six RAP covers in every year 1986-2024; Forest left out (SNU forest trend
    ##         not robust, Fig 8c; RAP least reliable in closed forest)
    ## groups: Grass / Shrub x annual / perennial (annual = mean RAP AFG > PFG over 1986-2024, as the long records),
    ##         Savanna
    ## status: 'greening' = SNU LAI trend > 0 and p < 0.05; 'control' = LAI trend n.s. (p >= 0.05);
    ##         'browning' = LAI trend < 0 and p < 0.05 (kept out of the comparison, counts printed) (2026-11-23)
    ## per pixel (OLS, per decade): LAI_trend (% of the pixel's mean), {AFG, PFG, SHR, TRE, LTR, BGR}_trend (pp) and _p
    ## figures (output in Result2\05_Grass_attribution\Greening_cover_SNU\):
    ##   Fig10c_greening_cover_{tag}  : a) mean BGR trend, greening vs control, 95% CI (1 x 1 deg block bootstrap)
    ##                                  b) heat map, rows = groups, columns = AFG, PFG, SHR, TRE; colour = greening trend,
    ##                                     text = greening trend and (greening - control), bold if |difference| >= 0.3
    ##   LAI_vs_cover_regression_{tag}: per group, area-weighted OLS over all pixels of the group
    ##                                  LAI_trend = a + sum b_c * cover_trend_c (c = AFG, PFG, SHR, TRE);
    ##                                  b_c = % LAI per decade per 1 pp per decade of cover c; 95% CI block bootstrap
    ##   RAP_cover_timeseries_{tag}   : area-weighted yearly mean RAP cover per group (BGR, AFG, PFG, SHR, TRE);
    ##                                  dotted lines at the Landsat 7 (1999) and Landsat 8 (2013) starts: check for steps
    ##   (earlier figures, not run any more: bare_ground, cover_trend, who_increased)
    ##   csv of each figure, and the pixel table Greening_cover_{tag}.df
    def __init__(self):
        super().__init__()
        self.outdir = this_root + rf'{R2}\05_Grass_attribution\Greening_cover_SNU\\'   ## 2026-11-23: window_config
        T.mk_dir(self.outdir, force=True)
        self.cover_list = ['AFG', 'PFG', 'SHR', 'TRE', 'LTR', 'BGR']
        self.cover_color = {'AFG': '#E6A23C', 'PFG': '#4C9A2A', 'SHR': '#8E5A2B', 'TRE': '#1F5F3A', 'LTR': '#B8A07E',
                            'BGR': '#9E9E9E'}
        self.group_list = ['Grass annual', 'Grass perennial', 'Shrub annual', 'Shrub perennial', 'Savanna']
        self.status_color = {'greening': '#0B4F6C', 'control': '#BDBDBD'}
        self.bold_diff = 0.3   ## pp per decade, |greening - control| shown in bold in the heat map
        self.veg_min = 5.   ## % total vegetation cover (AFG + PFG + SHR + TRE) below which LAI per cover is not computed
        self.co2_range = (2.9, 5.8)   ## CO2 fertilisation expectation, % per decade (Donohue et al. 2013)

    def run(self):
        df, ts = self.load()
        self.fig10c(df)
        self.lai_regression(df)
        self.rap_timeseries(ts)
        self.lai_per_cover(df)

    def run_lai_per_cover(self):
        ## 2026-11-23: only the LAI-per-cover figure (load is needed for the new columns)
        df, _ = self.load()
        self.lai_per_cover(df)

    def run_decomposition(self):
        ## 2026-11-23: decomposition figure from the saved pixel table (run run_lai_per_cover first)
        df = T.load_df(self.outdir + f'Greening_cover_{self.tag}.df')
        self.decomposition_figure(df)

    def decomposition_figure(self, df, res=0.25, n_min_cell=3):
        ## LAI trend (% of mean per decade) ~= cover gain (% of mean) + LAI per unit cover trend (2026-11-23)
        ##   cover gain = trend of total cover AFG + PFG + SHR + TRE (pp per decade) / its pixel mean x 100
        ## a) map, sig. greening pixels: share of the LAI trend from per-unit-cover greening = LAIc_trend / LAI_trend,
        ##    median per res-degree cell (cells with >= n_min_cell greening pixels); 0 = all cover gain, 1 = all per unit
        ## b) decomposition plane: per group, x = cover gain, y = per-unit-cover trend, area-weighted mean with 95% CI
        ##    (1 deg block bootstrap); open = LAI n.s., filled = sig. greening, arrow n.s. -> greening; dotted lines =
        ##    equal LAI trend (x + y), band = CO2 expectation
        ##    2026-11-23: + diamond = all pixels of the group (greening, n.s. and browning): the mean without selecting
        ##    pixels on their LAI trend (greening pixels are biased up, n.s. pixels down); also in the csv (status 'all')
        ## c) stacked split per group: cover gain + per-unit-cover trend, greening (solid) and LAI n.s. (pale);
        ##    dot = observed LAI trend
        ## output: Fig_decomposition_{tag}.png / .pdf / .csv, Fig_decomposition_map_{tag}.csv (cell values)
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        from matplotlib.colors import TwoSlopeNorm
        df = df[np.isfinite(df['LAIc_trend']) & (df['VEG_mean'] > 0)].copy()
        df['COV_rel_trend'] = df['VEG_trend'] / df['VEG_mean'] * 100
        df['resid'] = df['LAI_trend'] - df['COV_rel_trend'] - df['LAIc_trend']
        print('identity check, area-weighted mean of LAI - cover gain - per unit (should be near 0):',
              round(np.average(df['resid'], weights=df['area_weight']), 3))
        ## table
        rows = []
        for g in self.group_list:
            for st in ['greening', 'control', 'all']:   ## 2026-11-23: + all pixels of the group (no selection on LAI)
                d = df[df['group'] == g] if st == 'all' else df[(df['group'] == g) & (df['status'] == st)]
                if len(d) < self.n_min:
                    continue
                row = [g, st, len(d)]
                for col in ['LAI_trend', 'COV_rel_trend', 'LAIc_trend']:
                    row += list(self.wmean_ci(d, col))
                rows.append(row)
        cols = ['group', 'status', 'n_pix']
        for col in ['LAI', 'cover_gain', 'per_unit']:
            cols += [col, f'{col}_ci_low', f'{col}_ci_high']
        tab = pd.DataFrame(rows, columns=cols)
        tab.to_csv(self.outdir + f'Fig_decomposition_{self.tag}.csv', index=False)
        print(tab[['group', 'status', 'n_pix', 'LAI', 'cover_gain', 'per_unit']].round(2).to_string())
        ## map cells
        gr = df[df['status'] == 'greening'].copy()
        gr['share'] = gr['LAIc_trend'] / gr['LAI_trend']
        gr['cx'] = np.floor(gr['lon'] / res) * res
        gr['cy'] = np.floor(gr['lat'] / res) * res
        cell = gr.groupby(['cx', 'cy'])['share'].agg(['median', 'size']).reset_index()
        cell = cell[cell['size'] >= n_min_cell]
        cell.to_csv(self.outdir + f'Fig_decomposition_map_{self.tag}.csv', index=False)
        xe = np.arange(np.floor(df['lon'].min() / res) * res, df['lon'].max() + res, res)
        ye = np.arange(np.floor(df['lat'].min() / res) * res, df['lat'].max() + res, res)
        grid = np.full((len(ye) - 1, len(xe) - 1), np.nan)
        ix = np.clip(np.round((cell['cx'].values - xe[0]) / res).astype(int), 0, len(xe) - 2)
        iy = np.clip(np.round((cell['cy'].values - ye[0]) / res).astype(int), 0, len(ye) - 2)
        grid[iy, ix] = cell['median'].values
        col_g = {'Grass annual': '#E6A23C', 'Grass perennial': '#4C9A2A', 'Shrub annual': '#B07A4A',
                 'Shrub perennial': '#6B3E1E', 'Savanna': '#1F77B4'}
        with plt.rc_context(self.big_font):
            fig = plt.figure(figsize=(27, 9))
            ## a
            ax = fig.add_axes([0.0, 0.08, 0.27, 0.78], projection=ccrs.PlateCarree())
            ## 2026-11-23: five discrete classes instead of BrBG, so 'both' has its own colour (purple):
            ##   < 0 cover gain, leaf per cover falls | 0-1/3 mostly cover gain | 1/3-2/3 both | 2/3-1 mostly per unit
            ##   cover | > 1 per unit cover while cover falls
            from matplotlib.colors import ListedColormap, BoundaryNorm
            share_cls = ['#8C510A', '#D8B365', '#8E6BBF', '#5AB4AC', '#01665E']
            share_bnd = [-0.5, 0, 1 / 3, 2 / 3, 1, 1.5]
            im = ax.pcolormesh(xe, ye, np.clip(grid, -0.49, 1.49), cmap=ListedColormap(share_cls),
                               norm=BoundaryNorm(share_bnd, 5), transform=ccrs.PlateCarree(), shading='flat')
            ax.add_feature(cfeature.STATES, linewidth=0.4)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.set_extent([-124.6, -102, 31, 46], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.FixedLocator(range(-125, -100, 10))
            gl.ylocator = mticker.FixedLocator(range(30, 50, 5))
            gl.xlabel_style = {'size': 14}
            gl.ylabel_style = {'size': 14}
            cb = plt.colorbar(im, ax=ax, fraction=0.04)
            cb.set_ticks([-0.25, 1 / 6, 0.5, 5 / 6, 1.25])
            cb.set_ticklabels(['cover gain,\nleaf / cover falls', 'mostly\ncover gain', 'both', 'mostly\nper unit cover',
                               'per unit cover,\ncover falls'])
            cb.ax.tick_params(labelsize=13)
            ax.set_title(f'a) Sig. greening: share from per-unit-cover\ngreening ({res}° median)')
            ## b
            ax = fig.add_axes([0.35, 0.12, 0.30, 0.72])
            ax.axhspan(*self.co2_range, color='#FDEBD0', zorder=0, label='CO$_2$ expectation')
            xs_line = np.linspace(-3, 10, 20)
            for L in range(-2, 11, 2):
                ax.plot(xs_line, L - xs_line, ls=':', color='grey', lw=1)
            ## 2026-11-23: only the all-pixel mean of each group (diamond); greening / LAI n.s. stay in the csv and in c)
            sub = tab[tab['status'] == 'all'].set_index('group')
            for g in self.group_list:
                if g not in sub.index:
                    continue
                A = sub.loc[g]
                ax.errorbar(A['cover_gain'], A['per_unit'],
                            xerr=[[A['cover_gain'] - A['cover_gain_ci_low']], [A['cover_gain_ci_high'] - A['cover_gain']]],
                            yerr=[[A['per_unit'] - A['per_unit_ci_low']], [A['per_unit_ci_high'] - A['per_unit']]],
                            fmt='D', ms=16, color=col_g[g], mfc=col_g[g], mec='k', mew=1.5, capsize=4, lw=2,
                            label=f"{g}: LAI {A['LAI']:+.1f}", zorder=4)
                ax.annotate(g, (A['cover_gain'], A['per_unit']), xytext=(12, 10), textcoords='offset points',
                            fontsize=15, color=col_g[g], fontweight='bold')
            ax.axhline(0, color='k', lw=0.8)
            ax.axvline(0, color='k', lw=0.8)
            vals = np.concatenate([sub['cover_gain_ci_low'], sub['cover_gain_ci_high'], sub['per_unit_ci_low'],
                                   sub['per_unit_ci_high']])
            lo, hi = np.nanmin(vals) - 1.0, np.nanmax(vals) + 1.0
            ax.set_xlim(min(lo, -1.5), max(hi, 6))
            ax.set_ylim(min(lo, -1.5), self.co2_range[1] + 0.5)
            for L in range(-2, 11, 2):   ## label the equal-LAI lines at the top edge
                xt = L - (self.co2_range[1] + 0.2)
                if ax.get_xlim()[0] < xt < ax.get_xlim()[1]:
                    ax.text(xt, self.co2_range[1] + 0.2, f'LAI {L:+d}', fontsize=11, color='grey', ha='center')
            ax.set_xlabel('Cover gain (% of mean per decade)')
            ax.set_ylabel('LAI per unit cover trend\n(% per decade)')
            ax.set_title('b) all pixels of each group: LAI trend = cover gain + per unit cover\n'
                         'dotted = equal LAI trend; band = CO$_2$ expectation')
            ax.legend(frameon=False, fontsize=14, loc='upper left', bbox_to_anchor=(1.0, 1.0))
            ax.spines[['top', 'right']].set_visible(False)
            ## c
            ax = fig.add_axes([0.81, 0.12, 0.19, 0.72])
            xs = np.arange(len(self.group_list))
            for k, st in enumerate(['greening', 'control']):
                d = tab[tab['status'] == st].set_index('group').reindex(self.group_list)
                cov, unit = d['cover_gain'].values, d['per_unit'].values
                xx = xs + (k - 0.5) * 0.38
                a = 1 if st == 'greening' else 0.45
                ax.bar(xx, cov, 0.38, color='#8E5A2B', alpha=a, label='cover gain' if k == 0 else None)
                ax.bar(xx, unit, 0.38, bottom=np.where(unit >= 0, np.where(cov > 0, cov, 0), np.where(cov < 0, cov, 0)),
                       color='#2E8B57', alpha=a, label='per unit cover' if k == 0 else None)
                ax.scatter(xx, d['LAI'].values, color='k', s=35, zorder=3, label='LAI trend' if k == 0 else None)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels([g.replace(' ', '\n') for g in self.group_list], fontsize=13)
            ax.set_ylabel('% per decade')
            ax.set_title('c) solid = sig. greening\npale = LAI n.s.')
            ax.legend(frameon=False, fontsize=13, loc='upper right')
            ax.spines[['top', 'right']].set_visible(False)
            plt.savefig(self.outdir + f'Fig_decomposition_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_decomposition_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def lai_per_cover(self, df):
        ## does each unit of vegetation cover hold more leaf? (2026-11-23)
        ## per group, greening vs control (LAI n.s.): area-weighted mean trend (% per decade, 95% CI, 1 deg block
        ##   bootstrap) of a) GS LAI, b) LAI per unit vegetation cover (LAI / (AFG + PFG + SHR + TRE)),
        ##   plus total vegetation cover trend (pp per decade) in the csv; grey band = CO2 expectation
        ## output: LAI_per_cover_{tag}.png / .pdf / .csv
        cols = [('LAI_trend', 'GS LAI'), ('LAIc_trend', 'LAI per unit cover'), ('VEG_trend', 'total cover (pp)')]
        rows = []
        for g in self.group_list:
            for st in ['greening', 'control']:
                d = df[(df['group'] == g) & (df['status'] == st)]
                d = d[np.isfinite(d['LAIc_trend'])]
                if len(d) < self.n_min:
                    continue
                for col, name in cols:
                    m, lo, hi = self.wmean_ci(d, col)
                    rows.append([g, st, len(d), name, m, lo, hi])
        tab = pd.DataFrame(rows, columns=['group', 'status', 'n_pix', 'variable', 'trend', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + f'LAI_per_cover_{self.tag}.csv', index=False)
        print(tab.pivot_table(index=['group', 'status'], columns='variable', values='trend').round(2).to_string())
        x = np.arange(len(self.group_list))
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(1, 2, figsize=(19, 7), sharey=True)
            for ax, (col, name) in zip(axs, cols[:2]):
                ax.axhspan(*self.co2_range, color='#FDEBD0', zorder=0, label='CO$_2$ expectation')
                for k, st in enumerate(['greening', 'control']):
                    t = tab[(tab['status'] == st) & (tab['variable'] == name)].set_index('group').reindex(self.group_list)
                    v = t['trend'].values
                    ax.bar(x + (k - 0.5) * 0.38, v, width=0.38, color=self.status_color[st],
                           label='sig. greening' if st == 'greening' else 'LAI n.s.',
                           yerr=[v - t['ci_low'].values, t['ci_high'].values - v], capsize=4)
                    for xx, vv in zip(x + (k - 0.5) * 0.38, v):
                        if np.isfinite(vv):
                            ax.text(xx, vv + (0.3 if vv >= 0 else -0.3), f'{vv:+.1f}', ha='center',
                                    va='bottom' if vv >= 0 else 'top', fontsize=13)
                ax.axhline(0, color='k', lw=1)
                ax.set_xticks(x)
                ax.set_xticklabels([g.replace(' ', '\n') for g in self.group_list])
                ax.set_title(f'{"a" if col == "LAI_trend" else "b"}) {name} trend')
                ax.spines[['top', 'right']].set_visible(False)
            axs[0].set_ylabel('Trend 1986-2024 (% of mean / decade)')
            h, l = axs[0].get_legend_handles_labels()
            plt.tight_layout()
            fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
            plt.savefig(self.outdir + f'LAI_per_cover_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'LAI_per_cover_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def load(self):
        cmp = T.load_df(self.cmp_dff)
        info = cmp[cmp['landcover_type'].isin(['Grass', 'Shrub', 'Savanna'])][['lon', 'lat', 'area_weight',
                                                                               'landcover_type']]
        pix_set = set(info.index)
        snu = T.load_df(self.snu_dff)
        snu = snu[snu['pix'].isin(pix_set)]
        lai = self.matrix(snu, info, 'GS_LAI', relative=True)
        lai_abs = self.matrix(snu, info, 'GS_LAI')   ## 2026-11-23: for LAI per unit vegetation cover
        del snu
        rap = T.load_df(self.rap_dff)
        rap = rap[rap['pix'].isin(pix_set)]
        cov = {c: self.matrix(rap, info, f'RAP_{c}') for c in self.cover_list}
        del rap
        ok = np.isfinite(lai).all(axis=1)
        for c in self.cover_list:
            ok &= np.isfinite(cov[c]).all(axis=1)
        print('grass + shrub + savanna pixels:', len(info), '| with all years of SNU LAI and the six RAP covers:',
              int(ok.sum()))
        df = info[ok].copy()
        df['LAI_trend'], df['LAI_trend_p'] = self.trend(lai[ok])
        for c in self.cover_list:
            df[f'{c}_trend'], df[f'{c}_trend_p'] = self.trend(cov[c][ok])
            df[f'{c}_mean'] = np.mean(cov[c][ok], axis=1)
        ## 2026-11-23: LAI per unit vegetation cover = GS LAI / ((AFG + PFG + SHR + TRE) / 100), per year;
        ##   years with total cover < self.veg_min (%) -> NaN, pixel then has no trend
        veg = sum(cov[c][ok] for c in ['AFG', 'PFG', 'SHR', 'TRE'])
        with np.errstate(divide='ignore', invalid='ignore'):
            lai_c = np.where(veg >= self.veg_min, lai_abs[ok] / (veg / 100.), np.nan)
            lai_c = (lai_c / np.nanmean(lai_c, axis=1, keepdims=True) - 1) * 100   ## % of the pixel's mean
        df['VEG_mean'] = np.mean(veg, axis=1)
        df['VEG_trend'], df['VEG_trend_p'] = self.trend(veg)
        df['LAIc_trend'], df['LAIc_trend_p'] = self.trend(lai_c)
        print('pixels with LAI per cover in every year (total cover >=', self.veg_min, '%):',
              int(np.isfinite(df['LAIc_trend']).sum()), 'of', len(df))
        herb = np.where(df['AFG_mean'] > df['PFG_mean'], 'annual', 'perennial')
        df['group'] = np.where(df['landcover_type'] == 'Savanna', 'Savanna', df['landcover_type'].values + ' ' + herb)
        sig = df['LAI_trend_p'] < 0.05
        df['status'] = np.where(sig & (df['LAI_trend'] > 0), 'greening',
                                np.where(sig & (df['LAI_trend'] < 0), 'browning', 'control'))
        cnt = df.groupby(['group', 'status'])['area_weight'].agg(['size', 'sum']).unstack()
        print('pixels per group and status:\n', cnt['size'].to_string())
        area = cnt['sum'].div(cnt['sum'].sum(axis=1), axis=0) * 100
        print('area % per group (browning is left out of the comparison):\n', area.round(1).to_string())
        area.to_csv(self.outdir + f'status_area_{self.tag}.csv')
        T.save_df(df, self.outdir + f'Greening_cover_{self.tag}.df')
        ## yearly area-weighted mean cover per group (all pixels of the group), for the RAP consistency check
        w = df['area_weight'].values
        rows = []
        for g in self.group_list:
            m = (df['group'] == g).values
            for c in ['BGR', 'AFG', 'PFG', 'SHR', 'TRE']:
                y = np.sum(cov[c][ok][m] * w[m, None], axis=0) / np.sum(w[m])
                rows += [[g, c, yr, v] for yr, v in zip(self.year_list, y)]
        ts = pd.DataFrame(rows, columns=['group', 'cover', 'year', 'cover_%'])
        ts.to_csv(self.outdir + f'RAP_cover_timeseries_{self.tag}.csv', index=False)
        return df, ts

    def fig10c(self, df):
        from matplotlib.colors import TwoSlopeNorm
        covers = ['AFG', 'PFG', 'SHR', 'TRE']
        rows = []
        for g in self.group_list:
            for st in ['greening', 'control']:
                d = df[(df['group'] == g) & (df['status'] == st)]
                if len(d) < self.n_min:
                    continue
                for c in covers + ['BGR']:
                    m, lo, hi = self.wmean_ci(d, f'{c}_trend')
                    rows.append([g, st, len(d), c, m, lo, hi])
        tab = pd.DataFrame(rows, columns=['group', 'status', 'n_pix', 'cover', 'trend', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + f'Fig10c_greening_cover_{self.tag}.csv', index=False)
        print(tab.pivot_table(index=['group', 'status'], columns='cover', values='trend')[covers + ['BGR']]
              .round(2).to_string())

        def get(g, st, c, k='trend'):
            t = tab[(tab['group'] == g) & (tab['status'] == st) & (tab['cover'] == c)]
            return t[k].iloc[0] if len(t) else np.nan

        x = np.arange(len(self.group_list))
        with plt.rc_context(self.big_font):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={'width_ratios': [1, 1.15]})
            for k, st in enumerate(['greening', 'control']):
                v = np.array([get(g, st, 'BGR') for g in self.group_list])
                lo = np.array([get(g, st, 'BGR', 'ci_low') for g in self.group_list])
                hi = np.array([get(g, st, 'BGR', 'ci_high') for g in self.group_list])
                ax1.bar(x + (k - 0.5) * 0.38, v, width=0.38, color=self.status_color[st],
                        label='sig. greening' if st == 'greening' else 'LAI n.s.', yerr=[v - lo, hi - v], capsize=4)
            ax1.axhline(0, color='k', lw=1)
            ax1.set_xticks(x)
            ax1.set_xticklabels([g.replace(' ', '\n') for g in self.group_list])
            ax1.set_ylabel('Bare-ground trend (pp / decade)')
            ax1.set_title('a) Did bare ground decline?')
            ax1.legend(frameon=False, loc='lower left')
            ax1.spines[['top', 'right']].set_visible(False)
            G = np.array([[get(g, 'greening', c) for c in covers] for g in self.group_list])
            C = np.array([[get(g, 'control', c) for c in covers] for g in self.group_list])
            vmax = max(np.nanmax(np.abs(G)), 1)
            im = ax2.imshow(G, cmap='BrBG', norm=TwoSlopeNorm(0, -vmax / 2, vmax), aspect='auto')
            for i in range(len(self.group_list)):
                for j in range(len(covers)):
                    if not np.isfinite(G[i, j]):
                        continue
                    dd = G[i, j] - C[i, j]
                    ax2.text(j, i, f'{G[i, j]:+.1f}\n({dd:+.1f})', ha='center', va='center', fontsize=16,
                             fontweight='bold' if abs(dd) >= self.bold_diff else 'normal',
                             color='white' if abs(G[i, j]) > 0.65 * vmax else 'k')
            ax2.set_xticks(range(len(covers)))
            ax2.set_xticklabels(covers)
            ax2.set_yticks(range(len(self.group_list)))
            ax2.set_yticklabels(self.group_list)
            ax2.set_title(f'b) Cover trend in greening pixels\n(greening - LAI n.s.); bold |diff| >= {self.bold_diff}')
            cb = plt.colorbar(im, ax=ax2, fraction=0.045)
            cb.set_label('Cover trend (pp / decade)')
            plt.tight_layout()
            plt.savefig(self.outdir + f'Fig10c_greening_cover_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig10c_greening_cover_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def lai_regression(self, df, n_boot=200):
        ## per group: area-weighted OLS of LAI_trend on the four cover trends, all pixels of the group
        covers = ['AFG', 'PFG', 'SHR', 'TRE']

        def fit(d):
            X = np.column_stack([np.ones(len(d))] + [d[f'{c}_trend'].values for c in covers])
            sw = np.sqrt(d['area_weight'].values)
            return np.linalg.lstsq(X * sw[:, None], d['LAI_trend'].values * sw, rcond=None)[0][1:]

        rows = []
        rng = np.random.default_rng(0)
        for g in self.group_list:
            d = df[df['group'] == g]
            b = fit(d)
            block = (np.floor(d['lon'].values) * 1000 + np.floor(d['lat'].values)).astype(int)
            ids = np.unique(block)
            idx = {bb: np.where(block == bb)[0] for bb in ids}
            boot = np.array([fit(d.iloc[np.concatenate([idx[bb] for bb in rng.choice(ids, size=len(ids))])])
                             for _ in range(n_boot)])
            for k, c in enumerate(covers):
                rows.append([g, len(d), c, b[k], np.percentile(boot[:, k], 2.5), np.percentile(boot[:, k], 97.5)])
        tab = pd.DataFrame(rows, columns=['group', 'n_pix', 'cover', 'b_LAI_per_pp', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + f'LAI_vs_cover_regression_{self.tag}.csv', index=False)
        print(tab.round(3).to_string())
        x = np.arange(len(self.group_list))
        wd = 0.2
        with plt.rc_context(self.big_font):
            fig, ax = plt.subplots(figsize=(15, 6.5))
            for k, c in enumerate(covers):
                t = tab[tab['cover'] == c].set_index('group').reindex(self.group_list)
                v = t['b_LAI_per_pp'].values
                ax.bar(x + (k - 1.5) * wd, v, width=wd, color=self.cover_color[c], label=c,
                       yerr=[v - t['ci_low'].values, t['ci_high'].values - v], capsize=3)
            ax.axhline(0, color='k', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([g.replace(' ', '\n') for g in self.group_list])
            ax.set_ylabel('LAI trend per 1 pp cover trend\n(% / decade per pp / decade)')
            ax.set_title('Which cover gain turns into LAI? (multiple regression, 95% CI)')
            ax.legend(frameon=False, ncol=4, loc='upper right')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + f'LAI_vs_cover_regression_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'LAI_vs_cover_regression_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def rap_timeseries(self, ts):
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(2, 3, figsize=(19, 10), sharex=True)
            for ax, g in zip(axs.ravel(), self.group_list):
                for c in ['BGR', 'AFG', 'PFG', 'SHR', 'TRE']:
                    t = ts[(ts['group'] == g) & (ts['cover'] == c)]
                    ax.plot(t['year'], t['cover_%'], color=self.cover_color[c], lw=2.2, label=c)
                for yr in [1999, 2013]:
                    ax.axvline(yr, color='grey', lw=1.2, ls=':')
                ax.set_title(g)
                ax.spines[['top', 'right']].set_visible(False)
            for ax in axs[:, 0]:
                ax.set_ylabel('RAP cover (%)')
            axs.ravel()[-1].set_visible(False)
            h, l = axs[0, 0].get_legend_handles_labels()
            fig.legend(h, l, loc='center', bbox_to_anchor=(0.83, 0.27), frameon=False, fontsize=17,
                       title='dotted: Landsat 7 (1999), 8 (2013)', title_fontsize=14)
            plt.tight_layout()
            plt.savefig(self.outdir + f'RAP_cover_timeseries_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'RAP_cover_timeseries_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def wmean_ci(self, d, col):
        v = d[col].values
        w = d['area_weight'].values
        lo, hi = self.block_ci(d, v, w)
        return np.sum(v * w) / np.sum(w), lo, hi

    def bare_ground(self, df):
        cls = [('Sig. decline', '#5D4037'), ('n.s. decline', '#BCAAA4'), ('n.s. increase', '#C7E9F1'),
               ('Sig. increase', '#2C7FB8')]
        rows = []
        for g in self.group_list:
            for st in ['greening', 'control']:
                d = df[(df['group'] == g) & (df['status'] == st)]
                if len(d) < self.n_min:
                    continue
                x, p, w = d['BGR_trend'].values, d['BGR_trend_p'].values, d['area_weight'].values
                masks = [(p < 0.05) & (x < 0), (p >= 0.05) & (x < 0), (p >= 0.05) & (x >= 0), (p < 0.05) & (x >= 0)]
                m, lo, hi = self.wmean_ci(d, 'BGR_trend')
                row = [g, st, len(d), np.sum(w) / df.loc[df['group'] == g, 'area_weight'].sum() * 100, m, lo, hi]
                row += [np.sum(w[k]) / np.sum(w) * 100 for k in masks]
                rows.append(row)
        tab = pd.DataFrame(rows, columns=['group', 'status', 'n_pix', 'area_%_of_group', 'BGR_trend', 'ci_low',
                                          'ci_high'] + [f'{c}_%' for c, _ in cls])
        tab.to_csv(self.outdir + f'BGR_trend_greening_{self.tag}.csv', index=False)
        print(tab.round(2).to_string())
        with plt.rc_context(self.big_font):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1.4, 1]})
            labels = [f"{r['group']}\n{r['status']} ({r['area_%_of_group']:.0f}%)" for _, r in tab.iterrows()]
            y = np.arange(len(tab))[::-1]
            left = np.zeros(len(tab))
            for c, color in cls:
                ax1.barh(y, tab[f'{c}_%'], left=left, color=color, edgecolor='white', label=c)
                for yy, l, v in zip(y, left, tab[f'{c}_%']):
                    if v >= 8:
                        ax1.text(l + v / 2, yy, f'{v:.0f}', ha='center', va='center', fontsize=14,
                                 color='white' if c in ['Sig. decline', 'Sig. increase'] else 'k')
                left += tab[f'{c}_%'].values
            ax1.set_yticks(y)
            ax1.set_yticklabels(labels)
            ax1.set_xlim(0, 100)
            ax1.set_xlabel('Area (%)')
            ax1.set_title('a) RAP bare-ground trend class, 1986-2024')
            ax1.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.45, -0.1))
            ax1.spines[['top', 'right']].set_visible(False)
            xg = np.arange(len(self.group_list))
            for k, st in enumerate(['greening', 'control']):
                t = tab[tab['status'] == st].set_index('group').reindex(self.group_list)
                ax2.bar(xg + (k - 0.5) * 0.38, t['BGR_trend'], width=0.38, color=self.status_color[st], label=st,
                        yerr=[t['BGR_trend'] - t['ci_low'], t['ci_high'] - t['BGR_trend']], capsize=4)
            ax2.axhline(0, color='k', lw=1)
            ax2.set_xticks(xg)
            ax2.set_xticklabels([g.replace(' ', '\n') for g in self.group_list])
            ax2.set_ylabel('BGR trend (pp / decade)')
            ax2.set_title('b) Mean bare-ground trend, 95% CI')
            ax2.legend(frameon=False)
            ax2.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + f'BGR_trend_greening_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'BGR_trend_greening_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def cover_trend(self, df):
        rows = []
        for g in self.group_list:
            for st in ['greening', 'control']:
                d = df[(df['group'] == g) & (df['status'] == st)]
                if len(d) < self.n_min:
                    continue
                for c in self.cover_list:
                    m, lo, hi = self.wmean_ci(d, f'{c}_trend')
                    w = d['area_weight'].values
                    rows.append([g, st, len(d), c, m, lo, hi, np.sum(w * d[f'{c}_mean']) / np.sum(w)])
        tab = pd.DataFrame(rows, columns=['group', 'status', 'n_pix', 'cover', 'trend', 'ci_low', 'ci_high',
                                          'mean_cover_%'])
        tab.to_csv(self.outdir + f'cover_trend_greening_{self.tag}.csv', index=False)
        print(tab.pivot_table(index=['group', 'status'], columns='cover', values='trend')[self.cover_list]
              .round(2).to_string())
        x = np.arange(len(self.cover_list))
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
            for ax, g in zip(axs.ravel(), self.group_list):
                for k, st in enumerate(['greening', 'control']):
                    t = tab[(tab['group'] == g) & (tab['status'] == st)].set_index('cover').reindex(self.cover_list)
                    if t['trend'].isna().all():
                        continue
                    ax.bar(x + (k - 0.5) * 0.38, t['trend'], width=0.38, color=self.status_color[st],
                           label=f"{st} (n={int(t['n_pix'].iloc[0])})",
                           yerr=[t['trend'] - t['ci_low'], t['ci_high'] - t['trend']], capsize=3)
                ax.axhline(0, color='k', lw=1)
                ax.set_xticks(x)
                ax.set_xticklabels(self.cover_list)
                ax.set_title(g)
                ax.legend(frameon=False, loc='lower left')
                ax.spines[['top', 'right']].set_visible(False)
            for ax in axs[:, 0]:
                ax.set_ylabel('Cover trend (pp / decade)')
            plt.tight_layout()
            plt.savefig(self.outdir + f'cover_trend_greening_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'cover_trend_greening_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def who_increased(self, df):
        up = {c: ((df[f'{c}_trend_p'] < 0.05) & (df[f'{c}_trend'] > 0)).values for c in ['AFG', 'PFG', 'SHR', 'TRE']}
        woody = up['SHR'] | up['TRE']
        herb = up['AFG'] | up['PFG']
        cat = np.select([~woody & ~herb, ~woody & up['AFG'] & ~up['PFG'], ~woody & up['PFG'] & ~up['AFG'],
                         ~woody & up['AFG'] & up['PFG'], woody & ~herb, woody & herb],
                        ['none', 'AFG only', 'PFG only', 'AFG + PFG', 'woody only', 'woody + herbaceous'], default='none')
        cats = [('none', '#E0E0E0'), ('AFG only', '#E6A23C'), ('PFG only', '#4C9A2A'), ('AFG + PFG', '#A3C95A'),
                ('woody only', '#8E5A2B'), ('woody + herbaceous', '#C99A6B')]
        df = df.assign(increase=cat)
        rows = []
        for g in self.group_list:
            for st in ['greening', 'control']:
                d = df[(df['group'] == g) & (df['status'] == st)]
                if len(d) < self.n_min:
                    continue
                w = d['area_weight'].values
                rows.append([g, st, len(d)] + [np.sum(w[d['increase'].values == c]) / np.sum(w) * 100 for c, _ in cats])
        tab = pd.DataFrame(rows, columns=['group', 'status', 'n_pix'] + [f'{c}_%' for c, _ in cats])
        tab.to_csv(self.outdir + f'who_increased_greening_{self.tag}.csv', index=False)
        print(tab.round(1).to_string())
        with plt.rc_context(self.big_font):
            fig, ax = plt.subplots(figsize=(15, 7))
            xs = []
            for i, g in enumerate(self.group_list):
                for k, st in enumerate(['greening', 'control']):
                    xs.append((i * 2.6 + k, g, st))
            bottom = np.zeros(len(xs))
            for c, color in cats:
                vals = []
                for _, g, st in xs:
                    t = tab[(tab['group'] == g) & (tab['status'] == st)]
                    vals.append(t[f'{c}_%'].iloc[0] if len(t) else np.nan)
                vals = np.nan_to_num(np.array(vals))
                ax.bar([p for p, _, _ in xs], vals, bottom=bottom, width=0.85, color=color, edgecolor='white',
                       label=c)
                for (p, _, _), b, v in zip(xs, bottom, vals):
                    if v >= 7:
                        ax.text(p, b + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=14)
                bottom += vals
            ax.set_xticks([p for p, _, _ in xs])
            ax.set_xticklabels([st for _, _, st in xs], rotation=0, fontsize=14)
            for i, g in enumerate(self.group_list):
                ax.text(i * 2.6 + 0.5, -16, g, ha='center', va='top', fontsize=17, transform=ax.transData)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Area (%)')
            ax.set_title('Which RAP cover increased significantly (p < 0.05), 1986-2024')
            ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir + f'who_increased_greening_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'who_increased_greening_{self.tag}.pdf', bbox_inches='tight')
            plt.close()


class Composition_sensitivity(Growing_season_trend):
    ## after the cover composition of a pixel changed: does LAI gain more in equally wet years and lose less in
    ##   equally dry years? (2026-11-23)
    ## years  : P1 = 1986-2004, P2 = 2005-2024 (SNU LAI and RAP overlap)
    ## pixels : Grass / Shrub x annual / perennial and Savanna of the long attribution table
    ##          (long_attribution_data_v2_1986_2024.df, MODIS growing-season window), complete LAI and x in every year;
    ##          Forest left out
    ## x      : 'SPEI' = GS SPEI at the pixel's best scale; 'SM' = GS root-zone soil moisture percentile, as z-score of
    ##          the pixel's 1986-2024 years (same bins as SPEI)
    ## LAI anomaly: GS_LAI_rel detrended over 1986-2024 (removes the long greening), then in % of the period's mean LAI
    ##          (P2 LAI is higher, so absolute responses would grow by themselves)
    ## composition: share of AFG, PFG, woody (SHR + TRE) in total cover (AFG + PFG + SHR + TRE), mean P2 - mean P1 (pp);
    ##          period means, not yearly values (RAP uses Landsat greenness, so yearly cover follows the LAI anomaly)
    ##          class = the share that rose most if it rose >= up_thr; 'stable' if every |share change| < stable_thr;
    ##          the rest ('other') is left out
    ## per pixel and period:
    ##   bins   : mean anomaly in each x bin; wet gain = mean anomaly of years x > 0.5, dry loss = - mean anomaly of
    ##            years x < -0.5
    ##   hinge  : LAI' = a + b_wet max(x, 0) + b_dry min(x, 0); wet gain per unit x = b_wet, dry loss per unit = b_dry
    ##            (>= n_side_min years on each side of 0)
    ##   change = P2 - P1: wet gain > 0 -> LAI rises more in wet years; dry loss < 0 -> LAI drops less in dry years
    ## difference-in-differences: change of a class - change of 'stable' pixels; this removes what all pixels share
    ##   (CO2, background climate, AVHRR -> MODIS sensor change in SNU); 95% CI by joint 1 x 1 deg block bootstrap
    ## output: Result2\05_Grass_attribution\Composition_sensitivity_SNU\ :
    ##   Comp_sens_{x}_bins_{tag}.png/pdf/csv : LAI anomaly per x bin, P1 vs P2, per class (pixels with the bin in both
    ##                                          periods)
    ##   Comp_sens_{x}_did_{tag}.png/pdf/csv  : DiD of dry loss (x) and wet gain (y), a) hinge, b) bins; csv also per group
    ##   Comp_sens_{x}_pixels_{tag}.df        : pixel table (for maps later)
    ## run_cover_change (2026-11-23): per pixel, change of the sensitivity (P2 - P1) against the change of total cover
    ##   dVEG = mean (AFG + PFG + SHR + TRE) P2 - P1 (pp); sensitivity = OLS slope of the LAI anomaly on x (% per unit x),
    ##   dry side = b_dry, wet side = b_wet of the hinge fit; bins of dVEG per group, area-weighted mean and 95% CI
    ##   (1 x 1 deg block bootstrap); slope of the change of sensitivity on dVEG (area-weighted LS, same bootstrap)
    ##   the dVEG ~ 0 bin = change shared by pixels whose cover did not change (sensor, drier P2)
    ##   output: Comp_sens_{x}_cover_change_{tag}.png / .pdf / .csv, _slope.csv
    def __init__(self, x_var='SPEI'):
        Growing_season_trend.__init__(self, this_root + rf'{R2}\05_Grass_attribution\Composition_sensitivity_SNU\\')
        self.year_list = list(range(1986, 2025))
        self.tag = f'{self.year_list[0]}_{self.year_list[-1]}'
        self.period_dic = {'P1': (1986, 2004), 'P2': (2005, 2024)}
        self.data_dff = this_root + \
            rf'{R2}\03_response\long_attribution_1986_2024\long_attribution_data_v2_1986_2024.df'
        self.x_var = x_var
        self.x_col = {'SPEI': 'SPEI_best', 'SM': 'GS_P'}[x_var]
        self.x_label = {'SPEI': 'best-scale GS SPEI', 'SM': 'GS root-zone SM (z of percentile)'}[x_var]
        self.group_list = ['Grass annual', 'Grass perennial', 'Shrub annual', 'Shrub perennial', 'Savanna']
        self.rap_list = ['RAP_AFG', 'RAP_PFG', 'RAP_SHR', 'RAP_TRE']
        self.class_list = ['AFG share up', 'PFG share up', 'woody share up', 'stable']
        self.class_color = {'AFG share up': '#E6A23C', 'PFG share up': '#4C9A2A', 'woody share up': '#8E5A2B',
                            'stable': '#616161'}
        self.up_thr = 4.   ## pp of share, P2 - P1
        self.stable_thr = 2.   ## pp of share
        self.veg_min = 5.   ## % total cover, both periods
        self.bin_edges = [-np.inf, -1, -0.5, 0.5, 1, np.inf]
        self.bin_labels = ['< -1', '-1 to\n-0.5', '-0.5 to\n0.5', '0.5 to\n1', '> 1']
        self.n_side_min = 3
        self.n_min = 30
        self.dveg_edges = [-np.inf, -10, -5, -2, 2, 5, 10, np.inf]   ## pp, P2 - P1 total cover
        self.dveg_labels = ['< -10', '-10 to -5', '-5 to -2', '-2 to 2', '2 to 5', '5 to 10', '> 10']
        self.group_color = {'Grass annual': '#E6A23C', 'Grass perennial': '#4C9A2A', 'Shrub annual': '#C8A165',
                            'Shrub perennial': '#8E5A2B', 'Savanna': '#1F77B4'}
        self.big_font = {'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15,
                         'ytick.labelsize': 15, 'legend.fontsize': 15}

    def run(self):
        res = self.load()
        curves = self.bin_curves(res)
        tab = self.summary(res)
        self.fig_bins(curves)
        self.fig_did(tab)

    def load(self):
        from scipy.signal import detrend
        df = T.load_df(self.data_dff)
        df = df[df['group'].isin(self.group_list) & (df['year'] >= self.year_list[0]) &
                (df['year'] <= self.year_list[-1])]
        info = df.groupby('pix')[['lon', 'lat', 'area_weight', 'landcover_type', 'group']].first()

        def piv(col):
            return df.pivot(index='pix', columns='year', values=col).reindex(index=info.index,
                                                                             columns=self.year_list).values
        lai = piv('GS_LAI_rel')
        x = piv(self.x_col)
        cov = {c: piv(c) for c in self.rap_list}
        del df
        ok = np.isfinite(lai).all(axis=1) & np.isfinite(x).all(axis=1)
        print(self.x_var, '| pixels:', len(info), '| complete LAI and x:', int(ok.sum()))
        info, lai, x = info[ok].copy(), lai[ok], x[ok]
        cov = {c: v[ok] for c, v in cov.items()}
        if self.x_var == 'SM':
            x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
        yrs = np.array(self.year_list)
        pm = {p: (yrs >= y0) & (yrs <= y1) for p, (y0, y1) in self.period_dic.items()}
        ## LAI anomaly in % of the period's mean LAI
        lai_d = detrend(lai, axis=1)
        anom = np.full_like(lai_d, np.nan)
        for p, m in pm.items():
            level = 1 + np.mean(lai[:, m], axis=1, keepdims=True) / 100.
            anom[:, m] = lai_d[:, m] / level
        ## composition change
        veg = cov['RAP_AFG'] + cov['RAP_PFG'] + cov['RAP_SHR'] + cov['RAP_TRE']
        share = {'AFG': cov['RAP_AFG'], 'PFG': cov['RAP_PFG'], 'woody': cov['RAP_SHR'] + cov['RAP_TRE']}
        with np.errstate(divide='ignore', invalid='ignore'):
            share = {k: np.where(veg > 0, v / veg * 100, np.nan) for k, v in share.items()}
        res = info
        for k, v in share.items():
            res[f'd{k}'] = np.nanmean(v[:, pm['P2']], axis=1) - np.nanmean(v[:, pm['P1']], axis=1)
        res['dVEG'] = np.nanmean(veg[:, pm['P2']], axis=1) - np.nanmean(veg[:, pm['P1']], axis=1)   ## pp
        veg_ok = (np.nanmean(veg[:, pm['P1']], axis=1) >= self.veg_min) & \
                 (np.nanmean(veg[:, pm['P2']], axis=1) >= self.veg_min)
        D = res[['dAFG', 'dPFG', 'dwoody']].values
        name = np.array(['AFG share up', 'PFG share up', 'woody share up'])
        with np.errstate(invalid='ignore'):
            up = np.nanmax(D, axis=1)
            cls = np.where(np.nanmax(np.abs(D), axis=1) < self.stable_thr, 'stable',
                           np.where(up >= self.up_thr, name[np.nanargmax(np.where(np.isfinite(D), D, -999), axis=1)],
                                    'other'))
        cls = np.where(veg_ok & np.isfinite(D).all(axis=1), cls, 'no cover')
        res['cls'] = cls
        print('pixels per group and class:\n', res.groupby(['group', 'cls']).size().unstack(fill_value=0).to_string())
        ## per period: bins, wet gain / dry loss, hinge slopes
        k_bin = np.digitize(x, self.bin_edges[1:-1])   ## 0 .. 4
        for p, m in pm.items():
            a, s, kb = anom[:, m], x[:, m], k_bin[:, m]
            for k in range(len(self.bin_labels)):
                sel = kb == k
                with np.errstate(invalid='ignore', divide='ignore'):
                    res[f'{p}_bin{k}'] = np.where(sel.sum(axis=1) > 0, np.sum(np.where(sel, a, 0), axis=1) /
                                                  sel.sum(axis=1), np.nan)
                res[f'{p}_nyr_bin{k}'] = sel.sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                wet, dry = s > 0.5, s < -0.5
                res[f'{p}_wet_gain_bin'] = np.where(wet.sum(axis=1) > 0,
                                                    np.sum(np.where(wet, a, 0), axis=1) / wet.sum(axis=1), np.nan)
                res[f'{p}_dry_loss_bin'] = np.where(dry.sum(axis=1) > 0,
                                                    -np.sum(np.where(dry, a, 0), axis=1) / dry.sum(axis=1), np.nan)
            res[f'{p}_wet_gain_hinge'], res[f'{p}_dry_loss_hinge'] = self.hinge(a, s)
            sm_ = s - s.mean(axis=1, keepdims=True)
            res[f'{p}_sens'] = np.sum(sm_ * (a - a.mean(axis=1, keepdims=True)), axis=1) / np.sum(sm_ ** 2, axis=1)
        for v in ['wet_gain_bin', 'dry_loss_bin', 'wet_gain_hinge', 'dry_loss_hinge', 'sens']:
            res[f'd_{v}'] = res[f'P2_{v}'] - res[f'P1_{v}']
        ## how many dry / wet years each period has (do the two periods overlap in x?)
        w = res['area_weight'].values
        print('area-weighted mean number of years per x bin:')
        for p in pm:
            print(' ', p, [round(float(np.sum(res[f'{p}_nyr_bin{k}'].values * w) / np.sum(w)), 1)
                           for k in range(len(self.bin_labels))],
                  [b.replace('\n', ' ') for b in self.bin_labels])
        T.save_df(res, self.outdir + f'Comp_sens_{self.x_var}_pixels_{self.tag}.df')
        return res

    def run_cover_change(self):
        res = self.load()
        tab, slope = self.cover_change_bins(res)
        self.fig_cover_change(tab, slope)

    def cover_change_bins(self, res):
        metrics = [('d_sens', 'all years'), ('d_dry_loss_hinge', 'dry side'), ('d_wet_gain_hinge', 'wet side')]
        res = res[np.isfinite(res['dVEG'])].copy()
        res['dveg_bin'] = np.digitize(res['dVEG'].values, self.dveg_edges[1:-1])
        rows, srows = [], []
        for g in self.group_list:
            d = res[res['group'] == g]
            for col, name in metrics:
                dd = d[np.isfinite(d[col])]
                for k in range(len(self.dveg_labels)):
                    b = dd[dd['dveg_bin'] == k]
                    if len(b) < self.n_min:
                        continue
                    v, w = b[col].values, b['area_weight'].values
                    lo, hi = self.block_ci(b, v, w)
                    rows.append([g, name, k, self.dveg_labels[k], len(b), np.sum(v * w) / np.sum(w), lo, hi])
                if len(dd) >= self.n_min:
                    srows.append([g, name, len(dd), *self.slope_ci(dd, 'dVEG', col)])
        tab = pd.DataFrame(rows, columns=['group', 'side', 'bin', 'dVEG_bin', 'n_pix', 'd_sens', 'ci_low', 'ci_high'])
        slope = pd.DataFrame(srows, columns=['group', 'side', 'n_pix', 'slope', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + f'Comp_sens_{self.x_var}_cover_change_{self.tag}.csv', index=False)
        slope.to_csv(self.outdir + f'Comp_sens_{self.x_var}_cover_change_slope_{self.tag}.csv', index=False)
        print('pixels per group and dVEG bin:\n', res.groupby(['group', 'dveg_bin']).size().unstack(fill_value=0)
              .rename(columns=dict(enumerate(self.dveg_labels))).to_string())
        print(tab.pivot_table(index=['side', 'group'], columns='bin', values='d_sens').round(2).to_string())
        print('change of sensitivity per 1 pp of cover change:\n', slope.round(3).to_string())
        return tab, slope

    def slope_ci(self, d, xc, yc, n_boot=200):
        ## area-weighted LS slope of yc on xc, 95% CI by 1 x 1 deg block bootstrap
        x, y, w = d[xc].values, d[yc].values, d['area_weight'].values

        def fit(i):
            xm = np.sum(x[i] * w[i]) / np.sum(w[i])
            ym = np.sum(y[i] * w[i]) / np.sum(w[i])
            return np.sum(w[i] * (x[i] - xm) * (y[i] - ym)) / np.sum(w[i] * (x[i] - xm) ** 2)
        block = (np.floor(d['lon'].values) * 1000 + np.floor(d['lat'].values)).astype(int)
        ids = np.unique(block)
        idx = {k: np.where(block == k)[0] for k in ids}
        rng = np.random.default_rng(0)
        boot = [fit(np.concatenate([idx[k] for k in rng.choice(ids, size=len(ids))])) for _ in range(n_boot)]
        return fit(np.arange(len(x))), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    def fig_cover_change(self, tab, slope):
        sides = [('all years', 'a) all years: slope of LAI on x'), ('dry side', 'b) dry side: drop per unit x'),
                 ('wet side', 'c) wet side: rise per unit x')]
        x = np.arange(len(self.dveg_labels))
        off = np.linspace(-0.24, 0.24, len(self.group_list))
        sl = slope.set_index(['group', 'side'])
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(1, 3, figsize=(25, 7.5))
            for ax, (side, title) in zip(axs, sides):
                for o, g in zip(off, self.group_list):
                    t = tab[(tab['group'] == g) & (tab['side'] == side)]
                    if len(t) == 0:
                        continue
                    v = t['d_sens'].values
                    lab = g
                    if (g, side) in sl.index:
                        s_, lo, hi = sl.loc[(g, side), ['slope', 'ci_low', 'ci_high']]
                        lab = f'{g}: {s_:+.2f} [{lo:+.2f}, {hi:+.2f}]'
                    ax.errorbar(t['bin'].values + o, v, yerr=[v - t['ci_low'].values, t['ci_high'].values - v],
                                fmt='o-', color=self.group_color[g], ms=8, lw=2, capsize=3, label=lab)
                ax.axhline(0, color='k', lw=1)
                ax.axvspan(2.5, 3.5, color='#EEEEEE', zorder=0)
                ax.set_xticks(x)
                ax.set_xticklabels([l.replace(' to ', '\nto ') for l in self.dveg_labels])
                ax.set_xlabel('Total cover change, 2005-2024 minus 1986-2004 (pp)')
                ax.set_title(title)
                ax.spines[['top', 'right']].set_visible(False)
                ax.legend(frameon=False, fontsize=12, loc='best', title='slope per pp [95% CI]', title_fontsize=12)
            axs[0].set_ylabel('Change of sensitivity, P2 - P1\n(% LAI per unit x)')
            plt.tight_layout()
            plt.savefig(self.outdir + f'Comp_sens_{self.x_var}_cover_change_{self.tag}.png', dpi=300,
                        bbox_inches='tight')
            plt.savefig(self.outdir + f'Comp_sens_{self.x_var}_cover_change_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def hinge(self, a, s):
        ## per pixel LS of a = c + b_wet max(s, 0) + b_dry min(s, 0); returns b_wet, b_dry (NaN if too few years)
        X = np.stack([np.ones_like(s), np.maximum(s, 0), np.minimum(s, 0)], axis=2)
        XtX = np.einsum('pyi,pyj->pij', X, X)
        Xty = np.einsum('pyi,py->pi', X, a)
        ok = ((s > 0).sum(axis=1) >= self.n_side_min) & ((s < 0).sum(axis=1) >= self.n_side_min)
        beta = np.full((len(a), 3), np.nan)
        beta[ok] = np.linalg.solve(XtX[ok], Xty[ok][..., None])[..., 0]
        return beta[:, 1], beta[:, 2]

    def bin_curves(self, res):
        rows = []
        for c in self.class_list:
            d = res[res['cls'] == c]
            for k in range(len(self.bin_labels)):
                dd = d[np.isfinite(d[f'P1_bin{k}']) & np.isfinite(d[f'P2_bin{k}'])]
                if len(dd) < self.n_min:
                    continue
                for p in self.period_dic:
                    v, w = dd[f'{p}_bin{k}'].values, dd['area_weight'].values
                    lo, hi = self.block_ci(dd, v, w)
                    rows.append([c, k, self.bin_labels[k].replace('\n', ' '), p, len(dd), np.sum(v * w) / np.sum(w),
                                 lo, hi])
        tab = pd.DataFrame(rows, columns=['class', 'bin', 'x_bin', 'period', 'n_pix', 'anomaly', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + f'Comp_sens_{self.x_var}_bins_{self.tag}.csv', index=False)
        print(tab.pivot_table(index=['class', 'period'], columns='bin', values='anomaly').round(2).to_string())
        return tab

    def diff_ci(self, d, col, cls, n_boot=200):
        ## area-weighted mean of col in class cls minus that of 'stable', 95% CI by joint 1 x 1 deg block bootstrap
        d = d[np.isfinite(d[col]) & d['cls'].isin([cls, 'stable'])]
        a, b = (d['cls'] == cls).values, (d['cls'] == 'stable').values
        if a.sum() < self.n_min or b.sum() < self.n_min:
            return np.nan, np.nan, np.nan, int(a.sum())
        v, w = d[col].values, d['area_weight'].values

        def stat(idx):
            ia, ib = idx[a[idx]], idx[b[idx]]
            return np.sum(v[ia] * w[ia]) / np.sum(w[ia]) - np.sum(v[ib] * w[ib]) / np.sum(w[ib])
        block = (np.floor(d['lon'].values) * 1000 + np.floor(d['lat'].values)).astype(int)
        ids = np.unique(block)
        idx = {k: np.where(block == k)[0] for k in ids}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(n_boot):
            pick = np.concatenate([idx[k] for k in rng.choice(ids, size=len(ids))])
            if a[pick].sum() and b[pick].sum():
                boot.append(stat(pick))
        return stat(np.arange(len(d))), np.percentile(boot, 2.5), np.percentile(boot, 97.5), int(a.sum())

    def summary(self, res):
        ## raw change (P2 - P1) per class and DiD (class - stable), all groups together and per group
        rows = []
        cols = ['d_wet_gain_hinge', 'd_dry_loss_hinge', 'd_wet_gain_bin', 'd_dry_loss_bin']
        for g in ['All'] + self.group_list:
            d = res if g == 'All' else res[res['group'] == g]
            for col in cols:
                for c in self.class_list:
                    dd = d[(d['cls'] == c) & np.isfinite(d[col])]
                    if len(dd) < self.n_min:
                        continue
                    v, w = dd[col].values, dd['area_weight'].values
                    lo, hi = self.block_ci(dd, v, w)
                    rows.append([g, col, c, 'raw', len(dd), np.sum(v * w) / np.sum(w), lo, hi])
                    if c != 'stable':
                        m, lo, hi, n = self.diff_ci(d, col, c)
                        rows.append([g, col, c, 'DiD', n, m, lo, hi])
        tab = pd.DataFrame(rows, columns=['group', 'metric', 'class', 'kind', 'n_pix', 'value', 'ci_low', 'ci_high'])
        tab.to_csv(self.outdir + f'Comp_sens_{self.x_var}_did_{self.tag}.csv', index=False)
        print(tab[tab['group'] == 'All'].pivot_table(index=['class', 'kind'], columns='metric',
                                                      values='value').round(2).to_string())
        return tab

    def fig_bins(self, tab):
        x = np.arange(len(self.bin_labels))
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(1, 4, figsize=(24, 6.5), sharey=True)
            for ax, c in zip(axs, self.class_list):
                for p, color in [('P1', '#9E9E9E'), ('P2', self.class_color[c])]:
                    t = tab[(tab['class'] == c) & (tab['period'] == p)].set_index('bin').reindex(x)
                    ax.fill_between(x, t['ci_low'], t['ci_high'], color=color, alpha=0.25, lw=0)
                    ax.plot(x, t['anomaly'], 'o-', color=color, lw=2.5, ms=8,
                            label=f'{self.period_dic[p][0]}-{self.period_dic[p][1]}')
                ax.axhline(0, color='k', lw=1)
                ax.set_xticks(x)
                ax.set_xticklabels(self.bin_labels)
                n = tab[tab['class'] == c]['n_pix']
                ax.set_title(f'{c}\n(n = {int(n.max()) if len(n) else 0})')
                ax.set_xlabel(self.x_label)
                ax.spines[['top', 'right']].set_visible(False)
                ax.legend(frameon=False, loc='upper left')
            axs[0].set_ylabel('LAI anomaly (% of period mean)')
            plt.tight_layout()
            plt.savefig(self.outdir + f'Comp_sens_{self.x_var}_bins_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Comp_sens_{self.x_var}_bins_{self.tag}.pdf', bbox_inches='tight')
            plt.close()

    def fig_did(self, tab):
        t = tab[(tab['group'] == 'All') & (tab['kind'] == 'DiD')]
        st = tab[(tab['group'] == 'All') & (tab['class'] == 'stable')].set_index('metric')['value']
        with plt.rc_context(self.big_font):
            fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
            for ax, kind, title, unit in [(axs[0], 'hinge', 'a) per unit x (hinge fit)', '% per unit x'),
                                          (axs[1], 'bin', 'b) mean of years |x| > 0.5', '% of period mean')]:
                for c in self.class_list[:-1]:
                    gx = t[(t['class'] == c) & (t['metric'] == f'd_dry_loss_{kind}')]
                    gy = t[(t['class'] == c) & (t['metric'] == f'd_wet_gain_{kind}')]
                    if len(gx) == 0 or len(gy) == 0:
                        continue
                    vx, vy = gx['value'].values[0], gy['value'].values[0]
                    ax.errorbar(vx, vy, xerr=[[vx - gx['ci_low'].values[0]], [gx['ci_high'].values[0] - vx]],
                                yerr=[[vy - gy['ci_low'].values[0]], [gy['ci_high'].values[0] - vy]], fmt='o', ms=13,
                                color=self.class_color[c], capsize=5, lw=2, label=f'{c} (n = {gx["n_pix"].values[0]})')
                ax.axhline(0, color='k', lw=1)
                ax.axvline(0, color='k', lw=1)
                lim = np.nanmax(np.abs(ax.get_xlim() + ax.get_ylim()))
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                for tx, ty, s, ha, va in [(0.02, 0.98, 'more gain,\nless loss', 'left', 'top'),
                                          (0.98, 0.98, 'amplified\n(boom-bust)', 'right', 'top'),
                                          (0.02, 0.02, 'damped', 'left', 'bottom'),
                                          (0.98, 0.02, 'less gain,\nmore loss', 'right', 'bottom')]:
                    ax.text(tx, ty, s, transform=ax.transAxes, ha=ha, va=va, color='#757575', fontsize=15)
                ax.set_xlabel(f'Change of dry-year loss, P2 - P1\nminus stable pixels ({unit})')
                ax.set_ylabel(f'Change of wet-year gain, P2 - P1\nminus stable pixels ({unit})')
                ax.set_title(f'{title}\nstable pixels: dry {st.get(f"d_dry_loss_{kind}", np.nan):+.2f}, '
                             f'wet {st.get(f"d_wet_gain_{kind}", np.nan):+.2f}')
                ax.spines[['top', 'right']].set_visible(False)
            h, l = axs[0].get_legend_handles_labels()
            plt.tight_layout()
            fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
            plt.savefig(self.outdir + f'Comp_sens_{self.x_var}_did_{self.tag}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Comp_sens_{self.x_var}_did_{self.tag}.pdf', bbox_inches='tight')
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
    # Observed_trend_by_herb_cover().run()   ## 4b: observed trend by cover of 2003-2006 and of 2021-2024 (done)
    ## step 5: long-term (SNU 1986-2024): perennial grass expansion vs more leaf at the same cover
    # PFG_change_vs_LAI_trend().run()   ## 5a: LAI trend along the PFG cover trend, all grass and 3 x 3 strata (done)
    # PFG_change_vs_LAI_trend().light_blue_pfg()   ## 5b: PFG trend of sig. greening without sig. PFG increase (done)
    ## step 6: inside the greening pixels (SNU 1986-2024): did bare ground decline, which RAP cover increased
    # Greening_cover_change().run()   ## 6a: grass / shrub x annual / perennial + savanna; greening vs LAI n.s.;
    ##                                   LAI vs cover regression; RAP time series check (done 2026-11-23)
    # Greening_cover_change().run_lai_per_cover()   ## 6b: LAI per unit vegetation cover, greening vs LAI n.s. (done)
    # Greening_cover_change().run_decomposition()   ## 6c: cover gain vs per-unit-cover greening: map, plane, bars (done)
    ## step 7: does LAI respond differently to equally wet / dry years after the composition changed (P1 vs P2)
    # for x_var in ['SPEI', 'SM']:
    #     Composition_sensitivity(x_var).run()   ## 7a: bins and DiD vs stable pixels (done)
    ## step 8: same pixel, P2 - P1: did the sensitivity to dry / wet years change with the change of total cover?
    # for x_var in ['SPEI', 'SM']:
    #     Composition_sensitivity(x_var).run_cover_change()   ## 8a (done 2026-11-23)
    ## step 9: decomposition with the all-pixel mean (selection-bias check)
    Greening_cover_change().run_decomposition()   ## 9a: 6c figure + diamond = all pixels (2026-11-23)


if __name__ == '__main__':
    main()
