# coding='utf-8'
## CAMS CO2 inversion, batch 1: download and look at it (2026-11-25)
## why  : an independent, atmosphere-based check of the carbon side: did the land uptake of the western US weaken
##        after 2000? Inversions are poorly constrained at this scale (few CO2 stations in the West), so they are used
##        only for the whole region or 2-3 large sub-regions, as a consistency check, never per pixel against LAI
## data : CAMS global inversion-optimised greenhouse gas fluxes (ADS dataset cams-global-greenhouse-gas-inversion),
##        CO2 surface flux, surface-station inversion, monthly mean, version 'latest', 1 deg, global
##        request copied from the ADS download form (2026-11-25)
## login: ADS key in C:\Users\wenzhang1.BLUECAT\.adsapirc (url + key), kept apart from the CDS (ERA5) .cdsapirc;
##        this script only points cdsapi to that file and never reads or prints the key
## steps: 1 download one file per year (skips years already there; a year that is not available is reported)
##        2 open one year, print variables, units, dimensions, and map the annual mean land flux over the study area
## output: Data\CAMS_inversion\CAMS_CO2_flux_{year}.nc (or .zip as delivered), Result\Dataframe\CAMS_inversion\inspect\
##
## ---- CAMS_analysis (2026-11-25) ----
## files : one zip per year with 12 monthly netCDF (cams73_latest_co2_flux_surface_mm_{yyyymm}.nc), extracted once to
##         Data\CAMS_inversion\extract\{year}\
## flux  : posterior land-biosphere flux (variable with 'apos' and 'bio'), prior ('apri' and 'bio') for comparison;
##         sign: positive = source to the atmosphere, negative = uptake (NEE convention); converted to kg C per m2
##         per month from the units attribute (printed, run stops if the unit is not understood)
## region: the study area itself: each CAMS cell weighted by the area of the 0.05 deg LAI pixels inside it
##         (Coarse_lookup().pixel_table(): the pixels of all LAI analyses), split north / south of 38 N
##         -> regional total in Tg C per year (sum of flux x study area in the cell)
## caveat: the West has few CO2 stations; at this scale the posterior is close to the prior and only the regional
##         total and its year-to-year anomalies are read, never single cells; posterior - prior shows what the
##         atmospheric data changed
## years : 1979-2024 (2025 left out: the latest year of an inversion is the least constrained)
## figures ({R2}\04_water_carbon\CAMS_inversion\):
##   Fig_CAMS_NEE_ts         annual posterior NEE of the study area (prior dashed), trends 1982-2024 (dotted),
##                           1982-1999 (dashed), 2000-2024 (solid)
##   Fig_CAMS_NEE_regions    annual and growing-season (Apr-Sep) posterior NEE, north / south of 38 N
##   CAMS_NEE_annual.csv, CAMS_NEE_trend.csv, CAMS_NEE_vs_SPEI.csv (r with best-scale SPEI, raw and detrended)
from __Global__ import *
import zipfile
import calendar
import re
from ET_analysis import *   ## Coarse_lookup (LAI pixels), Time_series_base paths


class CAMS_base:
    def __init__(self):
        self.rc = os.path.expanduser(r'~\.adsapirc')
        self.dataset = 'cams-global-greenhouse-gas-inversion'
        self.datadir = data_root + rf'CAMS_inversion\\'
        self.outdir = result_root + rf'\Dataframe\CAMS_inversion\inspect\\'
        T.mk_dir(self.datadir, force=True)
        T.mk_dir(self.outdir, force=True)
        self.year_list = list(range(1979, 2026))
        self.extent = [-124.529, -102.079, 25.579, 48.979]

    def request(self, year):
        return {'variable': 'carbon_dioxide', 'quantity': 'surface_flux', 'input_observations': 'surface',
                'time_aggregation': 'monthly_mean', 'version': 'latest', 'year': [str(year)],
                'month': [f'{m:02d}' for m in range(1, 13)]}

    def year_file(self, year):
        for ext in ['.nc', '.zip']:
            f = self.datadir + f'CAMS_CO2_flux_{year}{ext}'
            if os.path.isfile(f) and os.path.getsize(f) > 0:
                return f
        return None


class CAMS_download(CAMS_base):
    def run(self):
        os.environ['CDSAPI_RC'] = self.rc            ## ADS key file, not the ERA5 one
        import cdsapi
        c = cdsapi.Client()
        print('server:', c.url)
        failed = []
        for year in self.year_list:
            if self.year_file(year):
                print('have', year)
                continue
            tmp = self.datadir + f'CAMS_CO2_flux_{year}.download'
            try:
                c.retrieve(self.dataset, self.request(year)).download(tmp)
            except Exception as e:
                failed.append((year, repr(e)[:300]))
                print('failed', year, repr(e)[:300])
                continue
            with open(tmp, 'rb') as fr:   ## name the file by what was delivered (zip starts with PK)
                ext = '.zip' if fr.read(2) == b'PK' else '.nc'
            os.replace(tmp, self.datadir + f'CAMS_CO2_flux_{year}{ext}')
            print('done', year, ext)
        print('failed years:', [y for y, _ in failed])


class CAMS_inspect(CAMS_base):
    def run(self, year=2010):
        f = self.year_file(year)
        if f is None:
            raise FileNotFoundError(f'no file for {year} in {self.datadir}')
        if f.endswith('.zip'):
            z = zipfile.ZipFile(f)
            print(f, 'contains:', z.namelist())
            ex = self.datadir + rf'extract\{year}\\'
            z.extractall(ex)
            f = [ex + n for n in z.namelist() if n.endswith('.nc')][0]
        ds = xarray.open_dataset(f)
        lines = [str(ds)] + [f'{v}: dims {ds[v].dims}, attrs {dict(ds[v].attrs)}' for v in ds.data_vars]
        txt = '\n'.join(lines)
        print(txt)
        open(self.outdir + f'inspect_{year}.txt', 'w').write(txt)
        self.quick_map(ds, year)

    def quick_map(self, ds, year):
        ## annual mean of the posterior land flux (the variable whose name has 'land' and not 'apri' / 'prior')
        cand = [v for v in ds.data_vars if 'land' in v.lower() and 'apri' not in v.lower() and 'prior' not in v.lower()]
        if not cand:
            print('no land flux variable found; variables:', list(ds.data_vars))
            return
        a = ds[cand[0]]
        latn = [d for d in a.dims if 'lat' in d.lower()][0]
        lonn = [d for d in a.dims if 'lon' in d.lower()][0]
        if float(a[lonn].max()) > 180:   ## 0..360 -> -180..180
            a = a.assign_coords({lonn: ((a[lonn] + 180) % 360) - 180}).sortby(lonn)
        a = a.sortby(latn)
        lon0, lon1, lat0, lat1 = self.extent
        a = a.sel({latn: slice(lat0 - 1, lat1 + 1), lonn: slice(lon0 - 1, lon1 + 1)})
        other = [d for d in a.dims if d not in (latn, lonn)]
        m = a.mean(dim=other) if other else a
        with plt.rc_context({'font.size': 22}):
            fig, ax = plt.subplots(figsize=(12, 11))
            v = np.nanpercentile(np.abs(m.values), 98)
            im = ax.pcolormesh(m[lonn], m[latn], m.values, cmap='RdBu_r', vmin=-v, vmax=v, shading='auto')
            cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.07)
            cb.set_label(f'{cand[0]}, {year} mean ({a.attrs.get("units", "")}); negative = uptake')
            ax.set_aspect(1 / np.cos(np.deg2rad(37)))
            plt.savefig(self.outdir + f'quicklook_{cand[0]}_{year}.png', dpi=150, bbox_inches='tight')
            plt.close()


class CAMS_analysis(CAMS_base):
    def __init__(self):
        super().__init__()
        self.extract_dir = self.datadir + 'extract\\'
        self.outdir2 = this_root + rf'\{R2}\04_water_carbon\CAMS_inversion\\'   ## 2026-11-25: was under Result\ by mistake
        T.mk_dir(self.outdir2, force=True)
        self.years = np.arange(1979, 2025)
        self.period_list = [(1982, 2024, ':'), (1982, 1999, '--'), (2000, 2024, '-')]
        self.font = {'font.size': 26, 'axes.titlesize': 28, 'axes.labelsize': 28, 'xtick.labelsize': 24,
                     'ytick.labelsize': 24, 'legend.fontsize': 24}
        self.spei_csv = this_root + rf'\{R2}\01_drought\SPEI_best_scale_1982_2024\Fig_SPEI_best_GS.csv'

    def run(self):
        grid = self.grid_info()
        w_s, w_n = self.study_area_weights(grid)
        tab = self.regional_series(grid, w_s, w_n)   ## about 5 min (552 monthly files)
        self.trends(tab)
        self.vs_spei(tab)
        self.fig_ts(tab)
        self.fig_regions(tab)

    def month_file(self, y, m):
        d = self.extract_dir + f'{y}\\'
        if not os.path.isdir(d) or not os.listdir(d):
            f = self.year_file(y)
            if f is None or not f.endswith('.zip'):
                raise FileNotFoundError(f'no zip for {y}')
            zipfile.ZipFile(f).extractall(d)
        fs = [x for x in os.listdir(d) if x.endswith(f'_{y}{m:02d}.nc')]
        if not fs:
            raise FileNotFoundError(f'no netCDF for {y}-{m:02d} in {d}')
        return d + fs[0]

    def pick(self, ds, must, avoid=()):
        c = [v for v in ds.data_vars if all(k in v.lower() for k in must) and not any(k in v.lower() for k in avoid)]
        return c[0] if c else None

    def grid_info(self):
        ds = xarray.open_dataset(self.month_file(2010, 7))
        print(ds)
        post = self.pick(ds, ['apos', 'bio']) or self.pick(ds, ['bio'], ['apri'])
        prior = self.pick(ds, ['apri', 'bio'])
        if post is None:
            raise ValueError(f'no posterior biosphere flux among {list(ds.data_vars)}')
        units = ds[post].attrs.get('units', '')
        latn = [d for d in ds[post].dims if 'lat' in d.lower()][0]
        lonn = [d for d in ds[post].dims if 'lon' in d.lower()][0]
        lat, lon = ds[latn].values, ds[lonn].values
        if 'area' in ds:
            area = np.array(ds['area'].values, dtype=float).squeeze()
        else:   ## spherical cell area from the cell centres
            dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
            R = 6371e3
            la = np.deg2rad(lat)[:, None]
            area = R ** 2 * np.deg2rad(dlon) * (np.sin(la + np.deg2rad(dlat) / 2) - np.sin(la - np.deg2rad(dlat) / 2))
            area = np.repeat(area, len(lon), axis=1)
        print(f'posterior: {post} | prior: {prior} | units: {units!r} | grid {len(lat)} x {len(lon)} '
              f'(dlat {lat[1] - lat[0]:.3f}, dlon {lon[1] - lon[0]:.3f})')
        ds.close()
        return {'post': post, 'prior': prior, 'units': units, 'lat': lat, 'lon': lon, 'latn': latn, 'lonn': lonn,
                'area': area}

    def to_kgC_m2_month(self, units, y, m):
        ## factor from the units attribute to kg C m-2 per month
        u = units.replace(' ', '').lower()
        f = 1.0
        if 'kgco2' in u or ('co2' in u and 'kgc' not in u):
            f *= 12.0 / 44.0
        if u.startswith('g') or 'gc' in u.split('kg')[0] and not u.startswith('kg'):
            f *= 1e-3
        days = calendar.monthrange(y, m)[1]
        if 's-1' in u or '/s' in u:
            f *= days * 86400
        elif 'day-1' in u or 'd-1' in u or '/day' in u:
            f *= days
        elif 'yr-1' in u or 'year-1' in u or '/yr' in u:
            f *= days / 365.25
        elif 'month-1' in u or 'mon-1' in u or '/month' in u:
            pass
        else:
            raise ValueError(f'unit not understood: {units!r}; tell Claude the unit')
        if 'm-2' not in u and 'm^-2' not in u and '/m2' not in u:
            raise ValueError(f'unit is not per m2: {units!r}; tell Claude the unit')
        return f

    def study_area_weights(self, grid):
        ## study area (m2) of each CAMS cell, south / north of 38 N, from the 0.05 deg LAI pixels
        info = Coarse_lookup().pixel_table()
        lat, lon = grid['lat'], grid['lon']
        lon_c = np.where(lon > 180, lon - 360, lon)
        dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
        i = np.argmin(np.abs(info['lat'].values[:, None] - lat[None, :]), axis=1)
        j = np.argmin(np.abs(info['lon'].values[:, None] - lon_c[None, :]), axis=1)
        px = (0.05 * 111.32e3) ** 2 * np.cos(np.deg2rad(info['lat'].values))   ## pixel area, m2
        w_s = np.zeros(grid['area'].shape)
        w_n = np.zeros(grid['area'].shape)
        south = info['lat'].values < 38
        np.add.at(w_s, (i[south], j[south]), px[south])
        np.add.at(w_n, (i[~south], j[~south]), px[~south])
        n = int(((w_s + w_n) > 0).sum())
        print(f'study pixels: {len(info)} | CAMS cells touched: {n} | study area {(w_s + w_n).sum() / 1e12:.2f} '
              f'million km2 (south {w_s.sum() / 1e12:.2f}, north {w_n.sum() / 1e12:.2f}) | cell size {dlat:.2f} x '
              f'{dlon:.2f} deg')
        return w_s, w_n

    def regional_series(self, grid, w_s, w_n):
        rows = []
        for y in tqdm(self.years, desc='CAMS years'):
            for m in range(1, 13):
                ds = xarray.open_dataset(self.month_file(y, m))
                f = self.to_kgC_m2_month(ds[grid['post']].attrs.get('units', grid['units']), y, m)
                row = [y, m]
                for v in [grid['post'], grid['prior']]:
                    if v is None:
                        row += [np.nan, np.nan]
                        continue
                    a = np.array(ds[v].values, dtype=float).squeeze() * f      ## kg C m-2 month-1
                    a = np.where(np.isfinite(a), a, 0)
                    row += [np.sum(a * w_s) / 1e9, np.sum(a * w_n) / 1e9]      ## Tg C per month
                ds.close()
                rows.append(row)
        mon = pd.DataFrame(rows, columns=['year', 'month', 'post_south', 'post_north', 'prior_south', 'prior_north'])
        mon['post_all'] = mon['post_south'] + mon['post_north']
        mon['prior_all'] = mon['prior_south'] + mon['prior_north']
        ann = mon.groupby('year').sum(numeric_only=True).drop(columns='month')
        gs = mon[mon['month'].between(4, 9)].groupby('year').sum(numeric_only=True).drop(columns='month')
        tab = ann.add_suffix('_ann').join(gs.add_suffix('_AprSep'))
        tab.to_csv(self.outdir2 + 'CAMS_NEE_annual.csv')
        mon.to_csv(self.outdir2 + 'CAMS_NEE_monthly.csv', index=False)
        print(tab[['post_all_ann', 'prior_all_ann', 'post_all_AprSep']].describe().round(1).to_string())
        return tab

    def trends(self, tab):
        from scipy import stats
        rows = []
        for col in ['post_all_ann', 'prior_all_ann', 'post_south_ann', 'post_north_ann', 'post_all_AprSep']:
            for t0, t1, _ in self.period_list:
                d = tab.loc[t0:t1, col]
                sl, ic, _, p, _ = stats.linregress(d.index.values, d.values)
                rows.append([col, f'{t0}-{t1}', d.mean(), sl * 10, p])
        out = pd.DataFrame(rows, columns=['series', 'period', 'mean_TgC_per_yr', 'trend_TgC_per_decade', 'p'])
        m1 = tab.loc[1982:1999, 'post_all_ann'].mean()
        m2 = tab.loc[2000:2024, 'post_all_ann'].mean()
        print(f'posterior NEE, study area: 1982-1999 {m1:.1f}, 2000-2024 {m2:.1f} Tg C per year (negative = uptake)')
        out.to_csv(self.outdir2 + 'CAMS_NEE_trend.csv', index=False)
        print(out.round(2).to_string())
        return out

    def vs_spei(self, tab):
        from scipy import stats
        sp = pd.read_csv(self.spei_csv).set_index('year')['mean']
        rows = []
        for col in ['post_all_ann', 'post_all_AprSep', 'prior_all_ann']:
            d = pd.concat([tab[col], sp], axis=1, join='inner').dropna()
            x, yv = d['mean'].values, d[col].values
            r, p = stats.pearsonr(x, yv)
            t = d.index.values.astype(float)
            xr_ = x - np.polyval(np.polyfit(t, x, 1), t)
            yr_ = yv - np.polyval(np.polyfit(t, yv, 1), t)
            r2, p2 = stats.pearsonr(xr_, yr_)
            rows.append([col, len(d), r, p, r2, p2])
        out = pd.DataFrame(rows, columns=['series', 'n_years', 'r', 'p', 'r_detrended', 'p_detrended'])
        out.to_csv(self.outdir2 + 'CAMS_NEE_vs_SPEI.csv', index=False)
        print('NEE vs best-scale SPEI (wetter -> more uptake gives negative r):')
        print(out.round(3).to_string())

    def fig_ts(self, tab):
        from scipy import stats
        c = '#6A3D9A'
        yrs = tab.index.values
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(20, 8))
            ax.plot(yrs, tab['prior_all_ann'], color='#9E9E9E', lw=2.5, ls='--', label='prior')
            ax.plot(yrs, tab['post_all_ann'], color=c, lw=3.5, label='posterior')
            text = []
            for t0, t1, ls in self.period_list:
                d = tab.loc[t0:t1, 'post_all_ann']
                sl, ic, _, p, _ = stats.linregress(d.index.values, d.values)
                ax.plot(d.index.values, sl * d.index.values + ic, color=c, lw=3, ls=ls)
                text.append(f'{t0}-{t1}: {sl * 10:+.1f} Tg C / decade (p = {p:.2f})')
            ax.text(0.01, 0.97, '\n'.join(text), transform=ax.transAxes, va='top', fontsize=22, color=c)
            ax.axhline(0, color='grey', lw=1)
            ax.axvline(1999.5, color='grey', lw=1.5, ls=':')
            ax.set_xlim(1982, yrs[-1])
            ax.set_ylabel('NEE (Tg C per year)\nnegative = uptake')
            ax.set_xlabel('Year')
            ax.set_title(getattr(self, 'ts_title', 'CAMS inversion (surface stations), land biosphere flux of the study area'),
                         loc='left')
            ax.legend(frameon=False, loc='lower right')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir2 + 'Fig_CAMS_NEE_ts.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir2 + 'Fig_CAMS_NEE_ts.pdf', bbox_inches='tight')
            plt.close()

    def fig_regions(self, tab):
        yrs = tab.index.values
        with plt.rc_context(self.font):
            fig, axs = plt.subplots(1, 2, figsize=(26, 8), sharey=False)
            for ax, suf, ttl in [(axs[0], 'ann', 'annual'), (axs[1], 'AprSep', 'April-September')]:
                for col, cl, lab in [('south', '#D9480F', 'south of 38 N'), ('north', '#2171B5', 'north of 38 N')]:
                    ax.plot(yrs, tab[f'post_{col}_{suf}'], color=cl, lw=3.5, label=lab)
                    ax.plot(yrs, tab[f'prior_{col}_{suf}'], color=cl, lw=2, ls='--', alpha=0.6)
                ax.axhline(0, color='grey', lw=1)
                ax.axvline(1999.5, color='grey', lw=1.5, ls=':')
                ax.set_xlim(1982, yrs[-1])
                ax.set_title(f'{ttl} (solid posterior, dashed prior)', loc='left')
                ax.set_ylabel('NEE (Tg C), negative = uptake')
                ax.spines[['top', 'right']].set_visible(False)
            axs[0].legend(frameon=False, loc='lower left')
            plt.tight_layout()
            plt.savefig(self.outdir2 + 'Fig_CAMS_NEE_regions.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir2 + 'Fig_CAMS_NEE_regions.pdf', bbox_inches='tight')
            plt.close()


class CarboScope_analysis(CAMS_analysis):
    ## 2026-11-25: Jena CarboScope nbetEXToc_v2025 (NBE-T inversion, extended station set, ocean prior; includes
    ##   disturbance and fire, like the CAMS 'bio' flux), the same study-area totals and figures as CAMS_analysis,
    ##   native CarboScope grid, plus a comparison with CAMS
    ## file  : Data\carbonscope\nc\nbetEXToc_v2025.flux.nc (6.4 GB, read lazily, only the western box is loaded)
    ## var   : co2flux_land (time axis 'mtime'); units from the attribute: per m2 (density) or per grid cell (then
    ##         divided by the cell area, variable dxyp / area or computed)
    ## output: {R2}\04_water_carbon\CarboScope_inversion\ (same file names as CAMS, prefix CarboScope_)
    def __init__(self):
        super().__init__()
        self.nc = data_root + r'carbonscope\nc\nbetEXToc_v2025.flux.nc'
        self.outdir2 = this_root + rf'\{R2}\04_water_carbon\CarboScope_inversion\\'
        T.mk_dir(self.outdir2, force=True)
        self.cams_csv = this_root + rf'\{R2}\04_water_carbon\CAMS_inversion\CAMS_NEE_annual.csv'
        self.var = 'co2flux_land'
        self.ts_title = 'CarboScope nbetEXToc v2025, land flux (incl. disturbance) of the study area'

    def run(self):
        ds, grid = self.open_box()
        w_s, w_n = self.study_area_weights(grid)
        tab = self.regional_series_cs(ds, grid, w_s, w_n)
        self.years = tab.index.values
        self.trends(tab)
        self.vs_spei(tab)
        self.fig_ts(tab)
        self.fig_regions(tab)
        self.fig_compare(tab)

    def open_box(self):
        ds = xarray.open_dataset(self.nc)
        print(ds)
        a = ds[self.var]
        print(self.var, dict(a.attrs))
        tn = [d for d in a.dims if 'time' in d.lower()][0]
        latn = [d for d in a.dims if 'lat' in d.lower()][0]
        lonn = [d for d in a.dims if 'lon' in d.lower()][0]
        lat, lon = ds[latn].values, ds[lonn].values
        lon_c = np.where(lon > 180, lon - 360, lon)
        li = np.where((lat > 24) & (lat < 51))[0]
        lj = np.where((lon_c > -128) & (lon_c < -99))[0]
        area_var = [v for v in ['dxyp', 'area', 'cell_area'] if v in ds]
        if area_var:
            area = np.array(ds[area_var[0]].values, dtype=float)
            area = area[li][:, None] * np.ones(len(lj)) if area.ndim == 1 else area[np.ix_(li, lj)]
        else:
            dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
            la = np.deg2rad(lat[li])[:, None]
            area = 6371e3 ** 2 * np.deg2rad(dlon) * (np.sin(la + np.deg2rad(dlat) / 2) - np.sin(la - np.deg2rad(dlat) / 2))
            area = np.repeat(area, len(lj), axis=1)
        grid = {'lat': lat[li], 'lon': lon[lj], 'area': area, 'li': li, 'lj': lj, 'tn': tn, 'latn': latn,
                'lonn': lonn, 'units': a.attrs.get('units', '')}
        print(f'time steps: {ds.sizes[tn]} ({str(ds[tn].values[0])[:10]} .. {str(ds[tn].values[-1])[:10]}) | box '
              f'{len(li)} x {len(lj)} cells | units {grid["units"]!r} | area from '
              f'{area_var[0] if area_var else "cell centres"}')
        return ds, grid

    def rate_factor(self, units):
        ## factor to Tg C per year from (value x study area / cell area) [per cell units] or (value x study area)
        ##   [per m2 units]; returns (factor, per_cell)
        u = units.replace(' ', '').lower()
        per_cell = not ('m-2' in u or '/m2' in u or 'm^-2' in u)
        f = 1.0
        if u.startswith('pgc') or 'pgc' in u:
            f *= 1e3                      ## Pg -> Tg
        elif u.startswith('tgc'):
            f *= 1.0
        elif u.startswith('gc') or u.startswith('g/'):
            f *= 1e-12
        elif u.startswith('kgc') or u.startswith('kg'):
            f *= 1e-9
        elif u.startswith('molc') or u.startswith('mol'):
            f *= 12e-12
        else:
            raise ValueError(f'mass unit not understood: {units!r}; tell Claude the unit')
        if 'co2' in u and 'c' not in u.replace('co2', ''):
            f *= 12 / 44
        if '/s' in u or 's-1' in u:
            f *= 365.25 * 86400
        elif '/day' in u or 'day-1' in u or 'd-1' in u:
            f *= 365.25
        elif '/month' in u or 'month-1' in u:
            f *= 12
        elif '/yr' in u or 'yr-1' in u or 'year' in u or '/a' in u:
            pass
        else:
            raise ValueError(f'time unit not understood: {units!r}; tell Claude the unit')
        return f, per_cell

    def regional_series_cs(self, ds, grid, w_s, w_n):
        f, per_cell = self.rate_factor(grid['units'])
        a = ds[self.var].isel({grid['latn']: grid['li'], grid['lonn']: grid['lj']})
        x = np.array(a.values, dtype=float)                       ## (time, lat, lon), only the box is read
        x = np.where(np.isfinite(x), x, 0)
        if per_cell:
            x = x / grid['area'][None]                            ## per m2
        t = pd.to_datetime(ds[grid['tn']].values)
        s = np.sum(x * w_s[None], axis=(1, 2)) * f                ## Tg C per year (rate) of the study area
        n = np.sum(x * w_n[None], axis=(1, 2)) * f
        d = pd.DataFrame({'time': t, 'south': s, 'north': n})
        d['year'], d['month'] = d['time'].dt.year, d['time'].dt.month
        d['all'] = d['south'] + d['north']
        mon = d.groupby(['year', 'month'])[['south', 'north', 'all']].mean().reset_index()
        mon.to_csv(self.outdir2 + 'CarboScope_NEE_monthly_rate.csv', index=False)
        ann = mon.groupby('year')[['south', 'north', 'all']].mean()                       ## Tg C per year
        gs = mon[mon['month'].between(4, 9)].groupby('year')[['south', 'north', 'all']].mean() * 0.5   ## Apr-Sep total
        ann = ann[mon.groupby('year').size() == 12]
        tab = pd.DataFrame(index=ann.index)
        for c in ['south', 'north', 'all']:
            tab[f'post_{c}_ann'] = ann[c]
            tab[f'prior_{c}_ann'] = np.nan
            tab[f'post_{c}_AprSep'] = gs[c]
            tab[f'prior_{c}_AprSep'] = np.nan
        tab = tab[tab.index <= 2024]
        tab.index.name = 'year'
        tab.to_csv(self.outdir2 + 'CarboScope_NEE_annual.csv')
        print(tab[['post_all_ann', 'post_south_ann', 'post_north_ann', 'post_all_AprSep']].round(1).to_string())
        return tab

    def trends(self, tab):
        out = super().trends(tab.assign(prior_all_ann=tab['post_all_ann']))   ## no prior in this file
        out = out[out['series'] != 'prior_all_ann']
        os.replace(self.outdir2 + 'CAMS_NEE_trend.csv', self.outdir2 + 'CarboScope_NEE_trend.csv')
        out.to_csv(self.outdir2 + 'CarboScope_NEE_trend.csv', index=False)
        return out

    def vs_spei(self, tab):
        super().vs_spei(tab.assign(prior_all_ann=tab['post_all_ann']))
        os.replace(self.outdir2 + 'CAMS_NEE_vs_SPEI.csv', self.outdir2 + 'CarboScope_NEE_vs_SPEI.csv')

    def fig_ts(self, tab):
        super().fig_ts(tab)
        for ext in ['png', 'pdf']:
            os.replace(self.outdir2 + f'Fig_CAMS_NEE_ts.{ext}', self.outdir2 + f'Fig_CarboScope_NEE_ts.{ext}')

    def fig_regions(self, tab):
        super().fig_regions(tab)
        for ext in ['png', 'pdf']:
            os.replace(self.outdir2 + f'Fig_CAMS_NEE_regions.{ext}', self.outdir2 + f'Fig_CarboScope_NEE_regions.{ext}')

    def fig_compare(self, tab):
        ## annual NEE of the study area, CAMS vs CarboScope, and their correlation (raw, detrended)
        from scipy import stats
        if not os.path.isfile(self.cams_csv):
            print('no CAMS csv, comparison skipped:', self.cams_csv)
            return
        cams = pd.read_csv(self.cams_csv, index_col=0)
        d = pd.concat({'CAMS': cams['post_all_ann'], 'CarboScope': tab['post_all_ann']}, axis=1).dropna()
        r, p = stats.pearsonr(d['CAMS'], d['CarboScope'])
        t = d.index.values.astype(float)
        dt = {k: d[k] - np.polyval(np.polyfit(t, d[k], 1), t) for k in d}
        r2, p2 = stats.pearsonr(dt['CAMS'], dt['CarboScope'])
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(20, 8))
            ax.plot(cams.index, cams['post_all_ann'], color='#6A3D9A', lw=3.5, label='CAMS (surface stations)')
            ax.plot(tab.index, tab['post_all_ann'], color='#E08214', lw=3.5, label='CarboScope nbetEXToc v2025')
            ax.axhline(0, color='grey', lw=1)
            ax.axvline(1999.5, color='grey', lw=1.5, ls=':')
            ax.set_xlim(1982, 2024)
            ax.text(0.01, 0.97, f'r = {r:.2f} (p = {p:.3f}), detrended r = {r2:.2f} (p = {p2:.3f}), '
                                f'{int(d.index.min())}-{int(d.index.max())}', transform=ax.transAxes, va='top', fontsize=22)
            ax.set_ylabel('NEE (Tg C per year)\nnegative = uptake')
            ax.set_xlabel('Year')
            ax.legend(frameon=False, loc='lower right')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir2 + 'Fig_inversions_compare.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir2 + 'Fig_inversions_compare.pdf', bbox_inches='tight')
            plt.close()
        print(f'CAMS vs CarboScope: r = {r:.2f} (p = {p:.3f}); detrended r = {r2:.2f} (p = {p2:.3f})')


class GCB_inversions(CarboScope_analysis):
    ## 2026-11-25: Global Carbon Budget 2025 inversion ensemble (14 atmospheric inversions, ICOS Carbon Portal,
    ##   Luijkx et al. 2025, doi:10.18160/VZ3B-GHYQ), 1 x 1 deg, monthly, 1976-2024 (members start in different years)
    ## file  : Data\GCB_inversions\GCP2025_inversions_1x1_*.nc (newest version on the ICOS page)
    ## var   : land_flux_only_fossil_cement_adjusted (posterior land flux of each member, fossil emissions harmonised
    ##         across members) and prior_flux_land; member dimension = the dimension that is not time / lat / lon
    ## method: as CAMS / CarboScope: native 1 deg, study-area total (cells weighted by the 0.05 deg LAI pixel area),
    ##         annual NEE per member (only years with 12 valid months)
    ## consensus: per member the trend 2001-2024 (period most members cover) and, for members starting by 1982,
    ##         1982-2024; how many members have a positive trend (weakening uptake); ensemble median and spread
    ## output: {R2}\04_water_carbon\GCB_inversions\
    ##   GCB_NEE_members.csv, GCB_member_trends.csv, Fig_GCB_NEE_ensemble.png, Fig_GCB_member_trends.png
    def __init__(self):
        super().__init__()
        d = data_root + r'GCB_inversions\\'
        fs = sorted(os.path.join(r_, f) for r_, _, ff in os.walk(d) for f in ff if f.endswith('.nc')) \
            if os.path.isdir(d) else []   ## 2026-11-25: also in the unzipped sub-folder
        if not fs:
            raise FileNotFoundError('put the GCB inversion netCDF in ' + d)
        self.nc = fs[-1]
        print('GCB file:', self.nc)
        ## 2026-11-25: label from the file name (GCP2026_... -> GCB2026), not hard-coded
        yr = re.search(r'GC[PB](\d{4})', os.path.basename(self.nc))
        self.tag = f'GCB{yr.group(1)}' if yr else 'GCB'
        self.outdir2 = this_root + rf'\{R2}\04_water_carbon\GCB_inversions\\'
        T.mk_dir(self.outdir2, force=True)
        self.var = 'land_flux_only_fossil_cement_adjusted'
        self.cs_csv = this_root + rf'\{R2}\04_water_carbon\CarboScope_inversion\CarboScope_NEE_annual.csv'
        self.trend_periods = [(2001, 2024), (1982, 2024), (1982, 1999), (2000, 2024)]

    def run(self):
        ds, grid = self.open_box()
        w_s, w_n = self.study_area_weights(grid)
        tab = self.member_series(ds, grid, w_s + w_n)
        tr = self.member_trends(tab)
        self.fig_ensemble(tab)
        self.fig_member_trends(tr)
        self.ensemble_vs_spei(tab)

    def member_series(self, ds, grid, w):
        f, per_cell = self.rate_factor(grid['units'])
        a = ds[self.var]
        mdim = [d for d in a.dims if d not in (grid['tn'], grid['latn'], grid['lonn'])]
        if len(mdim) != 1:
            raise ValueError(f'member dimension not found: dims {a.dims}')
        mdim = mdim[0]
        names = None
        for v in ds.variables:   ## a variable with the member names, if any
            if mdim in ds[v].dims and ds[v].ndim <= 2 and v != self.var and ds[v].dtype.kind in 'SUO':
                vals = ds[v].values
                ## 2026-11-25: names are stored with NUL padding between characters ('C\0\0\0A...'): drop the NULs
                names = [self.clean_name(x) if ds[v].ndim == 1 else ''.join(self.clean_name(c) for c in x)
                         for x in vals]
        n_mem = ds.sizes[mdim]
        names = names if names and len(names) == n_mem else [f'member_{k + 1:02d}' for k in range(n_mem)]
        print('members:', names)
        t = pd.to_datetime(ds[grid['tn']].values)
        sub = a.isel({grid['latn']: grid['li'], grid['lonn']: grid['lj']})
        rows = []
        for k, nm in enumerate(names):
            x = np.array(sub.isel({mdim: k}).transpose(grid['tn'], grid['latn'], grid['lonn']).values, dtype=float)
            valid = np.isfinite(x[:, w > 0]).all(axis=1)          ## month valid: every study cell has a value
            x = np.where(np.isfinite(x), x, 0)
            if per_cell:
                x = x / grid['area'][None]
            v = np.sum(x * w[None], axis=(1, 2)) * f               ## Tg C per year (rate)
            v[~valid] = np.nan
            rows.append(pd.DataFrame({'member': nm, 'year': t.year, 'month': t.month, 'rate': v}))
        mon = pd.concat(rows)
        g = mon.groupby(['member', 'year'])['rate']
        ann = (g.mean().where(g.count() == 12)).unstack('member')     ## Tg C per year, full years only
        ann = ann[ann.index <= 2024]
        ann.to_csv(self.outdir2 + 'GCB_NEE_members.csv')
        cov = ann.notna().apply(lambda c: f'{int(c[c].index.min())}-{int(c[c].index.max())}' if c.any() else 'none')
        print('years covered per member:')
        print(cov.to_string())
        return ann

    def clean_name(self, x):
        ## bytes or str -> plain text without NUL characters and blanks at the ends
        x = x.decode('utf-8', 'ignore') if isinstance(x, bytes) else str(x)
        return x.replace('\x00', '').strip()

    def member_trends(self, ann):
        from scipy import stats
        rows = []
        for m in ann.columns:
            for t0, t1 in self.trend_periods:
                d = ann.loc[t0:t1, m].dropna()
                if len(d) < 0.9 * (t1 - t0 + 1):
                    continue
                sl, ic, _, p, _ = stats.linregress(d.index.values, d.values)
                rows.append([m, f'{t0}-{t1}', len(d), d.mean(), sl * 10, p])
        tr = pd.DataFrame(rows, columns=['member', 'period', 'n_years', 'mean_TgC_per_yr', 'trend_TgC_per_decade', 'p'])
        tr.to_csv(self.outdir2 + 'GCB_member_trends.csv', index=False)
        for per, d in tr.groupby('period'):
            print(f'{per}: {len(d)} members | positive trend (weakening uptake) {int((d.trend_TgC_per_decade > 0).sum())}'
                  f' (sig. {int(((d.trend_TgC_per_decade > 0) & (d.p < 0.05)).sum())}) | negative '
                  f'{int((d.trend_TgC_per_decade < 0).sum())} (sig. {int(((d.trend_TgC_per_decade < 0) & (d.p < 0.05)).sum())})'
                  f' | median trend {d.trend_TgC_per_decade.median():+.1f} Tg C per decade')
        return tr

    def fig_ensemble(self, ann):
        yrs = ann.index.values
        med = ann.median(axis=1, skipna=True).where(ann.notna().sum(axis=1) >= 5)
        q1 = ann.quantile(0.25, axis=1).where(ann.notna().sum(axis=1) >= 5)
        q3 = ann.quantile(0.75, axis=1).where(ann.notna().sum(axis=1) >= 5)
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(20, 9))
            for m in ann.columns:
                ax.plot(yrs, ann[m], color='#BDBDBD', lw=1.2)
            ax.fill_between(yrs, q1, q3, color='#1B7837', alpha=0.25, lw=0, label=f'{self.tag} inversions, IQR')
            ax.plot(yrs, med, color='#1B7837', lw=4, label=f'{self.tag} median (>= 5 members)')
            for f_, col, lab in [(self.cams_csv, '#6A3D9A', 'CAMS'), (self.cs_csv, '#E08214', 'CarboScope v2025')]:
                if os.path.isfile(f_):
                    d = pd.read_csv(f_, index_col=0)['post_all_ann']
                    ax.plot(d.index, d.values, color=col, lw=2.5, ls='--', label=lab)
            ax.axhline(0, color='grey', lw=1)
            ax.axvline(1999.5, color='grey', lw=1.5, ls=':')
            ax.set_xlim(1980, 2024)
            ax.set_ylabel('NEE of the study area (Tg C per year)\nnegative = uptake')
            ax.set_xlabel('Year (grey: single members)')
            ax.legend(frameon=False, ncol=2, loc='upper left')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir2 + 'Fig_GCB_NEE_ensemble.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir2 + 'Fig_GCB_NEE_ensemble.pdf', bbox_inches='tight')
            plt.close()

    def fig_member_trends(self, tr):
        d = tr[tr['period'] == '2001-2024'].sort_values('trend_TgC_per_decade')
        if d.empty:
            return
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(14, 0.6 * len(d) + 3))
            col = ['#B2182B' if v > 0 else '#2166AC' for v in d['trend_TgC_per_decade']]
            ax.barh(range(len(d)), d['trend_TgC_per_decade'], color=col)
            for i, (v, p) in enumerate(zip(d['trend_TgC_per_decade'], d['p'])):
                ax.text(v, i, ' *' if p < 0.05 else '', va='center', ha='left' if v >= 0 else 'right', fontsize=26)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels(d['member'])
            ax.axvline(0, color='k', lw=1)
            ax.set_xlabel('NEE trend 2001-2024 (Tg C per decade)\npositive = weakening uptake, * p < 0.05')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.outdir2 + 'Fig_GCB_member_trends.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir2 + 'Fig_GCB_member_trends.pdf', bbox_inches='tight')
            plt.close()

    def ensemble_vs_spei(self, ann):
        from scipy import stats
        sp = pd.read_csv(self.spei_csv).set_index('year')['mean']
        rows = []
        for m in list(ann.columns) + ['median']:
            y = ann.median(axis=1).where(ann.notna().sum(axis=1) >= 5) if m == 'median' else ann[m]
            d = pd.concat([y.rename('nee'), sp], axis=1, join='inner').dropna()
            if len(d) < 10:
                continue
            t = d.index.values.astype(float)
            r, p = stats.pearsonr(d['mean'], d['nee'])
            xr_ = d['mean'] - np.polyval(np.polyfit(t, d['mean'], 1), t)
            yr_ = d['nee'] - np.polyval(np.polyfit(t, d['nee'], 1), t)
            r2, p2 = stats.pearsonr(xr_, yr_)
            rows.append([m, len(d), r, p, r2, p2])
        out = pd.DataFrame(rows, columns=['member', 'n_years', 'r', 'p', 'r_detrended', 'p_detrended'])
        out.to_csv(self.outdir2 + 'GCB_NEE_vs_SPEI.csv', index=False)
        print('NEE vs best-scale SPEI (negative r: wetter -> more uptake):')
        print(out.round(2).to_string())

    def fig_all_members(self):
        ## 2026-11-26: every member in its own colour (not grey), from the existing GCB_NEE_members.csv (no re-read of
        ##   the netCDF); top: 1980-2024; bottom: 2010-2024 zoom (most members start in 2015); black = median
        ##   (>= 5 members); dashed = names with 'Satellite'; legend shows each member's years
        ## anomaly: NEE minus the member's own 2015-2024 mean (the period all 16 members cover), Tg C per year; removes
        ##   the offsets between members (e.g. satellite vs in-situ, fossil prior); not % relative change, because NEE is
        ##   near zero and changes sign; trends are unchanged by the anomaly
        ## only GCB members (the standalone CAMS / CarboScope downloads are not on this figure)
        ## output: GCB_NEE_members_anomaly.csv, Fig_GCB_NEE_all_members.png / .pdf
        ann = pd.read_csv(self.outdir2 + 'GCB_NEE_members.csv', index_col=0)
        b0, b1 = 2015, 2024
        ann = ann - ann.loc[b0:b1].mean()
        ann.to_csv(self.outdir2 + 'GCB_NEE_members_anomaly.csv')
        yrs = ann.index.values
        med = ann.median(axis=1, skipna=True).where(ann.notna().sum(axis=1) >= 5)
        cols = plt.get_cmap('tab20').colors
        order = sorted(ann.columns, key=lambda m: ann[m].first_valid_index())   ## longest records first in the legend
        with plt.rc_context(self.font):
            fig, axs = plt.subplots(2, 1, figsize=(24, 18))
            for ax, (x0, x1) in zip(axs, [(1980, 2024), (2010, 2024)]):
                for i, m in enumerate(order):
                    s = ann[m].dropna()
                    ls = '--' if 'satellite' in m.lower() else '-'
                    ax.plot(s.index, s.values, color=cols[i % 20], lw=2.5, ls=ls, marker='o', ms=5,
                            label=f'{m} ({int(s.index.min())}-{int(s.index.max())})')
                ax.plot(yrs, med, color='k', lw=5, label=f'{self.tag} median (>= 5 members)')
                ax.axhline(0, color='grey', lw=1)
                if x0 < 2000:
                    ax.axvline(1999.5, color='grey', lw=1.5, ls=':')
                ax.set_xlim(x0 - 0.5, x1 + 0.5)
                v = ann.loc[x0:x1].values
                pad = 0.08 * (np.nanmax(v) - np.nanmin(v))
                ax.set_ylim(np.nanmin(v) - pad, np.nanmax(v) + pad)   ## y range from the years shown in this panel
                ax.set_ylabel(f'NEE anomaly (Tg C per year)\nvs {b0}-{b1} mean, negative = more uptake')
                ax.spines[['top', 'right']].set_visible(False)
            axs[0].set_title(f'{self.tag} inversions, study-area NEE anomaly, all members')
            axs[1].set_title('2010-2024 zoom')
            axs[1].set_xlabel('Year')
            axs[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=20)
            plt.tight_layout()
            plt.savefig(self.outdir2 + 'Fig_GCB_NEE_all_members.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir2 + 'Fig_GCB_NEE_all_members.pdf', bbox_inches='tight')
            plt.close()
        print('saved', self.outdir2 + 'Fig_GCB_NEE_all_members.png')


class GCB_tif_dic(GCB_inversions):
    ## 2026-11-26: GCB inversion members as monthly tifs (to look at in ArcGIS / QGIS) and per-pixel dic (same format
    ##   as Data_processing tif_to_dic: per_pix_dic_000.npy, key (row, col) -> monthly series)
    ## file  : the same GCP2026_inversions_1x1 netCDF as GCB_inversions (16 members in GCP2026)
    ## var   : land_flux_only_fossil_cement_adjusted (posterior land flux, NEE sign: positive = source, negative = uptake)
    ## unit  : converted to g C m-2 month-1 (from the units attribute via rate_factor, per cell -> per m2 if needed)
    ## grid  : native 1 deg, western box (lat 24-51, lon -128 to -99, from open_box), north-up; tif origin = upper-left
    ##         corner (cell centre minus half a cell); NaN written as -999999
    ##         no shp clip: at 1 deg the clip drops the border cells; mask with the study pixels later
    ## tif   : Data\GCB_inversions\tiff\{member}\YYYYMM01.tif, only months where the member has data
    ## dic   : Data\GCB_inversions\dic\{member}\per_pix_dic_000.npy, every month 1982-01 .. 2024-12 (516 values, NaN
    ##         where the member has no data), so all members line up in time
    def __init__(self):
        super().__init__()
        self.tifdir = data_root + r'GCB_inversions\tiff\\'
        self.dicdir = data_root + r'GCB_inversions\dic\\'
        T.mk_dir(self.tifdir, force=True)
        T.mk_dir(self.dicdir, force=True)
        self.year_list = list(range(1982, 2025))

    def run(self):
        self.nc_to_tif()
        self.tif_to_dic()

    def member_names(self, ds, mdim):
        names = None
        for v in ds.variables:
            if mdim in ds[v].dims and ds[v].ndim <= 2 and v != self.var and ds[v].dtype.kind in 'SUO':
                vals = ds[v].values
                names = [self.clean_name(x) if ds[v].ndim == 1 else ''.join(self.clean_name(c) for c in x)
                         for x in vals]
        n_mem = ds.sizes[mdim]
        return names if names and len(names) == n_mem else [f'member_{k + 1:02d}' for k in range(n_mem)]

    def nc_to_tif(self):
        ds, grid = self.open_box()
        f, per_cell = self.rate_factor(grid['units'])             ## value -> Tg C per year (x study area in m2)
        a = ds[self.var]
        mdim = [d for d in a.dims if d not in (grid['tn'], grid['latn'], grid['lonn'])][0]
        names = self.member_names(ds, mdim)
        print(f'{len(names)} members:', names)
        t = pd.to_datetime(ds[grid['tn']].values)
        lat = grid['lat']
        lon = np.where(grid['lon'] > 180, grid['lon'] - 360, grid['lon'])
        area = grid['area']
        flip = lat[0] < lat[-1]                                    ## tif rows must go north -> south
        if flip:
            lat = lat[::-1]
            area = area[::-1]
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]                              ## negative
        longitude_start = lon[0] - pixelWidth / 2
        latitude_start = lat[0] - pixelHeight / 2
        print(f'tif grid: {len(lat)} x {len(lon)}, origin ({longitude_start}, {latitude_start}), '
              f'cell {pixelWidth} x {pixelHeight}')
        sub = a.isel({grid['latn']: grid['li'], grid['lonn']: grid['lj']})
        for k, nm in enumerate(names):
            outdir = self.tifdir + nm + '\\'
            T.mk_dir(outdir, force=True)
            x = np.array(sub.isel({mdim: k}).transpose(grid['tn'], grid['latn'], grid['lonn']).values, dtype=float)
            if flip:
                x = x[:, ::-1, :]
            if per_cell:
                x = x / area[None]
            n = 0
            for i in tqdm(range(len(t)), desc=nm):
                arr = x[i]
                if not np.isfinite(arr).any():
                    continue
                days = calendar.monthrange(t[i].year, t[i].month)[1]
                arr = arr * f * 1e12 * days / 365.25               ## g C m-2 month-1
                arr[~np.isfinite(arr)] = -999999
                outf = outdir + f'{t[i].year}{t[i].month:02d}01.tif'
                ToRaster().array2raster(outf, longitude_start, latitude_start, pixelWidth, pixelHeight, arr,
                                        ndv=-999999)
                n += 1
            print(nm, n, 'tifs')

    def tif_to_dic(self):
        month_list = [f'{y}{m:02d}01.tif' for y in self.year_list for m in range(1, 13)]
        for nm in T.listdir(self.tifdir):
            fdir = self.tifdir + nm + '\\'
            if not os.path.isdir(fdir):
                continue
            outdir = self.dicdir + nm + '\\'
            T.mk_dir(outdir, force=True)
            have = set(T.listdir(fdir))
            ref = [x for x in have if x.endswith('.tif')][0]
            array0, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + ref)
            all_array = []
            for f in month_list:
                if f in have:
                    array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
                    array = np.array(array, dtype=float)
                    array[array < -999] = np.nan
                else:
                    array = np.full(np.shape(array0), np.nan)
                all_array.append(array)
            all_array = np.array(all_array)                        ## (month, row, col)
            row, col = all_array.shape[1], all_array.shape[2]
            temp_dic = {}
            for r in range(row):
                for c in range(col):
                    temp_dic[(r, c)] = all_array[:, r, c]
            np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)
            n_valid = int(np.isfinite(all_array).any(axis=(1, 2)).sum())
            print(f'{nm}: {row} x {col} pixels, {len(month_list)} months ({n_valid} with data) -> {outdir}')


def main():
    # CAMS_download().run()     ## 1: 1979-2025, one file per year (done 2026-11-25)
    # CAMS_inspect().run(2010)  ## 2: look at one year
    # CAMS_analysis().run()      ## 3: study-area NEE 1979-2024 (done 2026-11-25)
    # CarboScope_analysis().run()  ## 4: CarboScope nbetEXToc_v2025 + comparison (done 2026-11-25)
    # GCB_inversions().run()       ## 5: GCB2025 ensemble of 14 inversions (2026-11-25)
    GCB_tif_dic().run()            ## 6: GCB members -> monthly tif + per-pixel dic (2026-11-26)
    GCB_inversions().fig_all_members()   ## 7: all members in one time series figure (2026-11-26)


if __name__ == '__main__':
    main()
