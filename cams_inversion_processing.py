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
from __Global__ import *
import zipfile


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


def main():
    CAMS_download().run()     ## 1: 1979-2025, one file per year (rerun to fill failed years)
    # CAMS_inspect().run(2010)  ## 2: look at one year


if __name__ == '__main__':
    main()
