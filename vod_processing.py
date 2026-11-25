# coding='utf-8'
## VOD (vegetation optical depth), batch 1: download VODCA v2 and look at it (2026-11-25)
## why  : VOD sees canopy water + biomass with microwaves, independent of the optical LAI; if the canopy really held up
##        through the 2000-2024 drought, VOD should not collapse either (and its water part reacts to drought first)
## data : VODCA v2 (Zotta et al. 2024, ESSD 16, 4573), TU Wien research data repository, DOI 10.48436/t74ty-tcx62
##   CXKu : merged C + X + Ku band, 1987-2021, daily, 0.25 deg (upper canopy; the only multi-decade product)
##   L    : SMOS + SMAP, 2010-2021, 10-daily, 0.25 deg (whole canopy incl. branches and stems)
##   the two use different LPRM versions, so compare their changes, not their absolute values
## steps: 1 list the files of the record (sizes), nothing downloaded
##        2 download the files whose name contains one of the keys in self.keep (default CXKu and L), resumable
##        3 open one file, print its variables, dimensions, units, extent, and map one summer mean over the study area
## output: Data\VODCA_v2\ (downloads), Result\Dataframe\VODCA\inspect\ (printout and quick-look map)
from __Global__ import *
import zipfile


class VODCA_base:
    def __init__(self):
        self.api = 'https://researchdata.tuwien.ac.at/api/records/'
        self.record = 't74ty-tcx62'                  ## from the DOI; the latest version is resolved in list_files
        self.datadir = data_root + rf'VODCA_v2\\'
        self.outdir = result_root + rf'\Dataframe\VODCA\inspect\\'
        T.mk_dir(self.datadir, force=True)
        T.mk_dir(self.outdir, force=True)
        self.keep = ['CXKu', '_L_', 'L-band', 'VODCA_L']   ## file-name keys to download (case-insensitive)
        self.extent = [-124.529, -102.079, 25.579, 48.979]


class VODCA_download(VODCA_base):
    def list_files(self):
        r = requests.get(self.api + self.record + '/versions/latest', timeout=60)
        r.raise_for_status()
        rec = r.json()
        rid = rec['id']
        print('record:', rec['metadata'].get('title'), '| version:', rec['metadata'].get('version'), '| id:', rid,
              '| published:', rec['metadata'].get('publication_date'))
        f = requests.get(self.api + rid + '/files', timeout=60)
        f.raise_for_status()
        files = [(e['key'], e.get('size', 0), e['links']['content']) for e in f.json()['entries']]
        for k, s, _ in files:
            print(f'  {k:60s} {s / 1e9:8.2f} GB')
        print('total:', round(sum(s for _, s, _ in files) / 1e9, 2), 'GB')
        return files

    def run(self, download=False):
        files = self.list_files()
        sel = [x for x in files if any(k.lower() in x[0].lower() for k in self.keep)]
        print('selected by', self.keep, ':', [x[0] for x in sel], '|', round(sum(x[1] for x in sel) / 1e9, 2), 'GB')
        if not download:
            print('list only; run(download=True) to download the selected files')
            return
        for key, size, url in sel:
            self.get(key, size, url)

    def get(self, key, size, url):
        out = self.datadir + key
        done = os.path.getsize(out) if os.path.isfile(out) else 0
        if size and done == size:
            print('have', key)
            return
        head = {'Range': f'bytes={done}-'} if done else {}
        with requests.get(url, stream=True, headers=head, timeout=120) as r:
            r.raise_for_status()
            mode = 'ab' if done and r.status_code == 206 else 'wb'
            with open(out, mode) as fw, tqdm(total=size, initial=done if mode == 'ab' else 0, unit='B',
                                             unit_scale=True, desc=key) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fw.write(chunk)
                    bar.update(len(chunk))
        if size and os.path.getsize(out) != size:
            print('incomplete, run again to resume:', key)


class VODCA_inspect(VODCA_base):
    def run(self):
        f = self.first_nc()
        print('file:', f)
        ds = xarray.open_dataset(f)
        lines = [str(ds)]
        for v in ds.data_vars:
            a = ds[v]
            lines.append(f'{v}: dims {a.dims}, shape {a.shape}, attrs {dict(a.attrs)}')
        txt = '\n'.join(lines)
        print(txt)
        open(self.outdir + 'inspect_' + os.path.basename(f) + '.txt', 'w').write(txt)
        self.quick_map(ds, f)

    def first_nc(self):
        ## a netCDF somewhere under the download folder; a zip is opened and its first summer file extracted
        ncs = []
        for root, _, fs in os.walk(self.datadir):
            ncs += [os.path.join(root, x) for x in fs if x.endswith('.nc')]
        if ncs:
            return sorted(ncs)[len(ncs) // 2]
        zips = [self.datadir + x for x in os.listdir(self.datadir) if x.endswith('.zip')]
        if not zips:
            raise FileNotFoundError('no .nc or .zip in ' + self.datadir)
        z = zipfile.ZipFile(zips[0])
        names = [n for n in z.namelist() if n.endswith('.nc')]
        print(zips[0], ':', len(names), 'netCDF files, e.g.', names[:3], '...', names[-3:])
        pick = [n for n in names if '0715' in n or '-07-15' in n] or names[len(names) // 2:len(names) // 2 + 1]
        z.extract(pick[0], self.datadir + 'extract\\')
        return self.datadir + 'extract\\' + pick[0]

    def quick_map(self, ds, f):
        var = [v for v in ds.data_vars if 'vod' in v.lower()] or list(ds.data_vars)
        a = ds[var[0]]
        latn = [d for d in a.dims if 'lat' in d.lower()][0]
        lonn = [d for d in a.dims if 'lon' in d.lower()][0]
        a = a.sortby(latn)
        lon0, lon1, lat0, lat1 = self.extent
        a = a.sel({latn: slice(lat0, lat1), lonn: slice(lon0, lon1)})
        if a.ndim > 2:
            a = a.mean(dim=[d for d in a.dims if d not in (latn, lonn)], skipna=True)
        with plt.rc_context({'font.size': 22}):
            fig, ax = plt.subplots(figsize=(12, 11))
            im = ax.pcolormesh(a[lonn], a[latn], a.values, cmap='YlGn', shading='auto')
            cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.07)
            cb.set_label(f'{var[0]} ({a.attrs.get("units", "")})')
            ax.set_title(os.path.basename(f), fontsize=20)
            ax.set_aspect(1 / np.cos(np.deg2rad(37)))
            plt.savefig(self.outdir + 'quicklook_' + os.path.basename(f) + '.png', dpi=150, bbox_inches='tight')
            plt.close()
        print('valid pixels in the study area:', int(np.isfinite(a.values).sum()), 'of', a.size)


def main():
    # VODCA_download().run(download=False)   ## 1: list the files and sizes only
    VODCA_download().run(download=True)  ## 2: download CXKu and L (resumable)
    # VODCA_inspect().run()                ## 3: look at one file


if __name__ == '__main__':
    main()
