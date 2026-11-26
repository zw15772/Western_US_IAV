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
## trend : OLS per cell on the valid years (>= 80 % of the years of the period), % of the cell's mean per decade
## check : valid days per month in the growing season, per year, north and south of 38 N (TMI covers only < ~38 N
##         from 1998), with the sensor changes marked, to see whether sampling changes near 2000
## output: {R2}\04_water_carbon\VODCA_CXKu\
##   lookup_0.25deg.csv, VOD_GS_by_year.csv, VOD_trend.csv, VOD_group_summary.csv,
##   Fig_VOD_ts.png, Fig_VOD_trend_map.png, Fig_VOD_period_change.png, Fig_VOD_sampling.png
from __Global__ import *
import warnings
from ET_analysis import *


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




class VOD_monthly:
    def __init__(self):
        self.daily_dir = data_root + rf'VODCA_v2\VODCA_CXKu\nc\\'
        self.outdir = data_root + rf'VODCA_v2\VODCA_CXKu\CXKu_monthly_tif\\'   ## where the monthly tifs are now
        T.mk_dir(self.outdir + 'VOD', force=True)
        T.mk_dir(self.outdir + 'n_days', force=True)
        self.extent = [-125.0, -102.0, 25.5, 49.0]   ## lon0, lon1, lat0, lat1: study area on the 0.25 deg grid
        self.min_days = 5
        self.year_list = list(range(1987, 2022))

    def run(self):
        # first = True
        # for y in self.year_list:
        #     ydir = self.daily_dir + f'{y}\\'
        #     if not os.path.isdir(ydir):
        #         print('no folder', y)
        #         continue
        #     fs = sorted(f for f in os.listdir(ydir) if f.endswith('.nc'))
        #     for m in tqdm(range(1, 13), desc=str(y)):
        #         outf = self.outdir + rf'VOD\{y}{m:02d}01.tif'
        #         if os.path.isfile(outf):
        #             continue
        #         mf = [ydir + f for f in fs if f[13:20] == f'{y}-{m:02d}']   ## daily_images_YYYY-MM-DD.nc
        #         if not mf:
        #             print('no days', y, m)
        #             continue
        #         stack, lon, lat = [], None, None
        #         for f in mf:
        #             a, lon, lat = self.read_day(f, show=first)
        #             first = False
        #             stack.append(a)
        #         stack = np.array(stack)
        #         n = np.isfinite(stack).sum(axis=0)
        #         with warnings.catch_warnings():
        #             warnings.simplefilter('ignore', RuntimeWarning)
        #             mean = np.nanmean(stack, axis=0)
        #         mean[n < self.min_days] = np.nan
        #         self.write(outf, mean, lon, lat)
        #         self.write(self.outdir + rf'n_days\{y}{m:02d}01.tif', n.astype(float), lon, lat)
        self.tif_to_dic()

    def read_day(self, f, show=False):
        ds = xarray.open_dataset(f)
        if show:
            print(ds)
        var = 'VOD' if 'VOD' in ds else [v for v in ds.data_vars if 'vod' in v.lower()][0]
        a = ds[var]
        latn = [d for d in a.dims if 'lat' in d.lower()][0]
        lonn = [d for d in a.dims if 'lon' in d.lower()][0]
        a = a.squeeze(drop=True).sortby(latn, ascending=False)   ## north-up
        lon0, lon1, lat0, lat1 = self.extent
        a = a.sel({latn: slice(lat1, lat0), lonn: slice(lon0, lon1)})
        arr = np.array(a.values, dtype=float)
        arr[(arr < 0) | (arr > 5)] = np.nan          ## outside the physical VOD range / fill values
        out = arr, a[lonn].values, a[latn].values
        ds.close()
        return out

    def write(self, outf, arr, lon, lat):
        ## upper-left corner from the cell centres
        res_x = float(lon[1] - lon[0])
        res_y = float(lat[1] - lat[0])                ## negative (north-up)
        arr = np.where(np.isfinite(arr), arr, -999999)
        ToRaster().array2raster(outf, float(lon[0]) - res_x / 2, float(lat[0]) - res_y / 2, res_x, res_y, arr,
                                ndv=-999999)

    def tif_to_dic(self):

        fdir_all = join(data_root, 'VODCA_v2\VODCA_CXKu\\',   'CXKu_monthly_tif','VOD')
        outdir = join(data_root, 'VODCA_v2\VODCA_CXKu\\',   'dic')
        T.mk_dir(outdir, force=True)

        year_list = list(range(1982, 2025)) ##
        # 一般是2003 开始只有ppt是2002 因为要考虑冬天
        # 作为筛选条件

        all_array = []  #### so important  it should be go with T.mk_dic


        for f in T.listdir(fdir_all):
            print(f)

            if not f.endswith('.tif'):
                continue
            if int(f.split('.')[0][0:4]) not in year_list:
                continue


            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_all, f))
            array = np.array(array, dtype=float)


            # array_unify = array[:720][:720,
            #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
            # array_unify = array[:3600][:3600,
            #               :7200]


            array[array < -999] = np.nan
            # array_unify[array_unify > 10] = np.nan
            # array[array ==0] = np.nan

            # array[array < 0] = np.nan


            # plt.imshow(array)
            # plt.show()




            array_dryland = array
            # plt.imshow(array_dryland)
            # plt.show()

            all_array.append(array_dryland)

        row = len(all_array[0])
        col = len(all_array[0][0])
        key_list = []
        dic = {}

        for r in tqdm(range(row), desc='构造key'):  # 构造字典的键值，并且字典的键：值初始化
            for c in range(col):
                dic[(r, c)] = []
                key_list.append((r, c))
        # print(dic_key_list)

        for r in tqdm(range(row), desc='构造time series'):  # 构造time series
            for c in range(col):
                for arr in all_array:
                    value = arr[r][c]
                    dic[(r, c)].append(value)
                # print(dic)
        time_series = []
        flag = 0
        temp_dic = {}
        for key in tqdm(key_list, desc='output...'):  # 存数据
            flag = flag + 1
            time_series = dic[key]
            time_series = np.array(time_series)
            temp_dic[key] = time_series
            if flag % 10000 == 0:
                # print(flag)
                np.save(outdir + rf'/per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + rf'/per_pix_dic_%03d' % 0, temp_dic)



class VOD_analysis(GLEAM_Et):
    ## same steps as GLEAM_Et (ET_analysis.py), 0.25 deg, 1988-2021; reuses ols-free parts: masks, grid_map
    def __init__(self):
        super().__init__()
        self.var, self.unit = 'VOD', '-'
        self.monthly_dir = data_root + rf'VODCA_v2\VODCA_CXKu\CXKu_monthly_tif\\'
        self.dic_dir = data_root + rf'VODCA_v2\VODCA_CXKu\dic\\'
        self.outdir = self.result2w + rf'04_water_carbon\VODCA_CXKu\\'
        T.mk_dir(self.outdir, force=True)
        self.year_list = np.arange(1988, 2022)
        self.p1, self.p2 = (1988, 1999), (2000, 2021)
        self.period_dic = {'1988-2021': (1988, 2021), '1988-1999': self.p1, '2000-2021': self.p2}
        self.min_pix = 13          ## of the 25 possible 0.05 deg LAI pixels in a 0.25 deg cell
        self.min_year_share = 0.8  ## valid years needed for a trend

    def run(self):
        cube, geo = self.read_cube()
        lon0, res, lat0 = geo[0], geo[1], geo[3]
        nrow, ncol = cube.shape[2], cube.shape[3]
        look = Coarse_lookup().build(lon0, lat0, res, nrow, ncol, min_pix=self.min_pix)
        look.to_csv(self.outdir + f'lookup_{res:.2f}deg.csv', index=False)
        gs = self.season_values(cube, look)
        tr = self.trends(look, gs)
        summ = self.group_summary(look, gs, tr)
        self.fig_ts(look, gs)
        self.fig_trend_map(look, tr, geo, nrow, ncol)
        self.fig_period_change(look, tr, summ, geo, nrow, ncol)
        self.fig_sampling(look, geo)

    def month_list(self):
        ## dates of the dic values = the sorted monthly tif names (Wen's tif_to_dic)
        return sorted(f.split('.')[0] for f in os.listdir(self.monthly_dir + 'VOD') if f.endswith('.tif'))

    def read_cube(self):
        months = self.month_list()
        tif0 = self.monthly_dir + rf'VOD\{months[0]}.tif'
        array, x0, y0, pw, ph = ToRaster().raster2array(tif0)
        nrow, ncol = np.array(array).shape
        geo = (x0, abs(pw), 0, y0, 0, -abs(ph))
        idx = {m: i for i, m in enumerate(months)}
        need = [f'{y}{m:02d}01' for y in self.year_list for m in range(1, 13)]
        miss = [m for m in need if m not in idx]
        if miss:
            raise FileNotFoundError(f'monthly VOD tifs missing: {miss[:10]}')
        sel = np.array([idx[m] for m in need])
        dic = T.load_npy_dir(self.dic_dir)
        cube = np.full((len(self.year_list), 12, nrow, ncol), np.nan)
        for (r, c), v in dic.items():
            v = np.array(v, dtype=float)
            if len(v) != len(months):
                continue
            v = v[sel]
            v[(v < 0) | (v > 5) | ~np.isfinite(v)] = np.nan
            cube[:, :, r, c] = v.reshape(len(self.year_list), 12)
        print(f'VOD cube (year, month, row, col): {cube.shape} | dic values per cell: {len(months)} '
              f'({months[0]} .. {months[-1]}) | grid {geo}')
        return cube, geo

    def season_values(self, cube, look):
        r, c = look['r'].values, look['c'].values
        x = cube[:, :, r, c]                     ## (year, month, cell)
        mon = np.arange(1, 13)[:, None]
        win = (mon >= look['gs_start'].values[None, :]) & (mon <= look['gs_end'].values[None, :])
        xw = np.where(win[None], x, np.nan)
        n_ok = np.isfinite(xw).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            gs = np.nanmean(xw, axis=1)
        gs[n_ok < np.ceil(win.sum(axis=0)[None] / 2)] = np.nan
        out = pd.DataFrame(gs.T, columns=self.year_list)
        out.insert(0, 'c', c)
        out.insert(0, 'r', r)
        out.to_csv(self.outdir + 'VOD_GS_by_year.csv', index=False)
        return gs.T                              ## (cell, year)

    def ols_nan(self, y, yrs):
        ## per cell OLS on its valid years (>= min_year_share of the period)
        from scipy import stats
        slope = np.full(len(y), np.nan)
        p = np.full(len(y), np.nan)
        for i in range(len(y)):
            ok = np.isfinite(y[i])
            if ok.sum() < self.min_year_share * len(yrs):
                continue
            res = stats.linregress(yrs[ok], y[i][ok])
            slope[i], p[i] = res.slope, res.pvalue
        return slope, p

    def period_mean(self, gs, period):
        sel = (self.year_list >= period[0]) & (self.year_list <= period[1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(gs[:, sel], axis=1)

    def trends(self, look, gs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_all = np.nanmean(gs, axis=1)
        tr = look[['r', 'c', 'lat', 'lon', 'group']].copy()
        tr['VOD_GS_mean'] = mean_all
        for p, (y0, y1) in self.period_dic.items():
            sel = (self.year_list >= y0) & (self.year_list <= y1)
            s, pv = self.ols_nan(gs[:, sel], self.year_list[sel].astype(float))
            tr[f'trend_{p}'] = s * 10 / mean_all * 100
            tr[f'p_{p}'] = pv
        m1, m2 = self.period_mean(gs, self.p1), self.period_mean(gs, self.p2)
        tr['change_pct'] = (m2 - m1) / m1 * 100
        tr.to_csv(self.outdir + 'VOD_trend.csv', index=False)
        return tr

    def group_summary(self, look, gs, tr):
        w = np.cos(np.deg2rad(look['lat'].values))
        p1n, p2n = f'{self.p1[0]}-{self.p1[1]}', f'{self.p2[0]}-{self.p2[1]}'
        rows = []
        for k, m in self.masks(look).items():
            if m.sum() == 0:
                continue
            row = [k, int(m.sum())]
            for per in [self.p1, self.p2]:
                v = self.period_mean(gs[m], per)
                ok = np.isfinite(v)
                row.append(np.sum(v[ok] * w[m][ok]) / w[m][ok].sum())
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
        cols = ['group', 'n_cells', f'VOD_GS_{p1n}', f'VOD_GS_{p2n}']
        for p in self.period_dic:
            cols += [f'trend_{p}', f'sig_up_%_{p}', f'sig_down_%_{p}']
        cols.append('change_pct')
        tab = pd.DataFrame(rows, columns=cols)
        tab.to_csv(self.outdir + 'VOD_group_summary.csv', index=False)
        print(tab.round(3).to_string())
        return tab

    def fig_ts(self, look, gs):
        w = np.cos(np.deg2rad(look['lat'].values))
        groups = [g for g in self.group_list if self.masks(look)[g].sum() > 0]
        with plt.rc_context(self.font):
            fig, axs = plt.subplots(len(groups), 1, figsize=(20, 4.2 * len(groups)), sharex=True)
            for ax, g in zip(np.atleast_1d(axs), groups):
                m = self.masks(look)[g]
                x = gs[m]
                ts = np.array([np.sum(x[np.isfinite(x[:, j]), j] * w[m][np.isfinite(x[:, j])]) /
                               w[m][np.isfinite(x[:, j])].sum() for j in range(x.shape[1])])
                ax.plot(self.year_list, ts, color=self.color_dic[g], lw=3.5)
                for (y0, y1) in [self.p1, self.p2]:
                    sel = (self.year_list >= y0) & (self.year_list <= y1)
                    ax.hlines(ts[sel].mean(), y0, y1, color=self.color_dic[g], lw=2.5, ls='--')
                ax.axvline(self.p1[1] + 0.5, color='grey', ls=':', lw=2)
                ax.set_ylabel('VOD')
                ax.set_title(f'{g} (n = {int(m.sum())})', loc='left', fontweight='bold')
                ax.spines[['top', 'right']].set_visible(False)
            np.atleast_1d(axs)[-1].set_xlabel(f'Year (dashed: period means {self.p1[0]}-{self.p1[1]}, '
                                              f'{self.p2[0]}-{self.p2[1]})')
            fig.suptitle('VODCA v2 CXKu, growing-season mean (SNU window), 0.25 deg', y=1.0)
            plt.tight_layout()
            plt.savefig(self.outdir + 'Fig_VOD_ts.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_VOD_ts.pdf', bbox_inches='tight')
            plt.close()

    def fig_trend_map(self, look, tr, geo, nrow, ncol):
        import cartopy.crs as ccrs
        bounds = [-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6]
        with plt.rc_context(self.font):
            fig = plt.figure(figsize=(33, 11))
            for k, p in enumerate(self.period_dic):
                ax = fig.add_axes([k * 0.333, 0.12, 0.32, 0.8], projection=ccrs.PlateCarree())
                im = self.grid_map(ax, look, tr[f'trend_{p}'].values, geo, nrow, ncol, 'BrBG', bounds)
                s = (tr[f'p_{p}'].values < 0.05) & np.isfinite(tr[f'trend_{p}'].values)
                ax.scatter(look['lon'].values[s], look['lat'].values[s], s=10, color='k',
                           transform=ccrs.PlateCarree(), linewidths=0)
                ax.set_title(p, fontweight='bold')
            cax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds)
            cb.set_label('VOD trend (% of the 1988-2021 mean per decade), dots p < 0.05')
            plt.savefig(self.outdir + 'Fig_VOD_trend_map.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_VOD_trend_map.pdf', bbox_inches='tight')
            plt.close()

    def fig_period_change(self, look, tr, summ, geo, nrow, ncol):
        import cartopy.crs as ccrs
        bounds = [-15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15]
        lab = f'{self.p2[0]}-{self.p2[1]} vs {self.p1[0]}-{self.p1[1]}'
        with plt.rc_context(self.font):
            fig = plt.figure(figsize=(26, 11))
            ax = fig.add_axes([0.0, 0.12, 0.5, 0.8], projection=ccrs.PlateCarree())
            im = self.grid_map(ax, look, tr['change_pct'].values, geo, nrow, ncol, 'BrBG', bounds)
            ax.set_title(f'VOD, {lab}', fontweight='bold')
            cax = fig.add_axes([0.05, 0.06, 0.4, 0.03])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds)
            cb.set_label('change (%)')
            bx = fig.add_axes([0.62, 0.16, 0.36, 0.72])
            t = summ.set_index('group').reindex([g for g in self.group_list if g in summ['group'].values])
            y = np.arange(len(t))[::-1]
            v_all = t['change_pct'].values
            bx.barh(y, v_all, color=[self.color_dic[g] for g in t.index], height=0.7)
            for yi, v in zip(y, v_all):
                bx.text(v, yi, f' {v:+.1f}% ', va='center', ha='left' if v >= 0 else 'right', fontsize=22)
            span = max(np.nanmax(v_all), 0) - min(np.nanmin(v_all), 0)
            bx.set_xlim(min(np.nanmin(v_all), 0) - 0.3 * span, max(np.nanmax(v_all), 0) + 0.3 * span)
            bx.axvline(0, color='k', lw=1)
            bx.set_yticks(y)
            bx.set_yticklabels(t.index)
            bx.set_xlabel(f'VOD change, {lab} (%)')
            bx.spines[['top', 'right']].set_visible(False)
            plt.savefig(self.outdir + 'Fig_VOD_period_change.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + 'Fig_VOD_period_change.pdf', bbox_inches='tight')
            plt.close()

    def fig_sampling(self, look, geo):
        ## valid days per growing-season month, per year, cells north / south of 38 N
        r, c = look['r'].values, look['c'].values
        south = look['lat'].values < 38
        rows = []
        for y in self.year_list:
            vals = []
            for m in range(1, 13):
                f = self.monthly_dir + rf'n_days\{y}{m:02d}01.tif'
                a = np.array(ToRaster().raster2array(f)[0], dtype=float)
                a[a < 0] = np.nan
                vals.append(a[r, c])
            vals = np.array(vals)                    ## (month, cell)
            mon = np.arange(1, 13)[:, None]
            win = (mon >= look['gs_start'].values[None, :]) & (mon <= look['gs_end'].values[None, :])
            v = np.where(win, vals, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                cell = np.nanmean(v, axis=0)
            rows.append([y, np.nanmean(cell), np.nanmean(cell[south]), np.nanmean(cell[~south])])
        tab = pd.DataFrame(rows, columns=['year', 'all', 'south_38N', 'north_38N'])
        tab.to_csv(self.outdir + 'VOD_sampling_days.csv', index=False)
        events = [(1997.95, 'TMI (< ~38 N)'), (2002.45, 'AMSR-E'), (2011.8, 'AMSR-E ends'), (2012.55, 'AMSR2')]
        with plt.rc_context(self.font):
            fig, ax = plt.subplots(figsize=(20, 8))
            for col, cl, lab in [('all', '#222222', 'all cells'), ('south_38N', '#D9480F', 'south of 38 N'),
                                 ('north_38N', '#2171B5', 'north of 38 N')]:
                ax.plot(tab['year'], tab[col], color=cl, lw=3.5, label=lab)
            for x, lab in events:
                ax.axvline(x, color='grey', ls=':', lw=2)
                ax.text(x, ax.get_ylim()[1], f' {lab}', rotation=90, va='top', ha='left', fontsize=20, color='grey')
            ax.set_ylabel('valid days per month\n(growing season mean)')
            ax.set_xlabel('Year')
            ax.legend(frameon=False, loc='lower right')
            ax.spines[['top', 'right']].set_visible(False)
            plt.savefig(self.outdir + 'Fig_VOD_sampling.png', dpi=300, bbox_inches='tight')
            plt.close()
        print(tab.round(1).to_string())


def main():
    # VODCA_download().run(download=False)   ## 1: list the files and sizes only
    # VODCA_download().run(download=True)  ## 2: download CXKu and L (resumable)
    # VODCA_inspect().run()                ## 3: look at one file
    # VOD_monthly().run()                  ## 4: make monthly tifs and dic
    VOD_analysis().run()                 ## 5: trends and figures


if __name__ == '__main__':
    main()
