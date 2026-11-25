# coding='utf-8'
## Streamflow, batch 1: reference gauges, daily discharge, water-year metrics (2026-11-24)
## why  : SPEI says 2000-2024 was a long, severe drought; streamflow checks whether less water actually left the
##        basins, and (batch 2) whether the same precipitation now gives less runoff (runoff ratio Q / P)
## gauges: GAGES-II (Falcone 2011) reference class (CLASS == 'Ref', minimally disturbed by dams, diversions,
##        urbanisation), in the study extent; put the two GAGES-II files in Data\streamflow\gagesII\ (download from
##        USGS ScienceBase, "GAGES-II: Geospatial Attributes of Gages for Evaluating Streamflow"):
##          basinchar_and_report_sept_2011\gagesII_sept30_2011_conterm.xlsx   (sheets BasinID, Bas_Classif, ...)
##          boundaries-shapefiles-by-aggeco\bas_ref_all.shp                    (reference basin polygons, batch 2)
## daily Q: USGS, parameter 00060 (discharge, cubic feet per second), package dataretrieval (pip install
##        dataretrieval); one csv per gauge, kept, so a rerun only downloads what is missing
## water year: October of the previous year to September (WY 2024 = Oct 2023 - Sep 2024)
## metrics per gauge and water year (only years with >= 330 valid days):
##   Q_mm      : runoff depth, mm per year = mean flow / drainage area
##   Q7min_mm  : lowest 7-day mean flow (low flow), as mm per day
##   CT_doy    : centre of timing, day of the water year by which half of the annual flow has passed (snowmelt timing)
## gauges kept: >= 90 % of the water years 1961-2024 valid (same start as the drought analysis)
## output: Result\Dataframe\streamflow\
##   gauges_ref_west.csv   : selected reference gauges (id, name, lat, lon, area, state, ecoregion)
##   Q_annual.df           : gauge x water year metrics (long table)
##   gauges_kept.csv       : gauges that pass the completeness rule, with their share of valid years
from __Global__ import *


class Streamflow_base:
    def __init__(self):
        self.gages_dir = data_root + rf'streamflow\gagesII\\'
        self.daily_dir = data_root + rf'streamflow\USGS_daily\\'
        self.outdir = result_root + rf'\Dataframe\streamflow\\'
        T.mk_dir(self.daily_dir, force=True)
        T.mk_dir(self.outdir, force=True)
        self.extent = [-124.529, -102.079, 25.579, 48.979]   ## lon min, lon max, lat min, lat max (study grid)
        self.states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
        self.wy_first, self.wy_last = 1961, 2024
        self.min_days = 330           ## valid water year
        self.min_year_share = 0.9     ## valid water years needed in 1961-2024


class Gauge_select(Streamflow_base):
    ## GAGES-II reference gauges in the western states and inside the study extent
    def run(self):
        xlsx = self.gages_dir + r'basinchar_and_report_sept_2011\gagesII_sept30_2011_conterm.xlsx'
        sheets = pd.read_excel(xlsx, sheet_name=None, dtype={'STAID': str})
        print('GAGES-II sheets:', list(sheets.keys()))
        need = {'BasinID': ['STAID', 'STANAME', 'DRAIN_SQKM', 'LAT_GAGE', 'LNG_GAGE', 'STATE'],
                'Bas_Classif': ['STAID', 'CLASS', 'AGGECOREGION']}
        parts = []
        for sh, cols in need.items():
            if sh not in sheets:
                raise ValueError(f'sheet {sh} not found; sheets are {list(sheets.keys())}')
            d = sheets[sh]
            miss = [c for c in cols if c not in d.columns]
            if miss:
                raise ValueError(f'{sh}: columns {miss} not found; columns are {list(d.columns)}')
            parts.append(d[cols].set_index('STAID'))
        df = parts[0].join(parts[1], how='inner')
        df.index = df.index.str.zfill(8)
        n_all = len(df)
        lon0, lon1, lat0, lat1 = self.extent
        df = df[(df['CLASS'].str.strip() == 'Ref') & df['STATE'].isin(self.states) &
                df['LNG_GAGE'].between(lon0, lon1) & df['LAT_GAGE'].between(lat0, lat1)]
        df.index.name = 'site'
        df = df.rename(columns={'STANAME': 'name', 'DRAIN_SQKM': 'area_km2', 'LAT_GAGE': 'lat', 'LNG_GAGE': 'lon',
                                'STATE': 'state', 'CLASS': 'class', 'AGGECOREGION': 'ecoregion'})
        df.to_csv(self.outdir + 'gauges_ref_west.csv')
        print(f'GAGES-II gauges: {n_all} | reference, western states, in extent: {len(df)}')
        print(df.groupby('state').size().to_string())


class Download_Q(Streamflow_base):
    ## daily discharge 1960-10-01 .. 2024-09-30 (WY 1961-2024) per gauge, one csv each (date, Q_cfs, code)
    def run(self):
        sites = pd.read_csv(self.outdir + 'gauges_ref_west.csv', dtype={'site': str})['site'].tolist()
        todo = [s for s in sites if not os.path.isfile(self.daily_dir + f'{s}.csv')]
        print(f'gauges: {len(sites)} | already downloaded: {len(sites) - len(todo)} | to download: {len(todo)}')
        failed = []
        for s in tqdm(todo):
            d = None
            for attempt in range(3):
                try:
                    d = self.get_one(s)
                    break
                except Exception as e:
                    err = repr(e)
                    time.sleep(3 * (attempt + 1))
            if d is None:
                failed.append((s, err))
                continue
            d.to_csv(self.daily_dir + f'{s}.csv', index=False)   ## an empty table is saved too (no data), not retried
        print('failed:', len(failed))
        for s, e in failed[:20]:
            print(' ', s, e[:200])
        if failed:
            pd.DataFrame(failed, columns=['site', 'error']).to_csv(self.outdir + 'download_failed.csv', index=False)

    def get_one(self, s):
        start, end = f'{self.wy_first - 1}-10-01', f'{self.wy_last}-09-30'
        try:   ## legacy NWIS daily values service
            from dataretrieval import nwis
            d, _ = nwis.get_dv(sites=s, parameterCd='00060', start=start, end=end)
            if len(d) == 0:
                return pd.DataFrame(columns=['date', 'Q_cfs', 'code'])
            vcol = [c for c in d.columns if c.startswith('00060') and not c.endswith('_cd')][0]
            ccol = [c for c in d.columns if c.startswith('00060') and c.endswith('_cd')]
            out = pd.DataFrame({'date': pd.to_datetime(d.index).tz_localize(None).normalize(),
                                'Q_cfs': pd.to_numeric(d[vcol], errors='coerce').values,
                                'code': d[ccol[0]].astype(str).values if ccol else ''})
            return out
        except ImportError:
            raise
        except Exception as e_nwis:
            ## USGS is moving the daily values to the new Water Data API; dataretrieval.waterdata reads it
            from dataretrieval import waterdata
            d = waterdata.get_daily(monitoring_location_id=f'USGS-{s}', parameter_code='00060',
                                    statistic_id='00003', time=f'{start}/{end}')
            if isinstance(d, tuple):
                d = d[0]
            if len(d) == 0:
                return pd.DataFrame(columns=['date', 'Q_cfs', 'code'])
            tcol = 'time' if 'time' in d.columns else [c for c in d.columns if 'date' in c.lower() or 'time' in c.lower()][0]
            out = pd.DataFrame({'date': pd.to_datetime(d[tcol]).dt.tz_localize(None).dt.normalize(),
                                'Q_cfs': pd.to_numeric(d['value'], errors='coerce').values,
                                'code': d['approval_status'].astype(str).values if 'approval_status' in d.columns
                                else ''})
            return out


class Annual_Q(Streamflow_base):
    ## water-year metrics per gauge
    def run(self):
        g = pd.read_csv(self.outdir + 'gauges_ref_west.csv', dtype={'site': str}).set_index('site')
        rows = []
        for s in tqdm(g.index):
            f = self.daily_dir + f'{s}.csv'
            if not os.path.isfile(f):
                continue
            d = pd.read_csv(f, parse_dates=['date'])
            if len(d) == 0:
                continue
            rows.append(self.metrics(d, s, g.loc[s, 'area_km2']))
        df = pd.concat(rows, ignore_index=True)
        T.save_df(df, self.outdir + 'Q_annual.df')
        T.df_to_excel(df.head(1000), self.outdir + 'Q_annual_head.xlsx')
        ## completeness 1961-2024
        n_wy = self.wy_last - self.wy_first + 1
        share = df[df['valid']].groupby('site').size() / n_wy
        kept = g.join(share.rename('valid_year_share'), how='inner')
        kept = kept[kept['valid_year_share'] >= self.min_year_share]
        kept.to_csv(self.outdir + 'gauges_kept.csv')
        print(f'gauges with data: {df["site"].nunique()} | >= {self.min_year_share:.0%} valid water years '
              f'{self.wy_first}-{self.wy_last}: {len(kept)}')
        print(kept.groupby('state').size().to_string())

    def metrics(self, d, s, area_km2):
        d = d[np.isfinite(d['Q_cfs']) & (d['Q_cfs'] >= 0)].copy()   ## USGS fills missing / ice days with -999999
        d['wy'] = d['date'].dt.year + (d['date'].dt.month >= 10).astype(int)
        d = d[(d['wy'] >= self.wy_first) & (d['wy'] <= self.wy_last)]
        cms_to_mm_day = 86400 / (area_km2 * 1e6) * 1000     ## m3/s -> mm per day over the basin
        d['q_mm'] = d['Q_cfs'] * 0.0283168 * cms_to_mm_day
        out = []
        for wy, x in d.groupby('wy'):
            n = len(x)
            x = x.sort_values('date')
            q = x['q_mm'].values
            day = (x['date'] - pd.Timestamp(wy - 1, 10, 1)).dt.days.values + 1
            cum = np.cumsum(q)
            ct = day[np.searchsorted(cum, cum[-1] / 2)] if cum[-1] > 0 else np.nan
            q7 = pd.Series(q, index=x['date']).rolling('7D', min_periods=7).mean().min()
            out.append([s, wy, n, n >= self.min_days, q.mean() * 365.25, q7, ct])
        return pd.DataFrame(out, columns=['site', 'wy', 'n_days', 'valid', 'Q_mm', 'Q7min_mm', 'CT_doy'])


def main():
    Gauge_select().run()      ## 1: reference gauges in the west
    Download_Q().run()        ## 2: daily discharge (rerun to fill what failed)
    Annual_Q().run()          ## 3: water-year metrics, completeness


if __name__ == '__main__':
    main()
