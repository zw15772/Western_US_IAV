from __Global__ import *
from scipy import stats
from matplotlib.ticker import MaxNLocator


class Time_series_base():
    ## shared plotting; column name in df = f'{season}_{var}', e.g. spring_SPEI03, summer_SNU_LAI
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        self.dff = result_root + rf'\Dataframe\SPEI\SPEI.df'
        self.result2 = this_root + rf'\Result2\\'
        self.season_list = ['spring', 'summer']
        self.color_dic = {'spring': '#EA6E88', 'summer': '#48526F'}
        self.split_year = 2000
        self.end_year = 2024
        self.crop_threshold = 50   ## % cropland cover above which a pixel is excluded
        ## trend periods: (start, end, linestyle)
        self.trend_period_list = [(1982, 2024, ':'), (2000, 2024, '-')]
        self.ylabel_dic = {}   ## y-axis label per variable; default = variable name
        pass

    def load_df(self):
        df = T.load_df(self.dff)
        df = self.df_clean(df)
        return df

    def df_clean(self, df):
        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 45]
        ## exclude cropland: mean MCD12C1 cropland cover (IGBP 12 + 14) > 50%
        df = df[~(df['cropland_pct'] > self.crop_threshold)]
        return df

    def plot_time_series(self, df, var_list, start_year, outdir, fname):
        T.mk_dir(outdir, force=True)
        years = np.arange(start_year, self.end_year + 1)
        trend_list = []

        fig, axes = plt.subplots(len(var_list), 1, sharex=True, squeeze=False,
                                 figsize=(self.map_width * 1.5, self.map_height * 0.6 * max(len(var_list), 1.5)))
        axes = axes[:, 0]

        for ax, var in zip(axes, var_list):
            for season in self.season_list:
                col = f'{season}_{var}'
                if not col in df.columns:
                    continue
                mean_list = []
                std_list = []
                for year in years:
                    df_i = df[df['year'] == year]
                    vals = np.array(df_i[col].tolist(), dtype=float)
                    weight = np.array(df_i['area_weight'].tolist(), dtype=float)
                    weighted_mean = np.nansum(vals * weight) / np.nansum(weight * np.isfinite(vals))
                    weighted_var = np.nansum(weight * (vals - weighted_mean) ** 2) / np.nansum(weight * np.isfinite(vals))
                    mean_list.append(weighted_mean)
                    std_list.append(np.sqrt(weighted_var))
                mean_list = np.array(mean_list)
                std_list = np.array(std_list)
                color = self.color_dic[season]

                ax.plot(years, mean_list, color=color, lw=1.5, label=season.capitalize())
                ax.fill_between(years, mean_list - std_list, mean_list + std_list, color=color, alpha=0.15)

                ## linear trends: 1982-2024 (dotted) and 2000-2024 (solid)
                text = f'{season.capitalize()}'
                for t_start, t_end, ls in self.trend_period_list:
                    mask = (years >= t_start) & (years <= t_end) & np.isfinite(mean_list)
                    slope, intercept, r, p, se = stats.linregress(years[mask], mean_list[mask])
                    ax.plot(years[mask], slope * years[mask] + intercept, color=color, lw=2, linestyle=ls)
                    text += f'  {t_start}-{t_end}: {slope * 10:.2f}/10yr, p={p:.2f}'
                    trend_list.append([var, season, f'{t_start}-{t_end}', slope * 10, p])
                ax.text(0.01, 0.95 if season == 'spring' else 0.84, text,
                        transform=ax.transAxes, color=color, fontsize=7.5, va='top')

            if 'SPEI' in var:
                ax.axhline(0, color='grey', lw=0.5)
            if self.split_year > start_year:
                ax.axvspan(self.split_year, self.end_year, color='grey', alpha=0.1)
            ax.set_ylabel(self.ylabel_dic.get(var, var), fontsize=10)
            ax.set_xlim(start_year, self.end_year)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[0].legend(loc='upper right', frameon=False, ncol=2)
        axes[-1].set_xlabel('Year', fontsize=10)
        plt.tight_layout()
        plt.savefig(outdir + f'{fname}.png', dpi=300)
        plt.savefig(outdir + f'{fname}.pdf')
        plt.show()
        plt.close()

        df_trend = pd.DataFrame(trend_list, columns=['variable', 'season', 'period', 'slope_per_decade', 'p_value'])
        df_trend.to_csv(outdir + f'{fname}_trend.csv', index=False)
        print(df_trend)


class SPEI_time_series(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'01_drought\SPEI_timeseries\\'
        self.start_year = 1982
        self.main_list = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI48']
        self.other_list = ['SPEI01', 'SPEI09', 'SPEI15', 'SPEI18', 'SPEI21', 'SPEI27',
                           'SPEI30', 'SPEI33', 'SPEI36', 'SPEI39', 'SPEI42', 'SPEI45']
        pass

    def run(self):
        df = self.load_df()
        self.plot_time_series(df, self.main_list, self.start_year, self.outdir, 'Fig_SPEI_main')
        self.plot_time_series(df, self.other_list, self.start_year, self.outdir, 'FigS_SPEI_other')
        pass


class SNU_LAI_time_series(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'02_vegetation\SNU_LAI_timeseries\\'
        self.start_year = 1982
        self.var_list = ['SNU_LAI_relative_change']
        self.ylabel_dic = {'SNU_LAI_relative_change': 'Relative Change (%)'}
        self.LAI_threshold = 0.3   ## drop pixels whose multi-year mean LAI < 0.3
        pass

    def run(self):
        df = self.load_df()
        df = self.mask_low_LAI(df)
        self.plot_time_series(df, self.var_list, self.start_year, self.outdir, 'Fig_SNU_LAI')
        pass

    def mask_low_LAI(self, df):
        for season in self.season_list:
            pix_mean = df.groupby('pix')[f'{season}_SNU_LAI'].transform('mean')   ## multi-year mean per pixel
            df.loc[pix_mean < self.LAI_threshold, f'{season}_SNU_LAI_relative_change'] = np.nan
        return df


class MODIS_LAI_time_series(Time_series_base):
    def __init__(self):
        super().__init__()
        self.outdir = self.result2 + rf'02_vegetation\MODIS_LAI_timeseries\\'
        self.start_year = 2003
        self.var_list = ['MODIS_LAI_relative_change']
        self.ylabel_dic = {'MODIS_LAI_relative_change': 'Relative Change (%)'}
        self.trend_period_list = [(2003, 2024, '-')]   ## MODIS starts 2003: one trend only
        self.LAI_threshold = 0.3   ## drop pixels whose multi-year mean LAI < 0.3
        pass

    def run(self):
        df = self.load_df()
        df = self.mask_low_LAI(df)
        self.plot_time_series(df, self.var_list, self.start_year, self.outdir, 'Fig_MODIS_LAI')
        pass

    def mask_low_LAI(self, df):
        ## df has no raw MODIS LAI, so take the multi-year mean (2003-2024) from the npy
        for season in self.season_list:
            val_dic = T.load_npy(result_root + rf'\MODIS_LAI\{season}_LAI.npy')
            mean_dic = {pix: np.nanmean(np.array(vals, dtype=float)) for pix, vals in val_dic.items()}
            pix_mean = df['pix'].map(mean_dic)
            df.loc[~(pix_mean >= self.LAI_threshold), f'{season}_MODIS_LAI_relative_change'] = np.nan
        return df


## add a new variable: copy SNU_LAI_time_series, change outdir / start_year / var_list


def main():
    SPEI_time_series().run()
    SNU_LAI_time_series().run()
    MODIS_LAI_time_series().run()


if __name__ == '__main__':
    main()
