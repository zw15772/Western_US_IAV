import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= rf'D:\Western_US_IAV\Result\greening_analysis\SNU_LAI\\trend\SNU_LAI_trend.tif'
D=DIC_and_TIF(tif_template=tif_template)

class greening_analysis:
    def __init__(self):
        pass
    def run(self):
        self.relative_change()
        # self.trend_analysis()
        pass
    def relative_change(self):

        f = data_root+r'\SNU_LAI\spring_summer_season_LAI_mean\\spring_summer_season_LAI_mean.npy'
        outdir = result_root + rf'greening_analysis\SNU_LAI\\relative_change\\'
        Tools().mk_dir(outdir, force=True)


        dic = T.load_npy(f)

        zscore_dic = {}

        for pix in tqdm(dic):



            # print(len(dic[pix]))
            time_series = dic[pix]['summer']


            time_series = np.array(time_series)


            # print(len(time_series))

            if np.isnan(np.nanmean(time_series)):
                continue

            time_series = time_series
            mean = np.nanmean(time_series)
            relative_change = (time_series - mean) / mean * 100
            anomaly = time_series - mean
            zscore_dic[pix] = relative_change
          # plot
          #   plt.plot(time_series)
            # plt.plot(anomaly)
            # plt.legend(['anomaly'])
            # plt.show()

            # plt.plot(relative_change)
            # plt.legend(['relative_change'])
            # plt.legend([ 'relative_change'])
            # plt.show()

                ## save
        outf=outdir+rf'summer_July_Sept.npy'
        T.save_npy( zscore_dic, outf)

    def trend_analysis(self):

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        ##each window average trend


        fdir = result_root + r'greening_analysis/MODIS_LAI/relative_change/'
        outdir = result_root + r'greening_analysis/MODIS_LAI/relative_change/trend/'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0]
            # if os.path.isfile(outf + '_trend.tif'):
            #     continue
            print(outf)


            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix


                time_series = dic[pix]
                # print(time_series)
                time_series = np.array(time_series)
                print(len(time_series));exit()

                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                # if len(set(time_series)) == 1:
                #     continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                try:

                    # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series)), time_series)
                    trend_dic[pix] = slope
                    p_value_dic[pix] = p_value
                except:
                    continue

            arr_trend = D.pix_dic_to_spatial_arr(trend_dic)


            p_value_arr = D.pix_dic_to_spatial_arr(p_value_dic)
            fpath=result_root+rf'/greening_analysis/relative_change/trend/SNU_LAI_trend.tif'
            ll,lr,ul,ur=RasterIO_Func().get_tif_bounds(fpath)
            print(ll,lr,ul,ur)

            ax = plt.axes(projection=ccrs.PlateCarree())

            # --- 画趋势图 ---
            im = ax.imshow(
                arr_trend,
                cmap='RdBu',
                vmin=-1.5,
                vmax=1.5,
                extent=[-124.55, -102.04, 25.59,49],
                transform=ccrs.PlateCarree()
            )

            # --- 加 continent ---
            ax.add_feature(
                cfeature.LAND,
                facecolor='none',  #
                edgecolor='black',
                linewidth=0.5,
                zorder=2
            )
            ax.add_feature(cfeature.STATES, linewidth=0.3)

            lon_min_box = -125
            lon_max_box = -105
            lat_min_box = 30
            lat_max_box = 45

            rect = mpatches.Rectangle(
                (lon_min_box, lat_min_box),  # 左下角 (lon, lat)
                lon_max_box - lon_min_box,  # 宽度
                lat_max_box - lat_min_box,  # 高度
                linewidth=1.5,
                edgecolor='black',
                facecolor='none',
                transform=ccrs.PlateCarree(),  # ⭐关键
                zorder=10
            )

            ax.add_patch(rect)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Trend (relative_change, %)')

            plt.title(f)
            plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)


class area_weighted_average():
    def __init__(self):
        pass
    def run(self):
        # self.weighted_average_LAI_relative_change()
        self.weighted_average_LAICV()
        # self.weighted_average_LAI_percentile()




    def weighted_average_LAI_relative_change(self):  ###add weighted average LAI in dataframe
        df =result_root+rf'greening_analysis/Dataframe/greening_analysis.df'
        df = T.load_df(df)

        vars_to_weight = [
            'LAI_growing_season_relative_change',

        ]

        df['area_weight'] = np.cos(np.deg2rad(df['lat']))

        df_aw_year = (
            df
            .groupby('year')
            .apply(
                lambda x: pd.Series({
                    f'{v}_area_weighted':
                        (x[v] * x['area_weight']).sum() / x['area_weight'].sum()
                    for v in vars_to_weight
                })
            )
            .reset_index()
        )


        df = df.merge(df_aw_year, on='year', how='left')



        # plt.figure(figsize=(6, 4))
        #
        # plt.plot(
        #     df_aw_year['year'],
        #     df_aw_year['SNU_LAI_relative_change_area_weighted'],
        #     color='black',
        #     lw=2
        # )
        #
        # plt.xlabel('Year')
        # plt.ylabel('Area-weighted LAI change')
        # plt.title('Dryland vegetation change (area-weighted)')
        # plt.tight_layout()
        # plt.show()

        # df[df['year'] == 1982][
        #     ['SNU_LAI_relative_change_area_weighted',
        #      'LAI4g_relative_change_area_weighted',
        #      'composite_LAI_mean_relative_change_area_weighted',
        #      'GLOBMAP_LAI_relative_change_area_weighted',
        #
        #      ]
        # ].head()
        # T.print_head_n(df)


        outf=result_root+rf'/greening_analysis/Dataframe//greening_analysis_area_weighted.df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

        pass

    def weighted_average_LAICV(self):  ###add weighted average LAI in dataframe
        df =result_root+rf'IAV_analysis/Dataframe/CV_LAI.df'
        df = T.load_df(df)

        vars_to_weight = [
            'SNU_LAI_CV',


        ]

        df['area_weight'] = np.cos(np.deg2rad(df['lat']))

        df_aw_year = (
            df
            .groupby('window')
            .apply(
                lambda x: pd.Series({
                    f'{v}_area_weighted':
                        (x[v] * x['area_weight']).sum() / x['area_weight'].sum()
                    for v in vars_to_weight
                })
            )
            .reset_index()
        )


        df = df.merge(df_aw_year, on='window', how='left')



        # plt.figure(figsize=(6, 4))
        #
        # plt.plot(
        #     df_aw_year['year'],
        #     df_aw_year['SNU_LAI_relative_change_area_weighted'],
        #     color='black',
        #     lw=2
        # )
        #
        # plt.xlabel('Year')
        # plt.ylabel('Area-weighted LAI change')
        # plt.title('Dryland vegetation change (area-weighted)')
        # plt.tight_layout()
        # plt.show()

        # df[df['year'] == 1982][
        #     ['SNU_LAI_relative_change_area_weighted',
        #      'LAI4g_relative_change_area_weighted',
        #      'composite_LAI_mean_relative_change_area_weighted',
        #      'GLOBMAP_LAI_relative_change_area_weighted',
        #
        #      ]
        # ].head()
        # T.print_head_n(df)


        outf=result_root+rf'IAV_analysis/Dataframe/CV_LAI_area_weighted.df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

class PLOT_vegetation_change():
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        self.plot_time_series()
        pass

    def plot_time_series(self):
        dff=result_root + rf'\SPEI_Greening\Dataframe\Dataframe_1982_2024.df'
        df=T.load_df(dff)

        df=self.df_clean(df)


        year_list=list(range(1982, 2025))
        result_dic = {}
        eco_region_list = df['Ecoregion_level_II'].dropna().unique().tolist()
        eco_region_list.append('Western US')

        eco_region_list=['Western US','Western Cordillera','Upper Gila Mountains',
        'Warm Desert','Cold Desert','Western Sierra Madre Piedmont']

        for eco in eco_region_list:

            if eco == 'Western US':
                # 2. Use a single '=' for assignment, and handle the logic
                df_i = df.copy()
            else:
                df_i = df[df['Ecoregion_level_II'] == eco]

            pix_list = df_i['pix'].tolist()
            unique_pix_list = list(set(pix_list))
            spatial_dic = {}

            # for pix in unique_pix_list:
            #     spatial_dic[pix] = 1
            # arr = D.pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
            # plt.colorbar()
            # plt.title(f'{eco}')
            # plt.show()


            for season in ['spring_March_May', 'summer_July_Sept']:
                mean_dic = {}
                std_dic = {}

                for year in year_list:
                    df_ii = df_i[df_i['year'] == year]
                    ## scheme1
                    vals = np.array(df_ii[season].tolist(), dtype=float)
                    vals_len = len(vals)
                    weight = np.array(df_ii['area_weight'].tolist(), dtype=float)
                    weighted_mean = (
                            np.nansum(vals * weight)
                            / np.nansum(weight * np.isfinite(vals))
                    )
                    # weighted_mean=np.nanmean(vals)
                    # weighted_std = np.nanstd(vals)

                    #####加权方差
                    weighted_var = np.nansum(weight * (vals - weighted_mean) ** 2) / np.nansum(weight)

                    weighted_std = np.sqrt(weighted_var)

                    mean_dic[year] = weighted_mean
                    std_dic[year] = weighted_std
                    # print(weighted_std)

                result_dic[f'{eco}_{season}'] = mean_dic
                result_dic[f'{eco}_{season}_std'] = std_dic

                # 只存一次长度
                result_dic[f'{eco}_len'] = len(df_i)

            # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()

        # T.print_head_n(df_new);exit()

        flag = 0




        for eco in eco_region_list:
            plt.figure(figsize=(self.map_width, self.map_height))


            spring_vals = df_new[f'{eco}_spring_March_May']
            summer_vals = df_new[f'{eco}_summer_July_Sept']

            vals_len = df_new[f'{eco}_len'][0]


            plt.plot(year_list, spring_vals, label='Spring', linewidth=2)
            plt.plot(year_list, summer_vals, label='Summer', linewidth=2)


            slope_s, _, _, p_s, _ = stats.linregress(year_list, spring_vals)
            slope_sum, _, _, p_sum, _ = stats.linregress(year_list, summer_vals)

            stats_text = (
                f'Spring: slope={slope_s:.2f}, p={p_s:.2f}\n'
                f'Summer: slope={slope_sum:.2f}, p={p_sum:.2f}'
            )

            plt.text(0.95, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            plt.ylabel('SNU_LAI_relative_change (%)')
            plt.title(f'{eco}_n={vals_len}', fontsize=12)

            plt.legend()
            plt.grid(True, axis='x')

            plt.show()
            plt.close()


        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()

        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 45]
        #
        # df = df[df['landcover_classfication'] != 'Cropland']

        return df

def main():
    # greening_analysis().run()
    # area_weighted_average().run()
    PLOT_vegetation_change().run()




if __name__ == '__main__':
    main()