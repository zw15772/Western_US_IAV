import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= rf'D:\Western_US_IAV\Result\greening_analysis\relative_change\trend\SNU_LAI_trend.tif'
D=DIC_and_TIF(tif_template=tif_template)

class greening_analysis:
    def __init__(self):
        pass
    def run(self):
        self.relative_change()
        self.trend_analysis()
        pass
    def relative_change(self):

        f = data_root+r'MODIS_LAI\dic\annual_gs_dic_mean.npy'
        outdir = result_root + rf'greening_analysis\MODIS_LAI\\relative_change\\'
        Tools().mk_dir(outdir, force=True)

        outf = outdir + 'MODIS_LAI_mean_season1.npy'
        # print(outf);exit()


        dic = T.load_npy(f)

        zscore_dic = {}

        for pix in tqdm(dic):



            # print(len(dic[pix]))
            time_series = dic[pix]['season1']


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
          #   plt.plot(anomaly)
          #   # plt.legend(['anomaly'])
          #   plt.show()
          #
          #   plt.plot(relative_change)
          #   plt.legend(['relative_change'])
          #   plt.legend(['anomaly','relative_change'])
          #   plt.show()

                ## save
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
                # print(time_series)

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
                vmin=-1,
                vmax=1,
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

            # D.arr_to_tif(arr_trend, outf + '_trend.tif')
            # D.arr_to_tif(p_value_arr, outf + '_p_value.tif')
            #
            # np.save(outf + '_trend', arr_trend)
            # np.save(outf + '_p_value', p_value_arr)


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





def main():
    greening_analysis().run()
    # area_weighted_average().run()
    # PLOT_greening_IAV().run()
    # IAV_analysis().run()
    # LAImin_LAImax().run()




if __name__ == '__main__':
    main()