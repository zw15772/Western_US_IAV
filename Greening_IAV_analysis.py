import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= rf'D:\Western_US_IAV\Result\greening_analysis\relative_change\trend\SNU_LAI_trend.tif'
D=DIC_and_TIF(tif_template=tif_template)

class greening_analysis:
    def __init__(self):
        pass
    def run(self):
        # self.relative_change()
        self.trend_analysis()
        pass
    def relative_change(self):

        f = data_root+r'SNU_LAI/extract_growing_season_LAI_mean/growing_season_LAI_mean.npy'
        outdir = result_root + rf'greening_analysis/relative_change/'
        Tools().mk_dir(outdir, force=True)

        outf = outdir + 'SNU_LAI.npy'
        # print(outf);exit()


        dic = T.load_npy(f)

        zscore_dic = {}

        for pix in tqdm(dic):



            # print(len(dic[pix]))
            time_series = dic[pix]['growing_season']


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
        phenology_mask_f = data_root + rf'\basedata\Phenology_extraction\SeasType.tif'
        phenology_mask_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(phenology_mask_f)
        phenology_dic = D.spatial_arr_to_dic(phenology_mask_arr)



        fdir = result_root + r'greening_analysis/relative_change/'
        outdir = result_root + r'greening_analysis/relative_change/trend/'
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
                phenology_type=phenology_dic[pix]
                # print(phenology_type)
                if phenology_type == 3:
                    continue

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

class PLOT_greening_IAV:
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        # self.plot_relative_change_LAI()
        self.plot_CV_LAI()
        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['SeasType'] !=3]
        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 45]
        #
        # df = df[df['landcover_classfication'] != 'Cropland']


        return df
    def plot_relative_change_LAI(self):  ##### plot for 4 clusters

        df = T.load_df(
            result_root + rf'greening_analysis/Dataframe/greening_analysis_area_weighted.df')
        print(len(df))
        df=self.df_clean(df)
        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = D.pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()


        print(len(df))
        T.print_head_n(df)
        # exit()

        # create color list with one green and another 14 are grey

        color_list = ['black','green', 'blue',  'magenta', 'black','purple',  'purple', 'black', 'yellow', 'purple', 'pink', 'grey',
                      'brown', 'lime', 'teal', 'magenta']
        linewidth_list = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


        fig = plt.figure()
        i = 1


        variable_list = [
                         'LAI_growing_season_relative_change',
                         ]
        dic_label={'LAI_growing_season_relative_change':'SNU_LAI_growing_season_relative_change',}
        year_list=range(1982,2024)

        result_dic = {}

        for var in variable_list:
            mean_dic = {}
            for year in year_list:
                df_i = df[df['year'] == year]
                ## scheme1
                vals = np.array(df_i[f'{var}'].tolist(), dtype=float)
                weight = np.array(df_i['area_weight'].tolist(), dtype=float)
                weighted_mean_values = (
                        np.nansum(vals * weight)
                        / np.nansum(weight * np.isfinite(vals))
                )

                # print(var, year, weighted_mean_values)
                ## scheme2
                # vals = np.array(df_i[f'{var}_relative_change'].tolist(), dtype=float)
                # weighted_mean_values = np.nanmean(vals)

                mean_dic[year] = weighted_mean_values

            result_dic[var] = mean_dic


        # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()



        # T.print_head_n(df_new);exit()


        flag=0
        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            plt.plot(year_list, df_new[var], label=dic_label[var],linewidth=linewidth_list[flag], color=color_list[flag])
            flag=flag+1
            slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])
            print(var, f'{slope:.2f}', f'{p_value:.2f}')
        plt.ylabel('Relative change LAI (%)')

        plt.grid(True, axis='x')   # 只画竖线（随 x 刻度）

        plt.legend()
        plt.show()
        # out_pdf_fdir = result_root + rf'\Figure\\weighted_area\\Figure1a\\'
        # T.mk_dir(out_pdf_fdir, force=True)
        # plt.savefig(out_pdf_fdir + 'time_series_relative_change_mean.pdf', dpi=300, bbox_inches='tight')
        # plt.close()


    def plot_CV_LAI(self):  ##### plot for 4 clusters

        df = T.load_df(
            result_root + rf'IAV_analysis/Dataframe/CV_LAI_area_weighted.df')
        print(len(df))
        df = self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = D.pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()

        # create color list with one green and another 14 are grey

        color_list = ['black', 'green', 'blue', 'magenta', 'black', 'purple', 'purple', 'black', 'yellow', 'purple',
                      'pink', 'grey',
                      'brown', 'lime', 'teal', 'magenta']
        linewidth_list = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        variable_list = [
                         'SNU_LAI_CV',
                          ]
        dic_label = {'SNU_LAI_CV': 'SNU_LAI_CV',
                     }
        year_list = range(0, 29)

        result_dic = {}

        for var in variable_list:
            mean_dic={}
            for year in year_list:
                df_i = df[df['window'] == year]
                ## scheme1
                vals = np.array(df_i[f'{var}'].tolist(), dtype=float)
                weight=np.array(df_i['area_weight'].tolist(),dtype=float)
                weighted_mean_values = (
                        np.nansum(vals * weight)
                        / np.nansum(weight * np.isfinite(vals))
                )
                print(year,weighted_mean_values)
                ## scheme2
                # vals = np.array(df_i[f'{var}_detrend_CV_area_weighted'].tolist(), dtype=float)
                # weighted_mean_values = np.nanmean(vals)

                mean_dic[year] = weighted_mean_values

            result_dic[var] = mean_dic


        # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()
        # T.print_head_n(df_new);exit()

        flag = 0

        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            plt.plot(
                year_list,
                df_new[var],
                label=dic_label[var],
                linewidth=linewidth_list[flag],
                color=color_list[flag]
            )

            slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])
            print(var, f'{slope:.2f}', f'{p_value:.2f}')

            ## std

            flag = flag + 1
        ## if var == 'composite_LAI_CV': plot CI bar

        window_size = 15

        # set xticks with 1982-1997, 1998-2013,.. 2014-2020
        year_range = range(1982, 2025)
        year_range_str = []
        for year in year_range:

            start_year = year
            end_year = year + window_size - 1
            if end_year > 2025:
                break
            year_range_str.append(f'{start_year}-{end_year}')

        plt.xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right')
        plt.yticks(np.arange(8, 15, 2))

        plt.ylabel(f'CV_LAI (%/yr)')

        plt.legend(loc='upper left')

        # plt.show()
        plt.tight_layout()
        # out_pdf_fdir = result_root + rf'\FIGURE\weighted_area\\'
        # T.mk_dir(out_pdf_fdir, force=True)
        # plt.savefig(out_pdf_fdir + 'time_series_CV_mean.pdf', dpi=300, bbox_inches='tight')
        # plt.close()


        plt.legend()
        plt.show()

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






class IAV_analysis():
    def __init__(self):
        pass
    def run(self):
        # self.detrend()
        # self.extract_moving_window()
        # self.moving_window_CV_extraction_anaysis_LAI()
        self.trend_analysis()
        pass

    def detrend(self):
        phenology_mask_f = data_root + rf'SNU_LAI/Phenology_extraction/SeasType.tif'
        phenology_mask_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(phenology_mask_f)
        phenology_dic = D.spatial_arr_to_dic(phenology_mask_arr)

        fdir = data_root + rf'/SNU_LAI/extract_growing_season_LAI_mean/'
        outdir = result_root + rf'IAV_analysis/detrend/'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            print(f)

            outf = outdir + f.split('.')[0] + '_detrend.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

            detrend_zscore_dic = {}

            for pix in tqdm(dic):
                phenology_val=phenology_dic[pix]
                if phenology_val==3:
                    continue
                r, c = pix
                # print(len(dic[pix]))
                time_series = dic[pix]['growing_season']
                # print(time_series)
                time_series = np.array(time_series, dtype=float)
                # plt.plot(time_series)
                # plt.show()
                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.std(time_series) == 0:
                    continue
                ##### if count of nan is more than 50%, then skip
                if np.sum(np.isnan(time_series)) / len(time_series) > 0.5:
                    continue
                # mean = np.nanmean(time_series)
                # std=np.nanstd(time_series)
                # if std == 0:
                #     continue
                # delta_time_series = (time_series - mean) / std
                # if np.isnan(time_series).any():
                #     continue
                time_series = T.interp_nan(time_series)
                detrend_delta_time_series = T.detrend_vals(time_series)
                plt.plot(time_series)
                plt.plot(detrend_delta_time_series)
                plt.show()

                detrend_zscore_dic[pix] = detrend_delta_time_series

            np.save(outf, detrend_zscore_dic)

    def extract_moving_window(self):

        fdir_all = result_root + rf'/IAV_analysis/detrend//'
        outdir = result_root + rf'/IAV_analysis/moving_window_extraction//'

        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir_all):

            if not f.endswith('.npy'):
                continue
            if not 'detrend' in f:
                continue

            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)

            # if os.path.isfile(outf):
            #     continue


            dic = T.load_npy(fdir_all + f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                # time_series = dic[pix][mode]
                time_series = dic[pix]
                # plt.plot(time_series)
                # plt.show()

                time_series = np.array(time_series)
                # if T.is_all_nan(time_series):
                #     continue
                if len(time_series) == 0:
                    continue

                # time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                # print((len(time_series)))
                ## if all values are identical, then continue
                if np.nanmax(time_series) == np.nanmin(time_series):
                    continue

                # new_x_extraction_by_window[pix] = self.forward_window_extraction_detrend_anomaly(time_series, window)
                new_x_extraction_by_window[pix] = self.forward_window_extraction(time_series, window)

            T.save_npy(new_x_extraction_by_window, outf)

    def forward_window_extraction(self, x, window):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        # new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        new_x_extraction_by_window = []
        for i in range(len(x) + 1):
            if i + window >= len(x) + 1:
                continue
            else:
                anomaly = []
                relative_change_list = []
                x_vals = []
                for w in range(window):
                    x_val = (x[i + w])
                    x_vals.append(x_val)
                if np.isnan(np.nanmean(x_vals)):
                    continue

                # x_mean=np.nanmean(x_vals)

                # for i in range(len(x_vals)):
                #     if x_vals[0]==None:
                #         continue
                # x_anomaly=(x_vals[i]-x_mean)
                # relative_change = (x_vals[i] - x_mean) / x_mean

                # relative_change_list.append(x_vals)
                new_x_extraction_by_window.append(x_vals)
        return new_x_extraction_by_window

        pass

    def moving_window_CV_extraction_anaysis_LAI(self):
        window_size = 15

        fdir = result_root + rf'IAV_analysis/moving_window_extraction/'
        outdir = result_root + rf'IAV_analysis/moving_window_CV_extraction_anaysis/'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)

            outf = outdir + f.split('.')[0] + f'_CV.npy'
            print(outf)

            # if os.path.isfile(outf):
            #     continue

            new_x_extraction_by_window = {}
            trend_dic = {}
            p_value_dic = {}

            for pix in tqdm(dic):
                trend_list = []
                time_series_all = dic[pix]
                # print(len(time_series_all))
                if len(time_series_all) < 28:  ##
                    continue
                time_series_all = np.array(time_series_all)
                slides = len(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue


                    ### if all values are identical, then continue
                    time_series = time_series_all[ss]
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    # print(len(time_series))

                    if np.nanmean(time_series) == 0:
                        continue
                    cv = np.nanstd(time_series) / np.nanmean(time_series) * 100

                    trend_list.append(cv)
                    # print(trend_list)
                # plt.plot(trend_list)
                # plt.show()
                trend_dic[pix] = trend_list

            np.save(outf, trend_dic)
            T.open_path_and_file(outdir)


    def trend_analysis(self):

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        ##each window average trend
        phenology_mask_f = data_root + rf'SNU_LAI/Phenology_extraction/SeasType.tif'
        phenology_mask_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(phenology_mask_f)
        phenology_dic = D.spatial_arr_to_dic(phenology_mask_arr)

        fdir = result_root + r'IAV_analysis/moving_window_CV_extraction_anaysis/'
        outdir = result_root + r'IAV_analysis/moving_window_CV_extraction_anaysis/trend/'
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
                phenology_type=phenology_dic[pix]
                # print(phenology_type)
                if phenology_type == 3:
                    continue

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
                cmap='PRGn_r',
                vmin=-.8,
                vmax=.8,
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
            cbar.set_label('Trend')

            plt.title(f)
            plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)



class LAImin_LAImax:
    def run(self):
        # self.extract_growing_season_LAI_stats()
        # self.LAImax_LAImin_diff()
        self.trend_analysis()

    def extract_growing_season_LAI_stats(self):
        """
        For each pixel and each year, extract:
        - LAI_min, LAI_max
        - Percentiles: 95, 90, 80, 70, 30, 20, 10, 5
        """

        fdir = data_root + r'/SNU_LAI/extract_growing_season_monthly/'
        outdir = result_root + r'/LAImin_LAImax/raw/percentiles/'
        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)


        # --- containers ---
        lai_min_dic = {}
        lai_max_dic = {}

        percentiles = [95, 90, 80, 70, 30, 20, 10, 5]
        percentile_dic_all = {p: {} for p in percentiles}

        # --- main loop ---
        for pix in tqdm(spatial_dic):
            vals_growing_season = spatial_dic[pix]

            lai_min_list = []
            lai_max_list = []
            percentile_lists = {p: [] for p in percentiles}

            for val in vals_growing_season:
                if T.is_all_nan(val):
                    continue

                val = np.array(val, dtype=float)

                lai_min_list.append(np.nanmin(val))
                lai_max_list.append(np.nanmax(val))

                for p in percentiles:
                    percentile_lists[p].append(np.nanpercentile(val, p))

            # --- save to dicts ---
            lai_min_dic[pix] = lai_min_list
            lai_max_dic[pix] = lai_max_list

            for p in percentiles:
                percentile_dic_all[p][pix] = percentile_lists[p]

        # --- save files ---
        np.save(outdir + 'LAI_min.npy', lai_min_dic)
        np.save(outdir + 'LAI_max.npy', lai_max_dic)

        for p in percentiles:
            np.save(outdir + f'LAI_p{p}.npy', percentile_dic_all[p])



    def LAImax_LAImin_diff(self):
        fmin_path=data_root+rf'SNU_LAI/extract_growing_season_LAI_min/growing_season_LAI_min.npy'
        fmax_path=data_root+rf'SNU_LAI/extract_growing_season_LAI_max/growing_season_LAI_max.npy'
        dic_max=T.load_npy(fmax_path)
        dic_min=T.load_npy(fmin_path)
        spatial_dic={}
        for pix in tqdm(dic_max):
            if not pix in dic_min:
                continue
            vals_max=dic_max[pix]['growing_season']
            vals_max=np.array(vals_max)


            vals_min=dic_min[pix]['growing_season']
            vals_min=np.array(vals_min)
            vals_differences=vals_max-vals_min
            spatial_dic[pix]=vals_differences
        outdir=result_root + r'LAI_min_LAImax/raw/differences/'
        T.mk_dir(outdir, force=True)
        outf=outdir + r'LAImax_LAImin_difference.npy'

        T.save_npy(spatial_dic,outf)


        pass
    def trend_analysis(self):

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        ##each window average trend
        phenology_mask_f = data_root + rf'SNU_LAI/Phenology_extraction/SeasType.tif'
        phenology_mask_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(phenology_mask_f)
        phenology_dic = D.spatial_arr_to_dic(phenology_mask_arr)



        fdir = result_root + rf'LAImin_LAImax/raw/percentiles/'
        outdir = result_root + r'LAI_min_LAImax/raw/percentiles/trend/'
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
                phenology_type=phenology_dic[pix]
                # print(phenology_type)
                if phenology_type == 3:
                    continue

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
                vmin=-0.01,
                vmax=0.01,
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
            cbar.set_label('Trend')

            plt.title(f)
            plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)


def main():
    greening_analysis().run()
    # area_weighted_average().run()
    # PLOT_greening_IAV().run()
    # IAV_analysis().run()
    # LAImin_LAImax().run()




if __name__ == '__main__':
    main()