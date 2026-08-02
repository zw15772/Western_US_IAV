import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= rf'D:\Western_US_IAV\Data\basedata\200902.tif'
D=DIC_and_TIF(tif_template=tif_template)


class SPEI_Greening_categorize:
    def run(self):
        # self.categrize_2()
        self.plot_categorize()
        pass
    def categrize(self):
        ## wetting_greening
        ## drying_browning
        ## drying_greening
        ## wetting_browning
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt

        dff=result_root+rf'\SPEI_Greening\Dataframe\SPEI_Greening.df'
        df=T.load_df(dff)
        df=df.dropna()
        df['category'] = np.nan
        df['category'] = 5  # 5 = Not significant


        # 定义显著
        sig_lai = df['SNU_LAI_p_value'] < 0.05
        sig_spei = df['SPEI12_p_value'] < 0.05

        mask = sig_lai & sig_spei

        # 1 = Wetting-Greening
        df.loc[mask & (df['SPEI12_trend'] > 0) & (df['SNU_LAI_trend'] > 0), 'category'] = 1

        # 2 = Drying-Greening
        df.loc[mask & (df['SPEI12_trend'] < 0) & (df['SNU_LAI_trend'] > 0), 'category'] = 2

        # 3 = Wetting-Browning
        df.loc[mask & (df['SPEI12_trend'] > 0) & (df['SNU_LAI_trend'] < 0), 'category'] = 3

        # 4 = Drying-Browning
        df.loc[mask & (df['SPEI12_trend'] < 0) & (df['SNU_LAI_trend'] < 0), 'category'] = 4

        spatial_dic=T.df_to_spatial_dic(df,'category')
        array=D.pix_dic_to_spatial_arr(spatial_dic)

        fpath = result_root + rf'/greening_analysis/relative_change/trend/SNU_LAI_trend.tif'
        ll, lr, ul, ur = RasterIO_Func().get_tif_bounds(fpath)
        print(ll, lr, ul, ur)

        ax = plt.axes(projection=ccrs.PlateCarree())

        # --- 画趋势图 ---
        im = ax.imshow(
            array,
            cmap='RdBu',
            vmin=1,
            vmax=5,
            extent=[-124.55, -102.04, 25.59, 49],
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


        plt.show()

        # D.arr_to_tif(arr_trend, outf + '_trend.tif')
        # D.arr_to_tif(p_value_arr, outf + '_p_value.tif')
        #
        # np.save(outf + '_trend', arr_trend)
        # np.save(outf + '_p_value', p_value_arr)
        plt.imshow(array)
        plt.show()
        pass

    def categrize_2(self):

        import numpy as np
        import pandas as pd
        dff = result_root + rf'\SPEI_Greening\Dataframe\SPEI_Greening.df'
        df = T.load_df(dff)
        df = df.dropna()

        # ================================
        # Step 1: 定义 moisture regime
        # ================================

        df['moisture_group'] = np.nan

        # 1 = significant wetting
        df.loc[(df['SPEI12_trend'] > 0) &
               (df['SPEI12_p_value'] < 0.05),
        'moisture_group'] = 1

        # 2 = significant drying
        df.loc[(df['SPEI12_trend'] < 0) &
               (df['SPEI12_p_value'] < 0.05),
        'moisture_group'] = 2

        # 3 = stable moisture
        df.loc[df['SPEI12_p_value'] >= 0.05,
        'moisture_group'] = 3

        # ================================
        # Step 2: 定义 LAI response
        # ================================

        df['lai_group'] = 3  # 默认 3 = stable

        # 1 = significant greening
        df.loc[(df['SNU_LAI_trend'] > 0) &
               (df['SNU_LAI_p_value'] < 0.05),
        'lai_group'] = 1

        # 2 = significant browning
        df.loc[(df['SNU_LAI_trend'] < 0) &
               (df['SNU_LAI_p_value'] < 0.05),
        'lai_group'] = 2

        df['category_mean'] = (df['moisture_group'] - 1) * 3 + df['lai_group']

        category_labels = {
            1: 'Wetting - Greening',
            2: 'Wetting - Browning',
            3: 'Wetting - Stable LAI',
            4: 'Drying - Greening',
            5: 'Drying - Browning',
            6: 'Drying - Stable LAI',
            7: 'Stable Moisture - Greening',
            8: 'Stable Moisture - Browning',
            9: 'Stable Moisture - Stable LAI'
        }

        T.print_head_n(df)
        dff_new=result_root + rf'\SPEI_Greening\Dataframe\SPEI_Greening_mean.df'
        T.save_df(df, dff_new)
        T.df_to_excel(df,dff_new)
        spatial_dic=T.df_to_spatial_dic(df,'category_mean')
        array=D.pix_dic_to_spatial_arr(spatial_dic)
        outdir=result_root + rf'\SPEI_Greening\tif\\'
        Tools().mk_dir(outdir, force=True)
        D.arr_to_tif(array,outdir+rf'\SPEI_Greening_category_mean.tif')

        # ================================
        # Step 3: 统计比例
        # ================================

        # moisture_labels = {
        #     1: 'Significant Wetting',
        #     2: 'Significant Drying',
        #     3: 'Stable Moisture'
        # }
        #
        # lai_labels = {
        #     1: 'Greening',
        #     2: 'Browning',
        #     3: 'Stable LAI'
        # }
        #
        # print("\n===== Conditional LAI Response Under Different Moisture Regimes =====\n")
        #
        # for m in [1, 2, 3]:
        #
        #     sub = df[df['moisture_group'] == m]
        #
        #     total = len(sub)
        #
        #     if total == 0:
        #         continue
        #
        #     print(f"\n--- {moisture_labels[m]} (n = {total}) ---")
        #
        #     for l in [1, 2, 3]:
        #         count = len(sub[sub['lai_group'] == l])
        #         ratio = count / total * 100
        #
        #         print(f"{lai_labels[l]}: {ratio:.2f}%")
        #
        # pass

    def plot_categorize(self):
        dff=result_root + rf'\SPEI_Greening\Dataframe\SPEI_Greening_95percentile.df'
        df=T.load_df(dff)
        df=df.dropna()
        df=self.df_clean(df)
        # 计算比例
        table = pd.crosstab(
            df['moisture_group'],
            df['lai_group'],
            normalize='index'
        )

        # 保证顺序
        table = table.reindex(index=[1, 2, 3], columns=[1, 2, 3])

        data = table.values * 100  # 转百分比
        counts = df['moisture_group'].value_counts().sort_index()
        counts = counts.reindex([1, 2, 3])

        labels = ['Wetter', 'Dryer', 'Stable']

        x = np.arange(len(labels))  # 0,1,2
        width = 0.25

        fig, ax = plt.subplots(figsize=(5, 3))

        ax.bar(x - width, data[:, 0], width, label='Greening', color='#33a02c')
        ax.bar(x, data[:, 1], width, label='Browning', color='#e31a1c')
        ax.bar(x + width, data[:, 2], width, label='Stable LAI', color='#bdbdbd')

        for i, count in enumerate(counts):
            ax.text(
                x[i],
                102,  # 稍微高于100%
                f'n = {int(count)}',
                ha='center',
                va='bottom',
                fontsize=11
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('No. pixels (%)',fontsize=12)
        ax.set_ylim(0, 100)

        ax.legend(frameon=False)

        plt.tight_layout()
        plt.show()

        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['SeasType'] != 3]
        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 45]
        #
        # df = df[df['landcover_classfication'] != 'Cropland']
        return df

class SPEI_Greening_ecoregion:
    def __init__(self):

        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        # self.barplot_by_ecoregion_LAI()
        # self.barplot_by_ecoregion_SPEI()
        # self.weighted_average_variable()
        self.plot_time_series()
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

    def weighted_average_variable(self):  ###add weighted average LAI in dataframe
        dff=result_root + rf'\SPEI_Greening\Dataframe\Dataframe_area_weighted.df'
        df=T.load_df(dff)
        df=self.df_clean(df)

        vars_to_weight = [
            'SNU_LAI',

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


        outf=result_root + rf'\SPEI_Greening\Dataframe\Dataframe_area_weighted.df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

        pass



    def barplot_by_ecoregion_LAI(self):
        dff=result_root + rf'\SPEI_Greening\Dataframe\Dataframe.df'
        df=T.load_df(dff)
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
        result = []

        for eco in df['Ecoregion_level_II'].dropna().unique():

            subset = df[df['Ecoregion_level_II'] == eco]

            total = len(subset)

            if total == 0:
                continue

            sig_greening = ((subset['MODIS_LAI_mean_season1_trend'] > 0) &
                            (subset['MODIS_LAI_mean_season1_p_value'] < 0.05)).sum() / total

            non_sig_greening = ((subset['MODIS_LAI_mean_season1_trend'] > 0) &
                                (subset['MODIS_LAI_mean_season1_p_value'] >= 0.05)).sum() / total

            sig_browning = ((subset['MODIS_LAI_mean_season1_trend'] < 0) &
                            (subset['MODIS_LAI_mean_season1_p_value'] < 0.05)).sum() / total

            non_sig_browning = ((subset['MODIS_LAI_mean_season1_trend'] < 0) &
                                (subset['MODIS_LAI_mean_season1_p_value'] >= 0.05)).sum() / total

            result.append({
                'ecoregion': eco,
                'sig_greening': sig_greening,
                'non_sig_greening': non_sig_greening,
                'sig_browning': sig_browning,
                'non_sig_browning': non_sig_browning
            })
        result_df = pd.DataFrame(result)

        result_df = result_df.sort_values('sig_greening', ascending=False)

        plt.figure(figsize=(8, 6))

        plt.barh(result_df['ecoregion'],
                 result_df['sig_greening'],
                 color='darkgreen',
                 label='Sig Greening')

        plt.barh(result_df['ecoregion'],
                 result_df['non_sig_greening'],
                 left=result_df['sig_greening'],
                 color='lightgreen',
                 label='Non-sig Greening')

        left2 = result_df['sig_greening'] + result_df['non_sig_greening']

        plt.barh(result_df['ecoregion'],
                 result_df['sig_browning'],
                 left=left2,
                 color='darkred',
                 label='Sig Browning')

        plt.barh(result_df['ecoregion'],
                 result_df['non_sig_browning'],
                 left=left2 + result_df['sig_browning'],
                 color='salmon',
                 label='Non-sig Browning')

        plt.xlim(0, 1)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def barplot_by_ecoregion_SPEI(self):
        dff=result_root + rf'\SPEI_Greening\Dataframe\Dataframe.df'
        df=T.load_df(dff)
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
        result = []

        eco_order = [
            "Marine West Coast Forest",
            "Upper Gila Mountains",
            "Western Sierra Madre",
            "Western Cordillera",
            "West-Central Semiarid Prairies",
            "Western Sierra Madre Piedmont",
            "Cold Desert",
            "Mediterranean California",
            "Warm Desert",
            "South Central Semiarid Prairies",
            "Western Pacific Coastal Plain, Hills and Canyons",

        ]

        for eco in df['Ecoregion_level_II'].dropna().unique():

            subset = df[df['Ecoregion_level_II'] == eco]

            total = len(subset)

            if total == 0:
                continue

            sig_wetting = ((subset['SPEI12_annual_mean_trend'] > 0) &
                            (subset['SPEI12_annual_mean_p_value'] < 0.05)).sum() / total

            non_sig_wetting = ((subset['SPEI12_annual_mean_trend'] > 0) &
                                (subset['SPEI12_annual_mean_p_value'] >= 0.05)).sum() / total

            sig_drying = ((subset['SPEI12_annual_mean_trend'] < 0) &
                            (subset['SPEI12_annual_mean_p_value'] < 0.05)).sum() / total

            non_sig_drying= ((subset['SPEI12_annual_mean_trend'] < 0) &
                                (subset['SPEI12_annual_mean_p_value'] >= 0.05)).sum() / total

            result.append({
                'ecoregion': eco,
                'sig_wetting': sig_wetting,
                'non_sig_wetting': non_sig_wetting,
                'sig_drying': sig_drying,
                'non_sig_drying': non_sig_drying
            })
        result_df = pd.DataFrame(result)

        result_df['ecoregion'] = pd.Categorical(
            result_df['ecoregion'],
            categories=eco_order,
            ordered=True
        )

        result_df = result_df.sort_values('ecoregion')

        plt.figure(figsize=(8, 6))


        plt.barh(result_df['ecoregion'],
                 result_df['sig_wetting'],
                 color='darkblue',
                 label='Sig Wetting')

        plt.barh(result_df['ecoregion'],
                 result_df['non_sig_wetting'],
                 left=result_df['sig_wetting'],
                 color='lightblue',
                 label='Non-sig Wetting')

        left2 = result_df['sig_wetting'] + result_df['non_sig_wetting']

        plt.barh(result_df['ecoregion'],
                 result_df['sig_drying'],
                 left=left2,
                 color='red',
                 label='Sig Drying')

        plt.barh(result_df['ecoregion'],
                 result_df['non_sig_drying'],
                 left=left2 + result_df['sig_drying'],
                 color='orange',
                 label='Non-sig Drying')

        plt.xlim(0, 1)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()



class PLOT_vegetation_change():
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        self.plot_time_series_MODIS_record()
        # self.plot_time_series_SNU_record()
        pass

    def plot_time_series_MODIS_record(self):
        dff=result_root + rf'\SPEI_Greening\Dataframe\Dataframe.df'
        df=T.load_df(dff)

        df=self.df_clean(df)


        year_list=list(range(2003, 2025))
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
            for season in ['spring_LAI_anomaly', 'summer_LAI_anomaly']:
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
            plt.figure(figsize=(self.map_width*1.5, self.map_height))


            spring_vals = df_new[f'{eco}_spring_LAI_anomaly']
            summer_vals = df_new[f'{eco}_summer_LAI_anomaly']
            spring_std = df_new[f'{eco}_spring_LAI_anomaly_std']
            summer_std = df_new[f'{eco}_summer_LAI_anomaly_std']

            vals_len = df_new[f'{eco}_len'][0]




            slope_s, _, _, p_s, _ = stats.linregress(year_list, spring_vals)
            slope_sum, _, _, p_sum, _ = stats.linregress(year_list, summer_vals)
            color_spring = '#955F7C'

            plt.plot(
                year_list,
                spring_vals,
                color=color_spring,
                lw=2,
                label='Spring'
            )

            slope, intercept, r, p, _ = stats.linregress(year_list, spring_vals)
            years = np.array(year_list)

            plt.plot(
               year_list,
                slope * years + intercept,
                '--',
                color=color_spring,
                lw=2,
            )

            # -----------------------------
            # Summer
            # -----------------------------

            color_summer = '#07967F'

            plt.plot(
                year_list,
                summer_vals,
                color=color_summer,
                lw=2,
                label='Summer'
            )

            slope, intercept, r, p, _ = stats.linregress(year_list, summer_vals)

            plt.plot(
                year_list,
                slope * years + intercept,
                '--',
                color=color_summer,
                lw=2,

            )
            plt.fill_between(
                years,
                spring_vals - spring_std,
                spring_vals + spring_std,
                color=color_spring,
                alpha=0.2
            )

            plt.fill_between(
                years,
                summer_vals - summer_std,
                summer_vals + summer_std,
                color=color_summer,
                alpha=0.2
            )
            plt.legend()



            stats_text = (
                f'Spring: slope={slope_s:.2f}, p={p_s:.2f}\n'
                f'Summer: slope={slope_sum:.2f}, p={p_sum:.2f}'
            )

            plt.text(0.95, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                      )

            plt.ylabel('MODIS_LAI_anomaly(m2/m2)', fontsize=12)

            plt.title(f'{eco}_n={vals_len}', fontsize=12)


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
class PLOT_bar:
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        self.barplot_percentage()
        pass

    def barplot(self):
        ## plot
        dff=result_root+rf'\SPEI_Greening\Dataframe\Dataframe.df'
        season_list=['spring','summer']
        df = T.load_df(dff)
        df=self.df_clean(df)
        df_pixel = df.drop_duplicates(subset='pix')
        # for col in df.columns:
        #     print(col)
        # exit()
        print(df.columns.tolist())
        print(df_pixel.columns.tolist())
        eco_region_list = ['Western US', 'Western Cordillera', 'Upper Gila Mountains',
                           'Warm Desert', 'Cold Desert', 'Western Sierra Madre Piedmont']

        scale_list = np.arange(3, 49, 3)

        for season in season_list:

            for eco in eco_region_list:

                if eco == 'Western US':
                    df_i = df.copy()
                else:
                    df_i = df[df['Ecoregion_level_II'] == eco]

                # LAI
                df_lai = df_i[df_i[f' {season}_LAI_p_value'] < 0.05]
                lai_mean = df_lai[f' {season}_LAI_trend'].mean()

                # SPEI
                spei_trend = []

                for scale in scale_list:
                    df_SPEI = df_i[
                        df_i[f' {season}_SPEI{int(scale)}_p_value'] < 0.05
                        ]

                    vals = df_SPEI[f' {season}_SPEI{int(scale)}_trend']
                    spei_trend.append(vals.mean())

                # -----------------
                # Plot
                # -----------------

                plt.figure(figsize=(8, 4))

                plt.bar(
                    0,
                    lai_mean,
                    width=0.8,
                    color='#CFE3CA',
                    edgecolor='gray',
                    label='LAI'
                )

                x = np.arange(2, 2 + len(scale_list))

                plt.bar(
                    x,
                    spei_trend,
                    width=0.8,
                    color='#F6BC97',
                    edgecolor='gray',
                    label='SPEI'
                )

                plt.xticks(
                    [0] + list(x),
                    ['LAI'] + [str(i) for i in scale_list]
                )

                plt.ylabel('Trend')
                plt.xlabel('SPEI timescale (months)')
                ## add y=0
                plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
                plt.title(f'{eco} - {season}')

                plt.legend(frameon=False)
                plt.tight_layout()
                plt.show()



            pass

    def barplot_percentage(self):
        ## plot
        dff=result_root+rf'\SPEI_Greening\Dataframe\Dataframe.df'
        season_list=['spring','summer']
        df = T.load_df(dff)
        df=self.df_clean(df)
        df_pixel = df.drop_duplicates(subset='pix')
        # for col in df.columns:
        #     print(col)
        # exit()
        print(df.columns.tolist())
        print(df_pixel.columns.tolist())
        eco_region_list = ['Western US', 'Western Cordillera', 'Upper Gila Mountains',
                           'Warm Desert', 'Cold Desert', 'Western Sierra Madre Piedmont']

        scale_list = np.arange(3, 49, 3)

        for season in season_list:

            for eco in eco_region_list:

                if eco == 'Western US':
                    df_i = df.copy()
                else:
                    df_i = df[df['Ecoregion_level_II'] == eco]

                n_total = len(df_i)

                lai_green = np.sum(
                    (df_i[f' {season}_LAI_trend'] > 0) &
                    (df_i[f' {season}_LAI_p_value'] < 0.05)
                )

                lai_brown = np.sum(
                    (df_i[f' {season}_LAI_trend'] < 0) &
                    (df_i[f' {season}_LAI_p_value'] < 0.05)
                )

                lai_green_pct = lai_green / n_total * 100
                lai_brown_pct = lai_brown / n_total * 100

                ####################################
                # SPEI Drying Percentage
                ####################################

                spei_dry_pct = []
                spei_wet_pct = []

                for scale in scale_list:
                    dry = np.sum(
                        (df_i[f' {season}_SPEI{int(scale)}_trend'] < 0) &
                        (df_i[f' {season}_SPEI{int(scale)}_p_value'] < 0.05)
                    )

                    wet = np.sum(
                        (df_i[f' {season}_SPEI{int(scale)}_trend'] > 0) &
                        (df_i[f' {season}_SPEI{int(scale)}_p_value'] < 0.05)
                    )

                    spei_dry_pct.append(dry / n_total * 100)
                    spei_wet_pct.append(wet / n_total * 100)
                ####################################
                # Plot
                ####################################

                plt.figure(figsize=(6, 4))

                # ------------------------
                # LAI
                # ------------------------

                plt.bar(
                    0,
                    lai_green_pct,
                    color='forestgreen',
                    edgecolor='k',
                    width=0.6,
                    label='Greening'
                )

                plt.bar(
                    0,
                    -lai_brown_pct,
                    color='firebrick',
                    edgecolor='k',
                    width=0.6,
                    label='Browning'
                )

                # ------------------------
                # SPEI
                # ------------------------

                x = np.arange(2, 2 + len(scale_list))

                plt.bar(
                    x,
                    spei_wet_pct,
                    color='#4F9DDE',
                    edgecolor='k',
                    width=0.6,
                    label='Wetting'
                )

                plt.bar(
                    x,
                    -np.array(spei_dry_pct),
                    color='#F4A582',
                    edgecolor='k',
                    width=0.6,
                    label='Drying'
                )

                # ------------------------
                # Format
                # ------------------------

                plt.axhline(0, color='k', linewidth=1)

                plt.xticks(
                    [0] + list(x),
                    ['LAI'] + [str(i) for i in scale_list]
                )

                plt.ylabel('Percentage (%)')

                plt.xlabel('SPEI timescale (months)')

                plt.ylim(-100, 25)

                plt.title(f'{eco} ({season})')

                plt.legend(frameon=False, ncol=2)

                plt.tight_layout()

                plt.show()


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

    pass
class PLOT_SPEI():
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        self.plot_time_series_SPEI()
        # self.plot_time_series_SNU_record()
        pass

    def plot_time_series_SPEI(self):
        dff = result_root + rf'\SPEI_Greening\Dataframe\Dataframe.df'
        df = T.load_df(dff)

        df = self.df_clean(df)
        scale=3

        year_list = list(range(2003, 2025))
        result_dic = {}
        eco_region_list = df['Ecoregion_level_II'].dropna().unique().tolist()
        eco_region_list.append('Western US')

        eco_region_list = ['Western US', 'Western Cordillera', 'Upper Gila Mountains',
                           'Warm Desert', 'Cold Desert', 'Western Sierra Madre Piedmont']

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
            for season in [f'spring_SPEI{scale}', f'summer_SPEI{scale}']:
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
            plt.figure(figsize=(self.map_width * 1.5, self.map_height))

            spring_vals = df_new[f'{eco}_spring_SPEI{scale}']
            summer_vals = df_new[f'{eco}_summer_SPEI{scale}']
            spring_std = df_new[f'{eco}_spring_SPEI{scale}_std']
            summer_std = df_new[f'{eco}_summer_SPEI{scale}_std']

            vals_len = df_new[f'{eco}_len'][0]

            slope_s, _, _, p_s, _ = stats.linregress(year_list, spring_vals)
            slope_sum, _, _, p_sum, _ = stats.linregress(year_list, summer_vals)
            color_spring = '#EA6E88'

            plt.plot(
                year_list,
                spring_vals,
                color=color_spring,
                lw=2,
                label='Spring'
            )

            slope, intercept, r, p, _ = stats.linregress(year_list, spring_vals)
            years = np.array(year_list)

            plt.plot(
                year_list,
                slope * years + intercept,
                '--',
                color=color_spring,
                lw=2,
            )

            # -----------------------------
            # Summer
            # -----------------------------
            color_summer = '#48526F'

            plt.plot(
                year_list,
                summer_vals,
                color=color_summer,
                lw=2,
                label='Summer'
            )

            slope, intercept, r, p, _ = stats.linregress(year_list, summer_vals)

            plt.plot(
                year_list,
                slope * years + intercept,
                '--',
                color=color_summer,
                lw=2,

            )
            plt.fill_between(
                years,
                spring_vals - spring_std,
                spring_vals + spring_std,
                color=color_spring,
                alpha=0.2
            )

            plt.fill_between(
                years,
                summer_vals - summer_std,
                summer_vals + summer_std,
                color=color_summer,
                alpha=0.2
            )
            plt.legend()

            stats_text = (
                f'Spring: slope={slope_s:.2f}, p={p_s:.2f}\n'
                f'Summer: slope={slope_sum:.2f}, p={p_sum:.2f}'
            )

            plt.text(0.95, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     )

            plt.ylabel(f'SPEI{scale}', fontsize=12)

            plt.title(f'{eco}_n={vals_len}', fontsize=12)

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

class PLOT_WUE():
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        self.plot_time_series_GPP()

        pass

    def plot_time_series_WUE(self):
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


            mean_dic = {}
            std_dic = {}

            for year in year_list:
                df_ii = df_i[df_i['year'] == year]
                ## scheme1
                vals = np.array(df_ii['WUE_spring'].tolist(), dtype=float)

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

            result_dic[f'{eco}'] = mean_dic
            result_dic[f'{eco}_std'] = std_dic

            # 只存一次长度
            result_dic[f'{eco}_len'] = len(df_i)

            # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()

        # T.print_head_n(df_new);exit()

        flag = 0
        fig, ax = plt.subplots(3, 2, figsize=(10,6))
        ax = ax.flatten()

        for eco in eco_region_list:
            axes=ax[flag]



            vals = df_new[f'{eco}']
            std_vals = df_new[f'{eco}_std']

            vals_len = df_new[f'{eco}_len'][0]


            axes.plot(year_list, vals,   linewidth=2,color='blue', )

            # plt.fill_between(year_list,
            #                     vals - std_vals,
            #                     vals + std_vals,
            #
            #                  alpha=0.2)


            slope_s, _, _, p_s, _ = stats.linregress(year_list, vals)
            ## add trend line
            axes.plot(year_list, slope_s * np.array(year_list) + (vals[0] - slope_s * year_list[0]),
                     linestyle='--', color='blue', )


            stats_text = (
                f'WUE: slope={slope_s:.2f}, p={p_s:.2f}\n'

            )

            axes.text(0.95, 0.95, stats_text,
                     transform=axes.transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            axes.set_ylabel('LAI/precip,spring', fontsize=12)

            axes.set_title(f'{eco}_n={vals_len}', fontsize=12)



            axes.grid(True, axis='x')
            flag+=1




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
class PLOT_GPP:
    pass


    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        self.plot_time_series_GPP()

        pass

    def plot_time_series_GPP(self):
        dff=result_root + rf'\SPEI_Greening\Dataframe\Dataframe.df'
        df=T.load_df(dff)

        df=self.df_clean(df)


        year_list=list(range(2003, 2024))
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


            mean_dic = {}
            std_dic = {}

            for year in year_list:
                df_ii = df_i[df_i['year'] == year]
                ## scheme1
                vals = np.array(df_ii['summer_NEE'].tolist(), dtype=float)
                # vals=np.array(df_ii['summer_GPP_CFE-Hybrid'].tolist(), dtype=float)

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

            result_dic[f'{eco}'] = mean_dic
            result_dic[f'{eco}_std'] = std_dic

            # 只存一次长度
            result_dic[f'{eco}_len'] = len(df_i)

            # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()

        # T.print_head_n(df_new);exit()

        flag = 0
        fig, ax = plt.subplots(3, 2, figsize=(10,6))
        ax = ax.flatten()

        for eco in eco_region_list:
            axes=ax[flag]



            vals = df_new[f'{eco}']
            std_vals = df_new[f'{eco}_std']

            vals_len = df_new[f'{eco}_len'][0]


            axes.plot(year_list, vals,   linewidth=2,color='blue', )

            # plt.fill_between(year_list,
            #                     vals - std_vals,
            #                     vals + std_vals,
            #
            #                  alpha=0.2)



            slope_s, intercept_s, _, p_s, _ = stats.linregress(year_list, vals)
            ## add trend line
            trend_line = slope_s * np.array(year_list) + intercept_s
            axes.plot(year_list, trend_line, linestyle='--', color='red')


            stats_text = (
                f'NEE: slope={slope_s:.2f}, p={p_s:.2f}\n'

            )

            axes.text(0.95, 0.95, stats_text,
                     transform=axes.transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            axes.set_ylabel('GPP(gc/m2/month)_spring', fontsize=12)

            axes.set_title(f'{eco}_n={vals_len}', fontsize=12)



            axes.grid(True, axis='x')
            flag+=1




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

class PLOT_SNU_LAI:
    pass


    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        self.plot_time_series_GPP()

        pass

    def plot_time_series_GPP(self):
        dff=result_root + rf'\SPEI_Greening\Dataframe\Dataframe.df'
        df=T.load_df(dff)

        df=self.df_clean(df)


        year_list=list(range(2003, 2025))
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


            mean_dic = {}
            std_dic = {}

            for year in year_list:
                df_ii = df_i[df_i['year'] == year]
                ## scheme1
                vals = np.array(df_ii['summer_NEE'].tolist(), dtype=float)
                # vals=np.array(df_ii['summer_GPP_CFE-Hybrid'].tolist(), dtype=float)

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

            result_dic[f'{eco}'] = mean_dic
            result_dic[f'{eco}_std'] = std_dic

            # 只存一次长度
            result_dic[f'{eco}_len'] = len(df_i)

            # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()

        # T.print_head_n(df_new);exit()

        flag = 0
        fig, ax = plt.subplots(3, 2, figsize=(10,6))
        ax = ax.flatten()

        for eco in eco_region_list:
            axes=ax[flag]



            vals = df_new[f'{eco}']
            std_vals = df_new[f'{eco}_std']

            vals_len = df_new[f'{eco}_len'][0]


            axes.plot(year_list, vals,   linewidth=2,color='blue', )

            # plt.fill_between(year_list,
            #                     vals - std_vals,
            #                     vals + std_vals,
            #
            #                  alpha=0.2)


            slope_s, _, _, p_s, _ = stats.linregress(year_list, vals)
            ## add trend line
            axes.plot(year_list, slope_s * np.array(year_list) + (vals[0] - slope_s * year_list[0]),
                     linestyle='--', color='blue', )


            stats_text = (
                f'LAI: slope={slope_s:.2f}, p={p_s:.2f}\n'

            )

            axes.text(0.95, 0.95, stats_text,
                     transform=axes.transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            axes.set_ylabel('LAI(m2/m2)_summer', fontsize=12)

            axes.set_title(f'{eco}_n={vals_len}', fontsize=12)



            axes.grid(True, axis='x')
            flag+=1




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

class PLOT_Daymet:
    def __init__(self):
        pass
    def run(self):
        self.plot_barplot()
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


    def plot_barplot(self):
        dff = result_root + rf'\Daymet\Dataframe\Dataframe.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        variable_list = [
            'intensity',
            'maximum_dryspell',
            'fq',
            'amount',

        ]

        variable_list = [
            'vpd',


        ]

        # eco_region_list = ['Western US', 'Western Cordillera', 'Upper Gila Mountains',
        #                    'Warm Desert', 'Cold Desert', 'Western Sierra Madre Piedmont']

        eco_region_list = [1,2,3]

        df_plot = pd.DataFrame()
        season='summer'

        for eco in eco_region_list:


            df_i = df[df['summer_class_3'] == eco]

            for variable in variable_list:
                # col = f'{season}_rainfall_{variable}_trend'
                col = f'{variable}_{season}_npy_trend'

                tmp = pd.DataFrame({
                    'Region': eco,
                    'Variable': variable,
                    'Value': df_i[col]
                })

                df_plot = pd.concat([df_plot, tmp], ignore_index=True)

                # ---------- MODIS LAI ----------
                # tmp = pd.DataFrame({
                #     'Region': eco,
                #     'Variable': 'LAI',
                #     'Value': df_i[f'{season}_LAI_trend']
                # })
                #
                # df_plot = pd.concat([df_plot, tmp], ignore_index=True)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

        for ax, eco in zip(axes.flatten(), eco_region_list):
            sns.barplot(
                data=df_plot[df_plot['Region'] == eco],
                x='Variable',
                y='Value',
                palette='Set2',
                errorbar='se',
                ax=ax
            )

            ax.set_title(eco, fontsize=12)

            ax.set_xlabel('')
            ax.set_ylabel('Trend')
            ax.set_ylim(-0.02,0.02)

            ax.axhline(0, color='black', linewidth=0.8)

            ax.tick_params(axis='x', rotation=35)

        # 如果只有左边显示y轴
        for ax in axes[:, 1:].flatten():
            ax.set_ylabel('')

        plt.tight_layout()
        plt.show()






        pass

class PLOT_heatmap:
    def __init__(self):
        pass


    def run(self):
        self.heatmap()
        # self.count_map()
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

    def heatmap(self):
        dff=result_root+rf'\Dataframe\Trend_analysis\\Trend_analysis.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        for col in df.columns:
            print(col)


        x_var = 'summer_SPEI06_trend'
        y_var = 'summer_vpd_anomaly_trend'
        z_var = 'summer_LAI_anomaly_trend'

        # ==============================
        # 1. 提取数据
        # ==============================
        df_i = df[[x_var, y_var, z_var]].copy()
        df_i = df_i.replace([np.inf, -np.inf], np.nan).dropna()

        # 可选：去掉极端值，避免少数 outliers 拉伸坐标轴
        # for var in [x_var, y_var]:
        #     q_low = df_i[var].quantile(0.01)
        #     q_high = df_i[var].quantile(0.99)
        #     df_i = df_i[
        #         (df_i[var] >= q_low) &
        #         (df_i[var] <= q_high)
        #         ]
        SPEI=df_i[x_var].tolist()
        plt.hist(SPEI)
        plt.show()
        winter_ppt=df_i[y_var].tolist()
        plt.hist(winter_ppt)
        plt.show()

        # ==============================
        # 2. 设置 bin
        # ==============================
        n_bins = 10

        x_bins = np.linspace(
            -0.06,
            0.02,
            n_bins + 1
        )

        y_bins = np.linspace(
            -.4,
            .4,
            n_bins + 1
        )

        # ==============================
        # 3. 分箱
        # ==============================
        df_i['x_bin'] = pd.cut(
            df_i[x_var],
            bins=x_bins,
            labels=False,
            include_lowest=True
        )

        df_i['y_bin'] = pd.cut(
            df_i[y_var],
            bins=y_bins,
            labels=False,
            include_lowest=True
        )

        # ==============================
        # 4. 每个 bin 计算 mean LAI trend
        # ==============================
        heatmap = df_i.groupby(
            ['y_bin', 'x_bin'],
            observed=False
        )[z_var].mean().unstack()

        # 保证所有 bins 都存在
        heatmap = heatmap.reindex(
            index=range(n_bins),
            columns=range(n_bins)
        )

        count_map = df_i.groupby(
            ['y_bin', 'x_bin'],
            observed=False
        )[z_var].count().unstack()

        count_map = count_map.reindex(
            index=range(n_bins),
            columns=range(n_bins)
        ).fillna(0)

        # ==============================
        # 5. bin 中心
        # ==============================
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2

        # ==============================
        # 6. Plot
        # ==============================
        fig, ax = plt.subplots(figsize=(7, 6))

        # 让 0 两边颜色对称
        vmax = np.nanpercentile(
            np.abs(heatmap.values),
            95
        )

        im = ax.imshow(
            heatmap.values,
            origin='lower',
            aspect='auto',
            cmap='RdBu',
            vmin=-vmax,
            vmax=vmax,
            extent=[
                x_bins[0],
                x_bins[-1],
                y_bins[0],
                y_bins[-1]
            ]
        )

        # ==============================
        # 添加圆圈
        # circle size = pixel count
        # ==============================
        max_count = np.nanmax(count_map.values)

        for i in range(n_bins):
            for j in range(n_bins):

                count = count_map.iloc[i, j]

                if count == 0:
                    continue

                # 根据 pixel number 调整圆圈大小
                size = 300 * count / max_count

                ax.scatter(
                    x_centers[j],
                    y_centers[i],
                    s=size,
                    facecolors='none',
                    edgecolors='black',
                    linewidth=0.7
                )

        # ==============================
        # Reference lines
        # ==============================
        ax.axvline(
            0,
            color='black',
            linestyle='--',
            linewidth=1
        )

        ax.axhline(
            0,
            color='black',
            linestyle='--',
            linewidth=1
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Growing season LAI trend')

        ax.set_xlabel('SPEI trend')
        ax.set_ylabel('Rainfall frequency trend')

        plt.tight_layout()
        plt.show()

    def count_map(self):
        dff = result_root + rf'\Dataframe\Trend_analysis\\Trend_analysis.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        for col in df.columns:
            print(col)

        x_var = 'summer_SPEI06_trend'
        y_var = 'growing_season_rainfall_fq_5mm_anomaly_trend'
        z_var = 'summer_LAI_anomaly_trend'

        # ==============================
        # 1. 提取数据
        # ==============================
        df_i = df[[x_var, y_var, z_var]].copy()
        df_i = df_i.replace([np.inf, -np.inf], np.nan).dropna()

        # 可选：去掉极端值，避免少数 outliers 拉伸坐标轴
        # for var in [x_var, y_var]:
        #     q_low = df_i[var].quantile(0.01)
        #     q_high = df_i[var].quantile(0.99)
        #     df_i = df_i[
        #         (df_i[var] >= q_low) &
        #         (df_i[var] <= q_high)
        #         ]
        SPEI = df_i[x_var].tolist()
        plt.hist(SPEI)
        plt.show()
        winter_ppt = df_i[y_var].tolist()
        plt.hist(winter_ppt)
        plt.show()

        # ==============================
        # 2. 设置 bin
        # ==============================
        n_bins = 10

        x_bins = np.linspace(
            -0.06,
            0.02,
            n_bins + 1
        )

        y_bins = np.linspace(
            -.4,
            .4,
            n_bins + 1
        )

        # ==============================
        # 3. 分箱
        # ==============================
        df_i['x_bin'] = pd.cut(
            df_i[x_var],
            bins=x_bins,
            labels=False,
            include_lowest=True
        )

        df_i['y_bin'] = pd.cut(
            df_i[y_var],
            bins=y_bins,
            labels=False,
            include_lowest=True
        )
        heatmap = df_i.groupby(
            ['y_bin', 'x_bin'],
            observed=False
        )[z_var].mean().unstack()

        heatmap = heatmap.reindex(
            index=range(n_bins),
            columns=range(n_bins)
        )

        # ==============================
        # 4.1 每个 bin 的 pixel number
        # ==============================
        count_map = df_i.groupby(
            ['y_bin', 'x_bin'],
            observed=False
        )[z_var].count().unstack()

        count_map = count_map.reindex(
            index=range(n_bins),
            columns=range(n_bins)
        ).fillna(0)

        print(count_map)

        fig, ax = plt.subplots(figsize=(7, 6))

        im = ax.imshow(
            count_map.values,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            extent=[
                x_bins[0],
                x_bins[-1],
                y_bins[0],
                y_bins[-1]
            ]
        )

        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pixel number')

        ax.set_xlabel('SPEI trend')
        ax.set_ylabel('Rainfall frequency trend')

        plt.tight_layout()
        plt.show()


def main():
    # SPEI_Greening_categorize().run()
    # SPEI_Greening_ecoregion().run()
    # PLOT_vegetation_change().run()
    # PLOT_bar().run()
    # PLOT_Daymet().run()
    # PLOT_SPEI().run()
    # PLOT_WUE().run()
    # PLOT_GPP().run()
    PLOT_heatmap().run()
    # PLOT_SNU_LAI().run()

    pass




if __name__ == '__main__':
    main()