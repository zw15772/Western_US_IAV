import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
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
    def run(self):
        self.barplot_by_ecoregion()
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

    def barplot_by_ecoregion(self):
        dff=result_root + rf'\SPEI_Greening\Dataframe\Dataframe.df'
        df=T.load_df(dff)
        print(len(df))
        df=self.df_clean(df)
        print(len(df))
        bar_list=[]
        for ecoregion in df['Ecoregion_level_II'].dropna().unique():
            if ecoregion == np.nan:
                continue
            SNU_LAI_trend = df[df['Ecoregion_level_II'] == ecoregion]['SNU_LAI_trend']
            leng = len(SNU_LAI_trend)
            print(leng)
            if leng == 0:
                continue
            greening_ratio = (SNU_LAI_trend > 0).sum() / leng
            bar_list.append({
                'ecoregion': ecoregion,
                'greening_ratio': greening_ratio
            })


        print(bar_list)
        bar_df = pd.DataFrame(bar_list)
        bar_df = bar_df.sort_values('greening_ratio', ascending=False)

        plt.figure(figsize=(10, 6))

        plt.barh(bar_df['ecoregion'], bar_df['greening_ratio'])

        plt.xlabel('Greening Ratio')
        plt.xlim(0, 1)

        plt.gca().invert_yaxis()  # 最大在上面

        plt.tight_layout()
        plt.show()




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

def main():
    # SPEI_Greening_categorize().run()
    SPEI_Greening_ecoregion().run()
    pass




if __name__ == '__main__':
    main()