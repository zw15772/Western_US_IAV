from cmath import isnan

import matplotlib.pyplot as plt
import numpy as np

from SPEI_processing import SPEI_calculation
from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
D=DIC_and_TIF(tif_template=tif_template)


class coupling_anaysis:
    def __init__(self):
        pass
    def run(self):
        # self.calculating_correlation()
        # self.maxmum_correlation()
        self.heatmap()
    def calculating_correlation(self):
        ## here calculating correlation between SPEI and NDVI
        from scipy.stats import pearsonr
        import numpy as np


        MODIS_LAI_dir=data_root+rf'\MODIS_LAI\spring_summer_season_LAI_mean\\'
        SPEI_dir=data_root+rf'\Terraclimate\SPEI\\'
        MODIS_dic = T.load_npy_dir(MODIS_LAI_dir)
        scale_list=np.arange(3,49,3)
        for scale in scale_list:
            scale =int(scale)
            fdir=SPEI_dir+rf'\SPEI_{scale}_NOAA\spring_summer_season_SPEI_mean\\'
            SPEI_dic=T.load_npy_dir(fdir)

            r_dic = {}
            p_dic = {}
            outdir=result_root+rf'\coupling_anaysis\\summer\\'
            T.mk_dir(outdir, force=True)
            outf=outdir+rf'{scale}'

            for pix in tqdm(SPEI_dic.keys()):
                if pix not in MODIS_dic:
                    continue
                vals_SPEI=SPEI_dic[pix]['summer']
                vals_MODIS=MODIS_dic[pix]['summer']
                if len(vals_SPEI)!=22:
                    continue
                if len(vals_MODIS)!=22:
                    continue
                if len(vals_SPEI)!=len(vals_MODIS):
                    continue
                # print(vals_SPEI)
                # print(vals_MODIS)
                # print(np.isnan(np.nanmean(vals_SPEI)))
                if np.isnan(np.nanmean(vals_SPEI)):
                    continue
                if np.isnan(np.nanmean(vals_MODIS)):
                    continue
                vals_SPEI = np.asarray(vals_SPEI, dtype=float)
                vals_MODIS = np.asarray(vals_MODIS, dtype=float)

                mask = (~np.isnan(vals_SPEI)) & (~np.isnan(vals_MODIS))
                if mask.sum() >= 10:
                    r, p = pearsonr(vals_SPEI[mask], vals_MODIS[mask])
                else:
                    r, p = np.nan, np.nan



                r_dic[pix] = r
                p_dic[pix] = p

            T.save_npy(r_dic, outf + '_r.npy')
            T.save_npy(p_dic, outf + '_p.npy')

            D.pix_dic_to_tif(r_dic, outf + '_r.tif')
            D.pix_dic_to_tif(p_dic, outf + '_p.tif')

    def maxmum_correlation(self):
        ##
        fdir=result_root+rf'\coupling_anaysis\\spring\\'
        outdir=result_root+rf'\coupling_anaysis\\maximum_corr\\'
        T.mk_dir(outdir, force=True)
        array_list = []


        for f in sorted(T.listdir(fdir)):
            if not f.endswith('_r.tif'):
                continue



            array, originX, originY, pixelWidth, pixelHeight = \
                ToRaster().raster2array(join(fdir, f))

            array = np.array(array, dtype=float)

            array[array < -9999] = np.nan

            array_list.append(array)


        stack = np.stack(array_list, axis=0)  # (n_scale, row, col)

        # scale 对应关系
        scale_list = np.arange(3, 49, 3)  # [3,6,9,...,48]

        # 全是 NaN 的像素
        all_nan = np.all(np.isnan(stack), axis=0)


        # stack_abs = np.abs(stack)
        # stack[stack < 0] = np.nan
        stack[np.isnan(stack)] = -np.inf


        idx = np.nanargmax(stack, axis=0)


        # 最大绝对值对应的原始 r（保留正负号）
        max_r = np.take_along_axis(stack, idx[np.newaxis, :, :], axis=0)[0]

        # 对应的真正 scale
        optimal_scale = scale_list[idx]

        # 全 NaN 的像素恢复为 NaN
        max_r[all_nan] = np.nan
        optimal_scale = optimal_scale.astype(float)
        optimal_scale[all_nan] = np.nan
        ##

        D.arr_to_tif(max_r, outdir+'max_r_spring.tif')
        D.arr_to_tif(optimal_scale, outdir+'optimal_scale_spring.tif')

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

        dff=result_root+rf'\coupling_anaysis\Dataframe\\Dataframe.df'
        df=T.load_df(dff)
        print(len(df))
        df=self.df_clean(df)
        # print(len(df));exit()
        eco_region_list = df['Ecoregion_level_II'].dropna().unique().tolist()
        eco_region_list.append('Western US')




        # -----------------------------------
        # Region list
        # -----------------------------------
        eco_region_list = [
            'Western US',
            'Western Cordillera',
            'Upper Gila Mountains',
            'Warm Desert',
            'Cold Desert',
            'Western Sierra Madre Piedmont'
        ]

        scales = list(range(3, 49, 3))

        heatmap_df = pd.DataFrame(index=eco_region_list,
                                  columns=scales)

        for eco in eco_region_list:

            if eco == 'Western US':
                df_i =df.copy()

            else:
                df_i = df[df['Ecoregion_level_II'] == eco]

            for s in scales:
                r_col = f'{s}_r_spring'
                p_col = f'{s}_p_spring'
                # vals = df_i.loc[df_i[p_col] < 0.05, r_col]
                #
                # heatmap_df.loc[eco, s] = vals.mean()



                heatmap_df.loc[eco, s] = df_i[r_col].mean()

        heatmap_df = heatmap_df.astype(float)

        plt.figure(figsize=(11, 4.5))

        sns.heatmap(
            heatmap_df,
            cmap='RdBu',
            center=0,
            square=True,
            annot=True,
            fmt=".2f",
            linewidths=0.3,
            vmin=-0.5,
            vmax=0.5,
            cbar_kws={'label': 'Mean correlation spring(r)'}
        )


        plt.ylabel('')
        plt.tight_layout()
        plt.show()

def main():
    coupling_anaysis().run()


if __name__ == '__main__':
    main()









