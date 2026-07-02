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
        outdir=result_root+rf'\coupling_anaysis\\spring\\'
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

        # 用绝对值寻找最大值的位置
        stack_abs = np.abs(stack)
        stack_abs[np.isnan(stack_abs)] = -np.inf

        idx = np.argmax(stack_abs, axis=0)

        # 最大绝对值对应的原始 r（保留正负号）
        max_r = np.take_along_axis(stack, idx[np.newaxis, :, :], axis=0)[0]

        # 对应的真正 scale
        optimal_scale = scale_list[idx]

        # 全 NaN 的像素恢复为 NaN
        max_r[all_nan] = np.nan
        optimal_scale = optimal_scale.astype(float)
        optimal_scale[all_nan] = np.nan
        ##

        D.arr_to_tif(max_r, outdir+'max_r.tif')
        D.arr_to_tif(optimal_scale, outdir+'optimal_scale.tif')

    def heatmap(self):
        fdir=result_root+rf'\coupling_anaysis\\summer\\'


        pass



def main():
    coupling_anaysis().run()


if __name__ == '__main__':
    main()









