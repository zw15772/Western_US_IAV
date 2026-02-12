import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= rf'D:\Western_US_IAV\Data\Terraclimate\PET\extract_tif/195802.tif'
D=DIC_and_TIF(tif_template=tif_template)

class Processing_data:
    def run(self):
        # self.nc_to_tif_time_series_fast2()
        # self.extract_tif_from_shp()
        # self.differences_P_PET()
        self.tif_to_dic()



    def nc_to_tif_time_series_fast2(self):

        fdir=data_root + rf'\Terraclimate\Precip\tif\\'
        outdir=data_root + rf'Terraclimate\Precip\tif\\'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):


            outdir_name = f.split('.')[0].split('_')[-1]

            # exit()

            yearlist = list(range(1982, 2025))
            fpath = join(fdir,f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)


            outf = join(outdir,outdir_name+'.tif')
            array = nc_in['ppt']
            array = np.array(array).T

            array[array < 0] = np.nan
            longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.0416667, -0.0416667
            ToRaster().array2raster(outf, longitude_start, latitude_start,
                                    pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()

    pass

    def extract_tif_from_shp(self):
        shp_f=data_root + rf'basedata\\Western_US_bountry\\merged_western_US.shp'
        fdir= data_root + rf'Terraclimate\PET\tif\\'
        outdir=data_root + rf'Terraclimate\PET\extract_tif\\'
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=outdir+f

            ToRaster().clip_array(fpath, outf,shp_f)

        pass
    def differences_P_PET(self):
        fdir_P = data_root + rf'Terraclimate\Precip\extract_tif\\'
        fdir_PET = data_root + rf'Terraclimate\PET\extract_tif\\'
        outdir = data_root + rf'Terraclimate\P_PET\tif\\'
        T.mk_dir(outdir, force=True)
        for f_P in tqdm(sorted(os.listdir(fdir_P))):
            print(fdir_P+f_P)
            if not f_P.endswith('.tif'):
                continue
            array_P, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir_P + f_P)
            array_P = np.array(array_P, dtype=float)
            array_P[array_P < 0] = np.nan
            # plt.imshow(array_P)
            # plt.show()
            f_PET=fdir_PET+f_P
            print(f_PET)
            array_PET, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_PET)
            array_PET[array_PET < 0] = np.nan
            # plt.imshow(array_PET)
            # plt.show()
            array_PET = np.array(array_PET, dtype=float)
            array_P_PET=array_P-array_PET
            # print(np.shape(array_PET))
            # print(np.shape(array_P))
            # print(np.shape(array_P_PET))
            # exit()
            # plt.imshow(P_PET)
            # plt.show()
            print(outdir + f_P)

            D.arr_to_tif(array_P_PET, outdir + f_P)




        pass

    def tif_to_dic(self):

        fdir_all = data_root + rf'Terraclimate\Precip\extract_tif\\'
        outdir=data_root + rf'Terraclimate\Precip\dic\\'
        T.mk_dir(outdir, force=True)

        year_list = list(range(1958, 2025))
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
            array_unify = array[:3600][:3600,
                          :7200]

            array_unify[array_unify < -999] = np.nan
            # array_unify[array_unify > 10] = np.nan
            # array[array ==0] = np.nan

            array_unify[array_unify < 0] = np.nan  ####

            #
            #
            # plt.imshow(array_unify)
            # plt.show()
            # array_mask = np.array(array_mask, dtype=float)
            # plt.imshow(array_mask)
            # plt.show()

            array_dryland = array_unify
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
                np.save(outdir + rf'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + rf'per_pix_dic_%03d' % 0, temp_dic)


class SPEI_calculation:
    def run(self):
        self.calculate_SPEI()
        # self.compute_spei_NOAA()
        pass

    def calculate_SPEI(self):
        from scipy.stats import lognorm
        from scipy.stats import norm
        import numpy as np
        from scipy.stats import fisk, norm, genlogistic
        import numpy as np


        fdir = data_root + r'Terraclimate\P_PET\dic\\'
        outdir=data_root + r'Terraclimate\SPEI\SPEI_12\\'
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dic = np.load(join(fdir, f), allow_pickle=True).item()

            SPEI_12 = {}


            for pix in tqdm(dic):

                ts = np.array(dic[pix], dtype=float)
                # print(ts);

                # if np.all(np.isnan(ts)):
                #     continue

                # ---- Step 1: 12-month rolling sum ----
                ts12 = np.full_like(ts, np.nan)

                for i in range(11, len(ts)):
                    window = ts[i - 11:i + 1]
                    if np.sum(~np.isnan(window)) == 12:
                        ts12[i] = np.sum(window)

                # ---- Step 2: 只用非NaN值拟合 ----
                mask = np.isfinite(ts12)

                if mask.sum() < 30:
                    continue

                data = ts12[mask]
                # print(data)

                try:
                    # 拟合 log-logistic (用 lognorm 近似)
                    shape, loc, scale = genlogistic.fit(data)
                    cdf = genlogistic.cdf(data, shape, loc=loc, scale=scale)

                    # 概率截断，防止出现 +/- inf
                    cdf = np.clip(cdf, 0.0001, 0.9999)
                    spei_vals = norm.ppf(cdf)

                    # ---- Step 3: 填回完整时间序列 ----
                    result = np.full_like(ts12, np.nan)
                    result[mask] = spei_vals

                    SPEI_12[pix] = result

                    # plt.figure(figsize=(8, 5))
                    # plt.plot(data, spei_vals, '.')
                    # plt.xlabel("D12")
                    # plt.ylabel("SPEI12")
                    # plt.title("Transformation from D12 to SPEI12")
                    # plt.show()



                except:
                    continue

            outf=outdir+f.split('.')[0]+'.npy'
            T.save_npy(SPEI_12,outf)





        pass



    def compute_spei_NOAA(self):
        import numpy as np
        from climate_indices import indices, compute
        # import climate_indices
        # print(climate_indices.__version__);exit()

        print(dir(indices.Distribution))
        from tqdm import tqdm
        import os

        # 假设你的 T 和 data_root 已经定义
        fdir_PET = data_root + r'Terraclimate\PET\dic\\'
        fdir_Precip = data_root + r'Terraclimate\Precip\dic\\'
        outdir = data_root + r'Terraclimate\SPEI\SPEI_12_NOAA\\'
        T.mk_dir(outdir, force=True)


        for f in os.listdir(fdir_PET):
            if not f.endswith('.npy'):
                continue


            dic_PET = np.load(os.path.join(fdir_PET, f), allow_pickle=True).item()
            dic_Precip = np.load(os.path.join(fdir_Precip, f), allow_pickle=True).item()
            SPEI_12 = {}

            for pix in tqdm(dic_PET, desc=f"Processing {f}"):
                # 获取水分盈亏序列 (P - PET)
                ts_PET = np.array(dic_PET[pix], dtype=float)
                ts_Precip = np.array(dic_Precip[pix], dtype=float)

                # print(ts)

                if np.sum(np.isfinite(ts_PET)) < 360:
                    continue
                if np.sum(np.isfinite(ts_Precip))<360:
                    continue


                # ---- 使用 climate_indices 计算 SPEI ----
                try:
                    # indices.spei 函数要求输入：
                    # net_precipitation: P-PET 序列
                    # scale: 尺度的月数 (12)
                    # distribution: 分布类型 (通常使用 'log-logistic')
                    # periodicity: 周期性 ('monthly')
                    # fitting_params: 拟合参数（可选）

                    spei_vals = indices.spei(
                        precips_mm=ts_Precip,

                        pet_mm=ts_PET,
                        scale=12,
                        distribution=indices.Distribution.gamma,
                        periodicity=compute.Periodicity.monthly,
                        data_start_year=1958,  # 根据你的数据源调整
                        calibration_year_initial=1958,
                        calibration_year_final=2024 ) # 建议使用完整长度作为校准期



                    # 该函数返回的是一个掩码数组 (Masked Array)，我们转回普通 numpy 数组
                    SPEI_12[pix] = np.ma.filled(spei_vals, np.nan)

                except Exception as e:
                    print(f"Error at pixel {pix}: {e}")
                    continue

            # 保存结果
            save_path = os.path.join(outdir, f)
            np.save(save_path, SPEI_12)


def main():

    # Processing_data().run()
    SPEI_calculation().run()


    pass

if __name__ == '__main__':
    main()

