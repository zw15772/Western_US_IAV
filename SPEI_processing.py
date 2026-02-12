import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
D=DIC_and_TIF(tif_template=tif_template)

class Processing_data:
    def run(self):
        # self.nc_to_tif_time_series_fast2()
        # self.extract_tif_from_shp()
        # self.differences_P_PET()
        # self.resample()
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
    def resample(self):

        fdir =data_root+ rf'Terraclimate\PET\extract_tif\\'
        outdir =data_root+ rf'Terraclimate\PET\\resample\\'
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = fdir + f
            outf = outdir + f
            dataset = gdal.Open(fpath)

            try:
                gdal.Warp(outf, dataset, xRes=0.05, yRes=0.05, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass
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

        fdir_all = data_root + rf'Terraclimate\PET\resample\\'
        outdir=data_root + rf'Terraclimate\PET\dic\\'
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
        # self.calculate_SPEI()
        # self.compute_spei_NOAA()
        # self.extract_growing_season_monthly()
        # self.extract_growing_season_LAI_mean()
        self.trend_analysis()
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
        outdir = data_root + r'Terraclimate\SPEI\SPEI_12_NOAA\\dic\\'
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


    def extract_growing_season_monthly(self):
        fdir = data_root+rf'\Terraclimate\SPEI\SPEI_12_NOAA\\dic\\'

        outdir =data_root + r'Terraclimate\SPEI\SPEI_12_NOAA\extract_growing_season_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = data_root+rf'\basedata\4GST\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic = {}
        # for pix in phenology_dic:
        #     # print(phenology_dic[pix]);exit()
        #     val = phenology_dic[pix]['Onsets']
        #     try:
        #         val = float(val)
        #     except:
        #         continue
        #
        #     new_spatial_dic[pix] = val
        # spatial_array = D.pix_dic_to_spatial_arr(new_spatial_dic)
        # plt.imshow(spatial_array, interpolation='nearest', cmap='jet')
        # plt.show()
        # exit()
        spatial_dict_gs_count = {}

        for f in T.listdir(fdir):

            outf = outdir + f
            #
            # if os.path.isfile(outf):
            #     continue
            # print(outf)
            spatial_dict = dict(np.load(fdir + f, allow_pickle=True, encoding='latin1').item())
            dic_DOY = {15: 1,
                       30: 1,
                       45: 2,
                       60: 2,
                       75: 3,
                       90: 3,
                       105: 4,
                       120: 4,
                       135: 5,
                       150: 5,
                       165: 6,
                       180: 6,
                       195: 7,
                       210: 7,
                       225: 8,
                       240: 8,
                       255: 9,
                       270: 9,
                       285: 10,
                       300: 10,
                       315: 11,
                       330: 11,
                       345: 12,
                       360: 12,
                       }

            result_dic = {}

            for pix in tqdm(spatial_dict):
                if not pix in phenology_dic:
                    continue
                # print(pix)

                r, c = pix

                SeasType = phenology_dic[pix]['SeasType']
                if SeasType == 2:

                    SOS = phenology_dic[pix]['Onsets']
                    try:
                        SOS = float(SOS)

                    except:
                        continue

                    SOS = int(SOS)
                    SOS_monthly = dic_DOY[SOS]

                    EOS = phenology_dic[pix]['Offsets']
                    EOS = int(EOS)
                    EOS_monthly = dic_DOY[EOS]
                    # print(SOS_monthly,EOS_monthly)
                    # print(SOS,EOS)

                    time_series = spatial_dict[pix]

                    time_series = np.array(time_series)
                    if SOS_monthly > EOS_monthly:  ## south hemisphere
                        time_series_flatten = time_series.flatten()
                        time_series_reshape = time_series_flatten.reshape(-1, 12)
                        time_series_dict = {}
                        for y in range(len(time_series_reshape)):
                            if y + 1 == len(time_series_reshape):
                                break

                            time_series_dict[y] = np.concatenate(
                                (time_series_reshape[y][SOS_monthly - 1:], time_series_reshape[y + 1][:EOS_monthly]))

                    else:
                        time_series_flatten = time_series.flatten()
                        time_series_reshape = time_series_flatten.reshape(-1, 12)
                        time_series_dict = {}
                        for y in range(len(time_series_reshape)):
                            time_series_dict[y] = time_series_reshape[y][SOS_monthly - 1:EOS_monthly]
                    time_series_gs = []
                    for y in range(len(time_series_dict)):
                        time_series_gs.append(time_series_dict[y])
                    time_series_gs = np.array(time_series_gs)

                elif SeasType == 3:
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_gs = np.reshape(time_series, (-1, 12))

                elif SeasType == 1:
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_gs = np.reshape(time_series, (-1, 12))


                else:
                    SeasClss = phenology_dic[pix]['SeasClss']
                    print(SeasType, SeasClss)
                    continue
                spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                result_dic[pix] = time_series_gs
            # print(spatial_dict_gs_count)
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
            # # arr[arr<6] = np.nan
            # plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=12)
            # plt.colorbar()
            # plt.show()
            np.save(outf, result_dic)


    def extract_growing_season_LAI_mean(self):  ## extract LAI average
        fdir = data_root+r'Terraclimate\SPEI\SPEI_12_NOAA\extract_growing_season_monthly'

        outdir = data_root+r'\Terraclimate\SPEI\SPEI_12_NOAA\extract_growing_season_SPEI12_mean\\'


        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            ### annual year

            vals_growing_season = spatial_dic[pix][24:]
            print(vals_growing_season.shape[0])
            # plt.imshow(vals_growing_season)
            # plt.colorbar()
            # plt.show()
            growing_season_mean_list = []

            for val in vals_growing_season:
                if T.is_all_nan(val):
                    continue
                val = np.array(val)

                sum_growing_season = np.nanmean(val)

                growing_season_mean_list.append(sum_growing_season)

            result_dic[pix] = {
                'growing_season': growing_season_mean_list,
            }

        outf = outdir + 'growing_season_SPEI12_mean.npy'

        np.save(outf, result_dic)

    def trend_analysis(self):

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        ##each window average trend
        phenology_mask_f = data_root + rf'basedata\Phenology_extraction\SeasType.tif'
        phenology_mask_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(phenology_mask_f)
        phenology_dic = D.spatial_arr_to_dic(phenology_mask_arr)



        fdir = data_root + r'\Terraclimate\SPEI\SPEI_12_NOAA\extract_growing_season_SPEI12_mean\\'
        outdir = result_root + r'Terraclimate\SPEI\SPEI_12_NOAA\trend\\'
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

                time_series = dic[pix]['growing_season']
                # print(time_series)
                time_series = np.array(time_series)
                print(len(time_series))

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
            fpath=data_root + rf'basedata\Phenology_extraction\SeasType.tif'
            ll,lr,ul,ur=RasterIO_Func().get_tif_bounds(fpath)
            print(ll,lr,ul,ur)

            ax = plt.axes(projection=ccrs.PlateCarree())

            # --- 画趋势图 ---
            im = ax.imshow(
                arr_trend,
                cmap='RdBu',
                vmin=-0.05,
                vmax=0.05,
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
            cbar.set_label('Trend')

            plt.title(f)
            plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

class PLOT_SPEI:
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        self.plot_SPEI()

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
    def plot_SPEI(self):  ##### plot for 4 clusters

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
                         'SPEI12_mean',
                         ]
        dic_label={'SPEI12_mean':'SPEI12_mean',}
        year_list=range(1982,2024)

        mean_dic = {}
        std_dic = {}
        result_dic={}

        for var in variable_list:
            mean_dic = {}
            for year in year_list:
                df_i = df[df['year'] == year]
                ## scheme1
                vals = np.array(df_i[f'{var}'].tolist(), dtype=float)
                weight = np.array(df_i['area_weight'].tolist(), dtype=float)
                weighted_mean = (
                        np.nansum(vals * weight)
                        / np.nansum(weight * np.isfinite(vals))
                )

                # 加权方差
                weighted_var = np.nansum(weight * (vals - weighted_mean) ** 2) / np.nansum(weight)

                weighted_std = np.sqrt(weighted_var)

                mean_dic[year] = weighted_mean
                std_dic[year] = weighted_std
                print(weighted_std)


                # print(var, year, weighted_mean_values)
                ## scheme2
                # vals = np.array(df_i[f'{var}_relative_change'].tolist(), dtype=float)
                # weighted_mean_values = np.nanmean(vals)



            result_dic[var] = mean_dic
            result_dic[f'{var}_std'] = std_dic

        # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()



        # T.print_head_n(df_new);exit()

        flag=0
        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            mean_vals = df_new[var]
            std_vals = df_new[f'{var}_std']

            plt.plot(year_list, mean_vals,
                     label=dic_label[var],
                     linewidth=linewidth_list[flag],
                     color=color_list[flag])

            plt.fill_between(year_list,
                             mean_vals - std_vals,
                             mean_vals + std_vals,
                             color=color_list[flag],
                             alpha=0.2)

            slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, mean_vals)
            print(var, f'{slope:.4f}', f'{p_value:.4f}')

            flag += 1
        plt.ylabel('SPEI')

        plt.grid(True, axis='x')   # 只画竖线（随 x 刻度）

        plt.legend()
        plt.show()
        # out_pdf_fdir = result_root + rf'\Figure\\weighted_area\\Figure1a\\'
        # T.mk_dir(out_pdf_fdir, force=True)
        # plt.savefig(out_pdf_fdir + 'time_series_relative_change_mean.pdf', dpi=300, bbox_inches='tight')
        # plt.close()




def main():

    # Processing_data().run()
    # SPEI_calculation().run()
    PLOT_SPEI().run()


    pass

if __name__ == '__main__':
    main()

