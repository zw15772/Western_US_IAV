from cmath import isnan

import matplotlib.pyplot as plt
import numpy as np
import shap
from lytools import *
from matplotlib.pyplot import axes
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pprint import pprint
import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import zscore
from SPEI_processing import SPEI_calculation
from __Global__ import *
tif_template=  rf'D:\Western_US_IAV\Data\basedata\200902.tif'
D=DIC_and_TIF(tif_template=tif_template)


class coupling_anaysis:
    def __init__(self):
        pass
    def run(self):
        # self.calculating_correlation(
        self.heatmap()
    def calculating_correlation(self):
        ## here calculating correlation between SPEI and NDVI
        from scipy.stats import pearsonr


        SPEI_dir=result_root+rf'\Terraclimate\SPEI\\'

        season='summer'
        MODIS_LAI_fpath = result_root + rf'\detrend\MODIS_LAI\\{season}_LAI_detrend.npy'
        MODIS_dic = T.load_npy(MODIS_LAI_fpath)

        scale_list=np.arange(3,49,3)
        for scale in scale_list:
            scale =int(scale)
            fpath_SPEI=SPEI_dir+rf'{season}_SPEI{scale}.npy'
            SPEI_dic=T.load_npy(fpath_SPEI)

            r_dic = {}
            p_dic = {}
            outdir=result_root+rf'\coupling_anaysis\\{season}\\'
            T.mk_dir(outdir, force=True)
            outf=outdir+rf'{scale}'

            for pix in tqdm(SPEI_dic.keys()):
                if pix not in MODIS_dic:
                    continue
                vals_SPEI=SPEI_dic[pix]
                vals_MODIS=MODIS_dic[pix]
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
                r_col = f'{s}_r_summer'
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
            cbar_kws={'label': 'Mean correlation summer(r)'}
        )


        plt.ylabel('')
        plt.tight_layout()
        plt.show()

class Moving_window_coupling_analysis:
    def __init__(self):


        self.fdirX = result_root + rf'\Moving_window_coupling_analysis\\\moving_window_extraction\\'
        self.fdirY = result_root + rf'\Moving_window_coupling_analysis\\moving_window_extraction\\'


        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor


        pass

    def run(self):
        # self.moving_window_extraction()
        # self.moving_window_average_anaysis()
        # self.calculating_corr_temporal()
        # self.calculating_corr_temporal_function2()
        self.calculating_multiregression_sensitivity()
        # self.calculating_multiregression_sensitivity_block()
        # self.calculate_optimal_scale()
        # self.PLot_window_slices()
        # self.trend_analysis()
        # self.trend_analysis_block()



    def moving_window_extraction(self):

        fdir_all =result_root+ rf'\MODIS_LAI\MODIS_LAI\\'
        outdir = result_root + rf'\Moving_window_coupling_analysis\moving_window_extraction_10year\\'

        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir_all):
            if not 'summer' in f:
                continue

            if not f.endswith('.npy'):
                continue



            outf = outdir + f.split('.')[0] + '.npy'
            # print(outf);exit()


            if os.path.isfile(outf):
                continue
            # if os.path.isfile(outf):
            #     continue

            dic = T.load_npy(fdir_all+f)
            window = 10

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
        new_x_extraction_by_window=[]
        for i in range(len(x)+1):
            if i + window >= len(x)+1:
                continue
            else:
                anomaly = []
                relative_change_list=[]
                x_vals=[]
                for w in range(window):
                    x_val=(x[i + w])
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


    def moving_window_average_anaysis(self): ## each window calculating the average

        fdir = result_root + rf'\Moving_window_coupling_analysis\moving_window_extraction_5year\\'
        outdir = result_root + rf'\Moving_window_coupling_analysis\moving_window_extraction_average\\5year\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'LAI' in f:
                continue
            if 'detrend' in f:
                continue


            dic = T.load_npy(fdir + f)


            outf = outdir + f.split('.')[0] + f'_average.npy'
            print(outf)

            trend_dic = {}


            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                # plt.imshow(time_series_all)
                # plt.show()
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                slides = len(time_series_all)
                for ss in range(slides):


                    ### if all values are identical, then continue



                    time_series = time_series_all[ss]
                    # print(time_series)
                    # if np.nanmax(time_series) == np.nanmin(time_series):
                    #     continue
                    # print(len(time_series))
                    ##average

                    average=np.nanmean(time_series)
                    # print(average)

                    trend_list.append(average)
                print(trend_list)
                # plt.plot(trend_list)
                #
                # plt.show()

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)


    def calculating_corr_temporal(self):
        import numpy as np

        from tqdm import tqdm
        from scipy.stats import pearsonr
        season='summer'
        window_size=10

        # 假设这些是每个像素对应的字典，键是 pix，值是 (year, month)

        scale_list=['SPEI1','SPEI3', 'SPEI6', 'SPEI9',
                    'SPEI12','SPEI24',
                    'SPEI36',
                    'SPEI48',
                    ]
        # 假设这些是每个像素对应的字典，键是 pix，值是 (year, month)
        fdir = result_root + rf'Moving_window_coupling_analysis\moving_window_extraction_{window_size}year\\'

        out_corr = {}
        out_p_value = {}
        for scale in scale_list:
            fLAI = fdir + rf'\\{season}_LAI.npy'
            f_SPEI = fdir + rf'\\{season}_{scale}.npy'

            dic_LAI = T.load_npy(fLAI)
            dic_SPEI = T.load_npy(f_SPEI)
            outdir = result_root + rf'\Moving_window_coupling_analysis\output\\{window_size}year_trend\\'
            T.mk_dir(outdir, force=True)

            outcorr= outdir + rf'partial_corr_{scale}_{season}.npy'
            outpvalue= outdir + rf'partial_pvalue_{scale}_{season}.npy'
            if isfile(outcorr) and isfile(outpvalue):
                continue



            for pix in tqdm(dic_LAI):
                if pix not in dic_SPEI:
                    continue

                vals_LAI = np.array(dic_LAI[pix], dtype=float)
                vals_SPEI = np.array(dic_SPEI[pix], dtype=float)


                # 要求二维 [n_windows, n_years_in_window]
                if vals_LAI.ndim != 2:
                    continue

                n_windows, n_years = vals_LAI.shape

                beta_SPEI_list = []

                p_SPEI_list = []

                for w in range(n_windows):
                    y = vals_LAI[w, :]
                    x1 = vals_SPEI[w, :]

                    # 有效数据检查
                    if len(y) !=window_size or len(x1) !=window_size:

                        beta_SPEI_list.append(np.nan)

                        p_SPEI_list.append(np.nan)


                        continue




                    try:
                        r, p = pearsonr(x1, y)

                        beta_SPEI_list.append(r)
                        p_SPEI_list.append(p)

                    except:
                        beta_SPEI_list.append(np.nan)
                        p_SPEI_list.append(np.nan)


                # plt.plot(p_SPEI_list)
                # plt.show()


                out_corr[pix] = beta_SPEI_list
                out_p_value[pix] = p_SPEI_list

            # === 保存输出 ===


            T.save_npy(out_corr, outdir + rf'partial_corr_{scale}_{season}.npy')
            T.save_npy(out_p_value, outdir + rf'partial_p_{scale}_{season}.npy')

    def calculating_corr_temporal_function2(self):
        import numpy as np

        from tqdm import tqdm

        season = 'summer'


        fdir = result_root + r'\Moving_window_coupling_analysis\moving_window_extraction_5year\\'

        fLAI = fdir + rf'{season}_LAI_detrend.npy'

        file_dic = {
            'ppt': fdir + rf'ppt_winter_npy_detrend.npy',
            'intensity': fdir + rf'{season}_rainfall_intensity.npy',
            'temp': fdir + rf'tmax_{season}_npy_detrend.npy',
            # 'vpd': fdir + rf'vpd_{season}_detrend.npy',
            'rad': fdir + rf'srad_{season}_npy_detrend.npy',
        }

        dic_LAI = T.load_npy(fLAI)

        dic_var = {}

        for var in file_dic:
            dic_var[var] = T.load_npy(file_dic[var])



        outdir = result_root + r'\Moving_window_coupling_analysis\output\5year\\'
        T.mk_dir(outdir, force=True)

        ############################################
        # 保存结果
        ############################################

        var_list = list(file_dic.keys())

        result_corr = {v: {} for v in var_list}
        result_p = {v: {} for v in var_list}

        ############################################
        # 每个pixel
        ############################################

        for pix in tqdm(dic_LAI):

            if any(pix not in dic_var[v] for v in var_list):
                continue

            beta_dic = {v: [] for v in var_list}
            p_dic = {v: [] for v in var_list}

            ########################################
            # moving window
            ########################################
            vals_LAI = np.array(dic_LAI[pix], dtype=float)

            if vals_LAI.ndim != 2:
                continue

            n_windows, n_years = vals_LAI.shape

            skip = False
            var_arrays = {}

            for var in var_list:

                arr = np.array(dic_var[var][pix], dtype=float)

                if arr.ndim != 2:
                    skip = True
                    break

                if arr.shape != vals_LAI.shape:
                    print(f'{pix} {var}: {arr.shape} != {vals_LAI.shape}')
                    skip = True
                    break

                var_arrays[var] = arr

            if skip:
                continue

            for w in range(n_windows):

                data = {
                    'LAI': vals_LAI[w, :]
                }

                for var in var_list:
                    # print(var, np.array(dic_var[var][pix]).shape)
                    data[var] = np.array(dic_var[var][pix], dtype=float)[w, :]

                df_corr = pd.DataFrame(data).dropna()

                ####################################
                # 数据太少
                ####################################

                if len(df_corr) < 6:

                    for var in var_list:
                        beta_dic[var].append(np.nan)
                        p_dic[var].append(np.nan)

                    continue

                ####################################
                # 每个变量计算偏相关
                ####################################

                for var in var_list:

                    covar = [v for v in var_list if v != var]

                    try:

                        res = pg.partial_corr(
                            data=df_corr,
                            x=var,
                            y='LAI',
                            covar=covar,
                            method='pearson'
                        )

                        beta_dic[var].append(res['r'].iloc[0])
                        p_dic[var].append(res['p_val'].iloc[0])

                    except Exception as e:
                        # print('ncc')
                        # print(e)

                        beta_dic[var].append(np.nan)
                        p_dic[var].append(np.nan)

            ########################################
            # 保存pixel结果
            ########################################

            for var in var_list:
                result_corr[var][pix] = beta_dic[var]
                result_p[var][pix] = p_dic[var]
                # print(var)
                # print(len(result_corr[var][pix]))
                # print(len(result_p[var][pix]))

        ############################################
        # 输出
        ############################################

        for var in var_list:
            T.save_npy(
                result_corr[var],
                outdir + rf'partial_corr_{var}_{season}.npy'
            )

            T.save_npy(
                result_p[var],
                outdir + rf'partial_p_{var}_{season}.npy'
            )


    def calculating_multiregression_sensitivity_moving_window(self):

        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        import statsmodels.api as sm

        season = 'summer'

        fdir = result_root + r'\zscore\Moving_window\10year\\'

        fLAI = fdir + rf'{season}_LAI_zscore.npy'

        file_dic = {
            'ppt': fdir + rf'ppt_winter_npy_zscore.npy',
            'intensity': fdir + rf'{season}_rainfall_intensity_zscore.npy',
            'temp': fdir + rf'tmax_{season}_npy_zscore.npy',
            'rad': fdir + rf'srad_{season}_npy_zscore.npy',
            'SPEI3': fdir + rf'{season}_SPEI3.npy',
        }

        dic_LAI = T.load_npy(fLAI)

        dic_var = {}
        for var in file_dic:
            dic_var[var] = T.load_npy(file_dic[var])

        outdir = result_root + r'\Moving_window_coupling_analysis\output\10year_multiregression\\'
        T.mk_dir(outdir, force=True)

        var_list = list(file_dic.keys())

        ############################################
        # output
        ############################################

        result_beta = {v: {} for v in var_list}
        result_p = {v: {} for v in var_list}
        result_r2 = {}

        ############################################

        for pix in tqdm(dic_LAI):

            if any(pix not in dic_var[v] for v in var_list):
                continue

            vals_LAI = np.array(dic_LAI[pix], dtype=float)

            if vals_LAI.ndim != 2:
                continue

            n_windows, n_years = vals_LAI.shape

            skip = False

            for var in var_list:

                arr = np.array(dic_var[var][pix], dtype=float)

                if arr.ndim != 2:
                    skip = True
                    break

                if arr.shape != vals_LAI.shape:
                    skip = True
                    break

            if skip:
                continue

            beta_dic = {v: [] for v in var_list}
            p_dic = {v: [] for v in var_list}
            r2_list = []

            ############################################
            # moving window
            ############################################

            for w in range(n_windows):

                data = {

                    'LAI': vals_LAI[w, :]

                }

                for var in var_list:
                    data[var] = np.array(dic_var[var][pix], dtype=float)[w, :]

                df = pd.DataFrame(data).dropna()

                ############################################
                # enough samples?
                ############################################

                if len(df) < 10:

                    for var in var_list:
                        beta_dic[var].append(np.nan)
                        p_dic[var].append(np.nan)

                    r2_list.append(np.nan)

                    continue

                ############################################

                ############################################
                # regression
                ############################################

                X = df[var_list]

                X = sm.add_constant(X)

                y = df['LAI']

                try:

                    model = sm.OLS(y, X).fit()

                    for var in var_list:
                        beta_dic[var].append(model.params[var])

                        p_dic[var].append(model.pvalues[var])

                    r2_list.append(model.rsquared)

                except:

                    for var in var_list:
                        beta_dic[var].append(np.nan)
                        p_dic[var].append(np.nan)

                    r2_list.append(np.nan)

            ############################################
            # save pixel
            ############################################

            for var in var_list:
                result_beta[var][pix] = beta_dic[var]
                result_p[var][pix] = p_dic[var]

            result_r2[pix] = r2_list

        ############################################
        # save
        ############################################

        for var in var_list:
            T.save_npy(
                result_beta[var],
                outdir + rf'beta_{var}_{season}.npy'
            )

            T.save_npy(
                result_p[var],
                outdir + rf'p_value_{var}_{season}.npy'
            )

        T.save_npy(
            result_r2,
            outdir + rf'R2_{season}.npy'
        )

    def calculating_multiregression_sensitivity_block(self):


        import statsmodels.api as sm

        season = 'summer'

        ############################################################
        # input directory
        ############################################################

        fdir = result_root + r'\zscore\Moving_window\breakdown_10year\\'

        fdir_LAI = fdir + rf'{season}_LAI_zscore\\'

        file_dic = {

            'ppt': fdir + r'ppt_winter_npy_zscore\\',

            'intensity': fdir + rf'{season}_rainfall_intensity_zscore\\',

            'temp': fdir + rf'tmax_{season}_npy_zscore\\',

            'SPEI3': fdir + rf'{season}_SPEI3\\',

            'rad': fdir + rf'srad_{season}_npy_zscore\\',

        }

        var_list = list(file_dic.keys())

        ############################################################
        # output directory
        ############################################################

        outdir = result_root + r'\Moving_window_coupling_analysis\output\10year_multiregression\\'

        T.mk_dir(outdir, force=True)

        for var in var_list:
            T.mk_dir(outdir + rf'beta_{var}_{season}\\', force=True)

            T.mk_dir(outdir + rf'p_{var}_{season}\\', force=True)

        T.mk_dir(outdir + rf'R2_{season}\\', force=True)

        ############################################################
        # loop every spatial_dict_xxx.npy
        ############################################################

        file_list = sorted(os.listdir(fdir_LAI))

        for f in file_list:

            if not f.endswith('.npy'):
                continue



            ########################################################
            # load one block
            ########################################################

            dic_LAI = T.load_npy(os.path.join(fdir_LAI, f))

            dic_var = {}

            for var in var_list:
                dic_var[var] = T.load_npy(

                    os.path.join(file_dic[var], f)

                )

            ########################################################
            # output dictionary
            ########################################################

            result_beta = {v: {} for v in var_list}

            result_p = {v: {} for v in var_list}

            result_r2 = {}

            ########################################################
            # loop every pixel
            ########################################################

            for pix in tqdm(dic_LAI):

                ##############################################
                # check existence
                ##############################################

                if any(pix not in dic_var[v] for v in var_list):
                    continue

                vals_LAI = np.array(dic_LAI[pix], dtype=float)

                if vals_LAI.ndim != 2:
                    continue

                n_windows, n_years = vals_LAI.shape

                skip = False

                var_arrays = {}

                for var in var_list:

                    arr = np.array(dic_var[var][pix], dtype=float)

                    if arr.ndim != 2:
                        skip = True
                        break

                    if arr.shape != vals_LAI.shape:
                        print(f'{pix} {var}: {arr.shape} != {vals_LAI.shape}')

                        skip = True
                        break

                    var_arrays[var] = arr

                if skip:
                    continue

                ####################################################
                # initialize
                ####################################################

                beta_dic = {v: [] for v in var_list}

                p_dic = {v: [] for v in var_list}

                r2_list = []

                ####################################################
                # moving window regression
                ####################################################

                for w in range(n_windows):

                    ########################################
                    # response variable
                    ########################################

                    y = vals_LAI[w, :]

                    ########################################
                    # predictor matrix
                    ########################################

                    X = []

                    valid = np.ones(len(y), dtype=bool)

                    for var in var_list:
                        x = var_arrays[var][w, :]

                        valid &= np.isfinite(x)

                        X.append(x)

                    valid &= np.isfinite(y)

                    if np.sum(valid) < 10:
                        continue

                    ########################################
                    # remove NaN
                    ########################################

                    y_valid = y[valid]

                    X_valid = []

                    for x in X:
                        X_valid.append(x[valid])

                    X_valid = np.column_stack(X_valid)

                    ########################################
                    # constant
                    ########################################

                    X_valid = sm.add_constant(X_valid)

                    ########################################
                    # regression
                    ########################################

                    try:

                        model = sm.OLS(y_valid, X_valid).fit()

                    except:

                        continue

                    ########################################
                    # adjusted R2
                    ########################################

                    r2_list.append(model.rsquared_adj)

                    ########################################
                    # beta & p
                    ########################################

                    params = model.params[1:]

                    pvalues = model.pvalues[1:]

                    for i, var in enumerate(var_list):
                        beta_dic[var].append(params[i])

                        p_dic[var].append(pvalues[i])

                    ####################################################
                # save one pixel
                ####################################################

                if len(r2_list) == 0:
                    continue

                result_r2[pix] = np.array(r2_list)

                for var in var_list:
                    result_beta[var][pix] = np.array(beta_dic[var])

                    result_p[var][pix] = np.array(p_dic[var])

            ########################################################
            # save this spatial block
            ########################################################

            for var in var_list:
                fout = outdir + rf'beta_{var}_{season}\\{f}'

                np.save(fout, result_beta[var])

                fout = outdir + rf'p_{var}_{season}\\{f}'

                np.save(fout, result_p[var])

            fout = outdir + rf'R2_{season}\\{f}'

            np.save(fout, result_r2)





    def calculate_optimal_scale(self):

        import numpy as np
        from tqdm import tqdm

        season = 'summer'

        scale_list = [1,3, 6,9, 12,15, 18,21, 24, 27, 30,33, 36, 48]

        outdir = result_root + r'\Moving_window_coupling_analysis\output\\10year\\'

        ############################################
        # 读取所有scale的correlation
        ############################################

        corr_dic = {}

        for scale in scale_list:
            f = outdir + rf'partial_corr_SPEI{scale}_{season}.npy'

            corr_dic[scale] = T.load_npy(f)

        ############################################
        # 所有pixel
        ############################################

        pix_list = list(corr_dic[3].keys())

        optimal_scale_dic = {}
        optimal_corr_dic = {}

        for pix in tqdm(pix_list):

            ########################################
            # shape=(n_scale,n_window)
            ########################################

            corr_matrix = []

            for scale in scale_list:

                if pix not in corr_dic[scale]:
                    continue

                corr_matrix.append(corr_dic[scale][pix])

            corr_matrix = np.array(corr_matrix, dtype=float)

            if corr_matrix.ndim != 2:
                continue

            n_scale, n_window = corr_matrix.shape

            optimal_scale_list = []
            optimal_corr_list = []

            ########################################
            # 每一个window寻找最佳scale
            ########################################

            for w in range(n_window):

                corr = corr_matrix[:, w]

                if np.all(np.isnan(corr)):
                    optimal_scale_list.append(np.nan)
                    optimal_corr_list.append(np.nan)
                    continue

                ########################################
                # 最大绝对值相关
                ########################################

                # idx = np.nanargmax(np.abs(corr))
                idx = np.nanargmax(corr)

                optimal_scale_list.append(scale_list[idx])

                optimal_corr_list.append(corr[idx])

            optimal_scale_dic[pix] = optimal_scale_list
            optimal_corr_dic[pix] = optimal_corr_list

        ############################################
        # Save
        ############################################


        T.save_npy(
            optimal_scale_dic,
            outdir + rf'optimal_scale_{season}.npy'
        )

        T.save_npy(
            optimal_corr_dic,
            outdir + rf'optimal_corr_{season}.npy'
        )



    def trend_analysis(self):
        fdir = result_root + r'\Moving_window_coupling_analysis\output\10year_detrend\\'
        outdir = result_root + r'\Moving_window_coupling_analysis\output\\10year_multiregression\\trend\\'
        T.mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):


            fname = f.split('.')[0]


            fpath = join(fdir, f)
            dic = T.load_npy(fpath)
            result_dic = {}
            pvalue_result = {}
            for pix in dic:
                vals = dic[pix]
                vals = vals

                slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
                result_dic[pix] = slope
                pvalue_result[pix] = p_value
            D.pix_dic_to_tif(result_dic, outdir + f'{fname}_trend.tif')
            D.pix_dic_to_tif(pvalue_result, outdir + f'{fname}_pvalue.tif')

            pass

    def trend_analysis_block(self):
        fdir = result_root + r'\Moving_window_coupling_analysis\output\10year_multiregression_block\beta_intensity_summer\\'
        outdir = result_root + r'\Moving_window_coupling_analysis\output\\10year_multiregression_block\\trend\\'
        T.mk_dir(outdir, force=True)
        fname=fdir.split('\\')[-1]

        dic = T.load_npy_dir(fdir)
        result_dic = {}
        pvalue_result = {}
        for pix in dic:
            vals = dic[pix]


            slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
            result_dic[pix] = slope
            pvalue_result[pix] = p_value
        array=D.pix_dic_to_spatial_arr(result_dic)
        plt.imshow(array,vmin=-.1,vmax=.1)
        plt.show()
        # D.pix_dic_to_tif(result_dic, outdir + f'{fname}_trend.tif')
        # D.pix_dic_to_tif(pvalue_result, outdir + f'{fname}_pvalue.tif')



    def PLot_window_slices(self):

        fdir = result_root + r'Moving_window_coupling_analysis\output\10year_trend\\'
        outdir = result_root + r'\Moving_window_coupling_analysis\moving_window_extraction_10year_trend_slice\\'
        T.mk_dir(outdir, force=True)
        moving_window=10
        window_size=22-moving_window+1
        for f in os.listdir(fdir):
            if not 'partial_p' in f:
                continue

            fname = f.split('.')[0]


            fpath = join(fdir, f)
            dic = T.load_npy(fpath)
            result_dic = {}

            for w in range(window_size):
                for pix in dic:
                    vals = dic[pix][w]
                    vals = vals

                    result_dic[pix] = vals
                D.pix_dic_to_tif(result_dic, outdir + f'{fname}_{w}.tif')

            pass






class PLOT_temporal_change_corr:
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):

        # self.plot_SPEI_time_series()
        # self.heatmap()
        self.plot_barplot()
        # self.plot_time_series_MODIS_record()


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


    def plot_SPEI_time_series(self):
        df = T.load_df(
            result_root + rf'\Moving_window_coupling_analysis\Dataframe\Dataframe_5year.df')


        df = self.df_clean(df)


        eco_region_list = df['Ecoregion_level_II'].dropna().unique().tolist()
        eco_region_list.append('Western US')

        eco_region_list = ['Western US', 'Western Cordillera', 'Upper Gila Mountains',
                           'Warm Desert', 'Cold Desert', 'Western Sierra Madre Piedmont']
        season='summer'
        variable_list = [f'optimal_corr_summer']



        dic_label = {f'5year_partial_corr_SPEI48_{season}': 'SPEI48',
                     f'partial_corr_SPEI24_{season}': 'SPEI24',
                     f'partial_corr_SPEI36_{season}': 'SPEI36',
                     f'partial_corr_SPEI3_{season}': 'SPEI3',
                     f'partial_corr_SPEI12_{season}': 'SPEI12',
                     f'partial_corr_SPEI6_{season}': 'SPEI6',
                     f'optimal_corr_summer': 'Optimal scale'
                     }

        result_dic = {}

        for eco in eco_region_list:

            if eco == 'Western US':
                df_i = df.copy()
            else:
                df_i = df[df['Ecoregion_level_II'] == eco]

            for var in variable_list:

                mean_dic = {}
                std_dic = {}

                for year in sorted(df_i['year'].unique()):

                    df_year = df_i[df_i['year'] == year]

                    vals = df_year[var].values.astype(float)
                    weight = df_year['area_weight'].values.astype(float)

                    mask = np.isfinite(vals)

                    vals = vals[mask]
                    weight = weight[mask]

                    if len(vals) == 0:
                        mean_dic[year] = np.nan
                        std_dic[year] = np.nan
                        continue

                    ##########################
                    # weighted mean
                    ##########################

                    weighted_mean = np.sum(vals * weight) / np.sum(weight)

                    ##########################
                    # weighted std
                    ##########################

                    weighted_var = np.sum(
                        weight * (vals - weighted_mean) ** 2
                    ) / np.sum(weight)

                    weighted_std = np.sqrt(weighted_var)

                    mean_dic[year] = weighted_mean
                    std_dic[year] = weighted_std

                result_dic[f'{eco}_{dic_label[var]}'] = mean_dic
                result_dic[f'{eco}_{dic_label[var]}_std'] = std_dic

        df_new = pd.DataFrame(result_dic)
        df_new.index.name = 'year'
        df_new = df_new.reset_index()

        fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)

        axes = axes.flatten()

        for i, eco in enumerate(eco_region_list):

            ax = axes[i]

            for var in variable_list:
                label = dic_label[var]

                ax.plot(
                    df_new['year'],
                    df_new[f'{eco}_{label}'],
                    lw=2,
                    marker='o',
                    markersize=3,
                    label=label
                )

                # 如果想画标准差
                # std = df_new[f'{eco}_{label}_std']
                # ax.fill_between(
                #     df_new['year'],
                #     df_new[f'{eco}_{label}']-std,
                #     df_new[f'{eco}_{label}']+std,
                #     alpha=0.2
                # )

            ax.set_title(eco, fontsize=11)

            ax.axhline(0, ls='--', color='gray', lw=1)

            ax.set_ylim(-1, 1)

            ax.set_xlabel('Moving window')
            ax.set_ylabel('Correlation')

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc='upper center',
            ncol=len(variable_list),
            frameon=False
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.show()

    def heatmap(self):

        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        dff = result_root + r'\Moving_window_coupling_analysis\Dataframe\moving_window\Dataframe_10year_trend.df'
        df = T.load_df(dff)
        for col in df.columns:
            print(col)

        eco_region_list = [
            'Western US',
            'Western Cordillera',
            'Upper Gila Mountains',
            'Warm Desert',
            'Cold Desert',
            'Western Sierra Madre Piedmont'
        ]

        scale_list = [ 3, 6,9,  12,  24,  36, 48]
        print(df)

        ## 不要重复的行
        df_unique = df.drop_duplicates(subset='pix', keep='first')
        print(len(df_unique))




        fig, axes = plt.subplots(
            2,
            3,
            figsize=(8, 4),
            sharex=True,
            sharey=True
        )

        axes = axes.flatten()
        window_list=22+1-10

        for i, eco in enumerate(eco_region_list):

            ax = axes[i]

            if eco == 'Western US':
                df_i = df_unique.copy()
            else:
                df_i = df_unique[df['Ecoregion_level_II'] == eco]



            heatmap = []

            for scale in scale_list:


                # optimal=f'optimal_corr_summer'
                # col_p = f'partial_p_SPEI{scale}_{season}'


                corr_list = []

                for window in np.arange(int(window_list)):




                    vals = df_i[f'partial_corr_SPEI{scale}_summer_{window}']
                    # pval = df_i[f'partial_p_SPEI{scale}_summer_{window}']

                    # mask = (pval < 0.05) & np.isfinite(vals)
                    mask=np.isfinite(vals)
                    weight = np.array(df_i['area_weight'], dtype=float)



                    if np.sum(mask) == 0:
                        corr_list.append(np.nan)
                        continue

                    weighted_mean = np.sum(vals[mask] * weight[mask]) / np.sum(weight[mask])

                    corr_list.append(weighted_mean)

                heatmap.append(corr_list)

            heatmap = np.array(heatmap)

            sns.heatmap(
                heatmap[::-1],
                ax=ax,
                cmap='RdBu',
                # center=0,
                vmin=-.2,
                vmax=.8,
                xticklabels=np.arange(window_list),
                yticklabels=scale_list[::-1],
                cbar=(i == 5),
                cbar_kws={'label': 'Pearson r'}
            )

            ax.set_title(eco)

            ax.set_xlabel('Moving window')

            ax.set_ylabel('')

        plt.tight_layout()

        plt.show()

    def plot_barplot(self):

        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        dff = result_root + r'Moving_window_coupling_analysis\Dataframe\\Dataframe_10yeartrend.df'
        df = T.load_df(dff)

        eco_region_list = [
            'Western US',
            'Western Cordillera',
            'Upper Gila Mountains',
            'Warm Desert',
            'Cold Desert',
            'Western Sierra Madre Piedmont'
        ]

        scale_list = [3, 6, 9, 12,   36, 48]
        window_num = 22 + 1 - 10

        positive_dic = {}
        negative_dic = {}
        print(len(df))

        # df_unique = df.drop_duplicates(subset='pix', keep='last')
        # print(len(df_unique))

        for i, eco in enumerate(eco_region_list):



            if eco == 'Western US':
                df_i = df.copy()
            else:
                df_i = df[df['Ecoregion_level_II'] == eco]

            for scale in scale_list:

                positive_area = []
                negative_area = []

                df_scale = df_i[df_i[f'summer_SPEI{scale}_trend'] < 0].copy()
                print(len(df_scale))
                # print(len(df_i));exit()
                spatial_dic=T.df_to_spatial_dic(df,col_name='summer_SPEI3_trend',reduce_method=np.mean)
                array=D.pix_dic_to_spatial_arr(spatial_dic)
                plt.imshow(array, cmap='RdBu', vmin=-.8, vmax=.8)
                plt.show()
                # df_scale = df_i

                for window in range(window_num):


                    r = np.array(
                        df_scale[f'partial_corr_SPEI{scale}_summer_{window}'],
                        dtype=float
                    )

                    p = np.array(
                        df_scale[f'partial_p_SPEI{scale}_summer_{window}'],
                        dtype=float
                    )

                    valid = np.isfinite(r) & np.isfinite(p)

                    r = r[valid]
                    p = p[valid]

                    total_valid = len(r)

                    if total_valid == 0:
                        positive_area.append(np.nan)
                        negative_area.append(np.nan)
                        continue

                    # print(
                    #     scale,
                    # #     window,
                    #     len(df_scale),
                    #     np.sum((r > 0) & (p < 0.05)),
                    #     np.sum((r < 0) & (p < 0.05))
                    # );exit()

                    # Positive significant (%)
                    pos = np.sum((r > 0) & (p < 0.05)) / total_valid * 100

                    # Negative significant (%)
                    neg = np.sum((r < 0) & (p < 0.05)) / total_valid * 100

                    positive_area.append(pos)
                    negative_area.append(neg)
                positive_dic[scale] = positive_area
                negative_dic[scale] = negative_area

            fig, axes = plt.subplots(3, 2, figsize=(8, 6), sharey=True)

            axes = axes.flatten()

            years = np.arange(window_num)

            for i, scale in enumerate(scale_list):
                ax = axes[i]

                pos = np.array(positive_dic[scale])
                neg = np.array(negative_dic[scale])

                ax.bar(
                    years,
                    pos,
                    color='royalblue',
                    label='Positive'
                )

                ax.bar(
                    years,
                    -neg,
                    color='firebrick',
                    label='Negative'
                )

                ax.axhline(0, color='k', lw=0.8)

                ax.set_title(f'SPEI-{scale}')



                ax.set_ylim(-5, 35)

            axes[0].set_ylabel('Area (%)')

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=2)

            plt.tight_layout()
            plt.show()

        pass

    def plot_time_series_SPEI(self):
        dff = result_root + rf'\Moving_window_coupling_analysis\Dataframe\moving_window\Dataframe_5year.df'
        df = T.load_df(dff)

        df = self.df_clean(df)

        window_list = list(range(0, 18))
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
            for scale in [3,6,12,24,36,48]:
                mean_dic = {}
                std_dic = {}

                for year in window_list:
                    df_ii = df_i[df_i['window'] == year]
                    ## scheme1
                    vals = np.array(df_ii[f'summer_SPEI{scale}_average'].tolist(), dtype=float)
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

                result_dic[f'{eco}_SPEI{scale}_mean'] = mean_dic
                result_dic[f'{eco}_SPEI{scale}_std'] = std_dic

                # 只存一次长度
                result_dic[f'{eco}_len'] = len(df_i)

            # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()

        # T.print_head_n(df_new);exit()

        flag = 0

        for eco in eco_region_list:

            for scale in [3, 6, 12, 24, 36, 48]:
                plt.figure(figsize=(self.map_width * 1.5, self.map_height))


                summer_vals = df_new[f'{eco}_SPEI{scale}_mean']

                summer_std = df_new[f'{eco}_SPEI{scale}_std']

                vals_len = df_new[f'{eco}_len'][0]



                # -----------------------------
                # Summer
                # -----------------------------
                color_summer = '#48526F'

                plt.plot(
                    window_list,
                    summer_vals,
                    color=color_summer,
                    lw=2,
                    label='Summer'
                )

                slope, intercept, r, p, _ = stats.linregress(window_list, summer_vals)
                window_array = np.array(window_list)

                plt.plot(
                    window_list,
                    slope * window_array + intercept,
                    '--',
                    color=color_summer,
                    lw=2,

                )


                plt.fill_between(
                    window_list,
                    summer_vals - summer_std,
                    summer_vals + summer_std,
                    color=color_summer,
                    alpha=0.2
                )
                plt.legend()

                stats_text = (

                    f'Summer: slope={slope:.2f}, p={p:.2f}'
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

    def plot_time_series_MODIS_record(self):
        dff = result_root + rf'\Moving_window_coupling_analysis\Dataframe\moving_window\Dataframe_5year.df'
        df = T.load_df(dff)

        df = self.df_clean(df)

        window_list = list(range(0,18))
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

            mean_dic = {}
            std_dic = {}

            for year in window_list:
                df_ii = df_i[df_i['window'] == year]
                ## scheme1
                vals = np.array(df_ii['summer_LAI_anomaly_average'].tolist(), dtype=float)
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
                # print(
                #     eco,
                #     year,
                #     np.nanmean(vals),
                #     np.nanstd(vals),
                #     weighted_std,
                # );exit()

            result_dic[f'{eco}_summer_LAI_mean'] = mean_dic
            result_dic[f'{eco}_summer_LAI_std'] = std_dic

            # 只存一次长度
            result_dic[f'{eco}_len'] = len(df_i)

            # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()

        # T.print_head_n(df_new);exit()

        flag = 0

        for eco in eco_region_list:
            plt.figure(figsize=(self.map_width * 1.5, self.map_height))


            summer_vals = df_new[f'{eco}_summer_LAI_mean']

            summer_std = df_new[f'{eco}_summer_LAI_std']

            vals_len = df_new[f'{eco}_len'][0]


            # -----------------------------
            # Summer
            # -----------------------------

            color_summer = '#07967F'

            plt.plot(
                window_list,
                summer_vals,
                color=color_summer,
                lw=2,
                label='Summer'
            )

            slope, intercept, r, p, _ = stats.linregress(window_list, summer_vals)
            window_array=np.array(window_list)

            plt.plot(
                window_list,
                slope * window_array + intercept,
                '--',
                color=color_summer,
                lw=2,

            )


            plt.fill_between(
                window_list,
                summer_vals - summer_std,
                summer_vals + summer_std,
                color=color_summer,
                alpha=0.2
            )
            plt.legend()

            stats_text = (

                f'Summer: slope={slope:.2f}, p={p:.2f}'
            )

            plt.text(0.95, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     )

            plt.ylabel('MODIS_LAI_anomaly(m2/m2)', fontsize=12)

            plt.title(f'{eco}_n={vals_len}', fontsize=12)
            # plt.ylim(1.1,1.3)

            plt.grid(True, axis='x')

            plt.show()
            plt.close()

class Breakdown_data:
    def __init__(self):

        pass

    def run(self):
        # self.break_down_data()
        self.check_data()
        pass

    def break_down_data(self):
        fdir=result_root + rf'\zscore\Moving_window\10year\\'
        outdir =result_root + rf'\zscore\Moving_window\\\breakdown_10year\\'
        T.mkdir(outdir)
        for f in T.listdir(fdir):

            outdir_i=join(outdir, f.split('.')[0])
            T.mkdir(outdir_i)
            dic=T.load_npy(fdir+f)
            T.save_distributed_perpix_dic(dic, outdir_i, n=1000,prefix='spatial_dict',istqdm=True)

    pass

    def check_data(self):
        fdir = result_root + rf'\zscore\Moving_window\\\breakdown_10year\\'
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
        pass

class PLOT_bivariate():  ##
    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        self.__color_idx_array()
        pass

    def run(self):
        ## step 1
        self.bivariate_map()
        ## step 2
        # self.Figure_robinson_reprojection()  # convert figure to robinson and no need to plot robinson again

       ## step 3
        # self.statistic_pdf()


        pass
    def bivariate_map(self):  ## figure 1  ## LAImin and LAImax bivariate
        import xymap


        fdir_corr=result_root + rf'\bivariate\corr_10year_trend\\'
        fdir_SPEI=result_root + rf'\bivariate\SPEI_trend\\'

        outdir =result_root + rf'bivariate\\'

        T.mkdir(outdir)

        outtif = join(outdir,'SPEI48.tif')


        fpath1 = join(fdir_corr,'partial_corr_SPEI48_summer_trend.tif')

        fpath2 = join(fdir_SPEI,'summer_SPEI48_trend.tif')


        tif1_label, tif2_label = 'partial_corr_trend','SPEI_trend'

        #1
        # min1, max1 = -1, 1
        # min2, max2 = -1, 1

        #2
        min1, max1 = -.1, .1
        min2, max2 = 0, .1

        arr1 = ToRaster().raster2array(fpath1)[0]
        arr2 = ToRaster().raster2array(fpath2)[0]

        arr1[arr1<-9999] = np.nan
        arr2[arr2<-9999] = np.nan

        arr1_flattened = arr1.flatten()
        arr2_flattened = arr2.flatten()


        # plt.hist(arr1_flattened,bins=100)
        # plt.title('arr1')
        # plt.figure()
        # plt.hist(arr2_flattened,bins=100)
        # plt.title('arr2')
        # plt.show()

        # choice 1
        upper_left_color = (193,92,156)
        upper_right_color =(112, 196, 181)
        lower_left_color = (237, 125, 49)
        lower_right_color = (0, 0, 110)
        center_color = (240, 240, 240)

        ## CV greening option

        # upper_left_color = (194, 0, 120)
        # upper_right_color = (0,170,237)
        # lower_left_color = (233, 55, 43)
        # # lower_right_color = (160, 108, 168)
        # lower_right_color = (234, 233, 46)
        # center_color = (240, 240, 240)

        # upper_left_color = (0, 0, 110)
        # upper_right_color = (112, 196, 181)
        # lower_left_color = (237, 125, 49)
        #
        # lower_right_color = (193, 92, 156)
        # center_color = (240, 240, 240)


        xymap.Bivariate_plot_1(res = 4,
                         alpha = 255,
                         upper_left_color = upper_left_color, #
                         upper_right_color = upper_right_color, #
                         lower_left_color = lower_left_color, #
                         lower_right_color = lower_right_color, #
                         center_color = center_color).plot_bivariate(
                                                                    fpath1, fpath2,
                                                                    tif1_label, tif2_label,
                                                                    min1, max1,
                                                                    min2, max2,
                                                                    outtif,
                                                                    n_x = 5, n_y = 5
                                                                    )

        T.open_path_and_file(outdir)



    def Figure_robinson_reprojection(self):  # convert figure to robinson

        fdir_trend = result_root +  rf'\bivariate\std_mean\\'
        temp_root = result_root + rf'\bivariate\std_mean\\\\\\'
        outdir = result_root + rf'\bivariate\std_mean\\\\\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):
            if not 'std_mean' in f:
                continue

            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]

            fpath = fdir_trend + f
            outf=outdir + fname + '.tif'
            srcSRS=self.wkt_84()
            dstSRS=self.wkt_robinson()

            ToRaster().resample_reproj(fpath,outf, 50000, srcSRS=srcSRS, dstSRS=dstSRS)

            T.open_path_and_file(outdir)




    def wkt_robinson(self):
        wkt='''PROJCRS["World_Robinson",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["World_Robinson",
        METHOD["Robinson"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["ESRI",54030]]
        '''
        return wkt


    def wkt_84(self):
        wkt = '''GEOGCRS["WGS 84",
    ENSEMBLE["World Geodetic System 1984 ensemble",
        MEMBER["World Geodetic System 1984 (Transit)"],
        MEMBER["World Geodetic System 1984 (G730)"],
        MEMBER["World Geodetic System 1984 (G873)"],
        MEMBER["World Geodetic System 1984 (G1150)"],
        MEMBER["World Geodetic System 1984 (G1674)"],
        MEMBER["World Geodetic System 1984 (G1762)"],
        MEMBER["World Geodetic System 1984 (G2139)"],
        ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],
        ENSEMBLEACCURACY[2.0]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433]],
    CS[ellipsoidal,2],
        AXIS["geodetic latitude (Lat)",north,
            ORDER[1],
            ANGLEUNIT["degree",0.0174532925199433]],
        AXIS["geodetic longitude (Lon)",east,
            ORDER[2],
            ANGLEUNIT["degree",0.0174532925199433]],
    USAGE[
        SCOPE["Horizontal component of 3D system."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["EPSG",4326]]'''
        return wkt


    def RGBA_to_tif(self,blend_arr,outf,originX, originY, pixelWidth, pixelHeight):
        import PIL.Image as Image
        img = Image.fromarray(blend_arr.astype('uint8'), 'RGBA')
        img.save(outf)
        # define a projection and extent
        raster = gdal.Open(outf)
        geotransform = raster.GetGeoTransform()
        raster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRasterSRS = osr.SpatialReference()
        projection = self.wkt_84()
        # outRasterSRS.ImportFromEPSG(4326)
        # outRasterSRS.ImportFromEPSG(projection)
        # raster.SetProjection(outRasterSRS.ExportToWkt())
        raster.SetProjection(projection)
        pass




    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def statistic_pdf(self):
        dff= result_root + rf'\3mm\product_consistency\dataframe\\moving_window.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        vals_min=df['composite_LAI_std_trend'].tolist()
        vals_max=df['composite_LAI_mean_trend'].tolist()
        vals_min=np.array(vals_min)
        vals_max=np.array(vals_max)
        vals_min[vals_min>99]=np.nan
        vals_min[vals_min<-99]=np.nan
        vals_max[vals_max>99]=np.nan
        vals_max[vals_max<-99]=np.nan


        figure, ax = plt.subplots(figsize=(3,2),  )
        sns.kdeplot(vals_min, ax=ax, label=f'std trend', fill=False,color='#9fd7e9',linewidth=2)
        sns.kdeplot(vals_max, ax=ax, label=f'mean trend', fill=False,color='#fcd590',linewidth=2)
        # ax.legend()
        # === 添加均值线和标注 ===
        mean_min = np.nanmean(vals_min)
        mean_max = np.nanmean(vals_max)

        ax.axvline(mean_min, color='#9fd7e9', linestyle='--', linewidth=1)
        ax.axvline(mean_max, color='#fcd590', linestyle='--', linewidth=1)
        # plt.legend()

        # 在图顶写出均值
        # ylim = ax.get_ylim()
        # ax.text(mean_min, ylim[1] * 0.9, f'{mean_min:.4f}',
        #         color='#f599a1', rotation=90, va='top', ha='right', fontsize=8)
        # ax.text(mean_max, ylim[1] * 0.9, f'{mean_max:.4f}',
        #         color='#a577ad', rotation=90, va='top', ha='left', fontsize=8)

        plt.ylabel('')
        plt.xlim(-0.01, 0.01)
        # plt.tight_layout()
        # plt.show()
        outf=result_root + rf'\3mm\product_consistency\pdf\\std_mean_trend.pdf'
        plt.savefig(outf,bbox_inches='tight',pad_inches=0,dpi=300)







    def classify(self,row):
        if row['composite_LAI_CV_trend'] > 0 and row['composite_LAI_relative_change_mean_trend'] > 0:
            return 1
        elif row['composite_LAI_CV_trend'] > 0 and row['composite_LAI_relative_change_mean_trend'] < 0:
            return 2
        elif row['composite_LAI_CV_trend'] < 0 and row['composite_LAI_relative_change_mean_trend'] > 0:
            return 3
        elif row['composite_LAI_CV_trend'] < 0 and row['composite_LAI_relative_change_mean_trend'] < 0:
            return 4
        else:
            return 'Other'

        # Apply classification

    def bivariate_scheme3(self):

        import xymap

        fdir = result_root + rf'\3mm\extract_composite_phenology_year\trend\\'

        outdir = result_root + rf'\3mm\extract_composite_phenology_year\bivariate\\'

        T.mkdir(outdir)

        outtif = join(outdir, 'std_mean.tif')

        fpath1 = join(fdir, 'composite_LAI_std_trend.tif')

        fpath2 = join(fdir, 'composite_LAI_mean_trend.tif')

        tif1_label, tif2_label = 'std_trend', 'mean_trend'



        # 2

        bins_x = np.array([-np.inf, -0.01, 0, 0.01, np.inf])
        bins_y = np.array([-np.inf, -0.01, 0, 0.01, np.inf])

        arr1 = ToRaster().raster2array(fpath1)[0]
        arr2 = ToRaster().raster2array(fpath2)[0]

        arr1[arr1 < -9999] = np.nan
        arr2[arr2 < -9999] = np.nan

        dict1 = DIC_and_TIF().spatial_arr_to_dic(arr1)
        dict2 = DIC_and_TIF().spatial_arr_to_dic(arr2)
        dict_list = {'std_trend': dict1, 'mean_trend': dict2}
        df_new = T.spatial_dics_to_df(dict_list)
        df_new = df_new.dropna(how='any')
        ##
        T.print_head_n(df_new)

        arr_count = np.zeros((len(bins_x) - 1, len(bins_y) - 1)).flatten()
        arr = np.ones((360, 720, 4)) * 0
        for i, row in tqdm(df_new.iterrows(), total=len(df_new)):
            pix = row['pix']
            x = row[tif1_label]
            y = row[tif2_label]
            color_idx, color = self.get_color(x, y, bins_x, bins_y)
            arr_count[color_idx - 1] = arr_count[color_idx - 1] + 1
            # print('binsx',bins_x)
            # print('binsy',bins_y)
            # print(x,y,color_idx,color);exit()

            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            a = 255
            arr[pix] = np.array([r, g, b, a])
        self.RGBA_to_tif(arr, outtif, -180, 90, 0.5, -0.5)
        arr_count = arr_count.reshape((len(bins_x) - 1, len(bins_y) - 1))

        self.plot_legend(outdir,arr_count)

        T.open_path_and_file(outdir)


    def get_color(self, x, y, bins_x, bins_y):
        # color_array = [
        #     ['#f26d21 ', '#f598a1', '#ec1d8f', '#ec1d0f'],
        #     ['#c7e86e', '#7BC8F6', '#d3a3cb', '#f26d21'],
        #     ['#0e6b3f ', '#98cdd2', '#5d4a8d', '#f598a1'],
        #     ['#0e6b3f ', '#98cdd2', '#5d4a8d', '#7BC8F6'],
        # ][::-1]
        color_idx_array = self.color_idx_array
        color_dict = self.color_dict
        color_idx_array = np.array(color_idx_array)
        idx_x = np.digitize(x, bins_x) - 1
        idx_y = np.digitize(y, bins_y) - 1
        color_idx = color_idx_array[idx_y][idx_x]
        color = color_dict[color_idx]
        return color_idx, color

    def __color_idx_array(self):
        self.color_idx_array = [
                                   [1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]
                               ][::-1]
        self.color_dict = {
            4: '#3182bd',
            3: '#6baed6',
            2: '#bd9e39',
            1: '#8c6d31',
            12: '#e6550d',
            11: '#fd8d3c',
            10: '#ad494a',
            9: '#843c39',
            8: '#31a354',
            7: '#74c476',
            6: '#e7969c',
            5: '#d6616b',
            16: '#756bb1',
            15: '#9e9ac8',
            14: '#cedb9c',
            13: '#b5cf6b'
        }

    def plot_legend(self, outdir,count_arr):

        color_array = []
        for row in self.color_idx_array:
            color_array_i = []
            for col in row:
                color = self.color_dict[col]
                rgb_color = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)  ### convert hex to rgb
                color_array_i.append(rgb_color)
            color_array.append(color_array_i)
        color_array = np.array(color_array)[::-1]
        sns.heatmap(self.color_idx_array,annot=count_arr, fmt='g',alpha=0,annot_kws={'alpha':1,'color':'k'})
        plt.imshow(color_array)

        plt.show()
        # plt.savefig(join(outdir, 'legend.pdf'))
        # plt.close()
        pass






    def Figure2b_test(self):
        fdir = result_root + rf'\3mm\extract_composite_phenology_year\trend\\'

        outdir = result_root + rf'\3mm\extract_composite_phenology_year\bivariate\\'
        self.plot_legend(outdir)

        T.mkdir(outdir)

        outtif = join(outdir, 'CV_trend_test.tif')
        # outtif = join(outdir, 'LAImin_LAImax.tif')

        # fpath1 = join(fdir,'composite_LAI_detrend_relative_change_min_trend.tif')
        fpath1 = join(fdir, 'composite_LAI_CV_trend.tif')
        # fpath2 = join(fdir,'composite_LAI_detrend_relative_change_max_trend.tif')
        fpath2 = join(fdir, 'composite_LAI_relative_change_mean_trend.tif')

        # 1
        # tif1_label, tif2_label = 'LAImin_trend','LAImax_trend'
        # 2
        tif1_label, tif2_label = 'LAI_CV_trend', 'LAI_trend'

        # 2
        bins_x = np.array([-np.inf, -0.5, 0, 0.5, np.inf])
        bins_y = np.array([-np.inf, -0.3, 0, 0.3, np.inf])

        arr1 = ToRaster().raster2array(fpath1)[0]
        arr2 = ToRaster().raster2array(fpath2)[0]

        arr1[arr1 < -9999] = np.nan
        arr2[arr2 < -9999] = np.nan

        dict1 = DIC_and_TIF().spatial_arr_to_dic(arr1)
        dict2 = DIC_and_TIF().spatial_arr_to_dic(arr2)
        dict_list = {'LAI_CV_trend': dict1, 'LAI_trend': dict2}
        df_new = T.spatial_dics_to_df(dict_list)
        df_new = df_new.dropna(how='any')
        ##
        T.print_head_n(df_new)

        arr_count=np.zeros((len(bins_x)-1,len(bins_y)-1)).flatten()
        for i, row in tqdm(df_new.iterrows(), total=len(df_new)):
            pix = row['pix']
            x = row[tif1_label]
            y = row[tif2_label]
            color_idx, color = self.get_color(x, y, bins_x, bins_y)
            arr_count[color_idx-1]=arr_count[color_idx-1]+1

        arr_count=arr_count.reshape((len(bins_x)-1,len(bins_y)-1))
        arr_percentage=arr_count/np.nansum(arr_count)*100
        arr_log=np.log10(arr_count)
        fig,ax=plt.subplots(1,1,figsize=(5,5))

        plt.imshow(arr_percentage, cmap='GnBu', vmin=5, vmax=40)
        ## add line y=0 and x=0
        ax.axhline(y=1.5, color='k', linewidth=1)
        ax.axvline(x=1.5, color='k', linewidth=1)
        ## add label
        ax.set_xlabel('Trend in CVLAI (%/yr)', fontsize=10)
        ax.set_ylabel('Trend in LAI (%/yr)',fontsize=10)
        ## xtick is empty
        ax.set_xticks([])
        ax.set_yticks([])
        ## set colorbar name 'percentage
        # cbar = plt.colorbar()
        # cbar.ax.set_title('Percentage', fontsize=10)

        # plt.tight_layout()
        plt.colorbar()

        plt.show()

        # plt.savefig(outtif, dpi=300, bbox_inches='tight')
        # plt.close()

    pass


class categroy:
    def __init__(self):
        pass
    def run(self):
        self.categroy_analysis()
        pass

    def categroy_analysis(self):
        import numpy as np
        from matplotlib.colors import ListedColormap

        dff = result_root + rf'\coupling_anaysis\Dataframe\\Dataframe.df'
        df = T.load_df(dff)
        print(len(df))
        df = self.df_clean(df)
        for col in df.columns:
            print(col)

        season = 'summer'
        scale = 9

        # 重新梳理后的条件列表（核心聚焦于干旱加剧下的生态响应）
        conditions = [
            # 1. Drying + Greening (干旱加剧但显著变绿)
            (
                    (df[f' {season}_SPEI{scale}_trend'] < 0) &
                    (df[f' {season}_SPEI{scale}_p_value'] < 0.05) &
                    (df[f' {season}_LAI_trend'] > 0) &
                    (df[f' {season}_LAI_p_value'] < 0.05)
            ),

            # 2. Drying + LAI no change (干旱加剧但无显著变化 - 核心悖论区)
            (
                    (df[f' {season}_SPEI{scale}_trend'] < 0) &
                    (df[f' {season}_SPEI{scale}_p_value'] < 0.05) &
                    (df[f' {season}_LAI_p_value'] >= 0.05)
            ),

            # 3. Drying + Browning (干旱加剧且显著变黄/退化)
            (
                    (df[f' {season}_SPEI{scale}_trend'] < 0) &
                    (df[f' {season}_SPEI{scale}_p_value'] < 0.05) &
                    (df[f' {season}_LAI_trend'] < 0) &
                    (df[f' {season}_LAI_p_value'] < 0.05)
            ),

            # 4. Wetting overall (气候变湿区域作为整体对照)
            (
                    (df[f' {season}_SPEI{scale}_trend'] > 0) &
                    (df[f' {season}_SPEI{scale}_p_value'] < 0.05)
            )
        ]

        # 对应的数值标签 (1, 2, 3 对应干旱的三种命运，4 对应变湿区)
        choices = [1, 2, 3, 4]

        # 赋值（如果你不需要重复写 Class 和 summer_class，可以只保留其中一个）
        df['Class'] = np.select(conditions, choices, default=np.nan)
        df['summer_class'] = np.select(conditions, choices, default=np.nan)

        spatial_dic = T.df_to_spatial_dic(df, 'summer_class')
        arr = D.pix_dic_to_spatial_arr(spatial_dic)

        outdir = result_root + rf'\coupling_anaysis\\categroy_analysis\\'
        T.mk_dir(outdir, force=True)

        D.arr_to_tif(arr, outdir + rf'summer_class_{scale}.tif')


        #
        # cmap = ListedColormap([
        #     'white',  # background
        #     'firebrick',  # 1
        #     'gold',  # 2
        #     'forestgreen',  # 3
        #     'steelblue' , # 4
        #     'blue'
        #
        # ])
        #
        # plt.figure(figsize=(8, 6))
        # plt.imshow(arr,
        #            cmap=cmap,
        #            vmin=0,
        #            vmax=6,
        #            interpolation='nearest')
        #
        # cbar = plt.colorbar(ticks=[1, 2, 3, 4,5,6])
        # cbar.ax.set_yticklabels([
        #     'drying greening',
        #     'drying browning',
        #     'wetting greening',
        #     'wetting browning',
        #     'others'
        # ])
        #
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()



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

class SHAP():

    def __init__(self):
        self.y_variable = 'summer_LAI_detrend'

        self.this_class_png = result_root + rf'\SHAP\\png\\RF_{self.y_variable}\\'
        T.mk_dir(self.this_class_png, force=True)

        self.dff = result_root+rf'\SHAP\Dataframe\Dataframe_yearly.df'

        self.variable_list_rt()


        ##----------------------------------


        ####################

        self.x_variable_list = self.x_variable_list
        self.x_variable_range_dict = self.x_variable_range_dict_global

        pass

    def run(self):
        # self.check_df_attributes()
        # # #
        # # #
        # self.check_variables_ranges()
        # # #
        # self.show_colinear()
        # # self.check_spatial_plot()
        # # self.AIC_stepwise(self.dff)
        self.pdp_shap()
        # # # # # #
        # self.plot_pdp_shap()
        # self.plot_shaply_under_different_condition()
        # self.heatmap()
        # self.plot_bar_landcover()
        # self.shapely_df_generation()
        # self.plot_bar_shap()
        # self.plot_pdp_shap_test()
        # self.plot_pdp_shap_density_cloud()
        # self.plot_pdp_shap_density_cloud_individual()  ## paper use
        # self.plot_pdp_shap_density_cloud_individual_test()
        # self.plot_relative_importance()
        # self.plot_relative_importance_pie_plot()
    # self.plot_pdp_shap_all_models_SI()
        # self.plot_pdp_shap_all_models_main()
        # self.plot_heatmap_ranking()
        # self.plot_interaction_manual()
        # self.spatial_shapely_vs_aridity()
        # self.spatial_shapely()   ### spatial plot
        #
        #
        # self.variable_contributions()
        # self.plot_dominant_factors_bar()
        # self.plot_robinson()
        # self.max_contributions()
        # self.disentangle()


        pass

    def check_df_attributes(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print(df.columns.tolist())
        print(len(df))
        exit()
        pass

    def check_variables_ranges(self):

        dff = self.dff
        df = T.load_df(dff)
        df = self.df_clean(df)

        df = self.plot_hist(df)
        df = self.valid_range_df(df)
        # df = self.__select_extreme(df)
        # T.print_head_n(df)
        # exit()

        x_variable_list = self.x_variable_list
        print(len(x_variable_list))
        # exit()
        flag = 1

        for var in x_variable_list:
            print(flag, var)
            vals = df[var].tolist()
            plt.subplot(4, 4, flag)
            flag += 1
            plt.hist(vals, bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()


        pass
    def filter_percentile(self,df):
        dic_start={}
        dic_end={}

        x_list=self.x_variable_list

        percentiles = [5, 95]
        for x in x_list:
            values=df[x].tolist()
            values=np.array(values)
            values_mask=~np.isnan(values)
            values_nonan=values[values_mask]

            percentile_values = np.percentile(values_nonan, percentiles)
            dic_start[x]=percentile_values[0]
            dic_end[x]=percentile_values[1]
            df = df[(df[x] >= percentile_values[0]) & (df[x] <= percentile_values[1])]
        return df, dic_start, dic_end




    def AIC_stepwise(self,dff,initial_list=[],
                           threshold_in=0.01,
                           threshold_out=0.05,
                           verbose=True):
        import statsmodels.api as sm
        import pandas as pd

        import itertools


        df=T.load_df(dff)
        df=self.df_clean(df)
        df=df.dropna()


        x_list = self.x_variable_list


        y = df[self.y_variable]
        ## exclude nan

        X=df[x_list]

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_clean = X[mask]
        y_clean = y[mask]


        included = list(initial_list)
        while True:
            changed = False

            # forward step
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded, dtype=float)
            for new_column in excluded:
                model = sm.OLS(y_clean, sm.add_constant(pd.DataFrame(X_clean[included + [new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f'Add  {best_feature:20} with p-value {best_pval:.6f}')

            # backward step
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f'Drop {worst_feature:20} with p-value {worst_pval:.6f}')
            if not changed:
                break

        final_model = sm.OLS(y, sm.add_constant(X[included])).fit()
        if verbose:
            print("\nFinal model AIC:", final_model.aic)
            print("Selected variables:", included)
        return included, final_model



    def variable_list_rt(self):

        self.x_variable_list = [
             'srad_summer_npy_detrend',
            'ppt_winter_npy_detrend',
            # 'SM_L4_summer_npy_trend',
            # 'spring_LAI_detrend',

            'tmax_summer_npy_detrend',
            'ppt_summer_npy_detrend',

            # 'vpd_summer_npy_trend',
            # 'ppt_summer_npy_trend',
            # 'ppt_spring_npy_trend',
           'summer_rainfall_intensity',
           #  'tmean_mean',
            # 'ppt_winter_npy_trend',
            # 'soil_summer_npy_trend'



            ]
        self.x_variable_range_dict_global = {
             "post1_tmax_mean": [-1,-1.2],

                "during_SPI12_mean": [-3.5, -1.6],
                "pre_tmax_mean": [-1, 1.3],
                "CV_trend_before_5_year": [-3, 5],

                'post1_ppt_mean': [-.8, 1.3],

                'NDVI_pre2_mean': [-2.5,2.5],
                'historic_resilience':[-2.8,1.6],
                'pre_ppt_mean':[-1,1],
            'Aridity':[0.05,0.65],

                'AR1_trend_before_5_year':[-0.15,0.15],
                'historic_drought_mean_since1982':[-2,2],


        }




    def show_colinear(self, ):
        dff = self.dff
        df = T.load_df(dff)
        vars_list = self.x_variable_list
        df = df[vars_list]
        ## add LAI4g_raw
        # df['composite_LAI_beta_mean_zscore'] = T.load_df(dff)['composite_LAI_beta_mean_zscore']
        ## plot heat map to show the colinear variables

        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': 'Rainfall frequency (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': 'Heat event frequency (events/year)',
                    'cwdx80_05': 'Rooting zone water storage capacity (mm)',
                    'sand': 'Sand (g/kg)',

                    }

        import seaborn as sns
        fig, ax=plt.subplots(figsize=(8, 5))
        ### x tick label rotate
        vmin = -1
        vmax = 1


        sns.heatmap(df.corr(), annot=True, fmt=".2f",vmin=vmin, vmax=vmax,cmap="RdBu")
        plt.xticks(rotation=45)
        ax.set_yticks(np.arange(len(vars_list)) + 0.5)
        # ax.set_yticklabels(model_list[::-1], rotation=0, va='center')
        ##get name from dic
        # ax.set_yticklabels([name_dic[x] for x in vars_list], rotation=0, va='center')
        #
        # ax.set_xticks(np.arange(len(vars_list)) + 0.5)
        # ax.set_xticklabels([name_dic[x] for x in vars_list], rotation=45, ha='center')
        # ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    def discard_vif_vars(self, df, x_vars_list):
        ##################实时计算#####################
        vars_list_copy = copy.copy(x_vars_list)

        X = df[vars_list_copy]
        X = X.dropna()
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif.round(1)
        selected_vif_list = []
        for i in range(len(vif)):
            feature = vif['features'][i]
            VIF_val = vif['VIF Factor'][i]
            if VIF_val < 5.:
                selected_vif_list.append(feature)
        return selected_vif_list

        pass

    def plot_hist(self, df):
        # T.print_head_n(df)
        # exit()
        x_variable_list = self.x_variable_list
        ## combine x and y
        all_list = copy.copy(x_variable_list)
        all_list.append(self.y_variable)
        # print(all_list)
        # exit()
        for var in all_list:
            vals = df[var].tolist()
            vals = np.array(vals)
            # vals[vals<-500] = np.nan
            # vals[vals>500] = np.nan
            # vals = vals[~np.isnan(vals)]
            plt.hist(vals, bins=100)
            plt.title(var)
            plt.show()
        exit()
        return df

    def valid_range_df(self, df):

        print('original len(df):', len(df))
        for var in self.x_variable_list_CRU:

            if not var in df.columns:
                print(var, 'not in df')
                continue
            min, max = self.x_variable_range_dict[var]
            df = df[(df[var] >= min) & (df[var] <= max)]
        print('filtered len(df):', len(df))
        return df

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()

        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 30]
        df = df[df['lat'] < 45]
        # df = df[df["summer_class_3"].isin([1, 2, 3])]

        # eco_region_list = ['Western US', 'Western Cordillera', 'Upper Gila Mountains',
        #                    'Warm Desert', 'Cold Desert', 'Western Sierra Madre Piedmont']
        # #
        # df = df[df['landcover_classfication'] != 'Cropland']
        return df

    pass
        # #

    def check_spatial_plot(self):

        dff = self.dff
        df=T.load_df(dff)
        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        region_arr = DIC_and_TIF(pixelsize=.5).pix_dic_to_spatial_arr(unique_pix_list)
        plt.imshow(region_arr, cmap='jet', vmin=1, vmax=3,interpolation='nearest')
        plt.colorbar()
        plt.show()

    def pdp_shap(self):
        import joblib


        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap')

        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df = T.load_df(dff)
        df = self.df_clean(df)



        eco_region_list = [ 'Western Cordillera', 'Upper Gila Mountains',
                           'Warm Desert', 'Cold Desert', 'Western Sierra Madre Piedmont', ]
        eco_region_list = ['Western US',   ]

        for eco in eco_region_list:

            if eco == 'Western US':
                # 2. Use a single '=' for assignment, and handle the logic
                df_i = df.copy()
            else:
                df_i = df[df['Ecoregion_level_II'] == eco]

            pix_list = df_i['pix'].tolist()
            unique_pix_list = list(set(pix_list))
            spatial_dic={}

            for pix in unique_pix_list:
                spatial_dic[pix] = 1
            arr=D.pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
            plt.colorbar()
            plt.show()



            T.print_head_n(df_i)
            # print(len(df))
            # T.print_head_n(df)
            print('-' * 50)
            ## text select df the first 1000


            all_vars = copy.copy(x_variable_list)
            #
            #
            all_vars.append(y_variable)  # add the y variable to the list
            all_vars.append('pix')
            # all_vars.extend(['greening_trend_before_10year',
            #     'AR1_trend_before_10year',
            #     'CV_trend_before_10year',
            #     ])
            #
            #
            #
            all_vars_df = df_i[all_vars]  # get the dataframe with the x variables and the y variable

            all_vars_df = all_vars_df.dropna(subset=x_variable_list,  )
            all_vars_df = all_vars_df.dropna(subset=self.y_variable, )
            # print('len(all_vars_df):', len(all_vars_df));exit()


            ######


            pix_list = all_vars_df['pix'].tolist()
            # print(len(pix_list));exit()
            unique_pix_list = list(set(pix_list))
            spatial_dic = {}
            #
            for pix in unique_pix_list:
                spatial_dic[pix] = 1
            arr = D.pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.show()


            X = all_vars_df[x_variable_list].copy()
            Y = all_vars_df[y_variable]



            model, y, y_pred = self.__train_model(X, Y)  # train a Random Forests model

            explainer = shap.TreeExplainer(model)


            shap_values_all = explainer(X)

            # === 把 SHAP 加回 df ===
            for i, col in enumerate(x_variable_list):
                all_vars_df[f'shap_{col}'] = shap_values_all.values[:, i]

            outf=join(outdir, self.y_variable + '_shap_all.df')
            T.save_df(all_vars_df,outf)
            T.df_to_excel(all_vars_df, outf.replace('.df','.xlsx') )
            # exit()

            ## 这部分是 我存起来画 一部分图的

            # sample_size = min(5000, len(X))
            # X_sample = X.sample(sample_size, random_state=42)


            shap_values_samples = explainer(X)
            outf_shap = join(outdir, self.y_variable +  f'_{eco}_shap.npy')
            np.save(outf_shap, shap_values_samples.values)

            joblib.dump(
                {
                    "X": X,
                    "shap": shap_values_samples.values,
                    "columns": X.columns
                },
                join(outdir, 'shap_bundle.pkl')
            )


            # outpkl=join(outdir, self.y_variable + '.pkl')
            # T.save_dict_to_binary(shap_values_samples, outpkl)


            # shap_interaction_values = explainer.shap_interaction_values(X_sample)
            # feature_names = X_sample.columns.tolist()
            #
            # i_ar1 = feature_names.index("AR1_trend_before_10year")
            # i_spi = feature_names.index("post1_SPI12_mean")
            # interaction = shap_interaction_values[:, i_ar1, i_spi]
            #
            # plt.figure(figsize=(6, 5))
            #
            # plt.scatter(
            #     X_sample["AR1_trend_before_10year"],
            #     interaction,
            #     c=X_sample["post1_SPI12_mean"],
            #     cmap="RdBu",
            #     alpha=0.6
            # )
            #
            # plt.colorbar(label="Post SPI")
            # plt.xlabel("AR1")
            # plt.ylabel("Interaction effect (AR1 × SPI)")

            # plt.title("SHAP interaction")

            # plt.tight_layout()
            # plt.show()

            # save shap values


    def plot_pdp_shap(self):
        import joblib

        dff=self.dff

        df = T.load_df(dff)
        df = self.df_clean(df)
        df_temp, start_dic, end_dic = self.filter_percentile(df)
        fdir=join(self.this_class_png, 'pdp_shap')
        for f in T.listdir(fdir):

            if not f.endswith('.npy'):
                continue
            print(f)

            inf_shap = join(fdir, f)
            shap_values = np.load(inf_shap, allow_pickle=True)


            # print(shap_values);exit()
            x_variable_list = self.x_variable_list
            # pprint(shap_values)
            # exit()


            imp = np.abs(shap_values).mean(axis=0)
            # imp = np.abs(shap_values)
            # pprint(imp)
            # exit()

            imp_dict = dict(zip(x_variable_list, imp))
            # pprint(imp_dict)
            # exit()

            # 按importance排序
            sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)

            x_list = [i[0] for i in sorted_imp]
            y_list = [i[1] for i in sorted_imp]

            plt.figure()

            plt.barh(x_list[::-1], y_list[::-1],
                     color='grey', alpha=0.6)

            plt.xlabel("mean |SHAP value|", fontsize=12)
            plt.title("SHAP importance")

            plt.tight_layout()
            plt.show()

            # data = pd.read_pickle(
            #     join(self.this_class_png, 'pdp_shap', self.y_variable + '.pkl')
            # )
            file=join(self.this_class_png, 'pdp_shap', 'shap_bundle.pkl')
            bundle = joblib.load(file)
            data_X = bundle["X"]  # DataFrame (n_samples, n_features)
            shap_values = bundle["shap"]  # numpy array (n_samples, n_features)



            flag = 1
            centimeter_factor = 1 / 2.54
            plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))

            for x_var in x_list:

                idx = list(data_X.columns).index(x_var)

                data_i = data_X[x_var].values
                value_i = shap_values[:, idx]

                df_i = pd.DataFrame({
                    x_var: data_i,
                    'shap_v': value_i
                })


                start = start_dic[x_var]
                end = end_dic[x_var]

                bins = np.linspace(start, end, 50)

                df_group, bins_list_str = T.df_bin(df_i, x_var, bins)

                y_mean_list = []
                x_mean_list = []
                y_err_list = []

                df_i_copy = df_i[(df_i[x_var] > start) & (df_i[x_var] < end)]

                scatter_x_list = df_i_copy[x_var].tolist()
                scatter_y_list = df_i_copy['shap_v'].tolist()

                for name, df_group_i in df_group:

                    x_i = name[0].left

                    vals = df_group_i['shap_v'].tolist()

                    if len(vals) == 0:
                        continue

                    mean = np.nanmedian(vals)
                    err = np.nanstd(vals)

                    y_mean_list.append(mean)
                    x_mean_list.append(x_i)
                    y_err_list.append(err)

                plt.subplot(3, 3, flag)

                plt.scatter(
                    scatter_x_list,
                    scatter_y_list,
                    alpha=0.2,
                    c='gray',
                    marker='.',
                    s=1,
                    zorder=-1
                )

                y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=11)

                # name_dic = {'srad_summer_npy_trend': 'Incoming solar radiation trend (W/m²)',
                #             'soil_summer_npy_trend': 'Soil moisture trend(Kpa)',
                #             'ppt_winter_npy_anomaly':'Winter_precip anomaly (mm)',
                #             'ppt_summer_npy_anomaly':'Summer_precip anomaly (mm)',
                #             'tmax_summer_npy_trend':'Tmax trend(degree)',
                #             'summer_rainfall_amount_trend': 'Rainfall amount trend',
                # 'summer_rainfall_intensity_trend': 'Rainfall intensity trend',
                #             'ppt_winter_npy_mean':'Winter_precip mean',
                #             'ppt_winter_npy_trend':'Winter_precip trend',
                #             'tmean_mean':'MAT',
                #             'SWE_winter_npy_trend':'Winter SWE trend',
                #             'SM_L1_summer_npy_trend':'Surface SM trend',
                #             'SM_L4_summer_npy_trend':'Root zone SM trend',
                #             'ppt_summer_npy_trend':'Summer_precip trend',
                #             'summer_rainfall_fq_trend':'Rainfall frequency trend',
                #             'ppt_spring_npy_trend':'Spring_precip trend',
                #             'vpd_summer_npy_trend':'Summer VPD trend',
                #
                #             ' summer_SPEI3_trend':'Summer SPEI3 trend',
                #             ' spring_SPEI3_trend': 'Spring SPEI3 trend',
                #
                #
                #             }

                plt.plot(x_mean_list, y_mean_list, c='blue')

                # plt.xlabel(name_dic[x_var])
                plt.ylabel('Spring LAI anomaly (m2/m2)')

                flag += 1

                plt.ylim(-1,1)
            region=f.split('.')[0]

            plt.suptitle(region)

            plt.tight_layout()

            plt.show()
            # plt.savefig(outf,dpi=300)
            # plt.close()

    def plot_shaply_under_different_condition(self):
        ## read shaply values
        dff=join(self.this_class_png, 'pdp_shap','post_4_year_NDVI_SNU_anomal_detrend_shap_all.df')
        df=T.load_df(dff)

        x_variable_list =self.x_variable_list


        state_factor = 'CV_trend_before_10year'
        quantiles = np.linspace(0.05, 0.95, 7)
        ar1_bins = df[state_factor].quantile(quantiles).values
        plt.figure(figsize=(6, 5))
        for feature in x_variable_list:

            for i in range(len(ar1_bins) - 1):

                low = ar1_bins[i]
                high = ar1_bins[i + 1]

                df_temp = df[
                    (df[state_factor] >= low) &
                    (df[state_factor] < high)
                    ]

                if len(df_temp) < 50:
                    continue

                sns.regplot(
                    x=df_temp[feature],
                    y=df_temp[f'shap_{feature}'],
                    lowess=True,
                    scatter=False,
                    label=f'{round(low, 2)}–{round(high, 2)}'
                )

            plt.xlabel(feature)
            plt.ylabel('SHAPLY')

            plt.legend(title='AR1 percentile', fontsize=8)
            plt.ylim(-0.2,0.2)

            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        pass


    def plot_pdp_shap_density_cloud(self):
        x_variable_list = self.x_variable_list

        name_dic={'rainfall_intensity':'Rainfall intensity (mm/events)',
                  'rainfall_frenquency':'Rainfall frequency (events/year)',
                  'rainfall_seasonality_all_year':'Rainfall seasonality (unitless)',
                  'detrended_sum_rainfall_CV':r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                  'heat_event_frenquency':'Heat event frequency (events/year)',
                  'cwdx80_05':'Rooting zone water storage capacity (mm)',
                  'pi_average':'SM-Tcoupling (unitless)',
                  'fire_weighted_ecosystem_year_average':'Fire  (unitless)',
                  'rooting_depth':'Root depth (m)',

                  'sand':'Sand (g/kg)',

        }

        # inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        inf_shap = join(self.this_class_png, 'pdp_shap_beta', self.y_variable + '.shap.pkl')

        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])

        flag = 1
        centimeter_factor = 1 / 2.54
        # plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        fig, axs = plt.subplots(4, 2,
                                figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        # print(axs);exit()
        axs = axs.flatten()
        for x_var in x_list:
            shap_values_mat = shap_values[:, x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]

            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)

            percentiles = [5, 95]
            ## datapoints percentile
            percentile_values = np.percentile(scatter_x_list, percentiles)
            print(percentile_values)

            # plt.subplot(4, 3, flag)
            ax = axs[flag-1]
            ax.vlines(percentile_values, -7, 7, color='gray', linestyle='--', alpha=1)

            # ax2 = ax.twiny()  # Create a twin x-axis
            # ax2.set_xlim(ax.get_xlim())  # Match the limits with the main axis
            # ax2.set_xticks(percentile_values)  # Set percentile values as ticks
            # ax2.set_xticklabels([f'{p}%' for p in percentiles])  # Label with percentiles


            KDE_plot().plot_scatter(scatter_x_list, scatter_y_list,ax=ax )

            y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
            ax.plot(x_mean_list, y_mean_list, c='red', alpha=1)

            # ax.set_title(name_dic[x_var], fontsize=12)
            ax.set_xlabel(name_dic[x_var], fontsize=12)
            ax.set_ylabel(r'CV$_{\mathrm{LAI}}$ (%/year)', fontsize=12)

            flag += 1
            ax.set_ylim(-3, 3)
            # plt.show()


        plt.suptitle(self.y_variable)

        plt.tight_layout()
        plt.show()
        # plt.savefig(outf,dpi=300)
        # plt.close()

    def plot_pdp_shap_density_cloud_individual(self,line=False    ,scatter=True  ):
        from statsmodels.nonparametric.smoothers_lowess import lowess


        x_variable_list = self.x_variable_list

        name_dic={'post1_tmax_mean':'Tmax post drought',
                  'post1_ppt_mean':'Precip post drought',
                  'during_SPI12_mean':'drought severity',
                  'pre_tmax_mean':'Tmax pre drought',
                  'CV_trend_before_5_year':'CV trend before drought',
                  'Aridity':'Aridity',
                  'NDVI_pre2_mean':'NDVI pre drought',
                  'historic_resilience':'Historic Drought resilience',
                  'pre_ppt_mean':'Precip pre drought',
                  'AR1_trend_before_5_year':'AR1 trend before drought',
                  'historic_drought_mean_since1982':'Historic drought severity',





        }
        inf_shap = join(self.this_class_png, 'pdp_shap', self.y_variable + '.pkl')


        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])

        flag = 1
        centimeter_factor = 1 / 2.54
        # plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        # fig, axs = plt.subplots(4, 2,
        #                         figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        # print(axs);exit()
        # axs = axs.flatten()
        for x_var in x_list:
            shap_values_mat = shap_values[x_var]
            data_i = shap_values_mat
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]

            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)

            percentiles = [5, 95]
            ## datapoints percentile
            percentile_values = np.percentile(scatter_x_list, percentiles)
            print(percentile_values)

            # plt.subplot(4, 3, flag)
            # ax = axs[flag-1]
            # fig = plt.figure(figsize=(5*centimeter_factor,3*centimeter_factor))
            fig,ax = plt.subplots(1,1,figsize=(8*centimeter_factor,5*centimeter_factor))
            ax.vlines(percentile_values, -5, 5, color='gray', linestyle='--', alpha=1)

            # y_lims = {
            #     "post1_tmax_mean": [-0.5,0.5],
            #
            #     "during_SPI12_mean": [-3.5, -1.6],
            #     "pre_tmax_mean": [-1, 1.3],
            #     "CV_trend_before_5_year": [-3, 5],
            #
            #     'post1_ppt_mean': [-.8, 1.3],
            #
            #     'NDVI_pre2_mean': [-2.5,2.5],
            #     'historic_resilience':[-2.8,1.6],
            #     'pre_ppt_mean':[-1,1],
            #
            #     'AR1_trend_before_5_year':[-0.15,0.15],
            #     'historic_drought_mean_since1982':[-2,2],
            #
            # }

            if scatter:
                KDE_plot().plot_scatter(scatter_x_list, scatter_y_list,ax=ax )
                plt.axis('off')

            if line:
                # y_mean_list= lowess(y_mean_list, x_mean_list, frac=0.1)
                y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=13)
                # y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
                ax.plot(x_mean_list, y_mean_list, c='red', alpha=1)

                # ax.set_title(name_dic[x_var], fontsize=12)
                ax.set_xlabel(name_dic[x_var], fontsize=12)
                ax.set_ylabel(r'Beta (%/100mm)', fontsize=12)

            # flag += 1
            # ax.set_ylim(y_lims[x_var])
            ## add line when y=0
            # ax.axhline(0, c='black', linestyle='-', alpha=1)

            plt.show()

            # outdir=join(self.this_class_png, 'pdp_shap_beta2', 'pdf_cloud')
            # T.mk_dir(outdir, force=True)

            # outf = join(outdir, f'{x_var}.pdf')
            # plt.savefig(outf,dpi=300)
            # plt.close()


        #
        # plt.tight_layout()
        # plt.show()

    def plot_pdp_shap_density_cloud_individual_test(self,line=False    ,scatter=True  ):
        from statsmodels.nonparametric.smoothers_lowess import lowess


        x_variable_list = self.x_variable_list

        name_dic={
                  'rainfall_frenquency':'Rainfall frequency (events/year)',
                  'rainfall_seasonality_all_year':'Rainfall seasonality (unitless)',
            'heavy_rainfall_days':'Heavy rainfall days (days/year)',

                  'VPD':'VPD (kPa)',
                  'sum_rainfall':'Total rainfall (mm)',
                  'Aridity':'Aridity (unitless)',

                  'heat_event_frenquency':'Heat event frequency (events/year)',
            'FVC_relative_change_trend':'Changes in vegetation cover (%/yr)',



                  'fire_ecosystem_year':'Fire burn area(km2)',
                  'rooting_depth':'Rooting depth (cm)',
                  'rainfall_intensity_trend':'Rainfall intensity trend (mm/events/yr)',
                  'rainfall_frenquency_trend':'Rainfall frequency trend (events/yr)',
                  'cwdx80_05':'S0 (mm)',
                  'Burn_area_mean':'Fire burn area(km2)',
                  'Non tree vegetation_trend':'Changes in Short vegetation cover (%/yr)',

                  'fire_ecosystem_year_average_trend':'Fire burn area trend (km2/yr)',
                  'rainfall_seasonality_all_year_trend':'Rainfall seasonality trend (unitless/yr)',



        }
        inf_shap = join(self.this_class_png, 'pdp_shap_beta_ALL_sig2', self.y_variable + '.shap.pkl')


        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])

        flag = 1
        centimeter_factor = 1 / 2.54
        # plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        # fig, axs = plt.subplots(4, 2,
        #                         figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        # print(axs);exit()
        # axs = axs.flatten()
        for x_var in x_list:
            shap_values_mat = shap_values[:, x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]

            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)

            percentiles_95 = [5, 95]
            ## datapoints percentile
            percentile_95_values = np.percentile(scatter_x_list, percentiles_95)
            print(percentile_95_values)
            percentiles_75=[25,75]
            percentiles_75_values=np.percentile(scatter_x_list,percentiles_75)


            # plt.subplot(4, 3, flag)
            # ax = axs[flag-1]
            # fig = plt.figure(figsize=(5*centimeter_factor,3*centimeter_factor))
            fig,ax = plt.subplots(1,1,figsize=(8*centimeter_factor,5*centimeter_factor))
            # ax.vlines(percentile_values, -5, 5, color='gray', linestyle='--', alpha=1)
            ax.vlines(percentiles_75_values, -5, 5, color='gray', linestyle='--', alpha=1)


        ## set x_lims
            y_lims = {
                "rainfall_intensity": [-4, 4],
                "rainfall_frenquency": [-6, 6],
                "rainfall_seasonality_all_year": [-1.5, 1.5],
                "sand": [-0.5, 0.5],
                'soc':[-0.5,0.5],
                'VPD':[-5,5],
                'heavy_rainfall_days':[-6,6],
                'sum_rainfall':[-6,6],
                'VOD_detrend_min':[-3,3],
                'Aridity':[-1,1],
                'FVC_relative_change_trend':[-2,2],
                'heat_event_frenquency':[-1.5,1.5],

                'pi_average_trend': [-0.5,0.5],
                'fire_ecosystem_year':[-1,0.5],
                'rooting_depth':[-0.5,0.5],
                'cwdx80_05':[-0.5,0.5],

                'fire_ecosystem_year_average_trend':[-0.5,0.5],
                'rainfall_seasonality_all_year_trend':[-2,2],
                'rainfall_intensity_trend':[-1,1],
                'rainfall_frenquency_trend':[-1,1],
                'Burn_area_mean':[-1.5,1.5],

            }

            if scatter:
                KDE_plot().plot_scatter(scatter_x_list, scatter_y_list,ax=ax )
                plt.axis('off')

            if line:
                # y_mean_list= lowess(y_mean_list, x_mean_list, frac=0.1)
                y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=9)
                # y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
                ax.plot(x_mean_list, y_mean_list, c='red', alpha=1)
                ax.hlines(0, x_mean_list[0], x_mean_list[-1], color='gray', linestyle='--', alpha=1)


                # ax.set_title(name_dic[x_var], fontsize=12)
                ax.set_xlabel(name_dic[x_var], fontsize=12)
                ax.set_ylabel(r'Beta (%/100mm)', fontsize=12)

            # flag += 1
            ax.set_ylim(y_lims[x_var])
            ax.set_xlim(percentile_95_values[0],percentile_95_values[1])
            ## add line when y=0
            # ax.axhline(0, c='black', linestyle='-', alpha=1)

            # plt.show()

            outdir=join(self.this_class_png, 'pdp_shap_beta_ALL_sig2', 'pdf_cloud')
            T.mk_dir(outdir, force=True)

            outf = join(outdir, f'{x_var}.png')
            plt.savefig(outf,dpi=300)
            plt.close()


        #
        # plt.tight_layout()
        # plt.show()





    def plot_pdp_shap_all_models_main(self): ### plot all models in main
        fdir_all=results_root+rf'\3mm\SHAP\\'

        all_model_results = {}
        model_list = ['LAI4g',  'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai','LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',
                          'SDGVM_S2_lai',
                          'YIBs_S2_Monthly_lai']

        for model in model_list:

            fdir = join(fdir_all, rf'RF_{model}_detrend_CV_')

            for fdir_ii in T.listdir(fdir):


                for f in T.listdir(join(fdir, fdir_ii)):

                    if not '.shap.pkl' in f:
                        continue

                    inf_shap = join(fdir, fdir_ii, f)

                    shap_values = T.load_dict_from_binary(inf_shap)
                    print(shap_values)
                    x_list=['rainfall_intensity','rainfall_frenquency','detrended_sum_rainfall_CV','heat_event_frenquency', 'rainfall_seasonality_all_year',
                            'sand','cwdx80_05',]

                    # imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
                    # x_list = []
                    # y_list = []
                    # for key in imp_dict.keys():
                    #     x_list.append(key)
                    #     y_list.append(imp_dict[key])
                    result_dic_X = {}
                    result_dic_Y = {}
                    result_dic_err = {}
                    for x_var in x_list:
                        shap_values_mat = shap_values[:, x_var]
                        data_i = shap_values_mat.data
                        value_i = shap_values_mat.values
                        df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
                        # pprint(df_i);exit()
                        df_i_random = df_i.sample(n=len(df_i) )
                        df_i = df_i_random


                        start, end = self.x_variable_range_dict[x_var]

                        bins = np.linspace(start, end, 50)
                        df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
                        y_mean_list = []
                        x_mean_list = []
                        y_err_list = []
                        df_i_copy = copy.copy(df_i)
                        df_i_copy = df_i_copy[df_i_copy[x_var]>start]
                        df_i_copy = df_i_copy[df_i_copy[x_var]<end]

                        for name, df_group_i in df_group:
                            x_i = name[0].left

                            vals = df_group_i['shap_v'].tolist()

                            if len(vals) == 0:
                                continue
                            # mean = np.nanmean(vals)
                            mean = np.nanmedian(vals)
                            err = np.nanstd(vals)
                            y_mean_list.append(mean)
                            x_mean_list.append(x_i)
                            y_err_list.append(err)

                        result_dic_X[x_var] = x_mean_list
                        result_dic_Y[x_var] = y_mean_list
                        result_dic_err[x_var] = y_err_list
                    all_model_results[f]=result_dic_X,result_dic_Y,result_dic_err

            ### plot all models



        flag = 1
        centimeter_factor = 1 / 2.54
        rows=2
        cols=4

        color_list=['black', 'red', 'blue', 'purple', 'orange', 'greenyellow',  'gray',
                      'yellow', 'pink', 'brown', 'cyan', 'magenta', 'goldenrod', 'teal', 'lavender', 'maroon', 'navy',
                      'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive',
                      'silver', 'aqua', 'fuchsia']

        y_scale_list = [1,1,1,1,1]

        linewidth_list=[2]
        linewidth_list.extend([1]*20)
        alpha_list=[1]
        alpha_list.extend([0.6]*20)

        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': 'Rainfall frequency (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': 'Heat event frequency (events/year)',
                    'cwdx80_05': 'Rooting zone water storage capacity (mm)',

                    'sand': 'Sand (g/kg)',

                    }



        plt.figure(figsize=(cols * 8 * centimeter_factor, rows * 6 * centimeter_factor))
        y_lims = {
            "rainfall_intensity": [-15, 10],
            "rainfall_frenquency": [-15, 20],
            'detrended_sum_rainfall_CV': [-15, 40],
            "heat_event_frenquency": [-5, 8],
            "rainfall_seasonality_all_year": [-2, 10],
            "sand": [-10, 20],
            "cwdx80_05": [-5, 15],
        }


        for x_var in x_list:
            color_flag = 1
            plt.subplot(rows, cols, flag)

            for f in all_model_results.keys():

                result_dic_X,result_dic_Y,result_dic_err = all_model_results[f]

                x_mean_list = result_dic_X[x_var]
                y_mean_list = result_dic_Y[x_var]
                y_err_list = result_dic_err[x_var]

                y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)

                zorder_list=[1]
                zorder_list.extend([0]*20)

                plt.plot(x_mean_list, y_mean_list, c= color_list[color_flag-1], linewidth=linewidth_list[color_flag-1],zorder=zorder_list[color_flag-1])
                plt.xlabel(name_dic[x_var], fontsize=12)
                ## y_lims
                plt.ylim(y_lims[x_var])
                color_flag+=1
            flag += 1

    # plt.suptitle(self.y_variable)
            plt.tight_layout()
        plt.savefig(join(self.this_class_png, 'pdp_shap_all_models_SI.pdf'))
        plt.show()



    def rgb_to_hex(self,r, g, b):
        """
        Converts RGB color values (0-255) to a hexadecimal string.

        Args:
          r: The red component (integer, 0-255).
          g: The green component (integer, 0-255).
          b: The blue component (integer, 0-255).

        Returns:
          A string representing the hexadecimal color code (e.g., "#FFA501").
        """
        # Ensure values are within the valid range (0-255)
        if not all(0 <= x <= 255 for x in (r, g, b)):
            raise ValueError("RGB values must be between 0 and 255.")

        return f'#{r:02X}{g:02X}{b:02X}'

    def plot_relative_importance(self):  ## bar plot
        from matplotlib import cm
        import joblib

        ## here plot relative importance of each variable
        x_variable_list = self.x_variable_list

        name_dic = {'post1_tmax_mean': 'Tmax post drought',
                    'post1_ppt_mean': 'Precip post drought',
                    'during_SPI12_mean': 'Drought severity',
                    'pre1_tmax_mean': 'Tmax pre drought',
                    'CV_trend_before_5year': 'CV trend before drought',
                    'Aridity': 'Aridity',
                    'NDVI_pre2_mean': 'NDVI pre drought',
                    'historic_resilience': 'Historic Drought resilience',
                    'pre1_ppt_mean': 'Precip pre drought',
                    'AR1_trend_before_5year': 'AR1 trend before drought',
                    'historic_drought_mean_since1982': 'Historic drought severity',
                    'sand': 'Sand',

                    }



        ## read npy

        # pprint(shap_values);exit()
        # read npy1.   both correct here we use 2
        # inf_shap = join(self.this_class_png, 'pdp_shap', self.y_variable + '_shap.npy')
        # shap_values = np.load(inf_shap, allow_pickle=True)
        # x_variable_list = self.x_variable_list
        #
        # imp = np.abs(shap_values).mean(axis=0)
        #
        # imp_dict = dict(zip(x_variable_list, imp))

        # 按importance排序
        # imp_dict_sort = sorted(imp_dict.items(), key=lambda x: x[1], reverse=False)

        # x_list = [i[0] for i in imp_dict_sort]
        # y_list = [i[1] for i in imp_dict_sort]


        ## read pkl method 2

        file=join(self.this_class_png, 'pdp_shap', 'shap_bundle.pkl')

        inf_shap = joblib.load(file)

        # X = inf_shap["X"]
        shap_values = inf_shap["shap"]
        columns = inf_shap["columns"]

        shap_df = pd.DataFrame(shap_values, columns=columns)


        sum_abs_shap_dic = {}

        for col in shap_df.columns:
            sum_abs_shap_dic[col] = np.mean(np.abs(shap_df[col]))

        total_sum = sum(sum_abs_shap_dic.values())
        #
        relative_importance = {
            var: val / total_sum * 100
            for var, val in sum_abs_shap_dic.items()
        }
        #
        imp_dict_sort = sorted(
            relative_importance.items(),
            key=lambda x: x[1]
        )

        x_list_name_sort = [x[0] for x in imp_dict_sort]
        y_list = [x[1] for x in imp_dict_sort]
        #
        x_list = [name_dic[x] for x in x_list_name_sort]



        group_dic = {
            'Drought event climate': [
                'during_SPI12_mean',
                'post1_tmax_mean',
                'post1_ppt_mean',
                'pre1_tmax_mean',
                'pre1_ppt_mean',
            ],

            'Long-term background': [
                'Aridity',
                'sand',
                'historic_drought_mean_since1982',
            ],

            'Ecosystem state': [
                'NDVI_pre2_mean',
                'historic_resilience',
                'CV_trend_before_5year',
                'AR1_trend_before_5year',
            ]
        }


        x_list_name_sort = [x[0] for x in imp_dict_sort]



        var_group = {}
        for group, vars_ in group_dic.items():
            for v in vars_:
                var_group[v] = group

        group_color = {
            'Drought event climate': '#eca9aa',  # 红（气候冲击）
            'Long-term background': '#4575b4',  # 蓝（环境背景）
            'Ecosystem state': '#81c7bc'  # 绿（生态系统）
        }

        colors = [
            group_color[var_group[x]] if x in var_group else 'grey'
            for x in x_list_name_sort
        ]
        plt.barh(
            x_list,
            y_list,
            color=colors,
            alpha=0.8,
            edgecolor='black'
        )


        plt.xticks(fontsize=12)
        plt.xlabel('Importance (%)', fontsize=12)
        ## add text R2=0.89 in (0.5, 0.5)
        plt.text(10, 0.1, 'R2=0.66', fontsize=12)
        plt.tight_layout()
        #
        plt.show()
        ## Save pdf
        # plt.savefig(join(self.this_class_png, 'pdp_shap_beta_ALL_sig2', self.y_variable + '_importance_bar.pdf'), dpi=300,)




        pass

    def plot_relative_importance_pie_plot(self):  ## bar plot


        inf_shap = join(self.this_class_png, 'pdp_shap', self.y_variable + '.pkl')

        shap_values = T.load_dict_from_binary(inf_shap)
        sum_abs_shap_dic = {}

        for col in shap_values.columns:
            sum_abs_shap_dic[col] = np.mean(np.abs(shap_values[col]))

        total_sum = sum(sum_abs_shap_dic.values())

        relative_importance = {
            var: val / total_sum * 100
            for var, val in sum_abs_shap_dic.items()
        }



        group_dic = {
            'Drought event climate': [
                'during_SPI12_mean',
                'post1_tmax_mean',
                'post1_ppt_mean',
                'pre1_tmax_mean',
                'pre1_ppt_mean',
            ],

            'Long-term background': [
                'Aridity',
                'sand',
                'historic_drought_mean_since1982',
            ],

            'Ecosystem state': [
                'NDVI_pre2_mean',
                'historic_resilience',
                'CV_trend_before_5year',
                'AR1_trend_before_5year',
            ]
        }
        ## calculate group importance by summing variable importance in each group
        pie_dic={}
        for group, vars_ in group_dic.items():
            group_importance = sum([relative_importance[v] for v in vars_ if v in relative_importance])
            pie_dic[group] = group_importance



        group_color = {
            'Drought event climate': '#eca9aa',  # 红（气候冲击）
            'Long-term background': '#4575b4',  # 蓝（环境背景）
            'Ecosystem state': '#81c7bc'  # 绿（生态系统）
        }

        colors = [group_color[group] for group in pie_dic.keys()]
        labels = pie_dic.keys()
        sizes = [pie_dic[group] for group in pie_dic.keys()]
        plt.pie(sizes,  colors=colors, autopct='%1.1f%%', startangle=90)

        plt.xticks(fontsize=12)


        plt.tight_layout()
        #
        plt.show()
        ## Save pdf
        # plt.savefig(join(self.this_class_png, 'pdp_shap_beta_ALL_sig2', self.y_variable + '_importance_bar.pdf'), dpi=300,)




        pass




    def spatial_shapely_vs_aridity(self):  #### spatial plot

        dff = self.dff
        outdir =join(self.this_class_png, 'pdp_shap_beta11','spatial_shapely_sum')
        T.mk_dir(outdir, force=True)

        # T.open_path_and_file(outdir)
        # exit()

        x_variable_list = self.x_variable_list

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df_origin = T.load_df(dff)
        df_origin = self.df_clean(df_origin)
        # df_origin = self.valid_range_df(df_origin)
        # df_origin = df_origin.iloc(sample_indices)


        pix_list = T.get_df_unique_val_list(df_origin, 'pix')
        spatial_dict = {}
        for pix in pix_list:
            spatial_dict[pix] = 1
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr, interpolation='nearest', cmap='jet')
        # plt.colorbar()
        # plt.show()

        all_vars = copy.copy(x_variable_list)

        all_vars.append(y_variable)  # add the y variable to the list
        all_vars.append('pix')

        all_vars_df = df_origin[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='any')
        all_vars_df = all_vars_df.dropna(subset=self.y_variable, how='any')

        print(len(df_origin))
        x_variable_list = self.x_variable_list
        inf_shap = join(self.this_class_png, 'pdp_shap_beta11',self.y_variable + '.shap.pkl')
        # print(inf_shap);exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values.shape)
        T.print_head_n(df_origin)
        i=0
        for x_var in x_variable_list:
            print(x_var)

            # shap_values_mat = shap_values[:, x_var]

            col_name = f'{x_var}_shap'
            all_vars_df[col_name] = shap_values[:, x_var].values

            i+=1
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='all')
            # df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # arr = T.
        # T.print_head_n(df_origin)
        df_pix_dict = T.df_groupby(all_vars_df, 'pix')

        for xvar in x_variable_list:
            col_name = f'{xvar}_shap'
            spatial_dict = {}
            for pix in df_pix_dict:
                df_pix = df_pix_dict[pix]
                vals = df_pix[col_name].tolist()
                vals = np.array(vals)


                vals_abs_sum = np.sum(vals)
                vals_abs_sum_mean = vals_abs_sum / len(vals)
                spatial_dict[pix] = vals_abs_sum_mean
            outf = join(outdir, col_name + '.tif')
            DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(spatial_dict, outf)

        T.open_path_and_file(outdir)
        # exit()

    def variable_contributions(self):  ## each variable contribution and the max one
        r2 = .69
        fdir = join(self.this_class_png, 'pdp_shap_beta_ALL_sig', 'spatial_shapely')
        outdir = join(self.this_class_png,'pdp_shap_beta_ALL_sig', 'variable_contributions')
        T.mk_dir(outdir, force=True)
        all_spatial_dict = {}
        keys = []
        for f in T.listdir(fdir):
            # if 'sand' in f:
            #     continue
            # if 'cwdx' in f:
            #     continue
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)

            spatial_dict = DIC_and_TIF(pixelsize=.5).spatial_tif_to_dic(fpath)
            key = f.split('.')[0]
            print(key)
            all_spatial_dict[key] = spatial_dict
            keys.append(key)
        df = T.spatial_dics_to_df(all_spatial_dict)
        sum_val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            val_list = []
            for key in keys:
                val = row[key]
                # print(val)
                val_list.append(val)
            sum_val = np.sum(val_list)
            sum_val_list.append(sum_val)
        df['sum'] = sum_val_list
        new_key_dict = {}
        flag = 1
        for key in keys:
            df[key + '_contrib'] = df[key] / df['sum'] * 100 * r2
            new_key_dict[key + '_contrib'] = flag
            flag += 1
        # pprint(new_key_dict);exit()
        T.print_head_n(df)
        result_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            dict_i = {}
            for new_key in new_key_dict:
                val = row[new_key]
                dict_i[new_key] = val
            pix = row['pix']
            max_key = T.get_max_key_from_dict(dict_i)
            max_key_flag = new_key_dict[max_key]
            max_val = dict_i[max_key]
            result_dict[pix] = {'max_key': max_key, 'max_key_flag': max_key_flag, 'max_val': max_val}
        result_df = T.dic_to_df(result_dict, 'pix')
        outf_max_val = join(outdir, 'max_val_only.tif')
        outf_max_flag = join(outdir, 'max_flag_only.tif')
        max_val_dict = T.df_to_spatial_dic(result_df, 'max_val')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(max_val_dict, outf_max_val)
        max_flag_dict = T.df_to_spatial_dic(result_df, 'max_key_flag')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(max_flag_dict, outf_max_flag)

        legend_f = join(outdir, 'legend.txt')
        fw = open(legend_f, 'w')
        fw.write(str(new_key_dict))
        fw.close()

        T.open_path_and_file(outdir)

    def max_contributions(self):   #### no use
        fdir = join(self.this_class_png, 'pdp_shap_CV', 'variable_contributions')
        outdir = join(self.this_class_png,'pdp_shap_CV','variable_contributions')

        T.mk_dir(outdir, force=True)
        array_list = []
        variable_dict = {}
        flag = 0
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'max' in f:
                continue
            variable = f.split('.')[0]
            variable_dict[flag] = variable
            flag += 1

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir, f))
            array[array < -99] = np.nan

            array_list.append(array)
        pprint(variable_dict)
        # exit()



        array_list = np.array(array_list)
        max_index_matrix= []
        for r in tqdm(range(len(array_list[0]))):
            max_index_matrix_i = []
            for c in range(len(array_list[0][0])):
                vals_list = []
                for arr in array_list:
                    val = arr[r][c]
                    vals_list.append(val)
                if T.is_all_nan(vals_list):
                    max_index_matrix_i.append(np.nan)
                    continue
                max_index = np.argmax(vals_list)
                max_index_matrix_i.append(max_index)
            max_index_matrix.append(max_index_matrix_i)
        max_index = np.array(max_index_matrix)
        # max_index = np.nanargmax(array_list, axis=0)
        # max_index = np.array(max_index, dtype=float)



        # plt.imshow(max_index, interpolation='nearest',vmin=0,vmax=10)
        # plt.colorbar()
        # plt.show()
        # outf = join(outdir, 'max_variable.tif')
        # DIC_and_TIF(pixelsize=.5).arr_to_tif(max_index, outf)

    def plot_dominant_factors_bar(self):  ### insert bar plot
        dff=self.dff

        df=pd.read_pickle(dff)
        df=self.df_clean(df)
        df = df[df['composite_LAI_beta_trend_growing_season'] > 0]

        val_list=[1,2,3,4,5,6]
        dic_name={1:'Fire burn area',2:'Trends in vegetation cover',
                  3:'VPD',4:'heat_event_frenquency',
                  5:'Heavy rainfall days',
                  6:'Rainfall seasonality',


                  }

        color_dic = {'Fire burn area': 'red',
                     'Trends in vegetation cover': 'deepskyblue',
                     'VPD': '#ff6f00',
                     'heat_event_frenquency': '#bb3dc9',
                     'Heavy rainfall days': '#98e16e',
                     'Rainfall seasonality': '#455dca',


                     }
        percentage_list=[]
        percetage_dict={}
        for val in val_list:
            val=df[df['max_flag_only']==val]
            count=len(val)
            ## df=dfis nan
            percetage=count/np.count_nonzero(~np.isnan(df['max_flag_only']))*100
            print(dic_name[val['max_flag_only'].values[0]],percetage)
            percentage_list.append(percetage)


            percetage_dict[dic_name[val['max_flag_only'].values[0]]]=percetage
        print(sum(percentage_list))

        sorted_items = sorted(percetage_dict.items(), key=lambda x: x[1], reverse=True)

        # 拆分成 labels, values, colors
        sorted_labels = [item[0] for item in sorted_items]
        sorted_values = [item[1] for item in sorted_items]
        sorted_colors = [color_dic[label] for label in sorted_labels]
            ## ra

        fig, ax = plt.subplots(figsize=(3, 3))
        for label, value, color in zip(sorted_labels, sorted_values, sorted_colors):
            plt.bar(label, value, color=color)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Percentage')
        plt.tight_layout()
        # plt.show()
        #save pdf
        outdir = join(self.this_class_png, 'pdp_shap_beta_ALL_sig2')
        T.mk_dir(outdir, force=True)
        plt.savefig(join(outdir, 'percentage_dominant_factors.pdf'),dpi=300, bbox_inches='tight')


        ## plot



    def plot_robinson(self):

        # fdir_trend = result_root+rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result\npy_time_series\trend\\'
        fdir_trend = join(self.this_class_png, 'pdp_shap_beta_ALL_sig2','variable_contributions')
        temp_root = join(self.this_class_png, 'pdp_shap_beta_ALL_sig2','Robinson','temp')
        outdir = join(self.this_class_png, 'pdp_shap_beta_ALL_sig2','Robinson',)
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)


        fpath = join(fdir_trend, 'max_flag_only.tif')
        arr = DIC_and_TIF().spatial_tif_to_arr(fpath)
        # plt.imshow(arr, interpolation='nearest',cmap='jet')
        # plt.colorbar()
        # plt.show()

        plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
        m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=0.5, vmax=6.5, is_discrete=True, colormap_n=7,)

        # plt.show()
        outf = join(outdir, 'Robinson.pdf')
        plt.savefig(outf)


    def disentangle2(self):
        ## beta vs rainfall intensity vs domiant factors vs rainfall trend
        dff=rf'D:\Project3\Result\3mm\SHAP_beta\png\RF_composite_LAI_beta\pdp_shap_beta2\\Dataframe.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        # df=df[df['composite_LAI_beta_mean_p_value']<0.05]
        df=df[df['continent']=='Australia']
        # df=df[df['composite_LAI_beta_mean_trend']>0]
        rainfall_intensity_dominant_df=df[df['max_flag_only']==5]
        rainfall_intensity_trend=rainfall_intensity_dominant_df['rainfall_intensity_trend'].tolist()
        rainfall_intensity_values=rainfall_intensity_dominant_df['rainfall_intensity'].tolist()
        rainfall_intensity_values=np.array(rainfall_intensity_values)
        rainfall_intensity_values_mean=np.nanmean(rainfall_intensity_values,axis=1)
        beta=rainfall_intensity_dominant_df['composite_LAI_beta_mean_trend'].tolist()
        # df['rainfall_intensity_mean']=rainfall_intensity_values_mean

        ## plt


        plt.figure(figsize=(10, 5))
        sc = plt.scatter(
            rainfall_intensity_values_mean,
            beta,
            c=rainfall_intensity_trend,
            cmap='RdBu',
            alpha=0.7,
            vmin=-0.1,
            vmax=0.1,


        )
        plt.ylim(-2, 2)
        ## vmin and vmax

        plt.colorbar(sc, label='Rainfall Intensity Trend',)
        plt.xlabel('Rainfall Intensity Baseline')
        plt.ylabel('Beta Trend')


        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def disentangle(self):
        ## beta vs rainfall intensity vs domiant factors vs rainfall trend
        dff = rf'D:\Project3\Result\3mm\SHAP_beta\png\RF_composite_LAI_beta\pdp_shap_beta2\\Dataframe.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # df=df[df['composite_LAI_beta_mean_p_value']<0.05]
        # df = df[df['continent'] == 'Australia']
        # df=df[df['composite_LAI_beta_mean_trend']>0]
        rainfall_intensity_dominant_df = df[df['max_flag_only'] == 5]
        rainfall_intensity_trend = rainfall_intensity_dominant_df['rainfall_intensity_trend'].tolist()
        rainfall_intensity_values = rainfall_intensity_dominant_df['rainfall_intensity'].tolist()
        rainfall_intensity_values = np.array(rainfall_intensity_values)
        rainfall_intensity_values_mean=np.nanmean(rainfall_intensity_values,axis=0)

        beta = rainfall_intensity_dominant_df['composite_LAI_beta_mean_trend'].tolist()
        print(np.nanmean(beta))
        print(np.nanmean(rainfall_intensity_trend))


        ## plt
        plt.figure(figsize=(10, 5))
        plt.plot(rainfall_intensity_values_mean)


        plt.grid(True)
        plt.tight_layout()
        plt.show()


        pass
    def feature_importances_shap_values(self, shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        # for i in range(len(shap_values)):
        #     importances.append(np.abs(shap_values[i]).mean())
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))


        # Calculates the normalized version
        # importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        # feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
        # feature_importances_norm = {k: v for k, v in
        #                             sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
        # Prints the feature importances
        # for k, v in feature_importances.items():
        #     print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

        return feature_importances
        # return feature_importances_norm

    def __select_extreme(self, df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df

    def __train_model(self, X, y):
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score
        import numpy as np

        print(type(X))
        print(X.shape)
        print(X.dtypes)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        r2_list = []

        for train_index, test_index in kf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                booster='gbtree',

                n_estimators=1200,  # ↑
                max_depth=13,

                random_state=42,
                n_jobs=14,

            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)

        print("CV R2:", np.mean(r2_list))
        print("CV std:", np.std(r2_list))

        # 最后再用全部数据训练一个最终模型（给SHAP用）
        # =========================
        # Final model (用全部数据训练)
        # =========================
        final_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            booster="gbtree",
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=14
        )

        final_model.fit(X, y)

        y_pred = final_model.predict(X)

        r2_full = r2_score(y, y_pred)

        print("Full data R2:", r2_full)

        return final_model, y, y_pred

    def __train_model_bootstrap(self, X, y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''

        model = xgb.XGBRegressor(objective="reg:squarederror", booster='gbtree', n_estimators=100,
                               max_depth=15, eta=0.1, random_state=42, n_jobs=14,  )
        # model = RandomForestRegressor(n_estimators=200, random_state=42,n_jobs=14)
        # model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=12, max_depth=7)

        model.fit(X, y)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X)

        # print(len(y_pred))
        # plt.scatter(y_test, y_pred)
        # plt.show()
        r = stats.pearsonr(y_pred,y)

        r2 = r[0] ** 2
        print('r2:', r2)
        # exit()

        return model, y, y_pred





    def __train_model_RF(self, X, y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=1, test_size=0.) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # build a random forest model
        rf.fit(X, y)  # train the model
        coef = rf.feature_importances_
        imp_dict = {}
        for i in range(len(coef)):
            imp_dict[self.x_variable_list[i]] = coef[i]

        return imp_dict

    def benchmark_model(self, y, y_pred):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        plt.scatter(y, y_pred)
        plt.plot([0.6, 1.2], [0.6, 1.2], color='r', linestyle='-', linewidth=2)
        plt.ylabel('Predicted', size=20)
        plt.xlabel('Actual', size=20)
        plt.xlim(0.6, 1.2)
        plt.ylim(0.6, 1.2)
        plt.show()

class SHAP_classsification:
    def __init__(self):
        self.feature_cols = [
             'tmax_summer_npy_trend', 'soil_summer_npy_trend',
            'srad_summer_npy_trend', 'summer_rainfall_intensity_trend',
            'tmean_mean','ppt_winter_npy_mean', 'ppt_winter_npy_trend'
        ]
        self.dff = result_root + rf'\SHAP\Dataframe\\Dataframe.df'

        pass
    def run(self):
        # self.show_colinear()
        self.SHAP_classsification_function()
        self.pdp_plot()
        pass

    def show_colinear(self, ):
        dff = self.dff
        df = T.load_df(dff)
        vars_list = self.feature_cols
        df = df[vars_list]
        ## add LAI4g_raw
        # df['composite_LAI_beta_mean_zscore'] = T.load_df(dff)['composite_LAI_beta_mean_zscore']
        ## plot heat map to show the colinear variables



        import seaborn as sns
        fig, ax=plt.subplots(figsize=(8, 5))
        ### x tick label rotate
        vmin = -1
        vmax = 1


        sns.heatmap(df.corr(), annot=True, fmt=".2f",vmin=vmin, vmax=vmax,cmap="RdBu")
        plt.xticks(rotation=45)
        ax.set_yticks(np.arange(len(vars_list)) + 0.5)
        # ax.set_yticklabels(model_list[::-1], rotation=0, va='center')
        ##get name from dic
        # ax.set_yticklabels([name_dic[x] for x in vars_list], rotation=0, va='center')
        #
        # ax.set_xticks(np.arange(len(vars_list)) + 0.5)
        # ax.set_xticklabels([name_dic[x] for x in vars_list], rotation=45, ha='center')
        # ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    def SHAP_classsification_function(self):

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        import pandas as pd
        import joblib  # 用于保存模型

        dff=result_root+rf'\SHAP\Dataframe\\Dataframe.df'
        df=T.load_df(dff)
        feature_cols = self.feature_cols
        model_df = df[feature_cols + ['summer_class_3']].dropna()

        X = model_df[feature_cols]
        y = model_df['summer_class_3']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # 初始化随机森林分类器
        rf_clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        rf_clf.fit(X_train, y_train)

        # 评估模型准确率
        y_pred = rf_clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        # 提取特征重要性（Feature Importance）
        importances = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

        plt.figure(figsize=(8, 6))
        importances.plot(kind='barh', color='teal', edgecolor='black')

        plt.xlabel('Importance Score', fontsize=10)
        plt.ylabel('Environmental Features', fontsize=10)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # 【核心：保存训练好的模型到本地】
        outdir=result_root+rf'\SHAP\output\\'
        T.mk_dir(outdir, force=True)
        model_filename = join(outdir, 'rf_classifier_model.pkl')
        joblib.dump(rf_clf, model_filename)


    def pdp_plot(self):
        import joblib  #
        from sklearn.inspection import PartialDependenceDisplay
        pkl_path=result_root+rf'\SHAP\output\\rf_classifier_model.pkl'

        dff = result_root + rf'\SHAP\Dataframe\\Dataframe.df'
        df = T.load_df(dff)
        feature_cols =self.feature_cols
        X_data = df[feature_cols].dropna()


        rf_clf = joblib.load(pkl_path)

        features_to_plot = self.feature_cols

        # 3. 画出多分类的 PDP 概率响应曲线
        fig, ax = plt.subplots(figsize=(15, 5), ncols=len(features_to_plot))

        # target 参数可以指定你想看哪几个类别，或者让它把多分类的各类别曲线都画出来
        display = PartialDependenceDisplay.from_estimator(
            estimator=rf_clf,
            X=X_data,
            target=2.0,
            features=features_to_plot,
            kind='average',  # 画平均趋势线
            ax=ax,
            response_method='predict_proba'  # 画预测概率
        )

        plt.suptitle("Probability Response Curves by Ecological Class", fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()

def check_data():
    fdir=result_root+rf'\Daymet\\'
    result_dic={}
    for f in T.listdir(fdir):
        if not f.endswith('.npy'):
            continue
        print(f)
        spatial_dic=T.load_npy(join(fdir,f))
        for pix in spatial_dic:
            vals=spatial_dic[pix]
            result_dic[pix]=np.nanmean(vals)
        array=D.pix_dic_to_spatial_arr(result_dic)
        plt.imshow(array, cmap='Spectral', interpolation='nearest')
        plt.title(f)
        plt.show()


    pass
class Prepare_datasets_for_RF:
    def __init__(self):
        pass
    def run(self):
        pass


def main():
    # coupling_anaysis().run()
    # Breakdown_data().run()
    # Moving_window_coupling_analysis().run()
    # PLOT_temporal_change_corr().run()
    # PLOT_bivariate().run()

    # check_data()
    # categroy().run()
    SHAP().run()
    # SHAP_classsification().run()



if __name__ == '__main__':
    main()









