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



class Multiregression:
    def __init__(self):

        pass

    def run(self):
        # self.VIF()
        # self.trend_analysis()
        # self.contribution_analysis()
        # self.calculating_multiregression_sensitivity()
        # self.plot_sensitivity()
        self.PLOT_contribution_bar()
        # self.plot_distribution()
        pass

    def VIF(self):
        import pandas as pd
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        dff=result_root+rf'\\multiregression\Dataframe\\VIF.df'
        df=T.load_df(dff)
        season='growing_season'
        var_dic = {

            'intensity': rf'{season}_rainfall_intensity_zscore.npy',
            'temp': rf'tmax_{season}_npy_zscore_detrend.',
            'rad':  rf'srad_{season}_npy_zscore_detrend.npy',
            'ppt_winter':   rf'ppt_winter_npy_zscore_detrend.npy',
            'ppt':  rf'ppt_{season}_npy_zscore_detrend.npy',
        }

        print(df.head())
        print(df.shape)
        var_list = [
            rf'{season}_rainfall_intensity_zscore',
            rf'tmax_{season}_npy_zscore_detrend',
            rf'srad_{season}_npy_zscore_detrend',
            rf'ppt_winter_npy_zscore_detrend',
            rf'ppt_{season}_npy_zscore_detrend',
        ]

        # 去掉 NaN 和 inf
        df_vif = (
            df[var_list]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        vif = pd.DataFrame({
            "Variable": var_list,
            "VIF": [
                variance_inflation_factor(df_vif.values, i)
                for i in range(len(var_list))
            ]
        })

        print(vif.sort_values("VIF", ascending=False))

    def trend_analysis(self):

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        ##each window average trend

        fdir = result_root + r'\multiregression\input\input\zscore\\'
        outdir = result_root + r'\multiregression\\trend_analysis\\whole\\'
        Tools().mk_dir(outdir, force=True)
        season='growing_season'



        file_dic = {

            'rainfall intensity': fdir + rf'{season}_rainfall_intensity_zscore.npy',
            'tmax': fdir + rf'tmax_{season}_zscore.npy',
            'srad': fdir + rf'srad_{season}_zscore.npy',
            'ppt_winter': fdir + rf'ppt_winter_zscore.npy',
            'ppt': fdir + rf'ppt_{season}_zscore.npy',
        }

        for var in file_dic:
            dic = T.load_npy(file_dic[var])

            outf = outdir + rf'{var}'


            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix

                time_series = dic[pix]
                print(len(time_series))
                # plt.plot(time_series)
                # plt.show()
                time_series = np.array(time_series)
                # print(len(time_series));exit()

                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                # if len(set(time_series)) == 1:
                #     continue
                # print(time_series)

                # if np.nanstd(time_series) == 0:
                #     continue
                try:

                    # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series)), time_series)
                    # print(slope)

                    trend_dic[pix] = slope
                    p_value_dic[pix] = p_value
                except:
                    continue

            arr_trend = D.pix_dic_to_spatial_arr(trend_dic)
            p_value_arr = D.pix_dic_to_spatial_arr(p_value_dic)



            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

    def calculating_multiregression_sensitivity(self):

        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        import statsmodels.api as sm

        season = 'growing_season'

        fdir = result_root + r'multiregression\input\input\detrend_zscore\\\\'

        fLAI = fdir + rf'{season}_LAI_zscore_detrend.npy'

        file_dic = {

            'rainfall intensity': fdir + rf'{season}_rainfall_intensity_zscore.npy',
            'tmax': fdir + rf'tmax_{season}_zscore_detrend.npy',
            'srad': fdir + rf'srad_{season}_zscore_detrend.npy',
            'ppt_winter': fdir + rf'ppt_winter_zscore_detrend.npy',
            'ppt': fdir + rf'ppt_{season}_zscore_detrend.npy',
        }

        dic_LAI = T.load_npy(fLAI)

        dic_var = {}
        for var in file_dic:
            dic_var[var] = T.load_npy(file_dic[var])



        outdir = result_root + r'\multiregression\output\whole\\npy\\'
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

            for var in var_list:

                vals = np.array(dic_var[var][pix], dtype=float)

                if len(vals) != len(vals_LAI):
                    print(f'{pix} {var}: {len(vals)} != {len(vals_LAI)}')
                    continue



            # for v in ['ppt', 'temp', 'rad', 'intensity']:
            #     n_nan = 0
            #     n_total = 0
            #     for pix in dic_var[v]:
            #         arr = np.array(dic_var[v][pix], dtype=float)
            #         n_nan += np.isnan(arr).sum()
            #         n_total += arr.size
            #     print(v, n_nan / n_total)

            # print(len(dic_LAI))
            # print(len(dic_var['ppt']))
            # print(len(dic_var['temp']))
            # print(len(dic_var['rad']))
            # print(len(dic_var['intensity']))
            #
            # exit()

            beta_dic = {v: [] for v in var_list}
            p_dic = {v: [] for v in var_list}
            r2_list = []

            data = {

                # 'LAI': vals_LAI[0:11]
                # 'LAI': vals_LAI[11:],
                'LAI': vals_LAI,

            }

            for var in var_list:
                # data[var] = np.array(dic_var[var][pix], dtype=float)[11:]
                data[var] = np.array(dic_var[var][pix], dtype=float)

            df = pd.DataFrame(data)
            # T.print_head_n(df)
            df = pd.DataFrame(data).dropna()

            # for var in df.columns:
            #     if df[var].isna().sum() > 0:
            #         print(pix, var, df[var].isna().sum())
            # exit()

            ############################################
            # enough samples
            ############################################

            if len(df) < 10:
                print(pix, len(df))
                # exit()
                # T.print_head_n(df)

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



            model = sm.OLS(y, X).fit()

            for var in var_list:
                beta_dic[var].append(model.params[var])

                p_dic[var].append(model.pvalues[var])

            r2_list.append(model.rsquared)

            # except:
            #
            #     for var in var_list:
            #         beta_dic[var].append(np.nan)
            #         p_dic[var].append(np.nan)
            #
            #     r2_list.append(np.nan)

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
                outdir + rf'beta_{var}.npy'
            )

            T.save_npy(
                result_p[var],
                outdir + rf'p_value_{var}.npy'
            )

        T.save_npy(
            result_r2,
            outdir + rf'R2_{season}.npy'
        )


    def contribution_analysis(self):
        fdir_sensitivity=result_root + rf'\multiregression\output\first\\npy\\'
        fdir_trend=result_root + rf'\multiregression\trend_analysis\first\\'



        var_list = [
            f'ppt',
           f'ppt_winter',
            f'tmax',
            f'srad',
            f'rainfall intensity',
        ]
        # f_LAI=result_root + rf'multiregression\trend_analysis\second\\growing_season_LAI_zscore_trend.tif'
        # array_LAI, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_LAI)
        # array_LAI= np.array(array_LAI, dtype=float)
        #
        # val_dic_LAI = D.spatial_arr_to_dic(array_LAI)


        outdir = result_root + r'multiregression\Contribution\\first\\'
        T.mk_dir(outdir,force=True)

        for var in var_list:

            beta = T.load_npy(os.path.join(fdir_sensitivity, f'beta_{var}.npy'))
            array_climate, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(os.path.join(fdir_trend, f'{var}_trend.tif'))
            array_climate = np.array(array_climate, dtype=float)
            val_dic_climate= D.spatial_arr_to_dic(array_climate)

            contribution = {}

            pix_common = set(beta.keys()) & set(val_dic_climate.keys())

            for pix in pix_common:


                b = beta[pix][0]
                t = val_dic_climate[pix]


                if np.isnan(b) or np.isnan(t):
                    continue

                contribution[pix] = b * t
            array_contribution = D.pix_dic_to_spatial_arr(contribution)
            plt.imshow(array_contribution, cmap='RdBu',vmin=10, vmax=30)
            # plt.colorbar()
            # plt.show()
            D.arr_to_tif(array_contribution,  outdir + f'contribution_{var}.tif')

            T.save_npy(contribution,
                       os.path.join(outdir, f'{var}_contribution.npy'))



        pass


    def plot_sensitivity(self):
        fdir=result_root + rf'\multiregression\output\whole\\npy\\'
        outdir=result_root + rf'\multiregression\output\\whole\\tiff\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fname=f.split('.')[0]

            result_dic={}
            dic_LAI = T.load_npy(fdir + f)
            for pix in tqdm(dic_LAI):
                vals=np.array(dic_LAI[pix][0], dtype=float)
                result_dic[pix]=vals
            array=D.pix_dic_to_spatial_arr(result_dic)
            D.arr_to_tif(array,outdir + fname+'.tif')
            plt.imshow(array, cmap='Spectral', vmin=-1, vmax=1)
            plt.title(fname)
            plt.colorbar()
            plt.show()

            pass

        pass

    pass

    def PLOT_contribution_bar(self):
        dff=result_root + rf'multiregression\Dataframe\\contribution.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        periods=['first','second','whole']
        variable_list=['ppt_winter','ppt','tmax','srad','rainfall intensity']
        for period in periods:
            result_list=[]
            for var in variable_list:
                vals=df[f'contribution_{var}_{period}'].to_list()
                vals_mean=np.nanmean(vals)
                result_list.append(vals_mean)
            LAItrend=df[f'growing_season_LAI_zscore_trend_{period}'].to_list()
            result_list.append(np.nanmean(LAItrend))
            result_list = np.array(result_list)
            new_variable_list=variable_list.copy()
            new_variable_list.append('LAI trend')
            plt.figure(figsize=(6, 4))
            plt.bar(new_variable_list, result_list, color='steelblue')
            plt.title(period)
            plt.ylabel('Mean contribution')
            plt.xticks(rotation=30)
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

    def plot_distribution(self):
        dff=result_root + rf'\multiregression\Dataframe\\Dataframe.df'
        df=T.load_df(dff)
        df=self.df_clean(df)

        variable_list = [
            'beta_ppt_summer_first',
            'beta_ppt_summer_second'
        ]

        plt.figure(figsize=(6, 4))

        for var in variable_list:
            vals = np.array(df[var], dtype=float)
            vals = vals[~np.isnan(vals)]  # remove NaN

            print(var)
            print('N =', len(vals))
            print('Mean =', np.mean(vals))
            print('Median =', np.median(vals))

            plt.hist(
                vals,
                bins=100,
                alpha=0.5,
                density=True,
                label=var
            )

        plt.xlabel('Standardized regression coefficient (β)')
        plt.ylabel('Density')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        alpha = 0.05

        periods = ['first', 'second']

        positive = []
        negative = []
        nonsig = []
        for period in periods:
            beta = np.array(df[f'beta_ppt_summer_{period}'], dtype=float)
            p = np.array(df[f'p_value_ppt_summer_{period}'], dtype=float)

            mask = (~np.isnan(beta)) & (~np.isnan(p))

            beta = beta[mask]
            p = p[mask]

            total = len(beta)

            positive.append(np.sum((beta > 0) & (p < alpha)) / total * 100)
            negative.append(np.sum((beta < 0) & (p < alpha)) / total * 100)
            nonsig.append(np.sum(p >= alpha) / total * 100)

        fig, ax = plt.subplots(figsize=(5, 5))

        x = np.arange(2)

        ax.bar(x,
               positive,
               color='firebrick',
               label='Positive')

        ax.bar(x,
               negative,
               bottom=positive,
               color='steelblue',
               label='Negative')

        ax.bar(x,
               nonsig,
               bottom=np.array(positive) + np.array(negative),
               color='lightgray',
               label='Non-significant')

        ax.set_xticks(x)
        ax.set_xticklabels(['2003–2013', '2014–2024'])

        ax.set_ylabel('Area (%)')
        ax.set_ylim(0, 100)

        ax.legend(frameon=False)

        plt.tight_layout()
        plt.show()

class Partial_corr:
    def __init__(self):
        pass

    def run(self):
        # self.calculate_partial_corr()
        # self.plot_sensitivity()
        self.PLOT_corr_bar()

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

    def calculate_partial_corr(self):

        import pingouin as pg
        import pandas as pd
        from tqdm import tqdm
        import statsmodels.api as sm

        season = 'summer'

        fdir = result_root + rf'\detrend\{season}\\'

        fLAI = fdir + rf'{season}_LAI_anomaly_detrend.npy'

        file_dic = {

            'rainfall_intensity': fdir + rf'{season}_rainfall_intensity.npy',
            'tmax': fdir + rf'tmax_{season}_anomaly_detrend.npy',
            'srad': fdir + rf'srad_{season}_anomaly_detrend.npy',
            'ppt': fdir + rf'ppt_{season}_anomaly_detrend.npy',
            'ppt_winter': fdir + rf'ppt_winter_anomaly_detrend.npy',
        }

        dic_LAI = T.load_npy(fLAI)

        dic_var = {}
        for var in file_dic:
            dic_var[var] = T.load_npy(file_dic[var])

        outdir = result_root + rf'\calculate_partial_corr\output\\{season}\\npy\\'
        T.mk_dir(outdir, force=True)

        var_list = list(file_dic.keys())

        ############################################
        # output
        ############################################

        result_corr = {v: {} for v in var_list}
        result_p = {v: {} for v in var_list}


        ############################################

        for pix in tqdm(dic_LAI):

            if any(pix not in dic_var[v] for v in var_list):
                continue

            vals_LAI = np.array(dic_LAI[pix], dtype=float)

            for var in var_list:

                vals = np.array(dic_var[var][pix], dtype=float)

                if len(vals) != len(vals_LAI):
                    print(f'{pix} {var}: {len(vals)} != {len(vals_LAI)}')
                    continue

            # for v in ['ppt', 'temp', 'rad', 'intensity']:
            #     n_nan = 0
            #     n_total = 0
            #     for pix in dic_var[v]:
            #         arr = np.array(dic_var[v][pix], dtype=float)
            #         n_nan += np.isnan(arr).sum()
            #         n_total += arr.size
            #     print(v, n_nan / n_total)

            # print(len(dic_LAI))
            # print(len(dic_var['ppt']))
            # print(len(dic_var['temp']))
            # print(len(dic_var['rad']))
            # print(len(dic_var['intensity']))
            #
            # exit()

            corr_dic = {v: [] for v in var_list}
            p_dic = {v: [] for v in var_list}

            data = {

                # 'LAI': vals_LAI[0:11]
                'LAI': vals_LAI

            }

            for var in var_list:
                data[var] = np.array(dic_var[var][pix], dtype=float)

            df = pd.DataFrame(data)
            # T.print_head_n(df)
            df = pd.DataFrame(data).dropna()

            # for var in df.columns:
            #     if df[var].isna().sum() > 0:
            #         print(pix, var, df[var].isna().sum())
            # exit()

            ############################################
            # enough samples
            ############################################

            if len(df) < 10:
                print(pix, len(df))
                # exit()
                # T.print_head_n(df)

                for var in var_list:
                    result_corr[var][pix] = np.nan
                    result_p[var][pix] = np.nan


                continue

            ############################################

            ############################################
            # regression
            ############################################

            for var in var_list:

                covars = [v for v in var_list if v != var]

                try:

                    stats = pg.partial_corr(
                        data=df,
                        x='LAI',
                        y=var,
                        covar=covars,
                        method='pearson'
                    )

                    corr_dic[var].append(stats['r'].values[0])
                    p_dic[var].append(stats['p_val'].values[0])
                    # print(stats['p_val'].values[0]);exit()

                except:

                    corr_dic[var].append(np.nan)
                    p_dic[var].append(np.nan)
            ############################################
            # save pixel
            ############################################

            for var in var_list:
                result_corr[var][pix] = corr_dic[var]
                result_p[var][pix] = p_dic[var]

        ############################################
        # save
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

    def plot_sensitivity(self):
        season='growing_season'
        fdir=result_root + rf'\calculate_partial_corr\output\{season}\npy\\'
        outdir=result_root + rf'\calculate_partial_corr\output\\{season}\\tiff\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fname=f.split('.')[0]

            result_dic={}
            dic_LAI = T.load_npy(fdir + f)
            for pix in tqdm(dic_LAI):
                vals=np.array(dic_LAI[pix][0], dtype=float)
                result_dic[pix]=vals
            array=D.pix_dic_to_spatial_arr(result_dic)
            D.arr_to_tif(array,outdir + fname+'.tif')
            plt.imshow(array, cmap='Spectral', vmin=-1, vmax=1)
            plt.title(fname)
            plt.colorbar()
            plt.show()

            pass

    def PLOT_corr_bar(self):
        dff = result_root + rf'\calculate_partial_corr\Dataframe\\partial_corr.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        season='spring'


        variable_list = [
            rf'partial_corr_ppt_winter_{season}_whole',
            f'partial_corr_rainfall_intensity_{season}_whole',
            f'partial_corr_ppt_winter_{season}_whole',
        ]

        eco_region_list = ['Western US', 'Western Cordillera', 'Upper Gila Mountains',
                           'Warm Desert', 'Cold Desert', 'Western Sierra Madre Piedmont']

        for eco in eco_region_list:

            if eco == 'Western US':
                df_i = df.copy()
            else:
                df_i = df[df['Ecoregion_level_II'] == eco]

            result_list = []

            for var in variable_list:
                vals = df_i[var].to_numpy(dtype=float)

                vals = vals[~np.isnan(vals)]

                result_list.append(vals)

            plt.figure(figsize=(6, 5))

            plt.boxplot(
                result_list,
                labels=['GS ppt', 'Intensity', 'Winter ppt'],
                showfliers=False
            )

            plt.axhline(0, color='gray', linestyle='--')
            plt.ylabel('Partial correlation')
            plt.title(eco)
            plt.tight_layout()
            plt.show()

class SEM:
    def __init__(self):
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

    def run(self):
        # self.SEM_calculation()
        self.SEM_wet_dry()


    def SEM_calculation(self):
        from semopy import Model, calc_stats
        from scipy.stats import pearsonr

        dff=result_root+rf'\SEM\\Dataframe\\SEM.df'
        outdir=result_root + rf'\SEM\\output\\'
        T.mk_dir(outdir, force=True)
        df=T.load_df(dff)
        df=self.df_clean(df)

        df = df[[
            'SWE_winter_anomaly_detrend',
            'spring_rainfall_fq_5mm_anomaly_detrend',
            'ppt_summer_anomaly_detrend',
            'SM_L1_growing_season_anomaly_detrend',
            'summer_LAI_anomaly_detrend'
        ]].copy()

        df_sem = df.rename(columns={
            'SWE_winter_anomaly_detrend': 'SWE',
            'spring_rainfall_fq_5mm_anomaly_detrend': 'Rain_Fq',
            'ppt_summer_anomaly_detrend': 'Summer_PPT',
            'SM_L1_growing_season_anomaly_detrend': 'SM',
            'summer_LAI_anomaly_detrend': 'LAI'
        })

        df_corr = df_sem[
            ['Rain_Fq', 'Summer_PPT']
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()


        r, p = pearsonr(
            df_corr['Rain_Fq'],
            df_corr['Summer_PPT']
        )

        print('r =', r)
        print('p =', p)


        print(
            df_sem[
                ['SWE', 'Rain_Fq', 'Summer_PPT', 'SM', 'LAI']
            ].corr().round(2)
        )
        exit()

        model_desc = """

        # ==========================================
        # Soil moisture
        # ==========================================
        SM ~ SWE + Rain_Fq + Summer_PPT


        # ==========================================
        # Vegetation
        # ==========================================
        LAI ~ SM + Rain_Fq + Summer_PPT
        
        
        # ==========================================
        # Covariance among climate drivers
        # ==========================================
        SWE ~~ Rain_Fq
        SWE ~~ Summer_PPT
        Rain_Fq ~~ Summer_PPT
        
                """

        model = Model(model_desc)

        model.fit(
            df_sem[
                [
                    'SWE',
                    'Rain_Fq',
                    'Summer_PPT',
                    'SM',
                    'LAI'
                ]
            ]
        )

        # standardized path coefficients
        est = model.inspect(std_est=True)

        path_result = est[
            est['op'] == '~'
            ][
            [
                'lval',
                'rval',
                'Estimate',
                'Est. Std',
                'p-value'
            ]
        ]

        print(path_result)

        # model fit
        stats = calc_stats(model)

        print(stats.T)


        from semopy import semplot

        semplot(
            model,
            outdir+rf'SEM_path.png',
            plot_covs=True,
            std_ests=True,
            show=False
        )

    def SEM_wet_dry(self):
        from semopy import Model, calc_stats
        from scipy.stats import pearsonr

        dff = result_root + rf'\SEM\\Dataframe\\SEM.df'
        outdir = result_root + rf'\SEM\\output\\'
        T.mk_dir(outdir, force=True)
        df = T.load_df(dff)
        df = self.df_clean(df)

        df = df[[
            'SWE_winter_anomaly_detrend',
            'spring_rainfall_fq_5mm_anomaly_detrend',
            'ppt_summer_anomaly_detrend',
            'SM_L1_growing_season_anomaly_detrend',
            'summer_LAI_anomaly_detrend',
            'summer_SPEI06',
        ]].copy()

        df_sem = df.rename(columns={
            'SWE_winter_anomaly_detrend': 'SWE',
            'spring_rainfall_fq_5mm_anomaly_detrend': 'Rain_Fq',
            'ppt_summer_anomaly_detrend': 'Summer_PPT',
            'SM_L1_growing_season_anomaly_detrend': 'SM',
            'summer_LAI_anomaly_detrend': 'LAI',

        })


        df_dry = df_sem[df_sem['summer_SPEI06'] < -0.5].copy()
        df_wet = df_sem[df_sem['summer_SPEI06'] >= 0.5].copy()

        df_dic = {
            'wet': df_wet,
            'dry': df_dry,
        }

        for condition, df_ii in df_dic.items():

            #
            model_desc = """
    
            # ==========================================
            # Soil moisture
            # ==========================================
            SM ~ SWE + Rain_Fq + Summer_PPT
    
    
            # ==========================================
            # Vegetation
            # ==========================================
            LAI ~ SM + Rain_Fq + Summer_PPT
    
    
            # ==========================================
            # Covariance among climate drivers
            # ==========================================
            SWE ~~ Rain_Fq
            SWE ~~ Summer_PPT
            Rain_Fq ~~ Summer_PPT
    
                    """

            model = Model(model_desc)

            model.fit(
                df_ii[
                    [
                        'SWE',
                        'Rain_Fq',
                        'Summer_PPT',
                        'SM',
                        'LAI'
                    ]
                ]
            )

            # standardized path coefficients
            est = model.inspect(std_est=True)

            path_result = est[
                est['op'] == '~'
                ][
                [
                    'lval',
                    'rval',
                    'Estimate',
                    'Est. Std',
                    'p-value'
                ]
            ]

            print(path_result)

            # model fit
            stats = calc_stats(model)

            print(stats.T)

            from semopy import semplot

            semplot(
                model,
                outdir + rf'SEM_path_{condition}.png',
                plot_covs=True,
                std_ests=True,
                show=False
            )


def main():
    # Multiregression().run()
    # Partial_corr().run()
    SEM().run()

if __name__ == '__main__':
    main()
