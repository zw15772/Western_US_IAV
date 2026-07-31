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
        self.trend_analysis()
        # self.calculating_multiregression_sensitivity()
        # self.plot_sensitivity()
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

        fdir = result_root + r'\multiregression\input\\input1\\'
        outdir = result_root + r'\multiregression\\trend_analysis\\first\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0]
            # if os.path.isfile(outf + '_trend.tif'):
            #     continue
            # print(outf);exit()

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix

                time_series = dic[pix][0:11]
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

        fdir = result_root + r'\multiregression\input\\'

        fLAI = fdir + rf'{season}_LAI_zscore_detrend.npy'

        file_dic = {

            'intensity': fdir + rf'{season}_rainfall_intensity_zscore.npy',
            'temp': fdir + rf'tmax_{season}_npy_zscore_detrend.npy',
            'rad': fdir + rf'srad_{season}_npy_zscore_detrend.npy',
            'ppt_winter': fdir + rf'ppt_winter_npy_zscore_detrend.npy',
            'ppt': fdir + rf'ppt_{season}_npy_zscore_detrend.npy',
        }

        dic_LAI = T.load_npy(fLAI)

        dic_var = {}
        for var in file_dic:
            dic_var[var] = T.load_npy(file_dic[var])



        outdir = result_root + r'\multiregression\output\second\\npy\\'
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
                'LAI': vals_LAI[11:]

            }

            for var in var_list:
                data[var] = np.array(dic_var[var][pix], dtype=float)[11:]

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


    def plot_sensitivity(self):
        fdir=result_root + rf'\multiregression\output\second\\npy\\'
        outdir=result_root + rf'\multiregression\output\\second\\tiff\\'
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
        self.plot_sensitivity()

    def calculate_partial_corr(self):

        import pingouin as pg
        import pandas as pd
        from tqdm import tqdm
        import statsmodels.api as sm

        season = 'growing_season'

        fdir = result_root + r'\calculate_partial_corr\input\\'

        fLAI = fdir + rf'{season}_LAI_detrend.npy'

        file_dic = {

            'intensity': fdir + rf'{season}_rainfall_intensity.npy',
            'temp': fdir + rf'tmax_{season}_npy_detrend.npy',
            'rad': fdir + rf'srad_{season}_npy_detrend.npy',
            'ppt_growing_season': fdir + rf'ppt_{season}_npy_detrend.npy',
            'ppt_winter': fdir + rf'ppt_winter_npy_detrend.npy',
        }

        dic_LAI = T.load_npy(fLAI)

        dic_var = {}
        for var in file_dic:
            dic_var[var] = T.load_npy(file_dic[var])

        outdir = result_root + r'\calculate_partial_corr\output\second\\npy\\'
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
                'LAI': vals_LAI[11:]

            }

            for var in var_list:
                data[var] = np.array(dic_var[var][pix], dtype=float)[11:]

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
        fdir=result_root + rf'\calculate_partial_corr\output\second\\npy\\'
        outdir=result_root + rf'\calculate_partial_corr\output\\second\\tiff\\'
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

def main():
    Multiregression().run()
    # Partial_corr().run()

if __name__ == '__main__':
    main()
