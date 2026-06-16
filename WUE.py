import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
D=DIC_and_TIF(tif_template=tif_template)




class RUE_calculation:
    def run(self):
        # self.spring_summer_season_LAI_mean()
        # self.spring_summer_season_precip_mean()
        self.WUE()
        pass

    def spring_summer_season_LAI_mean(self):
        fdir = data_root + '\SNU_LAI\dic\\'
        outdir = data_root + '\SNU_LAI\spring_summer_season_LAI_mean\\'
        T.mk_dir(outdir, force=True)
        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}
        for pix in tqdm(spatial_dic):
            r, c = pix
            vals = spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals = np.array(vals)
            vals = np.reshape(vals, (-1, 12))
            # plt.imshow(vals)
            plt.show()
            spring_list = []
            summer_list = []

            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to may
                spring_val = np.nanmean(vals[i][2:5])
                ## july to sept
                summer_val = np.nanmean(vals[i][6:9])

                spring_list.append(spring_val)
                summer_list.append(summer_val)
            result_dic[pix] = {
                'spring': spring_list,
                'summer': summer_list,
            }
        outf = outdir + 'spring_summer_season_LAI_mean.npy'
        np.save(outf, result_dic)

    def spring_summer_season_precip_mean(self):
        fdir_precip=data_root+r'\Terraclimate\Precip\dic\\'

        outdir = data_root + '\Terraclimate\Precip\\spring_summer_season_precip_mean\\'
        T.mk_dir(outdir, force=True)
        spatial_dic = T.load_npy_dir(fdir_precip)
        result_dic = {}
        for pix in tqdm(spatial_dic):
            r, c = pix
            vals = spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals = np.array(vals)
            vals = np.reshape(vals, (-1, 12))
            # plt.imshow(vals)
            plt.show()
            spring_list = []
            summer_list = []

            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to may
                spring_val = np.nanmean(vals[i][2:5])

                ## july to sept
                summer_val = np.nanmean(vals[i][6:9])

                spring_list.append(spring_val)

                summer_list.append(summer_val)


            spring_list=np.array(spring_list)
            summer_list=np.array(summer_list)
            spring_list=spring_list[24:]
            summer_list=summer_list[24:]
            # print(len(spring_list),len(summer_list));exit()
            result_dic[pix] = {
                'spring': spring_list,
                'summer': summer_list,
            }
        outf = outdir + 'spring_summer_season_precip_mean.npy'
        np.save(outf, result_dic)

    pass

    def WUE(self):
        fdir_LAI=data_root+r'\SNU_LAI\spring_summer_season_LAI_mean\\'
        fdir_precip=data_root+r'\Terraclimate\Precip\spring_summer_season_precip_mean\\'
        outdir=result_root+r'\WUE\\'
        T.mk_dir(outdir,force=True)
        LAI_dic=T.load_npy(fdir_LAI+rf'spring_summer_season_LAI_mean.npy')
        precip_dic=T.load_npy(fdir_precip+rf'spring_summer_season_precip_mean.npy')
        WUE_dic={}

        for pix in tqdm(LAI_dic):

            LAI=LAI_dic[pix]['summer']
            if pix not in precip_dic:
                continue

            precip=precip_dic[pix]['summer']
            if np.isnan(np.nanmean(LAI)):
                continue
            if np.isnan(np.nanmean(precip)):
                continue
            LAI=np.array(LAI)
            precip=np.array(precip)

            WUE = np.where(precip != 0, LAI / precip, np.nan)

            # plt.plot(WUE)
            # plt.show()
            WUE_dic[pix]=WUE


        outf=outdir+rf'WUE_summer.npy'
        T.save_npy(WUE_dic,outf)


        pass

class GPP_calculation:
    def run(self):
        # self.spring_summer_season_GPP()
        self.trend_analysis()
        pass
    def spring_summer_season_GPP(self):
        fdir = data_root + 'ST_CFE-Hybrid_NT\dic\\'
        outdir = data_root + 'ST_CFE-Hybrid_NT\spring_summer_season_LAI_mean\\'
        T.mk_dir(outdir, force=True)
        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}
        for pix in tqdm(spatial_dic):
            r, c = pix
            vals = spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals = np.array(vals)
            vals = np.reshape(vals, (-1, 12))
            # plt.imshow(vals)
            plt.show()
            spring_list = []
            summer_list = []

            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to may
                spring_val = np.nanmean(vals[i][2:5])
                ##day to year
                spring_val=spring_val*31
                ## july to sept
                summer_val = np.nanmean(vals[i][6:9])
                summer_val=summer_val*31

                spring_list.append(spring_val)
                summer_list.append(summer_val)
            result_dic[pix] = {
                'spring': spring_list,
                'summer': summer_list,
            }
        outf = outdir + 'spring_summer_season_LAI_mean.npy'
        np.save(outf, result_dic)

    def trend_analysis(self):  ##each window average trend


        fdir = data_root + rf'\SNU_LAI\spring_summer_season_LAI_mean\\'
        outdir = result_root + rf'\SNU_LAI\\trend\spring\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            # if not 'DLEM_S2_lai' in f:
            #     continue

            outf = outdir + f.split('.')[0]
            # if os.path.isfile(outf + '_trend.tif'):
            #     continue
            # print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix

                    ## ignore the last one year

                # time_series = dic[pix][:-1]
                time_series = dic[pix]['spring'][0:39]
                time_series = np.array(time_series)
                # print(time_series)
                if np.isnan(time_series).all():
                    continue

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

            # plt.imshow(arr_trend, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)



def main():
    # RUE_calculation().run()
    GPP_calculation().run()


    pass

if __name__ == '__main__':
    main()