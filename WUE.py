import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
D=DIC_and_TIF(tif_template=tif_template)




class WUE_calculation:
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




def main():
    WUE_calculation().run()


    pass

if __name__ == '__main__':
    main()