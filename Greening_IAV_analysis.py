import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= '/Users/wenzhang/Downloads/Western US IAV/Data/SNU_LAI/extract_tif/199901.tif'
D=DIC_and_TIF(tif_template=tif_template)

class greening_analysis:
    def __init__(self):
        pass
    def run(self):
        # self.relative_change()
        self.trend_analysis()
        pass
    def relative_change(self):

        f = data_root+r'SNU_LAI/extract_growing_season_LAI_mean/growing_season_LAI_mean.npy'
        outdir = result_root + rf'greening_analysis/relative_change/'
        Tools().mk_dir(outdir, force=True)

        outf = outdir + 'LAI_growing_season_relative_change.npy'
        # print(outf);exit()


        dic = T.load_npy(f)

        zscore_dic = {}

        for pix in tqdm(dic):



            # print(len(dic[pix]))
            time_series = dic[pix]['growing_season']


            time_series = np.array(time_series)


            print(len(time_series))

            if np.isnan(np.nanmean(time_series)):
                continue

            time_series = time_series
            mean = np.nanmean(time_series)
            relative_change = (time_series - mean) / mean * 100
            anomaly = time_series - mean
            zscore_dic[pix] = relative_change
          # plot
          #   plt.plot(anomaly)
          #   # plt.legend(['anomaly'])
          #   plt.show()
          #
          #   plt.plot(relative_change)
          #   plt.legend(['relative_change'])
          #   plt.legend(['anomaly','relative_change'])
          #   plt.show()

                ## save
            T.save_npy( zscore_dic, outf)

    def trend_analysis(self):  ##each window average trend
        phenology_mask_f = data_root + rf'/basedata/Phenology_extraction/phenology_type.tif'
        phenology_mask_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(phenology_mask_f)
        phenology_dic = D.spatial_arr_to_dic(phenology_mask_arr)



        fdir = result_root + r'greening_analysis/relative_change/'
        outdir = result_root + r'greening_analysis/relative_change/trend/'
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

                time_series = dic[pix]
                # print(time_series)
                time_series = np.array(time_series)
                # print(time_series)

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
    greening_analysis().run()




if __name__ == '__main__':
    main()