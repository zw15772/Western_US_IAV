import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *


tif_template= '/Users/wenzhang/Downloads/Western US IAV/Data/SNU_LAI/extract_tif/200401.tif'
D=DIC_and_TIF(tif_template=tif_template)


class Data_processing:
    def run(self):
        # self.nc_to_tif_time_series_fast2()
        # self.extract_tif_from_shp()
        # self.tif_to_dic()
        ## 4 extract phenology based 4GST using GST_phenology_Wen.py
        ## 5 现在用SOS EOS extract growing season and return monthly data during growing season
        # self.extract_growing_season_monthly()
        # self.extract_growing_season_LAI_mean()
        # self.extract_growing_season_LAI_min()
        # self.extract_growing_season_LAI_max()
        self.spatial_plot()


        pass
    def nc_to_tif_time_series_fast2(self):

        fdir=rf'/Users/wenzhang/Downloads/Western US IAV/Data/SNU_LAI/nc/'
        outdir=rf'/Users/wenzhang/Downloads/Western US IAV/Data/SNU_LAI/tif/'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):


            outdir_name = f.split('.')[0].split('_')[-1]

            # exit()

            yearlist = list(range(1982, 2025))
            fpath = join(fdir,f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)


            outf = join(outdir,outdir_name+'.tif')
            array = nc_in['LAI']
            array = np.array(array).T

            array[array < 0] = np.nan
            longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.05, -0.05
            ToRaster().array2raster(outf, longitude_start, latitude_start,
                                    pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()

    def extract_tif_from_shp(self):
        shp_f=data_root + 'basedata/Western_US_bountry/merged_western_US.shp'
        fdir=data_root + '/SNU_LAI/tif/'
        outdir=data_root + '/SNU_LAI/extract_tif/'
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=outdir+f

            ToRaster().clip_array(fpath, outf,shp_f)

        pass

    def tif_to_dic(self):

        fdir_all = data_root + rf'/SNU_LAI/extract_tif/'
        outdir=data_root + '/SNU_LAI/dic/'
        T.mk_dir(outdir, force=True)

        year_list = list(range(1982, 2025))
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

            array_unify[array_unify < 0] = np.nan

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

    def extract_growing_season_monthly(self):
        fdir = data_root+rf'\Terraclimate\SPEI\SPEI_12\\dic\\'

        outdir =data_root + r'Terraclimate\SPEI\SPEI_12\extract_growing_season_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = data_root+rf'/SNU_LAI/4GST/4GST.npy'
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
        fdir = data_root+r'/SNU_LAI/extract_growing_season_monthly/'

        outdir = data_root+r'/SNU_LAI/extract_growing_season_LAI_mean/'


        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            ### annual year

            vals_growing_season = spatial_dic[pix]
            print(vals_growing_season.shape[1])
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

        outf = outdir + 'growing_season_LAI_mean.npy'

        np.save(outf, result_dic)


    def spatial_plot(self):
        f=data_root+r'/SNU_LAI/extract_growing_season_LAI_min/' + 'growing_season_LAI_min.npy'
        dic=T.load_npy(f)
        spatial_dic = {}
        for pix in tqdm(dic):
            r, c = pix
            vals_growing_season = dic[pix]['growing_season']
            spatial_dic[pix] = np.nanmean(vals_growing_season)
        arr=D.pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()


        pass

class convert_dic_to_tiff:   ### display in QGIS
    def run(self):
        self.add_nan()
        self.spatial_dict_to_tif()
    def add_nan(self):
        ## this function for NH 43 years and SH 42 years
        fpath = rf'/Users/wenzhang/Downloads/Western US IAV/Result/greening_analysis/relative_change/SNU_LAI.npy'
        spatial_dic = T.load_npy(fpath)
        len_dic={}
        for pix in tqdm(spatial_dic):
            data_len=len(spatial_dic[pix])
            len_dic[pix]=data_len

        arr = D.pix_dic_to_spatial_arr(len_dic)
        plt.imshow(arr)
        plt.show()
        spatial_dic_new={}
        for pix in tqdm(spatial_dic):
            r, c = pix
            vals = spatial_dic[pix]
            print(len(vals))

            if len(vals) == 42:
                vals = np.append(vals, np.nan)

            if len(vals) == 43:
                spatial_dic_new[pix] = vals

        outdir=result_root+r'/greening_analysis/convert_dic_to_tiff/relative_change/'
        T.mk_dir(outdir, force=True)
        outpath=outdir+r'/SNU_LAI.npy'
        T.save_npy( spatial_dic_new,outpath,)




    def spatial_dict_to_tif(self):
        phenology_mask_f = data_root + rf'SNU_LAI/Phenology_extraction/SeasType.tif'
        phenology_mask_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(phenology_mask_f)
        phenology_dic = D.spatial_arr_to_dic(phenology_mask_arr)

        fpath=result_root+rf'greening_analysis/convert_dic_to_tiff/relative_change/SNU_LAI.npy'
        spatial_dic=T.load_npy(fpath)

        len_dic = {}
        for pix in tqdm(spatial_dic):
            data_len = len(spatial_dic[pix])
            len_dic[pix] = data_len

        arr = D.pix_dic_to_spatial_arr(len_dic)
        plt.imshow(arr)
        plt.show()


        spatial_new={}
        for pix in tqdm(spatial_dic):
            phenology_type=phenology_dic[pix]
            if phenology_type==3:
                continue
            spatial_new[pix]=spatial_dic[pix]

        outdir=result_root+r'/greening_analysis/convert_dic_to_tiff/relative_change/tif/'
        T.mk_dir(outdir, force=True)
        D.pix_dic_to_tif_every_time_stamp(spatial_new, outdir, filename_list=list(range(1982,2025)))


    pass





class check_data:
    def run(self):
        self.plot_time_series()
        # self.check_spatial_coverage()
    def plot_time_series(self):
        f=data_root+rf'SNU_LAI/multi_year_mean_monthly/multi_year_mean_monthly.npy'
        dic=T.load_npy(f)
        for pix in dic:
            vals=dic[pix]

            if np.isnan(np.nanmean(vals)):
                continue
            print(len(vals))
            time_series = dic[pix]
            time_series=dic[pix]
            plt.plot(time_series)
            plt.show()

    def check_spatial_coverage(self):
        fdir = r'/Users/wenzhang/Downloads/Western US IAV/Data/SNU_LAI/dic/'
        dic = T.load_npy_dir(fdir)
        spatial_coverage = {}


        for pix in dic:
            vals = dic[pix]

            if np.isnan(np.nanmean(vals)):
                continue
            # print(len(vals))
            length = len(vals)
            spatial_coverage[pix] = length
        arr=D.pix_dic_to_spatial_arr(spatial_coverage)
        plt.imshow(arr,cmap='jet')
        plt.show()



def main():

     Data_processing().run()
    # area_weighted_average().run()

    # check_data().run()
    # convert_dic_to_tiff().run()

if __name__ == '__main__':
    main()