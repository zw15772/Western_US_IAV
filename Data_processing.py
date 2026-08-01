import matplotlib.pyplot as plt
import numpy as np
from lytools import *
from matplotlib.pyplot import summer
from rasterio.transform import from_bounds
import xarray as xr
from netCDF4 import Dataset
from rasterio.warp import Resampling, calculate_default_transform, reproject
import pyproj
from pprint import pprint
T = Tools()


from __Global__ import *
from Utils import *



tif_template= rf'D:\Western_US_IAV\Data\basedata\200902.tif'
D=DIC_and_TIF(tif_template=tif_template)


class Data_processing_vegetation:
    def run(self):
        # self.nc_to_tif_time_series_fast2()
        # self.nc_to_tif_time_series_fast2_VOD()
        # self.extract_tif_from_shp()
        # self.tif_to_dic()
        self.spring_season_LAI_mean()
        ## 4 extract phenology based 4GST using GST_phenology_Wen.py
        ## 5 现在用SOS EOS extract growing season and return monthly data during growing season
        # self.extract_growing_season_monthly()

        # self.spatial_plot()
        # self.plot_ecoregion()


        pass
    def nc_to_tif_time_series_fast2(self):

        fdir=rf'D:\Western_US_IAV\Data\VOD\\nc\\'
        outdir=rf'D:\Western_US_IAV\Data\VOD\tiff\\'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):


            outdir_name = f.split('.')[0].split('_')[-1]

            # exit()


            fpath = join(fdir,f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)


            outf = join(outdir,outdir_name+'.tif')
            array = nc_in['VOD']
            # plt.imshow(array[0])
            # plt.show()
            array = np.array(array).T


            # array[array < 0] = np.nan
            longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.25, -0.25
            ToRaster().array2raster(outf, longitude_start, latitude_start,
                                    pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()

    def nc_to_tif_time_series_fast2_VOD(self):
        from rasterio.transform import from_origin
        fdir = rf'D:\Western_US_IAV\Data\VOD\\nc\\'
        outdir = rf'D:\Western_US_IAV\Data\VOD\tiff\\'
        Tools().mk_dir(outdir, force=True)

        for f in tqdm(os.listdir(fdir)):

            fpath = join(fdir, f)
            nc_in = xarray.open_dataset(fpath)
            spei = nc_in['VOD']  # (time, lat, lon)
            lats = nc_in['lat'].values
            lons = nc_in['lon'].values
            time = nc_in['time'].values
            for i in range(len(lats)):
                print(lats[i+1]-lats[i])


            lat_res = abs(lats[1] - lats[0])
            lon_res = abs(lons[1] - lons[0])
            # print(lats[0], lats[-1]);exit()


            for i in range(len(time)):
                data = spei[i].values

                data = np.flipud(data)
                plt.imshow(data)
                plt.show()

                # 把 nan 设成 nodata
                data = data.astype(np.float32)

                year = str(nc_in['time.year'][i].values)
                month = str(nc_in['time.month'][i].values)
                month = int(month)

                outf = os.path.join(outdir, f'{year}{month:02d}.tif')

                longitude_start, latitude_start, pixelWidth, pixelHeight = -51.904212951660156 , 75.91789245605469, 0.25, -0.25
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, data, ndv=-999999)
                # exit()

    pass

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

    def spring_season_LAI_mean(self):
        fdir=data_root +rf'\SNU_LAI\dic\\'
        outdir=data_root +rf'\SNU_LAI\spring_summer_season_LAI_mean\\'
        T.mk_dir(outdir,force=True)
        spatial_dic=T.load_npy_dir(fdir)
        result_dic={}
        for pix in tqdm(spatial_dic):
            r,c=pix
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals=np.array(vals)
            vals=np.reshape(vals,(-1,12))
            # plt.imshow(vals)
            plt.show()
            spring_list=[]
            summer_list=[]

            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to may
                spring_val=np.nanmean(vals[i][2:5])
                ## july to sept
                summer_val=np.nanmean(vals[i][6:9])

                spring_list.append(spring_val)
                summer_list.append(summer_val)
            result_dic[pix]={
                'spring':spring_list,
                'summer':summer_list,
            }
        outf=outdir+'spring_summer_season_LAI_mean.npy'
        np.save(outf,result_dic)

    def extract_growing_season_monthly(self):
        fdir = data_root+rf'\MODIS_LAI\dic\\'

        outdir =data_root + r'\MODIS_LAI\extract_growing_season_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = data_root+rf'/MODIS_LAI/4GST/4GST.npy'
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

                        # lon, lat = D.pix_to_lon_lat(pix)
                        #

                        time_series_reshape = time_series_flatten.reshape(-1, 12)
                        plt.imshow(time_series_reshape)

                        # plt.title(f'lon:{lon}, lat:{lat},SOS_monthly:{SOS_monthly}, EOS_monthly:{EOS_monthly}')
                        plt.show()
                        plt.plot(time_series_reshape[0])
                        plt.show()
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
                    # plt.imshow(time_series_reshape)
                    #
                    # # plt.title(f'lon:{lon}, lat:{lat},SOS_monthly:{SOS_monthly}, EOS_monthly:{EOS_monthly}')
                    # plt.show()
                    # plt.plot(time_series_reshape[0])
                    # plt.show()

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
            # np.save(outf, result_dic)

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
                if len(vals_growing_season) == 42:
                    plt.plot(val)
                    plt.show()



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

    def plot_ecoregion(self):
        f = data_root + rf'basedata\Ecoregion\\Ecoregion_levelII_reprojected.shp'
        import geopandas as gpd
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        # 读取 shp
        gdf = gpd.read_file(f)

        # 看一下字段名
        print(gdf.columns)
        fig, ax = plt.subplots(figsize=(10, 8))
        # ax = plt.axes(projection=ccrs.PlateCarree())
        #
        # # 加大陆
        # ax.add_feature(cfeature.LAND, facecolor='lightgray')
        # ax.add_feature(cfeature.COASTLINE)

        gdf.plot(
            column='NA_L2NAME',
            categorical=True,
            legend=True,
            cmap='tab20',
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
        )



        plt.axis('off')
        # 移动 legend
        leg = ax.get_legend()
        leg.set_bbox_to_anchor((0.1, 0.2))
        leg._loc = 9  # upper center

        plt.tight_layout()
        plt.show()
        plt.tight_layout()
        plt.show()
class Data_processing_MODIS_LAI:
    def run(self):
        # self.modify_tif_metadata()
        # self.extract_tif_from_shp()
        # self.scale()
        # self.MVC()
        # self.raster_align()
        # self.tif_to_dic()
        self.spring_season_LAI_mean()
        self.growing_season()
        # self.trend_analysis()
        pass

    def modify_tif_metadata(self):
        from pprint import pprint
        from rasterio import Affine
        shp_f = data_root + 'basedata/Western_US_bountry/merged_western_US.shp'
        fdir = data_root + rf'\MODIS_LAI\\tif\\'
        outdir = data_root + rf'/MODIS_LAI/modify_tif_metadata/'
        T.mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):



            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=outdir+f
            with rasterio.open(fpath) as src:
                profile = src.profile
                affine=src.transform
                origin_x=affine.c
                # print(origin_x)
                affine_new=-179.07955750430673
                profile['transform']=Affine(affine.a,affine.b,affine_new,affine.d,affine.e,affine.f)
                with rasterio.open(outf, 'w', **profile) as dst:
                    dst.write(src.read())






    def extract_tif_from_shp(self):
        shp_f=data_root + 'basedata/Western_US_bountry/merged_western_US.shp'
        fdir=data_root + rf'\MODIS_LAI_0706\tiff\\'
        outdir=data_root + rf'/MODIS_LAI_0706/extract_tif/'
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):

            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=outdir+f

            ToRaster().clip_array(fpath, outf,shp_f)


        pass




    def scale(self):

        fdir = rf'D:\Western_US_IAV\Data\MODIS_LAI_0706\extract_tif\\'
        outdir = rf'D:\Western_US_IAV\Data\MODIS_LAI_0706\extract_tif_scaled\\'
        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)
            # array[array == 65535] = np.nan
            # array[array == 249] = np.nan
            array = array * 0.1
            array[array > 10] = np.nan
            array[array <= 0] = np.nan
            # array=array/10000
            # plt.imshow(array)
            # plt.show()



            outf = outdir + f
            ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, array)

    def filter_nan(self):
        fdir=data_root + '/WesternUS_MODIS_LAI_005deg_2000_2024/extract_tif/'
        outdir=data_root + '/WesternUS_MODIS_LAI_005deg_2000_2024/extract_tif_nan_filtered/'
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=outdir+f

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array = np.array(array, dtype=float)
            array[array==249] = np.nan
            array[array<=0] = np.nan
            ToRaster().array2raster(outf, originX, originY,
                                    pixelWidth, pixelHeight, array, ndv=-999999)
        pass
    def unify_TIFF(self):
        fdir_all=rf'D:\Western_US_IAV\Data\WesternUS_MODIS_LAI_005deg_2000_2024\extract_tif_scaled\\'
        outdir=rf'D:\Western_US_IAV\Data\WesternUS_MODIS_LAI_005deg_2000_2024\\unify\\'
        Tools().mk_dir(outdir, force=True)


        for f in os.listdir(join(fdir_all)):
            fpath=join(fdir_all,f)
            outpath=join(outdir,f)

            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            unify_tiff=DIC_and_TIF().unify_raster(fpath,outpath,0.05)

    def MVC(self):
        fdir=data_root + rf'\MODIS_LAI_0706\extract_tif_scaled\\'
        outdir=data_root + '/MODIS_LAI_0706/MVC/'
        T.mk_dir(outdir,force=True)
        Pre_Process().monthly_compose(fdir,outdir,method='max')

    # def MVC_wen(self):
    #     fdir = data_root + '/WesternUS_MODIS_LAI_005deg_2000_2024/extract_tif_scaled/'
    #     outdir = data_root + '/WesternUS_MODIS_LAI_005deg_2000_2024/MVC/'
    #     T.mk_dir(outdir, force=True)
    #     year_list=list(range(2003,2025))
    #     month_list=list(range(1,13))
    #
    #     for year in tqdm(year_list):
    #         for month in month_list:
    #
    #             f_list = []
    #             for f in T.listdir(fdir):
    #                 if not f.endswith('.tif'):
    #                     continue
    #                 if int(f.split('.')[0][0:4]) == year and int(f.split('.')[0][4:6]) == month:
    #                     f_list.append(join(fdir, f))
    #             if len(f_list) == 0:
    #                 continue
    #             outpath=join(outdir,rf'{year}{month:02d}.tif')

    def raster_align(self):
        fdir=data_root+rf'\MODIS_LAI_0706\MVC\\'
        outdir=join(data_root, 'MODIS_LAI_0706','raster_align' )
        T.mk_dir(outdir,force=True)
        reference_path=data_root+rf'\Terraclimate\ppt\extract_tif\\20020201.tif'
        for f in tqdm(os.listdir(fdir)):
            fpath=join(fdir, f)
            outpath=join(outdir, f)
            My_functions().align_tif(fpath,reference_path,outpath)
        pass
    def tif_to_dic(self):

        fdir_all = data_root + rf'\MODIS_LAI_0706\raster_align\\'
        outdir=data_root + '/MODIS_LAI_0706/dic/'
        T.mk_dir(outdir, force=True)

        year_list = list(range(2003, 2025))
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
            # array_unify = array[:3600][:3600,
            #               :7200]


            array[array < -999] = np.nan
            array[array > 10] = np.nan
            # array[array ==0] = np.nan

            array[array < 0] = np.nan


            # plt.imshow(array)
            # plt.show()

            # plt.imshow(array)
            # plt.show()

            array_dryland = array
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

    def spring_season_LAI_mean(self):
        fdir=data_root + '\MODIS_LAI_0706\dic\\'
        outdir=result_root + 'MODIS_LAI\\'
        T.mk_dir(outdir,force=True)
        spatial_dic=T.load_npy_dir(fdir)
        spring_result_dic={}
        summer_result_dic={}
        for pix in tqdm(spatial_dic):
            r,c=pix
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals=np.array(vals)
            vals=np.reshape(vals,(-1,12))
            # plt.imshow(vals)
            # plt.show()
            spring_list=[]
            summer_list=[]

            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to may
                spring_val=np.nanmean(vals[i][2:5])
                ## july to Oct
                summer_val=np.nanmean(vals[i][6:10])

                spring_list.append(spring_val)
                summer_list.append(summer_val)
            spring_result_dic[pix]=spring_list
            summer_result_dic[pix]=summer_list
        outspring=outdir+'spring_LAI.npy'
        np.save(outspring,spring_result_dic)
        outsummer=outdir+'summer_LAI.npy'
        np.save(outsummer,summer_result_dic)

    def growing_season(self):
        fdir=data_root + '\MODIS_LAI_0706\dic\\'
        outdir=result_root + 'MODIS_LAI\\'
        T.mk_dir(outdir,force=True)
        spatial_dic=T.load_npy_dir(fdir)
        growing_season_result_dic={}

        for pix in tqdm(spatial_dic):
            r,c=pix
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals=np.array(vals)
            vals=np.reshape(vals,(-1,12))
            # plt.imshow(vals)
            # plt.show()
            growing_season_list=[]


            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to oct
                growing_season_val=np.nanmean(vals[i][2:10])


                growing_season_list.append(growing_season_val)
            # plt.plot(growing_season_list)
            # plt.show()

            growing_season_result_dic[pix]=growing_season_list

        outspring=outdir+'growing_season_LAI.npy'
        np.save(outspring,growing_season_result_dic)






    def extract_growing_season_monthly(self):
        fdir = data_root + rf'\MODIS_LAI\dic\\'

        outdir = data_root + r'\MODIS_LAI\extract_growing_season_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = data_root + rf'/SNU_LAI/4GST/4GST.npy'
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

                        # lon, lat = D.pix_to_lon_lat(pix)
                        #

                        time_series_reshape = time_series_flatten.reshape(-1, 12)
                        # plt.imshow(time_series_reshape)
                        #
                        # plt.title(f'lon:{lon}, lat:{lat},SOS_monthly:{SOS_monthly}, EOS_monthly:{EOS_monthly}')
                        # plt.show()
                        # plt.plot(time_series_reshape[0])
                        # plt.show()
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
            # np.save(outf, result_dic)


    def extract_growing_season_LAI_mean(self):  ## extract LAI average
        fdir = data_root + r'/SNU_LAI/extract_growing_season_monthly/'

        outdir = data_root + r'/SNU_LAI/extract_growing_season_LAI_mean/'

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
                if len(vals_growing_season) == 42:
                    plt.plot(val)
                    plt.show()

                sum_growing_season = np.nanmean(val)

                growing_season_mean_list.append(sum_growing_season)

            result_dic[pix] = {
                'growing_season': growing_season_mean_list,
            }

        outf = outdir + 'growing_season_LAI_mean.npy'

        np.save(outf, result_dic)


    def spatial_plot(self):
        f = data_root + r'/SNU_LAI/extract_growing_season_LAI_min/' + 'growing_season_LAI_min.npy'
        dic = T.load_npy(f)
        spatial_dic = {}
        for pix in tqdm(dic):
            r, c = pix
            vals_growing_season = dic[pix]['growing_season']
            spatial_dic[pix] = np.nanmean(vals_growing_season)
        arr = D.pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()

        pass

    def trend_analysis(self):  ##each window average trend

        fdir = result_root + rf'\anomaly\\'
        outdir = result_root + rf'anomaly\\trend_analysis\\ '
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            # if not 'DLEM_S2_lai' in f:
            #     continue

            outf = outdir + f.split('.')[0]
            if os.path.isfile(outf + '_trend.tif'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix

                    ## ignore the last one year

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
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

class Data_processing_Terraclimate:
    def run(self):
        # self.download_all()
        # self.nc_to_tif_time_series_fast()
        # self.resample()
        # self.extract_tif_from_shp()
        # self.tif_to_dic()
        # self.spring_season_LAI_mean()
        # self.growing_season()
        # self.winter_precip()
        self.anomaly()
        # self.detrend()

        pass
    pass

    def download_all(self):
        params_list = []
        # product_list = ['ppt','vpd','tmax','tmin','soil','srad']

        product_list = ['ppt']
        year_list = list(range(2002, 2023))
        # year_list = list(range(1982, 1982))
        for product in product_list:
            for y in year_list:
                params_list.append([product, str(y)])
                params = [product, str(y)]
                # self.download(params)
        # print(len(params_list))
        # exit()
        MULTIPROCESS(self.download, params_list,istqdm=True).run(process=12,njobs=12, process_or_thread='t')
        # job_name = 'Terraclimate_download'
        # log_folder = join(logs_root,'download_data/Terraclimate/download_all')
        # init_job(job_name, params_list)
        # sumbit_jobs_array(self.download, params_list, log_folder, job_name=job_name,
        #                   job_number_limit=3,
        #                   parallel_process_per_task=10,
        #                   slurm_array_parallelism=3,
        #                   parallel_process_p_or_t='t',
        #                   cpus_per_task=2,
        #                   mem_gb=4,
        #                   timeout_min=10,
        #                   slurm_partition="general",
        #                   pbar_update_freq=1,
        #                   )
        # progress_bar_monitoring(job_name)

    def download(self, params):
        product, y = params
        outdir = join(data_root, 'TerraClimate', product)
        # print(outdir);exit()

        T.mkdir(outdir,True)


        url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{}_{}.nc'.format(product, y)
        # print(url)
        # exit()
        while 1:
            try:
                outf = join(outdir, '{}_{}.nc'.format(product, y))
                if os.path.isfile(outf):
                    return None
                req = requests.request('GET', url)
                content = req.content
                fw = open(outf, 'wb')
                fw.write(content)
                return None
            except Exception as e:
                print(url, 'error sleep 5s')
                time.sleep(5)
    def nc_to_tif_time_series_fast(self):
        '/data/home/wenzhang/Wen_Projects/Hotdrought_recovery/Data/TerraClimate/PPT'

        var='ppt'
        fdir = join(data_root, 'TerraClimate',f'{var}', 'nc')
        outdir = join(data_root, 'TerraClimate',f'{var}' ,'tiff')

        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):

            outdir_name = f.split('.')[0]
            # print(outdir_name)


            fpath = join(fdir, f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)
            time_bnds = nc_in['time']
            for t in range(len(time_bnds)):
                date = time_bnds[t]['time']
                date = pd.to_datetime(date.values)
                date_str = date.strftime('%Y%m%d')
                date_str = date_str.split()[0]
                outf = join(outdir, f'{date_str}.tif')
                array = nc_in[f'{var}'][t]

                array = np.array(array)
                # array[array < 0] = np.nan

                longitude_start = nc_in['lon'].values[0]
                latitude_start = nc_in['lat'].values[0]
                pixelWidth = nc_in['lon'].values[1] - nc_in['lon'].values[0]
                pixelHeight = nc_in['lat'].values[1] - nc_in['lat'].values[0]
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()



    def resample(self):
        fdir=join(data_root, 'TerraClimate','ppt', 'tiff')
        outdir=join(data_root, 'TerraClimate','ppt', 'resample')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath=join(fdir, f)
            outf=join(outdir, f)
            dataset = gdal.Open(fpath)


            try:
                gdal.Warp(outf, dataset, xRes=0.05, yRes=0.05, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass

    def extract_tif_from_shp(self):
        shp_f=data_root + 'basedata/Western_US_bountry/merged_western_US.shp'
        fdir=join(data_root, 'TerraClimate','ppt', 'resample')
        outdir=join(data_root, 'TerraClimate','ppt', 'extract_tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):

            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=join(outdir,f)
            if isfile(outf):
                continue

            ToRaster().clip_array(fpath, outf,shp_f)


        pass


    def tif_to_dic(self):

        fdir_all = join(data_root, 'TerraClimate', 'ppt', 'extract_tif')
        outdir = join(data_root, 'TerraClimate', 'ppt', 'dic')
        T.mk_dir(outdir, force=True)

        year_list = list(range(2002, 2025)) ##
        # 一般是2003 开始只有ppt是2002 因为要考虑冬天
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
            # array_unify = array[:3600][:3600,
            #               :7200]


            array[array < -999] = np.nan
            # array_unify[array_unify > 10] = np.nan
            # array[array ==0] = np.nan

            # array[array < 0] = np.nan


            # plt.imshow(array)
            # plt.show()




            array_dryland = array
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
                np.save(outdir + rf'/per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + rf'/per_pix_dic_%03d' % 0, temp_dic)

    def spring_season_LAI_mean(self):
        fdir=data_root + rf'\Terraclimate\srad\dic\\'
        outdir=result_root + rf'Terraclimate\\'
        T.mk_dir(outdir,force=True)
        var=fdir.split('\\')[-4]
        # print(var);exit()
        spatial_dic=T.load_npy_dir(fdir)
        spring_result_dic={}
        summer_result_dic={}
        for pix in tqdm(spatial_dic):
            r,c=pix
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals=np.array(vals)
            vals=np.reshape(vals,(-1,12))
            # plt.imshow(vals)
            # plt.show()
            spring_list=[]
            summer_list=[]

            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to may
                spring_val=np.nansum(vals[i][2:5])
                ## july to oct
                summer_val=np.nansum(vals[i][6:10])

                spring_list.append(spring_val)
                summer_list.append(summer_val)
            print(len(spring_list))
            spring_result_dic[pix]=spring_list
            summer_result_dic[pix]=summer_list
        outspring=outdir+rf'{var}_spring'
        outsummer=outdir+rf'{var}_summer'
        T.save_npy(spring_result_dic,outspring)
        T.save_npy(summer_result_dic,outsummer)

    def growing_season(self):
        variable_list=['srad','vpd','tmin','tmax','ppt','soil']
        fdir = data_root + rf'\Terraclimate\srad\dic\\'
        outdir = result_root + rf'Terraclimate\\growing_season\\'
        var = fdir.split('\\')[-4]
        T.mk_dir(outdir,force=True)
        spatial_dic=T.load_npy_dir(fdir)
        growing_season_result_dic={}

        for pix in tqdm(spatial_dic):
            r,c=pix
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals=np.array(vals)
            vals=np.reshape(vals,(-1,12))
            # plt.imshow(vals)
            # plt.show()
            growing_season_list=[]


            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to oct
                growing_season_val=np.nanmean(vals[i][2:10])


                growing_season_list.append(growing_season_val)
            print(len(growing_season_list))

            growing_season_result_dic[pix]=growing_season_list


        outspring=outdir+rf'{var}_growing_season'
        np.save(outspring,growing_season_result_dic)



    def winter_precip(self):
        fdir=data_root + rf'\Terraclimate\ppt\dic\\'
        outdir=result_root + rf'Terraclimate\\'
        T.mk_dir(outdir,force=True)
        var=fdir.split('\\')[-4]
        # print(var);exit()
        spatial_dic=T.load_npy_dir(fdir)
        winter_result_dic={}

        for pix in tqdm(spatial_dic):
            r,c=pix
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals=np.array(vals)
            vals=np.reshape(vals,(-1,12))
            # plt.imshow(vals)
            # plt.show()
            winter_list=[]

            for i in range(1, len(vals)):
                prev_oct_dec = vals[i - 1][10:12]  #  Nov Dec
                curr_jan_feb = vals[i][0:2]  # Jan Feb

                winter = np.concatenate([prev_oct_dec, curr_jan_feb])

                winter_mean = np.nansum(winter)  # 如果是降雨建议用sum
                winter_list.append(winter_mean)
            # print(len(winter_list))

            winter_result_dic[pix] = winter_list



        outwinter=outdir+rf'{var}_winter_npy'

        T.save_npy(winter_result_dic,outwinter)



    def anomaly(self):

        fdir = result_root + rf'\SM\\'
        outdir = result_root + rf'\\anomaly\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            # if not 'winter' in f:
            #     continue


            outf = outdir + f.split('.')[0]+'_anomaly.npy'
            # if isfile(outf):
            #     continue
            print(outf)

            dic = T.load_npy(fdir + f)

            zscore_dic = {}

            for pix in tqdm(dic):



                time_series = dic[pix]
                print(time_series)

                # # 检查 time_series 是否为 list 或 array（防止是 float/NaN）

                if not isinstance(time_series, (list, np.ndarray)):
                    print(f"{pix}: invalid time_series (not iterable): {time_series}")
                    continue

                time_series = np.array(time_series, dtype=float)
                # time_series = time_series[3:37]

                print(len(time_series))
                ## exclude nan




                if np.isnan(np.nanmean(time_series)):
                    continue
                # if np.nanmean(time_series) >999:
                #     continue
                if np.nanmean(time_series) <-999:
                    continue
                time_series = time_series
                mean = np.nanmean(time_series)
                zscore = (time_series - mean)




                zscore_dic[pix] = zscore


                plt.plot(time_series)
                #
                #
                # plt.plot(zscore)
                #
                # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)

    def detrend(self):

        fdir = result_root + rf'\anomaly\climate\\'
        outdir = result_root + rf'detrend\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'winter' in f:
                continue



            print(f)

            outf = outdir + f.split('.')[0] + '_detrend.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

            detrend_zscore_dic = {}

            for pix in tqdm(dic):



                r, c = pix
                # print(len(dic[pix]))
                time_series = dic[pix]
                print(len(time_series))
                # print(time_series)
                time_series = np.array(time_series, dtype=float)
                # plt.plot(time_series)
                # plt.show()
                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.std(time_series) == 0:
                    continue
                ##### if count of nan is more than 50%, then skip
                if np.sum(np.isnan(time_series)) / len(time_series) > 0.5:
                    continue
                # mean = np.nanmean(time_series)
                # std=np.nanstd(time_series)
                # if std == 0:
                #     continue
                # delta_time_series = (time_series - mean) / std
                # if np.isnan(time_series).any():
                #     continue
                time_series = T.interp_nan(time_series)
                detrend_delta_time_series = T.detrend_vals(time_series)
                # plt.plot(time_series)
                # plt.plot(detrend_delta_time_series)
                # plt.show()

                detrend_zscore_dic[pix] = detrend_delta_time_series

            np.save(outf, detrend_zscore_dic)


class calculating_mean_CV:
    def __init__(self):
        pass
    def run(self):
        self.calculating_mean_terrclimate()
        # self.calculating_mean_LAI()
        # self.calculating_CV()
        # self.calculating_annual_mean()

    def calculating_mean_terrclimate(self):
        fdir = result_root + rf'\Terraclimate\climate\\'
        outdir = result_root + rf'\\calculating_mean\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'tmean' in f:
                continue
            result_dic = {}
            if not f.endswith('.npy'):
                continue
            spatial_dic = T.load_npy(fdir + f)
            for pix in tqdm(spatial_dic):
                vals = spatial_dic[pix]
                mean_value = np.nanmean(vals)
                result_dic[pix] = mean_value
            outf = outdir + f.split('.')[0] + '_mean.npy'
            T.save_npy(result_dic,outf)


        pass

    def calculating_mean_LAI(self):
        fdir = result_root + rf'\MODIS_LAI\MODIS_LAI\\'
        outdir = result_root + rf'\\calculating_mean\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            result_dic = {}
            if not f.endswith('.npy'):
                continue
            spatial_dic = T.load_npy(fdir + f)
            for pix in tqdm(spatial_dic):
                vals = spatial_dic[pix]
                mean_value = np.nanmean(vals)
                result_dic[pix] = mean_value
            outf = outdir + f.split('.')[0] + '_mean.npy'
            T.save_npy(result_dic, outf)

        pass
    def calculating_CV(self):
        fdir=result_root + rf'\detrend\climate\\'
        outdir = result_root + rf'\calculating_CV\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'tmax' in f:
                continue
            result_dic = {}
            if not f.endswith('.npy'):
                continue
            dic= T.load_npy(fdir + f)
            for pix in tqdm(dic):
                vals = dic[pix]
                mean_value = np.nanmean(vals)
                std_value=np.nanstd(vals)
                result_dic[pix] = std_value/mean_value*100
            outf = outdir + f.split('.')[0] + '_CV.npy'
            T.save_npy(result_dic, outf)


        pass

    def calculating_annual_mean(self):
        fdir_tmax=data_root + rf'\Terraclimate\tmax\dic\\'
        fdir_tmin=data_root + rf'\Terraclimate\tmin\dic\\'

        outdir = data_root + rf'\Terraclimate\\tmean\\dic\\'
        Tools().mk_dir(outdir, force=True)
        tmax_dic=T.load_npy_dir(fdir_tmax)
        tmin_dic=T.load_npy_dir(fdir_tmin)
        result_dic={}
        for pix in tqdm(tmin_dic):
            if not pix in tmax_dic:
                continue
            tmax_vals=tmax_dic[pix]
            if np.isnan(np.nanmax(tmax_vals)):
                continue
            tmin_vals=tmin_dic[pix]
            tmax_reshape=np.reshape(tmax_vals,(-1,12))
            tmax_annual_mean=np.nanmean(tmax_reshape,axis=1)
            tmin_reshape=np.reshape(tmin_vals,(-1,12))
            tmin_annual_mean=np.nanmean(tmin_reshape,axis=1)


            tmean=(tmax_annual_mean+tmin_annual_mean)/2
            # plt.plot(tmax_annual_mean, 'r')
            # plt.plot(tmin_annual_mean, 'b')
            # plt.plot(tmean,'g')
            # plt.show()
            result_dic[pix]=tmean
        T.save_npy(result_dic,outdir+'dic.npy')









        pass

class Trend_analysis:
    def __init__(self):
        pass

    def run(self):
        self.trend_analysis()
        pass

    def trend_analysis(self):

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        ##each window average trend

        fdir = result_root + r'\anomaly\\'
        outdir = result_root + r'\anomaly\\trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            # if not 'ppt_winter_npy' in f:
            #     continue


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

            fpath = data_root + rf'\basedata\200902.tif'
            ll, lr, ul, ur = RasterIO_Func().get_tif_bounds(fpath)
            print(ll, lr, ul, ur)

            ax = plt.axes(projection=ccrs.PlateCarree())

           # # --- 画趋势图 ---
            im = ax.imshow(
                arr_trend,
                cmap='RdBu',
                vmin=-0.01,
                vmax=0.01,
                extent=[-124.55, -102.04, 25.59, 49],
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
    pass
class Data_processing_Daymet:
    def __init__(self):
        pass
    def run(self):
        # self.download()
        # self.daymet_nc_to_tif()
        # self.resample()
        # self.extract_tif_from_shp()
        # self.transform_to_blocks()
        # self.blocks_to_dict()
        # self.read_dict()
        # self.extract_spring_summer_rainfall_metrics() ## not use
        # self.extract_spring_summer_rainfall_intensity()
        # self.extract_spring_summer_rainfall_amount()
        self.extract_spring_summer_rainfall_fq()
        # self.extract_spring_summer_rainfall_dry_spell()
        # self.zscore()

        # self.trend_analysis()

        pass

    def download(self):

        import os
        import requests
        from requests.auth import HTTPBasicAuth

        # Earthdata账号
        username = "leeyang1991"
        password = "Asdfasdf911007"

        outdir = r"D:\Daymet\prcp"
        os.makedirs(outdir, exist_ok=True)

        base = "https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data"

        for year in range(2003, 2025):

            fname = f"daymet_v4_daily_na_prcp_{year}.nc"
            url = f"{base}/{fname}"

            outfile = os.path.join(outdir, fname)

            if os.path.exists(outfile):
                print(fname, "exists")
                continue

            print("Downloading", fname)

            r = requests.get(
                url,
                auth=HTTPBasicAuth(username, password),
                stream=True,
            )

            if r.status_code != 200:
                print(year, r.status_code)
                continue

            with open(outfile, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print("Finished!")



    def daymet_nc_to_tif(self):

        fdir = join(data_root, 'Daymet', 'prcp','nc')
        # outdir=rf'C:\\\Daymet\\prcp\\tiff'
        # print(outdir);exit()

        outdir = join(data_root, 'Daymet', 'prcp','tiff')
        T.mkdir(outdir, force=True)
        for f in os.listdir(fdir):

            year=f.split('.')[0].split('_')[-1]
            # print(year);exit()
            year=int(year)
            if year<2017:
                continue


            fpath=join(fdir,f)

            ncin = Dataset(fpath, 'r')
            ncin_xarr = xr.open_dataset(fpath)
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            time_list = ncin_xarr['time'][:]
            # pprint(time_list)
            # for t in time_list:
            vals_3d = ncin_xarr['prcp']
            lcc_crs = pyproj.CRS.from_string(
                "+proj=lcc +lat_1=25.0 +lat_2=60.0 +lat_0=42.5 +lon_0=-100.0 "
                "+x_0=0.0 +y_0=0.0 +ellps=GRS80 +units=m +no_defs"
            )
            wgs84_crs = DIC_and_TIF().wkt_84()
            transformer = pyproj.Transformer.from_crs(wgs84_crs, lcc_crs, always_xy=True)
            X, Y = transformer.transform(lon, lat)
            x_min, x_max = X.min(), X.max()
            y_min, y_max = Y.min(), Y.max()

            for i, value in tqdm(enumerate(vals_3d), total=len(time_list)):
                t = time_list[i]
                dt_obj = pd.to_datetime(t.values).to_pydatetime()
                year = dt_obj.year
                month = dt_obj.month
                day = dt_obj.day
                outf = join(outdir, f'{year:04d}{month:02d}{day:02d}.tif')
                if isfile(outf):
                    continue

                value = np.array(value)
                # value[value<=0] = np.nan
                height, width = value.shape

                # 构建原始 LCC 的仿射变换 (从左上角开始，x递增，y递减)
                src_transform = rasterio.transform.from_bounds(
                    x_min, y_min, x_max, y_max, width, height
                )

                dst_transform, dst_width, dst_height = calculate_default_transform(
                    lcc_crs,
                    wgs84_crs,
                    width,
                    height,
                    left=x_min,
                    bottom=y_min,
                    right=x_max,
                    top=y_max,
                )

                dst_value = np.ones((dst_height, dst_width), dtype=value.dtype) * np.nan

                reproject(
                    source=value,
                    destination=dst_value,
                    src_transform=src_transform,
                    src_crs=lcc_crs,
                    dst_transform=dst_transform,
                    dst_crs=wgs84_crs,
                    # resampling=Resampling.bilinear,  # 双线性插值，如果是分类数据请用 Resampling.nearest
                    resampling=Resampling.nearest,  # 双线性插值，如果是分类数据请用 Resampling.nearest
                    dst_nodata=np.nan
                )

                # dst_value
                with rasterio.open(
                        outf,
                        "w",
                        driver="GTiff",
                        height=dst_height,
                        width=dst_width,
                        count=1,
                        dtype=value.dtype,
                        crs=wgs84_crs,
                        transform=dst_transform,
                        nodata=np.nan,
                        bigtiff=True,
                ) as dst:
                    dst.write(dst_value, 1)




    pass

    def resample(self):
        fdir=join(data_root, 'Daymet','prcp', 'tiff')
        outdir=join(data_root, 'Daymet', 'prcp', 'resample')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath=join(fdir, f)
            outf=join(outdir, f)
            if isfile(outf):
                continue
            dataset = gdal.Open(fpath)


            try:
                gdal.Warp(outf, dataset, xRes=0.05, yRes=0.05, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass

    def extract_tif_from_shp(self):
        shp_f=data_root + 'basedata/Western_US_bountry/merged_western_US.shp'
        fdir=join(data_root, 'Daymet','prcp', 'resample')
        outdir=join(data_root, 'Daymet','prcp', 'extract_tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):

            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=join(outdir,f)

            ToRaster().clip_array(fpath, outf,shp_f)


        pass
    def transform_to_blocks(self):
        fdir=join(data_root, 'Daymet','prcp', 'extract_tif')
        outdir=join(data_root, 'Daymet','prcp', 'transform_tif')
        T.mk_dir(outdir,force=True)
        fpath_list=[]
        band_name_list=[]
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            fpath_list.append(fpath)
            band_name_list.append(f)
        Tif_loader_Height(fpath_list,block_height=20).transform_to_block(outdir,band_name_list=band_name_list)






        pass
    def blocks_to_dict(self):
        fdir = r'D:\Western_US_IAV\Data\Daymet\prcp\transform_tif'
        outdir = r'D:\Western_US_IAV\Data\Daymet\prcp\transform_dic'
        T.mk_dir(outdir,force=True)
        flist = T.listdir_full(fdir)
        Block_Handler(flist).transform_block_to_spatial_dict(outdir)


        pass

    def read_dict(self):
        fdir = r'D:\Western_US_IAV\Data\Daymet\prcp\transform_dic'
        for f in T.listdir(fdir):
            if not '12' in f:
                continue
            fpath = join(fdir, f)
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                print(pix)
                vals = spatial_dict[pix]
                plt.plot(vals)
                plt.title(pix)
                plt.show()

        pass

    def extract_spring_summer_rainfall_metrics(self):
        fdir = r'D:\Western_US_IAV\Data\Daymet\prcp\transform_dic'
        spring_result = {}
        summer_result = {}
        for f in T.listdir(fdir):
            # if not '05' in f:
            #     continue
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir, f)
            spatial_dict = T.load_npy(fpath)

            for pix in tqdm(spatial_dict):

                vals = np.array(spatial_dict[pix])

                # 每年365天
                vals = vals.reshape(-1, 365)

                spring_frequency_list = []
                spring_intensity_list = []
                spring_amount_list = []
                spring_dry_spell_list=[]

                summer_frequency_list = []
                summer_intensity_list = []
                summer_amount_list = []
                summer_dry_spell_list = []

                for i in range(vals.shape[0]):

                    # --------------------------
                    # Spring (Mar-May)
                    # --------------------------
                    spring_vals = vals[i, 59:151]

                    # 总降雨
                    spring_amount = np.nansum(spring_vals)

                    spring_dry_spell = self.longest_dry_spell(
                        spring_vals,
                        threshold=3
                    )

                    spring_dry_spell_list.append(spring_dry_spell)

                    # 雨日 (>3 mm)
                    spring_wet = spring_vals[spring_vals > 3]
                    spring_frequency = len(spring_wet)

                    # 雨强
                    if spring_frequency > 0:
                        spring_intensity = np.nanmean(spring_wet)
                    else:
                        spring_intensity = np.nan

                    spring_amount_list.append(spring_amount)
                    spring_frequency_list.append(spring_frequency)
                    spring_intensity_list.append(spring_intensity)

                    # --------------------------
                    # Summer (Jul-Oct)
                    # --------------------------
                    summer_vals = vals[i, 181:304]

                    summer_amount = np.nansum(summer_vals)

                    summer_wet = summer_vals[summer_vals > 3]
                    summer_frequency = len(summer_wet)
                    summer_dry_spell = self.longest_dry_spell(summer_vals,threshold=3)

                    if summer_frequency > 0:
                        summer_intensity = np.nanmean(summer_wet)
                    else:
                        summer_intensity = np.nan

                    summer_amount_list.append(summer_amount)
                    summer_frequency_list.append(summer_frequency)
                    summer_intensity_list.append(summer_intensity)
                    summer_dry_spell_list.append(summer_dry_spell)

                # 保存
                spring_result[pix] = {
                    'frequency': spring_frequency_list,
                    'intensity': spring_intensity_list,
                    'amount': spring_amount_list,
                    'maximum_dry_spell': spring_dry_spell_list,
                }

                summer_result[pix] = {
                    'frequency': summer_frequency_list,
                    'intensity': summer_intensity_list,
                    'amount': summer_amount_list,
                    'maximum_dry_spell': summer_dry_spell_list,
                }
        outdir = result_root + rf'\Daymet\\'
        T.mkdir(outdir,force=True)
        outf_spring=outdir+rf'spring_rainfall_metrics.npy'
        np.save(outf_spring,spring_result)
        outf_summer=outdir+rf'summer_rainfall_metrics.npy'
        np.save(outf_summer,summer_result)



                # # ---------- 测试画图 ----------
                # years = np.arange(2003, 2025)
                #
                # plt.figure(figsize=(10, 6))
                #
                # plt.subplot(411)
                # plt.plot(years, spring_frequency_list, '-o', label='Spring')
                # plt.plot(years, summer_frequency_list, '-o', label='Summer')
                # plt.ylabel('Frequency')
                # plt.legend()
                #
                # plt.subplot(412)
                # plt.plot(years, spring_intensity_list, '-o', label='Spring')
                # plt.plot(years, summer_intensity_list, '-o', label='Summer')
                # plt.ylabel('Intensity (mm/day)')
                #
                # plt.subplot(413)
                # plt.plot(years, spring_amount_list, '-o', label='Spring')
                # plt.plot(years, summer_amount_list, '-o', label='Summer')
                # plt.ylabel('Amount (mm)')
                # plt.xlabel('Year')
                #
                # plt.subplot(414)
                # plt.plot(years, spring_dry_spell_list, '-o', label='Spring')
                # plt.plot(years, summer_dry_spell_list, '-o', label='Summer')
                # plt.ylabel('Dry Spell (mm/day)')
                # plt.xlabel('Year')
                #
                # plt.tight_layout()
                # plt.show()

        pass

    def extract_spring_summer_rainfall_intensity(self):
        fdir = r'C:\Users\wenzhang1.BLUECAT\Desktop\\transform_dic'
        spring_result = {}
        summer_result = {}
        growing_season_result = {}
        for f in T.listdir(fdir):
            # if not '05' in f:
            #     continue
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir, f)
            spatial_dict = T.load_npy(fpath)

            for pix in tqdm(spatial_dict):

                vals = np.array(spatial_dict[pix])

                # 每年365天
                vals = vals.reshape(-1, 365)


                spring_intensity_list = []

                growing_season_intensity_list=[]

                summer_intensity_list = []


                for i in range(vals.shape[0]):

                    # --------------------------
                    # Spring (Mar-May)
                    # --------------------------
                    spring_vals = vals[i, 59:151]

                    # 雨日 (>3 mm)
                    spring_wet = spring_vals[spring_vals > 1]
                    spring_frequency = len(spring_wet)


                    # 雨强
                    if spring_frequency > 0:
                        spring_intensity = np.nanmean(spring_wet)
                    else:
                        spring_intensity = 0


                    spring_intensity_list.append(spring_intensity)

                    ##growing seson

                    growing_season_vals = vals[i, 59:304]

                    # 雨日 (>3 mm)
                    growing_season_wet = growing_season_vals[growing_season_vals > 1]
                    growing_season_frequency = len(growing_season_wet)

                    # 雨强
                    if growing_season_frequency > 0:
                        growing_season_intensity = np.nanmean(growing_season_wet)
                    else:
                        growing_season_intensity = 0

                    growing_season_intensity_list.append(growing_season_intensity)

                    # --------------------------
                    # Summer (Jul-Oct)
                    # --------------------------
                    summer_vals = vals[i, 181:304]


                    summer_wet = summer_vals[summer_vals > 1]
                    summer_frequency = len(summer_wet)


                    if summer_frequency > 0:
                        summer_intensity = np.nanmean(summer_wet)
                    else:
                        summer_intensity = 0


                    summer_intensity_list.append(summer_intensity)


                # 保存
                spring_result[pix] = spring_intensity_list
                growing_season_result[pix]=growing_season_intensity_list


                summer_result[pix] = summer_intensity_list
        outdir = result_root + rf'\Daymet\\'
        T.mkdir(outdir,force=True)
        outf_spring=outdir+rf'spring_rainfall_intensity.npy'
        np.save(outf_spring,spring_result)
        outf_summer=outdir+rf'summer_rainfall_intensity.npy'
        np.save(outf_summer,summer_result)
        outf_growing_season=outdir+rf'growing_season_rainfall_intensity.npy'
        np.save(outf_growing_season,growing_season_result)


    def extract_spring_summer_rainfall_fq(self):
        fdir = r'D:\Western_US_IAV\Data\Daymet\prcp\transform_dic'
        spring_result = {}
        summer_result = {}
        growing_result={}
        for f in T.listdir(fdir):
            # if not '05' in f:
            #     continue
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir, f)
            spatial_dict = T.load_npy(fpath)

            for pix in tqdm(spatial_dict):

                vals = np.array(spatial_dict[pix])

                # 每年365天
                vals = vals.reshape(-1, 365)

                spring_frequency_list = []
                growing_season_frequency_list=[]


                summer_frequency_list = []


                for i in range(vals.shape[0]):

                    # --------------------------
                    # Spring (Mar-May)
                    # --------------------------
                    spring_vals = vals[i, 59:151]


                    # 雨日 (>5 mm)
                    spring_wet = spring_vals[spring_vals > 5]
                    spring_frequency = len(spring_wet)
                    spring_frequency_list.append(spring_frequency)

                    ## growing season

                    growing_season_vals = vals[i, 59:304]
                    growing_season_wet = growing_season_vals[growing_season_vals > 5]
                    growing_season_frequency = len(growing_season_wet)
                    growing_season_frequency_list.append(growing_season_frequency)


############################  summer
                    summer_vals = vals[i, 181:304]

                    summer_wet = summer_vals[summer_vals > 5]
                    summer_frequency = len(summer_wet)

                    summer_frequency_list.append(summer_frequency)

                # 保存
                spring_result[pix] =spring_frequency_list

                summer_result[pix]=summer_frequency_list
                growing_result[pix]=growing_season_frequency_list
        outdir = result_root + rf'\Daymet\\'
        T.mkdir(outdir,force=True)
        outf_spring=outdir+rf'spring_rainfall_fq.npy'
        np.save(outf_spring,spring_result)
        outf_summer=outdir+rf'summer_rainfall_fq.npy'
        np.save(outf_summer,summer_result)
        outf_growing_season=outdir+rf'growing_season_rainfall_fq.npy'
        np.save(outf_growing_season,growing_result)

    def extract_spring_summer_rainfall_amount(self):
        fdir = r'D:\Western_US_IAV\Data\Daymet\prcp\transform_dic'
        spring_result = {}
        summer_result = {}
        for f in T.listdir(fdir):
            # if not '05' in f:
            #     continue
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir, f)
            spatial_dict = T.load_npy(fpath)

            for pix in tqdm(spatial_dict):

                vals = np.array(spatial_dict[pix])

                # 每年365天
                vals = vals.reshape(-1, 365)


                spring_amount_list = []

                summer_amount_list = []


                for i in range(vals.shape[0]):

                    # --------------------------
                    # Spring (Mar-May)
                    # --------------------------
                    spring_vals = vals[i, 59:151]

                    # 总降雨

                    spring_amount = np.nansum(spring_vals[ spring_vals > 3])  # 只计算降雨量大于3mm的总降雨量


                    spring_amount_list.append(spring_amount)


                    # --------------------------
                    # Summer (Jul-Oct)
                    # --------------------------
                    summer_vals = vals[i, 181:304]

                    summer_amount = np.nansum(summer_vals[summer_vals > 3])


                    summer_amount_list.append(summer_amount)

                # 保存
                spring_result[pix] = spring_amount_list

                summer_result[pix] = summer_amount_list

        outdir = result_root + rf'\Daymet\\'
        T.mkdir(outdir,force=True)
        outf_spring=outdir+rf'spring_rainfall_amount.npy'
        np.save(outf_spring,spring_result)
        outf_summer=outdir+rf'summer_rainfall_amount.npy'
        np.save(outf_summer,summer_result)


    def longest_dry_spell(self,vals, threshold=3):
        """
        vals: daily precipitation
        threshold: rainy day threshold (mm/day)

        return:
            maximum consecutive dry days
        """

        dry = vals < threshold

        max_len = 0
        current = 0

        for d in dry:

            if d:
                current += 1
                max_len = max(max_len, current)
            else:
                current = 0

        return max_len

    def extract_spring_summer_rainfall_dry_spell(self):
        fdir = r'D:\Western_US_IAV\Data\Daymet\prcp\transform_dic'
        spring_result = {}
        summer_result = {}
        for f in T.listdir(fdir):
            # if not '05' in f:
            #     continue
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir, f)
            spatial_dict = T.load_npy(fpath)

            for pix in tqdm(spatial_dict):

                vals = np.array(spatial_dict[pix])

                # 每年365天
                vals = vals.reshape(-1, 365)


                spring_dry_spell_list=[]


                summer_dry_spell_list = []

                for i in range(vals.shape[0]):

                    # --------------------------
                    # Spring (Mar-May)
                    # --------------------------
                    spring_vals = vals[i, 59:151]


                    spring_dry_spell = self.longest_dry_spell(
                        spring_vals,
                        threshold=3
                    )

                    spring_dry_spell_list.append(spring_dry_spell)



                    # --------------------------
                    # Summer (Jul-Oct)
                    # --------------------------
                    summer_vals = vals[i, 181:304]



                    summer_dry_spell = self.longest_dry_spell(summer_vals,threshold=3)




                    summer_dry_spell_list.append(summer_dry_spell)

                # 保存
                spring_result[pix] = spring_dry_spell_list

                summer_result[pix] = summer_dry_spell_list

        outdir = result_root + rf'\Daymet\\'
        T.mkdir(outdir,force=True)
        outf_spring=outdir+rf'spring_rainfall_maximum_dryspell.npy'
        np.save(outf_spring,spring_result)
        outf_summer=outdir+rf'summer_rainfall_maximum_dryspell.npy'
        np.save(outf_summer,summer_result)

    def zscore(self):

        fdir = result_root + rf'\MODIS_LAI\MODIS_LAI\\'
        outdir = result_root + rf'zscore\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'growing_season' in f:
                continue
            if not f.endswith('.npy'):
                continue


            outf = outdir + f.split('.')[0]+'_zscore.npy'
            # if isfile(outf):
            #     continue
            print(outf)

            dic = T.load_npy(fdir + f)

            zscore_dic = {}

            for pix in tqdm(dic):



                time_series = dic[pix]
                print(time_series)

                # # 检查 time_series 是否为 list 或 array（防止是 float/NaN）

                if not isinstance(time_series, (list, np.ndarray)):
                    print(f"{pix}: invalid time_series (not iterable): {time_series}")
                    continue

                time_series = np.array(time_series, dtype=float)
                # time_series = time_series[3:37]

                # print(len(time_series))
                ## exclude nan




                if np.isnan(np.nanmean(time_series)):
                    continue
                # if np.nanmean(time_series) >999:
                #     continue
                if np.nanmean(time_series) <-999:
                    continue
                time_series = time_series
                mean = np.nanmean(time_series)
                std= np.nanstd(time_series)
                zscore = (time_series - mean)/std


                zscore_dic[pix] = zscore

                #
                # plt.plot(time_series)
                # #
                # #
                # plt.plot(zscore)
                # #
                # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)

    def trend_analysis(self):

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        ##each window average trend

        fdir = result_root + r'\Daymet\\zscore\\'
        outdir = result_root + r'Daymet\\zscore\\trend_analysis\\'
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

            fpath = data_root + rf'\basedata\200902.tif'
            ll, lr, ul, ur = RasterIO_Func().get_tif_bounds(fpath)
            print(ll, lr, ul, ur)

            ax = plt.axes(projection=ccrs.PlateCarree())

           # # --- 画趋势图 ---
            im = ax.imshow(
                arr_trend,
                cmap='RdBu',
                vmin=-0.01,
                vmax=0.01,
                extent=[-124.55, -102.04, 25.59, 49],
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


    # def trend_analysis(self):  ## not use
    #
    #     import cartopy.crs as ccrs
    #     import cartopy.feature as cfeature
    #     import matplotlib.pyplot as plt
    #     ##each window average trend
    #
    #     fdir = result_root + r'\Daymet\\'
    #     outdir = result_root + r'\Daymet\\trend_analysis\\'
    #     Tools().mk_dir(outdir, force=True)
    #
    #     for f in os.listdir(fdir):
    #
    #         if not f.endswith('.npy'):
    #             continue
    #
    #         outf = join(outdir, f.split('.')[0])
    #         print(outf)
    #         # exit()
    #
    #         fpath = join(fdir, f)
    #
    #         dic = np.load(fpath, allow_pickle=True, encoding='latin1').item()
    #
    #         trend_dic = {}
    #         p_value_dic = {}
    #
    #         # 初始化四个指标
    #         variable_list = ['frequency', 'intensity', 'amount', 'maximum_dry_spell']
    #
    #         for var in variable_list:
    #             trend_dic[var] = {}
    #             p_value_dic[var] = {}
    #
    #         # ==============================
    #         # Calculate trend
    #         # ==============================
    #         for pix in tqdm(dic):
    #
    #             for var in variable_list:
    #
    #                 if var not in dic[pix]:
    #                     continue
    #
    #                 time_series = np.array(dic[pix][var], dtype=float)
    #
    #                 # 删除NaN
    #                 mask = ~np.isnan(time_series)
    #
    #                 if np.sum(mask) < 10:
    #                     continue
    #
    #                 x = np.arange(len(time_series))[mask]
    #                 y = time_series[mask]
    #
    #                 try:
    #                     slope, intercept, r, p = T.nan_line_fit(x, y)
    #
    #                     trend_dic[var][pix] = slope
    #                     p_value_dic[var][pix] = p
    #
    #                 except Exception:
    #                     continue
    #
    #         # ==============================
    #         # Save tif
    #         # ==============================
    #         for var in variable_list:
    #             arr_trend = D.pix_dic_to_spatial_arr(trend_dic[var])
    #             plt.imshow(arr_trend)
    #             plt.show()
    #             arr_p = D.pix_dic_to_spatial_arr(p_value_dic[var])
    #
    #             D.arr_to_tif(
    #                 arr_trend,
    #                 outf + f'_{var}_trend.tif'
    #             )
    #
    #             D.arr_to_tif(
    #                 arr_p,
    #                 outf + f'_{var}_p_value.tif'
    #             )
    #
    #
    #
    #         fpath = data_root + rf'\basedata\200902.tif'
    #         ll, lr, ul, ur = RasterIO_Func().get_tif_bounds(fpath)
    #         print(ll, lr, ul, ur)
    #
    #         ax = plt.axes(projection=ccrs.PlateCarree())
    #
    #        # # --- 画趋势图 ---
    #         im = ax.imshow(
    #             arr_trend,
    #             cmap='RdBu',
    #             vmin=-0.01,
    #             vmax=0.01,
    #             extent=[-124.55, -102.04, 25.59, 49],
    #             transform=ccrs.PlateCarree()
    #         )
    #
    #         # --- 加 continent ---
    #         ax.add_feature(
    #             cfeature.LAND,
    #             facecolor='none',  #
    #             edgecolor='black',
    #             linewidth=0.5,
    #             zorder=2
    #         )
    #         ax.add_feature(cfeature.STATES, linewidth=0.3)
    #
    #         lon_min_box = -125
    #         lon_max_box = -105
    #         lat_min_box = 30
    #         lat_max_box = 45
    #
    #         rect = mpatches.Rectangle(
    #             (lon_min_box, lat_min_box),  # 左下角 (lon, lat)
    #             lon_max_box - lon_min_box,  # 宽度
    #             lat_max_box - lat_min_box,  # 高度
    #             linewidth=1.5,
    #             edgecolor='black',
    #             facecolor='none',
    #             transform=ccrs.PlateCarree(),  # ⭐关键
    #             zorder=10
    #         )
    #
    #         ax.add_patch(rect)
    #         ax.set_xlabel('Longitude')
    #         ax.set_ylabel('Latitude')
    #
    #         cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    #         cbar.set_label('Trend')
    #
    #         plt.title(f)
    #         plt.show()
    #
    #         D.arr_to_tif(arr_trend, outf + '_trend.tif')
    #         D.arr_to_tif(arr_p, outf + '_p_value.tif')
    #
    #         np.save(outf + '_trend', arr_trend)
    #         np.save(outf + '_p_value', arr_p)
class Data_processing_ERA5land:

    def __init__(self):
        pass

    def run(self):

        # self.raster_align()

        # self.extract_tif_from_shp()
        # self.tif_to_dic()
        # self.winter_SWE()
        # self.spring_season_SM_mean()
        self.growing_season()

        pass

    def raster_align(self):  ## 这个函数包括了resample and align
        fdir = join(data_root, 'SWE', 'tiff_unify')
        outdir = join(data_root, 'SWE', 'raster_align')
        T.mk_dir(outdir,force=True)
        reference_path=data_root+rf'\basedata\\template_align.tif'
        for f in tqdm(os.listdir(fdir)):
            fpath=join(fdir, f)
            outpath=join(outdir, f)
            My_functions().align_tif(fpath,reference_path,outpath)
        pass



    def extract_tif_from_shp(self):
        shp_f=data_root + 'basedata/Western_US_bountry/merged_western_US.shp'
        fdir=join(data_root, 'SWE',  'raster_align')
        outdir=join(data_root, 'SWE',  'extract_tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):

            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=join(outdir,f)
            if isfile(outf):
                continue

            ToRaster().clip_array(fpath, outf,shp_f)


        pass



    def tif_to_dic(self):

        fdir_all = join(data_root, 'SWE',   'extract_tif')
        outdir = join(data_root, 'SWE',   'dic')
        T.mk_dir(outdir, force=True)

        year_list = list(range(2002, 2025))
        # 作为筛选条件

        all_array = []  #### so important  it should be go with T.mk_dic


        for f in T.listdir(fdir_all):
            print(f)

            if not f.endswith('.tif'):
                continue
            if int(f.split('_')[1][0:4]) not in year_list:
                continue


            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_all, f))
            array = np.array(array, dtype=float)


            # array_unify = array[:720][:720,
            #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
            # array_unify = array[:3600][:3600,
            #               :7200]


            array[array < -999] = np.nan
            # array_unify[array_unify > 10] = np.nan
            # array[array ==0] = np.nan

            # array[array < 0] = np.nan


            # plt.imshow(array)
            # plt.show()




            array_dryland = array
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
                np.save(outdir + rf'/per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + rf'/per_pix_dic_%03d' % 0, temp_dic)

    def winter_SWE(self):
        fdir=data_root + rf'\SWE\dic\\'
        outdir=result_root + rf'SWE\\'
        T.mk_dir(outdir,force=True)
        var='SWE'
        # print(var);exit()
        spatial_dic=T.load_npy_dir(fdir)
        winter_result_dic={}

        for pix in tqdm(spatial_dic):
            r,c=pix
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals=np.array(vals)
            vals=np.reshape(vals,(-1,12))
            # plt.imshow(vals)
            # plt.show()
            winter_list=[]

            for i in range(1, len(vals)):
                prev_oct_dec = vals[i - 1][11:12]  # Oct Nov Dec
                curr_jan_feb = vals[i][0:2]  # Jan Feb

                winter = np.concatenate([prev_oct_dec, curr_jan_feb])

                winter_mean = np.nanmean(winter)  # 如果是降雨建议用sum
                winter_list.append(winter_mean)
            # print(len(winter_list))

            winter_result_dic[pix] = winter_list



        outwinter=outdir+rf'{var}_winter'

        T.save_npy(winter_result_dic,outwinter)

    def growing_season(self):
        fdir = data_root + rf'\SM\L4\dic\\'
        outdir = result_root + rf'SM\\'
        T.mk_dir(outdir, force=True)
        var = 'SM_L4'
        # print(var);exit()
        spatial_dic = T.load_npy_dir(fdir)
        spring_result_dic = {}

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
            # plt.show()
            spring_list = []


            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to may
                spring_val = np.nansum(vals[i][2:10])


                spring_list.append(spring_val)

            spring_result_dic[pix] = spring_list

        outspring = outdir + rf'{var}_growing_season'

        T.save_npy(spring_result_dic, outspring)


    def spring_season_SM_mean(self):
        fdir=data_root + rf'\SM\L3\dic\\'
        outdir=result_root + rf'SM\\'
        T.mk_dir(outdir,force=True)
        var='SM_L3'
        # print(var);exit()
        spatial_dic=T.load_npy_dir(fdir)
        spring_result_dic={}
        summer_result_dic={}
        for pix in tqdm(spatial_dic):
            r,c=pix
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals=np.array(vals)
            vals=np.reshape(vals,(-1,12))
            # plt.imshow(vals)
            # plt.show()
            spring_list=[]
            summer_list=[]

            for i in range(len(vals)):
                # print(vals[i][2:5])
                ## march to may
                spring_val=np.nansum(vals[i][2:5])
                ## july to sept
                summer_val=np.nansum(vals[i][6:10])

                spring_list.append(spring_val)
                summer_list.append(summer_val)
            spring_result_dic[pix]=spring_list
            summer_result_dic[pix]=summer_list
        outspring=outdir+rf'{var}_spring'
        outsummer=outdir+rf'{var}_summer'
        T.save_npy(spring_result_dic,outspring)
        T.save_npy(summer_result_dic,outsummer)

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
        # self.plot_time_series()
        self.check_spatial_coverage()
    def plot_time_series(self):
        f=data_root+rf'\greening_analysis\relative_change\SNU_LAI_detrend.npy'
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
        fdir=result_root+rf'\anomaly\climate\\'

        for f in T.listdir(fdir):
            if not 'winter' in f:
                continue

            dic = T.load_npy(fdir+f)
            spatial_coverage = {}


            for pix in dic:
                vals = dic[pix]
                # if len(vals) == 42:
                #     plt.plot(vals)
                #     plt.show()

                if np.isnan(np.nanmean(vals)):
                    continue
                # print(len(vals))
                length = len(vals)
                spatial_coverage[pix] = length
            arr=D.pix_dic_to_spatial_arr(spatial_coverage)
            plt.imshow(arr,cmap='jet',vmin=21,vmax=22)
            plt.title(f)
            plt.colorbar()
            plt.show()

class Data_processing_carbonscope:
    def __init__(self):
        pass
    def run(self):
        # self.download_carbonscope()
        # self.nc_to_tif_time_series_fast()
        # self.resample()
        # self.extract_tif_from_shp()
        # self.aggregate_spring_summer_NEE()
        # self.tif_to_dic()

        self.plot_time_series()
    def download_carbonscope(self):
        url ='https://www.bgc-jena.mpg.de/CarboScope/INVERSION/OUTPUT/nbetEXToc_v2025/nbetEXToc_v2025.flux.nc'
        outdir =data_root+'carnonscope'
        T.mk_dir(outdir, force=True)
        T.download_file(url,
                        outf=None,
                        outdir=outdir,
                        num_threads =40,
                        )
    pass

    def nc_to_tif_time_series_fast(self):

        fdir = join(data_root, 'carbonscope', 'nc')
        outdir = join(data_root, 'carbonscope', 'tiff')

        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):

            outdir_name = f.split('.')[0]
            # print(outdir_name)


            fpath = join(fdir, f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)
            time_bnds = nc_in['mtime']
            for t in range(len(time_bnds)):
                date = time_bnds[t]['mtime']
                date = pd.to_datetime(date.values)
                date_str = date.strftime('%Y%m%d')
                date_str = date_str.split()[0]
                outf = join(outdir, f'{date_str}.tif')
                array = nc_in['co2flux_land'][t]

                array = np.array(array)
                # array[array < 0] = np.nan

                longitude_start = nc_in['lon'].values[0]
                latitude_start = nc_in['lat'].values[0]
                pixelWidth = nc_in['lon'].values[1] - nc_in['lon'].values[0]
                pixelHeight = nc_in['lat'].values[1] - nc_in['lat'].values[0]
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()

    def resample(self):
        fdir=join(data_root, 'carbonscope', 'tiff')
        outdir=join(data_root, 'carbonscope',  'resample')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            year=f.split('.')[0][0:4]
            year=int(year)
            if year<2003:
                continue
            if not f.endswith('.tif'):
                continue
            fpath=join(fdir, f)
            outf=join(outdir, f)
            if isfile(outf):
                continue
            dataset = gdal.Open(fpath)


            try:
                gdal.Warp(outf, dataset, xRes=0.05, yRes=0.05, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass

    def extract_tif_from_shp(self):
        shp_f=data_root + 'basedata/Western_US_bountry/merged_western_US.shp'
        fdir=join(data_root, 'carbonscope', 'resample')
        outdir=join(data_root, 'carbonscope', 'extract_tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):

            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)
            outf=join(outdir,f)
            if isfile(outf):
                continue

            ToRaster().clip_array(fpath, outf,shp_f)


        pass

    def aggregate_spring_summer_NEE(self):
        from datetime import datetime
        outdir=data_root+rf'/carbonscope/aggregate_spring_summer/'
        T.mk_dir(outdir,force=True)
        fdir=join(data_root, 'carbonscope', 'extract_tif')
        spring_month=[3,4,5]
        # summer_month=[7,8,9]

        file_list = sorted([f for f in os.listdir(fdir) if f.endswith('.tif')])

        years = range(2003, 2025)

        for year in tqdm(years):

            data = []
            profile = None

            for f in file_list:

                date = datetime.strptime(os.path.splitext(f)[0], "%Y%m%d")

                if date.year != year:
                    continue

                if date.month not in spring_month:
                    continue

                fpath = os.path.join(fdir, f)

                with rasterio.open(fpath) as src:

                    arr = src.read(1).astype(np.float32)

                    arr[arr < -9990] = np.nan

                    data.append(arr)

                    if profile is None:
                        profile = src.profile

            if len(data) == 0:
                print(year, "No files")
                continue

            out = np.nanmean(data, axis=0)
            D.arr_to_tif(out, os.path.join(outdir, f"spring_NEE_{year}.tif"))


    def tif_to_dic(self):

        fdir_all = join(data_root,'carbonscope', 'aggregate_spring_summer', 'spring' )
        outdir = join(data_root, 'carbonscope', 'dic_spring_annual')
        T.mk_dir(outdir, force=True)

        year_list = list(range(2003, 2025)) ##
        # 一般是2003 开始只有ppt是2002 因为要考虑冬天
        # 作为筛选条件

        all_array = []  #### so important  it should be go with T.mk_dic


        for f in T.listdir(fdir_all):
            print(f)

            if not f.endswith('.tif'):
                continue
            if int(f.split('_')[2][0:4]) not in year_list:
                continue


            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_all, f))
            array = np.array(array, dtype=float)


            # array_unify = array[:720][:720,
            #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
            # array_unify = array[:3600][:3600,
            #               :7200]


            array[array < -999] = np.nan
            # array_unify[array_unify > 10] = np.nan
            # array[array ==0] = np.nan

            # array[array < 0] = np.nan


            # plt.imshow(array)
            # plt.show()




            array_dryland = array
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
                np.save(outdir + rf'/per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + rf'/per_pix_dic_%03d' % 0, temp_dic)



    def plot_time_series(self):
        fdir = data_root + rf'\carbonscope\dic_summer_annual\\'
        dic = T.load_npy_dir(fdir)
        result_list=[]
        for pix in dic:
            vals = dic[pix]

            if np.isnan(np.nanmean(vals)):
                continue
            print(len(vals))
            time_series = dic[pix]
            result_list.append(time_series)
        vals_mean=np.nanmean(result_list, axis=0)
        yearlist=np.arange(2003, 2025)
        plt.plot(yearlist,vals_mean)
        plt.ylabel('summer NEE')
        plt.tight_layout()

        plt.show()

    pass


def main():

     # Data_processing_vegetation().run()
    # area_weighted_average().run()
    # Data_processing_MODIS_LAI().run()
    # Data_processing_Terraclimate().run()
    # calculating_mean_CV().run()
    Data_processing_Daymet().run()
    # Data_processing_ERA5land().run()
    # Data_processing_carbonscope().run()
    # Trend_analysis().run()
    # general_anaysis().run()


     # check_data().run()
    # convert_dic_to_tiff().run()

if __name__ == '__main__':
    main()