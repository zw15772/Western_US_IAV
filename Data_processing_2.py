# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
from openpyxl.styles.builtins import percent
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t


version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import re
import xarray

from osgeo import ogr
from osgeo import osr
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import copyreg
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
import sklearn
import random
# import h5py
from netCDF4 import Dataset
import shutil
import requests
from lytools import *
from osgeo import gdal

from osgeo import gdal

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from operator import itemgetter
from itertools import groupby
# import RegscorePy
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from mpl_toolkits.mplot3d import Axes3D
import pickle
from dateutil import relativedelta
from sklearn.inspection import permutation_importance
from pprint import pprint
T=Tools()
D = DIC_and_TIF(pixelsize=0.5)
centimeter_factor = 1/2.54


this_root = 'D:\Project3\\'
data_root = 'D:/Project3/Data/'
result_root = 'D:/Project3/Result/Nov/'
class Data_processing_2:

    def __init__(self):

        pass

    def run(self):

        # self.dryland_mask()
        # self.test_histogram()
        # self.resampleSOC()
        # self.reclassification_koppen()
        # self.aggregation_soil()
        # self.nc_to_tif_time_series_fast2()

        # self.resample()
        # self.scale()

        # self.aggregate()
        # self.aggregate_GlobMAP()
        # self.unify_TIFF()

        # self.extract_dryland_tiff()

        self.tif_to_dic()
        # self.interpolate_VCF()
        # self.interpolation()
        # self.mean()
        # self.mean_rainfall()
        # self.zscore()
        # self.anomaly()
        # self.composite_LAI()



        pass
    def dryland_mask(self):
        ## here we extract dryland tif
        fpath=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\aridity_index.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        NDVI_mask_f = rf'D:\Greening\Data\Base_data\\NDVI_mask.tif'
        array_NDVI_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)

        array[array_NDVI_mask < 0.1] = np.nan


        array[array >= 0.65] = np.nan


        outf=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\dryland_mask.tif'
        DIC_and_TIF().arr_to_tif(array, outf)

    def test_histogram(self):

        fpath = rf'E:\Project3\Data\ERA5_daily\dict\moving_window_average_anaysis\\detrended_annual_LAI4g_CV.npy'
        spatial_dic=np.load(fpath, allow_pickle=True, encoding='latin1').item()
        data_list=[]

        for pix in spatial_dic:
            val=spatial_dic[pix]
            data_list.append(val)

        data_list=np.array(data_list)
        ## histogram

        plt.hist(data_list,bins=50)

        plt.show()




    def resampleSOC(self):
        f=rf'E:\Project3\Data\Base_data\\continent.tif'

        outf = rf'E:\Project3\Data\Base_data\\continent_05.tif'

        dataset = gdal.Open(f)



        try:
            gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
        # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
        # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
        except Exception as e:
            pass

    def nc_to_tif(self):

        f=rf'E:\Project3\Data\GIMMS3g_plus_NDVI\\'
        outdir=rf'D:\Project3\Data\Base_data\HWSD\nc\\tif\\'
        Tools().mk_dir(outdir,force=True)

        nc = Dataset(fdir + f, 'r')

        print(nc)
        print(nc.variables.keys())
        # t = nc['time']
        # print(t)
        # start_year = int(t.units.split(' ')[-1].split('-')[0])

        # basetime = datetime.datetime(start_year, 1, 1)  # 告诉起始时间
        lat_list = nc['lat']
        lon_list = nc['lon']
        # lat_list=lat_list[::-1]  #取反
        print(lat_list[:])
        print(lon_list[:])

        origin_x = lon_list[0]  # 要为负数-180
        origin_y = lat_list[0]  # 要为正数90
        pix_width = lon_list[1] - lon_list[0]  # 经度0.5
        pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
        print(origin_x)
        print(origin_y)
        print(pix_width)
        print(pix_height)
        # SIF_arr_list = nc['SIF']
        SPEI_arr_list = nc['tmin']
        print(SPEI_arr_list.shape)
        print(SPEI_arr_list[0])
        # plt.imshow(SPEI_arr_list[5])
        # # plt.imshow(SPEI_arr_list[::])
        # plt.show()

        year = int(f.split('.')[-3][0:4])
        month = int(f.split('.')[-3][4:6])

        fname = '{}{:02d}.tif'.format(year, month)
        print(fname)
        newRasterfn = outdir + fname
        print(newRasterfn)
        longitude_start = origin_x
        latitude_start = origin_y
        pixelWidth = pix_width
        pixelHeight = pix_height
        # array = val
        # array=SPEI_arr_list[::-1]
        array = SPEI_arr_list

        array = np.array(array)
        # method 2
        array = array.T
        array = array * 0.001  ### GPP need scale factor
        array[array < 0] = np.nan

        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()
        ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, )

    def reclassification_koppen(self):
        f=rf'E:\Project3\Data\Base_data\Koeppen_tif\\Koeppen.tif_unify.tif'
        dic={0: 'Tropical', 1: 'Tropical', 2: 'Tropical', 3: 'Tropical', 4: 'Dry', 5: 'Dry', 6: 'Dry', 7: 'Dry', 8: 'Temperate', 9: 'Temperate', 10: 'Temperate',
         11: 'Temperate', 12: 'Temperate', 13: 'Temperate', 14: 'Temperate', 15: 'Temperate', 16: 'Temperate', 17: 'Continental', 18: 'Continental', 19: 'Continental', 20: 'Continental',
         21: 'Continental', 22: 'Continental', 23: 'Continental', 24: 'Continental', 25: 'Continental', 26: 'Continental', 27: 'Continental', 28: 'Polar', 29: 'Polar'}

        reclassification_dic = {
            'Tropical': 1,
            'Dry': 2,
            'Temperate': 3,
            'Continental': 4,
            'Polar': 5

        }
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        spatial_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        for pix in spatial_dic:
            val = spatial_dic[pix]
            if val not in dic:
                spatial_dic[pix] = np.nan
                continue
            if val<-99:
                spatial_dic[pix]=np.nan
                continue

            class_i = dic[val]
            spatial_dic[pix] = reclassification_dic[class_i]


        arr_new = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr_new, interpolation='nearest', cmap='jet')
        plt.colorbar()
        plt.title('Koeppen')
        plt.show()
        outf=rf'E:\Project3\Data\Base_data\Koeppen_tif\\Koeppen_reclassification.tif'
        DIC_and_TIF().arr_to_tif(arr_new, outf)

        pass

    def nc_to_tif_time_series_fast(self):

        fdir=rf'D:\Project3\Data\SM_T\unzip\\'
        outdir=rf'D:\Project3\Data\SM_T\\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):

            outdir_name = f.split('.')[0]
            # print(outdir_name)

            yearlist = list(range(1982, 2021))
            fpath = join(fdir,f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)
            time_bnds = nc_in['time_bnds']
            for t in range(len(time_bnds)):
                date = time_bnds[t]['time']
                date = pd.to_datetime(date.values)
                date_str = date.strftime('%Y%m%d')
                date_str = date_str.split()[0]
                outf = join(outdir, f'{date_str}.tif')
                array = nc_in['LAI'][t]*0.01
                array = np.array(array)
                array[array < 0] = np.nan
                longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.0833333333333, -0.0833333333333
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()




    def nc_to_tif_time_series_fast2(self):

        fdir=rf'D:\Project3\Data\GLEAM\\'
        outdir=rf'D:\Project3\Data\GLEAM\\TIFF_AE\\'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):
            if not 'E_1980-2022_GLEAM_v3.8a_MO'in f:
                continue



            outdir_name = f.split('.')[0]
            # print(outdir_name)

            yearlist = list(range(1982, 2021))
            fpath = join(fdir,f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)
            time_bnds = nc_in['time']
            for t in range(len(time_bnds)):
                date = time_bnds[t]['time']
                date = pd.to_datetime(date.values)
                date_str = date.strftime('%Y%m%d')
                date_str = date_str.split()[0]
                outf = join(outdir, f'{date_str}.tif')
                array = nc_in['E'][t]
                array = np.array(array)
                array[array < 0] = np.nan
                longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.25, -0.25
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, array, ndv=-999999)
                exit()


            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir+f, var_name='ndvi', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue




    def aggregation_soil(self):
        ## aggregation soil sand, CEC etc

        fdir = data_root+rf'Base_data\SoilGrid\SOIL_Grid_05_unify\\'

        product_list=['soc', 'sand', 'nitrogen', 'cec']

        for product_i in product_list:
            ##cec_15-30cm_mean_5000_05
            f_layer1 = fdir + rf'{product_i}_0-5cm_mean_5000_05.tif'
            f_layer2 = fdir + rf'{product_i}_5-15cm_mean_5000_05.tif'
            f_layer3 = fdir + rf'{product_i}_15-30cm_mean_5000_05.tif'


            array1, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer1)
            array2, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer2)
            array3, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer3)

            array1 = np.array(array1, dtype=float)*1/6
            array2 = np.array(array2, dtype=float)*1/3
            array3 = np.array(array3, dtype=float)*1/2

            aggregation_array =array1+array2+array3
            aggregation_array[aggregation_array < -99] = np.nan
            ToRaster().array2raster(fdir + rf'{product_i}.tif', originX, originY, pixelWidth, pixelHeight, aggregation_array, )



        pass
    def resample(self):
        fdir=rf'D:\Project3\Data\VCF\\'
        outdir=rf'D:\Project3\Data\VCF\\resample\\'
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath=fdir+f
            outf=outdir+f
            dataset = gdal.Open(fpath)

            try:
                gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass
    def scale(self):

        fdir = rf'D:\Project3\Data\GLOBMAP\resample\\'
        outdir = rf'D:\Project3\Data\GLOBMAP\scale\\'
        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)
            # array[array == 65535] = np.nan
            # array[array == 249] = np.nan
            array = array * 0.01
            array[array > 10] = np.nan
            array[array < 0] = np.nan
            # array=array/10000



            outf = outdir + f
            ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, array)

    def extract_dryland_tiff(self):
        self.datadir=rf'D:\Project3\Data\\'
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        fdir_all =data_root+ rf'\LAI4g\\'
        outdir = join(fdir_all, 'dryland_tiff_monthly\\')
        T.mk_dir(outdir, force=True)

        for fdir in T.listdir(fdir_all):
            print(fdir)
            if not 'monthly_compose' in fdir:
                continue
            for f in T.listdir(join(fdir_all, fdir)):

                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir_all, fdir, f)
                arr, originX, originY, pixelWidth, pixelHeight =self.raster2array(fpath)
                arr = arr.astype(float)
                arr[arr==0]=np.nan
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                fname=f.split('.')[0]
                # print(fname);exit()
                # outpath = join(outdir_i, fi)
                outpath = join(outdir, fname + '.tif')

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)

            pass

    def raster2array(self, rasterfn):
        '''
        create array from raster
        Agrs:
            rasterfn: tiff file path
        Returns:
            array: tiff data, an 2D array
        '''
        raster = gdal.Open(rasterfn)
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        array = np.asarray(array)
        del raster
        return array, originX, originY, pixelWidth, pixelHeight
    def aggregate(self):
        ##every four days to biweekly
        fdir=rf'D:\Project3\Data\GEODES_AVHRR_LAI\tif_average\\'
        outdir=rf'D:\Project3\Data\GEODES_AVHRR_LAI\aggregate\\'
        month_list=['01','02','03','04','05','06','07','08','09','10','11','12']
        yearlist=list(range(1982,2021))
        T.mk_dir(outdir, force=True)

        for year in yearlist:
            for month in month_list:
                data_aggregate_list=[]

                for f in T.listdir(fdir):
                    if not f.endswith('.tif'):
                        continue
                    if  int(f.split('.')[0][0:4])!=year:
                        continue
                    if  (f.split('.')[0][4:6])!=month:
                        continue



                    array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
                    array = np.array(array, dtype=float)
                    array[array < -99] = np.nan
                    data_aggregate_list.append(array)
                average_array = np.nanmax(data_aggregate_list, axis=0)
                outf=join(outdir,f'{year}{month}.tif')
                ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, average_array)




        pass

    def aggregate_GlobMAP(self):
        ##every four days to biweekly



        fdir=rf'D:\Project3\Data\GLOBMAP\scale\\'
        outdir=rf'D:\Project3\Data\GLOBMAP\aggregate\\'
        month_list=['01','02','03','04','05','06','07','08','09','10','11','12']
        yearlist=list(range(1982,2021))
        T.mk_dir(outdir, force=True)



        for year in yearlist:
            for month in month_list:
                data_aggregate_list=[]
                for f in tqdm(T.listdir(fdir)):
                    if not f.endswith('.tif'):
                        continue
                    date = f.split('.')[1][1:8]

                    datatime = self.parse_doy_date(int(date))
                    year_f = datatime.year
                    month_f = datatime.month
                    month_f = f'{month_f:02d}'
                    if year_f==year and month_f==month:


                        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
                        array = np.array(array, dtype=float)
                        array[array < -99] = np.nan
                        data_aggregate_list.append(array)
                average_array = np.nanmax(data_aggregate_list, axis=0)
                outf=join(outdir,f'{year}{month}.tif')
                ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, average_array)

    def parse_doy_date(self,date_int):
        from datetime import datetime, timedelta
        """Convert date in the format YYYYDDD to datetime"""
        year = int(str(date_int)[:4])
        doy = int(str(date_int)[4:])


        return datetime(year, 1, 1) + timedelta(days=doy - 1)




        pass


    def unify_TIFF(self):
        fdir_all=rf'D:\Project3\Data\GLOBMAP\aggregate\\'
        outdir=rf'D:\Project3\Data\GLOBMAP\unify\\'
        Tools().mk_dir(outdir, force=True)


        for f in os.listdir(join(fdir_all)):
            fpath=join(fdir_all,f)
            outpath=join(outdir,f)

            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            unify_tiff=DIC_and_TIF().unify_raster(fpath,outpath,0.5)



    def tif_to_dic(self):

        fdir_all = rf'D:\Project3\Data\CRU_monthly\Precip\dryland_tiff\\'
        outdir=rf'D:\Project3\Data\CRU_monthly\Precip\dic\\'
        T.mk_dir(outdir, force=True)

        year_list = list(range(1982, 2021))
        # 作为筛选条件

        all_array = []  #### so important  it should be go with T.mk_dic

        for f in os.listdir(fdir_all):
            if not f.endswith('.tif'):
                continue
            if int(f.split('.')[0][0:4]) not in year_list:
                continue

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_all, f))
            array = np.array(array, dtype=float)


            # array_unify = array[:720][:720,
            #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
            array_unify = array[:360][:360,
                          :720]

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
                np.save(outdir + '\\per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + '\\per_pix_dic_%03d' % 0, temp_dic)
    def mean(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f_mean=data_root+rf'\VCF\dryland_tiff\dic_interpolation\Tree cover.npy'

        dic_mean=T.load_npy(f_mean)
        # arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_mean)
        # dic_trend=DIC_and_TIF().spatial_arr_to_dic(arr)
        result_dic={}
        for pix in dic_mean:
            r, c = pix
            if r < 60:
                continue
            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            if dic_modis_mask[pix] == 12:
                continue

            val=dic_mean[pix]
            val_mean=np.nanmean(val)


            if np.isnan(val_mean) or np.isnan(val_mean):
                continue
            if val_mean==0:
                continue
            # relative_trend=val_trend/val_mean *100

            result_dic[pix]=val_mean
        arr=DIC_and_TIF().pix_dic_to_spatial_arr(result_dic)

        outf=data_root+rf'\VCF\dryland_tiff\dic_interpolation\\mean\\Tree cover_mean.npy'

        np.save(
            outf,
            result_dic)

        DIC_and_TIF().arr_to_tif(arr,outf.replace('.npy','.tif'))

    def mean_rainfall(self):
        fdir =result_root+rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not f.split('.')[0] in ['SM', 'sum_rainfall',
                                       'rainfall_intensity', 'CV_intraannual_rainfall',
                'rainfall_seasonality_all_year','']:
                continue
            f_mean=fdir+f

            dic_mean = T.load_npy(f_mean)

            result_dic = {}
            for pix in dic_mean:

                val = dic_mean[pix]['growing_season']
                val_mean = np.nanmean(val)

                result_dic[pix] = val_mean




            outdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\\mean_growing_season\\'
            T.mk_dir(outdir, force=True)

            np.save(
                outdir + f.split('.')[0] + '_mean.npy',
                result_dic)

    pass

    def anomaly(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\'
        outdir = result_root + rf'3mm\Multiregression\\\\anomaly\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'sum_rainfall' in f:
                continue



            outf = outdir + f.split('.')[0]+'_anomaly.npy'
            # if isfile(outf):
            #     continue
            # print(outf)

            dic = T.load_npy(fdir + f)

            zscore_dic = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue


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


                # plt.plot(time_series)
                # plt.legend(['raw'])
                # # plt.show()
                #
                # plt.plot(zscore)
                # plt.legend(['zscore'])
                # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)


    def zscore(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        model_list= [ 'CABLE-POP_S2_lai','CLASSIC_S2_lai',
                           'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                           'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                           'ORCHIDEE_S2_lai',

                           'YIBs_S2_Monthly_lai',

                     'TRENDY_ensemble_composite_time_series',
                    ]
        # model_list=['SNU_LAI','LAI4g','GLOBMAP_LAI']
        #
        # outdir = result_root + rf'\3mm\Multiregression\partial_correlation\Obs\obs_climate\input\Y\\'
        # Tools().mk_dir(outdir, force=True)
        for model in model_list:
            fdir = result_root + rf'\3mm\moving_window_multi_regression\TRENDY\multiresult_relative_change_detrend_growing_season\{model}\npy_time_series\\'
            outdir=result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\Input\X_new\\zscore\\'
            T.mk_dir(outdir, force=True)
            Tools().mk_dir(outdir, force=True)
            for f in os.listdir(fdir):

                dic = T.load_npy(fdir + f)
                outf= outdir + f.split('.')[0]+'_zscore.npy'

                zscore_dic = {}

                for pix in tqdm(dic):

                    if pix not in dic_dryland_mask:
                        continue


                    time_series = dic[pix]
                    # print(time_series)

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
                    zscore = (time_series - mean) / np.nanstd(time_series)




                    zscore_dic[pix] = zscore


                    # plt.plot(time_series)
                    # # plt.legend(['raw'])
                    # # plt.show()
                    #
                    #
                    # plt.plot(zscore)
                    # plt.legend(['zscore'])
                    # # plt.legend(['raw','zscore'])
                    # plt.show()

                    ## save
                np.save(outf, zscore_dic)

    def composite_LAI(self):

        infdir=result_root + rf'\3mm\moving_window_multi_regression\zscore\\'
        f_1=infdir+rf'LAI4g_zscore.npy'
        f_2=infdir+rf'GLOBMAP_LAI_zscore.npy'
        f_3=infdir+rf'SNU_LAI_zscore.npy'
        dic1=np.load(f_1,allow_pickle=True).item()
        dic2=np.load(f_2,allow_pickle=True).item()
        dic3=np.load(f_3,allow_pickle=True).item()
        average_dic= {}

        for pix in tqdm(dic1):
            if not pix in dic2:
                continue
            if not pix in dic3:
                continue
            value1=dic1[pix]
            value2=dic2[pix]
            value3=dic3[pix]


            value1=np.array(value1)
            value2=np.array(value2)
            value3=np.array(value3)
            if len(value1)!=38 or len(value2)!=38 or len(value3)!=38:
                print(pix,len(value1),len(value2),len(value3))
                continue
            print(value1,value2,value3)


            average_val=np.nanmedian([value1,value2,value3],axis=0)

            # print(average_val)
            if np.nanmean(average_val) >999:
                continue
            if np.nanmean(average_val) <-999:
                continue
            average_dic[pix]=average_val
            #
            # plt.plot(value1,color='blue')
            # plt.plot(value2,color='green')
            # plt.plot(value3,color='orange')
            # plt.plot(average_val,color='red')
            # plt.legend(['GlOBMAP','SNU','LAI4g','average'])
            # plt.show()

        outdir=result_root + rf'3mm\moving_window_multi_regression\zscore\\'
        Tools().mk_dir(outdir,force=True)

        np.save(outdir+'composite_LAI_zscore_median.npy',average_dic)

        pass
    def interpolate_VCF(self):
        fdir=rf'D:\Project3\Data\VCF\dryland_tiff\dic\Non vegetatated\\'
        outfir=rf'D:\Project3\Data\VCF\dryland_tiff\dic_interpolation\\'
        T.mk_dir(outfir,True)
        dic=T.load_npy_dir(fdir)
        year_list=list(range(1982,2017))
        year_list_actual=copy.copy(year_list)
        year_list_actual.remove(1994)
        year_list_actual.remove(2000)
        year_list_actual_reshape=np.reshape(year_list_actual,(-1,1))
        result_dic={}
        # print(year_list_actual);exit()
        for pix in dic:
            val=dic[pix]
            if T.is_all_nan(val) or np.isnan(val).sum() > 0:
                continue

            ## if any nan in val continue
            new_val=T.interp_nan(val)
            if len(new_val) != len(year_list_actual):
                print("长度不一致！", len(new_val), len(year_list_actual))
                continue
            val_reshape=np.reshape(new_val,(-1,1))
            # print(val)
            print(len(year_list_actual), len(val))  # 二者应相等

            model=LinearRegression()
            model.fit(year_list_actual_reshape,val_reshape)

            val_1994=model.predict([[1994]])[0]
            val_2000=model.predict([[2000]])[0]
            new_value=np.insert(val,12,val_1994)
            new_value=np.insert(new_value,18,val_2000)
            new_year_list=np.insert(year_list_actual,12,1994)
            new_year_list=np.insert(new_year_list,18,2000)

            #
            result_dic[pix]=new_value
        outf=outfir+f'''Non vegetatated.npy'''
        T.save_npy(result_dic,outf)
            # print(len(new_value))






        pass


    def interpolation(self):

        fdir = rf'E:\Project3\Data\Landsat\dic\\'
        dic = T.load_npy_dir(fdir)

        dict_clean = {}

        for pix in dic:
            r, c = pix
            vals = dic[pix]
            vals = np.array(vals)
            # vals_clip = vals[36:]
            vals_clip = vals
            vals_no_nan = vals_clip[~np.isnan(vals_clip)]
            ratio = len(vals_no_nan) / len(vals_clip)
            if ratio < 0.5:
                continue
            dict_clean[pix] = vals_clip

        mon_mean_dict = {}
        for pix in tqdm(dict_clean,desc='calculating long term mean'):
            vals = dict_clean[pix]
            vals_reshape = vals.reshape(-1, 12)
            vals_reshape_T = vals_reshape.T
            mon_mean_list = []
            for mon in vals_reshape_T:
                mon_mean = np.nanmean(mon)
                mon_mean_list.append(mon_mean)
            mon_mean_dict[pix] = mon_mean_list

        spatial_dic={}

        for pix in dict_clean:
            vals = dict_clean[pix]
            vals_reshape = vals.reshape(-1, 12)
            vals_reshape_T = vals_reshape.T
            # print(len(vals_reshape));exit()
            mon_vals_interpolated_T = []
            for i in range(len(vals_reshape_T)):
                mon_mean = mon_mean_dict[pix][i]
                mon_vals = vals_reshape_T[i]
                mon_vals = np.array(mon_vals)
                mon_vals[np.isnan(mon_vals)] = mon_mean
                mon_vals_interpolated_T.append(mon_vals)
            mon_vals_interpolated_T = np.array(mon_vals_interpolated_T)
            mon_vals_interpolated = mon_vals_interpolated_T.T
            mon_vals_interpolated_flatten = mon_vals_interpolated.flatten()
            vals_origin = dic[pix]
            spatial_dic[pix] = mon_vals_interpolated_flatten
            # plt.imshow(mon_vals_interpolated)
            # plt.figure(figsize=(15, 6))
            # plt.plot(mon_vals_interpolated_flatten)
            # plt.scatter(np.arange(0, len(vals_origin)), vals_origin, c='r')
            # plt.scatter(np.arange(0, len(mon_vals_interpolated_flatten)), mon_vals_interpolated_flatten, c='g',zorder=-1)
            # plt.show()
        np.save(fdir+'interpolated', spatial_dic)

    def nc_to_tif_template(self, fname, var_name, outdir, yearlist):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())
            # exit()
            time = ncin.variables['time'][:]

        except:
            raise UserWarning('File not supported: ' + fname)
        # lon,lat = np.nan,np.nan
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        print(basetime_unit)
        print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            try:
                                basetime_ = basetime.split('T')[0]
                                # print(basetime_)
                                basetime = datetime.datetime.strptime(basetime_, '%Y-%m-%d')
                                # print(basetime)
                            except:

                                raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            if year not in yearlist:
                continue
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = []
            lat_list = []
            value_list = []
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    lon_i = xx[i][j]
                    if lon_i > 180:
                        lon_i -= 360
                    lat_i = yy[i][j]
                    value_i = arr[i][j]
                    lon_list.append(lon_i)
                    lat_list.append(lat_i)
                    value_list.append(value_i)
            DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list, outpath)


pass




class Phenology():  ### plot site based phenology curve
    ## this function is to see phenology of NH, SH and tropical
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):

        self.phenology()
        pass

    def read_shp(self):
        fpath = r"C:\Users\wenzhang1\Desktop\point2.shp"
        df=T.read_point_shp(fpath, )
        return df

    def phenology(self):
        fdir_all = rf'E:\Project3\Data\LAI4g\\dic\\'
        spatial_dic=T.load_npy_dir(fdir_all)
        result_dic={}
        shp_df=self.read_shp()
        print(shp_df)
        lon_list=shp_df['point_x_pos'].to_list()
        lat_list=shp_df['point_y_pos'].to_list()
        pix_list=DIC_and_TIF().lon_lat_to_pix(lon_list, lat_list)


        for pix in pix_list:
            lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
            r,c=pix
            # if r<60:
            #     continu
            val=spatial_dic[pix]
            if T.is_all_nan(val):
                continue

            vals_reshape=val.reshape(-1,24)
            # plt.imshow(vals_reshape, interpolation='nearest', cmap='jet')
            # plt.colorbar()
            # plt.show()
            multiyear_mean = np.nanmean(vals_reshape, axis=0)
            #
            result_dic[pix]=multiyear_mean
            x=np.arange(0,24)
            xtick = [str(i) for i in x]
            plt.plot(x, multiyear_mean)
            plt.xticks(x, xtick)
            plt.xlabel('biweekly')
            plt.ylabel('LAI4g (m2/m2)')
            plt.title(f'lat:{lat},lon:{lon}')
            plt.show()
        # array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(result_dic)
        # plt.imshow(array, interpolation='nearest', cmap='jet')
        # plt.colorbar()
        #
        # plt.show()
            ## plt.plot(multiyear_mean)
            ## plt.show()





        pass

class build_moving_window_dataframe():
    def __init__(self):
        self.threshold = '1mm'
        self.this_class_arr = (
                    result_root +  rf'\bivariate\Dataframe\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'Dataframe.df'
    def run(self):
        df = self.__gen_df_init(self.dff)
        # df=self.build_df(df)
        # self.append_value(df)
        # df=self.append_attributes(df)
        # df=self.add_trend_to_df(df)
        # df=self.foo1(df)
        df=self.add_window_to_df(df)
        # df=self.add_interaction_to_df(df)
        # self.rescale_to_df(df)
        # self.add_fire(df)
        # self.add_sum_rainfall_mean(df)
        # self.add_short_vegetation_mean(df)
        # df=self.add_products_consistency_to_df(df)
        # df=self.rename_columns(df)
        # df=self.add_columns(df)
        # df=self.drop_field_df(df)
        self.show_field()

        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff)
    def show_field(self):
        df = T.load_df(self.dff)
        for col in df.columns:
            print(col)



    def __gen_df_init(self, file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df = self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self, file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass

    def build_df(self, df):

        fdir = result_root+rf'3mm\Multiregression\Multiregression_result_residual\OBS_zscore\Y\\'
        all_dic = {}

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'zscore' in f:
                continue

            fname = f.split('.')[0]

            fpath = fdir + f

            dic = T.load_npy(fpath)
            key_name = fname

            all_dic[key_name] = dic
        # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df

    def append_value(self, df):  ##补齐

        ## extract LAI4g

        for col in df.columns:
            if not 'LAI4g' in col:
                continue
            if 'CV' in col:
                continue

            vals_new = []


            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'append {col}'):
                pix = row['pix']
                r, c = pix
                if r<480:
                    continue
                vals = row[col]
                print(vals)
                if type(vals) == float:
                    vals_new.append(np.nan)
                    continue
                vals = np.array(vals)
                print(len(vals))
                # if len(vals)==23:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     # print(len(vals))
                # elif len(vals)==38:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     print(len(vals))
                if len(vals) == 23:

                    vals = np.append(vals, np.nan)
                    vals_new.append(vals)

                vals_new.append(vals)

                # exit()
            df[col] = vals_new

        return df

        pass

    def foo1(self, df):

        f = rf'D:\Project3\Result\Nov\Composite_LAI\CV\\\composite_LAI_mean_detrend_CV.npy'
        # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        # array = np.array(array, dtype=float)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)

        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]
            y = 0

            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                window=y
                # print(window)
                year.append(window)
                y += 1

        df['pix'] = pix_list



        df['window'] = year

        df['composite_LAI_mean_detrend_CV'] = change_rate_list
        return df
    def add_window_to_df(self, df):


        fdir=result_root+rf'\Composite_LAI\LAImin_LAImax\\'


        for f in os.listdir(fdir):
            if 'max' in f:
                continue
            if 'min' in f:
                continue



            variable= f.split('.')[0]

            print(variable)


            if not f.endswith('.npy'):
                continue

            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                window = row.window
                # pix = row.pix
                pix = row['pix']
                r, c = pix


                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue

                y = window

                vals = val_dic[pix]
                vals=np.array(vals)
                print(len(vals))
                # exit()
                # plt.plot(vals)
                # plt.show()

                # print(vals)
                # vals[vals>9999] = np.nan
                # vals[vals<-9999] = np.nan

                ##### if len vals is 38, the end of list add np.nan

                #
                if len(vals) == 24:
                    ## add twice nan at the end
                    # vals=np.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], vals,)
                    vals=np.append(vals,[np.nan])



                # if len(vals) !=24:
                #
                #     NDVI_list.append(np.nan)
                #     continue


                if len(vals) ==0:
                    NDVI_list.append(np.nan)
                    continue

                v1= vals[y-0]
                NDVI_list.append(v1)



            df[f'{variable}'] = NDVI_list
            # df[f'{variable}_growing_season'] = NDVI_list
        # exit()
        return df
    def rescale_to_df(self,df):
        # T.print_head_n(df);exit()



        df['sand']=df['sand']*100
        df['soc'] = df['soc'] *100
        # df['sand_rainfall_frenquency'] = df['sand'] * df['rainfall_frenquency_average_zscore']
        # df['sand_rainfall_intensity'] = df['sand'] * df['rainfall_intensity_average_zscore']

        return df

    # def add_interaction_to_df(self,df):
    #     # T.print_head_n(df);exit()
    #
    #     # df['CO2_rainfall']=df['CO2']*df['rainfall_intensity_average_zscore']
    #     df['FVC_relative_change'] = 'unknown'
    #     df.loc[df['sum_rainfall_trend'] > 0, 'wet_dry'] = 'wetting'
    #     df.loc[df['sum_rainfall_trend'] < 0, 'wet_dry'] = 'drying'
    #     return df

    def add_interaction_to_df(self,df):
        # T.print_head_n(df);exit()

        df['beta_CVrainfall_interaction']=df['composite_LAI_beta']*df['detrended_sum_rainfall_CV_zscore']

        return df



    pass

    def add_products_consistency_to_df(self,df):
        fdir=rf'E:\Project3\Result\3mm\bivariate_analysis\CV_products_comparison_bivariate\\'
        variable_list=['LAI4g_GIMMS_NDVI',
                       'LAI4g_NDVI',
                       'LAI4g_NDVI4g',


        ]
        for variable in variable_list:
            fpath=fdir+variable_list[0]+'.tif'

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array = np.array(array, dtype=float)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)



            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                if val > 99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)


            df[f'{variable}'] = val_list
        return df


        pass
    def append_attributes(self, df):  ## add attributes
        fdir =  result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\Y\\'
        var_list=['CV_intraannual_rainfall_ecosystem_year_zscore','CV_intraannual_rainfall_growing_season_zscore',
'detrended_sum_rainfall_ecosystem_year_CV_zscore','detrended_sum_rainfall_growing_season_CV_zscore',
                  'rainfall_frenquency_ecosystem_year_zscore','rainfall_frenquency_growing_season_zscore',
                  'LAI4g_sensitivity_zscore','GLOBMAP_LAI_sensitivity_zscore','composite_LAI_sensitivity_zscore',
                  'SNU_LAI_sensitivity_zscore'
                 ]
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue
            if not 'median' in f:
                continue



            # array=np.load(fdir+f)
            # dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic=T.load_npy(fdir+f)
            key_name = f.split('.')[0]
            # if not key_name in var_list:
            #     continue
            print(key_name)

            # df[key_name] = df['pix'].map(dic)
            # T.print_head_n(df)
            df=T.add_spatial_dic_to_df(df,dic,key_name)
        return df
    def add_fire(self,df):
        fdir=  data_root

        val_dic = T.load_npy_dir(fdir)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            # window = row.window
            # pix = row.pix
            pix = row['pix']
            r, c = pix

            if not pix in val_dic:
                val_list.append(np.nan)
                continue


            vals = val_dic[pix]['ecosystem_year']
            vals = np.array(vals)
            ## 10^6
            mean_burn_area=np.nansum(vals)/1000000
            if mean_burn_area < -99:
                val_list.append(np.nan)
                continue

            val_list.append(mean_burn_area)


        df['Burn_area_sum']=val_list



        pass

    def add_sum_rainfall_mean(self,df):
        fpath=  result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\sum_rainfall.npy'

        val_dic = T.load_npy(fpath)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            # window = row.window
            # pix = row.pix
            pix = row['pix']
            r, c = pix

            if not pix in val_dic:
                val_list.append(np.nan)
                continue


            vals = val_dic[pix]
            vals = np.array(vals)
            ## 10^6
            mean_burn_area=np.nanmean(vals)
            if mean_burn_area < -99:
                val_list.append(np.nan)
                continue

            val_list.append(mean_burn_area)


        df['sum_rainfall_mean']=val_list

    def add_short_vegetation_mean(self,df):
        fpath=  result_root+rf'3mm\VCF\moving_window_extraction\Non tree vegetation_average.npy'

        val_dic = T.load_npy(fpath)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            # window = row.window
            # pix = row.pix
            pix = row['pix']
            r, c = pix

            if not pix in val_dic:
                val_list.append(np.nan)
                continue


            vals = val_dic[pix]
            vals = np.array(vals)

            mean_burn_area=np.nanmean(vals)
            if mean_burn_area < -99:
                val_list.append(np.nan)
                continue

            val_list.append(mean_burn_area)


        df['Non tree vegetation_mean']=val_list
    def add_columns(self, df):
        df['window'] = df['window'].str.extract(r'(\d+)').astype(int)




        return df


    def rename_columns(self, df):
        df = df.rename(columns={'Non tree vegetation_average_zscore': 'Non_tree_vegetation_average_zscore',
                                'Tree cover_average_zscore': 'Tree_cover_average_zscore',
                                'Non vegetatated_average_zscore': 'Non_vegetatated_average_zscore',

                                }

                       )

        return df

    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[


                              'TRENDY_ensemble_sensitivity_zscore_mean',
            'TRENDY_ensemble_sensitivity_zscore_median',





                              ])
        return df








    def add_trend_to_df(self, df):
        fdir=result_root+rf'\bivariate\rainfallmin_rainfallmax\trend\\'
        for f in os.listdir(fdir):

            if not f.endswith('.tif'):
                continue
            print(f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            f_name = f.split('.')[0]
            print(f_name)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]

                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list
        return df

        pass

class build_dataframe():


    def __init__(self):

        self.this_class_arr = (
                result_root +  rf'\partial_correlation\Dataframe\3mm\\')
        # self.this_class_arr = (result_root+rf'\Multiregression_contribution\Obs\Dataframe\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'Obs.df'
        # self.this_class_arr = (result_root+rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\Dataframe\\')


        pass

    def run(self):


        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        df=self.foo2(df)
        # df=self.add_multiregression_to_df(df)
        # df=self.build_df(df)
        # df=self.build_df_monthly(df)
        # df=self.append_attributes(df)  ## 加属性
        # df=self.append_cluster(df)  ## 加属性
        # df=self.append_value(df)   ## insert or append value


        # df = self.add_detrend_zscore_to_df(df)
        # df=self.add_GPCP_lagged(df)
        # df=self.add_rainfall_characteristic_to_df(df)
        # df=self.add_lc_composition_to_df(df)
        # df=self.add_new_field_to_df(df)


        df=self.add_trend_to_df_trendy(df)  ### add different scenarios of mild, moderate, extreme
        # df=self.add_trend_to_df(df)
        # df=self.add_seasonality_to_df(df)
        # df=self.add_fire(df)

        # df=self.add_soil_to_df(df)
        # df=self.add_mean_to_df(df)
        # # # # df=self.add_interaction_to_df(df)

        # # #
        # df=self.add_aridity_to_df(df)
        # df=self.add_dryland_nondryland_to_df(df)
        # df=self.add_MODIS_LUCC_to_df(df)
        # df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        # df=self.add_landcover_classfication_to_df(df)
        # # # # # # # # # # df=self.dummies(df)
        # df=self.add_maxmium_LC_change(df)
        # df=self.add_row(df)
        # # # # # # # # # # # # # #
        # df=self.add_lat_lon_to_df(df)
        # df=self.add_continent_to_df(df)
        # df=self.add_residual_to_df(df)

        # # # #
        # df=self.add_rooting_depth_to_df(df)
        # #
        # df=self.add_area_to_df(df)


        # df=self.rename_columns(df)
        # df = self.drop_field_df(df)
        # df=self.remove_duplicate_columns(df)
        df=self.show_field(df)


        T.save_df(df, self.dff)

        self.__df_to_excel(df, self.dff)

    def __gen_df_init(self, file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df = self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self, file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass
    def build_df(self, df):

        fdir=rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\\'
        all_dic= {}
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            fname= f.split('.')[0]

            fpath=fdir+f

            dic = T.load_npy(fpath)
            key_name=fname
            print(key_name)
            all_dic[key_name]=dic
        # print(all_dic.keys())
        df=T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df

    def build_df_monthly(self, df):


        fdir = result_root+rf'extract_GS_return_monthly_data\individual_month_relative_change\X\\'
        all_dic= {}

        for fdir_ii in os.listdir(fdir):

            dic=T.load_npy(fdir+fdir_ii)

            key_name=fdir_ii.split('.')[0]
            all_dic[key_name] = dic
                # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df



    def append_attributes(self, df):  ## add attributes
        fdir = result_root+ rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue
            if not 'rainfall_intensity' in f:
                continue

            # array=np.load(fdir+f)
            # dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic=T.load_npy(fdir+f)
            key_name = f.split('.')[0]
            print(key_name)

            # df[key_name] = df['pix'].map(dic)
            # T.print_head_n(df)
            df=T.add_spatial_dic_to_df(df,dic,key_name)
        return df


    def append_cluster(self, df):  ## add attributes
        dic_label = {'sig_greening_sig_wetting': 1, 'sig_browning_sig_wetting': 2, 'non_sig_greening_sig_wetting': 3,

                     'non_sig_browning_sig_wetting': 4, 'sig_greening_sig_drying': 5, 'sig_browning_sig_drying': 6,

                     'non_sig_greening_sig_drying': 7, 'non_sig_browning_sig_drying': 8, np.nan: 0}

        #### reverse
        dic_label = {v: k for k, v in dic_label.items()}


        fdir = result_root+rf'Dataframe\anomaly_trends\\'
        for f in os.listdir(fdir):
            if not f.endswith('tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)

            # array=np.load(fdir+f)
            dic = DIC_and_TIF().spatial_arr_to_dic(array)

            key_name='label'
            for k in dic:
                if dic[k] <-99:
                    continue
                dic[k]=dic_label[dic[k]]

            df=T.add_spatial_dic_to_df(df,dic,key_name)

        return df






    def append_value(self, df):  ##补齐
        fdir = result_root + rf'growth_rate\\\growth_rate_raw\\'
        col_list=[]
        for f in os.listdir(fdir):
            if not 'LAI4g' in f:
                continue

            if not f.endswith('.npy'):
                continue


            col_name=f.split('.')[0]+'_growth_rate_raw'

            col_list.append(col_name)

        for col in col_list:
            vals_new=[]

            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'append {col}'):
                pix = row['pix']
                r, c = pix
                vals=row[col]
                if type(vals)==float:
                    vals_new.append(np.nan)
                    continue
                vals=np.array(vals)
                # if len(vals)==23:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     # print(len(vals))
                # elif len(vals)==38:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     print(len(vals))
                if len(vals)==38:

                    # vals=np.append(vals,np.nan)
                    ## append at the beginning
                    vals = np.insert(vals, 0, np.nan)


                vals_new.append(vals)

                # exit()
            df[col]=vals_new

        return df

        pass


    def foo1(self, df):

        f = rf'D:\Project3\Result\Nov\Composite_LAI\relative_change\\composite_LAI_median_relative_change.npy'
        # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        # array = np.array(array, dtype=float)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)


        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 1982
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y)
                y += 1


        df['pix'] = pix_list

        df['year'] = year
        # df['window'] = 'VPD_LAI4g_00'
        df['rainfall_intensity'] = change_rate_list
        return df

    def foo2(self, df):  # 新建trend

        f = result_root + rf'Multiregression_contribution\Obs\result_new\3mm\composite_LAI_mean\composite_LAI_mean_sensitivity_zscore.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(f)
        # val_array[val_array<-99]=np.nan
        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        # plt.imshow(val_array)
        # plt.colorbar()
        # plt.show()

        # exit()

        pix_list = []
        for pix in tqdm(val_dic):
            val = val_dic[pix]
            if np.isnan(val):
                continue
            pix_list.append(pix)
        df['pix'] = pix_list
        T.print_head_n(df)


        return df

    def add_multiregression_to_df(self, df):
        fdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\trend_ecosystem_year\\'
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'sum_rainfall' in f:
                continue



            print(f.split('.')[0])


            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val_list.append(val)
            df[f.split('.')[0]] = val_list
        return df



        pass

    def add_GPCP_lagged(self,df): ##
        fdir=result_root+rf'\extract_GS\OBS_LAI_extend\\'
        for f in os.listdir(fdir):
            if f.split('.')[0] not in ['GPCC','CRU','tmax']:
                continue

            spatial_dic = T.load_npy(fdir+f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year

                pix = row['pix']
                r, c = pix

                if not pix in spatial_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = spatial_dic[pix][1:]


                v1=vals[year-1983]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)
            col_name=f.split('.')[0]
            print(col_name)
            df[col_name]=NDVI_list

            return df
            pass

    def add_detrend_zscore_to_df(self, df):

        fdir=rf'D:\Project3\Result\Nov\Composite_LAI\relative_change\\'


        for f in os.listdir(fdir):



            variable= f.split('.')[0]


            print(variable)


            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year
                # pix = row.pix
                pix = row['pix']
                r, c = pix

                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = val_dic[pix]
                print(len(vals))

                ##### if len vals is 38, the end of list add np.nan

                # if len(vals) == 19:
                #     ##creast 19 nan
                #     nan_list = np.array([np.nan] * 19)
                #     vals=np.append(nan_list,vals)
                # if len(vals)==33 :
                #     nan_list=np.array([np.nan]*5)
                #     vals=np.append(vals,nan_list)

                if len(vals)==38:
                    vals = np.append(vals,np.nan)


                v1= vals[year - 1982]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}'] = NDVI_list
        # exit()
        return df



    def add_rainfall_characteristic_to_df(self, df):
        fdir = rf'E:\Data\ERA5_precip\ERA5_daily\dict\dry_spell\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            variable= f.split('.')[0]

            print(variable)


            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year
                # pix = row.pix
                pix = row['pix']
                r, c = pix


                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue


                vals = val_dic[pix]

                print(len(vals))
                ##### if len vals is 38, the end of list add np.nan

                # if len(vals) == 38:
                #     vals=np.append(vals,np.nan)
                #     v1 = vals[year - 1982]
                #     NDVI_list.append(v1)
                # if len(vals)==39:
                # v1 = vals[year - 1982]
                # v1 = vals[year - 1982]
                # if year < 2000:  ## fillwith nan
                #     NDVI_list.append(np.nan)
                #     continue


                v1= vals[year - 1982]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}'] = NDVI_list
        # exit()
        return df

    def add_new_field_to_df(self,df):
        # T.print_head_n(df);exit()

        # df['CO2_rainfall']=df['CO2']*df['rainfall_intensity_average_zscore']
        df['wet_dry'] = 'unknown'
        df.loc[df['sum_rainfall_trend'] > 0, 'wet_dry'] = 'wetting'
        df.loc[df['sum_rainfall_trend'] < 0, 'wet_dry'] = 'drying'
        return df



    def add_lc_composition_to_df(self, df):  ##add landcover composition to df


        fdir_all = data_root + rf'landcover_composition_DIC\\'

        all_dic = {}
        for fdir in os.listdir(fdir_all):
            fname=fdir.split('.')[0]

            dic={}
            for f in os.listdir(fdir_all+fdir):


                dicii = T.load_npy(fdir_all+fdir+'\\'+f)
                dic.update(dicii)


            all_dic[fname] = dic
        # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df
        # exit()

    def add_row(self, df):
        r_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def add_max_trend_to_df(self, df):

        fdir = data_root + rf'/Base_data/lc_trend/'
        for f in (os.listdir(fdir)):
            # print()
            if not 'max_trend' in f:
                continue
            if not f.endswith('.npy'):
                continue
            if 'p_value' in f:
                continue

            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val = val * 20
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list

        return df
    def add_continent_to_df(self, df):
        tiff = rf'D:\Project3\Data\Base_data\\continent_05.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'continent'

        dic_convert={1:'Africa',2:'Asia',3:'Australia',4: np.nan, 5:'South_America', 6: np.nan, 7:'Europe',8:'North_America',255: np.nan}

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            # print(val)

            val_convert=dic_convert[val]

            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val_convert)
        df[f_name] = val_list
        return df

    pass
    def add_lat_lon_to_df(self, df):
        D=DIC_and_TIF(pixelsize=0.5)
        df=T.add_lon_lat_to_df(df,D)
        return df


    def add_soil_texture_to_df(self, df):
        fdir = data_root + rf'\Base_data\\SoilGrid\SOIL_Grid_05_unify\\weighted_average\\'
        for f in (os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue

            tiff = fdir + f

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            fname=f.split('.')[0]
            print(fname)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[fname] = val_list
        return df
        pass

    def add_rooting_depth_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth_05.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'rooting_depth'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f_name] = val_list
        return df

        pass
    def add_Ndepostion_to_df(self, df):
        fdir='D:\Project3\Result\extract_GS\OBS_LAI_extend\\noy\\'
        val_dic = T.load_npy_dir(fdir)
        f_name = 'Noy'
        print(f_name)
        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            year = row.year
            # pix = row.pix
            pix = row['pix']
            r, c = pix

            if not pix in val_dic:
                NDVI_list.append(np.nan)
                continue

            vals = val_dic[pix]

            # print(len(vals))
            ##### if len vals is 38, the end of list add np.nan

            # if len(vals) == 38:
            #     vals=np.append(vals,np.nan)
            #     v1 = vals[year - 1982]
            #     NDVI_list.append(v1)
            # if len(vals)==39:
            # v1 = vals[year - 1982]
            # v1 = vals[year - 1982]
            v1 = vals[year - 0]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)

        df[f_name] = NDVI_list
        # exit()

        return df

    def add_area_to_df(self, df):
        area_dic=DIC_and_TIF(pixelsize=0.25).calculate_pixel_area()
        f_name = 'pixel_area'
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in area_dic:
                val_list.append(np.nan)
                continue
            val = area_dic[pix]
            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f_name] = val_list
        return df


    def add_ozone_to_df(self, df):
            f=rf'D:\Project3\Result\extract_GS\OBS_LAI_extend\\ozone.npy'
            val_dic = T.load_npy(f)
            f_name = 'ozone'
            print(f_name)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year_range
                # pix = row.pix
                pix = row['pix']
                r, c = pix

                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = val_dic[pix]

                # print(len(vals))
                ##### if len vals is 38, the end of list add np.nan

                if len(vals) == 37:
                    ## append 2 nan
                    vals=np.append(vals,np.nan)
                    vals=np.append(vals,np.nan)

                    v1 = vals[year - 0]
                    NDVI_list.append(v1)


            df[f_name] = NDVI_list
            # exit()

            return df

            pass
    def add_root_depth_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'rooting_depth'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f_name] = val_list
        return df

        pass
    def add_precipitation_CV_to_df(self, df):
        fdir='D:\Project3\Result\state_variables\\CV_monthly\\'
        for f in os.listdir(fdir):

            val_dic = T.load_npy(fdir + f)
            f_name = f.split('.')[0]+'_CV'
            print(f_name)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix'][0]
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list
        return df

        pass

    def add_events(self, df):
        fdir = result_root + rf'relative_change\events_extraction\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dict_ = T.load_npy(fdir + f)
            key_name = f.split('.')[0]+'_event_level'
            print(key_name)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                year= row['year']
                if not pix in dict_:
                    val_list.append(np.nan)
                    continue
                dic_i = dict_[pix]
                mode=np.nan

                for key in dic_i:
                    idx_list=dic_i[key]
                    for idx in idx_list:
                        yeari=idx+1982
                        if yeari==year:
                            mode=key
                            break
                val_list.append(mode)
            df[key_name] = val_list

        T.print_head_n(df)
        return df


    def add_trend_to_df_trendy(self,df):
        fdir_all = result_root + rf'\partial_correlation\Obs\result\3mm\\'
        for fdir in os.listdir(fdir_all):


            # for f in os.listdir(join(fdir_all,fdir)):
            #
            #
            #     if not f.endswith('.tif'):
            #         continue

            # #
            fdir_sig=fdir_all+fdir+'\\sig\\'
            # print(fdir_sig);exit()

            for f in os.listdir(fdir_sig):


                if not f.endswith('.tif'):
                    continue

                variable=(f.split('.')[0])
                if 'sensitivity' in variable:
                    fname=variable
                else:
                    fname=f'{fdir}_{variable}'
                print(fname)

                # fpath=join(fdir_all,fdir,f) ####### revise!!!!!!!
                fpath=join(fdir_sig,f)
                print(fpath)


                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                array = np.array(array, dtype=float)

                val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

                # val_array = np.load(fdir + f)
                # val_dic=T.load_npy(fdir+f)

                # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)


                val_list=[]
                for i,row in tqdm(df.iterrows(),total=len(df)):
                    pix=row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    val=val_dic[pix]
                    if val<-99:
                        val_list.append(np.nan)
                        continue
                    if val>99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(val)
                df[f'{fname}']=val_list

        return df

    def add_seasonality_to_df(self, df):
        f = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'


        val_dic = T.load_npy(f)



        f_name ='SeasType'
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]['SeasType']
            if val < -9999:
                val_list.append(np.nan)
                continue
            if val > 9999:
                val_list.append(np.nan)
                continue
            val_list.append(val)


        df[f'{f_name}'] = val_list


        return df

    def add_trend_to_df(self, df):
        fdir = result_root + rf'\Multiregression_contribution\Obs\input\Y\zscore\trend\\'

        variables_list = [
                          'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai','YIBs']

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue



            variable = (f.split('.')[0])
            print(variable)




            # if 'sensitivity' in variable:
            #     fname = variable
            # else:
            #     fname = f'composite_{variable}'
            # print(fname)



            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

            # val_array = np.load(fdir + f)
            # val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)

            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -9999:
                    val_list.append(np.nan)
                    continue
                if val > 9999:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)


            df[f'{f_name}'] = val_list


        return df

    def add_fire(self, df):
        fdir = result_root+rf'\Fire\extract_annual_growing_season_LAI_mean\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            fname=f.split('.')[0]

            val_dic = T.load_npy(fdir+f)

            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                # window = row.window
                # pix = row.pix
                pix = row['pix']
                r, c = pix

                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue

                vals = val_dic[pix]['growing_season']
                vals = np.array(vals)
                ## 10^6
                # mean_burn_area = np.nansum(vals) / 1000000 ## no use this
                mean_burn_area = np.nanmean(vals) / 1000000
                if mean_burn_area < -99:
                    val_list.append(np.nan)
                    continue

                val_list.append(mean_burn_area)

            df[f'{fname}'] = val_list
        return df

        pass

    def add_residual_to_df(self, df):


        print(df.columns.tolist())

        # df['residual_contrib'] = df['composite_LAI_detrend_CV_zscore_trend'] - (df['CV_intraannual_rainfall_ecosystem_year_contrib']+
        #                   df['detrended_sum_rainfall_CV_contrib']+df['Fire_sum_average_contrib']+df['composite_LAI_beta_mean_contrib'])

        df['LAI4g_residual_contrib'] = df['LAI4g_detrend_CV_zscore_trend'] - (df['LAI4g_sensitivity_zscore_contrib'] +
                                                                          df['LAI4g_detrended_sum_rainfall_ecosystem_year_CV_zscore_contrib'] + df['LAI4g_CV_intraannual_rainfall_ecosystem_year_zscore_contrib']

                                                                          )



        return df


    def add_soil_to_df(self, df):
        # fdir=data_root+rf'\Base_data\SoilGrid\SOIL_Grid_05_unify\weighted_average\\'


        fdir=data_root+rf'\Base_data\SoilGrid\SOIL_Grid_05_unify\weighted_average\\'

        for f in os.listdir(fdir):
            #
            if not f.endswith('.tif'):
                continue



            variable = (f.split('.')[0])
            print(variable)




            #
            # if variable not in variables_list:
            #     continue

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

            # val_array = np.load(fdir + f)
            # val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)

            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                # if val > 99:
                #     val_list.append(np.nan)
                #     continue
                val_list.append(val)


            df[f'{f_name}'] = val_list


        return df


    def add_mean_to_df(self, df):
        fdir=data_root+rf'\VCF\dryland_tiff\dic_interpolation\mean\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            variable = (f.split('.')[0])



            val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)

            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]

                if val < 0:
                    val_list.append(np.nan)
                    continue
                if val > 9999:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f'{f_name}'] = val_list



        return df

    def add_interaction_to_df(self, df):
        # T.print_head_n(df);exit()

        df['sum_rainfall_CVintra_interation']=df['sum_rainfall_mean']*df['CV_intraannual_rainfall_ecosystem_year']

        return df





    def rename_columns(self, df):
        df = df.rename(columns={'composite_LAI_detrend_CV_median_p_value': 'composite_LAI_median_detrend_CV_p_value',







                               }


                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=['SNU_LAI_detrend_CV_zscore_trend_percent',
                              'SNU_LAI_detrend_CV_zscore_p_value_percent',
                              'LAI4g_detrend_CV_zscore_trend_percent',
                              'LAI4g_detrend_CV_zscore_p_value_percent',
                              'GLOBMAP_LAI_detrend_CV_zscore_trend_percent',
                              'GLOBMAP_LAI_detrend_CV_zscore_p_value_percent',
                              'composite_LAI_median_detrend_CV_zscore_trend_percent',
                              'composite_LAI_median_detrend_CV_zscore_p_value_percent',
                              'composite_LAI_mean_detrend_CV_zscore_trend_percent',
                              'composite_LAI_mean_detrend_CV_zscore_p_value_percent',
















                              ])

        return df



    def add_NDVI_mask(self, df):
        f = data_root + rf'/Base_data/NDVI_mask.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df
    def add_MODIS_LUCC_to_df(self, df):
        f = data_root + rf'\Base_data\MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = f.split('.')[0]
        print(f_name)
        val_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)
        df['MODIS_LUCC'] = val_list
        return df



    def add_landcover_data_to_df(self, df):

        f = data_root + rf'\Base_data\\GLC\\glc2000_v1_1_05_deg_unify.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)

        df['landcover_GLC'] = val_list
        return df
    def add_landcover_classfication_to_df(self, df):

        val_list=[]
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            landcover=row['landcover_GLC']
            if landcover==0 or landcover==4:
                val_list.append('Evergreen')
            elif landcover==2 or landcover==3 or landcover==5:
                val_list.append('Deciduous')
            elif landcover==6:
                val_list.append('Mixed')
            elif landcover==11 or landcover==12:
                val_list.append('Shrub')
            elif landcover==13 or landcover==14:
                val_list.append('Grass')
            elif landcover==16 or landcover==17 or landcover==18:
                val_list.append('Cropland')
            elif landcover==19 :
                val_list.append('Bare')
            else:
                val_list.append(-999)
        df['landcover_classfication']=val_list

        return df

    def dummies(self,df):
        df=pd.get_dummies(df,columns=['landcover_classfication'])
        return df


    def add_maxmium_LC_change(self, df): ##

        f = data_root+rf'\Base_data\lc_trend\\max_trend.tif'

        array, origin, pixelWidth, pixelHeight, extent = ToRaster().raster2array(f)
        array[array <-99] = np.nan

        LC_dic =DIC_and_TIF().spatial_arr_to_dic(array)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix

            val= LC_dic[pix]
            df.loc[i,'LC_max'] = val
        return df

    def add_aridity_to_df(self,df):  ## here is original aridity index not classification

        f=data_root+rf'Base_data\\aridity_index_05\\aridity_index.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)

        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(fdir + f)

        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        f_name='Aridity'
        print(f_name)
        val_list=[]
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val=val_dic[pix]
            if val<-99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f'{f_name}']=val_list

        return df




    def add_AI_classfication(self, df):

        f = data_root + rf'\Base_data\aridity_index_05\\aridity_index.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            if val==0:
                label='Arid'
            elif val==1:
                label='Semi-Arid'
            elif val==2:
                label='Sub-Humid'
            elif val<-99:
                label=np.nan
            else:
                raise




            val_list.append(label)

        df['AI_classfication'] = val_list
        return df
    def add_dryland_nondryland_to_df(self, df):
        fpath=data_root+rf'\\Base_data\\aridity_index_05\\aridity_index.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue

            val = val_dic[pix]
            if val <= 0.65:
                label = 'Dryland'
            elif 0.65 < val <= 0.8:
                label = 'Sub-humid'
            elif 0.8 < val <= 1.5:
                label = 'Humid'
            elif val > 1.5:
                label = 'Very Humid'
            elif val < -99:
                label = np.nan
            else:
                raise
            val_list.append(label)
        df['Dryland_Humid'] = val_list

        return df


    def remove_duplicate_columns(self,df):

        T.print_head_n(df)
        duplicate_columns = df.columns[df.columns.duplicated()]
        print("重复的列名：", duplicate_columns)

        duplicated_mask = df.columns.duplicated(keep='first')

        # 只保留重复列（保留第二个）
        df = df.loc[:, duplicated_mask]

        T.print_head_n(df)
        return df

    def show_field(self, df):
        for col in df.columns:
            print(col)
        return df
        pass
class CO2_processing():  ## here CO2 processing

    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):

        # self.resample_CO2()
        # self.unify_TIFF()

        # self.plot_CO2()

        # self.per_pix()
        # self.interpolate1()
        self.check_spatial()

        # self.rename()
        pass

    def resample_CO2(self):
        fdir_all = rf'D:\Project3\Data\CO2\CO2_TIFF\original\\'

        outdir_all = rf'D:\Project3\Data\CO2\CO2_TIFF\resample_05\\'


        year = list(range(1982, 2021))
        # print(year)
        # exit()
        for fdir in tqdm(os.listdir(fdir_all)):
            outdir = outdir_all + fdir + '\\'
            T.mk_dir(outdir, force=True)
            for f in os.listdir(fdir_all + fdir):
                if not f.endswith('.tif'):
                    continue
                outf = outdir + f
                if os.path.isfile(outf):
                    continue

                if f.startswith('._'):
                    continue


                print(f)
                # exit()
                date = f.split('.')[0]

                # print(date)
                # exit()
                dataset = gdal.Open(fdir_all + fdir + '/' + f)
                # print(dataset.GetGeoTransform())
                original_x = dataset.GetGeoTransform()[1]
                original_y = dataset.GetGeoTransform()[5]

                # band = dataset.GetRasterBand(1)
                # newRows = dataset.YSize * 2
                # newCols = dataset.XSize * 2
                try:
                    gdal.Warp(outdir + '{}.tif'.format(date), dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass
    def unify_TIFF(self):
        fdir_all=rf'D:\Project3\Data\CO2\CO2_TIFF\\resample_05\\'
        outdir_all=rf'D:\Project3\Data\CO2\CO2_TIFF\\unify_05\\'

        T.mk_dir(outdir_all, force=True)


        for fdir in os.listdir(fdir_all):
            outdir = outdir_all + fdir + '\\'
            Tools().mk_dir(outdir, force=True)

            for f in tqdm(os.listdir(fdir_all+fdir)):
                fpath=join(fdir_all,fdir,f)

                outpath=join(outdir,f)

                if not f.endswith('.tif'):
                    continue
                if f.startswith('._'):
                    continue
                unify_tiff=DIC_and_TIF().unify_raster1(fpath,outpath,0.5)


    def plot_CO2(self):
        fdir=rf'D:\Project3\Data\CO2\dic\monthly_historic\\'
        result_dic={}
        len_dic = {}

        dic = T.load_npy_dir(fdir)
        for pix in dic:
            vals = dic[pix]
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            len_vals = len(vals)

            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(vals)), vals)
            result_dic[pix] = slope
            len_dic[pix] = len_vals

        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(arr, interpolation='nearest', cmap='jet')
        plt.colorbar()
        plt.show()


    def per_pix(self):

        # 设置数据路径
        tif_dir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245\\'
        outdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245_perpix\\'
        T.mkdir(outdir)
        Pre_Process().data_transform(tif_dir,outdir)

    def interpolate1(self):
        perpix_fdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245_perpix\\'
        tif_dir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245\\'
        outdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245_interpoolation\\'
        T.mkdir(outdir)
        year_list = []
        for f in T.listdir(tif_dir):
            if not f.endswith('.tif'):
                continue
            year = f.split('.')[0][0:4]
            year = int(year) - 1982
            year_list.append(year)
        year_list = sorted(list(set(year_list)))
        year_list = np.array(year_list)
        # print(date_list);exit()
        param_list = []

        for f in T.listdir(perpix_fdir):
            params = (perpix_fdir, outdir, f, year_list)
            param_list.append(params)
            # self.kernel_interpolate1(params)
        MULTIPROCESS(self.kernel_interpolate1, param_list).run(process=16)


    def kernel_interpolate1(self,params):
        perpix_fdir,outdir,f,year_list = params
        fpath = join(perpix_fdir, f)
        outpath = join(outdir, f)
        spatial_dict = T.load_npy(fpath)
        K = KDE_plot()
        spatial_dict_interp = {}

        for pix in spatial_dict:
            vals = spatial_dict[pix]
            vals = np.array(vals)
            vals[vals<-999] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue
            if len(vals) == 0:
                continue



            vals_reshape = vals.reshape(-1, 12)
            # print(len(vals_reshape))
            vals_reshape_T = vals_reshape.T
            vals_mon_interp = []
            for mon in range(12):
                vals_mon = vals_reshape_T[mon]
                # print(len(vals_mon))
                # interp = interpolate.interp1d(year_list, vals_mon, kind='linear')
                # a,b,r,p = T.nan_line_fit(year_list, vals_mon)
                # print(year_list)
                # print(len(year_list))
                # print(len(vals_mon));exit()
                a, b, r, p = K.linefit(year_list, vals_mon)
                y = a * (2014 - 1982) + b
                vals_mon_interp.append(y)

                # KDE_plot().plot_fit_line(a, b, r, p, year_list)
                # plt.scatter(year_list, vals_mon)
                # vals_dict = T.dict_zip(year_list,vals_mon)
                # pprint(vals_dict)
                # print(interp(2014-1982))
                # print((vals_dict[2015-1982]+vals_dict[2013-1982])/2)
                # print(y)
                # interpolate = interp1d.
                # plt.show()
            vals_reshape = np.insert(vals_reshape, (2014 - 1982), vals_mon_interp, axis=0)
            vals_interp = vals_reshape.flatten()
            spatial_dict_interp[pix] = vals_interp
            # date_list = []
            # for year in range(1982, 2021):
            #     for mon in range(1, 13):
            #         date = datetime.datetime(year, mon, 1)
            #         date_list.append(date)
            # plt.plot(date_list,vals_interp)
            # plt.show
        T.save_npy(spatial_dict_interp, outpath)

        # 将插值后的数据保存到磁盘
    def check_spatial(self):
        fdir=rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\phenology_year_extraction\\'

        dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in dic:
            vals = dic[pix]['ecosystem_year']
            if np.isnan(np.nanmean(vals)):
                print(pix)
            average = np.nanmean(vals)
            result_dic[pix] = average
        array=DIC_and_TIF().pix_dic_to_spatial_arr(result_dic)
        plt.imshow(array)
        plt.colorbar()
        plt.show()



    def rename(self):
        fdir=rf'D:\Project3\Data\CO2\CO2_TIFF\unify\historic_SSP245\\'

        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            date=f.split('.')[0][6:]
            if  date=='01':
                continue
            else:
                # print(f)

                ## replace other number with 01
                fnew=join(f.split('.')[0][0:6]+'01.'+f.split('.')[1])
                print(fnew)




            os.rename(os.path.join(fdir, f), os.path.join(fdir, fnew))

        pass


class visualize_SHAP():

    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'

        self.spatial_dic = {0: 'CO2_ecosystem_year_shap',
                       1: 'detrended_sum_rainfall_growing_season_CV_shap',
                       2: 'heat_event_frenquency_growing_season_shap',
                       3: 'rainfall_frenquency_growing_season_shap',
                       4: 'rainfall_frenquency_non_growing_season_shap',
                       5: 'rainfall_intensity_growing_season_shap',
                       6: 'rainfall_intensity_non_growing_season_shap',
                       7: 'rainfall_seasonality_all_year_ecosystem_year_shap',
                       8: 'rooting_depth_shap',
                       9: 'silt_shap'}

        self.regroup_dic = {0: 'CO2',
                       1: 'Interanual_rainfall_CV',
                       2: 'Heat_event_frenquency',
                       3: 'Intraanual_rainfall',
                       4: 'Intraanual_rainfall',
                       5: 'Intraanual_rainfall',
                       6: 'Intraanual_rainfall',
                       7: 'Intraanual_rainfall',
                       8: 'Rooting_depth',
                       9: 'Silt'}

        pass

    def run(self):

        self.dominant_factor_shifting()
        pass
    def important_bar_ploting(self):  ##### plot for 4 clusters
        pass
    def SHAP_ploting(self):
        pass

    def spatial_importance_factor(self):  ##### plot for 4 clusters
        pass

    def dominant_factor_shifting(self):
        ## dominant factor spatial map regroup in 4 groups 1 interannual rainfall CV
        ## 2 intraannual rainfall 3 silt 5 root depth

        regroup_dic = {}

        period_list = ['first', 'second']
        result_period={}
        for period in period_list:
            dic = {}
            fpath=rf'D:\Project3\Result\ERA5_025\1mm\SHAP_CO2_{period}_Decade\png\pdp_shap_CV\shapely_contribution\\max_variable.tif'


            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)


            dic['CO2']=0
            dic['Interanual_rainfall_CV']=0
            dic['Heat_event_frenquency']=0
            dic['Intraanual_rainfall']=0
            dic['Silt']=0
            dic['Rooting_depth']=0

            array = np.array(array)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)


            for pix in val_dic:
                vals = val_dic[pix]
                if vals<-99:
                    continue
                result_dic = self.regroup_dic[vals]

                dic[result_dic] = dic[result_dic] + 1

            result_period[period]=dic

        ## dic to df
        df=pd.DataFrame(result_period)
        ##

        df.plot.barh()

        plt.show()




        pass

class PLOT_dataframe():

    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass

    def run(self):

        # self.plot_anomaly_LAI_based_on_cluster()
        self.plot_moving_window()
        pass

    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season.df')
        print(len(df))
        df=self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        # exit()

        #create color list with one green and another 14 are grey

        color_list=['grey']*16
        color_list[0]='green'

        color_list=['green','blue','red','orange','aqua','purple', 'black', 'yellow', 'purple', 'pink', 'grey', 'brown','lime','teal','magenta']
        linewidth_list=[1]*16
        linewidth_list[0]=2


        fig = plt.figure()
        i=1


        variable_list=['phenology_LAI_mean']

        for product in variable_list:
            # print('=========')
            # print(product)
            # print(df_continent.columns.tolist())
            # exit()
            # T.print_head_n(df_continent)
            # exit()


            vals = df[product].tolist()
            # pixel_area_sum = df_continent['pixel_area'].sum()


            # print(vals)

            vals_nonnan=[]
            for val in vals:

                if type(val)==float: ## only screening
                    continue
                if len(val) ==0:
                    continue


                vals_nonnan.append(list(val))


            ###### calculate mean
            vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
            vals_mean=np.nanmean(vals_mean,axis=0)
            # vals_mean=vals_mean/pixel_area_sum

            val_std=np.nanstd(vals_mean,axis=0)

            # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
            plt.plot(vals_mean,color=color_list[i],linewidth=linewidth_list[variable_list.index(product)])
            i = i + 1
            plt.legend()


        year_range = range(1983, 2021)
        xticks=range(0,len(year_range),4)
        plt.xticks(xticks,year_range[::4])
        plt.ylim(-8, 8)
        plt.xlabel('year')
        plt.ylabel(f'relative change (%)')
        plt.grid(which='major', alpha=0.5)
        plt.show()
        # out_pdf_fdir=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\pdf\\'
        # plt.savefig(out_pdf_fdir + 'time_series.pdf', dpi=300, bbox_inches='tight')
        # plt.close()
    def plot_moving_window(self):
        self.map_width = 15.3 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        df_window=T.load_df(rf'E:\Project3\Result\3mm\Dataframe\moving_window_relative_change\\moving_window_relative_change.df')
        T.print_head_n(df_window)
        df_window=self.df_clean(df_window)
        df_time_series=T.load_df(rf'E:\Project3\Result\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season.df')
        df_time_series=self.df_clean(df_time_series)
        variable_list=['LAI4g','NDVI4g','NDVI','GIMMS_plus_NDVI']
        for product in variable_list:
            values_window=df_window[f'{product}_mean'].tolist()
            value_list_window=[]
            for val in values_window:
                if type(val) == float:  ## only screening
                    continue
                if len(val) != 24:
                    continue

                value_list_window.append(val)

            average_value_window=np.array(value_list_window)

            average_value_window[average_value_window>100]=np.nan
            average_value_window[average_value_window<-100]=np.nan

            average_value_window=np.nanmean(average_value_window,axis=0)

            ###### 38 yearly time series
            vals_yearly = df_time_series[product].tolist()

            value_list_yearly = []
            for val in vals_yearly:
                if type(val) == float:  ## only screening
                    continue
                if len(val) != 38:
                    continue
                value_list_yearly.append(val)

            average_value_yearly = np.array(value_list_yearly)
            average_value_yearly[average_value_yearly > 100] = np.nan
            average_value_yearly[average_value_yearly < -100] = np.nan
            average_value_yearly = np.nanmean(average_value_yearly, axis=0)

            fig, ax1 = plt.subplots(figsize=(self.map_width, self.map_height))

            # Plot the first dataset (38-year data)
            year_list = range(1983, 2021)
            ax1.plot(year_list, average_value_yearly,  color='k', lw=1, markersize=3, linestyle='-',marker='o', fillstyle='none')
            slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, average_value_yearly)
            plt.plot(year_list, [slope * x + intercept for x in year_list],
                     linestyle='--', color='k', alpha=0.5)
            plt.text(0, 1, f'{slope:.2f}', transform=plt.gca().transAxes,
                     fontsize=10)

            ax1.set_ylabel("Relative change (%/yr)", color='k', fontsize=12, )
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.set_xticks(range(1983, 2021, 4))
            ax1.set_ylim(-6, 8)
            # plt.show()

            # Create a second y-axis
            # ax2 = ax1.twinx()
            ## plot fitting curve
            average_value_yearly_smooth = SMOOTH().smooth_convolve(average_value_yearly, window_len=15)
            ax1.plot(year_list, average_value_yearly_smooth, color='grey', lw=2, alpha=0.5)
            #
            # ax2.set_xticks(range(0, 24, 4))
            # window_size = 15
            #
            # years_window = np.arange(1983 + window_size // 2, 2020 - window_size // 2 + 1)
            #
            #
            # # Plot the second dataset
            # ax2.plot(years_window, average_value_window, color='green', lw=2, alpha=0.5)
            #
            # ax2.set_ylabel("relative change (%/yr)", color='green', fontsize=12)
            # ax2.tick_params(axis='y', labelcolor='green')
            # ax2.set_ylim(-5, 5)



            plt.title(f'{product}')
            ## add y=0

            ax1.axhline(y=0, color='grey', lw=1.2, ls='--')

            plt.show()

            # outdir = result_root + rf'3mm\relative_change_growing_season\whole_period\\plot\\'
            # T.mk_dir(outdir)
            # outf = join(outdir, f'{product}_time_series.pdf')
            # plt.savefig(outf)
            # plt.close()




        pass


    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df



class PLOT_data():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass

    def run(self):

        self.plot_anomaly_LAI_based_on_cluster()
        pass

    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(rf'D:\Project3\ERA5_025\Dataframe\LAI4g_CV_continent\\LAI4g_CV_continent.df')
        print(len(df))
        df=self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        # exit()




        #create color list with one green and another 14 are grey

        color_list=['grey']*16
        color_list[0]='green'

        color_list=['green','blue','red','orange','aqua','purple', 'black', 'yellow', 'purple', 'pink', 'grey', 'brown','lime','teal','magenta']
        linewidth_list=[1]*16
        linewidth_list[0]=2


        fig = plt.figure()
        i=1


        variable_list=['detrended_growing_season_LAI_mean_CV']


        for continent in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America',  'global']:


            if continent=='global':
                df_continent=df
            else:

                df_continent = df[df['continent'] == continent]



            for product in variable_list:
                # print('=========')
                # print(product)
                # print(df_continent.columns.tolist())
                # exit()
                # T.print_head_n(df_continent)
                # exit()


                vals = df_continent[product].tolist()
                # pixel_area_sum = df_continent['pixel_area'].sum()


                # print(vals)

                vals_nonnan=[]
                for val in vals:


                    if type(val)==float: ## only screening
                        continue
                    if len(val) ==0:
                        continue
                    # val[val<-99]=np.nan

                    if not len(val) == 23:
                        ## add nan to the end of the list
                        for j in range(1):
                            val=np.append(val,np.nan)
                        # print(val)
                        # print(len(val))


                    vals_nonnan.append(list(val))

                        # exit()
                    # print(type(val))
                    # print(len(val))
                    # print(vals)

                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                # vals_mean=vals_mean/pixel_area_sum

                val_std=np.nanstd(vals_mean,axis=0)

                # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                plt.plot(vals_mean,label=continent,color=color_list[i],linewidth=linewidth_list[variable_list.index(product)])
                i = i + 1
                plt.legend()

        plt.xticks(range(0, 23, 4))
        window_size = 15

        # set xticks with 1982-1997, 1998-2013,.. 2014-2020
        year_range = range(1982, 2021)
        year_range_str = []
        for year in year_range:

            start_year = year
            end_year = year + window_size - 1
            if end_year > 2020:
                break
            year_range_str.append(f'{start_year}-{end_year}')
        plt.xticks(range(len(year_range_str))[::3], year_range_str[::3], rotation=45, ha='right')
        plt.xticks(range(0, 23, 3))



        plt.xlabel('window')

        plt.ylabel(f'LAI CV (%)')


        plt.grid(which='major', alpha=0.5)

        # plt.show()
        out_pdf_fdir=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\pdf\\'
        plt.savefig(out_pdf_fdir + 'time_series.pdf', dpi=300, bbox_inches='tight')
        plt.close()




    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 120]
        df = df[df['Aridity'] < 0.65]
        # df = df[df['LC_max'] < 20]

        df = df[df['MODIS_LUCC'] != 12]


        df = df[df['landcover_classfication'] != 'Cropland']

        return df
class greening_analysis():
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass

    def run(self):

        # self.relative_change()
        # self.anomaly()
        # self.anomaly_two_period()
        # self.long_term_mean()
        # self.weighted_average_LAI()
        # self.plot_time_series()
        # self.plot_time_series_spatial()
        # self.annual_growth_rate()
        # self.trend_analysis_simply_linear()
        # self.trend_analysis_TS()
        # self.heatmap()
        # self.heatmap()
        self.plot_robinson()
        # self.statistic_trend_bar()
        # self.plot_significant_percentage_area()
        # self.plot_significant_percentage_area_two_period()
        # self.plot_spatial_barplot_period()
        # self.plot_spatial_histgram_period()
        # self.stacked_bar_plot()
        # self.statistic_analysis()
        # self.plot_robinson()
        pass

    def relative_change(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'3mm\extract_GLOBMAP_phenology_year\\'
        outdir = result_root + rf'3mm\extract_GLOBMAP_phenology_year\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            if  'detrend' in f:
                continue
            if 'zscore' in f:
                continue



            outf = outdir + f.split('.')[0]+'_relative_change.npy'
            if isfile(outf):
                continue
            print(outf)

            dic = T.load_npy(fdir + f)

            zscore_dic = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                # print(len(dic[pix]))
                time_series = dic[pix]['growing_season']


                time_series = np.array(time_series)
                # time_series = time_series[3:37]

                print(len(time_series))

                if np.isnan(np.nanmean(time_series)):
                    continue

                time_series = time_series
                mean = np.nanmean(time_series)
                relative_change = (time_series - mean) / mean * 100
                anomaly = time_series - mean
                zscore_dic[pix] = relative_change
                # plot
                # plt.plot(anomaly)
                # plt.legend(['anomaly'])
                # plt.show()
                #
                # plt.plot(relative_change)
                # plt.legend(['relative_change'])
                # # plt.legend(['anomaly','relative_change'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)

    def long_term_mean(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\extraction_LAI4g\\'
        outdir = result_root + rf'3mm\mean_growing_season\\\\whole_period\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            if 'detrend' in f:
                continue


            outf = outdir + f.split('.')[0]
            if isfile(outf):
                continue
            print(outf)

            dic = T.load_npy(fdir + f)

            zscore_dic = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                # print(len(dic[pix]))
                time_series = dic[pix]['growing_season']


                time_series = np.array(time_series)
                # time_series = time_series[3:37]

                print(len(time_series))

                if np.isnan(np.nanmean(time_series)):
                    continue

                time_series = time_series
                mean = np.nanmean(time_series)


                zscore_dic[pix] = mean
                # plt.plot(time_series)
                #
                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                # plt.show()

                ## save

            DIC_and_TIF().pix_dic_to_tif(zscore_dic,  outf.split('.')[0] + '.tif')

            np.save(outf, zscore_dic)

    def anomaly(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\extraction_LAI4g\\'
        outdir = result_root + rf'3mm\anomaly_growing_season\\\\whole_period\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            if 'detrend' in f:
                continue

            outf = outdir + f.split('.')[0]
            if isfile(outf):
                continue
            print(outf)

            dic = T.load_npy(fdir + f)

            zscore_dic = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                # print(len(dic[pix]))
                time_series = dic[pix]['growing_season']


                time_series = np.array(time_series)


                print(len(time_series))

                if np.isnan(np.nanmean(time_series)):
                    continue

                time_series = time_series
                mean = np.nanmean(time_series)
                relative_change = (time_series - mean)

                zscore_dic[pix] = relative_change
                # plot
                # plt.plot(time_series)
                #
                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)
    def anomaly_two_period(self):
        fdir = result_root + rf'3mm\relative_change_growing_season\whole_period\\'
        for f in os.listdir(fdir):

            if 'detrend' in f:
                continue
            outdir = result_root + rf'3mm\relative_change_growing_season\\two_period\\'
            Tools().mk_dir(outdir, force=True)
            outf = outdir + f.split('.')[0]
            dic = T.load_npy(fdir + f)
            result_dic_first = {}
            result_dic_second = {}

            for pix in tqdm(dic):
                time_series = dic[pix]
                time_series_first = time_series[0:19]
                time_series_second = time_series[19:]

                result_dic_first[pix] = time_series_first
                result_dic_second[pix] = time_series_second

            np.save(outf+'_first',result_dic_first)
            np.save(outf+'_second',result_dic_second)


        pass

    def weighted_average_LAI(self):  ###add weighted average LAI in dataframe
        df =result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season_landsat.df'
        df = T.load_df(df)
        df_clean = self.df_clean(df)
        # print(len(df_clean))
        variable='Landsat'


        # 去除异常值（根据业务需求设定阈值）
        df_clean_ii = df_clean[(df_clean[f'{variable}'] > -50) & (df_clean[f'{variable}'] < 50)]
        # print(len(df_clean_ii));exit()
        # Step 1: 计算纬度权重
        df_clean_ii['latitude_weight'] = np.cos(np.radians(df_clean_ii['lat']))
        # Step 2: 按年份对权重进行归一化
        # 确保每一年干旱区的权重总和为1
        df_clean_ii['normalized_weight'] = df_clean_ii.groupby('year')['latitude_weight'].transform(lambda x: x / x.sum())

        # print(df_clean_ii.groupby('year')['normalized_weight'].sum())
        # exit()

        # Step 3: 计算加权平均LAI
        # weighted_avg_lai = df_clean_ii.groupby('year')['LAI_relative_change'].apply(lambda x: (x * df_clean_ii['normalized_weight']).sum())
        df_clean_ii[f'weighted_avg_{variable}_contribution'] = df_clean_ii[f'{variable}'] * df_clean_ii['normalized_weight']

        weighted_avg_variable_per_year = (
            df_clean_ii.groupby('year')[f'weighted_avg_{variable}_contribution'].sum().reset_index(name=f'weighted_avg_{variable}')
        )

        df_clean_ii = df_clean_ii.merge(weighted_avg_variable_per_year, on='year', how='left')
        # df_clean_ii[f'weighted_avg_{variable}'] = df_clean_ii[weighted_avg_lai_per_year]
        T.print_head_n(df_clean_ii)

        # exit()
        # T.print_head_n(df_clean_ii);exit()
        outf=result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season_landsat.df'
        T.save_df(df_clean_ii,outf)
        T.df_to_excel(df_clean_ii, outf)

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
    def plot_time_series(self):
        df=T.load_df(rf'D:\Project3\Result\3mm\product_consistency\dataframe\\relative_change.df')
        df=self.df_clean(df)

        year_range = range(1983, 2021)
        result_dic = {}

        variable_list=['NDVI4g','GIMMS_plus_NDVI','NDVI',]
        # variable_list=['LAI4g']
        variable_list = ['LAI4g','SNU_LAI_relative_change','GLOBMAP_LAI_relative_change',
                         'composite_LAI_relative_change_mean',
                         'composite_LAI_relative_change_median']


        for var in variable_list:

            result_dic[var] = {}
            data_dic = {}

            for year in year_range:
                df_i = df[df['year'] == year]
                # vals = df_i[f'weighted_avg_{var}'].tolist()
                vals = df_i[f'{var}'].tolist()
                data_dic[year] = np.nanmean(vals)
            result_dic[var] = data_dic
        ##dic to df

        df_new = pd.DataFrame(result_dic)

        T.print_head_n(df_new)
        ##calculate slope and intercept
        slope_dic={}
        intercept_dic={}
        year_range_dic={}
        data_range_dic={}

        for var in variable_list:
            if var=='GOSIF':
                year_range_temp = range(2002, 2021)
                vals=df_new[var].tolist()
                print(vals)
                vals_temp=vals[19:]
                data_range_dic[var] = vals_temp
                year_range_dic[var]=year_range_temp
            elif var=='landsat':
                year_range_temp = range(1987, 2021)
                vals=df_new[var].tolist()
                data_range_dic[var] = vals
                year_range_dic[var] = year_range_temp

            else:
                year_range_temp = range(1983, 2021)
                vals=df_new[var].tolist()
                data_range_dic[var] = vals
                year_range_dic[var] = year_range_temp


            slope, intercept, r_value, p_value, std_err = stats.linregress(year_range_dic[var], data_range_dic[var])
            # slope = round(slope, 2)

            slope_dic[var]=slope
            intercept_dic[var]=intercept
        #     print(slope,intercept)
        # exit()
        ## plot subplot
        fig, ax1 = plt.subplots(2, 2, figsize=(self.map_width*3, self.map_height*1.1))
        flag = 0

        for var in variable_list:

            ax1 = plt.subplot(3, 2, flag + 1)


            # Plot the first dataset (38-year data)
            year_list = range(1983, 2021)
            ax1.plot(year_range_dic[var],data_range_dic[var] , color='k', lw=1, markersize=3, linestyle='-', marker='o',
                     fillstyle='none')
            slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, data_range_dic[var])
            ax1.plot(year_list, [slope * x + intercept for x in year_list],
                     linestyle='--', color='k', alpha=0.5)
            ax1.text(0, 1, f'{slope:.2f}', transform=plt.gca().transAxes,
                     fontsize=10)

            ax1.set_ylabel("Relative change (%/yr)", color='k', fontsize=10, font='Arial')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.set_xticks(range(1983, 2021, 4))
            ax1.set_xticklabels(range(1983, 2021, 4), rotation=45)
            # ax1.set_ylim(-8, 8)

            ## plot fitting curve
            average_value_yearly_smooth = SMOOTH().smooth_convolve(data_range_dic[var], window_len=15)
            ax1.plot(year_list, average_value_yearly_smooth, color='grey',
                     lw=2, alpha=0.5,label='Fitting curve')
            plt.ylim(-10,10)
            plt.legend()

            ## add y=0

            ax1.axhline(y=0, color='grey', lw=1.2, ls='--')
            flag += 1
            plt.title(var)
        plt.tight_layout()
        plt.show()

        # outdir = result_root + rf'3mm\relative_change_growing_season\\whole_period\\plot\\'
        # T.mk_dir(outdir)
        # outf = outdir + rf'{var}_all_15.pdf'

        # plt.savefig(outf)
        # plt.close()
        # T.open_path_and_file(outdir)



    def plot_time_series_spatial(self):
        fdir = result_root + rf'3mm\relative_change_growing_season\\'

        for f in os.listdir(fdir):
            if not 'LAI' in f:
                continue
            outf = fdir + f
            print(outf)
            dic = np.load(outf, allow_pickle=True).item()
            all_data = []


            for pix in dic:
                time_series = dic[pix]
                # print(time_series)
                if np.isnan(np.nanmean(time_series)):
                    continue

                all_data.append(time_series)

            all_data = np.array(all_data)
            average_data = np.nanmean(all_data, axis=0)


            plt.plot(average_data)
            plt.show()





    def annual_growth_rate(self):
        ## input raw data becuase time2-time1
        fdir=data_root + rf'\TRENDY\S2\extract_phenology_LAI_mean\\'
        outdir=result_root + rf'\3mm\annual_growth_rate\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'NIRv' in f:
                continue
            dict=np.load(fdir+f,allow_pickle=True).item()
            growth_rate_dic={}
            for pix in tqdm(dict):
                time_series=dict[pix]['growing_season']
                # print(len(time_series))
                if len(time_series)==0:
                    continue
                # print(time_series)
                growth_rate_time_series=np.zeros(len(time_series)-1)
                for i in range(len(time_series)-1):

                    growth_rate_time_series[i]=(time_series[i+1]-time_series[i])/time_series[i]*100
                growth_rate_dic[pix]=growth_rate_time_series
            np.save(outdir+f,growth_rate_dic)

        pass

    def trend_analysis_TS(self):
        from scipy.stats import theilslopes
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'\\3mm\extract_SNU_LAI_phenology_year\\'
        outdir = result_root + rf'3mm\extract_SNU_LAI_phenology_year\\trend_analysis_TS\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            print(f)

            if 'detrend' in f:
                continue


            if not f.endswith('.npy'):
                continue

            outf=outdir+f.split('.')[0]
            if os.path.isfile(outf+'_trend.tif'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r,c=pix
                if r<60:
                    continue
                landcover_value=crop_mask[pix]
                if landcover_value==16 or landcover_value==17 or landcover_value==18:
                    continue
                if dic_modis_mask[pix]==12:
                    continue

                time_series=dic[pix]['growing_season']
                # print(time_series)


                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series))==1:
                    continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                # try:
                #
                #         # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                #         slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
                    ## Theil-Sen regression
                slope, intercept, p_value = self.TS_trend_analysis(time_series)
                # plt.scatter(np.arange(len(time_series)), time_series)
                # print(slope, intercept, p_value)
                # plt.show()
                trend_dic[pix] = slope
                p_value_dic[pix] = p_value
                # except:
                #     continue

            arr_trend = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(trend_dic)
            arr_trend_dryland = arr_trend * array_mask

            # p_value_arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(p_value_dic)
            # p_value_arr_dryland = p_value_arr * array_mask


            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()
            # exit()

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend_dryland, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.5).arr_to_tif(p_value_arr_dryland, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend_dryland)
            # np.save(outf + '_p_value', p_value_arr_dryland)
    def TS_trend_analysis(self,vals):
        x = np.arange(len(vals))
        regressor = TheilSenRegressor()
        regressor.fit(x.reshape(-1, 1), vals)
        slope = regressor.coef_[0]
        intercept = regressor.intercept_
        y_pred = regressor.predict(x.reshape(-1, 1))
        plt.scatter(x, y_pred)
        p_value = self.one_sample_t_test(x, vals,regressor)

        return slope, intercept, p_value

    def trend_analysis_simply_linear(self):
        from scipy.stats import theilslopes
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'3mm\relative_change_growing_season\TRENDY\\'
        outdir = result_root + rf'3mm\relative_change_growing_season\\\\TRENDY\\trend_3\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):



            if not f.endswith('.npy'):
                continue

            outf=outdir+f.split('.')[0]
            # if os.path.isfile(outf+'_trend.tif'):
            #     continue
            # print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r,c=pix
                if r<60:
                    continue
                landcover_value=crop_mask[pix]
                if landcover_value==16 or landcover_value==17 or landcover_value==18:
                    continue
                if dic_modis_mask[pix]==12:
                    continue

                time_series=dic[pix]
                # print(time_series)


                if len(time_series) == 0:
                    continue
                if isnan(np.nanmean(time_series)):
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series))==1:
                    continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                try:

                        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                        slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
                        trend_dic[pix] = slope
                        p_value_dic[pix] = p_value


                except:
                    continue



            arr_trend = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(trend_dic)
            arr_trend_dryland = arr_trend * array_mask

            p_value_arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(p_value_dic)
            p_value_arr_dryland = p_value_arr * array_mask


            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.3, vmax=0.3)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()
            # # exit()

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend_dryland, outf + '_trend.tif')
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(p_value_arr_dryland, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend_dryland)
            np.save(outf + '_p_value', p_value_arr_dryland)


    def one_sample_t_test(self,X, y, reg):
        from sklearn.linear_model import LinearRegression
        beta_hat = [reg.intercept_] + reg.coef_.tolist()
        n = len(y)
        # compute the p-values
        # from scipy.stats import t
        # add ones column
        X1 = np.column_stack((np.ones(n), X))
        # standard deviation of the noise.
        sigma_hat = np.sqrt(np.sum(np.square(y - X1 @ beta_hat)) / (n - X1.shape[1]))
        # estimate the covariance matrix for beta
        beta_cov = np.linalg.inv(X1.T @ X1)
        # the t-test statistic for each variable from the formula from above figure
        t_vals = beta_hat / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))
        # compute 2-sided p-values.
        p_vals = t.sf(np.abs(t_vals), n - X1.shape[1]) * 2


        return p_vals


    def gen_robinson_template(self):
        pass
    def plot_robinson(self):

        # fdir_trend = result_root+rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result\npy_time_series\trend\\'
        fdir_trend = result_root+rf'Composite_LAI\relative_change\trend\\'
        temp_root = result_root+rf'\TRENDY\S2\relative_change\relative_change\trend_analysis_relative_change\\temp_plot\\'
        outdir = result_root+rf'FIGURE\\Figure1a\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):
            if not 'composite' in f:
                continue
            if 'p_value' in f:
                continue

            if not f.endswith('.tif'):
                continue


            fname = f.split('.')[0]
            fname_p_value = fname.replace('trend', 'p_value')
            print(fname_p_value)
            fpath = fdir_trend + f
            p_value_f = fdir_trend + fname_p_value+'.tif'
            print(p_value_f)
            # exit()
            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=-1, vmax=1, is_discrete=True, colormap_n=9,)

            Plot_Robinson().plot_Robinson_significance_scatter(m,p_value_f,temp_root,0.05, s=0.5, marker='.')
            # plt.title(f'{fname}')
            # plt.show()
            outf = outdir + f+'.pdf'
            plt.savefig(outf)
            plt.close()
            T.open_path_and_file(outdir)
            exit()

    def statistic_trend_bar(self):
        fdir = result_root + rf'3mm\product_consistency\relative_change\Trend\\'
        variable_list=['GLOBMAP_LAI_relative_change','LAI4g','composite_LAI_relative_change_mean','SNU_LAI_relative_change']
        for variable in variable_list:
            f_trend_path=fdir+f'{variable}_trend.tif'
            f_pvalue_path=fdir+f'{variable}_p_value.tif'
            result_dic={}

            arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_path)
            arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_pvalue_path)
            arr_corr[arr_corr<-99]=np.nan
            arr_corr[arr_corr>99]=np.nan
            arr_corr=arr_corr[~np.isnan(arr_corr)]

            arr_pvalue[arr_pvalue<-99]=np.nan
            arr_pvalue[arr_pvalue>99]=np.nan
            arr_pvalue=arr_pvalue[~np.isnan(arr_pvalue)]
            ## corr negative and positive
            arr_corr = arr_corr.flatten()
            arr_pvalue = arr_pvalue.flatten()
            arr_pos=len(arr_corr[arr_corr>0])/len(arr_corr)*100
            arr_neg=len(arr_corr[arr_corr<0])/len(arr_corr)*100


            ## significant positive and negative
            ## 1 is significant and 2 positive or negative

            mask_pos = (arr_corr > 0) & (arr_pvalue < 0.05)
            mask_neg = (arr_corr < 0) & (arr_pvalue < 0.05)


            # 满足条件的像元数
            count_positive_sig = np.sum(mask_pos)
            count_negative_sig = np.sum(mask_neg)

            # 百分比
            significant_positive = (count_positive_sig / len(arr_corr)) * 100
            significant_negative = (count_negative_sig / len(arr_corr)) * 100
            result_dic = {

                'sig neg': significant_negative,
                'non sig neg': arr_neg,
                'non sig pos': arr_pos,
                'sig pos': significant_positive



            }
            # df_new=pd.DataFrame(result_dic,index=[variable])
            # ## plot
            # df_new=df_new.T
            # df_new=df_new.reset_index()
            # df_new.columns=['Variable','Percentage']
            # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
            # plt.show()
            color_list = [
                '#844000',
                'lightgray',

                'lightgray',
                '#064c6c',
            ]

            # color_list = [
            #     '#844000',
            #     '#fc9831',
            #     '#fffbd4',
            #     '#86b9d2',
            #     '#064c6c',
            # ]

            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]

            fig, ax = plt.subplots(1, 1, figsize=(3, 3))


            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage')
                plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            # plt.show()
            outdir=result_root+rf'\3mm\FIGURE\Figure1_greening\\'
            plt.savefig(outdir+f'{variable}.pdf', bbox_inches='tight', dpi=300)
            plt.close()


    def heatmap(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_LAI_trend = result_root + rf'3mm\relative_change_growing_season\trend\\phenology_LAI_mean_trend.tif'
        f_LAI_sensitivity_precipitation=result_root+rf'\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\npy_time_series\\trend\\\sum_rainfall_trend.tif'
        f_precip_trend=result_root + rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\trend_ecosystem_year\\sum_rainfall_trend.tif'

        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_LAI_trend)

        arr_LAI_trend[arr_LAI_trend < -999] = np.nan

        arr_LAI_sensitivity_precip, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_LAI_sensitivity_precipitation)
        arr_precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_precip_trend)
        arr_LAI_sensitivity_precip[arr_LAI_sensitivity_precip < -999] = np.nan
        arr_precip_trend[arr_precip_trend < -999] = np.nan
        arr_LAI_trend=np.array(arr_LAI_trend)
        arr_LAI_sensitivity_precip=np.array(arr_LAI_sensitivity_precip)
        arr_precip_trend=np.array(arr_precip_trend)

        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        dic_arr_LAI_sensitivity_precip=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_sensitivity_precip)
        dic_precip_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_precip_trend)

        result_dic={
            'LAI_trend':dic_LAI_trend,
            'LAI_sensitivity_precip_trend':dic_arr_LAI_sensitivity_precip,
            'intra_Preci_CV_trend':dic_precip_trend
        }
        # plt.hist(result_dic['LAI_trend'].values())
        # plt.show()
        # plt.hist(result_dic['LAI_sensitivity_precip_trend'].values())
        # plt.show()
        # plt.hist(result_dic['intra_Preci_CV_trend'].values())
        # plt.show()


        df=T.spatial_dics_to_df(result_dic)
        T.print_head_n(df)
        x_var = 'LAI_sensitivity_precip_trend'
        y_var = 'intra_Preci_CV_trend'
        z_var = 'LAI_trend'
        bin_x = [ -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
        bin_y = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()

        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)

        T.plot_df_bin_2d_matrix(matrix_dict,-.3,.3,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False)

        plt.colorbar()
        plt.show()
        matrix_dict, x_ticks_list, y_ticks_list = self.df_bin_2d_sample_size(df, val_col_name=z_var,
                                                                             col_name_x=x_var,
                                                                             col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        T.plot_df_bin_2d_matrix(matrix_dict, 0, 300, x_ticks_list, y_ticks_list, cmap='RdBu',
                                is_only_return_matrix=False)

        plt.colorbar()
        plt.show()


    #     plt.savefig(result_root + rf'Data_frame\\Frequency\\Trendy_{region}.pdf', dpi=300, )
    #     plt.close()

    def heatmap2(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_LAI_trend = result_root + rf'\3mm\relative_change_growing_season\trend\\phenology_LAI_mean_trend.tif'
        f_aridity=data_root+rf'Base_data\aridity_index_05\dryland_mask05.tif'
        f_precip_trend=result_root + rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\trend_ecosystem_year\\sum_rainfall_trend.tif'

        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_LAI_trend)

        arr_LAI_trend[arr_LAI_trend<-999]=np.nan
        # plt.imshow(arr_LAI_trend)
        # plt.show()

        aridity, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_aridity)
        aridity[aridity<-999]=np.nan
        aridity[aridity>999]=np.nan
        # plt.imshow(aridity)
        # plt.show()

        precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_precip_trend)
        precip_trend[precip_trend<-999]=np.nan
        precip_trend[precip_trend>999]=np.nan
        # plt.imshow(precip_trend)
        # plt.show()


        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        aridty_dic=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(aridity)
        precip_trend_dic=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(precip_trend)

        result_dic={
            'LAI_trend':dic_LAI_trend,
            'aridity':aridty_dic,
            'precip_trend':precip_trend_dic
        }
        df=T.spatial_dics_to_df(result_dic)
        df=df.dropna()
        T.print_head_n(df)
        x_var = 'aridity'
        y_var = 'precip_trend'
        z_var = 'LAI_trend'
        bin_x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
        bin_x=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
        bin_y = [-1,-0.5,0,0.5,1]

        # percentile_list=np.linspace(0,100,9)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()


        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)
        T.plot_df_bin_2d_matrix(matrix_dict, -.2, .2, x_ticks_list, y_ticks_list, cmap='RdBu',
                                is_only_return_matrix=False)
        plt.colorbar()
        plt.figure()


        matrix_dict, x_ticks_list, y_ticks_list = self.df_bin_2d_sample_size(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)


        T.plot_df_bin_2d_matrix(matrix_dict,0,300,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False)

        plt.colorbar()
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()
    def df_bin_2d_sample_size(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
        df_group_y, _ = T.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = T.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict,x_ticks_list,y_ticks_list


    def plot_significant_percentage_area_two_period(self):  ### insert bar plot for all spatial map to calculate percentage

        dff=result_root+rf'3mm\Dataframe\anomaly\\anomaly.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # T.print_head_n(df); exit()
        ##plt histogram of LAI
        variable='LAI4g'
        df=df[df[f'{variable}_trend_whole']<0.1]
        df=df[df[f'{variable}_trend_whole']>-0.1]
        period_list = ['first', 'second']
        result_dict = {}
        for period in period_list:

            significant_browning_count = 0
            non_significant_browning_count = 0
            significant_greening_count = 0
            non_significant_greening_count = 0
            vals_p_value = df[f'{variable}_{period}_p_value_relative_change'].values

            for i in range(len(vals_p_value)):
                if vals_p_value[i] < 0.05:
                    if df[f'{variable}_{period}_trend_relative_change'].values[i] > 0:
                        significant_greening_count = significant_greening_count + 1
                    else:
                        significant_browning_count = significant_browning_count + 1
                else:
                    if df[f'{variable}_{period}_trend_relative_change'].values[i] > 0:
                        non_significant_browning_count = non_significant_browning_count + 1
                    else:
                        non_significant_greening_count = non_significant_greening_count + 1
                ## plot bar
            ##calculate percentage
            significant_greening_percentage = significant_greening_count / len(vals_p_value)*100
            non_significant_greening_percentage = non_significant_greening_count / len(vals_p_value)*100
            significant_browning_percentage = significant_browning_count / len(vals_p_value)*100
            non_significant_browning_percentage = non_significant_browning_count / len(vals_p_value)*100

            count = [non_significant_browning_percentage,significant_browning_percentage, significant_greening_percentage,

                     non_significant_greening_percentage]
            print(count)
            result_dict[period] = count


        labels = ['non_significant_browning','significant_browning', 'significant_greening',
                  'non_significant_greening']
        color_list=['navajowhite','chocolate','navy','lightblue',]
        ##gap = 0.1
        df_new=pd.DataFrame(result_dict)
        df_new_T=df_new.T


        df_new_T.plot.barh( stacked=True, color=color_list,legend=False,width=0.1,)
        ## add legend
        plt.legend(labels)

        plt.ylabel('Percentage (%)')
        plt.tight_layout()

        # plt.show()
        ## Save
        outdir = result_root+rf'3mm\relative_change_growing_season\two_period\plot\\'
        outf = join(outdir, f'{variable}_significant_percentage_area_two_period.pdf')
        plt.savefig(outf)
        plt.close()


    def plot_significant_percentage_area(self):  ### insert bar plot for all spatial map to calculate percentage

        dff=result_root+rf'\3mm\bivariate_analysis\Dataframe\\Trend.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # T.print_head_n(df); exit()
        ##plt histogram of LAI
        variable='SNU_LAI'
        df=df[df[f'{variable}_relative_change_trend']<50]
        df=df[df[f'{variable}_relative_change_trend']>-50]


        vals_p_value = df[f'{variable}_relative_change_p_value'].values
        significant_browning_count = 0
        non_significant_browning_count = 0
        significant_greening_count = 0
        non_significant_greening_count = 0

        for i in range(len(vals_p_value)):
            if vals_p_value[i] < 0.05:
                if df[f'{variable}_relative_change_trend'].values[i] > 0:
                    significant_greening_count = significant_greening_count + 1
                else:
                    significant_browning_count = significant_browning_count + 1
            else:
                if df[f'{variable}_relative_change_trend'].values[i] > 0:
                    non_significant_browning_count = non_significant_browning_count + 1
                else:
                    non_significant_greening_count = non_significant_greening_count + 1
            ## plot bar
        ##calculate percentage
        significant_greening_percentage = significant_greening_count / len(vals_p_value)*100
        non_significant_greening_percentage = non_significant_greening_count / len(vals_p_value)*100
        significant_browning_percentage = significant_browning_count / len(vals_p_value)*100
        non_significant_browning_percentage = non_significant_browning_count / len(vals_p_value)*100

        count = [non_significant_browning_percentage,significant_browning_percentage, significant_greening_percentage,

                 non_significant_greening_percentage]
        print(count)
        labels = ['non_significant_browning','significant_browning', 'significant_greening',
                  'non_significant_greening']
        color_list=['navajowhite','chocolate','navy','lightblue',]
        ##gap = 0.1
        df_new=pd.DataFrame({'count':count})
        df_new_T=df_new.T


        df_new_T.plot.barh( stacked=True, color=color_list,legend=False,width=0.1,)
        ## add legend
        plt.legend(labels)

        plt.ylabel('Percentage (%)')
        plt.tight_layout()

        plt.show()


    def plot_spatial_barplot_period(self):

        dff = result_root + rf'3mm\Dataframe\Trend\\Trend.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        ##plt histogram of LAI
        df=df[df['LAI4g_trend']<30]
        df=df[df['LAI4g_trend']>-30]
        period_list=['1983_2001','2001_2020',]
        result_dict = {}
        for period in period_list:

            significant_browning_count = 0
            non_significant_browning_count = 0
            significant_greening_count = 0
            non_significant_greening_count = 0

            for i,row in df.iterrows():

                vals_p_value = row[f'LAI4g_{period}_p_value']
                if vals_p_value < 0.05:
                    if row[f'LAI4g_{period}_trend'] > 0:
                        significant_greening_count = significant_greening_count + 1
                    else:
                        significant_browning_count = significant_browning_count + 1
                else:
                    if row[f'LAI4g_{period}_trend'] < 0:
                        non_significant_browning_count = non_significant_browning_count + 1
                    else:
                        non_significant_greening_count = non_significant_greening_count + 1
                ## plot bar
            ##calculate percentage
            significant_greening_percentage = significant_greening_count / len(df)*100
            non_significant_greening_percentage = non_significant_greening_count / len(df)*100
            significant_browning_percentage = significant_browning_count / len(df)*100
            non_significant_browning_percentage = non_significant_browning_count / len(df)*100

            count = [non_significant_browning_percentage,significant_browning_percentage, significant_greening_percentage,

                     non_significant_greening_percentage]
            result_dict[period]=count
        print(result_dict)

        labels = ['non_significant_browning','significant_browning', 'significant_greening',
                  'non_significant_greening']
        color_list=['chocolate','navajowhite','lightblue','navy']
        ##gap = 0.1

        df_new=pd.DataFrame(result_dict)

        df_new.plot(kind='bar', stacked=False, color=color_list, edgecolor='black', figsize=(3, 3),legend=True)
        plt.ylabel('Percentage (%)')
        plt.xticks(np.arange(0,4),labels)
        plt.tight_layout()


        plt.show()



    def plot_spatial_histgram_period(self):
        from scipy.stats import gaussian_kde
        dff=result_root+rf'3mm\Dataframe\anomaly\\anomaly.df'
        df=T.load_df(dff)
        df = self.df_clean(df)
        variable='GIMMS_plus_NDVI'

        ##plt histogram of LAI
        # plt.hist(df['LAI4g_trend_whole'],bins=100)
        # plt.show();exit()
        df=df[df[f'{variable}_trend_whole']<0.1]
        df=df[df[f'{variable}_trend_whole']>-0.1]
        first_period=df[f'{variable}_first_trend_relative_change'].values
        first_period_vals=np.array(first_period)
        first_period_vals[first_period_vals<-1]=np.nan
        first_period_vals[first_period_vals > 1] = np.nan
        first_period_vals=first_period_vals[~np.isnan(first_period_vals)]

        second_period=df[f'{variable}_second_trend_relative_change'].values
        second_period_vals=np.array(second_period)
        second_period_vals[second_period_vals<-1]=np.nan
        second_period_vals[second_period_vals > 1] = np.nan
        second_period_vals = second_period_vals[~np.isnan(second_period_vals)]


       ## plot smoothed density

        first_period_vals_df = pd.DataFrame()
        second_period_vals_df = pd.DataFrame()
        first_period_vals_df['x'] = first_period_vals
        second_period_vals_df['x'] = second_period_vals
        ## plot smoothed density
        # kde_1983_2001 = gaussian_kde(first_period_vals)
        # kde_2002_2020 = gaussian_kde(second_period_vals)
        # x=np.linspace(-1,1,100)
        # plt.plot(x, kde_1983_2001(x), label="1983–2001", color='red', lw=2, alpha=0.8)
        # plt.plot(x, kde_2002_2020(x), label="2002–2020", color='green', lw=2, alpha=0.8)

        first_period_vals_df['weight'] = np.ones_like(first_period_vals) / len(first_period_vals)
        second_period_vals_df['weight'] = np.ones_like(second_period_vals) / len(second_period_vals)
        sns.kdeplot(data=first_period_vals_df, x='x', weights=first_period_vals_df.weight, shade=True, color='red', label='1983–2001')
        sns.kdeplot(data=second_period_vals_df, x='x', weights=second_period_vals_df.weight, shade=True, color='green', label='2002–2020')
        # plt.plot(x1, y1, color='red', label='1983–2001', linewidth=2)
        # plt.plot(x2, y2, color='green', label='2002–2020', linewidth=2)

        plt.xlabel(f'{variable} trend ( %/year)')
        plt.ylabel('Probability')
        plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(-1, 1)
        outdir = rf'E:\Project3\Result\3mm\relative_change_growing_season\two_period\\plot\\'
        T.mkdir(outdir)
        outf = join(outdir, f'{variable}_trend_2_periods_hist.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)

        plt.show()

        pass

    def stacked_bar_plot(self):
        dff = result_root + rf'3mm\Dataframe\Trend\\Trend.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        ##plt histogram of LAI

        T.print_head_n(df)
        period_list = [ '2002_2020','1983_2001', ]
        result_dict = {}
        trend_bins = [-20,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,20]
        color_list=[  'orange','red','brown','limegreen', 'green', 'deepskyblue', 'blue','navy']
        result_dict_browning = {}
        result_dict_greening = {}
        bin_labels = [(trend_bins[i], trend_bins[i + 1]) for i in range(len(trend_bins) - 1)]

        for period in period_list:
            df=df[df[f'LAI4g_{period}_p_value']<0.05]



            browning_vals = []
            greening_vals = []
            name_list = []
            df=df[df[f'LAI4g_{period}_trend'] > -30]
            df=df[df[f'LAI4g_{period}_trend'] < 30]

            for bin in bin_labels:
                name = bin
                df_group_i = df[df[f'LAI4g_{period}_trend']>bin[0]]
                df_group_i = df_group_i[df_group_i[f'LAI4g_{period}_trend']<=bin[1]]

                vals = len(df_group_i)/len(df)*100
                print(bin,vals)

                if bin[1] < 0:  # Browning bins (left of 0)
                    browning_vals.append(-vals)
                    greening_vals.append(0)
                else:  # Greening bins (right of 0)
                    browning_vals.append(0)
                    greening_vals.append(vals)
                # 使用正值表示 greening
                name_list.append(name)

            sorted_first_three = sorted(browning_vals[:3])
            browning_vals=sorted_first_three+browning_vals[3:]



            result_dict_browning[period] = browning_vals
            result_dict_greening[period] = greening_vals

    # Create new DataFrames for browning and greening

        df_browning = pd.DataFrame(result_dict_browning, index=name_list)
        df_greening = pd.DataFrame(result_dict_greening, index=name_list)

        # Transpose for horizontal bar plotting
        df_browning_T = df_browning.T
        df_greening_T = df_greening.T

        # Plot the stacked bar chart
        fig, ax = plt.subplots(figsize=(6, 4))

        df_browning_T.plot(kind='barh', stacked=True, width=0.3, ax=ax, color=color_list, legend=False,alpha=0.7)
        df_greening_T.plot(kind='barh', stacked=True, width=0.3, ax=ax, color=color_list, legend=False,alpha=0.7)

        # Add a vertical line at 0 to separate browning and greening
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
        plt.xlim(-5, 100)

        # Add labels and title
        plt.xlabel("Percentage")
        plt.ylabel("Period")


        plt.show()
    def statistic_analysis(self):  ##calculate spatial average trends for three decades
        dff=result_root+rf'3mm\Dataframe\Trend\\Trend.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        T.print_head_n(df)

        df_first_decade = df[df[f'LAI4g_2002_2020_trend']>-99]
        df_second_decade = df[df[f'LAI4g_1983_2001_trend']>-99]
        trend_first = df_first_decade[f'LAI4g_2002_2020_trend'].tolist()
        trend_second = df_second_decade[f'LAI4g_1983_2001_trend'].tolist()
        trend_first = np.array(trend_first)
        trend_second = np.array(trend_second)
        # trend_first = trend_first[~np.isnan(trend_first)]
        # trend_second = trend_second[~np.isnan(trend_second)]
        trend_first=np.nanmean(trend_first)
        trend_second=np.nanmean(trend_second)
        print(trend_first,trend_second)

        pass









class Plot_Robinson:
    def __init__(self):
        # plt.figure(figsize=(15.3 * centimeter_factor, 8.2 * centimeter_factor))
        self.map_width = 15.3 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass

    def robinson_template(self):
        '''
                :param fpath: tif file
                :param is_reproj: if True, reproject file from 4326 to Robinson
                :param res: resolution, meter
                '''
        color_list = [
             '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        # Blue represents high values, and red represents low values.
        plt.figure(figsize=(self.map_width, self.map_height))
        m = Basemap(projection='robin', lon_0=0, lat_0=90., resolution='c')

        m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        # for obj in meridict:
        #     line = meridict[obj][0][0]
        coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        polys = m.fillcontinents(color='#FFFFFF', lake_color='#EFEFEF', zorder=90)
    def plot_Robinson_significance_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5, s=20,
                                           c='k', marker='x',
                                           zorder=100, res=2):

        fpath_clip = fpath_p + 'clip.tif'
        fpath_spatial_dict = DIC_and_TIF(tif_template=fpath_p).spatial_tif_to_dic(fpath_p)
        D_clip = DIC_and_TIF(tif_template=fpath_p)
        D_clip_lon_lat_pix_dict = D_clip.spatial_tif_to_lon_lat_dic(temp_root)
        fpath_clip_spatial_dict_clipped = {}
        for pix in fpath_spatial_dict:
            lon, lat = D_clip_lon_lat_pix_dict[pix]
            fpath_clip_spatial_dict_clipped[pix] = fpath_spatial_dict[pix]
        DIC_and_TIF(tif_template=fpath_p).pix_dic_to_tif(fpath_clip_spatial_dict_clipped, fpath_clip)
        fpath_resample = fpath_clip + 'resample.tif'
        ToRaster().resample_reproj(fpath_clip, fpath_resample, res=res)
        fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        self.Robinson_reproj(fpath_resample, fpath_resample_ortho, res=res * 10000)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        # lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        # lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        # arr_reproj, originX_reproj, originY_reproj, pixelWidth_reproj, pixelHeight_reproj = ToRaster().raster2array(fpath_resample_ortho)
        # lon_list_reproj = np.arange(originX_reproj, originX_reproj + pixelWidth_reproj * arr_reproj.shape[1], pixelWidth_reproj)
        # lat_list_reproj = np.arange(originY_reproj, originY_reproj + pixelHeight_reproj * arr_reproj.shape[0], pixelHeight_reproj)
        # arr = m.transform_scalar(arr, lon_list, lat_list[::-1], len(lon_list_reproj), len(lat_list_reproj))
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr > sig_level] = np.nan
        # plt.figure()
        # plt.imshow(arr,interpolation='nearest',cmap='jet')
        # plt.show()
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        #
        os.remove(fpath_clip)
        os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)

        spatial_dict = D_resample.spatial_arr_to_dic(arr)
        lon_lat_pix_dict = D_resample.spatial_tif_to_lon_lat_dic(temp_root)
        # keys = spatial_dict.keys()

        lon_list = []
        lat_list = []
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            lon, lat = lon_lat_pix_dict[pix]
            lon_list.append(lon)
            lat_list.append(lat)
        lon_list = np.array(lon_list)
        lat_list = np.array(lat_list)
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        lon_list = lon_list + pixelWidth / 2
        lat_list = lat_list + pixelHeight / 2
        # print(lon_list)
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)

        return m


    def plot_Robinson(self, fpath, ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,
                      res=25000, is_discrete=False, colormap_n=11):
        '''
        :param fpath: tif file
        :param is_reproj: if True, reproject file from 4326 to Robinson
        :param res: resolution, meter
        ## trend color list
        '''
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        ### CV list

        # color_list = [
        #     '#008837',
        #     '#a6dba0',
        #     '#f7f7f7',
        #     '#c2a5cf',
        #     '#7b3294',
        # ]
        # std_list=[ '#e66101',
        #            '#fdb863',
        #            '#f7f7f7',
        #            '#b2abd2',
        #            '#5e3c99',
        #
        # ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        elif type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        # print(np.shape(arr))
        # plt.imshow(arr)
        # plt.show()
        if not is_reproj:
            arr_reproj, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            lon_list_reproj = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list_reproj = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        else:
            fpath_robinson = self.Robinson_reproj(fpath, fpath + '_robinson-reproj.tif', res=res)
            arr_reproj, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_robinson)
            lon_list_reproj = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list_reproj = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            # print(originX, originY, pixelWidth, pixelHeight)
            arr_reproj[arr_reproj<-9999] = np.nan
            # plt.imshow(arr_reproj,interpolation='nearest')
            # plt.show()
            os.remove(fpath_robinson)
            # print(fpath_robinson)
            # exit()
        # originY1 = copy.copy(originY)
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr_m = ma.masked_where(np.isnan(arr), arr)
        # originX = 0
        # originY = originY * 2
        # originY = 0

        # lon_list, lat_list = np.meshgrid(lon_list, lat_list)
        # print(lon_list.shape)
        # plt.imshow(arr_m)
        # plt.show()
        # exit()
        m = Basemap(projection='robin', lon_0=0, lat_0=90., ax=ax, resolution='c')
        # print(lon_list)
        # print(lat_list)
        # m = Basemap(projection='robin', lon_0=0,ax=ax, resolution='c')
        arr_m = m.transform_scalar(arr_m,lon_list,lat_list[::-1],len(lon_list_reproj)*1,len(lat_list_reproj)*1,order=0)
        # m.transform_vector()
        # plt.imshow(arr_m,interpolation='nearest')
        # plt.show()

        # ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax, )
        ret = m.imshow(arr_m[::-1], cmap=cmap, zorder=99, vmin=vmin, vmax=vmax,interpolation='nearest')
        # m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # plt.show()
        # meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        # meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        # for obj in meridict:
        #     line = meridict[obj][0][0]
        # coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        # polys = m.fillcontinents(color='#eeeeee', lake_color='#EFEFEF', zorder=90)
        # plt.show()
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0.05, shrink=0.5)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds,
                                                 orientation='horizontal')
            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret

    def Robinson_reproj(self, fpath, outf, res=50000):
        wkt = self.Robinson_wkt()
        srs = DIC_and_TIF().gen_srs_from_wkt(wkt)
        ToRaster().resample_reproj(fpath, outf, res, dstSRS=srs)
        return outf

    def Robinson_wkt(self):
        wkt = '''
        PROJCRS["Sphere_Robinson",
    BASEGEOGCRS["Unknown datum based upon the Authalic Sphere",
        DATUM["Not specified (based on Authalic Sphere)",
            ELLIPSOID["Sphere",6371000,0,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["Sphere_Robinson",
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
    ID["ESRI",53030]]'''
        return wkt
class bivariate_analysis():
    def __init__(self):
        pass
    def run(self):

        # self.xy_map()
        # self.plot_histogram()
        # self.plot_robinson()
        # self.generate_bivarite_map()

        self.generate_three_dimension_sensitivity_rainfall_greening() ##rainfall_trend +sensitivity+ greening
        # self.generate_three_dimension_growth_rate_greening_rainfall()
        # self.plot_three_dimension_pie()
        # self.plot_three_dimension_pie2()

        # self.plot_bar_greening_wetting() ## robust test between NDVI and LAI
        # self.plot_bar_greening_wetting2()
        # self.consistency_CV_LAI4g_NDVI()
        # self.CV_products_comparison_bivariate()
        # self.CV_products_comparison_bar()
        # self.scatterplot()
        # self.plot_figure1d()


        pass


    def xy_map(self): ##

        import xymap
        tif_sensitivity = result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\npy_time_series\trend\\sum_rainfall_trend.tif'
        tif_trends = result_root + rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\trend_ecosystem_year\\sum_rainfall_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\rainfall_sensitivity_trend.tif'
        T.mk_dir(result_root + rf'bivariate_analysis\\')
        tif1 = tif_trends
        tif2 = tif_sensitivity

        dic1=DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif1)
        dic2=DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif2)
        dics={'rainfall_trends':dic1,
        'rainfall_sensitivity':dic2}
        df=T.spatial_dics_to_df(dics)
        # print(df)
        df['is_rainfall_trend_positive'] = df['rainfall_trends'] > 0
        df['is_rainfall_sensitivity_positive'] = df['rainfall_sensitivity'] > 0
        print(df)
        label_list=[]
        for i ,row in df.iterrows():
            if row['is_rainfall_trend_positive'] and row['is_rainfall_sensitivity_positive']:
                label_list.append(1)
            elif row['is_rainfall_trend_positive'] and not row['is_rainfall_sensitivity_positive']:
                label_list.append(2)
            elif not row['is_rainfall_trend_positive'] and row['is_rainfall_sensitivity_positive']:
                label_list.append(3)
            else:
                label_list.append(4)

        df['label']=label_list
        result_dic=T.df_to_spatial_dic(df,'label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(result_dic,outtif)
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

    def plot_histogram(self):
        tiff=result_root + rf'3mm\bivariate_analysis\\rainfall_sensitivity_trend.tif'
        dic=DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tiff)

        positive_positive = 0
        positive_negative = 0
        negative_positive = 0
        negative_negative = 0
        flag=0

        for pix in dic:
            r,c=pix
            if r<60:
                continue
            vals=dic[pix]
            if np.isnan(vals):
                continue
            if vals==1:
                positive_positive+=1
            elif vals==2:
                positive_negative+=1
            elif vals==3:
                negative_positive+=1
            else:
                negative_negative+=1
            flag=flag+1
        plt.figure(figsize=(4,2.5))
        label=['positive_positive','positive_negative','negative_positive','negative_negative']
        positive_positive_percentage = positive_positive / flag *100
        positive_negative_percentage = positive_negative / flag*100
        negative_positive_percentage = negative_positive / flag *100
        negative_negative_percentage = negative_negative / flag*100
        plt.bar(label,[positive_positive_percentage,positive_negative_percentage,negative_positive_percentage,negative_negative_percentage],
                color=['r','y','g','b'],width=0.5)
        plt.ylabel('percentage')
        plt.show()

    def plot_robinson(self):

        fdir_trend = result_root+rf'\3mm\bivariate_analysis\\'
        temp_root = result_root+rf'\3mm\bivariate_analysis\\temp\\'
        outdir = result_root+rf'\3mm\bivariate_analysis\CV_products_comparison_bivariate\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):


            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]

            # fpath = fdir_trend + f
            # fpath = rf"E:\Project3\Result\3mm\bivariate_analysis\growth_rate_rainfall_greening.tif"
            fpath = rf"E:\Project3\Result\3mm\bivariate_analysis\CV_products_comparison_bivariate\LAI4g_NDVI.tif"
            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            # m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=8, is_discrete=True, colormap_n=8, cmap='RdYlBu_r',)

            color_list1 = [
                '#19449C',
                '#406BB3',
                '#6A3777',
                '#A769AC',
                '#157E3E',
                '#7DC466',
                '#98272F',
                '#F8B88D',
            ]

            color_list1 = [
                '#e7f4cb',
                '#b7dee3',
                '#75b1d3',
                '#2c7bb6',
                '#d7191c',
                '#ed6e43',
                '#fdba6e',
                '#fee8a4',
            ]
#### this is for consistency between NDVI and NDVI4g

            ## this is for consistency of CV between LAI4g and NDVI

            color_list3 = [
                '#a86ee1',
                '#d6cb38',
                '#4defef',
                '#de6e13',

            ]

            ## this is for dominant factor

            color_list4 = [
                '#7744ce',
                '#ff0105',
                '#0167ff',
                '#4def8e',
                '#01f7ff'

            ]


            my_cmap1 = T.cmap_blend(color_list1,n_colors=8)
            my_cmap2 = T.cmap_blend(color_list3,n_colors=5)
            # arr = ToRaster().raster2array(fpath)[0]
            # arr[arr<-999]=np.nan
            # plt.imshow(arr,cmap=my_cmap,vmin=1,vmax=8,interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=5, is_discrete=True, colormap_n=5, cmap=my_cmap2,)

            plt.title(f'{fname}')
            # plt.show()
            outf = outdir +'CV_products_LAI4g_NDVI.pdf'
            plt.savefig(outf)
            plt.close()
            exit()



    def generate_three_dimension_sensitivity_rainfall_greening(self):
        ##rainfall_trend +sensitivity+ greening
        dff=result_root+rf'\3mm\Dataframe\Trend\\Trend.df'

        df=T.load_df(dff)
        df = pd.DataFrame(df)
        # T.print_head_n(df)
        # exit()
        df=df[df['bivariate']>0]
        df=df[df['LAI4g_1983_2020_p_value']<0.05]


        # Create a new column 'label' with the values 'greening' or 'browning' based on the 'trend' column
        df["greening_label"] = df["LAI4g_1983_2020_trend"].apply(lambda x: "greening" if x >= 0 else "browning")

        category_mapping = {
            ("greening", 1): 1,
            ("greening", 2): 2,
            ("greening", 3): 3,
            ("greening", 4): 4,
            ("browning", 1): 5,
            ("browning", 2): 6,
            ("browning", 3): 7,
            ("browning", 4): 8,
        }

        # Apply the mapping to create a new column 'new_category'
        df["three_dimension"] = df.apply(lambda row: category_mapping[(row["greening_label"], row["bivariate"])], axis=1)
        T.save_df(df, result_root+rf'\3mm\Dataframe\Trend\\Trend.df')
        T.df_to_excel(df, result_root+rf'\3mm\Dataframe\Trend\\Trend.xlsx')

        # Display the result
        print(df)
        outdir=result_root+rf'\3mm\Dataframe\bivariate_analysis\\'
        outf=outdir+rf'\\three_dimension.tif'

        spatial_dic=T.df_to_spatial_dic(df, 'three_dimension')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(spatial_dic,outf)
        ##save pdf
        # outf = outdir + rf'\\three_dimension.pdf'
        # plt.savefig(outf)
        # plt.close()

    def generate_three_dimension_wetting_snesitivity_greening(self):
        ##rainfall_trend +sensitivity+ greening
        dff = result_root + rf'\3mm\Dataframe\Trend\\Trend.df'

        df = T.load_df(dff)
        df = pd.DataFrame(df)
        # T.print_head_n(df)
        # exit()
        df = df[df['bivariate'] > 0]
        df = df[df['LAI4g_1983_2020_p_value'] < 0.05]

        # Create a new column 'label' with the values 'greening' or 'browning' based on the 'trend' column
        df["greening_label"] = df["LAI4g_1983_2020_trend"].apply(lambda x: "greening" if x >= 0 else "browning")

        category_mapping = {
            ("greening", 1): 1,
            ("greening", 2): 2,
            ("greening", 3): 3,
            ("greening", 4): 4,
            ("browning", 1): 5,
            ("browning", 2): 6,
            ("browning", 3): 7,
            ("browning", 4): 8,
        }

        # Apply the mapping to create a new column 'new_category'
        df["three_dimension"] = df.apply(lambda row: category_mapping[(row["greening_label"], row["bivariate"])],
                                         axis=1)
        T.save_df(df, result_root + rf'\3mm\Dataframe\Trend\\Trend.df')
        T.df_to_excel(df, result_root + rf'\3mm\Dataframe\Trend\\Trend.xlsx')

        # Display the result
        print(df)
        outdir = result_root + rf'\3mm\Dataframe\bivariate_analysis\\'
        outf = outdir + rf'\\three_dimension.tif'

        spatial_dic = T.df_to_spatial_dic(df, 'three_dimension')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(spatial_dic, outf)
        ##save pdf
        # outf = outdir + rf'\\three_dimension.pdf'
        # plt.savefig(outf)
        # plt.close()

    def generate_bivarite_map(self):  ##

        import xymap
        tif_rainfall = result_root + rf'\D:\Project3\Result\3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend\detrended_sum_rainfall_CV_trend.tif'
        # tif_CV=  result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        tif_sensitivity= result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year_SNU_LAI\npy_time_series\\sum_rainfall_detrend_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\interannual_CVrainfall_beta.tif'
        T.mk_dir(result_root + rf'bivariate_analysis\\')
        tif1 = tif_rainfall
        tif2 = tif_sensitivity

        dic1 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif1)
        dic2 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif2)
        dics = {'interannual_CVrainfall': dic1,
                'beta': dic2}
        df = T.spatial_dics_to_df(dics)
        # print(df)
        df['interannual_CVrainfall_increase'] = df['interannual_CVrainfall'] > 0
        df['beta_increase'] = df['beta'] > 0

        print(df)
        label_list = []
        for i, row in df.iterrows():
            if row['interannual_CVrainfall_increase'] and row['beta_increase']:
                label_list.append(1)
            elif row['interannual_CVrainfall_increase'] and not row['beta_increase']:
                label_list.append(2)
            elif not row['interannual_CVrainfall_increase'] and row['beta_increase']:
                label_list.append(3)
            else:
                label_list.append(4)

        df['label'] = label_list
        result_dic = T.df_to_spatial_dic(df, 'label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(result_dic, outtif)

    pass

    def plot_three_dimension_pie(self):  ## plot greening rainfall sensitivity and rainfall trend

        dff=result_root+rf'\3mm\Dataframe\Trend\\Trend.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        # T.print_head_n(df);exit()


        # Classify greening and browning trends
        df=df[df['LAI4g_trend'] <30]
        df=df[df['LAI4g_trend'] >-30]
        print(len(df))

        df=df[df['LAI4g_p_value']<0.05]
        # T.print_head_n(df);exit()
        greening_counts = len(df[df['greening_label']=='greening'])
        browning_counts = len(df[df['greening_label']=='browning'])

        # Prepare data for pie chart with two layers
        grouped = df.groupby(["greening_label", "bivariate"]).size()  # Inner layer: greening and browning
        proportions = grouped.unstack(level=0).fillna(0)  # Fill missing groups with 0
        proportions["greening proportion"] = proportions['greening'] / greening_counts
        proportions["browning proportion"] = proportions['browning'] / browning_counts
        T.print_head_n(proportions)
        browning_size_list = proportions["browning"].tolist()
        greening_size_list = proportions["greening"].tolist()
        size_list = browning_size_list + greening_size_list
        browning_labels_list = proportions["browning proportion"].tolist()
        greening_labels_list = proportions["greening proportion"].tolist()
        labels_list = browning_labels_list + greening_labels_list
        labels_list = np.array(labels_list) * 100
        labels_list = np.round(labels_list, 1)
        outer_colors = ['#e7f4cb','#b7dee3', '#75b1d3', '#2c7bb6', '#d7191c', '#ed6e43','#fdba6e', '#fee8a4',   ][::-1]

        plt.pie(size_list, labels=labels_list,colors=outer_colors,radius=1.3,
                wedgeprops=dict(width=0.3, edgecolor='w'),)
        greening_total_size = proportions["greening"].sum()
        browning_total_size = proportions["browning"].sum()
        plt.pie([browning_total_size, greening_total_size],

                colors=['brown', 'lightgreen'],
                radius=1,
                wedgeprops=dict(width=.3, edgecolor='w'))
        plt.show()
        exit(99)
        # print(proportions.columns);exit()
        # labels_list = proportions['greening_label']
        # plt.pie(proportions["greening proportion"], labels=, autopct='%1.1f%%')
        # plt.show()
        # outer_colors = ['#A3E4D7', '#76D7C4', '#48C9B0', '#1ABC9C', '#F7DC6F', '#F4D03F', '#F1C40F', '#D4AC0D']

        # Inner pie chart data
        # inner_counts = [greening_counts, browning_counts]
        # inner_labels = ["Greening", "Browning"]
        # inner_colors = ['lightgreen', 'brown']

    def plot_three_dimension_pie2(self):  ## plot greening rainfall growthrate

        dff=result_root+rf'\3mm\Dataframe\Trend\\Trend.df'
        df=T.load_df(dff)
        # T.print_head_n(df);exit()
        outer_label_list = ['wetting_greening_accelerate','wetting_greening_slowdown','wetting_browning_accelerate','wetting_browning_slowdown',
        'drying_greening_accelerate','drying_greening_slowdown','drying_browning_accelerate','drying_browning_slowdown']

        # Classify greening and browning trends
        df=df[df['LAI4g_1983_2020_trend'] <30]
        df=df[df['LAI4g_1983_2020_trend'] >-30]
        print(len(df))

        df=df[df['LAI4g_1983_2020_p_value']<0.05]
        # T.print_head_n(df);exit()
        vals=df['growth_rate_rainfall_greening'].values
        vals=np.array(vals)
        # vals[vals<-99]=np.nan
        # vals[vals>99]=np.nan
        vals=vals[~np.isnan(vals)]

        dic_label = {1: 'wetting_greening_accelerate', 2: 'wetting_greening_slowdown', 3: 'wetting_browning_accelerate', 4: 'wetting_browning_slowdown',
        5: 'drying_greening_accelerate', 6: 'drying_greening_slowdown', 7: 'drying_browning_accelerate', 8: 'drying_browning_slowdown'}
        val_list=[]
        group_list=['wetting','drying']
        categories_list=['greening_accelerate','greening_slowdown','browning_accelerate','browning_slowdown']
        vals=df['growth_rate_rainfall_greening'].values


        for i in range(1,9):
            count=len(vals[vals==i])
            percent=count/len(vals)*100

            val_list.append(percent)
        ## plot bar
        plt.bar(outer_label_list, val_list)

        plt.ylabel('Percent (%)')
        plt.show()


        # for i in range(1,9):
        #     # dic_label[i]=dic_label[i]+'_'+str(len(vals[vals==i]))
        #     count = len(vals[vals==i])
        #     val_list.append(count)
        #
        # print(val_list)
        # # plt.pie(val_list,labels = outer_label_list,autopct='%1.1f%%')
        # plt.pie(val_list, labels=outer_label_list, autopct='%1.1f%%', startangle=90,
        #         wedgeprops=dict(width=0.4))  # width 控制环的宽度
        # plt.show()

    def plot_bar_greening_wetting(self):
        dff=result_root+rf'\3mm\Dataframe\wetting_greening_bivariable\\wetting_greening_bivariable.df'
        variable='NDVI'
        color_list3 = [
            '#2b83ba',
            '#d4c0dc',
            '#c1e4bd',
            '#f3bb2d',

        ]


        df=T.load_df(dff)
        df=self.df_clean(df)
        df=df[df[f'{variable}_p_value'] <0.05]
        vals = df[f'greening_wetting_{variable}'].values
        vals = np.array(vals)
        # vals[vals<-99]=np.nan
        # vals[vals>99]=np.nan
        vals = vals[~np.isnan(vals)]
        dic_label={1: 'wetting_greening', 2: 'wetting_browning', 3: 'drying_greening', 4: 'drying_browning'}

        val_list=[]
        fig=plt.figure(figsize=(3,2.5))

        for i in range(1,5):
            count=len(vals[vals==i])
            percent=count/len(vals)*100

            val_list.append(percent)
        x = np.array([0.2, 0.4, 0.6, 0.8])

        ## plot bar
        ## gap is 0.5 ## all bars gap is 0
        plt.bar(x, val_list, color=color_list3,width=0.1,align='center')

        plt.ylabel('Percent of Area (%)')
        plt.tight_layout()

        # plt.xticks(range(1, 5), dic_label.values())
        # plt.show()
        ## save
        outf=result_root+rf'\3mm\Dataframe\wetting_greening_bivariable\\bar_{variable}.pdf'
        plt.savefig(outf)
        T.open_path_and_file(outf)


        pass




    def CV_products_comparison_bivariate(self):  ## generate CV map

        import xymap
        tif_rainfall = result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        tif_greening= result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\detrended_GIMMS_plus_NDVI_CV_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\CV_products_comparison_bivariate\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\LAI4g_GIMMS_NDVI.tif'
        T.mk_dir(result_root + rf'bivariate_analysis\\')
        tif1 = tif_rainfall
        tif2 = tif_greening

        dic1 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif1)
        dic2 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif2)
        dics = {'LAI4g_CV_trend': dic1,
                'NDVI_CV_trend': dic2}
        df = T.spatial_dics_to_df(dics)
        ## filter r<60


        # print(df)
        df['is_increasing_LAI4g'] = df['LAI4g_CV_trend'] > 0
        df['is_increasing_NDVI'] = df['NDVI_CV_trend'] > 0
        print(df)
        label_list = []
        for i, row in df.iterrows():
            pix=row['pix']


            if row['is_increasing_LAI4g'] and row['is_increasing_NDVI']:
                label_list.append(1)
            elif row['is_increasing_LAI4g'] and not row['is_increasing_NDVI']:
                label_list.append(2)
            elif not row['is_increasing_LAI4g'] and row['is_increasing_NDVI']:
                label_list.append(3)
            else:
                label_list.append(4)

        df['label'] = label_list
        result_dic = T.df_to_spatial_dic(df, 'label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(result_dic, outtif)

    def CV_products_comparison_bar(self):
        dff=result_root+rf'\3mm\Dataframe\CV_products_comparison\\CV_products_comparison.df'
        df=T.load_df(dff)
        T.print_head_n(df)
        df=self.df_clean(df)
        df.dropna()
        dic_label={1:'Inc_Inc',2:'Inc_Dec',3:'Dec_Inc',4:'Dec_Dec'}
        color_list3 = [
            '#a86ee1',
            '#d6cb38',
            '#4defef',
            '#de6e13',

        ]


        variables = [ 'LAI4g_NDVI','LAI4g_GIMMS_NDVI', 'LAI4g_NDVI4g',]
        for variable in variables:
            val_list = []
            vals = df[variable].values
            for i in range(1, 5):
                count = len(vals[vals == i])
                percent = count / len(vals) * 100

                val_list.append(percent)
            x = np.array([0.2, 0.4, 0.6, 0.8])

            print(val_list)

            ## plot bar
            ## gap is 0.5 ## all bars gap is 0
            fig = plt.figure(figsize=(3, 2.5))
            plt.bar(x, val_list, color=color_list3, width=0.1, align='center')

            plt.ylabel('Percent of Area (%)')
            plt.tight_layout()

            # plt.xticks(range(0, 1), dic_label.values())
            # plt.show()
            ## save
            outdir = result_root + rf'\3mm\bivariate_analysis\CV_products_comparison_bivariate\\barplot\\'

            T.mk_dir(outdir)
            outf =outdir+rf'{variable}.pdf'
            plt.savefig(outf)
            plt.close()

        T.open_path_and_file(outdir)

        pass

    def scatterplot(self):
        # dff=rf'D:\Project3\Result\3mm\Dataframe\Trend\\Trend.df'
        # df=T.load_df(dff)
        # T.print_head_n(df);exit()
        # df=self.df_clean(df)
        trend_tif = r"D:\Project3\Result\3mm\relative_change_growing_season\TRENDY\trend_analysis_simple_linear_0206\LAI4g_trend.tif"
        cv_tif = r"D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\LAI4g_detrend_CV_trend.tif"

        spatial_trend_dict = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(trend_tif)
        spatial_cv_dict = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(cv_tif)
        df = T.spatial_dics_to_df({'LAI4g_trend': spatial_trend_dict, 'LAI4g_detrend_CV_trend': spatial_cv_dict})


        # df=df[df['LAI4g_p_value']<0.05]
        # df.dropna()
        greening_trend=df['LAI4g_trend'].values

        CV_trend=df['LAI4g_detrend_CV_trend'].values
        KDE_plot().plot_scatter(greening_trend,CV_trend)

        # plt.scatter(greening_trend,CV_trend)
        plt.xlabel('Greening Trend')
        plt.ylabel('CV Trend')
        # plt.ylim(-0.5,1.5)
        # plt.xlim(-0.5,1.5)
        plt.show()
        pass

    def plot_figure1d(self):
        tiff_f=rf'D:\Project3\Result\3mm\bivariate_analysis\three_dimension.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff_f)
        dic_label={1:'greening_wetting_posbeta',2:'greening_wetting_negbeta',3:'greening_drying_posbeta',4:'greening_drying_negbeta',
                   5:'browning_wetting_posbeta',6:'browning_wetting_negbeta',7:'browning_drying_posbeta',8:'browning_drying_negbeta'}
        dic={}



        greening_list=[]
        browning_list=[]
        for i in range(1,9):
            if i<5:
                array_i=array==i
                count = np.sum(array_i)
                greening_list.append(count)
            else:
                array_i = array == i
                count = np.sum(array_i)
                browning_list.append(count)
        greening_sum = np.sum(greening_list)
        browning_sum = np.sum(browning_list)

        ['#e7f4cb','#b7dee3', '#75b1d3', '#2c7bb6', '#d7191c', '#ed6e43','#fdba6e', '#fee8a4',]

        color_list2 = [
             '#75b1d3',

            '#e7f4cb',



            '#fdba6e',
            '#d7191c',

        ]

        color_list1 = [
            '#2c7bb6',


            '#b7dee3',





            '#fee8a4',
            '#ed6e43',
        ]

            ## count the number of pixels
        for i in range(1,9):

            if i<5:
                array_i = array == i
                count = np.sum(array_i)
                dic[i] = count/greening_sum*100
            else:
                array_i = array == i
                count = np.sum(array_i)
                dic[i] = count/browning_sum*100
        pprint(dic)


        dic[5]=-dic[5]
        dic[6]=-dic[6]
        dic[7]=-dic[7]
        dic[8]=-dic[8]

        group_labels = ['Wetting', 'Drying', 'Wetting', 'Drying']
        group_data = [
            [dic[1], dic[2]],
            [dic[3], dic[4]],
            [dic[5], dic[6]],
            [dic[7], dic[8]],
        ]

        # 拆分上下部分


        bottoms = [0 if v[0] >= 0 else v[0] for v in group_data]
        heights1 = [v[0] for v in group_data]
        heights2 = [v[1] for v in group_data]

        x = np.arange(len(group_labels))
        bar_width = 0.5

        fig, ax = plt.subplots(figsize=(12 * centimeter_factor,  8 * centimeter_factor))

        # 第一层

        bars1 = ax.bar(x, heights1, width=bar_width, color=color_list1)

        # 第二层堆叠
        bars2 = ax.bar(x, heights2, width=bar_width, bottom=heights1, color=color_list2)

        # 坐标轴 & 图例
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('Percentage (%)')

        ax.axhline(0, color='black', linewidth=0.8)
        plt.show()


        # plt.tight_layout()
        # plt.savefig(rf'D:\Project3\Result\3mm\FIGURE\three_dimension_statistics.pdf')






class multi_regression_temporal_patterns():  ## this is for address beta
    def __init__(self):

        self.fdirX = rf'D:\Project3\Result\3mm\Multiregression_temporal_patterns\\'
        self.fdirY = rf'D:\Project3\Result\3mm\Multiregression_temporal_patterns\\'

        self.y_var = ['LAI4g_detrend_CV']
        self.xvar = [ 'rainfall_intensity', 'rainfall_seasonality_all_year', 'detrended_sum_rainfall_CV', 'heat_event_frenquency','rainfall_frenquency','FVC',]

        self.multi_regression_result_dir = rf'D:\Project3\Result\3mm\Multiregression_temporal_patterns\\'
        T.mk_dir(self.multi_regression_result_dir, force=True)

        self.multi_regression_result_f = self.multi_regression_result_dir + 'multi_regression_result.npy'

        pass

    def run(self):

        # step 1 build dataframe

        # df=self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var)

        # # # step 2 cal correlation
        # self.cal_multi_regression_beta()

        # step 3 plot
        self.plt_multi_regression_result(self.multi_regression_result_f,self.y_var[0])

        ## step 4 convert m2/m2/ppm to %/100ppm
        # self.convert_CO2_sensitivity_unit()

        # step 5
        # self.calculate_trend_contribution()

        pass

    def build_df(self, fdir_X, fdir_Y, fx_list, fy):

        df = pd.DataFrame()

        filey = fdir_Y + fy[0] + '.npy'
        print(filey)

        dic_y = T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            if yvals[0] == None:
                continue

            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        for xvar in fx_list:

            # print(var_name)
            x_val_list = []
            filex = fdir_X + xvar + '.npy'
            # filex = fdir_X + xvar + f'_{period}.npy'

            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix]
                xvals = np.array(xvals)
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue

                xvals = T.interp_nan(xvals)
                if xvals[0] == None:
                    x_val_list.append([])
                    continue

                x_val_list.append(xvals)

            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list
        T.print_head_n(df)
        ## save df
        T.save_df(df, self.multi_regression_result_dir + fy[0] + '.df')
        T.df_to_excel(df, self.multi_regression_result_dir + fy[0] + '.xlsx')

        return df

    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r

    def cal_multi_regression_beta(self):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd
        import joblib

        df = T.load_df(self.multi_regression_result_dir + 'LAI4g_detrend_CV.df')

        x_var_list = self.xvar

        outf = self.multi_regression_result_f

        multi_derivative = {}

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # print(row);exit()
            pix = row.pix

            y_vals = row['y']
            # y_vals = T.remove_np_nan(y_vals)
            # y_vals = T.interp_nan(y_vals)
            if len(y_vals) == 0:
                continue

            # y_vals_detrend = signal.detrend(y_vals)
            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:

                x_vals = row[x]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue

                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue

                df_new[x] = x_vals

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            if len(x_var_list_valid) < 2:
                continue
            # T.print_head_n(df_new)

            df_new['y'] = y_vals  # nodetrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1, how='all')

            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)


            df_new = df_new.dropna()
            ## build multiregression model and consider interactioon

            linear_model = LinearRegression()
            # print(df_new['y'])

            linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])

            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix] = coef_dic
        T.save_npy(multi_derivative, outf)

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir, y_var):

        NDVI_mask_f = rf'D:/Project3/Data/Base_data/aridity_index_05\\dryland_mask05.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = rf'D:/Project3/Data//Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f =  rf'D:/Project3/Data//Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f = self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:


            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            # print(var_i)
            spatial_dic = {}
            for pix in dic:
                r, c = pix
                if r < 60:
                    continue

                landcover_value = crop_mask[pix]

                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            print(var_i)
            # plt.imshow(arr)
            # plt.colorbar()
            # plt.show()
            outf = self.multi_regression_result_dir + rf'\\{var_i}.tif'

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, outf)




    def convert_CO2_sensitivity_unit(self):
        period_list = ['1982_2020']
        for period in period_list:
            CO2_sensitivity_f = result_root + rf'multi_regression\\anomaly\\{period}\\CO2_LAI4g_{period}.tif'
            average_LAI4g_f = result_root + rf'\state_variables\\\\LAI4g_{period}.npy'
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(CO2_sensitivity_f)
            arr[arr < -99] = np.nan
            dic_CO2_sensitivity = DIC_and_TIF().spatial_arr_to_dic(arr)

            dic_LAI4g_average = T.load_npy(average_LAI4g_f)

            for pix in dic_CO2_sensitivity:
                CO2_sensitivity = dic_CO2_sensitivity[pix]
                CO2_sensitivity = np.array(CO2_sensitivity, dtype=float)
                if np.isnan(CO2_sensitivity):
                    continue
                if not pix in dic_LAI4g_average:
                    continue
                LAI_average = dic_LAI4g_average[pix]
                LAI_average = np.array(LAI_average, dtype=float)

                if np.isnan(LAI_average):
                    continue
                CO2_sensitivity = CO2_sensitivity / LAI_average * 100
                if CO2_sensitivity < -99999:
                    continue
                dic_CO2_sensitivity[pix] = CO2_sensitivity
            arr_new = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic_CO2_sensitivity)
            arr_new[arr_new < -99] = np.nan
            arr_new[arr_new > 99] = np.nan

            # plt.imshow(arr_new)
            # plt.colorbar()
            # plt.show()

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_new, f'{CO2_sensitivity_f.replace(".tif", "_scale.tif")}')
            # DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_CO2_sensitivity, f'{CO2_sensitivity_f.replace(".tif","_new.tif")}')
            # T.save_npy(dic_CO2_sensitivity, CO2_sensitivity_f.replace('.tif', '.npy'))

    def calculate_trend_contribution(self):
        ## here I would like to calculate the trend contribution of each variable
        ## the trend contribution is defined as the slope of the linear regression between the variable and the target variable mutiplied by trends of the variable
        ## load the trend of each variable
        ## load the trend of the target variable
        ## load multi regression result
        ## calculate the trend contribution
        trend_dir = result_root + rf'\trend_analysis\anomaly\OBS_extend\\'

        selected_vairables_list = [
            'CRU_trend',
            'CO2_trend',
            'tmax_trend',
            'VPD_trend',
        ]

        trend_dict = {}
        for variable in selected_vairables_list:
            fpath = join(trend_dir, f'{variable}.npy')
            array = np.load(fpath, allow_pickle=True)
            array[array < -9999] = np.nan
            spatial_dict = D.spatial_arr_to_dic(array)
            for pix in tqdm(spatial_dict, desc=variable):
                r, c = pix
                if r < 120:
                    continue
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue
                if not pix in trend_dict:
                    trend_dict[pix] = {}
                key = variable.replace('_trend', '')
                trend_dict[pix][key] = spatial_dict[pix]

        f = self.multi_regression_result_f
        print(f)
        print(isfile(f))
        # exit()
        dic_multiregression = T.load_npy(f)
        var_list = []
        for pix in dic_multiregression:

            # landcover_value = crop_mask[pix]
            # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
            #     continue

            vals = dic_multiregression[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        # print(var_list)
        # exit()
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic_multiregression:
                if not pix in trend_dict:
                    continue

                dic_i = dic_multiregression[pix]
                if not var_i in dic_i:
                    continue
                val_multireg = dic_i[var_i]
                val_trend = trend_dict[pix][var_i]
                val_contrib = val_multireg * val_trend
                spatial_dic[pix] = val_contrib
            arr_contrib = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr_contrib, cmap='RdBu', interpolation='nearest')
            plt.colorbar()
            plt.title(var_i)
            plt.show()
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_contrib,
                                                   f'{self.multi_regression_result_dir}\\{var_i}_trend_contribution.tif')

class multi_regression():  ###linaer regression for CO2 effects.
    def __init__(self):
        self.this_root = 'E:\Project3\\'
        self.data_root = 'E:/Project3/Data/'
        self.result_root = 'E:/Project3/Result/'

        self.fdirX=self.result_root+rf'E:\Project3\Result\3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\'
        self.fdir_Y=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_trend_growing_season\\'

        self.xvar_list = ['CO2','detrended_sum_rainfall','Tmax',]
        self.y_var = ['detrended_growing_season_LAI_mean_CV']
        pass

    def run(self):

        outdir = self.result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\\'
        T.mk_dir(outdir, force=True)

        # # ####step 1 build dataframe


        df= self.build_df(self.fdirX, self.fdir_Y, self.xvar_list, self.y_var)

        self.cal_multi_regression_beta(df,self.xvar_list)  # 修改参数
        # ###step 2 crate individial files
        self.plt_multi_regression_result(outdir,self.y_var)
#



    def build_df(self, fdir_X, fdir_Y, xvar_list,y_var):

        df = pd.DataFrame()
        dic_y=T.load_npy(fdir_Y+y_var[0]+'.npy')
        pix_list = []
        y_val_list=[]

        for pix in dic_y:
            r,c= pix


            if len(dic_y[pix]) == 0:
                continue
            vals = dic_y[pix]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals = np.array(vals,dtype=float)


            vals[vals>999.0] = np.nan
            vals[vals<-999.0] = np.nan

            pix_list.append(pix)
            y_val_list.append(vals)

        df['pix'] = pix_list
        df['y'] = y_val_list

        ##df histogram
        # build x

        for xvar in xvar_list:


            x_val_list = []
            x_arr = T.load_npy(fdir_X+xvar+'.npy')
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                # print(len(x_arr[pix]))
                if len(x_arr[pix]) < self.window:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix]
                vals = np.array(vals)
                vals = np.array(vals, dtype=float)
                vals[vals > 999] = np.nan
                vals[vals < -999] = np.nan
                if len(vals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)

            df[xvar] = x_val_list
            df['CO2_precip'] = df['sum_rainfall']*df['CO2']




        return df



    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r


    def cal_multi_regression_beta(self, df, x_var_list):

        multi_derivative = {}
        multi_pvalue = {}


        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            r,c=pix

            y_vals = row['y']
            y_vals[y_vals<-999]=np.nan
            y_vals = T.remove_np_nan(y_vals)
            if len(y_vals) == 0:
                continue
            y_vals = np.array(y_vals)
            # y_vals_detrend=signal.detrend(y_vals)


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals = T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0] == None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals  # nodetrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1, how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            linear_model = LinearRegression()
            # print(df_new['y'])

            linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])
            # coef_ = np.array(linear_model.coef_) / y_mean
            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            ## pvalue
            X=df_new[x_var_list_valid_new]
            Y=df_new['y']
            try:
                sse = np.sum((linear_model.predict(X) -Y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])

                se = np.array([
                    np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                    for i in range(sse.shape[0])
                ])

                t = coef_ / se
                p = 2 * (1 - stats.t.cdf(np.abs(t), Y.shape[0] - X.shape[1]))
            except:
                p=np.nan

            multi_derivative[pix] = coef_dic
            multi_pvalue[pix] = p

        T.save_npy(multi_derivative, outf)
        T.save_npy(multi_pvalue, outf.replace('.npy', '_pvalue.npy'))

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var):
        fdir = multi_regression_result_dir
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if 'pvalue' in f:
                continue
            print(f)

            w=f.split('\\')[-1].split('.')[0][-2:]


            w=int(w)

            dic = T.load_npy(fdir+f)
            var_list = []
            for pix in dic:
                # print(pix)
                vals = dic[pix]
                for var_i in vals:
                    var_list.append(var_i)
            var_list = list(set(var_list))
            for var_i in var_list:
                spatial_dic = {}
                for pix in dic:
                    dic_i = dic[pix]
                    if not var_i in dic_i:
                        continue
                    val = dic_i[var_i]
                    spatial_dic[pix] = val
                arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
                outdir=fdir+'TIFF\\'
                T.mk_dir(outdir)
                outf=outdir+f.replace('.npy','')

                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, outf + f'_{var_i}.tif')
                std = np.nanstd(arr)
                mean = np.nanmean(arr)
                vmin = mean - std
                vmax = mean + std
                # plt.figure()
                # arr[arr > 0.1] = 1
                # plt.imshow(arr,vmin=-0.5,vmax=0.5)
                #
                # plt.title(var_i)
                # plt.colorbar()

            # plt.show()



    pass


    def df_clean(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        df=df[df['LC_max']<10]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df
class TRENDY_trend:

    def run(self):
        # self.unzip()
        # self.nc_to_tif_SNU_NIRV_NDVI()
        # self.scale_GOSIF()
        # self.resample()
        # self.unify_TIFF()
        # self.extract_dryland_tiff()
        # self.tif_to_dic()
        # self.extract_phenology_year()
        # self.extract_phenology_LAI_mean()
        # self.relative_change()
        # self.TRENDY_ensemble()
        # self.weighted_average_LAI()
        # self.trend_analysis()
        # self.plt_basemap()
        # self.plot_bar_trend()

        pass

    def unzip(self):
        import gzip
        dic={'M01':'01','M02':'02','M03':'03',
             'M04':'04','M05':'05','M06':'06',
             'M07':'07','M08':'08','M09':'09',
             'M10':'10','M11':'11','M12':'12'}



        fdir = data_root+rf'\GOSIF\monthly\\'
        outdir = data_root+rf'\GOSIF\unzip\\'
        T.mk_dir(outdir)
        for f in T.listdir(fdir):


            f_in = join(fdir,f)
            year = f.split('_')[1][0:4]
            month=f.split('.')[1]
            month_conv = dic[month]
            foutname = f'{year}{month_conv}.tif'
            # print(foutname)
            # exit()
            f_out = join(outdir,foutname)

            ## ungz

            with gzip.open(f_in, 'rb') as f_in:
                with open(f_out, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


    def nc_to_tif_time_series(self):


        fdir=data_root+rf'NDVI_SNU\SNU_NDVI_v1-20250101T033602Z-001\SNU_NDVI_v1\nc\\'
        outdir=data_root+rf'NDVI_SNU\SNU_NDVI_v1-20250101T033602Z-001\SNU_NDVI_v1\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        for f in os.listdir(fdir):

            outdir_name = f.split('.')[0]
            print(outdir_name)

            yearlist = list(range(1982, 2021))


            try:
                self.nc_to_tif_template(fdir+f, var_name='NDVI', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue

    def scale_GOSIF(self):
        fdir = rf'E:\Project3\Data\GOSIF\unzip\\'
        outdir = rf'E:\Project3\Data\GOSIF\\scale\\'
        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)
            array[array == 32767] = np.nan
            array[array == 32766] = np.nan
            array = array * 0.0001

            array[array < 0] = np.nan


            outf = outdir + f
            ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, array)

    def nc_to_tif_template(self, fname, var_name, outdir, yearlist):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())
            # exit()
            time=ncin.variables['time'][:]

        except:
            raise UserWarning('File not supported: ' + fname)
        # lon,lat = np.nan,np.nan
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        print(basetime_unit)
        print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            try:
                                basetime_ = basetime.split('T')[0]
                                # print(basetime_)
                                basetime = datetime.datetime.strptime(basetime_, '%Y-%m-%d')
                                # print(basetime)
                            except:

                                raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            if year not in yearlist:
                continue
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = []
            lat_list = []
            value_list = []
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    lon_i = xx[i][j]
                    if lon_i > 180:
                        lon_i -= 360
                    lat_i = yy[i][j]
                    value_i = arr[i][j]
                    lon_list.append(lon_i)
                    lat_list.append(lat_i)
                    value_list.append(value_i)
            DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list, outpath)

    def nc_to_tif_SNU_NIRV_NDVI(self):  ##

        fdir = data_root + rf'NDVI_SNU\SNU_NDVI_v1\nc\\'
        outdir = data_root + rf'NDVI_SNU\SNU_NDVI_v1\TIFF\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.endswith('.nc'):
                continue

            outf=outdir+f.split('.')[0]+'.tif'
            if os.path.isfile(outf):
                continue

            nc = Dataset(fdir + f, 'r')

            print(nc)
            print(nc.variables.keys())

            # lat_list = nc['x']
            # lon_list = nc['y']
            # # lat_list=lat_list[::-1]  #取反
            # print(lat_list[:])
            # print(lon_list[:])
            #
            origin_x = -180  # 要为负数180
            origin_y = 90
            pix_width = 0.05 ##  分辨率
            pix_height = -0.05


            SPEI_arr_list = nc['NDVI']
            print(SPEI_arr_list.shape)
            print(SPEI_arr_list[0])
            # plt.imshow(SPEI_arr_list[5])
            # plt.imshow(SPEI_arr_list[::])
            # plt.show()

            year=int(f.split('_')[3][0:4])
            month=int(f.split('_')[3][4:6])

            fname = '{}{:02d}.tif'.format(year, month)
            print(fname)
            newRasterfn = outdir + fname
            print(newRasterfn)

            # array = val
            # array=SPEI_arr_list[::-1]
            array = SPEI_arr_list

            array = np.array(array)
            # method 2
            array = array.T
            # array=array* 0.001  ### GPP need scale factor
            array[array < 0] = np.nan

            # plt.imshow(array,vmin=0,vmax=1)
            # plt.colorbar()
            # plt.show()

            ToRaster().array2raster(newRasterfn, origin_x, origin_y, pix_width, pix_height, array)

    def resample(self):
        fdir_all = data_root + rf'TRENDY\\S2\\TIFF\\'
        for fdir in tqdm(os.listdir(fdir_all)):
            if not 'GOSIF' in fdir:
                continue


            outdir = data_root+rf'TRENDY\S2\\resample\\{fdir}\\'
            if os.path.isdir(outdir):
                continue

            T.mk_dir(outdir, force=True)
            year = list(range(1982, 2021))
            # print(year)
            # exit()
            for f in tqdm(os.listdir(fdir_all + fdir + '\\'), ):
                if not f.endswith('.tif'):
                    continue

                if f.startswith('._'):
                    continue


                print(f)
                # exit()
                date = f.split('.')[0]
                date_2 = date.split('_')[-1]
                print(date_2)

                # print(date)
                # exit()
                dataset = gdal.Open(fdir_all + fdir + '\\' + f)
                # print(dataset.GetGeoTransform())
                original_x = dataset.GetGeoTransform()[1]
                original_y = dataset.GetGeoTransform()[5]

                # band = dataset.GetRasterBand(1)
                # newRows = dataset.YSize * 2
                # newCols = dataset.XSize * 2
                try:
                    gdal.Warp(outdir + '{}.tif'.format(date_2), dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass



    def unify_TIFF(self):
        fdir_all=data_root+rf'TRENDY\\\S2\resample\\'
        for fdir in os.listdir(fdir_all):
            if not 'GOSIF' in fdir:
                continue

            outdir = data_root + rf'TRENDY\\\S2\\TIFF_unify\\{fdir}\\'
            Tools().mk_dir(outdir, force=True)

            for f in os.listdir(join(fdir_all,fdir)):
                fpath=join(fdir_all,fdir,f)

                outpath=join(outdir,f)
                print(outpath)

                if not f.endswith('.tif'):
                    continue
                if f.startswith('._'):
                    continue
                unify_tiff=DIC_and_TIF().unify_raster(fpath,outpath)
    def extract_dryland_tiff(self):
        self.datadir=data_root
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        fdir_all =data_root + rf'TRENDY\S2\TIFF_unify\\'

        for fdir in T.listdir(fdir_all):
            outdir=data_root+rf'TRENDY\S2\dryland\\{fdir}\\'
            T.mk_dir(outdir, force=True)
            if not 'GOSIF' in fdir:
                continue


            fdir_i = join(fdir_all, fdir)


            for fi in tqdm(T.listdir(fdir_i), desc=fdir):
                if not fi.endswith('.tif'):
                    continue
                fpath = join(fdir_i, fi)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)

                arr = arr[:360][:360,
                              :720]


                arr[arr<0]=np.nan
                arr[arr>7]=np.nan
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                outpath = join(outdir, fi)
                if os.path.exists(outpath):
                    continue

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)

        pass

    def tif_to_dic(self):

        fdir_all =data_root+rf'TRENDY\S2\dryland\\'
        year_list = list(range(1982, 2021))


        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'DLEM_S2_lai' in fdir:
                continue
            outdir = data_root+rf'TRENDY\S2\\dic\\{fdir}\\'
            T.mk_dir(outdir, force=True)
            all_array = []
            for f in os.listdir(join(fdir_all,fdir)):

                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:
                    continue


                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_all, fdir, f))
                array = np.array(array, dtype=float)


                # array_unify = array[:720][:720,
                #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                array_unify = array[:360][:360,
                              :720]
                # array_unify[array_unify < -999] = np.nan
                # array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan

                array_unify[array_unify < 0] = np.nan

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
                    np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def extract_phenology_year(self):
        fdir_all = data_root+rf'TRENDY\S2\dic\\'
        for fdir in T.listdir(fdir_all):
            if not 'GOSIF' in fdir:
                continue

            outfdir = data_root+rf'TRENDY\S2\extract_phenology_year\\{fdir}\\'
            T.mk_dir(outfdir, force=True)


            f_phenology = rf'E:\Project3\Data\LAI4g\4GST\\4GST.npy'
            phenology_dic = T.load_npy(f_phenology)
            for f in T.listdir(fdir_all+fdir):

                outf = join(outfdir, f)
                #
                # if os.path.isfile(outf):
                #     continue
                # print(outf)
                spatial_dict = dict(np.load(join(fdir_all, fdir, f), allow_pickle=True, encoding='latin1').item())
                dic_DOY={15: 0,
                         30: 0,
                         45: 1,
                         60: 1,
                         75: 2,
                         90: 2,
                         105: 3,
                         120: 3,
                         135: 4,
                         150: 4,
                         165: 5,
                         180: 5,
                         195: 6,
                         210: 6,
                         225: 7,
                         240: 7,
                         255: 8,
                         270: 8,
                         285: 9,
                         300: 9,
                         315: 10,
                         330: 10,
                         345: 11,
                         360: 11,}


                result_dic = {}
                for pix in tqdm(spatial_dict):
                    if not pix in phenology_dic:
                        continue

                    r, c = pix

                    SeasType=phenology_dic[pix]['SeasType']
                    if SeasType==2:

                        SOS=phenology_dic[pix]['Onsets']
                        try:
                            SOS=float(SOS)

                        except:
                            continue

                        SOS=int(SOS)
                        SOS_biweekly=dic_DOY[SOS]

                        EOS=phenology_dic[pix]['Offsets']
                        EOS=int(EOS)
                        EOS_biweekly=dic_DOY[EOS]


                        time_series = spatial_dict[pix]
                        time_series = np.array(time_series)
                        time_series_flatten = time_series.flatten()
                        time_series_flatten_extraction = time_series_flatten[(EOS_biweekly + 1):-(12 - EOS_biweekly - 1)]
                        time_series_flatten_extraction_reshape = time_series_flatten_extraction.reshape(-1, 12)
                        non_growing_season_list = []
                        growing_season_list = []
                        for vals in time_series_flatten_extraction_reshape:
                            if T.is_all_nan(vals):
                                continue
                            ## non-growing season +growing season is 365

                            non_growing_season = vals[0:SOS_biweekly]
                            growing_season = vals[SOS_biweekly:]
                            # print(growing_season)
                            non_growing_season_list.append(non_growing_season)
                            growing_season_list.append(growing_season)
                        # print(len(growing_season_list))
                        non_growing_season_list = np.array(non_growing_season_list)
                        growing_season_list = np.array(growing_season_list)


                    elif SeasType==3:
                        # SeasClass=phenology_dic[pix]['SeasClss']
                        # ## whole year is growing season
                        # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
                        # print(lat,lon)
                        # print(SeasType)
                        # print(SeasClass)
                        time_series = spatial_dict[pix]
                        time_series = np.array(time_series)
                        time_series_flatten = time_series.flatten()
                        time_series_flatten_extraction = time_series_flatten[12:]
                        time_series_flatten_extraction_reshape = time_series_flatten_extraction.reshape(-1, 12)
                        non_growing_season_list = []
                        growing_season_list = time_series_flatten_extraction_reshape

                    else:
                        SeasClss=phenology_dic[pix]['SeasClss']
                        print(SeasType,SeasClss)
                        continue

                    result_dic[pix]={'SeasType':SeasType,
                        'non_growing_season':non_growing_season_list,
                                  'growing_season':growing_season_list,
                                  'ecosystem_year':time_series_flatten_extraction_reshape}

                np.save(outf, result_dic)

    def extract_phenology_LAI_mean(self):  ## extract LAI average
        fdir_all = data_root + rf'TRENDY\S2\extract_phenology_year\\'
        for fdir in T.listdir(fdir_all):
            outdir = data_root+rf'TRENDY\S2\extract_phenology_LAI_mean\\'

            T.mk_dir(outdir, force=True)

            outf =data_root+rf'TRENDY\S2\extract_phenology_LAI_mean\\{fdir}.npy'

            spatial_dic = T.load_npy_dir(fdir_all+fdir)
            result_dic = {}

            for pix in tqdm(spatial_dic):
                ### ui==if northern hemisphere
                r, c = pix

                ### annual year

                vals = spatial_dic[pix]['ecosystem_year']
                vals_growing_season = spatial_dic[pix]['growing_season']
                vals_non_growing_season = spatial_dic[pix]['non_growing_season']

                ecosystem_mean_list = []
                growing_season_mean_list = []
                non_growing_season_mean_list = []

                for val in vals:
                    if T.is_all_nan(val):
                        continue
                    val = np.array(val)
                    sum_annaul = np.nanmean(val)
                    ecosystem_mean_list.append(sum_annaul)

                for val in vals_growing_season:
                    if T.is_all_nan(val):
                        continue
                    val = np.array(val)

                    sum_growing_season = np.nanmean(val)
                    growing_season_mean_list.append(sum_growing_season)

                for val in vals_non_growing_season:
                    if T.is_all_nan(val):
                        continue
                    val = np.array(val)

                    sum_non_growing_season = np.nanmean(val)
                    non_growing_season_mean_list.append(sum_non_growing_season)

                result_dic[pix] = {'ecosystem_year': ecosystem_mean_list,
                                   'growing_season': growing_season_mean_list,
                                   'non_growing_season': non_growing_season_mean_list}



            np.save(outf, result_dic)


    def relative_change(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = data_root + rf'TRENDY\S2\extract_phenology_LAI_mean\\'
        outdir=result_root+rf'3mm\relative_change_growing_season\\'

        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.split('.')[0] in ['NIRv','NDVI']:
                continue
            outf = outdir + f.split('.')[0] + f'_2002_2020.npy'

            print(outf)



            dic = T.load_npy(fdir+f)
            zscore_dic = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                # print(len(dic[pix]))
                time_series = dic[pix]['growing_season']
                # print(len(time_series))
                time_series=time_series[19:]


                # exit()

                time_series = np.array(time_series)
                # time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue

                time_series = time_series
                mean = np.nanmean(time_series)
                relative_change = (time_series - mean) / mean * 100

                zscore_dic[pix] = relative_change
                # plot
                # plt.plot(time_series)

                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)

    def weighted_average_LAI(self):  ###add weighted average LAI in dataframe
        df =result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season_yearly.df'
        df = T.load_df(df)
        df_clean = self.df_clean(df)
        # variable_list=['LAI_relative_change','GOSIF','NDVI','NIRv','CABLE-POP_S2_lai','CLASSIC_S2_lai',
        #                'CLM5','DLEM_S2_lai','IBIS_S2_lai','ISAM_S2_lai',
        #                'ISBA-CTRIP_S2_lai','JSBACH_S2_lai',
        #                'JULES_S2_lai','LPJ-GUESS_S2_lai','LPX-Bern_S2_lai',
        #                'ORCHIDEE_S2_lai',
        #                'SDGVM_S2_lai',
        #                'YIBs_S2_Monthly_lai']

        variable_list = ['LAI4g', 'GOSIF', 'NDVI', 'NIRv', ]




        for var in variable_list:

            # df_clean_ii = df_clean[(df_clean[var] > -50) & (df_clean[var] < 50)]
            # print(len(df_clean_ii));exit()
            # Step 1: 计算纬度权重
            df_clean_ii = df_clean

            df_clean_ii[f'latitude_weight'] = np.cos(np.radians(df_clean_ii['lat']))
            # Step 2: 按年份对权重进行归一化
            # 确保每一年干旱区的权重总和为1
            df_clean_ii[f'normalized_weight'] = df_clean_ii.groupby('year')['latitude_weight'].transform(
                lambda x: x / x.sum())
            print(df_clean_ii.groupby('year')['normalized_weight'].sum())
            # exit()

            # Step 3: 计算加权平均LAI
            # weighted_avg_lai = df_clean_ii.groupby('year')['LAI_relative_change'].apply(lambda x: (x * df_clean_ii['normalized_weight']).sum())
            df_clean_ii[f'weighted_avg_contribution_{var}'] = df_clean_ii[var] * df_clean_ii[f'normalized_weight']

            weighted_avg_lai_per_year = (
                df_clean_ii.groupby('year')[f'weighted_avg_contribution_{var}'].sum().reset_index(name=f'weighted_avg_{var}')
            )

            df_clean[f'weighted_avg_{var}'] = weighted_avg_lai_per_year[f'weighted_avg_{var}']
            # T.print_head_n(df_clean_ii)

        # exit()
        # T.print_head_n(df_clean_ii);exit()
        outf=result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season_yearly.df'
        T.save_df(df_clean, outf)
        T.df_to_excel(df_clean, outf)

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

    def trend_analysis(self):  ##each window average trend

        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'3mm\relative_change_growing_season\\TRENDY\\'
        outdir = result_root+rf'\3mm\relative_change_growing_season\TRENDY\trend_3\\'
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
                if r < 60:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                    ## ignore the last one year

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
                time_series = np.array(time_series)
                print(time_series)
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

    def TRENDY_ensemble(self):

        model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',

                      'YIBs_S2_Monthly_lai']

        fdir = result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis\\'
        arr_list = []

        for model in model_list:
            fpath = fdir + model + '_trend.tif'

            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr[arr>99]=np.nan
            arr[arr<-99]=np.nan
            # plt.imshow(arr, cmap='jet', vmin=-1, vmax=1)
            # plt.colorbar()
            # plt.show()
            arr_list.append(arr)

        arr_ensemble = np.nanmean(arr_list, axis=0)
        arr_ensemble[arr_ensemble>99]=np.nan
        arr_ensemble[arr_ensemble<-99]=np.nan
        plt.imshow(arr_ensemble, cmap='jet', vmin=-1, vmax=1)

        plt.colorbar()
        plt.show()


        DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_ensemble,
                                              result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis\TRENDY_ensemble_mean_trend.tif')






    def plt_basemap(self):
        fdir = result_root+rf'3mm\relative_change_growing_season\TRENDY\trend_analysis\\'

        count = 1
        fig = plt.figure(figsize=(10, 16))
        variables_list = ['LAI4g_trend', 'CABLE-POP_S2_lai_trend', 'CLASSIC_S2_lai_trend',
                     'CLM5_trend', 'DLEM_S2_lai_trend', 'IBIS_S2_lai_trend', 'ISAM_S2_lai_trend',
                     'ISBA-CTRIP_S2_lai_trend', 'JSBACH_S2_lai_trend',
                     'JULES_S2_lai_trend', 'LPJ-GUESS_S2_lai_trend', 'LPX-Bern_S2_lai_trend',
                     'ORCHIDEE_S2_lai_trend',
                     'SDGVM_S2_lai_trend',
                     'YIBs_S2_Monthly_lai_trend']
        for variable in variables_list:
            # print(f.split('.')[0]); exit()
            f=fdir+variable+'.tif'

            print(f)

            outf = fdir + f.split('.')[0]
            if os.path.isfile(outf + '_trend.tif'):
                continue
            # print(outf)

            fpath = join(fdir, f )

            ax = plt.subplot(4, 4, count)
            count = count + 1

            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = Tools().mask_999999_arr(arr, warning=False)
            # plt.imshow(arr, cmap='PiYG', interpolation='nearest', vmin=-50, vmax=50)
            # plt.show()
            arr = arr[:60]
            # arr_m = ma.masked_where(np.isnan(arr), arr)
            lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            # plt.imshow(arr,interpolation='nearest', cmap='PiYG', vmin=-.1, vmax=.1)
            # plt.show()
            # print(lat_list);exit()
            # plt.show()
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180, resolution='i',
                        ax=ax)

            ret = m.pcolormesh(lon_list, lat_list, arr, cmap='RdBu', zorder=1, vmin=-1, vmax=1)
            coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
            ## set basemap size

            plt.title(variable)
            plt.imshow(arr, cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
            # m.pcolormesh(lon_list, lat_list, arr_m, cmap='PiYG', zorder=-1, vmin=-1, vmax=1)
            #
            plt.tight_layout()


            # cax = plt.axes([0.5 - 0.3 / 2, 0.1, 0.3, 0.02])
            # plt.colorbar(mappable=ret, ax=ax, orientation='horizontal')
            ## set colorbar (CV %/year)
            # colorbar = plt.colorbar(mappable=ret, ax=ax, orientation='horizontal')

            ## set name of colorbar
            # colorbar.set_label('CV %/year', fontsize=12)



            # plt.colorbar(ax=ax, cax=cax, orientation='horizontal', extend='both')
        outdir=result_root+rf'3mm\relative_change_growing_season\TRENDY\fig\\'
        T.mk_dir(outdir, force=True)

        outf = join(outdir, 'trend_analysis.pdf')

        plt.savefig(outf, dpi=300, bbox_inches='tight')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        plt.show()









    pass

    def plot_bar_trend(self): ### all models comparision
        dff=result_root+rf'3mm\Dataframe\Trend\\Trend.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        variables_list = ['LAI4g_trend', 'CABLE-POP_S2_lai_trend', 'CLASSIC_S2_lai_trend',
                     'CLM5_trend', 'DLEM_S2_lai_trend', 'IBIS_S2_lai_trend', 'ISAM_S2_lai_trend',
                     'ISBA-CTRIP_S2_lai_trend', 'JSBACH_S2_lai_trend',
                     'JULES_S2_lai_trend', 'LPJ-GUESS_S2_lai_trend', 'LPX-Bern_S2_lai_trend',
                     'ORCHIDEE_S2_lai_trend',
                     'SDGVM_S2_lai_trend',
                     'YIBs_S2_Monthly_lai_trend']
        result_dict = {}
        boxlist=[]
        for variable in variables_list:
            val_list=df[variable].values
            val_array=np.array(val_list)
            val_array[val_array<-9999]=np.nan
            val_array[val_array>9999]=np.nan
            val_array = val_array[~np.isnan(val_array)]
            boxlist.append(val_array)
        plt.boxplot(boxlist,showfliers=False,labels=variables_list)
        plt.xticks(rotation=90)
        plt.ylabel('Trend (%) per year')
        plt.show()
        exit()

        for variable in variables_list:
            val_list = df[variable].values
            val_array = np.array(val_list)
            val_array[val_array < -9999] = np.nan
            val_array[val_array > 9999] = np.nan

            result_dict[variable]=val_array
        df_new=pd.DataFrame(result_dict)
        df_new.boxplot()
        # sns.violinplot(x='variable', y='value', data=pd.melt(df_new))

        plt.xticks(rotation=90)
        plt.ylabel('Trend (%) per year')
        plt.show()


        pass

    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + rf'\relative_change_growing_season\\TRENDY\\'
        outdir = result_root + rf'\TRENDY\\detrend_growing_season\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not '.npy' in f:
                continue


            outf = outdir + f.split('.')[0] + '.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

            detrend_zscore_dic = {}

            for pix in tqdm(dic):
                dryland_values = dic_dryland_mask[pix]
                if np.isnan(dryland_values):
                    continue
                r, c = pix

                # print(len(dic[pix]))
                time_series = dic[pix]
                print(len(time_series))

                # print(time_series)

                time_series = np.array(time_series)
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
                if r < 480:
                    # print(time_series)
                    ### interpolate
                    time_series = T.interp_nan(time_series)
                    # print(np.nanmean(time_series))
                    # plt.plot(time_series)

                    detrend_delta_time_series = signal.detrend(time_series) + np.nanmean(time_series)
                    # plt.plot(detrend_delta_time_series)
                    # plt.show()

                    detrend_zscore_dic[pix] = detrend_delta_time_series
                else:
                    time_series = time_series[0:38]
                    print(time_series)

                    if np.isnan(time_series).any():
                        continue
                    # print(time_series)
                    detrend_delta_time_series = signal.detrend(time_series) + np.nanmean(time_series)
                    ###add nan to the end if length is less than time_series
                    if len(detrend_delta_time_series) < 38:
                        detrend_delta_time_series = np.append(detrend_delta_time_series,
                                                              [np.nan] * (38 - len(detrend_delta_time_series)))

                        detrend_zscore_dic[pix] = detrend_delta_time_series

            np.save(outf, detrend_zscore_dic)


class TRENDY_CV:
    ## 1)

    def __init__(self):
        self.map_width = 15.3 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass

    def run(self):
        ## intra_CV_anaysis extract_LAI_phenology mean

        # self.detrend()
        # self.moving_window_extraction()
        # self.moving_window_CV_anaysis()
        # self.moving_window_mean_anaysis()
        # self.moving_window_max_min_anaysis()
        # self.trend_analysis()
        # self.TRENDY_ensemble()
        # self.TRENDY_ensemble_npy()
        # self.plot_robinson()
        # self.plt_basemap()

        # self.plot_CV_trend_bin() ## plot CV vs. trend in observations
        # self.plot_CV_trend_among_models() ## calculating mean obs and TRENDY
        self.plot_CV_trend_among_models2()  ## not calculating mean obs and TRENDY use it
        # self.bar_plot_continent()
        # self.CV_Aridity_gradient_plot()
        # self.plot_sign_between_LAI_NDVI()
        # self.plot_significant_percentage_area()
        # self.heatmap()
        # self.heatmap_count()



        pass


    def detrend(self): ## detrend LAI4g

        fdir = result_root+rf'3mm\relative_change_growing_season\TRENDY\\'
        outdir = result_root+rf'\3mm\relative_change_growing_season\\detrend\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            outf = outdir + f'{f.split(".")[0]}_detrend.npy'
            if isfile(outf):
                continue
            dict = T.load_npy(fdir + f)
            annual_spatial_dict = {}
            spatial_len_dic={}
            for pix in tqdm(dict):
                time_series = dict[pix]
                print(time_series)

                # if T.is_all_nan(time_series):
                #     continue
                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.isnan(time_series).any():
                    continue

                # plt.plot(time_series)

                #
                #
                detrended_annual_time_series = signal.detrend(time_series)+np.mean(time_series)
                # print((detrended_annual_time_series))
                # plt.plot(detrended_annual_time_series)
                # plt.show()

                annual_spatial_dict[pix] = detrended_annual_time_series
                spatial_len_dic[pix] = len(detrended_annual_time_series)
            arr_len=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_len_dic)
            plt.imshow(arr_len)
            plt.colorbar()
            plt.title(f)
            plt.show()

            np.save(outf, annual_spatial_dict)



        pass

    def moving_window_extraction(self):

        fdir = result_root+rf'\3mm\relative_change_growing_season\detrend\\'
        outdir = result_root+rf'\3mm\relative_change_growing_season\moving_window_extraction\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


            outf = outdir + f
            print(outf)
            if os.path.isfile(outf):
                continue

            dic = T.load_npy(fdir + f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                time_series = dic[pix]
                time_series = np.array(time_series)

                # time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                # print((len(time_series)))
                ### if all values are identical, then continue
                if np.nanmax(time_series) == np.nanmin(time_series):
                    continue

                # new_x_extraction_by_window[pix] = self.forward_window_extraction_detrend_anomaly(time_series, window)
                new_x_extraction_by_window[pix] = self.forward_window_extraction(time_series, window)

            T.save_npy(new_x_extraction_by_window, outf)

    def moving_window_max_min_anaysis(self):  ## each window calculating the average
        window_size = 15

        fdir =result_root+ rf'\3mm\relative_change_growing_season\moving_window_extraction\\'
        outdir = result_root+rf'\3mm\relative_change_growing_season\\moving_window_min_max_anaysis\\min\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'detrend' in f:
                continue


            dic = T.load_npy(fdir + f)

            slides = 38 - window_size + 1  ## revise!!
            outf = outdir + f.split('.')[0] + f'_min.npy'
            print(outf)

            trend_dic = {}

            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                for ss in range(slides):

                    ### if all values are identical, then continue
                    if len(time_series_all) < 24:
                        continue

                    time_series = time_series_all[ss]
                    # print(time_series)
                    # if np.nanmax(time_series) == np.nanmin(time_series):
                    #     continue
                    # print(len(time_series))
                    ##average
                    average = np.nanmin(time_series)
                    # print(average)

                    trend_list.append(average)

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)

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
    def moving_window_mean_anaysis(self):
        window_size=15
        fdir = result_root+rf'\3mm\relative_change_growing_season\moving_window_extraction\\'
        outdir = result_root+rf'\3mm\relative_change_growing_season\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)
            slides = 39-window_size
            outf = outdir + f.split('.')[0] + f'_mean.npy'
            print(outf)

            if os.path.isfile(outf):
                continue

            new_x_extraction_by_window = {}
            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                if len(time_series_all)<24:
                    continue
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue
                    # print((len(time_series)))
                    ### if all values are identical, then continue
                    time_series=time_series_all[ss]
                    # print(time_series)
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    # print(len(time_series))

                    if np.nanmean(time_series)==0:
                        continue
                    mean=np.nanmean(time_series)
                    trend_list.append(mean)

                trend_dic[pix]=trend_list

            np.save(outf, trend_dic)

            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')

    def moving_window_CV_anaysis(self):
        window_size=15
        fdir = result_root+rf'\3mm\extraction_GIMMS3g_plus_NDVI\dryland\moving_window_extraction\\'
        outdir = result_root+rf'\3mm\extraction_GIMMS3g_plus_NDVI\dryland\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)
            slides = 39-window_size
            outf = outdir + f.split('.')[0] + f'_CV.npy'
            print(outf)

            if os.path.isfile(outf):
                continue

            new_x_extraction_by_window = {}
            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                if len(time_series_all)<24:
                    continue
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue
                    # print((len(time_series)))
                    ### if all values are identical, then continue
                    time_series=time_series_all[ss]
                    # print(time_series)
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    # print(len(time_series))

                    if np.nanmean(time_series)==0:
                        continue
                    cv=np.nanstd(time_series)/np.nanmean(time_series)*100
                    trend_list.append(cv)

                trend_dic[pix]=trend_list

            np.save(outf, trend_dic)

            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')

    def TRENDY_ensemble_npy(self):

        model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',

                      'YIBs_S2_Monthly_lai']
        # model_list=['SNU_LAI','LAI4g','GLOBMAP_LAI']



        fdir = result_root + rf'\TRENDY\S2\15_year\moving_window_extraction_CV\\'

        result_dic={}

        for model in model_list:
            fpath = fdir + model + '_detrend_CV.npy'
            dic=T.load_npy(fpath)
            result_dic[model]=dic
        ## for each pixel calculate mean
        all_pixels = list(result_dic[model_list[0]].keys())
        ensemble_mean_dic = {}

        for pix in all_pixels:
            timeseries_list = []
            for model in model_list:
                if pix not in result_dic[model]:
                    continue
                vals = result_dic[model][pix]
                if len(vals)<24:
                    continue
                if vals is not None:
                    timeseries_list.append(np.array(vals))

            if timeseries_list:
                # stack 成 (13, T) 然后对 axis=0 平均
                timeseries_list = np.array(timeseries_list)
                for time_series in timeseries_list:
                    plt.plot(time_series)



                mean_ts = np.nanmean(timeseries_list, axis=0)

                # plt.plot(mean_ts,color='k',label='median')
                # plt.legend()
                #
                # plt.show()

                ensemble_mean_dic[pix] = mean_ts


        outf=result_root + rf'\TRENDY\S2\15_year\moving_window_extraction_CV\TRENDY_ensemble_mean_detrend_CV.npy'
        T.save_npy(ensemble_mean_dic, outf)


    def TRENDY_ensemble(self):

        model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',

                      'YIBs_S2_Monthly_lai']

        # model_list = [ 'GLOBMAP_LAI', 'LAI4g', 'SNU_LAI', ]

        fdir_all = result_root + rf'\partial_correlation\TRENDY\result\\\\'
        arr_list = []

        for model in model_list:
            fpath=join(fdir_all,model, 'sig_nomask', 'CV_daily_rainfall_average.tif')#fdir_all+model+'sig_nomask'+'_sensitivity.tif'
            # for f in os.listdir(fdir_all):



                #
                # if not f.endswith('.tif'):
                #     continue
                # fpath=fdir_all+f




            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr[arr > 99] = np.nan
            arr[arr < -99] = np.nan

            arr_list.append(arr)
        print(len(arr_list))

        arr_ensemble = np.nanmedian(arr_list, axis=0)
        arr_ensemble[arr_ensemble > 99] = np.nan
        arr_ensemble[arr_ensemble < -99] = np.nan
        plt.imshow(arr_ensemble, cmap='RdYlGn')
        plt.colorbar()
        plt.show()
        # outdir=result_root +  rf'\partial_correlation\TRENDY\result\TRENDY_ensemble_median\\'
        # T.mk_dir(outdir, force=True)
        # outf=outdir+'CV_daily_rainfall_average.tif'
        # DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_ensemble, outf)



    def trend_analysis(self):  ##each window average trend

        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'\3mm\relative_change_growing_season\moving_window_min_max_anaysis\max\\'
        outdir = result_root+rf'\3mm\relative_change_growing_season\moving_window_min_max_an aysis\max\\trend_analysis\\'
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
                if r < 60:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                    ## ignore the last one year

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
                time_series = np.array(time_series)
                time_series=time_series
                # print(time_series)
                if np.isnan(np.nanmean(time_series)):
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

    pass

    def plot_robinson(self):

        # fdir_trend = result_root+rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result\npy_time_series\trend\\'
        fdir_trend = result_root+rf'3mm\extract_composite_phenology_year\trend\\'
        temp_root = result_root+rf'3mm\extract_composite_phenology_year\temp_root\\'
        outdir = result_root+rf'\\3mm\extract_composite_phenology_year\\plot_robinson\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue
            if not 'mean' in f:
                continue

            fname = 'composite_LAI_mean_trend'
            fname_p_value = 'composite_LAI_mean_p_value'
            print(fname_p_value)
            fpath = fdir_trend+fname+'.tif'
            p_value_f = fdir_trend + fname_p_value + '.tif'
            print(p_value_f)
            # exit()
            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=-0.01, vmax=0.01, is_discrete=True, colormap_n=9,)

            Plot_Robinson().plot_Robinson_significance_scatter(m,p_value_f,temp_root,0.05, s=0.2, marker='.')
            # plt.title(f'{fname}')
            # plt.show()
            outf = outdir + f+'.pdf'
            plt.savefig(outf)
            plt.close()

    def plt_basemap(self):
        fdir = rf'E:\Project3\Data\TRENDY_LAI\trend_analysis\moving_window_CV\\'

        count = 1
        fig = plt.figure(figsize=(10, 15))
        for f in os.listdir(fdir):
            # if not 'CABLE' in f:
            #     continue
            if not f.endswith('.tif'):
                continue
            if 'p_value' in f:
                continue


            print(f)

            outf = fdir + f.split('.')[0]
            if os.path.isfile(outf + '_trend.tif'):
                continue
            print(outf)

            fpath = join(fdir, f )


            ax = plt.subplot(5, 3, count)
            count = count + 1


            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = Tools().mask_999999_arr(arr, warning=False)
            # plt.imshow(arr, cmap='PiYG', interpolation='nearest', vmin=-50, vmax=50)
            # plt.show()
            # arr = arr[:120]
            # arr_m = ma.masked_where(np.isnan(arr), arr)
            lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            # plt.imshow(arr,interpolation='nearest', cmap='PiYG', vmin=-.1, vmax=.1)
            # plt.show()
            # print(lat_list);exit()
            # plt.show()
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180, resolution='i',
                        ax=ax)

            ret = m.pcolormesh(lon_list, lat_list, arr, cmap='PiYG', zorder=1, vmin=-1, vmax=1)
            coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
            ## set basemap size


            plt.title(f.replace('_trend.tif', '').replace('_S2', '').replace('_lai', '').replace('_LAI', '').replace('_detrend', '').replace('_annual', ''))
            # plt.imshow(arr, cmap='PiYG', interpolation='nearest', vmin=-50, vmax=50)
            # m.pcolormesh(lon_list, lat_list, arr_m, cmap='PiYG', zorder=-1, vmin=-1, vmax=1)

            # plt.tight_layout()


            # cax = plt.axes([0.5 - 0.3 / 2, 0.1, 0.3, 0.02])
            # plt.colorbar(mappable=ret, ax=ax, orientation='horizontal')
            ## set colorbar (CV %/year)
            # colorbar = plt.colorbar(mappable=ret, ax=ax, orientation='horizontal')

            ## set name of colorbar
            # colorbar.set_label('CV %/year', fontsize=12)




            # plt.colorbar(ax=ax, cax=cax, orientation='horizontal', extend='both')
        outdir=rf'E:\Project3\Data\TRENDY_LAI\trend_analysis\moving_window_CV\\Figure\\'
        T.mk_dir(outdir, force=True)

        # outf = join(outdir, 'trend_analysis.pdf')
        #
        # plt.savefig(outf, dpi=300, bbox_inches='tight')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        plt.show()

    def plot_significant_percentage_area(self):  ### insert bar plot for all spatial map to calculate percentage

        dff = result_root + rf'\3mm\Dataframe\CVTrend_global\\\\CVTrend_global.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        ##plt histogram of LAI
        df=df[df['detrended_LAI4g_CV_trend_global']<100]
        df=df[df['detrended_LAI4g_CV_trend_global']>-100]


        vals_p_value = df['detrended_LAI4g_CV_p_value_global'].values
        significant_browning_count = 0
        non_significant_browning_count = 0
        significant_greening_count = 0
        non_significant_greening_count = 0

        for i in range(len(vals_p_value)):
            if vals_p_value[i] < 0.05:
                if df['detrended_LAI4g_CV_trend_global'].values[i] > 0:
                    significant_greening_count = significant_greening_count + 1
                else:
                    significant_browning_count = significant_browning_count + 1
            else:
                if df['detrended_LAI4g_CV_trend_global'].values[i] > 0:
                    non_significant_browning_count = non_significant_browning_count + 1
                else:
                    non_significant_greening_count = non_significant_greening_count + 1
            ## plot bar
        ##calculate percentage
        significant_greening_percentage = significant_greening_count / len(vals_p_value)*100
        non_significant_greening_percentage = non_significant_greening_count / len(vals_p_value)*100
        significant_browning_percentage = significant_browning_count / len(vals_p_value)*100
        non_significant_browning_percentage = non_significant_browning_count / len(vals_p_value)*100

        count = [non_significant_browning_percentage,significant_browning_percentage, significant_greening_percentage,

                 non_significant_greening_percentage]
        print(count)
        labels = ['non_significant_decreasing','significant_decreasing', 'significant_increasing',
                  'non_significant_increasing']
        color_list = [
            '#008837',
            '#a6dba0',
            '#7b3294',
            '#c2a5cf',

        ]
        ##gap = 0.1
        df_new=pd.DataFrame({'count':count})
        df_new_T=df_new.T


        df_new_T.plot.barh( stacked=True, color=color_list,legend=False,width=0.1,)
        ## add legend
        plt.legend(labels)

        plt.ylabel('Percentage (%)')
        plt.tight_layout()

        plt.show()


    pass
    def bar_plot_continent(self):
        ## plot non_dryland and dryland bar plot
        dff=result_root+rf'3mm\Dataframe\moving_window_CV\\moving_window_CV_new.df'
        df=T.load_df(dff)
        print(len(df))
        df=self.df_clean(df)
        df = df[df['LAI4g_detrend_CV_p_value'] < 0.05]
        print(len(df))

        # T.print_head_n(df);exit()

        # classfication_list = [ 'Dryland','Sub-humid','Humid','Very Humid']
        ## print column ==continent
        continent_list=df['continent'].unique().tolist()
        # print(continent_list)
        # exit()
        classfication_list = ['Dryland', 'Sub-humid', 'Humid', 'Very Humid']
        classfication_list=['Global','Australia','South_America','Africa',  'Asia', 'North_America', ]
        fig = plt.figure(figsize=(self.map_width / 1.2, self.map_height))
        for classfication in classfication_list:
            if classfication=='Global':
                df_i=df
            else:

                df_i = df[df['continent'] == classfication]
            vals = df_i['LAI4g_detrend_CV_trend'].tolist()
            vals=np.array(vals)
            vals[vals>99]=np.nan
            vals[vals<-99]=np.nan
            vals=vals[~np.isnan(vals)]
            count=len(vals)
            print(count)
            print(np.nanmean(vals))

            plt.bar(x=classfication, height=np.nanmean(vals), yerr=np.nanstd(vals), error_kw={'capsize': 3},color='#D3D3D3')
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.ylabel('Change in CVLAI (%/yr)')
        # plt.show()
        # outf = rf'C:\Users\wenzhang1\Desktop\transfer_figure\bar_plot_continent.pdf'
        # fig.savefig(outf, dpi=300)



        pass
    def CV_Aridity_gradient_plot(self):
        ## plot non_dryland and dryland bar plot
        dff=result_root+rf'3mm\Dataframe\moving_window_CV\\moving_window_CV_new.df'
        df=T.load_df(dff)
        print(len(df))
        df=self.df_clean(df)
        df = df[df['LAI4g_detrend_CV_p_value'] < 0.05]
        print(len(df))
        ## reclass
        threhold_list = [  0.1, 0.2,  0.3,  0.4, 0.5,  0.6, 0.65]
        label_list = []
        CV_list = []
        CV_list_mean = []
        CV_list_std = []
        result_dic = {}
        countlist=[]

        for i in range(len(threhold_list) - 1):
            threhold1 = threhold_list[i]
            threhold2 = threhold_list[i + 1]

            df_ii = df[(df['Aridity'] > threhold1) & (df['Aridity'] < threhold2)]

            CV_val = df_ii['LAI4g_detrend_CV_trend'].values
            CV_val = CV_val[~np.isnan(CV_val)]
            CV_val[CV_val > 99] = np.nan
            CV_val[CV_val < -99] = np.nan
            result_dic[f'{threhold1}-{threhold2}'] = CV_val
        ## bar boxplot
        for i in range(len(threhold_list) - 1):
            threhold1 = threhold_list[i]
            threhold2 = threhold_list[i + 1]

            CV_val = result_dic[f'{threhold1}-{threhold2}']
            CV_val_mean = np.nanmean(CV_val)
            CV_val_std = np.nanstd(CV_val)
            CV_list_mean.append(CV_val_mean)
            CV_list_std.append(CV_val_std)
            countlist.append(len(CV_val))

            CV_list.append(CV_val)
            label_list.append(f'{threhold1}-{threhold2}')
        ## plot violin plot
        x_ticks = np.linspace(0.05,0.65,len(label_list))
        print(countlist)
        fig=plt.figure(figsize=(self.map_width/1.2,self.map_height))



        plt.bar(x=x_ticks, height=CV_list_mean, yerr=CV_list_std, error_kw={'capsize': 3}, color='#D3D3D3',width=0.05)
        plt.xlabel('Aridity')
        plt.ylabel('Trend of CVLAI (%/yr)')
        plt.twinx()
        plt.twiny()

        # plt.hist(df['Aridity'].tolist(), bins=20, alpha=0.5, label='Positive', color='green', rwidth=0.9)
        AI_values = df['Aridity'].tolist()

        # x,y=Plot().plot_hist_smooth(AI_values,bins=threhold_list,alpha=0,interpolate_window=5)
        # plt.plot(x, y,)
        # plt.ylim(0.036, 0.12)


        # plt.xticks(np.arange(1, len(label_list) + 1), label_list, rotation=45)
        plt.tight_layout()
        plt.yticks([])
        plt.xticks([])
        # plt.show()
        outf=rf'C:\Users\wenzhang1\Desktop\transfer_figure\bar_plot_continent.pdf'
        fig.savefig(outf, dpi=300)



    def plot_CV_trend_bin(self):
        dff=result_root+rf'3mm\Dataframe\Trend_new\\Trend.df'
        df=T.load_df(dff)
        df=self.df_clean(df)

        lai_trends = df['sum_rainfall_sensitivity_trend'].values
        laicv_trends = df['LAI4g_detrend_CV_trend'].values

        # Remove NaN values
        lai_trends = lai_trends[~np.isnan(lai_trends)]
        laicv_trends = laicv_trends[~np.isnan(laicv_trends)]
        lai_trends[lai_trends > 99] = np.nan
        lai_trends[lai_trends < -99] = np.nan
        laicv_trends[laicv_trends > 99] = np.nan
        laicv_trends[laicv_trends < -99] = np.nan


        # plt.hist(laicv_trends, bins=10, alpha=0.5, label='LAICV Trends')
        # plt.hist(lai_trends, bins=10, alpha=0.5, label='LAI Trends')
        # plt.legend(loc='upper right')
        # plt.show()
        threhold_list = [-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2,2.4]

        label_list = []
        CV_list = []
        result_dic = {}

        for i in range(len(threhold_list) - 1):
            threhold1 = threhold_list[i]
            threhold2 = threhold_list[i + 1]

            df_ii = df[(df['sum_rainfall_sensitivity_trend'] > threhold1) & (df['sum_rainfall_sensitivity_trend'] < threhold2)]

            CV_val = df_ii['LAI4g_detrend_CV_trend'].values
            CV_val = CV_val[~np.isnan(CV_val)]
            CV_val[CV_val > 99] = np.nan
            CV_val[CV_val < -99] = np.nan
            result_dic[f'{threhold1}-{threhold2}'] = CV_val
        ## bar boxplot
        plt.figure(figsize=(self.map_width, self.map_height))
        for i in range(len(threhold_list) - 1):
            threhold1 = threhold_list[i]
            threhold2 = threhold_list[i + 1]

            CV_val = result_dic[f'{threhold1}-{threhold2}']

            CV_list.append(CV_val)
            label_list.append(f'{threhold1}-{threhold2}')
        ## plot violin plot

        plt.boxplot(CV_list, showmeans=True, showfliers=False)
        plt.xlabel('Beta (%/year)')
        plt.ylabel('Trend of CVLAI (%/year)')

        plt.xticks(np.arange(1, len(label_list) + 1), label_list, rotation=45)
        plt.tight_layout()
        plt.show()



    def plot_CV_trend_among_models(self):  ##calculating mean obs and TRENDY in program

        color_list = ['black','black', 'black',  'Lime',
                      '#f0988c', 'yellow', 'Aqua',
                      '#E7483D', '#ffff33','#984ea3',#ffff33',','#e41a1c',
                      '#9e9e9e', '#cfeaf1', 'green',
                      '#98cccb', 'DarkGoldenrod','DodgerBlue', 'black','#e66d50',]
        ## I want use set 3 color

        mark_size_list=[50]*3+[50]*13+[200]*2
        # alpha_list=[1]+[0.7]*3+[1]+[0.7]*12



        dff = result_root + rf'\Dataframe\\Trends_CV.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        T.print_head_n(df)
        print(df.columns.tolist())
        ## print column names
        # print(df.columns)
        # exit()
        marker_list=['^','s','P','X',]*4+['s']+['^']+['D']


        variables_list = ['LAI4g', 'GLOBMAP_LAI',
                          'SNU_LAI',
                    'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']
        vals_trend_list = []
        vals_CV_list = []
        err_trend_list = []
        err_CV_list = []


        for variable in variables_list:

            vals_trend = df[f'{variable}_relative_change_trend'].values
            vals_CV = df[f'{variable}_detrend_CV_trend'].values
            vals_trend[vals_trend>999] = np.nan
            # vals_trend[vals_CV_p_value > 0.05] = np.nan
            vals_CV[vals_CV>999] = np.nan
            vals_trend[vals_trend < -999] = np.nan
            vals_trend[vals_trend < -10] = np.nan
            vals_trend[vals_trend > 99] = np.nan
            vals_CV[vals_CV < -999] = np.nan
            # vals_CV[vals_CV_p_value > 0.05] = np.nan
            vals_trend = vals_trend[~np.isnan(vals_trend)]
            # print(variable,np.nanmean(vals_trend))
            # plt.hist(vals_trend,bins=100,color=color_list[0],alpha=0.5,edgecolor='k')
            # plt.title(variable)
            # plt.show()
            vals_CV = vals_CV[~np.isnan(vals_CV)]
            vals_trend_list.append(np.nanmean(vals_trend))
            vals_CV_list.append(np.nanmean(vals_CV))


            if variable in ['composite_LAI_mean','TRENDY_ensemble_mean']:
                err_trend_list.append(np.nanstd(vals_trend))
                err_CV_list.append(np.nanstd(vals_CV))
            else:
                err_trend_list.append(np.nan)
                err_CV_list.append(np.nan)
            print(variable, np.nanmean(vals_trend), np.nanmean(vals_CV))
        # exit()

        # === 计算 ensemble 和 obs mean ===
        obs_vars = ['LAI4g', 'GLOBMAP_LAI', 'SNU_LAI']
        model_vars = [v for v in variables_list if v not in obs_vars]

        obs_trend_mean = np.nanmean([vals_trend_list[variables_list.index(v)] for v in obs_vars])
        obs_CV_mean = np.nanmean([vals_CV_list[variables_list.index(v)] for v in obs_vars])

        model_trend_mean = np.nanmean([vals_trend_list[variables_list.index(v)] for v in model_vars])
        print('obs mean', obs_trend_mean, obs_CV_mean)
        model_CV_mean = np.nanmean([vals_CV_list[variables_list.index(v)] for v in model_vars])
        print('model mean', model_trend_mean, model_CV_mean)

        # 把 ensemble 加入列表末尾
        variables_plot = variables_list + ['OBS_mean', 'TRENDY_ensemble_mean']
        vals_trend_plot = vals_trend_list + [obs_trend_mean, model_trend_mean]
        vals_CV_plot = vals_CV_list + [obs_CV_mean, model_CV_mean]

        # plt.scatter(vals_CV_list,vals_trend_list,marker=marker_list,color=color_list[0],s=100)
        # plt.show()
        ##plot error bar
        # plt.figure(figsize=(self.map_width, self.map_height))

        plt.figure(figsize=(13*centimeter_factor, 10*centimeter_factor))

        # self.map_width = 13 * centimeter_factor
        # self.map_height = 8.2 * centimeter_factor

        err_trend_list = np.array(err_trend_list)
        err_CV_list = np.array(err_CV_list)
        for i, (x, y, marker,color,var,mark_size) in enumerate(zip(vals_trend_plot, vals_CV_plot, marker_list, color_list, variables_plot,mark_size_list)):
            plt.scatter(y, x, marker=marker,color=color_list[i], label=var, s=mark_size, edgecolors='black',)
            # plt.errorbar(y, x, xerr=err_trend_list[i], yerr=err_CV_list[i], fmt='none', color='grey', capsize=2, capthick=0.3,alpha=1)


            ##markerborderwidth=1

            plt.ylabel('Trends in LAI (%/yr)', fontsize=12)
            plt.xlabel('Trends in CVLAI (%/yr)', fontsize=12)
            plt.ylim(-0.3,.9)
            plt.xlim(-0.2, 0.5)
            plt.xticks(fontsize=12)
            ## xticks gap 0.05
            plt.yticks(np.arange(-0.2, .9, 0.2), fontsize=12)
            plt.yticks(fontsize=12)
            # plt.legend()
        ## save imagine
        plt.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
        plt.axvline(x=0.0, color='k', linestyle='--', linewidth=1)
        outdir=result_root + rf'\FIGURE\\Figure4\\'
        plt.savefig(outdir+ rf'\\LAI4g_detrend_CV_trend.pdf',  bbox_inches='tight')


        #
        # plt.show()

    def plot_CV_trend_among_models2(self):  ##here not calculating mean in program

        color_list = ['black', 'black', 'black', '#a1a9d0',
                      '#f0988c', '#b883d3', '#ffff33', '#c4a5de',
                      '#E7483D', '#984ea3', '#e41a1c',
                      '#9e9e9e', '#cfeaf1', '#f6cae5',
                      '#98cccb', '#5867AF', 'black', '#e66d50', ]
        ## I want use set 3 color

        mark_size_list = [200] * 1+[50] * 3 +[200] * 1+ [50] * 13
        # alpha_list=[1]+[0.7]*3+[1]+[0.7]*12

        dff = result_root + rf'\Dataframe\\Trends_CV\\Trends_CV.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        T.print_head_n(df)
        print(df.columns.tolist())
        ## print column names
        # print(df.columns)
        # exit()
        marker_list = ['^', 's', 'P', 'X', 'D'] * 4

        variables_list = ['composite_LAI_median','LAI4g', 'GLOBMAP_LAI',
                          'SNU_LAI',
                          'TRENDY_ensemble_median',
                          'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']
        vals_trend_list = []
        vals_CV_list = []
        err_trend_list = []
        err_CV_list = []

        for variable in variables_list:

            vals_trend = df[f'{variable}_relative_change_trend'].values
            vals_CV = df[f'{variable}_detrend_CV_trend'].values
            vals_trend[vals_trend > 999] = np.nan
            # vals_trend[vals_CV_p_value > 0.05] = np.nan
            vals_CV[vals_CV > 999] = np.nan
            vals_trend[vals_trend < -999] = np.nan
            vals_trend[vals_trend < -10] = np.nan
            vals_trend[vals_trend > 99] = np.nan
            vals_CV[vals_CV < -999] = np.nan
            # vals_CV[vals_CV_p_value > 0.05] = np.nan
            vals_trend = vals_trend[~np.isnan(vals_trend)]
            # print(variable,np.nanmean(vals_trend))
            # plt.hist(vals_trend,bins=100,color=color_list[0],alpha=0.5,edgecolor='k')
            # plt.title(variable)
            # plt.show()
            vals_CV = vals_CV[~np.isnan(vals_CV)]
            vals_trend_list.append(np.nanmean(vals_trend))
            vals_CV_list.append(np.nanmean(vals_CV))


        # exit()


        # 把 ensemble 加入列表末尾
        variables_plot = variables_list
        vals_trend_plot = vals_trend_list
        vals_CV_plot = vals_CV_list

        # plt.scatter(vals_CV_list,vals_trend_list,marker=marker_list,color=color_list[0],s=100)
        # plt.show()
        ##plot error bar
        # plt.figure(figsize=(self.map_width, self.map_height))

        plt.figure(figsize=(13 * centimeter_factor, 10 * centimeter_factor))

        # self.map_width = 13 * centimeter_factor
        # self.map_height = 8.2 * centimeter_factor

        err_trend_list = np.array(err_trend_list)
        err_CV_list = np.array(err_CV_list)
        for i, (x, y, marker, color, var, mark_size) in enumerate(
                zip(vals_trend_plot, vals_CV_plot, marker_list, color_list, variables_plot, mark_size_list)):
            plt.scatter(y, x, marker=marker, color=color_list[i], label=var, s=mark_size, edgecolors='black', )
            # plt.errorbar(y, x, xerr=err_trend_list[i], yerr=err_CV_list[i], fmt='none', color='grey', capsize=2, capthick=0.3,alpha=1)

            ##markerborderwidth=1

            plt.ylabel('Trends in LAI (%/yr)', fontsize=12)
            plt.xlabel('Trends in CVLAI (%/yr)', fontsize=12)
            plt.ylim(-0.3, .9)
            plt.xlim(-0.2, 0.5)
            plt.xticks(fontsize=12)
            ## xticks gap 0.05
            plt.yticks(np.arange(-0.2, .9, 0.2), fontsize=12)
            plt.yticks(fontsize=12)
            # plt.legend()
        ## save imagine
        plt.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
        plt.axvline(x=0.0, color='k', linestyle='--', linewidth=1)
        plt.savefig(result_root + rf'\FIGURE\\Figure3\\obs_TRENDY_CV_trends_median.pdf',  bbox_inches='tight')

        #
        # plt.show()




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

    def heatmap(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_sensitivity_trend = result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\npy_time_series\trend\\sum_rainfall_sensitivity_trend.tif'
        f_rainfall_trend=result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\ecosystem_year\trend\\\sum_rainfall_trend.tif'
        f_CVLAI=result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        outf = result_root + rf'\3mm\FIGURE\\heatmap2.pdf'
        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_CVLAI)

        arr_LAI_trend[arr_LAI_trend < -999] = np.nan

        arr_LAI_sensitivity_precip, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_sensitivity_trend)
        arr_precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_rainfall_trend)
        arr_LAI_sensitivity_precip[arr_LAI_sensitivity_precip < -999] = np.nan
        arr_precip_trend[arr_precip_trend < -999] = np.nan
        arr_LAI_trend=np.array(arr_LAI_trend)
        arr_LAI_sensitivity_precip=np.array(arr_LAI_sensitivity_precip)
        arr_precip_trend=np.array(arr_precip_trend)

        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        dic_arr_LAI_sensitivity_precip=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_sensitivity_precip)
        dic_precip_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_precip_trend)

        result_dic={
            'LAI_CV':dic_LAI_trend,
            'LAI_sensitivity_precip_trend':dic_arr_LAI_sensitivity_precip,
            'Preci_trend':dic_precip_trend
        }
        # plt.hist(result_dic['LAI_CV'].values())
        # plt.show()
        # plt.hist(result_dic['LAI_sensitivity_precip_trend'].values())
        # plt.show()
        # plt.hist(result_dic['Preci_trend'].values())
        # plt.show();exit()
        df=T.spatial_dics_to_df(result_dic)
        T.print_head_n(df)
        x_var = 'LAI_sensitivity_precip_trend'
        y_var = 'Preci_trend'
        z_var = 'LAI_CV'
        # bin_x = [ -0.6,-0.4,-0.2,0,0.2,0.4,0.6,]
        bin_x = np.linspace(-0.5, 1.5, 11)
        # bin_y = [ -4, -3, -2, -1, 0, 1, 2, 3, 4, ]
        bin_y = np.linspace(-3, 3, 13)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure(figsize=(self.map_width, self.map_height))

        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)

        my_cmap = T.cmap_blend(color_list = ['#000000','r', 'b'])
        my_cmap = 'GnBu'
        self.plot_df_bin_2d_matrix(matrix_dict,-1,1,x_ticks_list,y_ticks_list,cmap=my_cmap,
                              is_only_return_matrix=False)
        plt.colorbar()
        pprint(matrix_dict)


        matrix_dict_count, x_ticks_list, y_ticks_list = self.df_bin_2d_count(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        pprint(matrix_dict_count)
        scatter_size_dict = {
            (1,20): 5,
            (20,50): 20,
            (50,100): 50,
            (100,np.inf): 100
        }
        matrix_dict_count_normalized = {}
        # Normalize counts for circle size
        for key in matrix_dict_count:
            num = matrix_dict_count[key]
            for key2 in scatter_size_dict:
                if num >= key2[0] and num < key2[1]:
                    matrix_dict_count_normalized[key] = scatter_size_dict[key2]
                    break
        pprint(matrix_dict_count_normalized)
        reverse_x = list(range(len(bin_y)-1))[::-1]
        reverse_x_dict = {}
        for i in range(len(bin_y)-1):
            reverse_x_dict[i] = reverse_x[i]
        # print(reverse_x_dict);exit()
        for x,y in matrix_dict_count_normalized:
            plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='gray',edgecolors='none',alpha=.5)
        for x,y in matrix_dict_count_normalized:
            plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='none',edgecolors='gray',alpha=1)

        plt.xlabel('beta')
        plt.ylabel('Trend in Rainfall (mm/yr)')

        # plt.show()
        plt.savefig(outf)
        plt.close()




    #     plt.savefig(result_root + rf'Data_frame\\Frequency\\Trendy_{region}.pdf', dpi=300, )
    #     plt.close()


    def heatmap_count(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_sensitivity_trend = result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\npy_time_series\trend\\sum_rainfall_sensitivity_trend.tif'
        f_rainfall_trend=result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\ecosystem_year\trend\\\sum_rainfall_trend.tif'
        f_CVLAI=result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'

        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_CVLAI)

        arr_LAI_trend[arr_LAI_trend < -999] = np.nan

        arr_LAI_sensitivity_precip, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_sensitivity_trend)
        arr_precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_rainfall_trend)
        arr_LAI_sensitivity_precip[arr_LAI_sensitivity_precip < -999] = np.nan
        arr_precip_trend[arr_precip_trend < -999] = np.nan
        arr_LAI_trend=np.array(arr_LAI_trend)
        arr_LAI_sensitivity_precip=np.array(arr_LAI_sensitivity_precip)
        arr_precip_trend=np.array(arr_precip_trend)

        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        dic_arr_LAI_sensitivity_precip=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_sensitivity_precip)
        dic_precip_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_precip_trend)

        result_dic={
            'LAI_CV':dic_LAI_trend,
            'LAI_sensitivity_precip_trend':dic_arr_LAI_sensitivity_precip,
            'Preci_trend':dic_precip_trend
        }
        # plt.hist(result_dic['LAI_CV'].values())
        # plt.show()
        # plt.hist(result_dic['LAI_sensitivity_precip_trend'].values())
        # plt.show()
        # plt.hist(result_dic['Preci_trend'].values())
        # plt.show();exit()
        df=T.spatial_dics_to_df(result_dic)
        T.print_head_n(df)
        x_var = 'LAI_sensitivity_precip_trend'
        y_var = 'Preci_trend'
        z_var = 'LAI_CV'
        # bin_x = [ -0.6,-0.4,-0.2,0,0.2,0.4,0.6,]
        bin_x = np.linspace(-0.7, 1.5, 13)
        # bin_y = [ -4, -3, -2, -1, 0, 1, 2, 3, 4, ]
        bin_y = np.linspace(-3, 3, 13)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()

        matrix_dict,x_ticks_list,y_ticks_list = self.df_bin_2d_count(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)

        self.plot_df_bin_2d_matrix(matrix_dict,0,200,x_ticks_list,y_ticks_list,
                              is_only_return_matrix=False)

        plt.xlabel('beta')
        plt.ylabel('Trend in Rainfall (mm/yr)')

        plt.colorbar()
        plt.show()


    def df_bin_2d_count(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
        df_group_y, _ = self.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = self.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict,x_ticks_list,y_ticks_list

    def df_bin(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'],observed=True)
        bins_name = df_group.groups.keys()
        bins_name_list = list(bins_name)
        bins_list_str = [str(i) for i in bins_name_list]
        # for name,df_group_i in df_group:
        #     vals = df_group_i[col].tolist()
        #     mean = np.nanmean(vals)
        #     err,_,_ = self.uncertainty_err(SM)
        #     # x_list.append(name)
        #     y_list.append(mean)
        #     err_list.append(err)
        return df_group, bins_list_str

    def plot_df_bin_2d_matrix(self,matrix_dict,vmin,vmax,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False):
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])

    def plot_sign_between_LAI_NDVI(self):

        import xymap
        tif_rainfall = result_root + rf'3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        tif_greening = result_root + rf'3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\NDVI_detrend_CV_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\CV_LAI4g_NDVI.tif'
        T.mk_dir(result_root + rf'bivariate_analysis\\')
        tif1 = tif_rainfall
        tif2 = tif_greening

        dic1 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif1)
        dic2 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif2)
        dics = {'LAI4g_CV_trend': dic1,
                'NDVI_CV_trend': dic2}
        df = T.spatial_dics_to_df(dics)
        # print(df)
        df['is_increasing_LAI4g'] = df['LAI4g_CV_trend'] > 0
        df['is_increasing_NDVI'] = df['NDVI_CV_trend'] > 0
        print(df)
        label_list = []
        for i, row in df.iterrows():
            if row['is_increasing_LAI4g'] and row['is_increasing_NDVI']:
                label_list.append(1)
            elif row['is_increasing_LAI4g'] and not row['is_increasing_NDVI']:
                label_list.append(2)
            elif not row['is_increasing_LAI4g'] and row['is_increasing_NDVI']:
                label_list.append(3)
            else:
                label_list.append(4)

        df['label'] = label_list
        result_dic = T.df_to_spatial_dic(df, 'label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(result_dic, outtif)

        ## NDVI and LAI CV showing sign:
        pass

def rename():
    fdir=rf'D:\Project3\Result\3mm\moving_window_multi_regression\TRENDY\multiresult_relative_change_detrend_ecosystem_year\\'

    variable_list=['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
    'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
    'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
    'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
    'ORCHIDEE_S2_lai',

    'YIBs_S2_Monthly_lai']
    outdir=result_root+rf'\3mm\moving_window_multi_regression\\TRENDY\\multiresult_relative_change_detrend_ecosystem_year\\trend\\'
    T.mk_dir(outdir,force=True)

    for variable in variable_list:
        fdiri=join(fdir, variable,'trend')
        fpath=join(fdiri, 'sum_rainfall_detrend_trend.tif')

        fnew_name=variable+'_beta_trend.tif'
        outf=join(outdir, fnew_name)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, outf)




class check_data_distribution():
    def __init__(self):
        pass
    def run(self):
        # self.check_variables_ranges()
        self.plot_time_series()

    def check_variables_ranges(self):
        self.this_class_arr = (
                result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\Dataframe\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'statistics.df'

        dff = self.dff
        df = T.load_df(dff)
        df = self.df_clean(df)
        var_list=['sand_soil_grid','rooting_depth_05','cwdx80_05_soil_grid','Burn_area_sum',
                  'pi_growing_season_trend','SM_ecosystem_year_trend','sum_rainfall_trend']
        for var in var_list:
            vals = df[var].tolist()
            plt.hist(vals, bins=100)
            plt.title(var)
            plt.show()




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
    def plot_time_series(self):
        f=r'D:\Project3\Result\3mm\moving_window_multi_regression\moving_window\window_detrend_growing_season\\VPD_detrend.npy'
        dic=T.load_npy(f)
        for pix in dic:
            time_series=dic[pix]
            plt.plot(time_series)
            plt.show()
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








def main():
     # Data_processing_2().run()
    # # Phenology().run()
    build_dataframe().run()
    # build_moving_window_dataframe().run()

    # CO2_processing().run()
    # greening_analysis().run()
    # TRENDY_trend().run()
    # TRENDY_CV().run()
    # multi_regression_beta().run()
    # multi_regression_temporal_patterns().run()
    # bivariate_analysis().run()
    # visualize_SHAP().run()
    # PLOT_dataframe().run()
    # Plot_Robinson().robinson_template()
    # rename()
    # check_data_distribution().run()







if __name__ == '__main__':
    main()