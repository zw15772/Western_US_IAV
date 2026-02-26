import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
D=DIC_and_TIF(tif_template=tif_template)



class build_dataframe():


    def __init__(self):

        self.this_class_arr = (
                result_root +  rf'Terraclimate\SPEI\SPEI_12_NOAA\extreme_events\\')

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'wet_events_annual.df'


        pass

    def run(self):


        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        # df=self.foo2(df)

        # df=self.build_df(df)
        # df=self.build_df_monthly(df)
        # df=self.append_attributes(df)  ## 加属性
        # df=self.append_cluster(df)  ## 加属性
        # df=self.append_value(df)   ## insert or append value


        # df = self.add_detrend_zscore_to_df(df)

        # df=self.add_trend_to_df(df)
        # df=self.add_phenology_type_to_df(df)

        # df=self.add_mean_to_df(df)


        # # #
        # df=self.add_aridity_to_df(df)
        # df=self.add_dryland_nondryland_to_df(df)
        # df=self.add_MODIS_LUCC_to_df(df)
        # df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        # df=self.add_landcover_classfication_to_df(df)
        # # # # # # # # # # df=self.dummies(df)
        # df=self.add_maxmium_LC_change(df)
        df=self.add_row(df)
        df=self.add_Ecoregion_level_II_raster_to_df(df)
        # # # # # # # # # # # # # #
        df=self.add_lat_lon_to_df(df)
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
        f=result_root+rf'\greening_analysis\relative_change\\SNU_LAI.npy'



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
        fname=f.split('.')[0]


        df[fname] = change_rate_list
        return df

    def foo2(self, df):  # 新建trend

        f = result_root + rf'\greening_analysis\relative_change\trend\\SNU_LAI_trend.tif'
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



    def add_detrend_zscore_to_df(self, df):

        fdir=data_root+rf'\Terraclimate\SPEI\SPEI_12_NOAA\extract_growing_season_SPEI12_mean\\'


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

                vals = val_dic[pix]['growing_season']
                # print(vals)
                print(len(vals))

                ##### if len vals is 38, the end of list add np.nan

                # if len(vals) == 19:
                #     ##creast 19 nan
                #     nan_list = np.array([np.nan] * 19)
                #     vals=np.append(nan_list,vals)
                # if len(vals)==33 :
                #     nan_list=np.array([np.nan]*5)
                #     vals=np.append(vals,nan_list)

                if len(vals)==42:
                    vals = np.append(vals,np.nan)


                v1= vals[year - 1982]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}'] = NDVI_list
        # exit()
        return df


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
    def add_Ecoregion_level_II_raster_to_df(self, df):
        tiff =data_root+rf'\basedata\Ecoregion\\Ecoregion_level_II_raster.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'Ecoregion_level_II'

        dic_convert={10.1:'Cold Desert',6.2:'Western Cordillera',
                     9.4:'South Central Semiarid Prairies',
                     9.3:'West-Central Semiarid Prairies',
                     10.2:'Warm Desert', 11.1:'Mediterranean California',
                     13.1:'Upper Gila Mountains', 13.2:'Western Sierra Madre',
                     12.1:'Western Sierra Madre Piedmont',
                     7.1:'Marine West Coast Forest',
                     14.3:'Western Pacific Coastal Plain, Hills and Canyons',
                     -9999: np.nan
       }

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            # print(val)
            val=round(val,1)

            # print(val);exit()

            val_convert=dic_convert[val]

            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val_convert)
        df[f_name] = val_list
        return df

    pass

    def add_continent_to_df(self, df):
        tiff =data_root+rf'\basedata\Ecoregion\\Ecoregion_level_II_raster.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'Ecoregion_level_II'

        dic_convert={1:'Africa',2:'Asia',3:'Australia',4: np.nan, 5:'South_America', 6: np.nan, 7:'Europe',8:'North_America',-9999: np.nan}

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
        T.add_lon_lat_to_df(df,D)
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




    def add_phenology_type_to_df(self, df):
        f = data_root+rf'/basedata/Phenology_extraction/SeasType.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)

        val_dic = D.spatial_arr_to_dic(array)


        f_name ='SeasType'
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

    def add_trend_to_df(self, df):
        fdir = result_root + rf'\Terraclimate\SPEI\SPEI_12_NOAA\trend\\'


        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'growing_season_SPEI12_mean_p_value' in f:
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


    def rename_columns(self, df):
        df = df.rename(columns={rf'growing_season_SPEI12_mean_trend': 'SNU_LAI_relative_change',









                               }


                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=['category_9_percentile5',
                              'category_9_percentile95',

















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
class build_moving_window_dataframe():
    def __init__(self):

        self.this_class_arr = (
                    result_root +  rf'/IAV_analysis/Dataframe/')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'Dataframe.df'
    def run(self):
        df = self.__gen_df_init(self.dff)
        # df=self.build_df(df)
        # self.append_value(df)
        # df=self.append_attributes(df)
        # df=self.add_trend_to_df(df)
        # df=self.foo1(df)
        # df=self.add_window_to_df(df)
        # df=self.add_phenology_type_to_df(df)
        # df=self.add_row(df)
        df=self.add_lat_lon_to_df(df)



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

        f =result_root+ rf'IAV_analysis/moving_window_CV_extraction_anaysis/growing_season_LAI_mean_detrend_CV.npy'
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

        df['SNU_LAI_CV'] = change_rate_list
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

    def add_phenology_type_to_df(self, df):
        f = data_root+rf'/basedata/Phenology_extraction/SeasType.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)

        val_dic = D.spatial_arr_to_dic(array)


        f_name ='SeasType'
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

    def add_phenology_type_to_df(self, df):
        f = data_root+rf'/basedata/Phenology_extraction/SeasType.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)

        val_dic = D.spatial_arr_to_dic(array)


        f_name ='SeasType'
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

    def add_lat_lon_to_df(self, df):
        T.add_lon_lat_to_df(df, D)
        return df

    def add_row(self, df):
        r_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix
            r_list.append(r)
        df['row'] = r_list
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

class check_Data:
    def __init__(self):
        pass
    def run(self):
        self.spatial_plot()
    def spatial_plot(self):

        fdir=rf'D:\Western_US_IAV\Data\Terraclimate\SPEI\SPEI_12_NOAA\extract_growing_season_SPEI12_mean\\'
        spatial_len={}

        for f in os.listdir(fdir):
            dic=T.load_npy(fdir+f)
            for pix in dic:
                vals=dic[pix]['growing_season']
                length=len(vals)
                spatial_len[pix]=length
            array=D.pix_dic_to_spatial_arr(spatial_len)
            plt.imshow(array)
            plt.show()




def main ():
    build_dataframe().run()
    # build_moving_window_dataframe().run()
    # check_Data().run()
    pass

if __name__ == '__main__':
    main()