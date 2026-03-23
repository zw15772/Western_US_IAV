import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
D=DIC_and_TIF(tif_template=tif_template)
from datetime import datetime

import numpy as np
from scipy.signal import savgol_filter, find_peaks

class Phenology_extraction:
    def __init__(self,):
        pass
    def run(self):
        self.extract_phenology_run()

    def extract_phenology_run(self):
        dic= T.load_npy_dir(data_root + rf'\MODIS_LAI\LAI_climatology\dic\\')

        class_dic = {}

        sos1_dic = {}
        eos1_dic = {}


        sos2_dic = {}
        eos2_dic = {}
        result_dic={}


        for pix, ts in dic.items():

            if ts is None or np.isnan(ts).all():
                continue

            result = self.phenology_extract(ts)
            result_dic[pix]= result

            cls = result["class"]
            class_dic[pix] = cls

            # -------- season1 一定存在 --------
            s1 = result["season1"]

            sos1_dic[pix] = s1["sos"]
            eos1_dic[pix] = s1["eos"]

            # -------- season2 只有 class 2 才有 --------
            if cls == 2 and result["season2"] is not None:

                s2 = result["season2"]

                sos2_dic[pix] = s2["sos"]
                eos2_dic[pix] = s2["eos"]


            else:
                sos2_dic[pix] = np.nan
                eos2_dic[pix] = np.nan



        arr_class = D.pix_dic_to_spatial_arr(class_dic)

        arr_sos1 = D.pix_dic_to_spatial_arr(sos1_dic)
        arr_eos1 = D.pix_dic_to_spatial_arr(eos1_dic)


        arr_sos2 = D.pix_dic_to_spatial_arr(sos2_dic)
        arr_eos2 = D.pix_dic_to_spatial_arr(eos2_dic)

        outdir= data_root + rf'\MODIS_LAI\phenology_metrics\\'
        T.mk_dir(outdir,force=True)
        D.arr_to_tif(arr_class, outdir + r'classification.tif')
        D.arr_to_tif(arr_sos1, outdir + r'sos1.tif')
        D.arr_to_tif(arr_eos1, outdir + r'eos1.tif')

        D.arr_to_tif(arr_sos2, outdir + r'sos2.tif')
        D.arr_to_tif(arr_eos2, outdir + r'eos2.tif')


        T.save_npy(result_dic, outdir + r'phenology_result.npy')










    ## main function



    def is_peak_isolated(self,ts, peak_idx, threshold):

        n = len(ts)

        # 向左找 valley
        left = peak_idx
        found_left_valley = False
        for _ in range(n):
            left = (left - 1) % n
            if ts[left] <= threshold:
                found_left_valley = True
                break

        # 向右找 valley
        right = peak_idx
        found_right_valley = False
        for _ in range(n):
            right = (right + 1) % n
            if ts[right] <= threshold:
                found_right_valley = True
                break

        return found_left_valley and found_right_valley



    def smooth_series(self, ts):

        ts_interp = ts.copy()
        nan_mask = np.isnan(ts_interp)

        if nan_mask.any():
            ts_interp[nan_mask] = np.interp(
                np.flatnonzero(nan_mask),
                np.flatnonzero(~nan_mask),
                ts_interp[~nan_mask]
            )

        ts_smooth = savgol_filter(ts_interp, 21, 3, mode='wrap')
        return ts_smooth

    def extract_one_season(self,ts, peak_idx):

        n = len(ts)
        A = np.nanmax(ts) - np.nanmin(ts)
        threshold = np.nanmin(ts) + 0.2 * A

        # 向左找 SOS
        left = peak_idx
        for _ in range(n):
            if ts[left] <= threshold:
                break
            left = (left - 1) % n

        # 向右找 EOS
        right = peak_idx
        for _ in range(n):
            if ts[right] <= threshold:
                break
            right = (right + 1) % n

        return {
            "sos": left,
            "peak": peak_idx,
            "eos": right
        }

    # =========================
    # 提取 GS 指标
    # =========================
    def compute_gs_metrics(self,ts, season):

        sos = season["sos"]
        eos = season["eos"]

        if sos < eos:
            gs = ts[sos:eos]
        else:
            gs = np.concatenate([ts[sos:], ts[:eos]])

        return {
            "length": len(gs),
            "mean": np.nanmean(gs),
            "integral": np.nansum(gs),
            "max": np.nanmax(gs),
            "min": np.nanmin(gs)
        }
    # =========================
    # 主函数
    # =========================
    def phenology_extract(self,ts):
        ## class = 1 → evergreen
        # class = 2 → two seasons
        # class = 3 → single season

        ts = self.smooth_series(ts)
        n = len(ts)

        A = np.nanmax(ts) - np.nanmin(ts)
        # Evergreen

        if A < 0.1 and np.nanmean(ts) > 0.5:

            season1 = {

                "sos": 0,
                "peak": np.argmax(ts),
                "eos": n - 1
            }
            # plt.plot(ts)
            # plt.title('class 1')
            # plt.show()

            season1.update(self.compute_gs_metrics(ts, season1))

            return {
                "class": 1,
                "season1": season1,
                "season2": None
            }
            # -------------------
            # 找 peaks
            # -------------------
        peaks, _ = find_peaks(
            ts,
            prominence=0.1 * A,
            distance=15
        )


        # 没峰 → 单季全年
        if len(peaks) == 0:
            season1 = {
                "sos": 0,
                "peak": np.argmax(ts),
                "eos": n - 1
            }

            season1.update(self.compute_gs_metrics(ts, season1))

            return {
                "class": 3,
                "season1": season1,
                "season2": None
            }

        # 单峰
        if len(peaks) == 1:
            season1 = self.extract_one_season(ts, peaks[0])
            season1.update(self.compute_gs_metrics(ts, season1))
            # plt.plot(ts)
            # plt.title('class 3')
            # plt.show()

            return {
                "class": 3,
                "season1": season1,
                "season2": None
            }

        # -------------------
        # 判断双季
        # -------------------
        peak_vals = ts[peaks]
        idx = np.argsort(peak_vals)[::-1]

        p1 = peaks[idx[0]]
        p2 = peaks[idx[1]]

        left = min(p1, p2)
        right = max(p1, p2)

        valley = np.min(ts[left:right])

        cond_valley = (
                (ts[p1] - valley > 0.2 * A) and
                (ts[p2] - valley > 0.2 * A)
        )

        cond_distance = abs(p1 - p2) > 20

        threshold = np.nanmin(ts) + 0.2 * A

        # 判断两个峰是否都被 valley 包围
        isolated1 = self.is_peak_isolated(ts, p1, threshold)
        isolated2 =self.is_peak_isolated(ts, p2, threshold)

        if cond_valley and cond_distance and isolated1 and isolated2:

            s1 = self.extract_one_season(ts, p1)
            s2 = self.extract_one_season(ts, p2)

            seasons = sorted([s1, s2], key=lambda x: x["peak"])
            season1 = seasons[0]
            season2 = seasons[1]

            season1.update(self.compute_gs_metrics(ts, season1))
            season2.update(self.compute_gs_metrics(ts, season2))
            # plt.plot(ts)
            # plt.title('class 2')
            # plt.show()

            return {
                "class": 2,
                "season1": season1,
                "season2": season2
            }

        else:
            # plt.title('single_season')
            # plt.plot(ts)
            # plt.show()

            season1 = self.extract_one_season(ts, p1)
            season1.update(self.compute_gs_metrics(ts, season1))

            return {
                "class": 3,
                "season1": season1,
                "season2": None
            }

class Climatology_builder:
    def __init__(self):
        pass

    def run(self):
        # self.build_climatology()
        self.tif_to_dic()
        pass

    def build_climatology(self):
        from collections import defaultdict

        fdir = data_root + r'\MODIS_LAI\extract_tif_scaled\\'

        year_dic = defaultdict(list)

        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue

            year = f[:4]
            year_dic[year].append(f)
        # 打印每年数量
        print("Year counts:")
        for y in sorted(year_dic):
            print(y, len(year_dic[y]))

        template_doy = self.build_template(year_dic)

        all_years = []
        profile = None

        for y in sorted(year_dic):

            stack = self.align_one_year(fdir, y, year_dic[y], template_doy)

            if profile is None:
                _, profile = self.read_tif(os.path.join(fdir, year_dic[y][0]))

            all_years.append(stack)

        all_years = np.stack(all_years, axis=0)
        # shape: (years, time, H, W)

        climatology = np.nanmean(all_years, axis=0)
        # shape: (time, H, W)
        for i in range(climatology.shape[0]):
            out_path = data_root + rf'\MODIS_LAI\LAI_climatology\clim_{i+1:03d}.tif'
            arr= climatology[i]
            arr[np.isnan(arr)] = -9999
            D.arr_to_tif(arr, out_path)



    def build_template(self,year_dic):
            from datetime import datetime


        # 选一个时间最完整的年份
            best_year = max(year_dic.keys(), key=lambda y: len(year_dic[y]))

            template_doy = []

            for f in sorted(year_dic[best_year]):
                date = datetime.strptime(f[:8], "%Y%m%d")
                template_doy.append(date.timetuple().tm_yday)

            return template_doy

    def read_tif(self,path):

        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile

        arr[arr < -9990] = np.nan
        return arr, profile

    def align_one_year(self,folder, year, files, template_doy):
        from datetime import datetime

        year_files = sorted(files)
        year_doy = []

        for f in year_files:
            date = datetime.strptime(f[:8], "%Y%m%d")
            year_doy.append(date.timetuple().tm_yday)

        aligned_stack = []

        for doy in template_doy:

            if doy in year_doy:
                idx = year_doy.index(doy)
                arr, _ = self.read_tif(os.path.join(folder, year_files[idx]))
            else:
                # 如果缺失 → 填 NaN
                arr, _ = self.read_tif(os.path.join(folder, year_files[0]))
                arr[:] = np.nan

            aligned_stack.append(arr)

        aligned_stack = np.stack(aligned_stack, axis=0)

        return aligned_stack

    def tif_to_dic(self):
        fdir = data_root + r'\MODIS_LAI\LAI_climatology\\'
        outdir = data_root + rf'\MODIS_LAI\dic\\'
        T.mk_dir(outdir, force=True)

        all_array = []  #### so important  it should be go with T.mk_dic

        for f in T.listdir(fdir):
            print(f)

            if not f.endswith('.tif'):
                continue

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir, f))
            array = np.array(array, dtype=float)

            # array_unify = array[:720][:720,
            #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
            # array_unify = array[:3600][:3600,
            #               :7200]

            array[array < -999] = np.nan
            # array_unify[array_unify > 10] = np.nan
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


class check_data:
    def __init__(self):
        pass
    def run(self):
            self.plot_time_series()

    def plot_time_series(self):
        fdir = data_root + rf'\MODIS_LAI\dic\\'
        dic = T.load_npy_dir(fdir)
        for pix in dic:
            vals = dic[pix]

            if np.isnan(np.nanmean(vals)):
                continue
            print(len(vals))
            time_series = dic[pix]
            time_series = dic[pix]
            plt.plot(time_series)
            plt.show()

class Extraction_annual_growing_season:
    def __init__(self):
        pass
    def run(self):
        self.run_pipeline()


    def run_pipeline(self):

        outdir = data_root + r'\MODIS_LAI\dic\\'
        fdir = data_root + r'\MODIS_LAI\extract_tif_scaled\\'

        file_list = sorted([f for f in os.listdir(fdir) if f.endswith('.tif')])

        year_stack = {}

        # -----------------------------
        # 1️⃣ 构建 year_stack
        # -----------------------------
        for f in tqdm(file_list):

            date = datetime.strptime(f[:8], "%Y%m%d")
            year = date.year

            path = os.path.join(fdir, f)
            arr, *_ = ToRaster().raster2array(path)

            if year not in year_stack:
                year_stack[year] = []

            year_stack[year].append(arr.astype(np.float32))

        # -----------------------------
        # 2️⃣ stack + 插值（每年只做一次）
        # -----------------------------
        for year in tqdm(year_stack):

            stack = np.stack(year_stack[year])  # (time,row,col)

            if stack.shape[0] != 92:
                stack = self.interp_time_to_92(stack)

            year_stack[year] = stack

        # -----------------------------
        # 3️⃣ 加载 phenology
        # -----------------------------
        phenology_dic = T.load_npy(
            data_root + r'\MODIS_LAI\phenology_metrics\phenology_result.npy'
        )

        # -----------------------------
        # 4️⃣ 构建 annual GS
        # -----------------------------
        annual_gs_dic = self.build_annual_gs_dic(
            year_stack,
            phenology_dic
        )

        # -----------------------------
        # 5️⃣ 保存
        # -----------------------------
        os.makedirs(outdir, exist_ok=True)

        np.save(os.path.join(outdir, "annual_gs_dic_mean.npy"), annual_gs_dic)



    # =========================
    # 插值函数（时间维）
    # =========================
    def interp_time_to_92(self, data, min_len=85, min_valid_ratio=0.7):

        t, rows, cols = data.shape

        # 时间长度太短 → 丢弃整年
        if t < min_len:
            return None

        if t == 92:
            return data

        data_2d = data.reshape(t, -1)

        x_old = np.linspace(0, 1, t)
        x_new = np.linspace(0, 1, 92)

        data_interp = np.empty((92, data_2d.shape[1]), dtype=np.float32)

        for i in range(data_2d.shape[1]):

            ts = data_2d[:, i]

            valid_ratio = np.sum(~np.isnan(ts)) / t

            if valid_ratio < min_valid_ratio:
                data_interp[:, i] = np.nan
                continue

            if np.isnan(ts).all():
                data_interp[:, i] = np.nan
                continue

            if np.isnan(ts).any():
                mask = ~np.isnan(ts)
                ts = np.interp(x_old, x_old[mask], ts[mask])

            data_interp[:, i] = np.interp(x_new, x_old, ts)

        return data_interp.reshape(92, rows, cols)

    def build_annual_gs_dic(self, year_stack, phenology_dic):

        year_list = sorted(year_stack.keys())
        annual_gs_dic = {}

        # 获取行列数
        example_year = year_list[0]
        _, rows, cols = year_stack[example_year].shape

        for pix, result in tqdm(phenology_dic.items()):



            s1_series = []
            s2_series = []

            cls = result["class"]

            for year in year_list:
                row, col = pix



                data = year_stack[year]  # 已经是 92
                ts = data[:, row, col]


                if np.isnan(ts).all():
                    s1_series.append(np.nan)
                    s2_series.append(np.nan)
                    continue

                # -------- season1 --------
                s1 = result["season1"]
                sos1 = int(s1["sos"])
                eos1 = int(s1["eos"])

                if sos1 < eos1:
                    gs1 = ts[sos1:eos1]
                else:
                    gs1 = np.concatenate([ts[sos1:], ts[:eos1]])

                s1_series.append(np.nanmean(gs1))
                # s1_series.append(np.nanmax(gs1))

                # -------- season2 --------
                if cls == 2 and result["season2"] is not None:

                    s2 = result["season2"]
                    sos2 = int(s2["sos"])
                    eos2 = int(s2["eos"])

                    if sos2 < eos2:
                        gs2 = ts[sos2:eos2]
                    else:
                        gs2 = np.concatenate([ts[sos2:], ts[:eos2]])

                    s2_series.append(np.nanmean(gs2))
                    # s2_series.append(np.nanmax(gs2))
                else:
                    s2_series.append(np.nan)

            annual_gs_dic[pix] = {
                "season1": np.array(s1_series, dtype=np.float32),
                "season2": np.array(s2_series, dtype=np.float32)
            }

        return annual_gs_dic





        pass

class extraction_calendar_year_LAI:  ## this is test, no growing season just annual year

    def run(self):
        # self.run_interp_time_to_92()
        # self.tif_to_dic()

        # self.relative_change()
        self.trend_analysis()
        pass
    def run_interp_time_to_92(self):

        outdir = data_root + r'\MODIS_LAI\calendar_year\\'
        T.mk_dir(outdir, force=True)
        fdir = data_root + r'\MODIS_LAI\extract_tif_scaled\\'

        file_list = sorted([f for f in os.listdir(fdir) if f.endswith('.tif')])

        year_stack = {}

        # -----------------------------
        # 1️⃣ 构建 year_stack
        # -----------------------------
        for f in tqdm(file_list):

            date = datetime.strptime(f[:8], "%Y%m%d")
            year = date.year

            path = os.path.join(fdir, f)
            arr, *_ = ToRaster().raster2array(path)

            if year not in year_stack:
                year_stack[year] = []

            year_stack[year].append(arr.astype(np.float32))

        # -----------------------------
        # 2️⃣ stack + 插值（每年只做一次）
        # -----------------------------
        for year in tqdm(year_stack):

            stack = np.stack(year_stack[year])  # (time,row,col)

            if stack.shape[0] != 92:
                stack = self.interp_time_to_92(stack)

            year_stack[year] = stack


        # for year in year_stack:
        #
        #     calendar_year_LAI=np.nanmax(year_stack[year],axis=0)
        #     out_path = data_root + rf'\MODIS_LAI\calendar_year\{year:03d}.tif'
        #     arr= calendar_year_LAI
        #     arr[np.isnan(arr)] = -9999
        #     D.arr_to_tif(arr, out_path)
        #     plt.imshow(calendar_year_LAI)

        for year in year_stack:

            gs_LAI=np.nanmax(year_stack[year][30:77],axis=0)   ## May 1st to oct 31
            out_path = data_root + rf'\MODIS_LAI\growing_season\{year:03d}.tif'
            arr= gs_LAI
            arr[np.isnan(arr)] = -9999
            D.arr_to_tif(arr, out_path)
            plt.imshow(gs_LAI)


        # -----------------------------


    def interp_time_to_92(self, data, min_len=85, min_valid_ratio=0.7):

        t, rows, cols = data.shape

        # 时间长度太短 → 丢弃整年
        if t < min_len:
            return None

        if t == 92:
            return data

        data_2d = data.reshape(t, -1)

        x_old = np.linspace(0, 1, t)
        x_new = np.linspace(0, 1, 92)

        data_interp = np.empty((92, data_2d.shape[1]), dtype=np.float32)

        for i in range(data_2d.shape[1]):

            ts = data_2d[:, i]

            valid_ratio = np.sum(~np.isnan(ts)) / t

            if valid_ratio < min_valid_ratio:
                data_interp[:, i] = np.nan
                continue

            if np.isnan(ts).all():
                data_interp[:, i] = np.nan
                continue

            if np.isnan(ts).any():
                mask = ~np.isnan(ts)
                ts = np.interp(x_old, x_old[mask], ts[mask])

            data_interp[:, i] = np.interp(x_new, x_old, ts)

        return data_interp.reshape(92, rows, cols)


    def tif_to_dic(self):

        fdir_all = data_root + rf'\MODIS_LAI\growing_season\tiff_mean\\'
        outdir=data_root + '\MODIS_LAI\cgrowing_season\dic_mean\\'
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
            # array_unify[array_unify > 10] = np.nan
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



    def relative_change(self):

        fdir=data_root + r'\MODIS_LAI\growing_season\dic_mean\\'
        outdir = result_root + rf'greening_analysis\MODIS_LAI\\growing_season\\relative_change\\'
        Tools().mk_dir(outdir, force=True)

        outf = outdir + 'MODIS_LAI_growing_season_mean.npy'
        # print(outf);exit()


        dic = T.load_npy_dir(fdir)

        zscore_dic = {}

        for pix in tqdm(dic):



            # print(len(dic[pix]))
            time_series = dic[pix]


            time_series = np.array(time_series)


            # print(len(time_series));exit()

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

    def trend_analysis(self):

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        ##each window average trend


        fdir = result_root + r'greening_analysis/MODIS_LAI/growing_season/relative_change/'
        outdir = result_root + r'greening_analysis/MODIS_LAI/growing_season/relative_change//trend/'
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
            fpath=result_root+rf'\greening_analysis\MODIS_LAI\phenology_growing_Season\trend\MODIS_LAI_max_season1_trend.tif'
            ll,lr,ul,ur=RasterIO_Func().get_tif_bounds(fpath)
            print(ll,lr,ul,ur)

            ax = plt.axes(projection=ccrs.PlateCarree())

            # --- 画趋势图 ---
            im = ax.imshow(
                arr_trend,
                cmap='RdBu',
                vmin=-1,
                vmax=1,
                extent=[-124.55, -102.04, 25.59,49],
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
            cbar.set_label('Trend (relative_change, %)')

            plt.title(f)
            plt.show()

            # D.arr_to_tif(arr_trend, outf + '_trend.tif')
            # D.arr_to_tif(p_value_arr, outf + '_p_value.tif')
            #
            # np.save(outf + '_trend', arr_trend)
            # np.save(outf + '_p_value', p_value_arr)








def main():

   # Climatology_builder().run()
   # Phenology_extraction().run()
   # Extraction_annual_growing_season().run()
   extraction_calendar_year_LAI().run()
   # check_data().run()




if __name__ == '__main__':
    main()