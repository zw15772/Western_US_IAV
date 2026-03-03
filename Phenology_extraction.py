import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
D=DIC_and_TIF(tif_template=tif_template)

import numpy as np
from scipy.signal import savgol_filter, find_peaks

class Phenology_extraction:
    def __init__(self, D):
        pass
    def run(self):
        fdir= data_root + rf'\MODIS_LAI\extract_tif_scaled\LAI_ts.npy'
        self.phenology_extract()
        pass



    # =========================
    # 1️⃣ 多年平均
    # =========================
    def climatology_mean(self,lai_ts):
        from datetime import datetime





        for f in sorted(year_dic['2003']):
            date = datetime.strptime(f[:8], "%Y%m%d")
            template_dates.append(date.timetuple().tm_yday)

        year = '2022'
        year_dates = []
        year_files = sorted(year_dic[year])

        for f in year_files:
            date = datetime.strptime(f[:8], "%Y%m%d")
            year_dates.append(date.timetuple().tm_yday)

        aligned_stack = []

        for doy in template_dates:
            if doy in year_dates:
                idx = year_dates.index(doy)
                arr,originX,originY, pixelWidth, pixelHeight = ToRaster().raster2array((year_files[idx]))
            else:
                arr = np.full((H, W), np.nan)

            aligned_stack.append(arr)

        aligned_stack = np.stack(aligned_stack, axis=0)


        return np.nanmean(lai_ts, axis=0)

        # =========================
        # 2️⃣ 平滑
        # =========================

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

        # =========================
        # 3️⃣ 主函数   ##class = 1  → evergreen
        # class = 2  → two seasons
        # class = 3  → single season
        # =========================

    def phenology_extract(self, lai_ts):

        clim = self.climatology_mean(lai_ts)
        ts = self.smooth_series(clim)

        n = len(ts)
        A = np.max(ts) - np.min(ts)

        # -------------------
        # Evergreen
        # -------------------
        if A < 0.1:
            return {
                "class": 1,
                "n_seasons": 0,
                "seasons": []
            }

        # -------------------
        # 找 peaks
        # -------------------
        peaks, props = find_peaks(
            ts,
            prominence=0.15 * A,
            distance=15
        )

        if len(peaks) == 0:
            return {
                "class": 1,
                "n_seasons": 0,
                "seasons": []
            }

        # 如果只有一个峰 → single
        if len(peaks) == 1:
            seasons = [self.extract_one_season(ts, peaks[0])]
            return {
                "class": 3,
                "n_seasons": 1,
                "seasons": seasons
            }

        # 如果 >=2 个峰 → 判断是否 two seasons
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

        if cond_valley and cond_distance:
            season1 = self.extract_one_season(ts, p1)
            season2 = self.extract_one_season(ts, p2)

            return {
                "class": 2,
                "n_seasons": 2,
                "seasons": [season1, season2]
            }
        else:
            season = self.extract_one_season(ts, p1)
            return {
                "class": 3,
                "n_seasons": 1,
                "seasons": [season]
            }

        # =========================
        # 单个 season 提取
        # =========================

    def extract_one_season(self, ts, peak_idx):

        n = len(ts)
        A = np.max(ts) - np.min(ts)
        threshold = np.min(ts) + 0.2 * A

        # 向左找 SOS
        left = peak_idx
        while ts[left] > threshold:
            left -= 1
            if left < 0:
                left = n - 1
                break

        # 向右找 EOS
        right = peak_idx
        while ts[right] > threshold:
            right += 1
            if right >= n:
                right = 0
                break

        sos = left
        eos = right

        return {
            "sos": sos,
            "peak": peak_idx,
            "eos": eos
        }

        # =========================
        # 提取 GS（支持跨年）
        # =========================

    def extract_gs_values(ts, sos, eos):

        if sos < eos:
            gs = ts[sos:eos]
        else:
            gs = np.concatenate([ts[sos:], ts[:eos]])

        return {
            "length": len(gs),
            "mean": np.mean(gs),
            "integral": np.sum(gs),
            "max": np.max(gs),
            "min": np.min(gs)
        }


class Climatology_builder:
    def __init__(self):
        pass

    def run(self):
        self.build_climatology()
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

class check_data:
    def __init__(self):
        pass
    def run(self):
            self.plot_time_series()

    def plot_time_series(self):
        f = data_root + rf'\MODIS_LAI\LAI_climatology\LAI_climatology.npy'
        dic = T.load_npy(f)
        for pix in dic:
            vals = dic[pix]

            if np.isnan(np.nanmean(vals)):
                continue
            print(len(vals))
            time_series = dic[pix]
            time_series = dic[pix]
            plt.plot(time_series)
            plt.show()


def main():
   # Phenology_extraction().run()
   Climatology_builder().run()
   check_data().run()




if __name__ == '__main__':
    main()