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
        fdir= data_root + rf'basedata\Phenology_extraction\LAI_ts.npy'
        self.phenology_extract()
        pass



    # =========================
    # 1️⃣ 多年平均
    # =========================
    def climatology_mean(self,lai_ts):

        """
        lai_ts: shape (years, 91)
        """
        return np.nanmean(lai_ts, axis=0)

    # =========================
    # 2️⃣ 平滑
    # =========================
    def smooth_series(self,ts):

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
    def phenology_extract(self,lai_ts):

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
    def extract_one_season(self,ts, peak_idx):

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


def main():
    phenology = Phenology_extraction().run()



if __name__ == '__main__':
    main()