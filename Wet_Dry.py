import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= data_root + rf'basedata\Phenology_extraction\SeasType.tif'
D=DIC_and_TIF(tif_template=tif_template)

import numpy as np
import pandas as pd
from scipy.ndimage import label
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class Pick_wet_dry:
    def run(self):
        f='D:\Western_US_IAV\Data\Terraclimate\SPEI\SPEI_12_NOAA\calculating_annual_mean\\'
        SPEI_dic= T.load_npy(f + 'SPEI12_annual_mean_WUS_2003_2024.npy')
        # 1️⃣ 转3D
        average_dic={}
        for pix in SPEI_dic:
            SPEI_values= SPEI_dic[pix]
            average_dic[pix]=np.nanmean(SPEI_values)
        array=D.pix_dic_to_spatial_arr(average_dic)
        rows, cols = array.shape[0], array.shape[1]
        spei = self.dic_to_3d(SPEI_dic, rows, cols)
        lat, lon = self.get_lat_lon_from_tif()

        # print(lat.min(), lat.max())
        # print(lon.min(), lon.max())
        # exit()

        # ===== Step 1 =====


        # ===== Step 2 =====
        year_list = range(2003, 2025)
        years = np.array(year_list)

        events, labeled_dry, labeled_wet = self.detect_voxel_events(spei, years)

        # ===== Step 3 =====
        df = pd.DataFrame(events)
        results = self.get_top_events(df)




        # # ===== Step 4（画图）=====
        # self.plot_top_events_map(results['top_S_dry'], labeled_dry, lat, lon, cols,'Top S Dry')
        # self.plot_top_events_map(results['top_A_dry'], labeled_dry, lat, lon, cols,'Top A Dry')
        # self.plot_top_events_map(results['top_T_dry'], labeled_dry, lat, lon, cols,'Top T Dry')
        #
        # self.plot_top_events_map(results['top_S_wet'], labeled_wet, lat, lon, cols,'Top S Wet')
        # self.plot_top_events_map(results['top_A_wet'], labeled_wet, lat, lon, cols,'Top A Wet')
        # self.plot_top_events_map(results['top_T_wet'], labeled_wet, lat, lon, cols,'Top T Wet')

        ## plot event time

        self.plot_event_with_time(results['top_S_dry'], labeled_dry, lat, lon, cols,'Top S Dry with Time')
        self.plot_event_with_time(results['top_S_wet'], labeled_wet, lat, lon, cols,'Top S Wet with Time')


    def get_event_mask(self,labeled, event_id):
        return labeled == event_id



    def dic_to_3d(self, dic, rows, cols):

        sample_pix = list(dic.keys())[0]
        T = len(dic[sample_pix])

        data = np.full((T, rows, cols), np.nan, dtype=np.float32)

        for (r, c), ts in dic.items():
            data[:, r, c] = ts

        return data

    def detect_voxel_events(self,spei, years, dry_th=-1.5, wet_th=1.5,
                            min_area=100, min_duration=1):

        structure = np.ones((3, 3, 3))

        events = []

        # ===== drought =====
        dry_mask = spei < dry_th
        labeled_dry, num_dry = label(dry_mask, structure)

        for eid in tqdm(range(1, num_dry + 1)):

            idx = np.where(labeled_dry == eid)
            t, y, x = idx

            if len(t) == 0:
                continue

            duration = len(np.unique(t))
            area = len(t)
            area = len(np.unique(list(zip(y, x))))

            if duration < min_duration or area < min_area:
                continue

            vals = spei[idx]

            events.append({
                'type': 'dry',
                'event_id': f'dry_{eid}',
                'eid_num': eid,
                'start_year': years[np.min(t)],
                'end_year': years[np.max(t)],
                'duration': duration,
                'area': area,
                'severity': np.nansum(vals),
            })

        # ===== wet =====
        wet_mask = spei > wet_th
        labeled_wet, num_wet = label(wet_mask, structure)

        for eid in tqdm(range(1, num_wet + 1)):

            idx = np.where(labeled_wet == eid)
            t, y, x = idx

            if len(t) == 0:
                continue

            duration = len(np.unique(t))
            area = len(t)

            if duration < min_duration or area < min_area:
                continue

            vals = spei[idx]

            events.append({
                'type': 'wet',
                'event_id': f'wet_{eid}',
                'eid_num': eid,
                'start_year': years[np.min(t)],
                'end_year': years[np.max(t)],
                'duration': duration,
                'area': area,
                'severity': np.nansum(vals),
            })

        return events, labeled_dry, labeled_wet

    def get_top_events(df):

        df = df.copy()

        # 👉 drought severity 转正
        df['severity_adj'] = df['severity']
        df.loc[df['type'] == 'dry', 'severity_adj'] *= -1

        results = {}

        # ===== S =====
        results['top_S_dry'] = df[df['type'] == 'dry'] \
            .sort_values('severity_adj', ascending=False).head(10)

        results['top_S_wet'] = df[df['type'] == 'wet'] \
            .sort_values('severity_adj', ascending=False).head(10)

        # ===== A =====
        results['top_A_dry'] = df[df['type'] == 'dry'] \
            .sort_values('area', ascending=False).head(10)

        results['top_A_wet'] = df[df['type'] == 'wet'] \
            .sort_values('area', ascending=False).head(10)

        # ===== T =====
        results['top_T_dry'] = df[df['type'] == 'dry'] \
            .sort_values('duration', ascending=False).head(10)

        results['top_T_wet'] = df[df['type'] == 'wet'] \
            .sort_values('duration', ascending=False).head(10)

        results['df'] = df

        return results

    def get_top_events(self,df):

        df = df.copy()

        # 👉 drought severity 转正
        df['severity_adj'] = df['severity']
        df.loc[df['type'] == 'dry', 'severity_adj'] *= -1

        results = {}

        # ===== S =====
        results['top_S_dry'] = df[df['type'] == 'dry'] \
            .sort_values('severity_adj', ascending=False).head(10)

        results['top_S_wet'] = df[df['type'] == 'wet'] \
            .sort_values('severity_adj', ascending=False).head(10)

        # ===== A =====
        results['top_A_dry'] = df[df['type'] == 'dry'] \
            .sort_values('area', ascending=False).head(10)

        results['top_A_wet'] = df[df['type'] == 'wet'] \
            .sort_values('area', ascending=False).head(10)

        # ===== T =====
        results['top_T_dry'] = df[df['type'] == 'dry'] \
            .sort_values('duration', ascending=False).head(10)

        results['top_T_wet'] = df[df['type'] == 'wet'] \
            .sort_values('duration', ascending=False).head(10)

        results['df'] = df

        T.save_df(results['df'], result_root + rf'greening_analysis\Dataframe\wet_dry_events.df')
        T.df_to_excel(results['df'], result_root + rf'greening_analysis\Dataframe\wet_dry_events.xlsx')

        return results

    def plot_top_events_map(self,top_df, labeled, lat, lon, cols, title):

        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.coastlines()

        cmap = plt.cm.tab10

        for i, (_, row) in enumerate(top_df.iterrows()):
            eid = row['eid_num']

            mask_3d = (labeled == eid)
            mask_2d = np.any(mask_3d, axis=0)

            y, x = np.where(mask_2d)

            # 🔥 核心修复！！
            flat_idx = y * cols + x

            ax.scatter(lon[flat_idx], lat[flat_idx],
                       s=3,
                       color=cmap(i),
                       label=f'{i + 1}')

        ax.set_title(title)
        plt.legend(title='Rank', bbox_to_anchor=(1.05, 1))
        plt.show()

    import rasterio
    import numpy as np

    def get_lat_lon_from_tif(self,):
        tif_path=tif_template

        with rasterio.open(tif_path) as src:
            transform = src.transform
            rows, cols = src.height, src.width

            row_idx = np.arange(rows)
            col_idx = np.arange(cols)

            col_grid, row_grid = np.meshgrid(col_idx, row_idx)

            lon, lat = rasterio.transform.xy(transform, row_grid, col_grid)

            lon = np.array(lon)
            lat = np.array(lat)


        return lat, lon


    def plot_event_with_time(self,top_df, labeled, lat, lon, cols, title):

        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)

        cmap = plt.cm.tab10

        legend_text = ""

        for i, (_, row) in enumerate(top_df.iterrows()):
            eid = row['eid_num']

            # ===== voxel mask =====
            mask_3d = (labeled == eid)
            mask_2d = np.any(mask_3d, axis=0)

            y, x = np.where(mask_2d)

            # 👉 flat index（你现在的情况）
            flat_idx = y * cols + x

            xs = lon[flat_idx]
            ys = lat[flat_idx]

            # ===== 画空间 =====
            ax.scatter(xs, ys, s=3, color=cmap(i))

            # ===== 计算中心位置 =====
            cx = np.mean(xs)
            cy = np.mean(ys)

            # ===== 标注编号 =====
            ax.text(cx, cy, f"{i + 1}", fontsize=10,
                    ha='center', weight='bold')

            # ===== 记录时间（右下角）=====
            legend_text += f"#{i + 1}  {int(row['start_year'])}-{int(row['end_year'])}\n"

        # ===== 标题 =====
        ax.set_title(title)

        # ===== 右侧 legend（关键！）=====
        plt.figtext(0.85, 0.2, legend_text, fontsize=10)

        plt.show()
    pass


def main ():
    Pick_wet_dry().run()

if __name__ == '__main__':
    main()


