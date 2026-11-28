# coding='utf-8'
## Soil water holding capacity for the driver model (2026-11-28)
## data   : SoilGrids 2.0 (ISRIC, 250 m), mean of sand, clay (g/kg x 10 -> %), soc (dg/kg -> SOC % = value / 100),
##          depths 0-5, 5-15, 15-30, 30-60, 60-100 cm; read straight from the ISRIC VRT over the internet (/vsicurl/):
##          only the study area is downloaded, already averaged onto the 0.05 deg template grid (basedata\200902.tif)
## AWC    : Saxton & Rawls (2006) pedotransfer per layer, S, C as fractions, OM = 1.724 x SOC (%):
##          theta1500t = -0.024 S + 0.487 C + 0.006 OM + 0.005 S OM - 0.013 C OM + 0.068 S C + 0.031
##          theta1500  = theta1500t + (0.14 theta1500t - 0.02)
##          theta33t   = -0.251 S + 0.195 C + 0.011 OM + 0.006 S OM - 0.027 C OM + 0.452 S C + 0.299
##          theta33    = theta33t + (1.283 theta33t^2 - 0.374 theta33t - 0.015)
##          AWC_layer = theta33 - theta1500 (m3 m-3);  AWC_0_100 = sum AWC_layer x thickness (mm, 0-100 cm)
##          check: loam (S 0.4, C 0.2, OM 2.5) -> 0.28 - 0.14 = 0.14 m3 m-3, as in Saxton & Rawls Table 3
## output : {data_root}SoilGrids\tif\{var}_{depth}_mean_005.tif   (sand, clay in %, soc in g/kg, template grid)
##          {data_root}SoilGrids\tif\AWC_0_100cm_mm.tif, sand_0_100cm.tif, clay_0_100cm.tif (depth-weighted)
##          {result_root}Dataframe\Soil\soil_AWC.df   per pixel (row, col): AWC_0_100 (mm), sand, clay, SOC (0-100 cm)
##          {result_root}Dataframe\Soil\Fig_soil_AWC.png   quick-look maps
from __init__ import *
from __Global__ import *
from osgeo import gdal


class Soil_AWC():
    def __init__(self):
        self.url = 'https://files.isric.org/soilgrids/latest/data/{var}/{var}_{depth}_mean.vrt'
        self.var_list = ['sand', 'clay', 'soc']
        self.depth_list = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
        self.thick = np.array([50, 100, 150, 300, 400])   ## mm
        self.factor = {'sand': 10., 'clay': 10., 'soc': 10.}   ## stored value / factor -> %, %, g/kg
        self.template = data_root + rf'basedata\200902.tif'
        self.tif_dir = data_root + rf'SoilGrids\tif\\'
        self.df_dir = result_root + rf'\Dataframe\Soil\\'
        T.mk_dir(self.tif_dir, force=True)
        T.mk_dir(self.df_dir, force=True)

    def run(self):
        self.download_warp()
        self.awc()

    def grid(self):
        ds = gdal.Open(self.template)
        gt = ds.GetGeoTransform()
        nx, ny = ds.RasterXSize, ds.RasterYSize
        return gt, nx, ny, [gt[0], gt[3] + gt[5] * ny, gt[0] + gt[1] * nx, gt[3]]

    def download_warp(self):
        ## SoilGrids VRT -> template grid, area average; existing files are skipped
        gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '120')
        gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '5')
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        gt, nx, ny, bounds = self.grid()
        print('template: origin', gt[0], gt[3], '| size', nx, ny)
        for var in self.var_list:
            for depth in self.depth_list:
                fout = self.tif_dir + f'{var}_{depth}_mean_005.tif'
                if os.path.isfile(fout):
                    continue
                src = '/vsicurl/' + self.url.format(var=var, depth=depth)
                t0 = time.time()
                gdal.Warp(fout, src, dstSRS='EPSG:4326', outputBounds=bounds, width=nx, height=ny,
                          resampleAlg='average', srcNodata=-32768, dstNodata=-9999, outputType=gdal.GDT_Float32,
                          creationOptions=['COMPRESS=LZW'], multithread=True)
                print(var, depth, 'done in', round(time.time() - t0), 's')

    def read(self, fname):
        arr = gdal.Open(self.tif_dir + fname).ReadAsArray().astype(float)
        arr[arr == -9999] = np.nan
        return arr

    def awc(self):
        gt, nx, ny, bounds = self.grid()
        w = self.thick / self.thick.sum()
        awc_mm = np.zeros((ny, nx))
        mean = {v: np.zeros((ny, nx)) for v in self.var_list}
        for k, depth in enumerate(self.depth_list):
            S = self.read(f'sand_{depth}_mean_005.tif') / self.factor['sand'] / 100   ## fraction
            C = self.read(f'clay_{depth}_mean_005.tif') / self.factor['clay'] / 100
            soc = self.read(f'soc_{depth}_mean_005.tif') / self.factor['soc']        ## g/kg
            OM = 1.724 * soc / 10                                                   ## %
            t15 = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
            t15 = t15 + (0.14 * t15 - 0.02)
            t33 = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
            t33 = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
            awc_mm += np.clip(t33 - t15, 0, None) * self.thick[k]
            mean['sand'] += S * 100 * w[k]
            mean['clay'] += C * 100 * w[k]
            mean['soc'] += soc * w[k]
        for name, arr in [('AWC_0_100cm_mm', awc_mm), ('sand_0_100cm', mean['sand']), ('clay_0_100cm', mean['clay']),
                          ('soc_0_100cm', mean['soc'])]:
            self.save_tif(arr, gt, self.tif_dir + f'{name}.tif')
        rows, cols = np.where(np.isfinite(awc_mm))
        df = pd.DataFrame({'AWC_0_100': awc_mm[rows, cols], 'sand': mean['sand'][rows, cols],
                           'clay': mean['clay'][rows, cols], 'SOC': mean['soc'][rows, cols]},
                          index=pd.Index(list(zip(rows.tolist(), cols.tolist())), name='pix', tupleize_cols=False))
        T.save_df(df, self.df_dir + 'soil_AWC.df')
        print(df.describe().round(2).to_string())
        self.quick_look(awc_mm, mean)

    def save_tif(self, arr, gt, fout):
        drv = gdal.GetDriverByName('GTiff')
        ds = drv.Create(fout, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
        ds.SetGeoTransform(gt)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.WriteArray(np.where(np.isfinite(arr), arr, -9999))
        band.SetNoDataValue(-9999)
        ds = None

    def quick_look(self, awc_mm, mean):
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        for ax, (arr, title) in zip(axs, [(awc_mm, 'AWC 0-100 cm (mm)'), (mean['sand'], 'sand 0-100 cm (%)'),
                                          (mean['clay'], 'clay 0-100 cm (%)')]):
            im = ax.imshow(arr, cmap='Spectral', vmin=np.nanpercentile(arr, 2), vmax=np.nanpercentile(arr, 98))
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title, fontsize=22)
            ax.axis('off')
        plt.savefig(self.df_dir + 'Fig_soil_AWC.png', dpi=200, bbox_inches='tight')
        plt.close()


def main():
    Soil_AWC().run()


if __name__ == '__main__':
    main()
