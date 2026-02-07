from __Global__ import *

class Data_processing:
    def run(self):
        self.nc_to_tif_time_series_fast2()
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



def main():

    Data_processing().run()

if __name__ == '__main__':
    main()