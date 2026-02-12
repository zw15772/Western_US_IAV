import matplotlib.pyplot as plt
import numpy as np


from __Global__ import *
tif_template= '/Users/wenzhang/Downloads/Western US IAV/Data/SNU_LAI/extract_tif/199901.tif'
D=DIC_and_TIF(tif_template=tif_template)



class Phenology:

    def __init__(self):
        pass

    def run(self):

        #
        # self.multi_year_mean_monthly()
        # self.interpolate()

        # self.GST()
        # self.plot_4GST_df()
        # self.read_4GST_df()
        self.plot_4GST_npy()  ### plot SOS and EOS


        pass





    def multi_year_mean_monthly(self):
        fdir_all =data_root+ rf'SNU_LAI/dic/'
        spatial_dic=T.load_npy_dir(fdir_all)
        result_dic={}


        for pix in spatial_dic:

            val=spatial_dic[pix]
            if T.is_all_nan(val):
                continue

            vals_reshape=val.reshape(-1,12)
            # print(vals_reshape.shape)
            # plt.imshow(vals_reshape, interpolation='nearest', cmap='jet')
            # plt.colorbar()
            # plt.show()
            multiyear_mean = np.nanmean(vals_reshape, axis=0)
            #
            result_dic[pix]=multiyear_mean
            x=np.arange(0,12)
            # xtick = [str(i) for i in x]
            # plt.plot(x, multiyear_mean)
            # plt.xticks(x, xtick)
            # plt.xlabel('monthly')
            # plt.ylabel('LAI4g (m2/m2)')
            #
            # plt.show()
        outdir=data_root+rf'SNU_LAI/multi_year_mean_monthly/'

        Tools().mk_dir(outdir, force=True)

        np.save(outdir+'multi_year_mean_monthly.npy', result_dic)

    def interpolate(self):
        f =data_root+rf'SNU_LAI/multi_year_mean_monthly/multi_year_mean_monthly.npy'
        spatial_dic=T.load_npy(f)
        plot_dic={}
        result_dic={}

        # 原始月份中心（1–12）
        x_month = np.arange(12)

        # biweekly 目标点（24）
        x_biweekly = np.linspace(0, 11, 24)

        for pix, val in spatial_dic.items():
            val = np.asarray(val)

            if T.is_all_nan(val):
                continue

            # 至少要有一些有效值
            if np.isfinite(val).sum() < 4:
                continue

            # 插值（忽略 NaN）
            valid = np.isfinite(val)
            y_interp = np.interp(
                x_biweekly,
                x_month[valid],
                val[valid]
            )
            # plt.plot(np.arange(len(y_interp)), y_interp, label=pix)
            # plt.plot(val)
            # plt.show()


            result_dic[pix] = y_interp
            plot_dic[pix] = len(y_interp)
        # arr=D.pix_dic_to_spatial_arr(plot_dic)
        # plt.imshow(arr)
        # plt.show()

        outdir=data_root+rf'SNU_LAI/multi_year_mean_monthly/'

        outf=outdir+'multi_year_mean_monthly_interpolate.npy'
        T.save_npy(result_dic, outf)

#*****************************************
# COMPUTE ONSET/OFFSET
#*****************************************

    def GST(self):
        f = data_root+rf'SNU_LAI/multi_year_mean_monthly/multi_year_mean_monthly_interpolate.npy'
        outdir =data_root+ rf'SNU_LAI/4GST/'
        T.mk_dir(outdir, force=True)
        spatial_dic = T.load_npy(f)
        spatial_dic_result = {}
        for pix in tqdm(spatial_dic):
            r,c=pix

            LAI = spatial_dic[pix]
            # plt.plot(LAI)
            # plt.show()

            LAI[LAI<-9999]=np.nan
            if T.is_all_nan(LAI):
                continue
            # print(LAI)


            LAI1d_1, LAI1d_2, LAI1d_3, LAI1d_4,knan,LAI1d,days= self.divide_LAI_pieces(LAI)
            if LAI1d_1 is None:
                continue

            SeasType,SeasClss=self.compute_linear_regression_type_assign_new(LAI, LAI1d_1, LAI1d_2, LAI1d_3, LAI1d_4, knan,LAI1d)
            SeasType, Onsets, Offsets = self.SOS_EOS(pix,LAI1d, SeasType,SeasClss,days)
            print(SeasType, Onsets, Offsets)
            result_dict_i = {
                'SeasType':SeasType,
                'Onsets':Onsets,
                'Offsets':Offsets,
                'SeasClss':SeasClss
            }

            spatial_dic_result[pix] = result_dict_i

        np.save(join(outdir, '4GST.npy'), spatial_dic_result)
        df_result = T.dic_to_df(spatial_dic_result,'pix')
        outf = join(outdir, '4GST.df')
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)



    def divide_LAI_pieces(self,LAI1):
        # ********** Divide the LAI timeseries in 4 pieces *******
        if len(LAI1) != 24:
            raise ValueError("Expected biweekly (24) LAI, got monthly input")
        if np.nanmean(LAI1) > 0:
            day = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300,
                   315, 330, 345, 360]


            LAI1dr = LAI1
            knan = 0
            # identify LAI maximum
            for t in range(len(LAI1dr)):
                if np.isnan(LAI1dr[t]):
                    LAI1dr[t] = 0
                    knan = knan + 1
                if LAI1dr[t] == np.nanmax(LAI1dr):
                    midPos = t
            LAI1d = np.zeros(shape=(len(LAI1dr)))
            days = np.zeros(shape=(len(day)))
            # shift of time series to place maximum LAI in the time series center
            if midPos == 11:
                LAI1d = LAI1dr
                days = day
            elif midPos > 11:
                dMP = midPos - 11
                for l in range(len(LAI1dr) - dMP):
                    LAI1d[l] = LAI1dr[l + dMP]
                    days[l] = day[l + dMP]
                for l in range(dMP):
                    LAI1d[len(LAI1dr) - dMP + l] = LAI1dr[l]
                    days[len(day) - dMP + l] = day[l]
            else:
                dMP = 11 - midPos
                for l in range(dMP):
                    LAI1d[l] = LAI1dr[len(LAI1dr) - dMP + l]
                    days[l] = day[len(days) - dMP + l]
                for l in range(len(LAI1dr) - dMP):
                    LAI1d[dMP + l] = LAI1dr[l]
                    days[dMP + l] = day[l]

            LAI1d_1 = np.zeros(shape=(7))
            LAI1d_2 = np.zeros(shape=(7))
            LAI1d_3 = np.zeros(shape=(7))
            LAI1d_4 = np.zeros(shape=(7))
            for i1 in range(7):
                i2 = i1 + 6
                i3 = i1 + 12
                i4 = i1 + 18
                print("i1:", i1, "i3:", i3, "LAI1d.shape:", LAI1d.shape)
                LAI1d_1[i1] = LAI1d[i1]
                LAI1d_2[i1] = LAI1d[i2]
                LAI1d_3[i1] = LAI1d[i3]
                if i1 == 6:
                    LAI1d_4[i1] = LAI1d[0]
                else:
                    LAI1d_4[i1] = LAI1d[i4]

       # *** set to zero the nan points  *******
            for i in range(7):
                if np.isnan(LAI1d_1[i]):
                    LAI1d_1[i] = 0
                if np.isnan(LAI1d_2[i]):
                    LAI1d_2[i] = 0
                if np.isnan(LAI1d_3[i]):
                    LAI1d_3[i] = 0
                if np.isnan(LAI1d_4[i]):
                    LAI1d_4[i] = 0
            return LAI1d_1, LAI1d_2, LAI1d_3, LAI1d_4,knan,LAI1d,days
        else:
            return [None] * 7

    def compute_linear_regression_type_assign_new(self, LAI1, LAI1d_1, LAI1d_2, LAI1d_3, LAI1d_4, knan, LAI1dr):
        """
        根据 LAI 年序列的分段线性回归结果，划分季节类型

        Type 1: Evergreen (常绿，幅度小)
        Type 2: Single season (一个生长季)
        Type 3: Double season - 跨年双峰
        Type 4: Double season - 一年内双峰
        """

        # 时间分段
        time1 = np.arange(7)
        time2 = time1 + 6
        time3 = time1 + 12
        time4 = time1 + 18

        # 线性回归
        linreg1, _, _, _, _ = stats.linregress(time1, LAI1d_1)
        linreg2, _, _, _, _ = stats.linregress(time2, LAI1d_2)
        linreg3, _, _, _, _ = stats.linregress(time3, LAI1d_3)
        linreg4, _, _, _, _ = stats.linregress(time4, LAI1d_4)

        # 统计量
        LAImin = np.nanmin(LAI1)
        LAImax = np.nanmax(LAI1)
        LAImean = np.nanmean(LAI1)
        SeasAmpl = LAImax - LAImin

        # 初始化
        SeasType = 0
        SeasClss = 0

        # -------- 常绿类 --------
        if SeasAmpl < 0.2 * LAImean:  # 原来是 0.2，可以调整阈值
            SeasType = 1

        else:
            # 缺失太多 / 平直 → 单季
            if linreg1 == 0.0 or knan >= len(LAI1dr) / 2:
                SeasType = 2

            # 一年内双峰
            elif linreg1 >= 0 and linreg2 <= 0 and linreg3 >= 0 and linreg4 <= 0:
                SeasType = 4
                SeasClss = 1

            # 跨年双峰
            elif linreg1 <= 0 and linreg2 >= 0 and linreg3 <= 0 and linreg4 >= 0:
                SeasType = 3
                SeasClss = 4

            # 默认单季
            else:
                SeasType = 2

        return SeasType, SeasClss

    def compute_linear_regression_type_assign(self, LAI1, LAI1d_1, LAI1d_2, LAI1d_3, LAI1d_4,knan,LAI1dr):

        # ********** Divide the time axis in 4 pieces *******

        time1 = np.zeros(shape=(7))
        time2 = np.zeros(shape=(7))
        time3 = np.zeros(shape=(7))
        time4 = np.zeros(shape=(7))

        for i in range(7):
            time1[i] = i
            time2[i] = time1[i] + 6
            time3[i] = time1[i] + 12
            time4[i] = time1[i] + 18

        ## ********** Linear regression ******

        linreg1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(time1, LAI1d_1)
        linreg2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(time2, LAI1d_2)
        linreg3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(time3, LAI1d_3)
        linreg4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(time4, LAI1d_4)


        # ********** Setting the tipe depending on the linear regression gradient sign
        # Type 3.1) / \ / \;
        # Type 3.4) \ / \ /;
        # Type 2) - ... ; ... > half points; else
        # Type 1) evergreen areas:
        # regions where seasonal LAI amplitude is lower than 25 percentage of LAI minimum

        # ******** Compute the "Type" of LAI seasonal cycle ****
        # Type 1) evergreen phenology case
        #           __       __
        #      ____/  \   __/  \_____
        #              \_/
        #
        # Type 2) basic case with one growing season
        #             _____
        #            /     \
        #           /       \
        #      ____/         \_____
        #
        # Type 3.1) two growing season, in the same year
        #             __
        #            /  \    __
        #           /    \__/  \
        #      ____/            \__
        #
        # Type 3.4) two growing season, in two years
        #             __
        #      _     /  \     __
        #       \   /    \   /
        #        \_/      \_/
        #
        #
        # This method divides the annual LAI timeseries in 4 pieces
        # each of 3 months, with one overlapping step.
        # Then, the linear regression is computed for each of the 4 pieces
        # and a Seasonal Type is assigned depending on the directions
        # (positive or negative linear regression gradient) of the
        # four linear regressions

        # ******* compute time minimum ******
        LAImin = np.nanmin(LAI1, axis=0)

        # ******* compute time maximum ******
        LAImax = np.nanmax(LAI1, axis=0)

        # ****** compute annual mean ******
        LAImean = np.nanmean(LAI1, axis=0)

        # ****** compute Seasonal Amplitude *****
        SeasAmpl = LAImax - LAImin
        SeasClss = 0
        SeasType = 0


        if SeasAmpl < 0.25 * LAImean:
            SeasType = 1
        else:
            if linreg1 == 0.0:
                SeasType = 2
            elif knan >= len(LAI1dr) / 2:
                SeasType = 2
                ### two growing season in one year
            elif linreg1 >= 0 and linreg2 <= 0 and linreg3 >= 0 and linreg4 <= 0:
                SeasType= 3
                SeasClss = 1
                ## two growing season in two years
            elif linreg1 <= 0 and linreg2 >= 0 and linreg3 <= 0 and linreg4 >= 0:
                SeasType = 3
                SeasClss = 4
            else:
                SeasType = 2

        return SeasType,SeasClss

    def SOS_EOS(self, pix,LAI1d, SeasType,SeasClss,days):

        r,c = pix
        lon,lat=D.pix_to_lon_lat(pix)
        ## two demensional array for onsets and offsets
        Onsets = np.array([0.0, 0.0])
        Offsets = np.array([0.0, 0.0])

        # ===== 新增：Evergreen 情况 =====
        if SeasType == 1:
            # Evergreen：不定义 SOS / EOS
            # 但保留 SeasType = 1
            return SeasType, 0, 365

        # ******* compute the Onset and Offset in each gridpoint *****

        Ab = LAI1d
        for t in range(len(Ab)):
            if np.isnan(Ab[t]):
                Ab[t] = 0
        if SeasType == 2:
            mx = np.max(Ab)
            mn = np.min(Ab)
            thrs = mn + 0.2 * (mx - mn)
            if mx < 0.01:
                Onset = 0
                Offset = 0
            else:
                tr = np.where(Ab >= thrs)
                trar = tr[0]
                if len(trar) == 24:
                    Onset = 0
                    Offset = 0
                else:
                    # operations to smooth the line and avoid one-step perturbation to disturb onset and offset detection
                    kont = np.zeros(shape=(len(trar)))
                    for i in range(len(trar) - 1):
                        if trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        else:
                            kont[i] = 0
                    if trar[len(trar) - 1] - trar[len(trar) - 2] <= 2:
                        kont[len(trar) - 1] = 1
                    else:
                        kont[len(trar) - 1] = 0

                    kont2 = 0
                    for i in range(len(kont)):
                        if kont[i] == 0:
                            kont2 = kont2 + 1
                            break
                        else:
                            kont2 = kont2 + 1

                    if kont2 == len(trar):
                        trar2 = np.zeros(shape=(kont2))
                        for i in range(kont2):
                            trar2[i] = trar[i]
                    elif kont[0] == 0:
                        trar2 = np.zeros(shape=(len(kont) - 1))
                        for i in range(len(kont) - 1):
                            trar2[i] = trar[i + 1]
                    else:
                        valbc2 = 0
                        valac2 = 0
                        for i in range(len(kont)):
                            if i <= kont2 - 1:
                                valbc2 = valbc2 + 1
                            else:
                                valac2 = valac2 + 1
                        if valbc2 > valac2:
                            trar2 = np.zeros(shape=(kont2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i]
                        else:
                            trar2 = np.zeros(shape=(valac2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i + kont2]

                    Onset = trar2[0]
                    Offset = trar2[len(trar2) - 1]

                    Onsets = days[int(Onset)]
                    Offsets = days[int(Offset)]

                    if lat >= 0:
                        if days[11] >= 120 and days[11] <= 270:
                            SeasClss1 = 1
                        else:
                            SeasClss1 = 2
                    else:
                        if days[11] >= 120 and days[11] <= 270:
                            SeasClss1= 2
                        else:
                            SeasClss1 = 1


        elif SeasType == 3 and SeasClss == 1:

            # determine the central minimum that divides the two cycles
            LAI_cnt = Ab[7:18]
            C31min = np.min(LAI_cnt)
            for i in range(len(Ab)):
                if Ab[i] == C31min and i >= 7 and i < 18:
                    brk_pos = i

            # analyse the first cycle as a shorter type 1 cycle
            LAIc1 = Ab[:brk_pos + 1]
            daysc1 = days[:brk_pos + 1]

            mx = np.max(LAIc1)
            mn = np.min(LAIc1)
            thrs = mn + 0.2 * (mx - mn)
            if mx < 0.01:
                Onsets[0] = 0
                Offsets[0] = 0
            else:
                tr = np.where(LAIc1 >= thrs)
                trar = tr[0]
                if len(trar) == len(LAIc1):
                    Onsets[0] = 0
                    Offsets[0] = 0
                else:
                    kont = np.zeros(shape=(len(trar)))
                    for i in range(len(trar) - 1):
                        if trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        elif trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        else:
                            kont[i] = 0
                    if trar[len(trar) - 1] - trar[len(trar) - 2] <= 2:
                        kont[len(trar) - 1] = 1
                    else:
                        kont[len(trar) - 1] = 0

                    kont2 = 0
                    for i in range(len(kont)):
                        if kont[i] == 0:
                            kont2 = kont2 + 1
                            break
                        else:
                            kont2 = kont2 + 1

                    if kont2 == len(trar):
                        trar2 = np.zeros(shape=(kont2))
                        for i in range(kont2):
                            trar2[i] = trar[i]
                    elif kont[0] == 0:
                        trar2 = np.zeros(shape=(len(kont) - 1))
                        for i in range(len(kont) - 1):
                            trar2[i] = trar[i + 1]
                    else:
                        valbc2 = 0
                        valac2 = 0
                        for i in range(len(kont)):
                            if i <= kont2 - 1:
                                valbc2 = valbc2 + 1
                            else:
                                valac2 = valac2 + 1
                        if valbc2 > valac2:
                            trar2 = np.zeros(shape=(kont2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i]
                        else:
                            trar2 = np.zeros(shape=(valac2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i + kont2]

                    Onset = trar2[0]
                    Offset = trar2[len(trar2) - 1]

                    Onsets[0] = daysc1[int(Onset)]
                    Offsets[0] = daysc1[int(Offset)]

            # analyse the second cycle as a shorter type 1 cycle
            LAIc2 = Ab[brk_pos - 1:]
            daysc2 = days[brk_pos - 1:]

            mx = np.max(LAIc2)
            mn = np.min(LAIc2)
            thrs = mn + 0.2 * (mx - mn)
            if mx < 0.01:
                Onsets[1] = 0
                Offsets[1] = 0
            else:
                tr = np.where(LAIc2 >= thrs)
                trar = tr[0]
                if len(trar) == len(LAIc2):
                    Onsets[1] = 0
                    Offsets[1] = 0
                else:
                    kont = np.zeros(shape=(len(trar)))
                    for i in range(len(trar) - 1):
                        if trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        elif trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        else:
                            kont[i] = 0
                    if trar[len(trar) - 1] - trar[len(trar) - 2] <= 2:
                        kont[len(trar) - 1] = 1
                    else:
                        kont[len(trar) - 1] = 0

                    kont2 = 0
                    for i in range(len(kont)):
                        if kont[i] == 0:
                            kont2 = kont2 + 1
                            break
                        else:
                            kont2 = kont2 + 1

                    if kont2 == len(trar):
                        trar2 = np.zeros(shape=(kont2))
                        for i in range(kont2):
                            trar2[i] = trar[i]
                    elif kont[0] == 0:
                        trar2 = np.zeros(shape=(len(kont) - 1))
                        for i in range(len(kont) - 1):
                            trar2[i] = trar[i + 1]
                    else:
                        valbc2 = 0
                        valac2 = 0
                        for i in range(len(kont)):
                            if i <= kont2 - 1:
                                valbc2 = valbc2 + 1
                            else:
                                valac2 = valac2 + 1
                        if valbc2 > valac2:
                            trar2 = np.zeros(shape=(kont2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i]
                        else:
                            trar2 = np.zeros(shape=(valac2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i + kont2]

                    Onset = trar2[0]
                    Offset = trar2[len(trar2) - 1]

                    Onsets[1] = daysc2[int(Onset)]
                    Offsets[1] = daysc2[int(Offset)]
            # check on the order of cycles, in case invert them
            # lon, lat
            # print(Onsets);exit()
            if not np.isnan(Onsets[0]):
                if Onsets[0] > Onsets[1]:
                    tmpN0 = Onsets[0]
                    tmpN1 = Onsets[1]
                    tmpF0 = Offsets[0]
                    tmpF1 = Offsets[1]
                    Onsets[0] = tmpN1
                    Onsets[1] = tmpN0
                    Offsets[0] = tmpF1
                    Offsets[1] = tmpF0

        elif SeasType == 3 and SeasClss== 4:

            # determine the two cycles: one in the central 13 points and one using the 13 lateral points
            LAIc1 = Ab[6:19]
            daysc1 = days[6:19]

            LAIc2_a = Ab[:7]
            LAIc2_b = Ab[18:]
            daysc2_a = days[:7]
            daysc2_b = days[18:]
            LAIc2 = np.zeros(shape=(len(LAIc2_a) + len(LAIc2_b)))
            daysc2 = np.zeros(shape=(len(LAIc2_a) + len(LAIc2_b)))

            for ii in range(len(LAIc2_b)):
                LAIc2[ii] = LAIc2_b[ii]
                daysc2[ii] = daysc2_b[ii]

            for ii in range(len(LAIc2) - len(LAIc2_b)):
                jj = len(LAIc2_b) + ii
                LAIc2[jj] = LAIc2_a[ii]
                daysc2[jj] = daysc2_a[ii]

            #                                jj = 0
            #                                for ii in range(len(Ab)):
            #                                        if ii < 7 or ii >= 18:
            #                                                LAIc2[jj] = Ab[ii]
            #                                                daysc2[jj] = days[ii]
            #                                                jj = jj+1

            # analyse the first cycle as a shorter type 1 cycle
            mx = np.max(LAIc1)
            mn = np.min(LAIc1)
            thrs = mn + 0.2 * (mx - mn)
            if mx < 0.01:
                Onsets[0] = 0
                Offsets[0] = 0
            else:
                tr = np.where(LAIc1 >= thrs)
                trar = tr[0]
                if len(trar) == len(LAIc1):
                    Onsets[0] = 0
                    Offsets[0] = 0
                else:
                    kont = np.zeros(shape=(len(trar)))
                    for i in range(len(trar) - 1):
                        if trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        elif trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        else:
                            kont[i] = 0
                    if trar[len(trar) - 1] - trar[len(trar) - 2] <= 2:
                        kont[len(trar) - 1] = 1
                    else:
                        kont[len(trar) - 1] = 0

                    kont2 = 0
                    for i in range(len(kont)):
                        if kont[i] == 0:
                            kont2 = kont2 + 1
                            break
                        else:
                            kont2 = kont2 + 1

                    if kont2 == len(trar):
                        trar2 = np.zeros(shape=(kont2))
                        for i in range(kont2):
                            trar2[i] = trar[i]
                    elif kont[0] == 0:
                        trar2 = np.zeros(shape=(len(kont) - 1))
                        for i in range(len(kont) - 1):
                            trar2[i] = trar[i + 1]
                    else:
                        valbc2 = 0
                        valac2 = 0
                        for i in range(len(kont)):
                            if i <= kont2 - 1:
                                valbc2 = valbc2 + 1
                            else:
                                valac2 = valac2 + 1
                        if valbc2 > valac2:
                            trar2 = np.zeros(shape=(kont2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i]
                        else:
                            trar2 = np.zeros(shape=(valac2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i + kont2]

                    Onset = trar2[0]
                    Offset = trar2[len(trar2) - 1]

                    Onsets[0] = daysc1[int(Onset)]
                    Offsets[0] = daysc1[int(Offset)]

            # analyse the second cycle as a shorter type 1 cycle
            mx = np.max(LAIc2)
            mn = np.min(LAIc2)
            thrs = mn + 0.2 * (mx - mn)
            if mx < 0.01:
                Onsets[1] = 0
                Offsets[1] = 0
            else:
                tr = np.where(LAIc2 >= thrs)
                trar = tr[0]
                if len(trar) == len(LAIc2):
                    Onsets[1] = 0
                    Offsets[1] = 0
                else:
                    kont = np.zeros(shape=(len(trar)))
                    for i in range(len(trar) - 1):
                        if trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        elif trar[i + 1] - trar[i] <= 2:
                            kont[i] = 1
                        else:
                            kont[i] = 0
                    if trar[len(trar) - 1] - trar[len(trar) - 2] <= 2:
                        kont[len(trar) - 1] = 1
                    else:
                        kont[len(trar) - 1] = 0

                    kont2 = 0
                    for i in range(len(kont)):
                        if kont[i] == 0:
                            kont2 = kont2 + 1
                            break
                        else:
                            kont2 = kont2 + 1

                    if kont2 == len(trar):
                        trar2 = np.zeros(shape=(kont2))
                        for i in range(kont2):
                            trar2[i] = trar[i]
                    elif kont[0] == 0:
                        trar2 = np.zeros(shape=(len(kont) - 1))
                        for i in range(len(kont) - 1):
                            trar2[i] = trar[i + 1]
                    else:
                        valbc2 = 0
                        valac2 = 0
                        for i in range(len(kont)):
                            if i <= kont2 - 1:
                                valbc2 = valbc2 + 1
                            else:
                                valac2 = valac2 + 1
                        if valbc2 > valac2:
                            trar2 = np.zeros(shape=(kont2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i]
                        else:
                            trar2 = np.zeros(shape=(valac2))
                            for i in range(len(trar2)):
                                trar2[i] = trar[i + kont2]

                    Onset = trar2[0]
                    Offset = trar2[len(trar2) - 1]

                    Onsets[1] = daysc2[int(Onset)]
                    Offsets[1] = daysc2[int(Offset)]

            # check on the order of cycles, in case invert them
            if not np.isnan(Onsets[0]) and not np.isnan(Onsets[1]):
                if Onsets[0] > Onsets[1]:
                    tmpN0 = Onsets[0]
                    tmpN1 = Onsets[1]
                    tmpF0 = Offsets[0]
                    tmpF1 = Offsets[1]
                    Onsets[0] = tmpN1
                    Onsets[1] = tmpN0
                    Offsets[0] = tmpF1
                    Offsets[1] = tmpF0

        else:
            SeasType = np.nan
            Onsets[0] = np.nan
            Onsets[1] = np.nan
            Offsets[0] = np.nan
            Offsets[1] = np.nan
        # if Onsets > Offsets:
        # print([pix,SeasType, Onsets, Offsets]);exit()
        # print(
        #     "mx:", mx,
        #     "mn:", mn,
        #     "thrs:", thrs,
        #     "len(trar):", len(trar),
        #     "Ab:", Ab
        # );exit()
        return SeasType, Onsets, Offsets

    def plot_4GST_df(self):
        fpath = data_root+rf'/SNU_LAI/4GST/4GST.df'
        df = T.load_df(fpath)
        df_2 = df[df['SeasType'] == 1]
        onset_list  = []
        offset_list = []
        for i, row in df_2.iterrows():
            onset = row['Onsets']
            offset = row['Offsets']
            # print(onset)
            try:
                onset = float(onset)
                offset = float(offset)
                onset_list.append(onset)
                offset_list.append(offset)

            except:
                onset_list.append(np.nan)
                offset_list.append(np.nan)
                continue
        df_2['Onsets'] = onset_list
        df_2['Offsets'] = offset_list
        # T.print_head_n(df, 10);exit()
        Onsets = T.df_to_spatial_dic(df_2, 'Onsets')
        Offsets = T.df_to_spatial_dic(df_2, 'Offsets')
        # pprint(Onsets);exit()
        arr_onset = D.pix_dic_to_spatial_arr(Onsets)
        arr_offset = D.pix_dic_to_spatial_arr(Offsets)
        plt.imshow(arr_onset, interpolation='nearest', cmap='jet',vmin=0,vmax=100)
        # plt.colorbar()
        # plt.title('onset')
        # plt.figure()
        # plt.imshow(arr_offset, interpolation='nearest', cmap='jet',vmin=30,vmax=300)
        # plt.colorbar()
        # plt.title('offset')
        plt.show()
        pass
    def read_4GST_df(self):
        fpath= join(result_root, rf'LAI4g_phenology\4GST.df')
        df = T.load_df(fpath)
        ## get seasonal type length
        df_1 = df[df['SeasType'] == 1]

        print('df_1',len(df_1))
        T.print_head_n(df_1)
        df_2 = df[df['SeasType'] == 2]
        print('df_2',len(df_2))
        T.print_head_n(df_2)
        df_3 = df[df['SeasType'] == 3]
        print('df_3',len(df_3))
        T.print_head_n(df_3)

    def plot_4GST_npy(self):  ##
        f= rf'/Users/wenzhang/Downloads/Western US IAV/Data/SNU_LAI/4GST/4GST.npy'
        spatial_dic = T.load_npy(f)
        result_dic = {}
        vals_list = []

        for pix in spatial_dic:
            val=spatial_dic[pix]['Offsets']
            # val = spatial_dic[pix]['SeasType']
            print(pix,val)
            try:
                val=float(val)
            except:
                continue
            vals_list.append(val)

            result_dic[pix]=val
        vals_list = np.array(vals_list)
        ## get unique values
        vals_list = np.unique(vals_list)
        print(vals_list)

        arr = D.pix_dic_to_spatial_arr(result_dic)
        outdir=data_root+rf'SNU_LAI/Phenology_extraction/'
        T.mk_dir(outdir,force=True)
        outtif=outdir+'Offsets.tif'
        D.arr_to_tif(arr, outtif)


        # plt.imshow(arr, interpolation='nearest', cmap='jet',vmin=0,vmax=4)
        # plt.colorbar()
        # plt.title('leaf senescence')
        # plt.show()


        pass


# def test()

def test():
    # this
    # test
    pass



def main():

    Phenology().run()

    pass

if __name__ == '__main__':
    main()




