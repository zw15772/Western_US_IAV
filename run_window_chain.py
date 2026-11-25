# coding='utf-8'
## Rerun of the long analyses (SNU LAI 1982-2024 and everything built on it) with the growing-season window of
##   window_config.py (GS_WINDOW = 'SNU' -> data files *_snuwin, outputs in Result2_snuwin) (2026-11-23)
## steps run in order; if one stops with an error, fix it and set START to that step number to go on from there
## nothing here changes the MODIS-window results in Result2
from window_config import *
import growing_season_extraction as gse
import Time_series_plotting as tsp
import bivariate as biv
import SM_sufficiency as sms
import Attribution_analysis as att
import Grass_analysis as gra
import SNU_phenology as phe

START = 0


def greening_cover():
    gc = gra.Greening_cover_change()
    gc.run()                 ## Fig 6 / 6c / LAI per cover, pixel table
    gc.run_decomposition()   ## Fig 6b decomposition + all-pixel means


step_list = [
    ## 0 season table and GS LAI with the active window
    ('SNU growing-season table', lambda: gse.Growing_season_SNU().run() if GS_WINDOW == 'SNU' else None),
    ('GS LAI SNU', lambda: gse.GS_LAI_SNU().run()),
    ('GS LAI GIMMS', lambda: gse.GS_LAI_GIMMS().run()),
    ## 1 drought (builds SPEI_GS_1958_2024{WIN}.df first): Fig 1, step change
    ('SPEI 1982-2024', lambda: tsp.SPEI_1982_plot().run()),
    ## 2 best SPEI scale, decoupling: Fig 3, Fig 5
    ('best scale SNU', lambda: biv.Best_scale_long('SNU', 1982, 2024).run()),
    ('decoupling SNU', lambda: biv.Decoupling_long('SNU', 1982, 2024).run()),
    ('best-scale SPEI figures', lambda: tsp.Best_scale_SPEI_1982().run()),
    ## 3 LAI: Fig 2, Fig 4, robustness with GIMMS
    ('SNU LAI figures', lambda: tsp.SNU_LAI_time_series().run()),
    ('GIMMS LAI figures', lambda: tsp.GIMMS_LAI_time_series().run()),
    ('SNU vs GIMMS time series', lambda: tsp.LAI_same_period_compare(1982, 2020).ts_overlay()),
    ('SNU vs GIMMS 2000-2020', lambda: tsp.LAI_same_period_compare(2000, 2020).run()),
    ## 4 soil moisture, attribution: Fig S7, Fig 7, Fig 7b
    ('SM sufficiency', lambda: sms.SM_sufficiency(1982, 2024).run()),
    ('long attribution', lambda: att.Long_attribution(1986, 2024).run()),
    ('attribution maps', lambda: att.Long_attribution(1986, 2024).run_maps()),
    ## 5 greening composition, decomposition, phenology: Fig 6, 6b, 6c, phenology
    ('greening cover + decomposition', greening_cover),
    ('phenology', lambda: phe.SNU_phenology().run()),
]


def main():
    print('window:', GS_WINDOW, '| data files *' + WIN, '| outputs in', R2)
    for i, (name, func) in enumerate(step_list):
        if i < START:
            continue
        print(f'\n========== step {i}: {name} ==========')
        func()
    print('\nall steps done')


if __name__ == '__main__':
    main()
