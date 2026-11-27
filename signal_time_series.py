# coding='utf-8'
## Simple-signal figure: regional time series of every product on one time axis, 1982-2024 (2026-11-25)
## why  : first look at the plain signals before any bars or maps: does vegetation follow the drought step around 2000?
## panels (shared x axis, dotted line at 2000):
##   a  GS SPEI at one fixed scale (2026-11-26: was each pixel's best scale), area-weighted mean +- 1 SD (spatial)
##                                                                  <- drought_area_by_year.csv (Drought_analysis)
##      Figure 1 / Fig_all_signals_ts: self.ts_scale (SPEI12: of 6 vs 12 months, 12 is the best scale for the larger
##      area, 20 % vs 15 %); Fig_all_signals_ts_{scale}: panel a and the area panel b at the same scale
##   b  GS LAI, % of each pixel's mean: SNU 1982-2024, GIMMS 1982-2020 (each product's own pixels, as Fig 2)
##                                                                  <- GS_SNU / GS_GIMMS{WIN}.df, recomputed here
##   c  GS VOD, % of each cell's mean: CXKu 1988-2021, L band 2010-2021 shifted to the CXKu 2010-2021 level
##                                                                  <- Fig_VOD_ts_all.csv, Fig_VOD_L_ts_all.csv
##   d  NEE (Tg C per year, + = release), GCB inversion ensemble: median and IQR of the members (years with >= 5
##      members, 2001-2024), plus the two members that start before 2000 (CAMS, CarboScope)
##      2026-11-26: absolute NEE kept (median + IQR, CAMS and CarboScope with their trends); the text of the median is
##      the member agreement on the 2001-2024 trend (6/6 ...) instead of the median trend; all members in SI
## talk (2026-11-26): Fig_talk_signals_{scale}  Figure 2 drawn at slide size (13 x 7.4 in, 15-20 pt), a | b, c | d,
##                  e across the bottom; short trend text (stars only), for the PPT
## SI (2026-11-26): Fig_SI_SPEI_scales_ts  GS SPEI03-48 time series, one row per scale, mean +- 1 SD, trends + p
##                  Fig_SI_wet_dry_area    % area wet (up) / dry (down), one row per scale, 1982-2024
##                                                                  <- GCB_NEE_members.csv (GCB_inversions)
##      (2026-11-25: was CAMS and our own CarboScope download minus their means; our CarboScope sits ~90 Tg C / yr
##       above the GCB CarboScope member, so the GCB versions are used)
## each series: trends over its whole record (dotted), up to 1999 (dashed), 2000 to its end (solid), slope in the panel
## figures: Fig_SPEI_LAI_ts       panels a + b (the drought vs greenness figure)
##          Fig_all_signals_ts    panels a-d
##          Fig_signal_series.csv every plotted series; Fig_signal_trends.csv every trend
## output : Result2_snuwin\00_overview\signal_time_series\
## run after Drought_analysis, VOD_analysis, VOD_L_analysis and GCB_inversions
from Time_series_plotting import *


class Signal_time_series(Long_LAI_time_series):
    def __init__(self):
        super().__init__('SNU', 1982, 2024)
        self.outdir = self.result2w + rf'00_overview\signal_time_series\\'
        T.mk_dir(self.outdir, force=True)
        self.years = np.arange(1982, 2025)
        self.ts_scale = 'SPEI12'   ## 2026-11-26: fixed scale for Figure 1, no best scale
        self.vod_csv = self.result2w + rf'04_water_carbon\VODCA_CXKu\Fig_VOD_ts_all.csv'
        self.vod_l_csv = self.result2w + rf'04_water_carbon\VODCA_L\Fig_VOD_L_ts_all.csv'
        self.gcb_csv = self.result2w + rf'04_water_carbon\GCB_inversions\GCB_NEE_members.csv'
        self.gcb_min_members = 5
        self.gcb_trend = (2001, 2024)  ## 2026-11-26: period for the member agreement text
        self.big_font = {'font.size': 17, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 16,
                         'ytick.labelsize': 16, 'legend.fontsize': 16}
        self.style = {'SPEI best scale': '#D84315', 'SNU': '#2E7D32', 'GIMMS': '#5B9BD5', 'VOD CXKu': '#6D4C41',
                      'VOD L band': '#C9A227', 'GCB median': '#1B7837', 'CAMS (GCB)': '#6A3D9A',
                      'CarboScope (GCB)': '#E08A1E'}
        self.style.update({sc: '#D84315' for sc in ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI36', 'SPEI48']})
        self.lw = {'CAMS (GCB)': 1.4, 'CarboScope (GCB)': 1.4}   ## single members thinner than the median
        ## 2026-11-26: wet / dry area panel (b) of Figure 2, one figure per SPEI scale, from Drought_analysis
        self.area_csv = self.result2w + rf'01_drought\drought_analysis\drought_area_by_year.csv'
        self.area_scales = ['SPEI03', 'SPEI06', 'SPEI12', 'SPEI24', 'SPEI36', 'SPEI48']
        self.wet_classes = [('W1 moderate, SPEI > 1', 'SPEI > 1', '#6BAED6'), ('W2 severe, SPEI > 1.5', 'SPEI > 1.5', '#2171B5'),
                            ('W3 extreme, SPEI > 2', 'SPEI > 2', '#08306B')]
        self.dry_classes = [('D1 moderate, SPEI < -1', 'SPEI < -1', '#F4A259'), ('D2 severe, SPEI < -1.5', 'SPEI < -1.5', '#D9480F'),
                            ('D3 extreme, SPEI < -2', 'SPEI < -2', '#7F1D1D')]

    def run(self):
        s = self.series()
        pd.DataFrame(s).reindex(self.years).rename_axis('year').to_csv(self.outdir + 'Fig_signal_series.csv')
        rows = []
        panels = self.panel_list(self.ts_scale)
        self.figure(s, panels[:2], 'Fig_SPEI_LAI_ts', rows)
        rows = []
        self.figure(s, panels, 'Fig_all_signals_ts', rows)
        self.area_figures(s)   ## 2026-11-26: Figure 2 with the wet / dry area panel, one per scale
        self.si_figures(s)     ## 2026-11-26: SI figures
        for sc in self.area_scales:   ## 2026-11-26: talk version of Figure 2, one per scale
            if sc in s:
                self.talk_figure(s, sc)
        tab = pd.DataFrame(rows, columns=['series', 'period', 'slope_per_decade', 'p_value'])
        tab.to_csv(self.outdir + 'Fig_signal_trends.csv', index=False)
        print(tab.round(3).to_string())

    def panel_list(self, scale):
        ## (panel letter, y label, series names, zero line, show SD band)
        return [('a', f'GS {scale}', [scale], True, True),
                ('b', 'GS LAI\n(% of mean)', ['SNU', 'GIMMS'], True, False),
                ('c', 'GS VOD\n(% of mean)', ['VOD CXKu', 'VOD L band'], True, False),
                ('d', 'NEE, GCB inversions\n(Tg C / yr, + = release)', ['GCB median', 'CAMS (GCB)', 'CarboScope (GCB)'],
                 True, True)]

    def series(self):
        ## name -> pd.Series indexed by year (value), plus name + ' sd' where a band is drawn
        s = {}
        ## 2026-11-26: GS SPEI at fixed scales (area-weighted mean and spatial SD, group All) from Drought_analysis
        tab = pd.read_csv(self.area_csv)
        self.area_tab = tab[(tab['group'] == 'All') & (tab['year'] >= 1982) & (tab['year'] <= 2024)]
        for sc, d in self.area_tab.groupby('scale'):
            d = d.set_index('year').sort_index()
            s[sc] = d['mean_SPEI']
            if 'sd_SPEI' in d:
                s[sc + ' sd'] = d['sd_SPEI']
            else:
                print('no sd_SPEI in', self.area_csv, '-> rerun drought_analysis for the SD band')
        for name, y0, y1 in [('SNU', 1982, 2024), ('GIMMS', 1982, 2020)]:
            lai = Long_LAI_time_series(name, y0, y1)
            df = lai.load()
            yrs = np.arange(y0, y1 + 1)
            s[name] = pd.Series(lai.wmean(df, 'growing_LAI_rel', yrs), index=yrs)
        vod = pd.read_csv(self.vod_csv).set_index('year')['mean']
        vl = pd.read_csv(self.vod_l_csv).set_index('year')['mean']
        common = vl.index.intersection(vod.index)
        s['VOD CXKu'] = vod
        s['VOD L band'] = vl + (vod.loc[common].mean() - vl.loc[common].mean())   ## same level as CXKu 2010-2021
        g = pd.read_csv(self.gcb_csv, index_col=0)
        g.columns = [str(c).replace('\x00', '').strip() for c in g.columns]   ## member names had NUL padding
        g = g[(g.index >= 1982) & (g.index <= 2024)]
        self.gcb_members = g   ## 2026-11-26: for the member agreement text
        n = g.notna().sum(axis=1)
        s['GCB median'] = g.median(axis=1).where(n >= self.gcb_min_members)
        s['GCB median lo'] = g.quantile(0.25, axis=1).where(n >= self.gcb_min_members)
        s['GCB median hi'] = g.quantile(0.75, axis=1).where(n >= self.gcb_min_members)
        for name, key in [('CAMS (GCB)', 'cams_insitu'), ('CarboScope (GCB)', 'carboscope')]:
            col = [c for c in g.columns if c.lower().replace('-', '_') == key]
            if col:
                s[name] = g[col[0]]
            else:
                print('GCB member not found:', key, '| members:', list(g.columns))
                s[name] = pd.Series(dtype=float)
        print('GCB members:', list(g.columns), '| years with >=', self.gcb_min_members, 'members:',
              int(s['GCB median'].notna().sum()))
        return s

    def trends(self, y, name, rows):
        ## whole record (dotted), up to 1999 (dashed), 2000 to the end (solid); a period needs >= 8 years
        y = y.dropna()
        yrs = y.index.values.astype(int)
        t0, t1 = yrs[0], yrs[-1]
        out = []
        for a, b, ls in [(t0, t1, ':'), (t0, 1999, '--'), (2000, t1, '-')]:
            m = (yrs >= a) & (yrs <= b)
            if m.sum() < 8 or (ls != ':' and t0 >= 2000):
                continue
            sl, ic, _, p, _ = stats.linregress(yrs[m], y.values[m])
            out.append((a, b, ls, sl, ic, p))
            rows.append([name, f'{a}-{b}', sl * 10, p])
        return out

    def figure(self, s, panels, fname, rows):
        n = len(panels)
        with plt.rc_context(self.big_font):
            fig, axes = plt.subplots(n, 1, sharex=True, figsize=(16, 3.9 * n))   ## 2026-11-26: 14 -> 16 wide, room for the p values
            axes = np.atleast_1d(axes)
            first_area = True   ## 2026-11-26: legend on the first area panel only
            for ax, (letter, ylab, names, zero, band) in zip(axes, panels):
                if names[0].startswith('area:'):   ## 2026-11-26: wet / dry area bars
                    self.area_panel(ax, names[0].split(':')[1], legend=first_area)
                    first_area = False
                    ax.axvline(2000, color='grey', lw=1.2, ls=':')
                    ax.set_ylabel(ylab)
                    ax.text(-0.09, 1.0, letter, transform=ax.transAxes, fontsize=22, fontweight='bold', va='top')
                    ax.spines[['top', 'right']].set_visible(False)
                    continue
                text = []
                for name in names:
                    y = s[name].dropna()
                    color = self.style[name]
                    if y.empty:
                        continue
                    ax.plot(y.index, y.values, color=color, lw=self.lw.get(name, 2.2), label=name)
                    if band and name + ' sd' in s:
                        sd = s[name + ' sd'].reindex(y.index)
                        ax.fill_between(y.index, y - sd, y + sd, color=color, alpha=0.13, lw=0)
                    if band and name + ' lo' in s:   ## GCB: interquartile range of the members
                        ax.fill_between(y.index, s[name + ' lo'].reindex(y.index), s[name + ' hi'].reindex(y.index),
                                        color=color, alpha=0.2, lw=0)
                    tl = self.trends(y, name, rows)
                    for a, b, ls, sl, ic, p in tl:
                        xx = np.arange(a, b + 1)
                        ax.plot(xx, sl * xx + ic, color=color, lw=2, ls=ls)
                    unit = '/dec' if name.startswith('SPEI') else (
                        ' Tg C/dec' if 'GCB' in name else ' %/dec')
                    if name == 'GCB median':   ## 2026-11-26: member agreement instead of the median trend
                        text.append((self.gcb_agreement(), color))
                        continue
                    ## 2026-11-26: p value written for every period, not only the stars
                    text.append((name + ': ' + ' | '.join(f'{a}-{b} {sl * 10:+.2f}{unit} ({self.p_text(p)}){self.star(p)}'
                                                         for a, b, ls, sl, ic, p in tl), color))
                for k, (t, color) in enumerate(text):
                    ax.text(0.01, 0.04 + 0.11 * (len(text) - 1 - k), t, transform=ax.transAxes, fontsize=13,
                            color=color, va='bottom')
                if zero:
                    ax.axhline(0, color='grey', lw=0.8)
                ax.axvline(2000, color='grey', lw=1.2, ls=':')
                ax.set_ylabel(ylab)
                ax.text(-0.09, 1.0, letter, transform=ax.transAxes, fontsize=22, fontweight='bold', va='top')
                ax.spines[['top', 'right']].set_visible(False)
                if len(names) > 1:
                    ax.legend(frameon=False, loc='upper left', ncol=len(names))
                ylo, yhi = ax.get_ylim()
                ax.set_ylim(ylo - 0.35 * (yhi - ylo) * len(text) / 2, yhi)   ## room for the trend text
            axes[-1].set_xlim(1982, 2024)
            axes[-1].set_xlabel('Year')
            plt.tight_layout()
            plt.savefig(self.outdir + f'{fname}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'{fname}.pdf', bbox_inches='tight')
            plt.close()
        print('saved', fname)

    def star(self, p):
        ## slope per decade with significance marks
        return '**' if p < 0.01 else ('*' if p < 0.05 else '')

    def area_figures(self, s):
        ## Figure 2 with panel b = % of area wet (SPEI > 1 / 1.5 / 2, up) and dry (SPEI < -1 / -1.5 / -2, down), group
        ##   All, 1982-2024; classes are nested (SPEI < -1 includes < -1.5), so the bars are drawn on top of each other
        ## output: Fig_all_signals_ts_{scale}.png / .pdf
        ## 2026-11-26: panel a at the same scale as the area panel
        for sc in self.area_scales:
            if sc not in set(self.area_tab['scale']):
                print(f'{sc} not in {self.area_csv}; add it to Drought_analysis.scale_list and rerun')
                continue
            panels = self.panel_list(sc)
            pp = [panels[0], ('b', f'{sc} area (%)\nwet up, dry down', [f'area:{sc}'], True, False)]
            pp += [(chr(ord(p[0]) + 1),) + tuple(p[1:]) for p in panels[1:]]   ## b c d -> c d e
            self.figure(s, pp, f'Fig_all_signals_ts_{sc}', [])

    def area_panel(self, ax, sc, legend=True):
        d = self.area_tab[self.area_tab['scale'] == sc].set_index('year').sort_index()
        for col, lab, color in self.wet_classes:
            ax.bar(d.index, d[col], width=0.8, color=color, label=lab)
        for col, lab, color in self.dry_classes:
            ax.bar(d.index, -d[col], width=0.8, color=color, label=lab)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylim(-100, 100)
        ax.set_yticks([-100, -50, 0, 50, 100])
        ax.set_yticklabels(['100', '50', '0', '50', '100'])
        if legend:
            ax.legend(frameon=False, ncol=3, loc='upper right', fontsize=13)

    def gcb_agreement(self, short=False):
        ## members covering >= 90 % of self.gcb_trend: how many have a positive / negative NEE trend (and p < 0.05)
        t0, t1 = self.gcb_trend
        tr = []
        for m in self.gcb_members.columns:
            d = self.gcb_members.loc[t0:t1, m].dropna()
            if len(d) < 0.9 * (t1 - t0 + 1):
                continue
            sl, _, _, p, _ = stats.linregress(d.index.values, d.values)
            tr.append((sl * 10, p))
        if not tr:
            return 'GCB: no member covers ' + f'{t0}-{t1}'
        tr = np.array(tr)
        pos, neg = tr[:, 0] > 0, tr[:, 0] < 0
        sig = tr[:, 1] < 0.05
        if short:   ## 2026-11-26: talk version
            return (f'{int(pos.sum())}/{len(tr)} inversions covering {t0}-{t1} positive '
                    f'({int((pos & sig).sum())} with p<0.05), {tr[:, 0].min():+.1f} to {tr[:, 0].max():+.1f}')
        return (f'GCB median: {int(pos.sum())}/{len(tr)} inversions covering {t0}-{t1} have a positive trend '
                f'({int((pos & sig).sum())} with p<0.05), {int(neg.sum())} negative; member trends '
                f'{tr[:, 0].min():+.0f} to {tr[:, 0].max():+.0f} Tg C/dec')

    def talk_figure(self, s, sc):
        ## 2026-11-26: Figure 2 at slide size so the text stays readable from the back of the room
        rc = {'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14}
        with plt.rc_context(rc):
            fig = plt.figure(figsize=(13, 8.4))
            gs = fig.add_gridspec(3, 2, hspace=1.25, wspace=0.42, height_ratios=[1, 1, 1.4])   ## 2026-11-26: wider gap, c text ran into d
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]),
                   fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])]
            n_txt = [self.talk_lines(axs[0], s, [sc], sc, 2)]
            self.area_panel(axs[1], sc, legend=False)
            axs[1].set_yticks([-100, 0, 100])
            axs[1].set_yticklabels(['100', '0', '100'])
            axs[1].set_ylabel('Area (%)')
            axs[1].text(0.99, 0.88, 'wet', transform=axs[1].transAxes, color='#08306B', fontsize=15, ha='right')
            axs[1].text(0.01, 0.04, 'dry', transform=axs[1].transAxes, color='#7F1D1D', fontsize=15)
            n_txt.append(0)
            n_txt.append(self.talk_lines(axs[2], s, ['SNU', 'GIMMS'], 'LAI (%)', 2))
            n_txt.append(self.talk_lines(axs[3], s, ['VOD CXKu', 'VOD L band'], 'VOD (%)', 2))
            n_txt.append(self.talk_lines(axs[4], s, ['GCB median', 'CAMS (GCB)', 'CarboScope (GCB)'], 'NEE (Tg C/yr)', 1))
            units = ['trend per decade', '', 'trend in % per decade', 'trend in % per decade', 'trend in Tg C per decade']
            for k, ax in enumerate(axs):
                if units[k]:   ## unit of the trend numbers, once per panel, next to the letter
                    ax.annotate(units[k], (0, 1), xycoords='axes fraction', xytext=(0, 4 + 19 * max(n_txt[k], 1)),
                                textcoords='offset points', fontsize=13, color='#666666', va='bottom')
                ax.axvline(2000, color='grey', ls=':', lw=1.2)
                ax.set_xlim(1982, 2024)
                ax.spines[['top', 'right']].set_visible(False)
                ax.annotate('abcde'[k], (-0.09 if k < 4 else -0.045, 1), xycoords='axes fraction',
                            xytext=(0, 4 + 19 * max(n_txt[k], 1)), textcoords='offset points', fontsize=20,
                            fontweight='bold', va='bottom')
            plt.savefig(self.outdir + f'Fig_talk_signals_{sc}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.outdir + f'Fig_talk_signals_{sc}.pdf', bbox_inches='tight')
            plt.close()
        print('saved', f'Fig_talk_signals_{sc}')

    def talk_lines(self, ax, s, names, ylab, nd):
        ## lines + trends of one talk panel (whole record dotted, 2000-end solid); short trend text with stars only
        short = {'GCB median': 'GCB median (IQR)', 'CAMS (GCB)': 'CAMS', 'CarboScope (GCB)': 'CarboScope',
                 'VOD CXKu': 'CXKu', 'VOD L band': 'L band'}
        handles, text = [], []
        for name in names:
            y = s[name].dropna()
            if y.empty:
                continue
            color = self.style[name]
            if name + ' sd' in s:
                sd = s[name + ' sd'].reindex(y.index)
                ax.fill_between(y.index, y - sd, y + sd, color=color, alpha=0.15, lw=0)
            if name + ' lo' in s:
                ax.fill_between(y.index, s[name + ' lo'].reindex(y.index), s[name + ' hi'].reindex(y.index),
                                color=color, alpha=0.2, lw=0)
            h, = ax.plot(y.index, y.values, color=color, lw=1.8 if name in self.lw else 2.8, label=short.get(name, name))
            handles.append(h)
            tl = [t for t in self.trends(y, name, []) if t[1] != 1999]   ## 2026-11-26: whole record and 2000-end only
            for a, b, ls, sl, ic, p in tl:
                xx = np.arange(a, b + 1)
                ax.plot(xx, sl * xx + ic, color=color, lw=2.2, ls=ls)
            if name == 'GCB median':
                text.append(('GCB median (IQR): ' + self.gcb_agreement(short=True), color))
                continue
            lab = '' if len(names) == 1 else short.get(name, name).replace(' (IQR)', '') + '   '
            ## 2026-11-26: exact p value for every trend, stars kept
            text.append((lab + '   '.join(f'{str(a)[2:]}-{str(b)[2:]}: {sl * 10:+.{nd}f} ({self.p_text(p)}){self.star(p)}'
                                          for a, b, ls, sl, ic, p in tl), color))
        ## trend text above the panel (more room for the lines); the coloured series names replace the legend;
        pos = [(k, 0) for k in range(len(text))]
        n_row = max(r for r, _ in pos) + 1 if text else 0
        for (t, color), (r, x) in zip(text, pos):
            ax.annotate(t, (x, 1), xycoords='axes fraction', xytext=(0, 4 + 19 * (n_row - 1 - r)),
                        textcoords='offset points', fontsize=14, color=color, va='bottom')
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_ylabel(ylab)
        return n_row

    def si_figures(self, s):
        ## 2026-11-26: SI 1 = GS SPEI time series at every scale; SI 2 = wet / dry area at every scale
        scales = [sc for sc in self.area_scales if sc in s]
        letters = 'abcdefgh'
        self.figure(s, [(letters[k], f'GS {sc}', [sc], True, True) for k, sc in enumerate(scales)],
                    'Fig_SI_SPEI_scales_ts', [])
        self.figure(s, [(letters[k], f'{sc} area (%)\nwet up, dry down', [f'area:{sc}'], True, False)
                        for k, sc in enumerate(scales)], 'Fig_SI_wet_dry_area', [])

    def p_text(self, p):
        ## 2026-11-26: p value as text, e.g. p=0.24, p<0.001
        return 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'


def main():
    Signal_time_series().run()


if __name__ == '__main__':
    main()
