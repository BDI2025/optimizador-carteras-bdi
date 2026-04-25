# ============================================================
#  OPTIMIZADOR DE CARTERA PROFESIONAL â STREAMLIT
#  BDI Consultora de Inversiones
#  VersiÃ³n 2.0 â Modelo de Markowitz con Frontera Eficiente
# ============================================================
import warnings
warnings.filterwarnings('ignore')
import io

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize
from datetime import datetime

# âââââââââââââââââââââââââââââââââââââââââââââ
#  PAGE CONFIG  (debe ir primero)
# âââââââââââââââââââââââââââââââââââââââââââââ
st.set_page_config(
    page_title="BDI â Optimizador de Carteras",
    page_icon="â¡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# âââââââââââââââââââââââââââââââââââââââââââââ
#  PALETA CORPORATIVA BDI (oficial)
# âââââââââââââââââââââââââââââââââââââââââââââ
BDI_GREEN    = '#137247'   # Verde principal BDI
BDI_CHARCOAL = '#323232'   # Fondo oscuro
BDI_CREAM    = '#EFEDEA'   # Texto claro
BDI_TEAL     = '#17BEBB'   # Turquesa
BDI_LIME     = '#B5E61D'   # Lima / acento

BDI_DARK_BG  = '#1c1c1c'
BDI_CARD_BG  = '#282828'
BDI_BORDER   = '#404040'
BDI_GREEN_DK = '#0d4d2e'
BDI_MUTED    = '#9e9e9e'

PALETTE = [
    BDI_TEAL, BDI_LIME, '#ffa726', '#ef5350',
    '#ab47bc', '#42a5f5', '#26a69a', '#d4e157',
    '#ff7043', '#7e57c2', '#26c6da', '#66bb6a',
]

PORT_COLORS = [BDI_GREEN, BDI_TEAL, BDI_LIME, '#ffa726', '#ab47bc']

plt.rcParams.update({
    'figure.facecolor': BDI_DARK_BG,
    'axes.facecolor':   '#232323',
    'axes.edgecolor':   BDI_BORDER,
    'axes.labelcolor':  BDI_CREAM,
    'axes.titlecolor':  BDI_CREAM,
    'xtick.color':      BDI_MUTED,
    'ytick.color':      BDI_MUTED,
    'text.color':       BDI_CREAM,
    'grid.color':       '#383838',
    'grid.linewidth':   0.5,
    'legend.facecolor': '#282828',
    'legend.edgecolor': BDI_BORDER,
    'legend.labelcolor': BDI_CREAM,
    'font.family':      'DejaVu Sans',
})

# âââââââââââââââââââââââââââââââââââââââââââââ
#  CSS PERSONALIZADO
# âââââââââââââââââââââââââââââââââââââââââââââ
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #1c1c1c; color: #EFEDEA; }

    /* Sidebar â fondo claro BDI */
    [data-testid="stSidebar"] {
        background-color: #f0f7f2 !important;
        border-right: 3px solid #137247;
    }
    [data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
        background-color: #f0f7f2 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span:not([class*="color"]),
    [data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {
        color: #1c1c1c !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #137247 !important; }
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stNumberInput > div > div > input,
    [data-testid="stSidebar"] .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        color: #1c1c1c !important;
        border: 1px solid #137247 !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] span { color: #1c1c1c !important; }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label { color: #1c1c1c !important; }
    [data-testid="stSidebar"] hr { border-color: #137247 !important; opacity: 0.3; }
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stCaption { color: #137247 !important; }
    [data-testid="stSidebar"] code {
        background-color: #e8f5e9 !important;
        color: #137247 !important;
        border-radius: 3px !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #137247, #0d4d2e);
        color: #EFEDEA !important;
        border: 1px solid #17BEBB;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #B5E61D, #8aaa14) !important;
        color: #1c1c1c !important;
        border-color: #B5E61D !important;
    }
    /* NÃºmeros del slider en sidebar */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"] {
        color: #137247 !important;
    }

    /* TÃ­tulos globales */
    h1, h2, h3 { color: #B5E61D !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #282828;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 10px 14px;
    }
    [data-testid="stMetricValue"] {
        color: #B5E61D !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] { color: #17BEBB !important; font-size: 0.85rem !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #232323;
        border-bottom: 2px solid #137247;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #EFEDEA;
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        color: #B5E61D !important;
        border-bottom: 2px solid #B5E61D !important;
        background-color: #2a2a2a !important;
    }

    /* BotÃ³n principal */
    .stButton > button {
        background: linear-gradient(135deg, #137247, #0d4d2e);
        color: #EFEDEA;
        border: 1px solid #17BEBB;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #B5E61D, #8aaa14);
        color: #1c1c1c !important;
        border-color: #B5E61D;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #282828 !important;
        color: #EFEDEA !important;
        border: 1px solid #404040 !important;
        border-radius: 6px !important;
    }

    /* Slider */
    [data-testid="stSlider"] > div { color: #EFEDEA; }

    /* Checkbox */
    [data-testid="stCheckbox"] { color: #EFEDEA; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #404040; border-radius: 8px; }

    /* Divisor */
    hr { border-color: #404040; }

    /* Banner header */
    .bdi-header {
        background: linear-gradient(135deg, #0d4d2e 0%, #137247 50%, #0e4a5a 100%);
        border: 1px solid #17BEBB;
        border-radius: 14px;
        padding: 2rem 2.4rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Card de portfolio */
    .port-card {
        border-radius: 8px;
        padding: 0.7rem 1.4rem;
        margin: 0.7rem 0 0.3rem 0;
    }

    /* Info box para explicaciones */
    .info-box {
        background-color: #1e2e22;
        border-left: 4px solid #137247;
        border-radius: 0 8px 8px o;
        padding: 1rem 1.4rem;
        margin-bottom: 1.2rem;
        color: #EFEDEA;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    /* Instructivo pasos */
    .step-box {
        background-color: #252525;
        border: 1px solid #404040;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin: 0.5rem 0;
        color: #EFEDEA;
        font-size: 0.93rem;
        line-height: 1.6;
    }
    .step-num {
        display: inline-block;
        background: #137247;
        color: #EFEDEA;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-right: 10px;
        font-size: 0.95rem;
    }

    /* Ideas box */
    .ideas-box {
        background: linear-gradient(135deg, #0d2e1e, #1a3a2e);
        border: 1px solid #137247;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 2rem;
        color: #EFEDEA;
    }
    .idea-item {
        background: #232323;
        border-left: 4px solid #17BEBB;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.3rem;
        margin: 0.8rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Advertencia legal */
    .legal-warning {
        background-color: #1e1a00;
        border: 1px solid #ffa000;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-top: 2rem;
        color: #EFEDEA;
        font-size: 0.84rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# âââââââââââââââââââââââââââââââââââââââââââââ
#  FUNCIONES AUXILIARES
# âââââââââââââââââââââââââââââââââââââââââââââ
def pct(x):
    return f"{x * 100:.2f}%"

def add_bdi_watermark(fig):
    fig.text(0.99, 0.01, 'BDI Consultora de Inversiones',
             ha='right', va='bottom', fontsize=7,
             color=BDI_TEAL, alpha=0.5, style='italic')
    fig.text(0.01, 0.01, datetime.now().strftime('%d/%m/%Y'),
             ha='left', va='bottom', fontsize=7,
             color=BDI_MUTED, alpha=0.5)

def port_stats(w, mean_ret, cov_mat, rf):
    ret    = np.dot(w, mean_ret)
    vol    = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe

def calc_max_drawdown(series):
    cum      = (1 + series).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    return dd.min()

def calc_sortino(series, rf_daily):
    excess    = series - rf_daily
    neg_ret   = excess[excess < 0]
    downside  = np.sqrt((neg_ret ** 2).mean()) * np.sqrt(252)
    ann_excess = excess.mean() * 252
    return ann_excess / downside if downside > 0 else np.nan

def calc_cagr(cum_series):
    days  = (cum_series.index[-1] - cum_series.index[0]).days
    years = days / 365.25
    total = cum_series.iloc[-1] + 1
    return total ** (1 / years) - 1 if years > 0 else np.nan

# âââââââââââââââââââââââââââââââââââââââââââââ
#  GENERADOR DE REPORTE PDF
# âââââââââââââââââââââââââââââââââââââââââââââ
def generate_pdf_report(assets, portfolios, metricas, port_daily, cum_port, cum_assets,
                        corr_matrix, sim_vol_arr, sim_ret_arr, sim_sharpe_arr,
                        vol_fe, ret_range, vol_min, vol_sharpe, vol_eq, ret_eq,
                        ret_sharpe, ret_vol, sharpe_val, rf, anios, data_start, data_end,
                        w_obj_arr, vol_obj, ret_obj_real, obj_label,
                        has_custom_weights, vol_custom, ret_custom, cliente_nombre=""):
    """Genera un informe PDF completo con 8 pÃ¡ginas y marca BDI."""
    num_assets = len(assets)
    buf        = io.BytesIO()
    fecha_hoy  = datetime.now().strftime('%d/%m/%Y  %H:%M')
    AW, AH     = 11.69, 8.27   # landscape A4

    def _wmark(fig):
        fig.text(0.99, 0.01, 'BDI Consultora de Inversiones',
                 ha='right', va='bottom', fontsize=7, color=BDI_TEAL, alpha=0.5, style='italic')
        fig.text(0.01, 0.01, fecha_hoy,
                 ha='left',  va='bottom', fontsize=7, color=BDI_MUTED, alpha=0.5)

    with PdfPages(buf) as pdf:

        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        # PÃGINA 1 â CARÃTULA (portrait A4)
        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(BDI_DARK_BG)

        ax_b = fig.add_axes([0.04, 0.83, 0.92, 0.12])
        ax_b.set_facecolor(BDI_GREEN_DK)
        for sp in ax_b.spines.values():
            sp.set_edgecolor(BDI_TEAL); sp.set_linewidth(2)
        ax_b.axis('off')
        ax_b.text(0.5, 0.70, 'â¡  BDI â OPTIMIZADOR DE CARTERAS v2.0',
                  ha='center', va='center', fontsize=17, fontweight='bold',
                  color=BDI_CREAM, transform=ax_b.transAxes)
        ax_b.text(0.5, 0.28, 'Modelo de Markowitz con Frontera Eficiente',
                  ha='center', va='center', fontsize=11, color=BDI_LIME,
                  transform=ax_b.transAxes)

        ax_i = fig.add_axes([0.04, 0.55, 0.92, 0.26])
        ax_i.set_facecolor(BDI_CARD_BG)
        for sp in ax_i.spines.values():
            sp.set_edgecolor(BDI_BORDER); sp.set_linewidth(1)
        ax_i.axis('off')
        info_rows = []
        if cliente_nombre:
            info_rows.append(('ð¤ Cliente:', cliente_nombre))
        info_rows += [
            ('ð Fecha:', fecha_hoy),
            ('ð PerÃ­odo:', f'{anios} aÃ±os  ({data_start} â {data_end})'),
            ('ð Activos:', ', '.join(assets)),
            ('ð¼ PortFolios:', ', '.join(portfolios.keys())),
        ]
        y0 = 0.90
        for lbl, val in info_rows:
            ax_i.text(0.03, y0, lbl, fontsize=9, fontweight='bold',
                      color=BDI_TEAL, transform=ax_i.transAxes, va='top')
            ax_i.text(0.25, y0, val, fontsize=9, color=BDI_CREAM,
                      transform=ax_i.transAxes, va='top')
            y0 -= 0.17

        ax_t = fig.add_axes([0.04, 0.18, 0.92, 0.34])
        ax_t.set_facecolor(BDI_CARD_BG)
        ax_t.axis('off')
        ax_t.text(0.5, 0.96, 'ð Resumen de Resultados',
                  ha='center', va='top', fontsize=11, fontweight='bold',
                  color=BDI_LIME, transform=ax_t.transAxes)
        cols_h = ['Portfolio', 'Retorno', 'Volatilidad', 'Sharpe', 'CAGR', 'Max DD']
        col_xs = [0.01, 0.22, 0.37, 0.52, 0.66, 0.81]
        for j, (h, cx) in enumerate(zip(cols_h, col_xs)):
            ax_t.text(cx, 0.82, h, fontsize=8.5, fontweight='bold',
                      color=BDI_TEAL, transform=ax_t.transAxes, va='top')
        for i, (name, m) in enumerate(metricas.items()):
            yrow = 0.70 - i * 0.152
            c_row = PORT_COLORS[i % len(PORT_COLORS)]
            row_v = [name, pct(m['Retorno Anual']), pct(m['Volatilidad']),
                     f"{m['Sharpe Ratio']:.3f}", pct(m['CAGR']), pct(m['MÃ¡x. Drawdown'])]
            for j, (v, cx) in enumerate(zip(row_v, col_xs)):
                ax_t.text(cx, yrow, v, fontsize=8.5, va='top',
                          color=c_row if j == 0 else BDI_CREAM,
                          fontweight='bold' if j == 0 else 'normal',
                          transform=ax_t.transAxes)

        fig.text(0.5, 0.13, 'BDI Consultora de Inversiones  Â·  bdiconsultora@gmail.com  Â·  Mariano Ricciardi',
                 ha='center', fontsize=9, color=BDI_TEAL, style='italic')
        fig.text(0.5, 0.10, 'â ï¸ AnÃ¡lisis informativo. No constituye recomendaciÃ³n de inversiÃ³n.',
                 ha='center', fontsize=8, color=BDI_MUTED)
        fig.text(0.5, 0.07, f'Generado: {fecha_hoy}',
                 ha='center', fontsize=8, color=BDI_MUTED, alpha=0.7)
        pdf.savefig(fig, facecolor=BDI_DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        # PÃGINA 2 â ESPACIO DE MARKOWITZ
        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        fig, ax = plt.subplots(figsize=(AW, AH))
        fig.patch.set_facecolor(BDI_DARK_BG)
        sc = ax.scatter(sim_vol_arr * 100, sim_ret_arr * 100,
                        c=sim_sharpe_arr, cmap='plasma', alpha=0.22, s=4, zorder=1)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('Sharpe Ratio', color=BDI_CREAM, fontsize=10)
        cb.ax.yaxis.set_tick_params(color=BDI_CREAM, labelsize=9)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=BDI_CREAM)
        valid = ~np.isnan(vol_fe)
        ax.plot(vol_fe[valid] * 100, ret_range[valid] * 100,
                color=BDI_TEAL, linewidth=3.5, zorder=6,
                label='Frontera Eficiente', solid_capstyle='round')
        vcr = np.linspace(vol_min * 0.7, vol_sharpe * 1.3, 80)
        ax.plot(vcr * 100, rf * 100 + sharpe_val * vcr * 100,
                '--', color=BDI_LIME, linewidth=1.8, alpha=0.9, zorder=4, label='CML')
        pts = [(vol_sharpe, ret_sharpe, '*', 350, BDI_LIME,   f'MÃ¡x. Sharpe ({sharpe_val:.2f})'),
               (vol_min,    ret_vol,    'D', 130, '#ef5350', f'MÃ­n. Vol. ({pct(vol_min)})'),
               (vol_eq,     ret_eq,     's', 130, BDI_TEAL,  f'Equip. ({pct(ret_eq)})')]
        if w_obj_arr is not None:
            pts.append((vol_obj, ret_obj_real, 'P', 150, '#ffa726',
                        f'{obj_label} ({pct(ret_obj_real)})'))
        if has_custom_weights and not np.isnan(vol_custom):
            pts.append((vol_custom, ret_custom, '^', 150, '#ab47bc',
                        f'Personalizada ({pct(ret_custom)})'))
        for vp, rp, mk, sz, cl, lb in pts:
            ax.scatter(vp * 100, rp * 100, marker=mk, s=sz, color=cl,
                       zorder=10, edgecolors='white', linewidth=0.8, label=lb)
        ax.axhline(rf * 100, color=BDI_MUTED, linewidth=0.6, linestyle=':',
                   alpha=0.5, label=f'Rf={pct(rf)}')
        xp = (sim_vol_arr.max() - sim_vol_arr.min()) * 100 * 0.12
        yp = (sim_ret_arr.max() - sim_ret_arr.min()) * 100 * 0.15
        ax.set_xlim(sim_vol_arr.min() * 100 - xp, sim_vol_arr.max() * 100 + xp)
        ax.set_ylim(sim_ret_arr.min() * 100 - yp, sim_ret_arr.max() * 100 + yp)
        ax.set_title('Espacio de Portfolios â Modelo de Markowitz',
                     fontsize=14, fontweight='bold', color=BDI_CREAM, pad=12)
        ax.set_xlabel('Volatilidad Anual (%)', fontsize=11)
        ax.set_ylabel('Retorno Anual (%)', fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=9, framealpha=0.9, loc='upper left')
        ax.grid(True, alpha=0.18)
        _wmark(fig); plt.tight_layout()
        pdf.savefig(fig, facecolor=BDI_DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        # PÃGINA 3 â COMPOSICIÃN (hasta 3 donuts por pÃ¡gina)
        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        all_ports = list(portfolios.items())
        for bs in range(0, len(all_ports), 3):
            batch = all_ports[bs:bs + 3]
            nb = len(batch)
            fig, axes_d = plt.subplots(1, nb, figsize=(AW, AH))
            fig.patch.set_facecolor(BDI_DARK_BG)
            if nb == 1:
                axes_d = [axes_d]
            for ax, (name, w) in zip(axes_d, batch):
                mask   = w > 0.005
                lbs    = [assets[i] for i in range(num_assets) if mask[i]]
                szs    = w[mask]
                other  = 1 - szs.sum()
                if other > 0.001:
                    lbs.append('Otros'); szs = np.append(szs, other)
                cols_d = PALETTE[:len(lbs)]
                wedges, _, atxts = ax.pie(
                    szs, labels=None, colors=cols_d,
                    autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                    startangle=90, pctdistance=0.75,
                    wedgeprops={'linewidth': 1.5, 'edgecolor': BDI_DARK_BG, 'width': 0.60},
                )
                for at in atxts:
                    at.set_fontsize(9); at.set_color('white'); at.set_fontweight('bold')
                m = metricas[name]
                ax.text(0,  0.12, pct(m['Retorno Anual']),
                        ha='center', va='center', fontsize=13, fontweight='bold', color=BDI_LIME)
                ax.text(0, -0.15, f"Sharpe: {m['Sharpe Ratio']:.2f}",
                        ha='center', va='center', fontsize=9, color=BDI_CREAM)
                ax.text(0, -0.34, f"Vol: {pct(m['Volatilidad'])}",
                        ha='center', va='center', fontsize=9, color=BDI_TEAL)
                ax.legend(wedges, [f"{lb} ({s*100:.1f}%)" for lb, s in zip(lbs, szs)],
                          loc='lower center', bbox_to_anchor=(0.5, -0.38),
                          fontsize=7, framealpha=0.6, ncol=2)
                ax.set_title(name, fontsize=10, fontweight='bold', color=BDI_CREAM, pad=10)
            fig.suptitle('ComposiciÃ³n de Portfolios Ãptimos',
                         fontsize=13, fontweight='bold', color=BDI_CREAM, y=1.01)
            _wmark(fig); plt.tight_layout()
            pdf.savefig(fig, facecolor=BDI_DARK_BG, bbox_inches='tight')
            plt.close(fig)

        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        # PÃGINA 4 â RENDIMIENTO ACUMULADO
        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(AW, AH))
        fig.patch.set_facecolor(BDI_DARK_BG)
        for i, col in enumerate(cum_port.columns):
            ax1.plot(cum_port.index, cum_port[col] * 100,
                     label=col, linewidth=2.0, color=PORT_COLORS[i % len(PORT_COLORS)])
        ax1.axhline(0, color=BDI_MUTED, linewidth=0.5, linestyle='--', alpha=0.4)
        ax1.set_title('Rendimiento Acumulado â Portfolios',
                      fontsize=12, fontweight='bold', color=BDI_CREAM)
        ax1.set_ylabel('Rend. Acumulado (%)', fontsize=10)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax1.legend(fontsize=9, framealpha=0.85)
        ax1.grid(True, alpha=0.18)
        for i, col in enumerate(cum_assets.columns):
            ax2.plot(cum_assets.index, cum_assets[col] * 100,
                     label=col, linewidth=1.5,
                     color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax2.axhline(0, color=BDI_MUTED, linewidth=0.5, linestyle='--', alpha=0.4)
        ax2.set_title('Rendimiento Acumulado â Activos Individuales',
                      fontsize=12, fontweight='bold', color=BDI_CREAM)
        ax2.set_ylabel('Rend. Acumulado (%)', fontsize=10)
        ax2.set_xlabel('Fecha', fontsize=10)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax2.legend(fontsize=8, framealpha=0.80,
                   bbox_to_anchor=(1.01, 1), loc='upper left')
        ax2.grid(True, alpha=0.18)
        _wmark(fig); plt.tight_layout()
        pdf.savefig(fig, facecolor=BDI_DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        # PÃGINA 5 â MÃTRICAS COMPARATIVAS
        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        pn_p  = list(portfolios.keys())
        mp_p  = [('Retorno Anual', 'Retorno Anual (%)', True),
                 ('Volatilidad',   'Volatilidad (%)',   False),
                 ('Sharpe Ratio',  'Sharpe Ratio',      True),
                 ('MÃ¡x. Drawdown', 'MÃ¡x. Drawdown (%)', False)]
        fig, axes_m = plt.subplots(2, 2, figsize=(AW, AH))
        fig.patch.set_facecolor(BDI_DARK_BG)
        for ax, (col, ylabel, hb) in zip(axes_m.flatten(), mp_p):
            vals = [metricas[p][col] for p in pn_p]
            sc   = 100 if '%' in ylabel else 1
            bars = ax.bar(pn_p, [v * sc for v in vals],
                          color=PORT_COLORS[:len(pn_p)],
                          edgecolor=BDI_DARK_BG, linewidth=1.0, width=0.5)
            bidx = np.argmax(vals) if hb else np.argmin(vals)
            bars[bidx].set_edgecolor(BDI_LIME); bars[bidx].set_linewidth(2.5)
            for bar, val in zip(bars, vals):
                lbl = f'{val*100:.1f}%' if '%' in ylabel else f'{val:.2f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(bar.get_height()) * 0.03,
                        lbl, ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=BDI_CREAM)
            ax.set_title(ylabel, fontsize=11, fontweight='bold', color=BDI_CREAM)
            if '%' in ylabel:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
            ax.set_xticklabels(pn_p, rotation=15, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.18)
        fig.suptitle('ComparaciÃ³n de MÃ©tricas â Portfolios Ãptimos',
                     fontsize=13, fontweight='bold', color=BDI_CREAM, y=1.01)
        _wmark(fig); plt.tight_layout()
        pdf.savefig(fig, facecolor=BDI_DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        # PÃGINA 6 â CORRELACIÃN
        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        fig, ax = plt.subplots(figsize=(max(7, num_assets + 2), max(6, num_assets + 1)))
        fig.patch.set_facecolor(BDI_DARK_BG)
        cmap_r = LinearSegmentedColormap.from_list(
            'BDI_R',
            ['#ffebee', '#ffcdd2', '#ef9a9a', '#e57373',
             '#ef5350', '#e53935', '#c62828', '#b71c1c', '#7f0000'], N=256)
        sns.heatmap(corr_matrix, ax=ax, annot=True, fmt='.2f',
                    annot_kws={'size': 9, 'weight': 'bold'},
                    cmap=cmap_r, vmin=0, vmax=1,
                    linewidths=0.6, linecolor=BDI_DARK_BG, square=True,
                    cbar_kws={'shrink': 0.8, 'label': 'CorrelaciÃ³n', 'pad': 0.02})
        for to in ax.texts:
            try:
                v = float(to.get_text())
                to.set_color('#1c1c1c' if v < 0.55 else '#EFEDEA')
            except Exception:
                pass
        ax.set_title('Matriz de CorrelaciÃ³n entre Activos',
                     fontsize=13, fontweight='bold', color=BDI_CREAM, pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=9, color=BDI_CREAM)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=9, color=BDI_CREAM)
        _wmark(fig); plt.tight_layout()
        pdf.savefig(fig, facecolor=BDI_DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        # PÃGINA 7 â CAGR COMPARATIVO
        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        cagr_a  = {a: calc_cagr(cum_assets[a]) * 100 for a in assets}
        cagr_p  = {p: calc_cagr(cum_port[p])   * 100 for p in port_daily.columns}
        all_n   = list(cagr_a) + list(cagr_p)
        all_v   = list(cagr_a.values()) + list(cagr_p.values())
        all_ty  = (['Activo'] * len(cagr_a) +
                   ['Personalizada' if p == 'Personalizada' else 'Portfolio'
                    for p in cagr_p])
        tc      = {'Activo': BDI_TEAL, 'PortFolio': BDI_LIME, 'Personalizada': '#ab47bc'}
        sd      = sorted(zip(all_v, all_n, all_ty), reverse=True)
        sv, sn, sty = zip(*sd)
        fig, ax = plt.subplots(figsize=(max(9, len(all_n) * 1.1), AH))
        fig.patch.set_facecolor(BDI_DARK_BG)
        bars_c = ax.bar(sn, sv, color=[tc.get(t, BDI_MUTED) for t in sty],
                        edgecolor=BDI_DARK_BG, linewidth=1.0, width=0.6)
        for bar, val in zip(bars_c, sv):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.4 if val >= 0 else -1.2),
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=BDI_CREAM)
        ax.axhline(0, color=BDI_MUTED, linewidth=0.7, linestyle='--', alpha=0.5)
        ax.set_title('CAGR Anual Comparativo â Activos y Portfolios',
                     fontsize=13, fontweight='bold', color=BDI_CREAM, pad=10)
        ax.set_ylabel('CAGR Anual (%)', fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.set_xticklabels(sn, rotation=30, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.18)
        ax.legend(handles=[mpatches.Patch(color=c, label=t) for t, c in tc.items()],
                  fontsize=9, framealpha=0.85)
        _wmark(fig); plt.tight_layout()
        pdf.savefig(fig, facecolor=BDI_DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        # PÃGINA 8 â AVISO LEGAL (portrait A4)
        # ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(BDI_DARK_BG)
        ax_d = fig.add_axes([0.08, 0.28, 0.84, 0.52])
        ax_d.set_facecolor(BDI_CARD_BG)
        for sp in ax_d.spines.values():
            sp.set_edgecolor('#ffa000'); sp.set_linewidth(1.5)
        ax_d.axis('off')
        fig.text(0.5, 0.84, 'â ï¸  ADVERTENCIA LEGAL',
                 ha='center', fontsize=16, fontweight='bold', color='#ffa000')
        lines_d = [
            'Este anÃ¡lisis es de carÃ¡cter exclusivamente informativo y educativo.',
            'No constituye asesoramiento financiero ni una recomendaciÃ³n de inversiÃ³n.',
            'Los rendimientos pasados no garantizan resultados futuros.',
            '',
            'Las proyecciones y simulaciones presentadas tienen fines Ãºnicamente',
            'ilustrativos y estÃ¡n basadas en datos histÃ³ricos de acceso pÃºblico.',
            '',
            'Toda decisiÃ³n de inversiÃ³n debe tomarse con asesoramiento',
            'profesional apropiado y considerando el perfil de riesgo individual.',
        ]
        yld = 0.88
        for line in lines_d:
            ax_d.text(0.5, yld, line, ha='center', va='top', fontsize=10,
                      color=BDI_CREAM, transform=ax_d.transAxes)
            yld -= 0.10
        ax_d.text(0.5, 0.08,
                  'BDI Consultora de Inversiones  Â·  bdiconsultora@gmail.com  Â·  Mariano Ricciardi',
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  color=BDI_TEAL, transform=ax_d.transAxes)
        fig.text(0.5, 0.10, f'Reporte generado: {fecha_hoy}',
                 ha='center', fontsize=9, color=BDI_MUTED)
        pdf.savefig(fig, facecolor=BDI_DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # Metadata del PDF
        d = pdf.infodict()
        d['Title']   = 'BDI â AnÃ¡lisis y OptimizaciÃ³n de Cartera'
        d['Author']  = 'BDI Consultora de Inversiones â Mariano Ricciardi'
        d['Subject'] = f'Portafolio: {", ".join(H
        d['Creator'] = 'BDI Optimizador de Carteras v2.0'

    buf.seek(0)
    return buf.getvalue()

# âââââââââââââââââââââââââââââââââââââââââââââ
#  HEADER PRINCIPAL
# âââââââââââââââââââââââââââââââââââââââââââââ
st.markdown("""
<div class="bdi-header">
    <h1 style="color:#EFEDEA; margin:0; font-size:2.4rem; letter-spacing:3px; font-weight:800;">
        â¡ BDI â OPTIMIZADOR DE CARTERAS v2.0
    </h1>
    <p style="color:#B5E61D; margin:0.5rem 0 0.2rem 0; font-size:1.05rem; font-weight:600; letter-spacing:1px;">
        Modelo de Markowitz con Frontera Eficiente
    </p>
    <p style="color:#EFEDEA; margin:0; font-size:0.9rem; opacity:0.85;">
        BDI Consultora de Inversiones â Mariano Ricciardi
    </p>
</div>
""", unsafe_allow_html=True)

# âââââââââââââââââââââââââââââââââââââââââââââ
#  SIDEBAR â CONFIGURACIÃN
# âââââââââââââââââââââââââââââââââââââââââââââ
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:0.8rem 0 0.4rem 0;
                background:linear-gradient(135deg,#0d4d2e,#137247);
                border-radius:10px; margin-bottom:0.5rem;">
        <h2 style="color:#B5E61D; font-size:1.2rem; margin:0;">âï¸ ConfiguraciÃ³n</h2>
        <p style="color:#EFEDEA; font-size:0.82rem; margin:0.2rem 0 0 0;">ParÃ¡metros de optimizaciÃ³n</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">ð Activos a analizar</p>', unsafe_allow_html=True)
    tickers_input = st.text_area(
        "Tickers separados por coma",
        value="JPM:20, AAPL:30, MSFT:50, GOOGL:5, META:10, V:5",
        height=85,
        help="UsÃ¡ el mismo formato de ticker que figura en Yahoo Finance. Ej: JPM, AAPL, MSFT, MELI, GOOGL, SPY, GLD, XOM.\n\n"
             "Para agregar una cartera personalizada indicÃ¡ el peso (%): JPM:30, AAPL:25, MSFT:20, GOOGL:15, MELI:10",
    )
    st.caption(
        "ð¡ **Tip:** PodÃ©s especificar pesos para una cartera personalizada. "
        "Ej: `JPM:30, AAPL:25, MSFT:20, GOOGL:15, MELI:10`"
    )

    st.markdown("---")
    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">ð PerÃ­odo de anÃ¡lisis</p>', unsafe_allow_html=True)
    anios = st.slider("AÃ±os de historia", min_value=1, max_value=15, value=5)

    st.markdown("---")
    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">ð ParÃ¡metros de mercado</p>', unsafe_allow_html=True)
    rf_pct = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0, max_value=30.0,
        value=4.5, step=0.1,
        help="Ej: 4.5 equivale al 4.5% anual",
    )
    rf = rf_pct / 100

    st.markdown("---")
    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">ð¯ Retorno objetivo <em style="color:#555;font-size:0.85em;">(opcional)</em></p>', unsafe_allow_html=True)
    usar_objetivo = st.checkbox("Activar retorno objetivo", value=False)
    ret_obj       = None
    objetivo_activo = False
    if usar_objetivo:
        ret_obj_pct = st.number_input(
            "Retorno objetivo anual (%)",
            min_value=-50.0, max_value=300.0,
            value=15.0, step=0.5,
        )
        ret_obj         = ret_obj_pct / 100
        objetivo_activo = True

    st.markdown("---")
    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">âï¸ Restricciones de pesos</p>', unsafe_allow_html=True)
    min_peso = st.slider("Peso mÃ­nimo por activo (%)", 0, 50, 0) / 100
    max_peso = st.slider("Peso mÃ¡ximo por activo (%)", 10, 100, 100) / 100
    if max_peso < min_peso:
        max_peso = min_peso

    st.markdown("---")
    run_button = st.button("ð  EJECUTAR ANÃLISIS", type="primary")

# âââââââââââââââââââââââââââââââââââââââââââââ
#  ESTADO INICIAL (pantalla de bienvenida + instructivo)
# âââââââââââââââââââââââââââââââââââââââââââââ
if not run_button and 'results_ready' not in st.session_state:

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div style="text-align:center; padding:2rem 1rem 1rem 1rem;">
            <p style="font-size:4rem; margin:0;">ð</p>
            <h3 style="color:#17BEBB; margin:1rem 0 0.5rem 0; font-size:1.5rem;">
                Listo para optimizar tu cartera
            </h3>
            <p style="font-size:1rem; margin:0; color:#EFEDEA; line-height:1.7;">
                IngresÃ¡ los activos, configurÃ¡ los parÃ¡metros<br/>
                y presionÃ¡ <strong style="color:#B5E61D;">ð EJECUTAR ANÃLISIS</strong><br/>
                para obtener tu portafolio Ã³ptimo.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:1.5rem;">
            <strong style="color:#B5E61D;">ð Tickers de ejemplo (formato Yahoo Finance):</strong><br/><br/>
            <strong style="color:#17BEBB;">Acciones USA:</strong>
            JPM &nbsp;Â·&nbsp; AAPL &nbsp;Â·&nbsp; MSFT &nbsp;Â·&nbsp; GOOGL &nbsp;Â·&nbsp; AMZN &nbsp;Â·&nbsp; NVDA &nbsp;Â·&nbsp; XOM<br/>
            <strong style="color:#17BEBB;">ETFs:</strong>
            SPY &nbsp;Â·&nbsp; QQQ &nbsp;Â·&nbsp; IWM &nbsp;Â·&nbsp; GLD &nbsp;Â·&nbsp; TLT<br/>
            <strong style="color:#17BEBB;">LatinoamÃ©rica:</strong>
            MELI &nbsp;Â·&nbsp; NU &nbsp;Â·&nbsp; PBR<br/><br/>
            <strong style="color:#B5E61D;">ð¼ Cartera personalizada con pesos:</strong><br/>
            <span style="color:#EFEDEA; font-size:0.9rem;">
                AgregÃ¡ el peso (%) despuÃ©s de cada ticker con <code>:</code><br/>
                Ej: <code>JPM:30, AAPL:25, MSFT:20, GOOGL:15, MELI:10</code>
            </span><br/><br/>
            <span style="color:#9e9e9e; font-size:0.85rem;">
                â ï¸ IngresÃ¡ los tickers exactamente como aparecen en
                <strong>Yahoo Finance</strong> para asegurar la descarga de datos.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <h3 style="color:#B5E61D; margin-top:1.5rem; font-size:1.2rem;">
            ð Â¿CÃ³mo usar el optimizador?
        </h3>
        """, unsafe_allow_html=True)

        pasos = [
            ("1", "IngresÃ¡ los tickers", "En el panel izquierdo, escribÃ­ los sÃ­mbolos de los activos separados por coma (ej: <strong>AAPL, MSFT, JPM</strong>). UsÃ¡ el formato de Yahoo Finance."),
            ("2", "ElegÃ­ el perÃ­odo", "SeleccionÃ¡ cuÃ¡ntos aÃ±os de historia histÃ³rica querÃ©s analizar (1 a 15 aÃ±os). MÃ¡s aÃ±os = mayor robustez estadÃ­stica."),
            ("3", "ConfigurÃ¡ parÃ¡metros", "AjustÃ¡ la tasa libre de riesgo (referencia: tasa de la Fed o bono del Tesoro) y las restricciones de peso por activo."),
            ("4", "EjecutÃ¡ el anÃ¡lisis", "PresionÃ¡ <strong style='color:#B5E61D;'>ð EJECUTAR ANÃLISIS</strong>. Se calcularÃ¡n los portafolios Ã³ptimos por Sharpe, mÃ­nima volatilidad, equiponderado y âopcionalmenteâ retorno objetivo."),
            ("5", "ExplorÃ¡ los resultados", "NavegÃ¡ las pestaÃ±as: Espacio de Markowitz, ComposiciÃ³n, Rendimiento, MÃ©tricas, CorrelaciÃ³n y CAGR."),
        ]

        for num, titulo, desc in pasos:
            st.markdown(f"""
            <div class="step-box">
                <span class="step-num">{num}</span>
                <strong style="color:#17BEBB;">{titulo}</strong><br/>
                <span style="margin-left:38px; display:block; margin-top:4px;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.stop()

# âââââââââââââââââââââââââââââââââââââââââââââ
#  ANÃLISIS PRINCIPAL
# âââââââââââââââââââââââââââââââââââââââââââââ
if run_button:
    st.session_state.pop('pdf_ready', None)
    st.session_state.pop('pdf_bytes', None)

    # ââ Parse tickers (soporta formato TICKER:PESO para cartera personalizada) ââ
    raw_items = [item.strip() for item in tickers_input.split(',') if item.strip()]
    tickers = []
    custom_weights_input = {}   # {TICKER: peso_porcentaje}
    has_custom_weights   = False

    for item in raw_items:
        if ':' in item:
            parts  = item.split(':', 1)
            ticker = parts[0].strip().upper()
            try:
                weight_pct = float(parts[1].strip().rstrip('%'))
                custom_weights_input[ticker] = weight_pct
                has_custom_weights = True
            except ValueError:
                pass
            if ticker:
                tickers.append(ticker)
        else:
            t = item.strip().upper()
            if t:
                tickers.append(t)

    # Deduplicar conservando orden + limitar a 50
    seen = set()
    tickers_unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            tickers_unique.append(t)
    tickers = tickers_unique[:50]

    if not tickers:
        st.error("â IngresÃ¡ al menos 1 ticker.")
        st.stop()

    # ââ Descarga âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    with st.spinner("ð¡ Descargando datos de mercado desde Yahoo Finance..."):
        raw = yf.download(tickers, period=f"{anios}y", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw['Close']
        else:
            data = raw[['Close']] if 'Close' in raw.columns else raw
        data = data.dropna(axis=1, how='any')

    if data.shape[1] == 0:
        st.error("â Sin datos disponibles. RevisÃ¡ los tickers e intentÃ¡ de nuevo.")
        st.stop()

    assets      = list(data.columns)
    descartados = [t for t in tickers if t not in assets]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.success(f"â **{len(assets)} activos cargados:** {' Â· '.join(assets)}")
    with c2:
        if descartados:
            st.warning(f"â ï¸ Descartados: {', '.join(descartados)}")

    st.caption(
        f"ð PerÃ­odo: {data.index[0].strftime('%d/%m/%Y')} â "
        f"{data.index[-1].strftime('%d/%m/%Y')} Â· {len(data)} ruedas"
    )

    # ââ EstadÃ­sticas âââââââââââââââââââââââââââââââââââââââââââââââââââ
    returns      = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252
    corr_matrix  = returns.corr()
    num_assets   = len(assets)

    bounds           = tuple((min_peso, max_peso) for _ in range(num_assets))
    constraints_base = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    w0               = np.full(num_assets, 1.0 / num_assets)

    def _ps(w):
        return port_stats(w, mean_returns, cov_matrix, rf)

    def neg_sharpe(w): return -_ps(w)[2]
    def min_vol_fn(w): return  _ps(w)[1]

    # ââ Optimizaciones âââââââââââââââââââââââââââââââââââââââââââââââââ
    with st.spinner("ð¢ Ejecutando optimizaciones (Sharpe, Min-Vol, Equiponderado)..."):
        opt_s  = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints_base)
        w_sharpe = np.abs(opt_s.x) / np.abs(opt_s.x).sum()
        ret_sharpe, vol_sharpe, sharpe_val = _ps(w_sharpe)

        opt_v  = minimize(min_vol_fn, w0, method='SLSQP', bounds=bounds, constraints=constraints_base)
        w_vol  = np.abs(opt_v.x) / np.abs(opt_v.x).sum()
        ret_vol, vol_min, sharpe_minvol = _ps(w_vol)

        w_eq = np.full(num_assets, 1.0 / num_assets)
        ret_eq, vol_eq, sharpe_eq = _ps(w_eq)

        w_obj_arr = None
        ret_obj_real = vol_obj = sharpe_obj = np.nan
        obj_label = None

        if objetivo_activo:
            constraints_obj = list(constraints_base) + [
                {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - ret_obj}
            ]
            opt_o = minimize(min_vol_fn, w0, method='SLSQP',
                             bounds=bounds, constraints=constraints_obj)
            if opt_o.success:
                w_obj_arr = np.abs(opt_o.x) / np.abs(opt_o.x).sum()
                ret_obj_real, vol_obj, sharpe_obj = _ps(w_obj_arr)
                obj_label = f'Objetivo {pct(ret_obj)}'
            else:
                st.warning("â ï¸ No fue posible alcanzar el retorno objetivo con las restricciones dadas.")

    # ââ Frontera eficiente âââââââââââââââââââââââââââââââââââââââââââââ
    with st.spinner("ð Calculando frontera eficiente (100 puntos)..."):
        ret_range = np.linspace(ret_vol, max(mean_returns) * 1.05, 100)
        vol_fe    = []
        for target in ret_range:
            cons = list(constraints_base) + [
                {'type': 'eq', 'fun': lambda x, t=target: np.dot(x, mean_returns) - t}
            ]
            res = minimize(min_vol_fn, w0, method='SLSQP', bounds=bounds, constraints=cons)
            vol_fe.append(res.fun if res.success else np.nan)
        vol_fe = np.array(vol_fe)

    # ââ SimulaciÃ³n Monte Carlo âââââââââââââââââââââââââââââââââââââââââ
    with st.spinner("ð² Simulando 50 000 portafolios aleatorios..."):
        N_SIM = 50_000
        sim_ret_arr, sim_vol_arr, sim_sharpe_arr = [], [], []
        for _ in range(N_SIM):
            w = np.random.dirichlet(np.ones(num_assets))
            if min_peso > 0 and np.any(w < min_peso):
                continue
            r, v, s = _ps(w)
            sim_ret_arr.append(r)
            sim_vol_arr.append(v)
            sim_sharpe_arr.append(s)
        sim_ret_arr    = np.array(sim_ret_arr)
        sim_vol_arr    = np.array(sim_vol_arr)
        sim_sharpe_arr = np.array(sim_sharpe_arr)

    # ââ Construir diccionario de portfolios ââââââââââââââââââââââââââââ
    portfolios = {
        'MÃ¡x. Sharpe':     w_sharpe,
        'MÃ­n. Volatilidad':w_vol,
        'Equiponderado':   w_eq,
    }
    if w_obj_arr is not None:
        portfolios[obj_label] = w_obj_arr

    # ââ Cartera personalizada (si se ingresaron pesos) âââââââââââââââââ
    ret_custom = vol_custom = sharpe_custom = np.nan
    if has_custom_weights:
        avail = {t: custom_weights_input[t] for t in assets if t in custom_weights_input}
        if avail:
            total_pesos = sum(avail.values())
            if total_pesos > 0:
                w_custom = np.array([avail.get(a, 0.0) / total_pesos for a in assets])
                portfolios['Personalizada'] = w_custom
                ret_custom, vol_custom, sharpe_custom = _ps(w_custom)
                sin_peso = [a for a in assets if a not in avail]
                if sin_peso:
                    st.info(f"â¹ï¸ Activos sin peso especificado asignados a 0%: {', '.join(sin_peso)}")
            else:
                st.warning("â ï¸ Los pesos indicados suman 0. Se omite la cartera personalizada.")
        else:
            st.warning("â ï¸ NingÃºn ticker con peso coincide con los activos descargados. Se omite la cartera personalizada.")

    rf_daily   = rf / 252
    port_daily = pd.DataFrame({n: returns[assets].dot(w) for n, w in portfolios.items()})
    cum_port   = (1 + port_daily).cumprod() - 1
    cum_assets = (1 + returns[assets]).cumprod() - 1

    metricas = {}
    for name, w in portfolios.items():
        r, v, s = _ps(w)
        ser = port_daily[name]
        metricas[name] = {
            'Retorno Anual': r,
            'Volatilidad':   v,
            'Sharpe Ratio':  s,
            'Sortino Ratio': calc_sortino(ser, rf_daily),
            'MÃ¡x. Drawdown': calc_max_drawdown(ser),
            'CAGR':          calc_cagr(cum_port[name]),
        }

    # ââ Guardar resultados en session state para persistencia âââââââââ
    st.session_state['_an'] = dict(
        assets=assets, portfolios=portfolios, metricas=metricas,
        port_daily=port_daily, cum_port=cum_port, cum_assets=cum_assets,
        corr_matrix=corr_matrix, sim_vol_arr=sim_vol_arr,
        sim_ret_arr=sim_ret_arr, sim_sharpe_arr=sim_sharpe_arr,
        vol_fe=vol_fe, ret_range=ret_range,
        vol_min=vol_min, vol_sharpe=vol_sharpe, vol_eq=vol_eq,
        ret_eq=ret_eq, ret_sharpe=ret_sharpe, ret_vol=ret_vol,
        sharpe_val=sharpe_val, rf=rf, anios=anios,
        data_start=data.index[0].strftime('%d/%m/%Y'),
        data_end=data.index[-1].strftime('%d/%m/%Y'),
        w_obj_arr=w_obj_arr, vol_obj=vol_obj, ret_obj_real=ret_obj_real,
        ret_obj=ret_obj,
        obj_label=obj_label, has_custom_weights=has_custom_weights,
        vol_custom=vol_custom, ret_custom=ret_custom,
        num_assets=len(assets),
    )
    st.session_state['results_ready'] = True

# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  SECCIÃN DE RESULTADOS  (persiste entre reruns â fuera del if run_button)
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
if st.session_state.get('results_ready') and '_an' in st.session_state:
    _an            = st.session_state['_an']
    assets         = _an['assets']
    portfolios     = _an['portfolios']
    metricas       = _an['metricas']
    port_daily     = _an['port_daily']
    cum_port       = _an['cum_port']
    cum_assets     = _an['cum_assets']
    corr_matrix    = _an['corr_matrix']
    sim_vol_arr    = _an['sim_vol_arr']
    sim_ret_arr    = _an['sim_ret_arr']
    sim_sharpe_arr = _an['sim_sharpe_arr']
    vol_fe         = _an['vol_fe']
    ret_range      = _an['ret_range']
    vol_min        = _an['vol_min']
    vol_sharpe     = _an['vol_sharpe']
    vol_eq         = _an['vol_eq']
    ret_eq         = _an['ret_eq']
    ret_sharpe     = _an['ret_sharpe']
    ret_vol        = _an['ret_vol']
    sharpe_val     = _an['sharpe_val']
    rf             = _an['rf']
    anios          = _an['anios']
    data_start     = _an['data_start']
    data_end       = _an['data_end']
    w_obj_arr      = _an['w_obj_arr']
    vol_obj        = _an['vol_obj']
    ret_obj_real   = _an['ret_obj_real']
    ret_obj        = _an['ret_obj']
    obj_label      = _an['obj_label']
    has_custom_weights = _an['has_custom_weights']
    vol_custom     = _an['vol_custom']
    ret_custom     = _an['ret_custom']
    num_assets     = _an['num_assets']

    st.markdown("---")

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  SECCIÃN 1 â MÃTRICAS PRINCIPALES
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    st.markdown("## ð Resumen de Portfolios Ãptimos")

    for i, (name, m) in enumerate(metricas.items()):
        color = PORT_COLORS[i % len(PORT_COLORS)]
        st.markdown(
            f'<div class="port-card" style="border-left:4px solid {color}; background:#282828;">'
            f'<strong style="color:{color}; font-size:1.05rem;">ð {name}</strong></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(6)
        cols[0].metric("Retorno Anual",  pct(m['Retorno Anual']))
        cols[1].metric("Volatilidad",    pct(m['Volatilidad']))
        cols[2].metric("Sharpe Ratio",   f"{m['Sharpe Ratio']:.3f}")
        cols[3].metric("Sortino Ratio",  f"{m['Sortino Ratio']:.3f}")
        cols[4].metric("CAGR",           pct(m['CAGR']))
        cols[5].metric("MÃ¡x. Drawdown",  pct(m['MÃ¡x. Drawdown']))

    st.markdown("---")

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  SECCIÃN 2 â GRÃFICOS EN TABS
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    st.markdown("## ð AnÃ¡lisis GrÃ¡fico")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "ð Espacio Markowitz",
        "ð¥§ ComposiciÃ³n",
        "ð PortFolios",
        "ð Activos",
        "ð MÃ©tricas",
        "ð¥ CorrelaciÃ³n",
        "ð CAGR",
        "ð Educativo",
    ])

    # ââ Tab 1 â Espacio de Markowitz ââââââââââââââââââââââââââââââââââ
    with tab1:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð Â¿QuÃ© es el Espacio de Markowitz?</strong><br/>
            Este grÃ¡fico muestra el universo de todos los portafolios posibles formados con los activos seleccionados.
            Cada punto representa una combinaciÃ³n de pesos distinta. El color indica el <strong>Sharpe Ratio</strong>
            (relaciÃ³n retorno/riesgo). La <strong>Curva de Frontera Eficiente</strong> delimita los portafolios que
            maximizan el retorno para cada nivel de riesgo. La <strong>LÃ­nea de Mercado de Capitales (CML)</strong>
            parte de la tasa libre de riesgo y toca la frontera en el punto de mÃ¡ximo Sharpe.
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(13, 8))

        # Nube de portafolios simulados
        sc = ax.scatter(sim_vol_arr * 100, sim_ret_arr * 100,
                        c=sim_sharpe_arr, cmap='plasma',
                        alpha=0.30, s=5, zorder=1)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('Sharpe Ratio', color=BDI_CREAM, fontsize=12)
        cb.ax.yaxis.set_tick_params(color=BDI_CREAM, labelsize=10)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=BDI_CREAM)

        # Frontera eficiente â lÃ­nea gruesa y destacada
        valid = ~np.isnan(vol_fe)
        ax.plot(vol_fe[valid] * 100, ret_range[valid] * 100,
                color=BDI_TEAL, linewidth=4.5, zorder=6,
                label='Frontera Eficiente', solid_capstyle='round')

        # CML â solo en rango relevante (cerca de los datos)
        vol_cml_range = np.linspace(vol_min * 0.7, vol_sharpe * 1.3, 80)
        cml_line = rf * 100 + sharpe_val * vol_cml_range * 100
        ax.plot(vol_cml_range * 100, cml_line, '--', color=BDI_LIME, linewidth=2,
                alpha=0.9, zorder=4, label='CML (LÃ­nea de Mercado de Capitales)')

        # Marcadores de portfolios â mÃ¡s grandes y con anotaciones
        portfolios_plot = [
            (vol_sharpe, ret_sharpe, '*',  500, BDI_LIME,    f'MÃ¡x. Sharpe\n({sharpe_val:.2f})'),
            (vol_min,    ret_vol,    'D',  200, '#ef5350',   f'MÃ­n. Vol.\n({pct(vol_min)})'),
            (vol_eq,     ret_eq,     's',  200, BDI_TEAL,    f'Equip.\n({pct(ret_eq)})'),
        ]
        if w_obj_arr is not None:
            portfolios_plot.append(
                (vol_obj, ret_obj_real, 'P', 220, '#ffa726', f'Obj. {pct(ret_obj)}\n({pct(ret_obj_real)})')
            )
        if 'Personalizada' in portfolios and not np.isnan(vol_custom):
            portfolios_plot.append(
                (vol_custom, ret_custom, '^', 240, '#ab47bc', f'Personalizada\n({pct(ret_custom)})')
            )

        # Calcular offsets para anotaciones automÃ¡ticas
        x_range = (sim_vol_arr.max() - sim_vol_arr.min()) * 100
        y_range = (sim_ret_arr.max() - sim_ret_arr.min()) * 100

        for i, (vol_p, ret_p, marker, sz, color, label) in enumerate(portfolios_plot):
            ax.scatter(vol_p * 100, ret_p * 100,
                       marker=marker, s=sz, color=color, zorder=10,
                       edgecolors='white', linewidth=1.2)
            # AnotaciÃ³n con flecha
            x_off = x_range * 0.05 * (1 if i % 2 == 0 else -1)
            y_off = y_range * 0.06 * (1 if i < 2 else -1)
            ax.annotate(
                label,
                xy=(vol_p * 100, ret_p * 100),
                xytext=(vol_p * 100 + x_off, ret_p * 100 + y_off),
                fontsize=10, fontweight='bold', color=color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='#282828', ec=color, alpha=0.85, lw=1.2),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                zorder=11,
            )

        ax.axhline(rf * 100, color=BDI_MUTED, linewidth=0.8,
                   linestyle=':', alpha=0.6, label=f'Rf = {pct(rf)}')

        # Zoom al rango de datos relevante con padding
        x_pad = x_range * 0.15
        y_pad = y_range * 0.20
        ax.set_xlim(sim_vol_arr.min() * 100 - x_pad, sim_vol_arr.max() * 100 + x_pad)
        ax.set_ylim(sim_ret_arr.min() * 100 - y_pad, sim_ret_arr.max() * 100 + y_pad)

        ax.set_title('Espacio de Portfolios â Modelo de Markowitz',
                     fontsize=16, fontweight='bold', color=BDI_CREAM, pad=16)
        ax.set_xlabel('Volatilidad Anual (%)', fontsize=12, labelpad=10)
        ax.set_ylabel('Retorno Anual (%)', fontsize=12, labelpad=10)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=11, framealpha=0.9, loc='upper left',
                  fancybox=True, shadow=False)
        ax.grid(True, alpha=0.25)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ââ Tab 2 â ComposiciÃ³n (Donut charts) ââââââââââââââââââââââââââââ
    with tab2:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð¥§ ComposiciÃ³n de los portafolios Ã³ptimos</strong><br/>
            Cada grÃ¡fico de dona muestra la asignaciÃ³n de pesos recomendada para cada estrategia de optimizaciÃ³n.
            El nÃºmero central indica el <strong>retorno anual esperado</strong>. Debajo se muestran el
            <strong>Sharpe Ratio</strong> y la <strong>volatilidad</strong> del portafolio. La tabla al pie
            detalla los porcentajes exactos por activo.
        </div>
        """, unsafe_allow_html=True)

        n_ports = len(portfolios)
        fig, axes = plt.subplots(1, n_ports, figsize=(5.5 * n_ports, 7))
        if n_ports == 1:
            axes = [axes]

        for ax, (name, w) in zip(axes, portfolios.items()):
            mask   = w > 0.005
            labels = [assets[i] for i in range(num_assets) if mask[i]]
            sizes  = w[mask]
            other  = 1 - sizes.sum()
            if other > 0.001:
                labels.append('Otros')
                sizes = np.append(sizes, other)
            colors = PALETTE[:len(labels)]

            wedges, _, autotexts = ax.pie(
                sizes, labels=None, colors=colors,
                autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
                startangle=90, pctdistance=0.75,
                wedgeprops={'linewidth': 2, 'edgecolor': BDI_DARK_BG, 'width': 0.62},
            )
            for at in autotexts:
                at.set_fontsize(12)      # agrandado de 8 â 12
                at.set_color('white')
                at.set_fontweight('bold')

            m = metricas[name]
            ax.text(0,  0.12, pct(m['Retorno Anual']),
                    ha='center', va='center', fontsize=17,
                    fontweight='bold', color=BDI_LIME)
            ax.text(0, -0.13, f"Sharpe: {m['Sharpe Ratio']:.2f}",
                    ha='center', va='center', fontsize=10, color=BDI_CREAM)
            ax.text(0, -0.32, f"Vol: {pct(m['Volatilidad'])}",
                    ha='center', va='center', fontsize=10, color=BDI_TEAL)
            ax.legend(wedges, [f"{l} ({s*100:.1f}%)" for l, s in zip(labels, sizes)],
                      loc='lower center', bbox_to_anchor=(0.5, -0.30),
                      fontsize=10, framealpha=0.75, ncol=2)  # agrandado de 8 â 10
            ax.set_title(name, fontsize=12, fontweight='bold', color=BDI_CREAM, pad=14)

        fig.suptitle('ComposiciÃ³n de Portfolios Ãptimos',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, y=1.02)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### ð Tabla de pesos por activo")
        df_pesos = pd.DataFrame(
            {name: [f"{w[i]*100:.1f}%" for i in range(num_assets)]
             for name, w in portfolios.items()},
            index=assets,
        )
        st.dataframe(df_pesos, use_container_width=True)

    # ââ Tab 3 â Rendimiento acumulado: portfolios ââââââââââââââââââââââ
    with tab3:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð Rendimiento acumulado de los portafolios</strong><br/>
            Este grÃ¡fico muestra cÃ³mo habrÃ­a evolucionado una inversiÃ³n inicial en cada portafolio a lo largo
            del perÃ­odo histÃ³rico analizado. Permite comparar visualmente la performance relativa de cada
            estrategia de optimizaciÃ³n en distintos contextos de mercado (subas, bajas, lateralizaciones).
            El valor final de cada lÃ­nea se indica al extremo derecho del grÃ¡fico.
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(13, 6))
        for i, col in enumerate(cum_port.columns):
            ax.plot(cum_port.index, cum_port[col] * 100,
                    label=col, linewidth=2.5,
                    color=PORT_COLORS[i % len(PORT_COLORS)])
        ax.axhline(0, color=BDI_MUTED, linewidth=0.6, linestyle='--', alpha=0.5)
        ax.fill_between(cum_port.index, 0, cum_port.iloc[:, 0] * 100,
                        alpha=0.06, color=BDI_GREEN)
        for i, (col, val) in enumerate((cum_port.iloc[-1] * 100).items()):
            ax.annotate(f' {val:.1f}%', xy=(cum_port.index[-1], val),
                        fontsize=9, color=PORT_COLORS[i % len(PORT_COLORS)],
                        va='center', fontweight='bold')
        ax.set_title('Rendimiento Acumulado â PortFolios Ãptimos',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, pad=12)
        ax.set_xlabel('Fecha', fontsize=11, labelpad=8)
        ax.set_ylabel('Rendimiento Acumulado (%)', fontsize=11, labelpad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=10, framealpha=0.85)
        ax.grid(True, alpha=0.25)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ââ Tab 4 â Rendimiento acumulado: activos individuales âââââââââââ
    with tab4:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð Rendimiento acumulado por activo individual</strong><br/>
            VisualizÃ¡ cÃ³mo se desempeÃ±Ã³ cada activo de forma independiente durante el perÃ­odo analizado.
            Esto permite identificar quÃ© instrumentos lideraron el crecimiento, cuÃ¡les tuvieron mayor
            volatilidad y cÃ³mo interactuaron entre sÃ­. La diversificaciÃ³n busca combinarlos para suavizar
            las caÃ­das sin sacrificar retorno.
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(13, 6))
        for i, col in enumerate(cum_assets.columns):
            ax.plot(cum_assets.index, cum_assets[col] * 100,
                    label=col, linewidth=1.8,
                    color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax.axhline(0, color=BDI_MUTED, linewidth=0.6, linestyle='--', alpha=0.5)
        final_a = cum_assets.iloc[-1] * 100
        for i, (col, val) in enumerate(final_a.sort_values(ascending=False).items()):
            ax.annotate(
                f' {col}: {val:.0f}%',
                xy=(cum_assets.index[-1], val),
                fontsize=8,
                color=PALETTE[list(cum_assets.columns).index(col) % len(PALETTE)],
                va='center',
            )
        ax.set_title('Rendimiento Acumulado â Activos Individuales',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, pad=12)
        ax.set_xlabel('Fecha', fontsize=11, labelpad=8)
        ax.set_ylabel('Rendimiento Acumulado (%)', fontsize=11, labelpad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=9, framealpha=0.8,
                  bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(True, alpha=0.25)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ââ Tab 5 â MÃ©tricas comparativas (barras) ââââââââââââââââââââââââ
    with tab5:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð ComparaciÃ³n de mÃ©tricas entre portafolios</strong><br/>
            Los cuatro paneles comparan las mÃ©tricas clave de cada estrategia: <strong>Retorno Anual</strong>
            (cuÃ¡nto creciÃ³ en promedio), <strong>Volatilidad</strong> (nivel de riesgo o fluctuaciÃ³n),
            <strong>Sharpe Ratio</strong> (retorno ajustado por riesgo; mayor es mejor) y
            <strong>MÃ¡ximo Drawdown</strong> (peor caÃ­da desde un pico; menos negativo es mejor).
            El borde dorado resalta el mejor portafolio en cada categorÃ­a.
        </div>
        """, unsafe_allow_html=True)

        port_names    = list(portfolios.keys())
        metricas_plot = [
            ('Retorno Anual',  'Retorno Anual (%)',     True),
            ('Volatilidad',    'Volatilidad Anual (%)', False),
            ('Sharpe Ratio',   'Sharpe Ratio',          True),
            ('MÃ¡x. Drawdown',  'MÃ¡ximo Drawdown (%)',   False),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, (col, ylabel, higher_better) in zip(axes.flatten(), metricas_plot):
            vals  = [metricas[p][col] for p in port_names]
            scale = 100 if '%' in ylabel else 1
            bars  = ax.bar(port_names, [v * scale for v in vals],
                           color=PORT_COLORS[:len(port_names)],
                           edgecolor=BDI_DARK_BG, linewidth=1.2, width=0.5)
            best_idx = np.argmax(vals) if higher_better else np.argmin(vals)
            bars[best_idx].set_edgecolor(BDI_LIME)
            bars[best_idx].set_linewidth(2.8)
            for bar, val in zip(bars, vals):
                lbl = f'{val*100:.1f}%' if '%' in ylabel else f'{val:.2f}'
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.03,
                    lbl, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=BDI_CREAM,
                )
            ax.set_title(ylabel, fontsize=12, fontweight='bold', color=BDI_CREAM)
            if '%' in ylabel:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
            ax.set_xticklabels(port_names, rotation=15, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.25)

        fig.suptitle('ComparaciÃ³n de MÃ©tricas â Portfolios Ãptimos',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, y=1.01)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### ð Tabla de mÃ©tricas")
        df_met = pd.DataFrame(metricas).T
        fmt_pct   = ['Retorno Anual', 'Volatilidad', 'MÃ¡x. Drawdown', 'CAGR']
        fmt_ratio = ['Sharpe Ratio', 'Sortino Ratio']
        df_disp   = df_met.copy()
        for c in fmt_pct:
            df_disp[c] = df_disp[c].apply(pct)
        for c in fmt_ratio:
            df_disp[c] = df_disp[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_disp, use_container_width=True)

    # ââ Tab 6 â Matriz de correlaciÃ³n âââââââââââââââââââââââââââââââââ
    with tab6:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð¥ Matriz de correlaciÃ³n entre activos</strong><br/>
            Muestra el grado de movimiento conjunto entre pares de activos. Un valor cercano a
            <strong>+1</strong> (rojo oscuro) indica que los activos suben y bajan juntos, lo que reduce
            el beneficio de diversificaciÃ³n. Un valor cercano a <strong>0</strong> (rojo claro) indica
            baja correlaciÃ³n, lo que es ideal para reducir el riesgo total del portafolio.
            Buscar activos con baja correlaciÃ³n entre sÃ­ es la clave del modelo de Markowitz.
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(max(8, num_assets + 2), max(6, num_assets + 1)))

        # Mapa de color rojo: claro = baja correlaciÃ³n, oscuro = alta correlaciÃ³n
        cmap_red = LinearSegmentedColormap.from_list(
            'BDI_RED',
            ['#ffebee', '#ffcdd2', '#ef9a9a', '#e57373',
             '#ef5350', '#e53935', '#c62828', '#b71c1c', '#7f0000'],
            N=256
        )

        sns.heatmap(
            corr_matrix, ax=ax,
            annot=True, fmt='.2f',
            annot_kws={'size': 10, 'weight': 'bold'},
            cmap=cmap_red, vmin=0, vmax=1,
            linewidths=0.8, linecolor=BDI_DARK_BG, square=True,
            cbar_kws={'shrink': 0.8, 'label': 'CorrelaciÃ³n', 'pad': 0.02},
        )

        # Texto adaptivo: oscuro sobre celdas claras, claro sobre celdas oscuras
        for text_obj in ax.texts:
            try:
                val = float(text_obj.get_text())
                text_obj.set_color('#1c1c1c' if val < 0.55 else '#EFEDEA')
            except Exception:
                pass

        ax.set_title('Matriz de CorrelaciÃ³n entre Activos',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, pad=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=10, color=BDI_CREAM)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=10, color=BDI_CREAM)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ââ Tab 7 â CAGR comparativo ââââââââââââââââââââââââââââââââââââââ
    with tab7:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð CAGR â Tasa de Crecimiento Anual Compuesta</strong><br/>
            El CAGR representa la tasa a la que una inversiÃ³n habrÃ­a crecido aÃ±o a aÃ±o de forma constante
            para llegar al valor final observado. A diferencia del retorno simple, el CAGR toma en cuenta
            el efecto del interÃ©s compuesto. Es la mÃ©trica mÃ¡s adecuada para comparar el crecimiento
            real de distintos activos y portafolios en el largo plazo.
        </div>
        """, unsafe_allow_html=True)

        cagr_activos = {a: calc_cagr(cum_assets[a]) * 100 for a in assets}
        cagr_ports   = {p: calc_cagr(cum_port[p])   * 100 for p in port_daily.columns}
        all_names    = list(cagr_activos) + list(cagr_ports)
        all_vals     = list(cagr_activos.values()) + list(cagr_ports.values())
        def _port_type(name):
            if name == 'Personalizada':
                return 'Personalizada'
            return 'PortFolio'
        all_types    = ['Activo'] * len(cagr_activos) + [_port_type(p) for p in cagr_ports]
        type_color   = {'Activo': BDI_TEAL, 'PortFolio': BDI_LIME, 'Personalizada': '#ab47bc'}

        sorted_data  = sorted(zip(all_vals, all_names, all_types), reverse=True)
        s_vals, s_names, s_types = zip(*sorted_data)

        fig, ax = plt.subplots(figsize=(max(10, len(all_names) * 1.2), 7))
        bars = ax.bar(s_names, s_vals,
                      color=[type_color.get(t, BDI_MUTED) for t in s_types],
                      edgecolor=BDI_DARK_BG, linewidth=1.2, width=0.6)
        for bar, val in zip(bars, s_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.5 if val >= 0 else -1.5),
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=BDI_CREAM)
        ax.axhline(0, color=BDI_MUTED, linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_title('CAGR Anual Comparativo â Activos y PortFolios',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, pad=12)
        ax.set_ylabel('CAGR Anual (%)', fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.set_xticklabels(s_names, rotation=35, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.25)
        legend_elems = [mpatches.Patch(color=c, label=t) for t, c in type_color.items()]
        ax.legend(handles=legend_elems, fontsize=10, framealpha=0.85)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ââ Tab 8 â MÃ³dulo Educativo ââââââââââââââââââââââââââââââââââââââ
    with tab8:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð MÃ³dulo Educativo â Fundamentos del AnÃ¡lisis de Cartera</strong><br/>
            Esta secciÃ³n explica las fÃ³rmulas matemÃ¡ticas que utiliza el optimizador y cÃ³mo interpretar cada resultado.
            PodÃ©s usarla como referencia para entender quÃ© significa cada nÃºmero y cÃ³mo tomar mejores decisiones.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ð¬ 1. Modelo de Markowitz â TeorÃ­a Moderna de Portfolio", expanded=True):
            st.markdown("""
            **Â¿QuÃ© es?** La TeorÃ­a Moderna de Portfolio (Harry Markowitz, 1952) sostiene que el riesgo y el retorno
            de una cartera dependen no solo de los activos individuales, sino de la **correlaciÃ³n entre ellos**.
            La clave es que combinando activos poco correlacionados se puede reducir el riesgo **sin sacrificar retorno**.

            **Retorno esperado del portfolio:**
            """)
            st.latex(r"E(R_p) = \sum_{i=1}^{n} w_i \cdot E(R_i)")
            st.markdown("Donde $w_i$ es el peso del activo $i$ y $E(R_i)$ su retorno anualizado histÃ³rico.")

            st.markdown("**Varianza (riesgo cuadrÃ¡tico) del portfolio:**")
            st.latex(r"\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w} = \sum_{i=1}^{n}\sum_{j=1}^{n} w_i \cdot w_j \cdot \sigma_{ij}")
            st.markdown("""
            Donde $\Sigma$ es la **matriz de covarianza** y $\sigma_{ij}$ la covarianza entre los activos $i$ y $j$.

            **Volatilidad (riesgo) del portfolio:**
            """)
            st.latex(r"\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}")
            st.markdown("""
            ð¡ **Beneficio de la diversificaciÃ³n:** Si dos activos no estÃ¡n perfectamente correlacionados
            ($\\rho_{ij} < 1$), la volatilidad del portfolio es **menor** que el promedio ponderado de las
            volatilidades individuales. Eso es el "free lunch" de invertir: reducir riesgo sin sacrificar retorno.
            """)

        with st.expander("ð 2. Frontera Eficiente y Problema de OptimizaciÃ³n"):
            st.markdown("""
            **Frontera Eficiente:** Conjunto de portfolios que maximizan el retorno esperado
            para cada nivel de riesgo. Todo portfolio *debajo* o *a la derecha* de la frontera es subÃ³ptimo
            â existe un portfolio mejor con el mismo riesgo o menor riesgo con igual retorno.

            **Problema de optimizaciÃ³n (mÃ­nima volatilidad para retorno objetivo $R^*$):**
            """)
            st.latex(r"\min_{\mathbf{w}} \quad \sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}")
            st.latex(r"\text{sujeto a:} \quad \sum_{i=1}^{n} w_i = 1 \quad \text{(pesos suman 1)}")
            st.latex(r"\quad\quad\quad\quad \mathbf{w}^T \boldsymbol{\mu} = R^* \quad \text{(retorno objetivo)}")
            st.latex(r"\quad\quad\quad\quad w_i \geq 0 \quad \text{(sin posiciones cortas)}")
            st.markdown("""
            El optimizador usa **SLSQP** (Sequential Least Squares Programming), resolviendo este problema
            para 100 valores distintos de $R^*$ entre el mÃ­nimo y el mÃ¡ximo retorno posible.
            Cada soluciÃ³n es un punto de la frontera eficiente.

            ð¡ **QuÃ© muestra el grÃ¡fico:** La nube de puntos son 50,000 portfolios aleatorios (Monte Carlo).
            La curva de color turquesa es la frontera eficiente. Cualquier punto a la izquierda o arriba de ella
            es **inalcanzable** con los activos disponibles.
            """)

        with st.expander("â­ 3. Sharpe Ratio â Retorno ajustado por riesgo"):
            st.markdown("""
            Desarrollado por William Sharpe (1966), mide cuÃ¡nto retorno extra obtenemos por cada unidad de
            riesgo asumido, en relaciÃ³n a una inversiÃ³n libre de riesgo (ej: bono del Tesoro).
            """)
            st.latex(r"S = \frac{E(R_p) - R_f}{\sigma_p}")
            st.markdown("""
            Donde $R_f$ es la tasa libre de riesgo (configurada en el panel lateral).

            **InterpretaciÃ³n prÃ¡ctica:**

            | Sharpe | EvaluaciÃ³n |
            |--------|-----------|
            | < 0    | Peor que invertir en el activo libre de riesgo |
            | 0 â 0.5 | Aceptable |
            | 0.5 â 1 | Bueno |
            | > 1    | Excelente |
            | > 2    | Muy difÃ­cil de sostener en el tiempo |

            ð¡ El **Portfolio de MÃ¡ximo Sharpe** (estrella â en el grÃ¡fico) es el punto donde la
            **LÃ­nea de Mercado de Capitales (CML)** es tangente a la frontera eficiente.
            Es el portafolio "racionalmente Ã³ptimo" segÃºn la teorÃ­a.
            """)

        with st.expander("ð 4. Sortino Ratio â Penaliza solo la volatilidad negativa"):
            st.markdown("""
            Variante del Sharpe que distingue entre volatilidad "buena" (hacia arriba) y "mala" (hacia abajo),
            usando solo el **downside risk** en el denominador.
            """)
            st.latex(r"Sortino = \frac{E(R_p) - R_f}{\sigma_{down}}")
            st.latex(r"\sigma_{down} = \sqrt{\frac{\sum_{t:\, R_t < R_f}(R_t - R_f)^2}{T}} \times \sqrt{252}")
            st.markdown("""
            ð¡ **CuÃ¡ndo es mÃ¡s Ãºtil que el Sharpe:** Si los retornos del portfolio tienen **asimetrÃ­a positiva**
            (subas grandes, bajas pequeÃ±as), el Sortino serÃ¡ mayor que el Sharpe. Un Sortino > Sharpe indica
            que la volatilidad total estÃ¡ sesgada hacia el lado positivo â buena seÃ±al.
            """)

        with st.expander("ð 5. CAGR â Tasa de Crecimiento Anual Compuesta"):
            st.markdown("""
            El CAGR representa la tasa constante anual que llevarÃ­a una inversiÃ³n de su valor inicial
            al valor final observado, incorporando el efecto del **interÃ©s compuesto**.
            """)
            st.latex(r"CAGR = \left(\frac{V_f}{V_i}\right)^{\frac{1}{n}} - 1")
            st.markdown("""
            Donde $n$ es el nÃºmero de aÃ±os del perÃ­odo analizado.

            **Â¿Por quÃ© el CAGR es mejor que el retorno promedio?**

            Ejemplo: Un activo sube +100% el aÃ±o 1 y cae -50% el aÃ±o 2.
            """)
            st.latex(r"\text{Retorno promedio} = \frac{+100\% + (-50\%)}{2} = +25\% \quad \text{(engaÃ±oso)}")
            st.latex(r"CAGR = \sqrt{2 \times 0.5} - 1 = 0\% \quad \text{(refleja la realidad)}")
            st.markdown("ð¡ El CAGR siempre es â¤ al promedio aritmÃ©tico. La diferencia entre ambos crece con la volatilidad.")

        with st.expander("ð 6. MÃ¡ximo Drawdown â Peor caÃ­da desde un mÃ¡ximo histÃ³rico"):
            st.markdown("""
            Mide la mayor pÃ©rdida porcentual sufrida entre un pico y el valle subsiguiente
            en el perÃ­odo analizado. Es la mÃ©trica del **peor escenario**.
            """)
            st.latex(r"MDD = \min_{t} \frac{V_t - \max_{\tau \leq t} V_\tau}{\max_{\tau \leq t} V_\tau}")
            st.markdown("""
            ð¡ **CÃ³mo usarlo:** Un MDD de -35% significa que en algÃºn momento del perÃ­odo, el inversor
            habrÃ­a visto su cartera caer un 35% desde su mÃ¡ximo anterior. Es la pregunta clave:
            *Â¿PodrÃ­a aguantar esa caÃ­da sin vender?* Si la respuesta es no, el portfolio tiene
            demasiado riesgo para tu perfil.
            """)

        with st.expander("ð² 7. SimulaciÃ³n Monte Carlo"):
            st.markdown("""
            Para construir la nube de puntos del espacio de Markowitz, se generan **50,000 portfolios aleatorios**
            usando la distribuciÃ³n de Dirichlet (que garantiza pesos positivos que suman 1):
            """)
            st.latex(r"\mathbf{w} \sim \mathrm{Dir}(\mathbf{1}) \implies w_i \geq 0,\; \sum_i w_i = 1")
            st.markdown("""
            Para cada portfolio aleatorio se calculan retorno, volatilidad y Sharpe:
            """)
            st.latex(r"(E(R_p),\; \sigma_p,\; S_p) \quad \forall\; \mathbf{w} \text{ simulado}")
            st.markdown("""
            El color de cada punto en el grÃ¡fico indica el **Sharpe Ratio** (mÃ¡s amarillo = mejor).
            La frontera eficiente aparece como el "borde superior izquierdo" de esta nube â
            es el lÃ­mite de lo que es alcanzable con los activos disponibles.
            """)

        with st.expander("ðºï¸ 8. CÃ³mo elegir el portfolio segÃºn tu perfil"):
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.markdown("""
                <div class="info-box">
                    <strong style="color:#ef5350;">ð´ Perfil Conservador</strong><br/><br/>
                    ElegÃ­ el <strong>Portfolio de MÃ­nima Volatilidad</strong>.<br/><br/>
                    Menos riesgo de caÃ­das Â· Menor Drawdown histÃ³rico Â· Sacrifica algo de retorno Â·
                    Ideal para horizontes cortos o alta aversiÃ³n al riesgo.
                </div>
                """, unsafe_allow_html=True)
            with col_e2:
                st.markdown("""
                <div class="info-box">
                    <strong style="color:#B5E61D;">ð¡ Perfil Moderado</strong><br/><br/>
                    ElegÃ­ el <strong>Portfolio de MÃ¡ximo Sharpe</strong>.<br/><br/>
                    Mejor relaciÃ³n retorno/riesgo Â· Balance entre crecimiento y estabilidad Â·
                    El "Ã³ptimo racional" de Markowitz Â· Ideal para horizontes de 3â5 aÃ±os.
                </div>
                """, unsafe_allow_html=True)
            with col_e3:
                st.markdown("""
                <div class="info-box">
                    <strong style="color:#17BEBB;">ðµ Perfil Agresivo</strong><br/><br/>
                    UsÃ¡ <strong>Retorno Objetivo o Cartera Personalizada</strong>.<br/><br/>
                    Mayor exposiciÃ³n al crecimiento Â· Mayor volatilidad esperada Â·
                    Mayor potencial de retorno Â· Ideal para horizontes mayores a 7 aÃ±os.
                </div>
                """, unsafe_allow_html=True)
            st.markdown("""
            <div class="legal-warning" style="margin-top:1rem;">
                â ï¸ <strong>LimitaciÃ³n del modelo:</strong> Markowitz se basa en retornos histÃ³ricos y
                asume que las correlaciones son estables. En perÃ­odos de crisis, las correlaciones
                aumentan (los activos "caen juntos"), reduciendo el beneficio de la diversificaciÃ³n
                exactamente cuando mÃ¡s se necesita. UsÃ¡ estos resultados como guÃ­a, no como verdad absoluta.
            </div>
            """, unsafe_allow_html=True)

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  REPORTE PDF DESCARGABLE
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    st.markdown("---")
    st.markdown("## ð¥ Reporte PDF Descargable")

    pdf_col_l, pdf_col_r = st.columns([1.4, 1])

    with pdf_col_l:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">ð Informe ejecutivo con marca BDI</strong><br/>
            GenerÃ¡ un PDF profesional con todos los anÃ¡lisis, listo para presentar a clientes
            o guardar como respaldo del estudio.<br/><br/>
            <strong style="color:#17BEBB;">El reporte incluye (8 pÃ¡ginas):</strong><br/>
            &nbsp;Â·&nbsp; Portada BDI con resumen de mÃ©tricas<br/>
            &nbsp;Â·&nbsp; Espacio de Markowitz con Frontera Eficiente y CML<br/>
            &nbsp;Â·&nbsp; ComposiciÃ³n de portfolios (grÃ¡ficos de dona)<br/>
            &nbsp;Â·&nbsp; Rendimiento acumulado â portfolios y activos<br/>
            &nbsp;Â·&nbsp; ComparaciÃ³n de mÃ©tricas (4 paneles)<br/>
            &nbsp;Â·&nbsp; Matriz de correlaciÃ³n entre activos<br/>
            &nbsp;Â·&nbsp; CAGR comparativo anual<br/>
            &nbsp;Â·&nbsp; Aviso legal
        </div>
        """, unsafe_allow_html=True)

    with pdf_col_r:
        st.markdown("<br/>", unsafe_allow_html=True)
        cliente_nombre = st.text_input(
            "ð¤ Nombre del cliente (opcional)",
            value="",
            key="cliente_pdf",
            placeholder="Ej: Juan PÃ©rez â Perfil Moderado",
        )

        if st.button("ð Generar Reporte PDF", use_container_width=True, key="btn_gen_pdf"):
            with st.spinner("ð Generando reporte profesional BDI..."):
                pdf_bytes = generate_pdf_report(
                    assets=assets, portfolios=portfolios, metricas=metricas,
                    port_daily=port_daily, cum_port=cum_port, cum_assets=cum_assets,
                    corr_matrix=corr_matrix, sim_vol_arr=sim_vol_arr,
                    sim_ret_arr=sim_ret_arr, sim_sharpe_arr=sim_sharpe_arr,
                    vol_fe=vol_fe, ret_range=ret_range,
                    vol_min=vol_min, vol_sharpe=vol_sharpe, vol_eq=vol_eq,
                    ret_eq=ret_eq, ret_sharpe=ret_sharpe, ret_vol=ret_vol,
                    sharpe_val=sharpe_val, rf=rf, anios=anios,
                    data_start=data_start,
                    data_end=data_end,
                    w_obj_arr=w_obj_arr, vol_obj=vol_obj, ret_obj_real=ret_obj_real,
                    obj_label=obj_label, has_custom_weights=has_custom_weights,
                    vol_custom=vol_custom, ret_custom=ret_custom,
                    cliente_nombre=cliente_nombre,
                )
                st.session_state['pdf_bytes'] = pdf_bytes
                st.session_state['pdf_ready'] = True

        if st.session_state.get('pdf_ready'):
            fname = f"BDI_Cartera_{datetime.now().strftime('%Y%m%d')}.pdf"
            st.download_button(
                label="â¬ï¸ Descargar Reporte PDF",
                data=st.session_state['pdf_bytes'],
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )
            st.success(f"â PDF listo Â· {fname}")

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  ADVERTENCIA LEGAL
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    st.markdown("""
    <div class="legal-warning">
        â ï¸ <strong>ADVERTENCIA LEGAL:</strong>
        Este anÃ¡lisis es de carÃ¡cter exclusivamente informativo y educativo.
        No constituye asesoramiento financiero ni una recomendaciÃ³n de inversiÃ³n.
        Los rendimientos pasados no garantizan resultados futuros.<br/>
        <em>BDI Consultora de Inversiones Â· bdiconsultora@gmail.com Â· Mariano Ricciardi</em>
    </div>
    """, unsafe_allow_html=True)

    # (results_ready ya fue guardado en el bloque de cÃ³mputo)
