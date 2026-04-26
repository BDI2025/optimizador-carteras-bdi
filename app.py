# ============================================================
#  OPTIMIZADOR DE CARTERA PROFESIONAL — STREAMLIT
#  BDI Consultora de Inversiones
#  Versión 2.0 — Modelo de Markowitz con Frontera Eficiente
# ============================================================
import warnings
warnings.filterwarnings('ignore')
import io
import textwrap

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

# ─────────────────────────────────────────────
#  PAGE CONFIG  (debe ir primero)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BDI — Optimizador de Carteras",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  PALETA CORPORATIVA BDI (oficial)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #1c1c1c; color: #EFEDEA; }

    /* Sidebar — fondo claro BDI */
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
    /* Números del slider en sidebar */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"] {
        color: #137247 !important;
    }

    /* Títulos globales */
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

    /* Botón principal */
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
        border-radius: 0 8px 8px 0;
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

# ─────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  GENERADOR DE REPORTE PDF
# ─────────────────────────────────────────────
def generate_pdf_report(assets, portfolios, metricas, port_daily, cum_port, cum_assets,
                        corr_matrix, sim_vol_arr, sim_ret_arr, sim_sharpe_arr,
                        vol_fe, ret_range, vol_min, vol_sharpe, vol_eq, ret_eq,
                        ret_sharpe, ret_vol, sharpe_val, rf, anios, data_start, data_end,
                        w_obj_arr, vol_obj, ret_obj_real, obj_label,
                        has_custom_weights, vol_custom, ret_custom, cliente_nombre=""):
    """Informe PDF — Opcion B Moderno Premium — fondo blanco, BDI branding."""
    num_assets = len(assets)
    buf        = io.BytesIO()
    fecha_hoy  = datetime.now().strftime('%d/%m/%Y')
    AW, AH     = 11.69, 8.27   # landscape A4

    # ── Paleta para fondo blanco ──────────────────────────────────────
    WHITE  = '#FFFFFF'
    DARK   = '#1A1A1A'
    MUTED  = '#666666'
    LIGHT  = '#F5F5F5'
    BORDER = '#DEDEDE'

    # Colores de portafolios sobre fondo blanco (lima → oliva oscuro)
    PCW = [BDI_GREEN, BDI_TEAL, '#6e9900', '#e88c00', '#8e44ad']

    def _footer(fig):
        fig.text(0.5, 0.013, 'BDI Consultora de Inversiones  ·  hola@bdiconsultora.com',
                 ha='center', va='bottom', fontsize=7,
                 color=BDI_GREEN, alpha=0.80, style='italic')
        fig.text(0.02, 0.013, fecha_hoy,
                 ha='left', va='bottom', fontsize=7, color=MUTED, alpha=0.7)

    def _ax_off(ax, bg=WHITE):
        ax.set_facecolor(bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')

    def _left_border(ax, color, width=0.012):
        ax.add_patch(mpatches.Rectangle(
            (0, 0), width, 1, transform=ax.transAxes,
            facecolor=color, edgecolor='none', zorder=3))

    def _card_bg(ax, bg='#F9FAFB'):
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.5)

    def _section_strip(fig, bottom, height, title, subtitle='', accent=BDI_GREEN):
        ax_s = fig.add_axes([0, bottom, 1, height])
        _ax_off(ax_s, LIGHT)
        ax_s.add_patch(mpatches.Rectangle(
            (0, 0), 0.005, 1, transform=ax_s.transAxes,
            facecolor=accent, edgecolor='none'))
        ax_s.text(0.025, 0.70, title, fontsize=13, fontweight='bold',
                  color=accent, va='top', transform=ax_s.transAxes)
        if subtitle:
            ax_s.text(0.025, 0.25, subtitle, fontsize=8.5,
                      color=MUTED, va='top', transform=ax_s.transAxes)

    with PdfPages(buf) as pdf:

        # ══════════════════════════════════════════════════════════════
        # PAGINA 1 — PORTADA (Portrait A4)
        # ══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(WHITE)

        # Header gradient verde → teal — más alto, incluye el título
        ax_h = fig.add_axes([0, 0.72, 1, 0.28])
        ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1)
        cmap_gt = LinearSegmentedColormap.from_list('gt', [BDI_GREEN, BDI_TEAL])
        ax_h.imshow(np.linspace(0, 1, 256).reshape(1, -1),
                    aspect='auto', cmap=cmap_gt, extent=[0, 1, 0, 1], zorder=0)
        ax_h.axis('off')
        # Logo BDI (parte superior izquierda del header)
        ax_h.text(0.06, 0.97, 'BDI', fontsize=34, fontweight='bold',
                  color=WHITE, va='top')
        ax_h.text(0.06, 0.76, 'CONSULTORA DE INVERSIONES', fontsize=8.5,
                  color=BDI_LIME, va='top', fontweight='bold')
        ax_h.text(0.06, 0.64, 'hola@bdiconsultora.com', fontsize=7.5,
                  color=WHITE, va='top', alpha=0.90)
        # Separador horizontal
        ax_h.axhline(0.54, color=WHITE, linewidth=0.6, alpha=0.35, xmin=0.05, xmax=0.95)
        # Título dentro del header (fondo verde)
        ax_h.text(0.06, 0.48, 'Informe de Optimizacion de Portafolio de Inversiones',
                  fontsize=18, fontweight='bold', color=WHITE, va='top')
        ax_h.text(0.06, 0.23, 'Modelo de Markowitz con Frontera Eficiente',
                  fontsize=10, color=BDI_LIME, va='top', alpha=0.95, fontstyle='italic')

        # Info box — ajustado al nuevo header
        ax_i = fig.add_axes([0.06, 0.37, 0.88, 0.33])
        _card_bg(ax_i, LIGHT)
        _ax_off(ax_i, LIGHT)
        _left_border(ax_i, BDI_GREEN, 0.009)
        info_rows = []
        if cliente_nombre:
            info_rows.append(('Cliente:', cliente_nombre))
        info_rows += [
            ('Fecha:', fecha_hoy),
            ('Periodo:', f'{anios} anos  ({data_start}  a  {data_end})'),
            ('Activos:', textwrap.fill(', '.join(assets), width=60)),
            ('Portafolios:', textwrap.fill(', '.join(portfolios.keys()), width=60)),
        ]
        y0 = 0.88
        for lbl, val in info_rows:
            ax_i.text(0.025, y0, lbl, fontsize=9, fontweight='bold',
                      color=BDI_GREEN, va='top', transform=ax_i.transAxes)
            lines_val = val.split('\n')
            for k, line in enumerate(lines_val):
                ax_i.text(0.22, y0 - k * 0.14, line, fontsize=9, color=DARK,
                          va='top', transform=ax_i.transAxes)
            y0 -= 0.17 + max(0, len(lines_val) - 1) * 0.14

        # Tabla resumen
        ax_tb = fig.add_axes([0.06, 0.10, 0.88, 0.26])
        _ax_off(ax_tb)
        ax_tb.text(0, 0.97, 'Resumen de resultados', fontsize=10.5,
                   fontweight='bold', color=BDI_GREEN, va='top', transform=ax_tb.transAxes)
        ax_tb.plot([0, 1], [0.88, 0.88], color=BDI_GREEN, linewidth=1.0,
                   transform=ax_tb.transAxes)
        cols_h = ['Portafolio', 'Retorno Anual', 'Volatilidad', 'Sharpe', 'CAGR', 'Max DD']
        col_xs = [0.01, 0.22, 0.38, 0.52, 0.64, 0.78]
        for j, (h, cx) in enumerate(zip(cols_h, col_xs)):
            ax_tb.text(cx, 0.82, h, fontsize=8, fontweight='bold',
                       color=MUTED, va='top', transform=ax_tb.transAxes)
        ax_tb.plot([0, 1], [0.76, 0.76], color=BORDER, linewidth=0.5,
                   transform=ax_tb.transAxes)
        for i, (name, m) in enumerate(metricas.items()):
            yrow = 0.70 - i * 0.17
            if i % 2 == 0:
                ax_tb.add_patch(mpatches.Rectangle(
                    (0, yrow - 0.05), 1, 0.17, transform=ax_tb.transAxes,
                    facecolor=LIGHT, edgecolor='none'))
            c_row = PCW[i % len(PCW)]
            row_v = [name, pct(m['Retorno Anual']), pct(m['Volatilidad']),
                     f"{m['Sharpe Ratio']:.3f}", pct(m['CAGR']), pct(m['Máx. Drawdown'])]
            for j, (v, cx) in enumerate(zip(row_v, col_xs)):
                ax_tb.text(cx, yrow, v, fontsize=8.5, va='top',
                           color=c_row if j == 0 else DARK,
                           fontweight='bold' if j == 0 else 'normal',
                           transform=ax_tb.transAxes)

        _footer(fig)
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 2 — LOS 4 PORTAFOLIOS (Portrait A4)
        # ══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(WHITE)

        _section_strip(fig, 0.91, 0.07,
                       'Que representan los 4 portafolios',
                       'Cada portafolio responde a un criterio de optimizacion distinto '
                       'y se adapta a un perfil de inversor especifico.',
                       BDI_GREEN)

        PORT_INFO = {
            'Min. Volatilidad':  ('Min. Volatilidad',  BDI_TEAL,    'Conservador',   '#E0F7FA', '#004D55'),
            'Mín. Volatilidad':  ('Mín. Volatilidad',  BDI_TEAL,    'Conservador',   '#E0F7FA', '#004D55'),
            'Máx. Sharpe':       ('Máx. Sharpe',       BDI_GREEN,   'Arriesgado',    '#E8F5E9', '#1B5E20'),
            'Max. Sharpe':       ('Max. Sharpe',        BDI_GREEN,   'Arriesgado',    '#E8F5E9', '#1B5E20'),
            'Equiponderado':     ('Equiponderado',      '#6e9900',   'Moderado',      '#F1F8E9', '#2E5500'),
            'Personalizada':     ('Personalizada',      '#8e44ad',   'A medida',      '#F3E5F5', '#4A148C'),
        }
        GENERIC_PORT = ('Objetivo',    '#e88c00',   'Objetivo',    '#FFF8E1', '#7B3A00')

        DESC_PORT = {
            'Conservador':   (
                'Minimiza la varianza total del portafolio. Selecciona las ponderaciones '
                'que reducen al maximo las fluctuaciones del valor, sin importar el retorno '
                'esperado. Ideal para quienes priorizan la proteccion del capital y tienen '
                'baja tolerancia a la volatilidad.'
            ),
            'Arriesgado':    (
                'Maximiza el Ratio de Sharpe, es decir, el retorno adicional por unidad de '
                'riesgo asumida. Es la cartera mas eficiente segun la Teoria Moderna de '
                'Portafolios. Recomendada para inversores con alta tolerancia al riesgo '
                'que buscan el maximo crecimiento en el largo plazo.'
            ),
            'Moderado':      (
                'Asigna el mismo peso a cada activo sin ninguna optimizacion matematica: '
                'si hay 6 activos, cada uno recibe exactamente el 16.7% del portafolio. '
                'Estrategia simple, robusta y transparente. Ideal para perfiles moderados '
                'que buscan diversificacion sin concentrar en ningun activo particular.'
            ),
            'A medida':      (
                'Distribucion definida directamente por el usuario, reflejando sus '
                'convicciones, preferencias personales y experiencia propia. Permite '
                'comparar la eleccion individual contra las carteras optimizadas '
                'matematicamente y evaluar el costo o beneficio de esa decision.'
            ),
            'Objetivo':      (
                'Construida buscando un retorno objetivo especifico dentro de la frontera '
                'eficiente. Balancea el retorno esperado y el riesgo para alcanzar la '
                'meta de rentabilidad indicada con el menor nivel de volatilidad posible.'
            ),
        }

        port_list = list(portfolios.keys())
        # Layout vertical — 4 cards apiladas una debajo de la otra
        CARD_H   = 0.195
        CARD_GAP = 0.018
        MARGIN_L = 0.05
        CARD_W   = 0.90
        START_Y  = 0.905

        for idx, pname in enumerate(port_list[:4]):
            info_t = PORT_INFO.get(pname, None)
            if info_t:
                _, pc, profile, bb, bf = info_t
            else:
                _, pc, profile, bb, bf = GENERIC_PORT
            desc = DESC_PORT.get(profile, DESC_PORT['Objetivo'])
            cy   = START_Y - (idx + 1) * CARD_H - idx * CARD_GAP

            ax_c = fig.add_axes([MARGIN_L, cy, CARD_W, CARD_H])
            _card_bg(ax_c, '#FAFAFA')
            _ax_off(ax_c, '#FAFAFA')
            _left_border(ax_c, pc, 0.012)

            # Nombre + badge de perfil en la misma línea
            ax_c.text(0.030, 0.88, pname, fontsize=10.5, fontweight='bold',
                      color=pc, va='top', transform=ax_c.transAxes)
            ax_c.text(0.300, 0.88, f'  Perfil: {profile}', fontsize=8,
                      fontweight='bold', color=bf, va='top',
                      transform=ax_c.transAxes,
                      bbox=dict(boxstyle='round,pad=0.28', facecolor=bb, edgecolor='none'))
            wrapped = textwrap.fill(desc, width=105)
            ax_c.text(0.030, 0.65, wrapped, fontsize=8, color=DARK,
                      va='top', transform=ax_c.transAxes, linespacing=1.42)

            m = metricas.get(pname)
            if m:
                note = (f"Retorno anual: {pct(m['Retorno Anual'])}   |   "
                        f"Volatilidad: {pct(m['Volatilidad'])}   |   "
                        f"Sharpe: {m['Sharpe Ratio']:.2f}   |   "
                        f"CAGR: {pct(m['CAGR'])}")
                ax_c.text(0.030, 0.09, note, fontsize=7.5, color=MUTED,
                          va='bottom', transform=ax_c.transAxes,
                          style='italic')

        _footer(fig)
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 3 — GLOSARIO DE METRICAS (Portrait A4)
        # ══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(WHITE)

        _section_strip(fig, 0.92, 0.06,
                       'Glosario de metricas del analisis',
                       'Definicion de cada indicador utilizado para comparar y evaluar los portafolios.',
                       BDI_TEAL)

        METRICS_DEF = [
            {
                'name':    'Ratio de Sharpe',
                'cat':     'Rentabilidad / Riesgo',
                'cat_bg':  '#E8F5E9', 'cat_fg': '#1B5E20',
                'color':   BDI_GREEN,
                'desc':    ('Mide el retorno adicional obtenido por cada unidad de riesgo asumida, '
                            'usando la tasa libre de riesgo como referencia. Un Sharpe mayor a 1 '
                            'es bueno; mayor a 2 se considera excelente. Permite comparar '
                            'portafolios con distintos niveles de riesgo en igualdad de condiciones.'),
                'formula': r'$S = \dfrac{R_p - R_f}{\sigma_p}$',
                'fdetail': r'$R_p$: retorno portafolio   $R_f$: tasa libre de riesgo   $\sigma_p$: volatilidad del portafolio',
            },
            {
                'name':    'CAGR — Tasa de Crecimiento Anual Compuesta',
                'cat':     'Crecimiento',
                'cat_bg':  '#E0F7FA', 'cat_fg': '#006064',
                'color':   BDI_TEAL,
                'desc':    ('Indica a que tasa creció el portafolio en promedio por año durante '
                            'el periodo analizado. A diferencia del retorno simple, el CAGR '
                            'considera el efecto del interes compuesto, reflejando el verdadero '
                            'crecimiento anualizado de la inversion inicial.'),
                'formula': r'$CAGR = \left(\dfrac{V_f}{V_i}\right)^{1/n} - 1$',
                'fdetail': r'$V_f$: valor final   $V_i$: valor inicial   $n$: años del periodo',
            },
            {
                'name':    'Volatilidad Anual',
                'cat':     'Riesgo',
                'cat_bg':  '#FFF8E1', 'cat_fg': '#7B3A00',
                'color':   '#e88c00',
                'desc':    ('Desviacion estandar anualizada de los retornos diarios. Cuantifica '
                            'la incertidumbre o variacion esperada en el valor del portafolio. '
                            'A mayor volatilidad, mayor incertidumbre sobre los resultados '
                            'futuros. Los portafolios conservadores buscan minimizar este valor.'),
                'formula': r'$\sigma_{anual} = \sigma_{diaria} \times \sqrt{252}$',
                'fdetail': r'$\sigma_{diaria}$: desvio estandar de retornos diarios   $252$: dias habiles por año',
            },
            {
                'name':    'Maximo Drawdown',
                'cat':     'Perdida Maxima',
                'cat_bg':  '#FFEBEE', 'cat_fg': '#B71C1C',
                'color':   '#c0392b',
                'desc':    ('Caida porcentual maxima registrada desde el pico mas alto hasta el '
                            'punto mas bajo dentro del periodo analizado. Es el indicador clave '
                            'para evaluar la resistencia de un portafolio ante crisis o '
                            'correcciones de mercado. Un drawdown menor implica mayor proteccion.'),
                'formula': r'$MaxDD = \dfrac{Valle - Pico}{Pico}$',
                'fdetail': r'$Valle$: valor minimo posterior al pico   $Pico$: maximo historico previo',
            },
        ]

        # 4 cards a lo largo de la pagina
        M_START = 0.91
        M_H     = 0.195
        M_GAP   = 0.022

        for idx, md in enumerate(METRICS_DEF):
            cy_m = M_START - (idx + 1) * M_H - idx * M_GAP
            ax_m = fig.add_axes([0.05, cy_m, 0.90, M_H])
            _card_bg(ax_m, '#FAFAFA')
            _ax_off(ax_m, '#FAFAFA')
            _left_border(ax_m, md['color'], 0.010)

            ax_m.text(0.025, 0.92, md['name'], fontsize=9.5, fontweight='bold',
                      color=md['color'], va='top', transform=ax_m.transAxes)
            ax_m.text(0.025, 0.70, md['cat'], fontsize=7, fontweight='bold',
                      color=md['cat_fg'], va='top', transform=ax_m.transAxes,
                      bbox=dict(boxstyle='round,pad=0.26',
                                facecolor=md['cat_bg'], edgecolor='none'))
            wrapped_m = textwrap.fill(md['desc'], width=88)
            ax_m.text(0.025, 0.54, wrapped_m, fontsize=7.8, color=DARK,
                      va='top', transform=ax_m.transAxes, linespacing=1.42)
            # Fórmula con LaTeX (mathtext)
            ax_m.add_patch(mpatches.Rectangle(
                (0.025, 0.0), 0.95, 0.24, transform=ax_m.transAxes,
                facecolor=LIGHT, edgecolor='none'))
            ax_m.text(0.50, 0.18, md['formula'], fontsize=13,
                      color=md['color'], va='top', ha='center',
                      transform=ax_m.transAxes)
            ax_m.text(0.040, 0.06, md['fdetail'], fontsize=7,
                      color=MUTED, va='top', transform=ax_m.transAxes)

        _footer(fig)
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 4 — ESPACIO DE MARKOWITZ (Landscape, fondo blanco)
        # ══════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(AW, AH))
        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor('#F8F9FA')
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=DARK)
        sc = ax.scatter(sim_vol_arr * 100, sim_ret_arr * 100,
                        c=sim_sharpe_arr, cmap='YlGn', alpha=0.30, s=4, zorder=1)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('Sharpe Ratio', color=DARK, fontsize=10)
        cb.ax.yaxis.set_tick_params(color=DARK, labelsize=9)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=DARK)
        valid = ~np.isnan(vol_fe)
        ax.plot(vol_fe[valid] * 100, ret_range[valid] * 100,
                color=BDI_GREEN, linewidth=3.5, zorder=6,
                label='Frontera Eficiente', solid_capstyle='round')
        vcr = np.linspace(vol_min * 0.7, vol_sharpe * 1.3, 80)
        ax.plot(vcr * 100, rf * 100 + sharpe_val * vcr * 100,
                '--', color=BDI_TEAL, linewidth=1.8, alpha=0.9, zorder=4, label='CML')
        pts4 = [(vol_sharpe, ret_sharpe, '*', 350, BDI_GREEN,
                 f'Max Sharpe ({sharpe_val:.2f})'),
                (vol_min,    ret_vol,    'D', 130, BDI_TEAL,
                 f'Min Vol ({pct(vol_min)})'),
                (vol_eq,     ret_eq,     's', 130, '#6e9900',
                 f'Equip ({pct(ret_eq)})')]
        if w_obj_arr is not None:
            pts4.append((vol_obj, ret_obj_real, 'P', 150, '#e88c00',
                         f'{obj_label} ({pct(ret_obj_real)})'))
        if has_custom_weights and not np.isnan(vol_custom):
            pts4.append((vol_custom, ret_custom, '^', 150, '#8e44ad',
                         f'Personalizada ({pct(ret_custom)})'))
        for vp, rp, mk, sz, cl, lb in pts4:
            ax.scatter(vp * 100, rp * 100, marker=mk, s=sz, color=cl,
                       zorder=10, edgecolors='white', linewidth=0.8, label=lb)
        ax.axhline(rf * 100, color=MUTED, linewidth=0.6, linestyle=':',
                   alpha=0.6, label=f'Rf={pct(rf)}')
        xp = (sim_vol_arr.max() - sim_vol_arr.min()) * 100 * 0.12
        yp = (sim_ret_arr.max() - sim_ret_arr.min()) * 100 * 0.15
        ax.set_xlim(sim_vol_arr.min() * 100 - xp, sim_vol_arr.max() * 100 + xp)
        ax.set_ylim(sim_ret_arr.min() * 100 - yp, sim_ret_arr.max() * 100 + yp)
        ax.set_title('Espacio de Portafolios — Modelo de Markowitz',
                     fontsize=14, fontweight='bold', color=DARK, pad=12)
        ax.set_xlabel('Volatilidad Anual (%)', fontsize=11, color=DARK)
        ax.set_ylabel('Retorno Anual (%)',     fontsize=11, color=DARK)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=9, framealpha=1.0, loc='upper left',
                  facecolor=WHITE, edgecolor=BORDER, labelcolor=DARK)
        ax.grid(True, alpha=0.28, color=BORDER)
        _footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 5 — COMPOSICION (donuts 2×2, portrait A4, fondo blanco)
        # ══════════════════════════════════════════════════════════════
        all_ports = list(portfolios.items())
        port_keys = list(portfolios.keys())
        for bs in range(0, len(all_ports), 4):
            batch = all_ports[bs:bs + 4]
            nb    = len(batch)
            nrows = (nb + 1) // 2
            ncols = min(2, nb)
            fig, axes_d = plt.subplots(nrows, ncols, figsize=(8.27, 11.69))
            fig.patch.set_facecolor(WHITE)
            # Aplanar axes siempre en lista
            if nb == 1:
                axes_d = [axes_d]
            elif nrows == 1:
                axes_d = list(axes_d)
            else:
                axes_d = [ax for row in axes_d for ax in (row if hasattr(row, '__iter__') else [row])]
            # Ocultar axes sobrantes si nb es impar
            for ax_extra in axes_d[nb:]:
                ax_extra.axis('off')

            for ax, (name, w) in zip(axes_d, batch):
                ax.set_facecolor(WHITE)
                mask = w > 0.005
                lbs  = [assets[i] for i in range(num_assets) if mask[i]]
                szs  = w[mask]
                other = 1 - szs.sum()
                if other > 0.001:
                    lbs.append('Otros'); szs = np.append(szs, other)
                cols_d = PALETTE[:len(lbs)]
                wedges, _, atxts = ax.pie(
                    szs, labels=None, colors=cols_d,
                    autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                    startangle=90, pctdistance=0.78,
                    wedgeprops={'linewidth': 1.5, 'edgecolor': WHITE, 'width': 0.58},
                )
                for at in atxts:
                    at.set_fontsize(9); at.set_color(DARK); at.set_fontweight('bold')
                m    = metricas[name]
                pidx = port_keys.index(name) if name in port_keys else 0
                c_p  = PCW[pidx % len(PCW)]
                # Solo retorno en el centro
                ax.text(0, 0, pct(m['Retorno Anual']),
                        ha='center', va='center', fontsize=15,
                        fontweight='bold', color=c_p)
                ax.text(0, -0.20, 'Retorno anual',
                        ha='center', va='center', fontsize=7.5, color=MUTED)
                # Sharpe y Vol ABAJO del gráfico (fuera del donut)
                ax.text(0.5, -0.05,
                        f"Sharpe: {m['Sharpe Ratio']:.2f}   |   Vol: {pct(m['Volatilidad'])}",
                        ha='center', va='top', fontsize=9, color=DARK,
                        fontweight='bold', transform=ax.transAxes)
                # Leyenda con texto oscuro
                ax.legend(wedges, [f"{lb} ({s*100:.1f}%)" for lb, s in zip(lbs, szs)],
                          loc='lower center', bbox_to_anchor=(0.5, -0.22),
                          fontsize=9, framealpha=1.0, ncol=2,
                          facecolor=WHITE, edgecolor=BORDER, labelcolor=DARK)
                ax.set_title(name, fontsize=11, fontweight='bold', color=c_p, pad=12)
            fig.suptitle('Composicion de Portafolios Optimizados',
                         fontsize=13, fontweight='bold', color=DARK, y=1.01)
            _footer(fig)
            plt.tight_layout(h_pad=4.0, w_pad=3.0)
            pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
            plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 6 — RENDIMIENTO ACUMULADO (fondo blanco)
        # ══════════════════════════════════════════════════════════════
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(AW, AH))
        fig.patch.set_facecolor(WHITE)
        for ax_sub in [ax1, ax2]:
            ax_sub.set_facecolor('#F8F9FA')
            for sp in ax_sub.spines.values():
                sp.set_color(BORDER)
            ax_sub.tick_params(colors=DARK)
        for i, col in enumerate(cum_port.columns):
            ax1.plot(cum_port.index, cum_port[col] * 100,
                     label=col, linewidth=2.0, color=PCW[i % len(PCW)])
        ax1.axhline(0, color=MUTED, linewidth=0.5, linestyle='--', alpha=0.5)
        ax1.set_title('Rendimiento Acumulado — Portafolios',
                      fontsize=12, fontweight='bold', color=DARK)
        ax1.set_ylabel('Rend. Acumulado (%)', fontsize=10, color=DARK)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax1.legend(fontsize=9, framealpha=1.0, facecolor=WHITE, edgecolor=BORDER, labelcolor=DARK)
        ax1.grid(True, alpha=0.28, color=BORDER)
        for i, col in enumerate(cum_assets.columns):
            ax2.plot(cum_assets.index, cum_assets[col] * 100,
                     label=col, linewidth=1.5,
                     color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax2.axhline(0, color=MUTED, linewidth=0.5, linestyle='--', alpha=0.5)
        ax2.set_title('Rendimiento Acumulado — Activos Individuales',
                      fontsize=12, fontweight='bold', color=DARK)
        ax2.set_ylabel('Rend. Acumulado (%)', fontsize=10, color=DARK)
        ax2.set_xlabel('Fecha', fontsize=10, color=DARK)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax2.legend(fontsize=9, framealpha=1.0, facecolor=WHITE, edgecolor=BORDER,
                   labelcolor=DARK, bbox_to_anchor=(1.01, 1), loc='upper left')
        ax2.grid(True, alpha=0.28, color=BORDER)
        _footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 7 — METRICAS COMPARATIVAS (barras, fondo blanco)
        # ══════════════════════════════════════════════════════════════
        pn_p = list(portfolios.keys())
        mp_p = [('Retorno Anual', 'Retorno Anual (%)', True),
                ('Volatilidad',   'Volatilidad (%)',   False),
                ('Sharpe Ratio',  'Ratio de Sharpe',   True),
                ('Máx. Drawdown', 'Máx. Drawdown (%)', False)]
        fig, axes_m = plt.subplots(2, 2, figsize=(AW, AH))
        fig.patch.set_facecolor(WHITE)
        for ax, (col, ylabel, hb) in zip(axes_m.flatten(), mp_p):
            ax.set_facecolor('#F8F9FA')
            for sp in ax.spines.values():
                sp.set_color(BORDER)
            ax.tick_params(colors=DARK)
            vals = [metricas[p][col] for p in pn_p]
            sc   = 100 if '%' in ylabel else 1
            bars = ax.bar(pn_p, [v * sc for v in vals],
                          color=PCW[:len(pn_p)],
                          edgecolor=WHITE, linewidth=1.0, width=0.5)
            bidx = np.argmax(vals) if hb else np.argmin(vals)
            bars[bidx].set_edgecolor(BDI_GREEN); bars[bidx].set_linewidth(2.5)
            for bar, val in zip(bars, vals):
                lbl = f'{val*100:.1f}%' if '%' in ylabel else f'{val:.2f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(bar.get_height()) * 0.03,
                        lbl, ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=DARK)
            ax.set_title(ylabel, fontsize=11, fontweight='bold', color=DARK)
            if '%' in ylabel:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
            ax.set_xticklabels(pn_p, rotation=15, ha='right', fontsize=8, color=DARK)
            ax.grid(axis='y', alpha=0.28, color=BORDER)
        fig.suptitle('Comparacion de Metricas — Portafolios Optimizados',
                     fontsize=13, fontweight='bold', color=DARK, y=1.01)
        _footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 8 — CORRELACION (fondo blanco, colormap verde)
        # ══════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(max(7, num_assets + 2), max(6, num_assets + 1)))
        fig.patch.set_facecolor(WHITE)
        cmap_g = LinearSegmentedColormap.from_list(
            'BDI_G',
            ['#ffffff', '#c8e6c9', '#81c784', '#4caf50',
             '#2e7d32', '#1b5e20', '#137247'], N=256)
        sns.heatmap(corr_matrix, ax=ax, annot=True, fmt='.2f',
                    annot_kws={'size': 9, 'weight': 'bold'},
                    cmap=cmap_g, vmin=0, vmax=1,
                    linewidths=0.6, linecolor=WHITE, square=True,
                    cbar_kws={'shrink': 0.8, 'label': 'Correlacion', 'pad': 0.02})
        for to in ax.texts:
            try:
                v = float(to.get_text())
                to.set_color(DARK if v < 0.60 else WHITE)
            except Exception:
                pass
        ax.set_title('Matriz de Correlacion entre Activos',
                     fontsize=13, fontweight='bold', color=DARK, pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=9, color=DARK)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=9, color=DARK)
        _footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 9 — CAGR COMPARATIVO (fondo blanco)
        # ══════════════════════════════════════════════════════════════
        cagr_a  = {a: calc_cagr(cum_assets[a]) * 100 for a in assets}
        cagr_p  = {p: calc_cagr(cum_port[p])   * 100 for p in port_daily.columns}
        all_n   = list(cagr_a) + list(cagr_p)
        all_v   = list(cagr_a.values()) + list(cagr_p.values())
        all_ty  = (['Activo'] * len(cagr_a) +
                   ['Personalizada' if p == 'Personalizada' else 'Portafolio'
                    for p in cagr_p])
        tc_w    = {'Activo': BDI_TEAL, 'Portafolio': BDI_GREEN, 'Personalizada': '#8e44ad'}
        sd      = sorted(zip(all_v, all_n, all_ty), reverse=True)
        sv, sn, sty = zip(*sd)
        fig, ax = plt.subplots(figsize=(max(9, len(all_n) * 1.1), AH))
        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor('#F8F9FA')
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=DARK)
        bars_c = ax.bar(sn, sv, color=[tc_w.get(t, MUTED) for t in sty],
                        edgecolor=WHITE, linewidth=1.0, width=0.6)
        for bar, val in zip(bars_c, sv):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.4 if val >= 0 else -1.2),
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=DARK)
        ax.axhline(0, color=MUTED, linewidth=0.7, linestyle='--', alpha=0.5)
        ax.set_title('CAGR Anual Comparativo — Activos y Portafolios',
                     fontsize=13, fontweight='bold', color=DARK, pad=10)
        ax.set_ylabel('CAGR Anual (%)', fontsize=10, color=DARK)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.set_xticklabels(sn, rotation=30, ha='right', fontsize=9, color=DARK)
        ax.grid(axis='y', alpha=0.28, color=BORDER)
        ax.legend(handles=[mpatches.Patch(color=c, label=t) for t, c in tc_w.items()],
                  fontsize=9, framealpha=0.96, facecolor=WHITE, edgecolor=BORDER)
        _footer(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 10 — MODULO EDUCATIVO (Portrait A4) — 8 conceptos
        # ══════════════════════════════════════════════════════════════
        EDU = [
            {
                'title': '1. Teoria Moderna de Portafolios (Markowitz, 1952)',
                'color': BDI_GREEN,
                'text':  ('Harry Markowitz demostro que es posible construir portafolios que '
                          'maximizan el retorno esperado para cada nivel de riesgo dado. La '
                          'clave: los activos se evaluan por su aporte al riesgo total del '
                          'portafolio, no de forma aislada. Esta teoria es la base de la '
                          'optimizacion cuantitativa moderna.'),
                'formula': r'$\min_w \; w^\top \Sigma \, w \quad \text{s.a.} \quad w^\top\mu = \mu_{obj},\quad \sum_i w_i = 1$',
                'fdetail': r'$w$: pesos   $\Sigma$: matriz de covarianza   $\mu$: retornos esperados',
            },
            {
                'title': '2. La Frontera Eficiente',
                'color': BDI_TEAL,
                'text':  ('Conjunto de portafolios que ofrecen el maximo retorno posible para '
                          'cada nivel de riesgo, o el minimo riesgo para cada nivel de retorno. '
                          'Cualquier portafolio por debajo de la frontera es suboptimo: existe '
                          'otro con igual retorno y menor riesgo.'),
                'formula': r'$FE = \left\{\,(\sigma_p,\,\mu_p)\;\middle|\;\forall p^\prime:\;\sigma_{p^\prime} < \sigma_p \Rightarrow \mu_{p^\prime} < \mu_p\,\right\}$',
                'fdetail': r'$\sigma_p$: volatilidad del portafolio   $\mu_p$: retorno esperado',
            },
            {
                'title': '3. Diversificacion y Reduccion del Riesgo',
                'color': '#6e9900',
                'text':  ('Combinar activos no correlacionados reduce el riesgo total sin '
                          'sacrificar retorno. El "milagro" de la diversificacion: el riesgo '
                          'del portafolio puede ser menor que el promedio de riesgos individuales '
                          'si los activos no se mueven en perfecta sincronia.'),
                'formula': r'$\sigma_p^2 = \sum_i \sum_j w_i\, w_j\, \sigma_{ij}$',
                'fdetail': r'$\sigma_{ij}$: covarianza entre activos $i$ y $j$   $w_i, w_j$: pesos respectivos',
            },
            {
                'title': '4. Correlacion entre Activos',
                'color': '#e88c00',
                'text':  ('La correlacion mide como se mueven dos activos entre si. Valores '
                          'cercanos a +1: se mueven juntos (poca diversificacion). Valores '
                          'cercanos a -1: se mueven opuesto (maxima diversificacion). El objetivo '
                          'es combinar activos con baja correlacion entre si.'),
                'formula': r'$\rho_{ij} = \dfrac{\sigma_{ij}}{\sigma_i \cdot \sigma_j}$',
                'fdetail': r'$\rho_{ij}$: correlacion   $\sigma_{ij}$: covarianza   $\sigma_i, \sigma_j$: volatilidades individuales',
            },
            {
                'title': '5. Ratio de Sharpe y Linea de Mercado de Capitales (CML)',
                'color': BDI_GREEN,
                'text':  ('La CML conecta el activo libre de riesgo con el portafolio de maximo '
                          'Sharpe. Todo inversor racional elegira un punto sobre esta linea, '
                          'combinando el activo libre de riesgo con el portafolio de mercado '
                          'segun su tolerancia al riesgo.'),
                'formula': r'$E[R_p] = R_f + S \cdot \sigma_p$',
                'fdetail': r'$E[R_p]$: retorno esperado   $R_f$: tasa libre de riesgo   $S$: Sharpe   $\sigma_p$: volatilidad objetivo',
            },
            {
                'title': '6. CAGR vs. Retorno Aritmetico',
                'color': BDI_TEAL,
                'text':  ('El retorno aritmetico promedia las ganancias y perdidas anuales. '
                          'El CAGR mide el crecimiento real compuesto: cuanto multiplico el '
                          'capital. Para periodos largos, el CAGR siempre es menor o igual al '
                          'retorno aritmetico; la diferencia crece con la volatilidad.'),
                'formula': r'$CAGR \approx \mu_{arit} - \dfrac{\sigma^2}{2}$',
                'fdetail': r'$\mu_{arit}$: retorno medio aritmetico   $\sigma^2$: varianza anual del portafolio',
            },
            {
                'title': '7. Drawdown y Recuperacion',
                'color': '#c0392b',
                'text':  ('El drawdown es la perdida desde un pico. Su importancia radica en '
                          'el tiempo de recuperacion: una caida del 50% requiere un rebote del '
                          '100% para volver al punto de partida. Portafolios con menor drawdown '
                          'son mas resilientes y psicologicamente mas sostenibles.'),
                'formula': r'$R_{nec} = \dfrac{1}{1 + MaxDD} - 1$',
                'fdetail': r'$MaxDD = -0.50$ implica necesitar $+100\%$ para recuperar el capital',
            },
            {
                'title': '8. Rebalanceo Periodico',
                'color': '#8e44ad',
                'text':  ('Con el tiempo, los activos crecen distinto desviando los pesos del '
                          'objetivo original. Rebalancear implica vender lo que subio y comprar '
                          'lo que bajo para volver a la asignacion deseada. En mercados volatiles, '
                          'el rebalanceo puede mejorar el retorno de largo plazo.'),
                'formula': r'$w_i^{actual} = \dfrac{V_i}{\sum_j V_j} \quad\Rightarrow\quad \delta_i = w_i^{obj} - w_i^{actual}$',
                'fdetail': r'$V_i$: valor actual del activo $i$   $\delta_i$: ajuste necesario en el peso',
            },
        ]

        # 4 cards por pagina, 2 paginas total
        EDU_H   = 0.192
        EDU_GAP = 0.018
        EDU_MARGIN = 0.05

        for page_e in range(2):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor(WHITE)

            # Header de la pagina educativa
            ax_edu_hdr = fig.add_axes([0, 0.92, 1, 0.08])
            ax_edu_hdr.set_facecolor(BDI_TEAL)
            ax_edu_hdr.set_xlim(0, 1); ax_edu_hdr.set_ylim(0, 1)
            ax_edu_hdr.axis('off')
            page_label = f'Modulo Educativo  —  Parte {page_e + 1} de 2'
            ax_edu_hdr.text(0.05, 0.72, page_label, fontsize=13, fontweight='bold',
                            color=WHITE, va='top')
            ax_edu_hdr.text(0.05, 0.28, 'Conceptos, formulas y principios de la inversion cuantitativa',
                            fontsize=8.5, color=BDI_LIME, va='top', fontweight='bold')

            cards_this = EDU[page_e * 4: page_e * 4 + 4]
            E_START = 0.91

            for idx_e, card_e in enumerate(cards_this):
                cy_e = E_START - (idx_e + 1) * EDU_H - idx_e * EDU_GAP
                ax_e = fig.add_axes([EDU_MARGIN, cy_e, 1 - 2 * EDU_MARGIN, EDU_H])
                _card_bg(ax_e, '#FAFAFA')
                _ax_off(ax_e, '#FAFAFA')
                _left_border(ax_e, card_e['color'], 0.010)

                ax_e.text(0.025, 0.93, card_e['title'], fontsize=9.5, fontweight='bold',
                          color=card_e['color'], va='top', transform=ax_e.transAxes)
                wrapped_e = textwrap.fill(card_e['text'], width=90)
                ax_e.text(0.025, 0.72, wrapped_e, fontsize=7.8, color=DARK,
                          va='top', transform=ax_e.transAxes, linespacing=1.42)
                # Formula box con LaTeX (mathtext)
                ax_e.add_patch(mpatches.Rectangle(
                    (0.025, 0.00), 0.95, 0.24, transform=ax_e.transAxes,
                    facecolor=LIGHT, edgecolor='none'))
                ax_e.text(0.50, 0.19, card_e['formula'], fontsize=12,
                          color=card_e['color'], va='top', ha='center',
                          transform=ax_e.transAxes)
                ax_e.text(0.040, 0.07, card_e['fdetail'], fontsize=7,
                          color=MUTED, va='top', transform=ax_e.transAxes)

            _footer(fig)
            pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
            plt.close(fig)

        # ══════════════════════════════════════════════════════════════
        # PAGINA 12 — DISCLAIMER (Portrait A4) — texto corrido
        # ══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(WHITE)

        # Header rojo disclaimer
        ax_dh = fig.add_axes([0, 0.88, 1, 0.12])
        ax_dh.set_facecolor('#c0392b')
        ax_dh.set_xlim(0, 1); ax_dh.set_ylim(0, 1)
        ax_dh.axis('off')
        ax_dh.text(0.06, 0.80, 'Aviso Legal y Disclaimer', fontsize=15,
                   fontweight='bold', color=WHITE, va='top')
        ax_dh.text(0.06, 0.34, 'Este informe fue generado automaticamente y tiene caracter exclusivamente informativo y educativo.',
                   fontsize=8, color='#ffcdd2', va='top')

        # Texto corrido — un solo bloque de texto fluido
        ax_disc = fig.add_axes([0.08, 0.10, 0.84, 0.76])
        _ax_off(ax_disc, WHITE)

        DISC_ITEMS = [
            ('No constituye asesoramiento financiero.',
             'Este informe no constituye ni debe interpretarse como asesoramiento financiero, '
             'de inversion, legal, impositivo ni de ninguna otra naturaleza profesional. '
             'La informacion presentada tiene fines unicamente informativos y educativos. '
             'BDI Consultora no tiene relacion de mandato ni gestion de activos con el usuario.'),
            ('Rendimientos pasados no garantizan resultados futuros.',
             'Las proyecciones, simulaciones y analisis historicos presentados no constituyen '
             'garantia de rendimiento futuro. Los mercados financieros son dinamicos y estan '
             'sujetos a factores macroeconomicos, politicos y de liquidez impredecibles '
             'que pueden afectar significativamente los resultados reales.'),
            ('Riesgo de perdida de capital.',
             'Toda inversion en mercados financieros conlleva riesgos, incluyendo la posible '
             'perdida parcial o total del capital invertido. El inversor debe evaluar '
             'cuidadosamente su tolerancia al riesgo, su horizonte temporal y su situacion '
             'patrimonial antes de tomar cualquier decision de inversion.'),
            ('Precision de los datos.',
             'Los datos historicos utilizados provienen de fuentes publicas (Yahoo Finance u '
             'otras) y pueden contener inexactitudes, ajustes por dividendos o splits no '
             'reflejados. BDI Consultora no se responsabiliza por errores en los datos '
             'fuente ni por las consecuencias de decisiones basadas en este analisis.'),
            ('Consulte con un asesor certificado.',
             'Antes de tomar cualquier decision de inversion, se recomienda consultar con '
             'un asesor financiero certificado que conozca su situacion patrimonial, fiscal '
             'y sus objetivos financieros personales. Este informe no reemplaza el '
             'asesoramiento profesional individualizado.'),
        ]

        y_cur = 0.97
        LINE_SP = 1.55
        WRAP_W  = 88
        for title_d, body_d in DISC_ITEMS:
            # Título en negrita
            ax_disc.text(0.0, y_cur, title_d, fontsize=8, fontweight='bold',
                         color='#c0392b', va='top', transform=ax_disc.transAxes)
            y_cur -= 0.048
            # Cuerpo envuelto
            wrapped_d = textwrap.fill(body_d, width=WRAP_W)
            n_lines = wrapped_d.count('\n') + 1
            ax_disc.text(0.0, y_cur, wrapped_d, fontsize=7.5, color=DARK,
                         va='top', transform=ax_disc.transAxes, linespacing=LINE_SP)
            y_cur -= n_lines * 0.052 + 0.032

        # BDI contact block
        ax_bdi = fig.add_axes([0.06, 0.03, 0.88, 0.08])
        ax_bdi.set_facecolor(BDI_GREEN)
        ax_bdi.set_xlim(0, 1); ax_bdi.set_ylim(0, 1)
        ax_bdi.axis('off')
        ax_bdi.text(0.05, 0.82, 'BDI Consultora de Inversiones',
                    fontsize=11, fontweight='bold', color=WHITE, va='top',
                    transform=ax_bdi.transAxes)
        ax_bdi.text(0.05, 0.44, 'hola@bdiconsultora.com',
                    fontsize=9, color=BDI_LIME, va='top', transform=ax_bdi.transAxes)
        ax_bdi.text(0.05, 0.12, f'Informe generado el {fecha_hoy}  —  BDI Optimizador de Carteras v2.0',
                    fontsize=7, color=WHITE, alpha=0.80, va='top',
                    transform=ax_bdi.transAxes)

        _footer(fig)
        pdf.savefig(fig, facecolor=WHITE, bbox_inches='tight')
        plt.close(fig)

        # Metadata del PDF
        d = pdf.infodict()
        d['Title']   = 'BDI — Analisis y Optimizacion de Cartera'
        d['Author']  = 'BDI Consultora de Inversiones'
        d['Subject'] = f'Portafolio: {", ".join(assets)}'
        d['Creator'] = 'BDI Optimizador de Carteras v2.0'

    buf.seek(0)
    return buf.getvalue()

# ─────────────────────────────────────────────
#  HEADER PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class="bdi-header">
    <h1 style="color:#EFEDEA; margin:0; font-size:2.4rem; letter-spacing:3px; font-weight:800;">
        ⚡ BDI — OPTIMIZADOR DE CARTERAS v2.0
    </h1>
    <p style="color:#B5E61D; margin:0.5rem 0 0.2rem 0; font-size:1.05rem; font-weight:600; letter-spacing:1px;">
        Modelo de Markowitz con Frontera Eficiente
    </p>
    <p style="color:#EFEDEA; margin:0; font-size:0.9rem; opacity:0.85;">
        BDI Consultora de Inversiones — Mariano Ricciardi
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR — CONFIGURACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:0.8rem 0 0.4rem 0;
                background:linear-gradient(135deg,#0d4d2e,#137247);
                border-radius:10px; margin-bottom:0.5rem;">
        <h2 style="color:#B5E61D; font-size:1.2rem; margin:0;">⚙️ Configuración</h2>
        <p style="color:#EFEDEA; font-size:0.82rem; margin:0.2rem 0 0 0;">Parámetros de optimización</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">📌 Activos a analizar</p>', unsafe_allow_html=True)
    tickers_input = st.text_area(
        "Tickers separados por coma",
        value="JPM:20, AAPL:30, MSFT:50, GOOGL:5, META:10, V:5",
        height=85,
        help="Usá el mismo formato de ticker que figura en Yahoo Finance. Ej: JPM, AAPL, MSFT, MELI, GOOGL, SPY, GLD, XOM.\n\n"
             "Para agregar una cartera personalizada indicá el peso (%): JPM:30, AAPL:25, MSFT:20, GOOGL:15, MELI:10",
    )
    st.caption(
        "💡 **Tip:** Podés especificar pesos para una cartera personalizada. "
        "Ej: `JPM:30, AAPL:25, MSFT:20, GOOGL:15, MELI:10`"
    )

    st.markdown("---")
    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">📅 Período de análisis</p>', unsafe_allow_html=True)
    anios = st.slider("Años de historia", min_value=1, max_value=15, value=5)

    st.markdown("---")
    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">📈 Parámetros de mercado</p>', unsafe_allow_html=True)
    rf_pct = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0, max_value=30.0,
        value=4.5, step=0.1,
        help="Ej: 4.5 equivale al 4.5% anual",
    )
    rf = rf_pct / 100

    st.markdown("---")
    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">🎯 Retorno objetivo <em style="color:#555;font-size:0.85em;">(opcional)</em></p>', unsafe_allow_html=True)
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
    st.markdown('<p style="color:#137247; font-weight:700; margin-bottom:4px;">⚖️ Restricciones de pesos</p>', unsafe_allow_html=True)
    min_peso = st.slider("Peso mínimo por activo (%)", 0, 50, 0) / 100
    max_peso = st.slider("Peso máximo por activo (%)", 10, 100, 100) / 100
    if max_peso < min_peso:
        max_peso = min_peso

    st.markdown("---")
    run_button = st.button("🚀  EJECUTAR ANÁLISIS", type="primary")

# ─────────────────────────────────────────────
#  ESTADO INICIAL (pantalla de bienvenida + instructivo)
# ─────────────────────────────────────────────
if not run_button and 'results_ready' not in st.session_state:

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div style="text-align:center; padding:2rem 1rem 1rem 1rem;">
            <p style="font-size:4rem; margin:0;">📊</p>
            <h3 style="color:#17BEBB; margin:1rem 0 0.5rem 0; font-size:1.5rem;">
                Listo para optimizar tu cartera
            </h3>
            <p style="font-size:1rem; margin:0; color:#EFEDEA; line-height:1.7;">
                Ingresá los activos, configurá los parámetros<br/>
                y presioná <strong style="color:#B5E61D;">🚀 EJECUTAR ANÁLISIS</strong><br/>
                para obtener tu portafolio óptimo.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:1.5rem;">
            <strong style="color:#B5E61D;">📌 Tickers de ejemplo (formato Yahoo Finance):</strong><br/><br/>
            <strong style="color:#17BEBB;">Acciones USA:</strong>
            JPM &nbsp;·&nbsp; AAPL &nbsp;·&nbsp; MSFT &nbsp;·&nbsp; GOOGL &nbsp;·&nbsp; AMZN &nbsp;·&nbsp; NVDA &nbsp;·&nbsp; XOM<br/>
            <strong style="color:#17BEBB;">ETFs:</strong>
            SPY &nbsp;·&nbsp; QQQ &nbsp;·&nbsp; IWM &nbsp;·&nbsp; GLD &nbsp;·&nbsp; TLT<br/>
            <strong style="color:#17BEBB;">Latinoamérica:</strong>
            MELI &nbsp;·&nbsp; NU &nbsp;·&nbsp; PBR<br/><br/>
            <strong style="color:#B5E61D;">💼 Cartera personalizada con pesos:</strong><br/>
            <span style="color:#EFEDEA; font-size:0.9rem;">
                Agregá el peso (%) después de cada ticker con <code>:</code><br/>
                Ej: <code>JPM:30, AAPL:25, MSFT:20, GOOGL:15, MELI:10</code>
            </span><br/><br/>
            <span style="color:#9e9e9e; font-size:0.85rem;">
                ⚠️ Ingresá los tickers exactamente como aparecen en
                <strong>Yahoo Finance</strong> para asegurar la descarga de datos.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <h3 style="color:#B5E61D; margin-top:1.5rem; font-size:1.2rem;">
            📋 ¿Cómo usar el optimizador?
        </h3>
        """, unsafe_allow_html=True)

        pasos = [
            ("1", "Ingresá los tickers", "En el panel izquierdo, escribí los símbolos de los activos separados por coma (ej: <strong>AAPL, MSFT, JPM</strong>). Usá el formato de Yahoo Finance."),
            ("2", "Elegí el período", "Seleccioná cuántos años de historia histórica querés analizar (1 a 15 años). Más años = mayor robustez estadística."),
            ("3", "Configurá parámetros", "Ajustá la tasa libre de riesgo (referencia: tasa de la Fed o bono del Tesoro) y las restricciones de peso por activo."),
            ("4", "Ejecutá el análisis", "Presioná <strong style='color:#B5E61D;'>🚀 EJECUTAR ANÁLISIS</strong>. Se calcularán los portafolios óptimos por Sharpe, mínima volatilidad, equiponderado y —opcionalmente— retorno objetivo."),
            ("5", "Explorá los resultados", "Navegá las pestañas: Espacio de Markowitz, Composición, Rendimiento, Métricas, Correlación y CAGR."),
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

# ─────────────────────────────────────────────
#  ANÁLISIS PRINCIPAL
# ─────────────────────────────────────────────
if run_button:
    st.session_state.pop('pdf_ready', None)
    st.session_state.pop('pdf_bytes', None)

    # ── Parse tickers (soporta formato TICKER:PESO para cartera personalizada) ──
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
        st.error("❌ Ingresá al menos 1 ticker.")
        st.stop()

    # ── Descarga ───────────────────────────────────────────────────────
    with st.spinner("📡 Descargando datos de mercado desde Yahoo Finance..."):
        raw = yf.download(tickers, period=f"{anios}y", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw['Close']
        else:
            data = raw[['Close']] if 'Close' in raw.columns else raw
        data = data.dropna(axis=1, how='any')

    if data.shape[1] == 0:
        st.error("❌ Sin datos disponibles. Revisá los tickers e intentá de nuevo.")
        st.stop()

    assets      = list(data.columns)
    descartados = [t for t in tickers if t not in assets]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.success(f"✅ **{len(assets)} activos cargados:** {' · '.join(assets)}")
    with c2:
        if descartados:
            st.warning(f"⚠️ Descartados: {', '.join(descartados)}")

    st.caption(
        f"📅 Período: {data.index[0].strftime('%d/%m/%Y')} → "
        f"{data.index[-1].strftime('%d/%m/%Y')} · {len(data)} ruedas"
    )

    # ── Estadísticas ───────────────────────────────────────────────────
    returns      = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252
    corr_matrix  = returns.corr()
    num_assets   = len(assets)

    bounds           = tuple((min_peso, max_peso) for _ in range(num_assets))
    constraints_base = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1]},)
    w0               = np.full(num_assets, 1.0 / num_assets)

    def _ps(v):
        return port_stats(w, mean_returns, cov_matrix, rf)

    def neg_sharpe(w): return -_ps(w)Z[2]
    def min_vol_fn(w): return  _ps(w)Z[1]

    # ── Optimizaciones ─────────────────────────────────────────────────
    with st.spinner("🔢 Ejecutando optimizaciones (Sharpe, Min-Vol, Equiponderado)..."):
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
                st.warning("⚠️ No fue posible alcanzar el retorno objetivo con las restricciones dadas.")

    # ── Frontera eficiente ─────────────────────────────────────────────
    with st.spinner("📈 Calculando frontera eficiente (100 puntos)..."):
        ret_range = np.linspace(ret_vol, max(mean_returns) * 1.05, 100)
        vol_fe    = []
        for target in ret_range:
            cons = list(constraints_base) + [
                {'type': 'eq', 'fun': lambda x, t=target: np.dot(x, mean_returns) - t}
            ]
            res = minimize(min_vol_fn, w0, method='SLSQP', bounds=bounds, constraints=cons)
            vol_fe.append(res.fun if res.success else np.nan)
        vol_fe = np.array(vol_fe)

    # ── Simulación Monte Carlo ─────────────────────────────────────────
    with st.spinner"🎲 Simulando 50 000 portafolios aleatorios..."):
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

    # ── Construir diccionario de portfolios ────────────────────────────
    portfolios = {
        'Máx. Sharpe':     w_sharpe,
        'Mín. Volatilidad':w_vol,
        'Equiponderado':   w_eq,
    }
    if w_obj_arr is not None:
        portfolios[obj_label] = w_obj_arr

    # ── Cartera personalizada (si se ingresaron pesos) ─────────────────
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
                    st.info(fℹ️ Activos sin peso especificado asignados a 0%: {', '.join(sin_peso)}")
            else:
                st.warning("⚠️ Los pesos indicados suman 0. Se omite la cartera personalizada.")
        else:
            st.warning("⚠️ Ningún ticker con peso coincide con los activos descargados. Se omite la cartera personalizada.")

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
            'Máx. Drawdown': calc_max_drawdown(ser),
            'CAGR':          calc_cagr(cum_port[name]),
        }

    # ── Guardar resultados en session state para persistencia ─────────
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

# ─────────────────────────────────────────────────────────────────────
#  SECCIÓN DE RESULTADS  (persiste entre reruns — fuera del if run_button)
# ─────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────
    #  SECCIÓN 1 — MÉTRICAS PRINCIPALES
    # ─────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Resumen de Portfolios Óptimos")

    for i, (name, m) in enumerate(metricas.items()):
        color = PORT_COLORS[i % len(PORT_COLORS)]
        st.markdown(
            f'<div class="port-card" style="border-left:4px solid {color}; background:#282828;">'
            f'<strong style="color:{color}; font-size:1.05rem;">📁 {name}</strong></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(6)
        cols[0].metric("Retorno Anual",  pct(m['Retorno Anual']))
        cols[1].metric("Volatilidad",    pct(m['Volatilidad']))
        cols[2].metric("Sharpe Ratio",   f"{m['Sharpe Ratio']:.3f}")
        cols[3].metric("Sortino Ratio",  f"{m['Sortino Ratio']:.3f}")
        cols[4].metric("CAGR",           pct(m['CAGR']))
        cols[5].metric("Máx. Drawdown",  pct(m['Máx. Drawdown']))

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    #  SECCIÓN 2 — GRÁFICOS EN TABS
    # ─────────────────────────────────────────────────────────────────
    st.markdown("## 📈 Análisis Gráfico")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🌐 Espacio Markowitz",
        "🥧 Composición",
        "📈 Portfolios",
        "📈 Activos",
        "📊 Métricas",
        "🔥 Correlación",
        "📊 CAGR",
        "📚 Educativo",
    ])

    # ── Tab 1 — Espacio de Markowitz ──────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">🌐 ¿Qué es el Espacio de Markowitz?</strong><br/>
            Este gráfico muestra el universo de todos los portafolios posibles formados con los activos seleccionados.
            Cada punto representa una combinación de pesos distinta. El color indica el <strong>Sharpe Ratio</strong>
            (relación retorno/riesgo). La <strong>Curva de Frontera Eficiente</strong> delimita los portafolios que
            maximizan el retorno para cada nivel de riesgo. La <strong>Línea de Mercado de Capitales (CML)</strong>
            parte de la tasa libre de riesgo y toca la frontera en el punto de máximo Sharpe.
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

        # Frontera eficiente — línea gruesa y destacada
        valid = ~np.isnan(vol_fe)
        ax.plot(vol_fe[valid] * 100, ret_range[valid] * 100,
                color=BDI_TEAL, linewidth=4.5, zorder=6,
                label='Frontera Eficiente', solid_capstyle='round')

        # CML — solo en rango relevante (cerca de los datos)
        vol_cml_range = np.linspace(vol_min * 0.7, vol_sharpe * 1.3, 80)
        cml_line = rf * 100 + sharpe_val * vol_cml_range * 100
        ax.plot(vol_cml_range * 100, cml_line, '--', color=BDI_LIME, linewidth=2,
                alpha=0.9, zorder=4, label='CML (Línea de Mercado de Capitales)')

        # Marcadores de portfolios — más grandes y con anotaciones
        portfolios_plot = [
            (vol_sharpe, ret_sharpe, '*',  500, BDI_LIME,    f'Máx. Sharpe\n({sharpe_val:.2f})'),
            (vol_min,    ret_vol,    'D',  200, '#ef5350',   f'Mín. Vol.\n({pct(vol_min)})'),
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

        # Calcular offsets para anotaciones automáticas
        x_range = (sim_vol_arr.max() - sim_vol_arr.min()) * 100
        y_range = (sim_ret_arr.max() - sim_ret_arr.min()) * 100

        for i, (vol_p, ret_p, marker, sz, color, label) in enumerate(portfolios_plot):
            ax.scatter(vol_p * 100, ret_p * 100,
                       marker=marker, s=sz, color=color, zorder=10,
                       edgecolors='white', linewidth=1.2)
            # Anotación con flecha
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

        ax.set_title('Espacio de Portfolios — Modelo de Markowitz',
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

    # ── Tab 2 — Composición (Donut charts) ────────────────────────────
    with tab2:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">🥧 Composición de los portafolios óptimos</strong><br/>
            Cada gráfico de dona muestra la asignación de pesos recomendada para cada estrategia de optimización.
            El número central indica el <strong>retorno anual esperado</strong>. Debajo se muestran el
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
                at.set_fontsize(12)      # agrandado de 8 → 12
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
                      fontsize=10, framealpha=0.75, ncol=2)  # agrandado de 8 → 10
            ax.set_title(name, fontsize=12, fontweight='bold', color=BDI_CREAM, pad=14)

        fig.suptitle('Composición de Portfolios Óptimos',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, y=1.02)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### 📋 Tabla de pesos por activo")
        df_pesos = pd.DataFrame(
            {name: [f"{w[i]*100:.1f}%" for i in range(num_assets)]
             for name, w in portfolios.items()},
            index=assets,
        )
        st.dataframe(df_pesos, use_container_width=True)

    # ── Tab 3 — Rendimiento acumulado: portfolios ──────────────────────
    with tab3:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">📈 Rendimiento acumulado de los portafolios</strong><br/>
            Este gráfico muestra cómo habría evolucionado una inversión inicial en cada portafolio a lo largo
            del período histórico analizado. Permite comparar visualmente la performance relativa de cada
            estrategia de optimización en distintos contextos de mercado (subas, bajas, lateralizaciones).
            El valor final de cada línea se indica al extremo derecho del gráfico.
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
        ax.set_title('Rendimiento Acumulado — Portfolios Óptimos',
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

    # ── Tab 4 — Rendimiento acumulado: activos individuales ───────────
    with tab4:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">📈 Rendimiento acumulado por activo individual</strong><br/>
            Visualizá cómo se desempeñó cada activo de forma independiente durante el período analizado.
            Esto permite identificar qué instrumentos lideraron el crecimiento, cuáles tuvieron mayor
            volatilidad y cómo interactuaron entre sí. La diversificación busca combinarlos para suavizar
            las caídas sin sacrificar retorno.
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
        ax.set_title('Rendimiento Acumulado — Activos Individuales',
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

    # ── Tab 5 — Métricas comparativas (barras) ────────────────────────
    with tab5:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">📊 Comparación de métricas entre portafolios</strong><br/>
            Los cuatro paneles comparan las métricas clave de cada estrategia: <strong>Retorno Anual</strong>
            (cuánto creció en promedio), <strong>Volatilidad</strong> (nivel de riesgo o fluctuación),
            <strong>Sharpe Ratio</strong> (retorno ajustado por riesgo; mayor es mejor) y
            <strong>Máximo Drawdown</strong> (peor caída desde un pico; menos negativo es mejor).
            El borde dorado resalta el mejor portafolio en cada categoría.
        </div>
        """, unsafe_allow_html=True)

        port_names    = list(portfolios.keys())
        metricas_plot = [
            ('Retorno Anual',  'Retorno Anual (%)',     True),
            ('Volatilidad',    'Volatilidad Anual (%)', False),
            ('Sharpe Ratio',   'Sharpe Ratio',          True),
            ('Máx. Drawdown',  'Máximo Drawdown (%)',   False),
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

        fig.suptitle('Comparación de Métricas — Portfolios Óptimos',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, y=1.01)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### 📋 Tabla de métricas")
        df_met = pd.DataFrame(metricas).T
        fmt_pct   = ['Retorno Anual', 'Volatilidad', 'Máx. Drawdown', 'CAGR']
        fmt_ratio = ['Sharpe Ratio', 'Sortino Ratio']
        df_disp   = df_met.copy()
        for c in fmt_pct:
            df_disp[c] = df_disp[c].apply(pct)
        for c in fmt_ratio:
            df_disp[c] = df_disp[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_disp, use_container_width=True)

    # ── Tab 6 — Matriz de correlación ─────────────────────────────────
    with tab6:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">🔥 Matriz de correlación entre activos</strong><br/>
            Muestra el grado de movimiento conjunto entre pares de activos. Un valor cercano a
            <strong>+1</strong> (rojo oscuro) indica que los activos suben y bajan juntos, lo que reduce
            el beneficio de diversificación. Un valor cercano a <strong>0</strong> (rojo claro) indica
            baja correlación, lo que es ideal para reducir el riesgo total del portafolio.
            Buscar activos con baja correlación entre sí es la clave del modelo de Markowitz.
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(max(8, num_assets + 2), max(6, num_assets + 1)))

        # Mapa de color rojo: claro = baja correlación, oscuro = alta correlación
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
            cbar_kws={'shrink': 0.8, 'label': 'Correlación', 'pad': 0.02},
        )

        # Texto adaptivo: oscuro sobre celdas claras, claro sobre celdas oscuras
        for text_obj in ax.texts:
            try:
                val = float(text_obj.get_text())
                text_obj.set_color('#1c1c1c' if val < 0.55 else '#EFEDEA')
            except Exception:
                pass

        ax.set_title('Matriz de Correlación entre Activos',
                     fontsize=15, fontweight='bold', color=BDI_CREAM, pad=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=10, color=BDI_CREAM)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=10, color=BDI_CREAM)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 7 — CAGR comparativo ──────────────────────────────────────
    with tab7:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">📊 CAGR — Tasa de Crecimiento Anual Compuesta</strong><br/>
            El CAGR representa la tasa a la que una inversión habría crecido año a año de forma constante
            para llegar al valor final observado. A diferencia del retorno simple, el CAGR toma en cuenta
            el efecto del interés compuesto. Es la métrica más adecuada para comparar el crecimiento
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
            return 'Portfolio'
        all_types    = ['Activo'] * len(cagr_activos) + [_port_type(p) for p in cagr_ports]
        type_color   = {'Activo': BDI_TEAL, 'Portfolio': BDI_LIME, 'Personalizada': '#ab47bc'}

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
        ax.set_title('CAGR Anual Comparativo — Activos y Portfolios',
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

    # ── Tab 8 — Módulo Educativo ──────────────────────────────────────
    with tab8:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">📚 Módulo Educativo — Fundamentos del Análisis de Cartera</strong><br/>
            Esta sección explica las fórmulas matemáticas que utiliza el optimizador y cómo interpretar cada resultado.
            Podés usarla como referencia para entender qué significa cada número y cómo tomar mejores decisiones.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔬 1. Modelo de Markowitz — Teoría Moderna de Portfolio", expanded=True):
            st.markdown("""
            **¿Qué es?** La Teoría Moderna de Portfolio (Harry Markowitz, 1952) sostiene que el riesgo y el retorno
            de una cartera dependen no solo de los activos individuales, sino de la **correlación entre ellos**.
            La clave es que combinando activos poco correlacionados se puede reducir el riesgo **sin sacrificar retorno**.

            **Retorno esperado del portfolio:**
            """)
            st.latex(r"E(R_p) = \sum_{i=1}^{n} w_i \cdot E(R_i)")
            st.markdown("Donde $w_i$ es el peso del activo $i$ y $E(R_i)$ su retorno anualizado histórico.")

            st.markdown("**Varianza (riesgo cuadrático) del portfolio:**")
            st.latex(r"\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w} = \sum_{i=1}^{n}\sum_{j=1}^{n} w_i \cdot w_j \cdot \sigma_{ij}")
            st.markdown("""
            Donde $\Sigma$ es la **matriz de covarianza** y $\sigma_{ij}$ la covarianza entre los activos $i$ y $j$.

            **Volatilidad (riesgo) del portfolio:**
            """)
            st.latex(r"\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}")
            st.markdown("""
            💡 **Beneficio de la diversificación:** Si dos activos no están perfectamente correlacionados
            ($\\rho_{ij} < 1$), la volatilidad del portfolio es **menor** que el promedio ponderado de las
            volatilidades individuales. Eso es el "free lunch" de invertir: reducir riesgo sin sacrificar retorno.
            """)

        with st.expander("📈 2. Frontera Eficiente y Problema de Optimización"):
            st.markdown("""
            **Frontera Eficiente:** Conjunto de portfolios que maximizan el retorno esperado
            para cada nivel de riesgo. Todo portfolio *debajo* o *a la derecha* de la frontera es subóptimo
            — existe un portfolio mejor con el mismo riesgo o menor riesgo con igual retorno.

            **Problema de optimización (mínima volatilidad para retorno objetivo $R^*$):**
            """)
            st.latex(r"\min_{\mathbf{w}} \quad \sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}")
            st.latex(r"\text{sujeto a:} \quad \sum_{i=1}^{n} w_i = 1 \quad \text{(pesos suman 1)}")
            st.latex(r"\quad\quad\quad\quad \mathbf{w}^T \boldsymbol{\mu} = R^* \quad \text{(retorno objetivo)}")
            st.latex(r"\quad\quad\quad\quad w_i \geq 0 \quad \text{(sin posiciones cortas)}")
            st.markdown("""
            El optimizador usa **SLSQP** (Sequential Least Squares Programming), resolviendo este problema
            para 100 valores distintos de $R^*$ entre el mínimo y el máximo retorno posible.
            Cada solución es un punto de la frontera eficiente.

            💡 **Qué muestra el gráfico:** La nube de puntos son 50,000 portfolios aleatorios (Monte Carlo).
            La curva de color turquesa es la frontera eficiente. Cualquier punto a la izquierda o arriba de ella
            es **inalcanzable** con los activos disponibles.
            """)

        with st.expander("⭐ 3. Sharpe Ratio — Retorno ajustado por riesgo"):
            st.markdown("""
            Desarrollado por William Sharpe (1966), mide cuánto retorno extra obtenemos por cada unidad de
            riesgo asumido, en relación a una inversión libre de riesgo (ej: bono del Tesoro).
            """)
            st.latex(r"S = \frac{E(R_p) - R_f}{\sigma_p}")
            st.markdown("""
            Donde $R_f$ es la tasa libre de riesgo (configurada en el panel lateral).

            **Interpretación práctica:**

            | Sharpe | Evaluación |
            |--------|-----------|
            | < 0    | Peor que invertir en el activo libre de riesgo |
            | 0 – 0.5 | Aceptable |
            | 0.5 – 1 | Bueno |
            | > 1    | Excelente |
            | > 2    | Muy difícil de sostener en el tiempo |

            💡 El **Portfolio de Máximo Sharpe** (estrella ★ en el gráfico) es el punto donde la
            **Línea de Mercado de Capitales (CML)** es tangente a la frontera eficiente.
            Es el portafolio "racionalmente óptimo" según la teoría.
            """)

        with st.expander("📉 4. Sortino Ratio — Penaliza solo la volatilidad negativa"):
            st.markdown("""
            Variante del Sharpe que distingue entre volatilidad "buena" (hacia arriba) y "mala" (hacia abajo),
            usando solo el **downside risk** en el denominador.
            """)
            st.latex(r"Sortino = \frac{E(R_p) - R_f}{\sigma_{down}}")
            st.latex(r"\sigma_{down} = \sqrt{\frac{\sum_{t:\, R_t < R_f}(R_t - R_f)^2}{T}} \times \sqrt{252}")
            st.markdown("""
            💡 **Cuándo es más útil que el Sharpe:** Si los retornos del portfolio tienen **asimetría positiva**
            (subas grandes, bajas pequeñas), el Sortino será mayor que el Sharpe. Un Sortino > Sharpe indica
            que la volatilidad total está sesgada hacia el lado positivo — buena señal.
            """)

        with st.expander("📊 5. CAGR — Tasa de Crecimiento Anual Compuesta"):
            st.markdown("""
            El CAGR representa la tasa constante anual que llevaría una inversión de su valor inicial
            al valor final observado, incorporando el efecto del **interés compuesto**.
            """)
            st.latex(r"CAGR = \left(\frac{V_f}{V_i}\right)^{\frac{1}{n}} - 1")
            st.markdown("""
            Donde $n$ es el número de años del período analizado.

            **¿Por qué el CAGR es mejor que el retorno promedio?**

            Ejemplo: Un activo sube +100% el año 1 y cae -50% el año 2.
            """)
            st.latex(r"\text{Retorno promedio} = \frac{+100\% + (-50\%)}{2} = +25\% \quad \text{(engañoso)}")
            st.latex(r"CAGR = \sqrt{2 \times 0.5} - 1 = 0\% \quad \text{(refleja la realidad)}")
            st.markdown("💡 El CAGR siempre es ≤ al promedio aritmético. La diferencia entre ambos crece con la volatilidad.")

        with st.expander("📉 6. Máximo Drawdown — Peor caída desde un máximo histórico"):
            st.markdown("""
            Mide la mayor pérdida porcentual sufrida entre un pico y el valle subsiguiente
            en el período analizado. Es la métrica del **peor escenario**.
            """)
            st.latex(r"MDD = \min_{t} \frac{V_t - \max_{\tau \leq t} V_\tau}{\max_{\tau \leq t} V_\tau}")
            st.markdown("""
            💡 **Cómo usarlo:** Un MDD de -35% significa que en algún momento del período, el inversor
            habría visto su cartera caer un 35% desde su máximo anterior. Es la pregunta clave:
            *¿Podría aguantar esa caída sin vender?* Si la respuesta es no, el portfolio tiene
            demasiado riesgo para tu perfil.
            """)

        with st.expander("🎲 7. Simulación Monte Carlo"):
            st.markdown("""
            Para construir la nube de puntos del espacio de Markowitz, se generan **50,000 portfolios aleatorios**
            usando la distribución de Dirichlet (que garantiza pesos positivos que suman 1):
            """)
            st.latex(r"\mathbf{w} \sim \mathrm{Dir}(\mathbf{1}) \implies w_i \geq 0,\; \sum_i w_i = 1")
            st.markdown("""
            Para cada portfolio aleatorio se calculan retorno, volatilidad y Sharpe:
            """)
            st.latex(r"(E(R_p),\; \sigma_p,\; S_p) \quad \forall\; \mathbf{w} \text{ simulado}")
            st.markdown("""
            El color de cada punto en el gráfico indica el **Sharpe Ratio** (más amarillo = mejor).
            La frontera eficiente aparece como el "borde superior izquierdo" de esta nube —
            es el límite de lo que es alcanzable con los activos disponibles.
            """)

        with st.expander("🗺️ 8. Cómo elegir el portfolio según tu perfil"):
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.markdown("""
                <div class="info-box">
                    <strong style="color:#ef5350;">🔴 Perfil Conservador</strong><br/><br/>
                    Elegí el <strong>PortFolio de Mínima Volatilidad</strong>.<br/><br/>
                    Menos riesgo de caídas · Menor Drawdown histórico · Sacrifica algo de retorno ·
                    Ideal para horizontes cortos o alta aversión al riesgo.
                </div>
                """, unsafe_allow_html=True)
            with col_e2:
                st.markdown("""
                <div class="info-box">
                    <strong style="color:#B5E61D;">🟡 Perfil Moderado</strong><br/><br/>
                    Elegí el <strong>Portfolio de Máximo Sharpe</strong>.<br/><br/>
                    Mejor relación retorno/riesgo · Balance entre crecimiento y estabilidad ·
                    El "óptimo racional" de Markowitz · Ideal para horizontes de 3–5 años.
                </div>
                """, unsafe_allow_html=True)
            with col_e3:
                st.markdown("""
                <div class="info-box">
                    <strong style="color:#17BEBB;">🔵 Perfil Agresivo</strong><br/><br/>
                    Usá <strong>Retorno Objetivo o Cartera Personalizada</strong>.<br/><br/>
                    Mayor exposición al crecimiento · Mayor volatilidad esperada ·
                    Mayor potencial de retorno · Ideal para horizontes mayores a 7 años.
                </div>
                """, unsafe_allow_html=True)
            st.markdown("""
            <div class="legal-warning" style="margin-top:1rem;">
                ⚠️ <strong>Limitación del modelo:</strong> Markowitz se basa en retornos históricos y
                asume que las correlaciones son estables. En períodos de crisis, las correlaciones
                aumentan (los activos "caen juntos"), reduciendo el beneficio de la diversificación
                exactamente cuando más se necesita. Usá estos resultados como guía, no como verdad absoluta.
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    #  REPORTE PDF DESCARGABLE
    # ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📥 Reporte PDF Descargable")

    pdf_col_l, pdf_col_r = st.columns([1.4, 1])

    with pdf_col_l:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#B5E61D;">📄 Informe ejecutivo con marca BDI</strong><br/>
            Generá un PDF profesional con todos los análisis, listo para presentar a clientes
            o guardar como respaldo del estudio.<br/><br/>
            <strong style="color:#17BEBB;">El reporte incluye (8 páginas):</strong><br/>
            &nbsp;·&nbsp; Portada BDI con resumen de métricas<br/>
            &nbsp;·&nbsp; Espacio de Markowitz con Frontera Eficiente y CML<br/>
            &nbsp;·&nbsp; Composición de portfolios (gráficos de dona)<br/>
            &nbsp;·&nbsp; Rendimiento acumulado — portfolios y activos<br/>
            &nbsp;·&nbsp; Comparación de métricas (4 paneles)<br/>
            &nbsp;·&nbsp; Matriz de correlación entre activos<br/>
            &nbsp;·&nbsp; CAGR comparativo anual<br/>
            &nbsp;·&nbsp; Aviso legal
        </div>
        """, unsafe_allow_html=True)

    with pdf_col_r:
        st.markdown("<br/>", unsafe_allow_html=True)
        cliente_nombre = st.text_input(
            "👤 Nombre del cliente (opcional)",
            value="",
            key="cliente_pdf",
            placeholder="Ej: Juan Pérez — Perfil Moderado",
        )

        if st.button("📄 Generar Reporte PDF", use_container_width=True, key="btn_gen_pdf"):
            with st.spinner("📄 Generando reporte profesional BDI..."):
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
    ;           )
                st.session_state['pdf_bytes'] = pdf_bytes
                st.session_state['pdf_ready'] = True

        if st.session_state.get('pdf_ready'):
            fname = f"BDI_Cartera_{datetime.now().strftime('%Y%m%d')}.pdf"
            st.download_button(
                label="⬇️ Descargar Reporte PDF",
                data=st.session_state['pdf_bytes'],
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )
            st.success(f"✅ PDF listo · {fname}")

    # ─────────────────────────────────────────────────────────────────
    #  ADVERTENCIA LEGAL
    # ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="legal-warning">
        ⚠️ <strong>ADVERTENCIA LEGAL:</strong>
        Este análisis es de carácter exclusivamente informativo y educativo.
        No constituye asesoramiento financiero ni una recomendación de inversión.
        Los rendimientos pasados no garantizan resultados futuros.<br/>
        <em>BDI Consultora de Inversiones · bdiconsultora@gmail.com · Mariano Ricciardi</em>
    </div>
    """, unsafe_allow_html=True)

    # (results_ready ya fue guardado en el bloque de cómputo)
