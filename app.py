# ============================================================
#  OPTIMIZADOR DE CARTERA PROFESIONAL — STREAMLIT
#  BDI Consultora de Inversiones
#  Versión 2.0 — Modelo de Markowitz con Frontera Eficiente
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG  (debe ir primero)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BDI — Optimizador de Carteras",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  PALETA CORPORATIVA BDI
# ─────────────────────────────────────────────
BDI_DARK      = '#0a1628'
BDI_NAVY      = '#1a3a5c'
BDI_BLUE      = '#1565c0'
BDI_LIGHTBLUE = '#90caf9'
BDI_GOLD      = '#f0c040'
BDI_GOLD2     = '#ffa000'
BDI_RED       = '#e53935'
BDI_GREEN     = '#43a047'
BDI_TEAL      = '#00897b'
BDI_PURPLE    = '#8e24aa'
BDI_GREY      = '#37474f'
BDI_LIGHTGREY = '#b0bec5'

PALETTE = [
    BDI_GOLD, BDI_LIGHTBLUE, BDI_GREEN, BDI_RED,
    BDI_TEAL, BDI_PURPLE, BDI_GOLD2, '#ef5350',
    '#26c6da', '#d4e157', '#ff7043', '#ab47bc',
]

PORT_COLORS = [BDI_GOLD, BDI_RED, BDI_TEAL, BDI_GREEN]

plt.rcParams.update({
    'figure.facecolor': BDI_DARK,
    'axes.facecolor':   '#0d1f35',
    'axes.edgecolor':   '#2a4a6b',
    'axes.labelcolor':  BDI_LIGHTGREY,
    'axes.titlecolor':  'white',
    'xtick.color':      BDI_LIGHTGREY,
    'ytick.color':      BDI_LIGHTGREY,
    'text.color':       'white',
    'grid.color':       '#1e3a50',
    'grid.linewidth':   0.6,
    'legend.facecolor': '#0d1f35',
    'legend.edgecolor': '#2a4a6b',
    'legend.labelcolor':'white',
    'font.family':      'DejaVu Sans',
})

# ─────────────────────────────────────────────
#  CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #0a1628; color: #ffffff; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1f35;
        border-right: 1px solid #2a4a6b;
    }

    /* Títulos */
    h1, h2, h3 { color: #f0c040 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #0d1f35;
        border: 1px solid #2a4a6b;
        border-radius: 8px;
        padding: 10px 14px;
    }
    [data-testid="stMetricValue"] {
        color: #f0c040 !important;
        font-size: 1.3rem !important;
    }
    [data-testid="stMetricLabel"] { color: #90caf9 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1f35;
        border-bottom: 2px solid #2a4a6b;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #90caf9;
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #f0c040 !important;
        border-bottom: 2px solid #f0c040 !important;
        background-color: #112038 !important;
    }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0a47a0);
        color: white;
        border: 1px solid #90caf9;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #f0c040, #ffa000);
        color: #0a1628 !important;
        border-color: #f0c040;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #0d1f35 !important;
        color: white !important;
        border: 1px solid #2a4a6b !important;
        border-radius: 6px !important;
    }

    /* Slider */
    [data-testid="stSlider"] > div { color: #90caf9; }

    /* Checkbox */
    [data-testid="stCheckbox"] { color: #b0bec5; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #2a4a6b; border-radius: 8px; }

    /* Divisor */
    hr { border-color: #2a4a6b; }

    /* Banner header */
    .bdi-header {
        background: linear-gradient(135deg, #0d1f35 0%, #1a3a5c 60%, #0d1f35 100%);
        border: 1px solid #2a4a6b;
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Card de portfolio */
    .port-card {
        border-radius: 8px;
        padding: 0.7rem 1.4rem;
        margin: 0.7rem 0 0.3rem 0;
    }

    /* Advertencia legal */
    .legal-warning {
        background-color: #121208;
        border: 1px solid #ffa000;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-top: 2rem;
        color: #b0bec5;
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
             color=BDI_LIGHTBLUE, alpha=0.5, style='italic')
    fig.text(0.01, 0.01, datetime.now().strftime('%d/%m/%Y'),
             ha='left', va='bottom', fontsize=7,
             color=BDI_LIGHTGREY, alpha=0.5)

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
#  HEADER PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class="bdi-header">
    <h1 style="color:#f0c040; margin:0; font-size:2.3rem; letter-spacing:2px;">
        ⚡ BDI — OPTIMIZADOR DE CARTERAS
    </h1>
    <p style="color:#90caf9; margin:0.4rem 0 0 0; font-size:1.05rem; font-weight:500;">
        Modelo de Markowitz con Frontera Eficiente &nbsp;·&nbsp; v2.0
    </p>
    <p style="color:#b0bec5; margin:0.2rem 0 0 0; font-size:0.85rem;">
        BDI Consultora de Inversiones
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR — CONFIGURACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:0.8rem 0 0.4rem 0;">
        <h2 style="color:#f0c040; font-size:1.25rem; margin:0;">⚙️ Configuración</h2>
        <p style="color:#90caf9; font-size:0.82rem; margin:0.2rem 0 0 0;">Parámetros de optimización</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**📌 Activos a analizar**")
    tickers_input = st.text_area(
        "Tickers separados por coma",
        value="JPM, AAPL, MSFT, GOOGL, MELI",
        height=85,
        help="Ejemplos: JPM, AAPL, MSFT, MELI, GOOGL, GGAL.BA, SPY, GLD",
    )

    st.markdown("---")
    st.markdown("**📅 Período de análisis**")
    anios = st.slider("Años de historia", min_value=1, max_value=15, value=5)

    st.markdown("---")
    st.markdown("**📈 Parámetros de mercado**")
    rf_pct = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0, max_value=30.0,
        value=4.5, step=0.1,
        help="Ej: 4.5 equivale al 4.5% anual",
    )
    rf = rf_pct / 100

    st.markdown("---")
    st.markdown("**🎯 Retorno objetivo** *(opcional)*")
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
    st.markdown("**⚖️ Restricciones de pesos**")
    min_peso = st.slider("Peso mínimo por activo (%)", 0, 50, 0) / 100
    max_peso = st.slider("Peso máximo por activo (%)", 10, 100, 100) / 100
    if max_peso < min_peso:
        max_peso = min_peso

    st.markdown("---")
    run_button = st.button("🚀  EJECUTAR ANÁLISIS", type="primary")

# ─────────────────────────────────────────────
#  ESTADO INICIAL (pantalla de bienvenida)
# ─────────────────────────────────────────────
if not run_button and 'results_ready' not in st.session_state:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#b0bec5;">
        <p style="font-size:4.5rem; margin:0;">📊</p>
        <h3 style="color:#90caf9; margin:1rem 0 0.5rem 0;">Listo para optimizar</h3>
        <p style="font-size:1rem; margin:0;">
            Configurá los parámetros en el panel izquierdo<br/>
            y presioná <strong style="color:#f0c040;">🚀 EJECUTAR ANÁLISIS</strong>
        </p>
        <br/>
        <p style="font-size:0.85rem; color:#4a6a8a;">
            Ejemplos de tickers: JPM &nbsp;·&nbsp; AAPL &nbsp;·&nbsp; MSFT &nbsp;·&nbsp;
            GOOGL &nbsp;·&nbsp; MELI &nbsp;·&nbsp; GGAL.BA &nbsp;·&nbsp;
            XOM &nbsp;·&nbsp; GLD &nbsp;·&nbsp; SPY &nbsp;·&nbsp; BRK-B
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  ANÁLISIS PRINCIPAL
# ─────────────────────────────────────────────
if run_button:

    # ── Parse tickers ──────────────────────────────────────────────────
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()][:50]
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
    constraints_base = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    w0               = np.full(num_assets, 1.0 / num_assets)

    def _ps(w):
        return port_stats(w, mean_returns, cov_matrix, rf)

    def neg_sharpe(w): return -_ps(w)[2]
    def min_vol_fn(w): return  _ps(w)[1]

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
    with st.spinner("🎲 Simulando 50 000 portafolios aleatorios..."):
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

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    #  SECCIÓN 1 — MÉTRICAS PRINCIPALES
    # ─────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Resumen de Portfolios Óptimos")

    for i, (name, m) in enumerate(metricas.items()):
        color = PORT_COLORS[i % len(PORT_COLORS)]
        st.markdown(
            f'<div class="port-card" style="border-left:4px solid {color}; background:#0d1f35;">'
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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🌐 Espacio Markowitz",
        "🥧 Composición",
        "📈 Portfolios",
        "📈 Activos",
        "📊 Métricas",
        "🔥 Correlación",
        "📊 CAGR",
    ])

    # ── Tab 1 — Espacio de Markowitz ──────────────────────────────────
    with tab1:
        fig, ax = plt.subplots(figsize=(13, 8))
        sc = ax.scatter(sim_vol_arr * 100, sim_ret_arr * 100,
                        c=sim_sharpe_arr, cmap='plasma',
                        alpha=0.35, s=4, zorder=1)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('Sharpe Ratio', color='white', fontsize=11)
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

        valid = ~np.isnan(vol_fe)
        ax.plot(vol_fe[valid] * 100, ret_range[valid] * 100,
                color=BDI_LIGHTBLUE, linewidth=3, zorder=5, label='Frontera Eficiente')

        vol_cml  = np.linspace(0, np.nanmax(vol_fe) * 130, 100)
        cml_line = rf * 100 + sharpe_val * vol_cml
        ax.plot(vol_cml, cml_line, '--', color=BDI_GOLD, linewidth=1.8,
                alpha=0.8, zorder=4, label='CML (Línea de Mercado de Capitales)')

        ax.scatter(vol_sharpe * 100, ret_sharpe * 100,
                   marker='*', s=450, color=BDI_GOLD, zorder=10,
                   edgecolors='white', linewidth=0.8,
                   label=f'Máx. Sharpe  ({sharpe_val:.2f})')
        ax.scatter(vol_min * 100, ret_vol * 100,
                   marker='D', s=160, color=BDI_RED, zorder=10,
                   edgecolors='white', linewidth=0.8, label='Mín. Volatilidad')
        ax.scatter(vol_eq * 100, ret_eq * 100,
                   marker='s', s=120, color=BDI_TEAL, zorder=10,
                   edgecolors='white', linewidth=0.8, label='Equiponderado')
        if w_obj_arr is not None:
            ax.scatter(vol_obj * 100, ret_obj_real * 100,
                       marker='P', s=200, color=BDI_GREEN, zorder=10,
                       edgecolors='white', linewidth=0.8,
                       label=f'Retorno Obj. ({pct(ret_obj)})')

        ax.axhline(rf * 100, color=BDI_LIGHTGREY, linewidth=0.8,
                   linestyle=':', alpha=0.6, label=f'Rf = {pct(rf)}')
        ax.set_title('Espacio de Portfolios — Modelo de Markowitz',
                     fontsize=16, fontweight='bold', color='white', pad=16)
        ax.set_xlabel('Volatilidad Anual (%)', fontsize=12, labelpad=10)
        ax.set_ylabel('Retorno Anual (%)', fontsize=12, labelpad=10)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=9, framealpha=0.85, loc='upper left')
        ax.grid(True, alpha=0.3)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 2 — Composición (Donut charts) ────────────────────────────
    with tab2:
        n_ports = len(portfolios)
        fig, axes = plt.subplots(1, n_ports, figsize=(5 * n_ports, 6))
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
                wedgeprops={'linewidth': 2, 'edgecolor': BDI_DARK, 'width': 0.6},
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color('white')
                at.set_fontweight('bold')

            m = metricas[name]
            ax.text(0,  0.10, pct(m['Retorno Anual']),
                    ha='center', va='center', fontsize=14,
                    fontweight='bold', color=BDI_GOLD)
            ax.text(0, -0.12, f"Sharpe: {m['Sharpe Ratio']:.2f}",
                    ha='center', va='center', fontsize=9, color=BDI_LIGHTGREY)
            ax.text(0, -0.30, f"Vol: {pct(m['Volatilidad'])}",
                    ha='center', va='center', fontsize=9, color=BDI_LIGHTBLUE)
            ax.legend(wedges, [f"{l} ({s*100:.1f}%)" for l, s in zip(labels, sizes)],
                      loc='lower center', bbox_to_anchor=(0.5, -0.28),
                      fontsize=8, framealpha=0.7, ncol=2)
            ax.set_title(name, fontsize=11, fontweight='bold', color='white', pad=12)

        fig.suptitle('Composición de Portfolios Óptimos',
                     fontsize=15, fontweight='bold', color='white', y=1.02)
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
        fig, ax = plt.subplots(figsize=(13, 6))
        for i, col in enumerate(cum_port.columns):
            ax.plot(cum_port.index, cum_port[col] * 100,
                    label=col, linewidth=2.5,
                    color=PORT_COLORS[i % len(PORT_COLORS)])
        ax.axhline(0, color=BDI_LIGHTGREY, linewidth=0.6, linestyle='--', alpha=0.5)
        ax.fill_between(cum_port.index, 0, cum_port.iloc[:, 0] * 100,
                        alpha=0.06, color=BDI_GOLD)
        for i, (col, val) in enumerate((cum_port.iloc[-1] * 100).items()):
            ax.annotate(f' {val:.1f}%', xy=(cum_port.index[-1], val),
                        fontsize=9, color=PORT_COLORS[i % len(PORT_COLORS)],
                        va='center', fontweight='bold')
        ax.set_title('Rendimiento Acumulado — Portfolios Óptimos',
                     fontsize=15, fontweight='bold', color='white', pad=12)
        ax.set_xlabel('Fecha', fontsize=11, labelpad=8)
        ax.set_ylabel('Rendimiento Acumulado (%)', fontsize=11, labelpad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=10, framealpha=0.85)
        ax.grid(True, alpha=0.3)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 4 — Rendimiento acumulado: activos individuales ───────────
    with tab4:
        fig, ax = plt.subplots(figsize=(13, 6))
        for i, col in enumerate(cum_assets.columns):
            ax.plot(cum_assets.index, cum_assets[col] * 100,
                    label=col, linewidth=1.8,
                    color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax.axhline(0, color=BDI_LIGHTGREY, linewidth=0.6, linestyle='--', alpha=0.5)
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
                     fontsize=15, fontweight='bold', color='white', pad=12)
        ax.set_xlabel('Fecha', fontsize=11, labelpad=8)
        ax.set_ylabel('Rendimiento Acumulado (%)', fontsize=11, labelpad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=9, framealpha=0.8,
                  bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 5 — Métricas comparativas (barras) ────────────────────────
    with tab5:
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
                           edgecolor=BDI_DARK, linewidth=1.2, width=0.5)
            best_idx = np.argmax(vals) if higher_better else np.argmin(vals)
            bars[best_idx].set_edgecolor(BDI_GOLD)
            bars[best_idx].set_linewidth(2.5)
            for bar, val in zip(bars, vals):
                lbl = f'{val*100:.1f}%' if '%' in ylabel else f'{val:.2f}'
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.03,
                    lbl, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='white',
                )
            ax.set_title(ylabel, fontsize=12, fontweight='bold', color='white')
            if '%' in ylabel:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
            ax.set_xticklabels(port_names, rotation=15, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Comparación de Métricas — Portfolios Óptimos',
                     fontsize=15, fontweight='bold', color='white', y=1.01)
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
        fig, ax = plt.subplots(figsize=(max(8, num_assets + 2), max(6, num_assets + 1)))
        cmap_bdi = LinearSegmentedColormap.from_list(
            'BDI', ['#1565c0', '#0d2137', '#b71c1c'], N=256
        )
        sns.heatmap(
            corr_matrix, ax=ax,
            annot=True, fmt='.2f',
            annot_kws={'size': 10, 'color': 'white', 'weight': 'bold'},
            cmap=cmap_bdi, vmin=-1, vmax=1,
            linewidths=0.8, linecolor=BDI_DARK, square=True,
            cbar_kws={'shrink': 0.8, 'label': 'Correlación', 'pad': 0.02},
        )
        ax.set_title('Matriz de Correlación entre Activos',
                     fontsize=15, fontweight='bold', color='white', pad=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=10, color='white')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=10, color='white')
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 7 — CAGR comparativo ──────────────────────────────────────
    with tab7:
        cagr_activos = {a: calc_cagr(cum_assets[a]) * 100 for a in assets}
        cagr_ports   = {p: calc_cagr(cum_port[p])   * 100 for p in port_daily.columns}
        all_names    = list(cagr_activos) + list(cagr_ports)
        all_vals     = list(cagr_activos.values()) + list(cagr_ports.values())
        all_types    = ['Activo'] * len(cagr_activos) + ['PortFolio'] * len(cagr_ports)
        type_color   = {'Activo': BDI_LIGHTBLUE, 'PortFolio': BDI_GOLD}

        sorted_data  = sorted(zip(all_vals, all_names, all_types), reverse=True)
        s_vals, s_names, s_types = zip(*sorted_data)

        fig, ax = plt.subplots(figsize=(max(10, len(all_names) * 1.2), 7))
        bars = ax.bar(s_names, s_vals,
                      color=[type_color[t] for t in s_types],
                      edgecolor=BDI_DARK, linewidth=1.2, width=0.6)
        for bar, val in zip(bars, s_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.5 if val >= 0 else -1.5),
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='white')
        ax.axhline(0, color=BDI_LIGHTGREY, linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_title('CAGR Anual Comparativo — Activos y Portfolios',
                     fontsize=15, fontweight='bold', color='white', pad=12)
        ax.set_ylabel('CAGR Anual (%)', fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.set_xticklabels(s_names, rotation=35, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        legend_elems = [mpatches.Patch(color=c, label=t) for t, c in type_color.items()]
        ax.legend(handles=legend_elems, fontsize=10, framealpha=0.8)
        add_bdi_watermark(fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────
    #  ADVERTENCIA LEGAL
    # ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="legal-warning">
        ⚠️ <strong>ADVERTENCIA LEGAL:</strong>
        Este análisis es de carácter exclusivamente informativo y educativo.
        No constituye asesoramiento financiero ni una recomendación de inversión.
        Los rendimientos pasados no garantizan resultados futuros.<br/>
        <em>BDI Consultora de Inversiones · bdiconsultora@gmail.com</em>
    </div>
    """, unsafe_allow_html=True)

    # Guardar flag para no repetir pantalla de bienvenida
    st.session_state['results_ready'] = True
