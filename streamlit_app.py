import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

SCALE_FACTOR = 1_000  # El archivo está en miles de USD

st.set_page_config(
    page_title='Análisis de Portafolio',
    page_icon=':bar_chart:',
    layout='wide',
)

# Archivo CSV (AJUSTA ESTA RUTA CUANDO LO SUBAS A STREAMLIT CLOUD)
PORTFOLIO_FILE = (
    Path(__file__).parent
    / 'data'
    / 'Corporate Data for GCP 31Aug25 - CBA Advisory-Restricted name_V2.csv'
)

# ============================================
# CARGA Y LIMPIEZA DE DATOS
# ============================================

@st.cache_data
def load_portfolio_data(file_path: Path, last_modified: float):
    """
    Lee y limpia el dataset del portafolio corporativo.
    """

    df = pd.read_csv(file_path)

    # 1. LIMPIAR NOMBRES DE COLUMNAS
    df.columns = [col.strip() for col in df.columns]

    # 2. Nombre REAL correcto de la columna de monto:
    amount_col = 'US $ Equiv'

    # 3. LIMPIAR Y CONVERTIR MONTOS
    df[amount_col] = (
        df[amount_col]
        .astype(str)
        .str.replace('[^0-9.-]', '', regex=True)
        .replace('', pd.NA)
    )
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
    df[amount_col] = df[amount_col] * SCALE_FACTOR

    # 4. Normalizar dimensiones categóricas
    for col in ['Country', 'Segment', 'Product Type', 'Sector', 'Sector 2']:
        if col in df.columns:
            df[col] = df[col].fillna('Sin especificar')

    return df

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def format_currency(value: float) -> str:
    return f"{value:,.0f}"

def concentration_ratio(df, column, top=10):
    if df.empty: return 0
    grouped = df.groupby(column)['US $ Equiv'].sum().sort_values(ascending=False)
    return grouped.head(top).sum() / grouped.sum()

def hhi(df, column):
    if df.empty: return 0
    g = df.groupby(column)['US $ Equiv'].sum()
    s = g / g.sum()
    return float((s**2).sum() * 10_000)

# ============================================
# RENDER: KPIs
# ============================================

def render_kpis(df):
    total = df['US $ Equiv'].sum()
    n = len(df)
    avg = df['US $ Equiv'].mean() if n else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Exposición total (US$)", format_currency(total))
    c2.metric("Número de registros", f"{n:,}")
    c3.metric("Ticket promedio (US$)", format_currency(avg))

# ============================================
# RENDER: CONCENTRACIÓN
# ============================================

def render_concentration(df):
    st.subheader("Indicadores de concentración")

    dims = [
        ('Country', 'País'),
        ('Sector', 'Sector'),
        ('Segment', 'Segmento'),
    ]

    for col, label in dims:
        if col not in df.columns:
            continue

        with st.expander(f"Concentración por {label}"):
            h = hhi(df, col)
            cr3 = concentration_ratio(df, col, top=3) * 100
            cr10 = concentration_ratio(df, col, top=10) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("HHI", f"{h:,.0f}")
            c2.metric("CR3", f"{cr3:.1f}%")
            c3.metric("CR10", f"{cr10:.1f}%")

# ============================================
# RENDER: BREAKDOWNS (BARRAS + PIE)
# ============================================

def render_breakdown(df, column, title, label, include_pie=True):
    if column not in df.columns:
        return

    st.subheader(title)

    options = sorted(df[column].unique())
    col1, col2 = st.columns(2)

    # ----- BARRAS -----
    with col1:
        sel = st.multiselect(f"{label} visibles", options, default=options)
        selection = sel if sel else options
        df_f = df[df[column].isin(selection)]

        g = df_f.groupby(column)['US $ Equiv'].sum().reset_index()
        total = g['US $ Equiv'].sum()
        g['Porcentaje'] = g['US $ Equiv'] / total

        chart = (
            alt.Chart(g)
            .mark_bar(color="#2563eb")
            .encode(
                x=alt.X("US $ Equiv:Q", title="Exposición (US$)"),
                y=alt.Y(f"{column}:N", sort="-x", title=label),
                tooltip=[
                    f"{column}:N",
                    alt.Tooltip("US $ Equiv:Q", format=",.0f"),
                    alt.Tooltip("Porcentaje:Q", format=".1%"),
                ]
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # ----- PIE -----
    if include_pie:
        with col2:
            sel2 = st.multiselect(f"{label} (Pie)", options, default=options)
            selection2 = sel2 if sel2 else options
            df_p = df[df[column].isin(selection2)]

            g2 = df_p.groupby(column)['US $ Equiv'].sum().reset_index()
            g2['Porcentaje'] = g2['US $ Equiv'] / g2['US $ Equiv'].sum()

            pie = (
                alt.Chart(g2)
                .mark_arc()
                .encode(
                    theta="US $ Equiv:Q",
                    color=f"{column}:N",
                    tooltip=[
                        f"{column}:N",
                        alt.Tooltip("US $ Equiv:Q", format=",.0f"),
                        alt.Tooltip("Porcentaje:Q", format=".1%"),
                    ],
                )
            )
            st.altair_chart(pie, use_container_width=True)

# ============================================
# RENDER: HEATMAP
# ============================================

def render_heatmap(df):
    if not {'Country', 'Sector'}.issubset(df.columns):
        return

    st.subheader("Mapa de calor País vs Sector")

    g = df.groupby(['Country', 'Sector'])['US $ Equiv'].sum().reset_index()

    heat = (
        alt.Chart(g)
        .mark_rect()
        .encode(
            x=alt.X("Sector:N"),
            y=alt.Y("Country:N"),
            color=alt.Color("US $ Equiv:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Country", "Sector", alt.Tooltip("US $ Equiv:Q", format=",.0f")],
        )
        .properties(height=400)
    )
    st.altair_chart(heat, use_container_width=True)

# ============================================
# RENDER: TOP / BOTTOM
# ============================================

def render_top_bottom(df, n=10):
    st.subheader("Top / Bottom operaciones")

    amt = "US $ Equiv"
    ordered = df.sort_values(amt, ascending=False)

    top = ordered.head(n)
    bottom = ordered[ordered[amt] > 0].tail(n)

    for d in (top, bottom):
        d[amt] = d[amt].apply(format_currency)

    c1, c2 = st.columns(2)
    c1.write(f"### Top {n}")
    c1.dataframe(top, use_container_width=True)

    c2.write(f"### Bottom {n}")
    c2.dataframe(bottom, use_container_width=True)

# ============================================
# APP PRINCIPAL
# ============================================

df = load_portfolio_data(
    PORTFOLIO_FILE,
    PORTFOLIO_FILE.stat().st_mtime,
)

st.title("Análisis del Portafolio Corporativo")
st.caption("Filtros dinámicos, concentración, KPIs y desglose por dimensiones.")

# ------------------------------
# FILTROS LATERALES
# ------------------------------

fdf = df.copy()

if "Country" in fdf.columns:
    countries = sorted(fdf["Country"].unique())
    sel = st.sidebar.multiselect("País", countries, default=countries)
    fdf = fdf[fdf["Country"].isin(sel)]

if "Segment" in fdf.columns:
    seg = sorted(fdf["Segment"].unique())
    sel = st.sidebar.multiselect("Segmento", seg, default=seg)
    fdf = fdf[fdf["Segment"].isin(sel)]

if "Product Type" in fdf.columns:
    prod = sorted(fdf["Product Type"].unique())
    sel = st.sidebar.multiselect("Tipo de producto", prod, default=prod)
    fdf = fdf[fdf["Product Type"].isin(sel)]

if "Sector" in fdf.columns:
    sec = sorted(fdf["Sector"].unique())
    sel = st.sidebar.multiselect("Sector", sec, default=sec)
    fdf = fdf[fdf["Sector"].isin(sel)]

if fdf.empty:
    st.warning("No hay datos para mostrar con estos filtros.")
    st.stop()

# ------------------------------
# SECCIONES
# ------------------------------

st.header("Resumen General")
render_kpis(fdf)

st.divider()

render_concentration(fdf)

st.divider()

st.header("Desglose por Dimensiones")

render_breakdown(fdf, "Country", "Exposición por país", "País")
render_breakdown(fdf, "Segment", "Exposición por segmento", "Segmento")
render_breakdown(fdf, "Product Type", "Exposición por tipo de producto", "Tipo de producto")
render_breakdown(fdf, "Sector", "Exposición por sector", "Sector")
render_breakdown(fdf, "Sector 2", "Exposición por Sector 2", "Sector 2")

st.divider()

render_heatmap(fdf)

st.divider()

render_top_bottom(fdf)

st.divider()

st.header("Detalle Completo")
st.dataframe(fdf, use_container_width=True)
