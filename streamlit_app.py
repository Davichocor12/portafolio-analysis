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

# Archivo CSV
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
    df = pd.read_csv(file_path)

    # 1. LIMPIAR NOMBRES DE COLUMNAS
    df.columns = [col.strip() for col in df.columns]

    # 2. Renombrar columna EXACTA del archivo
    if "US $ Equiv" not in df.columns:
        df = df.rename(columns={' US $ Equiv ': 'US $ Equiv'})

    # 3. COLUMNA DE MONTO
    df['US $ Equiv'] = (
        df['US $ Equiv']
        .astype(str)
        .str.replace('[^0-9.-]', '', regex=True)
        .replace('', pd.NA)
    )

    df['US $ Equiv'] = pd.to_numeric(df['US $ Equiv'], errors='coerce').fillna(0)
    df['US $ Equiv'] = df['US $ Equiv'] * SCALE_FACTOR

    # 4. Normalizar categorías (SIN NULOS)
    categorical_cols = ['Country', 'Segment', 'Product Type', 'Sector', 'Sector 2']
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().fillna("Sin especificar")
            df[c].replace("nan", "Sin especificar", inplace=True)

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
    st.subheader(title)

    options = sorted(df[column].unique())
    col1, col2 = st.columns(2)

    # ----- BARRAS -----
    with col1:
        sel = st.multiselect(f"{label} visibles", options, default=options)
        vals = sel if sel else options

        df_f = df[df[column].isin(vals)]
        if df_f.empty:
            st.info("No hay datos para mostrar en esta gráfica.")
            return

        g = (
            df_f.groupby(column)['US $ Equiv']
            .sum()
            .reset_index()
            .sort_values('US $ Equiv', ascending=False)
        )

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
                    alt.Tooltip("Porcentaje:Q", format=".1%")
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # ----- PIE -----
    if include_pie:
        with col2:
            sel2 = st.multiselect(f"{label} (Pie)", options, default=options)
            vals2 = sel2 if sel2 else options

            df_p = df[df[column].isin(vals2)]
            if df_p.empty:
                st.info("No hay datos para mostrar en el pie chart.")
                return

            g2 = (
                df_p.groupby(column)['US $ Equiv']
                .sum()
                .reset_index()
            )

            pie = (
                alt.Chart(g2)
                .mark_arc()
                .encode(
                    theta="US $ Equiv:Q",
                    color=f"{column}:N",
                    tooltip=[
                        f"{column}:N",
                        alt.Tooltip("US $ Equiv:Q", format=",.0f")
                    ],
                )
            )
            st.altair_chart(pie, use_container_width=True)


# ============================================
# RENDER: HEATMAP
# ============================================

def render_heatmap(df):
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
        .properties(height=380)
    )

    st.altair_chart(heat, use_container_width=True)

# ============================================
# RENDER: TOP / BOTTOM
# ============================================

def render_top_bottom(df, n=10):
    st.subheader("Top / Bottom operaciones")

    amt = "US $ Equiv"
    ordered = df.sort_values(amt, ascending=False)

    top = ordered.head(n).copy()
    bottom = ordered[ordered[amt] > 0].tail(n).copy()

    if top.empty or bottom.empty:
        st.info("No hay suficientes datos para top/bottom.")
        return

    top[amt] = top[amt].apply(format_currency)
    bottom[amt] = bottom[amt].apply(format_currency)

    c1, c2 = st.columns(2)
    c1.markdown("### Top 10")
    c1.dataframe(top, use_container_width=True)

    c2.markdown("### Bottom 10 (solo valores > 0)")
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

filters = {
    "Country": "País",
    "Segment": "Segmento",
    "Product Type": "Tipo de producto",
    "Sector": "Sector",
}

for col, label in filters.items():
    options = sorted(fdf[col].unique())
    sel = st.sidebar.multiselect(label, options, default=options)
    fdf = fdf[fdf[col].isin(sel)]

if fdf.empty:
    st.warning("No hay datos disponibles con esta combinación de filtros.")
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

