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
    """Cargar el portafolio asegurando compatibilidad de archivos.

    Se intenta leer primero el CSV esperado; si no existe o falla la lectura,
    se busca un XLSX alterno con el mismo nombre. En caso de error se muestra
    un mensaje claro en la app para evitar que quede en blanco.
    """

    def read_csv_safe(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, on_bad_lines="skip")
        except UnicodeDecodeError:
            return pd.read_csv(path, on_bad_lines="skip", encoding="latin-1")

    if file_path.exists():
        try:
            df = read_csv_safe(file_path)
        except Exception:
            df = None
    else:
        df = None

    if df is None:
        alt_file = file_path.with_suffix(".xlsx")
        if alt_file.exists():
            try:
                df = pd.read_excel(alt_file)
            except Exception as e:
                st.error(f"No se pudo leer el archivo de portafolio: {e}")
                st.stop()
        else:
            st.error(
                "No se encontró el archivo de portafolio (CSV o XLSX) en la carpeta data."
            )
            st.stop()

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
            df[c] = df[c].replace("nan", "Sin especificar")

    # 5. ORR numérico para filtros de riesgo
    if 'ORR' in df.columns:
        df['ORR_num'] = pd.to_numeric(df['ORR'], errors='coerce')

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


def render_portfolio_summary(total_full: float, df_filtered: pd.DataFrame):
    """Mostrar comparación entre el portafolio completo y el filtrado actual."""

    filtered_total = df_filtered['US $ Equiv'].sum()
    delta_abs = filtered_total - total_full
    delta_pct = (filtered_total / total_full - 1) * 100 if total_full else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Portafolio total (US$)", format_currency(total_full))
    c2.metric(
        "Exposición filtrada (US$)",
        format_currency(filtered_total),
        delta=f"{delta_abs:,.0f}",
        delta_color="off",
    )
    c3.metric("Variación vs total", f"{delta_pct:.1f}%")

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

        g = g[g['US $ Equiv'] > 0]
        if g.empty:
            st.info("Solo hay valores en cero para esta selección.")
            return

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

            g2 = g2[g2['US $ Equiv'] > 0]
            if g2.empty:
                st.info("Solo hay valores en cero para esta selección.")
                return

            total_pie = g2['US $ Equiv'].sum()
            if total_pie <= 0:
                st.info("No hay valores positivos para el pie chart.")
                return

            g2['Porcentaje'] = g2['US $ Equiv'] / total_pie

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

            labels = (
                alt.Chart(g2)
                .mark_text(radius=120, size=12)
                .encode(
                    theta="US $ Equiv:Q",
                    text=alt.Text("Porcentaje:Q", format=".1%"),
                    color=f"{column}:N",
                )
            )

            st.altair_chart(pie + labels, use_container_width=True)


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

total_portfolio = df['US $ Equiv'].sum()

# ------------------------------
# FILTROS LATERALES
# ------------------------------

# Usar solo exposiciones distintas de cero para los filtros principales
fdf = df[df['US $ Equiv'] != 0].copy()

filters = {
    "Country": "País",
    "Segment": "Segmento",
    "Product Type": "Tipo de producto",
    "Sector": "Sector",
}

with st.sidebar.expander("Filtros (selección múltiple)", expanded=True):
    for col, label in filters.items():
        options = sorted(fdf[col].unique())
        sel = st.multiselect(label, options, default=options)
        fdf = fdf[fdf[col].isin(sel)]

    # Filtro de riesgo por ORR (mayor número = mayor riesgo)
    if 'ORR_num' in fdf.columns:
        orr_vals = fdf['ORR_num'].dropna()
        if not orr_vals.empty:
            min_orr = float(orr_vals.min())
            max_orr = float(orr_vals.max())
            step = 0.5 if (orr_vals % 1 != 0).any() else 1.0
            selected_min = st.slider(
                "ORR mínimo (riesgo)",
                min_value=min_orr,
                max_value=max_orr,
                value=min_orr,
                step=step,
                help="Muestra únicamente operaciones con ORR igual o superior al valor seleccionado.",
            )
            fdf = fdf[fdf['ORR_num'].notna() & (fdf['ORR_num'] >= selected_min)]
        else:
            st.info("No hay datos numéricos de ORR para filtrar.")

if fdf.empty:
    st.warning("No hay datos disponibles con esta combinación de filtros.")
    st.stop()

# ------------------------------
# SECCIONES
# ------------------------------

st.header("Resumen General")
render_portfolio_summary(total_portfolio, fdf)
st.markdown("### KPIs filtrados")
render_kpis(fdf)

st.divider()

plot_df = fdf[fdf['US $ Equiv'] > 0]

if plot_df.empty:
    st.warning("Solo hay valores en cero después de aplicar los filtros; no se mostrarán gráficos.")
else:
    render_concentration(plot_df)

    st.divider()

    st.header("Desglose por Dimensiones")

    render_breakdown(plot_df, "Country", "Exposición por país", "País")
    render_breakdown(plot_df, "Segment", "Exposición por segmento", "Segmento")
    render_breakdown(plot_df, "Product Type", "Exposición por tipo de producto", "Tipo de producto")
    render_breakdown(plot_df, "Sector", "Exposición por sector", "Sector")
    render_breakdown(plot_df, "Sector 2", "Exposición por Sector 2", "Sector 2")

    st.divider()

    render_heatmap(plot_df)

    st.divider()

    render_top_bottom(plot_df)

    st.divider()

st.header("Detalle Completo")
st.dataframe(fdf, use_container_width=True)

