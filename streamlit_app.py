import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

CATEGORY10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

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
    / 'Corporate Data for GCP 31Aug25 - CBA Advisory-Restricted name_V4.csv'
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
    categorical_cols = ['Country', 'Segment', 'Product Type', 'Sector', 'Sector 2', 'Delinq band']
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().fillna("Sin especificar")
            df[c] = df[c].replace("nan", "Sin especificar")

    # 4.1. Columna simplificada de morosidad
    if 'Delinq band' in df.columns:
        df['Delinq band simple'] = df['Delinq band'].apply(
            lambda x: "No" if str(x).strip().lower() == "clean" else "Sí"
        )

    # 5. ORR numérico para filtros de riesgo
    if 'ORR' in df.columns:
        df['ORR_num'] = pd.to_numeric(df['ORR'], errors='coerce')

    # 6. Fecha de madurez normalizada
    if 'Maturity date' in df.columns:
        df['Maturity date'] = pd.to_datetime(
            df['Maturity date'], errors='coerce', dayfirst=False
        )
        # Fechas sentinela (p. ej. 1-Jan-9999) se consideran no informadas
        df.loc[df['Maturity date'].dt.year >= 9999, 'Maturity date'] = pd.NaT

    return df

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def format_currency(value: float) -> str:
    return f"{value:,.0f}"

def weighted_avg_orr(group: pd.DataFrame) -> float:
    weights = group["US $ Equiv"]
    values = group["ORR_num"]

    if weights.sum() > 0:
        return float((values * weights).sum() / weights.sum())
    return float(values.mean()) if not values.empty else 0.0

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


def calendar_year_bucket(maturity_date: pd.Timestamp, today: pd.Timestamp) -> str:
    """Asignar bucket por año calendario para la fecha de madurez."""

    if pd.isna(maturity_date):
        return "Sin fecha"
    if maturity_date < today:
        return "Vencido"
    return str(maturity_date.year)


def render_maturity_analysis(df):
    st.header("Análisis de temporalidad (Maturity date)")

    if 'Maturity date' not in df.columns:
        st.info("No hay columna de fecha de madurez para analizar.")
        return

    maturity_df = df[(df['Maturity date'].notna()) & (df['US $ Equiv'] > 0)].copy()
    if maturity_df.empty:
        st.info("No hay fechas de madurez válidas con exposición positiva.")
        return

    today = pd.Timestamp.today().normalize()
    maturity_df['Días a vencimiento'] = (
        maturity_df['Maturity date'] - today
    ).dt.days

    max_maturity = maturity_df['Maturity date'].max()
    maturity_df['Bucket de plazo'] = maturity_df['Maturity date'].apply(
        calendar_year_bucket, args=(today,)
    )

    total_exposure = maturity_df['US $ Equiv'].sum()
    weights = maturity_df['US $ Equiv']
    wal_days = (
        (maturity_df['Días a vencimiento'] * weights).sum() / total_exposure
        if total_exposure
        else float('nan')
    )
    wal_years = wal_days / 365 if pd.notna(wal_days) else float('nan')
    weighted_maturity_date = (
        (today + pd.Timedelta(days=wal_days)) if pd.notna(wal_days) else None
    )

    matured_exposure = maturity_df.loc[
        maturity_df['Días a vencimiento'] < 0, 'US $ Equiv'
    ].sum()
    matured_share = (matured_exposure / total_exposure * 100) if total_exposure else 0

    upcoming_df = maturity_df[maturity_df['Días a vencimiento'] >= 0]
    next_maturity = (
        upcoming_df.loc[upcoming_df['Días a vencimiento'].idxmin(), 'Maturity date']
        if not upcoming_df.empty
        else None
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Weighted Average Life (años)", f"{wal_years:.2f}" if pd.notna(wal_years) else "N/D")
    c2.metric(
        "Madurez promedio ponderada",
        weighted_maturity_date.strftime("%d-%b-%Y") if weighted_maturity_date else "N/D",
    )
    c3.metric(
        "Exposición vencida",
        format_currency(matured_exposure),
        f"{matured_share:.1f}% del total",
    )
    c4.metric(
        "Próximo vencimiento",
        next_maturity.strftime("%d-%b-%Y") if next_maturity else "N/D",
    )
    c5.metric(
        "Madurez máxima",
        max_maturity.strftime("%d-%b-%Y") if pd.notna(max_maturity) else "N/D",
    )

    bucket_order = ["Vencido"] + [
        str(year) for year in range(today.year, max_maturity.year + 1)
    ]
    bucket_summary = (
        maturity_df.groupby('Bucket de plazo')['US $ Equiv']
        .sum()
        .reset_index(name='Exposición')
    )
    bucket_summary['Participación'] = bucket_summary['Exposición'] / total_exposure
    bucket_summary['Año (t=0 actual)'] = bucket_summary['Bucket de plazo'].apply(
        lambda x: (int(x) - today.year) if str(x).isdigit() else None
    )
    bucket_summary['Bucket de plazo'] = pd.Categorical(
        bucket_summary['Bucket de plazo'], bucket_order
    )
    bucket_summary = bucket_summary.sort_values('Bucket de plazo').dropna(
        subset=['Bucket de plazo']
    )

    st.markdown("### Exposición por plazo (años calendario)")
    st.caption(
        f"Buckets definidos por año calendario a partir del 1-Jan-{today.year}. "
        "Las sumas de los buckets corresponden a años completos."
    )
    bucket_display = bucket_summary.assign(
        Exposición=bucket_summary['Exposición'].apply(format_currency),
        **{
            "Año (t=0 actual)": bucket_summary['Año (t=0 actual)'].apply(
                lambda v: f"{int(v):,}" if pd.notna(v) else "N/A"
            )
        },
        Participación=bucket_summary['Participación'].apply(lambda x: f"{x:.1%}"),
    )[
        ["Bucket de plazo", "Año (t=0 actual)", "Exposición", "Participación"]
    ]

    st.dataframe(bucket_display, use_container_width=True)

    maturity_df['Año vencimiento'] = maturity_df['Maturity date'].dt.year
    year_summary = (
        maturity_df.groupby('Año vencimiento')['US $ Equiv']
        .sum()
        .reset_index(name='Exposición')
    )
    year_summary = year_summary[year_summary['Exposición'] > 0]

    if not year_summary.empty:
        chart = (
            alt.Chart(year_summary)
            .mark_bar(color="#6b21a8")
            .encode(
                x=alt.X('Año vencimiento:O', title='Año de vencimiento'),
                y=alt.Y('Exposición:Q', title='Exposición (US$)'),
                tooltip=[
                    alt.Tooltip('Año vencimiento:O', title='Año'),
                    alt.Tooltip('Exposición:Q', format=",.0f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)

def render_portfolio_summary(total_full: float, df_filtered: pd.DataFrame):
    """Mostrar comparación entre el portafolio completo y el filtrado actual."""

    filtered_total = df_filtered['US $ Equiv'].sum()
    participation = (filtered_total / total_full * 100) if total_full else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Portafolio total (US$)", format_currency(total_full))
    c2.metric("Exposición filtrada (US$)", format_currency(filtered_total))
    c3.metric("Participación del portafolio", f"{participation:.1f}%")

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
                    alt.Tooltip("Porcentaje:Q", format=".1%"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown(
            f"**Total {label.lower()} seleccionado:** {format_currency(total)}",
        )

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

            domain = g2[column].tolist()
            palette = [CATEGORY10[i % len(CATEGORY10)] for i in range(len(domain))]
            color_map = dict(zip(domain, palette))
            color_encoding = alt.Color(
                f"{column}:N", scale=alt.Scale(domain=domain, range=palette), legend=None
            )

            pie = (
                alt.Chart(g2)
                .mark_arc()
                .encode(
                    theta="US $ Equiv:Q",
                    color=color_encoding,
                    tooltip=[
                        f"{column}:N",
                        alt.Tooltip("US $ Equiv:Q", format=",.0f"),
                        alt.Tooltip("Porcentaje:Q", format=".1%"),
                    ],
                )
            )

            pie_col, legend_col = st.columns([0.6, 0.4])
            with pie_col:
                st.altair_chart(pie, use_container_width=True)

            legend_lines = []
            for _, row in g2.sort_values('Porcentaje', ascending=False).iterrows():
                color = color_map.get(row[column], "#999999")
                legend_lines.append(
                    (
                        "<div style='display:flex;align-items:center;margin-bottom:6px;'>"
                        f"<span style='width:14px;height:14px;display:inline-block;"
                        f"border-radius:3px;background:{color};margin-right:8px;'></span>"
                        f"<span>{row[column]}: {(row['Porcentaje'] * 100):.1f}%</span>"
                        "</div>"
                    )
                )
            with legend_col:
                st.markdown("".join(legend_lines), unsafe_allow_html=True)


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


def render_orr_heatmap(df):
    st.subheader("Mapa de calor ORR ponderado País vs Sector")

    if 'ORR_num' not in df.columns:
        st.info("No hay datos de ORR disponibles para calcular el mapa de calor.")
        return

    df_orr = df[(df['ORR_num'].notna()) & (df['US $ Equiv'] > 0)].copy()
    if df_orr.empty:
        st.info("No hay registros con ORR y exposición positiva para esta selección.")
        return

    grouped = (
        df_orr.groupby(['Country', 'Sector'])
        .apply(
            lambda g: pd.Series(
                {
                    "ORR ponderado": weighted_avg_orr(g),
                    "Exposición": g['US $ Equiv'].sum(),
                }
            )
        )
        .reset_index()
    )

    grouped = grouped[grouped['Exposición'] > 0]
    if grouped.empty:
        st.info("No hay exposición positiva para graficar en el mapa de calor de ORR.")
        return

    heat = (
        alt.Chart(grouped)
        .mark_rect()
        .encode(
            x=alt.X("Sector:N"),
            y=alt.Y("Country:N"),
            color=alt.Color("ORR ponderado:Q", scale=alt.Scale(scheme="greens")),
            tooltip=[
                "Country",
                "Sector",
                alt.Tooltip("ORR ponderado:Q", format=".2f"),
                alt.Tooltip("Exposición:Q", format=",.0f"),
            ],
        )
        .properties(height=380)
    )

    st.altair_chart(heat, use_container_width=True)

# ============================================
# RENDER: ORR POR DIMENSIÓN
# ============================================

def render_orr_by_dimension(df):
    st.subheader("ORR por dimensión")

    if 'ORR_num' not in df.columns:
        st.info("No hay datos de ORR disponibles para graficar.")
        return

    dims = {
        "Country": "País",
        "Segment": "Segmento",
        "Product Type": "Tipo de producto",
        "Sector": "Sector",
        "Sector 2": "Sector 2",
    }

    dimension = st.selectbox(
        "Dimensión para graficar ORR",
        options=list(dims.keys()),
        format_func=lambda x: dims[x],
        key="orr_dimension_selector",
    )

    # Selección de categorías específicas dentro de la dimensión
    category_options = sorted(df[dimension].unique())
    selected_categories = st.multiselect(
        f"{dims[dimension]} a mostrar",
        category_options,
        default=category_options,
        key=f"orr_categories_{dimension}",
    )

    df_selected = df[df[dimension].isin(selected_categories)]
    if df_selected.empty:
        st.info("No hay datos disponibles para la selección realizada.")
        return

    # Exposición total por categoría (incluye registros sin ORR)
    exposure_total = (
        df_selected.groupby(dimension)["US $ Equiv"].sum().reset_index(name="Exposición total")
    )
    exposure_total = exposure_total[exposure_total["Exposición total"] > 0]

    if exposure_total.empty:
        st.info("No hay exposición positiva para mostrar en esta dimensión.")
        return

    # Datos con ORR disponibles para el cálculo ponderado
    df_orr = df_selected[df_selected['ORR_num'].notna()].copy()
    if df_orr.empty:
        st.info("No hay datos de ORR disponibles para graficar.")
        return

    orr_grouped = (
        df_orr.groupby(dimension)
        .apply(
            lambda g: pd.Series(
                {
                    "ORR ponderado": weighted_avg_orr(g),
                    "Exposición con ORR": g["US $ Equiv"].sum(),
                }
            )
        )
        .reset_index()
    )

    # Unir exposición total con el ORR ponderado, conservando solo las categorías con ORR
    grouped = exposure_total.merge(orr_grouped, on=dimension, how="inner")
    if grouped.empty:
        st.info("Las categorías seleccionadas no tienen ORR disponible.")
        return

    grouped = grouped.sort_values("ORR ponderado", ascending=False)

    chart = (
        alt.Chart(grouped)
        .mark_bar(color="#10b981")
        .encode(
            x=alt.X("ORR ponderado:Q", title="ORR ponderado por exposición"),
            y=alt.Y(f"{dimension}:N", sort="-x", title=dims[dimension]),
            tooltip=[
                f"{dimension}:N",
                alt.Tooltip("ORR ponderado:Q", format=".2f"),
                alt.Tooltip("Exposición total:Q", format=",.0f"),
                alt.Tooltip("Exposición con ORR:Q", format=",.0f"),
            ],
        )
    )

    st.altair_chart(chart, use_container_width=True)


def render_exposure_by_dimension(df):
    st.subheader("Exposición (US$) por dimensión")

    dims = {
        "Country": "País",
        "Segment": "Segmento",
        "Product Type": "Tipo de producto",
        "Sector": "Sector",
        "Sector 2": "Sector 2",
        "Delinq band": "Delinq band (rangos)",
        "Delinq band simple": "¿Con morosidad? (Sí/No)",
    }

    available_dims = [d for d in dims if d in df.columns]
    if not available_dims:
        st.info("No hay dimensiones disponibles para graficar.")
        return

    dimension = st.selectbox(
        "Dimensión para graficar exposición",
        options=available_dims,
        format_func=lambda x: dims[x],
        key="exposure_dimension_selector",
    )

    category_options = sorted(df[dimension].unique())
    selected_categories = st.multiselect(
        f"{dims[dimension]} a mostrar",
        category_options,
        default=category_options,
        key=f"exposure_categories_{dimension}",
    )

    df_selected = df[df[dimension].isin(selected_categories)]
    if df_selected.empty:
        st.info("No hay datos disponibles para la selección realizada.")
        return

    exposure = (
        df_selected.groupby(dimension)["US $ Equiv"].sum().reset_index(name="Exposición")
    )
    exposure = exposure[exposure["Exposición"] > 0]

    if exposure.empty:
        st.info("No hay exposición positiva para mostrar en esta dimensión.")
        return

    exposure = exposure.sort_values("Exposición", ascending=False)

    chart = (
        alt.Chart(exposure)
        .mark_bar(color="#1d4ed8")
        .encode(
            x=alt.X("Exposición:Q", title="Exposición (US$)"),
            y=alt.Y(f"{dimension}:N", sort="-x", title=dims[dimension]),
            tooltip=[f"{dimension}:N", alt.Tooltip("Exposición:Q", format=",.0f")],
        )
    )

    st.altair_chart(chart, use_container_width=True)

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

plot_df = fdf[fdf['US $ Equiv'] > 0]

if plot_df.empty:
    st.warning("Solo hay valores en cero después de aplicar los filtros; no se mostrarán gráficos.")
else:
    st.header("Desglose por Dimensiones")

    render_exposure_by_dimension(plot_df)
    st.divider()

    render_orr_by_dimension(plot_df)
    st.divider()

    render_breakdown(plot_df, "Country", "Exposición por país", "País")
    render_breakdown(plot_df, "Segment", "Exposición por segmento", "Segmento")
    render_breakdown(plot_df, "Product Type", "Exposición por tipo de producto", "Tipo de producto")
    render_breakdown(plot_df, "Sector", "Exposición por sector", "Sector")
    render_breakdown(plot_df, "Sector 2", "Exposición por Sector 2", "Sector 2")
    render_breakdown(plot_df, "Delinq band", "Exposición por Delinq band", "Delinq band")

    st.divider()

    render_heatmap(plot_df)

    render_orr_heatmap(plot_df)

    st.divider()

    render_maturity_analysis(plot_df)

    st.divider()

    render_top_bottom(plot_df)

    st.divider()

st.header("Detalle Completo")
st.dataframe(fdf, use_container_width=True)

