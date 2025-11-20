import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# =========================
# CONFIGURACIÓN GENERAL
# =========================

SCALE_FACTOR = 1_000  # El archivo viene en miles de dólares

st.set_page_config(
    page_title='Análisis de portafolio',
    page_icon=':bar_chart:',
    layout='wide',
)

# Ruta del archivo (igual a como lo tenías, ajusta si es necesario)
PORTFOLIO_FILE = (
    Path(__file__).parent
    / 'data'
    / 'Corporate Data for GCP 31Aug25 - CBA Advisory-Restricted name_V2.csv'
)


# =========================
# CARGA Y LIMPIEZA DE DATOS
# =========================

@st.cache_data
def load_portfolio_data(file_path: Path, last_modified: float):
    """
    Lee y limpia la base de datos del portafolio corporativo.

    El parámetro `last_modified` hace que se invalide la caché
    cuando el archivo cambia.
    """
    df = pd.read_csv(file_path)

    # Limpiar espacios en nombres de columnas
    df.columns = [col.strip() for col in df.columns]

    # Normalizar columna de monto
    amount_col = 'US $ Equiv'
    df[amount_col] = (
        df[amount_col]
        .astype(str)
        .str.replace('[^0-9.-]', '', regex=True)
        .replace('', pd.NA)
    )
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
    df[amount_col] = df[amount_col] * SCALE_FACTOR

    # Rellenar dimensiones categóricas clave
    for column in ['Country', 'Segment', 'Product Type', 'Sector', 'Sector 2']:
        if column in df.columns:
            df[column] = df[column].fillna('Sin especificar')

    return df


# =========================
# FUNCIONES AUXILIARES
# =========================

def format_currency(value: float) -> str:
    """Retorna el valor en formato moneda con separador de miles."""
    return f"{value:,.0f}"


def concentration_ratio(df: pd.DataFrame, column: str, top_n: int = 10) -> float:
    """
    Calcula el CR_n (Concentration Ratio): porcentaje de exposición
    concentrada en las top_n categorías de una dimensión.
    """
    if df.empty:
        return 0.0

    grouped = df.groupby(column)['US $ Equiv'].sum().sort_values(ascending=False)
    total = grouped.sum()
    if total == 0:
        return 0.0

    top_sum = grouped.head(top_n).sum()
    return float(top_sum / total)


def hhi(df: pd.DataFrame, column: str) -> float:
    """
    Calcula el índice de Herfindahl–Hirschman (HHI) para una dimensión.
    Se escala a 0–10.000 como en análisis de competencia.
    """
    if df.empty:
        return 0.0

    grouped = df.groupby(column)['US $ Equiv'].sum()
    total = grouped.sum()
    if total == 0:
        return 0.0

    shares = grouped / total
    return float((shares.pow(2).sum()) * 10_000)


# =========================
# BLOQUES DE RENDERIZADO
# =========================

def render_kpis(df: pd.DataFrame):
    """Muestra métricas clave del dataset filtrado."""
    total_exposure = df['US $ Equiv'].sum()
    accounts = len(df)
    average_ticket = df['US $ Equiv'].mean() if accounts else 0

    col1, col2, col3 = st.columns(3)
    col1.metric('Exposición total (US$)', format_currency(total_exposure))
    col2.metric('Número de registros', f"{accounts:,}")
    col3.metric('Ticket promedio (US$)', format_currency(average_ticket))


def render_concentration_section(df: pd.DataFrame):
    """Muestra indicadores de concentración (CR y HHI) por dimensión."""
    st.subheader('Indicadores de concentración y diversificación')

    dims = [
        ('Country', 'País'),
        ('Sector', 'Sector'),
        ('Segment', 'Segmento'),
    ]

    for col_name, label in dims:
        if col_name not in df.columns:
            continue

        hhi_val = hhi(df, col_name)
        cr3 = concentration_ratio(df, col_name, top_n=3) * 100
        cr10 = concentration_ratio(df, col_name, top_n=10) * 100

        with st.expander(f'{label}: concentración'):
            c1, c2, c3 = st.columns(3)
            c1.metric('HHI', f"{hhi_val:,.0f}")
            c2.metric('CR3', f"{cr3:,.1f}%")
            c3.metric('CR10', f"{cr10:,.1f}%")

            st.caption(
                f"• HHI mide concentración (0–10.000). "
                f"Valores > 2.500 suelen indicar alta concentración.\n"
                f"• CR3/CR10 muestran qué % del portafolio se concentra "
                f"en las 3/10 categorías principales de {label.lower()}."
            )


def render_breakdown(
    df: pd.DataFrame,
    column: str,
    title: str,
    filter_label: str,
    include_pie: bool = False,
):
    """
    Desglose por dimensión con:
    - Filtro independiente para barras
    - Opcionalmente, gráfico de pie
    - Tabla formateada
    """
    if column not in df.columns:
        return

    st.subheader(title)

    choices = sorted(df[column].unique())

    left_col, right_col = st.columns(2)

    # ------------------------
    # BARRAS
    # ------------------------
    with left_col:
        bar_selection = st.multiselect(
            f'{filter_label} visibles (barras)',
            choices,
            default=choices,
            key=f"filter_bar_{column}",
        )
        bar_values = bar_selection if bar_selection else choices
        bar_filtered = df[df[column].isin(bar_values)]

        bar_grouped = (
            bar_filtered.groupby(column)['US $ Equiv']
            .sum()
            .reset_index()
            .sort_values('US $ Equiv', ascending=False)
        )

        total_exposure = bar_grouped['US $ Equiv'].sum()
        bar_grouped['Porcentaje de exposición'] = (
            bar_grouped['US $ Equiv'] / total_exposure * 100
        ).fillna(0)

        bar_chart = (
            alt.Chart(bar_grouped)
            .mark_bar(color='#2563eb')
            .encode(
                x=alt.X('US $ Equiv:Q', title='Exposición (US$)'),
                y=alt.Y(f'{column}:N', sort='-x', title=filter_label),
                tooltip=[
                    alt.Tooltip(f'{column}:N', title=filter_label),
                    alt.Tooltip('US $ Equiv:Q', title='Exposición (US$)', format=',.0f'),
                    alt.Tooltip(
                        'Porcentaje de exposición:Q',
                        title='Porcentaje',
                        format='.1f%'
                    ),
                ],
            )
        )

        st.altair_chart(bar_chart, use_container_width=True)

    # ------------------------
    # PIE
    # ------------------------
    if include_pie:
        with right_col:
            pie_selection = st.multiselect(
                f'{filter_label} visibles (pie)',
                choices,
                default=choices,
                key=f"filter_pie_{column}",
            )
            pie_values = pie_selection if pie_selection else choices
            pie_filtered = df[df[column].isin(pie_values)]

            pie_grouped = (
                pie_filtered.groupby(column)['US $ Equiv']
                .sum()
                .reset_index()
                .sort_values('US $ Equiv', ascending=False)
            )

            pie_total = pie_grouped['US $ Equiv'].sum()
            pie_grouped['Porcentaje de exposición'] = (
                pie_grouped['US $ Equiv'] / pie_total * 100
            ).fillna(0)

            pie_chart = (
                alt.Chart(pie_grouped)
                .mark_arc()
                .encode(
                    theta=alt.Theta('US $ Equiv:Q', stack=True),
                    color=alt.Color(f'{column}:N', title=filter_label),
                    tooltip=[
                        alt.Tooltip(f'{column}:N', title=filter_label),
                        alt.Tooltip('US $ Equiv:Q', title='Exposición (US$)', format=',.0f'),
                        alt.Tooltip(
                            'Porcentaje de exposición:Q',
                            title='Porcentaje',
                            format='.1f%'
                        ),
                    ],
                )
            )

            st.altair_chart(pie_chart, use_container_width=True)

    # ------------------------
    # TABLA
    # ------------------------
    display_df = bar_grouped.rename(columns={'US $ Equiv': 'Exposición (US$)'})
    display_df['Exposición (US$)'] = display_df['Exposición (US$)'].apply(
        format_currency
    )
    display_df['Porcentaje de exposición'] = display_df[
        'Porcentaje de exposición'
    ].map(lambda x: f"{x:.1f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_heatmap_country_sector(df: pd.DataFrame):
    """Mapa de calor País × Sector por exposición."""
    if not {'Country', 'Sector'}.issubset(df.columns):
        return

    st.subheader('Mapa de calor País vs Sector')

    grouped = (
        df.groupby(['Country', 'Sector'])['US $ Equiv']
        .sum()
        .reset_index()
    )

    if grouped.empty:
        st.info("No hay datos para construir el mapa de calor con los filtros actuales.")
        return

    heatmap = (
        alt.Chart(grouped)
        .mark_rect()
        .encode(
            x=alt.X('Sector:N', title='Sector'),
            y=alt.Y('Country:N', title='País'),
            color=alt.Color(
                'US $ Equiv:Q',
                title='Exposición (US$)',
                scale=alt.Scale(scheme='blues')
            ),
            tooltip=[
                alt.Tooltip('Country:N', title='País'),
                alt.Tooltip('Sector:N', title='Sector'),
                alt.Tooltip('US $ Equiv:Q', title='Exposición (US$)', format=',.0f'),
            ],
        )
        .properties(height=400)
    )

    st.altair_chart(heatmap, use_container_width=True)


def render_top_bottom_operations(df: pd.DataFrame, n: int = 10):
    """Muestra las top y bottom operaciones por exposición."""
    st.subheader(f'Top y bottom {n} operaciones por exposición')

    if df.empty:
        st.info("No hay operaciones para mostrar con los filtros actuales.")
        return

    amount_col = 'US $ Equiv'

    cols_show = [
        'Short name',
        'Country',
        'Segment',
        'Product Type',
        'Sector',
        'Sector 2',
        amount_col,
    ]
    cols_show = [c for c in cols_show if c in df.columns]

    ranked = df.sort_values(amount_col, ascending=False)

    top = ranked.head(n)[cols_show].copy()
    bottom = ranked[ranked[amount_col] > 0].tail(n)[cols_show].copy()

    for d in (top, bottom):
        if amount_col in d.columns:
            d[amount_col] = d[amount_col].apply(format_currency)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"###### Top {n} por exposición")
        st.dataframe(top, use_container_width=True, hide_index=True)

    with col2:
        st.markdown(f"###### Bottom {n} por exposición (mayores a 0)")
        st.dataframe(bottom, use_container_width=True, hide_index=True)


# =========================
# APP PRINCIPAL
# =========================

portfolio_df = load_portfolio_data(
    PORTFOLIO_FILE,
    PORTFOLIO_FILE.stat().st_mtime,
)

st.title('Análisis del portafolio corporativo')
st.caption('Explora la exposición y concentración por país, segmento, producto y sector.')

st.divider()

# -------------------------
# FILTROS GLOBALES
# -------------------------

st.sidebar.header("Filtros globales")

filtered_df = portfolio_df.copy()

if 'Country' in portfolio_df.columns:
    countries = sorted(portfolio_df['Country'].unique())
    selected_countries = st.sidebar.multiselect(
        'País',
        countries,
        default=countries,
    )
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

if 'Segment' in portfolio_df.columns:
    segments = sorted(portfolio_df['Segment'].unique())
    selected_segments = st.sidebar.multiselect(
        'Segmento',
        segments,
        default=segments,
    )
    if selected_segments:
        filtered_df = filtered_df[filtered_df['Segment'].isin(selected_segments)]

if 'Product Type' in portfolio_df.columns:
    products = sorted(portfolio_df['Product Type'].unique())
    selected_products = st.sidebar.multiselect(
        'Tipo de producto',
        products,
        default=products,
    )
    if selected_products:
        filtered_df = filtered_df[filtered_df['Product Type'].isin(selected_products)]

if 'Sector' in portfolio_df.columns:
    sectors = sorted(portfolio_df['Sector'].unique())
    selected_sectors = st.sidebar.multiselect(
        'Sector',
        sectors,
        default=sectors,
    )
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]

# Si con los filtros se vacía el dataset
if filtered_df.empty:
    st.warning("No hay datos con la combinación de filtros seleccionada.")
    st.stop()

# -------------------------
# RESUMEN GENERAL
# -------------------------

st.header('Resumen general')
render_kpis(filtered_df)

st.divider()

# -------------------------
# CONCENTRACIÓN
# -------------------------

render_concentration_section(filtered_df)

st.divider()

# -------------------------
# DESGLOSE POR DIMENSIONES
# -------------------------

st.header('Desglose por dimensiones')

render_breakdown(
    filtered_df,
    'Country',
    'Exposición por país',
    'País',
    include_pie=True,
)
render_breakdown(
    filtered_df,
    'Segment',
    'Exposición por segmento',
    'Segmento',
    include_pie=True,
)
render_breakdown(
    filtered_df,
    'Product Type',
    'Exposición por tipo de producto',
    'Tipo de producto',
    include_pie=True,
)
render_breakdown(
    filtered_df,
    'Sector',
    'Exposición por sector',
    'Sector',
    include_pie=True,
)
render_breakdown(
    filtered_df,
    'Sector 2',
    'Exposición por Sector 2',
    'Sector 2',
    include_pie=True,
)

st.divider()

# -------------------------
# MAPA DE CALOR PAÍS × SECTOR
# -------------------------

render_heatmap_country_sector(filtered_df)

st.divider()

# -------------------------
# TOP / BOTTOM OPERACIONES
# -------------------------

render_top_bottom_operations(filtered_df, n=10)

st.divider()

# -------------------------
# DETALLE DE OPERACIONES
# -------------------------

st.header('Detalle de operaciones filtradas')
st.dataframe(filtered_df, use_container_width=True)
