import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title='Análisis de portafolio',
    page_icon=':bar_chart:',
    layout='wide',
)

PORTFOLIO_FILE = (
    Path(__file__).parent
    / 'data'
    / 'Corporate Data for GCP 31Aug25 - CBA Advisory-Restricted name_V2.csv'
)


@st.cache_data
def load_portfolio_data(file_path: Path, last_modified: float):
    """Read and clean the corporate portfolio dataset.

    The ``last_modified`` parameter ensures Streamlit invalidates the cache
    whenever the underlying data file is updated.
    """

    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]

    amount_col = 'US $ Equiv'
    df[amount_col] = (
        df[amount_col]
        .astype(str)
        .str.replace('[^0-9.-]', '', regex=True)
        .replace('', pd.NA)
    )
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)

    for column in ['Country', 'Segment', 'Product Type', 'Sector', 'Sector 2']:
        df[column] = df[column].fillna('Sin especificar')

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to the dataframe."""

    with st.sidebar:
        st.header('Filtros del portafolio')

        filter_options = {}
        for column, label in [
            ('Country', 'Países'),
            ('Segment', 'Segmentos'),
            ('Product Type', 'Tipos de producto'),
            ('Sector', 'Sectores'),
            ('Sector 2', 'Sector 2'),
        ]:
            choices = sorted(df[column].unique())
            selection = st.multiselect(
                label,
                choices,
                default=choices,
            )
            filter_options[column] = selection if selection else choices

        min_amount, max_amount = st.slider(
            'Rango de exposición (US$)',
            float(df['US $ Equiv'].min()),
            float(df['US $ Equiv'].max()),
            (float(df['US $ Equiv'].min()), float(df['US $ Equiv'].max())),
        )

    filtered_df = df[
        (df['Country'].isin(filter_options['Country']))
        & (df['Segment'].isin(filter_options['Segment']))
        & (df['Product Type'].isin(filter_options['Product Type']))
        & (df['Sector'].isin(filter_options['Sector']))
        & (df['Sector 2'].isin(filter_options['Sector 2']))
        & (df['US $ Equiv'].between(min_amount, max_amount))
    ]

    return filtered_df


def render_kpis(df: pd.DataFrame):
    """Display key metrics for the filtered dataset."""

    total_exposure = df['US $ Equiv'].sum()
    accounts = len(df)
    average_ticket = df['US $ Equiv'].mean() if accounts else 0

    col1, col2, col3 = st.columns(3)
    col1.metric('Exposición total (US$)', f"{total_exposure:,.0f}")
    col2.metric('Número de registros', f"{accounts:,}")
    col3.metric('Ticket promedio (US$)', f"{average_ticket:,.0f}")


def render_breakdown(df: pd.DataFrame, column: str, title: str):
    """Render a bar chart with exposure grouped by the provided column."""

    grouped = (
        df.groupby(column)['US $ Equiv']
        .sum()
        .reset_index()
        .sort_values('US $ Equiv', ascending=False)
    )

    st.subheader(title)
    st.bar_chart(grouped, x=column, y='US $ Equiv')
    st.dataframe(grouped, use_container_width=True, hide_index=True)


portfolio_df = load_portfolio_data(
    PORTFOLIO_FILE,
    PORTFOLIO_FILE.stat().st_mtime,
)

st.title('Análisis del portafolio corporativo')
st.caption('Explora la exposición y concentración por país, segmento y producto.')

filtered_portfolio = apply_filters(portfolio_df)

st.divider()
st.header('Resumen general')
render_kpis(filtered_portfolio)

st.divider()

st.header('Desglose por dimensiones')
render_breakdown(filtered_portfolio, 'Country', 'Exposición por país')
render_breakdown(filtered_portfolio, 'Segment', 'Exposición por segmento')
render_breakdown(filtered_portfolio, 'Product Type', 'Exposición por tipo de producto')
render_breakdown(filtered_portfolio, 'Sector', 'Exposición por sector')
render_breakdown(filtered_portfolio, 'Sector 2', 'Exposición por Sector 2')

st.divider()

st.header('Detalle de operaciones filtradas')
st.dataframe(filtered_portfolio, use_container_width=True)
