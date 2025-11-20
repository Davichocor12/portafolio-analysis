import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

SCALE_FACTOR = 1_000

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
    df[amount_col] = df[amount_col] * SCALE_FACTOR

    for column in ['Country', 'Segment', 'Product Type', 'Sector', 'Sector 2']:
        df[column] = df[column].fillna('Sin especificar')

    return df


def format_currency(value: float) -> str:
    """Return currency formatted with thousand separators."""

    return f"{value:,.0f}"


def render_kpis(df: pd.DataFrame):
    """Display key metrics for the filtered dataset."""

    total_exposure = df['US $ Equiv'].sum()
    accounts = len(df)
    average_ticket = df['US $ Equiv'].mean() if accounts else 0

    col1, col2, col3 = st.columns(3)
    col1.metric('Exposición total (US$)', format_currency(total_exposure))
    col2.metric('Número de registros', f"{accounts:,}")
    col3.metric('Ticket promedio (US$)', format_currency(average_ticket))


def render_breakdown(
    df: pd.DataFrame,
    column: str,
    title: str,
    filter_label: str,
    include_pie: bool = False,
):
    """Render a breakdown with filters, bar chart, optional pie and a formatted table."""

    st.subheader(title)

    choices = sorted(df[column].unique())

    left_col, right_col = st.columns(2)

    # Bar chart (independent filter)
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
                    alt.Tooltip('Porcentaje de exposición:Q', title='Porcentaje', format='.1f%'),
                ],
            )
        )

        st.altair_chart(bar_chart, use_container_width=True)

    # Pie chart (independent filter)
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
                        alt.Tooltip('Porcentaje de exposición:Q', title='Porcentaje', format='.1f%'),
                    ],
                )
            )

            st.altair_chart(pie_chart, use_container_width=True)

    display_df = bar_grouped.rename(columns={'US $ Equiv': 'Exposición (US$)'})
    display_df['Exposición (US$)'] = display_df['Exposición (US$)'].apply(
        format_currency
    )
    display_df['Porcentaje de exposición'] = display_df[
        'Porcentaje de exposición'
    ].map(lambda x: f"{x:.1f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


portfolio_df = load_portfolio_data(
    PORTFOLIO_FILE,
    PORTFOLIO_FILE.stat().st_mtime,
)

st.title('Análisis del portafolio corporativo')
st.caption('Explora la exposición y concentración por país, segmento y producto.')

st.divider()
st.header('Resumen general')
render_kpis(portfolio_df)

st.divider()

st.header('Desglose por dimensiones')
render_breakdown(
    portfolio_df,
    'Country',
    'Exposición por país',
    'País',
    include_pie=True,
)
render_breakdown(
    portfolio_df,
    'Segment',
    'Exposición por segmento',
    'Segmento',
    include_pie=True,
)
render_breakdown(
    portfolio_df,
    'Product Type',
    'Exposición por tipo de producto',
    'Tipo de producto',
    include_pie=True,
)
render_breakdown(
    portfolio_df,
    'Sector',
    'Exposición por sector',
    'Sector',
    include_pie=True,
)
render_breakdown(
    portfolio_df,
    'Sector 2',
    'Exposición por Sector 2',
    'Sector 2',
    include_pie=True,
)

st.divider()

st.header('Detalle de operaciones filtradas')
st.dataframe(portfolio_df, use_container_width=True)
