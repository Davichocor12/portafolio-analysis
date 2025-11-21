import base64
import streamlit as st
import pandas as pd
import altair as alt
from itertools import cycle
from pathlib import Path

CATEGORY10 = [
    "#032043",  # Deep navy blue
    "#be993c",  # Yellow
    "#032043",  # Dark blue
    "#45777e",  # Teal variant
    "#5d8291",  # Mid tone
    "#9ec5ca",  # Light blue
]

BRAND_COLORS = {
    "primary": "#032043",  # Deep navy blue (primary bars and filters)
    "secondary": "#45777e",  # Teal variant (buttons, highlights)
    "accent": "#be993c",  # Yellow (calls to action)
    "neutral": "#032043",  # Dark blue for primary text
    "muted": "#5d8291",  # Secondary text
    "highlight": "#9ec5ca",  # Soft highlights
}
BRAND_FONT = "Arial, sans-serif"

BAR_COLOR_SEQUENCE = ["#be993c", "#5d9ea7"]
BAR_COLOR_CYCLE = cycle(BAR_COLOR_SEQUENCE)

# ============================================
# GENERAL CONFIGURATION
# ============================================

SCALE_FACTOR = 1_000  # The file is in thousands of USD

st.set_page_config(
    page_title='Portfolio Analysis',
    page_icon=':bar_chart:',
    layout='wide',
)


def encode_logo(logo_path: Path) -> str | None:
    """Convert the logo to base64 so it can be embedded in HTML.

    Returns ``None`` if the file does not exist or cannot be read.
    """

    try:
        return base64.b64encode(logo_path.read_bytes()).decode()
    except FileNotFoundError:
        return None


def apply_brand_styling() -> str | None:
    """Apply typography and the corporate color palette.

    Returns the logo encoded in base64 so it can be used in the layout.
    """

    logo_path = Path(__file__).parent / "logo.jpg"
    logo_b64 = encode_logo(logo_path)

    st.markdown(
        f"""
        <style>
            :root {{
                --brand-primary: {BRAND_COLORS["primary"]};
                --brand-secondary: {BRAND_COLORS["secondary"]};
                --brand-accent: {BRAND_COLORS["accent"]};
                --brand-neutral: {BRAND_COLORS["neutral"]};
                --brand-muted: {BRAND_COLORS["muted"]};
                --brand-highlight: {BRAND_COLORS["highlight"]};
                --brand-font: {BRAND_FONT};
            }}

            html, body, .stApp {{
                font-family: var(--brand-font) !important;
                color: var(--brand-neutral);
            }}

            h1, h2 {{
                color: var(--brand-primary);
                font-weight: 700;
                letter-spacing: 0.02em;
            }}

            h3, h4 {{
                color: var(--brand-secondary);
                font-weight: 600;
            }}

            p, span, label, .stMarkdown {{
                color: var(--brand-neutral);
            }}

            .stCaption, .st-emotion-cache-1q7q0r2 {{
                color: var(--brand-muted) !important;
                font-size: 0.95rem;
            }}

            /* Metrics adjustments */
            [data-testid="stMetric"] label, [data-testid="stMetricLabel"] {{
                color: var(--brand-muted);
                font-size: 0.95rem;
            }}
            [data-testid="stMetricValue"] {{
                color: var(--brand-primary);
                font-weight: 700;
            }}

            /* Sidebar and controls */
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, rgba(41,67,92,0.08), rgba(63,127,129,0.06));
                padding-top: 12px;
            }}
            [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {{
                color: var(--brand-primary);
            }}
            [data-baseweb="select"] > div {{
                border-radius: 8px;
                border-color: var(--brand-primary) !important;
            }}
            [data-baseweb="tag"] {{
                background-color: var(--brand-primary) !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 600;
            }}
            [data-baseweb="tag"] * {{
                color: #ffffff !important;
            }}
            [data-baseweb="tag"] svg {{
                fill: #ffffff !important;
            }}
            [data-testid="stSidebar"] button {{
                background-color: var(--brand-accent) !important;
                color: #ffffff !important;
                font-weight: 700 !important;
            }}

            /* Buttons and selectors */
            .stButton > button, .stDownloadButton > button {{
                background-color: var(--brand-accent) !important;
                color: #ffffff !important;
                border-radius: 8px !important;
                border: none !important;
                font-weight: 700 !important;
            }}
            .stButton > button:hover, .stDownloadButton > button:hover {{
                background-color: var(--brand-secondary) !important;
                color: #ffffff !important;
            }}
            button {{
                background-color: var(--brand-accent) !important;
                color: #ffffff !important;
                border-radius: 8px !important;
                border: none !important;
                font-weight: 700;
            }}
            button:hover {{
                background-color: var(--brand-secondary) !important;
                color: #ffffff !important;
            }}

            /* Title with logo */
            .title-with-logo {{
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .title-with-logo h1 {{
                margin: 0;
            }}
            .title-logo {{
                height: 56px;
                width: auto;
                object-fit: contain;
            }}
            .page-caption {{
                color: var(--brand-muted);
                margin-top: 4px;
            }}

            /* Cards and soft dividers */
            .stApp > header {{
                background: linear-gradient(90deg, rgba(41,67,92,0.08), rgba(242,165,65,0.06));
            }}
            .stDataFrame, .stMarkdown {{
                border-radius: 8px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    return logo_b64


def next_bar_color() -> str:
    """Return the next bar color alternating through the requested palette."""

    return next(BAR_COLOR_CYCLE)

# Archivo CSV
PORTFOLIO_FILE = (
    Path(__file__).parent
    / 'data'
    / 'Corporate Data for GCP 31Aug25 - CBA Advisory-Restricted name_V4.csv'
)

# ============================================
# DATA LOADING AND CLEANUP
# ============================================

@st.cache_data
def load_portfolio_data(file_path: Path, last_modified: float):
    """Load the portfolio while handling alternative file formats.

    The function first tries to read the expected CSV; if it does not exist or
    fails to load, it looks for an XLSX with the same name. When loading fails
    a clear message is shown in the app so the page does not stay blank.
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
                st.error(f"The portfolio file could not be read: {e}")
                st.stop()
        else:
            st.error(
                "The portfolio file (CSV or XLSX) was not found in the data folder."
            )
            st.stop()

    if df is None:
        st.error("The portfolio file could not be read.")
        st.stop()

    # 1. CLEAN COLUMN NAMES
    df.columns = [col.strip() for col in df.columns]

    # 2. Rename the exact column from the source file
    if "US $ Equiv" not in df.columns:
        df = df.rename(columns={' US $ Equiv ': 'US $ Equiv'})

    # 3. AMOUNT COLUMN
    df['US $ Equiv'] = (
        df['US $ Equiv']
        .astype(str)
        .str.replace('[^0-9.-]', '', regex=True)
        .replace('', pd.NA)
    )

    df['US $ Equiv'] = pd.to_numeric(df['US $ Equiv'], errors='coerce').fillna(0)
    df['US $ Equiv'] = df['US $ Equiv'] * SCALE_FACTOR

    # 4. Normalize categories (NO NULLS)
    categorical_cols = ['Country', 'Segment', 'Product Type', 'Sector', 'Sector 2', 'Delinq band']
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().fillna("Not specified")
            df[c] = df[c].replace("nan", "Not specified")

    # 4.1. Simplified delinquency flag column
    if 'Delinq band' in df.columns:
        df['Delinq band simple'] = df['Delinq band'].apply(
            lambda x: "No" if str(x).strip().lower() == "clean" else "Yes"
        )

    # 5. Numeric ORR for risk filters
    if 'ORR' in df.columns:
        df['ORR_num'] = pd.to_numeric(df['ORR'], errors='coerce')

    # 6. Normalized maturity date
    if 'Maturity date' in df.columns:
        df['Maturity date'] = pd.to_datetime(
            df['Maturity date'], errors='coerce', dayfirst=False
        )
        # Sentinel dates (e.g., 1-Jan-9999) are considered not provided
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
    c1.metric("Total exposure (US$)", format_currency(total))
    c2.metric("Number of records", f"{n:,}")
    c3.metric("Average ticket (US$)", format_currency(avg))


def calendar_year_bucket(maturity_date: pd.Timestamp, today: pd.Timestamp) -> str:
    """Assign a calendar-year bucket for the maturity date."""

    if pd.isna(maturity_date):
        return "No date"
    if maturity_date < today:
        return "Expired"
    return str(maturity_date.year)


def render_maturity_analysis(df):
    st.header("Maturity timing analysis")

    if 'Maturity date' not in df.columns:
        st.info("There is no maturity date column to analyze.")
        return

    maturity_df = df[(df['Maturity date'].notna()) & (df['US $ Equiv'] > 0)].copy()
    if maturity_df.empty:
        st.info("There are no valid maturity dates with positive exposure.")
        return

    today = pd.Timestamp.today().normalize()
    maturity_df['Days to maturity'] = (
        maturity_df['Maturity date'] - today
    ).dt.days

    max_maturity = maturity_df['Maturity date'].max()
    maturity_df['Term bucket'] = maturity_df['Maturity date'].apply(
        calendar_year_bucket, args=(today,)
    )

    total_exposure = maturity_df['US $ Equiv'].sum()
    weights = maturity_df['US $ Equiv']
    wal_days = (
        (maturity_df['Days to maturity'] * weights).sum() / total_exposure
        if total_exposure
        else float('nan')
    )
    wal_years = wal_days / 365 if pd.notna(wal_days) else float('nan')
    weighted_maturity_date = (
        (today + pd.Timedelta(days=wal_days)) if pd.notna(wal_days) else None
    )

    matured_exposure = maturity_df.loc[
        maturity_df['Days to maturity'] < 0, 'US $ Equiv'
    ].sum()
    matured_share = (matured_exposure / total_exposure * 100) if total_exposure else 0

    upcoming_df = maturity_df[maturity_df['Days to maturity'] >= 0]
    next_maturity = (
        upcoming_df.loc[upcoming_df['Days to maturity'].idxmin(), 'Maturity date']
        if not upcoming_df.empty
        else None
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Weighted Average Life (years)", f"{wal_years:.2f}" if pd.notna(wal_years) else "N/A")
    c2.metric(
        "Weighted average maturity",
        weighted_maturity_date.strftime("%d-%b-%Y") if weighted_maturity_date else "N/A",
    )
    c3.metric(
        "Expired exposure",
        format_currency(matured_exposure),
        f"{matured_share:.1f}% of total",
    )
    c4.metric(
        "Next maturity",
        next_maturity.strftime("%d-%b-%Y") if next_maturity else "N/A",
    )
    c5.metric(
        "Maximum maturity",
        max_maturity.strftime("%d-%b-%Y") if pd.notna(max_maturity) else "N/A",
    )

    bucket_order = ["Expired"] + [
        str(year) for year in range(today.year, max_maturity.year + 1)
    ]
    bucket_summary = (
        maturity_df.groupby('Term bucket')['US $ Equiv']
        .sum()
        .reset_index(name='Exposure')
    )
    bucket_summary['Share'] = bucket_summary['Exposure'] / total_exposure
    bucket_summary['Year (t=0 current)'] = bucket_summary['Term bucket'].apply(
        lambda x: (int(x) - today.year) if str(x).isdigit() else None
    )
    bucket_summary['Term bucket'] = pd.Categorical(
        bucket_summary['Term bucket'], bucket_order
    )
    bucket_summary = bucket_summary.sort_values('Term bucket').dropna(
        subset=['Term bucket']
    )

    st.markdown("### Exposure by term (calendar years)")
    st.caption(
        f"Buckets are defined by calendar year starting on 1-Jan-{today.year}. "
        "Bucket totals correspond to full years."
    )
    bucket_display = bucket_summary.assign(
        Exposure=bucket_summary['Exposure'].apply(format_currency),
        **{
            "Year (t=0 current)": bucket_summary['Year (t=0 current)'].apply(
                lambda v: f"{int(v):,}" if pd.notna(v) else "N/A"
            )
        },
        Share=bucket_summary['Share'].apply(lambda x: f"{x:.1%}"),
    )[
        ["Term bucket", "Year (t=0 current)", "Exposure", "Share"]
    ]

    st.dataframe(bucket_display, use_container_width=True)

    maturity_df['Maturity year'] = maturity_df['Maturity date'].dt.year
    year_summary = (
        maturity_df.groupby('Maturity year')['US $ Equiv']
        .sum()
        .reset_index(name='Exposure')
    )
    year_summary = year_summary[year_summary['Exposure'] > 0]

    if not year_summary.empty:
        chart = (
            alt.Chart(year_summary)
            .mark_bar(color=next_bar_color())
            .encode(
                x=alt.X('Maturity year:O', title='Maturity year'),
                y=alt.Y('Exposure:Q', title='Exposure (US$)'),
                tooltip=[
                    alt.Tooltip('Maturity year:O', title='Year'),
                    alt.Tooltip('Exposure:Q', format=",.0f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)


def render_eligible_activity_analysis(df: pd.DataFrame):
    """Group exposure by Eligible Activity classification.

    All Eligible Activity columns are consolidated into a single cumulative
    dimension, regardless of the column number (1-6)."""

    st.header("Eligible Activities analysis")

    eligible_cols = [
        c for c in df.columns if c.lower().startswith("elegible activities")
    ]

    if not eligible_cols:
        st.info(
            "No Eligible Activities columns were found in the file for analysis."
        )
        return

    eligible_df = df[df['US $ Equiv'] > 0].copy()
    if eligible_df.empty:
        st.info("Only zero values are available; analysis cannot be displayed.")
        return

    melted = (
        eligible_df.reset_index()
        .melt(
            id_vars=['index', 'US $ Equiv'],
            value_vars=eligible_cols,
            var_name='Campo Eligible Activity',
            value_name='Eligible Activity',
        )
    )

    melted['Eligible Activity'] = (
        melted['Eligible Activity'].astype(str).str.strip()
    )

    valid_mask = (
        melted['Eligible Activity'].notna()
        & (melted['Eligible Activity'] != "")
        & (~melted['Eligible Activity'].isin(["nan", "0", "0.0"]))
    )
    melted = melted[valid_mask]

    if melted.empty:
        st.info("There are no non-zero or non-empty Eligible Activity values.")
        return

    melted = melted.drop_duplicates(subset=['index', 'Eligible Activity'])

    options = sorted(melted['Eligible Activity'].unique())
    selected = st.multiselect(
        "Eligible Activities to display",
        options,
        default=options,
        help=(
            "Classifications are cumulative across columns; selecting one category "
            "adds all operations that include it."
        ),
    )

    filtered = melted[melted['Eligible Activity'].isin(selected)] if selected else melted

    summary = (
        filtered.groupby('Eligible Activity')
        .agg(
            Operations=('index', 'nunique'),
            Exposure=('US $ Equiv', 'sum'),
        )
        .reset_index()
    )

    summary = summary[summary['Exposure'] > 0].sort_values('Exposure', ascending=False)

    if summary.empty:
        st.info("The selected categories do not have positive exposure.")
        return

    summary['Share'] = summary['Exposure'] / summary['Exposure'].sum()

    chart = (
            alt.Chart(summary)
            .mark_bar(color=next_bar_color())
            .encode(
                x=alt.X('Exposure:Q', title='Exposure (US$)'),
                y=alt.Y('Eligible Activity:N', sort='-x', title='Eligible Activity'),
                tooltip=[
                    'Eligible Activity:N',
                    alt.Tooltip('Exposure:Q', format=",.0f"),
                    alt.Tooltip('Operations:Q', title='Number of operations'),
                    alt.Tooltip('Share:Q', format=".1%"),
                ],
            )
        )

    st.altair_chart(chart, use_container_width=True)

    table_display = summary.assign(
        **{
            'Exposure (US$)': summary['Exposure'].apply(format_currency),
            'Share': summary['Share'].apply(lambda x: f"{x:.1%}"),
        }
    )[
        ['Eligible Activity', 'Operations', 'Exposure (US$)', 'Share']
    ]

    st.markdown("**Eligible Activity details**")
    st.dataframe(table_display, use_container_width=True)

def render_pvt_sector_analysis(df: pd.DataFrame):
    """Analyze exposure to sectors tagged as PVT in "Sector 2"."""

    st.header("Exposure to PVT sectors (Sector 2)")

    if "Sector 2" not in df.columns:
        st.info("The file does not contain the 'Sector 2' column for this analysis.")
        return

    df_pos = df[df["US $ Equiv"] > 0].copy()
    if df_pos.empty:
        st.info("Only zero values are available; analysis cannot be displayed.")
        return

    pvt_df = df_pos[df_pos["Sector 2"].str.contains("PVT", case=False, na=False)]
    if pvt_df.empty:
        st.info("No records in Sector 2 contain 'PVT'.")
        return

    total_filtered = df_pos["US $ Equiv"].sum()
    pvt_total = pvt_df["US $ Equiv"].sum()
    portfolio_share = (pvt_total / total_filtered * 100) if total_filtered else 0
    nominal_sum = pvt_df["US $ Equiv"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total PVT exposure (US$)", format_currency(pvt_total))
    c2.metric("PVT records", f"{len(pvt_df):,}")
    c3.metric("Portfolio share", f"{portfolio_share:.2f}%")
    st.markdown(f"**Nominal PVT sum (US$):** {format_currency(nominal_sum)}")

    summary = (
        pvt_df.groupby("Sector 2")
        .agg(
            Operations=("Sector 2", "count"),
            Exposure=("US $ Equiv", "sum"),
        )
        .reset_index()
    )

    summary = summary[summary["Exposure"] > 0].sort_values("Exposure", ascending=False)
    summary["PVT share"] = summary["Exposure"] / pvt_total

    chart = (
        alt.Chart(summary)
        .mark_bar(color=next_bar_color())
        .encode(
            x=alt.X("Exposure:Q", title="Exposure (US$)"),
            y=alt.Y("Sector 2:N", sort="-x", title="PVT category"),
            tooltip=[
                "Sector 2:N",
                alt.Tooltip("Exposure:Q", format=",.0f"),
                alt.Tooltip("Operations:Q", title="Number of operations"),
                alt.Tooltip("PVT share:Q", format=".1%"),
            ],
        )
    )

    st.altair_chart(chart, use_container_width=True)

    table_display = summary.assign(
        **{
            "Exposure (US$)": summary["Exposure"].apply(format_currency),
            "PVT share": summary["PVT share"].apply(lambda x: f"{x:.1%}"),
        }
    )[
        ["Sector 2", "Operations", "Exposure (US$)", "PVT share"]
    ]

    st.markdown("**PVT category details in Sector 2**")
    st.dataframe(table_display, use_container_width=True)


def render_portfolio_summary(total_full: float, df_filtered: pd.DataFrame):
    """Show a comparison between the full portfolio and the current filters."""

    filtered_total = df_filtered['US $ Equiv'].sum()
    participation = (filtered_total / total_full * 100) if total_full else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total portfolio (US$)", format_currency(total_full))
    c2.metric("Filtered exposure (US$)", format_currency(filtered_total))
    c3.metric("Portfolio share", f"{participation:.1f}%")

# ============================================
# RENDER: BREAKDOWNS (BARRAS + PIE)
# ============================================

def render_breakdown(df, column, title, label, include_pie=True, show_table=False):
    st.subheader(title)

    options = sorted(df[column].unique())
    selected_options = st.multiselect(
        f"{label} a mostrar", options, default=options, key=f"breakdown_{column}"
    )
    values = selected_options if selected_options else options

    col1, col2 = st.columns(2)

    # ----- BARRAS -----
    with col1:
        df_f = df[df[column].isin(values)]
        if df_f.empty:
            st.info("There is no data to display in this chart.")
            return

        g = (
            df_f.groupby(column)['US $ Equiv']
            .sum()
            .reset_index()
            .sort_values('US $ Equiv', ascending=False)
        )

        g = g[g['US $ Equiv'] > 0]
        if g.empty:
            st.info("Only zero values exist for this selection.")
            return

        total = g['US $ Equiv'].sum()
        g['Porcentaje'] = g['US $ Equiv'] / total

        chart = (
            alt.Chart(g)
            .mark_bar(color=next_bar_color())
            .encode(
                x=alt.X("US $ Equiv:Q", title="Exposure (US$)"),
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
            df_p = df[df[column].isin(values)]
            if df_p.empty:
                st.info("There is no data to display in the pie chart.")
                return

            g2 = (
                df_p.groupby(column)['US $ Equiv']
                .sum()
                .reset_index()
            )

            g2 = g2[g2['US $ Equiv'] > 0]
            if g2.empty:
                st.info("Only zero values exist for this selection.")
                return

            total_pie = g2['US $ Equiv'].sum()
            if total_pie <= 0:
                st.info("There are no positive values for the pie chart.")
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
                color = color_map.get(row[column], BRAND_COLORS["muted"])
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

    if show_table:
        table = (
            df[df['US $ Equiv'] > 0]
            .groupby(column)['US $ Equiv']
            .sum()
            .reset_index(name='Exposure')
            .sort_values('Exposure', ascending=False)
        )

        if table.empty:
            st.info(f"There is no positive exposure to show by {label.lower()}.")
            return

        total_table = table['Exposure'].sum()
        table_display = table.assign(
            **{
                "Exposure (US$)": table['Exposure'].apply(format_currency),
                "Share": table['Exposure'].apply(
                    lambda v: f"{(v / total_table):.1%}" if total_table else "0.0%"
                ),
            }
        )[[column, "Exposure (US$)", "Share"]]

        st.markdown(f"**Exposure details by {label.lower()}**")
        st.dataframe(table_display.rename(columns={column: label}), use_container_width=True)


# ============================================
# RENDER: HEATMAP
# ============================================

def render_heatmap(df):
    st.subheader("Country vs Sector heatmap")

    g = df.groupby(['Country', 'Sector'])['US $ Equiv'].sum().reset_index()

    heat = (
        alt.Chart(g)
        .mark_rect()
        .encode(
            x=alt.X("Sector:N"),
            y=alt.Y("Country:N"),
            color=alt.Color(
                "US $ Equiv:Q",
                scale=alt.Scale(range=["#e8eef2", BRAND_COLORS["secondary"], BRAND_COLORS["primary"]]),
            ),
            tooltip=["Country", "Sector", alt.Tooltip("US $ Equiv:Q", format=",.0f")],
        )
        .properties(height=380)
    )

    st.altair_chart(heat, use_container_width=True)


def render_orr_heatmap(df):
    st.subheader("Weighted ORR heatmap (Country vs Sector)")

    if 'ORR_num' not in df.columns:
        st.info("No ORR data is available to calculate the heatmap.")
        return

    df_orr = df[(df['ORR_num'].notna()) & (df['US $ Equiv'] > 0)].copy()
    if df_orr.empty:
        st.info("There are no records with ORR and positive exposure for this selection.")
        return

    grouped = (
        df_orr.groupby(['Country', 'Sector'])
        .apply(
            lambda g: pd.Series(
                {
                    "ORR ponderado": weighted_avg_orr(g),
                    "Exposure": g['US $ Equiv'].sum(),
                }
            )
        )
        .reset_index()
    )

    grouped = grouped[grouped['Exposure'] > 0]
    if grouped.empty:
        st.info("There is no positive exposure to plot in the ORR heatmap.")
        return

    heat = (
        alt.Chart(grouped)
        .mark_rect()
        .encode(
            x=alt.X("Sector:N"),
            y=alt.Y("Country:N"),
            color=alt.Color(
                "ORR ponderado:Q",
                scale=alt.Scale(range=["#fff5e6", BRAND_COLORS["highlight"], BRAND_COLORS["accent"]]),
            ),
            tooltip=[
                "Country",
                "Sector",
                alt.Tooltip("ORR ponderado:Q", format=".2f"),
                alt.Tooltip("Exposure:Q", format=",.0f"),
            ],
        )
        .properties(height=380)
    )

    st.altair_chart(heat, use_container_width=True)

# ============================================
# RENDER: ORR BY DIMENSION
# ============================================

def render_orr_by_dimension(df):
    st.subheader("ORR by dimension")

    if 'ORR_num' not in df.columns:
        st.info("No ORR data is available to plot.")
        return

    dims = {
        "Country": "Country",
        "Segment": "Segment",
        "Product Type": "Product type",
        "Sector": "Sector",
        "Sector 2": "Sector 2",
    }

    dimension = st.selectbox(
        "Dimension to plot ORR",
        options=list(dims.keys()),
        format_func=lambda x: dims[x],
        key="orr_dimension_selector",
    )

    # Select specific categories within the chosen dimension
    category_options = sorted(df[dimension].unique())
    selected_categories = st.multiselect(
        f"{dims[dimension]} to display",
        category_options,
        default=category_options,
        key=f"orr_categories_{dimension}",
    )

    df_selected = df[df[dimension].isin(selected_categories)]
    if df_selected.empty:
        st.info("No data is available for the selected options.")
        return

    # Total exposure per category (includes records without ORR)
    exposure_total = (
        df_selected.groupby(dimension)["US $ Equiv"].sum().reset_index(name="Total exposure")
    )
    exposure_total = exposure_total[exposure_total["Total exposure"] > 0]

    if exposure_total.empty:
        st.info("There is no positive exposure to show for this dimension.")
        return

    # Records with ORR available for the weighted calculation
    df_orr = df_selected[df_selected['ORR_num'].notna()].copy()
    if df_orr.empty:
        st.info("No ORR data is available to plot.")
        return

    orr_grouped = (
        df_orr.groupby(dimension)
        .apply(
            lambda g: pd.Series(
                {
                    "ORR ponderado": weighted_avg_orr(g),
                    "Exposure with ORR": g["US $ Equiv"].sum(),
                }
            )
        )
        .reset_index()
    )

    # Join total exposure with weighted ORR, keeping only categories with ORR
    grouped = exposure_total.merge(orr_grouped, on=dimension, how="inner")
    if grouped.empty:
        st.info("The selected categories do not have ORR available.")
        return

    grouped = grouped.sort_values("ORR ponderado", ascending=False)

    chart = (
        alt.Chart(grouped)
        .mark_bar(color=next_bar_color())
        .encode(
            x=alt.X("ORR ponderado:Q", title="Weighted ORR by exposure"),
            y=alt.Y(f"{dimension}:N", sort="-x", title=dims[dimension]),
            tooltip=[
                f"{dimension}:N",
                alt.Tooltip("ORR ponderado:Q", format=".2f"),
                alt.Tooltip("Total exposure:Q", format=",.0f"),
                alt.Tooltip("Exposure with ORR:Q", format=",.0f"),
            ],
        )
    )

    st.altair_chart(chart, use_container_width=True)


def render_exposure_by_dimension(df):
    st.subheader("Exposure (US$) by dimension")

    dims = {
        "Country": "Country",
        "Segment": "Segment",
        "Product Type": "Product type",
        "Sector": "Sector",
        "Sector 2": "Sector 2",
        "Delinq band": "Delinq band (ranges)",
        "Delinq band simple": "Delinquent? (Yes/No)",
    }

    available_dims = [d for d in dims if d in df.columns]
    if not available_dims:
        st.info("There are no dimensions available to plot.")
        return

    dimension = st.selectbox(
        "Dimension to plot exposure",
        options=available_dims,
        format_func=lambda x: dims[x],
        key="exposure_dimension_selector",
    )

    category_options = sorted(df[dimension].unique())
    selected_categories = st.multiselect(
        f"{dims[dimension]} to display",
        category_options,
        default=category_options,
        key=f"exposure_categories_{dimension}",
    )

    df_selected = df[df[dimension].isin(selected_categories)]
    if df_selected.empty:
        st.info("No data is available for the selected options.")
        return

    exposure = (
        df_selected.groupby(dimension)["US $ Equiv"].sum().reset_index(name="Exposure")
    )
    exposure = exposure[exposure["Exposure"] > 0]

    if exposure.empty:
        st.info("There is no positive exposure to show for this dimension.")
        return

    exposure = exposure.sort_values("Exposure", ascending=False)

    chart = (
        alt.Chart(exposure)
        .mark_bar(color=next_bar_color())
        .encode(
            x=alt.X("Exposure:Q", title="Exposure (US$)"),
            y=alt.Y(f"{dimension}:N", sort="-x", title=dims[dimension]),
            tooltip=[f"{dimension}:N", alt.Tooltip("Exposure:Q", format=",.0f")],
        )
    )

    st.altair_chart(chart, use_container_width=True)

# ============================================
# RENDER: TOP / BOTTOM
# ============================================

def render_top_bottom(df, n=10):
    st.subheader("Top / bottom operations")

    amt = "US $ Equiv"
    ordered = df.sort_values(amt, ascending=False)

    top = ordered.head(n).copy()
    bottom = ordered[ordered[amt] > 0].tail(n).copy()

    if top.empty or bottom.empty:
        st.info("There is not enough data for the top/bottom view.")
        return

    top[amt] = top[amt].apply(format_currency)
    bottom[amt] = bottom[amt].apply(format_currency)

    c1, c2 = st.columns(2)
    c1.markdown("### Top 10")
    c1.dataframe(top, use_container_width=True)

    c2.markdown("### Bottom 10 (only values > 0)")
    c2.dataframe(bottom, use_container_width=True)


# ============================================
# APP PRINCIPAL
# ============================================

logo_b64 = apply_brand_styling()

df = load_portfolio_data(
    PORTFOLIO_FILE,
    PORTFOLIO_FILE.stat().st_mtime,
)

title_html = """
<div class="title-with-logo">
    <h1>Corporate Portfolio Analysis</h1>
"""

if logo_b64:
    title_html += f'<img src="data:image/jpeg;base64,{logo_b64}" alt="Logo" class="title-logo" />'

title_html += "</div>"

st.markdown(title_html, unsafe_allow_html=True)
st.markdown(
    '<p class="page-caption">Dynamic filters, concentration, KPIs, and breakdown by dimensions.</p>',
    unsafe_allow_html=True,
)

total_portfolio = df['US $ Equiv'].sum()

# ------------------------------
# SIDEBAR FILTERS
# ------------------------------

# Use only non-zero exposures for the primary filters
fdf = df[df['US $ Equiv'] != 0].copy()

filters = {
    "Country": "Country",
    "Segment": "Segment",
    "Product Type": "Product type",
    "Sector": "Sector",
}

with st.sidebar.expander("Filters (multiple selection)", expanded=True):
    for col, label in filters.items():
        options = sorted(fdf[col].unique())
        state_key = f"filter_{col}"
        button_key = f"select_all_{col}"

        if state_key not in st.session_state:
            st.session_state[state_key] = options

        if st.button(
            "Select all",
            key=button_key,
            help=f"Select all values for {label}",
            use_container_width=True,
        ):
            st.session_state[state_key] = options

        sel = st.multiselect(label, options, default=st.session_state[state_key], key=state_key)
        fdf = fdf[fdf[col].isin(sel)]

if fdf.empty:
    st.warning("No data is available with this combination of filters.")
    st.stop()

# ------------------------------
# SECCIONES
# ------------------------------

st.header("General summary")
render_portfolio_summary(total_portfolio, fdf)
st.markdown("### Filtered KPIs")
render_kpis(fdf)

plot_df = fdf[fdf['US $ Equiv'] > 0]

if plot_df.empty:
    st.warning("Only zero values remain after applying filters; charts will not be shown.")
else:
    st.header("Breakdown by dimensions")

    render_exposure_by_dimension(plot_df)
    st.divider()

    render_orr_by_dimension(plot_df)
    st.divider()

    render_breakdown(plot_df, "Country", "Exposure by country", "Country")
    render_breakdown(plot_df, "Segment", "Exposure by segment", "Segment")
    render_breakdown(plot_df, "Product Type", "Exposure by product type", "Product type")
    render_breakdown(plot_df, "Sector", "Exposure by sector", "Sector")
    render_breakdown(
        plot_df, "Sector 2", "Exposure by Sector 2", "Sector 2", show_table=True
    )
    render_breakdown(plot_df, "Delinq band", "Exposure by Delinq band", "Delinq band")

    st.divider()

    render_heatmap(plot_df)

    render_orr_heatmap(plot_df)

    st.divider()

    render_maturity_analysis(plot_df)

    st.divider()

    render_eligible_activity_analysis(plot_df)

    st.divider()

    render_pvt_sector_analysis(plot_df)

    st.divider()

    render_top_bottom(plot_df)

    st.divider()

st.header("Full detail")
st.dataframe(fdf, use_container_width=True)

