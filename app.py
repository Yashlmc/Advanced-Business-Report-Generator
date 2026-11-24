import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from analysis import analyze_sales, clean_sales_data
from forecasting import detect_anomalies, forecast_revenue
from query_agent import ask_sales_question
from report_generator import generate_report, verify_report_for_export


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Advanced Business Report Generator",
    page_icon="📊",
    layout="wide",
)

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()

groq_client = Groq(api_key=API_KEY)


# ==========================================
# HEADER
# ==========================================

st.title("📊 Advanced Business Report Generator")

st.markdown(
    """
    An AI-powered business intelligence application featuring sales analysis,
    chronological ML revenue forecasting (with Naive baseline comparison),
    anomaly detection, natural-language data queries, and numerically grounded
    Groq AI reporting.
    """
)

st.divider()


# ==========================================
# CSV UPLOAD
# ==========================================

st.subheader("📁 Upload Sales Dataset")

uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
    elif os.path.exists("sample_sales.csv"):
        df = pd.read_csv("sample_sales.csv")
        dataset_name = "sample_sales.csv"
        st.info("Using default dataset: sample_sales.csv")
    else:
        st.warning("Please upload a CSV sales dataset.")
        st.stop()

    # Clear all cached results when a different dataset is loaded
    data_signature = (dataset_name, len(df), tuple(df.columns))

    if st.session_state.get("data_signature") != data_signature:
        for key in [
            "business_report",
            "report_verification",
            "query_result",
            "query_plan",
            "forecast",
            "forecast_metrics",
            "anomaly_data",
            "anomalies",
        ]:
            st.session_state.pop(key, None)

        st.session_state["data_signature"] = data_signature

    # ==========================================
    # DATA CLEANING & ANALYSIS
    # ==========================================

    clean_df = clean_sales_data(df)
    summary, monthly_sales, top_products = analyze_sales(clean_df)

    st.success(f"Dataset loaded successfully: {dataset_name}")
    st.caption(
        f"Valid records: {len(clean_df):,} | "
        f"Products: {clean_df['Product'].nunique():,} | "
        f"Period: {summary['reporting_period']['start']} to "
        f"{summary['reporting_period']['end']}"
    )

    with st.expander("View sales data preview"):
        st.dataframe(clean_df.head(20), use_container_width=True)

    # ==========================================
    # BUSINESS OVERVIEW
    # ==========================================

    st.divider()
    st.header("📈 Business Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue", f"₹{summary['total_revenue']:,.2f}")
    col2.metric("Units Sold", f"{summary['total_units_sold']:,}")
    col3.metric("Sales Records", f"{summary['number_of_sales_records']:,}")
    col4.metric("Products", f"{summary['number_of_products']:,}")
    col5.metric("Avg Sale Value", f"₹{summary['average_sale_value']:,.2f}")

    # ==========================================
    # HISTORICAL MONTHLY SALES CHART
    # ==========================================

    st.divider()
    st.header("📊 Historical Monthly Revenue")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(
        monthly_sales["Month"],
        monthly_sales["Revenue"],
        marker="o",
        linewidth=2,
        color="#1D4ED8",
        markersize=6,
    )
    ax.set_title("Monthly Sales Performance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (₹)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ==========================================
    # TOP PRODUCTS
    # ==========================================

    st.divider()
    st.header("🏆 Top Performing Products")
    st.dataframe(top_products, use_container_width=True, hide_index=True)

    # ==========================================
    # REVENUE FORECASTING + BASELINE
    # ==========================================

    st.divider()
    st.header("🔮 Revenue Forecasting (Random Forest vs Naive Baseline)")

    st.write(
        "Evaluates a lag-featured Random Forest Regressor against a "
        "chronological Naive Persistence baseline before forecasting future months."
    )

    forecast_period = st.selectbox(
        "Select forecast horizon (months)",
        [1, 2, 3, 4, 5, 6],
        index=2,
    )

    if st.button("Generate Revenue Forecast", use_container_width=True):
        try:
            with st.spinner("Training & backtesting forecasting model..."):
                forecast_df, metrics = forecast_revenue(
                    clean_df, periods=forecast_period
                )
                st.session_state["forecast"] = forecast_df
                st.session_state["forecast_metrics"] = metrics
        except Exception as e:
            st.error(str(e))

    if "forecast" in st.session_state:
        forecast_df = st.session_state["forecast"]
        metrics = st.session_state["forecast_metrics"]

        st.subheader(
            f"Chronological Hold-Out Evaluation (Last {metrics['holdout_months']} Months)"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Random Forest MAE", f"₹{metrics['rf_mae']:,.2f}")
        m2.metric("Naive Baseline MAE", f"₹{metrics['baseline_mae']:,.2f}")
        m3.metric(
            "Random Forest MAPE",
            f"{metrics['rf_mape']}%" if metrics["rf_mape"] is not None else "N/A",
        )
        m4.metric(
            "Naive Baseline MAPE",
            (
                f"{metrics['baseline_mape']}%"
                if metrics["baseline_mape"] is not None
                else "N/A"
            ),
        )

        st.subheader("Predicted Monthly Revenue")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        fig_f, ax_f = plt.subplots(figsize=(11, 4.2))
        ax_f.plot(
            monthly_sales["Month"],
            monthly_sales["Revenue"],
            marker="o",
            label="Historical Actual",
            color="#1D4ED8",
            linewidth=2,
        )
        ax_f.plot(
            forecast_df["Month"],
            forecast_df["Forecast_Revenue"],
            marker="s",
            linestyle="--",
            label="Random Forest Forecast",
            color="#0F766E",
            linewidth=2,
        )
        ax_f.set_title("Historical vs Forecasted Monthly Revenue", fontsize=13)
        ax_f.set_xlabel("Month")
        ax_f.set_ylabel("Revenue (₹)")
        ax_f.legend()
        ax_f.grid(True, linestyle="--", alpha=0.4)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_f)
        plt.close(fig_f)

    # ==========================================
    # ANOMALY DETECTION
    # ==========================================

    st.divider()
    st.header("🚨 Sales Anomaly Detection")

    st.write(
        "Isolation Forest flags months with unusually high or low revenue."
    )

    if st.button("Detect Sales Anomalies", use_container_width=True):
        try:
            with st.spinner("Detecting unusual sales patterns..."):
                monthly_anomalies, anomalies = detect_anomalies(clean_df)
                st.session_state["anomaly_data"] = monthly_anomalies
                st.session_state["anomalies"] = anomalies
        except Exception as e:
            st.error(str(e))

    if "anomaly_data" in st.session_state:
        anomalies = st.session_state["anomalies"]

        if anomalies.empty:
            st.success("No unusual monthly revenue patterns were detected.")
        else:
            st.warning(
                f"{len(anomalies)} unusual month(s) detected. "
                "These are statistical flags, not proof of a business problem."
            )
            st.dataframe(
                anomalies[["Month", "Revenue", "Status"]],
                use_container_width=True,
                hide_index=True,
            )

    # ==========================================
    # NATURAL LANGUAGE QUERY AGENT
    # ==========================================

    st.divider()
    st.header("💬 Ask Questions About Your Sales")

    st.write(
        "Ask questions in plain English. Groq converts your question into a "
        "validated JSON query plan executed safely in Pandas (zero eval/exec)."
    )

    question = st.text_input(
        "Enter your business question",
        placeholder="Which product generated the highest revenue?",
    )

    if st.button("Ask Sales Assistant", use_container_width=True):
        try:
            with st.spinner("Understanding your question..."):
                plan, result = ask_sales_question(
                    clean_df, question, groq_client
                )
                st.session_state["query_plan"] = plan
                st.session_state["query_result"] = result
        except Exception as e:
            st.error(f"Unable to process question: {str(e)}")

    if "query_result" in st.session_state:
        result = st.session_state["query_result"]
        st.subheader("Query Result")

        if isinstance(result["result"], list):
            result_df = pd.DataFrame(result["result"])
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        else:
            metric_name = result["metric"]
            val = result["result"]
            if metric_name in {"revenue", "average_sale_value"}:
                display_val = f"₹{val:,.2f}"
            elif metric_name == "records":
                display_val = f"{int(val):,}"
            else:
                display_val = f"{val:,.2f}"

            st.metric(metric_name.replace("_", " ").title(), display_val)

        with st.expander("View validated query plan"):
            st.json(st.session_state["query_plan"])

    # ==========================================
    # GROQ BUSINESS REPORT + GROUNDING GATE
    # ==========================================

    st.divider()
    st.header("🤖 AI-Generated Business Report")

    st.write(
        "Generate a professional Markdown report using calculated metrics, "
        "verified automatically against dataset values before export."
    )

    if st.button(
        "Generate Business Report",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Groq is preparing and verifying your report..."):
                report, verification = generate_report(
                    summary, clean_df, client=groq_client
                )
                st.session_state["business_report"] = report
                st.session_state["report_verification"] = verification
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")

    if "business_report" in st.session_state:
        report = st.session_state["business_report"]
        verification = st.session_state["report_verification"]

        st.subheader("Business Report")
        st.markdown(report)

        st.divider()
        st.subheader("🛡️ Numeric Grounding Verification")

        if verification["verified"]:
            verify_report_for_export(report, clean_df, summary)
            st.success(verification["message"])
            st.write(f"Figures checked: {verification['total_figures']}")

            st.download_button(
                label="📥 Download Verified Report",
                data=report,
                file_name="verified_business_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.error(
                "Report verification failed. Downloading this report is disabled."
            )
            st.write(f"Figures checked: {verification['total_figures']}")
            st.write("Unsupported figures:", verification["unsupported_figures"])

    # ==========================================
    # FOOTER
    # ==========================================

    st.divider()
    st.caption(
        "Advanced Business Report Generator | "
        "Python | Pandas | Scikit-learn | Groq AI | Streamlit"
    )

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Check your CSV columns, data types, and Python dependencies.")