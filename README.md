# Advanced Business Report Generator

An AI-powered Business Intelligence and automated reporting workbench built with **Python**, **Pandas**, **Scikit-Learn**, **Groq API**, and **Streamlit**. It converts raw sales CSV datasets into interactive KPI dashboards, chronological ML revenue forecasts (benchmarked against a Naive Persistence baseline), monthly anomaly flags, sandboxed natural-language data queries, and numerically verified executive Markdown reports.

## Key Capabilities

* **Automated Sales Analytics (`analysis.py`):** Cleans and validates uploaded CSV sales records, computes total revenue, units sold, average sale value, top-five product contribution shares, and month-over-month revenue growth percentages in a strict JSON-safe summary structure.

* **Chronological Revenue Forecasting (`forecasting.py`):** Trains a lag-featured `RandomForestRegressor` (lag-1, lag-2, lag-3, and time index) to forecast 1–6 months ahead and benchmarks it on a chronological hold-out split against a **Naive Persistence baseline** using MAE and MAPE.

* **Monthly Revenue Anomaly Detection (`forecasting.py`):** Applies `IsolationForest` to aggregated monthly revenue series (with zero-filled missing months) to flag statistically unusual spikes or drops.

* **Sandboxed Natural-Language Query Agent (`query_agent.py`):** Converts plain-English business questions into a restricted, validated JSON query plan (`aggregate`, `grouped`, `top_n`) executed purely via predefined Pandas operations—eliminating `eval`/`exec` and code-injection risks.

* **Numerically Grounded AI Reporting (`report_generator.py`, `grounding.py`):** Generates structured six-section Markdown business reports via Groq (`temperature=0.1`) using calculated metrics only, then audits every extracted number against approved dataset and summary figures (supporting international and Indian comma formats while ignoring structural headings and verified dates) before unlocking report download.

## Repository Structure

| File | Responsibility | 
| ----- | ----- | 
| `app.py` | Interactive Streamlit dashboard integrating KPIs, Matplotlib historical/forecast charts, anomaly tables, NL query UI, and the verified report export gate. | 
| `analysis.py` | Reusable `clean_sales_data()` validator and `analyze_sales()` KPI/trend engine with strict `NaN`-to-`None` JSON sanitization. | 
| `forecasting.py` | Monthly time-series preparation, chronological Persistence vs. Random Forest MAE/MAPE backtesting, multi-step forecasting, and Isolation Forest anomaly detection. | 
| `query_agent.py` | Groq JSON-mode query planner, strict schema/allowlist validator, and safe Pandas execution engine (`Product` and `Month` dimensions). | 
| `report_generator.py` | Lazy Groq client initialization, grounded business-report prompt orchestration, and mandatory export verification wrapper. | 
| `grounding.py` | Decimal-normalized numerical claim extractor and verifier blocking export whenever unsupported figures appear in the generated narrative. | 
| `sample_sales.csv` | Optional default sales dataset loaded automatically when no CSV is uploaded. | 

## Technology Stack

| Component | Technology | Role in System | 
| ----- | ----- | ----- | 
| Data Processing | Pandas & NumPy | CSV ingestion, type coercion, business-rule filtering, and monthly/product aggregations. | 
| Machine Learning | Scikit-Learn (`RandomForestRegressor`, `IsolationForest`) | Lag-based revenue forecasting, chronological MAE/MAPE evaluation, and unsupervised anomaly detection. | 
| LLM Inference | Groq API (`openai/gpt-oss-120b`) | Low-latency structured JSON query planning (`temperature=0`) and executive report synthesis (`temperature=0.1`). | 
| Numeric Grounding | Python `decimal` & `re` | Precision-safe verification of every numerical claim in generated reports against dataset ground truth. | 
| Visualization & UI | Streamlit & Matplotlib | Interactive web dashboard, historical vs. forecasted revenue charts, and controlled Markdown export. | 

## Input CSV Schema

Upload any UTF-8 CSV file containing the following required columns (at least 8 distinct months of history are required for forecasting and anomaly detection):

| Column | Type | Validation Rule | 
| ----- | ----- | ----- | 
| `Date` | Date string (`YYYY-MM-DD`) | Parsed via `pd.to_datetime`; invalid dates are dropped. | 
| `Product` | String | Whitespace-trimmed; blank values are dropped. | 
| `Quantity` | Numeric | Must be strictly greater than `0`. | 
| `UnitPrice` | Numeric | Must be greater than or equal to `0`. | 

## Quickstart

### 1. Prerequisites

* Python 3.10+

* A Groq API key (`GROQ_API_KEY`)

### 2. Installation

```
git clone <your-repository-url>
cd Business-Report-Generator
python -m pip install streamlit pandas numpy matplotlib scikit-learn groq python-dotenv

```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_actual_api_key
GROQ_MODEL=openai/gpt-oss-120b

```

### 4. Launch the Application

```
python -m streamlit run app.py

```

Upload your sales CSV (or place `sample_sales.csv` in the project directory) to view KPIs, run the Random Forest vs. Naive Baseline forecast, detect anomalies, query your data in natural language, and download a numerically verified Markdown business report.

## Verification & Safety Guardrails

* **Zero Code Execution in NL Queries:** The query agent never asks the LLM to write Python or SQL; only allowlisted actions (`aggregate`, `grouped`, `top_n`), metrics (`revenue`, `quantity`, `records`, `average_sale_value`), and dimensions (`product`, `month`) are executed.

* **Export-Gated Numeric Grounding:** `verify_report_for_export()` blocks Markdown download if the LLM introduces any unverified revenue figure, percentage, or projection absent from the dataset or summary.

* **Session-State Isolation:** Switching CSV files automatically clears cached forecasts, anomalies, query outputs, and reports to prevent cross-dataset leakage.

## Known Limitations & Roadmap

* **Minimum History Window:** Lag-3 forecasting and Isolation Forest anomaly detection require at least 8 months of historical sales data.

* **Profitability Metrics:** Cost-of-goods-sold (COGS) and margin columns are not assumed in the default schema, so the prompt strictly prohibits inferring profitability.

* **Future Enhancements:** Multi-product demand forecasting, seasonal decomposition (Prophet/LightGBM comparison), and PDF/Excel report export.