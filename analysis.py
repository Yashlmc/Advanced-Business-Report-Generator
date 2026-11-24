import pandas as pd


# ==================================================
# 1. REUSABLE DATA CLEANING FUNCTION
# ==================================================

def clean_sales_data(df):
    """
    Clean and validate sales data.

    Required columns:
        Date, Product, Quantity, UnitPrice

    Returns:
        Cleaned DataFrame containing:
        Date, Product, Quantity, UnitPrice,
        Revenue, Month
    """

    if df is None:
        raise ValueError("No sales data provided.")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    required_columns = [
        "Date",
        "Product",
        "Quantity",
        "UnitPrice",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["Product"] = df["Product"].astype("string").str.strip()

    df = df.dropna(subset=required_columns)

    df = df[
        (df["Product"] != "")
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] >= 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid sales records found after cleaning.")

    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df


# ==================================================
# 2. MAIN SALES ANALYSIS FUNCTION
# ==================================================

def analyze_sales(df):
    """
    Analyze sales data using Pandas.

    Returns:
        summary:
            JSON-safe dictionary containing business metrics.
        monthly_sales:
            DataFrame containing monthly revenue, units sold,
            and month-over-month growth percentage.
        top_products:
            DataFrame containing the top five products by revenue.
    """

    df = clean_sales_data(df)

    total_revenue = float(df["Revenue"].sum())
    total_quantity = float(df["Quantity"].sum())
    total_sales_records = int(len(df))
    average_sale_value = total_revenue / total_sales_records
    number_of_products = int(df["Product"].nunique())

    # ----------------------------------
    # TOP FIVE PRODUCTS
    # ----------------------------------

    product_sales = (
        df.groupby("Product")
        .agg(
            Revenue=("Revenue", "sum"),
            Units_Sold=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )

    product_sales["Revenue"] = product_sales["Revenue"].round(2)
    product_sales["Units_Sold"] = product_sales["Units_Sold"].round(2)

    if total_revenue > 0:
        product_sales["Revenue_Percentage"] = (
            product_sales["Revenue"] / total_revenue * 100
        ).round(2)
    else:
        product_sales["Revenue_Percentage"] = 0.0

    top_products = product_sales.head(5).copy().reset_index(drop=True)

    # ----------------------------------
    # MONTHLY SALES TREND
    # ----------------------------------

    monthly_sales = (
        df.groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Units_Sold=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("Month")
        .reset_index(drop=True)
    )

    monthly_sales["Revenue"] = monthly_sales["Revenue"].round(2)
    monthly_sales["Units_Sold"] = monthly_sales["Units_Sold"].round(2)
    monthly_sales["Growth_Percentage"] = (
        monthly_sales["Revenue"].pct_change() * 100
    ).round(2)

    # Convert NaN (e.g. first month's Growth_Percentage) to None
    # so json.dumps(..., allow_nan=False) never fails.
    monthly_sales_records = []
    for record in monthly_sales.to_dict(orient="records"):
        cleaned_record = {
            k: (None if pd.isna(v) else v)
            for k, v in record.items()
        }
        monthly_sales_records.append(cleaned_record)

    # ----------------------------------
    # HIGHEST AND LOWEST REVENUE MONTHS
    # ----------------------------------

    highest_month = monthly_sales.loc[monthly_sales["Revenue"].idxmax()]
    lowest_month = monthly_sales.loc[monthly_sales["Revenue"].idxmin()]

    highest_month_info = {
        "month": str(highest_month["Month"]),
        "revenue": round(float(highest_month["Revenue"]), 2),
    }

    lowest_month_info = {
        "month": str(lowest_month["Month"]),
        "revenue": round(float(lowest_month["Revenue"]), 2),
    }

    # ----------------------------------
    # REPORTING PERIOD
    # ----------------------------------

    reporting_period = {
        "start": df["Date"].min().strftime("%Y-%m-%d"),
        "end": df["Date"].max().strftime("%Y-%m-%d"),
    }

    # ----------------------------------
    # CREATE JSON-SAFE BUSINESS SUMMARY
    # ----------------------------------

    summary = {
        "total_revenue": round(total_revenue, 2),
        "total_units_sold": (
            int(total_quantity)
            if total_quantity.is_integer()
            else round(total_quantity, 2)
        ),
        "number_of_sales_records": total_sales_records,
        "average_sale_value": round(float(average_sale_value), 2),
        "number_of_products": number_of_products,
        "top_products": top_products.to_dict(orient="records"),
        "monthly_sales": monthly_sales_records,
        "highest_revenue_month": highest_month_info,
        "lowest_revenue_month": lowest_month_info,
        "reporting_period": reporting_period,
    }

    return summary, monthly_sales, top_products