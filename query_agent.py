import json
import os

from analysis import clean_sales_data


# ==================================================
# 1. ALLOWED QUERY OPERATIONS & COLUMN MAPPINGS
# ==================================================

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

ALLOWED_METRICS = {
    "revenue",
    "quantity",
    "records",
    "average_sale_value",
}

ALLOWED_DIMENSIONS = {
    "product",
    "month",
}

DIMENSION_COLUMN_MAP = {
    "product": "Product",
    "month": "Month",
}

ALLOWED_ACTIONS = {
    "aggregate",
    "grouped",
    "top_n",
}


# ==================================================
# 2. CREATE QUERY PLAN USING GROQ
# ==================================================

def create_query_plan(client, question, model=None):
    """
    Convert a natural-language sales question into a restricted
    JSON query plan using Groq. Never generates executable code.
    """

    if not isinstance(question, str):
        raise ValueError("Question must be a text string.")

    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    if len(question) > 1000:
        raise ValueError("Question is too long. Maximum 1000 characters.")

    prompt = f"""
Convert the user's sales question into a JSON query plan.

USER QUESTION:
{question}

Allowed actions:
- aggregate
- grouped
- top_n
- unsupported

Allowed metrics:
- revenue
- quantity
- records
- average_sale_value

Allowed dimensions:
- product
- month

Rules:
1. Never generate Python or SQL code.
2. Never generate expressions or commands.
3. Use only the allowed actions, metrics and dimensions.
4. Use aggregate for overall totals.
5. Use grouped for comparisons by product or month.
6. Use top_n for ranking products or months.
7. Default top_n limit to 5 (must be an integer between 1 and 10).
8. Use revenue for sales revenue questions.
9. Use quantity for units sold questions.
10. Use records for number of sales records.
11. Use average_sale_value for average sale questions.
12. Use product when grouping by products.
13. Use month when grouping by months.
14. If the question cannot be answered using these operations,
    return {{"action": "unsupported"}}.
15. Return valid JSON only.

Examples:
Question: Which product generated the highest revenue?
Response: {{"action": "top_n", "metric": "revenue", "dimension": "product", "limit": 1}}

Question: Show total revenue by product.
Response: {{"action": "grouped", "metric": "revenue", "dimension": "product"}}

Question: How many units were sold?
Response: {{"action": "aggregate", "metric": "quantity"}}

Question: Show the top three products by revenue.
Response: {{"action": "top_n", "metric": "revenue", "dimension": "product", "limit": 3}}

Question: What is the average sale value?
Response: {{"action": "aggregate", "metric": "average_sale_value"}}

Question: Show the highest revenue month.
Response: {{"action": "top_n", "metric": "revenue", "dimension": "month", "limit": 1}}
"""

    try:
        response = client.chat.completions.create(
            model=model or DEFAULT_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sales analytics query planner. "
                        "Convert business questions into restricted JSON query plans. "
                        "Never generate executable code. Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty response.")

        plan = json.loads(content)
        if not isinstance(plan, dict):
            raise ValueError("Invalid query plan returned by the model.")

        return plan

    except json.JSONDecodeError as e:
        raise ValueError(
            "Unable to parse the query plan returned by Groq."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to generate query plan: {str(e)}") from e


# ==================================================
# 3. VALIDATE QUERY PLAN
# ==================================================

def validate_query_plan(plan):
    """Validate structure and values of a model-generated query plan."""

    if not isinstance(plan, dict):
        raise ValueError("Query plan must be a dictionary.")

    action = plan.get("action")

    if action == "unsupported":
        raise ValueError("This question is not supported by the agent.")

    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported query action: {action}")

    metric = plan.get("metric")
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"Unsupported query metric: {metric}")

    dimension = plan.get("dimension")
    if action in {"grouped", "top_n"} and dimension not in ALLOWED_DIMENSIONS:
        raise ValueError(f"Unsupported grouping dimension: {dimension}")

    if action == "top_n":
        limit = plan.get("limit", 5)
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 10
        ):
            raise ValueError(
                "Invalid top_n limit. Limit must be an integer between 1 and 10."
            )

    return True


# ==================================================
# 4. EXECUTE QUERY PLAN USING PANDAS
# ==================================================

def execute_query_plan(df, plan):
    """
    Execute a validated query plan using predefined Pandas operations.
    No dynamic code execution is permitted.
    """

    if df is None or df.empty:
        raise ValueError("No sales data available for querying.")

    df = clean_sales_data(df)
    validate_query_plan(plan)

    action = plan["action"]
    metric = plan["metric"]
    dimension = plan.get("dimension")

    metric_map = {
        "revenue": ("Revenue", "sum"),
        "quantity": ("Quantity", "sum"),
        "records": ("Revenue", "count"),
        "average_sale_value": ("Revenue", "mean"),
    }

    column, operation = metric_map[metric]

    # ----------------------------------
    # AGGREGATE QUERY
    # ----------------------------------

    if action == "aggregate":
        if operation == "sum":
            value = df[column].sum()
        elif operation == "count":
            value = df[column].count()
        elif operation == "mean":
            value = df[column].mean()
        else:
            raise ValueError("Unsupported aggregation operation.")

        formatted_val = (
            int(value) if metric == "records" else round(float(value), 2)
        )

        return {
            "result": formatted_val,
            "metric": metric,
            "action": action,
        }

    # ----------------------------------
    # GROUPED / TOP_N QUERY
    # ----------------------------------

    dim_col = DIMENSION_COLUMN_MAP[dimension]

    grouped = (
        df.groupby(dim_col)
        .agg(Value=(column, operation))
        .reset_index()
        .sort_values("Value", ascending=False)
        .reset_index(drop=True)
    )

    if action == "top_n":
        limit = plan.get("limit", 5)
        grouped = grouped.head(limit)

    if metric == "records":
        grouped["Value"] = grouped["Value"].astype(int)
    else:
        grouped["Value"] = grouped["Value"].round(2)

    return {
        "result": grouped.to_dict(orient="records"),
        "metric": metric,
        "dimension": dimension,
        "action": action,
    }


# ==================================================
# 5. MAIN NATURAL LANGUAGE QUERY FUNCTION
# ==================================================

def ask_sales_question(df, question, client, model=None):
    """
    End-to-end natural-language sales query handler:
    validate -> plan -> validate plan -> execute in Pandas.
    """

    if not isinstance(question, str):
        raise ValueError("Question must be a text string.")

    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")
    if len(question) > 1000:
        raise ValueError("Question is too long. Maximum 1000 characters.")

    plan = create_query_plan(client, question, model=model)
    validate_query_plan(plan)
    result = execute_query_plan(df, plan)

    return plan, result