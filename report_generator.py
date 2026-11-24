import json
import os

from dotenv import load_dotenv
from groq import Groq

from grounding import validate_report_for_export, verify_report


# ==================================================
# 1. LAZY GROQ CLIENT INITIALIZATION
# ==================================================

load_dotenv()

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def get_groq_client(api_key=None):
    """
    Lazily initialize the Groq client so importing this module
    never crashes before Streamlit can display a UI error.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not found. Please add it to your .env file."
        )
    return Groq(api_key=key)


# ==================================================
# 2. GENERATE PROFESSIONAL SALES REPORT
# ==================================================

def generate_report(summary, df, client=None, model=None):
    """
    Generate a professional business report using calculated
    sales metrics and verify all numerical figures.

    Parameters:
        summary:
            JSON-safe dictionary of calculated metrics from analysis.py.
        df:
            Cleaned sales DataFrame used by grounding.py for verification.
        client:
            Optional initialized Groq client.
        model:
            Optional Groq model ID override.

    Returns:
        report:
            Generated Markdown business report.
        verification:
            Numeric verification results dictionary.
    """

    if not isinstance(summary, dict) or not summary:
        raise ValueError("Sales summary must be a non-empty dictionary.")

    if df is None or df.empty:
        raise ValueError("Sales DataFrame is empty or unavailable.")

    try:
        summary_json = json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as e:
        raise ValueError("Unable to serialize sales summary.") from e

    prompt = f"""
You are a professional business analyst.

Generate a detailed and professional business sales report
using ONLY the calculated metrics provided below.

SALES SUMMARY:
{summary_json}

REPORT STRUCTURE:
1. EXECUTIVE SUMMARY
- Summarize overall business performance using total revenue,
  total units sold, number of sales records, and average sale value.

2. PRODUCT PERFORMANCE
- Identify the top-performing products and compare their revenue
  using supplied figures.
- Highlight lower-revenue products where supported and discuss opportunities.

3. MONTHLY SALES ANALYSIS
- Identify the highest and lowest revenue months.
- Discuss monthly revenue trends using only supplied monthly figures
  and growth percentages.

4. BUSINESS INSIGHTS
- Highlight key sales observations, revenue opportunities, and challenges.
- Clearly distinguish actual observations from assumptions.

5. BUSINESS RECOMMENDATIONS
- Provide practical recommendations for marketing, inventory management,
  promotions, revenue improvement, and sales planning based strictly on
  available observations (no invented financial targets).

6. CONCLUSION
- Provide a concise overall business conclusion.

STRICT NUMERICAL GROUNDING RULES:
1. Use only numerical values explicitly provided in the sales summary.
2. Never invent numerical values, percentages, ratios, or future forecasts.
3. Do not introduce dates not supplied in the summary.
4. Do not claim products are profitable or unprofitable (cost data is absent).
5. Clearly distinguish facts from assumptions.
6. If information is unavailable, state that the available data does not establish it.
7. Do not introduce external market statistics.

FORMATTING INSTRUCTIONS:
- Use Markdown headings and concise bullet points.
- Return plain Markdown only.
"""

    groq_client = client or get_groq_client()

    try:
        response = groq_client.chat.completions.create(
            model=model or DEFAULT_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful professional business analyst. "
                        "Generate accurate sales reports using only supplied "
                        "business metrics. Never fabricate numerical or "
                        "financial information. Follow all grounding rules."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=2500,
        )

        report = response.choices[0].message.content
        if not report or not report.strip():
            raise ValueError("Groq returned an empty report.")

        report = report.strip()

    except Exception as e:
        raise RuntimeError(
            f"Unable to generate the business report: {str(e)}"
        ) from e

    try:
        verification = verify_report(report, df, summary)
    except Exception as e:
        raise RuntimeError(
            f"Report numerical verification failed: {str(e)}"
        ) from e

    return report, verification


# ==================================================
# 3. VALIDATE REPORT BEFORE EXPORT
# ==================================================

def verify_report_for_export(report, df, summary):
    """
    Perform mandatory numerical verification before report export.
    Raises ValueError if unsupported figures exist.
    """
    return validate_report_for_export(report, df, summary)