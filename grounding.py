import re
from decimal import Decimal, InvalidOperation

import pandas as pd


# ==================================================
# 1. STRIP NON-CLAIM STRUCTURAL NUMBERS & DATES
# ==================================================

def _strip_structural_tokens(text, approved_dates):
    """
    Remove Markdown section numbering, ordered-list prefixes,
    and verified dataset date strings (YYYY-MM-DD / YYYY-MM)
    so they are not falsely flagged as unsupported numerical claims.
    """

    cleaned = text

    # Strip verified dataset dates (longest first so YYYY-MM-DD precedes YYYY-MM)
    for date_str in sorted(approved_dates, key=len, reverse=True):
        cleaned = re.sub(rf"(?<!\d){re.escape(date_str)}(?!\d)", " ", cleaned)

    # Strip Markdown heading numbers: e.g. "## 1. Executive Summary"
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*\d+[\.\)]\s+", "", cleaned)

    # Strip numbered list prefixes at start of lines: e.g. "  1. Marketing strategies"
    cleaned = re.sub(r"(?m)^\s*\d+[\.\)]\s+", "", cleaned)

    return cleaned


def _collect_approved_dates(df, summary):
    """Collect valid date, month, and year strings present in the dataset/summary."""
    dates = set()

    if "Date" in df.columns:
        parsed_dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        for dt in parsed_dates:
            dates.add(dt.strftime("%Y-%m-%d"))
            dates.add(dt.strftime("%Y-%m"))
            dates.add(dt.strftime("%Y"))

    if "Month" in df.columns:
        for m in df["Month"].dropna().astype(str):
            dates.add(m)
            if len(m) >= 4 and m[:4].isdigit():
                dates.add(m[:4])

    period = summary.get("reporting_period")
    if isinstance(period, dict):
        for val in period.values():
            if isinstance(val, str) and val:
                dates.add(val)
                if len(val) >= 7:
                    dates.add(val[:7])
                if len(val) >= 4:
                    dates.add(val[:4])

    return dates


# ==================================================
# 2. EXTRACT NUMERIC VALUES FROM REPORT
# ==================================================

def extract_numbers(text, approved_dates=None):
    """
    Extract numerical claim values from report text.

    Supports international and Indian comma formats, decimals,
    currency prefixes, percentages, and numbers at sentence ends
    (e.g. "₹351,600.00." or "₹3,51,600.00.").
    """

    if not isinstance(text, str):
        raise ValueError("Report must be a text string.")

    cleaned_text = _strip_structural_tokens(text, approved_dates or set())

    # Allow trailing sentence period (followed by non-digit),
    # while preventing partial matches inside numbers or words.
    pattern = (
        r"(?<![\w.,])"
        r"[-+]?"
        r"(?:\d{1,3}(?:,\d{2,3})+|\d+)"
        r"(?:\.\d+)?"
        r"%?"
        r"(?![\w,]|\.\d)"
    )

    matches = re.findall(pattern, cleaned_text)
    numbers = []

    for item in matches:
        cleaned = item.replace(",", "").replace("%", "")
        try:
            value = Decimal(cleaned)
            if value.is_finite():
                numbers.append(value)
        except InvalidOperation:
            continue

    return numbers


# ==================================================
# 3. NORMALIZE NUMERIC VALUES
# ==================================================

def normalize_number(value):
    """
    Convert numeric values into normalized Decimal format
    so 100, 100.0, and 100.00 compare equal.
    """

    try:
        if isinstance(value, bool) or value is None:
            return None
        number = Decimal(str(value))
        if number.is_finite():
            normalized = number.normalize()
            if normalized == Decimal("-0"):
                return Decimal("0")
            return normalized
    except (InvalidOperation, ValueError, TypeError):
        return None

    return None


# ==================================================
# 4. COLLECT APPROVED NUMERIC VALUES
# ==================================================

def collect_approved_numbers(df, summary):
    """
    Collect approved numerical values from:
      1. Cleaned DataFrame numeric columns (raw and 2-decimal rounded).
      2. Business analysis summary dictionary (recursive).
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")

    if not isinstance(summary, dict):
        raise ValueError("Summary must be a dictionary.")

    approved = set()

    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        for value in df[column].dropna():
            norm = normalize_number(value)
            if norm is not None:
                approved.add(norm)
            norm_rounded = normalize_number(round(float(value), 2))
            if norm_rounded is not None:
                approved.add(norm_rounded)

    def collect_recursive(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                collect_recursive(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                collect_recursive(value)
        elif isinstance(obj, bool) or obj is None:
            return
        elif isinstance(obj, (int, float, Decimal)):
            if isinstance(obj, float) and pd.isna(obj):
                return
            norm = normalize_number(obj)
            if norm is not None:
                approved.add(norm)
            # Also approve absolute value for negative growth percentages
            if norm is not None and norm < 0:
                approved.add(abs(norm))

    collect_recursive(summary)

    # Approve top-N structural count (e.g. "top 5 products")
    top_products = summary.get("top_products")
    if isinstance(top_products, list) and top_products:
        approved.add(normalize_number(len(top_products)))

    return approved


# ==================================================
# 5. VERIFY REPORT NUMERICAL VALUES
# ==================================================

def verify_report(report, df, summary):
    """
    Verify numerical figures in a generated report against
    approved dataset and summary values.
    """

    if not isinstance(report, str):
        raise ValueError("Report must be a text string.")

    if not report.strip():
        raise ValueError("Cannot verify an empty report.")

    approved_dates = _collect_approved_dates(df, summary)
    approved_numbers = collect_approved_numbers(df, summary)
    reported_numbers = extract_numbers(report, approved_dates=approved_dates)

    unsupported = []
    for number in reported_numbers:
        normalized = normalize_number(number)
        if normalized not in approved_numbers:
            unsupported.append(str(number))

    verified = len(unsupported) == 0

    if verified:
        message = (
            "All detected numerical values match approved dataset values."
        )
    else:
        message = (
            "Some numerical values could not be matched to the dataset. "
            "Report export is blocked."
        )

    return {
        "verified": verified,
        "total_figures": len(reported_numbers),
        "unsupported_figures": unsupported,
        "message": message,
    }


# ==================================================
# 6. VERIFY REPORT BEFORE EXPORT
# ==================================================

def validate_report_for_export(report, df, summary):
    """
    Validate a report before allowing export.
    Raises ValueError if unsupported numbers exist.
    """

    verification = verify_report(report, df, summary)

    if not verification["verified"]:
        raise ValueError(
            "Report verification failed. Unsupported numerical values detected: "
            + ", ".join(verification["unsupported_figures"])
        )

    return verification