"""Pull 10-K filings from SEC EDGAR via `edgartools`.

Two outputs per (ticker, year):
  1. financials.json — income / balance / cashflow line items from **XBRL** (exact numbers,
     no OCR). These feed the knowledge graph as ground-truth facts.
  2. filing.html    — the raw filing document, handed to Docling for narrative parsing.

`edgartools` is imported lazily so the rest of the package works without it installed.
"""
from __future__ import annotations

import json
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from tenk.config import settings


def _ensure_identity() -> None:
    from edgar import set_identity

    set_identity(settings.sec_identity)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
def _find_10k(ticker: str, year: int):
    """Locate the 10-K whose fiscal period falls in `year`."""
    from edgar import Company

    company = Company(ticker)
    filings = company.get_filings(form="10-K")
    # Prefer the filing whose fiscal period (period_of_report) falls in `year`.
    fallback = None
    for f in filings:
        period = str(getattr(f, "period_of_report", "") or "")
        if period.startswith(str(year)):
            return f
        # Fallback: a 10-K *filed* during `year` (covers missing/odd period data).
        fdate = str(getattr(f, "filing_date", ""))
        if fallback is None and fdate.startswith(str(year)):
            fallback = f
    return fallback


def _statement_year_dict(df, year: int) -> dict:
    """Pull one fiscal year's top-level line items from an edgartools statement dataframe.

    The dataframe has a `label` column, metadata flags (`abstract`/`dimension`), and one
    column per period (e.g. "2023-09-30 (FY)"). We keep non-abstract, non-dimensional rows
    (the real line items, not the product/segment breakdowns) for the matching year.
    """
    col = next((c for c in df.columns if str(c).strip().startswith(str(year))), None)
    if col is None:
        return {}
    out: dict = {}
    for _, row in df.iterrows():
        if bool(row.get("abstract")) or bool(row.get("dimension")):
            continue
        label = str(row.get("label") or "").strip()
        val = row.get(col)
        if label and val is not None and val == val:  # val == val filters NaN
            out.setdefault(label, {})[str(year)] = float(val)
    return out


def _financials_for_year(ticker: str, year: int) -> dict:
    """Extract income / balance / cash-flow line items for `year` via edgartools XBRL.

    Uses the company-level financials (covers the most recent fiscal years and avoids the
    per-filing SGML fetch). Returns {statement: {line: {year: value}}}.
    """
    from edgar import Company

    fin = Company(ticker).get_financials()
    if fin is None:
        return {}
    statements = {
        "income_statement": fin.income_statement,
        "balance_sheet": fin.balance_sheet,
        "cash_flow": fin.cashflow_statement,
    }
    out: dict = {}
    for name, method in statements.items():
        try:
            data = _statement_year_dict(method().to_dataframe(), year)
            if data:
                out[name] = data
        except Exception:
            continue
    return out


def fetch_filing(ticker: str, year: int, raw_dir: Path | None = None) -> dict | None:
    """Download one 10-K; write filing.html + financials.json. Returns a manifest entry."""
    _ensure_identity()
    raw_dir = raw_dir or settings.raw_dir
    f = _find_10k(ticker, year)
    if f is None:
        print(f"  ! no 10-K found for {ticker} {year}")
        return None

    dest = raw_dir / ticker / str(year)
    dest.mkdir(parents=True, exist_ok=True)

    # narrative HTML (Docling parses this)
    html = ""
    for attr in ("html", "markdown", "text"):
        fn = getattr(f, attr, None)
        if callable(fn):
            try:
                html = fn()
                break
            except Exception:
                continue
    (dest / "filing.html").write_text(html or "", encoding="utf-8")

    # structured financials from XBRL
    financials = {}
    try:
        financials = _financials_for_year(ticker, year)
    except Exception as exc:
        print(f"  ! financials extraction failed for {ticker} {year}: {exc}")

    manifest = {
        "ticker": ticker,
        "year": year,
        "form": "10-K",
        "accession": str(getattr(f, "accession_no", "") or getattr(f, "accession_number", "")),
        "source_url": str(getattr(f, "filing_url", "") or getattr(f, "url", "")),
        "raw_path": str(dest / "filing.html"),
    }
    (dest / "financials.json").write_text(json.dumps(financials, indent=2), encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  ✓ {ticker} {year}: {len(html):,} chars narrative, {len(financials)} statements")
    return manifest


def fetch_corpus(tickers: list[str] | None = None, years: list[int] | None = None) -> list[dict]:
    """Download the full configured corpus. Returns the list of manifests."""
    tickers = tickers or settings.tickers
    years = years or settings.years
    manifests = []
    for ticker in tickers:
        for year in years:
            print(f"Fetching {ticker} {year} …")
            m = fetch_filing(ticker, year)
            if m:
                manifests.append(m)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    (settings.raw_dir / "corpus.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    return manifests
