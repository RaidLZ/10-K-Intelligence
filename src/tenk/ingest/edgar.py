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
    for f in filings:
        # filing_date is shortly after fiscal year end; match on report year.
        fdate = str(getattr(f, "filing_date", ""))
        if fdate.startswith(str(year)) or fdate.startswith(str(year + 1)):
            # prefer the filing reporting ON `year`
            period = str(getattr(f, "period_of_report", "") or "")
            if period.startswith(str(year)) or fdate.startswith(str(year + 1)):
                return f
    return None


def _financials_to_dict(tenk) -> dict:
    """Best-effort extraction of the three statements as {statement: {line: {year: value}}}."""
    out: dict = {}
    fin = getattr(tenk, "financials", None)
    if fin is None:
        return out
    statements = {
        "income_statement": getattr(fin, "income", None) or getattr(fin, "income_statement", None),
        "balance_sheet": getattr(fin, "balance_sheet", None),
        "cash_flow": getattr(fin, "cashflow", None) or getattr(fin, "cash_flow", None),
    }
    for name, stmt in statements.items():
        if stmt is None:
            continue
        try:
            df = stmt.to_dataframe() if hasattr(stmt, "to_dataframe") else stmt
            out[name] = json.loads(df.to_json(orient="index"))
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
        financials = _financials_to_dict(f.obj())
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
