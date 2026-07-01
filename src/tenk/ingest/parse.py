"""Parse a filing's narrative into structured elements with Docling.

Docling's TableFormer preserves merged cells, and its layout model gives us section
headings + page numbers — which we carry onto every element so chunks stay citable.
Output: a list of element dicts `{section, page, text, is_table}` written to
`processed/<ticker>/<year>/sections.json`.

Docling is imported lazily; if it is unavailable we fall back to a simple HTML→text split
so ingestion still degrades gracefully.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tenk.config import settings

# Canonical 10-K item headings, used to normalise detected section titles.
_ITEM_RE = re.compile(r"^\s*item\s+(\d+[a-z]?)\b[.\s:-]*(.*)", re.IGNORECASE)

# Standard 10-K item titles — used to label a heading Docling split down to just its
# number ("Item 1A." with the title on the next line), and to sanity-check real headings.
_ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data", "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants", "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules", "16": "Form 10-K Summary",
}
# Words that, when they follow "Item N.", mark an inline cross-reference, not a heading
# (e.g. "see Item 15. of this Annual Report …") — so we don't reset the section on them.
_XREF_STARTS = ("of ", "in ", "for ", "to ", "and ", "the ", "as ", "under ", "above", "below", "or ")


def normalise_section(heading: str) -> str:
    m = _ITEM_RE.match(heading or "")
    if m:
        num, rest = m.group(1).upper(), m.group(2).strip().rstrip(".")
        if not rest:
            rest = _ITEM_TITLES.get(num, "")
        return f"Item {num}. {rest}" if rest else f"Item {num}"
    return (heading or "").strip()[:120]


def section_heading(text: str) -> str | None:
    """Return a canonical `Item N. Title` if `text` is a real 10-K item heading, else None.

    Handles Docling emitting the heading as just its number (title on the next line) and
    rejects inline cross-references like "Item 15. of this Annual Report on Form 10-K".
    """
    t = (text or "").strip()
    m = _ITEM_RE.match(t)
    if not m:
        return None
    num, rest = m.group(1).upper(), m.group(2).strip().rstrip(".").strip()
    known = _ITEM_TITLES.get(num)
    if not rest:  # number-only heading → fill the canonical title
        return f"Item {num}. {known}" if known else f"Item {num}"
    # A long official heading (e.g. Item 7's full MD&A title) is still a heading if its
    # text begins with the item's canonical title — accept it regardless of length.
    if known and rest.lower().startswith(known.lower()[:12]):
        return f"Item {num}. {known}"
    if rest.lower().startswith(_XREF_STARTS):  # cross-reference, not a heading
        return None
    if len(t) <= 80:
        return f"Item {num}. {rest}"
    return None


def _parse_with_docling(html_path: Path) -> list[dict]:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter(allowed_formats=[InputFormat.HTML, InputFormat.PDF])
    doc = converter.convert(str(html_path)).document

    elements: list[dict] = []
    current_section = ""
    for item, _level in doc.iterate_items():
        label = str(getattr(item, "label", "")).lower()
        page = None
        prov = getattr(item, "prov", None)
        if prov:
            page = getattr(prov[0], "page_no", None)

        if "table" in label:
            try:
                text = item.export_to_markdown(doc)
            except Exception:
                try:
                    text = item.export_to_markdown()
                except Exception:
                    text = getattr(item, "text", "")
            if text.strip():
                elements.append(
                    {"section": current_section, "page": page, "text": text, "is_table": True}
                )
            continue

        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue

        # SEC HTML headings arrive as plain "text" items, so detect them by content;
        # explicit header/title labels (PDFs, other formats) are honoured too.
        heading = section_heading(text)
        if heading is not None:
            current_section = heading
            continue
        if "header" in label or "title" in label or "section" in label:
            current_section = normalise_section(text)
            continue

        elements.append(
            {"section": current_section, "page": page, "text": text, "is_table": False}
        )
    return elements


def _parse_fallback(html_path: Path) -> list[dict]:
    """No Docling: crude split on 10-K item headings."""
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", raw)            # strip tags
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?i)(item\s+\d+[a-z]?\b[.\s:-]*[^.]{0,80})", text)
    elements: list[dict] = []
    section = ""
    for part in parts:
        if _ITEM_RE.match(part or ""):
            section = normalise_section(part)
        elif part and part.strip():
            elements.append(
                {"section": section, "page": None, "text": part.strip(), "is_table": False}
            )
    return elements


def parse_filing(html_path: str | Path) -> list[dict]:
    html_path = Path(html_path)
    try:
        elements = _parse_with_docling(html_path)
        if elements:
            return elements
    except Exception as exc:
        print(f"  ! Docling parse failed ({exc}); using fallback splitter")
    return _parse_fallback(html_path)


def parse_corpus(manifests: list[dict] | None = None) -> None:
    """Parse every filing in the corpus manifest into processed/<ticker>/<year>/sections.json."""
    if manifests is None:
        corpus_file = settings.raw_dir / "corpus.json"
        manifests = json.loads(corpus_file.read_text()) if corpus_file.exists() else []
    for m in manifests:
        out = settings.processed_dir / m["ticker"] / str(m["year"])
        out.mkdir(parents=True, exist_ok=True)
        elements = parse_filing(m["raw_path"])
        (out / "sections.json").write_text(json.dumps(elements, indent=2), encoding="utf-8")
        # carry financials forward next to the parsed narrative
        fin_src = Path(m["raw_path"]).parent / "financials.json"
        if fin_src.exists():
            (out / "financials.json").write_text(fin_src.read_text(), encoding="utf-8")
        print(f"  ✓ parsed {m['ticker']} {m['year']}: {len(elements)} elements")
