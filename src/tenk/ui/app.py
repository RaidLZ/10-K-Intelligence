"""Streamlit chat UI for 10-K Intelligence.

    make ui                              # runs it inside this venv (recommended)
    python -m streamlit run src/tenk/ui/app.py

Shows a *live* trace of the pipeline — which model, how it routed, what it retrieved,
how Corrective-RAG graded/rewrote — alongside the cited answer, so the system explains
itself instead of being a black box.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` executes this file as a script, so the package root isn't on sys.path
# by default. Make `import tenk` work regardless of how the app is launched.
_SRC = Path(__file__).resolve().parents[2]  # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402  (must follow the sys.path bootstrap above)

from tenk import trace  # noqa: E402
from tenk.config import settings  # noqa: E402
from tenk.llm import get_llm  # noqa: E402

st.set_page_config(page_title="10-K Intelligence", page_icon="📑", layout="wide")

# ── provider badge colours (local vs. hosted) ────────────────────────────────
_PROVIDER_STYLE = {
    "ollama": ("🖥️", "#1a7f37", "local"),
    "azure": ("☁️", "#0b6bcb", "Azure OpenAI"),
    "openai": ("✨", "#6b3fd4", "OpenAI"),
}

st.markdown(
    """
    <style>
      .badge {display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;
              font-size:0.8rem;font-weight:600;margin-right:6px;}
      .chip  {display:inline-block;padding:1px 8px;border-radius:10px;background:#eef;
              color:#334;font-size:0.72rem;font-weight:600;margin-left:6px;}
      .step  {font-family:ui-monospace,monospace;font-size:0.86rem;line-height:1.6;}
    </style>
    """,
    unsafe_allow_html=True,
)


def md(text: str) -> str:
    """Escape `$` so Streamlit doesn't render `$...$` as LaTeX math (mangles dollar amounts)."""
    return (text or "").replace("$", "\\$")


def _provider_info() -> tuple[str, str, str, str]:
    """(icon, colour, label, model) for the active LLM — no network call."""
    llm = get_llm()
    icon, colour, label = _PROVIDER_STYLE.get(llm.provider, ("🤖", "#555", llm.provider))
    return icon, colour, label, llm.model


icon, colour, label, model = _provider_info()

st.title("📑 10-K Intelligence")
st.markdown(
    f"Cited, multi-hop Q&A over SEC 10-Ks · hybrid vector + GraphRAG &nbsp; "
    f"<span class='badge' style='background:{colour}'>{icon} {label} · {model}</span>",
    unsafe_allow_html=True,
)

EXAMPLES = [
    "How did Apple's R&D spend change from 2022 to 2024, and how does it compare to Microsoft's?",
    "What were NVIDIA's main risk factors in its 2024 10-K?",
    "Which company in the corpus had the highest net income in 2024?",
]

with st.sidebar:
    st.subheader("⚙️ Engine")
    st.markdown(
        f"<span class='badge' style='background:{colour}'>{icon} {label}</span> "
        f"<span class='chip'>{model}</span>",
        unsafe_allow_html=True,
    )
    st.caption("Set `LLM_PROVIDER` in `.env` (ollama · openai · azure) and restart to switch.")

    st.subheader("📚 Corpus")
    st.markdown(
        f"**{len(settings.tickers)}** companies · **{min(settings.years)}–{max(settings.years)}**\n\n"
        + " ".join(f"`{t}`" for t in settings.tickers)
    )

    st.subheader("🧭 Routing")
    st.markdown(
        "- **vector** — single fact from one filing\n"
        "- **graph** — compare / rank / year-over-year\n"
        "- **agentic** — chained sub-questions"
    )

    st.subheader("💡 Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex

question = st.chat_input("Ask about the filings…") or st.session_state.pop("q", None)

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        # ── live pipeline trace ──────────────────────────────────────────────
        with st.status("Running the pipeline…", expanded=True) as status:
            log = st.container()

            def on_step(s) -> None:
                glyph = trace.ICONS.get(s.name, "•")
                log.markdown(
                    f"<div class='step'>{glyph} <b>{s.name}</b> — {md(s.detail)} "
                    f"<span style='color:#999'>· {s.ms:.0f} ms</span></div>",
                    unsafe_allow_html=True,
                )

            try:
                from tenk.pipeline import answer_question

                ans = answer_question(question, on_step=on_step)
                total = sum(st_.ms for st_ in ans.steps)
                status.update(
                    label=f"Done · route **{ans.route}** · {total:.0f} ms", state="complete"
                )
            except Exception as exc:  # infra not up, models not pulled, etc.
                status.update(label="Pipeline error", state="error")
                st.error(
                    f"Couldn't answer: `{type(exc).__name__}: {exc}`\n\n"
                    "Check that Qdrant + Neo4j are up (`make up`), the corpus is indexed "
                    "(`make ingest && make index && make graph`), and — for local — that "
                    "Ollama is running with the model pulled."
                )
                st.stop()

        # ── answer + metrics ─────────────────────────────────────────────────
        st.markdown(md(ans.text))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", ans.route)
        c2.metric("Contexts", len(ans.contexts))
        c3.metric("Citations", len(ans.citations))
        c4.metric("Latency", f"{sum(s.ms for s in ans.steps):.0f} ms")

        # ── evidence ─────────────────────────────────────────────────────────
        if ans.citations:
            with st.expander(f"📎 {len(ans.citations)} sources"):
                for i, c in enumerate(ans.citations, 1):
                    line = f"**[{i}]** {c.label()}"
                    if c.source_url:
                        line += f" — [filing]({c.source_url})"
                    st.markdown(line)
                    st.caption(md(c.snippet))

        if ans.contexts:
            with st.expander(f"🔬 {len(ans.contexts)} retrieved contexts (with scores)"):
                for c in ans.contexts:
                    st.markdown(
                        f"`{c.retriever}` · **{c.chunk.citation_label()}** "
                        f"<span class='chip'>score {c.score:.2f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(md(c.chunk.text[:280]))

        st.caption(f"🧭 {ans.notes}")
