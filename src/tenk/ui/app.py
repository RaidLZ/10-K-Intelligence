"""Streamlit chat UI for 10-K Intelligence.

    streamlit run src/tenk/ui/app.py
"""
from __future__ import annotations

import streamlit as st

from tenk.config import settings

st.set_page_config(page_title="10-K Intelligence", page_icon="📑", layout="wide")

st.title("📑 10-K Intelligence")
st.caption(
    f"Cited, multi-hop Q&A over SEC 10-Ks · corpus: {', '.join(settings.tickers)} "
    f"({min(settings.years)}–{max(settings.years)}) · hybrid vector + GraphRAG"
)

EXAMPLES = [
    "How did Apple's R&D spend change from 2022 to 2024, and how does it compare to Microsoft's?",
    "What were NVIDIA's main risk factors in its 2024 10-K?",
    "Which company in the corpus had the highest net income in 2024?",
]

with st.sidebar:
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    st.markdown("---")
    st.markdown("**Routing**: single-hop → vector · cross-company/year → graph · chained → agentic")

question = st.chat_input("Ask about the filings…") or st.session_state.pop("q", None)

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Routing, retrieving, and verifying…"):
            from tenk.pipeline import answer_question

            ans = answer_question(question)
        st.markdown(ans.text)
        st.caption(f"🧭 route: **{ans.route}** · {ans.notes}")
        if ans.citations:
            with st.expander(f"📎 {len(ans.citations)} sources"):
                for i, c in enumerate(ans.citations, 1):
                    line = f"**[{i}]** {c.label()}"
                    if c.source_url:
                        line += f" — [filing]({c.source_url})"
                    st.markdown(line)
                    st.caption(c.snippet)
