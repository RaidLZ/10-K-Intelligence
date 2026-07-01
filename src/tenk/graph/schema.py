"""Neo4j connection + graph schema.

Schema (kept deliberately small and finance-shaped):

    (:Company {ticker, name})
      -[:FILED]-> (:Filing {id, ticker, year, form, source_url})
                    -[:REPORTS {value}]-> (:Metric {name})          # from XBRL (exact)
    (:Company) -[:HAS_SEGMENT]->   (:Segment {name})                # LLM-extracted
    (:Company) -[:MENTIONS_RISK]-> (:Risk {name})                   # LLM-extracted
    (:Company) -[:COMPETES_WITH]-> (:Company)                       # LLM-extracted
    (:Person {name}) -[:EXECUTIVE_OF|BOARD_OF]-> (:Company)         # LLM-extracted
"""
from __future__ import annotations

from tenk.config import settings

CONSTRAINTS = [
    "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
    "CREATE CONSTRAINT filing_id IF NOT EXISTS FOR (f:Filing) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT segment_name IF NOT EXISTS FOR (s:Segment) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT risk_name IF NOT EXISTS FOR (r:Risk) REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
]


class Graph:
    """Tiny wrapper over the Neo4j driver with a context manager."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        from neo4j import GraphDatabase

        kwargs = {"auth": (user or settings.neo4j_user, password or settings.neo4j_password)}
        try:
            # silence verbose server notifications (e.g. from exploratory LLM-generated Cypher)
            self._driver = GraphDatabase.driver(
                uri or settings.neo4j_uri, notifications_min_severity="OFF", **kwargs
            )
        except Exception:  # older drivers don't support the kwarg
            self._driver = GraphDatabase.driver(uri or settings.neo4j_uri, **kwargs)

    def run(self, cypher: str, **params):
        with self._driver.session() as session:
            return list(session.run(cypher, **params))

    def ensure_schema(self) -> None:
        for c in CONSTRAINTS:
            self.run(c)

    def wipe(self) -> None:
        self.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
