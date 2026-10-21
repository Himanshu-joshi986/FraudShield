"""
Phase 7 -- Knowledge graph store.

Real Cypher-queryable graph database holding the FraudShield entity
model. Two interchangeable backends behind one interface:

  - "kuzu"  -- VERIFIED WORKING (kuzu 0.11.3). Embedded, no server
               process, no JVM, no Docker. This is what produced the
               measured Phase 7 results in docs/EXPERIMENTS.md §8.
  - "neo4j" -- the store named in the spec. Its adapter uses the real
               `neo4j` Python driver and is selected via
               GRAPH_BACKEND=neo4j.

These are NOT "networkx pretending to be a graph DB". kuzu executes
genuine Cypher, including the multi-hop pattern matching that fraud-ring
discovery actually needs:

    MATCH (a:Entity)-[:SHARES]->(x)<-[:SHARES]-(b:Entity)

Entity model (Section 26):

    Customer -[:OWNS_ACCOUNT]->    Account
    Customer -[:USES_DEVICE]->     Device
    Customer -[:HAS_EMAIL]->       Email
    Customer -[:HAS_ADDRESS]->     Address
    Customer -[:FILED_VIA]->       Broker
    Customer -[:MADE]->            Transaction
    Customer -[:FILED]->           Claim
    Claim    -[:SUPPORTED_BY]->    Document
"""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KUZU_PATH = REPO_ROOT / "data" / "graph_db"

NODE_TABLES = {
    "Customer": "id STRING, name STRING, data_source STRING, scenario_id STRING",
    "Account": "id STRING",
    "Device": "id STRING",
    "Email": "id STRING",
    "Address": "id STRING",
    "Broker": "id STRING",
    "Transaction": "id STRING, amount DOUBLE, timestamp STRING, is_fraud INT64",
    "Claim": "id STRING, amount DOUBLE, timestamp STRING",
    "Document": "id STRING, invoice_number STRING, amount DOUBLE",
}

REL_TABLES = [
    ("OWNS_ACCOUNT", "Customer", "Account"),
    ("USES_DEVICE", "Customer", "Device"),
    ("HAS_EMAIL", "Customer", "Email"),
    ("HAS_ADDRESS", "Customer", "Address"),
    ("FILED_VIA", "Customer", "Broker"),
    ("MADE", "Customer", "Transaction"),
    ("FILED", "Customer", "Claim"),
    ("SUPPORTED_BY", "Claim", "Document"),
]

# Which shared entity types count as STRONG fraud-ring signal.
# Deliberately excludes Broker and Email: many legitimate claimants
# legitimately share a broker or an email *domain*, so treating those as
# strong signal produces a flood of false positives. This was learned the
# hard way -- see docs/EXPERIMENTS.md §8.
# Which shared entity types trigger the HARD override (escalate regardless
# of score). Deliberately narrower than STRONG_SHARED_TYPES: Address alone
# is real signal (contributes to graph_risk) but is NOT sufficient on its
# own to force an override -- shared housing (e.g. family members, a
# shared mailing address) is common among legitimate claimants. This was
# found during Phase 9 evaluation: an early version included Address in
# the override set and incorrectly escalated Scenario 6 (shared-address,
# ground-truth MEDIUM) to HIGH.
OVERRIDE_ELIGIBLE_TYPES = ["Account", "Device"]
STRONG_SHARED_TYPES = ["Account", "Device", "Address"]
WEAK_SHARED_TYPES = ["Broker", "Email"]

SHARED_REL_FOR_TYPE = {
    "Account": "OWNS_ACCOUNT",
    "Device": "USES_DEVICE",
    "Address": "HAS_ADDRESS",
    "Broker": "FILED_VIA",
    "Email": "HAS_EMAIL",
}


class GraphStore(ABC):
    name = "base"

    @abstractmethod
    def initialize_schema(self, reset: bool = False): ...

    @abstractmethod
    def add_customer(self, customer_id: str, name: str, data_source: str = "synthetic",
                     scenario_id: str | None = None): ...

    @abstractmethod
    def add_entity(self, entity_type: str, entity_id: str): ...

    @abstractmethod
    def link(self, rel: str, from_type: str, from_id: str, to_type: str, to_id: str): ...

    @abstractmethod
    def query(self, cypher: str, params: dict | None = None) -> list[dict]: ...

    @abstractmethod
    def close(self): ...

    # ---- shared, backend-independent fraud-investigation queries ----

    def find_connected_customers(self, customer_id: str, shared_types: list[str] | None = None) -> list[dict]:
        """
        Who else is connected to this customer through a shared entity?
        This is the core Investigation Agent query -- real 2-hop Cypher
        pattern matching, not a Python loop over an in-memory dict.
        """
        shared_types = shared_types or STRONG_SHARED_TYPES
        results = []
        for etype in shared_types:
            rel = SHARED_REL_FOR_TYPE[etype]
            cypher = f"""
                MATCH (a:Customer)-[:{rel}]->(x:{etype})<-[:{rel}]-(b:Customer)
                WHERE a.id = $cid AND b.id <> $cid
                RETURN b.id AS connected_customer, b.name AS connected_name,
                       x.id AS shared_entity, '{etype}' AS shared_type
            """
            results.extend(self.query(cypher, {"cid": customer_id}))
        return results

    def find_shared_entity_clusters(self, entity_type: str, min_customers: int = 2) -> list[dict]:
        """
        Which entities of this type are shared by >= min_customers?
        A single account/device used by 6 different customers is the
        classic fraud-ring fingerprint.
        """
        rel = SHARED_REL_FOR_TYPE[entity_type]
        cypher = f"""
            MATCH (c:Customer)-[:{rel}]->(x:{entity_type})
            WITH x.id AS entity_id, COUNT(DISTINCT c.id) AS n_customers,
                 COLLECT(DISTINCT c.id) AS customers
            WHERE n_customers >= $min_customers
            RETURN entity_id, n_customers, customers
            ORDER BY n_customers DESC
        """
        return self.query(cypher, {"min_customers": min_customers})

    def get_customer_subgraph(self, customer_id: str) -> dict:
        """Nodes + edges around one customer, for the dashboard's fraud graph."""
        nodes, edges = {}, []
        nodes[customer_id] = {"id": customer_id, "type": "Customer"}

        for etype in STRONG_SHARED_TYPES + WEAK_SHARED_TYPES:
            rel = SHARED_REL_FOR_TYPE[etype]
            direct = self.query(
                f"MATCH (c:Customer)-[:{rel}]->(x:{etype}) WHERE c.id = $cid RETURN x.id AS eid",
                {"cid": customer_id},
            )
            for row in direct:
                eid = row["eid"]
                nodes[eid] = {"id": eid, "type": etype}
                edges.append({"source": customer_id, "target": eid, "relation": rel})

                others = self.query(
                    f"MATCH (c:Customer)-[:{rel}]->(x:{etype}) WHERE x.id = $eid AND c.id <> $cid "
                    f"RETURN c.id AS cid2, c.name AS name2",
                    {"eid": eid, "cid": customer_id},
                )
                for other in others:
                    nodes[other["cid2"]] = {"id": other["cid2"], "type": "Customer",
                                              "name": other.get("name2")}
                    edges.append({"source": other["cid2"], "target": eid, "relation": rel})

        return {"nodes": list(nodes.values()), "edges": edges}


class KuzuGraphStore(GraphStore):
    """VERIFIED WORKING -- kuzu 0.11.3, embedded, no server required."""

    name = "kuzu"

    def __init__(self, db_path: str | Path | None = None):
        import kuzu
        self.db_path = Path(db_path or os.environ.get("KUZU_DB_PATH") or DEFAULT_KUZU_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._kuzu = kuzu
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

    def initialize_schema(self, reset: bool = False):
        if reset:
            self.conn = None
            self.db = None
            if self.db_path.exists():
                shutil.rmtree(self.db_path, ignore_errors=True) if self.db_path.is_dir() else self.db_path.unlink()
            self.db = self._kuzu.Database(str(self.db_path))
            self.conn = self._kuzu.Connection(self.db)

        for table, cols in NODE_TABLES.items():
            try:
                self.conn.execute(f"CREATE NODE TABLE {table}({cols}, PRIMARY KEY(id))")
            except RuntimeError:
                pass  # already exists

        for rel, src, dst in REL_TABLES:
            try:
                self.conn.execute(f"CREATE REL TABLE {rel}(FROM {src} TO {dst})")
            except RuntimeError:
                pass

    def add_customer(self, customer_id, name, data_source="synthetic", scenario_id=None):
        self.conn.execute(
            "MERGE (c:Customer {id: $id}) SET c.name = $name, "
            "c.data_source = $ds, c.scenario_id = $sid",
            {"id": customer_id, "name": name, "ds": data_source, "sid": scenario_id or ""},
        )

    def add_entity(self, entity_type, entity_id, **props):
        if entity_type not in NODE_TABLES:
            raise ValueError(f"Unknown entity type: {entity_type}")
        set_clause = ""
        params = {"id": entity_id}
        if props:
            parts = []
            for i, (k, v) in enumerate(props.items()):
                parts.append(f"n.{k} = $p{i}")
                params[f"p{i}"] = v
            set_clause = " SET " + ", ".join(parts)
        self.conn.execute(f"MERGE (n:{entity_type} {{id: $id}}){set_clause}", params)

    def link(self, rel, from_type, from_id, to_type, to_id):
        self.conn.execute(
            f"MATCH (a:{from_type}), (b:{to_type}) WHERE a.id = $fid AND b.id = $tid "
            f"MERGE (a)-[:{rel}]->(b)",
            {"fid": from_id, "tid": to_id},
        )

    def query(self, cypher, params=None):
        result = self.conn.execute(cypher, params or {})
        df = result.get_as_df()
        return df.to_dict("records")

    def close(self):
        self.conn = None
        self.db = None


class Neo4jGraphStore(GraphStore):
    """
    Connects to a Neo4j server with the real `neo4j` Python driver. To use:
        pip install neo4j
        docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
        export GRAPH_BACKEND=neo4j
    """

    name = "neo4j"

    def __init__(self, uri=None, user=None, password=None):
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError(
                "neo4j driver not installed. Install with `pip install neo4j`, "
                "or use GRAPH_BACKEND=kuzu (the verified embedded backend)."
            ) from e
        self.driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"),
                  password or os.environ.get("NEO4J_PASSWORD", "changeme")),
        )

    def initialize_schema(self, reset: bool = False):
        with self.driver.session() as session:
            if reset:
                session.run("MATCH (n) DETACH DELETE n")
            for label in NODE_TABLES:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    def add_customer(self, customer_id, name, data_source="synthetic", scenario_id=None):
        with self.driver.session() as session:
            session.run(
                "MERGE (c:Customer {id: $id}) SET c.name=$name, c.data_source=$ds, c.scenario_id=$sid",
                id=customer_id, name=name, ds=data_source, sid=scenario_id or "",
            )

    def add_entity(self, entity_type, entity_id, **props):
        with self.driver.session() as session:
            session.run(f"MERGE (n:{entity_type} {{id: $id}}) SET n += $props",
                        id=entity_id, props=props)

    def link(self, rel, from_type, from_id, to_type, to_id):
        with self.driver.session() as session:
            session.run(
                f"MATCH (a:{from_type} {{id:$fid}}), (b:{to_type} {{id:$tid}}) MERGE (a)-[:{rel}]->(b)",
                fid=from_id, tid=to_id,
            )

    def query(self, cypher, params=None):
        with self.driver.session() as session:
            return [dict(record) for record in session.run(cypher, **(params or {}))]

    def close(self):
        self.driver.close()


_BACKENDS = {"kuzu": KuzuGraphStore, "neo4j": Neo4jGraphStore}


def get_graph_store(backend: str | None = None, **kwargs) -> GraphStore:
    backend = (backend or os.environ.get("GRAPH_BACKEND") or "kuzu").lower()
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown graph backend '{backend}'. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[backend](**kwargs)
