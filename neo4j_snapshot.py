import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from base import Neo4jOrchestrator


DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "concurrent_base.json"


def export_neo4j_snapshot(
    db: Neo4jOrchestrator,
    out_path: Path = DEFAULT_SNAPSHOT_PATH,
) -> Path:
    """
    Export the current Neo4j graph (nodes + relationships) to a JSON file.

    This is a logical export (Cypher-level), not a Neo4j store dump.
    It is intended to recreate the same graph state for this project.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with db.driver.session() as session:
        nodes_result = session.run(
            """
            MATCH (n)
            RETURN labels(n) AS labels, properties(n) AS props
            """
        )
        nodes: List[Dict[str, Any]] = []
        for r in nodes_result:
            labels = list(r["labels"] or [])
            props = dict(r["props"] or {})
            nodes.append({"labels": labels, "props": props})

        rels_result = session.run(
            """
            MATCH (a)-[r]->(b)
            RETURN labels(a) AS start_labels,
                   properties(a) AS start_props,
                   type(r) AS type,
                   properties(r) AS props,
                   labels(b) AS end_labels,
                   properties(b) AS end_props
            """
        )
        rels: List[Dict[str, Any]] = []
        for r in rels_result:
            rels.append(
                {
                    "start": {
                        "labels": list(r["start_labels"] or []),
                        "props": dict(r["start_props"] or {}),
                    },
                    "type": r["type"],
                    "props": dict(r["props"] or {}),
                    "end": {
                        "labels": list(r["end_labels"] or []),
                        "props": dict(r["end_props"] or {}),
                    },
                }
            )

    snapshot = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format": "stateReasoning-1.neo4j_snapshot.v1",
        },
        "nodes": nodes,
        "relationships": rels,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    return out_path


def _node_identity(props: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Pick a stable identity for MERGE.
    For this project, every node should have an `id` property.
    """
    if "id" in props:
        return ("id", props["id"])
    raise ValueError("Snapshot node missing required 'id' property.")


def load_neo4j_snapshot(
    db: Neo4jOrchestrator,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    *,
    clear_first: bool = True,
) -> None:
    """
    Load a snapshot created by `export_neo4j_snapshot` and recreate the graph.
    """
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    # Avoid JSONDecodeError on empty/whitespace-only snapshot files.
    try:
        if snapshot_path.stat().st_size == 0:
            print(f"[Snapshot] '{snapshot_path}' is empty. Nothing to load.")
            return
    except OSError:
        # If we can't stat, fall through to open/load and let it raise.
        pass

    with open(snapshot_path, "r", encoding="utf-8") as f:
        raw = f.read()
        if not raw.strip():
            print(f"[Snapshot] '{snapshot_path}' has no JSON content. Nothing to load.")
            return
        snapshot = json.loads(raw)

    nodes = snapshot.get("nodes", [])
    rels = snapshot.get("relationships", [])

    with db.driver.session() as session:
        if clear_first:
            session.run("MATCH (n) DETACH DELETE n")

        # 1) Recreate nodes
        for n in nodes:
            labels = n.get("labels", [])
            props = n.get("props", {}) or {}
            key, val = _node_identity(props)
            labels_cypher = ":" + ":".join(labels) if labels else ""

            # MERGE by identity, then set full props
            session.run(
                f"""
                MERGE (n{labels_cypher} {{{key}: $val}})
                SET n += $props
                """,
                val=val,
                props=props,
            )

        # 2) Recreate relationships
        for rel in rels:
            start_props = (rel.get("start", {}) or {}).get("props", {}) or {}
            end_props = (rel.get("end", {}) or {}).get("props", {}) or {}
            rel_type = rel.get("type")
            rel_props = rel.get("props", {}) or {}

            if not rel_type:
                continue

            sk, sv = _node_identity(start_props)
            ek, ev = _node_identity(end_props)

            # Relationship type cannot be parameterized in Cypher, so it must be interpolated.
            session.run(
                f"""
                MATCH (a {{{sk}: $sv}})
                MATCH (b {{{ek}: $ev}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $rprops
                """,
                sv=sv,
                ev=ev,
                rprops=rel_props,
            )


if __name__ == "__main__":
    db = Neo4jOrchestrator(password="password")
    path = export_neo4j_snapshot(db)
    print(f"Snapshot written to: {path}")
    #load_neo4j_snapshot(db)   
