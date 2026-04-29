import asyncio

from base import Neo4jOrchestrator


async def main() -> None:
    db = Neo4jOrchestrator(password="password")
    try:
        # Reset graph.
        with db.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

        # Create one runnable task with an output node.
        with db.driver.session() as s:
            s.run(
                """
                MERGE (t:TaskNode {id: $tid})
                SET t.task_type='COMPUTE',
                    t.instruction='do',
                    t.description='',
                    t.status='PENDING',
                    t.unresolved_deps=0
                MERGE (d:DataNode {id: $out})
                SET d.status='UNRESOLVED'
                MERGE (t)-[:GENERATES]->(d)
                """,
                tid="[T1]",
                out="[OUT1]",
            )

        # Try to claim the same task concurrently from 2 workers.
        r1, r2 = await asyncio.gather(db.async_claim_next_task(1), db.async_claim_next_task(2))
        print("claim results:", r1, r2)

        with db.driver.session() as s:
            row = s.run(
                "MATCH (t:TaskNode {id: $tid}) RETURN t.status AS st, t.claimed_by AS by",
                tid="[T1]",
            ).single()
            print("db state:", dict(row.data()) if row else None)

    finally:
        await db.aclose()


if __name__ == "__main__":
    asyncio.run(main())

