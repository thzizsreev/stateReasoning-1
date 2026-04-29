import asyncio

from base import Neo4jOrchestrator
from main import run_execution_engine


class DummyLLM:
    def generate_json(self, system_prompt: str, user_prompt: str):
        # Deterministic "compute" result.
        if "[END_ORCHESTRATION]" in user_prompt:
            return {"reasoning": "terminal", "content": "done"}
        return {"reasoning": "ok", "content": 123}


async def main() -> None:
    db = Neo4jOrchestrator(password="password")
    try:
        # Reset graph.
        with db.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

        # Build a tiny chain:
        #   T1 (COMPUTE) -> OUT1
        #   T2 (COMPUTE) requires OUT1 -> END_ORCHESTRATION
        with db.driver.session() as s:
            s.run(
                """
                MERGE (t1:TaskNode {id:$t1})
                SET t1.task_type='COMPUTE', t1.instruction='compute OUT1', t1.description='',
                    t1.status='PENDING', t1.unresolved_deps=0
                MERGE (out1:DataNode {id:$out1})
                SET out1.status='UNRESOLVED'
                MERGE (t1)-[:GENERATES]->(out1)

                MERGE (t2:TaskNode {id:$t2})
                SET t2.task_type='COMPUTE', t2.instruction='compute END', t2.description='',
                    t2.status='PENDING', t2.unresolved_deps=1
                MERGE (end:DataNode {id:$end})
                SET end.status='UNRESOLVED'
                MERGE (t2)-[:GENERATES]->(end)
                MERGE (out1)-[:REQUIRES]->(t2)
                """,
                t1="[T1]",
                out1="[OUT1]",
                t2="[T2]",
                end="[END_ORCHESTRATION]",
            )

        await run_execution_engine(db, DummyLLM(), num_workers=2)
    finally:
        await db.aclose()


if __name__ == "__main__":
    asyncio.run(main())

