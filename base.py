import json

from neo4j import GraphDatabase
#import networkx as nx
from typing import List, Any
from pydantic import BaseModel

from schema import TaskNode, DataNode
# ==========================================
# 1. PYDANTIC SCHEMAS
# ==========================================
class MicroTask(BaseModel):
    task_id: str
    instruction: str
    required_skills: List[str]  
    dependencies: List[str]     
    output_key: str             

class TaskDecomposition(BaseModel):
    phase_name: str
    tasks: List[MicroTask]

# ==========================================
# 2. NETWORKX GRAPH ENGINE
# ==========================================
class DynamicTaskGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_data_node(self, node_id: str, content: Any):
        self.G.add_node(node_id, type="DataNode", content=content, status="READY")
        print(f"🟢 DATA INJECTED: {node_id} = {content}")

    def spawn_tasks(self, decomposition: TaskDecomposition):
        print(f"\n🧬 SPAWNING DAG FOR: {decomposition.phase_name}")
        for task in decomposition.tasks:
            self.G.add_node(
                task.task_id, 
                type="TaskNode", 
                instruction=task.instruction,
                skills=task.required_skills,
                output_key=task.output_key,
                status="PENDING"
            )
            for dep in task.dependencies:
                if not self.G.has_node(dep):
                    self.G.add_node(dep, type="DataNode", content=None, status="UNRESOLVED")
                self.G.add_edge(dep, task.task_id, relation="REQUIRES")
                
            if not self.G.has_node(task.output_key):
                self.G.add_node(task.output_key, type="DataNode", content=None, status="UNRESOLVED")
            self.G.add_edge(task.task_id, task.output_key, relation="CREATES")
            print(f"   ├─ Spawned: {task.task_id} (Needs: {task.dependencies} -> Creates: {task.output_key})")

    def get_executable_tasks(self) -> List[str]:
        executable = []
        for node, attr in self.G.nodes(data=True):
            if attr.get("type") == "TaskNode" and attr.get("status") == "PENDING":
                deps = [u for u, v, d in self.G.in_edges(node, data=True) if d['relation'] == 'REQUIRES']
                if all(self.G.nodes[dep].get("status") == "READY" for dep in deps):
                    executable.append(node)
        return executable

    def complete_task(self, task_id: str, result: Any):
        output_key = self.G.nodes[task_id]["output_key"]
        self.G.nodes[task_id]["status"] = "COMPLETE"
        self.G.nodes[output_key]["status"] = "READY"
        self.G.nodes[output_key]["content"] = result
        print(f"   └─ 🟢 GRAPH UPDATED: {output_key} Resolved.")

# ==========================================
# 2. NEO4J GRAPH MANAGER
# ==========================================
class Neo4jOrchestrator:
    def __init__(self, uri="neo4j://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        #self._clear_db()

    def _clear_db(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🧹 Graph Cleared.")

    def close(self) -> None:
        self.driver.close()

    @staticmethod
    def _normalize_id(x: str) -> str:
        x = (x or "").strip()
        if not x:
            return x
        if x.startswith("[") and x.endswith("]"):
            return x
        return f"[{x}]"

    def ensure_semaphore(self) -> None:
        query = """
        MERGE (s:Semaphore {id: 'GLOBAL'})
        ON CREATE SET s.pending_count = 0
        """
        with self.driver.session() as session:
            session.run(query)

    def recompute_semaphore(self) -> None:
        # pending_count = sum(unresolved_deps) for PENDING tasks (decrements as deps resolve)
        query = """
        MERGE (s:Semaphore {id: 'GLOBAL'})
        WITH s
        MATCH (t:TaskNode)
        WHERE t.status = 'PENDING'
        WITH s, sum(coalesce(t.unresolved_deps, 0)) AS total
        SET s.pending_count = total
        """
        with self.driver.session() as session:
            session.run(query)

    def upsert_data_node(self, data, content: Any | None = None, status: str = "COMPLETE") -> None:
        """
        Aggressively inserts or updates a DataNode.
        Automatically cascades and safely recomputes the unresolved_deps semaphore.
        """
        node_id = self._normalize_id(data.id)
        
        # Resolve content fallback
        final_content = content if content is not None else data.content
        content_str = json.dumps(final_content) if isinstance(final_content, (dict, list)) else str(final_content)

        query = """
        // 1. Find or create the DataNode
        MERGE (d:DataNode {id: $id})
        SET d.content = $content, 
            d.status = $status
            
        // 2. SEMAPHORE CASCADE: Find downstream tasks
        WITH d
        OPTIONAL MATCH (d)-[:REQUIRES]->(t:TaskNode)
        
        // 3. ATOMIC RECALCULATION
        // DISTINCT forces the stream back to 1 row per task, preventing double-counting
        WITH DISTINCT t
        WHERE t IS NOT NULL
        
        // Use OPTIONAL MATCH to safely handle tasks that might somehow have 0 deps
        OPTIONAL MATCH (dep:DataNode)-[:REQUIRES]->(t)
        WITH t, sum(CASE 
            WHEN dep IS NULL THEN 0 
            WHEN coalesce(dep.status, 'PENDING') = 'COMPLETE' THEN 0 
            ELSE 1 
        END) AS unresolved
        
        // 4. Update the task's semaphore
        SET t.unresolved_deps = unresolved
        """
        with self.driver.session() as session:
            session.run(query, id=node_id, content=content_str, status=status)
            
        self.recompute_semaphore() 
        print(f"[DATA] Injected & cascaded: {node_id}")

    def upsert_task(self, task) -> None:
        """
        Upserts a TaskNode, draws topology using FOREACH to prevent stream alteration,
        and safely handles zero-dependency tasks and optional target outputs.
        """
        task_id = self._normalize_id(task.id)
        deps = [self._normalize_id(d) for d in task.dependencies]
        target_output = (
            self._normalize_id(task.target_output) if getattr(task, "target_output", None) else None
        )
        parent_task = self._normalize_id(task.parent_task) if getattr(task, "parent_task", None) else None

        query = """
        // 1) Upsert TaskNode and basic properties
        MERGE (t:TaskNode {id: $task_id})
        SET t.instruction = $instruction,
            t.description = $description,
            t.task_type = $task_type,
            t.status = $status

        // 2) Output DataNode shell + GENERATES edge (Only triggers if target_output exists)
        WITH t
        FOREACH (_ IN CASE WHEN $target_output IS NULL THEN [] ELSE [1] END |
            MERGE (out:DataNode {id: $target_output})
            ON CREATE SET out.status = 'UNRESOLVED'
            MERGE (t)-[:GENERATES]->(out)
        )

        // 3) Parent-child relationship (Execution Lineage)
        WITH t
        FOREACH (_ IN CASE WHEN $parent_task IS NULL THEN [] ELSE [1] END |
            MERGE (p:TaskNode {id: $parent_task})
            MERGE (p)-[:CREATES]->(t)
        )

        // 4) Clean old dependency edges to allow for graph self-correction
        WITH t
        OPTIONAL MATCH (d_old:DataNode)-[r_old:REQUIRES]->(t)
        DELETE r_old

        // 5) SAFE DEPENDENCY INJECTION (Data Lineage)
        WITH t
        FOREACH (dep_id IN $dependencies |
            MERGE (d:DataNode {id: dep_id})
            ON CREATE SET d.status = 'UNRESOLVED'
            MERGE (d)-[:REQUIRES]->(t)
        )

        // 6) SEMAPHORE CALCULATION
        WITH t
        OPTIONAL MATCH (d2:DataNode)-[:REQUIRES]->(t)
        
        WITH t, sum(CASE 
            WHEN d2 IS NULL THEN 0
            WHEN coalesce(d2.status, 'UNRESOLVED') = 'COMPLETE' THEN 0 
            ELSE 1 
        END) AS unresolved
        
        SET t.unresolved_deps = unresolved
        """

        with self.driver.session() as session:
            session.run(
                query,
                task_id=task_id,
                instruction=task.instruction,
                description=getattr(task, "description", "") or "",
                task_type=task.task_type,
                status=getattr(task, "status", "PENDING"),
                parent_task=parent_task,
                target_output=target_output,
                dependencies=deps,
            )
        print(f"[TASK] Injected: {task_id} (Parent: {parent_task})")

    def mark_data_complete(self, node_id: str, content: Any) -> None:
        node_id = self._normalize_id(node_id)
        content_str = json.dumps(content) if isinstance(content, (dict, list)) else str(content)

        # Mark data complete and refresh affected tasks unresolved_deps, then recompute semaphore.
        query = """
        MERGE (d:DataNode {id: $id})
        SET d.content = $content, d.status = 'COMPLETE'

        WITH d
        MATCH (d)-[:REQUIRES]->(t:TaskNode)
        WITH collect(distinct t) AS tasks
        UNWIND tasks AS t2
        MATCH (dep:DataNode)-[:REQUIRES]->(t2)
        WITH t2, sum(CASE WHEN dep.status = 'COMPLETE' THEN 0 ELSE 1 END) AS unresolved
        SET t2.unresolved_deps = unresolved
        """
        with self.driver.session() as session:
            session.run(query, id=node_id, content=content_str)

        self.recompute_semaphore()

    def mark_task_complete(self, task_id: str) -> None:
        task_id = self._normalize_id(task_id)
        query = """
        MATCH (t:TaskNode {id: $id})
        SET t.status = 'COMPLETE'
        """
        with self.driver.session() as session:
            session.run(query, id=task_id)