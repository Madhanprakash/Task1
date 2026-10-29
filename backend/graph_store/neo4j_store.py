from neo4j import GraphDatabase
from backend.core.config import get_settings


class Neo4jStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.driver = GraphDatabase.driver(self.settings.neo4j_uri, auth=(self.settings.neo4j_username, self.settings.neo4j_password))

    def initialize(self) -> None:
        with self.driver.session(database=self.settings.neo4j_database) as s:
            for query in ["CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE", "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE", "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE"]: s.run(query).consume()

    def save_document(self, document: dict, chunks: list[dict], facts: list[dict]) -> None:
        with self.driver.session(database=self.settings.neo4j_database) as s:
            s.run("MERGE (d:Document {id:$id}) SET d += $doc", id=document["id"], doc={k:v for k,v in document.items() if k != "id"}).consume()
            s.run("UNWIND $chunks AS c MATCH (d:Document {id:$document_id}) MERGE (x:Chunk {id:c.id}) SET x += c MERGE (d)-[:CONTAINS]->(x)", chunks=chunks, document_id=document["id"]).consume()
            for fact in facts:
                entities = [{"key": " ".join(str(e.get("name", "")).lower().split()), "name": str(e.get("name", ""))[:200], "type": str(e.get("type", "Concept"))[:80]} for e in fact["entities"] if e.get("name")]
                s.run("UNWIND $e AS item MERGE (x:Entity {key:item.key}) SET x.name=item.name, x.type=item.type", e=entities).consume()
                s.run("UNWIND $e AS item MATCH (c:Chunk {id:$chunk}) MATCH (x:Entity {key:item.key}) MERGE (c)-[:MENTIONS]->(x)", e=entities, chunk=fact["chunk_id"]).consume()
                relations = [{"source":" ".join(str(r.get("source", "")).lower().split()), "target":" ".join(str(r.get("target", "")).lower().split()), "predicate":str(r.get("predicate", "RELATED_TO"))[:100], "confidence":float(r.get("confidence", 0.5))} for r in fact["relations"] if r.get("source") and r.get("target") and r.get("predicate")]
                s.run("UNWIND $r AS item MATCH (a:Entity {key:item.source}),(b:Entity {key:item.target}) MERGE (a)-[x:RELATED_TO {predicate:item.predicate, chunk_id:$chunk}]->(b) SET x.confidence=item.confidence", r=relations, chunk=fact["chunk_id"]).consume()

    def list_documents(self, offset: int, limit: int) -> tuple[list[dict], int]:
        with self.driver.session(database=self.settings.neo4j_database) as s:
            total = s.run("MATCH (d:Document) RETURN count(d) AS total").single()["total"]
            records = s.run("MATCH (d:Document) OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk) RETURN d{.id,.filename,.checksum,.status,.created_at,.updated_at,chunk_count:count(c)} AS d ORDER BY d.created_at DESC SKIP $offset LIMIT $limit", offset=offset, limit=limit)
            return [r["d"] for r in records], total

    def delete_document(self, identifier: str) -> bool:
        with self.driver.session(database=self.settings.neo4j_database) as s:
            result = s.run("MATCH (d:Document {id:$id}) OPTIONAL MATCH (d)-[:CONTAINS]->(c) DETACH DELETE d,c RETURN count(*) AS count", id=identifier).single()
            return bool(result and result["count"])

    def graph(self, limit: int) -> dict:
        with self.driver.session(database=self.settings.neo4j_database) as s:
            records = s.run("MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) RETURN a.key AS source,a.name AS source_name,b.key AS target,b.name AS target_name,r.predicate AS predicate LIMIT $limit", limit=limit)
            edges = [x.data() for x in records]
        nodes = {x["source"]: {"id":x["source"],"label":x["source_name"]} for x in edges} | {x["target"]: {"id":x["target"],"label":x["target_name"]} for x in edges}
        return {"nodes": list(nodes.values()), "edges": [{"source":x["source"], "target":x["target"], "label":x["predicate"]} for x in edges]}
