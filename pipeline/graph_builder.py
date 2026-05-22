"""
Stage 4: Knowledge Graph Builder — LazyGraphRAG style
Extracts entities and relationships from wiki pages,
builds community clusters, generates summaries.
Inspired by Microsoft's LazyGraphRAG (low-cost variant of GraphRAG).
"""
import json
from collections import defaultdict
from domain_config import inject_domain_into_graph_prompt


ENTITY_EXTRACTION_PROMPT = """You are a knowledge graph builder for research papers.

Given wiki pages from multiple research papers, extract:
1. Named entities: concepts, methods, datasets, metrics, problems, domains
2. Relationships between entities

Return ONLY a JSON object:
{
  "entities": [
    {
      "id": "unique_snake_case_id",
      "name": "Human readable name",
      "type": "concept|method|dataset|metric|problem|domain|model",
      "papers": ["paper titles that mention this entity"],
      "importance": "high|medium|low"
    }
  ],
  "relationships": [
    {
      "source": "entity_id",
      "target": "entity_id", 
      "relation": "extends|contradicts|evaluates_on|applied_to|compared_with|is_limited_by|enables|ignores",
      "paper": "paper title where this relationship appears",
      "evidence": "brief quote or paraphrase supporting this relationship"
    }
  ]
}"""


COMMUNITY_PROMPT = """You are a research analyst identifying thematic communities in a knowledge graph.

Given entities and relationships from research papers, group them into thematic communities.
Each community represents a cluster of closely related concepts, methods, or problems.

Return ONLY a JSON array:
[
  {
    "id": "community_1",
    "theme": "Short theme name (e.g., 'Transformer-based Methods', 'Efficiency Limitations')",
    "entities": ["entity_id_1", "entity_id_2"],
    "summary": "2-3 sentence summary of what this community represents and its significance",
    "gap_signal": "Any gaps or missing connections suggested by this community"
  }
]"""


def extract_entities_and_relations(wiki: dict, client, domain_config: dict = None) -> dict:
    """
    Extract entities and relationships from all wiki pages.
    """
    pages = wiki.get("pages", [])
    cross_links = wiki.get("cross_links", {})

    # Build a compact representation for entity extraction
    compact_wiki = []
    for p in pages:
        compact_wiki.append({
            "title": p.get("title", p.get("source_file", "Unknown")),
            "methods": p.get("methods", []),
            "key_concepts": p.get("key_concepts", []),
            "limitations": p.get("limitations", []),
            "future_work": p.get("future_work", []),
            "datasets": p.get("datasets", []),
            "key_findings": p.get("key_findings", []),
        })

    user_prompt = f"""Wiki pages from {len(pages)} research papers:

{json.dumps(compact_wiki, indent=2)}

Cross-paper connections already identified:
{json.dumps(cross_links, indent=2)}

Extract all entities and relationships for the knowledge graph."""

    raw = client.chat_json(
        inject_domain_into_graph_prompt(ENTITY_EXTRACTION_PROMPT, domain_config or {}),
        user_prompt, max_tokens=4000
    )

    try:
        graph_data = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                graph_data = json.loads(match.group())
            except Exception:
                graph_data = {"entities": [], "relationships": []}
        else:
            graph_data = {"entities": [], "relationships": []}

    return graph_data


def build_communities(graph_data: dict, client) -> list:
    """
    Group entities into thematic communities.
    Each community is a cluster of related concepts.
    """
    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])

    if not entities:
        return []

    user_prompt = f"""Knowledge graph with {len(entities)} entities and {len(relationships)} relationships:

ENTITIES:
{json.dumps(entities, indent=2)}

RELATIONSHIPS:
{json.dumps(relationships, indent=2)}

Group these into 3-6 thematic communities. Focus on identifying communities where gaps exist."""

    raw = client.chat_json(COMMUNITY_PROMPT, user_prompt, max_tokens=3000)

    try:
        communities = json.loads(raw)
        if not isinstance(communities, list):
            communities = []
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                communities = json.loads(match.group())
            except Exception:
                communities = []
        else:
            communities = []

    return communities


def find_orphan_concepts(graph_data: dict, wiki: dict) -> list:
    """
    Find concepts that appear only once or have no connections.
    Orphan concepts are strong gap signals — mentioned but never built upon.
    """
    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])

    # Count connections per entity
    connection_count = defaultdict(int)
    for rel in relationships:
        connection_count[rel.get("source", "")] += 1
        connection_count[rel.get("target", "")] += 1

    orphans = []
    for entity in entities:
        eid = entity.get("id", "")
        connections = connection_count.get(eid, 0)
        papers = entity.get("papers", [])
        # Orphan: important concept mentioned in only 1 paper with few connections
        if connections <= 1 and entity.get("importance") in ["high", "medium"] and len(papers) == 1:
            orphans.append({
                "entity": entity,
                "connections": connections,
                "signal": f"'{entity['name']}' is mentioned in only one paper with minimal connections — potential unexplored area",
            })

    return orphans


def find_missing_bridges(graph_data: dict, wiki: dict) -> list:
    """
    Find future_work in one paper that could be addressed by methods in another.
    These are high-value gap signals.
    """
    pages = wiki.get("pages", [])
    bridges = []

    # Check if cross_links already identified these
    cross_links = wiki.get("cross_links", {})
    limitation_bridges = cross_links.get("limitation_bridges", [])
    bridges.extend([
        {
            "type": "limitation_bridge",
            "description": f"'{b.get('limitation')}' (in {b.get('limitation_in')}) could be addressed by {b.get('solution')} (from {b.get('potential_solution_in')})",
            "source_paper": b.get("limitation_in"),
            "target_paper": b.get("potential_solution_in"),
        }
        for b in limitation_bridges
    ])

    return bridges


def build_knowledge_graph(wiki: dict, client, domain_config: dict = None) -> dict:
    """
    Full graph building pipeline:
    1. Extract entities + relationships
    2. Build communities
    3. Find orphans and bridges
    """
    graph_data = extract_entities_and_relations(wiki, client, domain_config=domain_config)
    communities = build_communities(graph_data, client)
    orphans = find_orphan_concepts(graph_data, wiki)
    bridges = find_missing_bridges(graph_data, wiki)

    return {
        "entities": graph_data.get("entities", []),
        "relationships": graph_data.get("relationships", []),
        "communities": communities,
        "orphan_concepts": orphans,
        "missing_bridges": bridges,
        "stats": {
            "entity_count": len(graph_data.get("entities", [])),
            "relationship_count": len(graph_data.get("relationships", [])),
            "community_count": len(communities),
            "orphan_count": len(orphans),
            "bridge_count": len(bridges),
        },
    }
