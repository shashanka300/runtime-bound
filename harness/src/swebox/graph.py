"""Code-intelligence backends for the three graph tools (spec section 6.3).

Graphs are NetworkX node-link JSON (directed multigraph, ``calls`` edges); embeddings are
``.npz`` archives mapping node id -> float32[256]. Both are loaded once per file and
cached, since every seed and phase of a task shares them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import orjson

MIN_FILE_BYTES = 100  # spec: tools are offered only if both files exceed 100 bytes


def available(graph_file: Path, emb_file: Path) -> bool:
    return all(p.exists() and p.stat().st_size > MIN_FILE_BYTES for p in (graph_file, emb_file))


@dataclass
class CodeIndex:
    graph: nx.MultiDiGraph
    names: list[str]
    matrix: np.ndarray  # row-normalised embeddings aligned with ``names``
    _row: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._row = {n: i for i, n in enumerate(self.names)}

    # -- symbol resolution (4 tiers) -----------------------------------------------------
    def resolve(self, name: str) -> str | None:
        name = name.strip()
        if name in self.graph:
            return str(name)
        suffix = [n for n in self.graph if n.endswith(("." + name, "/" + name))]
        if suffix:
            return str(min(suffix, key=len))
        low = name.lower()
        ci = [n for n in self.graph if n.lower() == low]
        if ci:
            return str(ci[0])
        sub = [n for n in self.graph if low in n.lower()]
        return str(min(sub, key=len)) if sub else None

    # -- tools --------------------------------------------------------------------------
    def neighbors(self, node: str, edge_type: str | None, max_neighbors: int) -> dict[str, Any]:
        resolved = self.resolve(node)
        if resolved is None:
            raise KeyError(f"symbol not found in code graph: {node!r}")
        want = edge_type.lower() if edge_type else None
        seen: dict[str, None] = {}
        for _, dst, data in self.graph.out_edges(resolved, data=True):
            if want is None or str(data.get("type", "")).lower() == want:
                seen[dst] = None
        for src, _, data in self.graph.in_edges(resolved, data=True):
            if want is None or str(data.get("type", "")).lower() == want:
                seen[src] = None
        neighbors = list(seen)[:max_neighbors]
        return {"node": resolved, "neighbors": neighbors, "count": len(neighbors)}

    def similar(self, query: str, k: int) -> dict[str, Any]:
        resolved = self.resolve(query)
        if resolved is None or resolved not in self._row:
            raise KeyError(
                f"no embedding for {query!r}; pass a class, function or module symbol name"
            )
        sims = self.matrix @ self.matrix[self._row[resolved]]
        order = np.argsort(-sims)
        results = []
        for i in order:
            name = self.names[int(i)]
            if name == resolved:
                continue
            results.append(
                {
                    "node_name": name,
                    "code": str(self.graph.nodes[name].get("text", ""))
                    if name in self.graph
                    else "",
                    "similarity": round(float(sims[int(i)]), 4),
                }
            )
            if len(results) >= k:
                break
        return {"query": query, "results": results, "count": len(results)}

    def subgraph(self, nodes: list[str]) -> dict[str, Any]:
        resolved = [r for n in nodes if (r := self.resolve(n)) is not None]
        sub = self.graph.subgraph(resolved)
        edges = [
            {"from": u, "to": v, "type": str(d.get("type", "")).upper()}
            for u, v, d in sub.edges(data=True)
        ]
        return {
            "nodes": list(sub.nodes),
            "edges": edges,
            "node_count": sub.number_of_nodes(),
            "edge_count": len(edges),
        }


@lru_cache(maxsize=16)
def load(graph_file: Path, emb_file: Path) -> CodeIndex:
    data = orjson.loads(graph_file.read_bytes())
    g = nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")
    with np.load(emb_file) as z:
        names = list(z.keys())
        mat = np.stack([z[n] for n in names]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat /= np.where(norms == 0, 1.0, norms)
    return CodeIndex(g, names, mat)
