from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any

import networkx as nx

from .domain import LoadState
from .motion import EdgeTravelTimeModel

# 一条候选路线
@dataclass(frozen=True)
class SpatialRoute:
    start_node_id: str
    end_node_id: str
    edge_ids: tuple[str, ...]
    free_flow_travel_ms: int


class RouteProvider:
    def __init__(self, model: dict[str, Any], travel_times: EdgeTravelTimeModel) -> None:
        self.edges = {item["id"]: item for item in model["edges"]}
        self.travel_times = travel_times
        self._edges_by_group_and_nodes = {
            (item["robotGroup"], item["start"], item["end"]): item
            for item in model["edges"]
        }

    # 生成候选路线
    def candidate_routes(
        self,
        robot_group: str,
        start_node_id: str,
        end_node_id: str,
        load_state: LoadState,
        limit: int,
        closed_edge_ids: frozenset[str] = frozenset(),
    ) -> tuple[SpatialRoute, ...]:
        # 要 0 条就不给；起点=终点就返回一条"空路线"
        if limit <= 0:
            return ()
        if start_node_id == end_node_id:
            return (SpatialRoute(start_node_id, end_node_id, (), 0),)
        # 生成有向图，边权重 = 这条边的总耗时
        graph = nx.DiGraph()
        for edge in sorted(self.edges.values(), key=lambda item: item["id"]):
            if edge["robotGroup"] != robot_group or edge["id"] in closed_edge_ids:
                continue
            graph.add_edge(
                edge["start"],
                edge["end"],
                edge_id=edge["id"],
                weight=self.travel_times.duration_ms(edge, load_state),
            )
        try:
            # 用 Yen 算法求前 K 条最短路线
            node_paths = islice(
                nx.shortest_simple_paths(
                    graph, start_node_id, end_node_id, weight="weight"
                ),
                limit,
            )
            routes: list[SpatialRoute] = []
            for nodes in node_paths:
                edge_ids = tuple(
                    graph[left][right]["edge_id"]
                    for left, right in zip(nodes, nodes[1:])
                )
                routes.append(
                    SpatialRoute(
                        start_node_id=start_node_id,
                        end_node_id=end_node_id,
                        edge_ids=edge_ids,
                        free_flow_travel_ms=sum(
                            self.travel_times.duration_ms(self.edges[edge_id], load_state)
                            for edge_id in edge_ids
                        ),
                    )
                )
            return tuple(routes)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return ()

    # 只取最快一条的耗时
    def shortest_travel_ms(
        self,
        robot_group: str,
        start_node_id: str,
        end_node_id: str,
        load_state: LoadState,
    ) -> int | None:
        routes = self.candidate_routes(
            robot_group, start_node_id, end_node_id, load_state, limit=1
        )
        return routes[0].free_flow_travel_ms if routes else None
