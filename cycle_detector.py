import torch
from collections import deque, defaultdict

class CycleDetector:
    def __init__(self, edge_index, num_nodes):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.adj_list = self._build_adj_list()

    def _build_adj_list(self):
        adj = defaultdict(list)
        edges = self.edge_index.t().tolist()
        for u, v in edges:
            adj[u].append(v)
        return adj

    def _bfs_shortest_path(self, start, target, ignore_u, ignore_v):
        queue = deque([(start, 0)])
        visited = set([start])
        while queue:
            node, dist = queue.popleft()
            if node == target:
                return dist
            for neighbor in self.adj_list[node]:
                if (node == ignore_u and neighbor == ignore_v) or (node == ignore_v and neighbor == ignore_u):
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def extract_cycle_mask(self, max_cycle_length=6):
        num_edges = self.edge_index.shape[1]
        cycle_mask = torch.zeros(num_edges, dtype=torch.float32)
        edges = self.edge_index.t().tolist()
        for i, (u, v) in enumerate(edges):
            shortest_path = self._bfs_shortest_path(u, v, u, v)
            if shortest_path <= max_cycle_length - 1:
                cycle_mask[i] = 1.0
        return cycle_mask
