import unittest
import numpy as np
import networkx as nx
from Vectorization.ImageGraphs.adjacent_regions_graph import adjacent_regions_graph

class TestAdjacentRegionsGraph(unittest.TestCase):

    def test_connectivity8(self):
        L = np.kron(np.array([[5, 10], [1.5, 20]]), np.ones((4, 3)))
        g = adjacent_regions_graph(L, 8)
        
        labels = [data['Label'] for node, data in g.nodes(data=True)]
        self.assertEqual(labels, [1.5, 5, 10, 20])
        
        # The node ordering might be different, so we need to check edges based on labels
        label_map = {data['Label']: node for node, data in g.nodes(data=True)}
        
        expected_edges = [{1.5, 5}, {1.5, 10}, {1.5, 20}, {5, 10}, {5, 20}, {10, 20}]
        actual_edges = [{g.nodes[u]['Label'], g.nodes[v]['Label']} for u, v in g.edges()]
        
        self.assertEqual(len(actual_edges), len(expected_edges))
        for edge in expected_edges:
            self.assertIn(edge, actual_edges)

    def test_connectivity4(self):
        L = np.kron(np.array([[5, 10], [1.5, 20]]), np.ones((4, 3)))
        g = adjacent_regions_graph(L, 4)
        
        labels = sorted([data['Label'] for node, data in g.nodes(data=True)])
        self.assertEqual(labels, [1.5, 5, 10, 20])
        
        label_map = {data['Label']: node for node, data in g.nodes(data=True)}
        
        expected_edges = [{1.5, 5}, {1.5, 20}, {5, 10}, {10, 20}]
        actual_edges = [{g.nodes[u]['Label'], g.nodes[v]['Label']} for u, v in g.edges()]
        
        self.assertEqual(len(actual_edges), len(expected_edges))
        for edge in expected_edges:
            self.assertIn(edge, actual_edges)

    def test_default_connectivity(self):
        L = np.kron(np.array([[5, 10], [1.5, 20]]), np.ones((4, 3)))
        g1 = adjacent_regions_graph(L, 8)
        g2 = adjacent_regions_graph(L)
        
        self.assertTrue(nx.is_isomorphic(g1, g2, node_match=lambda n1, n2: n1['Label'] == n2['Label']))

if __name__ == '__main__':
    unittest.main()
