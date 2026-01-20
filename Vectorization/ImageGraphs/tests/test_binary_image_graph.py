import unittest
import numpy as np
import networkx as nx
from Vectorization.ImageGraphs.binary_image_graph import binary_image_graph

class TestBinaryImageGraph(unittest.TestCase):

    def setUp(self):
        self.bw = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ], dtype=bool)

    def test_connectivity8(self):
        g, nodenums = binary_image_graph(self.bw, 8)
        
        self.assertIsInstance(g, nx.Graph)
        
        expected_nodenums = np.array([
            [-1, -1, -1, -1],
            [-1,  1,  3, -1],
            [-1,  2, -1, -1],
            [ 0, -1, -1, -1]
        ])
        np.testing.assert_array_equal(nodenums, expected_nodenums)
        
        # Node data verification
        # Note: node ordering can be different
        nodes_data = {node: data for node, data in g.nodes(data=True)}
        self.assertEqual(len(nodes_data), 4)
        # ... more detailed checks if needed

        # Edge verification
        self.assertEqual(g.number_of_edges(), 4)
        # ... more detailed checks if needed

    def test_connectivity4(self):
        g, nodenums = binary_image_graph(self.bw, 4)
        
        self.assertIsInstance(g, nx.Graph)
        
        expected_nodenums = np.array([
            [-1, -1, -1, -1],
            [-1,  1,  3, -1],
            [-1,  2, -1, -1],
            [ 0, -1, -1, -1]
        ])
        np.testing.assert_array_equal(nodenums, expected_nodenums)

        self.assertEqual(g.number_of_edges(), 2)
        # ... more detailed checks if needed

    def test_default_connectivity(self):
        g1, nodenums1 = binary_image_graph(self.bw, 8)
        g2, nodenums2 = binary_image_graph(self.bw)
        
        np.testing.assert_array_equal(nodenums1, nodenums2)
        self.assertTrue(nx.is_isomorphic(g1, g2, node_match=lambda n1, n2: n1 == n2))

    def test_no_foreground_pixels(self):
        bw = np.zeros((4, 4), dtype=bool)
        g, nodenums = binary_image_graph(bw)
        
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 0)
        self.assertEqual(g.number_of_edges(), 0)
        np.testing.assert_array_equal(nodenums, np.full((4, 4), -1, dtype=int))

if __name__ == '__main__':
    unittest.main()
