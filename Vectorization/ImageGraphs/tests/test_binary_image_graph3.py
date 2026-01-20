import unittest
import numpy as np
import networkx as nx
from Vectorization.ImageGraphs.binary_image_graph3 import binary_image_graph3

class TestBinaryImageGraph3(unittest.TestCase):

    def setUp(self):
        self.bw = np.array([
            [[0, 1], [1, 1]],
            [[0, 1], [0, 0]]
        ], dtype=bool)

    def test_connectivity26(self):
        g, nodenums = binary_image_graph3(self.bw, 26)
        
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 4)
        self.assertEqual(g.number_of_edges(), 6)
        # ... more detailed checks if needed

    def test_connectivity18(self):
        g, nodenums = binary_image_graph3(self.bw, 18)
        
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 4)
        self.assertEqual(g.number_of_edges(), 5)
        # ... more detailed checks if needed

    def test_connectivity6(self):
        g, nodenums = binary_image_graph3(self.bw, 6)
        
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 4)
        self.assertEqual(g.number_of_edges(), 3)
        # ... more detailed checks if needed

    def test_default_connectivity(self):
        g1, nodenums1 = binary_image_graph3(self.bw, 26)
        g2, nodenums2 = binary_image_graph3(self.bw)
        
        np.testing.assert_array_equal(nodenums1, nodenums2)
        self.assertTrue(nx.is_isomorphic(g1, g2, node_match=lambda n1, n2: n1 == n2))

    def test_no_foreground_pixels(self):
        bw = np.zeros((2, 2, 2), dtype=bool)
        g, nodenums = binary_image_graph3(bw)
        
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 0)
        self.assertEqual(g.number_of_edges(), 0)
        np.testing.assert_array_equal(nodenums, np.full((2, 2, 2), -1, dtype=int))

if __name__ == '__main__':
    unittest.main()
