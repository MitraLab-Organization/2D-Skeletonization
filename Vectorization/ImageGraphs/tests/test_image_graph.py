import unittest
import numpy as np
import networkx as nx
from Vectorization.ImageGraphs.image_graph import image_graph

class TestImageGraph(unittest.TestCase):

    def test_connectivity8(self):
        g = image_graph((2, 3), 8)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 6)
        self.assertEqual(g.number_of_edges(), 11)

    def test_default_connectivity(self):
        g1 = image_graph((2, 3), 8)
        g2 = image_graph((2, 3))
        self.assertTrue(nx.is_isomorphic(g1, g2))

    def test_connectivity4(self):
        g = image_graph((2, 3), 4)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 6)
        self.assertEqual(g.number_of_edges(), 7)

    def test_empty_image(self):
        g = image_graph((0, 1))
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 0)
        self.assertEqual(g.number_of_edges(), 0)

if __name__ == '__main__':
    unittest.main()
