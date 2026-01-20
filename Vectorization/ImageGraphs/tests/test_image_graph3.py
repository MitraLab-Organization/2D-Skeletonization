import unittest
import numpy as np
import networkx as nx
from Vectorization.ImageGraphs.image_graph3 import image_graph3

class TestImageGraph3(unittest.TestCase):

    def test_connectivity26(self):
        g = image_graph3((2, 3, 2), 26)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 12)
        self.assertEqual(g.number_of_edges(), 50)

    def test_default_connectivity(self):
        g1 = image_graph3((2, 3, 2), 26)
        g2 = image_graph3((2, 3, 2))
        self.assertTrue(nx.is_isomorphic(g1, g2))

    def test_connectivity18(self):
        g = image_graph3((2, 3, 2), 18)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 12)
        self.assertEqual(g.number_of_edges(), 42)

    def test_connectivity6(self):
        g = image_graph3((2, 3, 2), 6)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 12)
        self.assertEqual(g.number_of_edges(), 20)

    def test_empty_image(self):
        g = image_graph3((2, 0, 3))
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 0)
        self.assertEqual(g.number_of_edges(), 0)

if __name__ == '__main__':
    unittest.main()
