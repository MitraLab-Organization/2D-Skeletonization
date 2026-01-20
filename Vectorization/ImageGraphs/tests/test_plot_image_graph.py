import unittest
import networkx as nx
import matplotlib.pyplot as plt
from Vectorization.ImageGraphs.image_graph import image_graph
from Vectorization.ImageGraphs.image_graph3 import image_graph3
from Vectorization.ImageGraphs.plot_image_graph import plot_image_graph

class TestPlotImageGraph(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_basic_positive(self):
        g = image_graph((2, 3))
        fig, ax = plot_image_graph(g)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

    def test_3d_not_supported(self):
        g = image_graph3((2, 3, 2))
        with self.assertRaises(ValueError):
            plot_image_graph(g)

if __name__ == '__main__':
    unittest.main()
