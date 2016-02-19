from edge import Edge
import unittest

__author__ = 'Lene Preuss <lene.preuss@gmx.net>'


class EdgeTest(unittest.TestCase):

    def test_edge_as_string(self):
        self.assertEqual(str(Edge(1, 2, 3)), '1 -> 2 [len=3.00]')

    def test_vertices_can_be_chars(self):
        self.assertEqual(str(Edge('A', 'B', 3)), 'A -> B [len=3.00]')

    def test_vertices_can_be_unicode(self):
        self.assertEqual(str(Edge('Ä', '中', 3)), 'Ä -> 中 [len=3.00]')

    def test_zero_distance_fail(self):
        with self.assertRaises(ValueError):
            Edge(1, 2, 0)

    def test_negative_distance_fail(self):
        with self.assertRaises(ValueError):
            Edge(1, 2, -1)

    def test_same_vertex_fail(self):
        with self.assertRaises(ValueError):
            Edge(1, 1, 1)

    def test_from_string(self):
        edge = Edge.from_string('AB1')
        self.assertEqual('A', edge.tail)
        self.assertEqual('B', edge.head)
        self.assertEqual(1, edge.length)

    def test_from_string_float_length(self):
        edge = Edge.from_string('AB1.5')
        self.assertEqual(1.5, edge.length)

    def test_from_string_length_scientific_notation(self):
        edge = Edge.from_string('AB1.5e-2')
        self.assertEqual(1.5e-2, edge.length)

    def test_from_incomplete_string(self):
        with self.assertRaises(ValueError):
            Edge.from_string('AB')

    def test_from_string_with_wrong_length(self):
        with self.assertRaises(ValueError):
            Edge.from_string('ABx')

    def test_from_string_with_zero_length(self):
        with self.assertRaises(ValueError):
            Edge.from_string('AB0')
