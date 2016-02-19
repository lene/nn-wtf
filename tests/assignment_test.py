from directed_graph import DirectedGraph
from dijkstra import shortest_path, shortest_round_trip

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmx.net>'


class AssignmentTest(unittest.TestCase):

    def setUp(self):
        self.graph = DirectedGraph.from_edges('AB5 BC4 CD8 DC8 DE6 AD5 CE2 EB3 AE7')

    def test_answer_1(self):
        self.assertEqual(9, self.graph.distance('ABC'))

    def test_answer_2(self):
        self.assertEqual(5, self.graph.distance('AD'))

    def test_answer_3(self):
        self.assertEqual(13, self.graph.distance('ADC'))

    def test_answer_4(self):
        self.assertEqual(22, self.graph.distance('AEBCD'))

    def test_answer_5(self):
        with self.assertRaises(ValueError):
            self.graph.distance('AED')

    def test_answer_6(self):
        self.assertEqual(2, len(self.graph.find_route_up_to_num_stops('C', 'C', 3)))

    def test_answer_7(self):
        self.assertEqual(3, len(self.graph.find_route_with_exactly_num_stops('A', 'C', 4)))

    def test_answer_8(self):
        route = shortest_path(self.graph, 'A', 'C')
        self.assertEqual(9, self.graph.distance(route))

    def test_answer_9(self):
        route = shortest_round_trip(self.graph, 'B')
        self.assertEqual(9, self.graph.distance(route))

    def test_answer_10(self):
        self.assertEqual(7, len(self.graph.find_routes_up_to_length('C', 'C', 29)))
