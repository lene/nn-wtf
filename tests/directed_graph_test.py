from directed_graph import DirectedGraph
from edge import Edge

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmx.net>'


class GraphTest(unittest.TestCase):

    def setUp(self):
        """
        A simple directed acyclic graph which nonetheless is useful for testing some things you can see intuitively:

        A -1-> B -6-+
        |      |    |
        |      2    v
        4      |    D
        |      v    ^
        +----> C -3-+
        """
        self.graph = DirectedGraph.from_edges('AB1 AC4 BC2 BD6 CD3')

    def test_from_edges_vertices(self):
        graph = DirectedGraph.from_edges('121 134 232 246 343')
        self.assertEqual(4, len(graph.vertices))
        self.assertEqual({'1', '2', '3', '4'}, set(graph.vertices))

    def test_from_edges_vertices_chars(self):
        self.assertEqual(4, len(self.graph.vertices))
        self.assertEqual({'A', 'B', 'C', 'D'}, set(self.graph.vertices))

    def test_from_edges_vertices_unicode(self):
        graph = DirectedGraph.from_edges('ÄЖ1 Ä中4 Ж中2 Ж™6 中™3')
        self.assertEqual(4, len(graph.vertices))
        self.assertEqual({'Ä', 'Ж', '中', '™'}, set(graph.vertices))

    def test_from_edges_edges(self):
        for edges in self.graph.edges.values():
            for edge in edges:
                if Edge('A', 'B', 1) == edge:
                    return
        self.fail('Edge not found')

    def test_from_edges_ignores_commas(self):
        graph = DirectedGraph.from_edges('AB1, AC4, BC2, BD6, CD3')
        self.assertEqual({'A', 'B', 'C', 'D'}, set(graph.vertices))

    def test_direct_distance(self):
        self.assertEqual(1, self.graph.direct_distance('A', 'B'))
        self.assertEqual(4, self.graph.direct_distance('A', 'C'))
        self.assertEqual(2, self.graph.direct_distance('B', 'C'))
        self.assertEqual(6, self.graph.direct_distance('B', 'D'))
        self.assertEqual(3, self.graph.direct_distance('C', 'D'))

    def test_direct_distance_fail(self):
        with self.assertRaises(ValueError):
            self.graph.direct_distance('A', 'D')

    def test_distances_path(self):
        self.assertEqual(3, self.graph.distance(['A', 'B', 'C']))
        self.assertEqual(3, self.graph.distance('ABC'))

    def test_distance_equals_direct_distance_where_applicable(self):
        self.assertEqual(1, self.graph.distance('AB'))
        self.assertEqual(4, self.graph.distance('AC'))
        self.assertEqual(2, self.graph.distance('BC'))
        self.assertEqual(6, self.graph.distance('BD'))
        self.assertEqual(3, self.graph.distance('CD'))

    def test_distance_fail(self):
        with self.assertRaises(ValueError):
            self.graph.distance('AD')
        with self.assertRaises(ValueError):
            self.graph.distance('ABA')
        with self.assertRaises(ValueError):
            self.graph.distance('AX')

    def test_routes_acyclic_graph(self):
        routes = self.graph.find_routes_up_to_length('A', 'D', 30)
        self.assertEqual(3, len(routes))
        self.assertIn(['A', 'B', 'D'], routes)
        self.assertIn(['A', 'C', 'D'], routes)
        self.assertIn(['A', 'B', 'C', 'D'], routes)

    def test_routes_acyclic_graph_max_length(self):
        routes = self.graph.find_routes_up_to_length('A', 'D', 6)
        self.assertEqual(1, len(routes))
        self.assertIn(['A', 'B', 'C', 'D'], routes)

    def test_cyclic_graph(self):
        self.graph.add_edge(Edge('D', 'A', 10))
        routes = self.graph.find_routes_up_to_length('A', 'D', 30)
        self.assertGreater(len(routes), 3)
        self.assertTrue(all(self.graph.distance(route) <= 30 for route in routes))

    def test_round_trips(self):
        """
        If I add an edge to a DAG to make it cyclic, in particular from a sink vertex to the end point, I get a graph
        with round trips from the start point to itself, which have the length of the route from the start point to
        the sink, plus the length of the added edge.
        """
        dag_routes = self.graph.find_routes_up_to_length('A', 'D', 10)

        self.graph.add_edge(Edge('D', 'A', 10))
        routes = self.graph.find_routes_up_to_length('A', 'A', 20)

        self.assertEqual(len(dag_routes), len(routes))
        for route in dag_routes:
            self.assertIn(route + ['A'], routes)
            self.assertEqual(self.graph.distance(route)+10, self.graph.distance(route + ['A']))

    def test_max_stops(self):
        self.graph.add_edge(Edge('D', 'A', 10))
        routes = self.graph.find_route_up_to_num_stops('A', 'D', 2)
        self.assertEqual(2, len(routes))
        routes = self.graph.find_route_up_to_num_stops('A', 'D', 3)
        self.assertEqual(3, len(routes))

        routes = self.graph.find_route_up_to_num_stops('A', 'A', 3)
        self.assertEqual(2, len(routes))

    def test_exact_num_stops(self):
        routes = self.graph.find_route_with_exactly_num_stops('A', 'D', 2)
        self.assertEqual(2, len(routes))

    def test_exact_num_stops_2(self):
        routes = self.graph.find_route_with_exactly_num_stops('A', 'D', 3)
        self.assertEqual(1, len(routes))

    def test_from_edges(self):
        self._test_graph_ab1_ac2_bc2(DirectedGraph.from_edges('AB1 AC2 BC2'))

    def test_from_file(self):
        from tempfile import mkstemp
        from os import remove
        _, filename = mkstemp()

        with open(filename, 'w+') as file:
            file.write('AB1 AC2 BC2')

        try:
            self._test_graph_ab1_ac2_bc2(DirectedGraph.from_file(filename))
        finally:
            remove(filename)

    def _test_graph_ab1_ac2_bc2(self, graph):
        self.assertEqual(3, len(graph.vertices))
        self.assertEqual({'A', 'B', 'C'}, set(graph.vertices))
        self.assertEqual(3, len(graph.edges))
        self.assertEqual(2, len(graph.edges['A']))
        self.assertEqual(1, len(graph.edges['B']))
        self.assertEqual(0, len(graph.edges['C']))

    def _print_routes_with_distance(self, routes):
        from pprint import pprint
        pprint(
            [(route, self.graph.distance(route))
             for route in sorted(routes, key=lambda r: self.graph.distance(r))]
        )

from directed_graph import trip_time, find_trips_up_to_duration

class TripTimeTest(unittest.TestCase):

    def setUp(self):
        self.graph = DirectedGraph.from_edges('AB5 BC4 CD8 DC8 DE6 AD5 CE2 EB3 AE7')

    def test_simple_route(self):
        self.assertEqual(11, trip_time(self.graph, 'ABC'))

    def test_question_2(self):
        self.assertEqual(5, trip_time(self.graph, 'AD'))

    def test_question_3(self):
        self.assertEqual(15, trip_time(self.graph, 'ADC'))

    def test_question_4(self):
        self.assertEqual(28, trip_time(self.graph, 'AEBCD'))

    def test_zero_length(self):
        self.assertEqual(0, trip_time(self.graph, 'A'))

    def test_answer_5(self):
        with self.assertRaises(ValueError):
            trip_time(self.graph, 'AED')

    def test_find_trips_up_to_duration(self):
        graph = DirectedGraph.from_edges('AB1 AC4 BC2 BD6 CD3')
        self.assertEqual(2, len(find_trips_up_to_duration(graph, 'A', 'C', 5)))














