from directed_graph import DirectedGraph
from dijkstra import PriorityQueue, shortest_path, shortest_round_trip

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmx.net>'


class PriorityQueueTest(unittest.TestCase):

    def setUp(self):
        self.queue = PriorityQueue([(2, '2'), (1, '1'), (0, '0')])

    def test_len(self):
        self.assertEqual(3, len(self.queue))

    def test_pop_returns_correct_order(self):
        self.assertEqual('0', self.queue.pop())
        self.assertEqual('1', self.queue.pop())
        self.assertEqual('2', self.queue.pop())
        self.assertEqual(0, len(self.queue))

    def test_add_item(self):
        self.queue.add_item('-1', -1)
        self.assertEqual(4, len(self.queue))
        self.assertEqual('-1', self.queue.pop())
        self.test_pop_returns_correct_order()

    def test_change_priority(self):
        self.queue.change_priority('2', -1)
        self.assertEqual('2', self.queue.pop())
        self.assertEqual('0', self.queue.pop())
        self.assertEqual('1', self.queue.pop())


class DijkstraTest(unittest.TestCase):

    def setUp(self):
        """
        A simple directed cyclic graph which nonetheless is useful for testing some things you can see intuitively:

        +-------10-----+
        v              |
        A -1-> B -6-+  |
        |      |    |  |
        |      2    v  |
        4      |    D--+
        |      v    ^
        +----> C -3-+
        """
        self.graph = DirectedGraph.from_edges('AB1 AC4 BC2 BD6 CD3 DA10')

    def test_shortest_path_basecase(self):
        route = shortest_path(self.graph, 'A', 'D')
        self.assertEqual(6, self.graph.distance(route))

    def test_shortest_path_round_trip_stays_at_home(self):
        self.assertEqual(1, len(shortest_path(self.graph, 'A', 'A')))

    def test_shortest_round_trip(self):
        possible_trips = self.graph.find_route_up_to_num_stops('A', 'A', 4)
        shortest = shortest_round_trip(self.graph, 'A')
        for route in possible_trips:
            self.assertLessEqual(self.graph.distance(shortest), self.graph.distance(route))
        self.assertEqual(16, self.graph.distance(shortest))

    def test_shortest_round_trip_nonexistent(self):
        graph = DirectedGraph.from_edges('AB1 AC4 BC2 BD6 CD3')
        with self.assertRaises(ValueError):
            shortest_round_trip(graph, 'A')
